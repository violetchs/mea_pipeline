"""pi-VAE style latent-state analysis for binned MEA activity."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class PiVAEAnalysisConfig:
    input_dim: int
    cov_dim: int = 6
    latent_dim: int = 16
    hidden_dim: int = 96
    beta_kl: float = 2e-4
    bin_ms: float = 10.0
    observation_loss: str = "poisson"
    epochs: int = 12
    batch_size: int = 16
    learning_rate: float = 8e-4
    seed: int = 7
    device: str = "auto"


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
        from torch import nn
        from torch.nn.utils import clip_grad_norm_
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency.
        raise RuntimeError("pi-VAE analysis requires PyTorch. Install torch to use this dynamics model.") from exc
    return torch, F, nn, clip_grad_norm_, DataLoader, Dataset


def _choose_device(torch, name: str):
    requested = str(name or "auto").strip().lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for pi-VAE, but CUDA is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(torch, seed: int) -> None:
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _smooth_signal(x: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    kernel = np.ones(2 * radius + 1, dtype=np.float32)
    kernel /= max(float(kernel.sum()), 1.0)
    padded = np.pad(np.asarray(x, dtype=np.float32), (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def derive_pivae_covariates(counts: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(counts, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if values.ndim != 3:
        return np.zeros((0, 0, 6), dtype=np.float32)
    sample_count, bin_count, _channel_count = values.shape
    if sample_count == 0 or bin_count == 0:
        return np.zeros((sample_count, bin_count, 6), dtype=np.float32)
    pop = np.sum(np.maximum(values, 0.0), axis=2).reshape(-1).astype(np.float32)
    smooth_radius = max(1, min(8, pop.size // 8 if pop.size >= 8 else 1))
    pop_smooth = _smooth_signal(pop, smooth_radius)
    pop_delta = np.diff(pop_smooth, prepend=pop_smooth[0]).astype(np.float32)
    threshold = float(np.mean(pop) + 2.0 * np.std(pop))
    burst = (pop >= threshold).astype(np.float32)
    rel_t = np.tile(np.linspace(0.0, 1.0, bin_count, dtype=np.float32), sample_count)
    cov = np.stack(
        [
            pop / max(float(np.percentile(pop, 99.0)), 1.0),
            pop_smooth / max(float(np.percentile(pop_smooth, 99.0)), 1.0),
            pop_delta / max(float(np.percentile(np.abs(pop_delta), 99.0)), 1.0),
            burst,
            np.sin(2.0 * np.pi * rel_t),
            np.cos(2.0 * np.pi * rel_t),
        ],
        axis=1,
    )
    return cov.reshape(sample_count, bin_count, -1).astype(np.float32)


def _decoder_loading_matrix(model, cfg: PiVAEAnalysisConfig, torch, device) -> np.ndarray:
    latent_dim = int(cfg.latent_dim)
    cov_dim = int(cfg.cov_dim)
    with torch.no_grad():
        z0 = torch.zeros((1, latent_dim), dtype=torch.float32, device=device)
        c0 = torch.zeros((1, cov_dim), dtype=torch.float32, device=device)
        base = model.decode_rate(z0, c0)
        rows = []
        for dim in range(latent_dim):
            z = torch.zeros((1, latent_dim), dtype=torch.float32, device=device)
            z[:, dim] = 1.0
            rows.append((model.decode_rate(z, c0) - base).squeeze(0).detach().cpu().numpy())
    return np.nan_to_num(np.asarray(rows, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def fit_pivae_latent_states(
    activity_rate: np.ndarray,
    *,
    latent_dim: int = 16,
    bin_ms: float = 10.0,
    hidden_dim: int | None = None,
    epochs: int = 12,
    batch_size: int = 16,
    beta_kl: float = 2e-4,
    learning_rate: float = 8e-4,
    observation_loss: str = "poisson",
    seed: int = 7,
    device: str = "auto",
    cancel_check=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit a pi-VAE style conditional VAE and return latent/reconstructed rates.

    ``activity_rate`` is expected as ``samples x time_bins x channels`` in Hz.
    Internally it is converted to per-bin counts for the Poisson/NB likelihood.
    """

    torch, F, nn, clip_grad_norm_, DataLoader, Dataset = _require_torch()

    rates = np.nan_to_num(np.asarray(activity_rate, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if rates.ndim != 3:
        return np.zeros((0, 0, 0), dtype=float), np.zeros((0, 0, 0), dtype=float), {}
    sample_count, bin_count, channel_count = rates.shape
    if sample_count == 0 or bin_count == 0 or channel_count == 0:
        return np.zeros((sample_count, bin_count, 0), dtype=float), np.zeros_like(rates), {}

    bin_scale = max(1e-9, float(bin_ms) / 1000.0)
    counts = np.maximum(rates, 0.0) * bin_scale
    covariates = derive_pivae_covariates(counts)
    components = min(max(1, int(latent_dim)), max(1, channel_count))
    hidden = int(hidden_dim) if hidden_dim is not None else max(32, min(192, max(components * 4, channel_count * 2)))
    cfg = PiVAEAnalysisConfig(
        input_dim=int(channel_count),
        cov_dim=int(covariates.shape[2]),
        latent_dim=int(components),
        hidden_dim=int(hidden),
        beta_kl=float(beta_kl),
        bin_ms=float(bin_ms),
        observation_loss=str(observation_loss or "poisson").strip().lower(),
        epochs=max(1, int(epochs)),
        batch_size=max(1, int(batch_size)),
        learning_rate=float(learning_rate),
        seed=int(seed),
        device=str(device or "auto"),
    )
    if cfg.observation_loss not in {"poisson", "nb"}:
        cfg.observation_loss = "poisson"

    class _SequenceDataset(Dataset):
        def __init__(self, x: np.ndarray, c: np.ndarray) -> None:
            self.x = x.astype(np.float32, copy=False)
            self.c = c.astype(np.float32, copy=False)

        def __len__(self) -> int:
            return int(self.x.shape[0])

        def __getitem__(self, index: int):
            return torch.from_numpy(self.x[index]), torch.from_numpy(self.c[index])

    def _counts_to_state(x):
        return torch.log1p(x * (1000.0 / max(float(cfg.bin_ms), 1e-9)))

    def _poisson_nll_no_const(mu, y):
        mu = mu.clamp_min(1e-8)
        return mu - y * torch.log(mu)

    def _nb_nll_no_const(mu, theta, y):
        mu = mu.clamp_min(1e-8)
        theta = theta.clamp_min(1e-4)
        log_theta = torch.log(theta)
        log_theta_mu = torch.log(theta + mu)
        logprob = (
            torch.lgamma(y + theta)
            - torch.lgamma(theta)
            - torch.lgamma(y + 1.0)
            + theta * (log_theta - log_theta_mu)
            + y * (torch.log(mu) - log_theta_mu)
        )
        return -logprob

    class _PiVAEStyle(nn.Module):
        def __init__(self, model_cfg: PiVAEAnalysisConfig) -> None:
            super().__init__()
            d = int(model_cfg.input_dim)
            c = int(model_cfg.cov_dim)
            z = int(model_cfg.latent_dim)
            h = int(model_cfg.hidden_dim)
            self.encoder = nn.Sequential(nn.Linear(d + c, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
            self.to_mu = nn.Linear(h, z)
            self.to_logvar = nn.Linear(h, z)
            self.prior_net = nn.Sequential(nn.Linear(c, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
            self.prior_mu = nn.Linear(h, z)
            self.prior_logvar = nn.Linear(h, z)
            self.decoder = nn.Sequential(nn.Linear(z + c, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
            self.rate_decoder = nn.Linear(h, d)
            self.log_theta = nn.Parameter(torch.zeros(d))

        def encode(self, x, c):
            state = _counts_to_state(x)
            enc = self.encoder(torch.cat([state, c], dim=-1))
            return self.to_mu(enc), self.to_logvar(enc)

        def decode_rate(self, z, c):
            dec_h = self.decoder(torch.cat([z, c], dim=-1))
            rates_hz = F.softplus(self.rate_decoder(dec_h))
            return rates_hz * (float(cfg.bin_ms) / 1000.0)

        def forward(self, x, c):
            mu_z, logvar_z = self.encode(x, c)
            prior_h = self.prior_net(c)
            prior_mu = self.prior_mu(prior_h)
            prior_logvar = self.prior_logvar(prior_h)
            z = mu_z + torch.exp(0.5 * logvar_z) * torch.randn_like(mu_z)
            mu = self.decode_rate(z, c)
            if cfg.observation_loss == "nb":
                theta = F.softplus(self.log_theta).view(1, 1, -1)
                recon = _nb_nll_no_const(mu, theta, x).mean()
            else:
                recon = _poisson_nll_no_const(mu, x).mean()
            kl = 0.5 * torch.mean(
                prior_logvar
                - logvar_z
                + (logvar_z.exp() + (mu_z - prior_mu).square()) / prior_logvar.exp().clamp_min(1e-8)
                - 1.0
            )
            return {"loss": recon + float(cfg.beta_kl) * kl, "recon": recon, "kl": kl, "z": mu_z, "mu": mu}

        @torch.no_grad()
        def infer_latent(self, x, c):
            mu_z, _logvar_z = self.encode(x, c)
            return mu_z

        @torch.no_grad()
        def reconstruct(self, x, c):
            return self.decode_rate(self.infer_latent(x, c), c)

    _set_seed(torch, int(cfg.seed))
    run_device = _choose_device(torch, cfg.device)
    model = _PiVAEStyle(cfg).to(run_device)
    loader = DataLoader(_SequenceDataset(counts, covariates), batch_size=int(cfg.batch_size), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=1e-5)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for epoch in range(int(cfg.epochs)):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("pi-VAE analysis cancelled")
        model.train()
        rows = []
        for x_batch, c_batch in loader:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("pi-VAE analysis cancelled")
            opt.zero_grad(set_to_none=True)
            out = model(x_batch.to(run_device), c_batch.to(run_device))
            out["loss"].backward()
            clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            rows.append(
                {
                    "loss": float(out["loss"].detach().cpu()),
                    "recon": float(out["recon"].detach().cpu()),
                    "kl": float(out["kl"].detach().cpu()),
                }
            )
        mean_loss = float(np.mean([row["loss"] for row in rows])) if rows else float("inf")
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": mean_loss,
                "recon": float(np.mean([row["recon"] for row in rows])) if rows else float("inf"),
                "kl": float(np.mean([row["kl"] for row in rows])) if rows else 0.0,
            }
        )
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    latent_chunks = []
    recon_chunks = []
    eval_loader = DataLoader(_SequenceDataset(counts, covariates), batch_size=int(cfg.batch_size), shuffle=False)
    with torch.no_grad():
        for x_batch, c_batch in eval_loader:
            x_batch = x_batch.to(run_device)
            c_batch = c_batch.to(run_device)
            latent_chunks.append(model.infer_latent(x_batch, c_batch).detach().cpu().numpy())
            recon_chunks.append(model.reconstruct(x_batch, c_batch).detach().cpu().numpy())
    latent_states = np.concatenate(latent_chunks, axis=0).astype(float) if latent_chunks else np.zeros((sample_count, bin_count, components), dtype=float)
    reconstructed_counts = np.concatenate(recon_chunks, axis=0).astype(float) if recon_chunks else np.zeros_like(counts, dtype=float)
    reconstructed_rates = np.maximum(reconstructed_counts / bin_scale, 0.0)
    loadings = _decoder_loading_matrix(model, cfg, torch, run_device)
    params = {
        "method": "pi_vae",
        "latent_dim": int(components),
        "loadings": loadings,
        "mean": np.mean(rates.reshape((-1, channel_count)), axis=0) if rates.size else np.zeros(channel_count, dtype=float),
        "noise_variance": np.var(rates.reshape((-1, channel_count)) - reconstructed_rates.reshape((-1, channel_count)), axis=0) if rates.size else np.zeros(channel_count, dtype=float),
        "log_likelihood": np.zeros(0, dtype=float),
        "n_iter": int(cfg.epochs),
        "model_config": asdict(cfg),
        "training_history": history,
        "best_loss": float(best_loss if math.isfinite(best_loss) else 0.0),
        "observation_loss": str(cfg.observation_loss),
        "covariates": covariates,
    }
    return latent_states, reconstructed_rates, params
