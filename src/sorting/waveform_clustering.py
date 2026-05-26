"""Automatic waveform clustering for NEV spike waveforms."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


ProgressCallback = Optional[Callable[[int, str], None]]


@dataclass
class WaveformClusteringConfig:
    reduction_method: str = "pca"
    clustering_method: str = "kmeans"
    min_spikes: int = 25
    max_clusters: int = 4
    fixed_clusters: int = 2
    pca_components: int = 5
    ica_components: int = 5
    ica_max_iter: int = 300
    gmm_covariance_type: str = "full"
    dbscan_eps: float = 0.8
    dbscan_min_samples: int = 10
    min_silhouette: float = 0.08
    max_silhouette_samples: int = 1000
    random_state: int = 7


def cluster_nev_waveforms(
    data,
    config: WaveformClusteringConfig,
    progress: ProgressCallback = None,
    channels: Iterable[str] | None = None,
) -> Dict[str, object]:
    """Cluster NEV spike waveforms channel by channel.

    The returned structure is compatible with ``UnifiedMEAData.sorting``: each
    channel receives a labels array aligned to that channel's spike times and
    waveform rows.
    """

    available_channels = sorted(data.waveforms.keys(), key=_channel_sort_key)
    if channels is None:
        selected_channels = available_channels
    else:
        requested_channels = {str(channel) for channel in channels}
        selected_channels = [channel for channel in available_channels if channel in requested_channels]
    sorting: Dict[str, object] = {
        "method": "waveform_clustering",
        "params": config.__dict__.copy(),
        "channels": {},
        "summary": {},
    }

    if not selected_channels:
        return sorting

    for index, channel in enumerate(selected_channels):
        _emit(progress, int(index / len(selected_channels) * 95), f"Clustering {channel}")
        waveforms = np.asarray(data.waveforms[channel], dtype=float)
        noise_mask = _existing_noise_mask(data, channel, waveforms.shape[0] if waveforms.ndim >= 1 else 0)
        labels, embedding, info = _cluster_channel_preserving_noise(waveforms, noise_mask, config)
        sorting["channels"][channel] = {
            "method": "waveform_clustering",
            "labels": labels,
            "embedding": embedding,
            "cluster_count": int(info["cluster_count"]),
            "silhouette": info["silhouette"],
            "spike_count": int(waveforms.shape[0]) if waveforms.ndim >= 1 else 0,
            "clustered_spike_count": int(info.get("clustered_spike_count", waveforms.shape[0] if waveforms.ndim >= 1 else 0)),
            "noise_count": int(info.get("noise_count", 0)),
        }

    summary = _summarize(sorting["channels"])
    sorting["summary"] = summary
    _emit(progress, 100, f"Waveform clustering complete: {summary['total_clusters']} clusters")
    return sorting


def _existing_noise_mask(data, channel: str, expected_size: int) -> np.ndarray:
    if expected_size <= 0 or not isinstance(getattr(data, "sorting", None), dict):
        return np.zeros(max(0, expected_size), dtype=bool)
    payload = data.sorting.get(channel, {})
    if not isinstance(payload, dict):
        return np.zeros(expected_size, dtype=bool)
    labels = payload.get("waveform_cluster_labels")
    if labels is None:
        labels = payload.get("labels")
    if labels is None:
        return np.zeros(expected_size, dtype=bool)
    labels = np.asarray(labels, dtype=np.int32)
    if labels.size != expected_size:
        return np.zeros(expected_size, dtype=bool)
    return labels == -1


def _cluster_channel_preserving_noise(waveforms: np.ndarray, noise_mask: np.ndarray, config: WaveformClusteringConfig):
    noise_mask = np.asarray(noise_mask, dtype=bool)
    if waveforms.ndim == 2 and waveforms.shape[0] and noise_mask.size == waveforms.shape[0]:
        invalid_rows = ~np.all(np.isfinite(waveforms), axis=1)
        noise_mask = noise_mask | invalid_rows
    if waveforms.ndim != 2 or waveforms.shape[0] == 0 or noise_mask.size != waveforms.shape[0] or not np.any(noise_mask):
        labels, embedding, info = _cluster_channel(waveforms, config)
        info = {
            **info,
            "clustered_spike_count": int(waveforms.shape[0]) if waveforms.ndim >= 1 else 0,
            "noise_count": 0,
        }
        return labels, embedding, info

    keep_mask = ~noise_mask
    clustered_waveforms = waveforms[keep_mask]
    labels, embedding, info = _cluster_channel(clustered_waveforms, config)

    full_labels = np.full(waveforms.shape[0], -1, dtype=np.int32)
    full_labels[keep_mask] = labels
    if embedding.ndim == 2:
        full_embedding = np.full((waveforms.shape[0], embedding.shape[1]), np.nan, dtype=np.float32)
        full_embedding[keep_mask] = embedding
    else:
        full_embedding = np.zeros((waveforms.shape[0], 0), dtype=np.float32)

    info = {
        **info,
        "cluster_count": _cluster_count(full_labels),
        "clustered_spike_count": int(np.count_nonzero(keep_mask)),
        "noise_count": int(np.count_nonzero(noise_mask)),
    }
    return full_labels, full_embedding, info


def waveform_embedding(waveforms: np.ndarray, config: WaveformClusteringConfig) -> np.ndarray:
    """Return reduction-space coordinates for a channel's waveform matrix."""

    waveforms = np.asarray(waveforms, dtype=float)
    if waveforms.ndim != 2 or waveforms.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return _waveform_features(waveforms, config)


def _cluster_channel(waveforms: np.ndarray, config: WaveformClusteringConfig):
    if waveforms.ndim != 2 or waveforms.shape[0] == 0:
        return np.zeros(0, dtype=np.int32), np.zeros((0, 0), dtype=np.float32), {"cluster_count": 0, "silhouette": None}

    spike_count, sample_count = waveforms.shape
    if spike_count < config.min_spikes or sample_count < 2:
        embedding = _waveform_features(waveforms, config)
        return np.zeros(spike_count, dtype=np.int32), embedding, {"cluster_count": 1, "silhouette": None}

    features = _waveform_features(waveforms, config)
    if features.ndim != 2 or features.shape[1] == 0:
        return np.zeros(spike_count, dtype=np.int32), features, {"cluster_count": 1, "silhouette": None}
    labels, score = _cluster_features(features, config)
    labels = np.asarray(labels, dtype=np.int32)
    cluster_count = _cluster_count(labels)
    if cluster_count < 2:
        return np.zeros(spike_count, dtype=np.int32), features, {"cluster_count": 1, "silhouette": None}

    return _compact_labels(labels), features, {"cluster_count": cluster_count, "silhouette": score}


def _cluster_features(features: np.ndarray, config: WaveformClusteringConfig):
    features = _finite_features(features)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        return np.zeros(features.shape[0] if features.ndim == 2 else 0, dtype=np.int32), None
    method = config.clustering_method.lower()
    if method == "kmeans":
        return _best_partition(features, config, "kmeans")
    if method == "gmm":
        return _best_partition(features, config, "gmm")
    if method == "agglomerative":
        n_clusters = min(max(2, config.fixed_clusters), features.shape[0])
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(features)
        return labels, _score_or_none(features, labels, config)
    if method == "dbscan":
        labels = DBSCAN(eps=config.dbscan_eps, min_samples=config.dbscan_min_samples).fit_predict(features)
        return labels, _score_or_none(features, labels, config)
    raise ValueError(f"Unsupported clustering method: {config.clustering_method}")


def _best_partition(features: np.ndarray, config: WaveformClusteringConfig, method: str):
    max_clusters = min(config.max_clusters, features.shape[0])
    if max_clusters < 2:
        return np.zeros(features.shape[0], dtype=np.int32), None

    best_labels = np.zeros(features.shape[0], dtype=np.int32)
    best_score = -1.0

    for cluster_count in range(2, max_clusters + 1):
        if method == "kmeans":
            model = KMeans(n_clusters=cluster_count, n_init=10, random_state=config.random_state)
            labels = model.fit_predict(features).astype(np.int32)
        else:
            model = GaussianMixture(
                n_components=cluster_count,
                covariance_type=config.gmm_covariance_type,
                random_state=config.random_state,
            )
            labels = model.fit_predict(features).astype(np.int32)
        if _cluster_count(labels) < 2:
            continue
        score = _safe_silhouette(features, labels, config)
        if score > best_score:
            best_score = score
            best_labels = labels

    if best_score < config.min_silhouette:
        return np.zeros(features.shape[0], dtype=np.int32), None

    return best_labels, float(best_score)


def _waveform_features(waveforms: np.ndarray, config: WaveformClusteringConfig) -> np.ndarray:
    waveforms = np.nan_to_num(waveforms, nan=0.0, posinf=0.0, neginf=0.0)
    centered = waveforms - np.mean(waveforms[:, : min(8, waveforms.shape[1])], axis=1, keepdims=True)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = _finite_features(StandardScaler().fit_transform(centered))
    if not np.isfinite(scaled).all() or np.allclose(np.var(scaled, axis=0).sum(), 0.0):
        return _finite_features(scaled)
    method = config.reduction_method.lower()
    if method == "none":
        return _finite_features(scaled)
    if method == "pca":
        n_components = min(config.pca_components, scaled.shape[1], scaled.shape[0] - 1)
        if n_components < 1:
            return _finite_features(scaled)
        return _safe_pca_features(scaled, n_components)
    if method == "ica":
        n_components = min(config.ica_components, scaled.shape[1], scaled.shape[0] - 1)
        if n_components < 1:
            return _finite_features(scaled)
        try:
            return _safe_ica_features(scaled, n_components, config)
        except Exception:
            return _pca_fallback(scaled, n_components, config)
    raise ValueError(f"Unsupported reduction method: {config.reduction_method}")


def _pca_fallback(scaled: np.ndarray, n_components: int, config: WaveformClusteringConfig) -> np.ndarray:
    fallback_components = min(n_components, scaled.shape[1], scaled.shape[0] - 1)
    if fallback_components < 1:
        return _finite_features(scaled)
    return _safe_pca_features(scaled, fallback_components)


def _safe_pca_features(scaled: np.ndarray, n_components: int) -> np.ndarray:
    """PCA through a small covariance eigendecomposition.

    NEV waveforms usually have tens of samples and up to many tens of
    thousands of spikes. Solving the feature covariance matrix keeps PCA fast
    and avoids scikit-learn/SciPy SVD paths that can terminate the GUI process
    on some Windows BLAS builds.
    """

    x = _finite_features(scaled).astype(np.float64, copy=False)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] == 0:
        return _finite_features(x)
    n_components = min(int(n_components), x.shape[1], x.shape[0] - 1)
    if n_components < 1:
        return _finite_features(x)

    x = x - np.mean(x, axis=0, keepdims=True)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if np.allclose(np.var(x, axis=0).sum(), 0.0):
        return _finite_features(x)

    try:
        covariance = (x.T @ x) / max(x.shape[0] - 1, 1)
        covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
        values, vectors = np.linalg.eigh(covariance)
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return _finite_features(x)

    order = np.argsort(values)[::-1]
    order = order[:n_components]
    components = vectors[:, order]
    for index in range(components.shape[1]):
        pivot = int(np.argmax(np.abs(components[:, index])))
        if components[pivot, index] < 0:
            components[:, index] *= -1.0
    return _finite_features(x @ components)


def _safe_ica_features(scaled: np.ndarray, n_components: int, config: WaveformClusteringConfig) -> np.ndarray:
    """Numerically guarded FastICA variant using only NumPy operations.

    scikit-learn's FastICA can occasionally terminate the process through
    lower-level BLAS/LAPACK paths on ill-conditioned waveform matrices. This
    implementation keeps the same high-level fixed-point ICA idea but guards
    whitening rank, thread count, iteration count, and all non-finite values.
    """

    x = _finite_features(scaled).astype(np.float64, copy=False)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2:
        return _finite_features(x)
    x = x - np.mean(x, axis=0, keepdims=True)
    if np.allclose(np.var(x, axis=0).sum(), 0.0):
        return _finite_features(x)

    sample_count = x.shape[0]
    try:
        _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return _pca_fallback(x, n_components, config)
    if singular_values.size == 0:
        return _pca_fallback(x, n_components, config)

    tolerance = max(np.finfo(float).eps * max(x.shape) * singular_values[0], 1e-12)
    rank = int(np.count_nonzero(singular_values > tolerance))
    n_components = min(int(n_components), rank, vt.shape[0])
    if n_components < 1:
        return _pca_fallback(x, max(1, int(config.ica_components)), config)

    whitening = vt[:n_components].T / np.maximum(singular_values[:n_components], tolerance)
    x_white = np.sqrt(max(sample_count - 1, 1)) * (x @ whitening)
    x_white = np.nan_to_num(x_white, nan=0.0, posinf=0.0, neginf=0.0)

    rng = np.random.default_rng(config.random_state)
    w = rng.normal(size=(n_components, n_components))
    w = _sym_decorrelate(w)
    max_iter = min(max(int(config.ica_max_iter), 1), 1000)
    for _ in range(max_iter):
        wx = x_white @ w.T
        gx = np.tanh(np.clip(wx, -20.0, 20.0))
        g_prime = 1.0 - gx**2
        w_new = (gx.T @ x_white) / sample_count - np.mean(g_prime, axis=0)[:, None] * w
        w_new = _sym_decorrelate(w_new)
        convergence = np.max(np.abs(np.abs(np.diag(w_new @ w.T)) - 1.0))
        w = w_new
        if not np.isfinite(convergence) or convergence < 1e-5:
            break

    features = x_white @ w.T
    if not np.isfinite(features).all() or np.allclose(np.var(features, axis=0).sum(), 0.0):
        return _pca_fallback(x, n_components, config)
    return _finite_features(features)


def _sym_decorrelate(matrix: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        values, vectors = np.linalg.eigh(matrix @ matrix.T)
    except np.linalg.LinAlgError:
        q, _ = np.linalg.qr(matrix)
        return q
    values = np.maximum(values, 1e-12)
    return (vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T) @ matrix


def _finite_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        return np.zeros((0, 0), dtype=np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _score_or_none(features: np.ndarray, labels: np.ndarray, config: WaveformClusteringConfig):
    if _cluster_count(labels) < 2:
        return None
    return _safe_silhouette(features, labels, config)


def _cluster_count(labels: np.ndarray) -> int:
    unique = set(int(label) for label in np.unique(labels))
    unique.discard(-1)
    return len(unique)


def _safe_silhouette(features: np.ndarray, labels: np.ndarray, config: WaveformClusteringConfig) -> float:
    try:
        if features.shape[0] > config.max_silhouette_samples:
            rng = np.random.default_rng(config.random_state)
            indices = rng.choice(features.shape[0], size=config.max_silhouette_samples, replace=False)
            return float(silhouette_score(features[indices], labels[indices]))
        return float(silhouette_score(features, labels))
    except ValueError:
        return -1.0


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    unique = sorted(int(label) for label in np.unique(labels) if int(label) != -1)
    remap = {label: index for index, label in enumerate(unique)}
    return np.asarray([-1 if int(label) == -1 else remap[int(label)] for label in labels], dtype=np.int32)


def _summarize(channels: Dict[str, Dict[str, object]]) -> Dict[str, int]:
    sorted_channels = 0
    total_clusters = 0
    total_spikes = 0
    for payload in channels.values():
        cluster_count = int(payload.get("cluster_count", 0))
        spike_count = int(payload.get("spike_count", 0))
        if spike_count:
            sorted_channels += 1
            total_spikes += spike_count
            total_clusters += cluster_count
    return {
        "sorted_channels": sorted_channels,
        "total_clusters": total_clusters,
        "total_spikes": total_spikes,
    }


def _channel_sort_key(channel: str):
    suffix = "".join(char for char in channel if char.isdigit())
    return (channel.rstrip(suffix), int(suffix) if suffix else -1, channel)


def _emit(progress: ProgressCallback, percent: int, message: str) -> None:
    if progress:
        progress(percent, message)
