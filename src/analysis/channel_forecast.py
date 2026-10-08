"""Compact reader for channel-forecast NPZ prediction files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import re

import numpy as np
from scipy.stats import wasserstein_distance_nd


REQUIRED_KEYS = {
    "channel_rate_pred",
    "target_channel_rate_raw",
    "context_channel_rate_raw",
    "channel_xy",
}


@dataclass(frozen=True)
class ChannelForecastSummary:
    path: Path
    predicted_rate: np.ndarray  # samples x channels; future-bin mean
    observed_rate: np.ndarray  # samples x channels; future-bin mean
    observed_global_rate: np.ndarray  # samples x (history bins + future bins)
    predicted_global_rate: np.ndarray  # samples x forecast bins
    channel_xy: np.ndarray  # channels x 2
    history_bin_count: int
    forecast_bin_count: int
    sample_labels: tuple[str, ...]
    stimulation_electrodes: tuple[tuple[int, ...], ...]
    stimulus_at_forecast_start: np.ndarray
    first_stimulus_bin: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.predicted_rate.shape[0])

    @property
    def channel_count(self) -> int:
        return int(self.predicted_rate.shape[1])


def load_channel_forecast_summary(
    path: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> ChannelForecastSummary:
    """Read a forecast NPZ and retain only values needed by the viewer.

    Large 3-D tensors are reduced while loading, so sample navigation does not
    retain the multi-gigabyte source arrays in memory.
    """
    source = Path(path)
    if source.suffix.lower() != ".npz":
        raise ValueError("Channel forecast input must be an .npz file")
    if not source.is_file():
        raise FileNotFoundError(source)

    def emit(message: str) -> None:
        if progress is not None:
            progress(message)

    with np.load(source, allow_pickle=False) as archive:
        missing = sorted(REQUIRED_KEYS.difference(archive.files))
        if missing:
            raise ValueError("Not a channel-forecast NPZ; missing: " + ", ".join(missing))

        emit("Reducing predicted future response")
        predicted = np.asarray(archive["channel_rate_pred"], dtype=np.float32)
        if predicted.ndim != 3:
            raise ValueError("Predicted rates must have shape samples × bins × channels")
        sample_count, forecast_bins, channel_count = predicted.shape
        valid = np.ones((sample_count, forecast_bins), dtype=bool)
        if "target_bin_valid_mask" in archive:
            candidate = np.asarray(archive["target_bin_valid_mask"], dtype=bool)
            if candidate.shape == valid.shape:
                valid = candidate
        valid_3d = valid[:, :, None]
        denom = np.maximum(np.sum(valid, axis=1, dtype=np.float32), 1.0)[:, None]
        predicted_rate = np.sum(np.where(valid_3d, predicted, 0.0), axis=1, dtype=np.float64) / denom
        predicted_global = np.sum(np.where(valid_3d, predicted, 0.0), axis=2, dtype=np.float64)
        del predicted

        emit("Reducing observed future response")
        observed = np.asarray(archive["target_channel_rate_raw"], dtype=np.float32)
        if observed.shape != (sample_count, forecast_bins, channel_count):
            raise ValueError("Observed rates must match the predicted samples, bins, and channels")
        observed_rate = np.sum(np.where(valid_3d, observed, 0.0), axis=1, dtype=np.float64) / denom
        future_global = np.sum(np.where(valid_3d, observed, 0.0), axis=2, dtype=np.float64)
        del observed

        emit("Reducing historical global response")
        context = np.asarray(archive["context_channel_rate_raw"], dtype=np.float32)
        if context.ndim != 3 or context.shape[0] != sample_count or context.shape[2] != channel_count:
            raise ValueError("Context rate must share the forecast sample and channel axes")
        history_global = np.sum(context, axis=2, dtype=np.float64)
        del context

        emit("Reading electrode coordinates")
        coordinates = np.asarray(archive["channel_xy"], dtype=np.float32)
        if coordinates.ndim == 3:
            coordinates = coordinates[0]
        if coordinates.shape != (channel_count, 2):
            raise ValueError("channel_xy must have shape samples × channels × 2 or channels × 2")

        labels = tuple(str(index + 1) for index in range(sample_count))
        if "target_start_time" in archive:
            starts = np.asarray(archive["target_start_time"])
            if starts.shape == (sample_count,):
                labels = tuple(f"{index + 1} | {float(value):.3f} s" for index, value in enumerate(starts))

        stimulation_electrodes = tuple(() for _ in range(sample_count))
        if "target_stim_site_id" in archive:
            site_ids = np.asarray(archive["target_stim_site_id"]).astype(str)
            if site_ids.shape == (sample_count,):
                stimulation_electrodes = tuple(
                    tuple(int(value) for value in re.findall(r"\d+", site_id))
                    if site_id.strip().lower() != "none" else ()
                    for site_id in site_ids
                )
        onset = np.zeros((sample_count, forecast_bins), dtype=bool)
        if "target_stim_features" in archive:
            features = np.asarray(archive["target_stim_features"])
            if features.ndim == 3 and features.shape[:2] == onset.shape and features.shape[2] > 0:
                onset = np.asarray(features[:, :, 0] > 0, dtype=bool)
        first_stimulus_bin = np.where(np.any(onset, axis=1), np.argmax(onset, axis=1), -1).astype(np.int16)

    return ChannelForecastSummary(
        path=source,
        predicted_rate=np.asarray(predicted_rate, dtype=np.float32),
        observed_rate=np.asarray(observed_rate, dtype=np.float32),
        observed_global_rate=np.asarray(np.concatenate((history_global, future_global), axis=1), dtype=np.float32),
        predicted_global_rate=np.asarray(predicted_global, dtype=np.float32),
        channel_xy=np.asarray(coordinates, dtype=np.float32),
        history_bin_count=int(history_global.shape[1]),
        forecast_bin_count=int(forecast_bins),
        sample_labels=labels,
        stimulation_electrodes=stimulation_electrodes,
        stimulus_at_forecast_start=np.asarray(onset[:, 0], dtype=bool),
        first_stimulus_bin=first_stimulus_bin,
    )


def global_response_lines(summary: ChannelForecastSummary, sample_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time in ms, observed 100+20 response, and predicted 20 response."""
    index = min(max(int(sample_index), 0), summary.sample_count - 1)
    history = summary.history_bin_count
    future = summary.forecast_bin_count
    time_ms = np.arange(history + future, dtype=np.float32) * 5.0
    observed = summary.observed_global_rate[index]
    predicted = np.full(history + future, np.nan, dtype=np.float32)
    predicted[history:] = summary.predicted_global_rate[index]
    return time_ms, observed, predicted


def spatial_response_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    coordinates: np.ndarray,
    *,
    gaussian_sigma: float = 6.0,
    transport_cell_size: float = 20.0,
) -> dict[str, float]:
    """Compare two response maps using physical electrode coordinates.

    The correlation is computed after Gaussian spatial weighting. Wasserstein
    distance uses 20 × 20 electrode-unit bins (11 × 6 cells) to keep exact
    2-D transport practical during interactive sample browsing.
    """
    prediction = np.asarray(predicted, dtype=float).ravel()
    truth = np.asarray(observed, dtype=float).ravel()
    xy = np.asarray(coordinates, dtype=float)
    valid = np.isfinite(prediction) & np.isfinite(truth) & np.isfinite(xy).all(axis=1)
    if np.count_nonzero(valid) < 2:
        return {"weighted_correlation": float("nan"), "wasserstein_distance": float("nan"), "centroid_distance": float("nan")}
    prediction = np.maximum(prediction[valid], 0.0)
    truth = np.maximum(truth[valid], 0.0)
    xy = xy[valid]

    squared_distance = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=2)
    weights = np.exp(-squared_distance / (2.0 * max(float(gaussian_sigma), 1e-6) ** 2))
    weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)
    smooth_prediction = weights @ prediction
    smooth_truth = weights @ truth
    if np.ptp(smooth_prediction) > 0 and np.ptp(smooth_truth) > 0:
        weighted_correlation = float(np.corrcoef(smooth_prediction, smooth_truth)[0, 1])
    else:
        weighted_correlation = float("nan")

    prediction_total = float(np.sum(prediction))
    truth_total = float(np.sum(truth))
    if prediction_total <= 0 or truth_total <= 0:
        return {"weighted_correlation": weighted_correlation, "wasserstein_distance": float("nan"), "centroid_distance": float("nan")}
    prediction_center = np.sum(xy * prediction[:, None], axis=0) / prediction_total
    truth_center = np.sum(xy * truth[:, None], axis=0) / truth_total
    centroid_distance = float(np.linalg.norm(prediction_center - truth_center))

    cell_x = np.clip(np.floor(xy[:, 0] / transport_cell_size).astype(int), 0, 10)
    cell_y = np.clip(np.floor(xy[:, 1] / transport_cell_size).astype(int), 0, 5)
    prediction_cells = np.zeros((6, 11), dtype=float)
    truth_cells = np.zeros((6, 11), dtype=float)
    np.add.at(prediction_cells, (cell_y, cell_x), prediction)
    np.add.at(truth_cells, (cell_y, cell_x), truth)
    support = (prediction_cells + truth_cells) > 0
    rows, columns = np.nonzero(support)
    support_xy = np.column_stack(((columns + 0.5) * transport_cell_size, (rows + 0.5) * transport_cell_size))
    wasserstein_distance = float(wasserstein_distance_nd(
        support_xy,
        support_xy,
        prediction_cells[support],
        truth_cells[support],
    ))
    return {
        "weighted_correlation": weighted_correlation,
        "wasserstein_distance": wasserstein_distance,
        "centroid_distance": centroid_distance,
    }
