"""Maxwell HD-MEA neuronal-unit footprint analysis.

The implementation follows the recording-electrode selection strategy described
by Kobayashi et al. (Nat Commun 2024, DOI: 10.1038/s41467-024-53505-w):
activity is scanned across the HD-MEA, neuronal units are represented by four
high-amplitude electrodes, and unit centers are kept at least 100 um apart.

The current project usually has already-detected spike times and spike-aligned
waveforms rather than a full MaxLab Live activity-scan export. This module
therefore reconstructs the same analysis from the unified data structure:
average spike amplitudes are estimated per electrode, spatially separated unit
centers are selected, four-electrode unit cores are built around each center,
and coincident spike-triggered waveforms are used to form the footprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from scipy.spatial import cKDTree


ProgressCallback = Optional[Callable[[int, str], None]]


@dataclass
class MaxwellFootprintConfig:
    selection_preference: str = "amplitude"
    unit_count: int = 256
    electrodes_per_unit: int = 4
    min_spacing_um: float = 100.0
    core_radius_um: float = 70.0
    min_firing_rate_hz: float = 0.0
    min_spike_amplitude_uv: float = 0.0
    min_spikes: int = 20
    activity_scan_spikes_per_electrode: int = 200
    coincidence_window_ms: float = 2.0
    min_core_matches: int = 1
    core_waveform_corr_threshold: float = 0.45
    core_amplitude_pattern_corr_threshold: float = 0.0
    min_footprint_spikes: int = 10
    amplitude_threshold_uv: float = 8.0
    max_scan_channels: int = 0
    max_triggers: int = 2000
    neighbor_radius_um: float = 42.0
    local_corr_threshold: float = 0.15
    background_removal: bool = True
    background_removal_size: int = 2
    denoise_enabled: bool = True
    denoise_min_amplitude_uv: float = 10.0
    denoise_min_snr: float = 4.0
    denoise_max_amplitude_uv: float = 1000.0
    denoise_min_negative_to_positive: float = 0.5


def run_maxwell_footprint_analysis(
    data,
    config: MaxwellFootprintConfig,
    progress: ProgressCallback = None,
) -> Dict[str, object]:
    """Build neuronal-unit footprints from Maxwell spike events and waveforms."""

    _emit(progress, 2, "Preparing Maxwell activity map")
    channel_info = _channel_info(data, config)
    if not channel_info:
        return _empty_result(config, "No Maxwell channels with waveforms and electrode coordinates were found.")

    targets = _select_unit_centers(channel_info, config)
    if not targets:
        return _empty_result(config, "No neuronal-unit centers passed the activity thresholds.")

    coords = np.asarray([[item["x_um"], item["y_um"]] for item in channel_info], dtype=float)
    tree = cKDTree(coords)
    by_channel = {item["channel"]: item for item in channel_info}
    all_channels = [item["channel"] for item in channel_info]

    results = []
    skipped = 0
    total = max(1, len(targets))
    for index, target in enumerate(targets):
        start = 5.0 + index / total * 90.0
        end = 5.0 + (index + 1) / total * 90.0
        _emit(progress, int(start), f"Building unit footprint {index + 1}/{len(targets)}")
        core_channels = _core_electrodes(target, channel_info, tree, config)
        scan_channels = _scan_channels(target, channel_info, all_channels, config)
        unit_events = _unit_events(data, target["channel"], core_channels, config)
        if unit_events["times"].size < config.min_footprint_spikes:
            skipped += 1
            continue
        scan_info = [by_channel[channel] for channel in scan_channels if channel in by_channel]
        footprint = _extract_footprint(data, unit_events["times"], scan_info, config, progress, start, end)
        if int(footprint.get("mask_electrode_count", 0)) <= 0:
            skipped += 1
            continue
        unit = {
            "id": int(len(results)),
            "target": target,
            "selection": {
                "center_channel": target["channel"],
                "core_channels": core_channels,
                "candidate_event_count": int(unit_events.get("candidate_count", 0)),
                "event_count": int(unit_events["times"].size),
                "core_match_counts": unit_events["match_counts"],
                "core_amplitude_matrix": unit_events["amplitudes"],
                "core_waveform_corr_matrix": unit_events["waveform_corrs"],
                "core_amplitude_pattern_corrs": unit_events["pattern_corrs"],
                "method": "Kobayashi 2024 neuronal-unit footprint",
            },
            "footprint": footprint,
        }
        results.append(unit)

    summary = {
        "target_count": int(len(targets)),
        "analyzed_units": int(len(results)),
        "skipped_units": int(skipped),
        "method_note": (
            "Kobayashi 2024-style neuronal-unit analysis: select high-average-amplitude "
            "electrodes at >=100 um spacing, use waveform-consistent multi-electrode unit "
            "cores, and construct coincident spike-triggered footprints."
        ),
    }
    _emit(progress, 100, f"Footprint analysis complete: {len(results)} units")
    return {
        "method": "maxwell_neuronal_unit_footprint",
        "params": dict(config.__dict__),
        "targets": targets,
        "units": results,
        "neurons": results,
        "summary": summary,
    }


def _channel_info(data, config: MaxwellFootprintConfig) -> list[dict]:
    meta_map = data.meta.get("channel_map", {}) if isinstance(data.meta, dict) else {}
    duration = _duration_s(data)
    output = []
    for channel, raw_waveforms in data.waveforms.items():
        spikes = np.asarray(data.spikes.get(channel, []), dtype=float)
        waveforms = np.asarray(raw_waveforms, dtype=np.float32)
        if waveforms.ndim != 2 or spikes.size != waveforms.shape[0] or spikes.size == 0:
            continue
        coord = _channel_coordinates(channel, meta_map)
        if coord is None:
            continue
        valid_mask = _valid_spike_mask(waveforms, config)
        if np.count_nonzero(valid_mask) < config.min_spikes:
            continue
        filtered_waveforms = waveforms[valid_mask]
        filtered_spikes = spikes[valid_mask]
        scan_waveforms = _activity_waveforms(filtered_waveforms, config.activity_scan_spikes_per_electrode)
        amplitudes = _negative_amplitudes(scan_waveforms)
        mean_amp = float(np.nanmean(amplitudes)) if amplitudes.size else 0.0
        median_amp = float(np.nanmedian(amplitudes)) if amplitudes.size else 0.0
        template = _template(filtered_waveforms)
        output.append(
            {
                "channel": str(channel),
                "electrode": coord[2],
                "x_um": float(coord[0]),
                "y_um": float(coord[1]),
                "spike_count": int(filtered_spikes.size),
                "raw_spike_count": int(spikes.size),
                "denoised_fraction": float(filtered_spikes.size / max(1, spikes.size)),
                "firing_rate_hz": float(filtered_spikes.size / max(duration, 1e-9)),
                "mean_spike_amplitude_uv": mean_amp,
                "median_spike_amplitude_uv": median_amp,
                "template": template.astype(np.float32),
            }
        )
    return sorted(output, key=lambda item: _channel_sort_key(item["channel"]))


def _select_unit_centers(channel_info: list[dict], config: MaxwellFootprintConfig) -> list[dict]:
    candidates = [
        item
        for item in channel_info
        if item["spike_count"] >= config.min_spikes
        and item["firing_rate_hz"] >= config.min_firing_rate_hz
        and item["mean_spike_amplitude_uv"] >= config.min_spike_amplitude_uv
    ]
    if config.selection_preference == "firing_rate":
        candidates.sort(key=lambda item: (item["firing_rate_hz"], item["mean_spike_amplitude_uv"]), reverse=True)
    else:
        candidates.sort(key=lambda item: (item["mean_spike_amplitude_uv"], item["firing_rate_hz"]), reverse=True)

    selected = []
    for candidate in candidates:
        if len(selected) >= max(1, int(config.unit_count)):
            break
        if any(_distance_um(candidate, existing) < config.min_spacing_um for existing in selected):
            continue
        selected.append(dict(candidate))
    return selected


def _core_electrodes(
    target: dict,
    channel_info: list[dict],
    tree: cKDTree,
    config: MaxwellFootprintConfig,
) -> list[str]:
    coords = np.asarray([[item["x_um"], item["y_um"]] for item in channel_info], dtype=float)
    nearby_indices = tree.query_ball_point(
        np.asarray([target["x_um"], target["y_um"]], dtype=float),
        r=max(float(config.core_radius_um), 17.5),
    )
    nearby = [channel_info[int(index)] for index in nearby_indices]
    if len(nearby) < int(config.electrodes_per_unit):
        _, indices = tree.query(
            np.asarray([target["x_um"], target["y_um"]], dtype=float),
            k=min(max(1, int(config.electrodes_per_unit) * 3), len(channel_info)),
        )
        nearby = [channel_info[int(index)] for index in np.atleast_1d(indices)]
    nearby.sort(
        key=lambda item: (
            item["channel"] != target["channel"],
            -float(item.get("mean_spike_amplitude_uv", 0.0)),
            _distance_um(item, target),
        )
    )
    core = [target["channel"]]
    for item in nearby:
        channel = item["channel"]
        if channel not in core:
            core.append(channel)
        if len(core) >= max(1, int(config.electrodes_per_unit)):
            break
    return core


def _scan_channels(target: dict, channel_info: list[dict], all_channels: list[str], config: MaxwellFootprintConfig) -> list[str]:
    if not config.max_scan_channels or len(channel_info) <= int(config.max_scan_channels):
        return all_channels
    scan_info = sorted(channel_info, key=lambda item: _distance_um(item, target))[: int(config.max_scan_channels)]
    return [item["channel"] for item in scan_info]


def _unit_events(data, center_channel: str, core_channels: list[str], config: MaxwellFootprintConfig) -> dict:
    center_spikes = np.asarray(data.spikes.get(center_channel, []), dtype=float)
    center_waveforms = np.asarray(data.waveforms.get(center_channel, []), dtype=np.float32)
    if center_spikes.size == 0:
        return _empty_unit_events(len(core_channels))
    center_valid = _valid_spike_mask(center_waveforms, config)
    valid_indices = np.flatnonzero(center_valid)
    if valid_indices.size == 0:
        return _empty_unit_events(len(core_channels))
    if valid_indices.size > int(config.max_triggers) > 0:
        keep = valid_indices[np.linspace(0, valid_indices.size - 1, int(config.max_triggers), dtype=int)]
    else:
        keep = valid_indices
    candidate_times = center_spikes[keep]
    window_s = float(config.coincidence_window_ms) / 1000.0
    spikes_by_channel = {channel: np.asarray(data.spikes.get(channel, []), dtype=float) for channel in core_channels}
    waveforms_by_channel = {channel: np.asarray(data.waveforms.get(channel, []), dtype=np.float32) for channel in core_channels}
    valid_by_channel = {
        channel: _valid_spike_mask(waveforms_by_channel[channel], config)
        for channel in core_channels
    }

    candidates = []
    min_matches = max(1, int(config.min_core_matches))
    for trigger in candidate_times:
        row = []
        matches = []
        match_count = 0
        for channel in core_channels:
            spikes = spikes_by_channel[channel]
            waveforms = waveforms_by_channel[channel]
            idx = _nearest_spike_index(spikes, float(trigger), window_s)
            if idx is None or waveforms.ndim != 2 or idx >= waveforms.shape[0]:
                row.append(0.0)
                matches.append(None)
                continue
            valid_mask = valid_by_channel.get(channel)
            if valid_mask is not None and idx < valid_mask.size and not bool(valid_mask[idx]):
                row.append(0.0)
                matches.append(None)
                continue
            amplitude = _negative_amplitude(waveforms[idx])
            row.append(amplitude)
            matches.append({"channel": channel, "index": int(idx), "amplitude": float(amplitude)})
            match_count += 1
        if match_count >= min_matches:
            candidates.append(
                {
                    "time": float(trigger),
                    "matches": matches,
                    "amplitudes": row,
                    "initial_match_count": int(match_count),
                }
            )

    if not candidates:
        return _empty_unit_events(len(core_channels))

    core_templates = _core_templates_from_candidates(candidates, core_channels, waveforms_by_channel)
    median_pattern = _median_amplitude_pattern(candidates, len(core_channels))
    waveform_corr_threshold = float(config.core_waveform_corr_threshold)
    pattern_corr_threshold = float(config.core_amplitude_pattern_corr_threshold)

    event_times = []
    match_counts = []
    amplitude_rows = []
    waveform_corr_rows = []
    pattern_corrs = []
    for candidate in candidates:
        row = []
        corr_row = []
        matches = 0
        for channel, match in zip(core_channels, candidate["matches"]):
            if match is None:
                row.append(0.0)
                corr_row.append(0.0)
                continue
            waveforms = waveforms_by_channel[channel]
            idx = int(match["index"])
            template = core_templates.get(channel)
            corr = _template_correlation(waveforms[idx], template) if template is not None else 0.0
            if corr < waveform_corr_threshold:
                row.append(0.0)
                corr_row.append(float(corr))
                continue
            matches += 1
            row.append(float(match["amplitude"]))
            corr_row.append(float(corr))
        pattern_corr = _amplitude_pattern_correlation(row, median_pattern)
        if matches < min_matches:
            continue
        if pattern_corr_threshold > -1.0 and matches >= 2 and pattern_corr < pattern_corr_threshold:
            continue

        event_times.append(float(candidate["time"]))
        match_counts.append(int(matches))
        amplitude_rows.append(row if len(row) == len(core_channels) else [0.0])
        waveform_corr_rows.append(corr_row if len(corr_row) == len(core_channels) else [0.0])
        pattern_corrs.append(float(pattern_corr))

    return {
        "times": np.asarray(event_times, dtype=np.float32),
        "match_counts": match_counts,
        "amplitudes": np.asarray(amplitude_rows, dtype=np.float32),
        "waveform_corrs": np.asarray(waveform_corr_rows, dtype=np.float32),
        "pattern_corrs": np.asarray(pattern_corrs, dtype=np.float32),
        "candidate_count": int(len(candidates)),
    }


def _empty_unit_events(core_count: int = 0) -> dict:
    width = max(0, int(core_count))
    return {
        "times": np.zeros(0, dtype=np.float32),
        "match_counts": [],
        "amplitudes": np.zeros((0, width), dtype=np.float32),
        "waveform_corrs": np.zeros((0, width), dtype=np.float32),
        "pattern_corrs": np.zeros(0, dtype=np.float32),
        "candidate_count": 0,
    }


def _core_templates_from_candidates(
    candidates: list[dict],
    core_channels: list[str],
    waveforms_by_channel: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    templates = {}
    for channel_index, channel in enumerate(core_channels):
        waveforms = waveforms_by_channel.get(channel)
        if waveforms is None or waveforms.ndim != 2:
            continue
        indices = []
        for candidate in candidates:
            matches = candidate.get("matches", [])
            if channel_index >= len(matches):
                continue
            match = matches[channel_index]
            if match is None:
                continue
            idx = int(match["index"])
            if 0 <= idx < waveforms.shape[0]:
                indices.append(idx)
        if indices:
            templates[channel] = _template(waveforms[np.asarray(indices, dtype=np.int32)])
    return templates


def _median_amplitude_pattern(candidates: list[dict], width: int) -> np.ndarray:
    if not candidates or width <= 0:
        return np.zeros(max(0, int(width)), dtype=np.float32)
    rows = np.zeros((len(candidates), int(width)), dtype=np.float32)
    for index, candidate in enumerate(candidates):
        values = np.asarray(candidate.get("amplitudes", []), dtype=np.float32)
        length = min(values.size, rows.shape[1])
        if length:
            rows[index, :length] = values[:length]
    rows = np.where(rows > 0.0, rows, np.nan)
    with np.errstate(invalid="ignore"):
        pattern = np.nanmedian(rows, axis=0)
    return np.nan_to_num(pattern, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _amplitude_pattern_correlation(row, pattern: np.ndarray) -> float:
    values = np.asarray(row, dtype=float)
    reference = np.asarray(pattern, dtype=float)
    size = min(values.size, reference.size)
    if size < 2:
        return 1.0
    values = values[:size]
    reference = reference[:size]
    mask = np.isfinite(values) & np.isfinite(reference) & (values > 0.0) & (reference > 0.0)
    if np.count_nonzero(mask) < 2:
        return 1.0
    left = values[mask] - np.mean(values[mask])
    right = reference[mask] - np.mean(reference[mask])
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12 or not np.isfinite(denom):
        return 1.0
    corr = float(np.dot(left, right) / denom)
    return corr if np.isfinite(corr) else 0.0


def _extract_footprint(
    data,
    trigger_spikes: np.ndarray,
    channel_info: list[dict],
    config: MaxwellFootprintConfig,
    progress: ProgressCallback = None,
    progress_start: float = 0.0,
    progress_end: float = 100.0,
) -> dict:
    trigger_spikes = np.asarray(trigger_spikes, dtype=float)
    if trigger_spikes.size > int(config.max_triggers) > 0:
        trigger_spikes = trigger_spikes[np.linspace(0, trigger_spikes.size - 1, int(config.max_triggers), dtype=int)]

    entries = []
    total_channels = max(1, len(channel_info))
    update_step = max(1, total_channels // 40)
    for item_index, item in enumerate(channel_info):
        if item_index % update_step == 0:
            value = progress_start + (progress_end - progress_start) * item_index / total_channels
            _emit(progress, int(value), f"Extracting footprint: {item_index + 1}/{total_channels} electrodes")
        channel = item["channel"]
        spikes = np.asarray(data.spikes.get(channel, []), dtype=float)
        waveforms = np.asarray(data.waveforms.get(channel, []), dtype=np.float32)
        if spikes.size == 0 or waveforms.ndim != 2:
            continue
        indices, latencies = _matched_waveform_indices(spikes, trigger_spikes, float(config.coincidence_window_ms) / 1000.0)
        in_bounds = indices < waveforms.shape[0]
        indices = indices[in_bounds]
        latencies = latencies[in_bounds] if latencies.size == in_bounds.size else latencies[: indices.size]
        if indices.size:
            valid_mask = _valid_spike_mask(waveforms, config)
            keep = valid_mask[indices] if valid_mask.size > int(np.max(indices)) else np.zeros(indices.size, dtype=bool)
            indices = indices[keep]
            latencies = latencies[keep] if latencies.size == keep.size else latencies[: indices.size]
        if indices.size < int(config.min_footprint_spikes):
            continue
        template = _template(waveforms[indices])
        amplitude = float(np.nanmax(template) - np.nanmin(template)) if template.size else 0.0
        latency = float(np.nanmedian(latencies[: indices.size])) if latencies.size else 0.0
        entries.append(
            {
                "channel": channel,
                "electrode": item["electrode"],
                "x_um": float(item["x_um"]),
                "y_um": float(item["y_um"]),
                "spike_count": int(indices.size),
                "amplitude_uv": amplitude,
                "latency_ms": latency,
                "coincidence_fraction": float(indices.size / max(1, trigger_spikes.size)),
                "template": template.astype(np.float32),
                "mask": bool(amplitude >= config.amplitude_threshold_uv),
            }
        )

    _apply_local_template_mask(entries, config)
    if config.background_removal:
        _apply_background_removal(entries, config.background_removal_size, config.neighbor_radius_um)
    mask_count = int(sum(1 for entry in entries if entry.get("mask")))
    _emit(progress, int(progress_end), "Footprint extracted")
    return {
        "entries": entries,
        "total_spikes": int(trigger_spikes.size),
        "footprint_completeness": float(mask_count / max(1, len(channel_info))),
        "mask_electrode_count": mask_count,
    }


def _matched_waveform_indices(spikes: np.ndarray, triggers: np.ndarray, window_s: float) -> tuple[np.ndarray, np.ndarray]:
    if spikes.size == 0 or triggers.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    positions = np.searchsorted(spikes, triggers, side="left")
    left = np.clip(positions - 1, 0, spikes.size - 1)
    right = np.clip(positions, 0, spikes.size - 1)
    left_lag = spikes[left] - triggers
    right_lag = spikes[right] - triggers
    use_right = np.abs(right_lag) < np.abs(left_lag)
    indices = np.where(use_right, right, left)
    lags = np.where(use_right, right_lag, left_lag)
    valid = np.abs(lags) <= window_s
    if not np.any(valid):
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    indices = indices[valid]
    lags = lags[valid]
    unique_indices, first = np.unique(indices, return_index=True)
    return unique_indices.astype(np.int32), (lags[first] * 1000.0).astype(np.float32)


def _apply_local_template_mask(entries: list[dict], config: MaxwellFootprintConfig) -> None:
    masked = [entry for entry in entries if entry.get("mask")]
    if not masked:
        return
    for entry in masked:
        neighbors = [
            other
            for other in masked
            if other is not entry and _distance_um(entry, other) <= float(config.neighbor_radius_um)
        ]
        if not neighbors:
            continue
        best_corr = max(_template_correlation(entry.get("template"), other.get("template")) for other in neighbors)
        entry["local_template_corr"] = float(best_corr)
        if best_corr < float(config.local_corr_threshold):
            entry["mask"] = False


def _apply_background_removal(entries: list[dict], min_size: int, neighbor_radius_um: float) -> None:
    masked_indices = [index for index, entry in enumerate(entries) if entry.get("mask")]
    if not masked_indices:
        return
    index_set = set(masked_indices)
    visited = set()
    keep = set()
    for start in masked_indices:
        if start in visited:
            continue
        stack = [start]
        component = []
        visited.add(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for other in list(index_set - visited):
                if _distance_um(entries[current], entries[other]) <= float(neighbor_radius_um):
                    visited.add(other)
                    stack.append(other)
        if len(component) >= max(1, int(min_size)):
            keep.update(component)
    for index in masked_indices:
        entries[index]["mask"] = index in keep


def _nearest_spike_index(spikes: np.ndarray, time_s: float, window_s: float) -> int | None:
    if spikes.size == 0:
        return None
    center = int(np.searchsorted(spikes, time_s, side="left"))
    best = None
    best_delta = None
    for idx in (center - 1, center):
        if 0 <= idx < spikes.size:
            delta = abs(float(spikes[idx] - time_s))
            if delta <= window_s and (best_delta is None or delta < best_delta):
                best = idx
                best_delta = delta
    return best


def _template(waveforms: np.ndarray) -> np.ndarray:
    waveforms = np.asarray(waveforms, dtype=np.float32)
    if waveforms.ndim == 1:
        waveforms = waveforms.reshape(1, -1)
    if waveforms.ndim != 2 or waveforms.size == 0:
        return np.zeros(0, dtype=np.float32)
    if waveforms.shape[0] > 800:
        waveforms = waveforms[np.linspace(0, waveforms.shape[0] - 1, 800, dtype=int)]
    centered = _center_waveforms(waveforms)
    template = np.nanmean(centered, axis=0).astype(np.float32)
    return np.nan_to_num(template, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _center_waveforms(waveforms: np.ndarray) -> np.ndarray:
    waveforms = np.asarray(waveforms, dtype=np.float32)
    if waveforms.ndim != 2:
        return np.zeros((0, 0), dtype=np.float32)
    baseline_window = waveforms[:, : min(8, waveforms.shape[1])]
    baseline = np.zeros((waveforms.shape[0], 1), dtype=np.float32)
    valid_rows = np.any(np.isfinite(baseline_window), axis=1)
    if np.any(valid_rows):
        baseline[valid_rows, 0] = np.nanmedian(baseline_window[valid_rows], axis=1).astype(np.float32)
    centered = waveforms - np.nan_to_num(baseline, nan=0.0, posinf=0.0, neginf=0.0)
    return np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _activity_waveforms(waveforms: np.ndarray, limit: int) -> np.ndarray:
    waveforms = np.asarray(waveforms, dtype=np.float32)
    if waveforms.ndim != 2 or waveforms.shape[0] <= max(1, int(limit)):
        return waveforms
    indices = np.linspace(0, waveforms.shape[0] - 1, max(1, int(limit)), dtype=int)
    return waveforms[indices]


def _negative_amplitudes(waveforms: np.ndarray) -> np.ndarray:
    waveforms = _center_waveforms(np.asarray(waveforms, dtype=np.float32))
    if waveforms.ndim != 2 or waveforms.size == 0:
        return np.zeros(0, dtype=np.float32)
    return np.abs(np.nanmin(waveforms, axis=1)).astype(np.float32)


def _valid_spike_mask(waveforms: np.ndarray, config: MaxwellFootprintConfig) -> np.ndarray:
    waveforms = np.asarray(waveforms, dtype=np.float32)
    if waveforms.ndim != 2 or waveforms.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    finite = np.all(np.isfinite(waveforms), axis=1)
    if not bool(config.denoise_enabled):
        return finite

    centered = _center_waveforms(waveforms)
    negative = np.abs(np.nanmin(centered, axis=1))
    positive = np.nanmax(centered, axis=1)
    peak_to_peak = np.nanmax(centered, axis=1) - np.nanmin(centered, axis=1)
    baseline = centered[:, : min(8, centered.shape[1])]
    baseline_rms = np.sqrt(np.nanmean(baseline * baseline, axis=1))
    baseline_rms = np.nan_to_num(baseline_rms, nan=0.0, posinf=np.inf, neginf=np.inf)

    min_amplitude = max(float(config.min_spike_amplitude_uv), float(config.denoise_min_amplitude_uv))
    snr_threshold = np.maximum(min_amplitude, float(config.denoise_min_snr) * np.maximum(baseline_rms, 1e-6))
    max_amplitude = float(config.denoise_max_amplitude_uv)
    polarity_ratio = float(config.denoise_min_negative_to_positive)

    valid = finite
    valid &= np.isfinite(negative) & np.isfinite(positive) & np.isfinite(peak_to_peak)
    valid &= negative >= snr_threshold
    valid &= peak_to_peak >= min_amplitude
    if max_amplitude > 0:
        valid &= peak_to_peak <= max_amplitude
    if polarity_ratio > 0:
        valid &= negative >= np.maximum(positive, 0.0) * polarity_ratio
    return valid.astype(bool, copy=False)


def _negative_amplitude(waveform: np.ndarray) -> float:
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        values = _negative_amplitudes(waveform)
        return float(np.nanmean(values)) if values.size else 0.0
    if waveform.size == 0:
        return 0.0
    baseline_window = waveform[: min(8, waveform.size)]
    finite_baseline = baseline_window[np.isfinite(baseline_window)]
    baseline = float(np.median(finite_baseline)) if finite_baseline.size else 0.0
    centered = waveform - (0.0 if not np.isfinite(baseline) else float(baseline))
    finite = centered[np.isfinite(centered)]
    return float(abs(np.nanmin(finite))) if finite.size else 0.0


def _template_correlation(left, right) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    size = min(left.size, right.size)
    if size < 3:
        return 0.0
    left = left[:size] - np.nanmean(left[:size])
    right = right[:size] - np.nanmean(right[:size])
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12 or not np.isfinite(denom):
        return 0.0
    corr = float(np.dot(left, right) / denom)
    return corr if np.isfinite(corr) else 0.0


def _channel_coordinates(channel: str, channel_map: dict) -> tuple[float, float, int | None] | None:
    payload = channel_map.get(channel, {}) if isinstance(channel_map, dict) else {}
    if not isinstance(payload, dict):
        payload = {}
    electrode = payload.get("electrode")
    if electrode is None:
        import re

        match = re.search(r"_e(\d+)$", str(channel))
        electrode = int(match.group(1)) if match else None
    try:
        x_um = float(payload.get("x_um", payload.get("x")))
        y_um = float(payload.get("y_um", payload.get("y")))
    except (TypeError, ValueError):
        if electrode is None:
            return None
        electrode = int(electrode)
        row = electrode // 220
        col = electrode % 220
        x_um = col * 17.5
        y_um = row * 17.5
    if not np.isfinite(x_um) or not np.isfinite(y_um):
        return None
    return float(x_um), float(y_um), int(electrode) if electrode is not None else None


def _duration_s(data) -> float:
    if isinstance(data.meta, dict):
        try:
            duration = float(data.meta.get("duration_s", 0.0))
            if duration > 0:
                return duration
        except (TypeError, ValueError):
            pass
    maximum = 0.0
    for values in data.spikes.values():
        values = np.asarray(values, dtype=float)
        if values.size:
            maximum = max(maximum, float(np.nanmax(values)))
    return max(maximum, 1.0)


def _distance_um(left: dict, right: dict) -> float:
    return float(np.hypot(float(left["x_um"]) - float(right["x_um"]), float(left["y_um"]) - float(right["y_um"])))


def _channel_sort_key(channel: str):
    text = str(channel)
    suffix = "".join(char for char in text if char.isdigit())
    return (text.rstrip(suffix), int(suffix) if suffix else -1, text)


def _empty_result(config: MaxwellFootprintConfig, reason: str) -> Dict[str, object]:
    return {
        "method": "maxwell_neuronal_unit_footprint",
        "params": dict(config.__dict__),
        "targets": [],
        "units": [],
        "neurons": [],
        "summary": {"target_count": 0, "analyzed_units": 0, "reason": reason},
    }


def _emit(progress: ProgressCallback, value: int, message: str) -> None:
    if progress is not None:
        progress(max(0, min(100, int(value))), message)
