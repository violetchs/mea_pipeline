"""PySide6 GUI entry point for the MEA pipeline."""

import copy
import colorsys
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

try:
    from PySide6.QtCore import QLineF, QPointF, QObject, QRunnable, QRectF, Qt, QThreadPool, QTimer, Signal, Slot
    from PySide6.QtGui import QAction, QColor, QFont, QImage, QPalette, QPainter, QPen, QPolygonF, QRadialGradient, QWheelEvent
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QProgressDialog,
        QPushButton,
        QScrollBar,
        QSizePolicy,
        QSlider,
        QSpinBox,
        QDoubleSpinBox,
        QStackedWidget,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - exercised by manual GUI startup.
    raise SystemExit(
        "PySide6 is required for the GUI. Install dependencies with: pip install -r requirements.txt"
    ) from exc

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.colors import Normalize
from matplotlib.widgets import LassoSelector
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans as SkKMeans
from sklearn.decomposition import PCA as SkPCA
from sklearn.manifold import TSNE

try:
    from .channel_map import (
        ChannelMap,
        default_channel_map,
        electrode_id,
        list_channel_maps,
        load_channel_map,
        normalize_channel_name,
        save_channel_map,
        validate_channel_map,
    )
    from ..mea_io import (
        MEAReader,
        UnifiedMEAData,
        filter_unified_by_wells,
        list_axion_spk_wells,
        read_axion_spk,
        read_blackrock_nev,
        read_maxwell_h5,
        save_spike_train_npz,
        save_unified_npz,
    )
    from ..pipeline import MEAPipeline, PipelineConfig, PipelineResult
    from ..sorting import MaxwellFootprintConfig, WaveformClusteringConfig, cluster_nev_waveforms, run_maxwell_footprint_analysis, waveform_embedding
    from ..visualization import Visualizer
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gui.channel_map import (
        ChannelMap,
        default_channel_map,
        electrode_id,
        list_channel_maps,
        load_channel_map,
        normalize_channel_name,
        save_channel_map,
        validate_channel_map,
    )
    from mea_io import (
        MEAReader,
        UnifiedMEAData,
        filter_unified_by_wells,
        list_axion_spk_wells,
        read_axion_spk,
        read_blackrock_nev,
        read_maxwell_h5,
        save_spike_train_npz,
        save_unified_npz,
    )
    from pipeline import MEAPipeline, PipelineConfig, PipelineResult
    from sorting import MaxwellFootprintConfig, WaveformClusteringConfig, cluster_nev_waveforms, run_maxwell_footprint_analysis, waveform_embedding
    from visualization import Visualizer


def _channel_sort_key(channel: str):
    text = str(channel)
    maxwell_electrode = re.fullmatch(r"(well\d+)_e(\d+)(.*)", text, flags=re.IGNORECASE)
    if maxwell_electrode:
        return (
            maxwell_electrode.group(1).lower(),
            0,
            int(maxwell_electrode.group(2)),
            maxwell_electrode.group(3).lower(),
            text,
        )
    maxwell_channel = re.fullmatch(r"(well\d+)_ch(\d+)(.*)", text, flags=re.IGNORECASE)
    if maxwell_channel:
        return (
            maxwell_channel.group(1).lower(),
            1,
            int(maxwell_channel.group(2)),
            maxwell_channel.group(3).lower(),
            text,
        )
    suffix = "".join(char for char in text if char.isdigit())
    return (text.rstrip(suffix), int(suffix) if suffix else -1, text)


def _prefer_waveform_channel(spike_series, waveform_series) -> str:
    if not spike_series:
        return ""
    waveform_keys = set(waveform_series or {})
    for label, _times in spike_series:
        if label in waveform_keys:
            return label
    for label, times in spike_series:
        if np.asarray(times).size:
            return label
    return spike_series[0][0]


def _default_maxwell_channel_map() -> ChannelMap | None:
    base_map = load_channel_map("maxwell_map")
    if base_map is None:
        electrodes = {}
        rows = 120
        cols = 220
        pitch_um = 17.5
        for row in range(rows):
            for col in range(cols):
                electrode_int = 221 + row * cols + col
                electrode = f"e{electrode_int}"
                electrodes[electrode] = {
                    "channel": "",
                    "reference": False,
                    "electrode": electrode_int,
                    "grid_row": row,
                    "grid_col": col,
                    "x_um": (col + 1) * pitch_um,
                    "y_um": (row + 1) * pitch_um,
                    "aliases": [electrode, str(electrode_int)],
                    "routed": False,
                }
        return ChannelMap(name="maxwell_map", rows=rows, cols=cols, electrodes=electrodes)

    electrodes = copy.deepcopy(base_map.electrodes)
    for payload in electrodes.values():
        if not isinstance(payload, dict):
            continue
        payload["channel"] = ""
        payload["routed"] = False
        payload.pop("source_channel", None)
        payload.pop("recording", None)
        payload.pop("data_group", None)
    return ChannelMap(name="maxwell_map", rows=base_map.rows, cols=base_map.cols, electrodes=electrodes)


def _maxwell_channel_map_from_unified(data: UnifiedMEAData) -> ChannelMap | None:
    if not isinstance(data.meta, dict):
        return _default_maxwell_channel_map()
    raw_map = data.meta.get("channel_map")
    base_map = _default_maxwell_channel_map()
    if base_map is None:
        return None
    if not isinstance(raw_map, dict) or not raw_map:
        return base_map

    electrodes = copy.deepcopy(base_map.electrodes)

    for channel_name, payload in raw_map.items():
        if not isinstance(payload, dict):
            continue
        electrode = payload.get("electrode")
        try:
            electrode_int = int(electrode)
        except (TypeError, ValueError):
            match = re.search(r"_e(\d+)$", str(channel_name), flags=re.IGNORECASE)
            if not match:
                continue
            electrode_int = int(match.group(1))
        if electrode_int < 0:
            continue

        electrode_key = f"e{electrode_int}"
        entry = electrodes.setdefault(electrode_key, {"channel": "", "reference": False})
        entry["channel"] = str(channel_name)
        entry["routed"] = True
        for field in ("source_channel", "well", "recording", "x_um", "y_um", "x", "y"):
            if field in payload:
                entry[field] = payload[field]

        aliases = entry.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = [aliases]
        alias_candidates = [str(channel_name), f"e{electrode_int}", str(electrode_int)]
        well = payload.get("well")
        if well:
            alias_candidates.append(f"{well}_e{electrode_int}")
        source_channel = payload.get("source_channel")
        if source_channel is not None and well:
            alias_candidates.append(f"{well}_ch{source_channel}")
        seen = {str(alias) for alias in aliases}
        for alias in alias_candidates:
            if alias and alias not in seen:
                aliases.append(alias)
                seen.add(alias)
        entry["aliases"] = aliases

    return ChannelMap(name="maxwell_map", rows=base_map.rows, cols=base_map.cols, electrodes=electrodes)


def _well_sort_key(well: str):
    text = str(well)
    match = re.match(r"([A-Za-z]+)(\d+)$", text)
    if match:
        return (match.group(1).upper(), int(match.group(2)), text)
    return (text, -1, text)


def _spike_trains_from_unified(data: UnifiedMEAData):
    return [data.spikes[channel] for channel in sorted(data.channels(), key=_channel_sort_key)]


def _spike_series_from_unified(data: UnifiedMEAData):
    return [
        (channel, np.asarray(data.spikes[channel], dtype=float))
        for channel in sorted(data.channels(), key=_channel_sort_key)
    ]


def _unit_display_label(unit: int) -> str:
    return "noise" if unit == -1 else f"unit {unit}"


def _sorting_labels_for_raster(data: UnifiedMEAData, channel: str, spike_count: int):
    if not isinstance(data.sorting, dict):
        return None
    sorting = data.sorting.get(channel, {})
    if not isinstance(sorting, dict):
        return None

    labels = sorting.get("waveform_cluster_labels")
    if labels is None:
        if data.meta.get("source") == "blackrock_nev":
            return None
        labels = sorting.get("labels")
    if labels is None:
        return None

    labels = np.asarray(labels, dtype=np.int32)
    if labels.size != spike_count:
        return None
    if "waveform_cluster_labels" in sorting or np.unique(labels).size > 1:
        return labels
    return None


def _raster_series_from_unified(data: UnifiedMEAData, include_noise: bool = True):
    series = []
    has_units = False
    for channel in sorted(data.channels(), key=_channel_sort_key):
        spikes = np.asarray(data.spikes[channel], dtype=float)
        labels = _sorting_labels_for_raster(data, channel, spikes.size)
        if labels is not None:
            has_units = True
            for unit in sorted(int(value) for value in np.unique(labels)):
                if unit == -1 and not include_noise:
                    continue
                series.append((f"{channel} {_unit_display_label(unit)}", spikes[labels == unit]))
        else:
            series.append((channel, spikes))
    return series, has_units


def _raster_waveforms_from_unified(data: UnifiedMEAData, include_noise: bool = True):
    waveforms_by_row = {}
    for channel in sorted(data.channels(), key=_channel_sort_key):
        waveforms = data.waveforms.get(channel)
        if waveforms is None:
            continue

        channel_waveforms = np.asarray(waveforms)
        spikes = np.asarray(data.spikes[channel], dtype=float)
        labels = _sorting_labels_for_raster(data, channel, spikes.size)
        if labels is not None and channel_waveforms.shape[0] == spikes.size:
            for unit in sorted(int(value) for value in np.unique(labels)):
                if unit == -1 and not include_noise:
                    continue
                row_label = f"{channel} {_unit_display_label(unit)}"
                waveforms_by_row[row_label] = channel_waveforms[labels == unit]
        else:
            waveforms_by_row[channel] = channel_waveforms
    return waveforms_by_row


def _format_time_tick(value: float) -> str:
    return f"{value:.3f}"


_HEATMAP_COLOR_STOPS = [
    (0.0, "#000000"),
    (0.18, "#02030b"),
    (0.34, "#07127a"),
    (0.46, "#005cff"),
    (0.50, "#22c55e"),
    (0.62, "#00d3ff"),
    (0.72, "#00e68a"),
    (0.92, "#d7f51f"),
    (1.0, "#dc2626"),
]


def _activity_heatmap_color(intensity: float) -> QColor:
    value = max(0.0, min(1.0, float(intensity)))
    for (left_pos, left_color), (right_pos, right_color) in zip(_HEATMAP_COLOR_STOPS, _HEATMAP_COLOR_STOPS[1:]):
        if value <= right_pos:
            span = max(right_pos - left_pos, 1e-12)
            fraction = (value - left_pos) / span
            start = QColor(left_color)
            stop = QColor(right_color)
            red = int(round(start.red() + (stop.red() - start.red()) * fraction))
            green = int(round(start.green() + (stop.green() - start.green()) * fraction))
            blue = int(round(start.blue() + (stop.blue() - start.blue()) * fraction))
            return QColor(red, green, blue)
    return QColor(_HEATMAP_COLOR_STOPS[-1][1])


def _available_channels_for_data(data, data_kind: str):
    if isinstance(data, UnifiedMEAData):
        channels = {str(channel) for channel in data.channels()}
        if data.meta.get("source") == "blackrock_nev":
            labels = data.meta.get("electrode_labels", {})
            waveform_headers = data.meta.get("waveform_headers", {})
            for raw_id in set(labels) | set(waveform_headers):
                try:
                    electrode_number = int(raw_id)
                except (TypeError, ValueError):
                    continue
                if 1 <= electrode_number <= 64:
                    channels.add(f"chan{electrode_number}")
                    label = labels.get(raw_id)
                    if label:
                        channels.add(str(label))
        if data.meta.get("source") == "axion_spk":
            for row in range(1, 9):
                for col in range(1, 9):
                    channels.add(f"r{row}c{col}")
        return sorted(channels, key=_channel_sort_key)
    if data_kind == "array" and data is not None:
        array = np.asarray(data)
        if array.ndim == 1:
            return ["chan1"]
        return [f"chan{index + 1}" for index in range(array.shape[0])]
    return []


def _display_indices(length: int, max_count: int):
    if length <= max_count:
        return np.arange(length)
    return np.linspace(0, length - 1, max_count, dtype=int)


def _normalized_channel_keys(channels) -> set[str]:
    keys = set()
    for channel in channels or []:
        text = str(channel or "").strip()
        if not text:
            continue
        keys.add(normalize_channel_name(text))
        if "_" in text:
            keys.add(normalize_channel_name(text.split("_", 1)[1]))
    return {key for key in keys if key}


def _waveform_time_axis(sample_count: int, sampling_rate):
    if sampling_rate:
        return np.arange(sample_count, dtype=float) / float(sampling_rate) * 1000.0
    return np.arange(sample_count, dtype=float)


def _cluster_color_map(labels):
    unique = sorted(int(label) for label in np.unique(labels))
    colors = {}
    color_index = 0
    for label in unique:
        if label == -1:
            colors[label] = "#94a3b8"
        else:
            colors[label] = f"C{color_index % 10}"
            color_index += 1
    return colors


def _base_channel_from_raster_label(label: str) -> str:
    text = str(label)
    for marker in (" unit ", " noise"):
        if marker in text:
            return text.split(marker, 1)[0]
    return text


def _detect_burst_intervals(
    spike_series,
    bin_ms: float = 10.0,
    smooth_ms: float = 50.0,
    threshold_z: float = 4.0,
    min_duration_ms: float = 30.0,
    merge_gap_ms: float = 30.0,
    min_spikes: int = 5,
    cancel_check=None,
):
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Burst detection cancelled")
    all_spikes = [
        np.asarray(times, dtype=float)
        for _, times in spike_series
        if np.asarray(times, dtype=float).size
    ]
    if not all_spikes:
        return []

    spikes = np.sort(np.concatenate(all_spikes))
    if spikes.size < 3:
        return []
    start = float(spikes.min())
    stop = float(spikes.max())
    bin_s = max(0.001, float(bin_ms) / 1000.0)
    if stop <= start + bin_s:
        return []

    edges = np.arange(start, stop + bin_s, bin_s)
    if edges.size < 3:
        return []
    counts, _ = np.histogram(spikes, bins=edges)
    if counts.size == 0 or counts.max() <= 0:
        return []

    smooth_bins = max(1, int(round(float(smooth_ms) / float(bin_ms))))
    if smooth_bins > 1:
        kernel = np.ones(smooth_bins, dtype=float) / float(smooth_bins)
        rate = np.convolve(counts.astype(float), kernel, mode="same")
    else:
        rate = counts.astype(float)

    quiet_cutoff = float(np.percentile(rate, 70))
    quiet_rate = rate[rate <= quiet_cutoff]
    if quiet_rate.size < max(3, min(10, rate.size)):
        quiet_rate = rate

    baseline = float(np.median(quiet_rate))
    mad = float(np.median(np.abs(quiet_rate - baseline)))
    robust_spread = 1.4826 * mad
    quiet_spread = float(np.std(quiet_rate))
    poisson_spread = float(np.sqrt(max(baseline, 0.0)))
    spread = max(robust_spread, quiet_spread, poisson_spread * 0.5)
    positive_rate = rate[rate > 0]
    if spread > 1e-9:
        high_threshold = baseline + float(threshold_z) * spread
    elif positive_rate.size:
        percentile = min(99.0, max(50.0, 50.0 + float(threshold_z) * 10.0))
        high_threshold = float(np.percentile(positive_rate, percentile))
    else:
        return []

    if not np.isfinite(high_threshold):
        return []
    high_threshold = max(1.0, float(high_threshold))
    low_threshold = max(0.25, baseline + 0.35 * (high_threshold - baseline), high_threshold * 0.25)
    low_threshold = min(low_threshold, high_threshold)
    high_active = rate >= high_threshold
    low_active = rate >= low_threshold
    min_bins = max(1, int(np.ceil(float(min_duration_ms) / float(bin_ms))))
    intervals = []
    index = 0
    while index < high_active.size:
        if index % 4096 == 0 and cancel_check is not None and cancel_check():
            raise InterruptedError("Burst detection cancelled")
        if not high_active[index]:
            index += 1
            continue
        run_start = index
        while run_start > 0 and low_active[run_start - 1]:
            run_start -= 1
        run_stop = index + 1
        while run_stop < low_active.size and low_active[run_stop]:
            run_stop += 1
        index = run_stop
        if run_stop - run_start >= min_bins:
            start_s = float(edges[run_start])
            stop_s = float(edges[min(run_stop, edges.size - 1)])
            spike_count = int(np.count_nonzero((spikes >= start_s) & (spikes <= stop_s)))
            if spike_count >= int(min_spikes):
                intervals.append((start_s, stop_s))

    if not intervals:
        density_intervals = _density_burst_intervals(
            spikes,
            min_duration_s=max(bin_s, float(min_duration_ms) / 1000.0),
            min_spikes=int(min_spikes),
            merge_gap_s=max(0.0, float(merge_gap_ms) / 1000.0),
            cancel_check=cancel_check,
        )
        intervals.extend(density_intervals)
    if not intervals:
        return []

    merge_gap_s = max(0.0, float(merge_gap_ms) / 1000.0)
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for index, (start_s, stop_s) in enumerate(intervals[1:]):
        if index % 4096 == 0 and cancel_check is not None and cancel_check():
            raise InterruptedError("Burst detection cancelled")
        prev_start, prev_stop = merged[-1]
        if start_s - prev_stop <= merge_gap_s:
            merged[-1] = (prev_start, max(prev_stop, stop_s))
        else:
            merged.append((start_s, stop_s))
    return merged


def _density_burst_intervals(spikes: np.ndarray, min_duration_s: float, min_spikes: int, merge_gap_s: float, cancel_check=None):
    if spikes.size < min_spikes or min_spikes <= 1:
        return []
    intervals = []
    stop_index = 0
    for start_index, start_s in enumerate(spikes):
        if start_index % 4096 == 0 and cancel_check is not None and cancel_check():
            raise InterruptedError("Burst detection cancelled")
        if stop_index < start_index:
            stop_index = start_index
        while stop_index < spikes.size and spikes[stop_index] <= start_s + min_duration_s:
            stop_index += 1
        if stop_index - start_index >= min_spikes:
            intervals.append((float(start_s), float(spikes[stop_index - 1])))
    if not intervals:
        return []
    merged = [intervals[0]]
    for index, (start_s, stop_s) in enumerate(intervals[1:]):
        if index % 4096 == 0 and cancel_check is not None and cancel_check():
            raise InterruptedError("Burst detection cancelled")
        prev_start, prev_stop = merged[-1]
        if start_s - prev_stop <= merge_gap_s:
            merged[-1] = (prev_start, max(prev_stop, stop_s))
        else:
            merged.append((start_s, stop_s))
    return merged


def _burst_sequence_payload(spike_series, burst_intervals):
    labels = np.asarray([label for label, _ in spike_series], dtype=object)
    intervals = np.asarray(
        [(float(start), float(stop)) for start, stop in burst_intervals if float(stop) > float(start)],
        dtype=float,
    ).reshape(-1, 2)
    relative_spike_times = np.empty((intervals.shape[0], labels.size), dtype=object)

    for burst_index, (start_s, stop_s) in enumerate(intervals):
        for row_index, (_, times) in enumerate(spike_series):
            spike_times = np.asarray(times, dtype=float)
            lo = int(np.searchsorted(spike_times, start_s, side="left"))
            hi = int(np.searchsorted(spike_times, stop_s, side="right"))
            relative_spike_times[burst_index, row_index] = (spike_times[lo:hi] - start_s).astype(np.float64)

    return {
        "format": "mea_pipeline_burst_sequences_v1",
        "time_unit": "seconds",
        "labels": labels,
        "burst_intervals_s": intervals,
        "relative_spike_times_s": relative_spike_times,
        "burst_count": int(intervals.shape[0]),
        "row_count": int(labels.size),
    }


def _stim_tail_keep_mask(times: np.ndarray, stim_times: np.ndarray, window_s: float) -> np.ndarray:
    values = np.asarray(times, dtype=float)
    events = np.asarray(stim_times, dtype=float)
    events = events[np.isfinite(events)]
    if values.size == 0 or events.size == 0 or window_s <= 0:
        return np.ones(values.shape, dtype=bool)
    events.sort()
    indices = np.searchsorted(events, values)
    artifact = np.zeros(values.shape, dtype=bool)
    tolerance = window_s + 1e-12
    right = indices < events.size
    if np.any(right):
        artifact[right] |= np.abs(events[indices[right]] - values[right]) <= tolerance
    left = indices > 0
    if np.any(left):
        artifact[left] |= np.abs(values[left] - events[indices[left] - 1]) <= tolerance
    return ~artifact


def _filter_spike_series_stim_tail(spike_series, stim_times, window_ms: float):
    stim_values = np.asarray(stim_times if stim_times is not None else [], dtype=float)
    window_s = max(0.0, float(window_ms) / 1000.0)
    filtered = []
    masks = {}
    removed = 0
    for label, times in spike_series:
        values = np.asarray(times, dtype=float)
        keep = _stim_tail_keep_mask(values, stim_values, window_s)
        masks[str(label)] = keep
        removed += int(values.size - np.count_nonzero(keep))
        filtered.append((label, values[keep]))
    return filtered, masks, removed


def _burst_delay_channel_series(spike_series, max_channels: int | None = None):
    channel_chunks: dict[str, list[np.ndarray]] = {}
    for label, times in spike_series:
        text = str(label)
        if " noise" in text.lower():
            continue
        channel = _base_channel_from_raster_label(text)
        values = np.asarray(times, dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            channel_chunks.setdefault(channel, []).append(values)

    channel_series = []
    for channel, chunks in channel_chunks.items():
        merged = np.unique(np.concatenate(chunks)) if len(chunks) > 1 else np.unique(chunks[0])
        if merged.size:
            channel_series.append((channel, merged))
    channel_series.sort(key=lambda item: (-item[1].size, _channel_sort_key(item[0])))
    if max_channels is not None and int(max_channels) > 0:
        channel_series = channel_series[: max(2, int(max_channels))]
    return [channel for channel, _ in channel_series], [times for _, times in channel_series]


def _burst_delay_limited_intervals(burst_intervals, burst_window_ms: float = 0.0):
    window_s = max(0.0, float(burst_window_ms)) / 1000.0
    intervals = []
    for start, stop in burst_intervals:
        start_s = float(start)
        stop_s = float(stop)
        if window_s > 0:
            stop_s = min(stop_s, start_s + window_s)
        if stop_s > start_s:
            intervals.append((start_s, stop_s))
    return intervals


def _burst_delay_first_spike_matrix(spike_series, burst_intervals, max_channels: int | None = None, burst_window_ms: float = 0.0):
    channels, channel_trains = _burst_delay_channel_series(spike_series, max_channels)
    channel_series = list(zip(channels, channel_trains))

    intervals = _burst_delay_limited_intervals(burst_intervals, burst_window_ms)
    first_times = np.full((len(intervals), len(channel_series)), np.nan, dtype=np.float64)
    for burst_index, (start_s, stop_s) in enumerate(intervals):
        for channel_index, (_, times) in enumerate(channel_series):
            lo = int(np.searchsorted(times, start_s, side="left"))
            if lo < times.size and times[lo] <= stop_s:
                first_times[burst_index, channel_index] = float(times[lo] - start_s)
    return [channel for channel, _ in channel_series], intervals, first_times


def _source_interval_delay_matches(
    source_times: np.ndarray,
    target_times: np.ndarray,
    max_abs_delay_ms: float,
    min_abs_delay_ms: float = 0.0,
    intervals: list[tuple[float, float]] | None = None,
):
    source_times = np.asarray(source_times, dtype=float)
    target_times = np.asarray(target_times, dtype=float)
    source_times = np.sort(source_times[np.isfinite(source_times)])
    target_times = np.sort(target_times[np.isfinite(target_times)])
    if source_times.size == 0 or target_times.size == 0 or (intervals is None and source_times.size < 2):
        return np.zeros((0, 3), dtype=float)

    source_starts, source_nexts, row_ids = _source_interval_candidates(source_times, intervals)
    return _source_interval_delay_matches_from_candidates(
        source_starts,
        source_nexts,
        row_ids,
        target_times,
        max_abs_delay_ms,
        min_abs_delay_ms,
    )


def _source_interval_candidates(
    source_times: np.ndarray,
    intervals: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_times = np.asarray(source_times, dtype=float)
    source_times = np.sort(source_times[np.isfinite(source_times)])
    if source_times.size < 2:
        if intervals is None or source_times.size == 0:
            empty = np.array([], dtype=float)
            return empty, empty, empty.astype(int)

    if intervals is None:
        return source_times[:-1], source_times[1:], np.zeros(source_times.size - 1, dtype=int)

    start_chunks = []
    next_chunks = []
    row_chunks = []
    for row_index, (start_s, stop_s) in enumerate(intervals):
        start_s = float(start_s)
        stop_s = float(stop_s)
        if stop_s <= start_s:
            continue
        lo = int(np.searchsorted(source_times, start_s, side="left"))
        hi = int(np.searchsorted(source_times, stop_s, side="right"))
        local_source = source_times[lo:hi]
        if local_source.size == 0:
            continue
        next_sources = np.empty(local_source.size, dtype=float)
        if local_source.size > 1:
            next_sources[:-1] = local_source[1:]
        next_sources[-1] = stop_s
        start_chunks.append(local_source)
        next_chunks.append(next_sources)
        row_chunks.append(np.full(local_source.size, int(row_index), dtype=int))

    if not start_chunks:
        empty = np.array([], dtype=float)
        return empty, empty, empty.astype(int)
    return np.concatenate(start_chunks), np.concatenate(next_chunks), np.concatenate(row_chunks)


def _source_interval_delay_matches_from_candidates(
    source_starts: np.ndarray,
    source_nexts: np.ndarray,
    row_ids: np.ndarray,
    target_times: np.ndarray,
    max_abs_delay_ms: float,
    min_abs_delay_ms: float = 0.0,
) -> np.ndarray:
    source_starts = np.asarray(source_starts, dtype=float)
    source_nexts = np.asarray(source_nexts, dtype=float)
    row_ids = np.asarray(row_ids, dtype=int)
    target_times = np.asarray(target_times, dtype=float)
    target_times = np.sort(target_times[np.isfinite(target_times)])
    if source_starts.size == 0 or target_times.size == 0:
        return np.zeros((0, 3), dtype=float)

    max_delay_s = max(0.0, float(max_abs_delay_ms)) / 1000.0
    min_delay_s = max(0.0, float(min_abs_delay_ms)) / 1000.0
    target_positions = np.searchsorted(target_times, source_starts, side="right")
    valid = target_positions < target_times.size
    if not np.any(valid):
        return np.zeros((0, 3), dtype=float)

    source_candidates = source_starts[valid]
    next_sources = source_nexts[valid]
    row_candidates = row_ids[valid]
    target_candidates = target_times[target_positions[valid]]
    delays_s = target_candidates - source_candidates
    valid_delays = (target_candidates < next_sources) & (delays_s >= min_delay_s)
    if max_delay_s > 0:
        valid_delays &= delays_s <= max_delay_s
    if not np.any(valid_delays):
        return np.zeros((0, 3), dtype=float)
    return np.column_stack(
        [
            source_candidates[valid_delays],
            target_candidates[valid_delays],
            row_candidates[valid_delays].astype(float, copy=False),
        ]
    ).astype(float, copy=False)


def _source_interval_delay_values(
    source_times: np.ndarray,
    target_times: np.ndarray,
    max_abs_delay_ms: float,
    min_abs_delay_ms: float = 0.0,
    intervals: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    matches = _source_interval_delay_matches(source_times, target_times, max_abs_delay_ms, min_abs_delay_ms, intervals)
    if matches.size == 0:
        return np.array([], dtype=float)
    return ((matches[:, 1] - matches[:, 0]) * 1000.0).astype(float, copy=False)


def _burst_delay_pair_values(
    first_times: np.ndarray,
    reference_index: int,
    target_index: int,
    max_abs_delay_ms: float,
    min_abs_delay_ms: float = 0.0,
):
    if first_times.ndim != 2 or first_times.shape[0] == 0:
        return np.array([], dtype=float)
    delays_ms = (first_times[:, target_index] - first_times[:, reference_index]) * 1000.0
    delays_ms = delays_ms[np.isfinite(delays_ms)]
    min_delay = max(0.0, float(min_abs_delay_ms))
    if min_delay > 0:
        delays_ms = delays_ms[np.abs(delays_ms) >= min_delay]
    if max_abs_delay_ms > 0:
        delays_ms = delays_ms[np.abs(delays_ms) <= float(max_abs_delay_ms)]
    return delays_ms.astype(float, copy=False)


BURST_DELAY_TABLE_ROW_LIMIT = 1000
BURST_DELAY_COMBO_CHANNEL_LIMIT = 1200
BURST_DELAY_DEFAULT_CHANNEL_LIMIT = 256
BURST_DELAY_MAX_ACTIVE_PER_BURST = 384


def _burst_delay_all_pair_values(
    first_times: np.ndarray,
    max_abs_delay_ms: float,
    min_abs_delay_ms: float = 0.0,
    max_values: int = 250000,
):
    if first_times.ndim != 2 or first_times.shape[1] < 2:
        return np.array([], dtype=float)
    max_lag_s = max(1.0, float(max_abs_delay_ms)) / 1000.0
    min_lag_s = max(0.0, float(min_abs_delay_ms)) / 1000.0
    max_values = max(1, int(max_values))
    chunks = []
    collected = 0
    participation = np.count_nonzero(np.isfinite(first_times), axis=0)
    for burst_times in np.asarray(first_times, dtype=float):
        active = np.flatnonzero(np.isfinite(burst_times))
        if active.size < 2:
            continue
        if active.size > BURST_DELAY_MAX_ACTIVE_PER_BURST:
            order = np.argsort(participation[active], kind="mergesort")[::-1]
            active = active[order[:BURST_DELAY_MAX_ACTIVE_PER_BURST]]
        order = np.argsort(burst_times[active], kind="mergesort")
        sorted_times = burst_times[active][order]
        right_edge = 1
        for left_pos, left_time in enumerate(sorted_times[:-1]):
            if collected >= max_values:
                break
            right_edge = max(right_edge, left_pos + 1)
            while right_edge < sorted_times.size and sorted_times[right_edge] - left_time <= max_lag_s:
                right_edge += 1
            if right_edge <= left_pos + 1:
                continue
            delays_ms = (sorted_times[left_pos + 1 : right_edge] - left_time) * 1000.0
            if min_lag_s > 0:
                delays_ms = delays_ms[delays_ms >= min_lag_s * 1000.0]
                if delays_ms.size == 0:
                    continue
            remaining = max_values - collected
            if delays_ms.size > remaining:
                delays_ms = delays_ms[_display_indices(delays_ms.size, remaining)]
            chunks.append(delays_ms.astype(float, copy=False))
            collected += int(delays_ms.size)
        if collected >= max_values:
            break
    if not chunks:
        return np.array([], dtype=float)
    values = np.concatenate(chunks)
    if values.size > max_values:
        values = values[_display_indices(values.size, max_values)]
    return values.astype(float, copy=False)


def _burst_delay_aligned_pairs(
    channels,
    first_times: np.ndarray,
    max_abs_delay_ms: float,
    min_abs_delay_ms: float,
    bin_ms: float,
    min_peak_count: int,
    min_peak_fraction: float,
    min_peak_to_background: float,
    cancel_check=None,
    progress_callback=None,
):
    if first_times.ndim != 2 or first_times.shape[1] < 2 or not channels:
        return []
    max_lag = max(1.0, float(max_abs_delay_ms))
    min_lag = max(0.0, float(min_abs_delay_ms))
    bin_width = max(0.1, float(bin_ms))
    channel_count = min(len(channels), first_times.shape[1])
    if channel_count < 2:
        return []

    participation = np.count_nonzero(np.isfinite(first_times[:, :channel_count]), axis=0)
    eligible = np.flatnonzero(participation >= max(1, int(min_peak_count)))
    if eligible.size < 2:
        return []

    if eligible.size < channel_count:
        local_times = first_times[:, eligible]
        local_to_original = eligible.astype(int, copy=False)
        local_participation = participation[eligible]
    else:
        local_times = first_times[:, :channel_count]
        local_to_original = np.arange(channel_count, dtype=int)
        local_participation = participation[:channel_count]

    max_lag_s = max_lag / 1000.0
    bin_count = max(1, int(np.floor((2.0 * max_lag) / bin_width)) + 1)
    pair_total: dict[tuple[int, int], int] = {}
    pair_bins: dict[tuple[int, int], dict[int, int]] = {}
    pair_bin_sums: dict[tuple[tuple[int, int], int], float] = {}
    pair_bin_sumsq: dict[tuple[tuple[int, int], int], float] = {}
    truncated_bursts = 0

    for burst_index, burst_times in enumerate(np.asarray(local_times, dtype=float)):
        if burst_index % 4 == 0:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Burst delay analysis cancelled")
            if progress_callback is not None:
                progress = 35 + int(55 * burst_index / max(1, local_times.shape[0]))
                progress_callback(min(94, progress), "Scanning burst-local delay candidates...")
        active_local = np.flatnonzero(np.isfinite(burst_times))
        if active_local.size < 2:
            continue
        if active_local.size > BURST_DELAY_MAX_ACTIVE_PER_BURST:
            order = np.argsort(local_participation[active_local], kind="mergesort")[::-1]
            active_local = active_local[order[:BURST_DELAY_MAX_ACTIVE_PER_BURST]]
            truncated_bursts += 1
        order = np.argsort(burst_times[active_local], kind="mergesort")
        active_local = active_local[order]
        sorted_times = burst_times[active_local]
        right_edge = 1
        for left_pos, left_time in enumerate(sorted_times[:-1]):
            right_edge = max(right_edge, left_pos + 1)
            while right_edge < sorted_times.size and sorted_times[right_edge] - left_time <= max_lag_s:
                right_edge += 1
            if right_edge <= left_pos + 1:
                continue
            earlier_original = int(local_to_original[active_local[left_pos]])
            later_originals = local_to_original[active_local[left_pos + 1 : right_edge]]
            delays_ms = (sorted_times[left_pos + 1 : right_edge] - left_time) * 1000.0
            for later_original, delay_ms in zip(later_originals, delays_ms):
                later_original = int(later_original)
                if earlier_original == later_original:
                    continue
                low = min(earlier_original, later_original)
                high = max(earlier_original, later_original)
                signed_delay = float(delay_ms if earlier_original == low else -delay_ms)
                if not np.isfinite(signed_delay) or abs(signed_delay) < min_lag or abs(signed_delay) > max_lag:
                    continue
                bin_index = int(np.floor((signed_delay + max_lag) / bin_width + 0.5))
                bin_index = max(0, min(bin_count - 1, bin_index))
                pair_key = (low, high)
                pair_total[pair_key] = pair_total.get(pair_key, 0) + 1
                counts = pair_bins.setdefault(pair_key, {})
                counts[bin_index] = counts.get(bin_index, 0) + 1
                stats_key = (pair_key, bin_index)
                pair_bin_sums[stats_key] = pair_bin_sums.get(stats_key, 0.0) + signed_delay
                pair_bin_sumsq[stats_key] = pair_bin_sumsq.get(stats_key, 0.0) + signed_delay * signed_delay

    if cancel_check is not None and cancel_check():
        raise InterruptedError("Burst delay analysis cancelled")

    results = []
    for pair_key, counts_by_bin in pair_bins.items():
        if not counts_by_bin:
            continue
        peak_index, peak_count = max(counts_by_bin.items(), key=lambda item: (item[1], -abs(item[0])))
        peak_count = int(peak_count)
        if peak_count < int(min_peak_count):
            continue
        total_count = int(pair_total.get(pair_key, peak_count))
        non_peak_counts = [count for index, count in counts_by_bin.items() if index != peak_index]
        zero_bins = max(0, bin_count - 1 - len(non_peak_counts))
        if zero_bins:
            non_peak_counts.extend([0] * min(zero_bins, 32))
        background = float(np.median(non_peak_counts)) if non_peak_counts else 0.0
        background = max(background, 0.5)
        peak_fraction = float(peak_count) / float(max(1, total_count))
        peak_to_background = float(peak_count) / background
        if peak_fraction < float(min_peak_fraction):
            continue
        if peak_to_background < float(min_peak_to_background):
            continue
        delay_window_indices = range(max(0, peak_index - 2), min(bin_count, peak_index + 3))
        delay_window_count = int(sum(counts_by_bin.get(index, 0) for index in delay_window_indices))
        delay_window_sum = float(
            sum(pair_bin_sums.get((pair_key, index), 0.0) for index in delay_window_indices)
        )
        signed_delay = delay_window_sum / float(max(1, delay_window_count))
        if signed_delay == 0.0:
            signed_delay = -max_lag + float(peak_index) * bin_width
        if signed_delay <= 0:
            output_reference, output_target = pair_key[1], pair_key[0]
            output_delay_ms = -signed_delay
        else:
            output_reference, output_target = pair_key[0], pair_key[1]
            output_delay_ms = signed_delay
        if output_delay_ms <= 0:
            continue
        sumsq = float(sum(pair_bin_sumsq.get((pair_key, index), 0.0) for index in delay_window_indices))
        variance = max(0.0, sumsq / float(max(1, delay_window_count)) - signed_delay * signed_delay)
        results.append(
            {
                "reference_index": int(output_reference),
                "target_index": int(output_target),
                "reference": str(channels[output_reference]),
                "target": str(channels[output_target]),
                "delay_ms": float(output_delay_ms),
                "peak_center_ms": abs(float(-max_lag + float(peak_index) * bin_width)),
                "peak_count": peak_count,
                "delay_window_count": delay_window_count,
                "total_count": total_count,
                "peak_fraction": peak_fraction,
                "peak_to_background": peak_to_background,
                "background_count": background,
                "std_ms": float(np.sqrt(variance)),
                "truncated_bursts": int(truncated_bursts),
            }
        )
    return sorted(
        results,
        key=lambda item: (
            -float(item["peak_count"]),
            -float(item["peak_fraction"]),
            -float(item["peak_to_background"]),
            abs(float(item["delay_ms"])),
            item["reference"],
            item["target"],
        ),
    )


def _spike_train_delay_aligned_pairs(
    channels,
    channel_trains,
    intervals: list[tuple[float, float]] | None,
    max_abs_delay_ms: float,
    min_abs_delay_ms: float,
    bin_ms: float,
    min_peak_count: int,
    min_peak_fraction: float,
    min_peak_to_background: float,
    cancel_check=None,
    progress_callback=None,
    mode: str = "burst_all",
):
    if len(channels) < 2 or len(channel_trains) < 2:
        return []
    max_lag = max(1.0, float(max_abs_delay_ms))
    min_lag = max(0.0, float(min_abs_delay_ms))
    bin_width = max(0.1, float(bin_ms))
    bin_count = max(1, int(np.floor(max_lag / bin_width)) + 1)
    train_count = min(len(channels), len(channel_trains))
    results = []
    train_lengths = np.asarray([np.asarray(channel_trains[index]).size for index in range(train_count)], dtype=int)
    eligible_indices = np.flatnonzero(train_lengths >= max(2, int(min_peak_count)))
    if eligible_indices.size < 2:
        return []
    total_pairs = max(1, int(eligible_indices.size) * max(0, int(eligible_indices.size) - 1))
    scanned = 0
    source_candidates_by_index = {
        int(index): _source_interval_candidates(
            channel_trains[int(index)],
            intervals,
        )
        for index in eligible_indices
    }

    for reference_index in eligible_indices:
        reference_index = int(reference_index)
        source_starts, source_nexts, row_ids = source_candidates_by_index[reference_index]
        if source_starts.size == 0:
            scanned += max(0, int(eligible_indices.size) - 1)
            continue
        for target_index in eligible_indices:
            target_index = int(target_index)
            if reference_index == target_index:
                continue
            scanned += 1
            if scanned % 64 == 0:
                if cancel_check is not None and cancel_check():
                    raise InterruptedError("Burst delay analysis cancelled")
                if progress_callback is not None:
                    progress = 35 + int(55 * scanned / total_pairs)
                    progress_callback(min(94, progress), "Scanning source-target spike train delays...")
            matches = _source_interval_delay_matches_from_candidates(
                source_starts,
                source_nexts,
                row_ids,
                channel_trains[target_index],
                max_lag,
                min_lag,
            )
            if matches.size == 0:
                continue
            values = ((matches[:, 1] - matches[:, 0]) * 1000.0).astype(float, copy=False)
            if values.size == 0:
                continue
            bin_indices = np.floor(values / bin_width + 0.5).astype(int)
            bin_indices = np.clip(bin_indices, 0, bin_count - 1)
            counts = np.bincount(bin_indices, minlength=bin_count)
            peak_index = int(np.argmax(counts))
            peak_count = int(counts[peak_index])
            if peak_count < int(min_peak_count):
                continue
            total_count = int(values.size)
            non_peak_counts = [int(count) for index, count in enumerate(counts) if index != peak_index]
            background = float(np.median(non_peak_counts)) if non_peak_counts else 0.0
            background = max(background, 0.5)
            peak_fraction = float(peak_count) / float(max(1, total_count))
            peak_to_background = float(peak_count) / background
            if peak_fraction < float(min_peak_fraction):
                continue
            if peak_to_background < float(min_peak_to_background):
                continue
            delay_window_indices = range(max(0, peak_index - 2), min(bin_count, peak_index + 3))
            window_mask = np.isin(bin_indices, list(delay_window_indices))
            window_values = values[window_mask]
            delay_ms = float(np.mean(window_values)) if window_values.size else float(peak_index * bin_width)
            if delay_ms <= 0:
                continue
            results.append(
                {
                    "reference_index": int(reference_index),
                    "target_index": int(target_index),
                    "reference": str(channels[reference_index]),
                    "target": str(channels[target_index]),
                    "delay_ms": delay_ms,
                    "peak_center_ms": float(peak_index * bin_width),
                    "peak_count": peak_count,
                    "delay_window_count": int(window_values.size),
                    "total_count": total_count,
                    "peak_fraction": peak_fraction,
                    "peak_to_background": peak_to_background,
                    "background_count": background,
                    "std_ms": float(np.std(window_values)) if window_values.size else 0.0,
                    "truncated_bursts": 0,
                    "mode": mode,
                }
            )

    return sorted(
        results,
        key=lambda item: (
            -float(item["peak_count"]),
            -float(item["peak_fraction"]),
            -float(item["peak_to_background"]),
            abs(float(item["delay_ms"])),
            item["reference"],
            item["target"],
        ),
    )


def _channel_map_positions(channel_map: ChannelMap | None):
    if channel_map is None:
        return {}, {}

    coordinate_entries = []
    for electrode, payload in channel_map.electrodes.items():
        if not isinstance(payload, dict):
            continue
        try:
            x_um = float(payload.get("x_um", payload.get("x")))
            y_um = float(payload.get("y_um", payload.get("y")))
        except (TypeError, ValueError):
            continue
        if np.isfinite(x_um) and np.isfinite(y_um):
            coordinate_entries.append((str(electrode), payload, x_um, y_um))

    lookup = {}
    electrode_positions = {}
    if coordinate_entries:
        xs = np.asarray([entry[2] for entry in coordinate_entries], dtype=float)
        ys = np.asarray([entry[3] for entry in coordinate_entries], dtype=float)
        xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
        ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
        for index, (electrode, payload, x_um, y_um) in enumerate(coordinate_entries):
            x = (float(x_um) - xmin) / max(xmax - xmin, 1e-6)
            y = (float(y_um) - ymin) / max(ymax - ymin, 1e-6)
            electrode_positions[electrode] = (x, y, payload)
            aliases = [str(payload.get("channel") or "").strip(), electrode]
            raw_aliases = payload.get("aliases", [])
            if isinstance(raw_aliases, (list, tuple)):
                aliases.extend(str(alias).strip() for alias in raw_aliases)
            for alias in aliases:
                if not alias:
                    continue
                keys = [normalize_channel_name(alias)]
                if "_" in alias:
                    keys.append(normalize_channel_name(alias.split("_", 1)[1]))
                for key in keys:
                    if key and key not in lookup:
                        lookup[key] = (x, y, electrode, index)
        return lookup, electrode_positions

    rows = max(1, int(getattr(channel_map, "rows", 8) or 8))
    cols = max(1, int(getattr(channel_map, "cols", 8) or 8))
    for index, (electrode, payload) in enumerate(channel_map.electrodes.items()):
        if not isinstance(payload, dict):
            continue
        match = re.fullmatch(r"([A-Za-z]+)(\d+)", str(electrode))
        if match:
            row = ord(match.group(1)[0].upper()) - ord("A")
            col = int(match.group(2)) - 1
        else:
            row = index // cols
            col = index % cols
        x = (col + 0.5) / float(cols)
        y = (row + 0.5) / float(rows)
        electrode_positions[str(electrode)] = (x, y, payload)
        aliases = [str(payload.get("channel") or "").strip(), str(electrode)]
        raw_aliases = payload.get("aliases", [])
        if isinstance(raw_aliases, (list, tuple)):
            aliases.extend(str(alias).strip() for alias in raw_aliases)
        for alias in aliases:
            if not alias:
                continue
            key = normalize_channel_name(alias)
            if key and key not in lookup:
                lookup[key] = (x, y, str(electrode), index)
    return lookup, electrode_positions


def _position_for_channel(channel: str, position_lookup: dict):
    candidates = [
        str(channel),
        _base_channel_from_raster_label(str(channel)),
    ]
    if "_" in str(channel):
        candidates.append(str(channel).split("_", 1)[1])
    for candidate in candidates:
        key = normalize_channel_name(candidate)
        if key in position_lookup:
            return position_lookup[key]
    return None


def _burst_total_spike_vectors(spike_series, burst_intervals, bin_ms: float = 5.0, window_ms: float = 0.0):
    intervals = [(float(start), float(stop)) for start, stop in burst_intervals if float(stop) > float(start)]
    bin_s = max(0.001, float(bin_ms) / 1000.0)
    if intervals and float(window_ms) > 0:
        window_s = max(bin_s, float(window_ms) / 1000.0)
    elif intervals:
        window_s = max(bin_s, max(stop - start for start, stop in intervals))
    else:
        window_s = bin_s
    bin_count = max(1, int(np.ceil(window_s / bin_s)))
    edges = np.arange(bin_count + 1, dtype=float) * bin_s
    vectors = np.zeros((len(intervals), bin_count), dtype=float)
    durations = np.asarray([stop - start for start, stop in intervals], dtype=float) if intervals else np.array([], dtype=float)
    all_series = [(label, np.asarray(times, dtype=float)) for label, times in spike_series]

    for burst_index, (start_s, stop_s) in enumerate(intervals):
        relative_chunks = []
        for _, times in all_series:
            if times.size == 0:
                continue
            lo = int(np.searchsorted(times, start_s, side="left"))
            hi = int(np.searchsorted(times, stop_s, side="right"))
            if hi > lo:
                relative_chunks.append(times[lo:hi] - start_s)
        if not relative_chunks:
            continue
        relative = np.concatenate(relative_chunks)
        relative = relative[(relative >= 0.0) & (relative <= window_s)]
        if relative.size:
            counts, _ = np.histogram(relative, bins=edges)
            vectors[burst_index, :] = counts.astype(float)
    centers_ms = (edges[:-1] + edges[1:]) * 500.0
    return intervals, centers_ms, vectors, durations


def _burst_activity_matrix(spike_series, burst_intervals, time_bin_ms: float = 5.0, window_ms: float = 0.0):
    labels = [label for label, _ in spike_series]
    intervals = [(float(start), float(stop)) for start, stop in burst_intervals if float(stop) > float(start)]
    bin_s = max(0.001, float(time_bin_ms) / 1000.0)
    if intervals and float(window_ms) > 0:
        window_s = max(bin_s, float(window_ms) / 1000.0)
    elif intervals:
        window_s = max(bin_s, max(stop - start for start, stop in intervals))
    else:
        window_s = bin_s
    bin_count = max(1, int(np.ceil(window_s / bin_s)))
    edges = np.arange(bin_count + 1, dtype=float) * bin_s
    matrix = np.zeros((len(intervals), len(labels), bin_count), dtype=float)
    if not intervals or not labels:
        return labels, intervals, matrix

    for burst_index, (start_s, stop_s) in enumerate(intervals):
        analysis_stop_s = start_s + window_s
        for row_index, (_, times) in enumerate(spike_series):
            spike_times = np.asarray(times, dtype=float)
            if spike_times.size == 0:
                continue
            lo = int(np.searchsorted(spike_times, start_s, side="left"))
            hi = int(np.searchsorted(spike_times, analysis_stop_s, side="right"))
            if hi <= lo:
                continue
            relative = spike_times[lo:hi] - start_s
            counts, _ = np.histogram(relative, bins=edges)
            matrix[burst_index, row_index, :] = counts.astype(float) / bin_s
    return labels, intervals, matrix


def _burst_correlation_analysis(
    spike_series,
    burst_intervals,
    time_bin_ms: float = 5.0,
    window_ms: float = 0.0,
    normalization: str = "per_burst",
    method: str = "template",
    channel_map: ChannelMap | None = None,
    cluster_count: int = 3,
    embedding_method: str = "pca",
    dtw_warp_bins: int = 2,
    latency_window_ms: float = 0.0,
    graph_window_ms: float = 10.0,
    block_threshold: float = 0.45,
):
    labels, intervals, activity = _burst_activity_matrix(spike_series, burst_intervals, time_bin_ms, window_ms)
    burst_count = activity.shape[0]
    method = str(method or "template").lower()
    if burst_count < 2:
        return {
            "labels": labels,
            "intervals": intervals,
            "activity": activity,
            "features": activity.reshape((burst_count, -1)) if activity.ndim == 3 else activity,
            "correlation": np.zeros((burst_count, burst_count), dtype=float),
            "order": np.arange(burst_count, dtype=int),
            "groups": np.ones(burst_count, dtype=int),
            "time_bin_ms": float(time_bin_ms),
            "window_ms": float(window_ms),
            "method": method,
            "block_threshold": float(block_threshold),
        }

    if method == "global_stats":
        features = _burst_global_stat_features(spike_series, intervals, time_bin_ms)
    elif method == "latency":
        features = _burst_latency_features(spike_series, intervals, latency_window_ms)
    elif method == "spatial":
        spatial = _burst_spatial_activity_matrix(spike_series, intervals, channel_map, time_bin_ms, window_ms)
        features = np.log1p(spatial.reshape((burst_count, -1)))
    elif method == "dtw":
        features = np.log1p(activity)
    elif method == "graph":
        features = _burst_propagation_graph_features(spike_series, intervals, graph_window_ms)
    else:
        features = np.log1p(activity.reshape((burst_count, -1)))

    if method == "dtw":
        correlation, order, groups = _dtw_correlation_order_groups(features, int(dtw_warp_bins), float(block_threshold))
        flat_features = features.reshape((burst_count, -1))
    else:
        flat_features = features.reshape((burst_count, -1)) if features.ndim > 2 else np.asarray(features, dtype=float)
        flat_features = _normalize_burst_features(flat_features, normalization)
        correlation = _feature_correlation(flat_features)
        if method == "template":
            groups = _kmeans_groups(flat_features, cluster_count)
            order = np.lexsort((np.arange(burst_count), groups))
        elif method == "embedding":
            embedding = _burst_embedding(flat_features, embedding_method)
            groups = _kmeans_groups(embedding, cluster_count)
            order = np.lexsort((np.arange(burst_count), groups))
        else:
            order, groups = _correlation_order_groups(correlation, float(block_threshold))

    return {
        "labels": labels,
        "intervals": intervals,
        "activity": activity,
        "features": flat_features,
        "correlation": correlation,
        "order": order,
        "groups": groups,
        "time_bin_ms": float(time_bin_ms),
        "window_ms": float(window_ms),
        "method": method,
        "block_threshold": float(block_threshold),
    }


def _normalize_burst_features(features: np.ndarray, normalization: str) -> np.ndarray:
    features = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    normalization = str(normalization or "per_burst").lower()
    if normalization == "per_burst":
        totals = np.sum(np.abs(features), axis=1, keepdims=True)
        return np.divide(features, totals, out=np.zeros_like(features), where=totals > 1e-12)
    if normalization == "unit_zscore":
        means = np.mean(features, axis=0, keepdims=True)
        stds = np.std(features, axis=0, keepdims=True)
        return np.divide(features - means, stds, out=np.zeros_like(features), where=stds > 1e-12)
    return features


def _scale_burst_trace_vectors(vectors: np.ndarray, mode: str) -> tuple[np.ndarray, str]:
    values = np.nan_to_num(np.asarray(vectors, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    mode = str(mode or "per_trace_peak").lower()
    if mode == "per_trace_peak":
        peaks = np.nanmax(np.abs(values), axis=1, keepdims=True) if values.size else np.zeros((values.shape[0], 1))
        return np.divide(values, peaks, out=np.zeros_like(values), where=peaks > 1e-12), "Normalized firing profile"
    if mode == "log":
        return np.log1p(np.maximum(values, 0.0)), "log1p(total spike count / bin)"
    if mode == "robust":
        scale = float(np.nanpercentile(values, 95.0)) if values.size else 0.0
        scale = max(scale, 1e-12)
        return np.clip(values / scale, 0.0, 1.0), "Robust scaled firing profile"
    return values, "Total spike count / bin"


def _feature_correlation(features: np.ndarray) -> np.ndarray:
    features = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    burst_count = features.shape[0] if features.ndim == 2 else 0
    if burst_count == 0:
        return np.zeros((0, 0), dtype=float)
    if burst_count == 1 or features.shape[1] == 0:
        return np.eye(burst_count, dtype=float)
    centered = features - np.mean(features, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 1e-12)
    correlation = normalized @ normalized.T
    correlation = np.clip(np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _correlation_order_groups(correlation: np.ndarray, threshold: float = 0.45):
    burst_count = correlation.shape[0]
    if burst_count < 3:
        return np.arange(burst_count, dtype=int), np.ones(burst_count, dtype=int)
    distance = np.clip(1.0 - correlation, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    if np.allclose(condensed, 0.0):
        return np.arange(burst_count, dtype=int), np.ones(burst_count, dtype=int)
    tree = linkage(condensed, method="average")
    return leaves_list(tree).astype(int), fcluster(tree, t=max(0.0, float(threshold)), criterion="distance").astype(int)


def _kmeans_groups(features: np.ndarray, cluster_count: int) -> np.ndarray:
    features = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    burst_count = features.shape[0] if features.ndim == 2 else 0
    if burst_count < 2:
        return np.ones(burst_count, dtype=int)
    distinct_count = np.unique(np.round(features, decimals=12), axis=0).shape[0]
    cluster_count = min(max(1, int(cluster_count)), burst_count, distinct_count)
    if cluster_count <= 1 or np.allclose(features, features[0]):
        return np.ones(burst_count, dtype=int)
    return SkKMeans(n_clusters=cluster_count, n_init=10, random_state=7).fit_predict(features).astype(int) + 1


def _burst_embedding(features: np.ndarray, embedding_method: str) -> np.ndarray:
    features = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    burst_count = features.shape[0] if features.ndim == 2 else 0
    if burst_count == 0:
        return np.zeros((0, 2), dtype=float)
    if burst_count < 3 or str(embedding_method).lower() == "pca":
        components = min(2, features.shape[1], burst_count)
        if components < 1:
            return np.zeros((burst_count, 2), dtype=float)
        embedded = SkPCA(n_components=components, random_state=7).fit_transform(features)
        if embedded.shape[1] == 1:
            embedded = np.column_stack([embedded[:, 0], np.zeros(burst_count)])
        return embedded[:, :2]
    perplexity = max(2, min(30, burst_count - 1))
    try:
        return TSNE(n_components=2, perplexity=perplexity, random_state=7, init="pca", learning_rate="auto").fit_transform(features)
    except Exception:
        return _burst_embedding(features, "pca")


def _burst_global_stat_features(spike_series, intervals, bin_ms: float) -> np.ndarray:
    all_series = [(label, np.asarray(times, dtype=float)) for label, times in spike_series]
    features = []
    for index, (start_s, stop_s) in enumerate(intervals):
        duration_s = max(stop_s - start_s, 1e-9)
        relative_spikes = []
        active_rows = 0
        for _, times in all_series:
            lo = int(np.searchsorted(times, start_s, side="left"))
            hi = int(np.searchsorted(times, stop_s, side="right"))
            if hi > lo:
                active_rows += 1
                relative_spikes.append(times[lo:hi] - start_s)
        spikes = np.sort(np.concatenate(relative_spikes)) if relative_spikes else np.array([], dtype=float)
        total = float(spikes.size)
        if spikes.size:
            bin_s = max(0.001, float(bin_ms) / 1000.0)
            bins = np.arange(0.0, duration_s + bin_s, bin_s)
            counts, edges = np.histogram(spikes, bins=bins if bins.size >= 2 else np.array([0.0, duration_s]))
            peak_index = int(np.argmax(counts)) if counts.size else 0
            peak_rate = float(counts[peak_index]) / bin_s if counts.size else 0.0
            peak_time = float((edges[peak_index] + edges[peak_index + 1]) / 2.0) if counts.size else 0.0
        else:
            peak_rate = 0.0
            peak_time = 0.0
        prev_ibi = (start_s - intervals[index - 1][0]) if index > 0 else 0.0
        next_ibi = (intervals[index + 1][0] - start_s) if index + 1 < len(intervals) else 0.0
        features.append([duration_s, total, active_rows, peak_rate, peak_time, duration_s - peak_time, prev_ibi, next_ibi])
    return np.asarray(features, dtype=float)


def _burst_latency_features(spike_series, intervals, window_ms: float) -> np.ndarray:
    labels = [label for label, _ in spike_series]
    if not intervals or not labels:
        return np.zeros((len(intervals), len(labels)), dtype=float)
    if float(window_ms) > 0:
        window_s = float(window_ms) / 1000.0
    else:
        window_s = max(stop - start for start, stop in intervals)
    features = np.full((len(intervals), len(labels)), window_s * 1000.0, dtype=float)
    for burst_index, (start_s, stop_s) in enumerate(intervals):
        analysis_stop = start_s + max(window_s, 1e-9)
        for row_index, (_, times) in enumerate(spike_series):
            values = np.asarray(times, dtype=float)
            lo = int(np.searchsorted(values, start_s, side="left"))
            if lo < values.size and values[lo] <= analysis_stop:
                features[burst_index, row_index] = (float(values[lo]) - start_s) * 1000.0
    return features


def _burst_spatial_activity_matrix(spike_series, intervals, channel_map: ChannelMap | None, time_bin_ms: float, window_ms: float) -> np.ndarray:
    bin_s = max(0.001, float(time_bin_ms) / 1000.0)
    if intervals and float(window_ms) > 0:
        window_s = max(bin_s, float(window_ms) / 1000.0)
    elif intervals:
        window_s = max(bin_s, max(stop - start for start, stop in intervals))
    else:
        window_s = bin_s
    bin_count = max(1, int(np.ceil(window_s / bin_s)))
    matrix = np.zeros((len(intervals), 64, bin_count), dtype=float)
    if channel_map is None:
        _, _, activity = _burst_activity_matrix(spike_series, intervals, time_bin_ms, window_ms)
        flattened_rows = min(activity.shape[1], 64) if activity.ndim == 3 else 0
        matrix[:, :flattened_rows, :] = activity[:, :flattened_rows, :]
        return matrix
    channel_to_index = {}
    for electrode, payload in channel_map.electrodes.items():
        channel = str(payload.get("channel") or "").strip()
        if not channel:
            continue
        row = ord(electrode[0].upper()) - ord("A")
        try:
            col = int(electrode[1:]) - 1
        except ValueError:
            continue
        if 0 <= row < 8 and 0 <= col < 8:
            channel_to_index[normalize_channel_name(channel)] = row * 8 + col
    edges = np.arange(bin_count + 1, dtype=float) * bin_s
    for burst_index, (start_s, _) in enumerate(intervals):
        analysis_stop_s = start_s + window_s
        for label, times in spike_series:
            electrode_index = channel_to_index.get(normalize_channel_name(_base_channel_from_raster_label(label)))
            if electrode_index is None:
                continue
            values = np.asarray(times, dtype=float)
            lo = int(np.searchsorted(values, start_s, side="left"))
            hi = int(np.searchsorted(values, analysis_stop_s, side="right"))
            if hi <= lo:
                continue
            counts, _ = np.histogram(values[lo:hi] - start_s, bins=edges)
            matrix[burst_index, electrode_index, :] += counts.astype(float) / bin_s
    return matrix


def _burst_propagation_graph_features(spike_series, intervals, graph_window_ms: float) -> np.ndarray:
    labels = [label for label, _ in spike_series]
    row_count = len(labels)
    features = np.zeros((len(intervals), row_count * row_count), dtype=float)
    window_s = max(0.0001, float(graph_window_ms) / 1000.0)
    for burst_index, (start_s, stop_s) in enumerate(intervals):
        first_latencies = np.full(row_count, np.nan, dtype=float)
        for row_index, (_, times) in enumerate(spike_series):
            values = np.asarray(times, dtype=float)
            lo = int(np.searchsorted(values, start_s, side="left"))
            if lo < values.size and values[lo] <= stop_s:
                first_latencies[row_index] = float(values[lo] - start_s)
        adjacency = np.zeros((row_count, row_count), dtype=float)
        for source in range(row_count):
            if not np.isfinite(first_latencies[source]):
                continue
            for target in range(row_count):
                if source == target or not np.isfinite(first_latencies[target]):
                    continue
                delta = first_latencies[target] - first_latencies[source]
                if 0.0 < delta <= window_s:
                    adjacency[source, target] = 1.0 - delta / window_s
        features[burst_index] = adjacency.reshape(-1)
    return features


def _dtw_correlation_order_groups(sequences: np.ndarray, warp_bins: int, block_threshold: float = 0.45):
    sequences = np.nan_to_num(np.asarray(sequences, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    burst_count = sequences.shape[0]
    if burst_count < 2:
        return np.eye(burst_count), np.arange(burst_count, dtype=int), np.ones(burst_count, dtype=int)
    distance = np.zeros((burst_count, burst_count), dtype=float)
    for i in range(burst_count):
        for j in range(i + 1, burst_count):
            value = _multichannel_dtw_distance(sequences[i].T, sequences[j].T, int(warp_bins))
            distance[i, j] = value
            distance[j, i] = value
    max_distance = float(np.max(distance))
    if max_distance <= 1e-12:
        correlation = np.ones((burst_count, burst_count), dtype=float)
    else:
        correlation = 1.0 - 2.0 * distance / max_distance
        correlation = np.clip(correlation, -1.0, 1.0)
        np.fill_diagonal(correlation, 1.0)
    order, groups = _correlation_order_groups(correlation, block_threshold)
    return correlation, order, groups


def _multichannel_dtw_distance(a: np.ndarray, b: np.ndarray, warp_bins: int) -> float:
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return 0.0
    band = max(abs(n - m), max(0, int(warp_bins)))
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        start = max(1, i - band)
        stop = min(m, i + band) + 1
        for j in range(start, stop):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / max(n + m, 1))


def _unit_spike_trains_from_unified(data: UnifiedMEAData, include_noise: bool = False):
    units = []
    if not isinstance(data.sorting, dict):
        return units

    for channel in sorted(data.channels(), key=_channel_sort_key):
        spikes = np.asarray(data.spikes.get(channel, []), dtype=float)
        sorting = data.sorting.get(channel, {})
        if not isinstance(sorting, dict):
            continue
        labels = sorting.get("waveform_cluster_labels")
        if labels is None:
            continue
        labels = np.asarray(labels, dtype=np.int32)
        if labels.size != spikes.size:
            continue

        for unit in sorted(int(value) for value in np.unique(labels)):
            if unit == -1 and not include_noise:
                continue
            mask = labels == unit
            unit_spikes = np.asarray(spikes[mask], dtype=float)
            units.append(
                {
                    "id": f"{channel} {_unit_display_label(unit)}",
                    "channel": channel,
                    "unit": unit,
                    "spikes": unit_spikes,
                }
            )
    return units


def _pair_lag_deltas(reference_spikes, target_spikes, window_s: float, max_reference_events: int | None = None):
    reference = np.asarray(reference_spikes, dtype=float)
    target = np.asarray(target_spikes, dtype=float)
    if max_reference_events is not None and reference.size > max_reference_events:
        reference = reference[_display_indices(reference.size, max_reference_events)]
    deltas = []
    for spike_time in reference:
        lo = int(np.searchsorted(target, spike_time - window_s, side="left"))
        hi = int(np.searchsorted(target, spike_time + window_s, side="right"))
        if hi > lo:
            deltas.append(target[lo:hi] - spike_time)
    if not deltas:
        return np.array([], dtype=float)
    return np.concatenate(deltas).astype(float)


def _temporal_coupling_pairs(
    units,
    window_ms: float = 100.0,
    bin_ms: float = 1.0,
    min_spikes: int = 5,
    max_reference_events: int = 2000,
):
    window_s = max(0.001, float(window_ms) / 1000.0)
    bin_s = max(0.0001, float(bin_ms) / 1000.0)
    bins = np.arange(-window_s, window_s + bin_s, bin_s)
    if bins.size < 3:
        bins = np.array([-window_s, 0.0, window_s], dtype=float)

    results = []
    valid_units = [unit for unit in units if np.asarray(unit["spikes"]).size >= min_spikes]
    for reference in valid_units:
        ref_spikes = np.asarray(reference["spikes"], dtype=float)
        for target in valid_units:
            if reference["id"] == target["id"]:
                continue
            target_spikes = np.asarray(target["spikes"], dtype=float)
            deltas = _pair_lag_deltas(ref_spikes, target_spikes, window_s, max_reference_events)
            if deltas.size == 0:
                continue

            hist, edges = np.histogram(deltas, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2.0
            positive = centers > 0
            if not np.any(positive):
                continue
            positive_hist = hist[positive]
            if positive_hist.size == 0 or int(positive_hist.max()) == 0:
                continue
            positive_centers = centers[positive]
            peak_local = int(np.argmax(positive_hist))
            peak_count = int(positive_hist[peak_local])
            peak_lag = float(positive_centers[peak_local])
            baseline = np.concatenate([hist[centers < 0], hist[(centers > 0) & (np.abs(centers - peak_lag) > 2 * bin_s)]])
            baseline_mean = float(np.mean(baseline)) if baseline.size else 0.0
            baseline_std = float(np.std(baseline)) if baseline.size else 0.0
            z_score = (peak_count - baseline_mean) / baseline_std if baseline_std > 0 else float(peak_count > 0) * peak_count
            near_peak = np.abs(deltas - peak_lag) <= max(bin_s, 2 * bin_s)
            lag_std_ms = float(np.std(deltas[near_peak]) * 1000.0) if np.count_nonzero(near_peak) > 1 else 0.0
            matched_ratio = float(np.count_nonzero((deltas > 0) & (np.abs(deltas - peak_lag) <= 2 * bin_s))) / max(1, ref_spikes.size)
            strength = float(peak_count * max(z_score, 0.0))
            results.append(
                {
                    "reference": reference,
                    "target": target,
                    "reference_id": reference["id"],
                    "target_id": target["id"],
                    "deltas": deltas,
                    "hist": hist,
                    "edges": edges,
                    "peak_lag_ms": peak_lag * 1000.0,
                    "peak_count": peak_count,
                    "z_score": float(z_score),
                    "lag_std_ms": lag_std_ms,
                    "matched_ratio": matched_ratio,
                    "strength": strength,
                }
            )
    return sorted(results, key=lambda item: (item["strength"], item["peak_count"]), reverse=True)


STIMULUS_RESPONSE_EXTENSIONS = {".nev", ".spk", ".h5", ".hdf5", ".npz"}


def _stimulus_response_supported_files(paths) -> list[Path]:
    files = []
    seen = set()
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            candidates = [
                child
                for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() in STIMULUS_RESPONSE_EXTENSIONS
            ]
        elif path.is_file() and path.suffix.lower() in STIMULUS_RESPONSE_EXTENSIONS:
            candidates = [path]
        else:
            candidates = []
        for candidate in candidates:
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(candidate)
    return sorted(files, key=lambda item: str(item).lower())


def _normalize_stimulus_parameter_key(key: str, unit: str = "") -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(key).strip().lower()).strip("_")
    unit_text = str(unit or "").replace("µ", "u").lower()
    aliases = {
        "el": "stim_electrode",
        "electrode": "stim_electrode",
        "freq": "frequency",
        "frequency": "frequency",
        "hz": "frequency",
        "amp": "amplitude",
        "ampl": "amplitude",
        "amplitude": "amplitude",
        "current": "amplitude",
        "dur": "duration",
        "duration": "duration",
        "width": "width",
        "pulse": "pulse",
        "intensity": "intensity",
        "level": "level",
    }
    base = aliases.get(text, text or "value")
    if unit_text in {"hz", "khz"} and base == "value":
        base = "frequency"
    elif unit_text in {"ua", "ma", "a", "mv", "v"} and base == "value":
        base = "amplitude"
    elif unit_text in {"us", "ms", "s"} and base == "value":
        base = "duration"
    return f"{base}_{unit_text}" if unit_text else base


def _extract_stimulus_parameters(path: str | Path) -> dict[str, object]:
    path_text = str(path)
    path = Path(path_text)
    split_parts = [part for part in re.split(r"[\\/]+", path_text) if part]
    if split_parts:
        text_parts = split_parts[-4:-1] + [Path(split_parts[-1]).stem]
    else:
        text_parts = [part for part in path.parts[-4:-1] if part] + [path.stem]
    text = " ".join(text_parts)
    params: dict[str, object] = {}
    pattern = re.compile(
        r"([A-Za-z]+)\s*[-_= ]?\s*(-?\d+(?:\.\d+)?)\s*(Hz|kHz|uA|µA|mA|A|mV|V|us|ms|s)?(?=[^A-Za-zµu]|$)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        key = _normalize_stimulus_parameter_key(match.group(1), match.group(3))
        try:
            params[key] = float(match.group(2))
        except ValueError:
            continue
    unit_only = re.compile(r"(?<![A-Za-z])(-?\d+(?:\.\d+)?)\s*(Hz|kHz|uA|µA|mA|A|mV|V|us|ms|s)\b", re.IGNORECASE)
    for match in unit_only.finditer(text):
        key = _normalize_stimulus_parameter_key("value", match.group(2))
        params.setdefault(key, float(match.group(1)))
    categorical = []
    for part in text_parts[-3:]:
        cleaned = re.sub(r"[_\-]+", " ", str(part)).strip()
        if cleaned and not re.search(r"\d", cleaned):
            categorical.append(cleaned)
    if categorical:
        params["condition"] = " / ".join(categorical[-2:])
    lower_text = text.lower()
    if re.search(r"(?<![a-z])spont(?![a-z])", lower_text):
        params["activity"] = "spont"
    if re.search(r"multi[\s_-]*site", lower_text):
        params["stim_mode"] = "multi-site"
        params.pop("site", None)
    if re.search(r"(?<![a-z])pre(?![a-z])", lower_text):
        params["period"] = "pre"
    elif re.search(r"(?<![a-z])after(?![a-z])", lower_text):
        params["period"] = "after"
    return params


def _stimulus_parameter_label(parameters: dict[str, object], path: str | Path) -> str:
    label_items = []
    for key in ("activity", "period", "stim_mode"):
        value = str(parameters.get(key, "")).strip()
        if value:
            label_items.append(f"{key}={value}")
    numeric_items = [
        (key, value)
        for key, value in parameters.items()
        if key not in {"condition"} and isinstance(value, (int, float, np.integer, np.floating))
    ]
    label_items.extend(f"{key}={float(value):g}" for key, value in sorted(numeric_items))
    if label_items:
        return ", ".join(label_items)
    condition = str(parameters.get("condition", "")).strip()
    if condition:
        return condition
    path = Path(path)
    return f"{path.parent.name} / {path.stem}" if path.parent.name else path.stem


def _load_spike_only_data(path: str | Path, cancel_check=None) -> UnifiedMEAData:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        data = read_maxwell_h5(path, cancel_check=cancel_check, extract_waveforms=False)
    elif suffix == ".nev":
        data = read_blackrock_nev(path, cancel_check=cancel_check)
    elif suffix == ".spk":
        data = read_axion_spk(path)
    else:
        data = MEAReader(path).load_data()
    if not isinstance(data, UnifiedMEAData):
        raise ValueError(f"Stimulus response analysis requires spike-event data: {path}")
    data.waveforms = {}
    return data


def _stimulus_response_record_from_data(
    path: str | Path,
    data: UnifiedMEAData,
    *,
    pre_ms: float = 200.0,
    response_ms: float = 1000.0,
    bin_ms: float = 5.0,
    artifact_ms: float = 0.0,
) -> dict:
    stim_times = np.asarray(data.stim_times, dtype=float)
    stim_times = np.sort(stim_times[np.isfinite(stim_times)])
    if stim_times.size == 0:
        raise ValueError(f"No stimulus timestamps found: {path}")
    try:
        channels = sorted([str(channel) for channel in data.channels()], key=_channel_sort_key)
    except Exception:
        channels = []
    if not channels:
        channels = sorted([str(channel) for channel in data.spikes.keys()], key=_channel_sort_key)
    channel_spikes = []
    spike_chunks = []
    for channel in channels:
        times = data.spikes.get(channel, [])
        values = np.asarray(times, dtype=float)
        values = np.sort(values[np.isfinite(values)])
        channel_spikes.append((channel, values))
        if values.size:
            spike_chunks.append(values)
    if not spike_chunks:
        raise ValueError(f"No spikes found: {path}")
    all_spikes = np.sort(np.concatenate(spike_chunks))
    channel_count = max(1, len(channel_spikes))
    pre_ms = max(0.0, float(pre_ms))
    response_ms = max(1.0, float(response_ms))
    artifact_ms = max(0.0, float(artifact_ms))
    response_counts = []
    baseline_counts = []
    latencies_ms = []
    trial_spikes_ms = []
    trial_channel_spikes_ms = []
    for stim_s in stim_times:
        window_start = float(stim_s) - pre_ms / 1000.0
        window_stop = float(stim_s) + response_ms / 1000.0
        channel_trial = []
        relative_chunks = []
        for _channel, values in channel_spikes:
            lo = int(np.searchsorted(values, window_start, side="left"))
            hi = int(np.searchsorted(values, window_stop, side="right"))
            relative = (values[lo:hi] - float(stim_s)) * 1000.0
            if artifact_ms > 0.0 and relative.size:
                relative = relative[np.abs(relative) > artifact_ms]
            relative = relative.astype(float, copy=False)
            channel_trial.append(relative)
            if relative.size:
                relative_chunks.append(relative)
        relative_ms = np.sort(np.concatenate(relative_chunks)) if relative_chunks else np.array([], dtype=float)
        trial_spikes_ms.append(relative_ms.astype(float, copy=False))
        trial_channel_spikes_ms.append(channel_trial)
        response_mask = (relative_ms >= 0.0) & (relative_ms <= response_ms)
        baseline_mask = (relative_ms < 0.0) & (relative_ms >= -pre_ms)
        response_counts.append(int(np.count_nonzero(response_mask)))
        baseline_counts.append(int(np.count_nonzero(baseline_mask)))
        if np.any(response_mask):
            latencies_ms.append(float(np.min(relative_ms[response_mask])))
    response_s = response_ms / 1000.0
    baseline_s = max(pre_ms / 1000.0, 1e-9)
    parameters = _extract_stimulus_parameters(path)
    return {
        "path": str(path),
        "file": Path(path).name,
        "condition": _stimulus_parameter_label(parameters, path),
        "parameters": parameters,
        "stim_count": int(stim_times.size),
        "channel_count": int(channel_count),
        "spike_count": int(all_spikes.size),
        "artifact_ms": float(artifact_ms),
        "channels": [channel for channel, _values in channel_spikes],
        "response_spikes_per_stim": float(np.mean(response_counts)) if response_counts else 0.0,
        "response_rate_hz_per_channel": float(np.sum(response_counts) / max(float(stim_times.size) * response_s * channel_count, 1e-9)),
        "baseline_rate_hz_per_channel": float(np.sum(baseline_counts) / max(float(stim_times.size) * baseline_s * channel_count, 1e-9)),
        "mean_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else np.nan,
        "trial_spikes_ms": trial_spikes_ms,
        "trial_channel_spikes_ms": trial_channel_spikes_ms,
    }


def _stimulus_response_group_records(records: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record.get("condition", "")), []).append(record)
    summaries = []
    for condition, items in sorted(grouped.items(), key=lambda item: item[0].lower()):
        stim_total = sum(int(item.get("stim_count", 0)) for item in items)
        rates = np.asarray([float(item.get("response_rate_hz_per_channel", 0.0)) for item in items], dtype=float)
        baseline = np.asarray([float(item.get("baseline_rate_hz_per_channel", 0.0)) for item in items], dtype=float)
        latencies = np.asarray([float(item.get("mean_latency_ms", np.nan)) for item in items], dtype=float)
        trial_spikes = []
        for item in items:
            for trial in item.get("trial_spikes_ms", []):
                trial_spikes.append(np.asarray(trial, dtype=float))
        summaries.append(
            {
                "condition": condition,
                "file_count": len(items),
                "stim_count": int(stim_total),
                "response_rate_mean": float(np.mean(rates)) if rates.size else 0.0,
                "response_rate_sem": float(np.std(rates, ddof=1) / np.sqrt(rates.size)) if rates.size > 1 else 0.0,
                "baseline_rate_mean": float(np.mean(baseline)) if baseline.size else 0.0,
                "mean_latency_ms": float(np.nanmean(latencies)) if np.any(np.isfinite(latencies)) else np.nan,
                "trial_spikes_ms": trial_spikes,
                "records": items,
            }
        )
    return summaries


def _fix_spinbox_hit_targets(root: QWidget) -> None:
    for spinbox in [*root.findChildren(QSpinBox), *root.findChildren(QDoubleSpinBox)]:
        spinbox.setCursor(Qt.CursorShape.ArrowCursor)
        spinbox.setMinimumHeight(30)
        spinbox.setStyleSheet(
            """
            QSpinBox, QDoubleSpinBox { padding-left: 6px; padding-right: 2px; }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 20px;
                height: 14px;
                border-left: 1px solid #cfd8e6;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 20px;
                height: 14px;
                border-left: 1px solid #cfd8e6;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                width: 7px;
                height: 7px;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                width: 7px;
                height: 7px;
            }
            """
        )


class ElectrodeMapCanvas(QWidget):
    electrode_selected = Signal(str)

    def __init__(self, channel_map: ChannelMap, parent=None):
        super().__init__(parent)
        self.channel_map = channel_map
        self.selected_electrode = "A1"
        self.available_channels = set()
        self.highlighted_electrodes: set[str] = set()
        self.stim_electrodes: set[str] = set()
        self.setMinimumSize(920, 620)

    def set_available_channels(self, channels) -> None:
        self.available_channels = _normalized_channel_keys(channels)
        self.update()

    def set_channel_map(self, channel_map: ChannelMap) -> None:
        self.channel_map = channel_map
        if self.selected_electrode not in channel_map.electrodes:
            self.selected_electrode = next(iter(channel_map.electrodes), "A1")
        self.update()

    def set_overlays(self, highlighted_electrodes=None, stim_electrodes=None) -> None:
        self.highlighted_electrodes = {str(electrode) for electrode in highlighted_electrodes or [] if str(electrode)}
        self.stim_electrodes = {str(electrode) for electrode in stim_electrodes or [] if str(electrode)}
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt override
        if self._uses_coordinate_layout():
            self._paint_coordinate_layout(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f8fafc"))

        centers = {
            electrode_id(row, col): self._center(row, col)
            for row in range(8)
            for col in range(8)
        }

        wire_pen = QPen(QColor("#9aa7bd"), 2)
        painter.setPen(wire_pen)
        for row in range(8):
            row_points = [centers[electrode_id(row, col)] for col in range(8)]
            for left, right in zip(row_points, row_points[1:]):
                painter.drawLine(left[0], left[1], right[0], right[1])
        for col in range(8):
            col_points = [centers[electrode_id(row, col)] for row in range(8)]
            for top, bottom in zip(col_points, col_points[1:]):
                painter.drawLine(top[0], top[1], bottom[0], bottom[1])

        for row in range(8):
            for col in range(8):
                electrode = electrode_id(row, col)
                channel = self.channel_map.channel_for(electrode)
                reference = self.channel_map.is_reference(electrode)
                x, y = centers[electrode]
                radius = self._radius()

                if electrode in self.stim_electrodes:
                    fill = QColor("#7c3aed")
                elif electrode in self.highlighted_electrodes:
                    fill = QColor("#dc2626")
                else:
                    fill = QColor("#36c986") if channel else QColor("#ef5f5f")
                painter.setBrush(fill)
                if electrode in self.stim_electrodes:
                    outline = QColor("#f59e0b")
                    outline_width = 5
                elif electrode in self.highlighted_electrodes:
                    outline = QColor("#7f1d1d")
                    outline_width = 5
                elif electrode == self.selected_electrode:
                    outline = QColor("#1d4ed8")
                    outline_width = 4
                else:
                    outline = QColor("#233044")
                    outline_width = 2
                painter.setPen(QPen(outline, outline_width))
                painter.drawEllipse(QRectF(x - radius, y - radius, radius * 2, radius * 2))

                painter.setPen(QPen(QColor("#0f172a"), 1))
                label = channel if channel else electrode
                font = QFont("Segoe UI", 8 if len(label) > 5 else 9, QFont.Bold)
                painter.setFont(font)
                painter.drawText(
                    QRectF(x - radius + 3, y - radius + 5, radius * 2 - 6, radius * 2 - 10),
                    Qt.AlignmentFlag.AlignCenter,
                    label,
                )

                if reference:
                    painter.setPen(QPen(QColor("#050505"), 4))
                    inset = radius * 0.45
                    painter.drawLine(x - inset, y - inset, x + inset, y + inset)
                    painter.drawLine(x + inset, y - inset, x - inset, y + inset)
                if electrode in self.stim_electrodes:
                    painter.setPen(QPen(QColor("#ffffff"), 2))
                    painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                    painter.drawText(QRectF(x - radius, y + radius * 0.25, radius * 2, radius * 0.65), Qt.AlignmentFlag.AlignCenter, "STIM")

        painter.end()

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        if self._uses_coordinate_layout():
            self._mouse_press_coordinate_layout(event)
            return

        pos = event.position() if hasattr(event, "position") else event.pos()
        px = float(pos.x())
        py = float(pos.y())
        radius = self._radius()

        for row in range(8):
            for col in range(8):
                electrode = electrode_id(row, col)
                x, y = self._center(row, col)
                if (px - x) ** 2 + (py - y) ** 2 <= radius**2:
                    self.selected_electrode = electrode
                    self.electrode_selected.emit(electrode)
                    self.update()
                    return

    def _radius(self) -> float:
        return min(self.width(), self.height()) / 23.0

    def _center(self, row: int, col: int) -> Tuple[float, float]:
        size = min(self.width(), self.height())
        margin = size * 0.11
        step = (size - margin * 2) / 7.0
        x_offset = (self.width() - size) / 2.0
        y_offset = (self.height() - size) / 2.0
        return x_offset + margin + col * step, y_offset + margin + row * step

    def _uses_coordinate_layout(self) -> bool:
        if self.channel_map.rows <= 8 and self.channel_map.cols <= 8:
            return False
        return any(
            isinstance(payload, dict)
            and ("x_um" in payload or "x" in payload)
            and ("y_um" in payload or "y" in payload)
            for payload in self.channel_map.electrodes.values()
        )

    def _coordinate_entries(self):
        entries = []
        for electrode, payload in self.channel_map.electrodes.items():
            if not isinstance(payload, dict):
                continue
            try:
                x_um = float(payload.get("x_um", payload.get("x")))
                y_um = float(payload.get("y_um", payload.get("y")))
            except (TypeError, ValueError):
                continue
            if np.isfinite(x_um) and np.isfinite(y_um):
                entries.append((str(electrode), payload, x_um, y_um))
        return entries

    def _coordinate_rect_and_bounds(self, entries):
        margin = 18.0
        footer = 34.0
        available_w = max(1.0, self.width() - margin * 2.0)
        available_h = max(1.0, self.height() - margin - footer)
        xs = np.asarray([entry[2] for entry in entries], dtype=float)
        ys = np.asarray([entry[3] for entry in entries], dtype=float)
        xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
        ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
        span_x = max(1.0, xmax - xmin)
        span_y = max(1.0, ymax - ymin)
        scale = min(available_w / span_x, available_h / span_y)
        draw_w = span_x * scale
        draw_h = span_y * scale
        rect = QRectF(margin + (available_w - draw_w) * 0.5, margin + (available_h - draw_h) * 0.5, draw_w, draw_h)
        return rect, (xmin, xmax, ymin, ymax)

    def _coordinate_point(self, rect: QRectF, bounds, x_um: float, y_um: float) -> QPointF:
        xmin, xmax, ymin, ymax = bounds
        x = rect.left() + (float(x_um) - xmin) / max(xmax - xmin, 1e-6) * rect.width()
        y = rect.top() + (float(y_um) - ymin) / max(ymax - ymin, 1e-6) * rect.height()
        return QPointF(float(x), float(y))

    def _paint_coordinate_layout(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#05070a"))
        entries = self._coordinate_entries()
        if not entries:
            painter.setPen(QPen(QColor("#cbd5e1"), 1))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No coordinate map")
            painter.end()
            return

        rect, bounds = self._coordinate_rect_and_bounds(entries)
        painter.setPen(QPen(QColor("#263244"), 1))
        painter.setBrush(QColor("#101820"))
        painter.drawRoundedRect(rect.adjusted(-3, -3, 3, 3), 4, 4)

        background_points = []
        recorded_points = []
        selected_point = None
        for electrode, payload, x_um, y_um in entries:
            point = self._coordinate_point(rect, bounds, x_um, y_um)
            recorded = self._is_recording_electrode(electrode, payload)
            if electrode == self.selected_electrode:
                selected_point = (point, electrode, payload)
            if recorded:
                recorded_points.append((point, electrode, payload))
            else:
                background_points.append((point, electrode, payload))

        radius = max(1.6, min(3.8, rect.width() / 520.0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#475569"))
        background_radius = max(0.75, radius * 0.55)
        for point, _, _ in background_points:
            painter.drawEllipse(
                QRectF(
                    point.x() - background_radius,
                    point.y() - background_radius,
                    background_radius * 2,
                    background_radius * 2,
                )
            )

        painter.setBrush(QColor("#18b7ff"))
        for point, _, _ in recorded_points:
            painter.drawEllipse(QRectF(point.x() - radius, point.y() - radius, radius * 2, radius * 2))

        painter.setBrush(Qt.BrushStyle.NoBrush)
        for point, electrode, _payload in recorded_points + background_points:
            if electrode in self.highlighted_electrodes:
                painter.setPen(QPen(QColor("#ef4444"), 2.4))
                painter.drawEllipse(QRectF(point.x() - radius * 2.3, point.y() - radius * 2.3, radius * 4.6, radius * 4.6))
            if electrode in self.stim_electrodes:
                painter.setPen(QPen(QColor("#f59e0b"), 2.6))
                painter.drawEllipse(QRectF(point.x() - radius * 3.0, point.y() - radius * 3.0, radius * 6.0, radius * 6.0))
                painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                painter.drawText(QRectF(point.x() + radius * 2.6, point.y() - 10, 60, 16), Qt.AlignmentFlag.AlignLeft, "STIM")

        if selected_point is not None:
            point, electrode, payload = selected_point
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#facc15"), 2.2))
            painter.drawEllipse(QRectF(point.x() - radius * 2.4, point.y() - radius * 2.4, radius * 4.8, radius * 4.8))
            aliases = payload.get("aliases", [])
            alias_text = ", ".join(str(alias) for alias in aliases[:2]) if isinstance(aliases, list) else ""
            source = payload.get("source_channel", "")
            label = str(electrode)
            if source != "":
                label += f" | channel {source}"
            if alias_text:
                label += f" | {alias_text}"
            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QPen(QColor("#e5e7eb"), 1))
            painter.drawText(QRectF(18, self.height() - 28, self.width() - 36, 18), Qt.AlignmentFlag.AlignLeft, label)

        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QPen(QColor("#e5e7eb"), 1))
        title = (
            f"{self.channel_map.name}: {len(entries)} electrodes | "
            f"recording {len(recorded_points)} | non-recording {max(0, len(entries) - len(recorded_points))}"
        )
        painter.drawText(QRectF(18, 4, self.width() - 36, 18), Qt.AlignmentFlag.AlignLeft, title)
        self._draw_coordinate_legend(painter)
        painter.end()

    def _is_recording_electrode(self, electrode: str, payload: dict) -> bool:
        if not self.available_channels:
            return False
        aliases = payload.get("aliases", [])
        candidates = [payload.get("channel", ""), electrode]
        if isinstance(aliases, (list, tuple)):
            candidates.extend(aliases)
        return bool(_normalized_channel_keys(candidates) & self.available_channels)

    def _draw_coordinate_legend(self, painter: QPainter) -> None:
        x = self.width() - 250
        y = 10
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(QColor("#94a3b8"), 1))
        painter.setBrush(QColor("#18b7ff"))
        painter.drawEllipse(QRectF(x, y + 3, 8, 8))
        painter.drawText(QRectF(x + 14, y, 90, 16), Qt.AlignmentFlag.AlignLeft, "recording")
        painter.setBrush(QColor("#334155"))
        painter.drawEllipse(QRectF(x + 104, y + 4, 7, 7))
        painter.drawText(QRectF(x + 116, y, 110, 16), Qt.AlignmentFlag.AlignLeft, "non-recording")

    def _mouse_press_coordinate_layout(self, event) -> None:
        entries = self._coordinate_entries()
        if not entries:
            return
        rect, bounds = self._coordinate_rect_and_bounds(entries)
        pos = event.position() if hasattr(event, "position") else event.pos()
        px = float(pos.x())
        py = float(pos.y())
        best = None
        best_distance = float("inf")
        for electrode, payload, x_um, y_um in entries:
            point = self._coordinate_point(rect, bounds, x_um, y_um)
            distance = (point.x() - px) ** 2 + (point.y() - py) ** 2
            if distance < best_distance:
                best = electrode
                best_distance = distance
        threshold = max(10.0, min(self.width(), self.height()) / 45.0)
        if best is not None and best_distance <= threshold**2:
            self.selected_electrode = best
            self.electrode_selected.emit(best)
            self.update()


class ChannelMapDialog(QDialog):
    def __init__(self, channel_map: ChannelMap | None = None, available_channels=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Channel Map")
        self.resize(1480, 900)
        self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)
        self.available_channels = [str(channel) for channel in available_channels or []]
        self.channel_map = channel_map or default_channel_map() or ChannelMap.new()
        self.selected_electrode = "A1" if "A1" in self.channel_map.electrodes else next(iter(self.channel_map.electrodes), "A1")

        self.canvas = ElectrodeMapCanvas(self.channel_map)
        self.canvas.selected_electrode = self.selected_electrode
        self.canvas.set_available_channels(self.available_channels)
        self.canvas.electrode_selected.connect(self._select_electrode)

        self.saved_maps = QComboBox()
        self._refresh_saved_maps()

        self.name_edit = QLineEdit(self.channel_map.name)
        self.electrode_label = QLabel(self.selected_electrode)
        self.electrode_label.setObjectName("CardTitle")

        self.channel_combo = QComboBox()
        self.channel_combo.setEditable(True)
        self._refresh_channel_choices()

        self.reference_check = QCheckBox("Reference electrode")
        self.reference_check.stateChanged.connect(self._set_reference_from_checkbox)

        set_channel = QPushButton("Set Channel")
        set_channel.clicked.connect(self._set_channel)
        clear_channel = QPushButton("Clear Channel")
        clear_channel.clicked.connect(self._clear_channel)
        new_map = QPushButton("New Map")
        new_map.clicked.connect(self._new_map)
        load_selected = QPushButton("Load Selected")
        load_selected.clicked.connect(self._load_selected_map)
        load_default = QPushButton("Load Default")
        load_default.clicked.connect(self._load_default_map)
        save = QPushButton("Save")
        save.clicked.connect(self._save_map)
        save_default = QPushButton("Save As Default")
        save_default.clicked.connect(self._save_as_default)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)

        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        self.validation_text.setMinimumHeight(170)

        right = QVBoxLayout()
        right.addWidget(QLabel("Map name"))
        right.addWidget(self.name_edit)
        right.addWidget(QLabel("Saved maps"))
        right.addWidget(self.saved_maps)
        right.addWidget(load_selected)
        right.addWidget(load_default)
        right.addSpacing(10)
        right.addWidget(QLabel("Selected electrode"))
        right.addWidget(self.electrode_label)
        right.addWidget(QLabel("Channel"))
        right.addWidget(self.channel_combo)
        right.addWidget(set_channel)
        right.addWidget(clear_channel)
        right.addWidget(self.reference_check)
        right.addSpacing(10)
        right.addWidget(new_map)
        right.addWidget(save)
        right.addWidget(save_default)
        right.addWidget(QLabel("Validation"))
        right.addWidget(self.validation_text, 1)
        right.addWidget(close)

        layout = QHBoxLayout(self)
        layout.addWidget(self.canvas, 5)
        layout.addLayout(right)

        self._select_electrode(self.selected_electrode)
        self._update_validation()
        _fix_spinbox_hit_targets(self)

    def _refresh_saved_maps(self) -> None:
        self.saved_maps.clear()
        self.saved_maps.addItems(list_channel_maps())

    def _refresh_channel_choices(self) -> None:
        current = self.channel_combo.currentText().strip() if hasattr(self, "channel_combo") else ""
        self.channel_combo.clear()
        self.channel_combo.addItem("")
        self.channel_combo.addItems(self.available_channels)
        if current:
            self.channel_combo.setCurrentText(current)

    def _set_map(self, channel_map: ChannelMap) -> None:
        self.channel_map = channel_map
        self.name_edit.setText(channel_map.name)
        self.canvas.set_channel_map(channel_map)
        if self.selected_electrode not in channel_map.electrodes:
            self.selected_electrode = self.canvas.selected_electrode
        self._select_electrode(self.selected_electrode)
        self._update_validation()

    def _select_electrode(self, electrode: str) -> None:
        self.selected_electrode = electrode
        self.electrode_label.setText(electrode)
        self.channel_combo.setCurrentText(self.channel_map.channel_for(electrode))
        self.reference_check.blockSignals(True)
        self.reference_check.setChecked(self.channel_map.is_reference(electrode))
        self.reference_check.blockSignals(False)

    def _set_channel(self) -> None:
        self.channel_map.set_channel(self.selected_electrode, self.channel_combo.currentText())
        self.canvas.update()
        self._update_validation()

    def _clear_channel(self) -> None:
        self.channel_map.set_channel(self.selected_electrode, "")
        self.channel_combo.setCurrentText("")
        self.canvas.update()
        self._update_validation()

    def _set_reference_from_checkbox(self, *_) -> None:
        self.channel_map.set_reference(self.selected_electrode, self.reference_check.isChecked())
        self.canvas.update()
        self._update_validation()

    def _new_map(self) -> None:
        name, accepted = QInputDialog.getText(self, "New Channel Map", "Map name")
        if not accepted:
            return
        name = name.strip() or "Untitled"
        self._set_map(ChannelMap.new(name))

    def _load_selected_map(self) -> None:
        name = self.saved_maps.currentText().strip()
        if not name:
            QMessageBox.information(self, "Channel Map", "No saved map is selected.")
            return
        loaded = load_channel_map(name)
        if loaded is None:
            QMessageBox.warning(self, "Channel Map", f"Map not found: {name}")
            self._refresh_saved_maps()
            return
        self._set_map(loaded)

    def _load_default_map(self) -> None:
        loaded = default_channel_map()
        if loaded is None:
            QMessageBox.information(self, "Channel Map", "No default map has been saved.")
            return
        self._set_map(loaded)

    def _save_map(self) -> None:
        self.channel_map.name = self.name_edit.text().strip() or "Untitled"
        save_channel_map(self.channel_map)
        self._refresh_saved_maps()
        self.saved_maps.setCurrentText(self.channel_map.name)
        self._update_validation()
        QMessageBox.information(self, "Channel Map", f"Saved map: {self.channel_map.name}")

    def _save_as_default(self) -> None:
        self.channel_map.name = self.name_edit.text().strip() or "Untitled"
        save_channel_map(self.channel_map, make_default=True)
        self._refresh_saved_maps()
        self.saved_maps.setCurrentText(self.channel_map.name)
        self._update_validation()
        QMessageBox.information(self, "Channel Map", f"Saved default map: {self.channel_map.name}")

    def _update_validation(self) -> None:
        report = validate_channel_map(self.channel_map, self.available_channels)
        lines = [
            f"Mapped electrodes: {report['mapped_count']}",
            f"Empty electrodes: {report['empty_electrode_count']}",
            f"Reference electrodes: {', '.join(report['reference_electrodes']) or 'None'}",
        ]

        duplicates = report["duplicates"]
        if duplicates:
            lines.append("Duplicate channel assignments:")
            for channel, electrodes in duplicates.items():
                lines.append(f"  {channel}: {', '.join(electrodes)}")

        unknown = report["unknown_channels"]
        if unknown:
            lines.append(f"Channels not found in loaded data: {', '.join(unknown[:20])}")
            if len(unknown) > 20:
                lines.append(f"  ... {len(unknown) - 20} more")

        unmapped = report["unmapped_channels"]
        if unmapped:
            lines.append(f"Loaded channels without electrode: {', '.join(unmapped[:20])}")
            if len(unmapped) > 20:
                lines.append(f"  ... {len(unmapped) - 20} more")

        lines.append("Status: valid" if report["is_valid"] else "Status: needs attention")
        self.validation_text.setPlainText("\n".join(lines))


class WorkerSignals(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)
    canceled = Signal(str)


def _create_progress_dialog(parent: QWidget, title: str, message: str, maximum: int = 0) -> QProgressDialog:
    maximum = int(maximum or 0)
    dialog = QProgressDialog(message, "Cancel", 0, 100 if maximum <= 0 else maximum, parent)
    dialog.setWindowTitle(title)
    dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.setValue(0)
    dialog._progress_cancel_requested = False
    dialog._progress_reported_value = 0
    dialog._progress_real_maximum = maximum
    dialog._progress_pulse_direction = 1
    dialog._progress_pulse_tick = 0
    dialog._progress_soft_ceiling = 100 if maximum <= 0 else max(0, maximum - 1)
    dialog._progress_title = str(title)
    dialog.canceled.connect(lambda: setattr(dialog, "_progress_cancel_requested", True))

    timer = QTimer(dialog)
    timer.setInterval(90)

    def pulse():
        if not dialog.isVisible():
            return
        real_maximum = int(getattr(dialog, "_progress_real_maximum", 0))
        current = int(dialog.value())
        if real_maximum <= 0:
            direction = int(getattr(dialog, "_progress_pulse_direction", 1))
            next_value = current + direction * 3
            if next_value >= 100:
                next_value = 100
                direction = -1
            elif next_value <= 0:
                next_value = 0
                direction = 1
            dialog._progress_pulse_direction = direction
            dialog.setValue(next_value)
            return
        tick = int(getattr(dialog, "_progress_pulse_tick", 0)) + 1
        dialog._progress_pulse_tick = tick
        if tick % 5:
            return
        reported = int(getattr(dialog, "_progress_reported_value", current))
        soft_target = int(getattr(dialog, "_progress_soft_ceiling", reported))
        soft_target = min(real_maximum - 1, max(reported, soft_target))
        if current < soft_target:
            dialog.setValue(current + 1)

    timer.timeout.connect(pulse)
    timer.start()
    dialog._progress_timer = timer
    dialog.show()
    QApplication.processEvents()
    return dialog


def _progress_cancel_requested(dialog: QProgressDialog | None) -> bool:
    if dialog is None:
        return False
    return bool(getattr(dialog, "_progress_cancel_requested", False) or dialog.wasCanceled())


def _set_progress_dialog(dialog: QProgressDialog | None, message: str | None = None, value: int | None = None) -> None:
    if dialog is None:
        return
    if message is not None:
        dialog.setLabelText(message)
    if value is not None:
        real_maximum = int(getattr(dialog, "_progress_real_maximum", dialog.maximum()))
        if real_maximum <= 0:
            dialog._progress_reported_value = 0
        else:
            value = max(0, min(real_maximum, int(value)))
            dialog._progress_reported_value = value
            dialog._progress_pulse_tick = 0
            title = str(getattr(dialog, "_progress_title", ""))
            if title == "Loading data" and 10 <= value < 90:
                dialog._progress_soft_ceiling = 90
            elif value >= real_maximum:
                dialog._progress_soft_ceiling = value
            else:
                dialog._progress_soft_ceiling = min(real_maximum - 1, value + max(1, real_maximum // 30))
            dialog.setValue(value)
    QApplication.processEvents()


def _close_progress_dialog(dialog: QProgressDialog | None) -> None:
    if dialog is None:
        return
    timer = getattr(dialog, "_progress_timer", None)
    if timer is not None:
        timer.stop()
    dialog.close()
    QApplication.processEvents()


class PipelineWorker(QRunnable):
    def __init__(self, input_path: str, config: PipelineConfig):
        super().__init__()
        self.input_path = input_path
        self.config = config
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            pipeline = MEAPipeline(self.config)
            result = pipeline.run(self.input_path, progress=self.signals.progress.emit)
            self.signals.finished.emit(result)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class SortingWorker(QRunnable):
    def __init__(self, data: UnifiedMEAData, config: WaveformClusteringConfig, channels=None):
        super().__init__()
        self.data = data
        self.config = config
        self.channels = list(channels) if channels is not None else None
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            sorting = cluster_nev_waveforms(
                self.data,
                self.config,
                progress=self.signals.progress.emit,
                channels=self.channels,
            )
            self.signals.finished.emit(sorting)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class MaxwellFootprintWorker(QRunnable):
    def __init__(self, data: UnifiedMEAData, config: MaxwellFootprintConfig):
        super().__init__()
        self.data = data
        self.config = config
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = run_maxwell_footprint_analysis(
                self.data,
                self.config,
                progress=self.signals.progress.emit,
            )
            self.signals.finished.emit(result)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class DataLoadWorker(QRunnable):
    def __init__(self, path: str, selected_wells=None):
        super().__init__()
        self.path = path
        self.selected_wells = selected_wells
        self.signals = WorkerSignals()
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return bool(self._cancel_requested)

    def _check_cancelled(self) -> None:
        if self._is_cancelled():
            raise InterruptedError("Data loading cancelled")

    @Slot()
    def run(self):
        try:
            suffix = Path(self.path).suffix.lower()
            self._check_cancelled()
            self.signals.progress.emit(5, "Opening data file...")
            if suffix == ".nev":
                self.signals.progress.emit(20, "Reading NEV file...")
                raw_data = read_blackrock_nev(self.path, cancel_check=self._is_cancelled)
                data_kind = "nev"
            elif suffix == ".spk":
                well_text = ", ".join(self.selected_wells) if self.selected_wells else "all wells"
                self.signals.progress.emit(20, f"Reading Axion SPK data ({well_text})...")
                self._check_cancelled()
                raw_data = read_axion_spk(self.path, wells=self.selected_wells)
                data_kind = "nev"
            elif suffix in {".h5", ".hdf5"}:
                self.signals.progress.emit(18, "Reading Maxwell H5 metadata...")
                self.signals.progress.emit(25, "Extracting spikes and waveforms...")
                raw_data = read_maxwell_h5(self.path, cancel_check=self._is_cancelled)
                data_kind = "nev"
            else:
                self.signals.progress.emit(20, "Reading data file...")
                self._check_cancelled()
                raw_data = MEAReader(self.path).load_data()
                data_kind = "nev" if isinstance(raw_data, UnifiedMEAData) else "array"
            self._check_cancelled()
            self.signals.progress.emit(92, "Preparing loaded data...")
            self.signals.finished.emit({"path": self.path, "raw_data": raw_data, "data_kind": data_kind})
        except InterruptedError as exc:
            self.signals.canceled.emit(str(exc) or "Data loading cancelled")
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class StimulusResponseWorker(QRunnable):
    def __init__(self, paths, *, pre_ms: float, response_ms: float, artifact_ms: float):
        super().__init__()
        self.paths = [str(path) for path in paths]
        self.pre_ms = float(pre_ms)
        self.response_ms = float(response_ms)
        self.artifact_ms = float(artifact_ms)
        self.signals = WorkerSignals()
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return bool(self._cancel_requested)

    @Slot()
    def run(self):
        try:
            records = []
            errors = []
            total = max(1, len(self.paths))
            for index, path in enumerate(self.paths):
                if self._is_cancelled():
                    raise InterruptedError("Stimulus response analysis cancelled")
                self.signals.progress.emit(5 + int(80 * index / total), f"Reading {Path(path).name}...")
                try:
                    data = _load_spike_only_data(path, cancel_check=self._is_cancelled)
                    record = _stimulus_response_record_from_data(
                        path,
                        data,
                        pre_ms=self.pre_ms,
                        response_ms=self.response_ms,
                        artifact_ms=self.artifact_ms,
                    )
                    records.append(record)
                except Exception as exc:
                    errors.append(f"{Path(path).name}: {exc}")
            if self._is_cancelled():
                raise InterruptedError("Stimulus response analysis cancelled")
            self.signals.progress.emit(92, "Preparing stimulus response comparison...")
            self.signals.finished.emit(
                {
                    "records": records,
                    "errors": errors,
                    "pre_ms": self.pre_ms,
                    "response_ms": self.response_ms,
                    "artifact_ms": self.artifact_ms,
                    "paths": self.paths,
                }
            )
        except InterruptedError as exc:
            self.signals.canceled.emit(str(exc) or "Stimulus response analysis cancelled")
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class StimulusResponseInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stimulus Response Files")
        self.resize(860, 520)
        self.paths: list[str] = []
        self.cached_payload: dict | None = None

        self.pre_ms = QDoubleSpinBox()
        self.pre_ms.setRange(0.0, 10000.0)
        self.pre_ms.setDecimals(1)
        self.pre_ms.setValue(200.0)
        self.pre_ms.setSuffix(" ms")
        self.response_ms = QDoubleSpinBox()
        self.response_ms.setRange(1.0, 60000.0)
        self.response_ms.setDecimals(1)
        self.response_ms.setValue(1000.0)
        self.response_ms.setSuffix(" ms")
        self.artifact_ms = QDoubleSpinBox()
        self.artifact_ms.setRange(0.0, 1000.0)
        self.artifact_ms.setDecimals(1)
        self.artifact_ms.setSingleStep(0.5)
        self.artifact_ms.setValue(1.0)
        self.artifact_ms.setSuffix(" ms")

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["File", "Condition", "Folder"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)

        add_files = QPushButton("Add Files")
        add_files.clicked.connect(self._add_files)
        add_folder = QPushButton("Add Folder")
        add_folder.clicked.connect(self._add_folder)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_selected)
        analyze = QPushButton("Analyze")
        analyze.setObjectName("PrimaryButton")
        analyze.clicked.connect(self.accept)
        self.open_raster_button = QPushButton("Open Raster")
        self.open_raster_button.setEnabled(False)
        self.psth_button = QPushButton("PSTH")
        self.psth_button.setEnabled(False)
        self.activation_curve_button = QPushButton("Activation curve")
        self.activation_curve_button.setEnabled(False)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Pre"))
        controls.addWidget(self.pre_ms)
        controls.addWidget(QLabel("Response"))
        controls.addWidget(self.response_ms)
        controls.addWidget(QLabel("Remove tail +/-"))
        controls.addWidget(self.artifact_ms)
        controls.addStretch(1)
        controls.addWidget(add_files)
        controls.addWidget(add_folder)
        controls.addWidget(remove)
        controls.addWidget(self.open_raster_button)
        controls.addWidget(self.psth_button)
        controls.addWidget(self.activation_curve_button)
        controls.addWidget(cancel)
        controls.addWidget(analyze)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Add spike-event files directly, or add folders recursively. Waveforms are not used."))
        layout.addWidget(self.table, 1)
        layout.addLayout(controls)
        _fix_spinbox_hit_targets(self)

    def _add_paths(self, paths) -> None:
        files = _stimulus_response_supported_files(paths)
        existing = set(self.paths)
        for path in files:
            path_text = str(path)
            if path_text not in existing:
                self.paths.append(path_text)
                existing.add(path_text)
        self.paths.sort(key=str.lower)
        self._refresh_table()

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add stimulus response files",
            "data",
            "Spike files (*.nev *.spk *.h5 *.hdf5 *.npz);;All files (*)",
        )
        self._add_paths(paths)

    def _add_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Add stimulus response folder", "data")
        if path:
            self._add_paths([path])

    def _remove_selected(self) -> None:
        selected_rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in selected_rows:
            if 0 <= row < len(self.paths):
                self.paths.pop(row)
        self._refresh_table()

    def _refresh_table(self) -> None:
        self.table.setRowCount(len(self.paths))
        for row, path_text in enumerate(self.paths):
            path = Path(path_text)
            params = _extract_stimulus_parameters(path)
            values = [path.name, _stimulus_parameter_label(params, path), str(path.parent)]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, path_text)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()

    def values(self) -> tuple[list[str], float, float, float]:
        return list(self.paths), float(self.pre_ms.value()), float(self.response_ms.value()), float(self.artifact_ms.value())

    def set_cached_payload(self, payload: dict | None) -> None:
        self.cached_payload = payload if isinstance(payload, dict) else None
        has_records = bool(self.cached_payload and self.cached_payload.get("records"))
        self.open_raster_button.setEnabled(has_records)
        self.psth_button.setEnabled(has_records)
        self.activation_curve_button.setEnabled(has_records)


class StimulusChannelMapWindow(QDialog):
    def __init__(self, channel_map: ChannelMap | None, selection_callback=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stimulus Channel Map")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.resize(1180, 880)
        self.channel_map = channel_map or ChannelMap.new("No channel map")
        self.selection_callback = selection_callback
        self.position_lookup, self.electrode_positions = _channel_map_positions(self.channel_map)
        self.available_channels: set[str] = set()
        self.selected_channels: set[str] = set()
        self.stim_electrodes: set[str] = set()
        self.record_label = ""
        self.canvas = ElectrodeMapCanvas(self.channel_map)
        self.canvas.setMinimumSize(1120, 800)
        self.canvas.electrode_selected.connect(self._electrode_selected)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas, 1)

    def set_context(self, *, available_channels=None, selected_channels=None, stim_electrodes=None, record_label: str = "") -> None:
        self.available_channels = _normalized_channel_keys(available_channels)
        self.selected_channels = _normalized_channel_keys(selected_channels)
        self.stim_electrodes = {str(electrode) for electrode in stim_electrodes or [] if str(electrode)}
        self.record_label = str(record_label or "")
        highlighted = self._electrodes_for_channels(self.selected_channels)
        self.canvas.set_available_channels(available_channels or [])
        self.canvas.set_overlays(highlighted_electrodes=highlighted, stim_electrodes=self.stim_electrodes)

    def _channels_for_electrodes(self, electrodes) -> list[str]:
        if self.channel_map is None:
            return []
        selected = []
        selected_set = {str(electrode) for electrode in electrodes}
        for electrode in selected_set:
            payload = self.channel_map.electrodes.get(electrode, {})
            if not isinstance(payload, dict):
                continue
            candidates = [payload.get("channel", ""), electrode]
            aliases = payload.get("aliases", [])
            if isinstance(aliases, (list, tuple)):
                candidates.extend(aliases)
            for candidate in candidates:
                keys = _normalized_channel_keys([candidate])
                matches = sorted(keys & self.available_channels)
                for key in matches:
                    if key not in selected:
                        selected.append(key)
        return selected

    def _electrodes_for_channels(self, channels) -> list[str]:
        normalized = _normalized_channel_keys(channels)
        electrodes = []
        for electrode, payload in self.channel_map.electrodes.items():
            if not isinstance(payload, dict):
                continue
            candidates = [payload.get("channel", ""), electrode]
            aliases = payload.get("aliases", [])
            if isinstance(aliases, (list, tuple)):
                candidates.extend(aliases)
            if _normalized_channel_keys(candidates) & normalized:
                electrodes.append(str(electrode))
        return electrodes

    def _electrode_selected(self, electrode: str) -> None:
        channels = self._channels_for_electrodes([electrode])
        if channels:
            self.selected_channels = set(channels)
            if self.selection_callback is not None:
                self.selection_callback(channels)


class StimulusResponseWindow(QDialog):
    RASTER_MAX_YTICKS = 36

    def __init__(self, payload: dict, parent=None, channel_map: ChannelMap | None = None):
        super().__init__(parent)
        self.payload = payload
        self.records = list(payload.get("records", []))
        self.errors = list(payload.get("errors", []))
        self.record_lookup = {index: record for index, record in enumerate(self.records)}
        self.trial_indices = {"left": 0, "right": 0}
        self.channel_map = self._channel_map_for_records(channel_map)
        self.position_lookup, self.electrode_positions = _channel_map_positions(self.channel_map)
        self.channel_map_window = None
        self.highlight_channels: set[str] = set()
        self.active_map_side = "left"
        self.raster_lassos = []
        self._raster_channel_points: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
        self._single_stim_electrodes = self._single_stim_electrodes_from_records()
        self.left_axis = None
        self.right_axis = None
        self.setWindowTitle("Stimulus Response Comparison")
        self.resize(1280, 820)

        self.status = QLabel()
        self.status.setObjectName("MutedText")
        self.left_file_combo = QComboBox()
        self.left_file_combo.setObjectName("StimulusLeftFile")
        self.right_file_combo = QComboBox()
        self.right_file_combo.setObjectName("StimulusRightFile")
        self.left_file_combo.currentIndexChanged.connect(lambda _index: self._file_changed("left"))
        self.right_file_combo.currentIndexChanged.connect(lambda _index: self._file_changed("right"))
        total_ms = max(1.0, float(payload.get("pre_ms", 0.0)) + float(payload.get("response_ms", 1.0)))
        self.display_window_ms = QDoubleSpinBox()
        self.display_window_ms.setRange(1.0, max(1.0, total_ms))
        self.display_window_ms.setDecimals(1)
        self.display_window_ms.setSingleStep(10.0)
        self.display_window_ms.setValue(total_ms)
        self.display_window_ms.setSuffix(" ms")
        self.display_window_ms.valueChanged.connect(lambda *_: self._draw_rasters())
        self.row_order_combo = QComboBox()
        self.row_order_combo.addItem("Local response", "local_response")
        self.row_order_combo.addItem("Electrode sequence", "electrode")
        self.row_order_combo.currentIndexChanged.connect(lambda *_: self._draw_rasters())
        self.lasso_channels = QCheckBox("Lasso channels")
        self.lasso_channels.toggled.connect(lambda *_: self._refresh_raster_lassos())
        self.map_button = QPushButton("Channel map")
        self.map_button.clicked.connect(self._show_channel_map)
        self.raster_canvas = FigureCanvas(Figure(figsize=(11, 6.4), tight_layout=True))
        self.raster_canvas.mpl_connect("scroll_event", self._raster_scrolled)
        self.error_box = QTextEdit()
        self.error_box.setReadOnly(True)
        self.error_box.setMaximumHeight(90)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Left file"))
        controls.addWidget(self.left_file_combo, 1)
        controls.addSpacing(16)
        controls.addWidget(QLabel("Right file"))
        controls.addWidget(self.right_file_combo, 1)
        controls.addSpacing(16)
        controls.addWidget(QLabel("Window"))
        controls.addWidget(self.display_window_ms)
        controls.addWidget(QLabel("Row order"))
        controls.addWidget(self.row_order_combo)
        controls.addWidget(self.lasso_channels)
        controls.addWidget(self.map_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addLayout(controls)
        layout.addWidget(self.raster_canvas, 1)
        layout.addWidget(QLabel("Skipped files / warnings"))
        layout.addWidget(self.error_box)

        self._populate()
        self.showMaximized()
        _fix_spinbox_hit_targets(self)

    def _channel_map_for_records(self, channel_map: ChannelMap | None) -> ChannelMap | None:
        if channel_map is not None and getattr(channel_map, "name", "") == "maxwell_map":
            return _default_maxwell_channel_map() or channel_map
        for record in self.records:
            channels = [str(channel) for channel in record.get("channels", [])]
            if any(re.search(r"(?:^|_)e\d+$", channel, flags=re.IGNORECASE) for channel in channels):
                return _default_maxwell_channel_map() or channel_map
        return channel_map

    def _is_multi_site_record(self, record: dict | None) -> bool:
        if not record:
            return False
        params = record.get("parameters", {})
        if str(params.get("stim_mode", "")).lower() == "multi-site":
            return True
        text = " ".join([str(record.get("file", "")), str(record.get("path", "")), str(record.get("condition", ""))]).lower()
        return bool(re.search(r"multi[\s_-]*site", text))

    def _single_stim_electrodes_from_records(self) -> list[str]:
        electrodes = []
        for record in self.records:
            if self._is_multi_site_record(record):
                continue
            electrode = self._resolve_stim_electrode(record.get("parameters", {}).get("stim_electrode"))
            if electrode and electrode not in electrodes:
                electrodes.append(electrode)
        return electrodes

    def _resolve_stim_electrode(self, value) -> str:
        if value is None or value == "":
            return ""
        try:
            number = float(value)
            text = str(int(number)) if abs(number - int(number)) < 1e-9 else f"{number:g}"
        except (TypeError, ValueError):
            text = str(value).strip()
        candidates = [text, f"e{text}", f"el{text}", f"chan{text}", f"ch{text}"]
        for candidate in candidates:
            if candidate in self.electrode_positions:
                return candidate
            position = _position_for_channel(candidate, self.position_lookup)
            if position is not None:
                return str(position[2])
        return text

    def _stim_electrodes_for_record(self, record: dict | None) -> list[str]:
        if record is None:
            return []
        if self._is_multi_site_record(record):
            return list(self._single_stim_electrodes)
        electrode = self._resolve_stim_electrode(record.get("parameters", {}).get("stim_electrode"))
        return [electrode] if electrode else []

    def _record_channels(self, side: str) -> list[str]:
        record = self._selected_record(side)
        channels, _trials = self._trial_channel_payload(record)
        return channels

    def _show_channel_map(self) -> None:
        if self.channel_map_window is None:
            self.channel_map_window = StimulusChannelMapWindow(self.channel_map, self._map_channels_selected, self)
        self._update_channel_map_context()
        self.channel_map_window.show()
        self.channel_map_window.raise_()
        self.channel_map_window.activateWindow()

    def _update_channel_map_context(self) -> None:
        if self.channel_map_window is None:
            return
        side = self.active_map_side if self.active_map_side in {"left", "right"} else "left"
        record = self._selected_record(side)
        self.channel_map_window.set_context(
            available_channels=self._record_channels(side),
            selected_channels=self.highlight_channels,
            stim_electrodes=self._stim_electrodes_for_record(record),
            record_label=self._record_label(record) if record else "",
        )

    def _map_channels_selected(self, channels) -> None:
        self._set_highlight_channels(channels, open_map=False)

    def _set_highlight_channels(self, channels, *, open_map: bool = False, side: str | None = None) -> None:
        normalized = _normalized_channel_keys(channels)
        if not normalized:
            return
        self.highlight_channels = normalized
        if side in {"left", "right"}:
            self.active_map_side = side
        self._draw_rasters()
        if open_map:
            self._show_channel_map()
        else:
            self._update_channel_map_context()

    def _x_limits(self) -> tuple[float, float]:
        pre_ms = max(0.0, float(self.payload.get("pre_ms", 0.0)))
        response_ms = max(1.0, float(self.payload.get("response_ms", 1.0)))
        total = max(1.0, pre_ms + response_ms)
        window = min(total, max(1.0, float(self.display_window_ms.value())))
        left = -pre_ms * window / total
        right = left + window
        return float(left), float(right)

    def _populate(self):
        self.status.setText(
            f"{len(self.records)} files analyzed; spont/pre {self.payload.get('pre_ms', 0):g} ms, "
            f"after {self.payload.get('response_ms', 0):g} ms, tail removed +/-{self.payload.get('artifact_ms', 0):g} ms"
        )
        self.error_box.setPlainText("\n".join(self.errors) if self.errors else "No skipped files.")
        for combo in (self.left_file_combo, self.right_file_combo):
            combo.blockSignals(True)
            combo.clear()
            for index, record in enumerate(self.records):
                combo.addItem(self._record_label(record), index)
            combo.blockSignals(False)
        if self.records:
            self.left_file_combo.setCurrentIndex(0)
            self.right_file_combo.setCurrentIndex(1 if len(self.records) > 1 else 0)
        self._draw_rasters()

    def _record_label(self, record: dict) -> str:
        file_name = str(record.get("file") or Path(str(record.get("path", ""))).name)
        condition = str(record.get("condition", "")).strip()
        label = f"{file_name} | {condition}" if condition and condition != file_name else file_name
        if self._is_multi_site_record(record):
            count = len(self._single_stim_electrodes)
            if count:
                label += f" | multi-site {count} sites"
        return label

    def _file_changed(self, side: str) -> None:
        self.trial_indices[side] = 0
        self.active_map_side = side
        self._draw_rasters()
        self._update_channel_map_context()

    def _selected_record(self, side: str) -> dict | None:
        combo = self.left_file_combo if side == "left" else self.right_file_combo
        index = combo.currentData()
        if index is None:
            index = combo.currentIndex()
        try:
            return self.record_lookup.get(int(index))
        except (TypeError, ValueError):
            return None

    def _trial_channel_payload(self, record: dict | None) -> tuple[list[str], list[list[np.ndarray]]]:
        if record is None:
            return [], []
        channels = [str(channel) for channel in record.get("channels", [])]
        trial_channels = record.get("trial_channel_spikes_ms", [])
        if channels and trial_channels:
            return channels, [
                [np.asarray(values, dtype=float) for values in trial]
                for trial in trial_channels
            ]
        aggregate_trials = record.get("trial_spikes_ms", [])
        if aggregate_trials:
            return ["all channels"], [[np.asarray(values, dtype=float)] for values in aggregate_trials]
        return channels, []

    def _ordered_trial_channel_payload(self, record: dict | None) -> tuple[list[str], list[list[np.ndarray]]]:
        channels, trials = self._trial_channel_payload(record)
        if not channels:
            return channels, trials
        order = self._channel_display_order(channels, trials)
        if order == list(range(len(channels))):
            return channels, trials
        ordered_channels = [channels[index] for index in order]
        ordered_trials = []
        for trial in trials:
            row_values = list(trial)
            if len(row_values) < len(channels):
                row_values.extend(np.array([], dtype=float) for _ in range(len(channels) - len(row_values)))
            ordered_trials.append([np.asarray(row_values[index], dtype=float) for index in order])
        return ordered_channels, ordered_trials

    def _channel_display_order(self, channels: list[str], trials: list[list[np.ndarray]]) -> list[int]:
        mode = str(self.row_order_combo.currentData() or "local_response") if hasattr(self, "row_order_combo") else "local_response"
        if mode == "electrode" or len(channels) <= 2:
            return list(range(len(channels)))
        return self._local_response_channel_order(channels, trials)

    def _channel_response_features(self, channels: list[str], trials: list[list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        pre_ms = max(0.0, float(self.payload.get("pre_ms", 0.0)))
        response_ms = max(1.0, float(self.payload.get("response_ms", 1.0)))
        channel_count = len(channels)
        positions = np.full((channel_count, 2), np.nan, dtype=float)
        features = np.zeros((channel_count, 5), dtype=float)
        first_latency: list[list[float]] = [[] for _ in channels]
        peak_values: list[list[float]] = [[] for _ in channels]
        response_counts = np.zeros(channel_count, dtype=float)
        baseline_counts = np.zeros(channel_count, dtype=float)
        responsive_trials = np.zeros(channel_count, dtype=float)
        valid_trials = 0

        for index, channel in enumerate(channels):
            position = _position_for_channel(channel, self.position_lookup)
            if position is not None:
                positions[index] = [float(position[0]), float(position[1])]

        for trial in trials:
            if not trial:
                continue
            valid_trials += 1
            for index in range(channel_count):
                values = np.asarray(trial[index] if index < len(trial) else [], dtype=float)
                if not values.size:
                    continue
                values = values[np.isfinite(values)]
                if not values.size:
                    continue
                response = values[(values >= 0.0) & (values <= response_ms)]
                baseline = values[(values < 0.0) & (values >= -pre_ms)] if pre_ms > 0.0 else np.array([], dtype=float)
                response_counts[index] += float(response.size)
                baseline_counts[index] += float(baseline.size)
                if response.size:
                    responsive_trials[index] += 1.0
                    first_latency[index].append(float(np.min(response)))
                    peak_values[index].extend(response.tolist())

        denominator = max(1, valid_trials)
        response_probability = responsive_trials / float(denominator)
        response_rate = response_counts / float(denominator)
        baseline_rate = baseline_counts / float(denominator)
        latency = np.full(channel_count, response_ms, dtype=float)
        peak_time = np.full(channel_count, response_ms, dtype=float)
        bin_count = max(4, min(30, int(np.ceil(response_ms / 10.0))))
        edges = np.linspace(0.0, response_ms, bin_count + 1)
        centers = (edges[:-1] + edges[1:]) * 0.5
        for index in range(channel_count):
            if first_latency[index]:
                latency[index] = float(np.median(first_latency[index]))
            if peak_values[index]:
                counts, _ = np.histogram(np.asarray(peak_values[index], dtype=float), bins=edges)
                if counts.size:
                    peak_time[index] = float(centers[int(np.argmax(counts))])

        features[:, 0] = response_probability
        features[:, 1] = np.log1p(response_rate)
        features[:, 2] = np.log1p(baseline_rate)
        features[:, 3] = latency / max(response_ms, 1e-9)
        features[:, 4] = peak_time / max(response_ms, 1e-9)
        return positions, features

    def _local_response_channel_order(self, channels: list[str], trials: list[list[np.ndarray]]) -> list[int]:
        count = len(channels)
        if count <= 2:
            return list(range(count))
        positions, features = self._channel_response_features(channels, trials)
        spatial = positions.copy()
        if np.any(~np.isfinite(spatial)):
            finite = np.isfinite(spatial).all(axis=1)
            fallback = np.arange(count, dtype=float)
            spatial[~finite, 0] = fallback[~finite] / max(count - 1, 1)
            spatial[~finite, 1] = 1.0

        if count > 1500:
            latency_bins = np.floor(np.clip(features[:, 3], 0.0, 1.0) * 12.0).astype(int)
            peak_bins = np.floor(np.clip(features[:, 4], 0.0, 1.0) * 12.0).astype(int)
            response_bins = np.floor(np.clip(features[:, 0], 0.0, 1.0) * 10.0).astype(int)
            return sorted(
                range(count),
                key=lambda index: (
                    latency_bins[index],
                    peak_bins[index],
                    -response_bins[index],
                    float(spatial[index, 1]),
                    float(spatial[index, 0]),
                    _channel_sort_key(channels[index]),
                ),
            )

        behavior = features.copy()
        med = np.nanmedian(behavior, axis=0)
        behavior = np.where(np.isfinite(behavior), behavior, med)
        spread = np.nanpercentile(behavior, 90, axis=0) - np.nanpercentile(behavior, 10, axis=0)
        spread = np.where(spread > 1e-9, spread, 1.0)
        behavior = (behavior - np.nanmedian(behavior, axis=0)) / spread
        spatial_spread = np.nanpercentile(spatial, 90, axis=0) - np.nanpercentile(spatial, 10, axis=0)
        spatial_spread = np.where(spatial_spread > 1e-9, spatial_spread, 1.0)
        spatial = (spatial - np.nanmedian(spatial, axis=0)) / spatial_spread

        behavior_delta = behavior[:, None, :] - behavior[None, :, :]
        spatial_delta = spatial[:, None, :] - spatial[None, :, :]
        behavior_distance = np.sqrt(np.sum(behavior_delta * behavior_delta, axis=2))
        spatial_distance = np.sqrt(np.sum(spatial_delta * spatial_delta, axis=2))
        behavior_scale = np.nanpercentile(behavior_distance, 90) or 1.0
        spatial_scale = np.nanpercentile(spatial_distance, 90) or 1.0
        distance = 0.58 * behavior_distance / max(behavior_scale, 1e-9) + 0.42 * spatial_distance / max(spatial_scale, 1e-9)
        distance = np.asarray(distance, dtype=float)
        distance[~np.isfinite(distance)] = 0.0
        np.fill_diagonal(distance, 0.0)
        try:
            tree = linkage(squareform(distance, checks=False), method="average", optimal_ordering=count <= 350)
            return [int(index) for index in leaves_list(tree)]
        except Exception:
            return sorted(
                range(count),
                key=lambda index: (
                    float(features[index, 3]),
                    float(features[index, 4]),
                    -float(features[index, 0]),
                    float(spatial[index, 1]),
                    float(spatial[index, 0]),
                    _channel_sort_key(channels[index]),
                ),
            )

    def _scroll_bounds(self, record: dict | None) -> tuple[int, int]:
        _channels, trials = self._trial_channel_payload(record)
        total = len(trials)
        return total, max(0, total - 1)

    def _raster_scrolled(self, event) -> None:
        if event.inaxes == self.left_axis:
            side = "left"
        elif event.inaxes == self.right_axis:
            side = "right"
        else:
            return
        record = self._selected_record(side)
        total, max_index = self._scroll_bounds(record)
        if total <= 1:
            return
        direction = -1 if getattr(event, "step", 0) > 0 else 1
        self.trial_indices[side] = int(np.clip(self.trial_indices[side] + direction, 0, max_index))
        self._draw_rasters()
        self._update_channel_map_context()

    def _refresh_raster_lassos(self) -> None:
        for lasso in self.raster_lassos:
            lasso.disconnect_events()
        self.raster_lassos = []
        if not self.lasso_channels.isChecked():
            return
        if self.left_axis is not None:
            self.raster_lassos.append(LassoSelector(self.left_axis, lambda vertices: self._finish_raster_lasso(vertices, "left")))
        if self.right_axis is not None:
            self.raster_lassos.append(LassoSelector(self.right_axis, lambda vertices: self._finish_raster_lasso(vertices, "right")))

    def _finish_raster_lasso(self, vertices, side: str) -> None:
        selected = []
        path = MplPath(vertices)
        xs, ys, channels = self._raster_channel_points.get(side, (np.array([], dtype=float), np.array([], dtype=float), []))
        if xs.size and ys.size:
            points = np.column_stack([xs, ys])
            mask = path.contains_points(points)
            selected = [channels[index] for index in np.flatnonzero(mask)]
        if not selected:
            record = self._selected_record(side)
            row_channels, _trials = self._ordered_trial_channel_payload(record)
            if row_channels:
                vertices_array = np.asarray(vertices, dtype=float)
                y_low = float(np.nanmin(vertices_array[:, 1]))
                y_high = float(np.nanmax(vertices_array[:, 1]))
                low = max(1, int(np.floor(min(y_low, y_high))))
                high = min(len(row_channels), int(np.ceil(max(y_low, y_high))))
                selected = row_channels[low - 1 : high]
        unique = []
        seen = set()
        for channel in selected:
            key = normalize_channel_name(channel)
            if key and key not in seen:
                unique.append(channel)
                seen.add(key)
        if unique:
            self._set_highlight_channels(unique, open_map=True, side=side)

    def _draw_one_raster(self, ax, record: dict | None, side: str) -> None:
        pre_ms = max(0.0, float(self.payload.get("pre_ms", 0.0)))
        response_ms = max(1.0, float(self.payload.get("response_ms", 1.0)))
        x_left, x_right = self._x_limits()
        ax.axvspan(max(x_left, -pre_ms), min(0.0, x_right), color="#dbeafe", alpha=0.45, linewidth=0)
        ax.axvspan(max(0.0, x_left), min(response_ms, x_right), color="#fee2e2", alpha=0.35, linewidth=0)
        ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.1, alpha=0.75)
        ax.set_xlim(x_left, x_right)
        ax.set_xlabel("Time from stimulus (ms)")
        ax.set_ylabel("Channel")
        if record is None:
            self._raster_channel_points[side] = (np.array([], dtype=float), np.array([], dtype=float), [])
            ax.text(0.5, 0.5, "No stimulus response data", transform=ax.transAxes, ha="center", va="center")
            return

        channels, trials = self._ordered_trial_channel_payload(record)
        total = len(trials)
        trial_index = int(np.clip(self.trial_indices.get(side, 0), 0, max(0, total - 1)))
        self.trial_indices[side] = trial_index
        title = self._record_label(record)
        ax.set_title(f"{title}\ntrial {trial_index + 1} / {total}" if total else title)
        if total and channels:
            trial = trials[trial_index]
            if len(trial) < len(channels):
                trial = list(trial) + [np.array([], dtype=float) for _ in range(len(channels) - len(trial))]
            elif len(trial) > len(channels):
                trial = trial[: len(channels)]
            normal_x = []
            normal_y = []
            highlight_x = []
            highlight_y = []
            point_x = []
            point_y = []
            point_channels = []
            for row_index, (channel, values) in enumerate(zip(channels, trial), start=1):
                values = np.asarray(values, dtype=float)
                if values.size:
                    values = values[(values >= x_left) & (values <= x_right)]
                if not values.size:
                    continue
                is_highlighted = bool(_normalized_channel_keys([channel]) & self.highlight_channels)
                target_x = highlight_x if is_highlighted else normal_x
                target_y = highlight_y if is_highlighted else normal_y
                target_x.extend(values.tolist())
                target_y.extend([float(row_index)] * int(values.size))
                point_x.extend(values.tolist())
                point_y.extend([float(row_index)] * int(values.size))
                point_channels.extend([channel] * int(values.size))
            if normal_x:
                y_values = np.asarray(normal_y, dtype=float)
                ax.vlines(normal_x, y_values - 0.38, y_values + 0.38, color="#111827", linewidth=0.72, alpha=0.94)
            if highlight_x:
                y_values = np.asarray(highlight_y, dtype=float)
                ax.vlines(highlight_x, y_values - 0.44, y_values + 0.44, color="#dc2626", linewidth=1.35, alpha=0.98)
            for row_index, channel in enumerate(channels, start=1):
                if _normalized_channel_keys([channel]) & self.highlight_channels:
                    ax.axhspan(row_index - 0.48, row_index + 0.48, color="#fee2e2", alpha=0.35, linewidth=0)
            self._raster_channel_points[side] = (
                np.asarray(point_x, dtype=float),
                np.asarray(point_y, dtype=float),
                list(point_channels),
            )
            ax.set_ylim(len(channels) + 0.75, 0.25)
            tick_indices = _display_indices(len(channels), self.RASTER_MAX_YTICKS)
            ax.set_yticks(tick_indices + 1)
            ax.set_yticklabels([channels[index] for index in tick_indices], fontsize=8)
        else:
            self._raster_channel_points[side] = (np.array([], dtype=float), np.array([], dtype=float), [])
            ax.text(0.5, 0.5, "No spikes in selected trial", transform=ax.transAxes, ha="center", va="center")
            ax.set_ylim(1.0, 0.0)
        if pre_ms > 0:
            pre_label_x = (max(x_left, -pre_ms) + min(0.0, x_right)) * 0.5
            ax.text(pre_label_x, 1.01, "spont / pre", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color="#1d4ed8")
        after_label_x = (max(0.0, x_left) + min(response_ms, x_right)) * 0.5
        ax.text(after_label_x, 1.01, "after", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color="#b91c1c")

    def _draw_rasters(self) -> None:
        figure = self.raster_canvas.figure
        figure.clear()
        axes = figure.subplots(1, 2, sharex=True)
        self.left_axis, self.right_axis = np.ravel(axes)
        self._draw_one_raster(self.left_axis, self._selected_record("left"), "left")
        self._draw_one_raster(self.right_axis, self._selected_record("right"), "right")
        figure.tight_layout()
        self.raster_canvas.draw_idle()
        self._refresh_raster_lassos()


class StimulusPSTHWindow(QDialog):
    def __init__(self, payload: dict, parent=None):
        super().__init__(parent)
        self.payload = payload
        self.records = list(payload.get("records", []))
        self.record_lookup = {index: record for index, record in enumerate(self.records)}
        self.setWindowTitle("Stimulus PSTH")
        self.resize(1180, 760)

        self.left_file_combo = QComboBox()
        self.right_file_combo = QComboBox()
        self.channel_combo = QComboBox()
        self.bin_ms = QDoubleSpinBox()
        self.bin_ms.setRange(0.1, 1000.0)
        self.bin_ms.setDecimals(1)
        self.bin_ms.setSingleStep(1.0)
        self.bin_ms.setValue(10.0)
        self.bin_ms.setSuffix(" ms")
        self.canvas = FigureCanvas(Figure(figsize=(10.5, 6.0), tight_layout=True))

        self.left_file_combo.currentIndexChanged.connect(self._selection_changed)
        self.right_file_combo.currentIndexChanged.connect(self._selection_changed)
        self.channel_combo.currentIndexChanged.connect(lambda *_: self._draw())
        self.bin_ms.valueChanged.connect(lambda *_: self._draw())

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Left file"))
        controls.addWidget(self.left_file_combo, 1)
        controls.addSpacing(12)
        controls.addWidget(QLabel("Right file"))
        controls.addWidget(self.right_file_combo, 1)
        controls.addSpacing(12)
        controls.addWidget(QLabel("Channel"))
        controls.addWidget(self.channel_combo)
        controls.addSpacing(12)
        controls.addWidget(QLabel("Bin"))
        controls.addWidget(self.bin_ms)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.canvas, 1)

        self._populate()
        _fix_spinbox_hit_targets(self)

    def _record_label(self, record: dict) -> str:
        file_name = str(record.get("file") or Path(str(record.get("path", ""))).name)
        condition = str(record.get("condition", "")).strip()
        return f"{file_name} | {condition}" if condition and condition != file_name else file_name

    def _populate(self) -> None:
        for combo in (self.left_file_combo, self.right_file_combo):
            combo.blockSignals(True)
            combo.clear()
            for index, record in enumerate(self.records):
                combo.addItem(self._record_label(record), index)
            combo.blockSignals(False)
        if self.records:
            self.left_file_combo.setCurrentIndex(0)
            self.right_file_combo.setCurrentIndex(1 if len(self.records) > 1 else 0)
        self._refresh_channel_combo()
        self._draw()

    def _selected_record(self, side: str) -> dict | None:
        combo = self.left_file_combo if side == "left" else self.right_file_combo
        index = combo.currentData()
        if index is None:
            index = combo.currentIndex()
        try:
            return self.record_lookup.get(int(index))
        except (TypeError, ValueError):
            return None

    def _selection_changed(self, *_args) -> None:
        current = str(self.channel_combo.currentData() or "")
        self._refresh_channel_combo(preferred=current)
        self._draw()

    def _refresh_channel_combo(self, preferred: str = "") -> None:
        records = [record for record in (self._selected_record("left"), self._selected_record("right")) if record is not None]
        channels = sorted(
            {str(channel) for record in records for channel in record.get("channels", [])},
            key=_channel_sort_key,
        )
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItem("All channels", "")
        for channel in channels:
            self.channel_combo.addItem(channel, channel)
        index = self.channel_combo.findData(preferred)
        self.channel_combo.setCurrentIndex(index if index >= 0 else 0)
        self.channel_combo.blockSignals(False)

    def _record_channel_trials(self, record: dict | None, channel: str) -> tuple[list[np.ndarray], int]:
        if record is None:
            return [], 1
        channels = [str(value) for value in record.get("channels", [])]
        trial_channels = record.get("trial_channel_spikes_ms", [])
        if not channel:
            rows = []
            for trial in trial_channels:
                chunks = [np.asarray(values, dtype=float) for values in trial if np.asarray(values, dtype=float).size]
                rows.append(np.sort(np.concatenate(chunks)) if chunks else np.array([], dtype=float))
            if rows:
                return rows, max(1, len(channels))
            aggregate = [np.asarray(values, dtype=float) for values in record.get("trial_spikes_ms", [])]
            return aggregate, max(1, int(record.get("channel_count", len(channels) or 1)))

        target_keys = _normalized_channel_keys([channel])
        channel_index = None
        for index, candidate in enumerate(channels):
            if _normalized_channel_keys([candidate]) & target_keys:
                channel_index = index
                break
        if channel_index is None:
            return [], 1
        rows = []
        for trial in trial_channels:
            values = trial[channel_index] if channel_index < len(trial) else []
            rows.append(np.asarray(values, dtype=float))
        return rows, 1

    def _psth_for_record(self, record: dict | None, channel: str):
        pre_ms = max(0.0, float(self.payload.get("pre_ms", 0.0)))
        response_ms = max(1.0, float(self.payload.get("response_ms", 1.0)))
        bin_ms = max(0.1, float(self.bin_ms.value()))
        edges = np.arange(-pre_ms, response_ms + bin_ms * 0.5, bin_ms, dtype=float)
        if edges.size == 0 or edges[-1] < response_ms:
            edges = np.append(edges, response_ms)
        if edges.size < 2:
            edges = np.array([-pre_ms, response_ms], dtype=float)
        trials, channel_count = self._record_channel_trials(record, channel)
        counts = np.zeros(edges.size - 1, dtype=float)
        valid_trials = 0
        for values in trials:
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                counts += np.histogram(values, bins=edges)[0]
            valid_trials += 1
        denom = max(1, valid_trials) * max(1, channel_count) * max(bin_ms / 1000.0, 1e-9)
        centers = (edges[:-1] + edges[1:]) * 0.5
        return centers, counts / denom, valid_trials, channel_count

    def _draw_one(self, ax, record: dict | None, side: str) -> None:
        pre_ms = max(0.0, float(self.payload.get("pre_ms", 0.0)))
        response_ms = max(1.0, float(self.payload.get("response_ms", 1.0)))
        channel = str(self.channel_combo.currentData() or "")
        ax.axvspan(-pre_ms, 0.0, color="#dbeafe", alpha=0.42, linewidth=0)
        ax.axvspan(0.0, response_ms, color="#fee2e2", alpha=0.34, linewidth=0)
        ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.1, alpha=0.75)
        if record is None:
            ax.text(0.5, 0.5, "No stimulus response data", transform=ax.transAxes, ha="center", va="center")
            ax.set_xlim(-pre_ms, response_ms)
            return
        centers, rates, trial_count, channel_count = self._psth_for_record(record, channel)
        width = max(0.1, float(self.bin_ms.value()) * 0.92)
        ax.bar(centers, rates, width=width, color="#2563eb" if side == "left" else "#dc2626", alpha=0.72, linewidth=0)
        ax.plot(centers, rates, color="#0f172a", linewidth=1.15, alpha=0.86)
        label = self._record_label(record)
        scope = channel or f"all channels ({channel_count})"
        ax.set_title(f"{label}\n{scope}, {trial_count} trials")
        ax.set_xlim(-pre_ms, response_ms)
        ax.set_xlabel("Time from stimulus (ms)")
        ax.set_ylabel("Rate (Hz/channel)" if not channel else "Rate (Hz)")
        ax.grid(True, axis="y", color="#e2e8f0", linewidth=0.8)

    def _draw(self) -> None:
        figure = self.canvas.figure
        figure.clear()
        axes = figure.subplots(1, 2, sharex=True)
        left_axis, right_axis = np.ravel(axes)
        self._draw_one(left_axis, self._selected_record("left"), "left")
        self._draw_one(right_axis, self._selected_record("right"), "right")
        top = max(float(left_axis.get_ylim()[1]), float(right_axis.get_ylim()[1]), 1.0)
        for axis in (left_axis, right_axis):
            axis.set_ylim(0.0, top)
        figure.tight_layout()
        self.canvas.draw_idle()


class StimulusActivationCurveWindow(QDialog):
    def __init__(self, payload: dict, parent=None):
        super().__init__(parent)
        self.payload = payload
        self.records = list(payload.get("records", []))
        self.setWindowTitle("Stimulus Activation Curve")
        self.resize(1120, 760)

        self.site_combo = QComboBox()
        self.channel_combo = QComboBox()
        response_ms = max(1.0, float(payload.get("response_ms", 1.0)))
        artifact_ms = max(0.0, float(payload.get("artifact_ms", 0.0)))
        default_start = min(response_ms, max(1.0, artifact_ms))
        default_stop = min(response_ms, max(default_start + 1.0, 50.0))
        self.evoked_start_ms = QDoubleSpinBox()
        self.evoked_start_ms.setRange(0.0, response_ms)
        self.evoked_start_ms.setDecimals(1)
        self.evoked_start_ms.setSingleStep(1.0)
        self.evoked_start_ms.setValue(default_start)
        self.evoked_start_ms.setSuffix(" ms")
        self.evoked_stop_ms = QDoubleSpinBox()
        self.evoked_stop_ms.setRange(0.1, max(0.1, response_ms))
        self.evoked_stop_ms.setDecimals(1)
        self.evoked_stop_ms.setSingleStep(1.0)
        self.evoked_stop_ms.setValue(default_stop)
        self.evoked_stop_ms.setSuffix(" ms")
        self.canvas = FigureCanvas(Figure(figsize=(10.5, 6.0), tight_layout=True))
        self.site_combo.currentIndexChanged.connect(lambda *_: self._draw())
        self.channel_combo.currentIndexChanged.connect(lambda *_: self._draw())
        self.evoked_start_ms.valueChanged.connect(lambda *_: self._draw())
        self.evoked_stop_ms.valueChanged.connect(lambda *_: self._draw())

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Stim site"))
        controls.addWidget(self.site_combo)
        controls.addSpacing(14)
        controls.addWidget(QLabel("Channel"))
        controls.addWidget(self.channel_combo)
        controls.addSpacing(14)
        controls.addWidget(QLabel("Evoked"))
        controls.addWidget(self.evoked_start_ms)
        controls.addWidget(QLabel("to"))
        controls.addWidget(self.evoked_stop_ms)
        controls.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.canvas, 1)

        self._populate()
        _fix_spinbox_hit_targets(self)

    def _populate(self) -> None:
        sites = sorted({self._stim_site(record) for record in self.records if self._stim_site(record)}, key=_channel_sort_key)
        channels = sorted(
            {str(channel) for record in self.records for channel in record.get("channels", [])},
            key=_channel_sort_key,
        )
        self.site_combo.blockSignals(True)
        self.site_combo.clear()
        self.site_combo.addItem("All sites", "")
        for site in sites:
            self.site_combo.addItem(site, site)
        self.site_combo.blockSignals(False)

        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItem("All channels", "")
        for channel in channels:
            self.channel_combo.addItem(channel, channel)
        self.channel_combo.blockSignals(False)
        self._draw()

    def _stim_site(self, record: dict) -> str:
        params = record.get("parameters", {}) if isinstance(record, dict) else {}
        value = params.get("stim_electrode") if isinstance(params, dict) else None
        if value in (None, ""):
            return ""
        try:
            number = float(value)
            return f"el{int(number)}" if abs(number - int(number)) < 1e-9 else f"el{number:g}"
        except (TypeError, ValueError):
            text = str(value).strip()
            return text if text.lower().startswith("el") else f"el{text}"

    def _amplitude_value(self, record: dict) -> tuple[float | None, str]:
        params = record.get("parameters", {}) if isinstance(record, dict) else {}
        if not isinstance(params, dict):
            return None, ""
        preferred = [
            key
            for key in params
            if re.search(r"(?:^|_)(?:amplitude|current|intensity|level)(?:_|$)", str(key), flags=re.IGNORECASE)
        ]
        for key in sorted(preferred, key=lambda item: (0 if str(item).startswith("amplitude") else 1, str(item))):
            value = params.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                label = str(key).replace("amplitude_", "").replace("current_", "")
                return float(value), label or str(key)
        return None, ""

    def _record_channel_trials(self, record: dict, channel: str) -> tuple[list[np.ndarray], int]:
        channels = [str(value) for value in record.get("channels", [])]
        trial_channels = record.get("trial_channel_spikes_ms", [])
        if not channel:
            rows = []
            for trial in trial_channels:
                chunks = [np.asarray(values, dtype=float) for values in trial if np.asarray(values, dtype=float).size]
                rows.append(np.sort(np.concatenate(chunks)) if chunks else np.array([], dtype=float))
            if rows:
                return rows, max(1, len(channels))
            aggregate = [np.asarray(values, dtype=float) for values in record.get("trial_spikes_ms", [])]
            return aggregate, max(1, int(record.get("channel_count", len(channels) or 1)))

        target_keys = _normalized_channel_keys([channel])
        channel_index = None
        for index, candidate in enumerate(channels):
            if _normalized_channel_keys([candidate]) & target_keys:
                channel_index = index
                break
        if channel_index is None:
            return [], 1
        rows = []
        for trial in trial_channels:
            values = trial[channel_index] if channel_index < len(trial) else []
            rows.append(np.asarray(values, dtype=float))
        return rows, 1

    def _record_trial_units(self, record: dict, channel: str) -> list[list[np.ndarray]]:
        channels = [str(value) for value in record.get("channels", [])]
        trial_channels = record.get("trial_channel_spikes_ms", [])
        if trial_channels:
            if channel:
                target_keys = _normalized_channel_keys([channel])
                channel_index = None
                for index, candidate in enumerate(channels):
                    if _normalized_channel_keys([candidate]) & target_keys:
                        channel_index = index
                        break
                if channel_index is None:
                    return []
                return [
                    [np.asarray(trial[channel_index] if channel_index < len(trial) else [], dtype=float)]
                    for trial in trial_channels
                ]
            rows = []
            channel_count = max(1, len(channels))
            for trial in trial_channels:
                row = []
                for index in range(channel_count):
                    row.append(np.asarray(trial[index] if index < len(trial) else [], dtype=float))
                rows.append(row)
            return rows
        return [[np.asarray(values, dtype=float)] for values in record.get("trial_spikes_ms", [])]

    def _evoked_window(self) -> tuple[float, float]:
        start = max(0.0, float(self.evoked_start_ms.value()))
        stop = max(0.1, float(self.evoked_stop_ms.value()))
        if stop <= start:
            stop = start + 0.1
        response_ms = max(1.0, float(self.payload.get("response_ms", 1.0)))
        return min(start, response_ms), min(stop, response_ms)

    def _baseline_corrected_evoked_strength(self, record: dict, channel: str) -> float:
        start_ms, stop_ms = self._evoked_window()
        window_ms = max(0.1, stop_ms - start_ms)
        pre_ms = max(0.0, float(self.payload.get("pre_ms", 0.0)))
        trials, _channel_count = self._record_channel_trials(record, channel)
        if not trials:
            return 0.0
        response_spikes = 0
        baseline_spikes = 0
        for values in trials:
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            response_spikes += int(np.count_nonzero((values >= start_ms) & (values <= stop_ms)))
            if pre_ms > 0.0:
                baseline_spikes += int(np.count_nonzero((values < 0.0) & (values >= -pre_ms)))
        trial_count = max(1, len(trials))
        response_per_stimulus = float(response_spikes) / float(trial_count)
        baseline_per_stimulus = 0.0
        if pre_ms > 0.0:
            baseline_per_stimulus = float(baseline_spikes) / float(trial_count) * window_ms / max(pre_ms, 1e-9)
        return response_per_stimulus - baseline_per_stimulus

    def activation_curve_data(self, *, site: str = "", channel: str = "") -> dict[str, dict[str, np.ndarray | str]]:
        grouped: dict[str, dict[float, list[float]]] = {}
        unit_label = ""
        for record in self.records:
            stim_site = self._stim_site(record)
            if not stim_site or (site and stim_site != site):
                continue
            amplitude, unit = self._amplitude_value(record)
            if amplitude is None:
                continue
            if unit and not unit_label:
                unit_label = unit
            value = self._baseline_corrected_evoked_strength(record, channel)
            grouped.setdefault(stim_site, {}).setdefault(float(amplitude), []).append(float(value))

        result = {}
        for stim_site, values_by_amp in sorted(grouped.items(), key=lambda item: _channel_sort_key(item[0])):
            amplitudes = np.asarray(sorted(values_by_amp), dtype=float)
            means = np.asarray([np.mean(values_by_amp[value]) for value in amplitudes], dtype=float)
            errors = np.asarray(
                [
                    (np.std(values_by_amp[value], ddof=1) / np.sqrt(len(values_by_amp[value]))) if len(values_by_amp[value]) > 1 else 0.0
                    for value in amplitudes
                ],
                dtype=float,
            )
            result[stim_site] = {"amplitude": amplitudes, "mean": means, "sem": errors, "unit": unit_label}
        return result

    def _sigmoid_fit(self, amplitudes: np.ndarray, responses: np.ndarray):
        x = np.asarray(amplitudes, dtype=float)
        y = np.asarray(responses, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size < 3 or np.unique(x).size < 3 or float(np.nanmax(y) - np.nanmin(y)) <= 0.0:
            return None
        span = max(float(np.max(x) - np.min(x)), 1e-9)
        ymin = float(np.nanmin(y))
        ymax = float(np.nanmax(y))
        amplitude = max(ymax - ymin, 1e-9)

        def model(values, bottom, top, x50, slope):
            return bottom + (top - bottom) / (1.0 + np.exp(-(values - x50) / max(float(slope), 1e-9)))

        try:
            params, _cov = curve_fit(
                model,
                x,
                y,
                p0=[ymin, ymax, float(np.median(x)), span / 4.0],
                bounds=(
                    [max(0.0, ymin - amplitude * 2.0), ymin, float(np.min(x) - span), span / 1000.0],
                    [ymax, ymax + amplitude * 2.0, float(np.max(x) + span), span * 10.0],
                ),
                maxfev=10000,
            )
        except Exception:
            return None
        draw_x = np.linspace(float(np.min(x)), float(np.max(x)), 160)
        draw_y = model(draw_x, *params)
        return draw_x, draw_y, params

    def _draw(self) -> None:
        site = str(self.site_combo.currentData() or "")
        channel = str(self.channel_combo.currentData() or "")
        curves = self.activation_curve_data(site=site, channel=channel)
        figure = self.canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        if not curves:
            ax.text(0.5, 0.5, "No amplitude series found for activation curve", transform=ax.transAxes, ha="center", va="center")
            ax.set_xlabel("Stimulus amplitude")
            ax.set_ylabel("Evoked spikes / stimulus")
            ax.set_ylim(0.0, 1.0)
            self.canvas.draw_idle()
            return
        colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#f97316", "#0891b2", "#be123c", "#4f46e5"]
        unit_label = ""
        y_values = []
        for index, (stim_site, curve) in enumerate(curves.items()):
            x = np.asarray(curve["amplitude"], dtype=float)
            y = np.asarray(curve["mean"], dtype=float)
            err = np.asarray(curve["sem"], dtype=float)
            y_values.extend((y - err).tolist())
            y_values.extend((y + err).tolist())
            unit_label = str(curve.get("unit") or unit_label)
            color = colors[index % len(colors)]
            ax.errorbar(x, y, yerr=err, marker="o", linewidth=1.8, capsize=3, color=color, label=stim_site)
            fit = self._sigmoid_fit(x, y)
            if fit is not None:
                fit_x, fit_y, params = fit
                ax.plot(fit_x, fit_y, color=color, linewidth=2.0, alpha=0.72, linestyle="--", label=f"{stim_site} sigmoid")
                ax.text(
                    float(fit_x[len(fit_x) // 2]),
                    float(np.nanmedian(fit_y)),
                    f"x50={params[2]:g}",
                    color=color,
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )
        finite_y = [float(value) for value in y_values if np.isfinite(value)]
        ymax = max(finite_y + [1.0])
        ymin = min(finite_y + [0.0])
        pad = max((ymax - ymin) * 0.08, 0.5)
        ax.set_ylim(min(0.0, ymin - pad), ymax + pad)
        ax.axhline(0.0, color="#64748b", linestyle="--", linewidth=1.0)
        ax.set_xlabel(f"Stimulus amplitude ({unit_label})" if unit_label else "Stimulus amplitude")
        ax.set_ylabel("Evoked spikes / stimulus")
        start_ms, stop_ms = self._evoked_window()
        if channel:
            ax.set_title(f"{channel}, evoked window {start_ms:g}-{stop_ms:g} ms")
        else:
            ax.set_title(f"All channels, evoked window {start_ms:g}-{stop_ms:g} ms")
        ax.grid(True, axis="y", color="#e2e8f0", linewidth=0.8)
        if len(curves) > 1:
            ax.legend(loc="best", fontsize=8)
        figure.tight_layout()
        self.canvas.draw_idle()


class AutoSortingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Sorting")
        self.setMinimumWidth(520)

        self.method = QComboBox()
        self.method.addItem("Waveform clustering (NEV)")
        self.method.setEnabled(False)

        self.reduction_method = QComboBox()
        self.reduction_method.addItem("PCA", "pca")
        self.reduction_method.addItem("ICA", "ica")
        self.reduction_method.addItem("None (scaled waveform)", "none")
        self.reduction_method.currentIndexChanged.connect(self._update_parameter_visibility)

        self.clustering_method = QComboBox()
        self.clustering_method.addItem("KMeans", "kmeans")
        self.clustering_method.addItem("Gaussian Mixture", "gmm")
        self.clustering_method.addItem("Agglomerative", "agglomerative")
        self.clustering_method.addItem("DBSCAN", "dbscan")
        self.clustering_method.currentIndexChanged.connect(self._update_parameter_visibility)

        self.max_clusters = QSpinBox()
        self.max_clusters.setRange(1, 12)
        self.max_clusters.setValue(4)

        self.fixed_clusters = QSpinBox()
        self.fixed_clusters.setRange(2, 12)
        self.fixed_clusters.setValue(2)

        self.pca_components = QSpinBox()
        self.pca_components.setRange(1, 20)
        self.pca_components.setValue(5)

        self.ica_components = QSpinBox()
        self.ica_components.setRange(1, 20)
        self.ica_components.setValue(5)

        self.ica_max_iter = QSpinBox()
        self.ica_max_iter.setRange(50, 5000)
        self.ica_max_iter.setValue(300)

        self.min_spikes = QSpinBox()
        self.min_spikes.setRange(2, 10000)
        self.min_spikes.setValue(25)

        self.gmm_covariance_type = QComboBox()
        self.gmm_covariance_type.addItems(["full", "tied", "diag", "spherical"])

        self.dbscan_eps = QDoubleSpinBox()
        self.dbscan_eps.setRange(0.001, 100.0)
        self.dbscan_eps.setDecimals(3)
        self.dbscan_eps.setSingleStep(0.05)
        self.dbscan_eps.setValue(0.8)

        self.dbscan_min_samples = QSpinBox()
        self.dbscan_min_samples.setRange(1, 1000)
        self.dbscan_min_samples.setValue(10)

        self.min_silhouette = QDoubleSpinBox()
        self.min_silhouette.setRange(0.0, 1.0)
        self.min_silhouette.setDecimals(3)
        self.min_silhouette.setSingleStep(0.01)
        self.min_silhouette.setValue(0.08)

        form = QFormLayout()
        form.addRow("Method", self.method)
        form.addRow("Reduction", self.reduction_method)
        form.addRow("Clustering", self.clustering_method)
        form.addRow("Min spikes/channel", self.min_spikes)

        self.pca_components_label = QLabel("PCA components")
        form.addRow(self.pca_components_label, self.pca_components)
        self.ica_components_label = QLabel("ICA components")
        form.addRow(self.ica_components_label, self.ica_components)
        self.ica_max_iter_label = QLabel("ICA max iterations")
        form.addRow(self.ica_max_iter_label, self.ica_max_iter)

        self.max_clusters_label = QLabel("Max clusters")
        form.addRow(self.max_clusters_label, self.max_clusters)
        self.fixed_clusters_label = QLabel("Cluster count")
        form.addRow(self.fixed_clusters_label, self.fixed_clusters)
        self.gmm_covariance_type_label = QLabel("GMM covariance")
        form.addRow(self.gmm_covariance_type_label, self.gmm_covariance_type)
        self.dbscan_eps_label = QLabel("DBSCAN eps")
        form.addRow(self.dbscan_eps_label, self.dbscan_eps)
        self.dbscan_min_samples_label = QLabel("DBSCAN min samples")
        form.addRow(self.dbscan_min_samples_label, self.dbscan_min_samples)
        self.min_silhouette_label = QLabel("Min silhouette")
        form.addRow(self.min_silhouette_label, self.min_silhouette)

        self.run_button = QPushButton("Run")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        actions = QHBoxLayout()
        actions.addStretch()
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.run_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(actions)
        self._update_parameter_visibility()
        _fix_spinbox_hit_targets(self)

    def get_config(self) -> WaveformClusteringConfig:
        return WaveformClusteringConfig(
            reduction_method=self.reduction_method.currentData(),
            clustering_method=self.clustering_method.currentData(),
            max_clusters=self.max_clusters.value(),
            fixed_clusters=self.fixed_clusters.value(),
            pca_components=self.pca_components.value(),
            ica_components=self.ica_components.value(),
            ica_max_iter=self.ica_max_iter.value(),
            min_spikes=self.min_spikes.value(),
            gmm_covariance_type=self.gmm_covariance_type.currentText(),
            dbscan_eps=self.dbscan_eps.value(),
            dbscan_min_samples=self.dbscan_min_samples.value(),
            min_silhouette=self.min_silhouette.value(),
        )

    def _update_parameter_visibility(self):
        reduction = self.reduction_method.currentData()
        clustering = self.clustering_method.currentData()

        self._set_row_visible(self.pca_components_label, self.pca_components, reduction == "pca")
        self._set_row_visible(self.ica_components_label, self.ica_components, reduction == "ica")
        self._set_row_visible(self.ica_max_iter_label, self.ica_max_iter, reduction == "ica")

        self._set_row_visible(self.max_clusters_label, self.max_clusters, clustering in {"kmeans", "gmm"})
        self._set_row_visible(self.fixed_clusters_label, self.fixed_clusters, clustering == "agglomerative")
        self._set_row_visible(self.gmm_covariance_type_label, self.gmm_covariance_type, clustering == "gmm")
        self._set_row_visible(self.dbscan_eps_label, self.dbscan_eps, clustering == "dbscan")
        self._set_row_visible(self.dbscan_min_samples_label, self.dbscan_min_samples, clustering == "dbscan")
        self._set_row_visible(self.min_silhouette_label, self.min_silhouette, clustering in {"kmeans", "gmm"})

    @staticmethod
    def _set_row_visible(label: QWidget, field: QWidget, visible: bool):
        label.setVisible(visible)
        field.setVisible(visible)


class MaxwellFootprintDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Maxwell Footprint Analysis")
        self.setMinimumWidth(520)

        self.selection_preference = QComboBox()
        self.selection_preference.addItem("Average Spike Amplitude", "amplitude")
        self.selection_preference.addItem("Firing Rate", "firing_rate")

        self.unit_count = QSpinBox()
        self.unit_count.setRange(1, 512)
        self.unit_count.setValue(256)

        self.electrodes_per_unit = QSpinBox()
        self.electrodes_per_unit.setRange(1, 16)
        self.electrodes_per_unit.setValue(4)

        self.min_spikes = QSpinBox()
        self.min_spikes.setRange(1, 100000)
        self.min_spikes.setValue(20)

        self.min_spacing = QDoubleSpinBox()
        self.min_spacing.setRange(0.0, 2000.0)
        self.min_spacing.setDecimals(1)
        self.min_spacing.setSingleStep(17.5)
        self.min_spacing.setValue(100.0)

        self.core_radius = QDoubleSpinBox()
        self.core_radius.setRange(17.5, 500.0)
        self.core_radius.setDecimals(1)
        self.core_radius.setSingleStep(17.5)
        self.core_radius.setValue(70.0)

        self.min_firing_rate = QDoubleSpinBox()
        self.min_firing_rate.setRange(0.0, 1000.0)
        self.min_firing_rate.setDecimals(3)
        self.min_firing_rate.setValue(0.0)

        self.min_amplitude = QDoubleSpinBox()
        self.min_amplitude.setRange(0.0, 10000.0)
        self.min_amplitude.setDecimals(1)
        self.min_amplitude.setValue(0.0)

        self.activity_scan_spikes = QSpinBox()
        self.activity_scan_spikes.setRange(10, 100000)
        self.activity_scan_spikes.setValue(200)

        self.coincidence_window = QDoubleSpinBox()
        self.coincidence_window.setRange(0.1, 20.0)
        self.coincidence_window.setDecimals(2)
        self.coincidence_window.setSingleStep(0.1)
        self.coincidence_window.setValue(2.0)

        self.min_core_matches = QSpinBox()
        self.min_core_matches.setRange(1, 16)
        self.min_core_matches.setValue(1)

        self.core_waveform_corr = QDoubleSpinBox()
        self.core_waveform_corr.setRange(-1.0, 1.0)
        self.core_waveform_corr.setDecimals(2)
        self.core_waveform_corr.setSingleStep(0.05)
        self.core_waveform_corr.setValue(0.45)

        self.core_pattern_corr = QDoubleSpinBox()
        self.core_pattern_corr.setRange(-1.0, 1.0)
        self.core_pattern_corr.setDecimals(2)
        self.core_pattern_corr.setSingleStep(0.05)
        self.core_pattern_corr.setValue(0.0)
        self.core_pattern_corr.setSpecialValueText("Disabled")

        self.min_footprint_spikes = QSpinBox()
        self.min_footprint_spikes.setRange(1, 100000)
        self.min_footprint_spikes.setValue(10)

        self.amplitude_threshold = QDoubleSpinBox()
        self.amplitude_threshold.setRange(0.0, 10000.0)
        self.amplitude_threshold.setDecimals(1)
        self.amplitude_threshold.setValue(8.0)

        self.background_removal = QCheckBox("Enable")
        self.background_removal.setChecked(True)

        self.background_size = QSpinBox()
        self.background_size.setRange(2, 10)
        self.background_size.setValue(3)

        self.neighbor_radius = QDoubleSpinBox()
        self.neighbor_radius.setRange(17.5, 500.0)
        self.neighbor_radius.setDecimals(1)
        self.neighbor_radius.setSingleStep(17.5)
        self.neighbor_radius.setValue(42.0)

        self.local_corr_threshold = QDoubleSpinBox()
        self.local_corr_threshold.setRange(-1.0, 1.0)
        self.local_corr_threshold.setDecimals(2)
        self.local_corr_threshold.setSingleStep(0.05)
        self.local_corr_threshold.setValue(0.15)

        self.max_triggers = QSpinBox()
        self.max_triggers.setRange(100, 100000)
        self.max_triggers.setValue(2000)

        self.max_scan_channels = QSpinBox()
        self.max_scan_channels.setRange(0, 30000)
        self.max_scan_channels.setValue(0)
        self.max_scan_channels.setSpecialValueText("All")

        self.denoise_enabled = QCheckBox("Enable")
        self.denoise_enabled.setChecked(True)

        self.denoise_min_amplitude = QDoubleSpinBox()
        self.denoise_min_amplitude.setRange(0.0, 10000.0)
        self.denoise_min_amplitude.setDecimals(1)
        self.denoise_min_amplitude.setValue(10.0)

        self.denoise_min_snr = QDoubleSpinBox()
        self.denoise_min_snr.setRange(0.0, 100.0)
        self.denoise_min_snr.setDecimals(2)
        self.denoise_min_snr.setValue(4.0)

        self.denoise_max_amplitude = QDoubleSpinBox()
        self.denoise_max_amplitude.setRange(0.0, 100000.0)
        self.denoise_max_amplitude.setDecimals(1)
        self.denoise_max_amplitude.setValue(1000.0)
        self.denoise_max_amplitude.setSpecialValueText("Disabled")

        self.denoise_polarity_ratio = QDoubleSpinBox()
        self.denoise_polarity_ratio.setRange(0.0, 10.0)
        self.denoise_polarity_ratio.setDecimals(2)
        self.denoise_polarity_ratio.setValue(0.5)

        form = QFormLayout()
        form.addRow("Selection preference", self.selection_preference)
        form.addRow("Number of units", self.unit_count)
        form.addRow("Core electrodes/unit", self.electrodes_per_unit)
        form.addRow("Min spacing between units (um)", self.min_spacing)
        form.addRow("Core search radius (um)", self.core_radius)
        form.addRow("Min firing rate (Hz)", self.min_firing_rate)
        form.addRow("Min average spike amplitude (uV)", self.min_amplitude)
        form.addRow("Min spikes/electrode", self.min_spikes)
        form.addRow("Activity scan spikes/electrode", self.activity_scan_spikes)
        form.addRow("Coincidence window (ms)", self.coincidence_window)
        form.addRow("Min core electrode matches", self.min_core_matches)
        form.addRow("Core waveform corr", self.core_waveform_corr)
        form.addRow("Core amplitude pattern corr", self.core_pattern_corr)
        form.addRow("Min footprint spikes/electrode", self.min_footprint_spikes)
        form.addRow("Electrode mask amplitude (uV)", self.amplitude_threshold)
        form.addRow("Neighbor radius (um)", self.neighbor_radius)
        form.addRow("Local template corr", self.local_corr_threshold)
        form.addRow("Max triggers/unit", self.max_triggers)
        form.addRow("Background removal", self.background_removal)
        form.addRow("Background removal size", self.background_size)
        form.addRow("Max scan channels", self.max_scan_channels)
        form.addRow("Denoise before footprint", self.denoise_enabled)
        form.addRow("Denoise min amplitude (uV)", self.denoise_min_amplitude)
        form.addRow("Denoise min SNR", self.denoise_min_snr)
        form.addRow("Denoise max p2p (uV)", self.denoise_max_amplitude)
        form.addRow("Denoise negative/positive", self.denoise_polarity_ratio)

        self.run_button = QPushButton("Run Footprint Analysis")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        actions = QHBoxLayout()
        actions.addStretch()
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.run_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(actions)
        _fix_spinbox_hit_targets(self)

    def get_config(self) -> MaxwellFootprintConfig:
        return MaxwellFootprintConfig(
            selection_preference=str(self.selection_preference.currentData()),
            unit_count=int(self.unit_count.value()),
            electrodes_per_unit=int(self.electrodes_per_unit.value()),
            min_spacing_um=float(self.min_spacing.value()),
            core_radius_um=float(self.core_radius.value()),
            min_firing_rate_hz=float(self.min_firing_rate.value()),
            min_spike_amplitude_uv=float(self.min_amplitude.value()),
            min_spikes=int(self.min_spikes.value()),
            coincidence_window_ms=float(self.coincidence_window.value()),
            min_core_matches=int(self.min_core_matches.value()),
            core_waveform_corr_threshold=float(self.core_waveform_corr.value()),
            core_amplitude_pattern_corr_threshold=float(self.core_pattern_corr.value()),
            min_footprint_spikes=int(self.min_footprint_spikes.value()),
            amplitude_threshold_uv=float(self.amplitude_threshold.value()),
            background_removal=bool(self.background_removal.isChecked()),
            background_removal_size=int(self.background_size.value()),
            max_scan_channels=int(self.max_scan_channels.value()),
            activity_scan_spikes_per_electrode=int(self.activity_scan_spikes.value()),
            neighbor_radius_um=float(self.neighbor_radius.value()),
            local_corr_threshold=float(self.local_corr_threshold.value()),
            max_triggers=int(self.max_triggers.value()),
            denoise_enabled=bool(self.denoise_enabled.isChecked()),
            denoise_min_amplitude_uv=float(self.denoise_min_amplitude.value()),
            denoise_min_snr=float(self.denoise_min_snr.value()),
            denoise_max_amplitude_uv=float(self.denoise_max_amplitude.value()),
            denoise_min_negative_to_positive=float(self.denoise_polarity_ratio.value()),
        )


class SettingsDialog(QDialog):
    def __init__(self, config: PipelineConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pipeline Settings")
        self.setMinimumWidth(420)
        self.config = config

        self.sampling_rate = QDoubleSpinBox()
        self.sampling_rate.setRange(1, 500000)
        self.sampling_rate.setValue(config.sampling_rate)
        self.sampling_rate.setSuffix(" Hz")

        self.filter_type = QComboBox()
        self.filter_type.addItems(["bandpass", "highpass", "lowpass", "none"])
        self.filter_type.setCurrentText(config.filter_type)

        self.low_cut = QDoubleSpinBox()
        self.low_cut.setRange(0.1, 100000)
        self.low_cut.setValue(config.low_cut)
        self.low_cut.setSuffix(" Hz")

        self.high_cut = QDoubleSpinBox()
        self.high_cut.setRange(0.1, 200000)
        self.high_cut.setValue(config.high_cut)
        self.high_cut.setSuffix(" Hz")

        self.outlier_threshold = QDoubleSpinBox()
        self.outlier_threshold.setRange(1, 20)
        self.outlier_threshold.setValue(config.outlier_threshold)

        self.normalize = QCheckBox("Enable channel-wise z-score normalization")
        self.normalize.setChecked(config.normalize)

        self.spike_threshold = QDoubleSpinBox()
        self.spike_threshold.setRange(1, 20)
        self.spike_threshold.setValue(config.spike_threshold)

        self.output_dir = QLineEdit(config.output_dir)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir)
        output_row.addWidget(browse)

        form = QFormLayout()
        form.addRow("Sampling rate", self.sampling_rate)
        form.addRow("Filter", self.filter_type)
        form.addRow("Low cut", self.low_cut)
        form.addRow("High cut", self.high_cut)
        form.addRow("Outlier threshold", self.outlier_threshold)
        form.addRow("", self.normalize)
        form.addRow("Spike threshold", self.spike_threshold)
        form.addRow("Output directory", output_row)

        save = QPushButton("Apply")
        save.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        actions = QHBoxLayout()
        actions.addStretch()
        actions.addWidget(cancel)
        actions.addWidget(save)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(actions)
        _fix_spinbox_hit_targets(self)

    def get_config(self) -> PipelineConfig:
        return PipelineConfig(
            sampling_rate=self.sampling_rate.value(),
            filter_type=self.filter_type.currentText(),
            low_cut=self.low_cut.value(),
            high_cut=self.high_cut.value(),
            outlier_threshold=self.outlier_threshold.value(),
            normalize=self.normalize.isChecked(),
            spike_threshold=self.spike_threshold.value(),
            output_dir=self.output_dir.text().strip() or "data/processed",
        )

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select output directory", self.output_dir.text())
        if path:
            self.output_dir.setText(path)


class PlotWindow(QDialog):
    def __init__(self, title: str, figure, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 560)
        layout = QVBoxLayout(self)
        layout.addWidget(FigureCanvas(figure))


class SpikeRasterCanvas(QWidget):
    wheel_zoom_requested = Signal(float, int)
    pan_requested = Signal(int)
    channel_selected = Signal(str)

    def __init__(self, spike_series, y_axis_label: str = "Channel", parent=None):
        super().__init__(parent)
        self.spike_series = [(label, np.asarray(times, dtype=float)) for label, times in spike_series]
        self.selected_channel = self.spike_series[0][0] if self.spike_series else ""
        self.y_axis_label = y_axis_label
        self.row_offset = 0
        self.visible_row_count = len(self.spike_series)
        self.window_start = 0.0
        self.window_duration = 10.0
        self.grid_step = 0.1
        self.burst_intervals = []
        self._burst_starts = np.array([], dtype=float)
        self._burst_stops = np.array([], dtype=float)
        self.stim_times = np.array([], dtype=float)
        self.playhead_time = None
        self.plot_left = self._preferred_left_margin()
        self.plot_right = 18
        self.plot_top = 24
        self.plot_bottom = 48
        self._drag_start_x = None
        self._drag_start_y = None
        self._drag_start_slider_ms = 0
        self._drag_moved = False
        self.setMinimumSize(880, 500)
        self.setMouseTracking(True)

    def _preferred_left_margin(self) -> int:
        labels = [label for label, _ in self.spike_series]
        if not labels:
            return 76
        metrics = self.fontMetrics()
        widest = max(metrics.horizontalAdvance(label) for label in labels)
        return int(min(240, max(76, widest + 40)))

    def set_spike_series(self, spike_series) -> None:
        self.spike_series = [(label, np.asarray(times, dtype=float)) for label, times in spike_series]
        labels = [label for label, _ in self.spike_series]
        if self.selected_channel not in labels:
            self.selected_channel = labels[0] if labels else ""
        self.plot_left = self._preferred_left_margin()
        self.set_visible_rows(self.row_offset, self.visible_row_count)
        self.update()

    @staticmethod
    def _row_center(top: int, plot_height: int, row: int, row_count: int) -> float:
        return top + plot_height / max(row_count, 1) * (row + 0.5)

    @staticmethod
    def _label_stride_for_rows(row_count: int, plot_height: int, label_height: int) -> int:
        if row_count <= 0:
            return 1
        max_labels = max(1, int(plot_height // max(1, label_height)))
        return max(1, int(np.ceil(row_count / max_labels)))

    def _visible_spike_series(self):
        if not self.spike_series:
            return []
        count = max(1, min(int(self.visible_row_count), len(self.spike_series)))
        offset = max(0, min(int(self.row_offset), len(self.spike_series) - count))
        return self.spike_series[offset : offset + count]

    def set_visible_rows(self, row_offset: int, visible_row_count: int) -> None:
        total_rows = len(self.spike_series)
        if total_rows == 0:
            self.row_offset = 0
            self.visible_row_count = 0
        else:
            self.visible_row_count = max(1, min(int(visible_row_count), total_rows))
            self.row_offset = max(0, min(int(row_offset), total_rows - self.visible_row_count))
        self.update()

    def set_selected_channel(self, channel: str) -> None:
        self.selected_channel = channel
        self.update()

    def set_view(self, start_s: float, duration_s: float, grid_step_s: float) -> None:
        self.window_start = max(0.0, float(start_s))
        self.window_duration = max(0.001, float(duration_s))
        self.grid_step = max(0.001, float(grid_step_s))
        self.update()

    def set_bursts(self, intervals) -> None:
        self.burst_intervals = sorted((float(start), float(stop)) for start, stop in intervals)
        if self.burst_intervals:
            self._burst_starts = np.asarray([item[0] for item in self.burst_intervals], dtype=float)
            self._burst_stops = np.asarray([item[1] for item in self.burst_intervals], dtype=float)
        else:
            self._burst_starts = np.array([], dtype=float)
            self._burst_stops = np.array([], dtype=float)
        self.update()

    def set_stim_times(self, stim_times) -> None:
        values = np.asarray(stim_times if stim_times is not None else [], dtype=float)
        values = values[np.isfinite(values)]
        values.sort()
        self.stim_times = values
        self.update()

    def set_playhead_time(self, playhead_time) -> None:
        try:
            value = float(playhead_time)
        except (TypeError, ValueError):
            value = np.nan
        self.playhead_time = value if np.isfinite(value) else None
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        left = self.plot_left
        right = self.plot_right
        top = self.plot_top
        bottom = self.plot_bottom
        plot_width = max(1, self.width() - left - right)
        plot_height = max(1, self.height() - top - bottom)
        start = self.window_start
        stop = self.window_start + self.window_duration

        burst_brush = QColor("#fecaca")
        burst_brush.setAlpha(95)
        painter.setPen(Qt.PenStyle.NoPen)
        burst_visible = False
        if self._burst_starts.size:
            first_burst = int(np.searchsorted(self._burst_stops, start, side="right"))
            last_burst = int(np.searchsorted(self._burst_starts, stop, side="left"))
            burst_iter = zip(self._burst_starts[first_burst:last_burst], self._burst_stops[first_burst:last_burst])
        else:
            burst_iter = ()
        for burst_start, burst_stop in burst_iter:
            overlap_start = max(start, burst_start)
            overlap_stop = min(stop, burst_stop)
            if overlap_stop <= overlap_start:
                continue
            x0 = left + (overlap_start - start) / self.window_duration * plot_width
            x1 = left + (overlap_stop - start) / self.window_duration * plot_width
            painter.fillRect(QRectF(x0, top, max(1.0, x1 - x0), plot_height), burst_brush)
            burst_visible = True

        painter.setPen(QPen(QColor("#d7deea"), 1))
        grid = np.arange(
            np.ceil(start / self.grid_step) * self.grid_step,
            stop + self.grid_step * 0.5,
            self.grid_step,
        )
        for tick in grid:
            x = left + (tick - start) / self.window_duration * plot_width
            painter.drawLine(int(x), top, int(x), top + plot_height)

        painter.setPen(QPen(QColor("#1f2937"), 1))
        painter.drawRect(left, top, plot_width, plot_height)

        stim_visible = False
        if self.stim_times.size:
            stim_lo = int(np.searchsorted(self.stim_times, start, side="left"))
            stim_hi = int(np.searchsorted(self.stim_times, stop, side="right"))
            visible_stim = self.stim_times[stim_lo:stim_hi]
            if visible_stim.size:
                stim_visible = True
                stim_pen = QPen(QColor("#f97316"), 1)
                stim_pen.setCosmetic(True)
                painter.setPen(stim_pen)
                painter.setBrush(QColor("#f97316"))
                for stim_time in visible_stim[:2000]:
                    x = left + (float(stim_time) - start) / self.window_duration * plot_width
                    painter.drawLine(QLineF(float(x), top, float(x), top + plot_height))
                    triangle = QPolygonF(
                        [
                            QPointF(float(x), top + 1),
                            QPointF(float(x) - 4.0, top + 9.0),
                            QPointF(float(x) + 4.0, top + 9.0),
                        ]
                    )
                    painter.drawPolygon(triangle)

        if burst_visible:
            painter.setFont(QFont("Segoe UI", 8))
            painter.setPen(QPen(QColor("#991b1b"), 1))
            painter.setBrush(QColor("#fecaca"))
            legend_x = left + plot_width - (178 if stim_visible else 104)
            painter.drawRect(int(legend_x), top + 8, 12, 8)
            painter.drawText(int(legend_x + 18), top + 4, 86, 18, Qt.AlignmentFlag.AlignLeft, "burst")
        if stim_visible:
            painter.setFont(QFont("Segoe UI", 8))
            painter.setPen(QPen(QColor("#f97316"), 1))
            painter.setBrush(QColor("#f97316"))
            legend_x = left + plot_width - 82
            painter.drawLine(QLineF(float(legend_x), top + 8.0, float(legend_x), top + 17.0))
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(float(legend_x), top + 5.0),
                        QPointF(float(legend_x) - 4.0, top + 12.0),
                        QPointF(float(legend_x) + 4.0, top + 12.0),
                    ]
                )
            )
            painter.drawText(int(legend_x + 12), top + 4, 64, 18, Qt.AlignmentFlag.AlignLeft, "stim")

        if self.playhead_time is not None and start <= self.playhead_time <= stop:
            x = left + (float(self.playhead_time) - start) / self.window_duration * plot_width
            playhead_pen = QPen(QColor("#0f172a"), 2)
            playhead_pen.setCosmetic(True)
            painter.setPen(playhead_pen)
            painter.drawLine(QLineF(float(x), top, float(x), top + plot_height))

        visible_series = self._visible_spike_series()
        channel_count = len(visible_series)
        if channel_count == 0:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No spike data")
            painter.end()
            return

        row_step = plot_height / max(channel_count, 1)
        label_font = QFont("Segoe UI", 8)
        painter.setFont(label_font)
        label_height = max(14, painter.fontMetrics().height() + 2)

        if row_step >= 6:
            painter.setPen(QPen(QColor("#eef2f7"), 1))
            for row in range(channel_count):
                y = self._row_center(top, plot_height, row, channel_count)
                painter.drawLine(left, int(round(y)), left + plot_width, int(round(y)))

        painter.setPen(QPen(QColor("#64748b"), 1))
        label_stride = self._label_stride_for_rows(channel_count, plot_height, label_height)
        for row, (channel, _) in enumerate(visible_series):
            y = self._row_center(top, plot_height, row, channel_count)
            if row % label_stride == 0:
                painter.drawText(
                    QRectF(6, y - label_height / 2, left - 14, label_height),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    channel,
                )

        default_spike = QColor("#2563eb")
        default_spike.setAlpha(235)
        selected_spike = QColor("#dc2626")
        selected_spike.setAlpha(245)
        for row, (channel, times) in enumerate(visible_series):
            if times.size == 0:
                continue
            spike_pen = QPen(selected_spike if channel == self.selected_channel else default_spike, 1)
            spike_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(spike_pen)
            lo = int(np.searchsorted(times, start, side="left"))
            hi = int(np.searchsorted(times, stop, side="right"))
            if hi <= lo:
                continue
            visible = times[lo:hi]
            xs = left + (visible - start) / self.window_duration * plot_width
            y_center = self._row_center(top, plot_height, row, channel_count)
            spike_half_height = max(1.0, min(row_step * 0.32, 5.0))
            y0 = int(y_center - spike_half_height)
            y1 = int(y_center + spike_half_height)
            xi_values = np.rint(xs).astype(np.int32, copy=False)
            xi_values = xi_values[(xi_values >= left) & (xi_values <= left + plot_width)]
            if xi_values.size == 0:
                continue
            xi_values = np.unique(xi_values)
            painter.drawLines([QLineF(float(xi), y0, float(xi), y1) for xi in xi_values])

        painter.setPen(QPen(QColor("#111827"), 1))
        painter.drawText(left, self.height() - 26, plot_width, 18, Qt.AlignmentFlag.AlignCenter, "Time (s)")
        painter.save()
        painter.translate(18, top + plot_height / 2)
        painter.rotate(-90)
        painter.drawText(-plot_height // 2, -4, plot_height, 18, Qt.AlignmentFlag.AlignCenter, self.y_axis_label)
        painter.restore()

        tick_font = QFont("Segoe UI", 8)
        painter.setFont(tick_font)
        painter.setPen(QPen(QColor("#475569"), 1))
        max_labels = max(2, plot_width // 90)
        label_stride = max(1, int(np.ceil(len(grid) / max_labels))) if len(grid) else 1
        for index, tick in enumerate(grid):
            if index % label_stride != 0 and index != len(grid) - 1:
                continue
            x = left + (tick - start) / self.window_duration * plot_width
            painter.drawText(
                int(x - 38),
                top + plot_height + 4,
                76,
                16,
                Qt.AlignmentFlag.AlignCenter,
                _format_time_tick(float(tick)),
            )

        painter.end()

    def wheelEvent(self, event: QWheelEvent):  # noqa: N802 - Qt override
        delta = event.angleDelta().y()
        if delta == 0:
            return
        pos = event.position() if hasattr(event, "position") else event.pos()
        plot_width = max(1, self.width() - self.plot_left - self.plot_right)
        fraction = (float(pos.x()) - self.plot_left) / plot_width
        fraction = min(1.0, max(0.0, fraction))
        direction = 1 if delta > 0 else -1
        self.wheel_zoom_requested.emit(fraction, direction)
        event.accept()

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position() if hasattr(event, "position") else event.pos()
            self._drag_start_x = float(pos.x())
            self._drag_start_y = float(pos.y())
            self._drag_start_slider_ms = int(round(self.window_start * 1000))
            self._drag_moved = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802 - Qt override
        if self._drag_start_x is None:
            super().mouseMoveEvent(event)
            return
        pos = event.position() if hasattr(event, "position") else event.pos()
        dx = float(pos.x()) - self._drag_start_x
        dy = float(pos.y()) - (self._drag_start_y or 0.0)
        if abs(dx) > 3 or abs(dy) > 3:
            self._drag_moved = True
        plot_width = max(1, self.width() - self.plot_left - self.plot_right)
        delta_ms = int(round((-dx / plot_width) * self.window_duration * 1000))
        self.pan_requested.emit(self._drag_start_slider_ms + delta_ms)
        event.accept()

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton and self._drag_start_x is not None:
            if not self._drag_moved:
                pos = event.position() if hasattr(event, "position") else event.pos()
                channel = self._channel_at_y(float(pos.y()))
                if channel:
                    self.channel_selected.emit(channel)
            self._drag_start_x = None
            self._drag_start_y = None
            self._drag_moved = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _channel_at_y(self, y: float) -> str:
        visible_series = self._visible_spike_series()
        channel_count = len(visible_series)
        if channel_count == 0:
            return ""
        plot_height = max(1, self.height() - self.plot_top - self.plot_bottom)
        if y < self.plot_top or y > self.plot_top + plot_height:
            return ""
        row = int((y - self.plot_top) / (plot_height / channel_count))
        row = min(channel_count - 1, max(0, row))
        return visible_series[row][0]


class SpikeWaveformCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.channel = ""
        self.waveforms = np.zeros((0, 0), dtype=float)
        self.sampling_rate = None
        self.max_traces = 320
        self.setMinimumHeight(210)

    def set_channel_waveforms(self, channel: str, waveforms, sampling_rate: float | None = None) -> None:
        self.channel = channel
        self.sampling_rate = sampling_rate
        if waveforms is None:
            self.waveforms = np.zeros((0, 0), dtype=float)
        else:
            array = np.asarray(waveforms, dtype=np.float32)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            self.waveforms = array
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        left = 62
        right = 18
        top = 26
        bottom = 34
        plot_width = max(1, self.width() - left - right)
        plot_height = max(1, self.height() - top - bottom)

        painter.setPen(QPen(QColor("#d7deea"), 1))
        painter.drawRect(left, top, plot_width, plot_height)
        painter.setPen(QPen(QColor("#111827"), 1))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        title = f"{self.channel} spike waveforms" if self.channel else "Spike waveforms"
        painter.drawText(left, 4, plot_width, 18, Qt.AlignmentFlag.AlignLeft, title)

        if self.waveforms.size == 0:
            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QPen(QColor("#64748b"), 1))
            painter.drawText(
                QRectF(left, top, plot_width, plot_height),
                Qt.AlignmentFlag.AlignCenter,
                "No waveform data for selected channel",
            )
            painter.end()
            return

        waveforms = self.waveforms
        if waveforms.ndim == 2:
            finite_rows = np.any(np.isfinite(waveforms), axis=1)
            waveforms = waveforms[finite_rows]
        if waveforms.size == 0:
            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QPen(QColor("#64748b"), 1))
            painter.drawText(
                QRectF(left, top, plot_width, plot_height),
                Qt.AlignmentFlag.AlignCenter,
                "No valid waveform data for selected channel",
            )
            painter.end()
            return
        if waveforms.shape[0] > self.max_traces:
            indices = np.linspace(0, waveforms.shape[0] - 1, self.max_traces, dtype=int)
            visible = waveforms[indices]
        else:
            visible = waveforms

        ymin = float(np.nanpercentile(visible, 1))
        ymax = float(np.nanpercentile(visible, 99))
        if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
            ymin = float(np.nanmin(visible))
            ymax = float(np.nanmax(visible))
        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0
        pad = (ymax - ymin) * 0.08
        ymin -= pad
        ymax += pad

        xs = np.linspace(left, left + plot_width, visible.shape[1])

        trace_color = QColor("#2563eb")
        trace_color.setAlpha(90)
        painter.setPen(QPen(trace_color, 1))
        for waveform in visible:
            ys = top + (ymax - waveform) / (ymax - ymin) * plot_height
            points = QPolygonF([QPointF(float(x), float(y)) for x, y in zip(xs, ys)])
            painter.drawPolyline(points)

        mean = np.nanmean(visible, axis=0)
        ys = top + (ymax - mean) / (ymax - ymin) * plot_height
        mean_pen = QPen(QColor("#1e3a8a"), 2)
        mean_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(mean_pen)
        painter.drawPolyline(QPolygonF([QPointF(float(x), float(y)) for x, y in zip(xs, ys)]))

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(QColor("#475569"), 1))
        painter.drawText(4, top - 4, left - 8, 16, Qt.AlignmentFlag.AlignRight, f"{ymax:.1f}")
        painter.drawText(4, top + plot_height - 12, left - 8, 16, Qt.AlignmentFlag.AlignRight, f"{ymin:.1f}")
        if self.sampling_rate:
            duration_ms = (waveforms.shape[1] - 1) / float(self.sampling_rate) * 1000.0
            x_label = f"Time (ms), 0.000 - {duration_ms:.3f}"
        else:
            x_label = "Time (sample index)"
        painter.drawText(left, self.height() - 24, plot_width, 16, Qt.AlignmentFlag.AlignCenter, x_label)
        painter.save()
        painter.translate(14, top + plot_height / 2)
        painter.rotate(-90)
        painter.drawText(-plot_height // 2, -4, plot_height, 16, Qt.AlignmentFlag.AlignCenter, "Voltage (uV)")
        painter.restore()
        trace_text = f"traces: {waveforms.shape[0]}"
        if visible.shape[0] < waveforms.shape[0]:
            trace_text += f" shown: {visible.shape[0]}"
        painter.drawText(left, self.height() - 18, plot_width, 16, Qt.AlignmentFlag.AlignRight, trace_text)
        painter.end()


class PopulationRateCanvas(QWidget):
    def __init__(self, spike_series, left_margin: int = 76, bin_ms: float = 20.0, parent=None):
        super().__init__(parent)
        self.spike_series = [(label, np.asarray(times, dtype=float)) for label, times in spike_series]
        self.left_margin = left_margin
        self.bin_s = max(0.001, float(bin_ms) / 1000.0)
        self.window_start = 0.0
        self.window_duration = 1.0
        self.centers = np.array([], dtype=float)
        self.rates = np.array([], dtype=float)
        self._all_centers = np.array([], dtype=float)
        self._all_rates = np.array([], dtype=float)
        self.burst_intervals = []
        self._burst_starts = np.array([], dtype=float)
        self._burst_stops = np.array([], dtype=float)
        self.stim_times = np.array([], dtype=float)
        self.playhead_time = None
        self._build_rate_cache()
        self.setMinimumHeight(120)

    def _build_rate_cache(self) -> None:
        all_times = [times for _, times in self.spike_series if times.size]
        if not all_times:
            self._all_centers = np.array([], dtype=float)
            self._all_rates = np.array([], dtype=float)
            return
        values = np.concatenate(all_times).astype(float, copy=False)
        values = values[np.isfinite(values)]
        if values.size == 0:
            self._all_centers = np.array([], dtype=float)
            self._all_rates = np.array([], dtype=float)
            return
        start = np.floor(float(values.min()) / self.bin_s) * self.bin_s
        stop = np.ceil(float(values.max()) / self.bin_s) * self.bin_s
        edges = np.arange(start, stop + self.bin_s * 1.5, self.bin_s)
        if edges.size < 2:
            edges = np.array([start, start + self.bin_s], dtype=float)
        counts, edges = np.histogram(values, bins=edges)
        self._all_centers = (edges[:-1] + edges[1:]) / 2.0
        self._all_rates = counts.astype(float) / self.bin_s / max(1, len(self.spike_series))

    def set_spike_series(self, spike_series) -> None:
        self.spike_series = [(label, np.asarray(times, dtype=float)) for label, times in spike_series]
        self._build_rate_cache()
        self.centers, self.rates = self._average_rate_trace(self.window_start, self.window_start + self.window_duration)
        self.update()

    def set_bursts(self, intervals) -> None:
        self.burst_intervals = sorted((float(start), float(stop)) for start, stop in intervals)
        if self.burst_intervals:
            self._burst_starts = np.asarray([item[0] for item in self.burst_intervals], dtype=float)
            self._burst_stops = np.asarray([item[1] for item in self.burst_intervals], dtype=float)
        else:
            self._burst_starts = np.array([], dtype=float)
            self._burst_stops = np.array([], dtype=float)
        self.update()

    def set_stim_times(self, stim_times) -> None:
        values = np.asarray(stim_times if stim_times is not None else [], dtype=float)
        values = values[np.isfinite(values)]
        values.sort()
        self.stim_times = values
        self.update()

    def set_playhead_time(self, playhead_time) -> None:
        try:
            value = float(playhead_time)
        except (TypeError, ValueError):
            value = np.nan
        self.playhead_time = value if np.isfinite(value) else None
        self.update()

    def set_view(self, start_s: float, duration_s: float) -> None:
        self.window_start = max(0.0, float(start_s))
        self.window_duration = max(0.001, float(duration_s))
        self.centers, self.rates = self._average_rate_trace(self.window_start, self.window_start + self.window_duration)
        self.update()

    def _average_rate_trace(self, start_s: float, stop_s: float):
        if self._all_centers.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        lo = int(np.searchsorted(self._all_centers, start_s, side="left"))
        hi = int(np.searchsorted(self._all_centers, stop_s, side="right"))
        return self._all_centers[lo:hi], self._all_rates[lo:hi]

    def paintEvent(self, event):  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        left = self.left_margin
        right = 18
        top = 18
        bottom = 30
        plot_width = max(1, self.width() - left - right)
        plot_height = max(1, self.height() - top - bottom)
        start = self.window_start
        stop = self.window_start + self.window_duration

        burst_brush = QColor("#fecaca")
        burst_brush.setAlpha(80)
        if self._burst_starts.size:
            first_burst = int(np.searchsorted(self._burst_stops, start, side="right"))
            last_burst = int(np.searchsorted(self._burst_starts, stop, side="left"))
            burst_iter = zip(self._burst_starts[first_burst:last_burst], self._burst_stops[first_burst:last_burst])
        else:
            burst_iter = ()
        for burst_start, burst_stop in burst_iter:
            overlap_start = max(start, burst_start)
            overlap_stop = min(stop, burst_stop)
            if overlap_stop <= overlap_start:
                continue
            x0 = left + (overlap_start - start) / self.window_duration * plot_width
            x1 = left + (overlap_stop - start) / self.window_duration * plot_width
            painter.fillRect(QRectF(x0, top, max(1.0, x1 - x0), plot_height), burst_brush)

        if self.stim_times.size:
            stim_lo = int(np.searchsorted(self.stim_times, start, side="left"))
            stim_hi = int(np.searchsorted(self.stim_times, stop, side="right"))
            visible_stim = self.stim_times[stim_lo:stim_hi]
            if visible_stim.size:
                stim_pen = QPen(QColor("#f97316"), 1)
                stim_pen.setCosmetic(True)
                painter.setPen(stim_pen)
                for stim_time in visible_stim[:2000]:
                    x = left + (float(stim_time) - start) / self.window_duration * plot_width
                    painter.drawLine(QLineF(float(x), top, float(x), top + plot_height))

        painter.setPen(QPen(QColor("#d7deea"), 1))
        painter.drawRect(left, top, plot_width, plot_height)
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(QColor("#475569"), 1))
        painter.drawText(6, top - 3, left - 12, 16, Qt.AlignmentFlag.AlignRight, "Avg rate")

        if self.centers.size == 0 or self.rates.size == 0:
            if self.playhead_time is not None and start <= self.playhead_time <= stop:
                x = left + (float(self.playhead_time) - start) / self.window_duration * plot_width
                painter.setPen(QPen(QColor("#0f172a"), 2))
                painter.drawLine(QLineF(float(x), top, float(x), top + plot_height))
            painter.drawText(QRectF(left, top, plot_width, plot_height), Qt.AlignmentFlag.AlignCenter, "No rate data")
            painter.end()
            return

        max_rate = float(np.nanmax(self.rates)) if np.isfinite(self.rates).any() else 0.0
        ymax = max(1.0, max_rate * 1.15)
        painter.setPen(QPen(QColor("#e2e8f0"), 1))
        for fraction in (0.25, 0.5, 0.75):
            y = top + plot_height * fraction
            painter.drawLine(left, int(y), left + plot_width, int(y))

        xs = left + (self.centers - start) / self.window_duration * plot_width
        ys = top + (ymax - self.rates) / ymax * plot_height
        points = QPolygonF([QPointF(float(x), float(y)) for x, y in zip(xs, ys)])
        rate_pen = QPen(QColor("#0f766e"), 2)
        rate_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(rate_pen)
        painter.drawPolyline(points)

        if self.playhead_time is not None and start <= self.playhead_time <= stop:
            x = left + (float(self.playhead_time) - start) / self.window_duration * plot_width
            playhead_pen = QPen(QColor("#0f172a"), 2)
            playhead_pen.setCosmetic(True)
            painter.setPen(playhead_pen)
            painter.drawLine(QLineF(float(x), top, float(x), top + plot_height))

        painter.setPen(QPen(QColor("#475569"), 1))
        painter.drawText(6, top + 4, left - 12, 14, Qt.AlignmentFlag.AlignRight, f"{ymax:.1f}")
        painter.drawText(left, self.height() - 22, plot_width, 16, Qt.AlignmentFlag.AlignCenter, "Time (s)")
        painter.drawText(left, self.height() - 20, plot_width, 14, Qt.AlignmentFlag.AlignRight, "Hz/unit")
        painter.end()


class ElectrodeHeatmapCanvas(QWidget):
    def __init__(self, channel_map: ChannelMap | None = None, parent=None):
        super().__init__(parent)
        self.channel_map = channel_map
        self.counts = {}
        self._target_counts = {}
        self._display_counts = {}
        self.well_counts = {}
        self._target_well_counts = {}
        self._display_well_counts = {}
        self.active_wells = set()
        self.scale_max_count = 0
        self.fast_mode = False
        self._coordinate_entries_cache = None
        self._coordinate_bounds_cache = None
        self._coordinate_lookup_cache = None
        self._routed_coordinate_entries_cache = None
        self._transition_timer = QTimer(self)
        self._transition_timer.setInterval(33)
        self._transition_timer.timeout.connect(self._advance_count_transition)
        self.setMinimumSize(320, 230)

    def set_channel_map(self, channel_map: ChannelMap | None) -> None:
        self.channel_map = channel_map
        self._invalidate_coordinate_cache()
        self.update()

    def _invalidate_coordinate_cache(self) -> None:
        self._coordinate_entries_cache = None
        self._coordinate_bounds_cache = None
        self._coordinate_lookup_cache = None
        self._routed_coordinate_entries_cache = None

    def _normalize_count_payload(self, counts: dict[str, int | float]):
        normalized = {}
        well_counts: dict[str, dict[str, int]] = {}
        for channel, count in counts.items():
            value = float(count)
            text = re.sub(r"\s+(?:unit\s+-?\d+|noise)$", "", str(channel).strip(), flags=re.IGNORECASE)
            match = re.fullmatch(r"([A-Za-z]+\d+)_(.+)", text.strip())
            well = match.group(1).upper() if match else ""
            electrode_channel = match.group(2) if match else text
            raw_key = normalize_channel_name(channel)
            normalized_channel = normalize_channel_name(electrode_channel)
            normalized[raw_key] = normalized.get(raw_key, 0) + value
            if normalized_channel != raw_key:
                normalized[normalized_channel] = normalized.get(normalized_channel, 0) + value
            if well:
                per_well = well_counts.setdefault(well, {})
                per_well[normalized_channel] = per_well.get(normalized_channel, 0) + value
            if "_" in text:
                suffix = text.split("_", 1)[1]
                normalized[normalize_channel_name(suffix)] = normalized.get(normalize_channel_name(suffix), 0) + value
        return normalized, well_counts

    def set_counts(self, counts: dict[str, int]) -> None:
        normalized, well_counts = self._normalize_count_payload(counts)
        if normalized == self._target_counts and well_counts == self._target_well_counts:
            return
        self._target_counts = normalized
        self._target_well_counts = well_counts
        self.counts = dict(normalized)
        self.well_counts = {well: dict(payload) for well, payload in well_counts.items()}
        if not self._transition_timer.isActive():
            self._transition_timer.start()
        self._advance_count_transition()

    @staticmethod
    def _blend_count_dict(current: dict[str, float], target: dict[str, float], alpha: float) -> tuple[dict[str, float], float]:
        blended = {}
        max_delta = 0.0
        for key in set(current) | set(target):
            current_value = float(current.get(key, 0.0))
            target_value = float(target.get(key, 0.0))
            next_value = current_value + (target_value - current_value) * alpha
            max_delta = max(max_delta, abs(target_value - next_value))
            if next_value > 0.01 or target_value > 0.01:
                blended[key] = next_value
        return blended, max_delta

    @classmethod
    def _blend_nested_count_dict(
        cls,
        current: dict[str, dict[str, float]],
        target: dict[str, dict[str, float]],
        alpha: float,
    ) -> tuple[dict[str, dict[str, float]], float]:
        blended = {}
        max_delta = 0.0
        for well in set(current) | set(target):
            payload, delta = cls._blend_count_dict(current.get(well, {}), target.get(well, {}), alpha)
            max_delta = max(max_delta, delta)
            if payload:
                blended[well] = payload
        return blended, max_delta

    def _advance_count_transition(self) -> None:
        alpha = 0.42 if self.fast_mode else 0.30
        self._display_counts, flat_delta = self._blend_count_dict(self._display_counts, self._target_counts, alpha)
        self._display_well_counts, nested_delta = self._blend_nested_count_dict(
            self._display_well_counts,
            self._target_well_counts,
            alpha,
        )
        self.active_wells = {
            well
            for well, payload in self._display_well_counts.items()
            if any(value > 0.01 for value in payload.values())
        }
        if max(flat_delta, nested_delta) <= 0.03:
            self._display_counts = dict(self._target_counts)
            self._display_well_counts = {well: dict(payload) for well, payload in self._target_well_counts.items()}
            self.active_wells = {well for well, payload in self.well_counts.items() if any(value > 0 for value in payload.values())}
            self._transition_timer.stop()
        self.update()

    def set_scale_max_count(self, max_count: int) -> None:
        self.scale_max_count = max(0, int(max_count))
        self.update()

    def set_fast_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self.fast_mode == enabled:
            return
        self.fast_mode = enabled
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#000000"))

        margin = 8.0
        colorbar_width = 12.0
        colorbar_gap = 12.0
        map_width = self.width() - margin * 2.0 - colorbar_width - colorbar_gap
        map_height = self.height() - margin * 2.0
        if map_width <= 20 or map_height <= 20:
            painter.end()
            return

        if self.channel_map is None:
            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QPen(QColor("#cbd5e1"), 1))
            painter.drawText(
                QRectF(8, 8, self.width() - 16, self.height() - 16),
                Qt.AlignmentFlag.AlignCenter,
                "No channel map",
            )
            painter.end()
            return

        display_well_counts = self._display_well_counts or self.well_counts
        display_counts = self._display_counts or self.counts
        if display_well_counts:
            selected_well = next(iter(self.active_wells)) if len(self.active_wells) == 1 else sorted(display_well_counts)[0]
            active_counts = display_well_counts.get(selected_well, {})
        else:
            active_counts = display_counts
        max_count = self.scale_max_count or max(active_counts.values(), default=0)

        rect = self._map_rect_for_channel_map(margin, map_width, map_height)
        self._draw_well_heatmap(painter, rect, active_counts, max_count)
        self._draw_coordinate_recording_overlay(painter, rect, active_counts)
        painter.setPen(QPen(QColor("#aeb4c0"), 1.1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)

        bar_left = margin + map_width + colorbar_gap
        bar_top = margin
        bar_height = map_height
        steps = max(32, int(bar_height))
        for index in range(steps):
            fraction = 1.0 - index / max(steps - 1, 1)
            painter.setPen(QPen(_activity_heatmap_color(fraction), 1))
            y = bar_top + index / max(steps - 1, 1) * bar_height
            painter.drawLine(QPointF(bar_left, y), QPointF(bar_left + colorbar_width, y))
        painter.setPen(QPen(QColor("#94a3b8"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(bar_left, bar_top, colorbar_width, bar_height))
        painter.end()

    def _coordinate_entries(self):
        if self._coordinate_entries_cache is not None:
            return self._coordinate_entries_cache
        if self.channel_map is None:
            return []
        entries = []
        for electrode, payload in self.channel_map.electrodes.items():
            if not isinstance(payload, dict):
                continue
            try:
                x_um = float(payload.get("x_um", payload.get("x")))
                y_um = float(payload.get("y_um", payload.get("y")))
            except (TypeError, ValueError):
                continue
            if np.isfinite(x_um) and np.isfinite(y_um):
                entries.append((str(electrode), payload, x_um, y_um))
        self._coordinate_entries_cache = entries
        return entries

    def _coordinate_bounds(self, entries=None):
        if entries is None and self._coordinate_bounds_cache is not None:
            return self._coordinate_bounds_cache
        entries = self._coordinate_entries() if entries is None else entries
        if not entries:
            return None
        xs = np.asarray([entry[2] for entry in entries], dtype=float)
        ys = np.asarray([entry[3] for entry in entries], dtype=float)
        xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
        ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
        if xmax <= xmin or ymax <= ymin:
            return None
        bounds = (xmin, xmax, ymin, ymax)
        if entries is self._coordinate_entries_cache:
            self._coordinate_bounds_cache = bounds
        return bounds

    def _coordinate_lookup(self):
        if self._coordinate_lookup_cache is not None:
            return self._coordinate_lookup_cache
        lookup = {}
        routed_entries = []
        entries = self._coordinate_entries()
        for index, (electrode, payload, x_um, y_um) in enumerate(entries):
            aliases = [str(payload.get("channel") or "").strip(), str(electrode)]
            raw_aliases = payload.get("aliases", [])
            if isinstance(raw_aliases, (list, tuple)):
                aliases.extend(str(alias).strip() for alias in raw_aliases)
            for alias in aliases:
                if not alias:
                    continue
                keys = [normalize_channel_name(alias)]
                if "_" in alias:
                    keys.append(normalize_channel_name(alias.split("_", 1)[1]))
                for key in keys:
                    if key and key not in lookup:
                        lookup[key] = (str(electrode), payload, float(x_um), float(y_um), index)
            if payload.get("routed") or payload.get("channel"):
                routed_entries.append((str(electrode), payload, float(x_um), float(y_um), index))
        self._coordinate_lookup_cache = lookup
        self._routed_coordinate_entries_cache = routed_entries
        return lookup

    def _map_rect_for_channel_map(self, margin: float, map_width: float, map_height: float) -> QRectF:
        bounds = self._coordinate_bounds()
        if bounds is None:
            size = min(map_width, map_height)
            return QRectF(margin + (map_width - size) / 2.0, margin + (map_height - size) / 2.0, size, size)

        xmin, xmax, ymin, ymax = bounds
        aspect = max(0.2, min(8.0, (xmax - xmin) / max(ymax - ymin, 1e-6)))
        draw_w = map_width
        draw_h = draw_w / aspect
        if draw_h > map_height:
            draw_h = map_height
            draw_w = draw_h * aspect
        return QRectF(
            margin + (map_width - draw_w) / 2.0,
            margin + (map_height - draw_h) / 2.0,
            draw_w,
            draw_h,
        )

    def _coordinate_point(self, rect: QRectF, bounds, x_um: float, y_um: float) -> QPointF:
        xmin, xmax, ymin, ymax = bounds
        x = rect.left() + (float(x_um) - xmin) / max(xmax - xmin, 1e-6) * rect.width()
        y = rect.top() + (float(y_um) - ymin) / max(ymax - ymin, 1e-6) * rect.height()
        return QPointF(float(x), float(y))

    def _draw_coordinate_recording_overlay(self, painter: QPainter, rect: QRectF, active_counts: dict[str, int]) -> None:
        self._coordinate_lookup()
        entries = self._routed_coordinate_entries_cache or []
        bounds = self._coordinate_bounds()
        if bounds is None:
            return

        painter.save()
        painter.setClipRect(rect)
        routed_pen = QPen(QColor(82, 94, 115, 110), 1.0)
        active_pen = QPen(QColor(205, 245, 255, 185), 1.3)
        for electrode, payload, x_um, y_um, _index in entries:
            point = self._coordinate_point(rect, bounds, x_um, y_um)
            lookup_channels = [str(payload.get("channel") or "").strip(), str(electrode)]
            aliases = payload.get("aliases", [])
            if isinstance(aliases, (list, tuple)):
                lookup_channels.extend(str(alias).strip() for alias in aliases)
            has_activity = False
            for channel in lookup_channels:
                if not channel:
                    continue
                if active_counts.get(normalize_channel_name(channel), 0) > 0:
                    has_activity = True
                    break
                if "_" in channel and active_counts.get(normalize_channel_name(channel.split("_", 1)[1]), 0) > 0:
                    has_activity = True
                    break
            painter.setPen(active_pen if has_activity else routed_pen)
            painter.drawPoint(point)
        painter.restore()

    def _draw_well_heatmap(self, painter: QPainter, rect: QRectF, active_counts: dict[str, int], max_count: int) -> None:
        painter.save()
        painter.setClipRect(rect)
        painter.fillRect(rect, QColor("#000000"))
        if max_count <= 0:
            painter.restore()
            return

        max_resolution = 112 if self.fast_mode else 180
        min_resolution = 72 if self.fast_mode else 96
        resolution = int(max(min_resolution, min(max_resolution, round(rect.width()))))
        field = self._continuous_heatmap_field(active_counts, max_count, resolution)
        if not np.any(field > 0):
            field = self._fallback_heatmap_field(active_counts, max_count, resolution)
        image = self._heatmap_field_image(field)
        painter.drawImage(rect, image)
        painter.restore()

    def _continuous_heatmap_field(self, active_counts: dict[str, int], max_count: int, resolution: int) -> np.ndarray:
        coordinate_field = self._coordinate_heatmap_field(active_counts, max_count, resolution)
        if np.any(coordinate_field > 0):
            return coordinate_field

        yy, xx = np.mgrid[0:resolution, 0:resolution]
        x = (xx.astype(np.float32) + 0.5) / float(resolution)
        y = (yy.astype(np.float32) + 0.5) / float(resolution)
        field = np.zeros((resolution, resolution), dtype=np.float32)
        points: list[tuple[int, int, float, float, float]] = []

        for row in range(8):
            for col in range(8):
                electrode = electrode_id(row, col)
                channel = self.channel_map.channel_for(electrode)
                if not channel:
                    continue
                count = active_counts.get(normalize_channel_name(channel), 0)
                if count <= 0:
                    continue
                intensity = min(1.0, float(count) / float(max_count))
                cx = (col + 0.5) / 8.0
                cy = (row + 0.5) / 8.0
                points.append((row, col, cx, cy, intensity))
                self._add_heat_kernel(field, x, y, cx, cy, intensity, row, col)

        point_lookup = {(row, col): (cx, cy, intensity) for row, col, cx, cy, intensity in points}
        for row, col, cx, cy, intensity in points:
            for d_row, d_col in ((0, 1), (1, 0), (1, 1), (1, -1)):
                other = point_lookup.get((row + d_row, col + d_col))
                if other is None:
                    continue
                ox, oy, other_intensity = other
                bridge = float(np.sqrt(intensity * other_intensity))
                if bridge <= 0.03:
                    continue
                self._add_heat_kernel(
                    field,
                    x,
                    y,
                    (cx + ox) * 0.5,
                    (cy + oy) * 0.5,
                    bridge * 0.42,
                    row + d_row * 3,
                    col + d_col * 3,
                    broad=True,
                )

        if points:
            weights = np.asarray([point[-1] for point in points], dtype=np.float32)
            centers_x = np.asarray([point[2] for point in points], dtype=np.float32)
            centers_y = np.asarray([point[3] for point in points], dtype=np.float32)
            centroid_x = float(np.average(centers_x, weights=weights))
            centroid_y = float(np.average(centers_y, weights=weights))
            self._add_heat_kernel(
                field,
                x,
                y,
                centroid_x,
                centroid_y,
                min(0.22, float(weights.mean()) * 0.26),
                11,
                17,
                broad=True,
            )

        field = np.clip(field, 0.0, 1.0)
        field = np.where(field < 0.035, 0.0, field)
        return np.power(field, 0.82, dtype=np.float32)

    def _coordinate_heatmap_field(self, active_counts: dict[str, int], max_count: int, resolution: int) -> np.ndarray:
        field = np.zeros((resolution, resolution), dtype=np.float32)
        if self.channel_map is None or max_count <= 0:
            return field

        lookup = self._coordinate_lookup()
        if not lookup:
            return field

        bounds = self._coordinate_bounds()
        if bounds is None:
            return field
        xmin, xmax, ymin, ymax = bounds
        active_points = []
        used_keys = set()
        for channel, count in active_counts.items():
            if count <= 0:
                continue
            key = normalize_channel_name(channel)
            if key in used_keys:
                continue
            entry = lookup.get(key)
            if entry is None and "_" in str(channel):
                entry = lookup.get(normalize_channel_name(str(channel).split("_", 1)[1]))
            if entry is None:
                continue
            used_keys.add(key)
            _electrode, _payload, x_um, y_um, index = entry
            intensity = min(1.0, float(count) / float(max_count))
            cx = (x_um - xmin) / (xmax - xmin)
            cy = (y_um - ymin) / (ymax - ymin)
            active_points.append((cx, cy, intensity, index))

        if not active_points:
            return field
        if self.fast_mode or len(active_points) > 96:
            return self._smoothed_point_heatmap(active_points, resolution)

        yy, xx = np.mgrid[0:resolution, 0:resolution]
        x = (xx.astype(np.float32) + 0.5) / float(resolution)
        y = (yy.astype(np.float32) + 0.5) / float(resolution)
        for cx, cy, intensity, index in active_points:
            self._add_heat_kernel(field, x, y, cx, cy, intensity, index // 220, index % 220)

        if len(active_points) > 1:
            points = np.asarray(active_points, dtype=np.float32)
            order = np.argsort(points[:, 0] + points[:, 1] * 2.0)
            step = max(1, len(order) // 180)
            for left_idx, right_idx in zip(order[::step], order[step::step]):
                left = points[int(left_idx)]
                right = points[int(right_idx)]
                distance = float(np.hypot(left[0] - right[0], left[1] - right[1]))
                if distance > 0.075:
                    continue
                bridge = float(np.sqrt(left[2] * right[2])) * 0.26
                if bridge <= 0.025:
                    continue
                seed = int(left[3] + right[3])
                self._add_heat_kernel(
                    field,
                    x,
                    y,
                    float((left[0] + right[0]) * 0.5),
                    float((left[1] + right[1]) * 0.5),
                    bridge,
                    seed // 220,
                    seed % 220,
                    broad=True,
                )

        if not np.any(field > 0):
            return field
        field = np.clip(field, 0.0, 1.0)
        field = np.where(field < 0.028, 0.0, field)
        return np.power(field, 0.82, dtype=np.float32)

    def _smoothed_point_heatmap(self, active_points, resolution: int) -> np.ndarray:
        field = np.zeros((resolution, resolution), dtype=np.float32)
        if not active_points:
            return field
        points = np.asarray(active_points, dtype=np.float32)
        xs = np.clip(np.rint(points[:, 0] * (resolution - 1)).astype(np.int32), 0, resolution - 1)
        ys = np.clip(np.rint(points[:, 1] * (resolution - 1)).astype(np.int32), 0, resolution - 1)
        weights = points[:, 2].astype(np.float32, copy=False)
        np.add.at(field, (ys, xs), weights)

        seeds = points[:, 3].astype(np.int32, copy=False)
        offset_x = ((seeds * 37) % 7 - 3).astype(np.int32)
        offset_y = ((seeds * 53) % 7 - 3).astype(np.int32)
        tail_x = np.clip(xs + offset_x, 0, resolution - 1)
        tail_y = np.clip(ys + offset_y, 0, resolution - 1)
        np.add.at(field, (tail_y, tail_x), weights * 0.18)

        sigma = 4.6 if self.fast_mode else 5.8
        field = gaussian_filter(field, sigma=sigma, mode="constant", truncate=3.0)
        if not np.any(field > 0):
            return field
        field *= float(2.0 * np.pi * sigma * sigma * 0.72)
        field = np.clip(field, 0.0, 1.0)
        field = np.where(field < 0.026, 0.0, field)
        return np.power(field, 0.82, dtype=np.float32)

    def _fallback_heatmap_field(self, active_counts: dict[str, int], max_count: int, resolution: int) -> np.ndarray:
        yy, xx = np.mgrid[0:resolution, 0:resolution]
        x = (xx.astype(np.float32) + 0.5) / float(resolution)
        y = (yy.astype(np.float32) + 0.5) / float(resolution)
        field = np.zeros((resolution, resolution), dtype=np.float32)
        for channel, count in active_counts.items():
            if count <= 0:
                continue
            match = re.fullmatch(r"(?:channel|chan|ch)?0*(\d+)", str(channel).strip().lower())
            if not match:
                continue
            index = int(match.group(1)) - 1
            if not 0 <= index < 64:
                continue
            row = index // 8
            col = index % 8
            intensity = min(1.0, float(count) / float(max_count))
            self._add_heat_kernel(field, x, y, (col + 0.5) / 8.0, (row + 0.5) / 8.0, intensity, row, col)
        if not np.any(field > 0):
            return field
        field = np.clip(field, 0.0, 1.0)
        field = np.where(field < 0.035, 0.0, field)
        return np.power(field, 0.82, dtype=np.float32)

    def _add_heat_kernel(
        self,
        field: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        cx: float,
        cy: float,
        intensity: float,
        row_seed: int,
        col_seed: int,
        *,
        broad: bool = False,
    ) -> None:
        angle = ((row_seed * 41 + col_seed * 23) % 180) * np.pi / 180.0
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        dx = x - float(cx)
        dy = y - float(cy)
        rx = dx * cos_a + dy * sin_a
        ry = -dx * sin_a + dy * cos_a
        base = 0.058 if not broad else 0.078
        sigma_x = base * (1.0 + 0.24 * ((row_seed * 7 + col_seed * 3) % 5) / 4.0)
        sigma_y = base * (0.92 + 0.28 * ((row_seed * 5 + col_seed * 11) % 5) / 4.0)
        field += float(intensity) * np.exp(-0.5 * ((rx / sigma_x) ** 2 + (ry / sigma_y) ** 2)).astype(np.float32)

        tail_angle = angle + np.pi * 0.42
        tail_x = float(cx) + 0.015 * np.cos(tail_angle)
        tail_y = float(cy) + 0.015 * np.sin(tail_angle)
        dx2 = x - tail_x
        dy2 = y - tail_y
        rx2 = dx2 * cos_a + dy2 * sin_a
        ry2 = -dx2 * sin_a + dy2 * cos_a
        field += float(intensity) * (0.10 if not broad else 0.06) * np.exp(
            -0.5 * ((rx2 / (sigma_x * 1.25)) ** 2 + (ry2 / (sigma_y * 0.95)) ** 2)
        ).astype(np.float32)

    def _heatmap_field_image(self, field: np.ndarray) -> QImage:
        image_data = self._heatmap_field_rgb(field)
        height, width, _ = image_data.shape
        return QImage(image_data.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()

    def _heatmap_field_rgb(self, field: np.ndarray) -> np.ndarray:
        stops = np.asarray(
            [
                (position, QColor(color).red(), QColor(color).green(), QColor(color).blue())
                for position, color in _HEATMAP_COLOR_STOPS
            ],
            dtype=np.float32,
        )
        values = np.clip(field.astype(np.float32), 0.0, 1.0)
        rgb = np.zeros((*values.shape, 3), dtype=np.float32)
        for left, right in zip(stops, stops[1:]):
            left_pos, right_pos = float(left[0]), float(right[0])
            mask = (values >= left_pos) & (values <= right_pos)
            if not np.any(mask):
                continue
            fraction = (values[mask] - left_pos) / max(right_pos - left_pos, 1e-6)
            rgb[mask] = left[1:] + (right[1:] - left[1:]) * fraction[:, None]
        return np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))

    def render_counts_rgb(self, counts: dict[str, int], resolution: int = 320, scale_max_count: int = 0) -> np.ndarray:
        normalized, well_counts = self._normalize_count_payload(counts)
        active_counts = well_counts[sorted(well_counts)[0]] if well_counts else normalized
        max_count = max(0, int(scale_max_count)) or max(active_counts.values(), default=0)
        resolution = max(32, int(resolution))
        if self.channel_map is None or max_count <= 0:
            return np.zeros((resolution, resolution, 3), dtype=np.uint8)
        field = self._continuous_heatmap_field(active_counts, max_count, resolution)
        if not np.any(field > 0):
            field = self._fallback_heatmap_field(active_counts, max_count, resolution)
        return self._heatmap_field_rgb(field)


class IBIWindow(QDialog):
    def __init__(self, burst_intervals, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inter-burst Interval")
        self.resize(760, 520)
        self.burst_intervals = [(float(start), float(stop)) for start, stop in burst_intervals]
        self.bin_ms = QSpinBox()
        self.bin_ms.setRange(10, 60000)
        self.bin_ms.setSingleStep(10)
        self.bin_ms.setValue(300)
        self.bin_ms.setSuffix(" ms")
        self.bin_ms.valueChanged.connect(self._draw)
        self.canvas = FigureCanvas(Figure(figsize=(7, 4), tight_layout=True))

        controls = QHBoxLayout()
        controls.addWidget(QLabel("IBI bin"))
        controls.addWidget(self.bin_ms)
        controls.addStretch()

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.canvas, 1)
        self._draw()
        _fix_spinbox_hit_targets(self)

    def _draw(self):
        figure = self.canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        if len(self.burst_intervals) < 2:
            ax.text(0.5, 0.5, "Need at least two detected bursts", ha="center", va="center")
        else:
            burst_starts = np.asarray([start for start, _ in self.burst_intervals], dtype=float)
            ibis_ms = np.diff(burst_starts) * 1000.0
            bin_ms = max(1.0, float(self.bin_ms.value()))
            max_ibi = max(bin_ms, float(np.nanmax(ibis_ms)))
            bins = np.arange(0.0, max_ibi + bin_ms * 1.5, bin_ms)
            ax.hist(ibis_ms, bins=bins, color="#ef4444", alpha=0.78, edgecolor="#991b1b")
            ax.set_title("Overall inter-burst intervals")
            ax.set_xlabel("IBI (ms)")
            ax.set_ylabel("Count")
        self.canvas.draw_idle()


class ISIWindow(QDialog):
    def __init__(self, spike_series, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inter-spike Interval")
        self.resize(760, 520)
        self.spike_series = [(label, np.asarray(times, dtype=float)) for label, times in spike_series]
        self.spike_lookup = {label: times for label, times in self.spike_series}
        self.unit_combo = QComboBox()
        self.unit_combo.addItems([label for label, _ in self.spike_series])
        self.unit_combo.currentIndexChanged.connect(self._draw)
        self.bin_ms = QSpinBox()
        self.bin_ms.setRange(1, 1000)
        self.bin_ms.setSingleStep(1)
        self.bin_ms.setValue(50)
        self.bin_ms.setSuffix(" ms")
        self.bin_ms.valueChanged.connect(self._draw)
        self.canvas = FigureCanvas(Figure(figsize=(7, 4), tight_layout=True))

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Unit"))
        controls.addWidget(self.unit_combo, 1)
        controls.addWidget(QLabel("ISI bin"))
        controls.addWidget(self.bin_ms)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.canvas, 1)
        self._draw()
        _fix_spinbox_hit_targets(self)

    def _draw(self):
        figure = self.canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        unit = self.unit_combo.currentText()
        times = np.asarray(self.spike_lookup.get(unit, []), dtype=float)
        if times.size < 2:
            ax.text(0.5, 0.5, "Need at least two spikes for ISI", ha="center", va="center")
        else:
            isi_ms = np.diff(np.sort(times)) * 1000.0
            isi_ms = isi_ms[(isi_ms >= 0.0) & (isi_ms <= 1000.0)]
            if isi_ms.size == 0:
                ax.text(0.5, 0.5, "No ISI values within 0-1000 ms", ha="center", va="center")
            else:
                bin_ms = max(1.0, float(self.bin_ms.value()))
                bins = np.arange(0.0, 1000.0 + bin_ms, bin_ms)
                ax.hist(isi_ms, bins=bins, color="#2563eb", alpha=0.78, edgecolor="#1e3a8a")
            ax.set_title(f"{unit} inter-spike intervals")
            ax.set_xlabel("ISI (ms)")
            ax.set_ylabel("Count")
            ax.set_xlim(0.0, 1000.0)
        self.canvas.draw_idle()


class BurstDelayWorker(QRunnable):
    def __init__(
        self,
        spike_series,
        burst_intervals,
        *,
        max_channels: int,
        max_lag_ms: float,
        min_lag_ms: float,
        delay_mode: str,
        burst_window_ms: float,
        bin_ms: float,
        min_peak_count: int,
        min_peak_fraction: float,
        min_peak_ratio: float,
    ):
        super().__init__()
        self.spike_series = [(label, np.asarray(times, dtype=float)) for label, times in spike_series]
        self.burst_intervals = [(float(start), float(stop)) for start, stop in burst_intervals]
        self.max_channels = int(max_channels)
        self.max_lag_ms = float(max_lag_ms)
        self.min_lag_ms = float(min_lag_ms)
        self.delay_mode = str(delay_mode)
        self.burst_window_ms = float(burst_window_ms)
        self.bin_ms = float(bin_ms)
        self.min_peak_count = int(min_peak_count)
        self.min_peak_fraction = float(min_peak_fraction)
        self.min_peak_ratio = float(min_peak_ratio)
        self.signals = WorkerSignals()
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return bool(self._cancel_requested)

    def _progress(self, value: int, message: str) -> None:
        self.signals.progress.emit(int(value), str(message))

    @Slot()
    def run(self):
        try:
            if self._is_cancelled():
                raise InterruptedError("Burst delay analysis cancelled")
            self._progress(8, "Preparing channel spike trains...")
            channels, channel_trains = _burst_delay_channel_series(
                self.spike_series,
                max_channels=self.max_channels,
            )
            intervals = _burst_delay_limited_intervals(self.burst_intervals, self.burst_window_ms)
            first_times = np.zeros((0, len(channels)), dtype=float)
            aligned_pairs = []

            if self.delay_mode == "burst_first":
                self._progress(12, "Preparing burst first-spike matrix...")
                channels, intervals, first_times = _burst_delay_first_spike_matrix(
                    self.spike_series,
                    self.burst_intervals,
                    max_channels=self.max_channels,
                    burst_window_ms=self.burst_window_ms,
                )
                _channels, channel_trains = _burst_delay_channel_series(
                    self.spike_series,
                    max_channels=self.max_channels,
                )
                if self._is_cancelled():
                    raise InterruptedError("Burst delay analysis cancelled")
                self._progress(32, "Scanning burst first-spike delay pairs...")
                aligned_pairs = _burst_delay_aligned_pairs(
                    channels,
                    first_times,
                    max_abs_delay_ms=self.max_lag_ms,
                    min_abs_delay_ms=self.min_lag_ms,
                    bin_ms=self.bin_ms,
                    min_peak_count=self.min_peak_count,
                    min_peak_fraction=self.min_peak_fraction,
                    min_peak_to_background=self.min_peak_ratio,
                    cancel_check=self._is_cancelled,
                    progress_callback=self._progress,
                )
            else:
                analysis_intervals = intervals if self.delay_mode == "burst_all" else None
                if self._is_cancelled():
                    raise InterruptedError("Burst delay analysis cancelled")
                self._progress(32, "Scanning source-target spike train delays...")
                aligned_pairs = _spike_train_delay_aligned_pairs(
                    channels,
                    channel_trains,
                    analysis_intervals,
                    max_abs_delay_ms=self.max_lag_ms,
                    min_abs_delay_ms=self.min_lag_ms,
                    bin_ms=self.bin_ms,
                    min_peak_count=self.min_peak_count,
                    min_peak_fraction=self.min_peak_fraction,
                    min_peak_to_background=self.min_peak_ratio,
                    cancel_check=self._is_cancelled,
                    progress_callback=self._progress,
                    mode=self.delay_mode,
                )
            self._progress(98, "Preparing visualizations...")
            self.signals.finished.emit(
                {
                    "channels": channels,
                    "intervals": intervals,
                    "first_times": first_times,
                    "channel_trains": channel_trains,
                    "delay_mode": self.delay_mode,
                    "aligned_pairs": aligned_pairs,
                }
            )
        except InterruptedError as exc:
            self.signals.canceled.emit(str(exc) or "Burst delay analysis cancelled")
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class BurstDelayWindow(QDialog):
    def __init__(
        self,
        spike_series,
        burst_intervals,
        parent=None,
        channel_map: ChannelMap | None = None,
        waveform_series=None,
        sampling_rate=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Burst Channel Delay")
        self.resize(1180, 720)
        self.spike_series = [(label, np.asarray(times, dtype=float)) for label, times in spike_series]
        self.burst_intervals = [(float(start), float(stop)) for start, stop in burst_intervals]
        self.waveform_series = {str(label): np.asarray(values) for label, values in (waveform_series or {}).items()}
        self.sampling_rate = sampling_rate
        self.channel_map = channel_map
        self.position_lookup, self.electrode_positions = _channel_map_positions(channel_map)
        self._map_recorded_xy, self._map_background_xy = self._channel_map_point_arrays()
        self.channels = []
        self.intervals = []
        self.first_times = np.zeros((0, 0), dtype=float)
        self.channel_trains = []
        self.delay_mode = "burst_first"
        self.aligned_pairs = []
        self.thread_pool = QThreadPool.globalInstance()
        self.active_delay_worker = None
        self.delay_progress = None
        self._analysis_pending = False
        self._map_ax = None
        self._map_channel_indices = np.array([], dtype=int)
        self._map_channel_xy = np.zeros((0, 2), dtype=float)
        self._map_channel_labels = []
        self._pair_combo_updating = False
        self._pair_anchor = None
        self._manual_reference_index = -1
        self._manual_target_index = -1
        self._manual_pair_active = False
        self._highlight_channel_index = -1

        self.max_channels = QSpinBox()
        available_channels = len(
            {
                _base_channel_from_raster_label(str(label))
                for label, times in self.spike_series
                if np.asarray(times, dtype=float).size and " noise" not in str(label).lower()
            }
        )
        self.max_channels.setRange(2, max(2, available_channels))
        self.max_channels.setValue(max(2, available_channels))
        self.max_channels.valueChanged.connect(self._mark_analysis_stale)

        self.max_lag_ms = QSpinBox()
        self.max_lag_ms.setRange(1, 10000)
        self.max_lag_ms.setSingleStep(5)
        self.max_lag_ms.setValue(100)
        self.max_lag_ms.setSuffix(" ms")
        self.max_lag_ms.valueChanged.connect(self._mark_analysis_stale)

        self.min_lag_ms = QDoubleSpinBox()
        self.min_lag_ms.setRange(0.0, 10000.0)
        self.min_lag_ms.setDecimals(1)
        self.min_lag_ms.setSingleStep(0.5)
        self.min_lag_ms.setValue(1.0)
        self.min_lag_ms.setSuffix(" ms")
        self.min_lag_ms.valueChanged.connect(self._mark_analysis_stale)

        self.burst_window_ms = QDoubleSpinBox()
        self.burst_window_ms.setRange(0.0, 100000.0)
        self.burst_window_ms.setDecimals(1)
        self.burst_window_ms.setSingleStep(5.0)
        self.burst_window_ms.setValue(0.0)
        self.burst_window_ms.setSuffix(" ms")
        self.burst_window_ms.valueChanged.connect(self._mark_analysis_stale)

        self.bin_ms = QDoubleSpinBox()
        self.bin_ms.setRange(0.1, 1000.0)
        self.bin_ms.setDecimals(1)
        self.bin_ms.setSingleStep(1.0)
        self.bin_ms.setValue(2.0)
        self.bin_ms.setSuffix(" ms")
        self.bin_ms.valueChanged.connect(self._mark_analysis_stale)

        self.min_peak_count = QSpinBox()
        self.min_peak_count.setRange(1, 100000)
        self.min_peak_count.setValue(25)
        self.min_peak_count.valueChanged.connect(self._mark_analysis_stale)

        self.min_peak_fraction = QDoubleSpinBox()
        self.min_peak_fraction.setRange(0.01, 1.0)
        self.min_peak_fraction.setDecimals(2)
        self.min_peak_fraction.setSingleStep(0.05)
        self.min_peak_fraction.setValue(0.50)
        self.min_peak_fraction.valueChanged.connect(self._mark_analysis_stale)

        self.min_peak_ratio = QDoubleSpinBox()
        self.min_peak_ratio.setRange(1.0, 1000.0)
        self.min_peak_ratio.setDecimals(1)
        self.min_peak_ratio.setSingleStep(1.0)
        self.min_peak_ratio.setValue(10.0)
        self.min_peak_ratio.valueChanged.connect(self._mark_analysis_stale)

        self.max_map_pairs = QSpinBox()
        self.max_map_pairs.setRange(1, 200)
        self.max_map_pairs.setValue(30)
        self.max_map_pairs.valueChanged.connect(lambda *_: self._draw_channel_map())

        self.delay_mode_combo = QComboBox()
        self.delay_mode_combo.addItem("Burst first spike", "burst_first")
        self.delay_mode_combo.addItem("Burst all spikes", "burst_all")
        self.delay_mode_combo.addItem("All spikes", "all_spikes")
        self.delay_mode_combo.currentIndexChanged.connect(self._mark_analysis_stale)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Selected pair", "pair")
        self.mode_combo.currentIndexChanged.connect(self._draw_histogram)

        self.map_pick_combo = QComboBox()
        self.map_pick_combo.addItem("Source", "source")
        self.map_pick_combo.addItem("Target", "target")

        self.manual_pair_check = QCheckBox("Manual pair")
        self.manual_pair_check.toggled.connect(self._manual_pair_toggled)

        self.significance_button = QPushButton("Significance")
        self.significance_button.clicked.connect(self._open_significance_dialog)
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self._draw)

        self.reference_combo = QComboBox()
        self.reference_combo.currentIndexChanged.connect(lambda *_: self._pair_selection_changed("source"))
        self.target_combo = QComboBox()
        self.target_combo.currentIndexChanged.connect(lambda *_: self._pair_selection_changed("target"))
        self.raster_align_combo = QComboBox()
        self.raster_align_combo.addItem("Burst onset", "onset")
        self.raster_align_combo.addItem("Source anchored", "source")
        self.raster_align_combo.currentIndexChanged.connect(self._draw_delay_raster)

        self.summary = QLabel()
        self.summary.setObjectName("MutedText")
        self.hist_canvas = FigureCanvas(Figure(figsize=(6, 3), tight_layout=True))
        self.delay_raster_canvas = FigureCanvas(Figure(figsize=(6, 3), tight_layout=True))
        self.waveform_canvas = FigureCanvas(Figure(figsize=(6, 2.4), tight_layout=True))
        self.map_canvas = FigureCanvas(Figure(figsize=(6, 5), tight_layout=False))
        self.map_canvas.mpl_connect("button_press_event", self._map_clicked)
        self.aligned_table = QTableWidget(0, 8)
        self.aligned_table.setHorizontalHeaderLabels(
            ["Source", "Target", "Delay ms", "Peak", "Total", "Peak frac", "Peak/bg", "Peak SD"]
        )
        self.aligned_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.aligned_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.aligned_table.itemSelectionChanged.connect(self._aligned_pair_selected)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Delay mode"))
        controls.addWidget(self.delay_mode_combo)
        controls.addWidget(QLabel("Max lag"))
        controls.addWidget(self.max_lag_ms)
        controls.addWidget(QLabel("Min delay"))
        controls.addWidget(self.min_lag_ms)
        controls.addWidget(QLabel("Burst window"))
        controls.addWidget(self.burst_window_ms)
        controls.addWidget(QLabel("Histogram bin"))
        controls.addWidget(self.bin_ms)
        controls.addWidget(self.significance_button)
        controls.addWidget(self.analyze_button)
        controls.addWidget(QLabel("Source"))
        controls.addWidget(self.reference_combo, 1)
        controls.addWidget(QLabel("Target"))
        controls.addWidget(self.target_combo, 1)
        controls.addWidget(QLabel("Raster align"))
        controls.addWidget(self.raster_align_combo)

        right_plots = QVBoxLayout()
        right_plots.addWidget(self.hist_canvas, 1)
        right_plots.addWidget(self.delay_raster_canvas, 1)
        right_plots.addWidget(self.waveform_canvas, 1)
        plot_area = QHBoxLayout()
        plot_area.addWidget(self.map_canvas, 3)
        plot_area.addLayout(right_plots, 2)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.summary)
        layout.addLayout(plot_area, 1)
        QTimer.singleShot(0, self._draw)
        _fix_spinbox_hit_targets(self)
        self.showMaximized()

    def _mark_analysis_stale(self, *_args) -> None:
        if not hasattr(self, "summary"):
            return
        if self.active_delay_worker is not None:
            self.summary.setText("Burst delay parameters changed; click Analyze after the current run finishes.")
            return
        current = self.summary.text().strip()
        if current and "click Analyze" not in current:
            self.summary.setText(f"{current} Parameters changed; click Analyze.")
        elif not current:
            self.summary.setText("Burst delay parameters changed; click Analyze.")

    def _open_significance_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Significance Parameters")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        peak_count = QSpinBox()
        peak_count.setRange(self.min_peak_count.minimum(), self.min_peak_count.maximum())
        peak_count.setValue(int(self.min_peak_count.value()))

        peak_fraction = QDoubleSpinBox()
        peak_fraction.setRange(self.min_peak_fraction.minimum(), self.min_peak_fraction.maximum())
        peak_fraction.setDecimals(2)
        peak_fraction.setSingleStep(0.05)
        peak_fraction.setValue(float(self.min_peak_fraction.value()))

        peak_ratio = QDoubleSpinBox()
        peak_ratio.setRange(self.min_peak_ratio.minimum(), self.min_peak_ratio.maximum())
        peak_ratio.setDecimals(1)
        peak_ratio.setSingleStep(1.0)
        peak_ratio.setValue(float(self.min_peak_ratio.value()))

        form.addRow("Min peak count", peak_count)
        form.addRow("Min peak fraction", peak_fraction)
        form.addRow("Min peak/background", peak_ratio)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        apply_button = QPushButton("Apply")
        cancel_button = QPushButton("Cancel")
        buttons.addWidget(cancel_button)
        buttons.addWidget(apply_button)
        layout.addLayout(buttons)
        apply_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        _fix_spinbox_hit_targets(dialog)

        if dialog.exec() != QDialog.Accepted:
            return
        self.min_peak_count.setValue(int(peak_count.value()))
        self.min_peak_fraction.setValue(float(peak_fraction.value()))
        self.min_peak_ratio.setValue(float(peak_ratio.value()))
        self._mark_analysis_stale()

    def _draw(self):
        if self.active_delay_worker is not None:
            self.summary.setText("Burst delay analysis is already running.")
            return
        self._analysis_pending = False
        self._set_controls_enabled(False)
        self.delay_progress = _create_progress_dialog(self, "Burst delay", "Starting burst delay analysis...", 100)
        worker = BurstDelayWorker(
            self.spike_series,
            self.burst_intervals,
            max_channels=int(self.max_channels.value()),
            max_lag_ms=float(self.max_lag_ms.value()),
            min_lag_ms=float(self.min_lag_ms.value()),
            delay_mode=str(self.delay_mode_combo.currentData() or "burst_first"),
            burst_window_ms=(
                float(self.burst_window_ms.value())
                if str(self.delay_mode_combo.currentData() or "burst_first") != "all_spikes"
                else 0.0
            ),
            bin_ms=float(self.bin_ms.value()),
            min_peak_count=int(self.min_peak_count.value()),
            min_peak_fraction=float(self.min_peak_fraction.value()),
            min_peak_ratio=float(self.min_peak_ratio.value()),
        )
        self.delay_progress.canceled.connect(worker.cancel)
        worker.signals.progress.connect(lambda value, message: _set_progress_dialog(self.delay_progress, message, value))
        worker.signals.finished.connect(lambda result, worker=worker: self._analysis_finished(result, worker))
        worker.signals.failed.connect(lambda details, worker=worker: self._analysis_failed(details, worker))
        worker.signals.canceled.connect(lambda message, worker=worker: self._analysis_canceled(message, worker))
        self.active_delay_worker = worker
        self.thread_pool.start(worker)

    def _set_controls_enabled(self, enabled: bool):
        for widget in [
            self.max_channels,
            self.max_lag_ms,
            self.min_lag_ms,
            self.burst_window_ms,
            self.bin_ms,
            self.min_peak_count,
            self.min_peak_fraction,
            self.min_peak_ratio,
            self.max_map_pairs,
            self.delay_mode_combo,
            self.map_pick_combo,
            self.mode_combo,
            self.manual_pair_check,
            self.significance_button,
            self.analyze_button,
            self.reference_combo,
            self.target_combo,
            self.raster_align_combo,
        ]:
            widget.setEnabled(enabled)

    def _finish_active_worker(self, worker: BurstDelayWorker):
        if self.active_delay_worker is worker:
            self.active_delay_worker = None
        _close_progress_dialog(self.delay_progress)
        self.delay_progress = None
        self._set_controls_enabled(True)

    def _analysis_finished(self, result: dict, worker: BurstDelayWorker):
        previous_reference = self.reference_combo.currentText()
        previous_target = self.target_combo.currentText()
        self._finish_active_worker(worker)
        self.channels = list(result.get("channels", []))
        self.intervals = list(result.get("intervals", []))
        self.first_times = np.asarray(result.get("first_times", np.zeros((0, 0))), dtype=float)
        self.channel_trains = [np.asarray(train, dtype=float) for train in result.get("channel_trains", [])]
        self.delay_mode = str(result.get("delay_mode", "burst_first"))
        self.aligned_pairs = list(result.get("aligned_pairs", []))
        truncated_bursts = max((int(item.get("truncated_bursts", 0)) for item in self.aligned_pairs), default=0)
        note = (
            f"{self._delay_mode_label()} analyzed {len(self.channels)} channels across {len(self.intervals)} bursts; "
            f"{len(self.aligned_pairs)} aligned delay patterns available for the map."
        )
        if truncated_bursts:
            note += f" {truncated_bursts} dense bursts were limited to the most active channels for responsiveness."
        burst_window_ms = float(self.burst_window_ms.value())
        if self.delay_mode != "all_spikes" and burst_window_ms > 0:
            note += f" Burst window: onset + {burst_window_ms:g} ms."
        self.summary.setText(note)
        self._refresh_channel_combos(previous_reference, previous_target)
        self._draw_histogram()
        self._draw_delay_raster()
        self._draw_waveforms()
        self._draw_channel_map()
        if self._analysis_pending:
            QTimer.singleShot(0, self._draw)

    def _analysis_failed(self, details: str, worker: BurstDelayWorker):
        self._finish_active_worker(worker)
        QMessageBox.critical(self, "Burst delay failed", details.splitlines()[-1] if details else "Unknown error")
        if self._analysis_pending:
            QTimer.singleShot(0, self._draw)

    def _analysis_canceled(self, message: str, worker: BurstDelayWorker):
        self._finish_active_worker(worker)
        self.summary.setText(message or "Burst delay analysis cancelled")
        if self._analysis_pending:
            QTimer.singleShot(0, self._draw)

    def closeEvent(self, event):  # noqa: N802 - Qt override
        if self.active_delay_worker is not None:
            self.active_delay_worker.cancel()
        _close_progress_dialog(self.delay_progress)
        self.delay_progress = None
        event.accept()

    def _channel_map_point_arrays(self):
        recorded_points = []
        background_points = []
        for _electrode, (x, y, payload) in self.electrode_positions.items():
            recorded = bool(payload.get("channel")) or bool(payload.get("routed"))
            target = recorded_points if recorded else background_points
            target.append((float(x), float(y)))
        recorded_xy = np.asarray(recorded_points, dtype=float) if recorded_points else np.zeros((0, 2), dtype=float)
        background_xy = np.asarray(background_points, dtype=float) if background_points else np.zeros((0, 2), dtype=float)
        return recorded_xy, background_xy

    def _refresh_channel_position_cache(self):
        indices = []
        points = []
        labels = []
        for index, channel in enumerate(self.channels):
            position = _position_for_channel(channel, self.position_lookup)
            if position is None:
                continue
            indices.append(index)
            points.append((float(position[0]), float(position[1])))
            labels.append(str(channel))
        self._map_channel_indices = np.asarray(indices, dtype=int) if indices else np.array([], dtype=int)
        self._map_channel_xy = np.asarray(points, dtype=float) if points else np.zeros((0, 2), dtype=float)
        self._map_channel_labels = labels

    def _refresh_channel_combos(self, previous_reference: str, previous_target: str, anchor: str | None = None):
        self._refresh_channel_position_cache()
        pairs = self._significant_pair_records()
        self._pair_combo_updating = True
        try:
            if not pairs:
                self.reference_combo.clear()
                self.target_combo.clear()
                self._pair_anchor = None
                return

            all_sources = sorted({reference for reference, _target, _result in pairs}, key=self._channel_index_sort_key)
            previous_reference_index = self._channel_index_for_label(previous_reference)
            previous_target_index = self._channel_index_for_label(previous_target)
            current_reference_index = self._combo_channel_index(self.reference_combo)
            current_target_index = self._combo_channel_index(self.target_combo)
            reference_index = previous_reference_index if previous_reference_index in all_sources else current_reference_index
            if reference_index not in all_sources:
                reference_index = -1

            target_options = []
            if reference_index >= 0:
                target_options = sorted(
                    {target for reference, target, _result in pairs if reference == reference_index},
                    key=self._channel_index_sort_key,
                )
            if anchor == "source":
                target_index = -1
            else:
                target_index = previous_target_index if previous_target_index in target_options else current_target_index
                if target_index not in target_options:
                    target_index = -1

            self._pair_anchor = "source" if reference_index >= 0 else None
            self._populate_pair_combo(self.reference_combo, all_sources, reference_index)
            self._populate_pair_combo(self.target_combo, target_options, target_index)
        finally:
            self._pair_combo_updating = False

    def _significant_pair_records(self) -> list[tuple[int, int, dict]]:
        records = []
        for result in self.aligned_pairs:
            try:
                reference_index = int(result.get("reference_index", -1))
                target_index = int(result.get("target_index", -1))
            except (TypeError, ValueError):
                continue
            if 0 <= reference_index < len(self.channels) and 0 <= target_index < len(self.channels):
                records.append((reference_index, target_index, result))
        return records

    def _channel_index_sort_key(self, index: int):
        label = self.channels[int(index)] if 0 <= int(index) < len(self.channels) else ""
        return _channel_sort_key(str(label))

    def _channel_index_for_label(self, label: str) -> int:
        if label in self.channels:
            return int(self.channels.index(label))
        return -1

    def _populate_pair_combo(self, combo: QComboBox, indices: list[int], selected_index: int) -> None:
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem("None", -1)
            for index in indices:
                combo.addItem(self.channels[index], index)
            item_index = combo.findData(int(selected_index))
            combo.setCurrentIndex(item_index if item_index >= 0 else 0)
        finally:
            combo.blockSignals(False)

    def _pair_result(self, reference_index: int, target_index: int) -> dict | None:
        for result_reference, result_target, result in self._significant_pair_records():
            if result_reference == int(reference_index) and result_target == int(target_index):
                return result
        return None

    def _selected_pair_result(self) -> dict | None:
        if self._manual_pair_active or self.manual_pair_check.isChecked():
            return self._manual_pair_result(self._manual_reference_index, self._manual_target_index)
        reference_index = self._combo_channel_index(self.reference_combo)
        target_index = self._combo_channel_index(self.target_combo)
        result = self._pair_result(reference_index, target_index)
        return result

    def _manual_pair_result(self, reference_index: int, target_index: int) -> dict | None:
        if reference_index < 0 or target_index < 0 or reference_index == target_index:
            return None
        if reference_index >= len(self.channels) or target_index >= len(self.channels):
            return None
        values = self._pair_delay_values(reference_index, target_index)
        if values.size == 0:
            delay_ms = 0.0
            peak_count = 0
            delay_window_count = 0
            total_count = 0
        else:
            bin_ms = max(0.1, float(self.bin_ms.value()))
            max_lag = max(1.0, float(self.max_lag_ms.value()))
            if self.delay_mode == "burst_first":
                bin_count = max(1, int(np.floor((2.0 * max_lag) / bin_ms)) + 1)
                bin_indices = np.floor((values + max_lag) / bin_ms + 0.5).astype(int)
            else:
                bin_count = max(1, int(np.floor(max_lag / bin_ms)) + 1)
                bin_indices = np.floor(values / bin_ms + 0.5).astype(int)
            bin_indices = np.clip(bin_indices, 0, bin_count - 1)
            counts = np.bincount(bin_indices, minlength=bin_count)
            peak_index = int(np.argmax(counts))
            delay_window_indices = range(max(0, peak_index - 2), min(bin_count, peak_index + 3))
            window_mask = np.isin(bin_indices, list(delay_window_indices))
            window_values = values[window_mask]
            delay_ms = float(np.mean(window_values)) if window_values.size else float(np.mean(values))
            peak_count = int(counts[peak_index])
            delay_window_count = int(window_values.size)
            total_count = int(values.size)
        return {
            "reference_index": int(reference_index),
            "target_index": int(target_index),
            "reference": str(self.channels[reference_index]),
            "target": str(self.channels[target_index]),
            "delay_ms": float(delay_ms),
            "peak_center_ms": float(delay_ms),
            "peak_count": int(peak_count),
            "delay_window_count": int(delay_window_count),
            "total_count": int(total_count),
            "peak_fraction": 0.0,
            "peak_to_background": 0.0,
            "background_count": 0.0,
            "std_ms": 0.0,
            "truncated_bursts": 0,
            "manual": True,
        }

    def _delay_mode_label(self) -> str:
        labels = {
            "burst_first": "Burst first-spike delay",
            "burst_all": "Burst all-spike delay",
            "all_spikes": "All-spike delay",
        }
        return labels.get(str(self.delay_mode), "Burst delay")

    def _selected_pair_delay_values(self, selected_pair: dict | None) -> np.ndarray:
        if selected_pair is None:
            return np.array([], dtype=float)
        try:
            reference_index = int(selected_pair["reference_index"])
            target_index = int(selected_pair["target_index"])
        except (KeyError, TypeError, ValueError):
            return np.array([], dtype=float)
        return self._pair_delay_values(reference_index, target_index)

    def _pair_delay_values(self, reference_index: int, target_index: int) -> np.ndarray:
        max_lag = max(1.0, float(self.max_lag_ms.value()))
        min_lag = max(0.0, float(self.min_lag_ms.value()))
        if self.delay_mode == "burst_first":
            if self.first_times.ndim != 2 or reference_index >= self.first_times.shape[1] or target_index >= self.first_times.shape[1]:
                return np.array([], dtype=float)
            return _burst_delay_pair_values(
                self.first_times,
                reference_index,
                target_index,
                max_lag,
                min_abs_delay_ms=min_lag,
            )
        if reference_index >= len(self.channel_trains) or target_index >= len(self.channel_trains):
            return np.array([], dtype=float)
        intervals = self.intervals if self.delay_mode == "burst_all" else None
        return _source_interval_delay_values(
            self.channel_trains[reference_index],
            self.channel_trains[target_index],
            max_lag,
            min_lag,
            intervals,
        )

    def _selected_pair_delay_matches(self, selected_pair: dict | None) -> np.ndarray:
        if selected_pair is None:
            return np.zeros((0, 3), dtype=float)
        try:
            reference_index = int(selected_pair["reference_index"])
            target_index = int(selected_pair["target_index"])
        except (KeyError, TypeError, ValueError):
            return np.zeros((0, 3), dtype=float)
        if reference_index >= len(self.channel_trains) or target_index >= len(self.channel_trains):
            return np.zeros((0, 3), dtype=float)
        intervals = self.intervals if self.delay_mode == "burst_all" else None
        return _source_interval_delay_matches(
            self.channel_trains[reference_index],
            self.channel_trains[target_index],
            max(1.0, float(self.max_lag_ms.value())),
            max(0.0, float(self.min_lag_ms.value())),
            intervals,
        )

    def _combo_channel_index(self, combo: QComboBox) -> int:
        data = combo.currentData()
        try:
            return int(data)
        except (TypeError, ValueError):
            return -1

    def _set_combo_channel_index(self, combo: QComboBox, channel_index: int) -> None:
        item_index = combo.findData(int(channel_index))
        if item_index < 0 and 0 <= int(channel_index) < len(self.channels):
            insert_at = max(0, combo.count() - 1 if combo.count() and combo.itemData(combo.count() - 1) == -1 else combo.count())
            combo.insertItem(insert_at, self.channels[int(channel_index)], int(channel_index))
            item_index = insert_at
        if item_index >= 0:
            combo.setCurrentIndex(item_index)

    def _map_clicked(self, event) -> None:
        if event.inaxes is not self._map_ax or event.xdata is None or event.ydata is None:
            return
        if self._map_channel_xy.size == 0:
            return
        display_points = self._map_ax.transData.transform(self._map_channel_xy)
        click = np.asarray([float(event.x), float(event.y)], dtype=float)
        distances = np.sum((display_points - click) ** 2, axis=1)
        nearest_pos = int(np.argmin(distances))
        nearest_distance = float(np.sqrt(distances[nearest_pos]))
        if nearest_distance > 12.0:
            return
        channel_index = int(self._map_channel_indices[nearest_pos])
        if channel_index < 0 or channel_index >= len(self.channels):
            return

        role = self.map_pick_combo.currentData()
        if event.button == 3:
            role = "target"
        elif event.button == 1 and role not in {"source", "target"}:
            role = "source"
        self._highlight_channel_index = int(channel_index)
        self._manual_pair_active = True
        self._set_manual_pair_channel(role, channel_index)
        self._draw_histogram()
        self._draw_delay_raster()
        self._draw_waveforms()
        self._draw_channel_map()

    def _pair_selection_changed(self, role: str) -> None:
        if self._pair_combo_updating:
            return
        self._manual_pair_active = False
        self._highlight_channel_index = -1
        if role == "target" and self._combo_channel_index(self.reference_combo) < 0:
            self._refresh_channel_combos("", "", anchor=None)
            self._draw_histogram()
            self._draw_delay_raster()
            self._draw_waveforms()
            self._draw_channel_map()
            return
        self._refresh_channel_combos(
            self.reference_combo.currentText(),
            self.target_combo.currentText(),
            anchor=role,
        )
        self._draw_histogram()
        self._draw_delay_raster()
        self._draw_waveforms()
        self._draw_channel_map()

    def _manual_pair_toggled(self):
        if self.manual_pair_check.isChecked():
            self._manual_pair_active = True
            self._manual_reference_index = self._combo_channel_index(self.reference_combo)
            self._manual_target_index = self._combo_channel_index(self.target_combo)
        else:
            self._manual_pair_active = False
        self._refresh_channel_combos(self.reference_combo.currentText(), self.target_combo.currentText(), anchor=self._pair_anchor)
        self._draw_histogram()
        self._draw_delay_raster()
        self._draw_waveforms()
        self._draw_channel_map()

    def _set_manual_pair_channel(self, role: str, channel_index: int) -> None:
        if role == "target":
            self._manual_target_index = int(channel_index)
            if self._manual_reference_index == self._manual_target_index:
                self._manual_reference_index = -1
        else:
            self._manual_reference_index = int(channel_index)
            if self._manual_target_index == self._manual_reference_index:
                self._manual_target_index = -1

    def _ensure_distinct_pair_selection(self) -> None:
        reference_index = self._combo_channel_index(self.reference_combo)
        target_index = self._combo_channel_index(self.target_combo)
        if reference_index < 0 or target_index < 0 or reference_index != target_index:
            return
        for index in range(len(self.channels)):
            if index != reference_index:
                self.target_combo.blockSignals(True)
                self._set_combo_channel_index(self.target_combo, index)
                self.target_combo.blockSignals(False)
                return

    def _select_matching_aligned_pair(self) -> None:
        reference_index = self._combo_channel_index(self.reference_combo)
        target_index = self._combo_channel_index(self.target_combo)
        if reference_index < 0 or target_index < 0:
            return
        for row, result in enumerate(self.aligned_pairs[:BURST_DELAY_TABLE_ROW_LIMIT]):
            if int(result.get("reference_index", -1)) == reference_index and int(result.get("target_index", -1)) == target_index:
                self.aligned_table.blockSignals(True)
                self.aligned_table.selectRow(row)
                self.aligned_table.blockSignals(False)
                return
        self.aligned_table.blockSignals(True)
        self.aligned_table.clearSelection()
        self.aligned_table.blockSignals(False)

    def _refresh_aligned_pairs(self):
        if self.delay_mode == "burst_first":
            self.aligned_pairs = _burst_delay_aligned_pairs(
                self.channels,
                self.first_times,
                max_abs_delay_ms=float(self.max_lag_ms.value()),
                min_abs_delay_ms=float(self.min_lag_ms.value()),
                bin_ms=float(self.bin_ms.value()),
                min_peak_count=int(self.min_peak_count.value()),
                min_peak_fraction=float(self.min_peak_fraction.value()),
                min_peak_to_background=float(self.min_peak_ratio.value()),
            )
        else:
            intervals = self.intervals if self.delay_mode == "burst_all" else None
            self.aligned_pairs = _spike_train_delay_aligned_pairs(
                self.channels,
                self.channel_trains,
                intervals,
                max_abs_delay_ms=float(self.max_lag_ms.value()),
                min_abs_delay_ms=float(self.min_lag_ms.value()),
                bin_ms=float(self.bin_ms.value()),
                min_peak_count=int(self.min_peak_count.value()),
                min_peak_fraction=float(self.min_peak_fraction.value()),
                min_peak_to_background=float(self.min_peak_ratio.value()),
                mode=self.delay_mode,
            )
        self._refresh_channel_combos(self.reference_combo.currentText(), self.target_combo.currentText(), anchor=self._pair_anchor)
        self._draw_histogram()
        self._draw_delay_raster()
        self._draw_waveforms()
        self._draw_channel_map()

    def _populate_aligned_table(self):
        self.aligned_table.blockSignals(True)
        try:
            display_pairs = self.aligned_pairs[:BURST_DELAY_TABLE_ROW_LIMIT]
            self.aligned_table.setRowCount(len(display_pairs))
            for row, result in enumerate(display_pairs):
                values = [
                    result["reference"],
                    result["target"],
                    f"{result['delay_ms']:.3f}",
                    str(result["peak_count"]),
                    str(result["total_count"]),
                    f"{result['peak_fraction']:.3f}",
                    f"{result['peak_to_background']:.1f}",
                    f"{result['std_ms']:.3f}",
                ]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if column == 0:
                        item.setData(Qt.ItemDataRole.UserRole, row)
                    self.aligned_table.setItem(row, column, item)
            self.aligned_table.resizeColumnsToContents()
        finally:
            self.aligned_table.blockSignals(False)

    def _aligned_pair_selected(self):
        items = self.aligned_table.selectedItems()
        if not items:
            return
        row = items[0].row()
        source_row = items[0].data(Qt.ItemDataRole.UserRole)
        if source_row is not None:
            try:
                row = int(source_row)
            except (TypeError, ValueError):
                pass
        if row < 0 or row >= len(self.aligned_pairs):
            return
        result = self.aligned_pairs[row]
        reference_index = int(result["reference_index"])
        target_index = int(result["target_index"])
        if reference_index >= len(self.channels) or target_index >= len(self.channels):
            return
        self._manual_pair_active = False
        self._highlight_channel_index = -1
        self.reference_combo.blockSignals(True)
        self.target_combo.blockSignals(True)
        self._set_combo_channel_index(self.reference_combo, reference_index)
        self._set_combo_channel_index(self.target_combo, target_index)
        self.reference_combo.blockSignals(False)
        self.target_combo.blockSignals(False)
        mode_index = self.mode_combo.findData("pair")
        if mode_index >= 0:
            self.mode_combo.setCurrentIndex(mode_index)
        self._draw_histogram()
        self._draw_delay_raster()
        self._draw_waveforms()
        self._draw_channel_map()

    def _delay_connected_components(self) -> list[list[int]]:
        adjacency: dict[int, set[int]] = {}
        for reference_index, target_index, _result in self._significant_pair_records():
            adjacency.setdefault(reference_index, set()).add(target_index)
            adjacency.setdefault(target_index, set()).add(reference_index)
        components = []
        visited = set()
        for start in sorted(adjacency, key=self._channel_index_sort_key):
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            component = []
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in sorted(adjacency.get(current, ()), key=self._channel_index_sort_key, reverse=True):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            components.append(sorted(component, key=self._channel_index_sort_key))
        return components

    def _component_colors(self, components: list[list[int]]) -> dict[int, str]:
        palette = [
            "#22c55e",
            "#38bdf8",
            "#f97316",
            "#e879f9",
            "#facc15",
            "#a78bfa",
            "#14b8a6",
            "#ef4444",
            "#84cc16",
            "#06b6d4",
            "#f43f5e",
            "#8b5cf6",
            "#10b981",
            "#f59e0b",
            "#0ea5e9",
            "#d946ef",
            "#65a30d",
            "#fb7185",
            "#2dd4bf",
            "#c084fc",
        ]
        colors = {}
        for component_index, component in enumerate(components):
            if component_index < len(palette):
                color = palette[component_index]
            else:
                hue = (component_index * 0.61803398875) % 1.0
                red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
                color = f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"
            for channel_index in component:
                colors[int(channel_index)] = color
        return colors

    def _first_spike_probability_window_ms(self) -> float:
        burst_window_ms = max(0.0, float(self.burst_window_ms.value())) if self.delay_mode != "all_spikes" else 0.0
        if burst_window_ms > 0:
            return float(burst_window_ms)
        durations = [float(stop - start) * 1000.0 for start, stop in self.intervals if float(stop) > float(start)]
        if durations:
            return max(1.0, max(durations))
        return max(1.0, float(self.max_lag_ms.value()))

    def _first_spike_peak_times_by_channel(self) -> dict[int, float]:
        if not self.channels or not self.intervals:
            return {}
        right_ms = self._first_spike_probability_window_ms()
        peak_times = {}
        for channel_index in range(len(self.channels)):
            fit = self._first_spike_probability_fit(channel_index, 0.0, right_ms)
            if fit is None or fit["spike_count"] <= 0 or not fit["centers_ms"].size:
                continue
            probabilities = np.asarray(fit["fit_probability"], dtype=float)
            if not probabilities.size or not np.any(np.isfinite(probabilities)):
                continue
            peak_index = int(np.nanargmax(probabilities))
            if probabilities[peak_index] <= 0:
                continue
            peak_times[int(channel_index)] = float(fit["centers_ms"][peak_index])
        return peak_times

    def _first_spike_peak_color(self, value_ms: float, low_ms: float, high_ms: float) -> str:
        if not np.isfinite(value_ms):
            return "#64748b"
        if high_ms <= low_ms:
            fraction = 0.5
        else:
            fraction = (float(value_ms) - float(low_ms)) / max(float(high_ms) - float(low_ms), 1e-9)
            fraction = max(0.0, min(1.0, fraction))
        blue = np.asarray([0, 22, 120], dtype=float)
        green = np.asarray([0, 230, 118], dtype=float)
        red = np.asarray([139, 0, 0], dtype=float)
        if fraction <= 0.5:
            local = fraction * 2.0
            color = blue + (green - blue) * local
        else:
            local = (fraction - 0.5) * 2.0
            color = green + (red - green) * local
        return f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}"

    def _channel_position_array(self, channel_indices: list[int]) -> np.ndarray:
        points = []
        for channel_index in channel_indices:
            if 0 <= int(channel_index) < len(self.channels):
                position = _position_for_channel(self.channels[int(channel_index)], self.position_lookup)
            else:
                position = None
            if position is None:
                points.append((np.nan, np.nan))
            else:
                points.append((float(position[0]), float(position[1])))
        return np.asarray(points, dtype=float) if points else np.zeros((0, 2), dtype=float)

    def _propagation_paths(self, components: list[list[int]]) -> list[dict]:
        pair_records = self._significant_pair_records()
        if not pair_records:
            return []
        pair_lookup = {(reference, target): result for reference, target, result in pair_records}
        paths = []
        for component in components:
            component = [int(index) for index in component]
            if len(component) < 10:
                continue
            local_index = {channel_index: pos for pos, channel_index in enumerate(component)}
            edges = [
                (reference, target, float(result.get("delay_ms", 0.0)))
                for reference, target, result in pair_records
                if reference in local_index and target in local_index and float(result.get("delay_ms", 0.0)) > 0
            ]
            if len(edges) < 2:
                continue
            outgoing: dict[int, list[tuple[int, dict]]] = {index: [] for index in component}
            in_degree = {index: 0 for index in component}
            out_degree = {index: 0 for index in component}
            for reference, target, _delay_ms in edges:
                result = pair_lookup.get((reference, target), {})
                outgoing.setdefault(reference, []).append((target, result))
                out_degree[reference] = out_degree.get(reference, 0) + 1
                in_degree[target] = in_degree.get(target, 0) + 1

            rows = []
            values = []
            for reference, target, delay_ms in edges:
                row = np.zeros(len(component), dtype=float)
                row[local_index[target]] = 1.0
                row[local_index[reference]] = -1.0
                rows.append(row)
                values.append(delay_ms)
            if not rows:
                continue
            matrix = np.vstack(rows)
            vector = np.asarray(values, dtype=float)
            anchored = np.vstack([matrix, np.ones(len(component), dtype=float)])
            anchored_values = np.concatenate([vector, np.array([0.0])])
            try:
                times_ms = np.linalg.lstsq(anchored, anchored_values, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            times_ms = times_ms - float(np.nanmin(times_ms))
            positions = self._channel_position_array(component)
            if not positions.size or not np.all(np.isfinite(positions)):
                continue

            def node_score(index: int) -> float:
                incoming = float(in_degree.get(index, 0))
                outgoing_count = float(out_degree.get(index, 0))
                return min(incoming, outgoing_count) + 0.12 * (incoming + outgoing_count)

            def edge_score(result: dict) -> float:
                peak_count = max(1.0, float(result.get("peak_count", 1.0)))
                peak_fraction = max(0.0, float(result.get("peak_fraction", 0.0)))
                peak_ratio = max(1.0, float(result.get("peak_to_background", 1.0)))
                return np.log1p(peak_count) * (1.0 + peak_fraction) * min(4.0, peak_ratio)

            sorted_nodes = sorted(component, key=lambda index: float(times_ms[local_index[index]]))
            start_candidates = sorted(
                [index for index in component if out_degree.get(index, 0) > 0],
                key=lambda index: (
                    out_degree.get(index, 0) - in_degree.get(index, 0),
                    out_degree.get(index, 0),
                    -float(times_ms[local_index[index]]),
                ),
                reverse=True,
            )
            end_candidates = sorted(
                [index for index in component if in_degree.get(index, 0) > 0],
                key=lambda index: (
                    in_degree.get(index, 0) - out_degree.get(index, 0),
                    in_degree.get(index, 0),
                    float(times_ms[local_index[index]]),
                ),
                reverse=True,
            )
            if not start_candidates or not end_candidates:
                continue
            start_candidates = start_candidates[: max(3, min(8, len(start_candidates)))]
            end_candidate_set = set(end_candidates[: max(3, min(8, len(end_candidates)))])

            best_score = {index: -np.inf for index in component}
            previous = {index: None for index in component}
            path_length = {index: 1 for index in component}
            for start in start_candidates:
                start_purity = max(0, out_degree.get(start, 0) - in_degree.get(start, 0))
                best_score[start] = max(best_score[start], float(start_purity + out_degree.get(start, 0)))

            for source in sorted_nodes:
                if not np.isfinite(best_score.get(source, -np.inf)):
                    continue
                source_time = float(times_ms[local_index[source]])
                for target, result in outgoing.get(source, []):
                    if target not in local_index:
                        continue
                    target_time = float(times_ms[local_index[target]])
                    if target_time <= source_time:
                        continue
                    score = best_score[source] + edge_score(result) + node_score(target)
                    if score > best_score.get(target, -np.inf):
                        best_score[target] = float(score)
                        previous[target] = source
                        path_length[target] = path_length[source] + 1

            viable_ends = [
                index
                for index in end_candidate_set
                if np.isfinite(best_score.get(index, -np.inf)) and path_length.get(index, 0) >= 3
            ]
            if not viable_ends:
                viable_ends = [
                    index
                    for index in component
                    if in_degree.get(index, 0) > out_degree.get(index, 0)
                    and np.isfinite(best_score.get(index, -np.inf))
                    and path_length.get(index, 0) >= 3
                ]
            if not viable_ends:
                continue
            end = max(
                viable_ends,
                key=lambda index: (
                    best_score[index],
                    in_degree.get(index, 0) - out_degree.get(index, 0),
                    float(times_ms[local_index[index]]),
                ),
            )
            path_nodes = []
            current = end
            while current is not None:
                path_nodes.append(int(current))
                current = previous.get(current)
            path_nodes.reverse()
            if len(path_nodes) < 3:
                continue
            path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
            paths.append(
                {
                    "component": component,
                    "layers": [[index] for index in path_nodes],
                    "times_ms": {component[pos]: float(times_ms[pos]) for pos in range(len(component))},
                    "edges": path_edges,
                    "pair_lookup": pair_lookup,
                }
            )
        return paths

    def _draw_channel_map(self):
        figure = self.map_canvas.figure
        if hasattr(figure, "set_layout_engine"):
            figure.set_layout_engine(None)
        else:
            figure.set_tight_layout(False)
        figure.clear()
        figure.patch.set_facecolor("#ffffff")
        figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        ax = figure.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_facecolor("#ffffff")
        self._map_ax = ax
        if not self.position_lookup:
            ax.text(0.5, 0.5, "No channel map coordinates available", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            self.map_canvas.draw_idle()
            return

        components = self._delay_connected_components()
        component_colors = self._component_colors(components)
        subset_indices = set(component_colors)
        propagation_paths = self._propagation_paths(components)
        peak_times = self._first_spike_peak_times_by_channel()
        peak_values = np.asarray(list(peak_times.values()), dtype=float) if peak_times else np.array([], dtype=float)
        finite_peak_values = peak_values[np.isfinite(peak_values)]
        peak_low = float(np.nanmin(finite_peak_values)) if finite_peak_values.size else 0.0
        peak_high = float(np.nanmax(finite_peak_values)) if finite_peak_values.size else 0.0
        peak_rank_fraction = {}
        if peak_times:
            sorted_peak_items = sorted(peak_times.items(), key=lambda item: (float(item[1]), self._channel_index_sort_key(item[0])))
            denominator = max(1, len(sorted_peak_items) - 1)
            for rank, (channel_index, _value) in enumerate(sorted_peak_items):
                peak_rank_fraction[int(channel_index)] = float(rank) / float(denominator)

        if self._map_background_xy.size:
            ax.scatter(
                self._map_background_xy[:, 0],
                self._map_background_xy[:, 1],
                s=3,
                color="#e5e7eb",
                alpha=0.9,
                linewidths=0,
                zorder=1,
                rasterized=True,
            )
        if self._map_recorded_xy.size:
            ax.scatter(
                self._map_recorded_xy[:, 0],
                self._map_recorded_xy[:, 1],
                s=7,
                color="#cbd5e1",
                alpha=0.85,
                linewidths=0,
                zorder=2,
                rasterized=True,
            )
        if self._map_channel_xy.size:
            point_colors = []
            point_sizes = []
            for channel_index in self._map_channel_indices:
                channel_index = int(channel_index)
                if channel_index in peak_times:
                    rank_fraction = peak_rank_fraction.get(channel_index)
                    if rank_fraction is None:
                        point_colors.append(self._first_spike_peak_color(peak_times[channel_index], peak_low, peak_high))
                    else:
                        point_colors.append(self._first_spike_peak_color(rank_fraction, 0.0, 1.0))
                else:
                    point_colors.append("#64748b")
                point_sizes.append(120 if channel_index in subset_indices else 78)
            ax.scatter(
                self._map_channel_xy[:, 0],
                self._map_channel_xy[:, 1],
                s=point_sizes,
                color=point_colors,
                edgecolors="#ffffff",
                marker="s",
                linewidths=0.8,
                alpha=0.96,
                zorder=3.2,
                rasterized=True,
            )
        selected_pair = self._selected_pair_result()
        selected_key = None
        if selected_pair is not None:
            selected_key = (int(selected_pair["reference_index"]), int(selected_pair["target_index"]))

        highlight_index = int(self._highlight_channel_index)
        for path_index, path in enumerate(propagation_paths):
            path_color = "#475569"
            for source, target in path.get("edges", []):
                if selected_key == (int(source), int(target)):
                    continue
                is_highlighted = highlight_index in {int(source), int(target)}
                source_pos = _position_for_channel(self.channels[int(source)], self.position_lookup)
                target_pos = _position_for_channel(self.channels[int(target)], self.position_lookup)
                if source_pos is None or target_pos is None:
                    continue
                ax.plot(
                    [float(source_pos[0]), float(target_pos[0])],
                    [float(source_pos[1]), float(target_pos[1])],
                    color="#111827",
                    linewidth=5.4 if is_highlighted else 4.0,
                    alpha=0.18 if is_highlighted else 0.06,
                    solid_capstyle="round",
                    zorder=4.5,
                )
                ax.annotate(
                    "",
                    xy=(float(target_pos[0]), float(target_pos[1])),
                    xytext=(float(source_pos[0]), float(source_pos[1])),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": "#d97706" if is_highlighted else path_color,
                        "lw": 3.0 if is_highlighted else 1.55,
                        "alpha": 0.96 if is_highlighted else 0.38,
                        "shrinkA": 8,
                        "shrinkB": 8,
                        "mutation_scale": 16 if is_highlighted else 10,
                    },
                    zorder=6.5 if is_highlighted else 4.8,
                )

        for reference_index, target_index, result in self._significant_pair_records():
            ref_pos = _position_for_channel(result["reference"], self.position_lookup)
            target_pos = _position_for_channel(result["target"], self.position_lookup)
            if ref_pos is None or target_pos is None:
                continue
            is_selected = selected_key == (reference_index, target_index)
            is_highlighted = highlight_index in {int(reference_index), int(target_index)}
            ax.annotate(
                "",
                xy=(float(target_pos[0]), float(target_pos[1])),
                xytext=(float(ref_pos[0]), float(ref_pos[1])),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#f97316" if is_selected else "#d97706" if is_highlighted else "#475569",
                    "lw": 3.0 if is_selected else 2.4 if is_highlighted else 1.1,
                    "alpha": 0.95 if is_selected or is_highlighted else 0.28,
                    "shrinkA": 5,
                    "shrinkB": 5,
                    "mutation_scale": 12 if is_selected or is_highlighted else 7,
                },
                zorder=7 if is_selected or is_highlighted else 4,
            )

        for component_index, component in enumerate(components):
            points = []
            for channel_index in component:
                if 0 <= channel_index < len(self.channels):
                    position = _position_for_channel(self.channels[channel_index], self.position_lookup)
                    if position is not None:
                        points.append((float(position[0]), float(position[1])))
            if not points:
                continue
            xy = np.asarray(points, dtype=float)
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                s=140,
                facecolors="none",
                edgecolors="#64748b",
                linewidths=0.9,
                alpha=0.32,
                zorder=6,
            )
            centroid = np.mean(xy, axis=0)
            ax.text(
                float(centroid[0]),
                float(centroid[1]),
                str(component_index + 1),
                color="#111827",
                fontsize=7,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=8,
            )

        for path_index, path in enumerate(propagation_paths):
            for layer_number, layer in enumerate(path.get("layers", []), start=1):
                positions = self._channel_position_array([int(index) for index in layer])
                if positions.size == 0 or not np.all(np.isfinite(positions)):
                    continue
                centroid = np.mean(positions, axis=0)
                ax.text(
                    float(centroid[0]),
                    float(centroid[1]) - 0.018,
                    f"P{path_index + 1}.{layer_number}",
                    color="#334155",
                    fontsize=6.5,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    zorder=10,
                    bbox={"boxstyle": "round,pad=0.10", "facecolor": "#ffffff", "edgecolor": "#94a3b8", "alpha": 0.82},
                )

        delay_note = f"{len(components)} sets; {len(propagation_paths)} paths"
        if finite_peak_values.size:
            delay_note += f"\nfirst-spike peak color: {peak_low:.1f}-{peak_high:.1f} ms"
        selected_pair = self._selected_pair_result()
        if selected_pair is None:
            if self._manual_pair_active or self.manual_pair_check.isChecked():
                source_label = (
                    self.channels[self._manual_reference_index]
                    if 0 <= self._manual_reference_index < len(self.channels)
                    else "None"
                )
                target_label = (
                    self.channels[self._manual_target_index]
                    if 0 <= self._manual_target_index < len(self.channels)
                    else "None"
                )
                delay_note += f"\nmanual source: {source_label}; target: {target_label}"
            else:
                delay_note += "\nSelect source, then target"
        else:
            ref_pos = _position_for_channel(selected_pair["reference"], self.position_lookup)
            target_pos = _position_for_channel(selected_pair["target"], self.position_lookup)
            if ref_pos is None or target_pos is None:
                ax.text(
                    0.5,
                    0.5,
                    "Selected pair is not present in the channel map",
                    ha="center",
                    va="center",
                    color="#334155",
                    transform=ax.transAxes,
                )
            else:
                rx, ry = float(ref_pos[0]), float(ref_pos[1])
                tx, ty = float(target_pos[0]), float(target_pos[1])
                delay = float(selected_pair.get("delay_ms", 0.0))
                pair_prefix = "manual" if selected_pair.get("manual") else "selected"
                delay_note += f"\n{pair_prefix}: {selected_pair['reference']} -> {selected_pair['target']} ({delay:.2f} ms)"
                ax.scatter([rx], [ry], s=64, color="#16a34a", edgecolors="#111827", linewidths=1.0, zorder=9)
                ax.scatter([tx], [ty], s=72, color="#dc2626", edgecolors="#111827", linewidths=1.0, zorder=9)

        ax.set_xlim(-0.015, 1.015)
        ax.set_ylim(1.015, -0.015)
        ax.set_aspect("auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(
            0.01,
            0.99,
            delay_note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.10", "facecolor": "#ffffff", "edgecolor": "#cbd5e1", "alpha": 0.88},
        )
        self.map_canvas.draw_idle()

    def _waveforms_for_channel(self, channel: str) -> np.ndarray:
        channel_text = str(channel)
        chunks = []
        for label, waveforms in self.waveform_series.items():
            label_text = str(label)
            if label_text != channel_text and _base_channel_from_raster_label(label_text) != channel_text:
                continue
            array = np.asarray(waveforms, dtype=float)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
                continue
            finite_rows = np.any(np.isfinite(array), axis=1)
            array = array[finite_rows]
            if array.size:
                chunks.append(array)
        if not chunks:
            return np.zeros((0, 0), dtype=float)
        width = min(chunk.shape[1] for chunk in chunks)
        if width <= 0:
            return np.zeros((0, 0), dtype=float)
        return np.vstack([chunk[:, :width] for chunk in chunks])

    def _waveform_display_subset(self, waveforms: np.ndarray, max_traces: int = 120) -> np.ndarray:
        array = np.asarray(waveforms, dtype=float)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            return np.zeros((0, 0), dtype=float)
        finite_rows = np.any(np.isfinite(array), axis=1)
        array = array[finite_rows]
        if array.shape[0] > max_traces:
            array = array[_display_indices(array.shape[0], max_traces)]
        return array

    def _draw_waveform_axis(
        self,
        ax,
        channel: str,
        role: str,
        waveforms: np.ndarray,
        color: str,
        mean_color: str,
        ylim: tuple[float, float] | None,
    ) -> None:
        ax.set_title(f"{role}: {channel}")
        ax.set_xlabel("Time (ms)" if self.sampling_rate else "Sample")
        ax.set_ylabel("Amplitude")
        if waveforms.size == 0:
            ax.text(0.5, 0.5, "No waveform data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        x = _waveform_time_axis(waveforms.shape[1], self.sampling_rate)
        for waveform in waveforms:
            ax.plot(x, waveform, color=color, linewidth=0.65, alpha=0.16)
        ax.plot(x, np.nanmean(waveforms, axis=0), color=mean_color, linewidth=2.0, label="mean")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.text(
            0.98,
            0.95,
            f"n={waveforms.shape[0]}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.85},
        )
        ax.legend(loc="best", fontsize=8)

    def _draw_waveforms(self):
        figure = self.waveform_canvas.figure
        figure.clear()
        selected_pair = self._selected_pair_result()
        if selected_pair is None:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, "Select source, then target", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            self.waveform_canvas.draw_idle()
            return

        source = str(selected_pair.get("reference", ""))
        target = str(selected_pair.get("target", ""))
        source_waveforms = self._waveform_display_subset(self._waveforms_for_channel(source))
        target_waveforms = self._waveform_display_subset(self._waveforms_for_channel(target))
        visible = [values.reshape(-1) for values in (source_waveforms, target_waveforms) if values.size]
        ylim = None
        if visible:
            samples = np.concatenate(visible)
            samples = samples[np.isfinite(samples)]
            if samples.size:
                ymin, ymax = np.nanpercentile(samples, [1, 99])
                if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
                    ymin = float(np.nanmin(samples))
                    ymax = float(np.nanmax(samples))
                if ymin == ymax:
                    ymin -= 1.0
                    ymax += 1.0
                pad = (float(ymax) - float(ymin)) * 0.08
                ylim = (float(ymin) - pad, float(ymax) + pad)

        axes = figure.subplots(1, 2, squeeze=False)[0]
        self._draw_waveform_axis(axes[0], source, "Source", source_waveforms, "#16a34a", "#15803d", ylim)
        self._draw_waveform_axis(axes[1], target, "Target", target_waveforms, "#dc2626", "#991b1b", ylim)
        self.waveform_canvas.draw_idle()

    def _delay_raster_points(self, selected_pair: dict | None):
        if selected_pair is None:
            return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=int), 0.0
        reference_index = int(selected_pair["reference_index"])
        target_index = int(selected_pair["target_index"])
        max_lag = max(1.0, float(self.max_lag_ms.value()))

        if self.delay_mode == "burst_first":
            if self.first_times.ndim != 2 or self.first_times.shape[0] == 0:
                return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=int), 0.0
            if reference_index >= self.first_times.shape[1] or target_index >= self.first_times.shape[1]:
                return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=int), 0.0

            source_ms = np.asarray(self.first_times[:, reference_index], dtype=float) * 1000.0
            target_ms = np.asarray(self.first_times[:, target_index], dtype=float) * 1000.0
            delay_ms = target_ms - source_ms
            min_lag = max(0.0, float(self.min_lag_ms.value()))
            valid = np.isfinite(source_ms) & np.isfinite(target_ms) & (np.abs(delay_ms) >= min_lag) & (np.abs(delay_ms) <= max_lag)
            rows = np.flatnonzero(valid).astype(int)
            if not rows.size:
                return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=int), 0.0
            source_x = source_ms[rows]
            target_x = target_ms[rows]
        else:
            matches = self._selected_pair_delay_matches(selected_pair)
            if matches.size == 0:
                return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=int), 0.0
            if self.delay_mode == "burst_all":
                rows = matches[:, 2].astype(int, copy=False)
                starts = np.asarray(
                    [self.intervals[row][0] if 0 <= row < len(self.intervals) else matches[index, 0] for index, row in enumerate(rows)],
                    dtype=float,
                )
                source_x = (matches[:, 0] - starts) * 1000.0
                target_x = (matches[:, 1] - starts) * 1000.0
            else:
                rows = np.arange(matches.shape[0], dtype=int)
                finite_starts = [
                    float(train[0])
                    for train in self.channel_trains
                    if np.asarray(train, dtype=float).size and np.isfinite(np.asarray(train, dtype=float)[0])
                ]
                baseline = min(finite_starts) if finite_starts else float(matches[0, 0])
                source_x = (matches[:, 0] - baseline) * 1000.0
                target_x = (matches[:, 1] - baseline) * 1000.0

        anchor_ms = 0.0
        if self.raster_align_combo.currentData() == "source":
            anchor_ms = max(5.0, min(50.0, max_lag * 0.2))
            target_x = anchor_ms + (target_x - source_x)
            source_x = np.full_like(target_x, anchor_ms, dtype=float)
        return source_x.astype(float), target_x.astype(float), rows.astype(int), float(anchor_ms)

    def _delay_raster_xlim(self, anchor_ms: float) -> tuple[float, float, str]:
        max_lag = max(1.0, float(self.max_lag_ms.value()))
        burst_window_ms = max(0.0, float(self.burst_window_ms.value())) if self.delay_mode != "all_spikes" else 0.0
        if self.raster_align_combo.currentData() == "source":
            span_ms = min(max_lag, burst_window_ms) if burst_window_ms > 0 else max_lag
            return float(anchor_ms - span_ms), float(anchor_ms + span_ms), "Time (ms, source anchored)"
        if self.delay_mode == "all_spikes":
            starts = []
            stops = []
            for train in self.channel_trains:
                values = np.asarray(train, dtype=float)
                values = values[np.isfinite(values)]
                if values.size:
                    starts.append(float(values[0]))
                    stops.append(float(values[-1]))
            if starts and stops:
                duration_ms = max(1.0, (max(stops) - min(starts)) * 1000.0)
            else:
                duration_ms = max_lag
            return 0.0, float(duration_ms), "Time from recording start (ms)"
        if burst_window_ms > 0:
            return 0.0, float(max(1.0, burst_window_ms)), f"Time from burst onset, first {burst_window_ms:g} ms"
        burst_duration_ms = max(
            [float(stop - start) * 1000.0 for start, stop in self.intervals if float(stop) > float(start)] or [max_lag]
        )
        return 0.0, float(max(max_lag, burst_duration_ms)), "Time from burst onset (ms)"

    def _channel_first_spike_times_ms(self, channel_index: int) -> np.ndarray:
        if channel_index < 0 or channel_index >= len(self.channels):
            return np.array([], dtype=float)
        if not self.intervals:
            return np.array([], dtype=float)
        if (
            self.delay_mode == "burst_first"
            and self.first_times.ndim == 2
            and self.first_times.shape[0] == len(self.intervals)
            and channel_index < self.first_times.shape[1]
        ):
            values_ms = np.asarray(self.first_times[:, channel_index], dtype=float) * 1000.0
            return values_ms[np.isfinite(values_ms)]
        if channel_index >= len(self.channel_trains):
            return np.array([], dtype=float)
        train = np.asarray(self.channel_trains[channel_index], dtype=float)
        train = np.sort(train[np.isfinite(train)])
        if train.size == 0:
            return np.array([], dtype=float)
        values = []
        for start_s, stop_s in self.intervals:
            lo = int(np.searchsorted(train, float(start_s), side="left"))
            if lo < train.size and train[lo] <= float(stop_s):
                values.append((float(train[lo]) - float(start_s)) * 1000.0)
        return np.asarray(values, dtype=float)

    def _first_spike_probability_fit(self, channel_index: int, left_ms: float, right_ms: float) -> dict | None:
        burst_count = len(self.intervals)
        if burst_count <= 0:
            return None
        left_ms = max(0.0, float(left_ms))
        right_ms = max(left_ms + 1.0, float(right_ms))
        bin_ms = max(0.5, float(self.bin_ms.value()))
        edges = np.arange(left_ms, right_ms + bin_ms * 0.5, bin_ms, dtype=float)
        if edges.size < 2 or edges[-1] < right_ms:
            edges = np.append(edges, right_ms)
        centers = (edges[:-1] + edges[1:]) * 0.5
        first_spikes = self._channel_first_spike_times_ms(channel_index)
        first_spikes = first_spikes[(first_spikes >= left_ms) & (first_spikes <= right_ms)]
        counts, _ = np.histogram(first_spikes, bins=edges)
        counts = counts.astype(float, copy=False)
        sigma_bins = max(0.75, min(4.0, 5.0 / bin_ms))
        fitted_counts = gaussian_filter(counts, sigma=sigma_bins, mode="nearest", truncate=3.0)
        probability = np.clip(fitted_counts / float(burst_count), 0.0, 1.0)
        count_se = np.sqrt(np.maximum(fitted_counts, 1e-9))
        lower = np.clip((fitted_counts - 1.96 * count_se) / float(burst_count), 0.0, 1.0)
        upper = np.clip((fitted_counts + 1.96 * count_se) / float(burst_count), 0.0, 1.0)
        observed = counts / float(burst_count)
        return {
            "centers_ms": centers,
            "observed_probability": observed,
            "fit_probability": probability,
            "lower_probability": lower,
            "upper_probability": upper,
            "burst_count": int(burst_count),
            "spike_count": int(first_spikes.size),
        }

    def _draw_first_spike_probability_overlay(self, ax, selected_pair: dict, left_ms: float, right_ms: float):
        if self.delay_mode == "all_spikes" or self.raster_align_combo.currentData() == "source":
            return None
        try:
            source_index = int(selected_pair["reference_index"])
            target_index = int(selected_pair["target_index"])
        except (KeyError, TypeError, ValueError):
            return None
        source_fit = self._first_spike_probability_fit(source_index, left_ms, right_ms)
        target_fit = self._first_spike_probability_fit(target_index, left_ms, right_ms)
        fits = [
            ("source", source_fit, "#16a34a", "#86efac"),
            ("target", target_fit, "#dc2626", "#fecaca"),
        ]
        if not any(fit is not None and fit["centers_ms"].size for _role, fit, _line, _fill in fits):
            return None
        prob_ax = ax.twinx()
        max_probability = 0.0
        for role, fit, line_color, fill_color in fits:
            if fit is None or not fit["centers_ms"].size:
                continue
            x = fit["centers_ms"]
            y = fit["fit_probability"]
            lower = fit["lower_probability"]
            upper = fit["upper_probability"]
            observed = fit["observed_probability"]
            max_probability = max(max_probability, float(np.nanmax(upper)) if upper.size else 0.0)
            prob_ax.fill_between(x, lower, upper, color=fill_color, alpha=0.20, linewidth=0)
            prob_ax.plot(x, y, color=line_color, linewidth=1.8, linestyle="-", label=f"{role} Poisson fit")
            prob_ax.scatter(x, observed, s=10, color=line_color, alpha=0.35, linewidths=0)
        prob_ax.set_ylabel("First-spike probability")
        prob_ax.set_ylim(0.0, min(1.0, max(0.05, max_probability * 1.15)))
        prob_ax.tick_params(axis="y", labelsize=8, colors="#475569")
        prob_ax.spines["right"].set_color("#94a3b8")
        return prob_ax

    def _draw_delay_raster(self):
        figure = self.delay_raster_canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        selected_pair = self._selected_pair_result()
        source_x, target_x, burst_rows, anchor_ms = self._delay_raster_points(selected_pair)
        if selected_pair is None:
            ax.text(0.5, 0.5, "Select source, then target", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            self.delay_raster_canvas.draw_idle()
            return
        if not burst_rows.size:
            ax.text(0.5, 0.5, "No source/target spike matches", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            self.delay_raster_canvas.draw_idle()
            return

        ax.scatter(source_x, burst_rows + 1, s=14, color="#16a34a", label="source", alpha=0.88)
        ax.scatter(target_x, burst_rows + 1, s=14, color="#dc2626", label="target", alpha=0.88)
        for sx, tx, row in zip(source_x, target_x, burst_rows):
            ax.plot([sx, tx], [row + 1, row + 1], color="#94a3b8", linewidth=0.65, alpha=0.42)
        left, right, xlabel = self._delay_raster_xlim(anchor_ms)
        if self.raster_align_combo.currentData() == "source":
            ax.axvline(anchor_ms, color="#16a34a", linewidth=1.0, linestyle="--", alpha=0.75)

        ax.set_xlim(left, right)
        ax.set_ylim(max(0.5, burst_rows.min() + 0.5), burst_rows.max() + 1.5)
        ax.invert_yaxis()
        ax.set_title(f"{self._delay_mode_label()} raster")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Event" if self.delay_mode == "all_spikes" else "Burst")
        prob_ax = self._draw_first_spike_probability_overlay(ax, selected_pair, left, right)
        handles, labels = ax.get_legend_handles_labels()
        if prob_ax is not None:
            probability_handles, probability_labels = prob_ax.get_legend_handles_labels()
            handles.extend(probability_handles)
            labels.extend(probability_labels)
        if handles:
            ax.legend(handles, labels, loc="best", fontsize=7)
        self.delay_raster_canvas.draw_idle()

    def _draw_histogram(self):
        figure = self.hist_canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        if len(self.channels) < 2:
            ax.text(0.5, 0.5, "No burst delay data", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            self.hist_canvas.draw_idle()
            return

        max_lag = max(1.0, float(self.max_lag_ms.value()))
        selected_pair = self._selected_pair_result()
        if selected_pair is None:
            values = np.array([], dtype=float)
            title = f"Selected-pair {self._delay_mode_label().lower()} distribution"
        else:
            values = self._selected_pair_delay_values(selected_pair)
            title = f"{selected_pair['reference']} -> {selected_pair['target']} {self._delay_mode_label().lower()}"

        if values.size == 0:
            message = "No matched spikes for selected pair" if selected_pair and selected_pair.get("manual") else "No matched bursts for selected significant pair"
            ax.text(0.5, 0.5, message, ha="center", va="center")
            ax.set_xlim((-max_lag, max_lag) if self.delay_mode == "burst_first" else (0.0, max_lag))
        else:
            bin_ms = max(0.1, float(self.bin_ms.value()))
            start = -max_lag if self.delay_mode == "burst_first" else 0.0
            bins = np.arange(start, max_lag + bin_ms * 1.5, bin_ms)
            ax.hist(values, bins=bins, color="#2563eb", alpha=0.78, edgecolor="#1e3a8a")
            mean_delay = float(selected_pair.get("delay_ms", np.mean(values)))
            ax.axvline(0.0, color="#475569", linewidth=1.0)
            ax.axvline(mean_delay, color="#dc2626", linewidth=1.5, linestyle="--", label=f"5-bin mean {mean_delay:.2f} ms")
            ax.legend(loc="best", fontsize=8)
            ax.text(
                0.98,
                0.96,
                f"n={values.size}\n5-bin mean={mean_delay:.2f} ms",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
            )
        ax.set_title(title)
        ax.set_xlabel("Delay (ms)")
        ax.set_ylabel("Count")
        ax.set_xlim((-max_lag, max_lag) if self.delay_mode == "burst_first" else (0.0, max_lag))
        self.hist_canvas.draw_idle()

class BurstClusteringWindow(QDialog):
    def __init__(self, spike_series, burst_intervals, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Burst Clustering")
        self.resize(1280, 760)
        self.spike_series = [(label, np.asarray(times, dtype=float)) for label, times in spike_series]
        self.burst_intervals = [(float(start), float(stop)) for start, stop in burst_intervals]
        self.current = None
        self.manual_groups = np.array([], dtype=np.int32)
        self._analysis_signature = None
        self.undo_stack = []
        self.lasso = None
        self.lasso_mode = None
        self.pending_assignment_label = 0
        self.embedding_ax = None

        self.bin_ms = QDoubleSpinBox()
        self.bin_ms.setRange(0.5, 500.0)
        self.bin_ms.setDecimals(1)
        self.bin_ms.setSingleStep(1.0)
        self.bin_ms.setValue(5.0)
        self.bin_ms.setSuffix(" ms")
        self.bin_ms.valueChanged.connect(self._draw)

        self.window_ms = QSpinBox()
        self.window_ms.setRange(0, 60000)
        self.window_ms.setSingleStep(10)
        self.window_ms.setValue(0)
        self.window_ms.setSuffix(" ms")
        self.window_ms.valueChanged.connect(self._draw)

        self.cluster_count = QSpinBox()
        self.cluster_count.setRange(1, 20)
        self.cluster_count.setValue(3)
        self.cluster_count.valueChanged.connect(self._draw)

        self.reducer = QComboBox()
        self.reducer.addItem("PCA", "pca")
        self.reducer.addItem("t-SNE", "tsne")
        self.reducer.currentIndexChanged.connect(self._draw)

        self.normalize = QComboBox()
        self.normalize.addItem("Per burst", "per_burst")
        self.normalize.addItem("Time-bin z-score", "unit_zscore")
        self.normalize.addItem("None", "none")
        self.normalize.currentIndexChanged.connect(self._draw)

        self.trace_scale = QComboBox()
        self.trace_scale.addItem("Shape (per burst peak)", "per_trace_peak")
        self.trace_scale.addItem("Log count", "log")
        self.trace_scale.addItem("Robust count", "robust")
        self.trace_scale.addItem("Raw count", "raw")
        self.trace_scale.currentIndexChanged.connect(self._draw)

        self.cluster_id = QSpinBox()
        self.cluster_id.setRange(0, 99)
        self.cluster_id.setValue(0)
        self.lasso_button = QPushButton("Assign Cluster")
        self.lasso_button.setCheckable(True)
        self.lasso_button.clicked.connect(self._start_lasso)
        self.noise_button = QPushButton("Assign Noise")
        self.noise_button.setCheckable(True)
        self.noise_button.clicked.connect(self._start_noise_lasso)
        self.recluster_clean_button = QPushButton("Recluster Clean")
        self.recluster_clean_button.clicked.connect(self._recluster_clean_bursts)
        self.hide_noise = QCheckBox("Hide noise")
        self.hide_noise.stateChanged.connect(lambda *_: self._redraw_current())
        self.cluster_filter = QComboBox()
        self.cluster_filter.currentIndexChanged.connect(lambda *_: self._redraw_current())
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self._undo_manual_assignment)
        self.undo_button.setEnabled(False)

        self.summary = QLabel()
        self.summary.setObjectName("MutedText")
        self.status = QLabel("Ready")
        self.status.setObjectName("MutedText")
        self.embedding_canvas = FigureCanvas(Figure(figsize=(6, 4), tight_layout=True))
        self.trace_canvas = FigureCanvas(Figure(figsize=(7, 4), tight_layout=True))

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Bin"))
        controls.addWidget(self.bin_ms)
        controls.addWidget(QLabel("Window"))
        controls.addWidget(self.window_ms)
        controls.addWidget(QLabel("Clusters"))
        controls.addWidget(self.cluster_count)
        controls.addWidget(QLabel("Reducer"))
        controls.addWidget(self.reducer)
        controls.addWidget(QLabel("Normalize"))
        controls.addWidget(self.normalize)
        controls.addWidget(QLabel("Trace scale"))
        controls.addWidget(self.trace_scale)
        controls.addStretch(1)

        manual_controls = QHBoxLayout()
        manual_controls.addWidget(QLabel("Assign cluster"))
        manual_controls.addWidget(self.cluster_id)
        manual_controls.addWidget(self.lasso_button)
        manual_controls.addWidget(self.noise_button)
        manual_controls.addWidget(self.recluster_clean_button)
        manual_controls.addWidget(self.hide_noise)
        manual_controls.addWidget(QLabel("Show"))
        manual_controls.addWidget(self.cluster_filter)
        manual_controls.addWidget(self.undo_button)
        manual_controls.addWidget(self.status, 1)

        plots = QHBoxLayout()
        plots.addWidget(self.embedding_canvas, 1)
        plots.addWidget(self.trace_canvas, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addLayout(manual_controls)
        layout.addWidget(self.summary)
        layout.addLayout(plots, 1)
        self._draw()
        _fix_spinbox_hit_targets(self)
        self.showMaximized()

    def _draw(self):
        if self.lasso_mode is not None:
            self._stop_lasso_mode("Assignment mode off")
        intervals, centers_ms, vectors, durations = _burst_total_spike_vectors(
            self.spike_series,
            self.burst_intervals,
            bin_ms=float(self.bin_ms.value()),
            window_ms=float(self.window_ms.value()),
        )
        burst_count = vectors.shape[0]
        feature_vectors = _normalize_burst_features(vectors, self.normalize.currentData())
        embedding = _burst_embedding(feature_vectors, self.reducer.currentData()) if burst_count else np.zeros((0, 2), dtype=float)
        auto_groups = _kmeans_groups(feature_vectors, int(self.cluster_count.value())) if burst_count else np.array([], dtype=int)
        signature = self._current_analysis_signature()
        if signature != self._analysis_signature or self.manual_groups.size != auto_groups.size:
            self.manual_groups = auto_groups.astype(np.int32, copy=True)
            self.undo_stack = []
            self.undo_button.setEnabled(False)
            self._analysis_signature = signature
        groups = self.manual_groups.astype(np.int32, copy=False)
        self.current = {
            "intervals": intervals,
            "centers_ms": centers_ms,
            "vectors": vectors,
            "features": feature_vectors,
            "embedding": embedding,
            "groups": groups,
            "durations": durations,
        }
        self._refresh_cluster_filter()
        self._draw_embedding(embedding, groups, intervals)
        self._draw_traces(centers_ms, vectors, groups)
        self._update_summary()

    def _current_analysis_signature(self):
        return (
            round(float(self.bin_ms.value()), 6),
            int(self.window_ms.value()),
            int(self.cluster_count.value()),
            str(self.reducer.currentData()),
            str(self.normalize.currentData()),
            len(self.burst_intervals),
        )

    def _redraw_current(self):
        if not self.current:
            self._draw()
            return
        self._draw_embedding(self.current["embedding"], self.current["groups"], self.current["intervals"])
        self._draw_traces(self.current["centers_ms"], self.current["vectors"], self.current["groups"])
        self._update_summary()

    def _update_summary(self):
        if not self.current:
            self.summary.setText("No burst clustering result")
            return
        vectors = np.asarray(self.current.get("vectors", []), dtype=float)
        groups = np.asarray(self.current.get("groups", []), dtype=np.int32)
        burst_count = int(vectors.shape[0]) if vectors.ndim >= 2 else int(groups.size)
        active = int(np.count_nonzero(np.sum(vectors, axis=1) > 0)) if vectors.ndim == 2 and vectors.size else 0
        vector_bins = int(vectors.shape[1]) if vectors.ndim == 2 else 0
        window_text = "auto" if self.window_ms.value() <= 0 else f"{self.window_ms.value()} ms"
        group_count = len(set(int(value) for value in groups if int(value) != -1)) if groups.size else 0
        noise_count = int(np.count_nonzero(groups == -1)) if groups.size else 0
        self.summary.setText(
            f"Bursts: {burst_count} | Active bursts: {active} | Vector length: {vector_bins} bins | "
            f"Window: {window_text} | Clusters: {group_count} | Noise: {noise_count}"
        )

    def _draw_embedding(self, embedding: np.ndarray, groups: np.ndarray, intervals):
        figure = self.embedding_canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        self.embedding_ax = ax
        if embedding.shape[0] < 2:
            ax.text(0.5, 0.5, "Need at least two bursts", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            visible_mask = self._visible_group_mask(groups)
            color_map = _cluster_color_map(groups)
            plotted = False
            for group in sorted(np.unique(groups)):
                indices = np.flatnonzero((groups == group) & visible_mask)
                if indices.size == 0:
                    continue
                plotted = True
                legend_label = "noise" if int(group) == -1 else f"cluster {int(group)}"
                ax.scatter(
                    embedding[indices, 0],
                    embedding[indices, 1],
                    s=42,
                    alpha=0.86,
                    color=color_map[int(group)],
                    label=legend_label,
                )
                for index in indices:
                    ax.text(
                        float(embedding[index, 0]),
                        float(embedding[index, 1]),
                        str(index + 1),
                        fontsize=7,
                        color="#111827",
                    )
            if plotted:
                ax.legend(loc="best", fontsize=8)
            else:
                ax.text(0.5, 0.5, "No bursts match the current cluster filter", ha="center", va="center")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
        ax.set_title("Burst spike-count vector embedding")
        self.embedding_canvas.draw_idle()

    def _draw_traces(self, centers_ms: np.ndarray, vectors: np.ndarray, groups: np.ndarray):
        figure = self.trace_canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        if vectors.shape[0] == 0:
            ax.text(0.5, 0.5, "No detected bursts", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            display_vectors, ylabel = _scale_burst_trace_vectors(vectors, self.trace_scale.currentData())
            color_map = _cluster_color_map(groups)
            visible_mask = self._visible_group_mask(groups)
            plotted = False
            for group in sorted(np.unique(groups)):
                indices = np.flatnonzero((groups == group) & visible_mask)
                if indices.size == 0:
                    continue
                plotted = True
                color = color_map[int(group)]
                draw_indices = indices[_display_indices(indices.size, 40)] if indices.size else indices
                for index in draw_indices:
                    ax.plot(centers_ms, display_vectors[index], color=color, alpha=0.18, linewidth=0.9)
                mean_trace = np.mean(display_vectors[indices], axis=0) if indices.size else np.zeros_like(centers_ms)
                legend_label = "noise" if int(group) == -1 else f"cluster {int(group)} mean"
                ax.plot(centers_ms, mean_trace, color=color, linewidth=2.4, label=legend_label)
            if plotted:
                ax.legend(loc="best", fontsize=8)
            else:
                ax.text(0.5, 0.5, "No bursts match the current cluster filter", ha="center", va="center")
            ax.set_xlabel("Time from burst onset (ms)")
            ax.set_ylabel(ylabel)
        ax.set_title("Cluster mean burst trajectories")
        self.trace_canvas.draw_idle()

    def _refresh_cluster_filter(self):
        previous = self.cluster_filter.currentData()
        if previous is None:
            previous = "all"
        labels = sorted(int(label) for label in np.unique(self.manual_groups)) if self.manual_groups.size else []
        self.cluster_filter.blockSignals(True)
        self.cluster_filter.clear()
        self.cluster_filter.addItem("All clusters", "all")
        for label in labels:
            text = "noise" if label == -1 else f"cluster {label}"
            self.cluster_filter.addItem(text, label)
        index = self.cluster_filter.findData(previous)
        self.cluster_filter.setCurrentIndex(index if index >= 0 else 0)
        self.cluster_filter.blockSignals(False)

    def _active_cluster_filter(self):
        value = self.cluster_filter.currentData()
        return None if value in (None, "all") else int(value)

    def _visible_group_mask(self, groups: np.ndarray) -> np.ndarray:
        labels = np.asarray(groups, dtype=np.int32)
        mask = np.ones(labels.shape[0], dtype=bool)
        if self.hide_noise.isChecked():
            mask &= labels != -1
        active_cluster = self._active_cluster_filter()
        if active_cluster is not None:
            mask &= labels == active_cluster
        return mask

    def _start_lasso(self):
        self._toggle_lasso_mode("cluster", int(self.cluster_id.value()))

    def _start_noise_lasso(self):
        self._toggle_lasso_mode("noise", -1)

    def _toggle_lasso_mode(self, mode: str, label: int):
        if self.lasso_mode == mode:
            self._stop_lasso_mode("Assignment mode off")
            return
        self.lasso_mode = mode
        self.pending_assignment_label = int(label)
        self._refresh_lasso_button_states()
        message = "Draw regions to mark noise; right-click to exit" if mode == "noise" else (
            f"Draw regions to assign cluster {label}; right-click to exit"
        )
        self._begin_lasso(message)

    def _begin_lasso(self, message: str):
        if self.current is None or self.current["embedding"].shape[0] == 0:
            QMessageBox.information(self, "Burst Clustering", "Need an embedding before manual cluster assignment.")
            self._stop_lasso_mode("Assignment mode off")
            return
        if self.embedding_ax is None:
            return
        if self.lasso is not None:
            self.lasso.disconnect_events()
        self.status.setText(message)
        self.lasso = LassoSelector(self.embedding_ax, self._finish_lasso)
        self.lasso.connect_event("button_press_event", self._lasso_button_press)

    def _lasso_button_press(self, event):
        if event.button == 3:
            self._stop_lasso_mode("Assignment mode off")

    def _stop_lasso_mode(self, message: str):
        if self.lasso is not None:
            self.lasso.disconnect_events()
            self.lasso = None
        self.lasso_mode = None
        self._refresh_lasso_button_states()
        self.status.setText(message)

    def _refresh_lasso_button_states(self):
        self.lasso_button.setChecked(self.lasso_mode == "cluster")
        self.noise_button.setChecked(self.lasso_mode == "noise")

    def _finish_lasso(self, vertices):
        if self.lasso is not None:
            self.lasso.disconnect_events()
            self.lasso = None
        if self.current is None:
            return
        points = np.asarray(self.current["embedding"], dtype=float)
        if points.ndim != 2 or points.shape[1] == 0:
            return
        if points.shape[1] == 1:
            points = np.column_stack([points[:, 0], np.zeros(points.shape[0])])
        else:
            points = points[:, :2]
        selected = np.zeros(points.shape[0], dtype=bool)
        finite_points = np.isfinite(points).all(axis=1)
        if np.any(finite_points):
            selected[finite_points] = MplPath(vertices).contains_points(points[finite_points])
        if self.manual_groups.size == selected.size:
            selected &= self._visible_group_mask(self.manual_groups)
        count = int(np.count_nonzero(selected))
        if count:
            self.undo_stack.append(self.manual_groups.copy())
            if len(self.undo_stack) > 50:
                self.undo_stack = self.undo_stack[-50:]
            self.manual_groups[selected] = int(self.pending_assignment_label)
            self.current["groups"] = self.manual_groups.astype(np.int32, copy=True)
            self.undo_button.setEnabled(True)
            self._refresh_cluster_filter()
            self._redraw_current()
        target = "noise" if self.pending_assignment_label == -1 else f"cluster {self.pending_assignment_label}"
        self.status.setText(f"Assigned {count} bursts to {target}")
        if self.lasso_mode is not None:
            mode = self.lasso_mode
            message = "Draw another region to mark noise; right-click to exit" if mode == "noise" else (
                f"Draw another region to assign cluster {self.pending_assignment_label}; right-click to exit"
            )
            self._begin_lasso(message)

    def _push_group_undo(self) -> None:
        self.undo_stack.append(self.manual_groups.copy())
        if len(self.undo_stack) > 50:
            self.undo_stack = self.undo_stack[-50:]
        self.undo_button.setEnabled(True)

    def _recluster_clean_bursts(self):
        if self.lasso_mode is not None:
            self._stop_lasso_mode("Assignment mode off")
        if self.current is None or self.manual_groups.size == 0:
            self.status.setText("No burst clustering result to recluster")
            return
        features = np.asarray(self.current.get("features", []), dtype=float)
        if features.ndim != 2 or features.shape[0] != self.manual_groups.size:
            self.status.setText("Cannot recluster: feature matrix is not aligned")
            return
        clean_mask = self.manual_groups != -1
        clean_count = int(np.count_nonzero(clean_mask))
        if clean_count < 2:
            self.status.setText("Need at least two non-noise bursts to recluster")
            return

        self._push_group_undo()
        new_groups = self.manual_groups.copy()
        new_groups[clean_mask] = _kmeans_groups(features[clean_mask], int(self.cluster_count.value()))
        self.manual_groups = new_groups.astype(np.int32, copy=False)
        self.current["groups"] = self.manual_groups.astype(np.int32, copy=True)
        self._refresh_cluster_filter()
        self._redraw_current()
        noise_count = int(np.count_nonzero(self.manual_groups == -1))
        self.status.setText(f"Reclustered {clean_count} non-noise bursts; kept {noise_count} noise bursts")

    def _undo_manual_assignment(self):
        if not self.undo_stack:
            self.status.setText("No manual clustering step to undo")
            self.undo_button.setEnabled(False)
            return
        self.manual_groups = self.undo_stack.pop()
        if self.current is not None:
            self.current["groups"] = self.manual_groups.astype(np.int32, copy=True)
        self.undo_button.setEnabled(bool(self.undo_stack))
        self._refresh_cluster_filter()
        self._redraw_current()
        self.status.setText("Undid last manual burst clustering change")


class BurstCorrelationWindow(QDialog):
    def __init__(self, spike_series, burst_intervals, parent=None, channel_map: ChannelMap | None = None):
        super().__init__(parent)
        self.setWindowTitle("Burst Correlation")
        self.resize(820, 640)
        self.spike_series = [(label, np.asarray(times, dtype=float)) for label, times in spike_series]
        self.burst_intervals = [(float(start), float(stop)) for start, stop in burst_intervals]
        self.channel_map = channel_map
        self.summary = QLabel()
        self.summary.setObjectName("MutedText")
        self.current_analysis = None
        self.current_order = np.array([], dtype=int)
        self.selected_pair = None
        self.matrix_ax = None

        self.method_combo = QComboBox()
        for label, key in [
            ("Global statistics", "global_stats"),
            ("Propagation latency", "latency"),
            ("Spatial propagation", "spatial"),
            ("Template matching", "template"),
            ("Embedding clustering", "embedding"),
            ("Dynamic time warping", "dtw"),
            ("Propagation graph", "graph"),
        ]:
            self.method_combo.addItem(label, key)
        self.method_combo.setCurrentIndex(self.method_combo.findData("template"))
        self.method_combo.currentIndexChanged.connect(self._method_changed)
        self.param_stack = QStackedWidget()
        self.param_pages = {}
        self._build_method_pages()
        self.canvas = FigureCanvas(Figure(figsize=(7, 5)))
        self.canvas.mpl_connect("button_press_event", self._matrix_clicked)
        self.sequence_canvas = FigureCanvas(Figure(figsize=(6, 5)))

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Method"))
        controls.addWidget(self.method_combo)
        controls.addStretch()

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.param_stack)
        layout.addWidget(self.summary)
        plot_area = QHBoxLayout()
        plot_area.addWidget(self.canvas, 1)
        plot_area.addWidget(self.sequence_canvas, 1)
        layout.addLayout(plot_area, 1)
        self._method_changed()
        self._draw()
        _fix_spinbox_hit_targets(self)
        self.showMaximized()

    def _build_method_pages(self):
        self.global_bin_ms = self._double_spin(1.0, 100.0, 10.0, " ms")
        self.global_block_threshold = self._threshold_spin()
        self.global_normalize = self._normalize_combo("unit_zscore")
        self._add_param_page(
            "global_stats",
            [("Stats bin", self.global_bin_ms), ("Block threshold", self.global_block_threshold), ("Normalize", self.global_normalize)],
        )

        self.latency_window_ms = self._spin(0, 5000, 0, " ms")
        self.latency_block_threshold = self._threshold_spin()
        self.latency_normalize = self._normalize_combo("unit_zscore")
        self._add_param_page(
            "latency",
            [("Window", self.latency_window_ms), ("Block threshold", self.latency_block_threshold), ("Normalize", self.latency_normalize)],
        )

        self.spatial_bin_ms = self._double_spin(1.0, 100.0, 5.0, " ms")
        self.spatial_window_ms = self._spin(0, 5000, 0, " ms")
        self.spatial_block_threshold = self._threshold_spin()
        self.spatial_normalize = self._normalize_combo("per_burst")
        self._add_param_page(
            "spatial",
            [
                ("Time bin", self.spatial_bin_ms),
                ("Window", self.spatial_window_ms),
                ("Block threshold", self.spatial_block_threshold),
                ("Normalize", self.spatial_normalize),
            ],
        )

        self.template_bin_ms = self._double_spin(1.0, 100.0, 5.0, " ms")
        self.template_window_ms = self._spin(0, 5000, 0, " ms")
        self.template_count = self._spin(1, 20, 3, "")
        self.template_normalize = self._normalize_combo("per_burst")
        self._add_param_page(
            "template",
            [
                ("Time bin", self.template_bin_ms),
                ("Window", self.template_window_ms),
                ("Templates", self.template_count),
                ("Normalize", self.template_normalize),
            ],
        )

        self.embedding_bin_ms = self._double_spin(1.0, 100.0, 5.0, " ms")
        self.embedding_window_ms = self._spin(0, 5000, 0, " ms")
        self.embedding_count = self._spin(1, 20, 3, "")
        self.embedding_reducer = QComboBox()
        self.embedding_reducer.addItem("PCA", "pca")
        self.embedding_reducer.addItem("t-SNE", "tsne")
        self.embedding_reducer.currentIndexChanged.connect(self._draw)
        self.embedding_normalize = self._normalize_combo("per_burst")
        self._add_param_page(
            "embedding",
            [
                ("Time bin", self.embedding_bin_ms),
                ("Window", self.embedding_window_ms),
                ("Clusters", self.embedding_count),
                ("Reducer", self.embedding_reducer),
                ("Normalize", self.embedding_normalize),
            ],
        )

        self.dtw_bin_ms = self._double_spin(1.0, 100.0, 5.0, " ms")
        self.dtw_window_ms = self._spin(0, 5000, 0, " ms")
        self.dtw_warp_bins = self._spin(0, 20, 2, " bins")
        self.dtw_block_threshold = self._threshold_spin()
        self.dtw_normalize = self._normalize_combo("per_burst")
        self._add_param_page(
            "dtw",
            [
                ("Time bin", self.dtw_bin_ms),
                ("Window", self.dtw_window_ms),
                ("Warp", self.dtw_warp_bins),
                ("Block threshold", self.dtw_block_threshold),
                ("Normalize", self.dtw_normalize),
            ],
        )

        self.graph_window_ms = self._double_spin(0.1, 200.0, 10.0, " ms")
        self.graph_block_threshold = self._threshold_spin()
        self.graph_normalize = self._normalize_combo("unit_zscore")
        self._add_param_page(
            "graph",
            [("Edge window", self.graph_window_ms), ("Block threshold", self.graph_block_threshold), ("Normalize", self.graph_normalize)],
        )

        self.time_bin_ms = self.template_bin_ms
        self.window_ms = self.template_window_ms
        self.normalize = self.template_normalize

    def _spin(self, minimum: int, maximum: int, value: int, suffix: str):
        field = QSpinBox()
        field.setRange(minimum, maximum)
        field.setSingleStep(1 if maximum <= 100 else 10)
        field.setValue(value)
        if suffix:
            field.setSuffix(suffix)
        field.valueChanged.connect(self._draw)
        return field

    def _double_spin(self, minimum: float, maximum: float, value: float, suffix: str):
        field = QDoubleSpinBox()
        field.setRange(minimum, maximum)
        field.setDecimals(1)
        field.setSingleStep(1.0)
        field.setValue(value)
        if suffix:
            field.setSuffix(suffix)
        field.valueChanged.connect(self._draw)
        return field

    def _threshold_spin(self):
        field = QDoubleSpinBox()
        field.setRange(0.01, 2.0)
        field.setDecimals(2)
        field.setSingleStep(0.05)
        field.setValue(0.45)
        field.valueChanged.connect(self._draw)
        return field

    def _normalize_combo(self, current: str):
        combo = QComboBox()
        combo.addItem("Per burst", "per_burst")
        combo.addItem("Unit z-score", "unit_zscore")
        combo.addItem("None", "none")
        combo.setCurrentIndex(max(0, combo.findData(current)))
        combo.currentIndexChanged.connect(self._draw)
        return combo

    def _add_param_page(self, key: str, rows):
        page = QWidget()
        layout = QFormLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        for label, widget in rows:
            layout.addRow(label, widget)
        self.param_pages[key] = page
        self.param_stack.addWidget(page)

    def _method_changed(self):
        key = self.method_combo.currentData()
        page = self.param_pages.get(key)
        if page is not None:
            self.param_stack.setCurrentWidget(page)
        self._draw()

    def _draw(self):
        figure = self.canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        self.matrix_ax = ax
        method = self.method_combo.currentData()
        params = self._current_params(method)
        analysis = _burst_correlation_analysis(
            self.spike_series,
            self.burst_intervals,
            channel_map=self.channel_map,
            method=method,
            **params,
        )
        self.current_analysis = analysis
        correlation = analysis["correlation"]
        order = analysis["order"]
        self.current_order = order
        intervals = analysis["intervals"]

        if correlation.shape[0] < 2:
            ax.text(0.5, 0.5, "Need at least two detected bursts", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            self.summary.setText("Detected bursts: 0" if not intervals else "Detected bursts: 1")
            self.canvas.draw_idle()
            self._draw_selected_burst_sequences()
            return

        ordered = correlation[np.ix_(order, order)]
        image = ax.imshow(ordered, vmin=-1.0, vmax=1.0, cmap="coolwarm", interpolation="nearest", aspect="auto")
        figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Correlation")

        ordered_groups = analysis["groups"][order]
        boundaries = np.flatnonzero(ordered_groups[1:] != ordered_groups[:-1]) + 0.5
        for boundary in boundaries:
            ax.axhline(boundary, color="#0f172a", linewidth=1.0, alpha=0.8)
            ax.axvline(boundary, color="#0f172a", linewidth=1.0, alpha=0.8)

        if self.selected_pair is None or any(index >= correlation.shape[0] for index in self.selected_pair):
            self.selected_pair = (int(order[0]), int(order[1] if len(order) > 1 else order[0]))
        ordered_positions = {int(burst): position for position, burst in enumerate(order)}
        if self.selected_pair[0] in ordered_positions and self.selected_pair[1] in ordered_positions:
            y = ordered_positions[self.selected_pair[0]]
            x = ordered_positions[self.selected_pair[1]]
            ax.scatter([x], [y], s=92, facecolors="none", edgecolors="#111827", linewidths=1.8)
            ax.scatter([y], [x], s=92, facecolors="none", edgecolors="#111827", linewidths=1.8)

        burst_labels = [f"B{index + 1}\n{intervals[index][0]:.2f}s" for index in order]
        tick_indices = _display_indices(len(order), 18)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([burst_labels[index] for index in tick_indices], rotation=90, fontsize=7)
        ax.set_yticks(tick_indices)
        ax.set_yticklabels([burst_labels[index] for index in tick_indices], fontsize=7)
        ax.set_title("Burst pattern correlation")
        ax.set_xlabel("Burst")
        ax.set_ylabel("Burst")

        group_count = len(set(int(value) for value in ordered_groups))
        mean_corr = float(np.mean(ordered[np.triu_indices_from(ordered, k=1)]))
        active_rows = int(np.count_nonzero(np.any(analysis["activity"] > 0, axis=(0, 2))))
        window_value = params.get("window_ms", 0.0)
        window_text = "auto" if not window_value else f"{window_value:g} ms"
        block_text = (
            f"threshold {params['block_threshold']:.2f}"
            if "block_threshold" in params
            else f"clusters {params.get('cluster_count', 'n/a')}"
        )
        self.summary.setText(
            f"Method: {self.method_combo.currentText()} | Bursts: {len(order)} | Active rows: {active_rows} | "
            f"Blocks: {group_count} ({block_text}) | Mean corr: {mean_corr:.3f} | Window: {window_text}"
        )
        self.canvas.draw_idle()
        self._draw_selected_burst_sequences()

    def _matrix_clicked(self, event):
        if self.current_analysis is None or event.inaxes is not self.matrix_ax:
            return
        if event.xdata is None or event.ydata is None or self.current_order.size == 0:
            return
        col = int(round(float(event.xdata)))
        row = int(round(float(event.ydata)))
        if row < 0 or col < 0 or row >= self.current_order.size or col >= self.current_order.size:
            return
        self.selected_pair = (int(self.current_order[row]), int(self.current_order[col]))
        self._draw()

    def _draw_selected_burst_sequences(self):
        figure = self.sequence_canvas.figure
        figure.clear()
        if self.current_analysis is None or self.selected_pair is None:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, "Click a matrix cell to compare two bursts", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            self.sequence_canvas.draw_idle()
            return

        intervals = self.current_analysis.get("intervals", [])
        labels = self.current_analysis.get("labels", [])
        if len(intervals) < 2 or not labels:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, "No burst sequence data", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            self.sequence_canvas.draw_idle()
            return

        first, second = self.selected_pair
        if first >= len(intervals) or second >= len(intervals):
            self.sequence_canvas.draw_idle()
            return

        row_indices = np.arange(len(labels), dtype=int)
        max_duration_ms = max((intervals[index][1] - intervals[index][0]) * 1000.0 for index in [first, second])
        max_duration_ms = max(1.0, float(max_duration_ms))
        axes = figure.subplots(2, 1, sharex=True)
        for ax, burst_index in zip(axes, [first, second]):
            start_s, stop_s = intervals[burst_index]
            for display_row, source_row in enumerate(row_indices):
                spike_times = np.asarray(self.spike_series[source_row][1], dtype=float)
                lo = int(np.searchsorted(spike_times, start_s, side="left"))
                hi = int(np.searchsorted(spike_times, stop_s, side="right"))
                if hi <= lo:
                    continue
                relative_ms = (spike_times[lo:hi] - start_s) * 1000.0
                ax.vlines(relative_ms, display_row - 0.36, display_row + 0.36, color="#2563eb", linewidth=0.9, alpha=0.95)
            ax.set_title(f"B{burst_index + 1}: {start_s:.3f}-{stop_s:.3f} s")
            ax.set_ylabel("Row")
            ax.set_ylim(row_indices.size - 0.5, -0.5)
            ax.set_xlim(0.0, max_duration_ms)
            tick_indices = _display_indices(row_indices.size, 10) if row_indices.size else np.array([], dtype=int)
            ax.set_yticks(tick_indices)
            ax.set_yticklabels([str(labels[row_indices[index]]) for index in tick_indices], fontsize=7)
        axes[-1].set_xlabel("Time from burst start (ms)")
        self.sequence_canvas.draw_idle()

    def _current_params(self, method: str) -> dict:
        if method == "global_stats":
            return {
                "time_bin_ms": float(self.global_bin_ms.value()),
                "block_threshold": float(self.global_block_threshold.value()),
                "normalization": self.global_normalize.currentData(),
            }
        if method == "latency":
            return {
                "latency_window_ms": float(self.latency_window_ms.value()),
                "block_threshold": float(self.latency_block_threshold.value()),
                "normalization": self.latency_normalize.currentData(),
            }
        if method == "spatial":
            return {
                "time_bin_ms": float(self.spatial_bin_ms.value()),
                "window_ms": float(self.spatial_window_ms.value()),
                "block_threshold": float(self.spatial_block_threshold.value()),
                "normalization": self.spatial_normalize.currentData(),
            }
        if method == "embedding":
            return {
                "time_bin_ms": float(self.embedding_bin_ms.value()),
                "window_ms": float(self.embedding_window_ms.value()),
                "cluster_count": int(self.embedding_count.value()),
                "embedding_method": self.embedding_reducer.currentData(),
                "normalization": self.embedding_normalize.currentData(),
            }
        if method == "dtw":
            return {
                "time_bin_ms": float(self.dtw_bin_ms.value()),
                "window_ms": float(self.dtw_window_ms.value()),
                "dtw_warp_bins": int(self.dtw_warp_bins.value()),
                "block_threshold": float(self.dtw_block_threshold.value()),
                "normalization": self.dtw_normalize.currentData(),
            }
        if method == "graph":
            return {
                "graph_window_ms": float(self.graph_window_ms.value()),
                "block_threshold": float(self.graph_block_threshold.value()),
                "normalization": self.graph_normalize.currentData(),
            }
        return {
            "time_bin_ms": float(self.template_bin_ms.value()),
            "window_ms": float(self.template_window_ms.value()),
            "cluster_count": int(self.template_count.value()),
            "normalization": self.template_normalize.currentData(),
        }


class HeatmapGifExportDialog(QDialog):
    def __init__(self, min_time: float, max_time: float, frame_step_ms: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Heatmap GIF")
        self.min_time = float(min_time)
        self.max_time = float(max_time)

        lower = min(self.min_time, self.max_time)
        upper = max(self.min_time, self.max_time)
        if upper <= lower:
            upper = lower + 1.0

        self.start_s = QDoubleSpinBox()
        self.start_s.setRange(lower, upper)
        self.start_s.setDecimals(3)
        self.start_s.setSingleStep(0.1)
        self.start_s.setValue(lower)
        self.start_s.setSuffix(" s")

        self.end_s = QDoubleSpinBox()
        self.end_s.setRange(lower, upper)
        self.end_s.setDecimals(3)
        self.end_s.setSingleStep(0.1)
        self.end_s.setValue(upper)
        self.end_s.setSuffix(" s")

        self.frame_step_ms = QSpinBox()
        self.frame_step_ms.setRange(10, 60000)
        self.frame_step_ms.setSingleStep(10)
        self.frame_step_ms.setValue(max(10, int(frame_step_ms)))
        self.frame_step_ms.setSuffix(" ms")

        export_button = QPushButton("Export")
        export_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        form = QFormLayout()
        form.addRow("Start time", self.start_s)
        form.addRow("End time", self.end_s)
        form.addRow("Frame step", self.frame_step_ms)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(cancel_button)
        buttons.addWidget(export_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)
        _fix_spinbox_hit_targets(self)

    def values(self) -> tuple[float, float, int]:
        start = float(self.start_s.value())
        end = float(self.end_s.value())
        if end < start:
            start, end = end, start
        return start, end, int(self.frame_step_ms.value())


class SpikeRasterWindow(QDialog):
    def __init__(
        self,
        title: str,
        spike_series,
        waveform_series=None,
        sampling_rate=None,
        parent=None,
        y_axis_label: str = "Channel",
        channel_map: ChannelMap | None = None,
        stim_times=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1280, 820)
        self.raw_spike_series = [(label, np.asarray(times, dtype=float)) for label, times in spike_series]
        self.spike_series = [(label, times.copy()) for label, times in self.raw_spike_series]
        self.spike_lookup = {label: times for label, times in self.spike_series}
        self._count_series = [(_base_channel_from_raster_label(label), times) for label, times in self.spike_series]
        self.raw_waveform_series = {label: np.asarray(values) for label, values in (waveform_series or {}).items()}
        self.waveform_series = dict(self.raw_waveform_series)
        self.sampling_rate = sampling_rate
        self.channel_map = channel_map
        self.analysis_windows = []
        self.selected_channel = _prefer_waveform_channel(self.spike_series, self.waveform_series)
        self._last_waveform_view = None
        self._last_waveform_refresh = 0.0
        self._waveform_min_interval_s = 0.12
        self._waveform_window_limit = 2400
        self._playback_time_ms = None
        self._internal_slider_update = False
        self.stim_times = np.asarray(stim_times if stim_times is not None else [], dtype=float)
        self.stim_times = self.stim_times[np.isfinite(self.stim_times)]
        self.stim_times.sort()
        all_times = [times for _, times in self.raw_spike_series if times.size]
        if self.stim_times.size:
            all_times.append(self.stim_times)
        if all_times:
            self.min_time = min(float(times.min()) for times in all_times)
            self.max_time = max(float(times.max()) for times in all_times)
        else:
            self.min_time = 0.0
            self.max_time = 1.0

        total_duration_ms = max(1, int(np.ceil((self.max_time - self.min_time) * 1000)))
        default_grid_ms = 100
        default_window_grids = max(10, min(100, total_duration_ms // default_grid_ms or 10))
        default_visible_rows = max(1, min(len(self.spike_series) or 1, 30))

        self.burst_intervals = []
        self.canvas = SpikeRasterCanvas(self.spike_series, y_axis_label=y_axis_label)
        self.canvas.set_bursts(self.burst_intervals)
        self.canvas.set_stim_times(self.stim_times)
        self.canvas.wheel_zoom_requested.connect(self._zoom_grid_at)
        self.canvas.pan_requested.connect(self._pan_to_absolute_ms)
        self.canvas.channel_selected.connect(self._select_channel)
        self.rate_canvas = PopulationRateCanvas(self.spike_series, left_margin=self.canvas.plot_left)
        self.rate_canvas.set_bursts(self.burst_intervals)
        self.rate_canvas.set_stim_times(self.stim_times)
        self.waveform_canvas = SpikeWaveformCanvas()
        self.heatmap_canvas = ElectrodeHeatmapCanvas(channel_map)
        self.heatmap_scale_count = 0
        self._heatmap_scale_cache: dict[float, int] = {}
        self._last_heatmap_refresh = 0.0
        self._heatmap_min_interval_s = 0.15
        self.row_scroll = QScrollBar(Qt.Orientation.Vertical)
        self.row_scroll.setSingleStep(1)
        self.row_scroll.valueChanged.connect(self._sync_visible_rows)

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(50)
        self.play_timer.timeout.connect(self._playback_step)
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self._toggle_playback)
        self.ibi_button = QPushButton("IBI")
        self.ibi_button.clicked.connect(self._open_ibi_window)
        self.isi_button = QPushButton("ISI")
        self.isi_button.clicked.connect(self._open_isi_window)
        self.burst_corr_button = QPushButton("Burst Corr")
        self.burst_corr_button.clicked.connect(self._open_burst_correlation_window)
        self.burst_delay_button = QPushButton("Burst Delay")
        self.burst_delay_button.clicked.connect(self._open_burst_delay_window)
        self.burst_cluster_button = QPushButton("Burst Cluster")
        self.burst_cluster_button.clicked.connect(self._open_burst_clustering_window)
        self.save_bursts_button = QPushButton("Save Bursts")
        self.save_bursts_button.clicked.connect(self._save_bursts)
        self.export_heatmap_gif_button = QPushButton("Export GIF")
        self.export_heatmap_gif_button.clicked.connect(self._export_heatmap_gif)
        self.raster_action_combo = QComboBox()
        self.raster_action_combo.addItem("Choose action...", "")
        self.raster_action_combo.addItem("IBI", "ibi")
        self.raster_action_combo.addItem("ISI", "isi")
        self.raster_action_combo.addItem("Burst correlation", "burst_corr")
        self.raster_action_combo.addItem("Burst delay", "burst_delay")
        self.raster_action_combo.addItem("Burst clustering", "burst_cluster")
        self.raster_action_combo.addItem("Save bursts", "save_bursts")
        self.raster_action_combo.addItem("Export heatmap GIF", "export_heatmap_gif")
        self.raster_action_combo.setMinimumWidth(170)
        self.raster_action_combo.activated.connect(self._raster_action_selected)

        self.window_grids = QSpinBox()
        self.window_grids.setRange(1, 500)
        self.window_grids.setSingleStep(1)
        self.window_grids.setValue(default_window_grids)
        self.window_grids.setFixedWidth(58)
        self.window_grids.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.window_grids.setCursor(Qt.CursorShape.ArrowCursor)
        self.window_grids.valueChanged.connect(self._update_slider_range)

        self.grid_ms = QSpinBox()
        self.grid_ms.setRange(1, 60000)
        self.grid_ms.setSingleStep(10)
        self.grid_ms.setValue(default_grid_ms)
        self.grid_ms.setFixedWidth(66)
        self.grid_ms.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.grid_ms.setCursor(Qt.CursorShape.ArrowCursor)
        self.grid_ms.valueChanged.connect(self._update_slider_range)

        self.heatmap_ms = QSpinBox()
        self.heatmap_ms.setRange(10, 5000)
        self.heatmap_ms.setSingleStep(10)
        self.heatmap_ms.setValue(100)
        self.heatmap_ms.setFixedWidth(70)
        self.heatmap_ms.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.heatmap_ms.setCursor(Qt.CursorShape.ArrowCursor)
        self.heatmap_ms.valueChanged.connect(self._heatmap_bin_changed)

        self.burst_bin_ms = QSpinBox()
        self.burst_bin_ms.setRange(1, 500)
        self.burst_bin_ms.setSingleStep(1)
        self.burst_bin_ms.setValue(10)
        self.burst_bin_ms.setFixedWidth(58)
        self.burst_bin_ms.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.burst_bin_ms.setCursor(Qt.CursorShape.ArrowCursor)
        self.burst_bin_ms.valueChanged.connect(self._refresh_bursts)

        self.burst_threshold_z = QDoubleSpinBox()
        self.burst_threshold_z.setRange(0.5, 20.0)
        self.burst_threshold_z.setSingleStep(0.5)
        self.burst_threshold_z.setDecimals(1)
        self.burst_threshold_z.setValue(4.0)
        self.burst_threshold_z.setFixedWidth(58)
        self.burst_threshold_z.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.burst_threshold_z.setCursor(Qt.CursorShape.ArrowCursor)
        self.burst_threshold_z.valueChanged.connect(self._refresh_bursts)

        self.burst_min_spikes = QSpinBox()
        self.burst_min_spikes.setRange(2, 1000)
        self.burst_min_spikes.setSingleStep(1)
        self.burst_min_spikes.setValue(5)
        self.burst_min_spikes.setFixedWidth(58)
        self.burst_min_spikes.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.burst_min_spikes.setCursor(Qt.CursorShape.ArrowCursor)
        self.burst_min_spikes.valueChanged.connect(self._refresh_bursts)

        self.hide_stim_tail = QCheckBox("Hide stim tail")
        self.hide_stim_tail.setEnabled(self.stim_times.size > 0)
        self.hide_stim_tail.stateChanged.connect(self._apply_stim_tail_filter)
        self.stim_tail_ms = QDoubleSpinBox()
        self.stim_tail_ms.setRange(0.1, 1000.0)
        self.stim_tail_ms.setDecimals(1)
        self.stim_tail_ms.setSingleStep(0.5)
        self.stim_tail_ms.setValue(1.0)
        self.stim_tail_ms.setFixedWidth(58)
        self.stim_tail_ms.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.stim_tail_ms.setCursor(Qt.CursorShape.ArrowCursor)
        self.stim_tail_ms.setEnabled(self.stim_times.size > 0)
        self.stim_tail_ms.valueChanged.connect(self._apply_stim_tail_filter)

        self.visible_rows = QSpinBox()
        self.visible_rows.setRange(1, max(1, len(self.spike_series)))
        self.visible_rows.setSingleStep(1)
        self.visible_rows.setValue(default_visible_rows)
        self.visible_rows.setFixedWidth(58)
        self.visible_rows.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.visible_rows.setCursor(Qt.CursorShape.ArrowCursor)
        self.visible_rows.valueChanged.connect(self._update_row_scroll_range)

        self.window_minus_button = self._step_button("-")
        self.window_minus_button.clicked.connect(lambda: self.window_grids.stepBy(-1))
        self.window_plus_button = self._step_button("+")
        self.window_plus_button.clicked.connect(lambda: self.window_grids.stepBy(1))
        self.grid_minus_button = self._step_button("-")
        self.grid_minus_button.clicked.connect(lambda: self.grid_ms.stepBy(-1))
        self.grid_plus_button = self._step_button("+")
        self.grid_plus_button.clicked.connect(lambda: self.grid_ms.stepBy(1))
        self.visible_rows_minus_button = self._step_button("-")
        self.visible_rows_minus_button.clicked.connect(lambda: self.visible_rows.stepBy(-1))
        self.visible_rows_plus_button = self._step_button("+")
        self.visible_rows_plus_button.clicked.connect(lambda: self.visible_rows.stepBy(1))

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setSingleStep(10)
        self.slider.setPageStep(max(10, self._window_ms() // 2))
        self.slider.valueChanged.connect(self._slider_value_changed)

        self.time_label = QLabel()
        self.time_label.setObjectName("MutedText")
        self.time_label.setFont(QFont("Consolas", 9))
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.time_label.setMinimumWidth(154)
        self.time_label.setFixedWidth(154)
        self.row_label = QLabel()
        self.row_label.setObjectName("MutedText")
        self.row_label.setMinimumWidth(112)
        self.row_label.setFixedWidth(112)

        playback_controls = QHBoxLayout()
        playback_controls.setSpacing(8)
        playback_controls.addWidget(self.play_button)
        playback_controls.addWidget(QLabel("Time"))
        self.slider.setMinimumWidth(420)
        playback_controls.addWidget(self.slider, 1)
        playback_controls.addWidget(self.time_label)

        parameter_controls = QHBoxLayout()
        parameter_controls.setSpacing(8)
        parameter_controls.addWidget(QLabel("Window"))
        parameter_controls.addWidget(self._number_stepper(self.window_minus_button, self.window_grids, self.window_plus_button))
        parameter_controls.addWidget(QLabel("grids"))
        parameter_controls.addWidget(QLabel("Grid"))
        parameter_controls.addWidget(self._number_stepper(self.grid_minus_button, self.grid_ms, self.grid_plus_button))
        parameter_controls.addWidget(QLabel("ms/grid"))
        parameter_controls.addWidget(QLabel("Heatmap"))
        parameter_controls.addWidget(self.heatmap_ms)
        parameter_controls.addWidget(QLabel("ms"))
        parameter_controls.addWidget(self.hide_stim_tail)
        parameter_controls.addWidget(self.stim_tail_ms)
        parameter_controls.addWidget(QLabel("ms"))
        parameter_controls.addWidget(QLabel("Burst bin"))
        parameter_controls.addWidget(self.burst_bin_ms)
        parameter_controls.addWidget(QLabel("ms"))
        parameter_controls.addWidget(QLabel("Burst z"))
        parameter_controls.addWidget(self.burst_threshold_z)
        parameter_controls.addWidget(QLabel("Min spikes"))
        parameter_controls.addWidget(self.burst_min_spikes)
        parameter_controls.addWidget(QLabel("Rows"))
        parameter_controls.addWidget(
            self._number_stepper(
                self.visible_rows_minus_button,
                self.visible_rows,
                self.visible_rows_plus_button,
            )
        )
        parameter_controls.addWidget(QLabel("visible"))
        parameter_controls.addWidget(self.row_label)
        parameter_controls.addStretch(1)
        parameter_controls.addWidget(QLabel("Action"))
        parameter_controls.addWidget(self.raster_action_combo)

        layout = QVBoxLayout(self)
        raster_area = QWidget()
        raster_layout = QHBoxLayout(raster_area)
        raster_layout.setContentsMargins(0, 0, 0, 0)
        raster_layout.setSpacing(6)
        raster_layout.addWidget(self.canvas, 1)
        raster_layout.addWidget(self.row_scroll)
        layout.addWidget(raster_area, 2)
        layout.addWidget(self.rate_canvas, 1)
        lower_area = QWidget()
        lower_layout = QHBoxLayout(lower_area)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        lower_layout.setSpacing(8)
        lower_layout.addWidget(self.waveform_canvas, 2)
        lower_layout.addWidget(self.heatmap_canvas, 1)
        layout.addWidget(lower_area, 1)
        layout.addLayout(playback_controls)
        layout.addLayout(parameter_controls)

        self._refresh_bursts()
        self._refresh_heatmap_scale()
        self._update_row_scroll_range()
        self._update_slider_range()
        if self.selected_channel:
            self._select_channel(self.selected_channel)
        _fix_spinbox_hit_targets(self)
        self.showMaximized()

    def _start_progress(self, title: str, message: str, maximum: int = 0) -> QProgressDialog:
        return _create_progress_dialog(self, title, message, maximum)

    def _finish_progress(self, dialog: QProgressDialog | None) -> None:
        _close_progress_dialog(dialog)

    def _raster_action_selected(self, index: int) -> None:
        action = str(self.raster_action_combo.itemData(int(index)) or "")
        if not action:
            return
        self.raster_action_combo.blockSignals(True)
        self.raster_action_combo.setCurrentIndex(0)
        self.raster_action_combo.blockSignals(False)
        handlers = {
            "ibi": self._open_ibi_window,
            "isi": self._open_isi_window,
            "burst_corr": self._open_burst_correlation_window,
            "burst_delay": self._open_burst_delay_window,
            "burst_cluster": self._open_burst_clustering_window,
            "save_bursts": self._save_bursts,
            "export_heatmap_gif": self._export_heatmap_gif,
        }
        handler = handlers.get(action)
        if handler is not None:
            handler()

    def _total_duration_ms(self) -> int:
        return max(1, int(np.ceil((self.max_time - self.min_time) * 1000)))

    def _short_data_window(self) -> bool:
        return self._total_duration_ms() <= self._window_ms()

    def _slider_maximum_ms(self) -> int:
        total_ms = self._total_duration_ms()
        window_ms = self._window_ms()
        if total_ms <= window_ms:
            return total_ms
        return max(0, total_ms - window_ms)

    def _window_start_offset_ms(self) -> int:
        if self._short_data_window():
            return 0
        return min(self.slider.value(), self._slider_maximum_ms())

    def _window_stop_offset_ms(self) -> int:
        return self._window_start_offset_ms() + self._window_ms()

    def _playhead_time_s(self) -> float | None:
        if self._playback_time_ms is None:
            return None
        return self.min_time + min(max(0, int(self._playback_time_ms)), self._total_duration_ms()) / 1000.0

    def _set_playback_time_ms(self, value_ms: int | float | None) -> None:
        if value_ms is None:
            self._playback_time_ms = None
            return
        self._playback_time_ms = int(min(max(0, round(float(value_ms))), self._total_duration_ms()))

    def _update_slider_range(self):
        window_ms = self._window_ms()
        self.slider.setMaximum(self._slider_maximum_ms())
        self.slider.setPageStep(max(1, window_ms // 2))
        if self._playback_time_ms is not None:
            self._set_playback_time_ms(self._playback_time_ms)
        if self.slider.maximum() == 0 and not self._short_data_window():
            self._stop_playback()
        self._refresh_heatmap_scale()
        self._update_view()

    def _update_row_scroll_range(self):
        total_rows = len(self.spike_series)
        visible_rows = max(1, min(self.visible_rows.value(), max(1, total_rows)))
        if self.visible_rows.value() != visible_rows:
            self.visible_rows.blockSignals(True)
            self.visible_rows.setValue(visible_rows)
            self.visible_rows.blockSignals(False)
        max_offset = max(0, total_rows - visible_rows)
        current = min(self.row_scroll.value(), max_offset)
        self.row_scroll.blockSignals(True)
        self.row_scroll.setRange(0, max_offset)
        self.row_scroll.setPageStep(visible_rows)
        self.row_scroll.setValue(current)
        self.row_scroll.blockSignals(False)
        self._sync_visible_rows()

    def _sync_visible_rows(self):
        total_rows = len(self.spike_series)
        visible_rows = max(1, min(self.visible_rows.value(), max(1, total_rows)))
        offset = min(self.row_scroll.value(), max(0, total_rows - visible_rows))
        self.canvas.set_visible_rows(offset, visible_rows)
        if total_rows:
            self.row_label.setText(f"Rows {offset + 1}-{min(total_rows, offset + visible_rows)} / {total_rows}")
        else:
            self.row_label.setText("Rows 0 / 0")

    def _apply_stim_tail_filter(self):
        if not hasattr(self, "hide_stim_tail"):
            return
        previous_selected = self.selected_channel
        if self.hide_stim_tail.isChecked() and self.stim_times.size:
            filtered, masks, removed = _filter_spike_series_stim_tail(
                self.raw_spike_series,
                self.stim_times,
                float(self.stim_tail_ms.value()),
            )
            self.spike_series = filtered
            self.waveform_series = {}
            for label, waveforms in self.raw_waveform_series.items():
                mask = masks.get(str(label))
                values = np.asarray(waveforms)
                if mask is not None and values.ndim >= 2 and values.shape[0] == mask.size:
                    self.waveform_series[label] = values[mask]
                else:
                    self.waveform_series[label] = values
            suffix = f"stim tail hidden: {removed} spikes"
        else:
            self.spike_series = [(label, times.copy()) for label, times in self.raw_spike_series]
            self.waveform_series = dict(self.raw_waveform_series)
            suffix = "stim tail shown"

        self.spike_lookup = {label: times for label, times in self.spike_series}
        self._count_series = [(_base_channel_from_raster_label(label), times) for label, times in self.spike_series]
        labels = [label for label, _ in self.spike_series]
        if previous_selected in labels:
            self.selected_channel = previous_selected
        else:
            self.selected_channel = _prefer_waveform_channel(self.spike_series, self.waveform_series)
        self.canvas.set_spike_series(self.spike_series)
        self.canvas.set_selected_channel(self.selected_channel)
        self.rate_canvas.set_spike_series(self.spike_series)
        self._heatmap_scale_cache.clear()
        self._last_waveform_view = None
        self._refresh_bursts()
        self._refresh_heatmap_scale()
        self._update_row_scroll_range()
        self._update_view(force_heatmap=True)
        self._refresh_waveforms_for_window(force=True)
        self.summary_text = suffix if hasattr(self, "summary_text") else suffix

    def _slider_value_changed(self):
        if not self._internal_slider_update:
            if self._short_data_window():
                self._set_playback_time_ms(self.slider.value())
            else:
                self._set_playback_time_ms(self._window_start_offset_ms())
            self._update_view(force_heatmap=True)
            return
        self._update_view()

    def _update_view(self, *, force_heatmap: bool = False):
        start_s = self.min_time + self._window_start_offset_ms() / 1000.0
        duration_s = self._window_ms() / 1000.0
        grid_s = self.grid_ms.value() / 1000.0
        self.canvas.set_view(start_s, duration_s, grid_s)
        self.rate_canvas.set_view(start_s, duration_s)
        playhead_time = self._playhead_time_s()
        self.canvas.set_playhead_time(playhead_time)
        self.rate_canvas.set_playhead_time(playhead_time)
        self.time_label.setText(f"{start_s:8.3f} - {start_s + duration_s:8.3f} s")
        self._refresh_waveforms_for_window()
        self._refresh_heatmap_for_view(start_s, duration_s, force=force_heatmap)

    def _refresh_heatmap_for_view(
        self,
        start_s: float | None = None,
        duration_s: float | None = None,
        *,
        force: bool = False,
    ):
        if start_s is None:
            start_s = self.min_time + self._window_start_offset_ms() / 1000.0
        if duration_s is None:
            duration_s = self._window_ms() / 1000.0
        playing = self.play_timer.isActive()
        self.heatmap_canvas.set_fast_mode(playing)
        now = time.monotonic()
        if playing and not force and (now - self._last_heatmap_refresh) < self._heatmap_min_interval_s:
            return
        self._last_heatmap_refresh = now
        heatmap_duration_s = max(0.001, self.heatmap_ms.value() / 1000.0)
        playhead_time = self._playhead_time_s()
        heatmap_stop_s = float(playhead_time) if playhead_time is not None else start_s + duration_s
        heatmap_start_s = max(start_s, heatmap_stop_s - heatmap_duration_s)
        self.heatmap_canvas.set_counts(self._window_channel_counts(heatmap_start_s, heatmap_stop_s))

    def _heatmap_bin_changed(self):
        self._refresh_heatmap_scale()
        self._refresh_heatmap_for_view(force=True)

    def _heatmap_gif_frame_times(self, start_s: float, end_s: float, frame_step_ms: int) -> np.ndarray:
        start_s = max(float(self.min_time), float(start_s))
        end_s = min(float(self.max_time), float(end_s))
        step_s = max(0.01, float(frame_step_ms) / 1000.0)
        if end_s < start_s:
            start_s, end_s = end_s, start_s
        if end_s <= start_s:
            return np.asarray([start_s], dtype=float)
        values = np.arange(start_s, end_s + step_s * 0.5, step_s, dtype=float)
        values = values[values <= end_s + 1e-9]
        if values.size == 0 or values[-1] < end_s:
            values = np.append(values, end_s)
        return values.astype(float, copy=False)

    def _refresh_heatmap_scale(self):
        heatmap_duration_s = max(0.001, self.heatmap_ms.value() / 1000.0)
        cache_key = round(float(heatmap_duration_s), 6)
        if cache_key in self._heatmap_scale_cache:
            self.heatmap_scale_count = self._heatmap_scale_cache[cache_key]
            self.heatmap_canvas.set_scale_max_count(self.heatmap_scale_count)
            return
        start = float(self.min_time)
        stop = float(self.max_time)
        scale = 0
        if stop > start:
            edges = np.arange(start, stop + heatmap_duration_s * 1.5, heatmap_duration_s, dtype=float)
            if edges.size < 2:
                edges = np.array([start, start + heatmap_duration_s], dtype=float)
            per_channel_bins: dict[str, np.ndarray] = {}
            for base_channel, values in self._count_series:
                if values.size == 0:
                    continue
                lo = int(np.searchsorted(values, edges[0], side="left"))
                hi = int(np.searchsorted(values, edges[-1], side="right"))
                if hi <= lo:
                    continue
                indices = np.searchsorted(values[lo:hi], edges)
                bin_counts = np.diff(indices).astype(np.int32, copy=False)
                existing = per_channel_bins.get(base_channel)
                if existing is None:
                    per_channel_bins[base_channel] = bin_counts
                else:
                    existing += bin_counts
            for bin_counts in per_channel_bins.values():
                if bin_counts.size:
                    scale = max(scale, int(bin_counts.max()))
        self.heatmap_scale_count = int(scale)
        self._heatmap_scale_cache[cache_key] = self.heatmap_scale_count
        self.heatmap_canvas.set_scale_max_count(self.heatmap_scale_count)

    def _export_heatmap_gif(self):
        if self.channel_map is None:
            QMessageBox.information(self, "Export Heatmap GIF", "Set a channel map before exporting a heatmap GIF.")
            return False

        dialog = HeatmapGifExportDialog(self.min_time, self.max_time, self.heatmap_ms.value(), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        start_s, end_s, frame_step_ms = dialog.values()
        frame_times = self._heatmap_gif_frame_times(start_s, end_s, frame_step_ms)
        if frame_times.size == 0:
            QMessageBox.information(self, "Export Heatmap GIF", "No frames are available for the selected time range.")
            return False

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export firing-rate heatmap GIF",
            str(Path("data") / "heatmap_activity.gif"),
            "GIF image (*.gif);;All files (*)",
        )
        if not path:
            return False
        if Path(path).suffix.lower() != ".gif":
            path = f"{path}.gif"

        try:
            from PIL import Image
        except ImportError:
            QMessageBox.critical(self, "Export Heatmap GIF failed", "Pillow is required to export animated GIF files.")
            return False

        self._refresh_heatmap_scale()
        progress = self._start_progress("Export Heatmap GIF", "Rendering heatmap frames...", int(frame_times.size))
        frames = []
        heatmap_duration_s = max(0.001, self.heatmap_ms.value() / 1000.0)
        selected_start_s = max(float(self.min_time), min(float(start_s), float(end_s)))
        resolution = max(128, min(512, int(max(self.heatmap_canvas.width(), self.heatmap_canvas.height(), 320))))
        try:
            for frame_index, stop_s in enumerate(frame_times):
                QApplication.processEvents()
                if _progress_cancel_requested(progress):
                    self._log("Heatmap GIF export cancelled")
                    return False
                window_start_s = max(selected_start_s, float(stop_s) - heatmap_duration_s)
                counts = self._window_channel_counts(window_start_s, float(stop_s))
                rgb = self.heatmap_canvas.render_counts_rgb(
                    counts,
                    resolution=resolution,
                    scale_max_count=self.heatmap_scale_count,
                )
                frames.append(Image.fromarray(rgb, mode="RGB"))
                _set_progress_dialog(progress, f"Rendering frame {frame_index + 1}/{frame_times.size}", frame_index + 1)

            if not frames:
                QMessageBox.information(self, "Export Heatmap GIF", "No frames were rendered for the selected time range.")
                return False
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=max(10, int(frame_step_ms)),
                loop=0,
                optimize=False,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export Heatmap GIF failed", str(exc))
            return False
        finally:
            self._finish_progress(progress)

        QMessageBox.information(self, "Export Heatmap GIF", f"Saved heatmap GIF:\n{path}")
        return True

    def _toggle_playback(self):
        if self.play_button.isChecked():
            total_ms = self._total_duration_ms()
            start_offset = self._window_start_offset_ms()
            stop_offset = min(total_ms, self._window_stop_offset_ms())
            if self._playback_time_ms is None:
                self._set_playback_time_ms(start_offset)
            elif self._playback_time_ms >= total_ms:
                self._set_playback_time_ms(start_offset)
            elif not (start_offset <= int(self._playback_time_ms) < stop_offset):
                self._set_playback_time_ms(start_offset)
            self.play_button.setText("Pause")
            self._update_view()
            self.play_timer.start()
        else:
            self._stop_playback()

    def _stop_playback(self):
        was_active = self.play_timer.isActive()
        if self.play_timer.isActive():
            self.play_timer.stop()
        if self.play_button.isChecked():
            self.play_button.blockSignals(True)
            self.play_button.setChecked(False)
            self.play_button.blockSignals(False)
        self.play_button.setText("Play")
        self.heatmap_canvas.set_fast_mode(False)
        if was_active:
            self._refresh_heatmap_for_view(force=True)
            self._refresh_waveforms_for_window(force=True)

    def _playback_step(self):
        step_ms = self.play_timer.interval()
        total_ms = self._total_duration_ms()
        if self._playback_time_ms is None:
            self._set_playback_time_ms(self._window_start_offset_ms())
        next_playhead = min(total_ms, int(self._playback_time_ms or 0) + step_ms)
        self._set_playback_time_ms(next_playhead)

        if self._short_data_window():
            self._set_slider_value_internal(min(self.slider.maximum(), int(self._playback_time_ms or 0)))
            self._refresh_heatmap_for_view(force=True)
        else:
            window_stop = self._window_stop_offset_ms()
            if next_playhead > window_stop and self.slider.value() < self.slider.maximum():
                target_start = min(
                    self.slider.maximum(),
                    max(self.slider.minimum(), next_playhead - self._window_ms()),
                )
                self._set_slider_value_internal(target_start)
                self._refresh_heatmap_for_view(force=True)
            else:
                self._update_view(force_heatmap=True)

        if next_playhead >= total_ms:
            self._stop_playback()

    def _window_ms(self) -> int:
        return max(1, self.window_grids.value() * self.grid_ms.value())

    def _refresh_bursts(self):
        spike_count = sum(np.asarray(times).size for _, times in self.spike_series)
        progress = None
        if len(self.spike_series) > 80 or spike_count > 500000:
            progress = self._start_progress("Burst detection", "Detecting bursts...", 0)
        def cancel_requested():
            if progress is None:
                return False
            QApplication.processEvents()
            return _progress_cancel_requested(progress)
        try:
            intervals = _detect_burst_intervals(
                self.spike_series,
                bin_ms=float(self.burst_bin_ms.value()),
                threshold_z=float(self.burst_threshold_z.value()),
                min_spikes=int(self.burst_min_spikes.value()),
                cancel_check=cancel_requested,
            )
            if cancel_requested():
                self._log("Burst detection cancelled")
                return
            self.burst_intervals = intervals
            self.canvas.set_bursts(self.burst_intervals)
            self.rate_canvas.set_bursts(self.burst_intervals)
        except InterruptedError as exc:
            self._log(str(exc) or "Burst detection cancelled")
        finally:
            self._finish_progress(progress)

    def _open_ibi_window(self):
        progress = self._start_progress("IBI", "Preparing IBI histogram...", 0)
        try:
            window = IBIWindow(self.burst_intervals, self)
        finally:
            self._finish_progress(progress)
        return self._show_analysis_window(window)

    def _open_isi_window(self):
        progress = self._start_progress("ISI", "Preparing ISI histogram...", 0)
        try:
            window = ISIWindow(self.spike_series, self)
        finally:
            self._finish_progress(progress)
        return self._show_analysis_window(window)

    def _open_burst_correlation_window(self):
        progress = self._start_progress("Burst correlation", "Computing burst correlation...", 0)
        try:
            window = BurstCorrelationWindow(self.spike_series, self.burst_intervals, self, self.channel_map)
        finally:
            self._finish_progress(progress)
        return self._show_analysis_window(window)

    def _open_burst_delay_window(self):
        for window in list(self.analysis_windows):
            if isinstance(window, BurstDelayWindow):
                window.show()
                window.raise_()
                window.activateWindow()
                return window
        window = BurstDelayWindow(
            self.spike_series,
            self.burst_intervals,
            self,
            self.channel_map,
            waveform_series=self.waveform_series,
            sampling_rate=self.sampling_rate,
        )
        return self._show_analysis_window(window)

    def _open_burst_clustering_window(self):
        progress = self._start_progress("Burst clustering", "Preparing burst clustering...", 0)
        try:
            window = BurstClusteringWindow(self.spike_series, self.burst_intervals, self)
        finally:
            self._finish_progress(progress)
        return self._show_analysis_window(window)

    def _save_bursts(self):
        if not self.burst_intervals:
            QMessageBox.information(self, "Save Bursts", "No detected bursts are available to save.")
            return False
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save burst sequences",
            str(Path("data") / "burst_sequences.npy"),
            "NumPy array (*.npy);;All files (*)",
        )
        if not path:
            return False
        if Path(path).suffix.lower() != ".npy":
            path = f"{path}.npy"
        progress = self._start_progress("Save bursts", "Preparing burst sequences...", 0)
        try:
            payload = _burst_sequence_payload(self.spike_series, self.burst_intervals)
            _set_progress_dialog(progress, "Saving burst sequences...")
            np.save(path, payload, allow_pickle=True)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return False
        finally:
            self._finish_progress(progress)
        QMessageBox.information(self, "Save Bursts", f"Saved {payload['burst_count']} bursts: {path}")
        return True

    def _show_analysis_window(self, window: QDialog):
        self.analysis_windows.append(window)
        window.finished.connect(lambda _: self._forget_analysis_window(window))
        window.show()
        return window

    def _forget_analysis_window(self, window: QDialog):
        if window in self.analysis_windows:
            self.analysis_windows.remove(window)

    def _window_channel_counts(self, start_s: float, stop_s: float) -> dict[str, int]:
        counts = {}
        for base_channel, values in self._count_series:
            lo = int(np.searchsorted(values, start_s, side="left"))
            hi = int(np.searchsorted(values, stop_s, side="right"))
            counts[base_channel] = counts.get(base_channel, 0) + max(0, hi - lo)
        return counts

    def _step_button(self, text: str) -> QPushButton:
        button = QPushButton(text)
        button.setFixedSize(28, 28)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setCursor(Qt.CursorShape.ArrowCursor)
        return button

    def _number_stepper(self, minus: QPushButton, field: QSpinBox, plus: QPushButton) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(minus)
        layout.addWidget(field)
        layout.addWidget(plus)
        widget.setFixedWidth(minus.width() + field.width() + plus.width() + 8)
        return widget

    def _zoom_grid_at(self, fraction: float, direction: int) -> None:
        old_window_ms = self._window_ms()
        anchor_ms = self.slider.value() + int(round(old_window_ms * fraction))
        step = max(1, self.grid_ms.singleStep() * 10)
        old_grid_ms = self.grid_ms.value()
        if direction > 0:
            new_grid_ms = max(self.grid_ms.minimum(), old_grid_ms - step)
        else:
            new_grid_ms = min(self.grid_ms.maximum(), old_grid_ms + step)
        if new_grid_ms == old_grid_ms:
            return

        self.grid_ms.blockSignals(True)
        self.grid_ms.setValue(new_grid_ms)
        self.grid_ms.blockSignals(False)

        new_window_ms = self._window_ms()
        new_max = self._slider_maximum_ms()
        self.slider.setMaximum(new_max)
        self.slider.setPageStep(max(1, new_window_ms // 2))
        target_start = anchor_ms - int(round(new_window_ms * fraction))
        self.slider.blockSignals(True)
        self.slider.setValue(min(new_max, max(0, target_start)))
        self.slider.blockSignals(False)
        self._update_view()

    def _pan_to_absolute_ms(self, absolute_start_ms: int) -> None:
        target_slider = int(round(absolute_start_ms - self.min_time * 1000))
        self.slider.setValue(min(self.slider.maximum(), max(self.slider.minimum(), target_slider)))

    def _select_channel(self, channel: str) -> None:
        self.selected_channel = channel
        self._last_waveform_view = None
        self.canvas.set_selected_channel(channel)
        self._refresh_waveforms_for_window(force=True)

    def _set_slider_value_internal(self, value: int) -> None:
        self._internal_slider_update = True
        try:
            self.slider.setValue(value)
        finally:
            self._internal_slider_update = False

    def _refresh_waveforms_for_window(self, *, force: bool = False) -> None:
        channel = self.selected_channel
        if not channel:
            return
        now = time.monotonic()
        if self.play_timer.isActive() and not force and (now - self._last_waveform_refresh) < self._waveform_min_interval_s:
            return

        waveforms = self.waveform_series.get(channel)
        times = self.spike_lookup.get(channel)
        if waveforms is None or times is None:
            key = (channel, None, None)
            if self._last_waveform_view == key:
                return
            self._last_waveform_view = key
            self._last_waveform_refresh = now
            self.waveform_canvas.set_channel_waveforms(channel, None, self.sampling_rate)
            return

        waveforms = np.asarray(waveforms)
        times = np.asarray(times, dtype=float)
        if waveforms.shape[0] != times.size:
            key = (channel, "all", int(waveforms.shape[0]))
            if self._last_waveform_view == key:
                return
            self._last_waveform_view = key
            self._last_waveform_refresh = now
            self.waveform_canvas.set_channel_waveforms(channel, waveforms, self.sampling_rate)
            return

        start = self.canvas.window_start
        stop = start + self.canvas.window_duration
        lo = int(np.searchsorted(times, start, side="left"))
        hi = int(np.searchsorted(times, stop, side="right"))
        key = (channel, lo, hi)
        if self._last_waveform_view == key:
            return
        self._last_waveform_view = key
        self._last_waveform_refresh = now
        if hi - lo > self._waveform_window_limit:
            indices = np.linspace(lo, hi - 1, self._waveform_window_limit, dtype=int)
            window_waveforms = waveforms[indices]
        else:
            window_waveforms = waveforms[lo:hi]
        self.waveform_canvas.set_channel_waveforms(channel, window_waveforms, self.sampling_rate)


class ResultsWindow(QDialog):
    def __init__(self, result: PipelineResult, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pipeline Results")
        self.resize(760, 520)

        tabs = QTabWidget()
        tabs.addTab(self._summary_tab(result), "Summary")
        tabs.addTab(self._stats_tab(result), "Statistics")
        tabs.addTab(self._spike_tab(result), "Spikes")

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)

    def _summary_tab(self, result: PipelineResult):
        widget = QWidget()
        text = QTextEdit()
        text.setReadOnly(True)
        raw_shape = " x ".join(str(v) for v in np.asarray(result.raw).shape)
        processed_shape = " x ".join(str(v) for v in np.asarray(result.processed).shape)
        text.setPlainText(
            "\n".join(
                [
                    f"Raw shape: {raw_shape}",
                    f"Processed shape: {processed_shape}",
                    f"Channels: {result.analysis['channel_count']}",
                    f"Samples: {result.analysis['sample_count']}",
                    f"Saved file: {result.output_path}",
                ]
            )
        )
        layout = QVBoxLayout(widget)
        layout.addWidget(text)
        return widget

    def _stats_tab(self, result: PipelineResult):
        widget = QWidget()
        text = QTextEdit()
        text.setReadOnly(True)
        stats = result.analysis["statistics"]
        lines = []
        for name, values in stats.items():
            preview = ", ".join(f"{value:.4g}" for value in np.asarray(values)[:12])
            lines.append(f"{name}: {preview}")
        text.setPlainText("\n".join(lines))
        layout = QVBoxLayout(widget)
        layout.addWidget(text)
        return widget

    def _spike_tab(self, result: PipelineResult):
        widget = QWidget()
        text = QTextEdit()
        text.setReadOnly(True)
        lines = [f"Channel {index + 1}: {len(spikes)} spikes" for index, spikes in enumerate(result.spikes)]
        text.setPlainText("\n".join(lines))
        layout = QVBoxLayout(widget)
        layout.addWidget(text)
        return widget


class NevResultsWindow(QDialog):
    def __init__(self, data: UnifiedMEAData, parent=None):
        super().__init__(parent)
        source = data.meta.get("source", "") if isinstance(data.meta, dict) else ""
        self.setWindowTitle("Axion SPK Results" if source == "axion_spk" else "NEV Results")
        self.resize(760, 520)

        tabs = QTabWidget()
        tabs.addTab(self._summary_tab(data), "Summary")
        tabs.addTab(self._channels_tab(data), "Channels")
        tabs.addTab(self._sorting_tab(data), "Sorting")

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)

    def _summary_tab(self, data: UnifiedMEAData):
        widget = QWidget()
        text = QTextEdit()
        text.setReadOnly(True)
        start, stop = data.time_range()
        spike_count = sum(values.size for values in data.spikes.values())
        waveform_shapes = [value.shape for value in data.waveforms.values()]
        waveform_shape = waveform_shapes[0] if waveform_shapes else "None"
        header = data.meta.get("basic_header", {})
        source = data.meta.get("source", "Unknown")
        lines = [
            f"File: {data.meta.get('file', '')}",
            f"Source: {source}",
            f"Channels: {len(data.spikes)}",
            f"Spikes: {spike_count}",
            f"Time range: {start:.6g} - {stop:.6g} s",
            f"Waveform sampling rate: {data.sr or 'Unknown'} Hz",
            f"Waveform shape per channel: {waveform_shape}",
            f"Stim/event packets: {data.stim_times.size}",
        ]
        if source == "axion_spk":
            lines.extend(
                [
                    f"Wells: {', '.join(data.meta.get('wells', [])) or 'Unknown'}",
                    f"Electrode sites: {data.meta.get('electrode_count', 'Unknown')}",
                ]
            )
        elif source == "maxwell_h5":
            lines.extend(
                [
                    f"Wells: {', '.join(data.meta.get('wells', [])) or 'Unknown'}",
                    f"Recordings: {', '.join(data.meta.get('recordings', [])) or 'Unknown'}",
                    f"Duration: {data.meta.get('duration_s', stop):.6g} s",
                    f"Stim artifact removal: +/-{data.meta.get('stim_artifact_window_ms', 0):g} ms, "
                    f"{data.meta.get('stim_artifact_removed_count', 0)} spikes",
                ]
            )
        else:
            lines.extend(
                [
                    f"NEV file spec: {header.get('file_spec', 'Unknown')}",
                    f"Data packet bytes: {header.get('bytes_in_data_packets', 'Unknown')}",
                    f"Stim markers: {data.stim_times.size}",
                ]
            )
        text.setPlainText("\n".join(lines))
        layout = QVBoxLayout(widget)
        layout.addWidget(text)
        return widget

    def _channels_tab(self, data: UnifiedMEAData):
        widget = QWidget()
        text = QTextEdit()
        text.setReadOnly(True)
        lines = []
        for channel in sorted(data.channels(), key=_channel_sort_key):
            spikes = data.spikes[channel]
            waveforms = data.waveforms.get(channel)
            waveform_text = f", waveform {waveforms.shape}" if waveforms is not None else ""
            lines.append(f"{channel}: {spikes.size} spikes{waveform_text}")
        text.setPlainText("\n".join(lines))
        layout = QVBoxLayout(widget)
        layout.addWidget(text)
        return widget

    def _sorting_tab(self, data: UnifiedMEAData):
        widget = QWidget()
        text = QTextEdit()
        text.setReadOnly(True)
        meta = data.sorting.get("_waveform_clustering", {}) if isinstance(data.sorting, dict) else {}
        summary = meta.get("summary", {}) if isinstance(meta, dict) else {}
        lines = [
            f"Method: {meta.get('method', 'None') if isinstance(meta, dict) else 'None'}",
            f"Sorted channels: {summary.get('sorted_channels', 0)}",
            f"Total clusters: {summary.get('total_clusters', 0)}",
            f"Total spikes: {summary.get('total_spikes', 0)}",
            "",
        ]
        for channel in sorted(data.channels(), key=_channel_sort_key):
            payload = data.sorting.get(channel, {}) if isinstance(data.sorting, dict) else {}
            labels = np.asarray(payload.get("waveform_cluster_labels", []))
            if labels.size:
                clusters = np.unique(labels).size
                silhouette = payload.get("silhouette")
                score = "n/a" if silhouette is None else f"{float(silhouette):.3f}"
                lines.append(f"{channel}: {clusters} clusters, {labels.size} spikes, silhouette {score}")
            else:
                lines.append(f"{channel}: not sorted")
        text.setPlainText("\n".join(lines))
        layout = QVBoxLayout(widget)
        layout.addWidget(text)
        return widget


class SortingResultsWindow(QDialog):
    def __init__(self, data: UnifiedMEAData, parent=None):
        super().__init__(parent)
        self.data = data
        self.setWindowTitle("Sorting Results")
        self.resize(1120, 760)

        self.channel_combo = QComboBox()
        self.sorted_channels = self._sorted_channels()
        self.channel_combo.addItems(self.sorted_channels)
        self.channel_combo.currentIndexChanged.connect(self._refresh_plots)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)

        self.embedding_canvas = FigureCanvas(Figure(figsize=(7, 4), tight_layout=True))
        self.waveform_canvas = FigureCanvas(Figure(figsize=(7, 4), tight_layout=True))

        tabs = QTabWidget()
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        summary_layout.addWidget(self.summary_text)
        tabs.addTab(summary_tab, "Summary")

        embedding_tab = QWidget()
        embedding_layout = QVBoxLayout(embedding_tab)
        embedding_layout.addWidget(self.embedding_canvas)
        tabs.addTab(embedding_tab, "Embedding")

        waveform_tab = QWidget()
        waveform_layout = QVBoxLayout(waveform_tab)
        waveform_layout.addWidget(self.waveform_canvas)
        tabs.addTab(waveform_tab, "Cluster Waveforms")

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Channel"))
        top.addWidget(self.channel_combo)
        top.addStretch()
        layout.addLayout(top)
        layout.addWidget(tabs)

        self._refresh_summary()
        self._refresh_plots()
        _fix_spinbox_hit_targets(self)

    def _sorted_channels(self):
        channels = []
        for channel in sorted(self.data.channels(), key=_channel_sort_key):
            payload = self.data.sorting.get(channel, {}) if isinstance(self.data.sorting, dict) else {}
            labels = np.asarray(payload.get("waveform_cluster_labels", []))
            if labels.size:
                channels.append(channel)
        return channels

    def _refresh_summary(self):
        meta = self.data.sorting.get("_waveform_clustering", {}) if isinstance(self.data.sorting, dict) else {}
        summary = meta.get("summary", {}) if isinstance(meta, dict) else {}
        params = meta.get("params", {}) if isinstance(meta, dict) else {}
        lines = [
            f"Method: {meta.get('method', 'None') if isinstance(meta, dict) else 'None'}",
            f"Reduction: {params.get('reduction_method', 'n/a')}",
            f"Clustering: {params.get('clustering_method', 'n/a')}",
            f"Sorted channels: {summary.get('sorted_channels', 0)}",
            f"Total clusters: {summary.get('total_clusters', 0)}",
            f"Total spikes: {summary.get('total_spikes', 0)}",
            "",
        ]
        for channel in self.sorted_channels:
            payload = self.data.sorting.get(channel, {})
            labels = np.asarray(payload.get("waveform_cluster_labels", []))
            silhouette = payload.get("silhouette")
            score = "n/a" if silhouette is None else f"{float(silhouette):.3f}"
            lines.append(f"{channel}: {np.unique(labels).size} clusters, {labels.size} spikes, silhouette {score}")
        self.summary_text.setPlainText("\n".join(lines))

    def _refresh_plots(self):
        channel = self.channel_combo.currentText()
        self._draw_embedding(channel)
        self._draw_cluster_waveforms(channel)

    def _draw_embedding(self, channel: str):
        figure = self.embedding_canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        payload = self.data.sorting.get(channel, {}) if isinstance(self.data.sorting, dict) else {}
        embedding = np.asarray(payload.get("embedding", []), dtype=float)
        labels = np.asarray(payload.get("waveform_cluster_labels", []), dtype=int)
        if embedding.ndim != 2 or embedding.shape[0] == 0 or embedding.shape[1] == 0 or labels.size == 0:
            ax.text(0.5, 0.5, "No embedding for selected channel", ha="center", va="center")
        else:
            if embedding.shape[1] == 1:
                x = embedding[:, 0]
                y = np.zeros_like(x)
            else:
                x = embedding[:, 0]
                y = embedding[:, 1]
            indices = _display_indices(labels.size, 6000)
            color_map = _cluster_color_map(labels)
            for label, color in color_map.items():
                cluster_indices = indices[labels[indices] == label]
                legend_label = "noise" if label == -1 else f"cluster {label}"
                ax.scatter(
                    x[cluster_indices],
                    y[cluster_indices],
                    color=color,
                    s=9,
                    alpha=0.78,
                    label=legend_label,
                )
            ax.set_title(f"{channel} reduction space")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.legend(loc="best", fontsize=8)
        self.embedding_canvas.draw_idle()

    def _draw_cluster_waveforms(self, channel: str):
        figure = self.waveform_canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        payload = self.data.sorting.get(channel, {}) if isinstance(self.data.sorting, dict) else {}
        labels = np.asarray(payload.get("waveform_cluster_labels", []), dtype=int)
        waveforms = np.asarray(self.data.waveforms.get(channel, []), dtype=float)
        if waveforms.ndim != 2 or labels.size != waveforms.shape[0]:
            ax.text(0.5, 0.5, "No clustered waveforms for selected channel", ha="center", va="center")
        else:
            x = _waveform_time_axis(waveforms.shape[1], self.data.sr)
            color_map = _cluster_color_map(labels)
            for label, color in color_map.items():
                cluster_waveforms = waveforms[labels == label]
                draw = cluster_waveforms[_display_indices(cluster_waveforms.shape[0], 90)]
                for waveform in draw:
                    ax.plot(x, waveform, color=color, alpha=0.10, linewidth=0.7)
                legend_label = "noise" if label == -1 else f"cluster {label}"
                ax.plot(x, np.mean(cluster_waveforms, axis=0), color=color, linewidth=2.0, label=legend_label)
            ax.set_title(f"{channel} clustered spike waveforms")
            ax.set_xlabel("Time (ms)" if self.data.sr else "Time (sample index)")
            ax.set_ylabel("Voltage (uV)")
            ax.legend(loc="best", fontsize=8)
        self.waveform_canvas.draw_idle()


class MaxwellFootprintResultsWindow(QDialog):
    def __init__(self, data: UnifiedMEAData, result: dict, parent=None):
        super().__init__(parent)
        self.data = data
        self.result = result
        self.setWindowTitle("Maxwell Footprint Analysis")
        self.resize(1400, 900)

        self.unit_combo = QComboBox()
        for neuron in result.get("units", result.get("neurons", [])):
            footprint = neuron.get("footprint", {})
            self.unit_combo.addItem(
                f"unit {neuron['id']} | {footprint.get('mask_electrode_count', 0)} els | {footprint.get('total_spikes', 0)} spikes",
                int(neuron["id"]),
            )
        self.unit_combo.currentIndexChanged.connect(self._draw_unit)
        self.summary_label = QLabel(self._summary_text())
        self.summary_label.setObjectName("MutedText")

        self.overview_canvas = FigureCanvas(Figure(figsize=(8, 4.5), tight_layout=True))
        self.unit_canvas = FigureCanvas(Figure(figsize=(8, 6), tight_layout=True))

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Neuron"))
        controls.addWidget(self.unit_combo)
        controls.addWidget(self.summary_label, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(QLabel("Method diagnostics"))
        layout.addWidget(self.overview_canvas, 1)
        layout.addWidget(QLabel("Selected unit core, footprint and waveforms"))
        layout.addWidget(self.unit_canvas, 2)

        self._draw_overview()
        self._draw_unit()
        self.showMaximized()

    def _summary_text(self) -> str:
        summary = self.result.get("summary", {})
        return (
            f"targets: {summary.get('target_count', 0)} | "
            f"analyzed units: {summary.get('analyzed_units', 0)} | "
            f"skipped: {summary.get('skipped_units', 0)}"
        )

    def _draw_overview(self):
        figure = self.overview_canvas.figure
        figure.clear()
        axes = figure.subplots(1, 3)
        targets = self.result.get("targets", [])
        neurons = self.result.get("units", self.result.get("neurons", []))

        ax = axes[0]
        if targets:
            xs = [target["x_um"] for target in targets]
            ys = [target["y_um"] for target in targets]
            amps = [max(target.get("mean_spike_amplitude_uv", 0.0), 1e-6) for target in targets]
            ax.scatter(xs, ys, s=np.clip(np.asarray(amps) * 1.2, 12, 120), c=amps, cmap="turbo", alpha=0.85)
        ax.set_title("Selected unit centers")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_aspect("equal", adjustable="box")

        ax = axes[1]
        if neurons:
            events = [neuron.get("selection", {}).get("event_count", 0) for neuron in neurons]
            mask_sizes = [neuron.get("footprint", {}).get("mask_electrode_count", 0) for neuron in neurons]
            ax.scatter(events, mask_sizes, color="#2563eb", alpha=0.85)
        ax.set_title("Core events vs footprint size")
        ax.set_xlabel("core events")
        ax.set_ylabel("masked electrodes")

        ax = axes[2]
        if neurons:
            mask_sizes = [neuron.get("footprint", {}).get("mask_electrode_count", 0) for neuron in neurons]
            ax.hist(mask_sizes, bins=20, color="#0f766e", alpha=0.78)
        ax.set_title("Electrode mask sizes")
        ax.set_xlabel("masked electrodes")
        ax.set_ylabel("unit count")

        self.overview_canvas.draw_idle()

    def _draw_unit(self):
        figure = self.unit_canvas.figure
        figure.clear()
        neurons = self.result.get("units", self.result.get("neurons", []))
        if not neurons:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, "No analyzed units", ha="center", va="center")
            ax.axis("off")
            self.unit_canvas.draw_idle()
            return

        neuron_id = self.unit_combo.currentData()
        neuron = next((item for item in neurons if int(item["id"]) == int(neuron_id)), neurons[0])
        selection = neuron.get("selection", {})
        footprint = neuron.get("footprint", {})
        entries = footprint.get("entries", [])
        selected = [entry for entry in entries if entry.get("mask")]

        grid = figure.add_gridspec(2, 3, height_ratios=[1.05, 1.2])
        ax_map = figure.add_subplot(grid[0, 0])
        ax_clusters = figure.add_subplot(grid[0, 1])
        ax_latency = figure.add_subplot(grid[0, 2])
        ax_templates = [figure.add_subplot(grid[1, index]) for index in range(3)]

        if entries:
            ax_map.scatter(
                [entry["x_um"] for entry in entries],
                [entry["y_um"] for entry in entries],
                s=8,
                color="#cbd5e1",
                alpha=0.35,
            )
        if selected:
            amplitudes = np.asarray([entry.get("amplitude_uv", 0.0) for entry in selected], dtype=float)
            scatter = ax_map.scatter(
                [entry["x_um"] for entry in selected],
                [entry["y_um"] for entry in selected],
                c=amplitudes,
                cmap="turbo",
                s=np.clip(amplitudes * 1.1, 28, 140),
                edgecolor="#111827",
                linewidth=0.7,
            )
            figure.colorbar(scatter, ax=ax_map, label="STA p2p (uV)")
            center_channel = selection.get("center_channel", "")
            for entry in selected[:14]:
                ax_map.text(entry["x_um"], entry["y_um"], entry["channel"].split("_")[-1], fontsize=7)
            center = next((entry for entry in entries if entry.get("channel") == center_channel), None)
            if center is not None:
                ax_map.scatter([center["x_um"]], [center["y_um"]], marker="x", s=90, c="#000000", linewidths=2.2)
        core_channels = set(selection.get("core_channels", []))
        if entries and core_channels:
            core_entries = [entry for entry in entries if entry.get("channel") in core_channels]
            if core_entries:
                ax_map.scatter(
                    [entry["x_um"] for entry in core_entries],
                    [entry["y_um"] for entry in core_entries],
                    marker="s",
                    s=80,
                    facecolors="none",
                    edgecolors="#f97316",
                    linewidth=1.4,
                    label="unit core",
                )
                ax_map.legend(loc="best", fontsize=8)
        ax_map.set_title(f"unit {neuron['id']} extracted footprint")
        ax_map.set_xlabel("x (um)")
        ax_map.set_ylabel("y (um)")
        ax_map.set_aspect("equal", adjustable="box")

        amplitude_matrix = np.asarray(selection.get("core_amplitude_matrix", []), dtype=float)
        if amplitude_matrix.ndim == 2 and amplitude_matrix.size:
            draw = amplitude_matrix
            if draw.shape[0] > 600:
                draw = draw[np.linspace(0, draw.shape[0] - 1, 600, dtype=int)]
            event_index = np.arange(draw.shape[0])
            labels = list(selection.get("core_channels", []))
            for column in range(draw.shape[1]):
                label = labels[column].split("_")[-1] if column < len(labels) else f"core {column + 1}"
                ax_clusters.scatter(event_index, draw[:, column], s=6, alpha=0.45, label=label)
            ax_clusters.legend(loc="best", fontsize=7)
        ax_clusters.set_title(
            "Four-electrode unit core\n"
            f"{selection.get('event_count', 0)} / {selection.get('candidate_event_count', 0)} refined events"
        )
        ax_clusters.set_xlabel("event index")
        ax_clusters.set_ylabel("negative amplitude (uV)")

        if selected:
            latencies = np.asarray([entry.get("latency_ms", 0.0) for entry in selected], dtype=float)
            order = np.argsort(latencies)
            ax_latency.bar(np.arange(len(order)), latencies[order], color="#0f766e")
            ax_latency.set_xticks(np.arange(len(order)))
            ax_latency.set_xticklabels([selected[index]["channel"].split("_")[-1] for index in order], rotation=70, fontsize=7)
        ax_latency.set_title("STA latency from trigger")
        ax_latency.set_ylabel("latency (ms)")

        for ax in ax_templates:
            ax.axis("off")
        for panel, entry in enumerate(selected[:9]):
            ax = ax_templates[panel // 3]
            template = np.asarray(entry.get("template", []), dtype=float)
            if template.size == 0:
                continue
            offset = panel % 3
            x = np.arange(template.size)
            y = template + offset * (np.nanmax(np.abs(template)) * 2.4 + 1.0)
            ax.plot(x, y, color="#1d4ed8", linewidth=1.1)
            ax.text(0, y[0], entry["channel"].split("_")[-1], fontsize=8, va="center")
            ax.axis("on")
            ax.set_yticks([])
            ax.set_title("Channel template small multiples" if panel == 0 else "")
            ax.set_xlabel("sample")

        self.unit_canvas.draw_idle()


class SortingWorkspaceWindow(QDialog):
    def __init__(self, data: UnifiedMEAData, parent=None):
        super().__init__(parent)
        self.data = data
        self.setWindowTitle("Sorting")
        self.resize(1280, 820)
        self.thread_pool = QThreadPool.globalInstance()
        self.current_embedding = np.zeros((0, 0), dtype=np.float32)
        self.current_labels = np.zeros(0, dtype=np.int32)
        self.undo_stack = {}
        self.active_sorting_workers = []
        self.sorting_progress = None
        self.lasso = None
        self.lasso_mode = None
        self.embedding_view_limits = None
        self.analysis_windows = []
        self.dirty = False

        self.channel_combo = QComboBox()
        self.channel_combo.addItems(sorted(data.channels(), key=_channel_sort_key))
        self.channel_combo.currentIndexChanged.connect(self._load_channel)

        self.params = AutoSortingDialog(self)
        self.params.setWindowFlags(Qt.WindowType.Widget)
        self.params.method.setVisible(False)
        self.params.method.setParent(None)
        self.params.run_button.setVisible(False)
        self.params.cancel_button.setVisible(False)

        self.run_auto_button = QPushButton("Run Auto Sorting")
        self.run_auto_button.setObjectName("PrimaryButton")
        self.run_auto_button.clicked.connect(self._run_auto_sorting)
        self.run_channel_button = QPushButton("Run Channel Sorting")
        self.run_channel_button.clicked.connect(self._run_channel_sorting)
        self.maxwell_footprint_button = QPushButton("Footprint Analysis")
        self.maxwell_footprint_button.clicked.connect(self._run_maxwell_footprint_analysis)
        source = data.meta.get("source", "") if isinstance(data.meta, dict) else ""
        self.maxwell_footprint_button.setVisible(source == "maxwell_h5")
        self.save_sorting_button = QPushButton("Save Sorting")
        self.save_sorting_button.clicked.connect(self._save_sorting)
        self.compute_embedding_button = QPushButton("Compute Embedding")
        self.compute_embedding_button.clicked.connect(self._compute_current_embedding)
        self.cluster_id = QSpinBox()
        self.cluster_id.setRange(0, 99)
        self.cluster_id.setValue(0)
        self.lasso_button = QPushButton("Assign Cluster")
        self.lasso_button.setCheckable(True)
        self.lasso_button.clicked.connect(self._start_lasso)
        self.noise_button = QPushButton("Assign Noise")
        self.noise_button.setCheckable(True)
        self.noise_button.clicked.connect(self._start_noise_lasso)
        self.hide_noise = QCheckBox("Hide noise")
        self.hide_noise.stateChanged.connect(lambda *_: self._draw_all())
        self.cluster_filter = QComboBox()
        self.cluster_filter.currentIndexChanged.connect(lambda *_: self._draw_all())
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self._undo_manual_assignment)
        self.undo_button.setEnabled(False)
        self.pending_assignment_label = 0
        self.status = QLabel("Ready")
        self.status.setObjectName("MutedText")
        self.status.setMinimumWidth(0)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        self.waveform_canvas = FigureCanvas(Figure(figsize=(6, 5), tight_layout=True))
        self.embedding_canvas = FigureCanvas(Figure(figsize=(6, 5), tight_layout=True))
        self.embedding_canvas.mpl_connect("scroll_event", self._embedding_scroll_zoom)
        self.embedding_ax = None

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Channel"))
        controls.addWidget(self.channel_combo)
        controls.addWidget(self.run_auto_button)
        controls.addWidget(self.run_channel_button)
        controls.addWidget(self.maxwell_footprint_button)
        controls.addWidget(self.save_sorting_button)
        controls.addWidget(self.compute_embedding_button)
        controls.addWidget(QLabel("Assign cluster"))
        controls.addWidget(self.cluster_id)
        controls.addWidget(self.lasso_button)
        controls.addWidget(self.noise_button)
        controls.addWidget(self.hide_noise)
        controls.addWidget(QLabel("Show"))
        controls.addWidget(self.cluster_filter)
        controls.addWidget(self.undo_button)
        controls.addWidget(self.status, 1)

        plots = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Spike waveforms"))
        left.addWidget(self.waveform_canvas, 1)
        right = QVBoxLayout()
        right.addWidget(QLabel("Reduction space"))
        right.addWidget(self.embedding_canvas, 1)
        plots.addLayout(left, 1)
        plots.addLayout(right, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addLayout(plots, 1)
        layout.addWidget(self.params)

        self._load_channel()
        _fix_spinbox_hit_targets(self)
        self.showMaximized()

    def _start_progress(self, title: str, message: str, maximum: int = 0) -> QProgressDialog:
        return _create_progress_dialog(self, title, message, maximum)

    def _update_progress(self, value: int, message: str) -> None:
        self.status.setText(message)
        _set_progress_dialog(self.sorting_progress, message, value)

    def _finish_progress(self) -> None:
        _close_progress_dialog(self.sorting_progress)
        self.sorting_progress = None

    def _config(self) -> WaveformClusteringConfig:
        return self.params.get_config()

    def _channel(self) -> str:
        return self.channel_combo.currentText()

    def _load_channel(self):
        if self.lasso_mode is not None:
            self._stop_lasso_mode("Assignment mode off")
        self.embedding_view_limits = None
        channel = self._channel()
        self.undo_button.setEnabled(bool(self.undo_stack.get(channel)))
        payload = self.data.sorting.get(channel, {}) if isinstance(self.data.sorting, dict) else {}
        self.current_labels = np.asarray(payload.get("waveform_cluster_labels", []), dtype=np.int32)
        self.current_embedding = np.asarray(payload.get("embedding", []), dtype=np.float32)
        waveforms = np.asarray(self.data.waveforms.get(channel, []))
        if waveforms.ndim == 2 and self.current_labels.size != waveforms.shape[0]:
            self.current_labels = np.zeros(waveforms.shape[0], dtype=np.int32)
        if self.current_embedding.ndim != 2 or self.current_embedding.shape[0] != self.current_labels.size:
            self._compute_current_embedding(update_status=False)
        self._refresh_cluster_filter()
        self._draw_all()

    def _run_auto_sorting(self):
        if not self.data.waveforms:
            QMessageBox.warning(self, "Sorting", "The loaded NEV file does not contain spike waveforms.")
            return
        self._set_sorting_buttons_enabled(False)
        self.status.setText("Auto sorting running...")
        self.sorting_progress = self._start_progress("Auto sorting", "Auto sorting running...", 100)
        worker = SortingWorker(self.data, self._config())
        worker.signals.progress.connect(self._update_progress)
        worker.signals.finished.connect(lambda sorting, worker=worker: self._sorting_worker_finished(sorting, worker))
        worker.signals.failed.connect(lambda details, worker=worker: self._sorting_worker_failed(details, worker))
        self.active_sorting_workers.append(worker)
        self.thread_pool.start(worker)

    def _run_channel_sorting(self):
        channel = self._channel()
        waveforms = np.asarray(self.data.waveforms.get(channel, []))
        if waveforms.ndim != 2 or waveforms.shape[0] == 0:
            QMessageBox.warning(self, "Sorting", f"{channel} does not contain spike waveforms.")
            return
        self._set_sorting_buttons_enabled(False)
        self.status.setText(f"Sorting {channel}...")
        self.sorting_progress = self._start_progress("Channel sorting", f"Sorting {channel}...", 100)
        worker = SortingWorker(self.data, self._config(), channels=[channel])
        worker.signals.progress.connect(self._update_progress)
        worker.signals.finished.connect(lambda sorting, worker=worker: self._sorting_worker_finished(sorting, worker))
        worker.signals.failed.connect(lambda details, worker=worker: self._sorting_worker_failed(details, worker))
        self.active_sorting_workers.append(worker)
        self.thread_pool.start(worker)

    def _run_maxwell_footprint_analysis(self):
        source = self.data.meta.get("source", "") if isinstance(self.data.meta, dict) else ""
        if source != "maxwell_h5":
            QMessageBox.information(self, "Maxwell Footprint Analysis", "Footprint analysis is only enabled for Maxwell H5 data.")
            return
        if not self.data.waveforms:
            QMessageBox.warning(self, "Maxwell Footprint Analysis", "Footprint analysis requires spike-aligned waveforms.")
            return
        dialog = MaxwellFootprintDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        self._set_sorting_buttons_enabled(False)
        self.status.setText("Maxwell footprint analysis running...")
        self.sorting_progress = self._start_progress("Maxwell Footprint Analysis", "Running footprint analysis...", 100)
        worker = MaxwellFootprintWorker(self.data, dialog.get_config())
        worker.signals.progress.connect(self._update_progress)
        worker.signals.finished.connect(lambda result, worker=worker: self._footprint_worker_finished(result, worker))
        worker.signals.failed.connect(lambda details, worker=worker: self._footprint_worker_failed(details, worker))
        self.active_sorting_workers.append(worker)
        self.thread_pool.start(worker)

    def _set_sorting_buttons_enabled(self, enabled: bool):
        self.run_auto_button.setEnabled(enabled)
        self.run_channel_button.setEnabled(enabled)
        self.maxwell_footprint_button.setEnabled(enabled)
        self.compute_embedding_button.setEnabled(enabled)
        self.channel_combo.setEnabled(enabled)

    def _footprint_worker_finished(self, result: dict, worker: MaxwellFootprintWorker):
        self._forget_sorting_worker(worker)
        self._finish_progress()
        self._set_sorting_buttons_enabled(True)
        self.data.sorting["_maxwell_footprint_analysis"] = result
        self.dirty = True
        summary = result.get("summary", {})
        self.status.setText(
            f"Maxwell footprint analysis complete: {summary.get('analyzed_units', 0)} units, "
            f"{summary.get('target_count', 0)} targets"
        )
        window = MaxwellFootprintResultsWindow(self.data, result, self)
        self._show_analysis_window(window)
        window.show()

    def _show_analysis_window(self, window: QDialog):
        self.analysis_windows.append(window)
        window.finished.connect(lambda *_: self._forget_analysis_window(window))
        return window

    def _forget_analysis_window(self, window: QDialog):
        if window in self.analysis_windows:
            self.analysis_windows.remove(window)

    def _footprint_worker_failed(self, details: str, worker: MaxwellFootprintWorker):
        self._forget_sorting_worker(worker)
        self._finish_progress()
        self._set_sorting_buttons_enabled(True)
        self.status.setText("Maxwell footprint analysis failed")
        QMessageBox.critical(self, "Maxwell footprint analysis failed", details.splitlines()[-1])

    def _auto_sorting_finished(self, sorting: dict):
        self._finish_progress()
        self._set_sorting_buttons_enabled(True)
        if self.lasso_mode is not None:
            self._stop_lasso_mode("Assignment mode off")
        self._apply_sorting(sorting)
        summary = sorting.get("summary", {})
        sorted_channels = summary.get("sorted_channels", 0)
        total_clusters = summary.get("total_clusters", 0)
        if sorted_channels == 1 and len(sorting.get("channels", {})) == 1:
            channel = next(iter(sorting.get("channels", {})))
            self.status.setText(f"Channel sorting complete: {channel}, {total_clusters} clusters")
        else:
            self.status.setText(f"Auto sorting complete: {sorted_channels} channels, {total_clusters} clusters")
        self._load_channel()

    def _auto_sorting_failed(self, details: str):
        self._finish_progress()
        self._set_sorting_buttons_enabled(True)
        self.status.setText("Auto sorting failed")
        QMessageBox.critical(self, "Sorting failed", details.splitlines()[-1])

    def _sorting_worker_finished(self, sorting: dict, worker: SortingWorker):
        self._forget_sorting_worker(worker)
        self._auto_sorting_finished(sorting)

    def _sorting_worker_failed(self, details: str, worker: SortingWorker):
        self._forget_sorting_worker(worker)
        self._auto_sorting_failed(details)

    def _forget_sorting_worker(self, worker: SortingWorker):
        if worker in self.active_sorting_workers:
            self.active_sorting_workers.remove(worker)

    def _save_sorting(self):
        default_name = "sorted_mea_data.npz"
        source = self.data.meta.get("file") if isinstance(self.data.meta, dict) else None
        if source:
            default_name = f"{Path(source).stem}_sorted.npz"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save sorted data",
            str(Path("data") / default_name),
            "Unified sorting data (*.npz);;All files (*)",
        )
        if not path:
            return False
        progress = self._start_progress("Save sorting", "Saving sorting result...", 0)
        try:
            saved = save_unified_npz(self.data, path)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return False
        finally:
            _close_progress_dialog(progress)
        self.status.setText(f"Saved sorting: {Path(saved).name}")
        self.status.setToolTip(str(saved))
        self.dirty = False
        return True

    def closeEvent(self, event):  # noqa: N802 - Qt override
        if not self.dirty:
            event.accept()
            return

        choice = QMessageBox.question(
            self,
            "Unsaved sorting",
            "The current sorting result has not been saved. Save before closing?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if choice == QMessageBox.StandardButton.Yes:
            if self._save_sorting():
                event.accept()
            else:
                event.ignore()
        elif choice == QMessageBox.StandardButton.No:
            event.accept()
        else:
            event.ignore()

    def _apply_sorting(self, sorting: dict):
        self.data.sorting["_waveform_clustering"] = {
            "method": sorting.get("method"),
            "params": sorting.get("params", {}),
            "summary": sorting.get("summary", {}),
        }
        for channel, payload in sorting.get("channels", {}).items():
            labels = np.asarray(payload.get("labels", []), dtype=np.int32)
            embedding = np.asarray(payload.get("embedding", []), dtype=np.float32)
            channel_sorting = self.data.sorting.setdefault(channel, {})
            channel_sorting["method"] = "waveform_clustering"
            channel_sorting["labels"] = labels
            channel_sorting["waveform_cluster_labels"] = labels
            channel_sorting["embedding"] = embedding
            channel_sorting["cluster_count"] = int(payload.get("cluster_count", 0))
            channel_sorting["silhouette"] = payload.get("silhouette")
        self.dirty = True

    def _compute_current_embedding(self, update_status=True):
        if self.lasso_mode is not None:
            self._stop_lasso_mode("Assignment mode off")
        channel = self._channel()
        waveforms = np.asarray(self.data.waveforms.get(channel, []), dtype=float)
        if waveforms.ndim != 2 or waveforms.shape[0] == 0:
            self.current_embedding = np.zeros((0, 0), dtype=np.float32)
            self.current_labels = np.zeros(0, dtype=np.int32)
            return
        progress = self._start_progress("Embedding", f"Computing embedding for {channel}...", 0) if update_status else None
        try:
            self.current_embedding = waveform_embedding(waveforms, self._config())
        except Exception as exc:
            self.status.setText("Embedding update failed")
            if update_status:
                QMessageBox.critical(self, "Embedding failed", str(exc))
            return
        finally:
            if progress is not None:
                _close_progress_dialog(progress)
        self.embedding_view_limits = None
        if self.current_labels.size != waveforms.shape[0]:
            self.current_labels = np.zeros(waveforms.shape[0], dtype=np.int32)
        self._save_current_channel()
        if update_status:
            self.status.setText(f"Embedding updated for {channel}")
        self._draw_all()

    def _save_current_channel(self):
        channel = self._channel()
        payload = self.data.sorting.setdefault(channel, {})
        payload["method"] = "manual_waveform_sorting"
        payload["waveform_cluster_labels"] = self.current_labels.astype(np.int32)
        payload["labels"] = self.current_labels.astype(np.int32)
        payload["embedding"] = self.current_embedding.astype(np.float32)
        payload["cluster_count"] = int(np.unique(self.current_labels).size) if self.current_labels.size else 0
        self.dirty = True
        self._refresh_cluster_filter()

    def _refresh_cluster_filter(self):
        previous = self.cluster_filter.currentData()
        if previous is None:
            previous = "all"
        labels = sorted(int(label) for label in np.unique(self.current_labels)) if self.current_labels.size else []
        self.cluster_filter.blockSignals(True)
        self.cluster_filter.clear()
        self.cluster_filter.addItem("All clusters", "all")
        for label in labels:
            text = "noise" if label == -1 else f"cluster {label}"
            self.cluster_filter.addItem(text, label)
        index = self.cluster_filter.findData(previous)
        self.cluster_filter.setCurrentIndex(index if index >= 0 else 0)
        self.cluster_filter.blockSignals(False)

    def _active_cluster_filter(self):
        value = self.cluster_filter.currentData()
        return None if value in (None, "all") else int(value)

    def _visible_label_mask(self, labels: np.ndarray) -> np.ndarray:
        mask = np.ones(labels.shape[0], dtype=bool)
        if self.hide_noise.isChecked():
            mask &= labels != -1
        active_cluster = self._active_cluster_filter()
        if active_cluster is not None:
            mask &= labels == active_cluster
        return mask

    def _start_lasso(self):
        self._toggle_lasso_mode("cluster", self.cluster_id.value())

    def _start_noise_lasso(self):
        self._toggle_lasso_mode("noise", -1)

    def _toggle_lasso_mode(self, mode: str, label: int):
        if self.lasso_mode == mode:
            self._stop_lasso_mode("Assignment mode off")
            return
        self.lasso_mode = mode
        self.pending_assignment_label = label
        self._refresh_lasso_button_states()
        message = "Draw regions to mark noise; right-click to exit" if mode == "noise" else (
            f"Draw regions to assign cluster {label}; right-click to exit"
        )
        self._begin_lasso(message)

    def _begin_lasso(self, message: str):
        if self.current_embedding.ndim != 2 or self.current_embedding.shape[0] == 0:
            QMessageBox.information(self, "Sorting", "Compute an embedding before manual cluster assignment.")
            return
        if self.lasso is not None:
            self.lasso.disconnect_events()
        self.status.setText(message)
        self.lasso = LassoSelector(self.embedding_ax, self._finish_lasso)
        self.lasso.connect_event("button_press_event", self._lasso_button_press)

    def _lasso_button_press(self, event):
        if event.button == 3:
            self._stop_lasso_mode("Assignment mode off")

    def _stop_lasso_mode(self, message: str):
        if self.lasso is not None:
            self.lasso.disconnect_events()
            self.lasso = None
        self.lasso_mode = None
        self._refresh_lasso_button_states()
        self.status.setText(message)

    def _refresh_lasso_button_states(self):
        self.lasso_button.setChecked(self.lasso_mode == "cluster")
        self.noise_button.setChecked(self.lasso_mode == "noise")

    def _finish_lasso(self, vertices):
        if self.lasso is not None:
            self.lasso.disconnect_events()
            self.lasso = None
        points = self._embedding_xy()
        selected = np.zeros(points.shape[0], dtype=bool)
        finite_points = np.isfinite(points).all(axis=1)
        if np.any(finite_points):
            selected[finite_points] = MplPath(vertices).contains_points(points[finite_points])
        if self.current_labels.size == selected.size:
            selected &= self._visible_label_mask(self.current_labels)
        count = int(np.count_nonzero(selected))
        if count:
            self._push_undo()
            self.current_labels[selected] = self.pending_assignment_label
            self._save_current_channel()
            self._draw_all()
            self.undo_button.setEnabled(True)
        target = "noise" if self.pending_assignment_label == -1 else f"cluster {self.pending_assignment_label}"
        self.status.setText(f"Assigned {count} spikes to {target}")
        if self.lasso_mode is not None:
            mode = self.lasso_mode
            message = "Draw another region to mark noise; right-click to exit" if mode == "noise" else (
                f"Draw another region to assign cluster {self.pending_assignment_label}; right-click to exit"
            )
            self._begin_lasso(message)

    def _push_undo(self):
        channel = self._channel()
        self.undo_stack.setdefault(channel, []).append(self.current_labels.copy())
        if len(self.undo_stack[channel]) > 50:
            self.undo_stack[channel] = self.undo_stack[channel][-50:]

    def _undo_manual_assignment(self):
        channel = self._channel()
        stack = self.undo_stack.get(channel, [])
        if not stack:
            self.status.setText("No manual sorting step to undo")
            self.undo_button.setEnabled(False)
            return
        self.current_labels = stack.pop()
        self._save_current_channel()
        self._draw_all()
        self.undo_button.setEnabled(bool(stack))
        self.status.setText(f"Undid last manual sorting change for {channel}")

    def _embedding_xy(self):
        if self.current_embedding.ndim != 2 or self.current_embedding.shape[1] == 0:
            row_count = self.current_embedding.shape[0] if self.current_embedding.ndim == 2 else 0
            return np.zeros((row_count, 2), dtype=np.float32)
        if self.current_embedding.shape[1] == 1:
            return np.column_stack([self.current_embedding[:, 0], np.zeros(self.current_embedding.shape[0])])
        return self.current_embedding[:, :2]

    def _embedding_scroll_zoom(self, event):
        if self.embedding_ax is None or event.inaxes is not self.embedding_ax:
            return
        xlim = self.embedding_ax.get_xlim()
        ylim = self.embedding_ax.get_ylim()
        if not np.isfinite([*xlim, *ylim]).all():
            return
        xdata = event.xdata if event.xdata is not None else (xlim[0] + xlim[1]) / 2.0
        ydata = event.ydata if event.ydata is not None else (ylim[0] + ylim[1]) / 2.0
        scale = 0.8 if event.button == "up" else 1.25
        new_width = max((xlim[1] - xlim[0]) * scale, 1e-9)
        new_height = max((ylim[1] - ylim[0]) * scale, 1e-9)
        rel_x = (xlim[1] - xdata) / max(xlim[1] - xlim[0], 1e-9)
        rel_y = (ylim[1] - ydata) / max(ylim[1] - ylim[0], 1e-9)
        new_xlim = (xdata - new_width * (1.0 - rel_x), xdata + new_width * rel_x)
        new_ylim = (ydata - new_height * (1.0 - rel_y), ydata + new_height * rel_y)
        self.embedding_ax.set_xlim(new_xlim)
        self.embedding_ax.set_ylim(new_ylim)
        self.embedding_view_limits = (new_xlim, new_ylim)
        self.embedding_canvas.draw_idle()

    def _draw_all(self):
        self._draw_waveforms()
        self._draw_embedding()

    def _draw_waveforms(self):
        figure = self.waveform_canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        channel = self._channel()
        waveforms = np.asarray(self.data.waveforms.get(channel, []), dtype=float)
        labels = self.current_labels
        if waveforms.ndim != 2 or labels.size != waveforms.shape[0]:
            ax.text(0.5, 0.5, "No waveforms", ha="center", va="center")
        else:
            x = _waveform_time_axis(waveforms.shape[1], self.data.sr)
            color_map = _cluster_color_map(labels)
            visible_mask = self._visible_label_mask(labels)
            plotted = False
            for label, color in color_map.items():
                cluster_mask = (labels == label) & visible_mask
                if not np.any(cluster_mask):
                    continue
                plotted = True
                cluster_waveforms = waveforms[cluster_mask]
                draw = cluster_waveforms[_display_indices(cluster_waveforms.shape[0], 100)]
                for waveform in draw:
                    ax.plot(x, waveform, color=color, alpha=0.10, linewidth=0.7)
                legend_label = "noise" if label == -1 else f"cluster {label}"
                ax.plot(x, np.mean(cluster_waveforms, axis=0), color=color, linewidth=2.0, label=legend_label)
            ax.set_xlabel("Time (ms)" if self.data.sr else "Time (sample index)")
            ax.set_ylabel("Voltage (uV)")
            ax.set_title(f"{channel} waveforms")
            if plotted:
                ax.legend(loc="best", fontsize=8)
            else:
                ax.text(0.5, 0.5, "No spikes match the current cluster filter", ha="center", va="center")
        self.waveform_canvas.draw_idle()

    def _draw_embedding(self):
        figure = self.embedding_canvas.figure
        figure.clear()
        self.embedding_ax = figure.add_subplot(111)
        ax = self.embedding_ax
        if self.current_embedding.ndim != 2 or self.current_embedding.shape[0] == 0 or self.current_embedding.shape[1] == 0:
            ax.text(0.5, 0.5, "No embedding", ha="center", va="center")
        else:
            points = self._embedding_xy()
            labels = self.current_labels if self.current_labels.size == points.shape[0] else np.zeros(points.shape[0], dtype=int)
            visible_mask = self._visible_label_mask(labels) & np.isfinite(points).all(axis=1)
            visible_indices = np.flatnonzero(visible_mask)
            indices = visible_indices[_display_indices(visible_indices.size, 8000)] if visible_indices.size else np.array([], dtype=int)
            color_map = _cluster_color_map(labels)
            plotted = False
            for label, color in color_map.items():
                cluster_indices = indices[labels[indices] == label]
                if cluster_indices.size == 0:
                    continue
                plotted = True
                legend_label = "noise" if label == -1 else f"cluster {label}"
                ax.scatter(
                    points[cluster_indices, 0],
                    points[cluster_indices, 1],
                    color=color,
                    s=10,
                    alpha=0.8,
                    label=legend_label,
                )
            if plotted:
                ax.legend(loc="best", fontsize=8)
            else:
                ax.text(0.5, 0.5, "No spikes match the current cluster filter", ha="center", va="center")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_title(f"{self._channel()} reduction space")
            if self.embedding_view_limits is not None:
                ax.set_xlim(self.embedding_view_limits[0])
                ax.set_ylim(self.embedding_view_limits[1])
        self.embedding_canvas.draw_idle()


class TemporalCouplingWindow(QDialog):
    def __init__(self, data: UnifiedMEAData, parent=None):
        super().__init__(parent)
        self.data = data
        self.setWindowTitle("Temporal Coupling")
        self.resize(1280, 820)
        self.units = _unit_spike_trains_from_unified(data, include_noise=False)
        self.all_results = []
        self.results = []

        self.window_ms = QSpinBox()
        self.window_ms.setRange(5, 1000)
        self.window_ms.setValue(100)
        self.window_ms.setSuffix(" ms")
        self.bin_ms = QDoubleSpinBox()
        self.bin_ms.setRange(0.1, 50.0)
        self.bin_ms.setDecimals(1)
        self.bin_ms.setValue(1.0)
        self.bin_ms.setSuffix(" ms")
        self.min_spikes = QSpinBox()
        self.min_spikes.setRange(1, 10000)
        self.min_spikes.setValue(5)
        self.max_pairs = QSpinBox()
        self.max_pairs.setRange(1, 500)
        self.max_pairs.setValue(80)
        self.max_pairs.valueChanged.connect(self._resort_results)

        self.sort_by = QComboBox()
        for label, key in [
            ("Strength", "strength"),
            ("Match", "matched_ratio"),
            ("Lag", "peak_lag_ms"),
            ("Peak", "peak_count"),
            ("Z", "z_score"),
            ("Lag SD", "lag_std_ms"),
            ("Reference", "reference_id"),
            ("Target", "target_id"),
        ]:
            self.sort_by.addItem(label, key)
        self.sort_by.currentIndexChanged.connect(self._resort_results)
        self.sort_order = QComboBox()
        self.sort_order.addItem("High to low", "desc")
        self.sort_order.addItem("Low to high", "asc")
        self.sort_order.currentIndexChanged.connect(self._resort_results)

        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.setObjectName("PrimaryButton")
        self.analyze_button.clicked.connect(self._analyze)
        self.status = QLabel()
        self.status.setObjectName("MutedText")

        self.pair_table = QTableWidget(0, 7)
        self.pair_table.setHorizontalHeaderLabels(
            ["Reference", "Target", "Lag ms", "Peak", "Z", "Lag SD ms", "Match"]
        )
        self.pair_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.pair_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.pair_table.itemSelectionChanged.connect(self._draw_selected_pair)
        self.pair_table.horizontalHeader().sectionClicked.connect(self._sort_by_table_header)

        self.correlogram_canvas = FigureCanvas(Figure(figsize=(6, 4), tight_layout=True))
        self.aligned_canvas = FigureCanvas(Figure(figsize=(6, 4), tight_layout=True))

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Window"))
        controls.addWidget(self.window_ms)
        controls.addWidget(QLabel("Bin"))
        controls.addWidget(self.bin_ms)
        controls.addWidget(QLabel("Min spikes"))
        controls.addWidget(self.min_spikes)
        controls.addWidget(QLabel("Max pairs"))
        controls.addWidget(self.max_pairs)
        controls.addWidget(QLabel("Sort by"))
        controls.addWidget(self.sort_by)
        controls.addWidget(self.sort_order)
        controls.addWidget(self.analyze_button)
        controls.addWidget(self.status, 1)

        plots = QVBoxLayout()
        plots.addWidget(QLabel("Cross-correlogram"))
        plots.addWidget(self.correlogram_canvas, 1)
        plots.addWidget(QLabel("Reference-aligned target raster"))
        plots.addWidget(self.aligned_canvas, 1)

        body = QHBoxLayout()
        body.addWidget(self.pair_table, 1)
        body.addLayout(plots, 2)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addLayout(body, 1)

        _fix_spinbox_hit_targets(self)
        self._analyze()
        self.showMaximized()

    def _analyze(self):
        if len(self.units) < 2:
            self.results = []
            self.pair_table.setRowCount(0)
            self.status.setText("Need at least two sorted non-noise units")
            self._draw_empty("Need at least two sorted non-noise units")
            return

        self.status.setText("Analyzing unit timing...")
        QApplication.processEvents()
        self.all_results = _temporal_coupling_pairs(
            self.units,
            window_ms=self.window_ms.value(),
            bin_ms=self.bin_ms.value(),
            min_spikes=self.min_spikes.value(),
        )
        self._sort_results()
        self._populate_table()
        self.status.setText(f"{len(self.units)} units, {len(self.results)} directed pairs")
        if self.results:
            self.pair_table.selectRow(0)
        else:
            self._draw_empty("No pair passed the current analysis settings")

    def _sort_results(self):
        key = self.sort_by.currentData() or "strength"
        reverse = self.sort_order.currentData() != "asc"

        def sort_value(result):
            value = result.get(key)
            if isinstance(value, str):
                return value.lower()
            if value is None:
                return -np.inf if reverse else np.inf
            try:
                number = float(value)
            except (TypeError, ValueError):
                return str(value).lower()
            return number if np.isfinite(number) else (-np.inf if reverse else np.inf)

        sorted_results = sorted(self.all_results, key=sort_value, reverse=reverse)
        self.results = sorted_results[: self.max_pairs.value()]

    def _resort_results(self):
        if not self.all_results:
            return
        self._sort_results()
        self._populate_table()
        self.pair_table.selectRow(0)

    def _sort_by_table_header(self, column: int):
        header_to_key = {
            0: "reference_id",
            1: "target_id",
            2: "peak_lag_ms",
            3: "peak_count",
            4: "z_score",
            5: "lag_std_ms",
            6: "matched_ratio",
        }
        key = header_to_key.get(column)
        if key is None:
            return
        index = self.sort_by.findData(key)
        if index >= 0:
            self.sort_by.setCurrentIndex(index)

    def _populate_table(self):
        self.pair_table.setRowCount(len(self.results))
        for row, result in enumerate(self.results):
            values = [
                result["reference_id"],
                result["target_id"],
                f"{result['peak_lag_ms']:.3f}",
                str(result["peak_count"]),
                f"{result['z_score']:.2f}",
                f"{result['lag_std_ms']:.3f}",
                f"{result['matched_ratio']:.3f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, row)
                self.pair_table.setItem(row, col, item)
        self.pair_table.resizeColumnsToContents()

    def _selected_result(self):
        items = self.pair_table.selectedItems()
        if not items:
            return None
        row = items[0].row()
        if row < 0 or row >= len(self.results):
            return None
        return self.results[row]

    def _draw_selected_pair(self):
        result = self._selected_result()
        if result is None:
            return
        self._draw_correlogram(result)
        self._draw_aligned_raster(result)

    def _draw_empty(self, message: str):
        for canvas in [self.correlogram_canvas, self.aligned_canvas]:
            figure = canvas.figure
            figure.clear()
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, message, ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            canvas.draw_idle()

    def _draw_correlogram(self, result: dict):
        figure = self.correlogram_canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        edges = result["edges"] * 1000.0
        hist = result["hist"]
        widths = np.diff(edges)
        ax.bar(edges[:-1], hist, width=widths, align="edge", color="#2563eb", alpha=0.75)
        ax.axvline(0, color="#475569", linewidth=1.0)
        ax.axvline(result["peak_lag_ms"], color="#dc2626", linewidth=1.5, linestyle="--")
        ax.set_title(f"{result['reference_id']} -> {result['target_id']}")
        ax.set_xlabel("Target spike lag from reference (ms)")
        ax.set_ylabel("Count")
        ax.text(
            0.98,
            0.95,
            f"peak {result['peak_lag_ms']:.3f} ms\nz {result['z_score']:.2f}\nlag SD {result['lag_std_ms']:.3f} ms",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#d7deea", "alpha": 0.9},
        )
        self.correlogram_canvas.draw_idle()

    def _draw_aligned_raster(self, result: dict):
        figure = self.aligned_canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        window_s = self.window_ms.value() / 1000.0
        reference_spikes = np.asarray(result["reference"]["spikes"], dtype=float)
        target_spikes = np.asarray(result["target"]["spikes"], dtype=float)
        if reference_spikes.size > 220:
            reference_spikes = reference_spikes[_display_indices(reference_spikes.size, 220)]

        rows = []
        for row, spike_time in enumerate(reference_spikes):
            lo = int(np.searchsorted(target_spikes, spike_time - window_s, side="left"))
            hi = int(np.searchsorted(target_spikes, spike_time + window_s, side="right"))
            lags = (target_spikes[lo:hi] - spike_time) * 1000.0
            if lags.size:
                rows.append((row, lags))

        for row, lags in rows:
            ax.vlines(lags, row - 0.38, row + 0.38, color="#1d4ed8", linewidth=1.05, alpha=0.95)
        ax.axvline(0, color="#475569", linewidth=1.0)
        ax.axvline(result["peak_lag_ms"], color="#dc2626", linewidth=1.3, linestyle="--")
        ax.set_xlim(-self.window_ms.value(), self.window_ms.value())
        ax.set_ylim(-1, max(1, len(reference_spikes)))
        ax.set_title(f"{result['target_id']} spikes aligned to {result['reference_id']}")
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("Reference event")
        self.aligned_canvas.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MEA Pipeline Studio")
        self.resize(1120, 720)
        self.thread_pool = QThreadPool.globalInstance()
        self.config = PipelineConfig()
        self.input_path = ""
        self.data_kind = ""
        self.raw_data = None
        self.result = None
        self.channel_map = _default_maxwell_channel_map() or default_channel_map()
        self.child_windows = []
        self.pipeline_progress = None
        self.active_load_worker = None
        self.active_stimulus_worker = None
        self.stimulus_response_dialog = None
        self.stimulus_response_payload = None

        self._build_ui()
        self._build_menu()
        _fix_spinbox_hit_targets(self)

    def _build_ui(self):
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(18)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(280)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setSpacing(12)

        title = QLabel("MEA Pipeline\nStudio")
        title.setObjectName("AppTitle")
        title.setFont(QFont("Segoe UI", 22, QFont.Bold))
        side_layout.addWidget(title)

        self.file_label = QLabel("No data file selected")
        self.file_label.setWordWrap(True)
        self.file_label.setObjectName("MutedText")
        side_layout.addWidget(self.file_label)

        self.open_button = QPushButton("Open Data")
        self.open_button.clicked.connect(self.open_data)
        self.stimulus_response_button = QPushButton("Stimulus Response Analysis")
        self.stimulus_response_button.clicked.connect(self.open_stimulus_response_analysis)
        self.preview_button = QPushButton("Raw Data Raster")
        self.preview_button.clicked.connect(self.preview_raw)
        self.preview_button.setEnabled(False)
        self.save_spike_train_button = QPushButton("Save Spike Train")
        self.save_spike_train_button.clicked.connect(self.save_spike_train)
        self.save_spike_train_button.setEnabled(False)
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.open_settings)
        self.channel_map_button = QPushButton("Channel Map")
        self.channel_map_button.clicked.connect(self.open_channel_map)
        self.sorting_button = QPushButton("Sorting")
        self.sorting_button.clicked.connect(self.open_sorting)
        self.temporal_button = QPushButton("Temporal Coupling")
        self.temporal_button.clicked.connect(self.open_temporal_coupling)
        self.temporal_button.setEnabled(False)
        self.run_button = QPushButton("Run Full Pipeline")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self.run_pipeline)
        self.run_button.setEnabled(False)
        self.results_button = QPushButton("Open Results")
        self.results_button.clicked.connect(self.open_results)
        self.results_button.setEnabled(False)

        for button in [
            self.stimulus_response_button,
            self.open_button,
            self.preview_button,
            self.save_spike_train_button,
            self.settings_button,
            self.channel_map_button,
            self.sorting_button,
            self.temporal_button,
            self.run_button,
            self.results_button,
        ]:
            button.setMinimumHeight(40)
            side_layout.addWidget(button)

        side_layout.addStretch()

        content = QFrame()
        content.setObjectName("Content")
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(16)

        header = QLabel("Loaded data preview")
        header.setObjectName("Header")
        header.setFont(QFont("Segoe UI", 20, QFont.Bold))
        content_layout.addWidget(header)

        self.data_preview = QTextEdit()
        self.data_preview.setReadOnly(True)
        self.data_preview.setMinimumHeight(260)
        self.data_preview.setPlaceholderText("Open a data file to show a summary.")
        content_layout.addWidget(self.data_preview, 2)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Activity log")
        self.log.setMinimumHeight(150)
        content_layout.addWidget(self.log, 1)
        self._update_data_preview()

        layout.addWidget(sidebar)
        layout.addWidget(content, 1)
        self.setCentralWidget(root)

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("File")
        open_action = QAction("Open Data", self)
        open_action.triggered.connect(self.open_data)
        file_menu.addAction(open_action)
        self.save_spike_train_action = QAction("Save Spike Train", self)
        self.save_spike_train_action.triggered.connect(self.save_spike_train)
        self.save_spike_train_action.setEnabled(False)
        file_menu.addAction(self.save_spike_train_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = self.menuBar().addMenu("View")
        results_action = QAction("Results", self)
        results_action.triggered.connect(self.open_results)
        view_menu.addAction(results_action)

        tools_menu = self.menuBar().addMenu("Tools")
        channel_map_action = QAction("Channel Map", self)
        channel_map_action.triggered.connect(self.open_channel_map)
        tools_menu.addAction(channel_map_action)
        tools_menu.addSeparator()
        sorting_action = QAction("Sorting", self)
        sorting_action.triggered.connect(self.open_sorting)
        tools_menu.addAction(sorting_action)
        temporal_action = QAction("Temporal Coupling", self)
        temporal_action.triggered.connect(self.open_temporal_coupling)
        tools_menu.addAction(temporal_action)
        stimulus_action = QAction("Stimulus Response Analysis", self)
        stimulus_action.triggered.connect(self.open_stimulus_response_analysis)
        tools_menu.addAction(stimulus_action)

    def _start_progress(self, title: str, message: str, maximum: int = 0) -> QProgressDialog:
        return _create_progress_dialog(self, title, message, maximum)

    def _progress_step(self, dialog: QProgressDialog | None, message: str, value: int | None = None) -> None:
        _set_progress_dialog(dialog, message, value)

    def _finish_progress(self, dialog: QProgressDialog | None) -> None:
        _close_progress_dialog(dialog)

    def open_stimulus_response_analysis(self):
        if self.active_stimulus_worker is not None:
            QMessageBox.information(self, "Stimulus Response", "Stimulus response analysis is already running.")
            return
        dialog = self._stimulus_response_analysis_dialog()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _stimulus_response_analysis_dialog(self) -> StimulusResponseInputDialog:
        if self.stimulus_response_dialog is None:
            dialog = StimulusResponseInputDialog(self)
            dialog.setWindowModality(Qt.WindowModality.NonModal)
            dialog.accepted.connect(self._start_stimulus_response_from_dialog)
            dialog.open_raster_button.clicked.connect(self._open_cached_stimulus_raster)
            dialog.psth_button.clicked.connect(self._open_cached_stimulus_psth)
            dialog.activation_curve_button.clicked.connect(self._open_cached_stimulus_activation_curve)
            self.stimulus_response_dialog = dialog
        return self.stimulus_response_dialog

    def _start_stimulus_response_from_dialog(self):
        if self.active_stimulus_worker is not None:
            QMessageBox.information(self, "Stimulus Response", "Stimulus response analysis is already running.")
            return
        dialog = self._stimulus_response_analysis_dialog()
        paths, pre_ms, response_ms, artifact_ms = dialog.values()
        if not paths:
            QMessageBox.information(self, "Stimulus Response", "Add at least one stimulus response file or folder.")
            dialog.show()
            return
        self.pipeline_progress = self._start_progress("Stimulus response", "Starting stimulus response analysis...", 100)
        worker = StimulusResponseWorker(paths, pre_ms=pre_ms, response_ms=response_ms, artifact_ms=artifact_ms)
        self.pipeline_progress.canceled.connect(worker.cancel)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(lambda payload, worker=worker: self._stimulus_response_finished(payload, worker))
        worker.signals.failed.connect(lambda details, worker=worker: self._stimulus_response_failed(details, worker))
        worker.signals.canceled.connect(lambda message, worker=worker: self._stimulus_response_canceled(message, worker))
        self.active_stimulus_worker = worker
        self.thread_pool.start(worker)

    def _stimulus_response_finished(self, payload: dict, worker: StimulusResponseWorker):
        if worker._is_cancelled():
            self._stimulus_response_canceled("Stimulus response analysis cancelled", worker)
            return
        if self.active_stimulus_worker is worker:
            self.active_stimulus_worker = None
        self._finish_progress(self.pipeline_progress)
        self.pipeline_progress = None
        records = list(payload.get("records", []))
        errors = list(payload.get("errors", []))
        if not records:
            details = "\n".join(errors[:12]) if errors else "No stimulus response files could be analyzed."
            QMessageBox.warning(self, "Stimulus Response", details)
            self._log("Stimulus response analysis produced no usable files")
            self._return_to_stimulus_analysis_dialog()
            return
        self.stimulus_response_payload = payload
        self._stimulus_response_analysis_dialog().set_cached_payload(payload)
        self._open_stimulus_raster_window(payload)
        self._log(f"Stimulus response analysis: {len(records)} files, {len(errors)} skipped")

    def _stimulus_response_failed(self, details: str, worker: StimulusResponseWorker):
        if self.active_stimulus_worker is worker:
            self.active_stimulus_worker = None
        self._finish_progress(self.pipeline_progress)
        self.pipeline_progress = None
        self._log(details)
        QMessageBox.critical(self, "Stimulus Response failed", details.splitlines()[-1] if details else "Unknown error")
        self._return_to_stimulus_analysis_dialog()

    def _stimulus_response_canceled(self, message: str, worker: StimulusResponseWorker):
        if self.active_stimulus_worker is worker:
            self.active_stimulus_worker = None
        self._finish_progress(self.pipeline_progress)
        self.pipeline_progress = None
        self._log(message or "Stimulus response analysis cancelled")
        self._return_to_stimulus_analysis_dialog()

    def _open_cached_stimulus_raster(self):
        if not self.stimulus_response_payload:
            QMessageBox.information(self, "Stimulus Response", "Run analysis before opening raster results.")
            return
        self._open_stimulus_raster_window(self.stimulus_response_payload)

    def _open_cached_stimulus_psth(self):
        if not self.stimulus_response_payload:
            QMessageBox.information(self, "Stimulus Response", "Run analysis before opening PSTH.")
            return
        self._open_stimulus_psth_window(self.stimulus_response_payload)

    def _open_cached_stimulus_activation_curve(self):
        if not self.stimulus_response_payload:
            QMessageBox.information(self, "Stimulus Response", "Run analysis before opening activation curve.")
            return
        self._open_stimulus_activation_curve_window(self.stimulus_response_payload)

    def _open_stimulus_raster_window(self, payload: dict):
        window = StimulusResponseWindow(payload, self, channel_map=self.channel_map)
        self._show_stimulus_result_window(window)

    def _open_stimulus_psth_window(self, payload: dict):
        window = StimulusPSTHWindow(payload, self)
        self._show_stimulus_result_window(window)

    def _open_stimulus_activation_curve_window(self, payload: dict):
        window = StimulusActivationCurveWindow(payload, self)
        self._show_stimulus_result_window(window)

    def _show_stimulus_result_window(self, window: QDialog):
        dialog = self._stimulus_response_analysis_dialog()
        dialog.hide()
        window.finished.connect(lambda _result: self._return_to_stimulus_analysis_dialog())
        self._show_child(window)

    def _return_to_stimulus_analysis_dialog(self):
        if self.stimulus_response_dialog is None:
            return
        self.stimulus_response_dialog.show()
        self.stimulus_response_dialog.raise_()
        self.stimulus_response_dialog.activateWindow()

    def open_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open MEA data",
            "data",
            "Data files (*.npy *.npz *.csv *.txt *.tsv *.nev *.spk *.h5 *.hdf5);;Array files (*.npy *.npz *.csv *.txt *.tsv);;Blackrock NEV (*.nev);;Axion SPK (*.spk);;Maxwell H5 (*.h5 *.hdf5);;All files (*)",
        )
        if not path:
            return
        selected_wells = None
        try:
            suffix = Path(path).suffix.lower()
            if suffix == ".spk":
                wells = list_axion_spk_wells(path)
                if wells:
                    selected_wells, accepted = self._select_wells(wells)
                    if not accepted:
                        return
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        self.preview_button.setEnabled(False)
        self.save_spike_train_button.setEnabled(False)
        self.save_spike_train_action.setEnabled(False)
        self.run_button.setEnabled(False)
        self.results_button.setEnabled(False)
        self.temporal_button.setEnabled(False)
        self.pipeline_progress = self._start_progress("Loading data", "Starting data load...", 100)
        worker = DataLoadWorker(path, selected_wells=selected_wells)
        self.pipeline_progress.canceled.connect(worker.cancel)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(lambda payload, worker=worker: self._data_load_finished(payload, worker))
        worker.signals.failed.connect(lambda details, worker=worker: self._data_load_failed(details, worker))
        worker.signals.canceled.connect(lambda message, worker=worker: self._data_load_canceled(message, worker))
        self.active_load_worker = worker
        self.thread_pool.start(worker)

    def _data_load_finished(self, payload: dict, worker: DataLoadWorker):
        if worker._is_cancelled():
            self._data_load_canceled("Data loading cancelled", worker)
            return
        if self.active_load_worker is worker:
            self.active_load_worker = None
        self._finish_progress(self.pipeline_progress)
        self.pipeline_progress = None
        path = str(payload.get("path", ""))
        self.raw_data = payload.get("raw_data")
        self.data_kind = str(payload.get("data_kind", ""))
        if isinstance(self.raw_data, UnifiedMEAData):
            filtered = self._maybe_select_loaded_well(self.raw_data)
            if filtered is None:
                self._update_data_preview()
                return
            self.raw_data = filtered
        self._apply_source_channel_map()

        self.input_path = path
        self.result = None
        self.file_label.setText(Path(path).name)
        self.preview_button.setEnabled(True)
        can_save_spike_train = isinstance(self.raw_data, UnifiedMEAData)
        self.save_spike_train_button.setEnabled(can_save_spike_train)
        self.save_spike_train_action.setEnabled(can_save_spike_train)
        self.run_button.setEnabled(self.data_kind == "array")
        self.results_button.setEnabled(self.data_kind == "nev")
        self.temporal_button.setEnabled(self.data_kind == "nev")
        self._log(f"Loaded {path}")
        if self.data_kind == "nev":
            spike_count = sum(values.size for values in self.raw_data.spikes.values())
            source = self.raw_data.meta.get("source", "spike file") if isinstance(self.raw_data.meta, dict) else "spike file"
            selected = self.raw_data.meta.get("selected_wells", []) if isinstance(self.raw_data.meta, dict) else []
            self._log(f"Spike source: {source}")
            if selected:
                self._log(f"Selected wells: {', '.join(selected)}")
            self._log(f"Spike channels: {len(self.raw_data.spikes)}")
            self._log(f"Spikes: {spike_count}")
            self._log("Spike-event files can be previewed and opened as results")
        else:
            self._log(f"Raw data shape: {np.asarray(self.raw_data).shape}")
        self._update_data_preview()
        self._validate_default_channel_map()

    def _data_load_failed(self, details: str, worker: DataLoadWorker):
        if self.active_load_worker is worker:
            self.active_load_worker = None
        self._finish_progress(self.pipeline_progress)
        self.pipeline_progress = None
        self.preview_button.setEnabled(self.raw_data is not None)
        can_save_spike_train = isinstance(self.raw_data, UnifiedMEAData)
        self.save_spike_train_button.setEnabled(can_save_spike_train)
        self.save_spike_train_action.setEnabled(can_save_spike_train)
        self.run_button.setEnabled(self.data_kind == "array")
        self.results_button.setEnabled(self.data_kind == "nev")
        self.temporal_button.setEnabled(self.data_kind == "nev")
        QMessageBox.critical(self, "Load failed", details.splitlines()[-1])

    def _data_load_canceled(self, message: str, worker: DataLoadWorker):
        if self.active_load_worker is worker:
            self.active_load_worker = None
        self._finish_progress(self.pipeline_progress)
        self.pipeline_progress = None
        self.preview_button.setEnabled(self.raw_data is not None)
        can_save_spike_train = isinstance(self.raw_data, UnifiedMEAData)
        self.save_spike_train_button.setEnabled(can_save_spike_train)
        self.save_spike_train_action.setEnabled(can_save_spike_train)
        self.run_button.setEnabled(self.data_kind == "array")
        self.results_button.setEnabled(self.data_kind == "nev")
        self.temporal_button.setEnabled(self.data_kind == "nev")
        self._log(message or "Data loading cancelled")

    def _select_wells(self, wells):
        ordered = sorted([str(well) for well in wells], key=_well_sort_key)
        if not ordered:
            return None, True
        choices = ordered + ["All wells"]
        choice, accepted = QInputDialog.getItem(
            self,
            "Select well",
            "Well for visualization and processing:",
            choices,
            0,
            False,
        )
        if not accepted:
            return None, False
        if choice == "All wells":
            return None, True
        return [choice], True

    def _maybe_select_loaded_well(self, data: UnifiedMEAData):
        wells = []
        if isinstance(data.meta, dict):
            wells = [str(well) for well in data.meta.get("wells", []) if str(well)]
        if len(wells) <= 1:
            return data
        selected_wells, accepted = self._select_wells(wells)
        if not accepted:
            return None
        return filter_unified_by_wells(data, selected_wells)

    def _apply_source_channel_map(self) -> None:
        if not isinstance(self.raw_data, UnifiedMEAData):
            return
        if not isinstance(self.raw_data.meta, dict):
            return
        source = self.raw_data.meta.get("source")
        if source == "axion_spk":
            axion_map = load_channel_map("axion_map")
            if axion_map is not None:
                self.channel_map = axion_map
            return
        if source == "maxwell_h5":
            maxwell_map = _maxwell_channel_map_from_unified(self.raw_data)
            if maxwell_map is not None:
                self.channel_map = maxwell_map
            return
        if source != "axion_spk":
            fallback = default_channel_map()
            if fallback is not None:
                self.channel_map = fallback
            return

    def _update_data_preview(self):
        self.data_preview.setPlainText(self._data_preview_text())

    def _data_preview_text(self) -> str:
        if self.raw_data is None:
            return "No data loaded.\n\nUse Open Data to load an array, NPZ, or Blackrock NEV file."

        lines = []
        if self.input_path:
            lines.append(f"File: {self.input_path}")
        lines.append(f"Kind: {self.data_kind or type(self.raw_data).__name__}")

        if isinstance(self.raw_data, UnifiedMEAData):
            data = self.raw_data
            channels = sorted(data.channels(), key=_channel_sort_key)
            spike_counts = {channel: int(np.asarray(data.spikes.get(channel, [])).size) for channel in channels}
            total_spikes = sum(spike_counts.values())
            _, max_time = data.time_range()
            waveform_channels = sorted(data.waveforms.keys(), key=_channel_sort_key)
            sorted_units = 0
            if isinstance(data.sorting, dict):
                for channel in channels:
                    labels = _sorting_labels_for_raster(data, channel, spike_counts.get(channel, 0))
                    if labels is not None:
                        sorted_units += len([label for label in np.unique(labels) if int(label) != -1])

            lines.extend(
                [
                    f"Sampling rate: {data.sr:g} Hz" if data.sr else "Sampling rate: n/a",
                    f"Duration: {max_time:.3f} s" if max_time else "Duration: n/a",
                    f"Channels: {len(channels)}",
                    f"Total spikes: {total_spikes}",
                    f"Waveform channels: {len(waveform_channels)}",
                    f"Sorted units: {sorted_units}",
                ]
            )
            if isinstance(data.meta, dict) and data.meta:
                source = data.meta.get("source")
                if source:
                    lines.append(f"Source: {source}")
                if source == "axion_spk":
                    wells = data.meta.get("wells") or []
                    lines.append(f"Wells: {', '.join(wells) if wells else 'n/a'}")
                    lines.append(f"Electrode sites: {data.meta.get('electrode_count', 'n/a')}")
                elif source == "maxwell_h5":
                    wells = data.meta.get("wells") or []
                    records = data.meta.get("event_records") or []
                    lines.append(f"Wells: {', '.join(wells) if wells else 'n/a'}")
                    lines.append(f"Stim/event packets: {data.stim_times.size}")
                    if data.stim_times.size:
                        lines.append(f"Stim range: {float(data.stim_times[0]):.3f} - {float(data.stim_times[-1]):.3f} s")
                    if "stim_artifact_removed_count" in data.meta:
                        lines.append(
                            f"Stim artifact removal: +/-{data.meta.get('stim_artifact_window_ms', 0):g} ms, "
                            f"{data.meta.get('stim_artifact_removed_count', 0)} spikes removed"
                        )
                    if records:
                        first_label = records[0].get("stim_label") or records[0].get("stim_message") or ""
                        if first_label:
                            lines.append(f"First stim: {first_label}")
            lines.append("")
            lines.append("Top channels by spike count:")
            for channel, count in sorted(spike_counts.items(), key=lambda item: item[1], reverse=True)[:12]:
                waveforms = data.waveforms.get(channel)
                waveform_text = ""
                if waveforms is not None:
                    waveform_text = f", waveforms {np.asarray(waveforms).shape}"
                lines.append(f"  {channel}: {count} spikes{waveform_text}")
            if len(channels) > 12:
                lines.append(f"  ... {len(channels) - 12} more channels")
            return "\n".join(lines)

        array = np.asarray(self.raw_data)
        lines.extend(
            [
                f"Shape: {array.shape}",
                f"Dtype: {array.dtype}",
                f"Dimensions: {array.ndim}",
                f"Values: {array.size}",
            ]
        )
        if array.ndim >= 2:
            lines.append(f"Channels/rows: {array.shape[0]}")
            lines.append(f"Samples/columns: {array.shape[1]}")
        if array.size:
            sample = array.reshape(-1)
            if sample.size > 100000:
                sample = sample[_display_indices(sample.size, 100000)]
            finite = np.isfinite(sample.astype(float, copy=False)) if np.issubdtype(sample.dtype, np.number) else np.array([])
            if finite.size:
                numeric = sample[finite].astype(float, copy=False)
                lines.append(f"Finite values in preview sample: {int(np.count_nonzero(finite))} / {sample.size}")
                if numeric.size:
                    lines.append(f"Preview min/max: {float(np.nanmin(numeric)):.6g} / {float(np.nanmax(numeric)):.6g}")
                    lines.append(f"Preview mean: {float(np.nanmean(numeric)):.6g}")
        return "\n".join(lines)

    def save_spike_train(self):
        if not isinstance(self.raw_data, UnifiedMEAData):
            QMessageBox.information(self, "Save Spike Train", "Load spike-event data before saving spike trains.")
            return

        default_name = "spike_train_data.npz"
        if self.input_path:
            default_name = f"{Path(self.input_path).stem}_spike_train.npz"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save spike train data",
            default_name,
            "Spike train data (*.npz);;All files (*)",
        )
        if not path:
            return
        if Path(path).suffix.lower() != ".npz":
            path = f"{path}.npz"

        progress = self._start_progress("Save Spike Train", "Saving spike train data...", 0)
        try:
            saved = save_spike_train_npz(self.raw_data, path)
        except Exception as exc:
            QMessageBox.critical(self, "Save Spike Train failed", str(exc))
            return
        finally:
            self._finish_progress(progress)

        self._log(f"Saved spike train data: {saved}")
        QMessageBox.information(
            self,
            "Save Spike Train",
            f"Saved spike train data without waveform arrays:\n{saved}",
        )

    def preview_raw(self):
        if self.raw_data is None:
            return
        progress = self._start_progress("Preparing raster", "Preparing raster data...", 4)
        try:
            if self.data_kind == "nev":
                self._progress_step(progress, "Building raster rows...", 1)
                raster_series, has_units = _raster_series_from_unified(self.raw_data, include_noise=False)
                self._progress_step(progress, "Attaching waveform data...", 2)
                waveform_series = _raster_waveforms_from_unified(self.raw_data, include_noise=False)
                self._progress_step(progress, "Creating raster window...", 3)
                window = SpikeRasterWindow(
                    "Raw Data Raster" if not has_units else "Raw Data Unit Raster",
                    raster_series,
                    waveform_series,
                    self.raw_data.sr,
                    self,
                    y_axis_label="Unit" if has_units else "Channel",
                    channel_map=self.channel_map,
                    stim_times=self.raw_data.stim_times,
                )
            else:
                self._progress_step(progress, "Rendering array preview...", 2)
                figure = Visualizer().plot_timeseries(self.raw_data)
                window = PlotWindow("Raw Data Raster", figure, self)
            self._progress_step(progress, "Opening raster window...", 4)
        except Exception as exc:
            QMessageBox.critical(self, "Preview failed", str(exc))
            return
        finally:
            self._finish_progress(progress)
        self._show_child(window)

    def open_settings(self):
        dialog = SettingsDialog(self.config, self)
        if dialog.exec() == QDialog.Accepted:
            self.config = dialog.get_config()
            self._log("Settings updated")

    def open_channel_map(self):
        available_channels = _available_channels_for_data(self.raw_data, self.data_kind)
        dialog = ChannelMapDialog(self.channel_map, available_channels, self)
        if dialog.exec() == QDialog.Accepted:
            self.channel_map = dialog.channel_map
            self._log(f"Channel map selected: {self.channel_map.name}")
            self._validate_default_channel_map()

    def open_sorting(self):
        if self.data_kind != "nev" or not isinstance(self.raw_data, UnifiedMEAData):
            QMessageBox.information(self, "Sorting", "Sorting currently requires loaded spike waveforms.")
            return
        if not self.raw_data.waveforms:
            QMessageBox.warning(self, "Sorting", "The loaded NEV file does not contain spike waveforms.")
            return
        progress = self._start_progress("Opening sorting", "Preparing sorting workspace...", 0)
        try:
            window = SortingWorkspaceWindow(self.raw_data, self)
        except Exception as exc:
            QMessageBox.critical(self, "Sorting failed", str(exc))
            return
        finally:
            self._finish_progress(progress)
        self._show_child(window)

    def open_temporal_coupling(self):
        if self.data_kind != "nev" or not isinstance(self.raw_data, UnifiedMEAData):
            QMessageBox.information(self, "Temporal Coupling", "Temporal coupling analysis requires loaded sorted spike data.")
            return
        progress = self._start_progress("Temporal coupling", "Collecting sorted units...", 3)
        units = _unit_spike_trains_from_unified(self.raw_data, include_noise=False)
        self._progress_step(progress, "Checking available units...", 1)
        if len(units) < 2:
            self._finish_progress(progress)
            QMessageBox.information(
                self,
                "Temporal Coupling",
                "Temporal coupling analysis requires at least two sorted non-noise units.",
            )
            return
        try:
            self._progress_step(progress, "Building temporal coupling window...", 2)
            window = TemporalCouplingWindow(self.raw_data, self)
            self._progress_step(progress, "Opening temporal coupling window...", 3)
        except Exception as exc:
            QMessageBox.critical(self, "Temporal Coupling failed", str(exc))
            return
        finally:
            self._finish_progress(progress)
        self._show_child(window)

    def run_pipeline(self):
        if not self.input_path or self.data_kind != "array":
            return
        self.run_button.setEnabled(False)
        self.results_button.setEnabled(False)
        self._log("Starting pipeline")
        self.pipeline_progress = self._start_progress("Running pipeline", "Starting pipeline...", 100)
        worker = PipelineWorker(self.input_path, self.config)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.failed.connect(self._on_failed)
        self.thread_pool.start(worker)

    def open_results(self):
        if self.data_kind == "nev" and isinstance(self.raw_data, UnifiedMEAData):
            progress = self._start_progress("Opening results", "Preparing result summary...", 5)
            try:
                window = NevResultsWindow(self.raw_data, self)
                self._progress_step(progress, "Building raster rows...", 1)
                raster_series, has_units = _raster_series_from_unified(self.raw_data)
                self._progress_step(progress, "Attaching waveform data...", 2)
                waveform_series = _raster_waveforms_from_unified(self.raw_data)
                self._progress_step(progress, "Creating raster window...", 3)
                source = self.raw_data.meta.get("source", "") if isinstance(self.raw_data.meta, dict) else ""
                prefix = "Axion SPK" if source == "axion_spk" else "NEV"
                raster = SpikeRasterWindow(
                    f"{prefix} Spike Raster" if not has_units else f"{prefix} Unit Raster",
                    raster_series,
                    waveform_series,
                    self.raw_data.sr,
                    self,
                    y_axis_label="Unit" if has_units else "Channel",
                    channel_map=self.channel_map,
                    stim_times=self.raw_data.stim_times,
                )
                self._progress_step(progress, "Opening result windows...", 5)
            except Exception as exc:
                QMessageBox.critical(self, "Results failed", str(exc))
                return
            finally:
                self._finish_progress(progress)
            self._show_child(window)
            self._show_child(raster)
            return

        if self.result is None:
            return
        progress = self._start_progress("Opening results", "Rendering pipeline results...", 4)
        try:
            window = ResultsWindow(self.result, self)
            self._progress_step(progress, "Rendering correlation plot...", 2)
            heatmap = PlotWindow("Channel Correlation", Visualizer().plot_results(self.result.analysis), self)
            self._progress_step(progress, "Rendering spike raster...", 3)
            raster = PlotWindow("Spike Raster", Visualizer().plot_raster(self.result.spikes), self)
            self._progress_step(progress, "Opening result windows...", 4)
        except Exception as exc:
            QMessageBox.critical(self, "Results failed", str(exc))
            return
        finally:
            self._finish_progress(progress)
        self._show_child(window)
        self._show_child(heatmap)
        self._show_child(raster)

    def _on_progress(self, value: int, message: str):
        self._log(f"{value}% - {message}")
        if self.pipeline_progress is not None:
            self._progress_step(self.pipeline_progress, message, max(0, min(100, int(value))))

    def _on_finished(self, result: PipelineResult):
        self._finish_progress(self.pipeline_progress)
        self.pipeline_progress = None
        self.result = result
        self.run_button.setEnabled(True)
        self.results_button.setEnabled(True)
        self._log(f"Completed. Output: {result.output_path}")
        self.open_results()

    def _on_failed(self, details: str):
        self._finish_progress(self.pipeline_progress)
        self.pipeline_progress = None
        self.run_button.setEnabled(True)
        self._log(details)
        QMessageBox.critical(self, "Pipeline failed", details.splitlines()[-1])

    def _validate_default_channel_map(self):
        if self.channel_map is None or self.raw_data is None:
            return
        channels = _available_channels_for_data(self.raw_data, self.data_kind)
        report = validate_channel_map(self.channel_map, channels)
        self._log(
            "Channel map validation: "
            f"{report['mapped_count']} mapped, "
            f"{len(report['unmapped_channels'])} loaded channels unmapped, "
            f"{len(report['unknown_channels'])} unknown assignments"
        )
        if report["duplicates"]:
            self._log(f"Channel map duplicates: {', '.join(report['duplicates'].keys())}")

    def _log(self, message: str):
        self.log.append(message)

    def _show_child(self, window: QDialog):
        self.child_windows.append(window)
        window.finished.connect(lambda _: self._forget_child(window))
        window.show()

    def _forget_child(self, window: QDialog):
        if window in self.child_windows:
            self.child_windows.remove(window)


def apply_theme(app: QApplication) -> None:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#f6f7fb"))
    palette.setColor(QPalette.WindowText, QColor("#172033"))
    palette.setColor(QPalette.Base, QColor("#ffffff"))
    palette.setColor(QPalette.AlternateBase, QColor("#eef2f7"))
    palette.setColor(QPalette.Text, QColor("#172033"))
    palette.setColor(QPalette.Button, QColor("#ffffff"))
    palette.setColor(QPalette.ButtonText, QColor("#172033"))
    palette.setColor(QPalette.Highlight, QColor("#2563eb"))
    app.setPalette(palette)
    app.setStyleSheet(
        """
        QMainWindow, QDialog { background: #f6f7fb; }
        QMenuBar { background: #ffffff; border-bottom: 1px solid #e3e8f2; }
        #Sidebar, #Content {
            background: #ffffff;
            border: 1px solid #e3e8f2;
            border-radius: 8px;
        }
        #Sidebar { padding: 8px; }
        #AppTitle { color: #13213a; }
        #Header { color: #13213a; }
        #CardTitle { font-size: 16px; font-weight: 700; color: #13213a; }
        #MutedText { color: #667085; }
        QPushButton {
            background: #ffffff;
            border: 1px solid #cfd8e6;
            border-radius: 6px;
            padding: 8px 12px;
            color: #172033;
            font-weight: 600;
        }
        QPushButton:hover { background: #f0f5ff; border-color: #9bbcf5; }
        QPushButton:disabled { color: #98a2b3; background: #f2f4f7; }
        #PrimaryButton {
            background: #2563eb;
            color: #ffffff;
            border-color: #2563eb;
        }
        #PrimaryButton:hover { background: #1d4ed8; }
        QTextEdit, QLineEdit, QComboBox {
            border: 1px solid #cfd8e6;
            border-radius: 6px;
            background: #ffffff;
            padding: 6px;
        }
        QSpinBox, QDoubleSpinBox {
            border: 1px solid #cfd8e6;
            border-radius: 6px;
            background: #ffffff;
            padding-left: 6px;
            padding-right: 2px;
            min-height: 30px;
        }
        QSpinBox::up-button, QDoubleSpinBox::up-button {
            subcontrol-origin: border;
            subcontrol-position: top right;
            width: 20px;
            height: 14px;
            border-left: 1px solid #cfd8e6;
        }
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            subcontrol-origin: border;
            subcontrol-position: bottom right;
            width: 20px;
            height: 14px;
            border-left: 1px solid #cfd8e6;
        }
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow,
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
            width: 7px;
            height: 7px;
        }
        """
    )


def main() -> int:
    app = QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
