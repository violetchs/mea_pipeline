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
    from PySide6.QtCore import QItemSelectionModel, QLineF, QPointF, QObject, QRunnable, QRectF, Qt, QThreadPool, QTimer, Signal, Slot
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
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QProgressDialog,
        QPushButton,
        QScrollArea,
        QScrollBar,
        QSpacerItem,
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
from sklearn.decomposition import FactorAnalysis as SkFactorAnalysis
from sklearn.decomposition import PCA as SkPCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

try:
    from ..analysis import (
        assign_feature_clusters,
        compute_similarity_matrix,
        create_generic_analysis_figure,
        fit_pivae_latent_states,
        hierarchical_order_and_groups,
        normalize_feature_matrix,
        reduce_feature_matrix,
        run_generic_matrix_analysis,
    )
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
    from . import visual_stimulus_package_builder as stimulus_builder
    from ..mea_io import (
        MEAReader,
        MEAWriter,
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
    from analysis import (
        assign_feature_clusters,
        compute_similarity_matrix,
        create_generic_analysis_figure,
        fit_pivae_latent_states,
        hierarchical_order_and_groups,
        normalize_feature_matrix,
        reduce_feature_matrix,
        run_generic_matrix_analysis,
    )
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
    from gui import visual_stimulus_package_builder as stimulus_builder
    from mea_io import (
        MEAReader,
        MEAWriter,
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


def _enable_standard_window_controls(window) -> None:
    if window is None:
        return
    window.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
    window.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
    window.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)


def _set_balanced_normal_geometry(window, width_fraction: float = 0.82, height_fraction: float = 0.82) -> None:
    if window is None:
        return
    screen = window.screen() or QApplication.primaryScreen()
    if screen is None:
        return
    available = screen.availableGeometry()
    current_size = window.size()
    width = max(1, int(current_size.width()))
    height = max(1, int(current_size.height()))
    if width < 320 or height < 240:
        hint = window.sizeHint()
        width = max(width, int(hint.width()), 960)
        height = max(height, int(hint.height()), 640)
    aspect = max(0.8, min(2.4, float(width) / max(1.0, float(height))))
    max_width = max(640, int(available.width() * float(width_fraction)))
    max_height = max(480, int(available.height() * float(height_fraction)))
    target_width = min(max_width, max(640, width))
    target_height = int(round(target_width / aspect))
    if target_height > max_height:
        target_height = max_height
        target_width = int(round(target_height * aspect))
    target_width = max(640, min(max_width, target_width))
    target_height = max(480, min(max_height, target_height))
    geometry = available
    geometry.setWidth(int(target_width))
    geometry.setHeight(int(target_height))
    geometry.moveCenter(available.center())
    window.setGeometry(geometry)
    window._balanced_normal_geometry = geometry


class AppDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        _enable_standard_window_controls(self)
        self.setSizeGripEnabled(True)

    def showMaximized(self):
        if not self.isMaximized():
            _set_balanced_normal_geometry(self)
        super().showMaximized()

    def showNormal(self):
        super().showNormal()
        geometry = getattr(self, "_balanced_normal_geometry", None)
        if geometry is not None:
            self.setGeometry(geometry)


def _progress_enabled_for_widget(widget: QWidget | None) -> bool:
    if widget is None or not widget.isVisible():
        return False
    return os.environ.get("QT_QPA_PLATFORM", "").lower() != "offscreen"


def _use_non_blocking_messages() -> bool:
    return os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen"


def _show_message_dialog(parent: QWidget | None, title: str, text: str, level: str = "info"):
    if not _use_non_blocking_messages():
        if level == "warning":
            return QMessageBox.warning(parent, title, text)
        if level == "critical":
            return QMessageBox.critical(parent, title, text)
        return QMessageBox.information(parent, title, text)

    dialog = QDialog(parent)
    _enable_standard_window_controls(dialog)
    dialog.setWindowTitle(title)
    dialog.setAttribute(Qt.WA_DeleteOnClose, True)
    dialog.resize(520, 220)
    layout = QVBoxLayout(dialog)
    label = QLabel(text)
    label.setWordWrap(True)
    if level == "critical":
        label.setStyleSheet("color: #b91c1c;")
    elif level == "warning":
        label.setStyleSheet("color: #9a3412;")
    layout.addWidget(label, 1)
    close_button = QPushButton("Close")
    close_button.clicked.connect(dialog.close)
    layout.addWidget(close_button)
    dialog.show()
    return dialog


def _show_info_message(parent: QWidget | None, title: str, text: str):
    return _show_message_dialog(parent, title, text, "info")


def _show_warning_message(parent: QWidget | None, title: str, text: str):
    return _show_message_dialog(parent, title, text, "warning")


def _show_error_message(parent: QWidget | None, title: str, text: str):
    return _show_message_dialog(parent, title, text, "critical")


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


def _axion_electrode_slot(
    electrode: object,
    payload: dict[str, object] | None = None,
) -> tuple[int | None, int | None, int | None, str]:
    payload = payload if isinstance(payload, dict) else {}
    row = payload.get("electrode_row")
    col = payload.get("electrode_col")
    try:
        row_int = int(row)
        col_int = int(col)
    except (TypeError, ValueError):
        row_int = None
        col_int = None

    electrode_text = str(payload.get("electrode") or electrode or "").strip()
    if (row_int is None or col_int is None) and electrode_text:
        match = re.fullmatch(r"r(\d+)c(\d+)", electrode_text, flags=re.IGNORECASE)
        if match:
            row_int = int(match.group(1))
            col_int = int(match.group(2))

    slot = None
    if row_int is not None and col_int is not None and 1 <= row_int <= 8 and 1 <= col_int <= 8:
        slot = (row_int - 1) * 8 + (col_int - 1)
    return row_int, col_int, slot, electrode_text


def _axion_well_grid_position(well: str, fallback_index: int = 0) -> tuple[int, int]:
    text = str(well or "").strip().upper()
    match = re.fullmatch(r"([A-Z]+)(\d+)", text)
    if match:
        row_label = match.group(1)
        try:
            col_number = int(match.group(2))
        except ValueError:
            col_number = 1
        row_index = 0
        for char in row_label:
            row_index = row_index * 26 + (ord(char) - ord("A") + 1)
        row_index = max(0, row_index - 1)
        return row_index, max(0, col_number - 1)
    return fallback_index // 3, fallback_index % 3


def _default_axion_channel_map(data: UnifiedMEAData | None = None) -> ChannelMap:
    raw_map = data.meta.get("channel_map", {}) if isinstance(getattr(data, "meta", None), dict) else {}
    raw_map = raw_map if isinstance(raw_map, dict) else {}
    wells = []
    if isinstance(getattr(data, "meta", None), dict):
        wells = [str(well) for well in data.meta.get("wells", []) if str(well).strip()]
    ordered_wells = sorted({well.strip() for well in wells if well.strip()}, key=_well_sort_key)
    default_wells = ["A1", "A2", "A3", "B1", "B2", "B3"]
    for well in default_wells:
        if len(ordered_wells) >= 6:
            break
        if well not in ordered_wells:
            ordered_wells.append(well)
    if not ordered_wells:
        ordered_wells = list(default_wells)

    rows = max(6, len(ordered_wells))
    cols = 64
    routed_lookup: dict[tuple[str, int], tuple[str, dict[str, object]]] = {}
    for channel_name, payload in raw_map.items():
        if not isinstance(payload, dict):
            continue
        well = str(payload.get("well") or "").strip()
        if not well and "_" in str(channel_name):
            well = str(channel_name).split("_", 1)[0].strip()
        if not well:
            continue
        row_int, col_int, slot, electrode_text = _axion_electrode_slot(channel_name, payload)
        if slot is None:
            continue
        routed_payload = dict(payload)
        if not routed_payload.get("electrode"):
            routed_payload["electrode"] = electrode_text or f"r{(slot // 8) + 1}c{(slot % 8) + 1}"
        if row_int is not None:
            routed_payload["electrode_row"] = row_int
        if col_int is not None:
            routed_payload["electrode_col"] = col_int
        routed_lookup[(well, slot)] = (str(channel_name), routed_payload)
        if well not in ordered_wells:
            ordered_wells.append(well)

    electrodes = {}
    electrode_pitch_um = 32.0
    well_gap_x_um = electrode_pitch_um * 2.0
    well_gap_y_um = electrode_pitch_um * 2.5
    for well_index, well in enumerate(ordered_wells):
        well_row, well_col = _axion_well_grid_position(well, well_index)
        block_origin_x = well_col * (8.0 * electrode_pitch_um + well_gap_x_um)
        block_origin_y = well_row * (8.0 * electrode_pitch_um + well_gap_y_um)
        for slot in range(cols):
            electrode_row = slot // 8 + 1
            electrode_col = slot % 8 + 1
            electrode_text = f"r{electrode_row}c{electrode_col}"
            electrode_key = f"{well}_slot{slot + 1:02d}"
            channel_name = ""
            routed = False
            payload = routed_lookup.get((well, slot))
            entry = {
                "channel": "",
                "reference": False,
                "well": well,
                "electrode": electrode_text,
                "electrode_row": electrode_row,
                "electrode_col": electrode_col,
                "grid_row": well_row * 8 + (electrode_row - 1),
                "grid_col": slot,
                "well_grid_row": well_row,
                "well_grid_col": well_col,
                "x_um": block_origin_x + electrode_col * electrode_pitch_um,
                "y_um": block_origin_y + electrode_row * electrode_pitch_um,
                "aliases": [
                    electrode_key,
                    f"{well}_{electrode_text}",
                    electrode_text,
                    f"{well}_slot{slot + 1}",
                ],
                "routed": False,
            }
            if payload is not None:
                channel_name, routed_payload = payload
                entry["channel"] = channel_name
                entry["routed"] = True
                routed = True
                for field in (
                    "code",
                    "well_index",
                    "well_row",
                    "well_col",
                    "electrode_index",
                    "mea_electrode",
                    "source_channel",
                    "recording",
                    "data_group",
                ):
                    if field in routed_payload:
                        entry[field] = routed_payload[field]
            aliases = entry.get("aliases", [])
            if not isinstance(aliases, list):
                aliases = [aliases]
            seen = {str(alias) for alias in aliases}
            if channel_name and channel_name not in seen:
                aliases.append(channel_name)
                seen.add(channel_name)
            if routed and "_" in channel_name:
                suffix = channel_name.split("_", 1)[1]
                if suffix and suffix not in seen:
                    aliases.append(suffix)
                    seen.add(suffix)
            entry["aliases"] = aliases
            electrodes[electrode_key] = entry

    return ChannelMap(name="axion_map", rows=rows, cols=cols, electrodes=electrodes)


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


def _axion_well_from_channel_label(label: str) -> str:
    base = _base_channel_from_raster_label(label).strip()
    match = re.match(r"([A-Za-z]+\d+)_", base)
    return match.group(1) if match else ""


def _axion_raster_well_groups(spike_series) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for label, _times in spike_series:
        well = _axion_well_from_channel_label(str(label))
        if not well:
            continue
        groups.setdefault(well, []).append(str(label))
    return {well: labels for well, labels in sorted(groups.items(), key=lambda item: _well_sort_key(item[0])) if labels}


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


def _resolve_channel_map_electrode(value, lookup: dict, positions: dict) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    candidates = [text]
    try:
        numeric = int(float(text.lstrip("eE")))
        candidates.extend([str(numeric), f"e{numeric}", f"chan{numeric}"])
    except (TypeError, ValueError):
        pass
    for candidate in candidates:
        if candidate in positions:
            return candidate
        found = _position_for_channel(candidate, lookup)
        if found is not None:
            return str(found[2])
    return None


def draw_maxwell_channel_map(
    ax,
    channel_map,
    *,
    recording_electrodes=None,
    stimulation_electrodes=None,
    electrode_metrics=None,
    selected_electrode=None,
    title="Channel map",
):
    lookup, positions = _channel_map_positions(channel_map)
    figure = ax.figure
    figure.patch.set_facecolor("#ffffff")
    ax.clear()
    ax.set_facecolor("#ffffff")
    ax.set_title(str(title or "Channel map"))

    state = {
        "positions": positions,
        "lookup": lookup,
        "point_lookup": [],
        "recording": set(),
        "stimulation": set(),
        "metrics": {},
        "selected": None,
    }
    if not positions:
        ax.text(0.5, 0.5, "No channel map", ha="center", va="center", transform=ax.transAxes, color="#64748b")
        ax.set_axis_off()
        return state

    entries = [(electrode, float(x), float(y), payload) for electrode, (x, y, payload) in positions.items()]
    xs = [entry[1] for entry in entries]
    ys = [entry[2] for entry in entries]
    ax.scatter(xs, ys, s=5, color="#cbd5e1", alpha=0.72, marker="o", linewidths=0, zorder=2)

    for electrode, x, y, payload in entries:
        state["point_lookup"].append({"electrode": electrode, "x": x, "y": y, "payload": payload})

    metric_lookup = {}
    for key, value in dict(electrode_metrics or {}).items():
        resolved = _resolve_channel_map_electrode(key, lookup, positions)
        if resolved is not None:
            metric_lookup[resolved] = value
    state["metrics"] = metric_lookup

    recording = {
        resolved
        for resolved in (_resolve_channel_map_electrode(value, lookup, positions) for value in (recording_electrodes or []))
        if resolved is not None
    }
    stimulation = {
        resolved
        for resolved in (_resolve_channel_map_electrode(value, lookup, positions) for value in (stimulation_electrodes or []))
        if resolved is not None
    }
    state["recording"] = recording
    state["stimulation"] = stimulation

    if recording:
        rec_points = [(positions[electrode][0], positions[electrode][1]) for electrode in sorted(recording) if electrode in positions]
        if rec_points:
            rec = np.asarray(rec_points, dtype=float)
            ax.scatter(rec[:, 0], rec[:, 1], s=24, color="#16a34a", marker="o", edgecolors="#064e3b", linewidths=0.45, zorder=4)

    if stimulation:
        stim_points = [(positions[electrode][0], positions[electrode][1]) for electrode in sorted(stimulation) if electrode in positions]
        if stim_points:
            stim = np.asarray(stim_points, dtype=float)
            ax.scatter(stim[:, 0], stim[:, 1], s=50, color="#dc2626", marker="o", edgecolors="#111827", linewidths=0.8, zorder=6)

    selected = _resolve_channel_map_electrode(selected_electrode, lookup, positions)
    if selected is not None and selected in positions:
        x, y, _payload = positions[selected]
        ax.scatter([x], [y], s=92, facecolors="none", edgecolors="#111827", marker="o", linewidths=1.8, zorder=8)
        state["selected"] = selected

    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="#111827", linewidth=1.15, zorder=3)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return state


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


def _non_overlapping_spike_windows(spike_series, window_ms: float) -> list[tuple[float, float]]:
    window_s = max(0.001, float(window_ms) / 1000.0)
    chunks = []
    for _label, times in spike_series:
        values = np.asarray(times, dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            chunks.append(values)
    if not chunks:
        return []
    all_times = np.sort(np.concatenate(chunks))
    start_s = float(all_times[0])
    stop_s = float(all_times[-1])
    if stop_s <= start_s:
        return [(start_s, start_s + window_s)]
    count = max(1, int(np.ceil((stop_s - start_s) / window_s)))
    return [(start_s + index * window_s, start_s + (index + 1) * window_s) for index in range(count)]


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
    mode = str(normalization or "per_burst").lower()
    mapped_mode = {
        "per_burst": "per_sample_l1",
        "unit_zscore": "feature_zscore",
        "none": "none",
    }.get(mode, mode)
    normalized, _meta = normalize_feature_matrix(features, method=mapped_mode)
    return normalized


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
    return compute_similarity_matrix(features, method="correlation")


def _correlation_order_groups(correlation: np.ndarray, threshold: float = 0.45):
    return hierarchical_order_and_groups(correlation, threshold=threshold, criterion="distance", linkage_method="average")


def _kmeans_groups(features: np.ndarray, cluster_count: int) -> np.ndarray:
    return assign_feature_clusters(features, method="kmeans", cluster_count=cluster_count)


def _burst_embedding(features: np.ndarray, embedding_method: str) -> np.ndarray:
    reduced = reduce_feature_matrix(features, method=str(embedding_method or "pca").lower(), n_components=2, standardize=False)
    coordinates = np.asarray(reduced.get("coordinates", np.zeros((0, 2), dtype=float)), dtype=float)
    if coordinates.ndim != 2:
        return np.zeros((0, 2), dtype=float)
    if coordinates.shape[1] == 1:
        coordinates = np.column_stack([coordinates[:, 0], np.zeros(coordinates.shape[0], dtype=float)])
    return coordinates[:, :2]


def _burst_trajectory_feature_transform(activity: np.ndarray, normalization: str) -> tuple[np.ndarray, dict]:
    values = np.nan_to_num(np.asarray(activity, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if values.ndim != 3:
        return np.zeros((0, 0, 0), dtype=float), {"mode": str(normalization or "none").lower()}
    vectors = np.transpose(values, (0, 2, 1))
    mode = str(normalization or "channel_zscore").lower()
    params = {"mode": mode}
    if mode == "log_channel_zscore":
        vectors = np.log1p(np.maximum(vectors, 0.0))
    if mode == "per_time_total":
        totals = np.sum(np.abs(vectors), axis=2, keepdims=True)
        params["totals"] = totals
        return np.divide(vectors, totals, out=np.zeros_like(vectors), where=totals > 1e-12), params
    if mode in {"channel_zscore", "log_channel_zscore"}:
        flat = vectors.reshape((-1, vectors.shape[2]))
        means = np.mean(flat, axis=0, keepdims=True)
        stds = np.std(flat, axis=0, keepdims=True)
        scaled = np.divide(flat - means, stds, out=np.zeros_like(flat), where=stds > 1e-12)
        params["mean"] = means.reshape((1, 1, -1))
        params["std"] = stds.reshape((1, 1, -1))
        return scaled.reshape(vectors.shape), params
    return vectors, params


def _burst_trajectory_inverse_features(states: np.ndarray, params: dict) -> np.ndarray:
    values = np.nan_to_num(np.asarray(states, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if values.ndim != 3:
        return np.zeros((0, 0, 0), dtype=float)
    mode = str((params or {}).get("mode", "none")).lower()
    if mode in {"channel_zscore", "log_channel_zscore"}:
        mean = np.asarray((params or {}).get("mean", 0.0), dtype=float)
        std = np.asarray((params or {}).get("std", 1.0), dtype=float)
        values = values * std + mean
        if mode == "log_channel_zscore":
            values = np.expm1(values)
        return np.maximum(values, 0.0)
    if mode == "per_time_total":
        totals = np.asarray((params or {}).get("totals", 1.0), dtype=float)
        return np.maximum(values * totals, 0.0)
    return values


def _burst_trajectory_features(activity: np.ndarray, normalization: str) -> np.ndarray:
    features, _params = _burst_trajectory_feature_transform(activity, normalization)
    return features


def _trajectory_dispersion(trajectories: np.ndarray) -> np.ndarray:
    points = np.nan_to_num(np.asarray(trajectories, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if points.ndim != 3 or points.shape[0] == 0:
        return np.zeros(0, dtype=float)
    centroids = np.mean(points, axis=0, keepdims=True)
    distances = np.linalg.norm(points - centroids, axis=2)
    return np.mean(distances, axis=0)


def _local_pca_manifold_diagnostics(points: np.ndarray, max_dim: int = 8, neighbor_count: int = 24, max_samples: int = 1200) -> dict:
    values = np.nan_to_num(np.asarray(points, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if values.ndim != 2 or values.shape[0] < 4 or values.shape[1] < 1:
        return {
            "dims": np.zeros(0, dtype=int),
            "mean_local_reconstruction_error": np.zeros(0, dtype=float),
            "mean_local_variance_ratio": np.zeros(0, dtype=float),
            "estimated_local_dim": np.zeros(0, dtype=float),
        }
    if values.shape[0] > max_samples:
        sample_indices = np.linspace(0, values.shape[0] - 1, int(max_samples), dtype=int)
        values = values[sample_indices]
    sample_count, feature_dim = values.shape
    k = min(max(4, int(neighbor_count)), sample_count - 1)
    if k < 2:
        return {
            "dims": np.zeros(0, dtype=int),
            "mean_local_reconstruction_error": np.zeros(0, dtype=float),
            "mean_local_variance_ratio": np.zeros(0, dtype=float),
            "estimated_local_dim": np.zeros(0, dtype=float),
        }
    max_dim = min(max(1, int(max_dim)), feature_dim, k)
    dims = np.arange(1, max_dim + 1, dtype=int)
    try:
        knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        knn.fit(values)
        neighbor_indices = knn.kneighbors(return_distance=False)[:, 1:]
    except Exception:
        return {
            "dims": np.zeros(0, dtype=int),
            "mean_local_reconstruction_error": np.zeros(0, dtype=float),
            "mean_local_variance_ratio": np.zeros(0, dtype=float),
            "estimated_local_dim": np.zeros(0, dtype=float),
        }

    reconstruction_errors = np.zeros(dims.size, dtype=float)
    explained_ratios = np.zeros(dims.size, dtype=float)
    estimated_local_dim = np.zeros(sample_count, dtype=float)
    threshold = 0.9

    for point_index in range(sample_count):
        neighborhood = values[neighbor_indices[point_index]]
        if neighborhood.ndim != 2 or neighborhood.shape[0] < 2:
            continue
        centered = neighborhood - np.mean(neighborhood, axis=0, keepdims=True)
        try:
            _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        energy = singular_values ** 2
        total_energy = float(np.sum(energy))
        if total_energy <= 1e-12:
            continue
        cumulative = np.cumsum(energy) / total_energy
        estimated_local_dim[point_index] = float(np.searchsorted(cumulative, threshold) + 1)
        local_mean = np.mean(neighborhood, axis=0)
        point_centered = values[point_index] - local_mean
        for dim_index, dim in enumerate(dims):
            basis = vt[:dim]
            coeff = point_centered @ basis.T
            reconstructed = local_mean + coeff @ basis
            reconstruction_errors[dim_index] += float(np.linalg.norm(values[point_index] - reconstructed))
            explained_ratios[dim_index] += float(cumulative[min(dim - 1, cumulative.size - 1)])

    scale = max(1, sample_count)
    return {
        "dims": dims,
        "mean_local_reconstruction_error": reconstruction_errors / scale,
        "mean_local_variance_ratio": explained_ratios / scale,
        "estimated_local_dim": estimated_local_dim,
    }


def _factor_analysis_latent_states(
    features_by_time: np.ndarray,
    latent_dim: int = 16,
    max_iter: int = 1000,
) -> tuple[np.ndarray, dict]:
    values = np.nan_to_num(np.asarray(features_by_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if values.ndim != 3:
        return np.zeros((0, 0, 0), dtype=float), {}
    burst_count, bin_count, channel_count = values.shape
    if burst_count == 0 or bin_count == 0 or channel_count == 0:
        return np.zeros((burst_count, bin_count, 0), dtype=float), {}
    flat = values.reshape((burst_count * bin_count, channel_count))
    sample_count = flat.shape[0]
    components = min(max(1, int(latent_dim)), channel_count, sample_count)
    if sample_count < 2 or np.allclose(flat, flat[0]):
        latent = np.zeros((burst_count, bin_count, components), dtype=float)
        params = {
            "method": "factor_analysis",
            "latent_dim": components,
            "loadings": np.zeros((components, channel_count), dtype=float),
            "mean": np.mean(flat, axis=0) if flat.size else np.zeros(channel_count, dtype=float),
            "noise_variance": np.zeros(channel_count, dtype=float),
            "log_likelihood": np.zeros(0, dtype=float),
            "n_iter": 0,
        }
        return latent, params
    model = SkFactorAnalysis(n_components=components, random_state=7, max_iter=max(10, int(max_iter)))
    latent_flat = model.fit_transform(flat)
    latent = np.nan_to_num(latent_flat.reshape((burst_count, bin_count, components)), nan=0.0, posinf=0.0, neginf=0.0)
    params = {
        "method": "factor_analysis",
        "latent_dim": components,
        "loadings": np.nan_to_num(np.asarray(model.components_, dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
        "mean": np.nan_to_num(np.asarray(model.mean_, dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
        "noise_variance": np.nan_to_num(np.asarray(model.noise_variance_, dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
        "log_likelihood": np.nan_to_num(np.asarray(getattr(model, "loglike_", []), dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
        "n_iter": int(getattr(model, "n_iter_", 0)),
    }
    return latent, params


def _fit_linear_latent_dynamics(
    latent_states: np.ndarray,
    observed_states: np.ndarray,
    latent_params: dict,
    normalization_params: dict,
) -> dict:
    latent = np.nan_to_num(np.asarray(latent_states, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    observed = np.nan_to_num(np.asarray(observed_states, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    params = latent_params or {}
    loadings = np.asarray(params.get("loadings", []), dtype=float)
    mean = np.asarray(params.get("mean", []), dtype=float)
    if latent.ndim != 3 or latent.shape[0] == 0 or latent.shape[1] < 2 or latent.shape[2] == 0:
        return {}
    if observed.ndim != 3 or observed.shape[:2] != latent.shape[:2] or loadings.ndim != 2 or mean.ndim != 1:
        return {}

    burst_count, bin_count, latent_dim = latent.shape
    source = latent[:, :-1, :].reshape((-1, latent_dim))
    target = latent[:, 1:, :].reshape((-1, latent_dim))
    if source.shape[0] == 0 or target.shape[0] == 0:
        return {}

    design = np.column_stack([source, np.ones(source.shape[0], dtype=float)])
    try:
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return {}
    transition = coef[:-1, :].T
    bias = coef[-1, :]

    one_step_latent = np.zeros_like(latent)
    one_step_latent[:, 0, :] = latent[:, 0, :]
    one_step_latent[:, 1:, :] = (source @ transition.T + bias).reshape((burst_count, bin_count - 1, latent_dim))

    model_latent = np.zeros_like(latent)
    model_latent[:, 0, :] = latent[:, 0, :]
    for time_index in range(1, bin_count):
        model_latent[:, time_index, :] = model_latent[:, time_index - 1, :] @ transition.T + bias

    def _reconstruct(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        reconstructed_flat = states.reshape((-1, states.shape[2])) @ loadings + mean
        reconstructed = reconstructed_flat.reshape(observed.shape)
        raw_reconstructed = _burst_trajectory_inverse_features(reconstructed, normalization_params or {})
        return reconstructed, raw_reconstructed

    one_step_reconstructed, one_step_raw_reconstructed = _reconstruct(one_step_latent)
    reconstructed, raw_reconstructed = _reconstruct(model_latent)

    centered_observed = observed - np.mean(observed, axis=(0, 1), keepdims=True)
    sst = float(np.sum(centered_observed ** 2))
    one_step_residual = observed - one_step_reconstructed
    rollout_residual = observed - reconstructed
    one_step_r2 = 1.0 - float(np.sum(one_step_residual ** 2)) / max(sst, 1e-12)
    rollout_r2 = 1.0 - float(np.sum(rollout_residual ** 2)) / max(sst, 1e-12)
    one_step_time_rmse = np.sqrt(np.mean(one_step_residual ** 2, axis=(0, 2))) if one_step_residual.size else np.zeros(bin_count, dtype=float)
    time_rmse = np.sqrt(np.mean(rollout_residual ** 2, axis=(0, 2))) if rollout_residual.size else np.zeros(bin_count, dtype=float)
    one_step_latent_rmse = float(np.sqrt(np.mean((latent[:, 1:, :] - one_step_latent[:, 1:, :]) ** 2))) if bin_count > 1 else 0.0
    rollout_latent_rmse = float(np.sqrt(np.mean((latent[:, 1:, :] - model_latent[:, 1:, :]) ** 2))) if bin_count > 1 else 0.0

    return {
        "method": "lds",
        "transition_matrix": transition,
        "transition_bias": bias,
        "one_step_latent_states": one_step_latent,
        "model_latent_states": model_latent,
        "one_step_reconstructed_states": one_step_reconstructed,
        "one_step_raw_reconstructed_states": one_step_raw_reconstructed,
        "reconstructed_states": reconstructed,
        "raw_reconstructed_states": raw_reconstructed,
        "one_step_r2": float(one_step_r2),
        "rollout_r2": float(rollout_r2),
        "one_step_time_rmse": one_step_time_rmse,
        "time_rmse": time_rmse,
        "one_step_latent_rmse": float(one_step_latent_rmse),
        "rollout_latent_rmse": float(rollout_latent_rmse),
    }


def _pivae_latent_states(
    raw_observed_states: np.ndarray,
    latent_dim: int = 16,
    time_bin_ms: float = 10.0,
    cancel_check=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    values = np.nan_to_num(np.asarray(raw_observed_states, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if values.ndim != 3:
        return np.zeros((0, 0, 0), dtype=float), np.zeros((0, 0, 0), dtype=float), {}
    if values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] == 0:
        return np.zeros((values.shape[0], values.shape[1], 0), dtype=float), np.zeros_like(values), {}
    latent_states, reconstructed_states, params = fit_pivae_latent_states(
        values,
        latent_dim=int(latent_dim),
        bin_ms=float(time_bin_ms),
        cancel_check=cancel_check,
    )
    return (
        np.nan_to_num(np.asarray(latent_states, dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
        np.maximum(np.nan_to_num(np.asarray(reconstructed_states, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), 0.0),
        params if isinstance(params, dict) else {},
    )


def _select_factor_analysis_channels(
    activity: np.ndarray,
    labels,
    min_total_activity: float = 1.0,
    min_active_bursts: int = 1,
    min_variance: float = 0.0,
    max_channels: int = 256,
) -> tuple[np.ndarray, list[str], dict]:
    values = np.nan_to_num(np.asarray(activity, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    channel_count = values.shape[1] if values.ndim == 3 else 0
    all_labels = [str(label) for label in labels]
    if channel_count == 0:
        return np.zeros(0, dtype=int), [], {
            "total_activity": np.zeros(0, dtype=float),
            "active_bursts": np.zeros(0, dtype=int),
            "variance": np.zeros(0, dtype=float),
            "score": np.zeros(0, dtype=float),
        }
    flat = np.transpose(values, (0, 2, 1)).reshape((-1, channel_count))
    total_activity = np.sum(np.maximum(values, 0.0), axis=(0, 2))
    active_bursts = np.count_nonzero(np.sum(np.maximum(values, 0.0), axis=2) > 0.0, axis=0)
    variance = np.var(flat, axis=0)
    score = total_activity * np.sqrt(np.maximum(variance, 0.0) + 1e-12)
    mask = (
        (total_activity >= max(0.0, float(min_total_activity)))
        & (active_bursts >= max(0, int(min_active_bursts)))
        & (variance >= max(0.0, float(min_variance)))
    )
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        fallback_count = min(channel_count, max(1, int(max_channels)))
        selected = np.argsort(score)[::-1][:fallback_count]
    else:
        max_channels = max(1, int(max_channels))
        if selected.size > max_channels:
            selected = selected[np.argsort(score[selected])[::-1][:max_channels]]
        else:
            selected = selected[np.argsort(score[selected])[::-1]]
    selected = np.asarray(selected, dtype=int)
    selected_labels = [all_labels[index] if index < len(all_labels) else f"channel{index + 1}" for index in selected]
    metrics = {
        "total_activity": total_activity,
        "active_bursts": active_bursts.astype(int),
        "variance": variance,
        "score": score,
        "selected_indices": selected,
    }
    return selected, selected_labels, metrics


def _burst_trajectory_analysis(
    spike_series,
    burst_intervals,
    time_bin_ms: float = 5.0,
    window_ms: float = 100.0,
    normalization: str = "channel_zscore",
    cluster_count: int = 3,
    early_bins: int = 3,
    latent_dim: int = 16,
    min_total_activity: float = 1.0,
    min_active_bursts: int = 1,
    min_variance: float = 0.0,
    max_channels: int = 256,
    selected_channel_indices: np.ndarray | None = None,
    analysis_scope: str = "burst",
    model_method: str = "fa",
    cancel_check=None,
) -> dict:
    labels, intervals, activity = _burst_activity_matrix(spike_series, burst_intervals, time_bin_ms, window_ms)
    burst_count = int(activity.shape[0]) if activity.ndim == 3 else 0
    bin_count = int(activity.shape[2]) if activity.ndim == 3 else 0
    centers_ms = (np.arange(bin_count, dtype=float) + 0.5) * max(0.001, float(time_bin_ms))
    if burst_count == 0 or bin_count == 0:
        return {
            "labels": labels,
            "intervals": intervals,
            "activity": activity,
            "features": np.zeros((burst_count, bin_count, len(labels)), dtype=float),
            "observed_states": np.zeros((burst_count, bin_count, len(labels)), dtype=float),
            "reconstructed_states": np.zeros((burst_count, bin_count, len(labels)), dtype=float),
            "raw_observed_states": np.zeros((burst_count, bin_count, len(labels)), dtype=float),
            "raw_reconstructed_states": np.zeros((burst_count, bin_count, len(labels)), dtype=float),
            "centers_ms": centers_ms,
            "groups": np.ones(burst_count, dtype=int),
            "dispersion": np.zeros(bin_count, dtype=float),
            "reconstruction_rmse": np.zeros(bin_count, dtype=float),
            "reconstruction_r2": 0.0,
            "early_mean_dispersion": 0.0,
            "late_mean_dispersion": 0.0,
            "latent_states": np.zeros((burst_count, bin_count, 0), dtype=float),
            "latent_params": {},
            "representation": str(model_method or "fa"),
            "state_projection": "pi-VAE 0D" if str(model_method or "fa").strip().lower() == "pivae" else "Factor Analysis 0D",
            "analysis_scope": str(analysis_scope or "burst"),
            "model_method": str(model_method or "fa").strip().lower(),
            "selected_channel_indices": np.zeros(0, dtype=int),
            "selected_labels": [],
            "channel_filter": {},
        }

    if selected_channel_indices is None:
        selected_indices, selected_labels, channel_filter = _select_factor_analysis_channels(
            activity,
            labels,
            min_total_activity=min_total_activity,
            min_active_bursts=min_active_bursts,
            min_variance=min_variance,
            max_channels=max_channels,
        )
    else:
        channel_count = activity.shape[1] if activity.ndim == 3 else 0
        selected_indices = np.asarray(selected_channel_indices, dtype=int)
        selected_indices = selected_indices[(selected_indices >= 0) & (selected_indices < channel_count)]
        selected_labels = [str(labels[int(index)]) for index in selected_indices]
        flat = np.transpose(np.nan_to_num(activity, nan=0.0, posinf=0.0, neginf=0.0), (0, 2, 1)).reshape((-1, channel_count)) if channel_count else np.zeros((0, 0), dtype=float)
        total_activity = np.sum(np.maximum(activity, 0.0), axis=(0, 2)) if channel_count else np.zeros(0, dtype=float)
        active_bursts = np.count_nonzero(np.sum(np.maximum(activity, 0.0), axis=2) > 0.0, axis=0) if channel_count else np.zeros(0, dtype=int)
        variance = np.var(flat, axis=0) if flat.size else np.zeros(channel_count, dtype=float)
        channel_filter = {
            "total_activity": total_activity,
            "active_bursts": active_bursts.astype(int),
            "variance": variance,
            "score": total_activity * np.sqrt(np.maximum(variance, 0.0) + 1e-12),
            "selected_indices": selected_indices,
        }
    filtered_activity = activity[:, selected_indices, :] if selected_indices.size else activity[:, :0, :]
    raw_observed_states = np.transpose(np.nan_to_num(filtered_activity, nan=0.0, posinf=0.0, neginf=0.0), (0, 2, 1))
    requested_model = str(model_method or "fa").strip().lower()
    if requested_model == "pivae":
        observed_states = raw_observed_states
        norm_params = {"mode": "none"}
        latent_states, raw_reconstructed_states, latent_params = _pivae_latent_states(
            raw_observed_states,
            latent_dim=latent_dim,
            time_bin_ms=time_bin_ms,
            cancel_check=cancel_check,
        )
        reconstructed_states = raw_reconstructed_states
        state_projection = f"pi-VAE {latent_states.shape[2] if latent_states.ndim == 3 else 0}D"
        representation = "pi_vae"
    else:
        observed_states, norm_params = _burst_trajectory_feature_transform(filtered_activity, normalization)
        latent_states, latent_params = _factor_analysis_latent_states(observed_states, latent_dim)
        loadings = np.asarray(latent_params.get("loadings", []), dtype=float)
        mean = np.asarray(latent_params.get("mean", []), dtype=float)
        if latent_states.size and loadings.ndim == 2 and mean.ndim == 1:
            reconstructed_flat = latent_states.reshape((-1, latent_states.shape[2])) @ loadings + mean
            reconstructed_states = reconstructed_flat.reshape(observed_states.shape)
        else:
            reconstructed_states = np.zeros_like(observed_states)
        raw_reconstructed_states = _burst_trajectory_inverse_features(reconstructed_states, norm_params)
        state_projection = f"Factor Analysis {latent_states.shape[2]}D"
        representation = "factor_analysis"
    residual = observed_states - reconstructed_states
    per_sample_rmse = np.sqrt(np.mean(residual ** 2, axis=2)) if residual.ndim == 3 and residual.shape[2] else np.zeros((burst_count, bin_count), dtype=float)
    reconstruction_rmse = np.mean(per_sample_rmse, axis=0) if per_sample_rmse.size else np.zeros(bin_count, dtype=float)
    sse = float(np.sum(residual ** 2))
    centered_observed = observed_states - np.mean(observed_states, axis=(0, 1), keepdims=True)
    sst = float(np.sum(centered_observed ** 2))
    reconstruction_r2 = 1.0 - sse / max(sst, 1e-12)

    trajectory_bins = latent_states.shape[1]
    early_count = max(1, min(int(early_bins), max(1, trajectory_bins)))
    start_features = np.mean(latent_states[:, :early_count, :], axis=1)
    groups = _kmeans_groups(start_features, int(cluster_count))
    dispersion = _trajectory_dispersion(latent_states)
    late_start = min(max(0, trajectory_bins - 1), max(early_count, int(np.floor(max(1, trajectory_bins) * 0.5))))
    early_mean = float(np.mean(dispersion[:early_count])) if dispersion.size else 0.0
    late_mean = float(np.mean(dispersion[late_start:])) if dispersion.size else 0.0
    return {
        "labels": labels,
        "selected_labels": selected_labels,
        "selected_channel_indices": selected_indices,
        "channel_filter": channel_filter,
        "intervals": intervals,
        "activity": activity,
        "features": observed_states,
        "observed_states": observed_states,
        "reconstructed_states": reconstructed_states,
        "raw_observed_states": raw_observed_states,
        "raw_reconstructed_states": raw_reconstructed_states,
        "normalization_params": norm_params,
        "centers_ms": centers_ms,
        "groups": groups,
        "dispersion": dispersion,
        "reconstruction_rmse": reconstruction_rmse,
        "reconstruction_r2": float(reconstruction_r2),
        "early_mean_dispersion": early_mean,
        "late_mean_dispersion": late_mean,
        "early_bins": early_count,
        "time_bin_ms": float(time_bin_ms),
        "window_ms": float(window_ms),
        "representation": representation,
        "analysis_scope": str(analysis_scope or "burst"),
        "state_projection": state_projection,
        "model_method": "pivae" if requested_model == "pivae" else "fa",
        "latent_states": latent_states,
        "latent_params": latent_params,
    }


def _aligned_weight_similarity(weight_matrices: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    matrices = [np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0) for matrix in weight_matrices]
    count = len(matrices)
    similarity = np.eye(count, dtype=float)
    aligned = [matrix.copy() for matrix in matrices]
    if count == 0:
        return np.zeros((0, 0), dtype=float), []
    reference = matrices[0]
    if reference.ndim != 2 or reference.size == 0:
        return similarity, aligned
    for index in range(1, count):
        matrix = matrices[index]
        if matrix.ndim != 2 or matrix.size == 0:
            continue
        dims = min(reference.shape[0], matrix.shape[0])
        channels = min(reference.shape[1], matrix.shape[1])
        if dims < 1 or channels < 1:
            continue
        ref_block = reference[:dims, :channels]
        matrix_block = matrix[:dims, :channels]
        try:
            u, _s, vt = np.linalg.svd(matrix_block @ ref_block.T, full_matrices=False)
            rotation = vt.T @ u.T
            aligned_block = rotation @ matrix_block
            aligned[index] = matrix.copy()
            aligned[index][:dims, :channels] = aligned_block
        except np.linalg.LinAlgError:
            aligned[index] = matrix.copy()
    for row in range(count):
        for col in range(row + 1, count):
            a = aligned[row]
            b = aligned[col]
            if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
                value = 0.0
            else:
                dims = min(a.shape[0], b.shape[0])
                channels = min(a.shape[1], b.shape[1])
                va = a[:dims, :channels].reshape(-1)
                vb = b[:dims, :channels].reshape(-1)
                va = va - np.mean(va)
                vb = vb - np.mean(vb)
                denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
                value = float(np.dot(va, vb) / denom) if denom > 1e-12 else 0.0
            similarity[row, col] = value
            similarity[col, row] = value
    return similarity, aligned


def _multi_file_factor_analysis_payload(
    paths,
    *,
    time_bin_ms: float = 10.0,
    window_ms: float = 300.0,
    model_method: str = "fa",
    normalization: str = "channel_zscore",
    latent_dim: int = 16,
    min_total_activity: float = 1.0,
    min_active_bursts: int = 1,
    min_variance: float = 0.0,
    max_channels: int = 256,
    burst_bin_ms: float = 10.0,
    burst_smooth_ms: float = 50.0,
    burst_threshold_z: float = 4.0,
    artifact_ms: float = 1.0,
    analysis_scope: str = "burst",
    cancel_check=None,
    progress=None,
) -> dict:
    files = _stimulus_response_supported_files(paths)
    model_method = str(model_method or "fa").strip().lower()
    if model_method not in {"fa", "lds", "pivae"}:
        model_method = "fa"
    prepared_records = []
    errors = []
    total = max(1, len(files))
    for index, path in enumerate(files):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Multi-file factor analysis cancelled")
        if progress is not None:
            progress(5 + int(82 * index / total), f"Reading {path.name}...")
        try:
            data = _load_spike_only_data(path, cancel_check=cancel_check)
            spike_series = _spike_series_from_unified(data)
            stim_times = np.asarray(getattr(data, "stim_times", []), dtype=float)
            stim_times = np.sort(stim_times[np.isfinite(stim_times)])
            artifact_ms = max(0.0, float(artifact_ms))
            artifact_removed = 0
            if artifact_ms > 0.0 and stim_times.size:
                spike_series, _artifact_masks, artifact_removed = _filter_spike_series_stim_tail(
                    spike_series,
                    stim_times,
                    artifact_ms,
                )
            if not spike_series:
                raise ValueError("No spike trains found")
            scope = str(analysis_scope or "burst")
            if scope == "all_windows":
                intervals = _non_overlapping_spike_windows(spike_series, window_ms)
            else:
                intervals = _detect_burst_intervals(
                    spike_series,
                    bin_ms=burst_bin_ms,
                    smooth_ms=burst_smooth_ms,
                    threshold_z=burst_threshold_z,
                    cancel_check=cancel_check,
                )
            if not intervals:
                raise ValueError("No analysis windows available" if scope == "all_windows" else "No bursts detected")
            labels, activity_intervals, activity = _burst_activity_matrix(spike_series, intervals, time_bin_ms, window_ms)
            params = _extract_stimulus_parameters(path)
            prepared_records.append(
                {
                    "path": str(path),
                    "file": Path(path).name,
                    "condition": _stimulus_parameter_label(params, path),
                    "parameters": params,
                    "spike_series": spike_series,
                    "intervals": activity_intervals,
                    "labels": [str(label) for label in labels],
                    "activity": activity,
                    "burst_count": len(activity_intervals),
                    "channel_count": len(spike_series),
                    "stim_count": int(stim_times.size),
                    "artifact_ms": float(artifact_ms),
                    "artifact_removed_spikes": int(artifact_removed),
                }
            )
        except Exception as exc:
            errors.append(f"{Path(path).name}: {exc}")
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Multi-file factor analysis cancelled")
    if progress is not None:
        progress(88, "Selecting global channel set...")
    all_labels = sorted(
        {str(label) for record in prepared_records for label in record.get("labels", [])},
        key=_channel_sort_key,
    )
    global_index = {label: index for index, label in enumerate(all_labels)}
    bin_count = 0
    if prepared_records:
        first_activity = np.asarray(prepared_records[0].get("activity", []), dtype=float)
        bin_count = int(first_activity.shape[2]) if first_activity.ndim == 3 else 0
    global_chunks = []
    for record in prepared_records:
        activity = np.asarray(record.get("activity", []), dtype=float)
        labels = [str(label) for label in record.get("labels", [])]
        if activity.ndim != 3:
            continue
        aligned_activity = np.zeros((activity.shape[0], len(all_labels), activity.shape[2]), dtype=float)
        for local_index, label in enumerate(labels):
            target_index = global_index.get(label)
            if target_index is not None:
                aligned_activity[:, target_index, :] = activity[:, local_index, :]
        record["global_activity"] = aligned_activity
        if aligned_activity.size:
            global_chunks.append(aligned_activity)
    if global_chunks:
        global_activity = np.concatenate(global_chunks, axis=0)
        selected_indices, selected_labels, global_channel_filter = _select_factor_analysis_channels(
            global_activity,
            all_labels,
            min_total_activity=min_total_activity,
            min_active_bursts=min_active_bursts,
            min_variance=min_variance,
            max_channels=max_channels,
        )
        selected_indices = np.asarray(sorted(int(index) for index in selected_indices), dtype=int)
        selected_labels = [all_labels[int(index)] for index in selected_indices]
        global_channel_filter = dict(global_channel_filter)
        global_channel_filter["selected_indices"] = selected_indices
    else:
        global_channel_filter = {}
        selected_indices = np.zeros(0, dtype=int)
        selected_labels = []

    if progress is not None:
        progress(92, "Fitting aligned latent models...")
    records = []
    selected_set = set(int(index) for index in selected_indices)
    for record in prepared_records:
        global_activity = np.asarray(record.get("global_activity", []), dtype=float)
        if global_activity.ndim != 3 or global_activity.shape[0] == 0 or not selected_set:
            errors.append(f"{record.get('file', 'unknown')}: no global selected channels available")
            continue
        fixed_series = []
        for index, label in enumerate(all_labels):
            fixed_series.append((label, np.array([], dtype=float)))
        analysis = _burst_trajectory_analysis(
            fixed_series,
            record.get("intervals", []),
            time_bin_ms=time_bin_ms,
            window_ms=window_ms,
            normalization=normalization,
            latent_dim=latent_dim,
            min_total_activity=min_total_activity,
            min_active_bursts=min_active_bursts,
            min_variance=min_variance,
            max_channels=max_channels,
            selected_channel_indices=selected_indices,
            analysis_scope=str(analysis_scope or "burst"),
        )
        analysis["activity"] = global_activity
        filtered_activity = global_activity[:, selected_indices, :] if selected_indices.size else global_activity[:, :0, :]
        raw_observed_states = np.transpose(np.nan_to_num(filtered_activity, nan=0.0, posinf=0.0, neginf=0.0), (0, 2, 1))
        if model_method == "pivae":
            observed_states = raw_observed_states
            norm_params = {"mode": "none"}
            latent_states, raw_reconstructed_states, latent_params = _pivae_latent_states(
                raw_observed_states,
                latent_dim=latent_dim,
                time_bin_ms=time_bin_ms,
                cancel_check=cancel_check,
            )
            reconstructed_states = raw_reconstructed_states
            state_projection = f"pi-VAE {latent_states.shape[2] if latent_states.ndim == 3 else 0}D"
            analysis_model_method = "pivae"
        else:
            observed_states, norm_params = _burst_trajectory_feature_transform(filtered_activity, normalization)
            latent_states, latent_params = _factor_analysis_latent_states(observed_states, latent_dim)
            loadings = np.asarray(latent_params.get("loadings", []), dtype=float)
            mean = np.asarray(latent_params.get("mean", []), dtype=float)
            if latent_states.size and loadings.ndim == 2 and mean.ndim == 1:
                reconstructed_flat = latent_states.reshape((-1, latent_states.shape[2])) @ loadings + mean
                reconstructed_states = reconstructed_flat.reshape(observed_states.shape)
            else:
                reconstructed_states = np.zeros_like(observed_states)
            raw_reconstructed_states = _burst_trajectory_inverse_features(reconstructed_states, norm_params)
            state_projection = f"Factor Analysis {latent_states.shape[2] if latent_states.ndim == 3 else 0}D"
            analysis_model_method = "fa"
        residual = observed_states - reconstructed_states
        per_sample_rmse = np.sqrt(np.mean(residual ** 2, axis=2)) if residual.ndim == 3 and residual.shape[2] else np.zeros((observed_states.shape[0], observed_states.shape[1]), dtype=float)
        fa_reconstruction_rmse = np.mean(per_sample_rmse, axis=0) if per_sample_rmse.size else np.zeros(bin_count, dtype=float)
        centered_observed = observed_states - np.mean(observed_states, axis=(0, 1), keepdims=True) if observed_states.size else observed_states
        sse = float(np.sum(residual ** 2))
        sst = float(np.sum(centered_observed ** 2))
        fa_reconstruction_r2 = 1.0 - sse / max(sst, 1e-12)
        dispersion = _trajectory_dispersion(latent_states)
        early_count = max(1, min(3, max(1, latent_states.shape[1] if latent_states.ndim == 3 else 1)))
        early_mean = float(np.mean(dispersion[:early_count])) if dispersion.size else 0.0
        late_start = min(max(0, dispersion.size - 1), max(early_count, int(np.floor(max(1, dispersion.size) * 0.5)))) if dispersion.size else 0
        late_mean = float(np.mean(dispersion[late_start:])) if dispersion.size else 0.0
        analysis.update(
            {
                "labels": all_labels,
                "selected_labels": selected_labels,
                "selected_channel_indices": selected_indices,
                "channel_filter": global_channel_filter,
                "features": observed_states,
                "observed_states": observed_states,
                "reconstructed_states": reconstructed_states,
                "raw_observed_states": raw_observed_states,
                "raw_reconstructed_states": raw_reconstructed_states,
                "normalization_params": norm_params,
                "latent_states": latent_states,
                "latent_params": latent_params,
                "reconstruction_rmse": fa_reconstruction_rmse,
                "reconstruction_r2": float(fa_reconstruction_r2),
                "fa_reconstruction_rmse": fa_reconstruction_rmse,
                "fa_reconstruction_r2": float(fa_reconstruction_r2),
                "dispersion": dispersion,
                "early_mean_dispersion": early_mean,
                "late_mean_dispersion": late_mean,
                "state_projection": state_projection,
                "analysis_scope": str(analysis_scope or "burst"),
                "model_method": analysis_model_method,
            }
        )
        if model_method == "lds":
            lds_result = _fit_linear_latent_dynamics(latent_states, observed_states, latent_params, norm_params)
            if lds_result:
                analysis.update(lds_result)
                analysis["reconstruction_rmse"] = np.asarray(lds_result.get("time_rmse", fa_reconstruction_rmse), dtype=float)
                analysis["reconstruction_r2"] = float(lds_result.get("rollout_r2", fa_reconstruction_r2))
                analysis["state_projection"] = f"LDS over FA latent {latent_states.shape[2] if latent_states.ndim == 3 else 0}D"
                analysis["model_method"] = "lds"
        record = dict(record)
        record.pop("spike_series", None)
        record.pop("activity", None)
        record.pop("global_activity", None)
        record["analysis"] = analysis
        record["selected_channel_count"] = len(selected_labels)
        record["latent_dim"] = int(latent_states.shape[2]) if latent_states.ndim == 3 else 0
        record["model_method"] = str(analysis.get("model_method", model_method))
        record["reconstruction_r2"] = float(analysis.get("reconstruction_r2", fa_reconstruction_r2))
        record["fa_reconstruction_r2"] = float(fa_reconstruction_r2)
        records.append(record)

    if progress is not None:
        progress(96, "Aligning factor loading matrices...")
    weights = [np.asarray((record.get("analysis", {}).get("latent_params", {}) or {}).get("loadings", []), dtype=float) for record in records]
    similarity, aligned_weights = _aligned_weight_similarity(weights)
    for record, aligned_weight in zip(records, aligned_weights):
        record["aligned_loadings"] = aligned_weight
    return {
        "records": records,
        "errors": errors,
        "paths": [str(path) for path in files],
        "w_similarity": similarity,
        "global_labels": all_labels,
        "global_selected_labels": selected_labels,
        "global_selected_channel_indices": selected_indices,
        "global_channel_filter": global_channel_filter,
        "time_bin_ms": float(time_bin_ms),
        "window_ms": float(window_ms),
        "model_method": model_method,
        "normalization": str(normalization),
        "latent_dim": int(latent_dim),
        "min_total_activity": float(min_total_activity),
        "min_active_bursts": int(min_active_bursts),
        "min_variance": float(min_variance),
        "max_channels": int(max_channels),
        "burst_bin_ms": float(burst_bin_ms),
        "burst_smooth_ms": float(burst_smooth_ms),
        "burst_threshold_z": float(burst_threshold_z),
        "artifact_ms": float(artifact_ms),
        "analysis_scope": str(analysis_scope or "burst"),
    }


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


DATA_FILE_EXTENSIONS = {".npy", ".npz", ".csv", ".txt", ".tsv", ".nev", ".spk", ".h5", ".hdf5"}
STIMULUS_RESPONSE_EXTENSIONS = {".nev", ".spk", ".h5", ".hdf5", ".npz"}


def _supported_files(paths, extensions: set[str]) -> list[Path]:
    files = []
    seen = set()
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            candidates = [
                child
                for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() in extensions
            ]
        elif path.is_file() and path.suffix.lower() in extensions:
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


def _data_file_supported_files(paths) -> list[Path]:
    return _supported_files(paths, DATA_FILE_EXTENSIONS)


def _stimulus_response_supported_files(paths) -> list[Path]:
    return _supported_files(paths, STIMULUS_RESPONSE_EXTENSIONS)


def _normalize_stimulus_parameter_key(key: str, unit: str = "") -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(key).strip().lower()).strip("_")
    unit_text = str(unit or "").replace("\u03bc", "u").lower()
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
        r"([A-Za-z]+)\s*[-_= ]?\s*(-?\d+(?:\.\d+)?)\s*(Hz|kHz|uA|纰孉|mA|A|mV|V|us|ms|s)?(?=[^A-Za-z纰寀]|$)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        key = _normalize_stimulus_parameter_key(match.group(1), match.group(3))
        try:
            params[key] = float(match.group(2))
        except ValueError:
            continue
    unit_only = re.compile(r"(?<![A-Za-z])(-?\d+(?:\.\d+)?)\s*(Hz|kHz|uA|纰孉|mA|A|mV|V|us|ms|s)\b", re.IGNORECASE)
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


class ChannelMapDialog(AppDialog):
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
            _show_info_message(self, "Channel Map", "No saved map is selected.")
            return
        loaded = load_channel_map(name)
        if loaded is None:
            _show_warning_message(self, "Channel Map", f"Map not found: {name}")
            self._refresh_saved_maps()
            return
        self._set_map(loaded)

    def _load_default_map(self) -> None:
        loaded = default_channel_map()
        if loaded is None:
            _show_info_message(self, "Channel Map", "No default map has been saved.")
            return
        self._set_map(loaded)

    def _save_map(self) -> None:
        self.channel_map.name = self.name_edit.text().strip() or "Untitled"
        save_channel_map(self.channel_map)
        self._refresh_saved_maps()
        self.saved_maps.setCurrentText(self.channel_map.name)
        self._update_validation()
        _show_info_message(self, "Channel Map", f"Saved map: {self.channel_map.name}")

    def _save_as_default(self) -> None:
        self.channel_map.name = self.name_edit.text().strip() or "Untitled"
        save_channel_map(self.channel_map, make_default=True)
        self._refresh_saved_maps()
        self.saved_maps.setCurrentText(self.channel_map.name)
        self._update_validation()
        _show_info_message(self, "Channel Map", f"Saved default map: {self.channel_map.name}")

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


def _defer_hide(widget: QWidget | None, delay_ms: int = 0) -> None:
    if widget is None:
        return
    QTimer.singleShot(max(0, int(delay_ms)), widget.hide)


def _unique_output_path(path: Path, used_lowercase_paths: set[str] | None = None) -> Path:
    candidate = Path(path)
    suffix = candidate.suffix
    stem = candidate.stem
    parent = candidate.parent
    used = used_lowercase_paths if used_lowercase_paths is not None else set()
    counter = 1
    while candidate.exists() or str(candidate).lower() in used:
        candidate = parent / f"{stem}_{counter}{suffix}"
        counter += 1
    used.add(str(candidate).lower())
    return candidate


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
                self.signals.progress.emit(25, "Extracting Maxwell spikes...")
                raw_data = read_maxwell_h5(self.path, cancel_check=self._is_cancelled, extract_waveforms=False)
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


def _load_gui_data_file(path: str | Path, *, selected_wells=None, cancel_check=None):
    path = Path(path)
    suffix = path.suffix.lower()
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Data loading cancelled")
    if suffix == ".nev":
        raw_data = read_blackrock_nev(path, cancel_check=cancel_check)
        data_kind = "nev"
    elif suffix == ".spk":
        raw_data = read_axion_spk(path, wells=selected_wells)
        data_kind = "nev"
    elif suffix in {".h5", ".hdf5"}:
        raw_data = read_maxwell_h5(path, cancel_check=cancel_check, extract_waveforms=False)
        data_kind = "nev"
    else:
        raw_data = MEAReader(path).load_data()
        data_kind = "nev" if isinstance(raw_data, UnifiedMEAData) else "array"
    return raw_data, data_kind


def _loaded_data_kind_label(data, data_kind: str = "") -> str:
    if isinstance(data, UnifiedMEAData) and isinstance(data.meta, dict):
        source = str(data.meta.get("source", "") or "")
        if source:
            return source
    return str(data_kind or type(data).__name__)


def _loaded_data_stats(data) -> tuple[int, int, int]:
    if isinstance(data, UnifiedMEAData):
        channels = len(data.channels())
        spikes = int(sum(np.asarray(values).size for values in data.spikes.values()))
        waveforms = len(data.waveforms)
        return channels, spikes, waveforms
    array = np.asarray(data) if data is not None else np.asarray([])
    channels = int(array.shape[0]) if array.ndim >= 1 else 0
    return channels, int(array.size), 0


def _loaded_data_activity_label(path: str | Path, data=None) -> str:
    params = _extract_stimulus_parameters(path)
    activity = str(params.get("activity", "")).strip().lower()
    if activity == "spont":
        return "Spontaneous"
    if isinstance(data, UnifiedMEAData):
        stim_times = np.asarray(getattr(data, "stim_times", []), dtype=float)
        if np.count_nonzero(np.isfinite(stim_times)) > 0:
            return "Stimulus"
        if isinstance(data.meta, dict):
            event_count = data.meta.get("event_count", 0)
            try:
                if int(event_count) > 0:
                    return "Stimulus"
            except (TypeError, ValueError):
                pass
    if "stim_electrode" in params or str(params.get("stim_mode", "")).strip().lower() == "multi-site":
        return "Stimulus"
    return "Spontaneous"


def _loaded_data_activity_sort_key(label: str) -> tuple[int, str]:
    text = str(label or "")
    order = {"Stimulus": 0, "Spontaneous": 1}
    return (order.get(text, 2), text.lower())


def _generic_analysis_matrix_from_record(
    record,
    *,
    view_mode: str = "auto",
    bin_ms: float = 10.0,
    burst_window_ms: float = 300.0,
    burst_threshold_z: float = 4.0,
    array_axis: str = "rows",
) -> tuple[np.ndarray, list[str], str]:
    """Convert a loaded database record into a generic samples x features matrix."""

    raw_data = (record or {}).get("raw_data")
    view_mode = str(view_mode or "auto").strip().lower()
    if isinstance(raw_data, UnifiedMEAData):
        spike_series = _spike_series_from_unified(raw_data)
        if not spike_series:
            return np.zeros((0, 0), dtype=float), [], "No spike-event channels"
        channel_labels = [str(label) for label, _times in spike_series]
        start_s, stop_s = raw_data.time_range()
        bin_s = max(1e-3, float(bin_ms) / 1000.0)
        if stop_s <= start_s:
            stop_s = start_s + bin_s
        edges = np.arange(start_s, stop_s + bin_s, bin_s, dtype=float)
        if edges.size < 2:
            edges = np.array([start_s, start_s + bin_s], dtype=float)
        rows = []
        for _label, times in spike_series:
            counts, _ = np.histogram(np.asarray(times, dtype=float), bins=edges)
            rows.append(np.asarray(counts, dtype=float))
        channel_time = np.vstack(rows) if rows else np.zeros((0, max(0, edges.size - 1)), dtype=float)
        burst_intervals = _detect_burst_intervals(
            spike_series,
            bin_ms=float(bin_ms),
            threshold_z=float(burst_threshold_z),
            min_spikes=5,
        )
        burst_labels, burst_intervals, burst_activity = _burst_activity_matrix(
            spike_series,
            burst_intervals,
            time_bin_ms=float(bin_ms),
            window_ms=float(burst_window_ms),
        )
        if view_mode in {"auto", "channel_time"}:
            description = f"Channel x time-bin spike-count matrix | channels={len(channel_labels)} | bins={channel_time.shape[1]} | bin={float(bin_ms):g} ms"
            return channel_time, channel_labels, description
        if view_mode == "time_channel":
            labels = [f"bin {index + 1}" for index in range(channel_time.shape[1])]
            description = f"Time-bin x channel spike-count matrix | bins={channel_time.shape[1]} | channels={len(channel_labels)} | bin={float(bin_ms):g} ms"
            return channel_time.T, labels, description
        if view_mode == "burst_flat":
            if burst_activity.ndim == 3 and burst_activity.shape[0] > 0:
                matrix = burst_activity.reshape((burst_activity.shape[0], burst_activity.shape[1] * burst_activity.shape[2]))
            else:
                matrix = np.zeros((0, 0), dtype=float)
            labels = [f"burst {index + 1}" for index in range(matrix.shape[0])]
            description = (
                f"Burst x flattened(channel,time) matrix | bursts={matrix.shape[0]} | channels={len(burst_labels)} | "
                f"bins/window={burst_activity.shape[2] if burst_activity.ndim == 3 else 0} | bin={float(bin_ms):g} ms | window={float(burst_window_ms):g} ms"
            )
            return matrix, labels, description
        if view_mode == "channel_burst":
            if burst_activity.ndim == 3 and burst_activity.shape[0] > 0:
                matrix = np.transpose(burst_activity, (1, 0, 2)).reshape((burst_activity.shape[1], burst_activity.shape[0] * burst_activity.shape[2]))
            else:
                matrix = np.zeros((0, 0), dtype=float)
            description = (
                f"Channel x flattened(burst,time) matrix | channels={len(burst_labels)} | bursts={burst_activity.shape[0] if burst_activity.ndim == 3 else 0} | "
                f"bins/window={burst_activity.shape[2] if burst_activity.ndim == 3 else 0} | bin={float(bin_ms):g} ms | window={float(burst_window_ms):g} ms"
            )
            return matrix, [str(label) for label in burst_labels], description
        description = f"Channel x time-bin spike-count matrix | channels={len(channel_labels)} | bins={channel_time.shape[1]} | bin={float(bin_ms):g} ms"
        return channel_time, channel_labels, description

    array = np.nan_to_num(np.asarray(raw_data, dtype=float), nan=0.0, posinf=0.0, neginf=0.0) if raw_data is not None else np.zeros((0, 0), dtype=float)
    if array.ndim == 0:
        array = array.reshape((1, 1))
    elif array.ndim == 1:
        array = array.reshape((1, -1))
    elif array.ndim > 2:
        first_dim = int(array.shape[0]) if array.shape else 0
        array = array.reshape((first_dim, -1))
    axis_mode = str(array_axis or "rows").strip().lower()
    if axis_mode == "columns" and array.ndim == 2:
        array = array.T
        labels = [f"column {index + 1}" for index in range(array.shape[0])]
        description = f"Array matrix | samples=columns | rows={array.shape[1]} | columns={array.shape[0]}"
        return np.asarray(array, dtype=float), labels, description
    labels = [f"row {index + 1}" for index in range(array.shape[0])] if array.ndim >= 2 else []
    description = f"Array matrix | samples=rows | rows={array.shape[0] if array.ndim >= 1 else 0} | features={array.shape[1] if array.ndim >= 2 else 0}"
    return np.asarray(array, dtype=float), labels, description


def _parse_optional_float_list(text: str) -> np.ndarray:
    values = []
    for token in re.split(r"[\s,;]+", str(text or "").strip()):
        if not token:
            continue
        values.append(float(token))
    return np.asarray(values, dtype=float)


def _parse_custom_time_windows(text: str, data: UnifiedMEAData) -> list[tuple[float, float, str]]:
    raw = str(text or "full").strip()
    start_s, stop_s = data.time_range()
    if stop_s <= start_s:
        finite = [np.asarray(times, dtype=float) for _label, times in _spike_series_from_unified(data) if np.asarray(times, dtype=float).size]
        if finite:
            stop_s = max(float(np.nanmax(values)) for values in finite if values.size)
        stop_s = max(stop_s, start_s + 1.0)
    if not raw or raw.lower() in {"full", "all", "*"}:
        return [(float(start_s), float(stop_s), f"{float(start_s):g}-{float(stop_s):g}s")]
    windows: list[tuple[float, float, str]] = []
    for chunk in re.split(r"[,;]+", raw):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:-|:|to)\s*([+-]?\d+(?:\.\d+)?)\s*$", chunk, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid time window: {chunk}")
        lo = float(match.group(1))
        hi = float(match.group(2))
        if hi <= lo:
            raise ValueError(f"Time window stop must be greater than start: {chunk}")
        windows.append((lo, hi, f"{lo:g}-{hi:g}s"))
    if not windows:
        raise ValueError("No valid time windows were defined")
    return windows


def _custom_channel_filter(spike_series, channels_text: str):
    requested = [token.strip() for token in re.split(r"[,;]+", str(channels_text or "")) if token.strip()]
    if not requested:
        return [(str(label), np.asarray(times, dtype=float)) for label, times in spike_series]
    requested_keys = {normalize_channel_name(token) for token in requested}
    requested_keys.update(normalize_channel_name(_base_channel_from_raster_label(token)) for token in requested)
    selected = []
    for label, times in spike_series:
        candidates = {
            normalize_channel_name(str(label)),
            normalize_channel_name(_base_channel_from_raster_label(str(label))),
        }
        if candidates & requested_keys:
            selected.append((str(label), np.asarray(times, dtype=float)))
    return selected


def _custom_spike_vector_matrix(spike_series, windows, analysis_type: str):
    mode = str(analysis_type or "firing_rate_vector")
    feature_labels = [str(label) for label, _times in spike_series]
    sample_labels = [str(label) for _start, _stop, label in windows]
    matrix = np.zeros((len(windows), len(spike_series)), dtype=float)
    for row, (start_s, stop_s, _label) in enumerate(windows):
        duration_s = max(float(stop_s) - float(start_s), 1e-9)
        for col, (_channel, times) in enumerate(spike_series):
            values = np.asarray(times, dtype=float)
            count = int(np.count_nonzero((values >= float(start_s)) & (values < float(stop_s))))
            matrix[row, col] = float(count) / duration_s if mode == "firing_rate_vector" else float(count)
    if mode == "firing_rate_vector":
        description = f"Custom vector firing rate | windows={len(windows)} | channels={len(spike_series)}"
    elif mode == "spike_count_vector":
        description = f"Custom vector spike count | windows={len(windows)} | channels={len(spike_series)}"
    else:
        raise ValueError(f"Unsupported custom analysis type: {analysis_type}")
    return matrix, sample_labels, feature_labels, description


def _processed_dataset_label(record, *, view_mode: str) -> str:
    path = Path(str((record or {}).get("path", "")))
    source_name = path.name or str((record or {}).get("name", "") or "dataset")
    return f"{source_name} | {str(view_mode or 'auto')}"


def _processed_dataset_commit(parameters: dict | None) -> tuple[str, str]:
    params = dict(parameters or {})
    dataset_type = str(params.get("dataset_type", params.get("view_mode", "dataset")) or "dataset")
    dataset_group = str(params.get("dataset_group", "generic") or "generic")
    origin = str(params.get("origin", "manual") or "manual")
    bin_ms = float(params.get("bin_ms", 10.0))
    burst_window_ms = float(params.get("burst_window_ms", 300.0))
    burst_threshold_z = float(params.get("burst_threshold_z", 4.0))
    array_axis = str(params.get("array_axis", "rows") or "rows")

    if dataset_type == "burst_flat":
        title = f"{dataset_group} | burst detection cached"
        detail = (
            f"Generated from spike trains with burst detection; "
            f"view={params.get('view_mode', 'burst_flat')}, bin={bin_ms:g} ms, "
            f"burst window={burst_window_ms:g} ms, burst z-threshold={burst_threshold_z:g}, origin={origin}."
        )
        return title, detail
    if dataset_type == "channel_time":
        title = f"{dataset_group} | channel-time spike counts"
        detail = (
            f"Generated from spike trains using time-bin counting; "
            f"view={params.get('view_mode', 'channel_time')}, bin={bin_ms:g} ms, origin={origin}."
        )
        return title, detail
    if dataset_group == "array":
        title = "array | matrix import"
        detail = (
            f"Generated from array-like source data; "
            f"view={params.get('view_mode', 'auto')}, sample axis={array_axis}, origin={origin}."
        )
        return title, detail
    title = f"{dataset_group} | processed dataset"
    detail = (
        f"Generated with view={params.get('view_mode', 'auto')}, "
        f"bin={bin_ms:g} ms, burst window={burst_window_ms:g} ms, "
        f"burst z-threshold={burst_threshold_z:g}, sample axis={array_axis}, origin={origin}."
    )
    return title, detail


def _processed_dataset_presets_for_record(record) -> list[dict]:
    data = record.get("raw_data") if isinstance(record, dict) else None
    if isinstance(data, UnifiedMEAData):
        return [
            {
                "dataset_type": "channel_time",
                "dataset_group": "raster",
                "view_mode": "channel_time",
                "bin_ms": 10.0,
                "burst_window_ms": 300.0,
                "burst_threshold_z": 4.0,
                "array_axis": "rows",
                "origin": "auto-raster",
                "display_name": "Channel x time",
            },
            {
                "dataset_type": "burst_flat",
                "dataset_group": "burst",
                "view_mode": "burst_flat",
                "bin_ms": 10.0,
                "burst_window_ms": 300.0,
                "burst_threshold_z": 4.0,
                "array_axis": "rows",
                "origin": "auto-raster",
                "display_name": "Burst trajectory matrix",
            },
        ]
    return [
        {
            "dataset_type": "array_rows",
            "dataset_group": "array",
            "view_mode": "auto",
            "bin_ms": 10.0,
            "burst_window_ms": 300.0,
            "burst_threshold_z": 4.0,
            "array_axis": "rows",
            "origin": "auto-raster",
            "display_name": "Array rows",
        }
    ]


class FileDatabaseLoadWorker(QRunnable):
    def __init__(self, paths, *, selected_wells_by_path=None):
        super().__init__()
        self.paths = [str(path) for path in paths]
        self.selected_wells_by_path = dict(selected_wells_by_path or {})
        self.signals = WorkerSignals()
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return bool(self._cancel_requested)

    @Slot()
    def run(self):
        records = []
        errors = []
        total = max(1, len(self.paths))
        try:
            for index, path_text in enumerate(self.paths):
                if self._is_cancelled():
                    raise InterruptedError("Data loading cancelled")
                path = Path(path_text)
                self.signals.progress.emit(
                    5 + int(87 * index / total),
                    f"Reading {path.name} ({index + 1}/{len(self.paths)})...",
                )
                try:
                    raw_data, data_kind = _load_gui_data_file(
                        path,
                        selected_wells=self.selected_wells_by_path.get(str(path)),
                        cancel_check=self._is_cancelled,
                    )
                    records.append({"path": str(path), "raw_data": raw_data, "data_kind": data_kind})
                except InterruptedError:
                    raise
                except Exception as exc:
                    errors.append(f"{path.name}: {exc}")
            if self._is_cancelled():
                raise InterruptedError("Data loading cancelled")
            self.signals.progress.emit(95, "Preparing file database...")
            self.signals.finished.emit({"records": records, "errors": errors})
        except InterruptedError as exc:
            self.signals.canceled.emit(str(exc) or "Data loading cancelled")
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class MaxwellWaveformLoadWorker(QRunnable):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.signals = WorkerSignals()
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return bool(self._cancel_requested)

    @Slot()
    def run(self):
        try:
            if self._is_cancelled():
                raise InterruptedError("Waveform loading cancelled")
            self.signals.progress.emit(10, "Opening Maxwell raw data...")
            data = read_maxwell_h5(self.path, cancel_check=self._is_cancelled, extract_waveforms=True)
            if self._is_cancelled():
                raise InterruptedError("Waveform loading cancelled")
            self.signals.progress.emit(95, "Preparing waveforms...")
            self.signals.finished.emit(data)
        except InterruptedError as exc:
            self.signals.canceled.emit(str(exc) or "Waveform loading cancelled")
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


class MultiFileFactorAnalysisWorker(QRunnable):
    def __init__(self, paths, parameters: dict):
        super().__init__()
        self.paths = [str(path) for path in paths]
        self.parameters = dict(parameters or {})
        self.signals = WorkerSignals()
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return bool(self._cancel_requested)

    @Slot()
    def run(self):
        try:
            payload = _multi_file_factor_analysis_payload(
                self.paths,
                cancel_check=self._is_cancelled,
                progress=self.signals.progress.emit,
                **self.parameters,
            )
            if self._is_cancelled():
                raise InterruptedError("Multi-file factor analysis cancelled")
            self.signals.finished.emit(payload)
        except InterruptedError as exc:
            self.signals.canceled.emit(str(exc) or "Multi-file factor analysis cancelled")
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class DataFilesInputDialog(AppDialog):
    def __init__(self, parent=None, initial_paths=None):
        super().__init__(parent)
        self.setWindowTitle("Open Data Files")
        self.resize(860, 520)
        self.paths: list[str] = []

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["File", "Type", "Folder"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)

        add_files = QPushButton("Add Files")
        add_files.clicked.connect(self._add_files)
        add_folder = QPushButton("Add Folder")
        add_folder.clicked.connect(self._add_folder)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_selected)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        load = QPushButton("Load")
        load.setObjectName("PrimaryButton")
        load.clicked.connect(self.accept)

        controls = QHBoxLayout()
        controls.addStretch(1)
        controls.addWidget(add_files)
        controls.addWidget(add_folder)
        controls.addWidget(remove)
        controls.addWidget(cancel)
        controls.addWidget(load)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Add data files directly, or add folders recursively. Maxwell H5 files load spikes first; waveforms are loaded later for sorting."))
        layout.addWidget(self.table, 1)
        layout.addLayout(controls)

        if initial_paths:
            self._add_paths(initial_paths)

    def _add_paths(self, paths) -> None:
        files = _data_file_supported_files(paths)
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
            "Add data files",
            "data",
            "Data files (*.npy *.npz *.csv *.txt *.tsv *.nev *.spk *.h5 *.hdf5);;Array files (*.npy *.npz *.csv *.txt *.tsv);;Spike files (*.nev *.spk *.h5 *.hdf5 *.npz);;All files (*)",
        )
        self._add_paths(paths)

    def _add_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Add data folder", "data")
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
            values = [path.name, path.suffix.lower().lstrip(".") or "file", str(path.parent)]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, path_text)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()

    def values(self) -> list[str]:
        return list(self.paths)


class _DatabaseAnalysisDialogBase(AppDialog):
    def _setup_database_table(self):
        self.database_sort_column = None
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["File", "Kind", "Label", "Channels", "Spikes", "Folder"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        table_header = self.table.horizontalHeader()
        table_header.setSectionsClickable(True)
        table_header.setSortIndicatorShown(False)
        table_header.sectionClicked.connect(self._database_header_clicked)

    def _set_records(self, records) -> None:
        selected_paths = set()
        if hasattr(self, "table") and self.table.rowCount():
            for index in self.table.selectedIndexes():
                item = self.table.item(index.row(), 0)
                if item is not None:
                    path_text = str(item.data(Qt.ItemDataRole.UserRole) or "")
                    if path_text:
                        selected_paths.add(path_text)
        self.records = list(records or [])
        self._ensure_database_order()
        if self.database_sort_column is not None:
            column = int(self.database_sort_column)
            self.records.sort(key=lambda record: (self._database_sort_value(record, column), self._database_default_sort_key(record)))
        self.table.setRowCount(len(self.records))
        for row, record in enumerate(self.records):
            path = Path(str(record.get("path", "")))
            data = record.get("raw_data")
            channels, spikes, _waveforms = _loaded_data_stats(data)
            values = [
                path.name,
                _loaded_data_kind_label(data, str(record.get("data_kind", ""))),
                _loaded_data_activity_label(path, data),
                str(channels),
                str(spikes),
                str(path.parent),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        self._restore_selected_paths(selected_paths)

    def _ensure_database_order(self) -> None:
        for index, record in enumerate(self.records):
            if isinstance(record, dict) and "_database_order" not in record:
                record["_database_order"] = index

    def _database_header_clicked(self, column: int) -> None:
        if not self.records:
            return
        if self.database_sort_column == int(column):
            self.database_sort_column = None
            self._reorder_database_records(lambda record: self._database_default_sort_key(record))
            return
        self.database_sort_column = int(column)
        self._reorder_database_records(
            lambda record: (self._database_sort_value(record, int(column)), self._database_default_sort_key(record))
        )

    def _database_default_sort_key(self, record: dict) -> tuple[int, str]:
        try:
            order = int(record.get("_database_order", 0))
        except (AttributeError, TypeError, ValueError):
            order = 0
        return (order, str((record or {}).get("path", "")).lower())

    def _database_sort_value(self, record: dict, column: int):
        path = Path(str((record or {}).get("path", "")))
        data = (record or {}).get("raw_data")
        channels, spikes, _waveforms = _loaded_data_stats(data)
        values = {
            0: str(path.name).lower(),
            1: _loaded_data_kind_label(data, str((record or {}).get("data_kind", ""))).lower(),
            2: _loaded_data_activity_sort_key(_loaded_data_activity_label(path, data)),
            3: int(channels),
            4: int(spikes),
            5: str(path.parent).lower(),
        }
        return values.get(int(column), str(path).lower())

    def _reorder_database_records(self, key) -> None:
        selected_paths = {
            str(self.records[row].get("path", ""))
            for row in self._selected_rows()
            if 0 <= row < len(self.records)
        }
        self.records.sort(key=key)
        self._set_records(self.records)
        if selected_paths:
            self.table.clearSelection()
            self._restore_selected_paths(selected_paths)

    def _restore_selected_paths(self, selected_paths) -> None:
        self.table.clearSelection()
        if not self.records:
            return
        selection_model = self.table.selectionModel()
        flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
        matched = False
        if selected_paths and selection_model is not None:
            for row, record in enumerate(self.records):
                if str(record.get("path", "")) in selected_paths:
                    selection_model.select(self.table.model().index(row, 0), flags)
                    if not matched:
                        self.table.setCurrentCell(row, 0)
                    matched = True
        if not matched:
            self.table.selectAll()

    def _selected_rows(self) -> list[int]:
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        return [row for row in rows if 0 <= row < len(self.records)]

    def _selected_records(self) -> list[dict]:
        return [self.records[row] for row in self._selected_rows()]

    def _selected_paths(self) -> list[str]:
        return [
            str(self.records[row].get("path", ""))
            for row in self._selected_rows()
            if 0 <= row < len(self.records) and self.records[row].get("path")
        ]


class StimulusDatabaseAnalysisDialog(_DatabaseAnalysisDialogBase):
    def __init__(self, records, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stimulus Response Database")
        self.resize(900, 560)
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
        self._setup_database_table()
        self._set_records(records)
        self.selected_count_label = QLabel()
        self.selected_count_label.setObjectName("MutedText")
        self.table.itemSelectionChanged.connect(self._update_selection_summary)

        analyze = QPushButton("Analyze")
        analyze.setObjectName("PrimaryButton")
        analyze.clicked.connect(self.accept)
        self.psth_button = QPushButton("PSTH")
        self.psth_button.setEnabled(False)
        self.activation_curve_button = QPushButton("Activation curve")
        self.activation_curve_button.setEnabled(False)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)

        controls_frame = QFrame()
        controls_frame.setObjectName("Panel")
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(12, 10, 12, 10)
        controls_layout.setSpacing(8)

        controls_grid = QGridLayout()
        controls_grid.setHorizontalSpacing(10)
        controls_grid.setVerticalSpacing(8)
        controls_grid.addWidget(QLabel("Pre"), 0, 0)
        controls_grid.addWidget(self.pre_ms, 0, 1)
        controls_grid.addWidget(QLabel("Response"), 0, 2)
        controls_grid.addWidget(self.response_ms, 0, 3)
        controls_grid.addWidget(QLabel("Remove tail +/-"), 1, 0)
        controls_grid.addWidget(self.artifact_ms, 1, 1)
        controls_grid.addWidget(QLabel("Pre is baseline before stimulus; Response is the post-stimulus analysis window."), 1, 2, 1, 2)
        controls_layout.addLayout(controls_grid)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        action_row.addWidget(self.psth_button)
        action_row.addWidget(self.activation_curve_button)
        action_row.addWidget(cancel)
        action_row.addWidget(analyze)
        controls_layout.addLayout(action_row)

        layout = QVBoxLayout(self)
        intro = QLabel("Select loaded database files for stimulus-response analysis, then set the visible pre/post window.")
        intro.setObjectName("MutedText")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        layout.addWidget(self.selected_count_label)
        layout.addWidget(self.table, 1)
        layout.addWidget(controls_frame)
        self._update_selection_summary()
        _fix_spinbox_hit_targets(self)

    def _set_records(self, records) -> None:
        super()._set_records(records)
        self._update_selection_summary()

    def _update_selection_summary(self) -> None:
        if not hasattr(self, "selected_count_label"):
            return
        selected = len(self._selected_paths()) if hasattr(self, "table") else 0
        total = len(getattr(self, "records", []))
        self.selected_count_label.setText(f"Selected files: {selected} / {total}")

    def values(self) -> tuple[list[str], float, float, float]:
        return self._selected_paths(), float(self.pre_ms.value()), float(self.response_ms.value()), float(self.artifact_ms.value())

    def set_cached_payload(self, payload: dict | None) -> None:
        self.cached_payload = payload if isinstance(payload, dict) else None
        has_records = bool(self.cached_payload and self.cached_payload.get("records"))
        self.psth_button.setEnabled(has_records)
        self.activation_curve_button.setEnabled(has_records)


class FactorAnalysisDatabaseDialog(_DatabaseAnalysisDialogBase):
    def __init__(self, records, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dynamics Analysis Database")
        self.resize(980, 600)
        self.cached_payload: dict | None = None

        self.bin_ms = QDoubleSpinBox()
        self.bin_ms.setRange(0.5, 200.0)
        self.bin_ms.setDecimals(1)
        self.bin_ms.setValue(10.0)
        self.bin_ms.setSuffix(" ms")
        self.bin_ms.setToolTip("Time bin for burst/activity vectors. Typical range: 5-20 ms.")
        self.window_ms = QDoubleSpinBox()
        self.window_ms.setRange(5.0, 60000.0)
        self.window_ms.setDecimals(1)
        self.window_ms.setValue(300.0)
        self.window_ms.setSuffix(" ms")
        self.window_ms.setToolTip("Analysis window per burst or per non-overlapping segment. Typical range: 100-500 ms.")
        self.analysis_scope = QComboBox()
        self.analysis_scope.addItem("Bursts", "burst")
        self.analysis_scope.addItem("All data windows", "all_windows")
        self.analysis_scope.setToolTip("Bursts uses detected burst intervals; All data windows splits the recording into consecutive windows.")
        self.model_method = QComboBox()
        self.model_method.addItem("Factor Analysis (FA)", "fa")
        self.model_method.addItem("Linear Dynamical System (LDS)", "lds")
        self.model_method.addItem("pi-VAE", "pivae")
        self.model_method.setToolTip("FA estimates latent states independently; LDS adds a temporal latent-state model; pi-VAE fits a conditional Poisson VAE.")
        self.normalize = QComboBox()
        self.normalize.addItem("Channel z-score", "channel_zscore")
        self.normalize.addItem("Log + channel z-score", "log_channel_zscore")
        self.normalize.addItem("Per time total", "per_time_total")
        self.normalize.addItem("None", "none")
        self.normalize.setToolTip("Preprocessing applied before fitting. Channel z-score is usually the safest default.")
        self.latent_dim = QSpinBox()
        self.latent_dim.setRange(1, 128)
        self.latent_dim.setValue(16)
        self.latent_dim.setToolTip("Number of latent factors/states. Typical range: 8-32 for routine exploration.")
        self.min_activity = QDoubleSpinBox()
        self.min_activity.setRange(0.0, 1_000_000.0)
        self.min_activity.setDecimals(2)
        self.min_activity.setValue(1.0)
        self.min_bursts = QSpinBox()
        self.min_bursts.setRange(0, 100000)
        self.min_bursts.setValue(1)
        self.min_var = QDoubleSpinBox()
        self.min_var.setRange(0.0, 1_000_000.0)
        self.min_var.setDecimals(6)
        self.min_var.setValue(0.0)
        self.max_channels = QSpinBox()
        self.max_channels.setRange(1, 20000)
        self.max_channels.setValue(256)
        self.burst_threshold = QDoubleSpinBox()
        self.burst_threshold.setRange(0.5, 20.0)
        self.burst_threshold.setDecimals(1)
        self.burst_threshold.setValue(4.0)
        self.artifact_ms = QDoubleSpinBox()
        self.artifact_ms.setRange(0.0, 1000.0)
        self.artifact_ms.setDecimals(1)
        self.artifact_ms.setSingleStep(0.5)
        self.artifact_ms.setValue(1.0)
        self.artifact_ms.setSuffix(" ms")
        self._setup_database_table()
        self._set_records(records)

        self.options_button = QPushButton("Filters...")
        self.options_button.clicked.connect(self._open_filter_dialog)
        self.options_summary = QLabel()
        self.options_summary.setObjectName("MutedText")
        self.options_summary.setWordWrap(True)

        params_frame = QFrame()
        params_frame.setObjectName("Panel")
        params = QGridLayout(params_frame)
        params.setContentsMargins(12, 10, 12, 10)
        params.setHorizontalSpacing(12)
        params.setVerticalSpacing(8)
        params.addWidget(QLabel("Bin"), 0, 0)
        params.addWidget(self.bin_ms, 0, 1)
        params.addWidget(QLabel("Window"), 0, 2)
        params.addWidget(self.window_ms, 0, 3)
        params.addWidget(QLabel("Scope"), 0, 4)
        params.addWidget(self.analysis_scope, 0, 5)
        params.addWidget(QLabel("Model"), 0, 6)
        params.addWidget(self.model_method, 0, 7)
        params.addWidget(QLabel("Normalize"), 1, 0)
        params.addWidget(self.normalize, 1, 1)
        params.addWidget(QLabel("Latent dim"), 1, 2)
        params.addWidget(self.latent_dim, 1, 3)
        params.addWidget(self.options_button, 1, 4)
        params.addWidget(self.options_summary, 1, 5, 1, 3)

        analyze = QPushButton("Analyze")
        analyze.setObjectName("PrimaryButton")
        analyze.clicked.connect(self.accept)
        self.open_result_button = QPushButton("Open Result")
        self.open_result_button.setEnabled(False)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.open_result_button)
        buttons.addWidget(cancel)
        buttons.addWidget(analyze)

        layout = QVBoxLayout(self)
        intro = QLabel("Select loaded database files, choose FA, LDS, or pi-VAE, then run dynamics analysis.")
        intro.setObjectName("MutedText")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        layout.addWidget(params_frame)
        layout.addWidget(self.table, 1)
        layout.addLayout(buttons)
        self._update_option_summary()
        _fix_spinbox_hit_targets(self)

    def values(self) -> tuple[list[str], dict]:
        return self._selected_paths(), {
            "time_bin_ms": float(self.bin_ms.value()),
            "window_ms": float(self.window_ms.value()),
            "analysis_scope": str(self.analysis_scope.currentData() or "burst"),
            "model_method": str(self.model_method.currentData() or "fa"),
            "normalization": str(self.normalize.currentData()),
            "latent_dim": int(self.latent_dim.value()),
            "min_total_activity": float(self.min_activity.value()),
            "min_active_bursts": int(self.min_bursts.value()),
            "min_variance": float(self.min_var.value()),
            "max_channels": int(self.max_channels.value()),
            "burst_threshold_z": float(self.burst_threshold.value()),
            "artifact_ms": float(self.artifact_ms.value()),
        }

    def set_cached_payload(self, payload: dict | None) -> None:
        self.cached_payload = payload if isinstance(payload, dict) else None
        self.open_result_button.setEnabled(bool(self.cached_payload and self.cached_payload.get("records")))

    def _update_option_summary(self) -> None:
        self.options_summary.setText(
            "Filters: "
            f"min activity {float(self.min_activity.value()):g}, "
            f"min bursts {int(self.min_bursts.value())}, "
            f"min var {float(self.min_var.value()):g}, "
            f"max ch {int(self.max_channels.value())}, "
            f"burst z {float(self.burst_threshold.value()):g}, "
            f"tail +/-{float(self.artifact_ms.value()):g} ms"
        )

    def _open_filter_dialog(self) -> None:
        dialog = QDialog(self)
        _enable_standard_window_controls(dialog)
        dialog.setWindowTitle("Dynamics Analysis Filters")
        layout = QVBoxLayout(dialog)

        intro = QLabel(
            "These parameters control channel screening, burst detection, and stimulus-tail cleanup.\n"
            "Use them only when the default latent-state fit is too noisy or too permissive."
        )
        intro.setObjectName("MutedText")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        min_activity = QDoubleSpinBox()
        min_activity.setRange(self.min_activity.minimum(), self.min_activity.maximum())
        min_activity.setDecimals(self.min_activity.decimals())
        min_activity.setValue(float(self.min_activity.value()))
        min_activity.setToolTip("Remove channels with very low total activity. Typical range: 0-10.")

        min_bursts = QSpinBox()
        min_bursts.setRange(self.min_bursts.minimum(), self.min_bursts.maximum())
        min_bursts.setValue(int(self.min_bursts.value()))
        min_bursts.setToolTip("Require a channel to participate in at least this many bursts/windows.")

        min_var = QDoubleSpinBox()
        min_var.setRange(self.min_var.minimum(), self.min_var.maximum())
        min_var.setDecimals(self.min_var.decimals())
        min_var.setValue(float(self.min_var.value()))
        min_var.setToolTip("Remove channels with nearly constant activity vectors. Typical range: 0-0.1.")

        max_channels = QSpinBox()
        max_channels.setRange(self.max_channels.minimum(), self.max_channels.maximum())
        max_channels.setValue(int(self.max_channels.value()))
        max_channels.setToolTip("Upper bound on fitted channels for speed and numerical stability.")

        burst_threshold = QDoubleSpinBox()
        burst_threshold.setRange(self.burst_threshold.minimum(), self.burst_threshold.maximum())
        burst_threshold.setDecimals(self.burst_threshold.decimals())
        burst_threshold.setValue(float(self.burst_threshold.value()))
        burst_threshold.setToolTip("Burst detection z-threshold when analysis scope is Bursts. Typical range: 3-6.")

        artifact_ms = QDoubleSpinBox()
        artifact_ms.setRange(self.artifact_ms.minimum(), self.artifact_ms.maximum())
        artifact_ms.setDecimals(self.artifact_ms.decimals())
        artifact_ms.setSingleStep(self.artifact_ms.singleStep())
        artifact_ms.setValue(float(self.artifact_ms.value()))
        artifact_ms.setSuffix(" ms")
        artifact_ms.setToolTip("Remove spikes within +/- this range around stimulation artifacts.")

        form.addRow("Min total activity", min_activity)
        form.addRow("Min active bursts/windows", min_bursts)
        form.addRow("Min variance", min_var)
        form.addRow("Max fitted channels", max_channels)
        form.addRow("Burst z-threshold", burst_threshold)
        form.addRow("Artifact tail window", artifact_ms)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        apply_button = QPushButton("Apply")
        buttons.addWidget(cancel)
        buttons.addWidget(apply_button)
        layout.addLayout(buttons)
        cancel.clicked.connect(dialog.reject)
        apply_button.clicked.connect(dialog.accept)
        _fix_spinbox_hit_targets(dialog)

        if dialog.exec() != QDialog.Accepted:
            return
        self.min_activity.setValue(float(min_activity.value()))
        self.min_bursts.setValue(int(min_bursts.value()))
        self.min_var.setValue(float(min_var.value()))
        self.max_channels.setValue(int(max_channels.value()))
        self.burst_threshold.setValue(float(burst_threshold.value()))
        self.artifact_ms.setValue(float(artifact_ms.value()))
        self._update_option_summary()


class GenericAnalysisDialog(_DatabaseAnalysisDialogBase):
    def __init__(self, records, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Analysis")
        self.resize(980, 620)
        self.cached_payload: dict | None = None

        self.analysis_type = QComboBox()
        self.analysis_type.addItem("Vector firing rate (Hz)", "firing_rate_vector")
        self.analysis_type.addItem("Vector spike count", "spike_count_vector")
        self.analysis_type.setToolTip("Basic function applied after file/channel/time-window selection.")

        self.time_windows = QLineEdit("full")
        self.time_windows.setPlaceholderText("full, or 0-10, 10-20")
        self.time_windows.setToolTip("Comma-separated windows in seconds. Use full for the whole recording.")
        self.channels = QLineEdit()
        self.channels.setPlaceholderText("blank = all channels; e.g. chan12, chan13")
        self.channels.setToolTip("Comma-separated channel labels. Matching is case-insensitive and accepts base channel names.")
        self.dataset_name = QLineEdit()
        self.dataset_name.setPlaceholderText("optional processed dataset name")
        self.x_values = QLineEdit()
        self.x_values.setPlaceholderText("optional x values, e.g. 0, 10, 20")
        self.x_label = QLineEdit("Time window")
        self.y_label = QLineEdit("")
        self.plot_mode = QComboBox()
        self.plot_mode.addItem("Auto", "auto")
        self.plot_mode.addItem("Line", "line")
        self.plot_mode.addItem("Bar", "bar")

        self._setup_database_table()
        self._set_records(records)

        controls_frame = QFrame()
        controls_frame.setObjectName("Panel")
        controls = QGridLayout(controls_frame)
        controls.setContentsMargins(12, 10, 12, 10)
        controls.setHorizontalSpacing(10)
        controls.setVerticalSpacing(8)
        controls.addWidget(QLabel("Basic function"), 0, 0)
        controls.addWidget(self.analysis_type, 0, 1)
        controls.addWidget(QLabel("Time windows"), 0, 2)
        controls.addWidget(self.time_windows, 0, 3, 1, 3)
        controls.addWidget(QLabel("Channels"), 1, 0)
        controls.addWidget(self.channels, 1, 1, 1, 5)
        controls.addWidget(QLabel("Name"), 2, 0)
        controls.addWidget(self.dataset_name, 2, 1, 1, 2)
        controls.addWidget(QLabel("Plot"), 2, 3)
        controls.addWidget(self.plot_mode, 2, 4)
        controls.addWidget(QLabel("X values"), 3, 0)
        controls.addWidget(self.x_values, 3, 1, 1, 2)
        controls.addWidget(QLabel("X label"), 3, 3)
        controls.addWidget(self.x_label, 3, 4)
        controls.addWidget(QLabel("Y label"), 4, 3)
        controls.addWidget(self.y_label, 4, 4)
        note = QLabel("Results are saved into the processed-data database and can be plotted immediately.")
        note.setObjectName("MutedText")
        note.setWordWrap(True)
        controls.addWidget(note, 4, 0, 1, 3)

        intro = QLabel(
            "Build a custom dataset from loaded files by choosing files, channels, and time windows, "
            "then run a basic analysis function such as firing-rate vectorization."
        )
        intro.setObjectName("MutedText")
        intro.setWordWrap(True)

        analyze = QPushButton("Run Custom Analysis")
        analyze.setObjectName("PrimaryButton")
        analyze.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(analyze)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(controls_frame)
        layout.addWidget(self.table, 1)
        layout.addLayout(buttons)
        _fix_spinbox_hit_targets(self)

    def _set_records(self, records) -> None:
        super()._set_records(records)

    def values(self) -> tuple[list[str], dict]:
        return self._selected_paths(), {
            "analysis_kind": "custom_basic",
            "analysis_type": str(self.analysis_type.currentData() or "firing_rate_vector"),
            "time_windows": self.time_windows.text().strip(),
            "channels": self.channels.text().strip(),
            "display_name": self.dataset_name.text().strip(),
            "x_values": self.x_values.text().strip(),
            "x_label": self.x_label.text().strip() or "Time window",
            "y_label": self.y_label.text().strip(),
            "plot_mode": str(self.plot_mode.currentData() or "auto"),
        }


class StimulusResponseInputDialog(AppDialog):
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


class MultiFileFactorAnalysisInputDialog(AppDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multi-file Factor Analysis")
        self.resize(940, 560)
        self.paths: list[str] = []
        self.cached_payload: dict | None = None

        self.bin_ms = QDoubleSpinBox()
        self.bin_ms.setRange(0.5, 200.0)
        self.bin_ms.setDecimals(1)
        self.bin_ms.setValue(10.0)
        self.bin_ms.setSuffix(" ms")
        self.window_ms = QDoubleSpinBox()
        self.window_ms.setRange(5.0, 60000.0)
        self.window_ms.setDecimals(1)
        self.window_ms.setValue(300.0)
        self.window_ms.setSuffix(" ms")
        self.analysis_scope = QComboBox()
        self.analysis_scope.addItem("Bursts", "burst")
        self.analysis_scope.addItem("All data windows", "all_windows")
        self.analysis_scope.setToolTip("Use detected bursts, or split each file into non-overlapping windows.")
        self.normalize = QComboBox()
        self.normalize.addItem("Channel z-score", "channel_zscore")
        self.normalize.addItem("Log + channel z-score", "log_channel_zscore")
        self.normalize.addItem("Per time total", "per_time_total")
        self.normalize.addItem("None", "none")
        self.latent_dim = QSpinBox()
        self.latent_dim.setRange(1, 128)
        self.latent_dim.setValue(16)
        self.min_activity = QDoubleSpinBox()
        self.min_activity.setRange(0.0, 1_000_000.0)
        self.min_activity.setDecimals(2)
        self.min_activity.setValue(1.0)
        self.min_bursts = QSpinBox()
        self.min_bursts.setRange(0, 100000)
        self.min_bursts.setValue(1)
        self.min_var = QDoubleSpinBox()
        self.min_var.setRange(0.0, 1_000_000.0)
        self.min_var.setDecimals(6)
        self.min_var.setValue(0.0)
        self.max_channels = QSpinBox()
        self.max_channels.setRange(1, 20000)
        self.max_channels.setValue(256)
        self.burst_threshold = QDoubleSpinBox()
        self.burst_threshold.setRange(0.5, 20.0)
        self.burst_threshold.setDecimals(1)
        self.burst_threshold.setValue(4.0)
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
        self.open_result_button = QPushButton("Open Result")
        self.open_result_button.setEnabled(False)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)

        params = QGridLayout()
        params.addWidget(QLabel("Bin"), 0, 0)
        params.addWidget(self.bin_ms, 0, 1)
        params.addWidget(QLabel("Window"), 0, 2)
        params.addWidget(self.window_ms, 0, 3)
        params.addWidget(QLabel("Scope"), 0, 4)
        params.addWidget(self.analysis_scope, 0, 5)
        params.addWidget(QLabel("Normalize"), 0, 6)
        params.addWidget(self.normalize, 0, 7)
        params.addWidget(QLabel("Latent dim"), 0, 8)
        params.addWidget(self.latent_dim, 0, 9)
        params.addWidget(QLabel("Min activity"), 1, 0)
        params.addWidget(self.min_activity, 1, 1)
        params.addWidget(QLabel("Min bursts"), 1, 2)
        params.addWidget(self.min_bursts, 1, 3)
        params.addWidget(QLabel("Min var"), 1, 4)
        params.addWidget(self.min_var, 1, 5)
        params.addWidget(QLabel("Max fit ch"), 1, 6)
        params.addWidget(self.max_channels, 1, 7)
        params.addWidget(QLabel("Burst z"), 2, 0)
        params.addWidget(self.burst_threshold, 2, 1)
        params.addWidget(QLabel("Remove tail +/-"), 2, 2)
        params.addWidget(self.artifact_ms, 2, 3)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(add_files)
        buttons.addWidget(add_folder)
        buttons.addWidget(remove)
        buttons.addWidget(self.open_result_button)
        buttons.addWidget(cancel)
        buttons.addWidget(analyze)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Add multiple spike-event files or folders. Each file is fit independently with Factor Analysis, then W is aligned to compare loading similarity."))
        layout.addLayout(params)
        layout.addWidget(self.table, 1)
        layout.addLayout(buttons)
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
            "Add multi-file FA data",
            "data",
            "Spike files (*.nev *.spk *.h5 *.hdf5 *.npz);;All files (*)",
        )
        self._add_paths(paths)

    def _add_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Add multi-file FA folder", "data")
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

    def values(self) -> tuple[list[str], dict]:
        return list(self.paths), {
            "time_bin_ms": float(self.bin_ms.value()),
            "window_ms": float(self.window_ms.value()),
            "analysis_scope": str(self.analysis_scope.currentData() or "burst"),
            "normalization": str(self.normalize.currentData()),
            "latent_dim": int(self.latent_dim.value()),
            "min_total_activity": float(self.min_activity.value()),
            "min_active_bursts": int(self.min_bursts.value()),
            "min_variance": float(self.min_var.value()),
            "max_channels": int(self.max_channels.value()),
            "burst_threshold_z": float(self.burst_threshold.value()),
            "artifact_ms": float(self.artifact_ms.value()),
        }

    def set_cached_payload(self, payload: dict | None) -> None:
        self.cached_payload = payload if isinstance(payload, dict) else None
        self.open_result_button.setEnabled(bool(self.cached_payload and self.cached_payload.get("records")))


class StimulusChannelMapWindow(AppDialog):
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


class StimulusResponseWindow(AppDialog):
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
        self.status.setWordWrap(True)
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
        self.display_window_ms.setToolTip("Visible raster window. Default shows pre-stimulus and post-stimulus together.")
        self.row_order_combo = QComboBox()
        self.row_order_combo.addItem("Local response", "local_response")
        self.row_order_combo.addItem("Electrode sequence", "electrode")
        self.row_order_combo.currentIndexChanged.connect(lambda *_: self._draw_rasters())
        self.row_order_combo.setToolTip("Local response groups nearby and behaviorally similar channels together.")
        self.view_settings_button = QPushButton("View settings...")
        self.view_settings_button.clicked.connect(self._open_view_settings_dialog)
        self.view_settings_summary = QLabel()
        self.view_settings_summary.setObjectName("MutedText")
        self.view_settings_summary.setWordWrap(True)
        self.lasso_channels = QCheckBox("Lasso channels")
        self.lasso_channels.toggled.connect(lambda *_: self._refresh_raster_lassos())
        self.map_button = QPushButton("Channel map")
        self.map_button.clicked.connect(self._show_channel_map)
        self.raster_canvas = FigureCanvas(Figure(figsize=(11, 6.4), tight_layout=True))
        self.raster_canvas.mpl_connect("scroll_event", self._raster_scrolled)
        self.error_box = QTextEdit()
        self.error_box.setReadOnly(True)
        self.error_box.setMaximumHeight(90)

        controls_frame = QFrame()
        controls_frame.setObjectName("Panel")
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(12, 10, 12, 10)
        controls_layout.setSpacing(8)

        controls_grid = QGridLayout()
        controls_grid.setHorizontalSpacing(10)
        controls_grid.setVerticalSpacing(8)
        controls_grid.addWidget(QLabel("Left file"), 0, 0)
        controls_grid.addWidget(self.left_file_combo, 0, 1)
        controls_grid.addWidget(QLabel("Right file"), 0, 2)
        controls_grid.addWidget(self.right_file_combo, 0, 3)
        controls_grid.addWidget(QLabel("Window"), 1, 0)
        controls_grid.addWidget(self.display_window_ms, 1, 1)
        controls_grid.addWidget(QLabel("Row order"), 1, 2)
        controls_grid.addWidget(self.row_order_combo, 1, 3)
        controls_grid.addWidget(self.view_settings_button, 2, 0)
        controls_grid.addWidget(self.view_settings_summary, 2, 1, 1, 3)
        controls_layout.addLayout(controls_grid)

        helper_row = QHBoxLayout()
        helper_row.addWidget(self.lasso_channels)
        helper_row.addWidget(self.map_button)
        helper_row.addWidget(QLabel("Scroll on either raster to switch trial."))
        helper_row.addStretch(1)
        controls_layout.addLayout(helper_row)

        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(controls_frame)
        layout.addWidget(self.raster_canvas, 1)
        layout.addWidget(QLabel("Skipped files / warnings"))
        layout.addWidget(self.error_box)

        self._update_view_settings_summary()
        self._populate()
        self.showMaximized()
        _fix_spinbox_hit_targets(self)

    def _update_view_settings_summary(self) -> None:
        self.view_settings_summary.setText(
            f"Visible window {float(self.display_window_ms.value()):g} ms; "
            f"row order = {'local response' if self.row_order_combo.currentData() == 'local_response' else 'electrode sequence'}."
        )

    def _open_view_settings_dialog(self) -> None:
        dialog = QDialog(self)
        _enable_standard_window_controls(dialog)
        dialog.setWindowTitle("Stimulus Response View Settings")
        layout = QVBoxLayout(dialog)

        intro = QLabel(
            "These settings only affect how rasters are displayed and compared.\n"
            "Use the main panel for file selection; use this dialog for view behavior."
        )
        intro.setObjectName("MutedText")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()

        display_window = QDoubleSpinBox()
        display_window.setRange(self.display_window_ms.minimum(), self.display_window_ms.maximum())
        display_window.setDecimals(self.display_window_ms.decimals())
        display_window.setSingleStep(self.display_window_ms.singleStep())
        display_window.setValue(float(self.display_window_ms.value()))
        display_window.setSuffix(" ms")
        display_window.setToolTip("Visible time span on the raster. Typically set to cover pre + after response.")

        row_order = QComboBox()
        row_order.addItem("Local response", "local_response")
        row_order.addItem("Electrode sequence", "electrode")
        row_order.setCurrentIndex(row_order.findData(self.row_order_combo.currentData()))
        row_order.setToolTip("Local response groups channels by spatial/response similarity; electrode sequence preserves map order.")

        form.addRow("Visible time window", display_window)
        form.addRow("Channel row order", row_order)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        apply_button = QPushButton("Apply")
        buttons.addWidget(cancel)
        buttons.addWidget(apply_button)
        layout.addLayout(buttons)
        cancel.clicked.connect(dialog.reject)
        apply_button.clicked.connect(dialog.accept)

        if dialog.exec() != QDialog.Accepted:
            return
        self.display_window_ms.setValue(float(display_window.value()))
        selected_index = self.row_order_combo.findData(row_order.currentData())
        if selected_index >= 0:
            self.row_order_combo.setCurrentIndex(selected_index)
        self._update_view_settings_summary()

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


class StimulusPSTHWindow(AppDialog):
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


class StimulusActivationCurveWindow(AppDialog):
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


class AutoSortingDialog(AppDialog):
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


class MaxwellFootprintDialog(AppDialog):
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


class SettingsDialog(AppDialog):
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


class PlotWindow(AppDialog):
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
        self._background_cache = None
        self._background_cache_key = None
        self._raster_cache = None
        self._raster_cache_key = None
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
        self._background_cache = None
        self._background_cache_key = None
        self._raster_cache = None
        self._raster_cache_key = None
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
        self._background_cache = None
        self._background_cache_key = None
        self._raster_cache = None
        self._raster_cache_key = None
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
        self._background_cache = None
        self._background_cache_key = None
        self._raster_cache = None
        self._raster_cache_key = None
        self.update()

    def set_stim_times(self, stim_times) -> None:
        values = np.asarray(stim_times if stim_times is not None else [], dtype=float)
        values = values[np.isfinite(values)]
        values.sort()
        self.stim_times = values
        self._background_cache = None
        self._background_cache_key = None
        self._raster_cache = None
        self._raster_cache_key = None
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

        stim_visible = False
        if self.stim_times.size:
            stim_lo = int(np.searchsorted(self.stim_times, start, side="left"))
            stim_hi = int(np.searchsorted(self.stim_times, stop, side="right"))
            stim_visible = bool(self.stim_times[stim_lo:stim_hi].size)

        cache_key = (
            int(round(start * 1000)),
            int(round(stop * 1000)),
            int(round(self.grid_step * 1000)),
            tuple((int(round(a * 1000)), int(round(b * 1000))) for a, b in self.burst_intervals),
            tuple(int(round(v * 1000)) for v in self.stim_times[:2000]),
            int(left),
            int(top),
            int(plot_width),
            int(plot_height),
            int(stim_visible),
        )
        if self._background_cache_key != cache_key or self._background_cache is None:
            background = QImage(self.size(), QImage.Format.Format_ARGB32_Premultiplied)
            background.fill(QColor("#ffffff"))
            bg = QPainter(background)
            bg.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            for burst_start, burst_stop in burst_iter:
                overlap_start = max(start, burst_start)
                overlap_stop = min(stop, burst_stop)
                if overlap_stop <= overlap_start:
                    continue
                x0 = left + (overlap_start - start) / self.window_duration * plot_width
                x1 = left + (overlap_stop - start) / self.window_duration * plot_width
                bg.fillRect(QRectF(x0, top, max(1.0, x1 - x0), plot_height), burst_brush)
                burst_visible = True

            bg.setPen(QPen(QColor("#d7deea"), 1))
            grid = np.arange(np.ceil(start / self.grid_step) * self.grid_step, stop + self.grid_step * 0.5, self.grid_step)
            for tick in grid:
                x = left + (tick - start) / self.window_duration * plot_width
                bg.drawLine(int(x), top, int(x), top + plot_height)

            bg.setPen(QPen(QColor("#1f2937"), 1))
            bg.drawRect(left, top, plot_width, plot_height)

            if stim_visible:
                visible_stim = self.stim_times[stim_lo:stim_hi]
                stim_pen = QPen(QColor("#f97316"), 1)
                stim_pen.setCosmetic(True)
                bg.setPen(stim_pen)
                bg.setBrush(QColor("#f97316"))
                for stim_time in visible_stim[:2000]:
                    x = left + (float(stim_time) - start) / self.window_duration * plot_width
                    bg.drawLine(QLineF(float(x), top, float(x), top + plot_height))
                    triangle = QPolygonF(
                        [
                            QPointF(float(x), top + 1),
                            QPointF(float(x) - 4.0, top + 9.0),
                            QPointF(float(x) + 4.0, top + 9.0),
                        ]
                    )
                    bg.drawPolygon(triangle)

            if burst_visible:
                bg.setFont(QFont("Segoe UI", 8))
                bg.setPen(QPen(QColor("#991b1b"), 1))
                bg.setBrush(QColor("#fecaca"))
                legend_x = left + plot_width - (178 if stim_visible else 104)
                bg.drawRect(int(legend_x), top + 8, 12, 8)
                bg.drawText(int(legend_x + 18), top + 4, 86, 18, Qt.AlignmentFlag.AlignLeft, "burst")
            if stim_visible:
                bg.setFont(QFont("Segoe UI", 8))
                bg.setPen(QPen(QColor("#f97316"), 1))
                bg.setBrush(QColor("#f97316"))
                legend_x = left + plot_width - 82
                bg.drawLine(QLineF(float(legend_x), top + 8.0, float(legend_x), top + 17.0))
                bg.drawPolygon(
                    QPolygonF(
                        [
                            QPointF(float(legend_x), top + 5.0),
                            QPointF(float(legend_x) - 4.0, top + 12.0),
                            QPointF(float(legend_x) + 4.0, top + 12.0),
                        ]
                    )
                )
                bg.drawText(int(legend_x + 12), top + 4, 64, 18, Qt.AlignmentFlag.AlignLeft, "stim")

            self._background_cache = background
            self._background_cache_key = cache_key
            bg.end()

        painter.drawImage(0, 0, self._background_cache)

        visible_series = self._visible_spike_series()
        channel_count = len(visible_series)
        if channel_count == 0:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No spike data")
            painter.end()
            return

        row_step = plot_height / max(channel_count, 1)
        raster_cache_key = (
            int(round(start * 1000)),
            int(round(stop * 1000)),
            int(round(self.grid_step * 1000)),
            int(self.row_offset),
            int(self.visible_row_count),
            int(left),
            int(top),
            int(plot_width),
            int(plot_height),
            int(self.width()),
            int(self.height()),
            self.y_axis_label,
            tuple((str(channel), id(times), int(np.asarray(times).size)) for channel, times in visible_series),
        )
        if self._raster_cache_key != raster_cache_key or self._raster_cache is None:
            raster_image = QImage(self.size(), QImage.Format.Format_ARGB32_Premultiplied)
            raster_image.fill(Qt.GlobalColor.transparent)
            rp = QPainter(raster_image)
            rp.setRenderHint(QPainter.RenderHint.Antialiasing, False)

            label_font = QFont("Segoe UI", 8)
            rp.setFont(label_font)
            label_height = max(14, rp.fontMetrics().height() + 2)
            rp.setPen(QPen(QColor("#64748b"), 1))
            label_stride = self._label_stride_for_rows(channel_count, plot_height, label_height)
            for row, (channel, _) in enumerate(visible_series):
                if row % label_stride != 0:
                    continue
                y = self._row_center(top, plot_height, row, channel_count)
                rp.drawText(
                    QRectF(6, y - label_height / 2, left - 14, label_height),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    channel,
                )

            default_spike = QColor("#2563eb")
            default_spike.setAlpha(235)
            spike_pen = QPen(default_spike, 1)
            spike_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            rp.setPen(spike_pen)
            for row, (_channel, times) in enumerate(visible_series):
                if times.size == 0:
                    continue
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
                rp.drawLines([QLineF(float(xi), y0, float(xi), y1) for xi in xi_values])

            rp.setPen(QPen(QColor("#111827"), 1))
            rp.drawText(left, self.height() - 26, plot_width, 18, Qt.AlignmentFlag.AlignCenter, "Time (s)")
            rp.save()
            rp.translate(18, top + plot_height / 2)
            rp.rotate(-90)
            rp.drawText(-plot_height // 2, -4, plot_height, 18, Qt.AlignmentFlag.AlignCenter, self.y_axis_label)
            rp.restore()

            tick_font = QFont("Segoe UI", 8)
            rp.setFont(tick_font)
            rp.setPen(QPen(QColor("#475569"), 1))
            grid = np.arange(np.ceil(start / self.grid_step) * self.grid_step, stop + self.grid_step * 0.5, self.grid_step)
            max_labels = max(2, plot_width // 90)
            label_stride = max(1, int(np.ceil(len(grid) / max_labels))) if len(grid) else 1
            for index, tick in enumerate(grid):
                if index % label_stride != 0 and index != len(grid) - 1:
                    continue
                x = left + (tick - start) / self.window_duration * plot_width
                rp.drawText(
                    int(x - 38),
                    top + plot_height + 4,
                    76,
                    16,
                    Qt.AlignmentFlag.AlignCenter,
                    _format_time_tick(float(tick)),
                )
            rp.end()
            self._raster_cache = raster_image
            self._raster_cache_key = raster_cache_key

        painter.drawImage(0, 0, self._raster_cache)

        selected_spike = QColor("#dc2626")
        selected_spike.setAlpha(245)
        selected_pen = QPen(selected_spike, 1)
        selected_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(selected_pen)
        for row, (channel, times) in enumerate(visible_series):
            if channel != self.selected_channel or times.size == 0:
                continue
            lo = int(np.searchsorted(times, start, side="left"))
            hi = int(np.searchsorted(times, stop, side="right"))
            if hi <= lo:
                break
            visible = times[lo:hi]
            xs = left + (visible - start) / self.window_duration * plot_width
            y_center = self._row_center(top, plot_height, row, channel_count)
            spike_half_height = max(1.0, min(row_step * 0.36, 5.5))
            y0 = int(y_center - spike_half_height)
            y1 = int(y_center + spike_half_height)
            xi_values = np.rint(xs).astype(np.int32, copy=False)
            xi_values = xi_values[(xi_values >= left) & (xi_values <= left + plot_width)]
            if xi_values.size:
                xi_values = np.unique(xi_values)
                painter.drawLines([QLineF(float(xi), y0, float(xi), y1) for xi in xi_values])
            break

        if self.playhead_time is not None and start <= self.playhead_time <= stop:
            x = left + (float(self.playhead_time) - start) / self.window_duration * plot_width
            playhead_pen = QPen(QColor("#0f172a"), 2)
            playhead_pen.setCosmetic(True)
            painter.setPen(playhead_pen)
            painter.drawLine(QLineF(float(x), top, float(x), top + plot_height))

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
        self._raw_counts = {}
        self._target_counts = {}
        self._display_counts = {}
        self._target_raw_counts = {}
        self._display_raw_counts = {}
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
        raw_counts = {str(channel): float(count) for channel, count in counts.items()}
        normalized, well_counts = self._normalize_count_payload(counts)
        if raw_counts == self._target_raw_counts and normalized == self._target_counts and well_counts == self._target_well_counts:
            return
        self._target_raw_counts = raw_counts
        self._target_counts = normalized
        self._target_well_counts = well_counts
        self._raw_counts = dict(raw_counts)
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
        self._display_raw_counts, raw_delta = self._blend_count_dict(self._display_raw_counts, self._target_raw_counts, alpha)
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
        if max(raw_delta, flat_delta, nested_delta) <= 0.03:
            self._display_raw_counts = dict(self._target_raw_counts)
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
        display_counts = self._display_raw_counts or self._raw_counts or self._display_counts or self.counts
        use_combined_counts = bool(self._coordinate_entries()) or self.channel_map.rows != 8 or self.channel_map.cols != 8
        if display_well_counts and not use_combined_counts:
            selected_well = next(iter(self.active_wells)) if len(self.active_wells) == 1 else sorted(display_well_counts)[0]
            active_counts = display_well_counts.get(selected_well, {})
        else:
            active_counts = display_counts
        max_count = self.scale_max_count or max(active_counts.values(), default=0)

        rect = self._map_rect_for_channel_map(margin, map_width, map_height)
        self._draw_well_heatmap(painter, rect, active_counts, max_count)
        self._draw_plate_well_overlay(painter, rect)
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
        exact_lookup = {}
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
                exact_lookup.setdefault(alias, (str(electrode), payload, float(x_um), float(y_um), index))
                keys = [normalize_channel_name(alias)]
                if "_" in alias:
                    keys.append(normalize_channel_name(alias.split("_", 1)[1]))
                for key in keys:
                    if key and key not in lookup:
                        lookup[key] = (str(electrode), payload, float(x_um), float(y_um), index)
            if payload.get("routed") or payload.get("channel"):
                routed_entries.append((str(electrode), payload, float(x_um), float(y_um), index))
        lookup["__exact__"] = exact_lookup
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

    def _count_for_entry(self, counts: dict[str, int], payload: dict[str, object], electrode: str) -> float:
        candidates = [
            str(payload.get("channel") or "").strip(),
            str(electrode or "").strip(),
        ]
        raw_aliases = payload.get("aliases", [])
        if isinstance(raw_aliases, (list, tuple)):
            candidates.extend(str(alias).strip() for alias in raw_aliases)
        exact_lookup = {}
        lookup = self._coordinate_lookup()
        if isinstance(lookup, dict):
            exact_lookup = lookup.get("__exact__", {})
        for candidate in candidates:
            if not candidate:
                continue
            if candidate in counts:
                return float(counts[candidate])
            if candidate in exact_lookup and candidate in counts:
                return float(counts[candidate])
        for candidate in candidates:
            if not candidate:
                continue
            key = normalize_channel_name(candidate)
            if key and key in counts:
                return float(counts[key])
            if "_" in candidate:
                suffix = candidate.split("_", 1)[1]
                key = normalize_channel_name(suffix)
                if key and key in counts:
                    return float(counts[key])
        return 0.0

    def _well_block_rects(self, rect: QRectF):
        entries = self._coordinate_entries()
        if not entries:
            return []
        blocks: dict[str, dict[str, object]] = {}
        for _electrode, payload, x_um, y_um in entries:
            if not isinstance(payload, dict):
                continue
            well = str(payload.get("well") or "").strip()
            if not well:
                continue
            block = blocks.setdefault(
                well,
                {
                    "xs": [],
                    "ys": [],
                    "well_grid_row": payload.get("well_grid_row"),
                    "well_grid_col": payload.get("well_grid_col"),
                },
            )
            block["xs"].append(float(x_um))
            block["ys"].append(float(y_um))
        if len(blocks) < 2:
            return []

        xs_all = np.asarray([x for block in blocks.values() for x in block["xs"]], dtype=float)
        ys_all = np.asarray([y for block in blocks.values() for y in block["ys"]], dtype=float)
        if xs_all.size == 0 or ys_all.size == 0:
            return []
        unique_xs = np.unique(np.round(xs_all, 6))
        unique_ys = np.unique(np.round(ys_all, 6))
        step_x = float(np.median(np.diff(unique_xs))) if unique_xs.size > 1 else 1.0
        step_y = float(np.median(np.diff(unique_ys))) if unique_ys.size > 1 else 1.0
        bounds = self._coordinate_bounds()
        if bounds is None:
            return []

        output = []
        for well, block in blocks.items():
            xs = np.asarray(block["xs"], dtype=float)
            ys = np.asarray(block["ys"], dtype=float)
            left_top = self._coordinate_point(rect, bounds, float(xs.min() - step_x * 0.55), float(ys.min() - step_y * 0.55))
            right_bottom = self._coordinate_point(rect, bounds, float(xs.max() + step_x * 0.55), float(ys.max() + step_y * 0.55))
            block_rect = QRectF(
                min(left_top.x(), right_bottom.x()),
                min(left_top.y(), right_bottom.y()),
                abs(right_bottom.x() - left_top.x()),
                abs(right_bottom.y() - left_top.y()),
            )
            output.append(
                (
                    well,
                    block_rect,
                    int(block.get("well_grid_row") or 0),
                    int(block.get("well_grid_col") or 0),
                )
            )
        return sorted(output, key=lambda item: (item[2], item[3], _well_sort_key(item[0])))

    def _draw_plate_well_overlay(self, painter: QPainter, rect: QRectF) -> None:
        block_rects = self._well_block_rects(rect)
        if not block_rects:
            return
        painter.save()
        painter.setClipRect(rect.adjusted(-2.0, -18.0, 2.0, 2.0))
        label_font = QFont("Segoe UI", 8)
        label_font.setBold(True)
        painter.setFont(label_font)
        border_pen = QPen(QColor(226, 232, 240, 150), 1.1)
        border_pen.setCosmetic(True)
        label_pen = QPen(QColor(241, 245, 249), 1)
        for well, block_rect, _row, _col in block_rects:
            painter.setPen(border_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(block_rect, 3, 3)
            label_rect = QRectF(block_rect.left(), max(rect.top() - 18.0, block_rect.top() - 18.0), block_rect.width(), 14.0)
            painter.setPen(label_pen)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, well)
        painter.restore()

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
            has_activity = self._count_for_entry(active_counts, payload, electrode) > 0.0
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
        exact_lookup = lookup.get("__exact__", {}) if isinstance(lookup, dict) else {}
        used_entries = set()
        for electrode, payload in self.channel_map.electrodes.items():
            if not isinstance(payload, dict):
                continue
            count = self._count_for_entry(active_counts, payload, str(electrode))
            if count <= 0:
                continue
            direct_entry = None
            channel_name = str(payload.get("channel") or "").strip()
            if channel_name:
                direct_entry = exact_lookup.get(channel_name)
            if direct_entry is None:
                direct_entry = exact_lookup.get(str(electrode))
            if direct_entry is None:
                for alias in payload.get("aliases", []) if isinstance(payload.get("aliases", []), (list, tuple)) else []:
                    direct_entry = exact_lookup.get(str(alias).strip())
                    if direct_entry is not None:
                        break
            if direct_entry is None:
                continue
            _electrode, _payload, x_um, y_um, index = direct_entry
            if index in used_entries:
                continue
            used_entries.add(index)
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
        active_counts = dict(counts)
        if not active_counts:
            active_counts = well_counts[sorted(well_counts)[0]] if well_counts else normalized
        max_count = max(0, int(scale_max_count)) or max(active_counts.values(), default=0)
        resolution = max(32, int(resolution))
        if self.channel_map is None or max_count <= 0:
            return np.zeros((resolution, resolution, 3), dtype=np.uint8)
        field = self._continuous_heatmap_field(active_counts, max_count, resolution)
        if not np.any(field > 0):
            field = self._fallback_heatmap_field(active_counts, max_count, resolution)
        return self._heatmap_field_rgb(field)


class IBIWindow(AppDialog):
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


class ISIWindow(AppDialog):
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


class BurstDelayWindow(AppDialog):
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
        self.delay_mode_combo.setToolTip("Delay definition. First spike is the strictest; all spikes is the most permissive.")

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
        self.advanced_button = QPushButton("Analysis settings...")
        self.advanced_button.clicked.connect(self._open_delay_settings_dialog)
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
        self.raster_align_combo.setToolTip("Burst onset keeps a common time origin; Source anchored aligns source spikes to a visible offset.")

        self.summary = QLabel()
        self.summary.setObjectName("MutedText")
        self.parameter_hint = QLabel()
        self.parameter_hint.setObjectName("MutedText")
        self.parameter_hint.setWordWrap(True)
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

        controls_frame = QFrame()
        controls_frame.setObjectName("Panel")
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(12, 10, 12, 10)
        controls_layout.setSpacing(8)

        controls_grid = QGridLayout()
        controls_grid.setHorizontalSpacing(10)
        controls_grid.setVerticalSpacing(8)
        controls_grid.addWidget(QLabel("Delay mode"), 0, 0)
        controls_grid.addWidget(self.delay_mode_combo, 0, 1)
        controls_grid.addWidget(QLabel("Source"), 0, 2)
        controls_grid.addWidget(self.reference_combo, 0, 3)
        controls_grid.addWidget(QLabel("Target"), 0, 4)
        controls_grid.addWidget(self.target_combo, 0, 5)
        controls_grid.addWidget(QLabel("Raster align"), 1, 0)
        controls_grid.addWidget(self.raster_align_combo, 1, 1)
        controls_grid.addWidget(self.advanced_button, 1, 2)
        controls_grid.addWidget(self.parameter_hint, 1, 3, 1, 3)
        controls_layout.addLayout(controls_grid)

        helper_row = QHBoxLayout()
        helper_row.setSpacing(8)
        helper_row.addWidget(self.manual_pair_check)
        helper_row.addWidget(QLabel("Map click"))
        helper_row.addWidget(self.map_pick_combo)
        helper_row.addWidget(self.significance_button)
        helper_row.addWidget(self.analyze_button)
        helper_row.addStretch(1)
        helper_row.addWidget(QLabel("Analyze runs only after you click Analyze."))
        controls_layout.addLayout(helper_row)

        right_plots = QVBoxLayout()
        right_plots.addWidget(self.hist_canvas, 1)
        right_plots.addWidget(self.delay_raster_canvas, 1)
        right_plots.addWidget(self.waveform_canvas, 1)
        plot_area = QHBoxLayout()
        plot_area.addWidget(self.map_canvas, 3)
        plot_area.addLayout(right_plots, 2)

        layout = QVBoxLayout(self)
        layout.addWidget(controls_frame)
        layout.addWidget(self.summary)
        layout.addLayout(plot_area, 1)
        self._update_delay_parameter_hint()
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
        _enable_standard_window_controls(dialog)
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
        self._update_delay_parameter_hint()
        self._mark_analysis_stale()

    def _update_delay_parameter_hint(self) -> None:
        self.parameter_hint.setText(
            f"Lag {int(self.max_lag_ms.value())} ms, min delay {float(self.min_lag_ms.value()):g} ms, "
            f"burst window {float(self.burst_window_ms.value()):g} ms, bin {float(self.bin_ms.value()):g} ms"
        )

    def _open_delay_settings_dialog(self) -> None:
        dialog = QDialog(self)
        _enable_standard_window_controls(dialog)
        dialog.setWindowTitle("Burst Delay Settings")
        layout = QVBoxLayout(dialog)

        intro = QLabel(
            "These parameters control the delay-search window and significance filter.\n"
            "Keep the main panel focused on source/target selection and visualization."
        )
        intro.setObjectName("MutedText")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()

        max_lag = QSpinBox()
        max_lag.setRange(self.max_lag_ms.minimum(), self.max_lag_ms.maximum())
        max_lag.setValue(int(self.max_lag_ms.value()))
        max_lag.setSuffix(" ms")
        max_lag.setToolTip("Maximum searched delay magnitude. Typical range: 20-200 ms.")

        min_lag = QDoubleSpinBox()
        min_lag.setRange(self.min_lag_ms.minimum(), self.min_lag_ms.maximum())
        min_lag.setDecimals(self.min_lag_ms.decimals())
        min_lag.setSingleStep(self.min_lag_ms.singleStep())
        min_lag.setValue(float(self.min_lag_ms.value()))
        min_lag.setSuffix(" ms")
        min_lag.setToolTip("Ignore ultra-small delays. Typical range: 0.5-5 ms.")

        burst_window = QDoubleSpinBox()
        burst_window.setRange(self.burst_window_ms.minimum(), self.burst_window_ms.maximum())
        burst_window.setDecimals(self.burst_window_ms.decimals())
        burst_window.setSingleStep(self.burst_window_ms.singleStep())
        burst_window.setValue(float(self.burst_window_ms.value()))
        burst_window.setSuffix(" ms")
        burst_window.setToolTip("Analyze only onset + X ms inside each burst. 0 means full burst.")

        bin_ms = QDoubleSpinBox()
        bin_ms.setRange(self.bin_ms.minimum(), self.bin_ms.maximum())
        bin_ms.setDecimals(self.bin_ms.decimals())
        bin_ms.setSingleStep(self.bin_ms.singleStep())
        bin_ms.setValue(float(self.bin_ms.value()))
        bin_ms.setSuffix(" ms")
        bin_ms.setToolTip("Delay histogram bin size. Typical range: 1-5 ms.")

        max_channels = QSpinBox()
        max_channels.setRange(self.max_channels.minimum(), self.max_channels.maximum())
        max_channels.setValue(int(self.max_channels.value()))
        max_channels.setToolTip("Upper bound on analyzed channels for speed.")

        max_map_pairs = QSpinBox()
        max_map_pairs.setRange(self.max_map_pairs.minimum(), self.max_map_pairs.maximum())
        max_map_pairs.setValue(int(self.max_map_pairs.value()))
        max_map_pairs.setToolTip("Maximum number of significant pairs rendered on the map.")

        form.addRow("Max lag", max_lag)
        form.addRow("Min delay", min_lag)
        form.addRow("Burst onset window", burst_window)
        form.addRow("Histogram bin", bin_ms)
        form.addRow("Max analyzed channels", max_channels)
        form.addRow("Max map pairs", max_map_pairs)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        apply_button = QPushButton("Apply")
        buttons.addWidget(cancel)
        buttons.addWidget(apply_button)
        layout.addLayout(buttons)
        cancel.clicked.connect(dialog.reject)
        apply_button.clicked.connect(dialog.accept)
        _fix_spinbox_hit_targets(dialog)

        if dialog.exec() != QDialog.Accepted:
            return
        self.max_lag_ms.setValue(int(max_lag.value()))
        self.min_lag_ms.setValue(float(min_lag.value()))
        self.burst_window_ms.setValue(float(burst_window.value()))
        self.bin_ms.setValue(float(bin_ms.value()))
        self.max_channels.setValue(int(max_channels.value()))
        self.max_map_pairs.setValue(int(max_map_pairs.value()))
        self._update_delay_parameter_hint()
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
            self.advanced_button,
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
        _show_error_message(self, "Burst delay failed", details.splitlines()[-1] if details else "Unknown error")
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

class BurstTrajectoryWindow(AppDialog):
    def __init__(
        self,
        spike_series,
        burst_intervals,
        parent=None,
        channel_map: ChannelMap | None = None,
        model_method: str = "fa",
    ):
        super().__init__(parent)
        self.setWindowTitle("Burst Trajectory")
        self.resize(1280, 760)
        self.spike_series = [(label, np.asarray(times, dtype=float)) for label, times in spike_series]
        self.burst_intervals = [(float(start), float(stop)) for start, stop in burst_intervals]
        self.channel_map = channel_map
        self.model_method = str(model_method or "fa").strip().lower()
        if self.model_method not in {"fa", "lds", "pivae"}:
            self.model_method = "fa"
        self.current = None
        self.metric_windows = []
        self._spatial_temporal_cache_signature = None
        self._spatial_temporal_cache = None

        self.bin_ms = QDoubleSpinBox()
        self.bin_ms.setRange(0.5, 200.0)
        self.bin_ms.setDecimals(1)
        self.bin_ms.setSingleStep(1.0)
        self.bin_ms.setValue(10.0)
        self.bin_ms.setSuffix(" ms")

        self.window_ms = QSpinBox()
        self.window_ms.setRange(5, 60000)
        self.window_ms.setSingleStep(10)
        self.window_ms.setValue(300)
        self.window_ms.setSuffix(" ms")

        self.analysis_scope = QComboBox()
        self.analysis_scope.addItem("Bursts", "burst")
        self.analysis_scope.addItem("All data windows", "all_windows")
        self.analysis_scope.setToolTip("Use detected bursts, or split the whole recording into non-overlapping windows.")
        self.analysis_scope.currentIndexChanged.connect(lambda *_: self._update_settings_summary())

        self.normalize = QComboBox()
        self.normalize.addItem("Channel z-score", "channel_zscore")
        self.normalize.addItem("Log + channel z-score", "log_channel_zscore")
        self.normalize.addItem("Per time total", "per_time_total")
        self.normalize.addItem("None", "none")
        self.normalize.setToolTip("Preprocessing for the state vector before FA fitting.")
        self.normalize.currentIndexChanged.connect(lambda *_: self._update_settings_summary())

        self.temporal_method = QComboBox()
        self.temporal_method.addItem("Linear", "linear")
        self.temporal_method.addItem("RBF kernel ridge", "rbf")
        self.temporal_method.addItem("kNN local average", "knn")
        self.temporal_method.addItem("Polynomial ridge", "poly")
        self.temporal_method.addItem("Random forest", "rf")
        self.temporal_method.addItem("Gradient boosting", "gb")
        self.temporal_method.addItem("MLP tanh", "mlp")
        self.temporal_method.setToolTip("Model used to predict z(t) from previous latent-state bins.")
        self.temporal_method.currentIndexChanged.connect(lambda *_: self._update_settings_summary())

        self.latent_dim = QSpinBox()
        self.latent_dim.setRange(1, 128)
        self.latent_dim.setValue(16)
        self.latent_dim.valueChanged.connect(lambda *_: self._update_settings_summary())

        self.history_bins = QSpinBox()
        self.history_bins.setRange(1, 20)
        self.history_bins.setValue(3)
        self.history_bins.setToolTip("Number of previous latent time bins used to predict the next latent state.")
        self.history_bins.valueChanged.connect(lambda *_: self._update_settings_summary())

        self.activity_similarity_weight = QDoubleSpinBox()
        self.activity_similarity_weight.setRange(0.0, 1.0)
        self.activity_similarity_weight.setDecimals(2)
        self.activity_similarity_weight.setSingleStep(0.05)
        self.activity_similarity_weight.setValue(0.78)
        self.activity_similarity_weight.setToolTip("Weight for activity-pattern similarity when detecting spatial-temporal regions.")

        self.spatial_similarity_weight = QDoubleSpinBox()
        self.spatial_similarity_weight.setRange(0.0, 1.0)
        self.spatial_similarity_weight.setDecimals(2)
        self.spatial_similarity_weight.setSingleStep(0.05)
        self.spatial_similarity_weight.setValue(0.22)
        self.spatial_similarity_weight.setToolTip("Weight for physical distance affinity when detecting spatial-temporal regions.")

        self.region_membership_threshold = QDoubleSpinBox()
        self.region_membership_threshold.setRange(0.0, 1.0)
        self.region_membership_threshold.setDecimals(2)
        self.region_membership_threshold.setSingleStep(0.05)
        self.region_membership_threshold.setValue(0.18)
        self.region_membership_threshold.setToolTip("Higher values keep only channels that are clearly regional members.")

        self.filter_values = {
            "min_activity": 1.0,
            "min_bursts": 1.0,
            "min_var": 0.0,
            "max_channels": 256.0,
        }

        self.selected_burst_value = 1
        self.display_param = QComboBox()
        self.display_param.addItem("Burst", "burst")
        self.display_param.setToolTip("Select which burst the comparison views show.")
        self.display_value = QSpinBox()
        self.display_value.valueChanged.connect(self._display_value_changed)
        self.display_param.currentIndexChanged.connect(self._display_param_changed)
        self.display_param.currentIndexChanged.connect(lambda *_: self._update_settings_summary())
        self._display_param_changed()

        self.rmse_order = QComboBox()
        self.rmse_order.addItem("RMSE high to low", "desc")
        self.rmse_order.addItem("RMSE low to high", "asc")
        self.rmse_order.currentIndexChanged.connect(self._draw_all_views)
        self.rmse_order.currentIndexChanged.connect(lambda *_: self._update_settings_summary())
        self.rmse_order.setToolTip("Controls the channel ordering in reconstruction views.")

        self.settings_button = QPushButton("Settings...")
        self.settings_button.clicked.connect(self._open_analysis_settings_dialog)
        self.settings_summary = QLabel()
        self.settings_summary.setObjectName("MutedText")
        self.settings_summary.setWordWrap(True)

        self.bin_ms.setToolTip("Time bin used to convert spikes into channel activity vectors. Typical range: 5-20 ms.")
        self.window_ms.setToolTip("Analysis window after burst onset, or fixed window size in all-window mode.")
        self.analysis_scope.setToolTip("Choose whether to fit only detected bursts or tile the full recording into non-overlapping windows.")
        self.normalize.setToolTip("Preprocessing applied before factor analysis. Per-burst is usually the best first pass.")
        self.latent_dim.setToolTip("Number of latent factors used by the FA model. Higher values capture more structure but can overfit.")
        self.temporal_method.setToolTip("Temporal model used to explain latent-state evolution across bins.")
        self.history_bins.setToolTip("How many previous latent-state bins are used by the temporal model.")
        self.display_value.setToolTip("Burst index displayed in the reconstruction views.")

        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self._draw)
        self.reconstruction_metrics_button = QPushButton("Recon Metrics")
        self.reconstruction_metrics_button.clicked.connect(self._show_reconstruction_metrics)
        self.weight_metrics_button = QPushButton("W Metrics")
        self.weight_metrics_button.clicked.connect(self._show_weight_metrics)
        self.temporal_model_button = QPushButton("Temporal Model")
        self.temporal_model_button.clicked.connect(self._show_temporal_model)
        self.trajectory_analysis_button = QPushButton("Trajectory")
        self.trajectory_analysis_button.clicked.connect(self._show_trajectory_analysis)
        self.normalized_time_button = QPushButton("Normalized Time")
        self.normalized_time_button.clicked.connect(self._show_normalized_time_analysis)
        self.spatial_temporal_button = QPushButton("Spatial-temporal")
        self.spatial_temporal_button.clicked.connect(self._show_spatial_temporal_analysis)
        self.summary = QLabel("Ready")
        self.summary.setObjectName("MutedText")
        self.raster_canvas = FigureCanvas(Figure(figsize=(9, 5.8), tight_layout=True))
        self.latent_canvas = FigureCanvas(Figure(figsize=(9, 3.2), tight_layout=True))
        self.psth_canvas = FigureCanvas(Figure(figsize=(4.2, 2.7), tight_layout=True))
        self.weight_canvas = FigureCanvas(Figure(figsize=(4.2, 3.2), tight_layout=True))

        controls_frame = QFrame()
        controls_frame.setObjectName("Panel")
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        controls_layout.setSpacing(8)

        parameter_grid = QGridLayout()
        parameter_grid.setHorizontalSpacing(12)
        parameter_grid.setVerticalSpacing(8)
        parameter_grid.addWidget(QLabel("Bin"), 0, 0)
        parameter_grid.addWidget(self.bin_ms, 0, 1)
        parameter_grid.addWidget(QLabel("Window"), 0, 2)
        parameter_grid.addWidget(self.window_ms, 0, 3)
        parameter_grid.addWidget(QLabel("Scope"), 0, 4)
        parameter_grid.addWidget(self.analysis_scope, 0, 5)
        parameter_grid.addWidget(QLabel("Latent dim"), 0, 6)
        parameter_grid.addWidget(self.latent_dim, 0, 7)
        parameter_grid.addWidget(QLabel("Temporal"), 0, 8)
        parameter_grid.addWidget(self.temporal_method, 0, 9)
        parameter_grid.addWidget(QLabel("Burst"), 1, 0)
        parameter_grid.addWidget(self.display_value, 1, 1)
        parameter_grid.addWidget(self.settings_button, 1, 2)
        parameter_grid.addWidget(self.settings_summary, 1, 3, 1, 7)
        controls_layout.addLayout(parameter_grid)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        action_row.addWidget(self.analyze_button)
        action_row.addWidget(self.reconstruction_metrics_button)
        action_row.addWidget(self.weight_metrics_button)
        action_row.addWidget(self.temporal_model_button)
        action_row.addWidget(self.trajectory_analysis_button)
        action_row.addWidget(self.normalized_time_button)
        action_row.addWidget(self.spatial_temporal_button)
        action_row.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        controls_layout.addLayout(action_row)

        left_plots = QVBoxLayout()
        left_plots.addWidget(self.raster_canvas, 3)
        left_plots.addWidget(self.latent_canvas, 2)

        right_plots = QVBoxLayout()
        right_plots.addWidget(self.psth_canvas, 1)
        right_plots.addWidget(self.weight_canvas, 1)

        plots = QHBoxLayout()
        plots.addLayout(left_plots, 3)
        plots.addLayout(right_plots, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(controls_frame)
        layout.addWidget(self.summary)
        layout.addLayout(plots, 1)
        self._update_settings_summary()
        self._draw()
        _fix_spinbox_hit_targets(self)
        self.showMaximized()

    def _draw(self):
        progress = _create_progress_dialog(self, "Dynamics analysis", "Fitting latent states...", 0) if _progress_enabled_for_widget(self) else None
        try:
            self.summary.setText("Analyzing latent states...")
            QApplication.processEvents()
            scope = str(self.analysis_scope.currentData() or "burst")
            intervals = (
                _non_overlapping_spike_windows(self.spike_series, float(self.window_ms.value()))
                if scope == "all_windows"
                else self.burst_intervals
            )
            self.current = _burst_trajectory_analysis(
                self.spike_series,
                intervals,
                time_bin_ms=float(self.bin_ms.value()),
                window_ms=float(self.window_ms.value()),
                normalization=str(self.normalize.currentData()),
                cluster_count=3,
                early_bins=3,
                latent_dim=int(self.latent_dim.value()),
                min_total_activity=float(self.filter_values["min_activity"]),
                min_active_bursts=int(round(self.filter_values["min_bursts"])),
                min_variance=float(self.filter_values["min_var"]),
                max_channels=int(round(self.filter_values["max_channels"])),
                analysis_scope=scope,
                model_method=self.model_method,
            )
            if self.model_method == "lds" and self.current:
                latent_states = np.asarray(self.current.get("latent_states", []), dtype=float)
                observed_states = np.asarray(self.current.get("observed_states", []), dtype=float)
                latent_params = self.current.get("latent_params", {}) or {}
                normalization_params = self.current.get("normalization_params", {}) or {}
                lds_result = _fit_linear_latent_dynamics(latent_states, observed_states, latent_params, normalization_params)
                if lds_result:
                    fa_reconstruction_rmse = np.asarray(self.current.get("reconstruction_rmse", []), dtype=float)
                    fa_reconstruction_r2 = float(self.current.get("reconstruction_r2", 0.0))
                    self.current["fa_reconstruction_rmse"] = fa_reconstruction_rmse
                    self.current["fa_reconstruction_r2"] = fa_reconstruction_r2
                    self.current.update(lds_result)
                    self.current["reconstruction_rmse"] = np.asarray(lds_result.get("time_rmse", fa_reconstruction_rmse), dtype=float)
                    self.current["reconstruction_r2"] = float(lds_result.get("rollout_r2", fa_reconstruction_r2))
                    self.current["state_projection"] = f"LDS over FA latent {latent_states.shape[2] if latent_states.ndim == 3 else 0}D"
                    self.current["model_method"] = "lds"
                else:
                    self.current["model_method"] = "fa"
            elif self.current:
                self.current["model_method"] = self.model_method if self.model_method == "pivae" else "fa"
        except Exception as exc:
            self.current = None
            self._spatial_temporal_cache_signature = None
            self._spatial_temporal_cache = None
            _close_progress_dialog(progress)
            self._draw_error(str(exc))
            self.summary.setText(f"Burst trajectory failed: {str(exc).splitlines()[-1] if str(exc) else type(exc).__name__}")
            return
        self._spatial_temporal_cache_signature = None
        self._spatial_temporal_cache = None
        _set_progress_dialog(progress, "Drawing views...")
        self._refresh_compare_ranges()
        self._draw_all_views()
        self._update_summary()
        _close_progress_dialog(progress)

    def _draw_all_views(self):
        self._draw_raster_comparison()
        self._draw_latent_heatmap()
        self._draw_psth_comparison()
        self._draw_weight_matrix()

    def _current_model_method(self) -> str:
        return str((self.current or {}).get("model_method", self.model_method or "fa")).strip().lower()

    def _display_latent_states(self) -> np.ndarray:
        analysis = self.current or {}
        model_method = self._current_model_method()
        if model_method == "lds":
            modeled = np.asarray(analysis.get("model_latent_states", []), dtype=float)
            if modeled.ndim == 3 and modeled.size:
                return modeled
        return np.asarray(analysis.get("latent_states", []), dtype=float)

    def _selected_burst_index(self, sample_count: int) -> int:
        return min(max(0, int(self.selected_burst_value) - 1), max(0, int(sample_count) - 1))

    def _update_settings_summary(self):
        self.settings_summary.setText(
            f"Preprocess: {self.normalize.currentText()}, history {int(self.history_bins.value())} bins, "
            f"RMSE order {self.rmse_order.currentText().lower()}. "
            f"Channel screen: act >= {float(self.filter_values['min_activity']):g}, "
            f"bursts >= {int(round(self.filter_values['min_bursts']))}, "
            f"var >= {float(self.filter_values['min_var']):g}, "
            f"max ch {int(round(self.filter_values['max_channels']))}. "
            f"Region weights: act {float(self.activity_similarity_weight.value()):.2f}, "
            f"space {float(self.spatial_similarity_weight.value()):.2f}, "
            f"membership {float(self.region_membership_threshold.value()):.2f}."
        )

    def _open_analysis_settings_dialog(self):
        dialog = QDialog(self)
        _enable_standard_window_controls(dialog)
        dialog.setWindowTitle("Burst Trajectory Settings")
        layout = QVBoxLayout(dialog)

        intro = QLabel(
            "Advanced settings for channel pre-screening and spatial-temporal region detection.\n"
            "Keep the top bar focused on the main latent-state analysis controls."
        )
        intro.setObjectName("MutedText")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()

        normalize_combo = QComboBox()
        for index in range(self.normalize.count()):
            normalize_combo.addItem(self.normalize.itemText(index), self.normalize.itemData(index))
        normalize_combo.setCurrentIndex(max(0, normalize_combo.findData(self.normalize.currentData())))
        normalize_combo.setToolTip("Preprocessing applied before factor analysis. Per-burst is often a good default.")

        history_bins = QSpinBox()
        history_bins.setRange(self.history_bins.minimum(), self.history_bins.maximum())
        history_bins.setSingleStep(self.history_bins.singleStep())
        history_bins.setValue(int(self.history_bins.value()))
        history_bins.setToolTip("Number of previous latent-state bins used by the temporal model.")

        rmse_order = QComboBox()
        for index in range(self.rmse_order.count()):
            rmse_order.addItem(self.rmse_order.itemText(index), self.rmse_order.itemData(index))
        rmse_order.setCurrentIndex(max(0, rmse_order.findData(self.rmse_order.currentData())))
        rmse_order.setToolTip("Channel ordering used by the reconstruction comparison panels.")

        min_activity = QDoubleSpinBox()
        min_activity.setRange(0.0, 1_000_000.0)
        min_activity.setDecimals(2)
        min_activity.setValue(float(self.filter_values["min_activity"]))
        min_activity.setToolTip("Remove channels with very low total activity. Typical range: 0-10.")

        min_bursts = QSpinBox()
        min_bursts.setRange(0, 100000)
        min_bursts.setValue(int(round(self.filter_values["min_bursts"])))
        min_bursts.setToolTip("Require a channel to appear in at least this many bursts/windows.")

        min_var = QDoubleSpinBox()
        min_var.setRange(0.0, 1_000_000.0)
        min_var.setDecimals(6)
        min_var.setValue(float(self.filter_values["min_var"]))
        min_var.setToolTip("Remove near-constant channels. Typical range: 0-0.1.")

        max_channels = QSpinBox()
        max_channels.setRange(1, 20000)
        max_channels.setValue(int(round(self.filter_values["max_channels"])))
        max_channels.setToolTip("Upper bound on fitted channels for speed and stability.")

        activity_weight = QDoubleSpinBox()
        activity_weight.setRange(0.0, 1.0)
        activity_weight.setDecimals(2)
        activity_weight.setSingleStep(0.05)
        activity_weight.setValue(float(self.activity_similarity_weight.value()))
        activity_weight.setToolTip("Weight of activity-pattern similarity in regional clustering.")

        spatial_weight = QDoubleSpinBox()
        spatial_weight.setRange(0.0, 1.0)
        spatial_weight.setDecimals(2)
        spatial_weight.setSingleStep(0.05)
        spatial_weight.setValue(float(self.spatial_similarity_weight.value()))
        spatial_weight.setToolTip("Weight of physical proximity in regional clustering.")

        membership_thr = QDoubleSpinBox()
        membership_thr.setRange(0.0, 1.0)
        membership_thr.setDecimals(2)
        membership_thr.setSingleStep(0.05)
        membership_thr.setValue(float(self.region_membership_threshold.value()))
        membership_thr.setToolTip("Higher threshold keeps only more region-specific channels. Typical range: 0.1-0.4.")

        form.addRow("Normalization", normalize_combo)
        form.addRow("Temporal history", history_bins)
        form.addRow("RMSE ordering", rmse_order)
        form.addRow("Min total activity", min_activity)
        form.addRow("Min active bursts/windows", min_bursts)
        form.addRow("Min variance", min_var)
        form.addRow("Max fitted channels", max_channels)
        form.addRow("Region activity weight", activity_weight)
        form.addRow("Region spatial weight", spatial_weight)
        form.addRow("Region membership threshold", membership_thr)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        apply_button = QPushButton("Apply")
        buttons.addWidget(cancel)
        buttons.addWidget(apply_button)
        layout.addLayout(buttons)
        cancel.clicked.connect(dialog.reject)
        apply_button.clicked.connect(dialog.accept)
        _fix_spinbox_hit_targets(dialog)

        if dialog.exec() != QDialog.Accepted:
            return
        self.normalize.setCurrentIndex(max(0, self.normalize.findData(normalize_combo.currentData())))
        self.history_bins.setValue(int(history_bins.value()))
        self.rmse_order.setCurrentIndex(max(0, self.rmse_order.findData(rmse_order.currentData())))
        self.filter_values["min_activity"] = float(min_activity.value())
        self.filter_values["min_bursts"] = float(min_bursts.value())
        self.filter_values["min_var"] = float(min_var.value())
        self.filter_values["max_channels"] = float(max_channels.value())
        self.activity_similarity_weight.setValue(float(activity_weight.value()))
        self.spatial_similarity_weight.setValue(float(spatial_weight.value()))
        self.region_membership_threshold.setValue(float(membership_thr.value()))
        self._update_settings_summary()

    def _display_param_changed(self):
        self.display_value.blockSignals(True)
        self.display_value.setRange(1, max(1, self.display_value.maximum()))
        self.display_value.setValue(int(self.selected_burst_value))
        self.display_value.blockSignals(False)

    def _display_value_changed(self, value: int):
        self.selected_burst_value = int(value)
        self._draw_all_views()

    def _refresh_compare_ranges(self):
        if not self.current:
            return
        observed = np.asarray(self.current.get("observed_states", []), dtype=float)
        burst_count = int(observed.shape[0]) if observed.ndim == 3 else 1
        self.selected_burst_value = min(max(1, int(self.selected_burst_value)), max(1, burst_count))
        self.display_value.blockSignals(True)
        self.display_value.setRange(1, max(1, burst_count))
        self.display_value.setValue(self.selected_burst_value)
        self.display_value.blockSignals(False)

    def _draw_error(self, message: str):
        for canvas in (self.raster_canvas, self.latent_canvas, self.psth_canvas, self.weight_canvas):
            figure = canvas.figure
            figure.clear()
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, message or "Burst trajectory failed", ha="center", va="center", wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])
            canvas.draw_idle()

    def _channel_rmse_order(self, observed: np.ndarray, reconstructed: np.ndarray, burst_index: int) -> tuple[np.ndarray, np.ndarray]:
        channel_count = observed.shape[2] if observed.ndim == 3 else 0
        residual = observed[burst_index] - reconstructed[burst_index] if channel_count else np.zeros((0, 0), dtype=float)
        channel_rmse = np.sqrt(np.mean(residual ** 2, axis=0)) if residual.size else np.zeros(channel_count, dtype=float)
        if self.rmse_order.currentData() == "asc":
            order = np.argsort(channel_rmse)
        else:
            order = np.argsort(channel_rmse)[::-1]
        return order, channel_rmse

    def _draw_raster_comparison(self):
        figure = self.raster_canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        analysis = self.current or {}
        model_method = self._current_model_method()
        raw_observed = np.asarray(analysis.get("raw_observed_states", []), dtype=float)
        raw_reconstructed = np.asarray(analysis.get("raw_reconstructed_states", []), dtype=float)
        observed = np.asarray(analysis.get("observed_states", []), dtype=float)
        reconstructed = np.asarray(analysis.get("reconstructed_states", []), dtype=float)
        labels = [str(label) for label in analysis.get("selected_labels", [])]
        if raw_observed.ndim != 3 or raw_reconstructed.shape != raw_observed.shape or raw_observed.shape[0] < 1 or raw_observed.shape[1] < 1:
            ax.text(0.5, 0.5, "No factor analysis reconstruction", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            burst_index = self._selected_burst_index(raw_observed.shape[0])
            window_ms = float(analysis.get("window_ms", self.window_ms.value()))
            if not np.isfinite(window_ms) or window_ms <= 0:
                window_ms = float(raw_observed.shape[1])
            if observed.ndim == 3 and reconstructed.shape == observed.shape:
                order, channel_rmse = self._channel_rmse_order(observed, reconstructed, burst_index)
            else:
                order = np.arange(raw_observed.shape[2], dtype=int)
                channel_rmse = np.zeros(raw_observed.shape[2], dtype=float)
            true_block = raw_observed[burst_index][:, order].T
            recon_block = raw_reconstructed[burst_index][:, order].T
            heatmap = np.vstack([true_block, recon_block])
            vmax = float(np.nanpercentile(heatmap, 98.0)) if heatmap.size else 1.0
            vmax = max(vmax, 1e-9)
            image = ax.imshow(
                heatmap,
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                vmin=0.0,
                vmax=vmax,
                extent=[0.0, window_ms, heatmap.shape[0], 0.0],
            )
            ax.axhline(true_block.shape[0] - 0.5, color="#111827", linewidth=1.0)
            ax.text(window_ms * 0.01, max(0.8, true_block.shape[0] * 0.05), "Observed", color="white", fontsize=8, va="top")
            ax.text(window_ms * 0.01, true_block.shape[0] + max(0.8, recon_block.shape[0] * 0.05), "Reconstructed", color="white", fontsize=8, va="top")
            title_prefix = "LDS rollout reconstruction" if model_method == "lds" else ("pi-VAE reconstruction" if model_method == "pivae" else "FA reconstruction")
            ax.set_title(f"Raw burst activity vs {title_prefix} | burst {burst_index + 1}")
            ax.set_ylabel("All selected channels")
            ax.set_xlim(0.0, window_ms)
            ax.set_xticks(np.linspace(0.0, window_ms, min(7, max(2, raw_observed.shape[1] + 1))))
            ax.set_xlabel("Time from burst onset (ms)")
            if labels and len(labels) == raw_observed.shape[2]:
                y_tick_indices = _display_indices(order.size, min(12, max(1, order.size)))
                y_ticks = np.r_[y_tick_indices + 0.5, y_tick_indices + len(order) + 0.5]
                y_labels = [f"{labels[int(order[index])]} ({channel_rmse[int(order[index])]:.3g})" for index in y_tick_indices]
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_labels + y_labels, fontsize=7)
            figure.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        self.raster_canvas.draw_idle()

    def _draw_latent_heatmap(self):
        figure = self.latent_canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        analysis = self.current or {}
        model_method = self._current_model_method()
        latent_states = self._display_latent_states()
        centers_ms = np.asarray(analysis.get("centers_ms", []), dtype=float)
        if latent_states.ndim == 3 and latent_states.shape[0]:
            burst_index = self._selected_burst_index(latent_states.shape[0])
            latent_block = latent_states[burst_index].T
            window_ms = float(analysis.get("window_ms", self.window_ms.value()))
            if not np.isfinite(window_ms) or window_ms <= 0:
                window_ms = float(latent_block.shape[1])
            vmax = float(np.nanpercentile(np.abs(latent_block), 98.0)) if latent_block.size else 1.0
            vmax = max(vmax, 1e-9)
            image = ax.imshow(
                latent_block,
                aspect="auto",
                interpolation="nearest",
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                extent=[0.0, window_ms, latent_block.shape[0], 0.0],
            )
            title = "Modeled latent state z(t)" if model_method == "lds" else ("pi-VAE latent state z(t)" if model_method == "pivae" else "Latent state z(t)")
            ax.set_title(f"{title} | burst {burst_index + 1}")
            ax.set_ylabel("Latent dim")
            ax.set_xlim(0.0, window_ms)
            ax.set_xticks(np.linspace(0.0, window_ms, min(7, max(2, latent_states.shape[1] + 1))))
            ax.set_xlabel("Time from burst onset (ms)")
            ax.set_yticks(np.arange(latent_block.shape[0]) + 0.5)
            ax.set_yticklabels([str(index + 1) for index in range(latent_block.shape[0])], fontsize=7)
            figure.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        else:
            ax.text(0.5, 0.5, "No latent state", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        self.latent_canvas.draw_idle()

    def _draw_psth_comparison(self):
        figure = self.psth_canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        analysis = self.current or {}
        model_method = self._current_model_method()
        observed = np.asarray(analysis.get("raw_observed_states", []), dtype=float)
        reconstructed = np.asarray(analysis.get("raw_reconstructed_states", []), dtype=float)
        centers_ms = np.asarray(analysis.get("centers_ms", []), dtype=float)
        if observed.ndim != 3 or reconstructed.shape != observed.shape or observed.shape[0] == 0:
            ax.text(0.5, 0.5, "No PSTH data", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            burst_index = self._selected_burst_index(observed.shape[0])
            observed_psth = np.mean(observed[burst_index], axis=1)
            reconstructed_psth = np.mean(reconstructed[burst_index], axis=1)
            if centers_ms.size != observed_psth.size:
                centers_ms = np.arange(observed_psth.size, dtype=float)
            ax.plot(centers_ms, observed_psth, color="#1d4ed8", linewidth=1.7, label="Observed PSTH")
            ax.plot(centers_ms, reconstructed_psth, color="#dc2626", linewidth=1.7, alpha=0.9, label="Reconstructed PSTH")
            title_prefix = "LDS rollout" if model_method == "lds" else ("pi-VAE reconstructed" if model_method == "pivae" else "reconstructed")
            ax.set_title(f"Raw observed vs {title_prefix} PSTH | burst {burst_index + 1}")
            ax.set_xlabel("Time from burst onset (ms)")
            ax.set_ylabel("Mean firing rate (Hz)")
            window_ms = float(analysis.get("window_ms", self.window_ms.value()))
            if np.isfinite(window_ms) and window_ms > 0:
                ax.set_xlim(0.0, window_ms)
            ax.legend(loc="best", fontsize=8)
        self.psth_canvas.draw_idle()

    def _draw_weight_matrix(self):
        figure = self.weight_canvas.figure
        figure.clear()
        weight_ax = figure.add_subplot(111)
        analysis = self.current or {}
        params = analysis.get("latent_params", {}) or {}
        loadings = np.asarray(params.get("loadings", []), dtype=float)
        labels = [str(label) for label in analysis.get("selected_labels", [])]
        if loadings.ndim != 2 or loadings.size == 0:
            weight_ax.text(0.5, 0.5, "No loading matrix", ha="center", va="center")
            weight_ax.set_xticks([])
            weight_ax.set_yticks([])
        else:
            vmax = float(np.nanpercentile(np.abs(loadings), 98.0)) if loadings.size else 1.0
            vmax = max(vmax, 1e-9)
            image = weight_ax.imshow(loadings, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            title = "pi-VAE decoder loading approximation" if self._current_model_method() == "pivae" else "Factor loading matrix W"
            weight_ax.set_title(title)
            weight_ax.set_ylabel("Latent dim")
            weight_ax.set_xlabel("Selected channel")
            if labels and len(labels) == loadings.shape[1]:
                tick_indices = _display_indices(loadings.shape[1], 8)
                weight_ax.set_xticks(tick_indices)
                weight_ax.set_xticklabels([labels[int(index)] for index in tick_indices], rotation=45, ha="right", fontsize=7)
            weight_ax.set_yticks(np.arange(loadings.shape[0]))
            weight_ax.set_yticklabels([str(index + 1) for index in range(loadings.shape[0])], fontsize=8)
            figure.colorbar(image, ax=weight_ax, fraction=0.025, pad=0.02)
        self.weight_canvas.draw_idle()

    def _show_metric_text(self, title: str, text: str):
        dialog = QDialog(self)
        _enable_standard_window_controls(dialog)
        dialog.setWindowTitle(title)
        dialog.resize(720, 520)
        layout = QVBoxLayout(dialog)
        viewer = QTextEdit()
        viewer.setReadOnly(True)
        viewer.setPlainText(text)
        layout.addWidget(viewer)
        button = QPushButton("Close")
        button.clicked.connect(dialog.accept)
        layout.addWidget(button)
        dialog.exec()

    def _show_metric_figure(self, title: str, summary: str, draw_callback):
        dialog = QDialog(self)
        _enable_standard_window_controls(dialog)
        dialog.setWindowTitle(title)
        dialog.resize(1040, 760)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        layout = QVBoxLayout(dialog)
        summary_label = QLabel(summary)
        summary_label.setObjectName("MutedText")
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)
        canvas = FigureCanvas(Figure(figsize=(10, 6.5), tight_layout=True))
        layout.addWidget(canvas, 1)
        button = QPushButton("Close")
        button.clicked.connect(dialog.close)
        layout.addWidget(button)
        draw_callback(canvas.figure)
        canvas.draw_idle()
        self.metric_windows.append(dialog)

        def _forget_window(_obj=None, window=dialog):
            if window in self.metric_windows:
                self.metric_windows.remove(window)

        dialog.destroyed.connect(_forget_window)
        dialog.show()
        return dialog

    def _show_reconstruction_metrics(self):
        if not self.current:
            _show_info_message(self, "Reconstruction metrics", "Run factor analysis first.")
            return
        observed = np.asarray(self.current.get("observed_states", []), dtype=float)
        reconstructed = np.asarray(self.current.get("reconstructed_states", []), dtype=float)
        rmse = np.asarray(self.current.get("reconstruction_rmse", []), dtype=float)
        r2 = float(self.current.get("reconstruction_r2", 0.0))
        latent_states = np.asarray(self.current.get("latent_states", []), dtype=float)
        if observed.ndim != 3 or reconstructed.shape != observed.shape:
            _show_info_message(self, "Reconstruction metrics", "No reconstruction data is available.")
            return
        residual = observed - reconstructed
        channel_rmse = np.sqrt(np.mean(residual ** 2, axis=(0, 1))) if residual.size else np.zeros(0, dtype=float)
        labels = [str(label) for label in self.current.get("selected_labels", [])]
        order = np.argsort(channel_rmse)[::-1]
        latent_dim_curve = self._latent_dim_reconstruction_curve(observed)
        flat_latent = latent_states.reshape((-1, latent_states.shape[2])) if latent_states.ndim == 3 and latent_states.shape[2] > 0 else np.zeros((0, 0), dtype=float)
        if flat_latent.ndim == 2 and flat_latent.shape[0] >= 2 and flat_latent.shape[1] >= 1:
            centered_latent = flat_latent - np.mean(flat_latent, axis=0, keepdims=True)
            covariance = centered_latent.T @ centered_latent / max(1, centered_latent.shape[0] - 1)
            try:
                latent_eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
            except np.linalg.LinAlgError:
                latent_eigenvalues = np.zeros(flat_latent.shape[1], dtype=float)
            latent_eigenvalues = np.clip(np.asarray(latent_eigenvalues, dtype=float), 0.0, None)
            latent_variance_ratio = latent_eigenvalues / max(float(np.sum(latent_eigenvalues)), 1e-12)
        else:
            latent_eigenvalues = np.zeros(0, dtype=float)
            latent_variance_ratio = np.zeros(0, dtype=float)
        local_diagnostics = _local_pca_manifold_diagnostics(flat_latent, max_dim=8, neighbor_count=20, max_samples=1000)
        summary = " | ".join([
            f"Global R2: {r2:.6g}",
            f"Mean time-bin RMSE: {float(np.mean(rmse)):.6g}" if rmse.size else "Mean time-bin RMSE: n/a",
            f"Median channel RMSE: {float(np.median(channel_rmse)):.6g}" if channel_rmse.size else "Median channel RMSE: n/a",
            f"Max channel RMSE: {float(np.max(channel_rmse)):.6g}" if channel_rmse.size else "Max channel RMSE: n/a",
        ])

        def _draw(figure):
            figure.clear()
            centers_ms = np.asarray(self.current.get("centers_ms", []), dtype=float)
            if centers_ms.size != rmse.size:
                centers_ms = np.arange(rmse.size, dtype=float)
            axes = figure.subplots(2, 3)
            ax = axes[0, 0]
            dims = np.asarray(latent_dim_curve.get("dims", []), dtype=int)
            r2_values = np.asarray(latent_dim_curve.get("r2", []), dtype=float)
            rmse_values = np.asarray(latent_dim_curve.get("mean_rmse", []), dtype=float)
            if dims.size:
                ax.plot(dims, r2_values, color="#1d4ed8", marker="o", linewidth=1.8, label="R2")
                ax.set_ylabel("R2", color="#1d4ed8")
                ax.tick_params(axis="y", labelcolor="#1d4ed8")
                ax.set_ylim(min(0.0, float(np.nanmin(r2_values)) - 0.02), min(1.02, max(0.05, float(np.nanmax(r2_values)) + 0.02)))
                rmse_ax = ax.twinx()
                rmse_ax.plot(dims, rmse_values, color="#dc2626", marker="s", linewidth=1.4, label="Mean RMSE")
                rmse_ax.set_ylabel("Mean RMSE", color="#dc2626")
                rmse_ax.tick_params(axis="y", labelcolor="#dc2626")
                ax.set_xticks(dims)
                ax.tick_params(axis="x", labelrotation=45)
            else:
                ax.text(0.5, 0.5, "No latent dimension scan", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_title("Latent dim vs reconstruction")
            ax.set_xlabel("Latent dim")

            ax = axes[0, 1]
            ax.plot(centers_ms, rmse, color="#1d4ed8", linewidth=1.8)
            ax.set_title("Time-bin reconstruction RMSE")
            ax.set_xlabel("Time from burst onset (ms)")
            ax.set_ylabel("RMSE")
            window_ms = float(self.current.get("window_ms", self.window_ms.value()))
            if np.isfinite(window_ms) and window_ms > 0:
                ax.set_xlim(0.0, window_ms)

            ax = axes[0, 2]
            top_order = order[: min(30, order.size)]
            top_labels = [labels[int(index)] if int(index) < len(labels) else f"ch{int(index) + 1}" for index in top_order]
            ax.bar(np.arange(top_order.size), channel_rmse[top_order], color="#dc2626", alpha=0.82)
            ax.set_title("Channel RMSE")
            ax.set_ylabel("RMSE")
            ax.set_xticks(np.arange(top_order.size))
            ax.set_xticklabels(top_labels, rotation=60, ha="right", fontsize=7)

            ax = axes[1, 0]
            residual_flat = residual.reshape(-1)
            if residual_flat.size:
                ax.hist(residual_flat, bins=60, color="#475569", alpha=0.78)
            ax.axvline(0.0, color="#111827", linewidth=1.0)
            ax.set_title("Residual distribution")
            ax.set_xlabel("Observed - reconstructed")
            ax.set_ylabel("Count")

            ax = axes[1, 1]
            observed_flat = observed.reshape(-1)
            reconstructed_flat = reconstructed.reshape(-1)
            if observed_flat.size:
                sample_count = min(12000, observed_flat.size)
                sample_indices = np.linspace(0, observed_flat.size - 1, sample_count, dtype=int)
                ax.scatter(observed_flat[sample_indices], reconstructed_flat[sample_indices], s=6, color="#2563eb", alpha=0.25, linewidths=0)
                limits = np.asarray([observed_flat[sample_indices], reconstructed_flat[sample_indices]], dtype=float)
                low = float(np.nanpercentile(limits, 1.0))
                high = float(np.nanpercentile(limits, 99.0))
                if np.isfinite(low) and np.isfinite(high) and high > low:
                    ax.plot([low, high], [low, high], color="#dc2626", linewidth=1.2)
                    ax.set_xlim(low, high)
                    ax.set_ylim(low, high)
            ax.set_title("Observed vs reconstructed state")
            ax.set_xlabel("Observed")
            ax.set_ylabel("Reconstructed")

            ax = axes[1, 2]
            ax.text(0.02, 0.98, "", transform=ax.transAxes)
            if latent_eigenvalues.size:
                components = np.arange(1, latent_eigenvalues.size + 1, dtype=int)
                ax.plot(components, latent_eigenvalues, color="#7c3aed", marker="o", linewidth=1.7, label="Global eigenvalue")
                if latent_variance_ratio.size:
                    cumulative = np.cumsum(latent_variance_ratio)
                    ratio_ax = ax.twinx()
                    ratio_ax.plot(components, cumulative, color="#059669", marker="s", linewidth=1.3, linestyle="--", label="Cumulative variance")
                    ratio_ax.set_ylabel("Cumulative variance", color="#059669")
                    ratio_ax.tick_params(axis="y", labelcolor="#059669")
                    ratio_ax.set_ylim(0.0, 1.02)
                ax.set_xticks(components)
                ax.set_title("Flattened z PCA eigenspectrum")
                ax.set_xlabel("Principal component")
                ax.set_ylabel("Eigenvalue")
            else:
                ax.text(0.5, 0.5, "No latent point cloud", ha="center", va="center")
                ax.set_title("Flattened z PCA eigenspectrum")
                ax.set_xticks([])
                ax.set_yticks([])

            inset_left = ax.inset_axes([0.10, 0.18, 0.36, 0.28])
            local_dims = np.asarray(local_diagnostics.get("dims", []), dtype=int)
            local_var = np.asarray(local_diagnostics.get("mean_local_variance_ratio", []), dtype=float)
            estimated_local_dim = np.asarray(local_diagnostics.get("estimated_local_dim", []), dtype=float)
            if local_dims.size:
                inset_left.plot(local_dims, local_var, color="#2563eb", marker="o", linewidth=1.2)
                inset_left.set_title("Local PCA variance", fontsize=7)
                inset_left.set_xlabel("Dim", fontsize=6)
                inset_left.set_ylabel("Explained", fontsize=6)
                inset_left.tick_params(axis="both", labelsize=6)
            else:
                inset_left.text(0.5, 0.5, "No local PCA", ha="center", va="center", fontsize=7)
                inset_left.set_xticks([])
                inset_left.set_yticks([])

            inset_right = ax.inset_axes([0.58, 0.18, 0.36, 0.28])
            valid_local_dims = estimated_local_dim[np.isfinite(estimated_local_dim) & (estimated_local_dim > 0)]
            if valid_local_dims.size:
                inset_right.hist(valid_local_dims, bins=np.arange(0.5, max(8.5, float(np.max(valid_local_dims)) + 1.5), 1.0), color="#7c3aed", alpha=0.8)
                mean_local_dim = float(np.mean(valid_local_dims))
                median_local_dim = float(np.median(valid_local_dims))
                inset_right.axvline(mean_local_dim, color="#dc2626", linewidth=1.0, linestyle="--", label=f"mean {mean_local_dim:.2f}")
                inset_right.axvline(median_local_dim, color="#16a34a", linewidth=1.0, linestyle=":", label=f"median {median_local_dim:.2f}")
                inset_right.set_title("Local intrinsic dim", fontsize=7)
                inset_right.set_xlabel("Dim", fontsize=6)
                inset_right.set_ylabel("Count", fontsize=6)
                inset_right.legend(loc="best", fontsize=6, frameon=False)
                inset_right.tick_params(axis="both", labelsize=6)
            else:
                inset_right.text(0.5, 0.5, "No local dim estimate", ha="center", va="center", fontsize=7)
                inset_right.set_xticks([])
                inset_right.set_yticks([])

        self._show_metric_figure("Reconstruction metrics", summary, _draw)

    def _latent_dim_reconstruction_curve(self, observed: np.ndarray) -> dict:
        values = np.nan_to_num(np.asarray(observed, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] == 0:
            return {"dims": np.zeros(0, dtype=int), "r2": np.zeros(0, dtype=float), "mean_rmse": np.zeros(0, dtype=float)}
        sample_count = int(values.shape[0] * values.shape[1])
        channel_count = int(values.shape[2])
        max_dim = min(96, sample_count, channel_count)
        if max_dim < 1:
            return {"dims": np.zeros(0, dtype=int), "r2": np.zeros(0, dtype=float), "mean_rmse": np.zeros(0, dtype=float)}
        if max_dim < 4:
            dims = np.arange(1, max_dim + 1, dtype=int)
        else:
            dims = np.arange(4, max_dim + 1, 4, dtype=int)
            if dims.size == 0 or int(dims[-1]) != max_dim:
                dims = np.r_[dims, max_dim]
            current_dim = min(max_dim, max(1, int(self.latent_dim.value())))
            if current_dim not in set(int(value) for value in dims):
                dims = np.sort(np.r_[dims, current_dim]).astype(int)
        r2_values = []
        rmse_values = []
        centered = values - np.mean(values, axis=(0, 1), keepdims=True)
        sst = float(np.sum(centered ** 2))
        for dim in dims:
            latent, params = _factor_analysis_latent_states(values, int(dim), max_iter=500)
            loadings = np.asarray(params.get("loadings", []), dtype=float)
            mean = np.asarray(params.get("mean", []), dtype=float)
            if latent.size and loadings.ndim == 2 and mean.ndim == 1:
                reconstructed_flat = latent.reshape((-1, latent.shape[2])) @ loadings + mean
                reconstructed = reconstructed_flat.reshape(values.shape)
            else:
                reconstructed = np.zeros_like(values)
            residual = values - reconstructed
            r2_values.append(float(1.0 - float(np.sum(residual ** 2)) / max(sst, 1e-12)))
            rmse_values.append(float(np.sqrt(np.mean(residual ** 2))) if residual.size else 0.0)
        return {
            "dims": np.asarray(dims, dtype=int),
            "r2": np.asarray(r2_values, dtype=float),
            "mean_rmse": np.asarray(rmse_values, dtype=float),
        }

    def _show_weight_metrics(self):
        if not self.current:
            _show_info_message(self, "Weight matrix metrics", "Run factor analysis first.")
            return
        params = self.current.get("latent_params", {}) or {}
        loadings = np.asarray(params.get("loadings", []), dtype=float)
        labels = [str(label) for label in self.current.get("selected_labels", [])]
        if loadings.ndim != 2 or loadings.size == 0:
            _show_info_message(self, "Weight matrix metrics", "No loading matrix is available.")
            return
        try:
            _u, singular_values, vt = np.linalg.svd(loadings, full_matrices=False)
        except np.linalg.LinAlgError:
            _show_info_message(self, "Weight matrix metrics", "SVD failed for the loading matrix.")
            return
        energy = singular_values ** 2
        total_energy = float(np.sum(energy))
        cumulative = np.cumsum(energy) / max(total_energy, 1e-12)
        rank = int(np.count_nonzero(singular_values > max(singular_values[0] if singular_values.size else 0.0, 1.0) * 1e-9))
        if total_energy > 1e-12:
            weights = energy / total_energy
            effective_rank = float(np.exp(-np.sum(weights * np.log(np.maximum(weights, 1e-12)))))
        else:
            effective_rank = 0.0
        factor_gram = loadings @ loadings.T
        factor_norms = np.sqrt(np.maximum(np.diag(factor_gram), 0.0))
        factor_overlap = np.divide(
            factor_gram,
            np.outer(factor_norms, factor_norms),
            out=np.zeros_like(factor_gram),
            where=np.outer(factor_norms, factor_norms) > 1e-12,
        )
        if rank > 0 and vt.size:
            row_basis = vt[:rank, :]
            channel_projection = row_basis.T @ row_basis
        else:
            row_basis = np.zeros((0, loadings.shape[1]), dtype=float)
            channel_projection = np.zeros((loadings.shape[1], loadings.shape[1]), dtype=float)
        summary = " | ".join([
            f"W shape: {loadings.shape[0]} x {loadings.shape[1]}",
            f"Rank: {rank}",
            f"Effective rank: {effective_rank:.4g}",
            f"Energy@min(3): {float(cumulative[min(2, cumulative.size - 1)]):.4g}" if cumulative.size else "Energy@min(3): n/a",
        ])

        def _draw(figure):
            figure.clear()
            axes = figure.subplots(2, 2)
            ax = axes[0, 0]
            if singular_values.size:
                x = np.arange(1, singular_values.size + 1)
                ax.plot(x, singular_values, marker="o", color="#1d4ed8", linewidth=1.8, label="singular value")
                ax.set_ylabel("Singular value", color="#1d4ed8")
                ax.tick_params(axis="y", labelcolor="#1d4ed8")
                energy_ax = ax.twinx()
                energy_ax.plot(x, cumulative, marker="s", color="#dc2626", linewidth=1.4, label="cumulative energy")
                energy_ax.set_ylabel("Cumulative energy", color="#dc2626")
                energy_ax.tick_params(axis="y", labelcolor="#dc2626")
                energy_ax.set_ylim(0.0, 1.02)
            ax.set_title("W subspace singular spectrum")
            ax.set_xlabel("Component")

            ax = axes[0, 1]
            image = ax.imshow(factor_overlap, aspect="auto", interpolation="nearest", cmap="seismic", vmin=-1.0, vmax=1.0)
            ax.set_title("Latent-factor overlap from W W^T")
            ax.set_xlabel("Latent factor")
            ax.set_ylabel("Latent factor")
            figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

            ax = axes[1, 0]
            if row_basis.size:
                vmax = float(np.nanpercentile(np.abs(row_basis), 98.0)) if row_basis.size else 1.0
                vmax = max(vmax, 1e-9)
                image = ax.imshow(row_basis, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-vmax, vmax=vmax)
                figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title("Orthonormal channel subspace basis V")
            ax.set_xlabel("Selected channel")
            ax.set_ylabel("Subspace axis")
            if labels and row_basis.shape[1] == len(labels):
                tick_indices = _display_indices(len(labels), 10)
                ax.set_xticks(tick_indices)
                ax.set_xticklabels([labels[int(index)] for index in tick_indices], rotation=45, ha="right", fontsize=7)

            ax = axes[1, 1]
            image = ax.imshow(channel_projection, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=1.0)
            ax.set_title("Channel-space projection P = V^T V")
            ax.set_xlabel("Selected channel")
            ax.set_ylabel("Selected channel")
            if labels and channel_projection.shape[0] == len(labels):
                tick_indices = _display_indices(len(labels), 8)
                ax.set_xticks(tick_indices)
                ax.set_yticks(tick_indices)
                ax.set_xticklabels([labels[int(index)] for index in tick_indices], rotation=45, ha="right", fontsize=7)
                ax.set_yticklabels([labels[int(index)] for index in tick_indices], fontsize=7)
            figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        self._show_metric_figure("W subspace metrics", summary, _draw)

    def _temporal_latent_model(self) -> dict:
        analysis = self.current or {}
        latent = np.asarray(analysis.get("latent_states", []), dtype=float)
        observed = np.asarray(analysis.get("observed_states", []), dtype=float)
        params = analysis.get("latent_params", {}) or {}
        loadings = np.asarray(params.get("loadings", []), dtype=float)
        mean = np.asarray(params.get("mean", []), dtype=float)
        if latent.ndim != 3 or latent.shape[0] == 0 or latent.shape[1] < 2 or latent.shape[2] == 0:
            return {}
        if observed.ndim != 3 or observed.shape[:2] != latent.shape[:2] or loadings.ndim != 2 or mean.ndim != 1:
            return {}

        history_bins = min(max(1, int(self.history_bins.value())), max(1, latent.shape[1] - 1))

        def _lagged_design(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            chunks = []
            targets = []
            burst_ids = []
            for time_index in range(history_bins, states.shape[1]):
                chunks.append(states[:, time_index - history_bins:time_index, :].reshape((states.shape[0], history_bins * states.shape[2])))
                targets.append(states[:, time_index, :])
                burst_ids.append(np.arange(states.shape[0], dtype=int))
            if not chunks:
                return (
                    np.zeros((0, history_bins * latent.shape[2]), dtype=float),
                    np.zeros((0, latent.shape[2]), dtype=float),
                    np.zeros(0, dtype=int),
                )
            return np.vstack(chunks), np.vstack(targets), np.concatenate(burst_ids)

        source, target, burst_ids = _lagged_design(latent)
        if source.shape[0] == 0 or target.shape[0] == 0:
            return {}
        burst_count = int(latent.shape[0])
        if burst_count >= 2:
            validation_burst_count = min(max(1, int(round(burst_count * 0.2))), max(1, burst_count - 1))
            validation_bursts = np.unique(np.linspace(0, burst_count - 1, validation_burst_count, dtype=int))
            validation_mask = np.isin(burst_ids, validation_bursts)
            if not np.any(validation_mask) or np.all(validation_mask):
                validation_mask = burst_ids == int(burst_count - 1)
        else:
            validation_mask = np.zeros(source.shape[0], dtype=bool)
            validation_bursts = np.zeros(0, dtype=int)
        train_mask = ~validation_mask
        if not np.any(train_mask):
            train_mask = np.ones(source.shape[0], dtype=bool)
            validation_mask = np.zeros(source.shape[0], dtype=bool)
            validation_bursts = np.zeros(0, dtype=int)

        train_source = source[train_mask]
        train_target = target[train_mask]
        val_source = source[validation_mask]
        val_target = target[validation_mask]

        design = np.column_stack([source, np.ones(source.shape[0], dtype=float)])
        try:
            coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        except np.linalg.LinAlgError:
            return {}
        dynamics = coef[:-1, :].T
        intercept = coef[-1, :]
        selected_method = str(self.temporal_method.currentData() or "linear")

        train_design = np.column_stack([train_source, np.ones(train_source.shape[0], dtype=float)])
        try:
            train_coef, *_ = np.linalg.lstsq(train_design, train_target, rcond=None)
        except np.linalg.LinAlgError:
            train_coef = coef
        train_dynamics = train_coef[:-1, :].T
        train_intercept = train_coef[-1, :]

        linear_predicted = np.zeros_like(latent)
        linear_predicted[:, :history_bins, :] = latent[:, :history_bins, :]
        for time_index in range(history_bins, latent.shape[1]):
            query = linear_predicted[:, time_index - history_bins:time_index, :].reshape((latent.shape[0], history_bins * latent.shape[2]))
            linear_predicted[:, time_index, :] = query @ train_dynamics.T + train_intercept

        max_train = min(3000, train_source.shape[0])
        train_indices = np.linspace(0, train_source.shape[0] - 1, max_train, dtype=int) if train_source.shape[0] > max_train else np.arange(train_source.shape[0], dtype=int)
        train_source = train_source[train_indices]
        train_target = train_target[train_indices]
        nonlinear_candidates = []
        gamma = 1.0 / max(1, latent.shape[2])

        def _as_2d_prediction(values, row_count: int) -> np.ndarray:
            predicted = np.asarray(values, dtype=float)
            if predicted.ndim == 1:
                predicted = predicted.reshape((row_count, 1))
            if predicted.shape != (row_count, latent.shape[2]):
                predicted = np.resize(predicted, (row_count, latent.shape[2]))
            return np.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0)

        def _rollout_from_predictor(predict_callback):
            predicted = np.zeros_like(latent)
            predicted[:, :history_bins, :] = latent[:, :history_bins, :]
            for time_index in range(history_bins, latent.shape[1]):
                query = predicted[:, time_index - history_bins:time_index, :].reshape((latent.shape[0], history_bins * latent.shape[2]))
                predicted[:, time_index, :] = _as_2d_prediction(predict_callback(query), query.shape[0])
            return predicted

        def _candidate_from_predictor(name: str, predict_callback, metadata: dict | None = None):
            train_one_step = _as_2d_prediction(predict_callback(train_source), train_source.shape[0]) if train_source.shape[0] else np.zeros((0, latent.shape[2]), dtype=float)
            train_one_step_rmse = float(np.sqrt(np.mean((train_target - train_one_step) ** 2))) if train_one_step.size else 0.0
            val_one_step = _as_2d_prediction(predict_callback(val_source), val_source.shape[0]) if val_source.shape[0] else np.zeros((0, latent.shape[2]), dtype=float)
            val_one_step_rmse = float(np.sqrt(np.mean((val_target - val_one_step) ** 2))) if val_one_step.size else float("nan")
            predicted = _rollout_from_predictor(predict_callback)
            reconstructed_flat = predicted.reshape((-1, predicted.shape[2])) @ loadings + mean
            reconstructed = reconstructed_flat.reshape(observed.shape)
            residual = observed - reconstructed
            time_rmse = np.sqrt(np.mean(residual ** 2, axis=(0, 2))) if residual.size else np.zeros(latent.shape[1], dtype=float)
            centered = observed - np.mean(observed, axis=(0, 1), keepdims=True)
            r2 = 1.0 - float(np.sum(residual ** 2)) / max(float(np.sum(centered ** 2)), 1e-12)
            return {
                "name": name,
                "predicted_latent": predicted,
                "one_step_rmse": float(val_one_step_rmse if np.isfinite(val_one_step_rmse) else train_one_step_rmse),
                "train_one_step_rmse": float(train_one_step_rmse),
                "val_one_step_rmse": float(val_one_step_rmse),
                "r2": float(r2),
                "time_rmse": time_rmse,
                "metadata": dict(metadata or {}),
            }

        def _fit_sklearn_candidate(name: str, estimator, fit_source=None, fit_target=None, metadata: dict | None = None):
            xs = train_source if fit_source is None else np.asarray(fit_source, dtype=float)
            ys = train_target if fit_target is None else np.asarray(fit_target, dtype=float)
            if xs.shape[0] < 2:
                return
            try:
                estimator.fit(xs, ys)
                nonlinear_candidates.append(_candidate_from_predictor(name, estimator.predict, metadata))
            except Exception:
                return

        kernel_alpha = np.zeros((0, latent.shape[2]), dtype=float)
        if selected_method == "rbf" and train_source.shape[0] >= 2:
            diffs = train_source[: min(512, train_source.shape[0])]
            pairwise = np.sum((diffs[:, None, :] - diffs[None, :, :]) ** 2, axis=2)
            positive_distances = pairwise[pairwise > 1e-12]
            if positive_distances.size:
                gamma = 1.0 / max(float(np.median(positive_distances)), 1e-12)
            train_sq = np.sum(train_source ** 2, axis=1, keepdims=True)
            kernel = np.exp(-gamma * np.maximum(train_sq + train_sq.T - 2.0 * train_source @ train_source.T, 0.0))
            ridge = 1e-3 * max(1.0, float(np.trace(kernel)) / max(1, kernel.shape[0]))
            try:
                kernel_alpha = np.linalg.solve(kernel + ridge * np.eye(kernel.shape[0]), train_target)
                def _rbf_predict(query):
                    query = np.asarray(query, dtype=float)
                    query_sq = np.sum(query ** 2, axis=1, keepdims=True)
                    train_sq_flat = train_sq.T
                    query_kernel = np.exp(-gamma * np.maximum(query_sq + train_sq_flat - 2.0 * query @ train_source.T, 0.0))
                    return query_kernel @ kernel_alpha

                nonlinear_candidates.append(
                    _candidate_from_predictor(
                        "RBF kernel ridge",
                        _rbf_predict,
                        {"gamma": float(gamma), "train_count": int(train_source.shape[0])},
                    )
                )
            except np.linalg.LinAlgError:
                kernel_alpha = np.zeros((0, latent.shape[2]), dtype=float)

        if selected_method == "knn" and train_source.shape[0] >= 3:
            neighbor_count = min(12, max(1, int(np.sqrt(train_source.shape[0]))))
            _fit_sklearn_candidate(
                "kNN local average",
                make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=neighbor_count, weights="distance")),
                metadata={"neighbors": int(neighbor_count), "train_count": int(train_source.shape[0])},
            )
        if selected_method == "poly" and train_source.shape[0] >= 6:
            _fit_sklearn_candidate(
                "Polynomial ridge",
                make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=1.0)),
                metadata={"degree": 2, "train_count": int(train_source.shape[0])},
            )
        if selected_method in {"rf", "gb"} and train_source.shape[0] >= 10:
            forest_count = min(1800, train_source.shape[0])
            forest_indices = np.linspace(0, train_source.shape[0] - 1, forest_count, dtype=int) if train_source.shape[0] > forest_count else np.arange(train_source.shape[0], dtype=int)
            if selected_method == "rf":
                _fit_sklearn_candidate(
                    "Random forest",
                    RandomForestRegressor(
                        n_estimators=80,
                        max_depth=12,
                        min_samples_leaf=2,
                        random_state=7,
                        n_jobs=1,
                    ),
                    fit_source=train_source[forest_indices],
                    fit_target=train_target[forest_indices],
                    metadata={"trees": 80, "train_count": int(forest_indices.size)},
                )
            if selected_method == "gb":
                boosting_count = min(1200, train_source.shape[0])
                boosting_indices = np.linspace(0, train_source.shape[0] - 1, boosting_count, dtype=int) if train_source.shape[0] > boosting_count else np.arange(train_source.shape[0], dtype=int)
                _fit_sklearn_candidate(
                    "Gradient boosting",
                    MultiOutputRegressor(
                        GradientBoostingRegressor(
                            n_estimators=90,
                            learning_rate=0.05,
                            max_depth=3,
                            random_state=7,
                        )
                    ),
                    fit_source=train_source[boosting_indices],
                    fit_target=train_target[boosting_indices],
                    metadata={"estimators": 90, "train_count": int(boosting_indices.size)},
                )
        if selected_method == "mlp" and train_source.shape[0] >= 20:
            mlp_count = min(1500, train_source.shape[0])
            mlp_indices = np.linspace(0, train_source.shape[0] - 1, mlp_count, dtype=int) if train_source.shape[0] > mlp_count else np.arange(train_source.shape[0], dtype=int)
            hidden = max(16, min(96, latent.shape[2] * 4))
            _fit_sklearn_candidate(
                "MLP tanh",
                make_pipeline(
                    StandardScaler(),
                    MLPRegressor(
                        hidden_layer_sizes=(hidden, hidden),
                        activation="tanh",
                        alpha=1e-3,
                        learning_rate_init=1e-3,
                        max_iter=250,
                        early_stopping=bool(mlp_indices.size >= 50),
                        random_state=7,
                    ),
                ),
                fit_source=train_source[mlp_indices],
                fit_target=train_target[mlp_indices],
                metadata={"hidden": int(hidden), "train_count": int(mlp_indices.size)},
            )

        if selected_method == "linear":
            linear_train_prediction = train_source @ train_dynamics.T + train_intercept if train_source.size else np.zeros((0, latent.shape[2]), dtype=float)
            linear_train_rmse = float(np.sqrt(np.mean((train_target - linear_train_prediction) ** 2))) if linear_train_prediction.size else 0.0
            linear_val_prediction = val_source @ train_dynamics.T + train_intercept if val_source.size else np.zeros((0, latent.shape[2]), dtype=float)
            linear_val_rmse = float(np.sqrt(np.mean((val_target - linear_val_prediction) ** 2))) if linear_val_prediction.size else float("nan")
            best_candidate = {
                "name": "Linear",
                "predicted_latent": linear_predicted,
                "one_step_rmse": float(linear_val_rmse if np.isfinite(linear_val_rmse) else linear_train_rmse),
                "train_one_step_rmse": float(linear_train_rmse),
                "val_one_step_rmse": float(linear_val_rmse),
                "r2": float("nan"),
                "time_rmse": np.zeros(latent.shape[1], dtype=float),
                "metadata": {},
            }
        elif nonlinear_candidates:
            best_candidate = min(
                nonlinear_candidates,
                key=lambda item: (
                    float(item.get("val_one_step_rmse", np.inf))
                    if np.isfinite(float(item.get("val_one_step_rmse", np.inf)))
                    else float(item.get("train_one_step_rmse", np.inf)),
                    -float(item["r2"]),
                ),
            )
        else:
            best_candidate = {
                "name": "Linear fallback",
                "predicted_latent": linear_predicted,
                "one_step_rmse": float("inf"),
                "train_one_step_rmse": float("inf"),
                "val_one_step_rmse": float("nan"),
                "r2": float("-inf"),
                "time_rmse": np.zeros(latent.shape[1], dtype=float),
                "metadata": {},
            }

        predicted = np.asarray(best_candidate.get("predicted_latent", linear_predicted), dtype=float)
        reconstructed_flat = predicted.reshape((-1, predicted.shape[2])) @ loadings + mean
        reconstructed = reconstructed_flat.reshape(observed.shape)
        raw_reconstructed = _burst_trajectory_inverse_features(reconstructed, analysis.get("normalization_params", {}) or {})
        residual = observed - reconstructed
        centered = observed - np.mean(observed, axis=(0, 1), keepdims=True)
        r2 = 1.0 - float(np.sum(residual ** 2)) / max(float(np.sum(centered ** 2)), 1e-12)
        time_rmse = np.sqrt(np.mean(residual ** 2, axis=(0, 2))) if residual.size else np.zeros(latent.shape[1], dtype=float)
        linear_reconstructed_flat = linear_predicted.reshape((-1, linear_predicted.shape[2])) @ loadings + mean
        linear_reconstructed = linear_reconstructed_flat.reshape(observed.shape)
        linear_residual = observed - linear_reconstructed
        linear_time_rmse = np.sqrt(np.mean(linear_residual ** 2, axis=(0, 2))) if linear_residual.size else np.zeros(latent.shape[1], dtype=float)
        linear_r2 = 1.0 - float(np.sum(linear_residual ** 2)) / max(float(np.sum(centered ** 2)), 1e-12)
        linear_latent_residual = train_target - (train_source @ train_dynamics.T + train_intercept)
        linear_latent_rmse = float(np.sqrt(np.mean(linear_latent_residual ** 2))) if linear_latent_residual.size else 0.0
        nonlinear_one_step_rmse = float(best_candidate.get("one_step_rmse", linear_latent_rmse))
        candidate_metrics = [
            {
                "name": str(item.get("name", "")),
                "one_step_rmse": float(item.get("one_step_rmse", 0.0)),
                "train_one_step_rmse": float(item.get("train_one_step_rmse", 0.0)),
                "val_one_step_rmse": float(item.get("val_one_step_rmse", float("nan"))),
                "r2": float(item.get("r2", 0.0)),
                "metadata": dict(item.get("metadata", {}) or {}),
            }
            for item in sorted(
                nonlinear_candidates,
                key=lambda item: (
                    float(item.get("val_one_step_rmse", np.inf))
                    if np.isfinite(float(item.get("val_one_step_rmse", np.inf)))
                    else float(item.get("train_one_step_rmse", np.inf))
                ),
            )
        ]
        return {
            "predicted_latent": predicted,
            "linear_predicted_latent": linear_predicted,
            "reconstructed_states": reconstructed,
            "raw_reconstructed_states": raw_reconstructed,
            "dynamics": dynamics,
            "intercept": intercept,
            "history_bins": int(history_bins),
            "gamma": float(gamma),
            "kernel_train_count": int(train_source.shape[0]) if kernel_alpha.size else 0,
            "best_method": str(best_candidate.get("name", "Linear fallback")),
            "selected_method": selected_method,
            "candidate_metrics": candidate_metrics,
            "train_burst_count": int(np.unique(burst_ids[train_mask]).size) if burst_ids.size else 0,
            "val_burst_count": int(np.unique(burst_ids[validation_mask]).size) if np.any(validation_mask) else 0,
            "validation_bursts": validation_bursts.astype(int).tolist(),
            "r2": float(r2),
            "linear_r2": float(linear_r2),
            "time_rmse": time_rmse,
            "linear_time_rmse": linear_time_rmse,
            "latent_rmse": float(nonlinear_one_step_rmse),
            "linear_latent_rmse": linear_latent_rmse,
            "train_latent_rmse": float(best_candidate.get("train_one_step_rmse", linear_latent_rmse)),
            "val_latent_rmse": float(best_candidate.get("val_one_step_rmse", float("nan"))),
        }

    def _show_temporal_model(self):
        if not self.current:
            _show_info_message(self, "Temporal model", "Run factor analysis first.")
            return
        model = self._temporal_latent_model()
        if not model:
            _show_info_message(self, "Temporal model", "At least two time bins and valid FA states are required.")
            return
        analysis = self.current or {}
        raw_observed = np.asarray(analysis.get("raw_observed_states", []), dtype=float)
        raw_reconstructed = np.asarray(model.get("raw_reconstructed_states", []), dtype=float)
        latent = np.asarray(analysis.get("latent_states", []), dtype=float)
        predicted_latent = np.asarray(model.get("predicted_latent", []), dtype=float)
        burst_index = self._selected_burst_index(raw_observed.shape[0] if raw_observed.ndim == 3 else 1)
        best_method = str(model.get("best_method", "Nonlinear"))
        selected_method = str(model.get("selected_method", best_method))
        history_bins = int(model.get("history_bins", self.history_bins.value()))
        candidate_metrics = list(model.get("candidate_metrics", []))
        ranked = ", ".join(
            f"{item.get('name', '')}: val {float(item.get('val_one_step_rmse', 0.0)):.4g}"
            if np.isfinite(float(item.get("val_one_step_rmse", float("nan"))))
            else f"{item.get('name', '')}: train {float(item.get('train_one_step_rmse', 0.0)):.4g}"
            for item in candidate_metrics[:4]
        )
        val_latent_rmse = float(model.get("val_latent_rmse", float("nan")))
        val_rmse_text = f"{val_latent_rmse:.6g}" if np.isfinite(val_latent_rmse) else "n/a"
        summary = " | ".join([
            f"Temporal latent dynamics: method={selected_method}",
            f"history bins: {history_bins}",
            f"train/val bursts: {int(model.get('train_burst_count', 0))}/{int(model.get('val_burst_count', 0))}",
            f"nonlinear R2: {float(model.get('r2', 0.0)):.6g}",
            f"linear R2: {float(model.get('linear_r2', 0.0)):.6g}",
            f"validation one-step z RMSE: {val_rmse_text}",
            f"candidate validation RMSE: {ranked}" if ranked else "candidate validation RMSE: n/a",
        ])

        def _draw(figure):
            figure.clear()
            axes = figure.subplots(2, 2)
            window_ms = float(analysis.get("window_ms", self.window_ms.value()))
            centers_ms = np.asarray(analysis.get("centers_ms", []), dtype=float)
            if raw_observed.ndim == 3 and raw_reconstructed.shape == raw_observed.shape:
                true_block = raw_observed[burst_index].T
                recon_block = raw_reconstructed[burst_index].T
                heatmap = np.vstack([true_block, recon_block])
                vmax = max(float(np.nanpercentile(heatmap, 98.0)) if heatmap.size else 1.0, 1e-9)
                ax = axes[0, 0]
                image = ax.imshow(heatmap, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=vmax, extent=[0.0, window_ms, heatmap.shape[0], 0.0])
                ax.axhline(true_block.shape[0] - 0.5, color="#111827", linewidth=1.0)
                ax.set_title(f"Raw data vs temporal reconstruction | {best_method} | history {history_bins} | sample {burst_index + 1}")
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Observed / temporal recon channels")
                figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

                ax = axes[0, 1]
                observed_psth = np.mean(raw_observed[burst_index], axis=1)
                reconstructed_psth = np.mean(raw_reconstructed[burst_index], axis=1)
                if centers_ms.size != observed_psth.size:
                    centers_ms = np.linspace(0.0, window_ms, observed_psth.size, endpoint=False)
                ax.plot(centers_ms, observed_psth, color="#1d4ed8", linewidth=1.8, label="Observed")
                ax.plot(centers_ms, reconstructed_psth, color="#dc2626", linewidth=1.5, label=f"{best_method} recon")
                ax.set_title("Raw PSTH reconstruction")
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Mean firing rate (Hz)")
                ax.legend(loc="best", fontsize=8)
            else:
                for ax in axes[0, :]:
                    ax.text(0.5, 0.5, "No raw reconstruction", ha="center", va="center")
                    ax.set_xticks([])
                    ax.set_yticks([])

            ax = axes[1, 0]
            if latent.ndim == 3 and predicted_latent.shape == latent.shape:
                show_dims = min(4, latent.shape[2])
                for dim in range(show_dims):
                    ax.plot(latent[burst_index, :, dim], linewidth=1.5, label=f"z{dim + 1}")
                    ax.plot(predicted_latent[burst_index, :, dim], linewidth=1.0, linestyle="--", label=f"pred z{dim + 1}")
                ax.set_title(f"Observed vs {best_method} predicted z | history {history_bins}")
                ax.set_xlabel("Time bin")
                ax.set_ylabel("Latent value")
                ax.legend(loc="best", fontsize=7, ncol=2)
            else:
                ax.text(0.5, 0.5, "No latent prediction", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])

            ax = axes[1, 1]
            time_rmse = np.asarray(model.get("time_rmse", []), dtype=float)
            linear_time_rmse = np.asarray(model.get("linear_time_rmse", []), dtype=float)
            if time_rmse.size:
                x = centers_ms if centers_ms.size == time_rmse.size else np.arange(time_rmse.size)
                ax.plot(x, time_rmse, color="#7c3aed", marker="o", linewidth=1.6, label=best_method)
                if linear_time_rmse.size == time_rmse.size:
                    ax.plot(x, linear_time_rmse, color="#64748b", marker="s", linewidth=1.2, linestyle="--", label="Linear")
                    ax.legend(loc="best", fontsize=8)
                if candidate_metrics:
                    ranking = "\n".join(
                        (
                            f"{index + 1}. {item.get('name', '')}: "
                            f"val {float(item.get('val_one_step_rmse', 0.0)):.3g}, "
                            f"train {float(item.get('train_one_step_rmse', 0.0)):.3g}"
                        )
                        if np.isfinite(float(item.get("val_one_step_rmse", float("nan"))))
                        else f"{index + 1}. {item.get('name', '')}: train {float(item.get('train_one_step_rmse', 0.0)):.3g}"
                        for index, item in enumerate(candidate_metrics[:5])
                    )
                    ax.text(
                        0.98,
                        0.98,
                        ranking,
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=8,
                        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
                    )
            ax.set_title("Temporal reconstruction RMSE over time")
            ax.set_xlabel("Time (ms)" if centers_ms.size == time_rmse.size else "Time bin")
            ax.set_ylabel("RMSE")

        self._show_metric_figure("Temporal latent model", summary, _draw)

    def _spatial_temporal_regions(self) -> dict:
        analysis = self.current or {}
        signature = (
            id(analysis),
            id(self.channel_map),
            round(float(self.activity_similarity_weight.value()), 4),
            round(float(self.spatial_similarity_weight.value()), 4),
            round(float(self.region_membership_threshold.value()), 4),
            round(float(self.bin_ms.value()), 4),
            round(float(self.window_ms.value()), 4),
        )
        if self._spatial_temporal_cache_signature == signature and isinstance(self._spatial_temporal_cache, dict):
            return self._spatial_temporal_cache
        if self.channel_map is None:
            return {}
        labels = [str(label) for label in analysis.get("selected_labels", [])]
        intervals = list(analysis.get("intervals", []))
        raw_observed = np.asarray(analysis.get("raw_observed_states", []), dtype=float)
        if raw_observed.ndim != 3 or raw_observed.shape[0] == 0 or raw_observed.shape[2] == 0 or not labels:
            return {}
        position_lookup, _all_positions = _channel_map_positions(self.channel_map)
        valid_indices = []
        positions = []
        for index, label in enumerate(labels):
            position = _position_for_channel(label, position_lookup)
            if position is None:
                continue
            valid_indices.append(index)
            positions.append((float(position[0]), float(position[1])))
        if len(valid_indices) < 4:
            return {}

        valid_indices = np.asarray(valid_indices, dtype=int)
        positions_array = np.asarray(positions, dtype=float)
        activity = raw_observed[:, :, valid_indices]
        flattened = np.transpose(activity, (2, 0, 1)).reshape((len(valid_indices), -1))
        centered = flattened - np.mean(flattened, axis=1, keepdims=True)
        std = np.std(centered, axis=1, keepdims=True)
        std[std < 1e-9] = 1.0
        normalized = centered / std
        if normalized.shape[0] < 4:
            return {}
        corr = np.corrcoef(normalized)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = np.clip(0.5 * (corr + corr.T), -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        spatial_distance = np.sqrt(np.sum((positions_array[:, None, :] - positions_array[None, :, :]) ** 2, axis=2))
        spatial_scale = float(np.nanmedian(spatial_distance[spatial_distance > 0])) if np.any(spatial_distance > 0) else 1.0
        spatial_scale = max(spatial_scale, 1e-6)
        spatial_affinity = np.exp(-0.5 * (spatial_distance / spatial_scale) ** 2)
        activity_weight = float(self.activity_similarity_weight.value())
        spatial_weight = float(self.spatial_similarity_weight.value())
        total_weight = activity_weight + spatial_weight
        if total_weight <= 1e-9:
            activity_weight, spatial_weight = 0.78, 0.22
            total_weight = 1.0
        activity_weight /= total_weight
        spatial_weight /= total_weight
        activity_similarity = np.maximum(corr, 0.0)
        similarity = activity_weight * activity_similarity + spatial_weight * spatial_affinity
        similarity = np.clip(0.5 * (similarity + similarity.T), 0.0, 1.0)
        np.fill_diagonal(similarity, 1.0)
        distance = np.clip(1.0 - similarity, 0.0, 1.0)
        if distance.shape[0] < 2:
            return {}
        condensed = squareform(distance, checks=False)
        if condensed.size == 0:
            return {}
        tree = linkage(condensed, method="average", optimal_ordering=distance.shape[0] <= 256)

        max_clusters = int(min(8, max(2, round(np.sqrt(len(valid_indices))))))
        best_assignments = None
        best_labels = None
        best_score = -np.inf
        candidate_metrics = []
        embedding = np.hstack([normalized, 0.55 * positions_array])
        for cluster_count in range(2, max_clusters + 1):
            labels_candidate = fcluster(tree, t=cluster_count, criterion="maxclust").astype(int)
            unique_labels = np.unique(labels_candidate)
            if unique_labels.size < 2:
                continue
            counts = np.asarray([int(np.sum(labels_candidate == label)) for label in unique_labels], dtype=int)
            if np.any(counts < 2):
                continue
            try:
                silhouette = float(silhouette_score(embedding, labels_candidate))
            except Exception:
                silhouette = float("nan")
            within_scores = []
            between_scores = []
            for label in unique_labels:
                member_mask = labels_candidate == label
                member_similarity = similarity[np.ix_(member_mask, member_mask)]
                if member_similarity.size > member_mask.sum():
                    upper = member_similarity[np.triu_indices(member_similarity.shape[0], k=1)]
                    if upper.size:
                        within_scores.append(float(np.mean(upper)))
                outer_mask = ~member_mask
                cross_similarity = similarity[np.ix_(member_mask, outer_mask)]
                if cross_similarity.size:
                    between_scores.append(float(np.mean(cross_similarity)))
            within_mean = float(np.mean(within_scores)) if within_scores else 0.0
            between_mean = float(np.mean(between_scores)) if between_scores else 0.0
            modularity_gap = within_mean - between_mean
            score = modularity_gap + 0.35 * (silhouette if np.isfinite(silhouette) else -1.0)
            candidate_metrics.append(
                {
                    "cluster_count": int(cluster_count),
                    "silhouette": float(silhouette) if np.isfinite(silhouette) else float("nan"),
                    "within_similarity": within_mean,
                    "between_similarity": between_mean,
                    "modularity_gap": modularity_gap,
                    "score": score,
                }
            )
            if score > best_score:
                best_score = score
                best_assignments = labels_candidate
                best_labels = unique_labels

        if best_assignments is None or best_labels is None or len(best_labels) < 2:
            return {}
        assignments = best_assignments
        initial_module_labels = [np.flatnonzero(assignments == label) for label in best_labels]
        if len(initial_module_labels) < 2:
            return {}

        membership_threshold = float(self.region_membership_threshold.value())
        channel_region_scores = np.full(len(valid_indices), np.nan, dtype=float)
        channel_region_margin = np.full(len(valid_indices), np.nan, dtype=float)
        filtered_module_labels = []
        filtered_label_ids = []
        channel_significant_mask = np.zeros(len(valid_indices), dtype=bool)
        for label_index, member_indices in enumerate(initial_module_labels):
            if member_indices.size < 2:
                continue
            other_indices = np.setdiff1d(np.arange(len(valid_indices), dtype=int), member_indices, assume_unique=False)
            cluster_keep = []
            for channel_index in member_indices:
                same_mask = member_indices[member_indices != channel_index]
                within_similarity = float(np.mean(similarity[channel_index, same_mask])) if same_mask.size else 1.0
                other_scores = []
                for other_label_indices in initial_module_labels:
                    if other_label_indices is member_indices or other_label_indices.size == 0:
                        continue
                    other_scores.append(float(np.mean(similarity[channel_index, other_label_indices])))
                best_other = float(max(other_scores)) if other_scores else 0.0
                margin = within_similarity - best_other
                channel_region_scores[channel_index] = within_similarity
                channel_region_margin[channel_index] = margin
                if within_similarity >= membership_threshold and margin >= max(0.05, membership_threshold * 0.35):
                    cluster_keep.append(int(channel_index))
                    channel_significant_mask[channel_index] = True
            if len(cluster_keep) >= 2:
                filtered_module_labels.append(np.asarray(cluster_keep, dtype=int))
                filtered_label_ids.append(label_index)

        module_labels = filtered_module_labels
        if len(module_labels) < 2:
            return {}
        retained_channels = int(np.sum(channel_significant_mask))

        bin_centers = np.asarray(analysis.get("centers_ms", []), dtype=float)
        if bin_centers.size != raw_observed.shape[1]:
            bin_centers = np.linspace(0.0, float(analysis.get("window_ms", self.window_ms.value())), raw_observed.shape[1], endpoint=False)

        channel_peak_times = np.full((raw_observed.shape[0], len(valid_indices)), np.nan, dtype=float)
        for burst_index in range(raw_observed.shape[0]):
            burst_block = np.asarray(activity[burst_index], dtype=float)
            if burst_block.ndim != 2:
                continue
            for channel_index in range(burst_block.shape[1]):
                trace = np.asarray(burst_block[:, channel_index], dtype=float)
                if trace.ndim != 1 or trace.size == 0 or not np.any(trace > 0):
                    continue
                threshold = max(np.nanpercentile(trace[trace > 0], 70.0), np.nanmax(trace) * 0.35)
                active_bins = np.flatnonzero(trace >= threshold)
                if active_bins.size == 0:
                    active_bins = np.flatnonzero(trace > 0)
                if active_bins.size == 0:
                    continue
                first_bin = int(active_bins[0])
                peak_bin = int(np.argmax(trace))
                stop_bin = min(max(first_bin + 1, peak_bin + 1), trace.size, bin_centers.size)
                if stop_bin <= first_bin:
                    stop_bin = min(first_bin + 1, trace.size, bin_centers.size)
                if stop_bin <= first_bin:
                    continue
                weight_slice = np.asarray(trace[first_bin:stop_bin], dtype=float)
                center_slice = np.asarray(bin_centers[first_bin:stop_bin], dtype=float)
                usable = min(weight_slice.size, center_slice.size)
                if usable <= 0:
                    continue
                weight_slice = weight_slice[:usable]
                center_slice = center_slice[:usable]
                if np.sum(weight_slice) > 0:
                    channel_peak_times[burst_index, channel_index] = float(np.average(center_slice, weights=weight_slice))
                else:
                    channel_peak_times[burst_index, channel_index] = float(bin_centers[first_bin])

        module_times = np.full((raw_observed.shape[0], len(module_labels)), np.nan, dtype=float)
        for burst_index in range(raw_observed.shape[0]):
            for module_index, member_indices in enumerate(module_labels):
                burst_block = np.asarray(activity[burst_index], dtype=float)
                if burst_block.ndim != 2:
                    continue
                channel_block = np.take(burst_block, member_indices, axis=1)
                if channel_block.ndim != 2 or channel_block.shape[0] == 0:
                    continue
                module_trace = np.mean(channel_block, axis=1)
                if not np.any(module_trace > 0):
                    continue
                threshold = max(np.nanpercentile(module_trace[module_trace > 0], 70.0), np.nanmax(module_trace) * 0.35)
                active_bins = np.flatnonzero(module_trace >= threshold)
                if active_bins.size == 0:
                    active_bins = np.flatnonzero(module_trace > 0)
                if active_bins.size == 0:
                    continue
                first_bin = int(active_bins[0])
                stop_bin = min(first_bin + 3, module_trace.size, bin_centers.size)
                if stop_bin <= first_bin:
                    continue
                weight_slice = np.asarray(module_trace[first_bin:stop_bin], dtype=float)
                center_slice = np.asarray(bin_centers[first_bin:stop_bin], dtype=float)
                usable = min(weight_slice.size, center_slice.size)
                if usable <= 0:
                    continue
                weight_slice = weight_slice[:usable]
                center_slice = center_slice[:usable]
                if weight_slice.size and np.sum(weight_slice) > 0:
                    module_times[burst_index, module_index] = float(np.average(center_slice, weights=weight_slice))
                else:
                    module_times[burst_index, module_index] = float(bin_centers[first_bin])

        module_centers = []
        module_sizes = []
        edges = []
        for module_index, member_indices in enumerate(module_labels):
            coords = positions_array[member_indices]
            module_centers.append(np.mean(coords, axis=0))
            module_sizes.append(int(member_indices.size))
        module_centers = np.asarray(module_centers, dtype=float)
        for source_index in range(len(module_labels)):
            for target_index in range(len(module_labels)):
                if source_index == target_index:
                    continue
                source_members = np.asarray(module_labels[source_index], dtype=int)
                target_members = np.asarray(module_labels[target_index], dtype=int)
                true_delays = []
                background_delays = []
                for burst_index in range(channel_peak_times.shape[0]):
                    source_times = channel_peak_times[burst_index, source_members]
                    target_times = channel_peak_times[burst_index, target_members]
                    source_valid = source_times[np.isfinite(source_times)]
                    target_valid = target_times[np.isfinite(target_times)]
                    if source_valid.size == 0 or target_valid.size == 0:
                        continue
                    burst_true = (target_valid[:, None] - source_valid[None, :]).reshape(-1)
                    if burst_true.size == 0:
                        continue
                    true_delays.extend(float(value) for value in burst_true[np.isfinite(burst_true)])
                    source_shuffled = np.random.permutation(source_valid)
                    target_shuffled = np.random.permutation(target_valid)
                    burst_background = (target_shuffled[:, None] - source_shuffled[None, :]).reshape(-1)
                    if burst_background.size:
                        background_delays.extend(float(value) for value in burst_background[np.isfinite(burst_background)])

                true_values = np.asarray(true_delays, dtype=float)
                background_values = np.asarray(background_delays, dtype=float)
                if true_values.size < max(12, raw_observed.shape[0] * 2) or background_values.size < max(12, raw_observed.shape[0] * 2):
                    continue
                min_delay = max(0.5 * float(self.bin_ms.value()), 1.0)
                positive_true = true_values[true_values > min_delay]
                if positive_true.size < max(8, int(np.ceil(true_values.size * 0.18))):
                    continue
                max_delay = max(float(analysis.get("window_ms", self.window_ms.value())), float(min_delay) + float(self.bin_ms.value()))
                bin_width = max(1.0, float(self.bin_ms.value()))
                bin_edges = np.arange(-max_delay, max_delay + bin_width * 1.5, bin_width, dtype=float)
                if bin_edges.size < 4:
                    continue
                true_hist, edges_ms = np.histogram(true_values, bins=bin_edges)
                background_hist, _ = np.histogram(background_values, bins=bin_edges)
                centers_ms = 0.5 * (edges_ms[:-1] + edges_ms[1:])
                positive_mask = centers_ms > min_delay
                positive_true_hist = true_hist[positive_mask]
                positive_background_hist = background_hist[positive_mask]
                positive_centers = centers_ms[positive_mask]
                if positive_true_hist.size == 0 or int(np.max(positive_true_hist)) <= 0:
                    continue
                peak_local_index = int(np.argmax(positive_true_hist))
                peak_center_ms = float(positive_centers[peak_local_index])
                peak_count = int(positive_true_hist[peak_local_index])
                background_count = float(positive_background_hist[peak_local_index]) if peak_local_index < positive_background_hist.size else 0.0
                if peak_count <= max(3.0, background_count + 1.0):
                    continue
                window_mask = np.abs(centers_ms - peak_center_ms) <= 2.0 * bin_width
                peak_window_values = true_values[np.abs(true_values - peak_center_ms) <= 2.0 * bin_width]
                if peak_window_values.size == 0:
                    peak_window_values = positive_true
                median_delay = float(np.median(positive_true))
                peak_neighborhood_mean = float(np.mean(peak_window_values))
                distribution_shift = float(np.median(true_values) - np.median(background_values))
                if distribution_shift <= 0:
                    continue
                background_window_values = background_values[np.abs(background_values - peak_center_ms) <= 2.0 * bin_width]
                peak_to_background = float(peak_count) / max(1.0, background_count)
                if peak_to_background < 1.5:
                    continue
                edges.append(
                    {
                        "source": source_index,
                        "target": target_index,
                        "delay_ms": median_delay,
                        "median_delay_ms": median_delay,
                        "peak_delay_ms": peak_center_ms,
                        "peak_window_mean_ms": peak_neighborhood_mean,
                        "peak_to_background": peak_to_background,
                        "peak_count": peak_count,
                        "background_peak_count": background_count,
                        "distribution_shift": distribution_shift,
                        "true_values": true_values,
                        "background_values": background_values,
                        "true_hist": true_hist,
                        "background_hist": background_hist,
                        "hist_edges": edges_ms,
                        "window_mask": window_mask,
                        "window_count": int(peak_window_values.size),
                        "background_window_count": int(background_window_values.size),
                    }
                )

        result = {
            "labels": labels,
            "valid_indices": valid_indices,
            "positions": positions_array,
            "assignments": assignments,
            "channel_region_scores": channel_region_scores,
            "channel_region_margin": channel_region_margin,
            "channel_significant_mask": channel_significant_mask,
            "modules": module_labels,
            "module_centers": module_centers,
            "module_sizes": module_sizes,
            "edges": edges,
            "intervals": intervals,
            "similarity": similarity,
            "distance": distance,
            "activity_similarity": activity_similarity,
            "spatial_affinity": spatial_affinity,
            "activity_weight": activity_weight,
            "spatial_weight": spatial_weight,
            "membership_threshold": membership_threshold,
            "retained_channels": retained_channels,
            "cluster_metrics": candidate_metrics,
            "module_times": module_times,
            "channel_peak_times": channel_peak_times,
        }
        self._spatial_temporal_cache_signature = signature
        self._spatial_temporal_cache = result
        return result

    def _show_spatial_temporal_analysis(self):
        progress = _create_progress_dialog(self, "Spatial-temporal analysis", "Detecting regions and propagation delays...", 0) if _progress_enabled_for_widget(self) else None
        analysis = self._spatial_temporal_regions()
        _close_progress_dialog(progress)
        if not analysis:
            summary = (
                "Spatial-temporal analysis unavailable | "
                "Need a channel map plus enough routed channels with usable burst activity. "
                "Try increasing the window, lowering region threshold, or using richer burst data."
            )

            def _draw_empty(figure):
                figure.clear()
                axes = figure.subplots(2, 2)
                for axis in axes.flat:
                    axis.set_xticks([])
                    axis.set_yticks([])
                    axis.set_frame_on(False)
                axes[0, 0].text(
                    0.5,
                    0.58,
                    "Spatial-temporal regions and directed propagation",
                    ha="center",
                    va="center",
                    fontsize=12,
                    weight="bold",
                )
                axes[0, 0].set_title("Spatial-temporal regions and directed propagation")
                axes[0, 0].text(
                    0.5,
                    0.38,
                    "No stable regional structure was detected for the current selection.",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
                axes[0, 1].text(0.5, 0.5, "Similarity matrix unavailable", ha="center", va="center")
                axes[1, 0].text(0.5, 0.5, "Cluster validation unavailable", ha="center", va="center")
                axes[1, 1].text(0.5, 0.5, "Delay distribution unavailable", ha="center", va="center")

            self._show_metric_figure("Spatial-temporal analysis", summary, _draw_empty)
            return

        module_count = len(analysis.get("modules", []))
        edge_count = len(analysis.get("edges", []))
        cluster_metrics = list(analysis.get("cluster_metrics", []))
        best_metric = max(cluster_metrics, key=lambda item: float(item.get("score", -np.inf))) if cluster_metrics else None
        summary_parts = [
            f"Spatial-temporal modules: {module_count}",
            f"directed module links: {edge_count}",
            f"weights: act {float(analysis.get('activity_weight', 0.0)):.2f} / space {float(analysis.get('spatial_weight', 0.0)):.2f}",
            f"region thr: {float(analysis.get('membership_threshold', 0.0)):.2f}",
            f"retained channels: {int(analysis.get('retained_channels', 0))}/{len(analysis.get('valid_indices', []))}",
        ]
        if best_metric is not None:
            summary_parts.append(
                f"selected clusters: {int(best_metric.get('cluster_count', module_count))} | silhouette: {float(best_metric.get('silhouette', float('nan'))):.3f} | modularity gap: {float(best_metric.get('modularity_gap', 0.0)):.3f}"
            )
        if edge_count:
            best_edge = max(
                list(analysis.get("edges", [])),
                key=lambda item: (
                    float(item.get("peak_to_background", 0.0)),
                    float(item.get("distribution_shift", 0.0)),
                ),
            )
            summary_parts.append(
                "best edge "
                f"{int(best_edge['source']) + 1}->{int(best_edge['target']) + 1} | "
                f"median {float(best_edge.get('median_delay_ms', 0.0)):.2f} ms | "
                f"peak {float(best_edge.get('peak_delay_ms', 0.0)):.2f} ms | "
                f"peak-win mean {float(best_edge.get('peak_window_mean_ms', 0.0)):.2f} ms"
            )
        summary = " | ".join(summary_parts)

        def _draw(figure):
            figure.clear()
            axes = figure.subplots(2, 2)
            ax = axes[0, 0]
            ax.set_facecolor("white")
            positions = np.asarray(analysis["positions"], dtype=float)
            module_centers = np.asarray(analysis["module_centers"], dtype=float)
            modules = list(analysis["modules"])
            edges = list(analysis["edges"])
            channel_significant_mask = np.asarray(analysis.get("channel_significant_mask", []), dtype=bool)
            channel_region_margin = np.asarray(analysis.get("channel_region_margin", []), dtype=float)

            if positions.size == 0:
                ax.text(0.5, 0.5, "No positioned channels", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
                return

            palette = [
                "#1d4ed8",
                "#dc2626",
                "#16a34a",
                "#ea580c",
                "#7c3aed",
                "#0f766e",
                "#be123c",
                "#4338ca",
            ]
            _all_lookup, electrode_positions = _channel_map_positions(self.channel_map)
            background_points = [
                (float(x), float(y))
                for x, y, _payload in electrode_positions.values()
                if np.isfinite(float(x)) and np.isfinite(float(y))
            ]
            background = np.asarray(background_points, dtype=float) if background_points else np.zeros((0, 2), dtype=float)
            if background.size:
                ax.scatter(background[:, 0], background[:, 1], s=24, color="#d1d5db", alpha=0.55, marker="s", linewidths=0)

            if channel_significant_mask.size == positions.shape[0]:
                rejected = positions[~channel_significant_mask]
                if rejected.size:
                    ax.scatter(
                        rejected[:, 0],
                        rejected[:, 1],
                        s=54,
                        color="#94a3b8",
                        alpha=0.85,
                        marker="s",
                        edgecolors="#ffffff",
                        linewidths=0.4,
                        label="Filtered out",
                    )

            for module_index, member_indices in enumerate(modules):
                color = palette[module_index % len(palette)]
                module_points = positions[member_indices]
                ax.scatter(
                    module_points[:, 0],
                    module_points[:, 1],
                    s=90,
                    color=color,
                    alpha=0.92,
                    marker="s",
                    edgecolors="white",
                    linewidths=0.6,
                    label=f"Region {module_index + 1}",
                )
                center = module_centers[module_index]
                ax.text(center[0], center[1], str(module_index + 1), color="white", ha="center", va="center", fontsize=8, weight="bold")

            for edge in edges:
                source = int(edge["source"])
                target = int(edge["target"])
                start = module_centers[source]
                stop = module_centers[target]
                delta = stop - start
                distance = float(np.hypot(delta[0], delta[1]))
                if distance <= 1e-9:
                    continue
                scale = min(0.18, 12.0 / max(distance, 1.0))
                start_pt = start + delta * scale
                stop_pt = stop - delta * scale
                peak_ratio = float(edge.get("peak_to_background", 1.0))
                peak_ratio = max(1.0, min(5.0, peak_ratio))
                shift = abs(float(edge.get("distribution_shift", 0.0)))
                line_width = 1.2 + 0.45 * peak_ratio + min(1.0, shift / max(float(self.bin_ms.value()), 1.0)) * 0.4
                ax.annotate(
                    "",
                    xy=(stop_pt[0], stop_pt[1]),
                    xytext=(start_pt[0], start_pt[1]),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": "#111827",
                        "lw": line_width,
                        "alpha": 0.7,
                    },
                )
                mid = 0.5 * (start_pt + stop_pt)
                ax.text(
                    mid[0],
                    mid[1],
                    f"med {float(edge.get('median_delay_ms', 0.0)):.1f}\npeak {float(edge.get('peak_delay_ms', 0.0)):.1f}\nmean5 {float(edge.get('peak_window_mean_ms', 0.0)):.1f}",
                    fontsize=7,
                    color="#111827",
                    ha="center",
                    va="center",
                    bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.85, "pad": 0.2},
                )

            ax.set_title("Spatial-temporal regions and directed propagation")
            ax.set_aspect("equal", adjustable="box")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(loc="upper right", fontsize=8, frameon=True)

            similarity_ax = axes[0, 1]
            similarity = np.asarray(analysis.get("similarity", []), dtype=float)
            assignments = np.asarray(analysis.get("assignments", []), dtype=int)
            if similarity.ndim == 2 and similarity.size:
                order = np.argsort(assignments, kind="stable")
                ordered = similarity[np.ix_(order, order)]
                image = similarity_ax.imshow(ordered, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
                similarity_ax.set_title("Channel similarity matrix (clusters sorted)")
                similarity_ax.set_xlabel("Channel (cluster-sorted)")
                similarity_ax.set_ylabel("Channel (cluster-sorted)")
                figure.colorbar(image, ax=similarity_ax, fraction=0.046, pad=0.04)
            else:
                similarity_ax.text(0.5, 0.5, "No similarity matrix", ha="center", va="center")
                similarity_ax.set_xticks([])
                similarity_ax.set_yticks([])

            cluster_ax = axes[1, 0]
            if cluster_metrics:
                cluster_counts = np.asarray([int(item.get("cluster_count", 0)) for item in cluster_metrics], dtype=int)
                silhouettes = np.asarray([float(item.get("silhouette", np.nan)) for item in cluster_metrics], dtype=float)
                gaps = np.asarray([float(item.get("modularity_gap", 0.0)) for item in cluster_metrics], dtype=float)
                cluster_ax.plot(cluster_counts, silhouettes, color="#1d4ed8", marker="o", linewidth=1.8, label="Silhouette")
                cluster_ax.plot(cluster_counts, gaps, color="#dc2626", marker="s", linewidth=1.6, label="Within-between gap")
                cluster_ax.set_title("Cluster-count validation")
                cluster_ax.set_xlabel("Cluster count")
                cluster_ax.set_ylabel("Score")
                cluster_ax.legend(loc="best", fontsize=8)
            else:
                cluster_ax.text(0.5, 0.5, "No cluster metrics", ha="center", va="center")
                cluster_ax.set_xticks([])
                cluster_ax.set_yticks([])

            if channel_region_margin.size:
                inset = cluster_ax.inset_axes([0.58, 0.12, 0.36, 0.34])
                valid_margin = channel_region_margin[np.isfinite(channel_region_margin)]
                if valid_margin.size:
                    inset.hist(valid_margin, bins=min(16, max(6, valid_margin.size)), color="#475569", alpha=0.85)
                    inset.axvline(float(analysis.get("membership_threshold", 0.0)) * 0.35, color="#dc2626", linestyle="--", linewidth=1.0)
                    inset.set_title("Margin", fontsize=7)
                    inset.tick_params(axis="both", labelsize=6)

            edge_ax = axes[1, 1]
            if edges:
                selected_edge = max(
                    edges,
                    key=lambda item: (
                        float(item.get("peak_to_background", 0.0)),
                        float(item.get("distribution_shift", 0.0)),
                        float(item.get("peak_count", 0.0)),
                    ),
                )
                true_hist = np.asarray(selected_edge.get("true_hist", []), dtype=float)
                background_hist = np.asarray(selected_edge.get("background_hist", []), dtype=float)
                hist_edges = np.asarray(selected_edge.get("hist_edges", []), dtype=float)
                if true_hist.size and background_hist.size and hist_edges.size == true_hist.size + 1:
                    centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
                    width = np.diff(hist_edges)
                    edge_ax.bar(hist_edges[:-1], background_hist, width=width, align="edge", color="#94a3b8", alpha=0.55, label="Background")
                    edge_ax.bar(hist_edges[:-1], true_hist, width=width, align="edge", color="#2563eb", alpha=0.65, label="True")
                    median_delay = float(selected_edge.get("median_delay_ms", 0.0))
                    peak_delay = float(selected_edge.get("peak_delay_ms", 0.0))
                    peak_window_mean = float(selected_edge.get("peak_window_mean_ms", 0.0))
                    edge_ax.axvline(median_delay, color="#dc2626", linewidth=1.6, linestyle="--", label="Median")
                    edge_ax.axvline(peak_delay, color="#16a34a", linewidth=1.6, linestyle="-.", label="Peak")
                    edge_ax.axvline(peak_window_mean, color="#7c3aed", linewidth=1.6, linestyle=":", label="Peak-window mean")
                    edge_ax.set_title(
                        f"Delay distribution vs background | {int(selected_edge['source']) + 1}->{int(selected_edge['target']) + 1}"
                    )
                    edge_ax.set_xlabel("Delay (ms)")
                    edge_ax.set_ylabel("Count")
                    edge_ax.legend(loc="best", fontsize=8)
                    edge_ax.text(
                        0.98,
                        0.98,
                        "\n".join(
                            [
                                f"median: {median_delay:.2f} ms",
                                f"peak: {peak_delay:.2f} ms",
                                f"peak-win mean: {peak_window_mean:.2f} ms",
                                f"peak/bg: {float(selected_edge.get('peak_to_background', 0.0)):.2f}",
                                f"shift: {float(selected_edge.get('distribution_shift', 0.0)):.2f} ms",
                            ]
                        ),
                        transform=edge_ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=8,
                        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
                    )
                else:
                    edge_ax.text(0.5, 0.5, "No delay histogram", ha="center", va="center")
                    edge_ax.set_xticks([])
                    edge_ax.set_yticks([])
            else:
                edge_ax.text(0.5, 0.5, "No significant directed edges", ha="center", va="center")
                edge_ax.set_xticks([])
                edge_ax.set_yticks([])

        self._show_metric_figure("Spatial-temporal analysis", summary, _draw)

    def _trajectory_analysis(self) -> dict:
        latent = np.asarray((self.current or {}).get("latent_states", []), dtype=float)
        if latent.ndim != 3 or latent.shape[0] < 1 or latent.shape[1] < 1 or latent.shape[2] < 1:
            return {}
        cluster_count = max(1, min(4, latent.shape[0]))
        trajectory_relative = latent - latent[:, :1, :]
        if not np.any(np.abs(trajectory_relative) > 1e-12):
            trajectory_relative = latent.copy()
        trajectory_features = trajectory_relative.reshape((latent.shape[0], -1))
        if trajectory_features.shape[0] >= 2 and trajectory_features.shape[1] >= 1:
            try:
                scaled_features = StandardScaler().fit_transform(trajectory_features)
            except Exception:
                scaled_features = trajectory_features
        else:
            scaled_features = trajectory_features
        scaled_features = np.nan_to_num(np.asarray(scaled_features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

        sample_count = int(scaled_features.shape[0])
        if sample_count >= 2:
            distance = np.sqrt(
                np.maximum(
                    0.0,
                    np.sum(scaled_features ** 2, axis=1, keepdims=True)
                    + np.sum(scaled_features ** 2, axis=1, keepdims=True).T
                    - 2.0 * (scaled_features @ scaled_features.T),
                )
            )
            np.fill_diagonal(distance, 0.0)
            condensed = squareform(distance, checks=False)
            tree = linkage(condensed, method="average", optimal_ordering=sample_count <= 256)
            groups = fcluster(tree, t=cluster_count, criterion="maxclust").astype(int)
            order = leaves_list(tree).astype(int)
        else:
            distance = np.zeros((sample_count, sample_count), dtype=float)
            groups = np.ones(sample_count, dtype=int)
            order = np.arange(sample_count, dtype=int)

        if scaled_features.shape[0] >= 2 and scaled_features.shape[1] >= 1 and not np.allclose(scaled_features, scaled_features[0, 0]):
            full_components = min(scaled_features.shape[0], scaled_features.shape[1])
            trajectory_pca_model = SkPCA(n_components=full_components, random_state=7)
            trajectory_scores = trajectory_pca_model.fit_transform(scaled_features)
            components = min(2, trajectory_scores.shape[1], scaled_features.shape[0])
            projection = trajectory_scores[:, :components]
            if projection.shape[1] == 1:
                projection = np.column_stack([projection[:, 0], np.zeros(projection.shape[0])])
            trajectory_pca_eigenvalues = np.asarray(getattr(trajectory_pca_model, "explained_variance_", np.zeros(0)), dtype=float)
            trajectory_pca_explained = np.asarray(getattr(trajectory_pca_model, "explained_variance_ratio_", np.zeros(0)), dtype=float)
        else:
            base = scaled_features[:, 0] if scaled_features.ndim == 2 and scaled_features.shape[1] else np.zeros(latent.shape[0], dtype=float)
            projection = np.column_stack([base, np.zeros(latent.shape[0])])
            trajectory_pca_eigenvalues = np.zeros(0, dtype=float)
            trajectory_pca_explained = np.zeros(0, dtype=float)
        unique_groups = sorted(int(value) for value in np.unique(groups))
        mean_trajectories = {}
        for group in unique_groups:
            mask = groups == group
            group_latent = latent[mask]
            centroid = np.mean(group_latent, axis=0)
            mean_trajectories[group] = centroid
        return {
            "groups": groups,
            "projection": projection,
            "distance_matrix": distance,
            "trajectory_order": order,
            "trajectory_features": scaled_features,
            "mean_trajectories": mean_trajectories,
            "trajectory_pca_eigenvalues": trajectory_pca_eigenvalues,
            "trajectory_pca_explained": trajectory_pca_explained,
        }

    def _resample_normalized_time(self, values: np.ndarray, steps: int) -> np.ndarray:
        array = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        steps = max(2, int(steps))
        if array.ndim == 1:
            array = array.reshape((-1, 1))
        if array.ndim != 2 or array.shape[0] == 0:
            return np.zeros((steps, 0), dtype=float)
        if array.shape[0] == 1:
            return np.repeat(array, steps, axis=0)
        old_t = np.linspace(0.0, 1.0, array.shape[0], dtype=float)
        new_t = np.linspace(0.0, 1.0, steps, dtype=float)
        out = np.zeros((steps, array.shape[1]), dtype=float)
        for dim in range(array.shape[1]):
            out[:, dim] = np.interp(new_t, old_t, array[:, dim])
        return out

    def _normalized_time_analysis(self, resample_steps: int = 40, max_k: int = 4) -> dict:
        latent = np.asarray((self.current or {}).get("latent_states", []), dtype=float)
        raw = np.asarray((self.current or {}).get("raw_observed_states", []), dtype=float)
        intervals = list((self.current or {}).get("intervals", []))
        if latent.ndim != 3 or latent.shape[0] < 1 or latent.shape[1] < 2 or latent.shape[2] < 1:
            return {}
        steps = max(6, int(resample_steps))
        dims = min(3, latent.shape[2])
        trajectories = []
        population = []
        meta = []
        for index in range(latent.shape[0]):
            trajectory = np.asarray(latent[index, :, :dims], dtype=float)
            if trajectory.shape[0] < 2:
                continue
            trajectory = trajectory - trajectory[:1, :]
            trajectories.append(self._resample_normalized_time(trajectory, steps))
            if raw.ndim == 3 and index < raw.shape[0]:
                population_trace = np.mean(np.maximum(raw[index], 0.0), axis=1)
            else:
                population_trace = np.linalg.norm(trajectory, axis=1)
            population.append(self._resample_normalized_time(population_trace, steps).reshape(steps))
            if index < len(intervals):
                start_s, stop_s = intervals[index]
                meta.append(
                    {
                        "burst_index": int(index),
                        "start_s": float(start_s),
                        "stop_s": float(stop_s),
                        "duration_ms": max(0.0, float(stop_s) - float(start_s)) * 1000.0,
                    }
                )
            else:
                meta.append({"burst_index": int(index), "start_s": 0.0, "stop_s": 0.0, "duration_ms": 0.0})
        if not trajectories:
            return {}
        trajs = np.stack(trajectories, axis=0)
        pop_trajs = np.stack(population, axis=0) if population else np.zeros((trajs.shape[0], steps), dtype=float)
        if trajs.shape[2] < 3:
            trajs = np.pad(trajs, ((0, 0), (0, 0), (0, 3 - trajs.shape[2])), mode="constant")
        flat = trajs.reshape((trajs.shape[0], -1))
        if flat.shape[0] >= 2 and flat.shape[1] >= 1:
            try:
                scaled = StandardScaler().fit_transform(flat)
            except Exception:
                scaled = flat
        else:
            scaled = flat
        scaled = np.nan_to_num(np.asarray(scaled, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

        best_score = -np.inf
        best_labels = np.zeros(scaled.shape[0], dtype=int)
        best_k = 1
        score_by_k: dict[str, float] = {}
        distinct_count = np.unique(np.round(scaled, decimals=10), axis=0).shape[0] if scaled.ndim == 2 and scaled.size else 1
        max_k = max(1, min(int(max_k), scaled.shape[0], int(distinct_count)))
        for k in range(1, max_k + 1):
            if k == 1 or scaled.shape[0] < 3:
                labels = np.zeros(scaled.shape[0], dtype=int)
                score = 0.0
            else:
                try:
                    labels = SkKMeans(n_clusters=k, n_init=10, random_state=7).fit_predict(scaled).astype(int)
                    score = float(silhouette_score(scaled, labels)) if len(set(int(value) for value in labels)) > 1 else 0.0
                except Exception:
                    labels = np.zeros(scaled.shape[0], dtype=int)
                    score = 0.0
            score_by_k[f"k_{k}"] = float(score)
            if score > best_score:
                best_score = float(score)
                best_labels = labels
                best_k = int(k)

        unique_labels = sorted(int(value) for value in np.unique(best_labels))
        mean_trajectories = {}
        mean_population = {}
        cluster_rows = []
        global_center = np.mean(scaled, axis=0) if scaled.size else np.zeros(0, dtype=float)
        for label in unique_labels:
            mask = best_labels == label
            cluster_flat = scaled[mask]
            center = np.mean(cluster_flat, axis=0) if cluster_flat.size else np.zeros_like(global_center)
            within = float(np.mean(np.linalg.norm(cluster_flat - center, axis=1))) if cluster_flat.size else 0.0
            between = float(np.linalg.norm(center - global_center)) if center.size else 0.0
            cluster_pop = pop_trajs[mask]
            mean_trajectories[label] = np.mean(trajs[mask], axis=0)
            mean_population[label] = np.mean(cluster_pop, axis=0) if cluster_pop.size else np.zeros(steps, dtype=float)
            peaks = np.max(cluster_pop, axis=1) if cluster_pop.size else np.zeros(0, dtype=float)
            auc = np.mean(cluster_pop, axis=1) if cluster_pop.size else np.zeros(0, dtype=float)
            cluster_rows.append(
                {
                    "cluster": int(label),
                    "count": int(np.count_nonzero(mask)),
                    "fraction": float(np.count_nonzero(mask) / max(best_labels.size, 1)),
                    "within_distance": within,
                    "between_distance_to_global": between,
                    "mean_peak": float(np.mean(peaks)) if peaks.size else 0.0,
                    "mean_auc": float(np.mean(auc)) if auc.size else 0.0,
                    "within_curve_std": float(np.mean(np.std(cluster_pop, axis=0))) if cluster_pop.size else 0.0,
                }
            )

        if scaled.shape[0] >= 2 and scaled.shape[1] >= 1:
            try:
                pca_model = SkPCA(n_components=min(scaled.shape[0], scaled.shape[1]), random_state=7)
                projection = pca_model.fit_transform(scaled)
                if projection.shape[1] == 1:
                    projection = np.column_stack([projection[:, 0], np.zeros(projection.shape[0])])
                eigenvalues = np.asarray(getattr(pca_model, "explained_variance_", np.zeros(0)), dtype=float)
                explained = np.asarray(getattr(pca_model, "explained_variance_ratio_", np.zeros(0)), dtype=float)
            except Exception:
                projection = np.column_stack([scaled[:, 0], np.zeros(scaled.shape[0])]) if scaled.shape[1] else np.zeros((scaled.shape[0], 2), dtype=float)
                eigenvalues = np.zeros(0, dtype=float)
                explained = np.zeros(0, dtype=float)
        else:
            projection = np.zeros((scaled.shape[0], 2), dtype=float)
            eigenvalues = np.zeros(0, dtype=float)
            explained = np.zeros(0, dtype=float)

        return {
            "trajectories": trajs,
            "population_trajectories": pop_trajs,
            "labels": best_labels,
            "best_k": int(best_k),
            "silhouette_by_k": score_by_k,
            "cluster_stats": {"clusters": cluster_rows},
            "mean_trajectories": mean_trajectories,
            "mean_population": mean_population,
            "projection": projection[:, :2],
            "trajectory_pca_eigenvalues": eigenvalues,
            "trajectory_pca_explained": explained,
            "normalized_time": np.linspace(0.0, 1.0, steps, dtype=float),
            "burst_meta": meta,
            "resample_steps": int(steps),
        }

    def _show_normalized_time_analysis(self):
        if not self.current:
            _show_info_message(self, "Normalized time", "Run dynamics analysis first.")
            return
        result = self._normalized_time_analysis()
        if not result:
            _show_info_message(self, "Normalized time", "No latent burst trajectories are available.")
            return
        trajs = np.asarray(result.get("trajectories", []), dtype=float)
        pop_trajs = np.asarray(result.get("population_trajectories", []), dtype=float)
        labels = np.asarray(result.get("labels", []), dtype=int)
        unique_labels = sorted(int(value) for value in np.unique(labels))
        time = np.asarray(result.get("normalized_time", []), dtype=float)
        best_k = int(result.get("best_k", 1))
        silhouette_by_k = result.get("silhouette_by_k", {}) or {}
        best_silhouette = float(silhouette_by_k.get(f"k_{best_k}", 0.0))
        summary = " | ".join(
            [
                f"Bursts: {trajs.shape[0]}",
                f"Normalized steps: {int(result.get('resample_steps', 0))}",
                f"Best k: {best_k}",
                f"Silhouette: {best_silhouette:.4g}",
                "Basis: latent trajectories aligned to burst onset and resampled to common length",
            ]
        )

        def _draw(figure):
            figure.clear()
            axes = figure.subplots(2, 2)
            colors = [f"C{index % 10}" for index in range(max(1, len(unique_labels)))]

            ax = axes[0, 0]
            projection = np.asarray(result.get("projection", []), dtype=float)
            if projection.ndim == 2 and projection.shape[0]:
                for color_index, label in enumerate(unique_labels):
                    mask = labels == label
                    ax.scatter(projection[mask, 0], projection[mask, 1], s=30, alpha=0.86, color=colors[color_index], label=f"cluster {label + 1}")
                ax.set_title("Normalized-time burst classes")
                ax.set_xlabel("Trajectory PC1")
                ax.set_ylabel("Trajectory PC2")
                ax.legend(loc="best", fontsize=8)
            else:
                ax.text(0.5, 0.5, "No projection", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])

            ax = axes[0, 1]
            plotted = False
            for color_index, label in enumerate(unique_labels):
                mean_traj = np.asarray(result.get("mean_trajectories", {}).get(label, []), dtype=float)
                if mean_traj.ndim != 2 or mean_traj.shape[0] == 0:
                    continue
                plotted = True
                ax.plot(time, mean_traj[:, 0], color=colors[color_index], linewidth=2.0, label=f"cluster {label + 1} z1")
                if mean_traj.shape[1] > 1:
                    ax.plot(time, mean_traj[:, 1], color=colors[color_index], linewidth=1.5, linestyle="--", label=f"cluster {label + 1} z2")
            if plotted:
                ax.set_title("Latent dimensions over normalized burst time")
                ax.set_xlabel("Normalized burst time")
                ax.set_ylabel("Relative latent state")
                ax.legend(loc="best", fontsize=7)
            else:
                ax.text(0.5, 0.5, "No mean trajectories", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])

            ax = axes[1, 0]
            for color_index, label in enumerate(unique_labels):
                cluster = pop_trajs[labels == label]
                if cluster.ndim != 2 or cluster.shape[0] == 0:
                    continue
                mean = np.mean(cluster, axis=0)
                std = np.std(cluster, axis=0)
                ax.plot(time, mean, color=colors[color_index], linewidth=2.0, label=f"cluster {label + 1}")
                ax.fill_between(time, mean - std, mean + std, color=colors[color_index], alpha=0.15, linewidth=0)
            ax.set_title("Population activity over normalized burst time")
            ax.set_xlabel("Normalized burst time")
            ax.set_ylabel("Mean firing rate (Hz)")
            ax.legend(loc="best", fontsize=8)

            ax = axes[1, 1]
            eigenvalues = np.asarray(result.get("trajectory_pca_eigenvalues", []), dtype=float)
            explained = np.asarray(result.get("trajectory_pca_explained", []), dtype=float)
            score_by_k = result.get("silhouette_by_k", {}) or {}
            if eigenvalues.size:
                count = min(10, eigenvalues.size)
                x = np.arange(1, count + 1, dtype=int)
                ax.plot(x, eigenvalues[:count], marker="o", color="#1d4ed8", linewidth=1.6, label="Eigenvalue")
                ax.set_xlabel("Principal component / k")
                ax.set_ylabel("Eigenvalue", color="#1d4ed8")
                ax.tick_params(axis="y", labelcolor="#1d4ed8")
                ax2 = ax.twinx()
                if explained.size:
                    ax2.plot(x, np.cumsum(explained[:count]) * 100.0, marker="s", color="#16a34a", linewidth=1.3, label="Cumulative %")
                k_values = []
                scores = []
                for key, value in score_by_k.items():
                    try:
                        k_values.append(int(str(key).split("_")[-1]))
                        scores.append(float(value))
                    except Exception:
                        pass
                if k_values:
                    ax2.plot(k_values, np.asarray(scores) * 100.0, marker="^", color="#dc2626", linewidth=1.2, linestyle="--", label="Silhouette x100")
                ax2.set_ylabel("Explained / silhouette", color="#334155")
                ax2.tick_params(axis="y", labelcolor="#334155")
                ax.set_title("Trajectory PCA and k selection")
            else:
                ax.text(0.5, 0.5, "No PCA spectrum", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])

        self._show_metric_figure("Normalized burst time analysis", summary, _draw)

    def _show_trajectory_analysis(self):
        if not self.current:
            _show_info_message(self, "Trajectory analysis", "Run factor analysis first.")
            return
        result = self._trajectory_analysis()
        if not result:
            _show_info_message(self, "Trajectory analysis", "No latent trajectories are available.")
            return
        latent = np.asarray(self.current.get("latent_states", []), dtype=float)
        centers_ms = np.asarray(self.current.get("centers_ms", []), dtype=float)
        groups = np.asarray(result.get("groups", []), dtype=int)
        unique_groups = sorted(int(value) for value in np.unique(groups))
        summary = " | ".join([
            f"Samples: {latent.shape[0]}",
            f"Time bins: {latent.shape[1]}",
            f"Latent dims: {latent.shape[2]}",
            f"Trajectory-structure clusters: {len(unique_groups)}",
            "Clustering basis: trajectory distance matrix",
        ])

        def _draw(figure):
            figure.clear()
            axes = figure.subplots(2, 2)
            projection = np.asarray(result.get("projection", []), dtype=float)
            distance_matrix = np.asarray(result.get("distance_matrix", []), dtype=float)
            trajectory_order = np.asarray(result.get("trajectory_order", []), dtype=int)
            ax = axes[0, 0]
            burst_index = np.arange(projection.shape[0], dtype=float)
            scatter = ax.scatter(
                projection[:, 0],
                projection[:, 1],
                s=28,
                alpha=0.88,
                c=burst_index,
                cmap="viridis",
                edgecolors="none",
            )
            ax.set_title("Trajectory-structure clusters")
            ax.set_xlabel("Trajectory PC1")
            ax.set_ylabel("Trajectory PC2")
            if projection.shape[0]:
                figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Burst order")

            ax = axes[0, 1]
            plotted = False
            for group in unique_groups:
                mask = groups == group
                if not np.any(mask):
                    continue
                color = f"C{(group - 1) % 10}"
                group_latent = latent[mask]
                for trajectory in group_latent:
                    trajectory = np.asarray(trajectory, dtype=float)
                    if trajectory.ndim != 2 or trajectory.shape[0] == 0:
                        continue
                    plotted = True
                    if trajectory.shape[1] >= 2:
                        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=1.0, alpha=0.24)
                    else:
                        ax.plot(np.arange(trajectory.shape[0]), trajectory[:, 0], color=color, linewidth=1.0, alpha=0.24)
                mean_trajectory = np.asarray(result.get("mean_trajectories", {}).get(group, []), dtype=float)
                if mean_trajectory.ndim == 2 and mean_trajectory.shape[0] > 0:
                    if mean_trajectory.shape[1] >= 2:
                        ax.plot(mean_trajectory[:, 0], mean_trajectory[:, 1], color=color, linewidth=2.2, label=f"cluster {group}")
                        ax.scatter(mean_trajectory[0, 0], mean_trajectory[0, 1], color=color, s=24)
                    else:
                        ax.plot(np.arange(mean_trajectory.shape[0]), mean_trajectory[:, 0], color=color, linewidth=2.2, label=f"cluster {group}")
            if plotted:
                ax.set_title("All latent trajectories")
                ax.set_xlabel("z1" if latent.shape[2] >= 2 else "Time bin")
                ax.set_ylabel("z2" if latent.shape[2] >= 2 else "z1")
                ax.legend(loc="best", fontsize=8)
            else:
                ax.text(0.5, 0.5, "No latent trajectories", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])

            ax = axes[1, 0]
            if distance_matrix.size:
                if trajectory_order.size == distance_matrix.shape[0]:
                    ordered = distance_matrix[np.ix_(trajectory_order, trajectory_order)]
                else:
                    ordered = distance_matrix
                image = ax.imshow(ordered, aspect="auto", interpolation="nearest", cmap="magma")
                ax.set_title("Trajectory distance matrix")
                ax.set_xlabel("Trajectory (cluster-sorted)")
                ax.set_ylabel("Trajectory (cluster-sorted)")
                figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, "No distance matrix", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])

            ax = axes[1, 1]
            eigenvalues = np.asarray(result.get("trajectory_pca_eigenvalues", []), dtype=float)
            explained = np.asarray(result.get("trajectory_pca_explained", []), dtype=float)
            ax.set_title("Trajectory-PCA eigenvalues and explained variance")
            if eigenvalues.size:
                cumulative = np.cumsum(explained) if explained.size else np.zeros(0, dtype=float)
                cumulative_cutoff = int(np.searchsorted(cumulative, 0.95, side="left") + 1) if cumulative.size else 0
                relative_cutoff = int(np.flatnonzero(eigenvalues >= max(float(eigenvalues[0]) * 0.01, 1e-12))[-1] + 1) if eigenvalues.size else 0
                display_count = max(3, min(eigenvalues.size, max(cumulative_cutoff, relative_cutoff)))
                component_index = np.arange(1, display_count + 1, dtype=int)
                shown_eigenvalues = eigenvalues[:display_count]
                shown_explained = explained[:display_count] if explained.size else np.zeros(0, dtype=float)
                shown_cumulative = cumulative[:display_count] if cumulative.size else np.zeros(0, dtype=float)
                ax2 = ax.twinx()
                ax.plot(component_index, shown_eigenvalues, color="#1d4ed8", marker="o", linewidth=1.8, label="Eigenvalue")
                if explained.size:
                    ax2.plot(component_index, shown_explained * 100.0, color="#dc2626", marker="s", linewidth=1.4, label="Explained %")
                    ax2.plot(component_index, shown_cumulative * 100.0, color="#16a34a", marker="^", linewidth=1.4, linestyle="--", label="Cumulative %")
                    ax2.axhline(90.0, color="#94a3b8", linewidth=1.0, linestyle=":")
                    ax2.axhline(95.0, color="#64748b", linewidth=1.0, linestyle="--")
                    ax2.set_ylabel("Explained variance (%)")
                    ax2.set_ylim(0.0, 100.0)
                    handles_left, labels_left = ax.get_legend_handles_labels()
                    handles_right, labels_right = ax2.get_legend_handles_labels()
                    ax.legend(handles_left + handles_right, labels_left + labels_right, loc="best", fontsize=8)
                ax.set_xlim(1, max(3, display_count))
                ax.text(
                    0.98,
                    0.98,
                    f"shown: {display_count}/{eigenvalues.size}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
                )
                ax.set_xlabel("Principal component")
                ax.set_ylabel("Eigenvalue")
            else:
                ax.text(0.5, 0.5, "No PCA spectrum", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])

        self._show_metric_figure("Latent trajectory analysis", summary, _draw)

    def _update_summary(self):
        if not self.current:
            self.summary.setText("No burst trajectory result")
            return
        model_method = self._current_model_method()
        latent_states = self._display_latent_states()
        burst_count = int(latent_states.shape[0]) if latent_states.ndim == 3 else 0
        bin_count = int(latent_states.shape[1]) if latent_states.ndim == 3 else 0
        selected_count = int(latent_states.shape[2]) if latent_states.ndim == 3 else 0
        total_count = len(self.current.get("labels", []))
        groups = np.asarray(self.current.get("groups", []), dtype=int)
        group_count = len(set(int(value) for value in groups)) if groups.size else 0
        early_mean = float(self.current.get("early_mean_dispersion", 0.0))
        late_mean = float(self.current.get("late_mean_dispersion", 0.0))
        ratio = late_mean / max(early_mean, 1e-12)
        projection = str(self.current.get("state_projection", ""))
        latent_params = self.current.get("latent_params", {}) or {}
        loading_shape = np.asarray(latent_params.get("loadings", []), dtype=float).shape
        if len(loading_shape) == 2:
            projection = f"{projection} | W {loading_shape[0]}x{loading_shape[1]} | EM {int(latent_params.get('n_iter', 0))}"
        r2 = float(self.current.get("reconstruction_r2", 0.0))
        scope = "All windows" if str(self.current.get("analysis_scope", "burst")) == "all_windows" else "Bursts"
        model_label = "LDS" if model_method == "lds" else ("pi-VAE" if model_method == "pivae" else "Factor Analysis")
        self.summary.setText(
            f"{model_label} | {projection} | Scope: {scope} | Channels: {selected_count}/{total_count} | Samples: {burst_count} | Bins: {bin_count} | Start groups: {group_count} | "
            f"R2: {r2:.4g} | Early dispersion: {early_mean:.4g} | Late dispersion: {late_mean:.4g} | Late/Early: {ratio:.3g}"
        )


class MultiFileFactorAnalysisWindow(AppDialog):
    def __init__(self, payload: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dynamics Analysis")
        self.resize(1280, 820)
        self.payload = payload if isinstance(payload, dict) else {}
        self.records = list(self.payload.get("records", []))

        self.file_combo = QComboBox()
        for index, record in enumerate(self.records):
            label = f"{index + 1}. {record.get('condition') or record.get('file')}"
            self.file_combo.addItem(label, index)
        self.file_combo.currentIndexChanged.connect(self._draw)
        self.file_combo.setToolTip("Choose which fitted file to inspect in the latent-state and loading plots.")
        self.summary = QLabel("")
        self.summary.setObjectName("MutedText")
        self.summary.setWordWrap(True)
        self.similarity_canvas = FigureCanvas(Figure(figsize=(5.8, 4.0), tight_layout=True))
        self.weight_canvas = FigureCanvas(Figure(figsize=(5.8, 4.0), tight_layout=True))
        self.latent_canvas = FigureCanvas(Figure(figsize=(6.6, 4.0), tight_layout=True))
        self.performance_canvas = FigureCanvas(Figure(figsize=(6.6, 3.0), tight_layout=True))

        controls_frame = QFrame()
        controls_frame.setObjectName("Panel")
        controls = QHBoxLayout(controls_frame)
        controls.setContentsMargins(12, 10, 12, 10)
        controls.setSpacing(10)
        controls.addWidget(QLabel("Inspect file"))
        controls.addWidget(self.file_combo, 1)
        controls.addWidget(QLabel("W is aligned to file 1 for cross-file comparison."))
        controls.addStretch(1)

        left = QVBoxLayout()
        left.addWidget(self.latent_canvas, 3)
        left.addWidget(self.performance_canvas, 2)
        right = QVBoxLayout()
        right.addWidget(self.similarity_canvas, 1)
        right.addWidget(self.weight_canvas, 1)
        plots = QHBoxLayout()
        plots.addLayout(left, 1)
        plots.addLayout(right, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(controls_frame)
        layout.addWidget(self.summary)
        layout.addLayout(plots, 1)
        self._draw()
        _fix_spinbox_hit_targets(self)

    def _selected_record_index(self) -> int:
        value = self.file_combo.currentData()
        try:
            return min(max(0, int(value)), max(0, len(self.records) - 1))
        except Exception:
            return 0

    def _record_labels(self) -> list[str]:
        labels = []
        for index, record in enumerate(self.records):
            text = str(record.get("condition") or record.get("file") or f"file {index + 1}")
            labels.append(text if len(text) <= 28 else text[:25] + "...")
        return labels

    def _draw(self):
        self._update_summary()
        self._draw_similarity()
        self._draw_weight_comparison()
        self._draw_latent_overview()
        self._draw_performance()

    def _update_summary(self):
        if not self.records:
            self.summary.setText("No dynamics-analysis records.")
            return
        errors = list(self.payload.get("errors", []))
        r2_values = np.asarray([float(record.get("reconstruction_r2", 0.0)) for record in self.records], dtype=float)
        selected_counts = [int(record.get("selected_channel_count", 0)) for record in self.records]
        removed = sum(int(record.get("artifact_removed_spikes", 0)) for record in self.records)
        stim_total = sum(int(record.get("stim_count", 0)) for record in self.records)
        scope = "All windows" if str(self.payload.get("analysis_scope", "burst")) == "all_windows" else "Bursts"
        model_method = str(self.payload.get("model_method", "fa")).lower()
        if model_method == "lds":
            one_step = np.asarray(
                [float((record.get("analysis", {}) or {}).get("one_step_r2", 0.0)) for record in self.records],
                dtype=float,
            )
            latent_rmse = np.asarray(
                [float((record.get("analysis", {}) or {}).get("rollout_latent_rmse", 0.0)) for record in self.records],
                dtype=float,
            )
            self.summary.setText(
                f"Model: LDS | files: {len(self.records)} | scope: {scope} | skipped: {len(errors)} | "
                f"mean rollout R2: {float(np.mean(r2_values)):.4g} | mean one-step R2: {float(np.mean(one_step)):.4g} | "
                f"mean latent RMSE: {float(np.mean(latent_rmse)):.4g} | "
                f"selected channels: {min(selected_counts) if selected_counts else 0}-{max(selected_counts) if selected_counts else 0} | "
                f"stim events: {stim_total} | tail +/-{float(self.payload.get('artifact_ms', 0.0)):g} ms removed spikes: {removed}"
            )
            return
        if model_method == "pivae":
            self.summary.setText(
                f"Model: pi-VAE | files: {len(self.records)} | scope: {scope} | skipped: {len(errors)} | mean R2: {float(np.mean(r2_values)):.4g} | "
                f"selected channels: {min(selected_counts) if selected_counts else 0}-{max(selected_counts) if selected_counts else 0} | "
                f"stim events: {stim_total} | tail +/-{float(self.payload.get('artifact_ms', 0.0)):g} ms removed spikes: {removed} | "
                f"decoder loadings aligned to file 1"
            )
            return
        self.summary.setText(
            f"Model: FA | files: {len(self.records)} | scope: {scope} | skipped: {len(errors)} | mean R2: {float(np.mean(r2_values)):.4g} | "
            f"selected channels: {min(selected_counts) if selected_counts else 0}-{max(selected_counts) if selected_counts else 0} | "
            f"stim events: {stim_total} | tail +/-{float(self.payload.get('artifact_ms', 0.0)):g} ms removed spikes: {removed} | "
            f"W aligned to file 1 for Pearson correlation"
        )

    def _draw_similarity(self):
        figure = self.similarity_canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        similarity = np.asarray(self.payload.get("w_similarity", []), dtype=float)
        labels = self._record_labels()
        if similarity.ndim != 2 or similarity.size == 0:
            ax.text(0.5, 0.5, "No W similarity", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            finite = similarity[np.isfinite(similarity)]
            if similarity.shape[0] > 1:
                off_diag = similarity[~np.eye(similarity.shape[0], dtype=bool)]
                display_values = off_diag[np.isfinite(off_diag)]
            else:
                display_values = finite
            if display_values.size:
                vmin = float(np.nanpercentile(display_values, 2.0))
                vmax = float(np.nanpercentile(display_values, 98.0))
            elif finite.size:
                vmin = float(np.nanmin(finite))
                vmax = float(np.nanmax(finite))
            else:
                vmin, vmax = -1.0, 1.0
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                vmin, vmax = -1.0, 1.0
            if vmax - vmin < 0.05:
                center = 0.5 * (vmin + vmax)
                vmin = center - 0.025
                vmax = center + 0.025
            vmin = max(-1.0, vmin)
            vmax = min(1.0, vmax)
            if vmax <= vmin:
                vmax = min(1.0, vmin + 0.05)
                vmin = max(-1.0, vmax - 0.05)
            image = ax.imshow(similarity, aspect="auto", interpolation="nearest", cmap="turbo", vmin=vmin, vmax=vmax)
            ax.set_title("Aligned W correlation")
            indices = np.arange(len(labels))
            ax.set_xticks(indices)
            ax.set_yticks(indices)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(labels, fontsize=7)
            figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        self.similarity_canvas.draw_idle()

    def _draw_weight_comparison(self):
        figure = self.weight_canvas.figure
        figure.clear()
        axes = figure.subplots(1, 2)
        if not self.records:
            for ax in axes:
                ax.text(0.5, 0.5, "No W", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            self.weight_canvas.draw_idle()
            return
        selected = self._selected_record_index()
        reference = np.asarray(self.records[0].get("aligned_loadings", []), dtype=float)
        current = np.asarray(self.records[selected].get("aligned_loadings", []), dtype=float)
        vmax = max(
            float(np.nanpercentile(np.abs(reference), 98.0)) if reference.size else 0.0,
            float(np.nanpercentile(np.abs(current), 98.0)) if current.size else 0.0,
            1e-9,
        )
        for ax, matrix, title in zip(axes, [reference, current], ["Reference aligned W", f"Selected aligned W | {selected + 1}"]):
            if matrix.ndim != 2 or matrix.size == 0:
                ax.text(0.5, 0.5, "No W", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel("Channel")
            ax.set_ylabel("Latent dim")
            figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        self.weight_canvas.draw_idle()

    def _draw_latent_overview(self):
        figure = self.latent_canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        if not self.records:
            ax.text(0.5, 0.5, "No latent states", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            self.latent_canvas.draw_idle()
            return
        selected = self._selected_record_index()
        analysis = self.records[selected].get("analysis", {}) or {}
        model_method = str(self.payload.get("model_method", "fa")).lower()
        latent_key = "model_latent_states" if model_method == "lds" else "latent_states"
        latent = np.asarray(analysis.get(latent_key, analysis.get("latent_states", [])), dtype=float)
        if latent.ndim != 3 or latent.size == 0:
            ax.text(0.5, 0.5, "No latent states", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            mean_time = np.mean(latent, axis=0).T
            vmax = float(np.nanpercentile(np.abs(mean_time), 98.0)) if mean_time.size else 1.0
            vmax = max(vmax, 1e-9)
            window_ms = float(analysis.get("window_ms", self.payload.get("window_ms", 300.0)))
            image = ax.imshow(mean_time, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-vmax, vmax=vmax, extent=[0.0, window_ms, mean_time.shape[0], 0.0])
            title = "Mean modeled z(t) across bursts" if model_method == "lds" else ("Mean pi-VAE z(t) across bursts" if model_method == "pivae" else "Mean z(t) across bursts")
            ax.set_title(f"{title} | {self.records[selected].get('condition') or self.records[selected].get('file')}")
            ax.set_xlabel("Time from burst onset (ms)")
            ax.set_ylabel("Latent dim")
            figure.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        self.latent_canvas.draw_idle()

    def _draw_performance(self):
        figure = self.performance_canvas.figure
        figure.clear()
        axes = figure.subplots(1, 2)
        labels = self._record_labels()
        model_method = str(self.payload.get("model_method", "fa")).lower()
        r2 = np.asarray([float(record.get("reconstruction_r2", 0.0)) for record in self.records], dtype=float)
        bursts = np.asarray([int(record.get("burst_count", 0)) for record in self.records], dtype=float)
        selected_channels = np.asarray([int(record.get("selected_channel_count", 0)) for record in self.records], dtype=float)
        x = np.arange(len(self.records))
        if model_method == "lds":
            one_step_r2 = np.asarray(
                [float((record.get("analysis", {}) or {}).get("one_step_r2", 0.0)) for record in self.records],
                dtype=float,
            )
            one_step_latent_rmse = np.asarray(
                [float((record.get("analysis", {}) or {}).get("one_step_latent_rmse", 0.0)) for record in self.records],
                dtype=float,
            )
            rollout_latent_rmse = np.asarray(
                [float((record.get("analysis", {}) or {}).get("rollout_latent_rmse", 0.0)) for record in self.records],
                dtype=float,
            )
            axes[0].bar(x - 0.18, r2, width=0.36, color="#2563eb", alpha=0.82, label="Rollout")
            axes[0].bar(x + 0.18, one_step_r2, width=0.36, color="#14b8a6", alpha=0.82, label="One-step")
            axes[0].set_title("LDS reconstruction R2")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            axes[0].set_ylabel("R2")
            axes[0].legend(loc="best", fontsize=8)
            axes[1].bar(x - 0.18, rollout_latent_rmse, width=0.36, color="#dc2626", alpha=0.8, label="Rollout z RMSE")
            axes[1].bar(x + 0.18, one_step_latent_rmse, width=0.36, color="#f59e0b", alpha=0.8, label="One-step z RMSE")
            axes[1].set_title("Latent-state prediction error")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            axes[1].set_ylabel("RMSE")
            axes[1].legend(loc="best", fontsize=8)
        else:
            axes[0].bar(x, r2, color="#2563eb", alpha=0.82)
            axes[0].set_title("pi-VAE reconstruction R2" if model_method == "pivae" else "Reconstruction R2")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            axes[0].set_ylabel("R2")
            axes[1].bar(x - 0.18, bursts, width=0.36, color="#16a34a", alpha=0.78, label="Bursts")
            axes[1].bar(x + 0.18, selected_channels, width=0.36, color="#f59e0b", alpha=0.78, label="Selected ch")
            axes[1].set_title("Data size")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            axes[1].legend(loc="best", fontsize=8)
        self.performance_canvas.draw_idle()


class GenericAnalysisWindow(AppDialog):
    def __init__(self, payload: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Analysis")
        self.resize(1220, 820)
        self.payload = payload if isinstance(payload, dict) else {}
        self.records = list(self.payload.get("records", []))

        self.file_combo = QComboBox()
        for index, record in enumerate(self.records):
            label = f"{index + 1}. {record.get('condition') or record.get('file')}"
            self.file_combo.addItem(label, index)
        self.file_combo.currentIndexChanged.connect(self._draw)

        self.summary = QLabel("")
        self.summary.setObjectName("MutedText")
        self.summary.setWordWrap(True)
        self.canvas = FigureCanvas(Figure(figsize=(10.5, 7.0), tight_layout=True))
        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setMinimumHeight(140)

        controls_frame = QFrame()
        controls_frame.setObjectName("Panel")
        controls = QHBoxLayout(controls_frame)
        controls.setContentsMargins(12, 10, 12, 10)
        controls.setSpacing(10)
        controls.addWidget(QLabel("Inspect file"))
        controls.addWidget(self.file_combo, 1)
        controls.addWidget(QLabel("Custom data -> basic function -> processed dataset -> auto plot"))
        controls.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(controls_frame)
        layout.addWidget(self.summary)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.detail)
        self._draw()
        _fix_spinbox_hit_targets(self)

    def _selected_record_index(self) -> int:
        value = self.file_combo.currentData()
        try:
            return min(max(0, int(value)), max(0, len(self.records) - 1))
        except Exception:
            return 0

    def _draw(self):
        self._update_summary()
        self._draw_figure()
        self._draw_detail()

    def _update_summary(self):
        if not self.records:
            self.summary.setText("No custom-analysis records.")
            return
        params = dict(self.payload.get("parameters", {}) or {})
        errors = list(self.payload.get("errors", []))
        if params.get("analysis_kind") == "custom_basic":
            self.summary.setText(
                " | ".join(
                    [
                        f"Files: {len(self.records)}",
                        f"Skipped: {len(errors)}",
                        f"Function: {params.get('analysis_type', 'n/a')}",
                        f"Windows: {params.get('time_windows', 'n/a')}",
                        f"Channels: {params.get('channels') or 'all'}",
                    ]
                )
            )
            return
        self.summary.setText(
            " | ".join(
                [
                    f"Files: {len(self.records)}",
                    f"Skipped: {len(errors)}",
                    f"View: {params.get('view_mode', 'n/a')}",
                    f"Normalization: {params.get('normalization', 'n/a')}",
                    f"Similarity: {params.get('similarity', 'n/a')}",
                    f"Reduction: {params.get('reduction', 'n/a')}",
                    f"Clustering: {params.get('clustering', 'n/a')}",
                ]
            )
        )

    def _draw_figure(self):
        if dict(self.payload.get("parameters", {}) or {}).get("analysis_kind") == "custom_basic":
            self._draw_custom_figure()
            return
        figure = create_generic_analysis_figure({}, title="Generic analysis")
        selected = self._selected_record_index()
        if 0 <= selected < len(self.records):
            result = dict(self.records[selected].get("analysis", {}) or {})
            title = str(self.records[selected].get("condition") or self.records[selected].get("file") or "Generic analysis")
            figure = create_generic_analysis_figure(result, title=title)
        old_canvas = self.canvas
        new_canvas = FigureCanvas(figure)
        layout = self.layout()
        if layout is not None:
            old_item = layout.itemAt(2)
            if old_item is not None:
                widget = old_item.widget()
                if widget is not None:
                    layout.replaceWidget(widget, new_canvas)
                    widget.setParent(None)
        self.canvas = new_canvas
        old_canvas.close()
        self.canvas.draw_idle()

    def _draw_custom_figure(self):
        figure = Figure(figsize=(10.5, 7.0), tight_layout=True)
        ax = figure.add_subplot(111)
        selected = self._selected_record_index()
        record = self.records[selected] if 0 <= selected < len(self.records) else {}
        processed = dict(record.get("processed", {}) or {})
        matrix = np.asarray(processed.get("matrix", []), dtype=float)
        plot_payload = dict(processed.get("plot_payload", {}) or {})
        sample_labels = list(processed.get("sample_labels", []))
        feature_labels = list(processed.get("feature_labels", []))
        if matrix.ndim != 2 or matrix.size == 0:
            ax.text(0.5, 0.5, "No plottable custom data", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            x = np.asarray(plot_payload.get("x", list(range(1, matrix.shape[0] + 1))), dtype=float)
            if x.size != matrix.shape[0]:
                x = np.arange(1, matrix.shape[0] + 1, dtype=float)
            mode = str(plot_payload.get("plot_mode", "auto") or "auto")
            if mode == "auto":
                mode = "bar" if matrix.shape[0] <= 8 else "line"
            if mode == "bar":
                width = 0.8 / max(1, matrix.shape[1])
                offsets = (np.arange(matrix.shape[1]) - (matrix.shape[1] - 1) / 2.0) * width
                for col in range(matrix.shape[1]):
                    label = feature_labels[col] if col < len(feature_labels) else f"series {col + 1}"
                    ax.bar(x + offsets[col], matrix[:, col], width=width, alpha=0.82, label=label)
            else:
                for col in range(matrix.shape[1]):
                    label = feature_labels[col] if col < len(feature_labels) else f"series {col + 1}"
                    ax.plot(x, matrix[:, col], marker="o", linewidth=1.8, label=label)
            if sample_labels and len(sample_labels) == matrix.shape[0]:
                ax.set_xticks(x)
                ax.set_xticklabels(sample_labels, rotation=30, ha="right")
            ax.set_xlabel(str(plot_payload.get("x_label", "X")))
            ax.set_ylabel(str(plot_payload.get("y_label", "Value")))
            ax.set_title(str(processed.get("name") or record.get("file") or "Custom analysis"))
            if matrix.shape[1] <= 12:
                ax.legend(loc="best", fontsize=8)
        old_canvas = self.canvas
        new_canvas = FigureCanvas(figure)
        layout = self.layout()
        if layout is not None:
            old_item = layout.itemAt(2)
            if old_item is not None:
                widget = old_item.widget()
                if widget is not None:
                    layout.replaceWidget(widget, new_canvas)
                    widget.setParent(None)
        self.canvas = new_canvas
        old_canvas.close()
        self.canvas.draw_idle()

    def _draw_detail(self):
        if not self.records:
            self.detail.setPlainText("No details available.")
            return
        selected = self._selected_record_index()
        record = self.records[selected]
        if dict(self.payload.get("parameters", {}) or {}).get("analysis_kind") == "custom_basic":
            processed = dict(record.get("processed", {}) or {})
            matrix = np.asarray(processed.get("matrix", []), dtype=float)
            self.detail.setPlainText(
                "\n".join(
                    [
                        f"File: {record.get('file')}",
                        f"Dataset: {processed.get('name', 'n/a')}",
                        f"Description: {processed.get('description', 'n/a')}",
                        f"Shape: {matrix.shape[0] if matrix.ndim >= 1 else 0} x {matrix.shape[1] if matrix.ndim == 2 else 0}",
                        f"Samples: {', '.join(list(processed.get('sample_labels', []))[:12])}",
                        f"Features: {', '.join(list(processed.get('feature_labels', []))[:12])}",
                    ]
                )
            )
            return
        analysis = dict(record.get("analysis", {}) or {})
        params = dict(analysis.get("parameters", {}) or {})
        prepared = analysis.get("prepared")
        groups = np.asarray(analysis.get("groups", []), dtype=int)
        counts_text = ", ".join(
            f"group {int(label)}: {int(count)}"
            for label, count in zip(*np.unique(groups, return_counts=True))
        ) if groups.size else "no groups"
        sample_count = int(getattr(prepared, "sample_count", 0)) if prepared is not None else 0
        feature_count = int(getattr(prepared, "feature_count", 0)) if prepared is not None else 0
        self.detail.setPlainText(
            "\n".join(
                [
                    f"File: {record.get('file')}",
                    f"Condition: {record.get('condition')}",
                    f"Matrix source: {record.get('matrix_description', 'n/a')}",
                    f"Samples x features: {sample_count} x {feature_count}",
                    f"View mode: {params.get('view_mode', 'n/a')}",
                    f"Normalization: {params.get('normalization', 'n/a')}",
                    f"Similarity: {params.get('similarity', 'n/a')}",
                    f"Reduction: {params.get('reduction', 'n/a')} ({params.get('reduction_dims', 'n/a')} dims)",
                    f"Clustering: {params.get('clustering', 'n/a')} / target groups {params.get('cluster_count', 'n/a')}",
                    f"Detected groups: {counts_text}",
                ]
            )
        )


class BurstClusteringWindow(AppDialog):
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
        self.bin_ms.valueChanged.connect(lambda *_: self._update_cluster_settings_summary())

        self.window_ms = QSpinBox()
        self.window_ms.setRange(0, 60000)
        self.window_ms.setSingleStep(10)
        self.window_ms.setValue(0)
        self.window_ms.setSuffix(" ms")
        self.window_ms.valueChanged.connect(self._draw)
        self.window_ms.valueChanged.connect(lambda *_: self._update_cluster_settings_summary())

        self.cluster_count = QSpinBox()
        self.cluster_count.setRange(1, 20)
        self.cluster_count.setValue(3)
        self.cluster_count.valueChanged.connect(self._draw)
        self.cluster_count.valueChanged.connect(lambda *_: self._update_cluster_settings_summary())

        self.reducer = QComboBox()
        self.reducer.addItem("PCA", "pca")
        self.reducer.addItem("t-SNE", "tsne")
        self.reducer.currentIndexChanged.connect(self._draw)
        self.reducer.currentIndexChanged.connect(lambda *_: self._update_cluster_settings_summary())

        self.normalize = QComboBox()
        self.normalize.addItem("Per burst", "per_burst")
        self.normalize.addItem("Time-bin z-score", "unit_zscore")
        self.normalize.addItem("None", "none")
        self.normalize.currentIndexChanged.connect(self._draw)
        self.normalize.currentIndexChanged.connect(lambda *_: self._update_cluster_settings_summary())

        self.trace_scale = QComboBox()
        self.trace_scale.addItem("Shape (per burst peak)", "per_trace_peak")
        self.trace_scale.addItem("Log count", "log")
        self.trace_scale.addItem("Robust count", "robust")
        self.trace_scale.addItem("Raw count", "raw")
        self.trace_scale.currentIndexChanged.connect(self._draw)
        self.trace_scale.currentIndexChanged.connect(lambda *_: self._update_cluster_settings_summary())
        self.trace_scale.setToolTip("Controls how burst trajectories are scaled before plotting.")

        self.analysis_settings_button = QPushButton("Analysis settings...")
        self.analysis_settings_button.clicked.connect(self._open_cluster_settings_dialog)
        self.analysis_settings_button.setToolTip("Choose the vectorization window and trace display scaling.")
        self.analysis_settings_summary = QLabel()
        self.analysis_settings_summary.setObjectName("MutedText")
        self.analysis_settings_summary.setWordWrap(True)

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
        controls.addWidget(self.analysis_settings_button)
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
        layout.addWidget(self.analysis_settings_summary)
        layout.addLayout(manual_controls)
        layout.addWidget(self.summary)
        layout.addLayout(plots, 1)
        self._update_cluster_settings_summary()
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

    def _update_cluster_settings_summary(self) -> None:
        window_text = "auto burst duration" if self.window_ms.value() <= 0 else f"{self.window_ms.value()} ms"
        normalize_label = self.normalize.currentText()
        trace_label = self.trace_scale.currentText()
        self.analysis_settings_summary.setText(
            "Feature build: "
            f"{self.bin_ms.value():g} ms bins within {window_text}. "
            "Preprocessing: "
            f"{normalize_label}. "
            "Trace display: "
            f"{trace_label}. "
            "Use PCA for stable overview; switch to t-SNE when you want local neighborhood structure."
        )

    def _open_cluster_settings_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Burst clustering settings")
        layout = QVBoxLayout(dialog)
        intro = QLabel(
            "These settings define how each burst is converted into a spike-count vector before clustering, "
            "and how the mean trajectories are displayed."
        )
        intro.setObjectName("MutedText")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        bin_field = QDoubleSpinBox()
        bin_field.setRange(self.bin_ms.minimum(), self.bin_ms.maximum())
        bin_field.setDecimals(self.bin_ms.decimals())
        bin_field.setSingleStep(self.bin_ms.singleStep())
        bin_field.setValue(self.bin_ms.value())
        bin_field.setSuffix(" ms")
        bin_field.setToolTip("Time bin used to convert each burst into a spike-count vector. Typical range: 2-20 ms.")
        form.addRow("Vector bin", bin_field)

        window_field = QSpinBox()
        window_field.setRange(self.window_ms.minimum(), self.window_ms.maximum())
        window_field.setSingleStep(self.window_ms.singleStep())
        window_field.setValue(self.window_ms.value())
        window_field.setSuffix(" ms")
        window_field.setToolTip("Fixed burst window. Set to 0 to use each burst's native duration.")
        form.addRow("Analysis window", window_field)

        trace_scale_field = QComboBox()
        for index in range(self.trace_scale.count()):
            trace_scale_field.addItem(self.trace_scale.itemText(index), self.trace_scale.itemData(index))
        trace_scale_field.setCurrentIndex(max(0, trace_scale_field.findData(self.trace_scale.currentData())))
        trace_scale_field.setToolTip("How to scale the trajectory panel: shape-emphasis, robust counts, log counts, or raw counts.")
        form.addRow("Trace scale", trace_scale_field)

        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        ok_button = QPushButton("Apply")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        buttons.addWidget(cancel_button)
        buttons.addWidget(ok_button)
        layout.addLayout(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.bin_ms.setValue(float(bin_field.value()))
        self.window_ms.setValue(int(window_field.value()))
        self.trace_scale.setCurrentIndex(max(0, self.trace_scale.findData(trace_scale_field.currentData())))
        self._update_cluster_settings_summary()

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
            _show_info_message(self, "Burst Clustering", "Need an embedding before manual cluster assignment.")
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


class BurstCorrelationWindow(AppDialog):
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


class HeatmapGifExportDialog(AppDialog):
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


class SpikeRasterWindow(AppDialog):
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
        channel_groups: dict[str, list[str]] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1280, 820)
        self.raw_spike_series = [(label, np.asarray(times, dtype=float)) for label, times in spike_series]
        self.raw_waveform_series = {label: np.asarray(values) for label, values in (waveform_series or {}).items()}
        self._filtered_spike_series_all = list(self.raw_spike_series)
        self._filtered_waveform_series_all = dict(self.raw_waveform_series)
        self.spike_series = list(self.raw_spike_series)
        self.spike_lookup = {label: times for label, times in self.spike_series}
        self._count_series = []
        self._count_series_by_channel = {}
        self._count_series_version = 0
        self._set_count_series_from_spike_series()
        self.waveform_series = dict(self.raw_waveform_series)
        self.sampling_rate = sampling_rate
        self.channel_map = channel_map
        self.analysis_windows = []
        self.selected_channel = _prefer_waveform_channel(self.spike_series, self.waveform_series)
        self.channel_groups = {}
        if isinstance(channel_groups, dict):
            available_labels = {str(label) for label, _times in self.raw_spike_series}
            for name, labels in channel_groups.items():
                selected = [str(label) for label in labels if str(label) in available_labels]
                if selected:
                    self.channel_groups[str(name)] = selected
        self._last_waveform_view = None
        self._last_waveform_refresh = 0.0
        self._waveform_min_interval_s = 0.12
        self._waveform_window_limit = 2400
        self._playback_time_ms = None
        self._internal_slider_update = False
        self._channel_count_cache = {}
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
        self.raster_action_combo.setToolTip("Choose a downstream analysis or export action for the current raster.")

        self.well_combo = None
        if self.channel_groups:
            self.well_combo = QComboBox()
            self.well_combo.addItem("All wells", "")
            for well in sorted(self.channel_groups, key=_well_sort_key):
                self.well_combo.addItem(str(well), str(well))
            self.well_combo.currentIndexChanged.connect(self._well_selection_changed)
            self.well_combo.setMinimumWidth(116)
            self.well_combo.setToolTip("Filter the raster rows to one Axion well or show all wells together.")

        self.window_grids = QSpinBox()
        self.window_grids.setRange(1, 500)
        self.window_grids.setSingleStep(1)
        self.window_grids.setValue(default_window_grids)
        self.window_grids.setFixedWidth(58)
        self.window_grids.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.window_grids.setCursor(Qt.CursorShape.ArrowCursor)
        self.window_grids.valueChanged.connect(self._update_slider_range)
        self.window_grids.valueChanged.connect(self._update_raster_settings_summary)
        self.window_grids.setToolTip("Number of grid bins shown in the current raster window. Typical range: 10-120.")

        self.grid_ms = QSpinBox()
        self.grid_ms.setRange(1, 60000)
        self.grid_ms.setSingleStep(10)
        self.grid_ms.setValue(default_grid_ms)
        self.grid_ms.setFixedWidth(66)
        self.grid_ms.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.grid_ms.setCursor(Qt.CursorShape.ArrowCursor)
        self.grid_ms.valueChanged.connect(self._update_slider_range)
        self.grid_ms.valueChanged.connect(self._update_raster_settings_summary)
        self.grid_ms.setToolTip("Milliseconds represented by each grid bin. Smaller values show finer timing detail.")

        self.heatmap_ms = QSpinBox()
        self.heatmap_ms.setRange(10, 5000)
        self.heatmap_ms.setSingleStep(10)
        self.heatmap_ms.setValue(100)
        self.heatmap_ms.setFixedWidth(70)
        self.heatmap_ms.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.heatmap_ms.setCursor(Qt.CursorShape.ArrowCursor)
        self.heatmap_ms.valueChanged.connect(self._heatmap_bin_changed)
        self.heatmap_ms.setToolTip("Time window used to integrate the heatmap around the current playhead. Typical range: 50-300 ms.")

        self.burst_bin_ms = QSpinBox()
        self.burst_bin_ms.setRange(1, 500)
        self.burst_bin_ms.setSingleStep(1)
        self.burst_bin_ms.setValue(10)
        self.burst_bin_ms.setFixedWidth(58)
        self.burst_bin_ms.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.burst_bin_ms.setCursor(Qt.CursorShape.ArrowCursor)
        self.burst_bin_ms.valueChanged.connect(self._refresh_bursts)
        self.burst_bin_ms.setToolTip("Population spike-count bin used for burst detection. Typical range: 5-20 ms.")

        self.burst_threshold_z = QDoubleSpinBox()
        self.burst_threshold_z.setRange(0.5, 20.0)
        self.burst_threshold_z.setSingleStep(0.5)
        self.burst_threshold_z.setDecimals(1)
        self.burst_threshold_z.setValue(4.0)
        self.burst_threshold_z.setFixedWidth(58)
        self.burst_threshold_z.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.burst_threshold_z.setCursor(Qt.CursorShape.ArrowCursor)
        self.burst_threshold_z.valueChanged.connect(self._refresh_bursts)
        self.burst_threshold_z.setToolTip("Z-score threshold for calling a burst from the population rate. Higher values are more selective.")

        self.burst_min_spikes = QSpinBox()
        self.burst_min_spikes.setRange(2, 1000)
        self.burst_min_spikes.setSingleStep(1)
        self.burst_min_spikes.setValue(5)
        self.burst_min_spikes.setFixedWidth(58)
        self.burst_min_spikes.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.burst_min_spikes.setCursor(Qt.CursorShape.ArrowCursor)
        self.burst_min_spikes.valueChanged.connect(self._refresh_bursts)
        self.burst_min_spikes.setToolTip("Minimum total spikes required inside a detected burst candidate.")

        self.hide_stim_tail = QCheckBox("Hide stim tail")
        self.hide_stim_tail.setEnabled(self.stim_times.size > 0)
        self.hide_stim_tail.stateChanged.connect(self._apply_stim_tail_filter)
        self.hide_stim_tail.stateChanged.connect(lambda *_: self._update_raster_settings_summary())
        self.hide_stim_tail.setToolTip("Remove spikes within a short artifact window after each stimulation event.")
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
        self.stim_tail_ms.valueChanged.connect(lambda *_: self._update_raster_settings_summary())
        self.stim_tail_ms.setToolTip("Artifact tail removed after each stimulation time. Typical range: 0.5-3 ms.")

        self.visible_rows = QSpinBox()
        self.visible_rows.setRange(1, max(1, len(self.spike_series)))
        self.visible_rows.setSingleStep(1)
        self.visible_rows.setValue(default_visible_rows)
        self.visible_rows.setFixedWidth(58)
        self.visible_rows.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.visible_rows.setCursor(Qt.CursorShape.ArrowCursor)
        self.visible_rows.valueChanged.connect(self._update_row_scroll_range)
        self.visible_rows.valueChanged.connect(self._update_raster_settings_summary)
        self.visible_rows.setToolTip("How many channels are shown at once in the raster viewport.")

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
        self.display_settings_button = QPushButton("Display settings...")
        self.display_settings_button.clicked.connect(self._open_raster_view_settings_dialog)
        self.display_settings_button.setToolTip("Heatmap, artifact-tail and other raster display options.")
        self.burst_settings_button = QPushButton("Burst settings...")
        self.burst_settings_button.clicked.connect(self._open_raster_burst_settings_dialog)
        self.burst_settings_button.setToolTip("Burst detection parameters used by burst-related analyses.")
        self.raster_settings_summary = QLabel()
        self.raster_settings_summary.setObjectName("MutedText")
        self.raster_settings_summary.setWordWrap(True)

        playback_controls = QHBoxLayout()
        playback_controls.setSpacing(8)
        playback_controls.addWidget(self.play_button)
        playback_controls.addWidget(QLabel("Time"))
        self.slider.setMinimumWidth(420)
        playback_controls.addWidget(self.slider, 1)
        playback_controls.addWidget(self.time_label)

        parameter_frame = QFrame()
        parameter_layout = QGridLayout(parameter_frame)
        parameter_layout.setContentsMargins(0, 0, 0, 0)
        parameter_layout.setHorizontalSpacing(10)
        parameter_layout.setVerticalSpacing(6)
        parameter_layout.addWidget(QLabel("Window"), 0, 0)
        parameter_layout.addWidget(
            self._number_stepper(self.window_minus_button, self.window_grids, self.window_plus_button),
            0,
            1,
        )
        parameter_layout.addWidget(QLabel("Grid"), 0, 2)
        parameter_layout.addWidget(self._number_stepper(self.grid_minus_button, self.grid_ms, self.grid_plus_button), 0, 3)
        parameter_layout.addWidget(QLabel("Rows"), 0, 4)
        parameter_layout.addWidget(
            self._number_stepper(
                self.visible_rows_minus_button,
                self.visible_rows,
                self.visible_rows_plus_button,
            ),
            0,
            5,
        )
        next_column = 6
        if self.well_combo is not None:
            parameter_layout.addWidget(QLabel("Well"), 0, next_column)
            parameter_layout.addWidget(self.well_combo, 0, next_column + 1)
            next_column += 2
        parameter_layout.addWidget(QLabel("Action"), 0, next_column)
        parameter_layout.addWidget(self.raster_action_combo, 0, next_column + 1)
        parameter_layout.addWidget(self.display_settings_button, 0, next_column + 2)
        parameter_layout.addWidget(self.burst_settings_button, 0, next_column + 3)
        parameter_layout.addWidget(self.row_label, 1, 0, 1, 2)
        parameter_layout.addWidget(self.raster_settings_summary, 1, 2, 1, max(8, next_column + 2))
        parameter_layout.setColumnStretch(next_column + 1, 1)

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
        layout.addWidget(parameter_frame)

        self._refresh_bursts()
        self._refresh_heatmap_scale()
        self._update_row_scroll_range()
        self._update_slider_range()
        self._update_raster_settings_summary()
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

    def _update_raster_settings_summary(self) -> None:
        window_ms = self._window_ms()
        stim_text = "off"
        if self.stim_times.size:
            stim_text = "on" if self.hide_stim_tail.isChecked() else "off"
            stim_text += f" ({self.stim_tail_ms.value():g} ms)"
        group_text = ""
        if self.well_combo is not None:
            current_well = str(self.well_combo.currentData() or "")
            group_text = f" Rows from: {current_well or 'all wells'}."
        self.raster_settings_summary.setText(
            "Core view: "
            f"{window_ms:g} ms window, {self.grid_ms.value()} ms/grid, {self.visible_rows.value()} visible rows. "
            "Display: "
            f"heatmap {self.heatmap_ms.value()} ms, stim-tail filter {stim_text}.{group_text} "
            "Burst detection: "
            f"{self.burst_bin_ms.value()} ms bin, z >= {self.burst_threshold_z.value():.1f}, "
            f"min spikes {self.burst_min_spikes.value()}."
        )

    def _display_labels_for_current_group(self) -> set[str] | None:
        if self.well_combo is None:
            return None
        selected = str(self.well_combo.currentData() or "")
        if not selected:
            return None
        labels = self.channel_groups.get(selected, [])
        return {str(label) for label in labels}

    def _apply_display_subset(self) -> None:
        allowed_labels = self._display_labels_for_current_group()
        previous_selected = self.selected_channel
        if allowed_labels is None:
            self.spike_series = list(self._filtered_spike_series_all)
            self.waveform_series = dict(self._filtered_waveform_series_all)
        else:
            self.spike_series = [
                (label, times)
                for label, times in self._filtered_spike_series_all
                if str(label) in allowed_labels
            ]
            self.waveform_series = {
                str(label): np.asarray(values)
                for label, values in self._filtered_waveform_series_all.items()
                if str(label) in allowed_labels
            }
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
        self._set_count_series_from_spike_series()
        self._last_waveform_view = None
        self._refresh_bursts()
        self._refresh_heatmap_scale()
        self._update_row_scroll_range()
        self._update_slider_range()
        self._update_view(force_heatmap=True)
        self._refresh_waveforms_for_window(force=True)
        self._update_raster_settings_summary()

    def _well_selection_changed(self) -> None:
        self._apply_display_subset()

    def _open_raster_view_settings_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Raster display settings")
        layout = QVBoxLayout(dialog)
        intro = QLabel(
            "Adjust the display-only parameters here. These settings control how the raster and heatmap are shown, "
            "without changing the downstream burst-detection thresholds."
        )
        intro.setWordWrap(True)
        intro.setObjectName("MutedText")
        layout.addWidget(intro)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        heatmap_field = QSpinBox()
        heatmap_field.setRange(self.heatmap_ms.minimum(), self.heatmap_ms.maximum())
        heatmap_field.setSingleStep(self.heatmap_ms.singleStep())
        heatmap_field.setValue(self.heatmap_ms.value())
        heatmap_field.setSuffix(" ms")
        heatmap_field.setToolTip("Integration window used to compute each heatmap frame.")
        form.addRow("Heatmap window", heatmap_field)

        hide_tail_field = QCheckBox("Hide stimulation artifact tail")
        hide_tail_field.setChecked(self.hide_stim_tail.isChecked())
        hide_tail_field.setEnabled(self.stim_times.size > 0)
        hide_tail_field.setToolTip("Remove spikes shortly after each stimulus marker.")
        form.addRow("Stim tail filter", hide_tail_field)

        stim_tail_field = QDoubleSpinBox()
        stim_tail_field.setRange(self.stim_tail_ms.minimum(), self.stim_tail_ms.maximum())
        stim_tail_field.setDecimals(self.stim_tail_ms.decimals())
        stim_tail_field.setSingleStep(self.stim_tail_ms.singleStep())
        stim_tail_field.setValue(self.stim_tail_ms.value())
        stim_tail_field.setSuffix(" ms")
        stim_tail_field.setEnabled(self.stim_times.size > 0)
        stim_tail_field.setToolTip("Artifact window removed after each stimulation event. Typical range: 0.5-3 ms.")
        form.addRow("Artifact window", stim_tail_field)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        ok_button = QPushButton("Apply")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        buttons.addWidget(cancel_button)
        buttons.addWidget(ok_button)
        layout.addLayout(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.heatmap_ms.setValue(int(heatmap_field.value()))
        self.hide_stim_tail.setChecked(hide_tail_field.isChecked())
        self.stim_tail_ms.setValue(float(stim_tail_field.value()))
        self._update_raster_settings_summary()

    def _open_raster_burst_settings_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Burst detection settings")
        layout = QVBoxLayout(dialog)
        intro = QLabel(
            "These parameters define how the population rate is converted into burst intervals. "
            "They affect burst overlays here and any burst-based analyses opened from this raster window."
        )
        intro.setWordWrap(True)
        intro.setObjectName("MutedText")
        layout.addWidget(intro)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        burst_bin_field = QSpinBox()
        burst_bin_field.setRange(self.burst_bin_ms.minimum(), self.burst_bin_ms.maximum())
        burst_bin_field.setSingleStep(self.burst_bin_ms.singleStep())
        burst_bin_field.setValue(self.burst_bin_ms.value())
        burst_bin_field.setSuffix(" ms")
        burst_bin_field.setToolTip("Population spike-count bin. Smaller bins preserve onset detail; larger bins smooth the rate.")
        form.addRow("Rate bin", burst_bin_field)

        threshold_field = QDoubleSpinBox()
        threshold_field.setRange(self.burst_threshold_z.minimum(), self.burst_threshold_z.maximum())
        threshold_field.setDecimals(self.burst_threshold_z.decimals())
        threshold_field.setSingleStep(self.burst_threshold_z.singleStep())
        threshold_field.setValue(self.burst_threshold_z.value())
        threshold_field.setToolTip("Z-score threshold above baseline population activity. Typical range: 3-6.")
        form.addRow("Burst z-threshold", threshold_field)

        min_spikes_field = QSpinBox()
        min_spikes_field.setRange(self.burst_min_spikes.minimum(), self.burst_min_spikes.maximum())
        min_spikes_field.setSingleStep(self.burst_min_spikes.singleStep())
        min_spikes_field.setValue(self.burst_min_spikes.value())
        min_spikes_field.setToolTip("Reject burst candidates with too few total spikes.")
        form.addRow("Minimum spikes", min_spikes_field)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        ok_button = QPushButton("Apply")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        buttons.addWidget(cancel_button)
        buttons.addWidget(ok_button)
        layout.addLayout(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.burst_bin_ms.setValue(int(burst_bin_field.value()))
        self.burst_threshold_z.setValue(float(threshold_field.value()))
        self.burst_min_spikes.setValue(int(min_spikes_field.value()))
        self._update_raster_settings_summary()

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
        if self.hide_stim_tail.isChecked() and self.stim_times.size:
            filtered, masks, removed = _filter_spike_series_stim_tail(
                self.raw_spike_series,
                self.stim_times,
                float(self.stim_tail_ms.value()),
            )
            self._filtered_spike_series_all = filtered
            filtered_waveforms = {}
            for label, waveforms in self.raw_waveform_series.items():
                mask = masks.get(str(label))
                values = np.asarray(waveforms)
                if mask is not None and values.ndim >= 2 and values.shape[0] == mask.size:
                    filtered_waveforms[label] = values[mask]
                else:
                    filtered_waveforms[label] = values
            self._filtered_waveform_series_all = filtered_waveforms
            suffix = f"stim tail hidden: {removed} spikes"
        else:
            self._filtered_spike_series_all = list(self.raw_spike_series)
            self._filtered_waveform_series_all = dict(self.raw_waveform_series)
            suffix = "stim tail shown"
        self._apply_display_subset()
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
            for base_channel, series_list in self._count_series_by_channel.items():
                for values in series_list:
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
            _show_info_message(self, "Export Heatmap GIF", "Set a channel map before exporting a heatmap GIF.")
            return False

        dialog = HeatmapGifExportDialog(self.min_time, self.max_time, self.heatmap_ms.value(), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        start_s, end_s, frame_step_ms = dialog.values()
        frame_times = self._heatmap_gif_frame_times(start_s, end_s, frame_step_ms)
        if frame_times.size == 0:
            _show_info_message(self, "Export Heatmap GIF", "No frames are available for the selected time range.")
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
            _show_error_message(self, "Export Heatmap GIF failed", "Pillow is required to export animated GIF files.")
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
                _show_info_message(self, "Export Heatmap GIF", "No frames were rendered for the selected time range.")
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
            _show_error_message(self, "Export Heatmap GIF failed", str(exc))
            return False
        finally:
            self._finish_progress(progress)

        _show_info_message(self, "Export Heatmap GIF", f"Saved heatmap GIF:\n{path}")
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

    def _open_burst_trajectory_window(self):
        progress = self._start_progress("Burst trajectory", "Preparing burst trajectory analysis...", 0)
        try:
            window = BurstTrajectoryWindow(self.spike_series, self.burst_intervals, self, self.channel_map)
        finally:
            self._finish_progress(progress)
        return self._show_analysis_window(window)

    def _save_bursts(self):
        if not self.burst_intervals:
            _show_info_message(self, "Save Bursts", "No detected bursts are available to save.")
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
            _show_error_message(self, "Save failed", str(exc))
            return False
        finally:
            self._finish_progress(progress)
        _show_info_message(self, "Save Bursts", f"Saved {payload['burst_count']} bursts: {path}")
        return True

    def _show_analysis_window(self, window: QDialog):
        self.analysis_windows.append(window)
        window.finished.connect(lambda _: self._forget_analysis_window(window))
        window.show()
        return window

    def _forget_analysis_window(self, window: QDialog):
        if window in self.analysis_windows:
            self.analysis_windows.remove(window)

    def _set_count_series_from_spike_series(self) -> None:
        self._count_series = [(_base_channel_from_raster_label(label), times) for label, times in self.spike_series]
        grouped: dict[str, list[np.ndarray]] = {}
        for base_channel, values in self._count_series:
            grouped.setdefault(base_channel, []).append(np.asarray(values, dtype=float))
        self._count_series_by_channel = grouped
        self._count_series_version += 1
        if hasattr(self, "_heatmap_scale_cache"):
            self._heatmap_scale_cache.clear()
        if hasattr(self, "_channel_count_cache"):
            self._channel_count_cache.clear()

    def _window_channel_counts(self, start_s: float, stop_s: float) -> dict[str, int]:
        cache_key = (round(float(start_s), 6), round(float(stop_s), 6), self._count_series_version)
        cached = self._channel_count_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        counts = {}
        for base_channel, series_list in self._count_series_by_channel.items():
            total = 0
            for values in series_list:
                lo = int(np.searchsorted(values, start_s, side="left"))
                hi = int(np.searchsorted(values, stop_s, side="right"))
                total += max(0, hi - lo)
            counts[base_channel] = total
        if len(self._channel_count_cache) > 48:
            self._channel_count_cache.clear()
        self._channel_count_cache[cache_key] = dict(counts)
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
        if channel == self.selected_channel and self._last_waveform_view is not None:
            if self.canvas.selected_channel != channel:
                self.canvas.set_selected_channel(channel)
            return
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


class ResultsWindow(AppDialog):
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


class NevResultsWindow(AppDialog):
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


class SortingResultsWindow(AppDialog):
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


class MaxwellFootprintResultsWindow(AppDialog):
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


class SortingWorkspaceWindow(AppDialog):
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
        self.selected_spike_index = None
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
        self.embedding_canvas.mpl_connect("button_press_event", self._embedding_clicked)
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
        self.selected_spike_index = None
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
            _show_warning_message(self, "Sorting", "The loaded NEV file does not contain spike waveforms.")
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
            _show_warning_message(self, "Sorting", f"{channel} does not contain spike waveforms.")
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
            _show_info_message(self, "Maxwell Footprint Analysis", "Footprint analysis is only enabled for Maxwell H5 data.")
            return
        if not self.data.waveforms:
            _show_warning_message(self, "Maxwell Footprint Analysis", "Footprint analysis requires spike-aligned waveforms.")
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
        _show_error_message(self, "Maxwell footprint analysis failed", details.splitlines()[-1])

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
        _show_error_message(self, "Sorting failed", details.splitlines()[-1])

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
            _show_error_message(self, "Save failed", str(exc))
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
            self.selected_spike_index = None
            return
        progress = self._start_progress("Embedding", f"Computing embedding for {channel}...", 0) if update_status else None
        try:
            self.current_embedding = waveform_embedding(waveforms, self._config())
        except Exception as exc:
            self.status.setText("Embedding update failed")
            if update_status:
                _show_error_message(self, "Embedding failed", str(exc))
            return
        finally:
            if progress is not None:
                _close_progress_dialog(progress)
        self.embedding_view_limits = None
        if self.current_labels.size != waveforms.shape[0]:
            self.current_labels = np.zeros(waveforms.shape[0], dtype=np.int32)
        if self.selected_spike_index is not None and not (0 <= int(self.selected_spike_index) < waveforms.shape[0]):
            self.selected_spike_index = None
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
            _show_info_message(self, "Sorting", "Compute an embedding before manual cluster assignment.")
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

    def _embedding_clicked(self, event):
        if self.lasso_mode is not None:
            return
        if self.embedding_ax is None or event.inaxes is not self.embedding_ax:
            return
        if getattr(event, "button", None) not in (1, None):
            return
        selected = self._nearest_visible_embedding_index(event)
        if selected is None:
            self.selected_spike_index = None
            self.status.setText("No spike selected")
        else:
            self.selected_spike_index = int(selected)
            label = int(self.current_labels[selected]) if self.current_labels.size > selected else 0
            label_text = "noise" if label == -1 else f"cluster {label}"
            self.status.setText(f"Selected spike {selected + 1} ({label_text})")
        self._draw_all()

    def _nearest_visible_embedding_index(self, event):
        if self.current_embedding.ndim != 2 or self.current_embedding.shape[0] == 0:
            return None
        points = self._embedding_xy()
        if points.size == 0:
            return None
        labels = self.current_labels if self.current_labels.size == points.shape[0] else np.zeros(points.shape[0], dtype=int)
        visible_mask = self._visible_label_mask(labels) & np.isfinite(points).all(axis=1)
        visible_indices = np.flatnonzero(visible_mask)
        if visible_indices.size == 0:
            return None
        visible_points = points[visible_indices]
        if getattr(event, "x", None) is not None and getattr(event, "y", None) is not None and self.embedding_ax is not None:
            display_points = self.embedding_ax.transData.transform(visible_points)
            event_xy = np.array([float(event.x), float(event.y)], dtype=float)
            distances = np.linalg.norm(display_points - event_xy, axis=1)
            nearest_pos = int(np.argmin(distances))
            if float(distances[nearest_pos]) <= 18.0:
                return int(visible_indices[nearest_pos])
            return None
        if event.xdata is None or event.ydata is None:
            return None
        data_xy = np.array([float(event.xdata), float(event.ydata)], dtype=float)
        spans = np.array(
            [
                max(abs(self.embedding_ax.get_xlim()[1] - self.embedding_ax.get_xlim()[0]), 1e-9),
                max(abs(self.embedding_ax.get_ylim()[1] - self.embedding_ax.get_ylim()[0]), 1e-9),
            ],
            dtype=float,
        )
        distances = np.linalg.norm((visible_points - data_xy) / spans, axis=1)
        nearest_pos = int(np.argmin(distances))
        if float(distances[nearest_pos]) <= 0.035:
            return int(visible_indices[nearest_pos])
        return None

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
            selected = self.selected_spike_index
            if selected is not None and 0 <= int(selected) < waveforms.shape[0] and visible_mask[int(selected)]:
                selected = int(selected)
                label = int(labels[selected]) if labels.size > selected else 0
                selected_color = "#f97316"
                ax.plot(
                    x,
                    waveforms[selected],
                    color=selected_color,
                    linewidth=2.6,
                    alpha=0.98,
                    label=f"selected spike {selected + 1}",
                    zorder=8,
                )
                ax.text(
                    0.98,
                    0.95,
                    f"spike {selected + 1}\n{'noise' if label == -1 else f'cluster {label}'}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": selected_color, "alpha": 0.92},
                )
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
            selected = self.selected_spike_index
            if selected is not None and 0 <= int(selected) < points.shape[0] and visible_mask[int(selected)]:
                selected = int(selected)
                ax.scatter(
                    [points[selected, 0]],
                    [points[selected, 1]],
                    s=120,
                    facecolors="none",
                    edgecolors="#111827",
                    linewidths=2.2,
                    zorder=9,
                )
                ax.scatter(
                    [points[selected, 0]],
                    [points[selected, 1]],
                    s=34,
                    color="#f97316",
                    edgecolors="#111827",
                    linewidths=0.9,
                    zorder=10,
                )
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_title(f"{self._channel()} reduction space")
            if self.embedding_view_limits is not None:
                ax.set_xlim(self.embedding_view_limits[0])
                ax.set_ylim(self.embedding_view_limits[1])
        self.embedding_canvas.draw_idle()


class TemporalCouplingWindow(AppDialog):
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
        self.bin_ms.setToolTip("Cross-correlogram bin size. Typical range: 0.5-2 ms.")
        self.min_spikes = QSpinBox()
        self.min_spikes.setRange(1, 10000)
        self.min_spikes.setValue(5)
        self.min_spikes.setToolTip("Minimum spike count required per unit to participate in pair analysis.")
        self.max_pairs = QSpinBox()
        self.max_pairs.setRange(1, 500)
        self.max_pairs.setValue(80)
        self.max_pairs.valueChanged.connect(self._resort_results)
        self.max_pairs.setToolTip("Maximum number of ranked directed pairs displayed.")

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

        self.settings_button = QPushButton("Analysis settings...")
        self.settings_button.clicked.connect(self._open_temporal_settings_dialog)
        self.settings_summary = QLabel()
        self.settings_summary.setObjectName("MutedText")
        self.settings_summary.setWordWrap(True)

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

        controls_frame = QFrame()
        controls_frame.setObjectName("Panel")
        controls = QGridLayout(controls_frame)
        controls.setContentsMargins(12, 10, 12, 10)
        controls.setHorizontalSpacing(10)
        controls.setVerticalSpacing(8)
        controls.addWidget(QLabel("Sort by"), 0, 0)
        controls.addWidget(self.sort_by, 0, 1)
        controls.addWidget(self.sort_order, 0, 2)
        controls.addWidget(self.settings_button, 0, 3)
        controls.addWidget(self.analyze_button, 0, 4)
        controls.addWidget(self.status, 0, 5)
        controls.addWidget(self.settings_summary, 1, 0, 1, 6)

        plots = QVBoxLayout()
        plots.addWidget(QLabel("Cross-correlogram"))
        plots.addWidget(self.correlogram_canvas, 1)
        plots.addWidget(QLabel("Reference-aligned target raster"))
        plots.addWidget(self.aligned_canvas, 1)

        body = QHBoxLayout()
        body.addWidget(self.pair_table, 1)
        body.addLayout(plots, 2)

        layout = QVBoxLayout(self)
        layout.addWidget(controls_frame)
        layout.addLayout(body, 1)

        self._update_temporal_settings_summary()
        _fix_spinbox_hit_targets(self)
        self._analyze()
        self.showMaximized()

    def _update_temporal_settings_summary(self) -> None:
        self.settings_summary.setText(
            f"Window {int(self.window_ms.value())} ms, bin {float(self.bin_ms.value()):g} ms, "
            f"min spikes {int(self.min_spikes.value())}, max displayed pairs {int(self.max_pairs.value())}."
        )

    def _open_temporal_settings_dialog(self) -> None:
        dialog = QDialog(self)
        _enable_standard_window_controls(dialog)
        dialog.setWindowTitle("Temporal Coupling Settings")
        layout = QVBoxLayout(dialog)

        intro = QLabel(
            "These parameters control the cross-correlogram time window and pair-screening rules.\n"
            "Keep the main panel focused on ranking and result inspection."
        )
        intro.setObjectName("MutedText")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()

        window_ms = QSpinBox()
        window_ms.setRange(self.window_ms.minimum(), self.window_ms.maximum())
        window_ms.setValue(int(self.window_ms.value()))
        window_ms.setSuffix(" ms")
        window_ms.setToolTip("Half-width of the lag window searched around each reference spike. Typical range: 20-200 ms.")

        bin_ms = QDoubleSpinBox()
        bin_ms.setRange(self.bin_ms.minimum(), self.bin_ms.maximum())
        bin_ms.setDecimals(self.bin_ms.decimals())
        bin_ms.setSingleStep(self.bin_ms.singleStep())
        bin_ms.setValue(float(self.bin_ms.value()))
        bin_ms.setSuffix(" ms")
        bin_ms.setToolTip("Histogram bin size for lag counts.")

        min_spikes = QSpinBox()
        min_spikes.setRange(self.min_spikes.minimum(), self.min_spikes.maximum())
        min_spikes.setValue(int(self.min_spikes.value()))
        min_spikes.setToolTip("Units below this spike count are skipped to avoid unstable estimates.")

        max_pairs = QSpinBox()
        max_pairs.setRange(self.max_pairs.minimum(), self.max_pairs.maximum())
        max_pairs.setValue(int(self.max_pairs.value()))
        max_pairs.setToolTip("Only the top-ranked directed pairs are shown in the table.")

        form.addRow("Lag window", window_ms)
        form.addRow("Histogram bin", bin_ms)
        form.addRow("Minimum spikes per unit", min_spikes)
        form.addRow("Max displayed pairs", max_pairs)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        apply_button = QPushButton("Apply")
        buttons.addWidget(cancel)
        buttons.addWidget(apply_button)
        layout.addLayout(buttons)
        cancel.clicked.connect(dialog.reject)
        apply_button.clicked.connect(dialog.accept)

        if dialog.exec() != QDialog.Accepted:
            return
        self.window_ms.setValue(int(window_ms.value()))
        self.bin_ms.setValue(float(bin_ms.value()))
        self.min_spikes.setValue(int(min_spikes.value()))
        self.max_pairs.setValue(int(max_pairs.value()))
        self._update_temporal_settings_summary()
        self._analyze()

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


class StimulusGenerationDialog(AppDialog):
    def __init__(self, records: list[dict], parent=None, channel_map: ChannelMap | None = None):
        super().__init__(parent)
        self.setWindowTitle("Stimulus Generation")
        self.resize(940, 640)
        self.records = list(records or [])
        self.channel_map = channel_map
        self.info = stimulus_builder.ExperimentInfo()
        self.groups = [stimulus_builder.ElectrodeGroup("group_A", self._default_electrodes())]
        self.protocols = [
            stimulus_builder.StimulusProtocol("feedback_single_150mV", "single_pulse", amplitude_mv=150.0),
            stimulus_builder.StimulusProtocol(
                "train_burst_20hz",
                "individual_burst",
                amplitude_mv=150.0,
                pulse_width_us=200.0,
                pulse_frequency_hz=20.0,
                pulses_per_burst=10,
            ),
            stimulus_builder.StimulusProtocol(
                "poisson_random_safe",
                "poisson_random_electrodes",
                amplitude_mv=150.0,
                pulse_width_us=200.0,
                pulses_per_burst=1,
                poisson_duration_s=300.0,
                lambda_mode="scale",
                random_seed=42,
            ),
        ]
        self.protocol_source_paths: dict[str, str] = {}
        if self.records:
            self.protocol_source_paths["poisson_random_safe"] = str(self.records[0].get("path", ""))
        self.blocks = [stimulus_builder.ExperimentBlock("group_A_150mV", "group_A", "feedback_single_150mV")]
        self.poisson_auto_groups: dict[str, str] = {}
        self.output_dir = Path("generated_visual_experiment")
        self.preview_map_selected = None
        self.preview_map_selection_artist = None
        self._preview_series_cache: dict[tuple, list[dict]] = {}
        self._preview_rate_cache: dict[tuple, tuple[list[int], list[float]]] = {}
        self._preview_recording_metrics_cache: dict[tuple, tuple[list[int], dict[int, dict]]] = {}
        self._preview_electrode_lookup_key = None
        self._preview_electrode_lookup: dict[str, int] = {}
        self.preview_window_start_ms = 0.0
        self.preview_window_ms = 5000.0
        self.preview_total_ms = 5000.0
        self.preview_raster_axis = None
        self.preview_raster_protocol = None
        self.preview_raster_series: list[dict] = []
        self.preview_raster_arrays: list[tuple[dict, np.ndarray]] = []
        self._preview_view_key = None
        self._preview_drag = None
        self._build_ui()
        self._refresh_all()

    def refresh_pipeline_context(self, records: list[dict], channel_map: ChannelMap | None = None) -> None:
        self.records = list(records or [])
        self.channel_map = channel_map or self.channel_map
        self._clear_preview_caches()
        self._refresh_source_combo()

    def _clear_preview_caches(self) -> None:
        self._preview_series_cache.clear()
        self._preview_rate_cache.clear()
        self._preview_recording_metrics_cache.clear()
        self._preview_electrode_lookup_key = None
        self._preview_electrode_lookup = {}

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.experiment_tab = QWidget()
        self.groups_tab = QWidget()
        self.protocols_tab = QWidget()
        self.blocks_tab = QWidget()
        self.preview_tab = QWidget()
        self.generate_tab = QWidget()
        self.tabs.addTab(self.experiment_tab, "Experiment")
        self.tabs.addTab(self.groups_tab, "Electrodes")
        self.tabs.addTab(self.protocols_tab, "Stimulus")
        self.tabs.addTab(self.blocks_tab, "Blocks")
        self.tabs.addTab(self.preview_tab, "Preview")
        self.tabs.addTab(self.generate_tab, "Generate")
        self._build_experiment_tab()
        self._build_groups_tab()
        self._build_protocols_tab()
        self._build_blocks_tab()
        self._build_preview_tab()
        self._build_generate_tab()

    def _build_experiment_tab(self) -> None:
        layout = QVBoxLayout(self.experiment_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        summary = QLabel(
            "Set the experiment identity and output roots used by the generated MaxWell package. "
            "Hardware defaults are kept unless you edit them later in the generated config files."
        )
        summary.setObjectName("MutedText")
        summary.setWordWrap(True)
        layout.addWidget(summary)
        form = QFormLayout()
        layout.addLayout(form)
        self.info_fields: dict[str, QLineEdit] = {}
        for label, key in [
            ("Experiment name", "name"),
            ("Culture ID", "culture_id"),
            ("DIV", "div"),
            ("Date", "date"),
            ("Recording prefix", "recording_prefix"),
            ("CFG path", "cfg_path"),
            ("Data root", "data_root"),
        ]:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            field = QLineEdit(str(getattr(self.info, key)))
            self.info_fields[key] = field
            row_layout.addWidget(field, 1)
            if key == "cfg_path":
                browse = QPushButton("Browse")
                browse.clicked.connect(self._browse_cfg)
                row_layout.addWidget(browse)
            form.addRow(label, row)
        for key in ["device", "event_threshold", "amplifier_gain", "cpp_runner", "spike_step", "max_stims", "sequence_name"]:
            self.info_fields[key] = QLineEdit(str(getattr(self.info, key)))

        self.info_texts: dict[str, QTextEdit] = {}
        for label, key in [
            ("Scientific question", "scientific_question"),
            ("Closed-loop logic", "closed_loop_logic"),
        ]:
            text = QTextEdit()
            text.setMaximumHeight(58)
            text.setPlainText(str(getattr(self.info, key)))
            self.info_texts[key] = text
            form.addRow(label, text)
        self.info_texts["expected_output"] = QTextEdit()
        self.info_texts["expected_output"].setPlainText(str(getattr(self.info, "expected_output")))
        hint = QLabel("Spontaneous data for random stimulation is selected from the current pipeline database in the Stimulus tab.")
        hint.setObjectName("MutedText")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)

    def _build_groups_tab(self) -> None:
        layout = QHBoxLayout(self.groups_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        left = QVBoxLayout()
        hint = QLabel("Electrode groups are named sets used by experiment blocks. Defaults come from the current channel map when possible.")
        hint.setObjectName("MutedText")
        hint.setWordWrap(True)
        left.addWidget(hint)
        self.group_table = QTableWidget(0, 2)
        self.group_table.setHorizontalHeaderLabels(["Group", "Electrodes"])
        self.group_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.group_table.itemSelectionChanged.connect(self._load_selected_group)
        left.addWidget(self.group_table, 1)
        layout.addLayout(left, 1)

        editor = QFrame()
        editor.setObjectName("Panel")
        edit_layout = QVBoxLayout(editor)
        edit_layout.setContentsMargins(10, 10, 10, 10)
        edit_layout.setSpacing(8)
        self.group_name = QLineEdit()
        self.group_electrodes = QLineEdit()
        form = QFormLayout()
        form.addRow("Group name", self.group_name)
        form.addRow("Electrodes", self.group_electrodes)
        edit_layout.addLayout(form)
        buttons = QHBoxLayout()
        save = QPushButton("Add / Update")
        save.clicked.connect(self._save_group)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_group)
        buttons.addWidget(save)
        buttons.addWidget(remove)
        edit_layout.addLayout(buttons)
        edit_layout.addStretch(1)
        layout.addWidget(editor)

    def _build_protocols_tab(self) -> None:
        layout = QHBoxLayout(self.protocols_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self.protocol_table = QTableWidget(0, 3)
        self.protocol_table.setHorizontalHeaderLabels(["Protocol", "Type", "Pipeline source"])
        self.protocol_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.protocol_table.itemSelectionChanged.connect(self._load_selected_protocol)
        self.protocol_table.setMaximumWidth(420)
        layout.addWidget(self.protocol_table, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumHeight(0)
        editor = QFrame()
        editor.setObjectName("Panel")
        editor.setMinimumHeight(0)
        edit_layout = QVBoxLayout(editor)
        edit_layout.setContentsMargins(10, 10, 10, 10)
        edit_layout.setSpacing(8)
        explanation = QLabel(
            "Define stimulation patterns. For poisson_random_electrodes, choose a spontaneous recording from the pipeline database; "
            "its firing rates are exported into the generated package."
        )
        explanation.setObjectName("MutedText")
        explanation.setWordWrap(True)
        edit_layout.addWidget(explanation)

        self.source_box = QGroupBox("Pipeline spontaneous source")
        source_layout = QVBoxLayout(self.source_box)
        source_layout.setContentsMargins(8, 8, 8, 8)
        source_hint = QLabel("Use a loaded spontaneous spike dataset as the rate template for random-electrode stimulation.")
        source_hint.setObjectName("MutedText")
        source_hint.setWordWrap(True)
        source_layout.addWidget(source_hint)
        self.source_combo = QComboBox()
        self.source_combo.currentIndexChanged.connect(self._source_selection_changed)
        source_layout.addWidget(self.source_combo)
        edit_layout.addWidget(self.source_box)

        main_box = QGroupBox("Protocol")
        form = QGridLayout(main_box)
        form.setContentsMargins(8, 8, 8, 8)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(5)
        self.protocol_fields: dict[str, QLineEdit] = {}
        self.protocol_field_labels: dict[str, QLabel] = {}
        self.protocol_type = QComboBox()
        self.protocol_type.addItems(list(stimulus_builder.PROTOCOL_TYPES))
        form.addWidget(QLabel("Type"), 0, 0)
        form.addWidget(self.protocol_type, 0, 1)
        compact_rows = [
            ("Name", "name"),
            ("Amplitude mV", "amplitude_mv"),
            ("Pulse width us", "pulse_width_us"),
            ("Pulse frequency Hz", "pulse_frequency_hz"),
            ("Pulses per burst", "pulses_per_burst"),
            ("Interpulse ms", "interpulse_interval_ms"),
            ("Burst count", "burst_count"),
            ("Burst interval ms", "burst_interval_ms"),
            ("Start ms", "start_ms"),
            ("DAC channel", "channel"),
            ("Poisson duration s", "poisson_duration_s"),
        ]
        for index, (label, key) in enumerate(compact_rows, start=1):
            field = QLineEdit()
            field.setMaximumWidth(150)
            self.protocol_fields[key] = field
            grid_row = index // 2
            grid_col = (index % 2) * 2
            label_widget = QLabel(label)
            self.protocol_field_labels[key] = label_widget
            form.addWidget(label_widget, grid_row, grid_col)
            form.addWidget(field, grid_row, grid_col + 1)
        for key in [
            "inter_phase_interval_us",
            "region_count",
            "max_candidate_electrodes",
            "lambda_scale",
            "lambda_floor_hz",
            "lambda_mean_hz",
            "lambda_std_hz",
            "random_seed",
        ]:
            self.protocol_fields[key] = QLineEdit()
        self.lambda_mode = QComboBox()
        self.lambda_mode.addItems(["scale", "normal"])
        self.lambda_mode.currentIndexChanged.connect(lambda *_: self._update_protocol_type_fields())
        lambda_row = (len(compact_rows) + 1) // 2
        self.lambda_mode_label = QLabel("Lambda mode")
        form.addWidget(self.lambda_mode_label, lambda_row, 2)
        form.addWidget(self.lambda_mode, lambda_row, 3)
        edit_layout.addWidget(main_box)

        self.advanced_toggle = QPushButton("Show advanced")
        self.advanced_toggle.setCheckable(True)
        self.advanced_box = QGroupBox("Advanced")
        self.advanced_box.setVisible(False)
        self.advanced_toggle.toggled.connect(lambda *_: self._update_protocol_advanced_visibility())
        advanced_form = QFormLayout(self.advanced_box)
        for label, key in [
            ("Region size", "region_count"),
            ("Max candidates", "max_candidate_electrodes"),
            ("Lambda scale", "lambda_scale"),
            ("Lambda floor Hz", "lambda_floor_hz"),
            ("Normal mean Hz", "lambda_mean_hz"),
            ("Normal std Hz", "lambda_std_hz"),
            ("Random seed", "random_seed"),
            ("Inter-phase us", "inter_phase_interval_us"),
        ]:
            advanced_form.addRow(label, self.protocol_fields[key])
            self.protocol_field_labels[key] = advanced_form.labelForField(self.protocol_fields[key])
            self.protocol_fields[key].setMaximumWidth(160)
        edit_layout.addWidget(self.advanced_toggle)
        edit_layout.addWidget(self.advanced_box)

        self.custom_points = QTextEdit()
        self.custom_points.setMaximumHeight(64)
        self.custom_points.setPlaceholderText("time_ms, amplitude_mv, duration_us, channel")
        self.custom_points_label = QLabel("Custom sequence")
        edit_layout.addWidget(self.custom_points_label)
        edit_layout.addWidget(self.custom_points)
        self.protocol_notes = QTextEdit()
        self.protocol_notes.setMaximumHeight(46)
        edit_layout.addWidget(QLabel("Notes"))
        edit_layout.addWidget(self.protocol_notes)
        buttons = QHBoxLayout()
        save = QPushButton("Add / Update")
        save.clicked.connect(self._save_protocol)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_protocol)
        preview = QPushButton("Preview")
        preview.clicked.connect(self._preview_form_protocol)
        buttons.addWidget(save)
        buttons.addWidget(remove)
        buttons.addWidget(preview)
        edit_layout.addLayout(buttons)
        scroll.setWidget(editor)
        layout.addWidget(scroll, 1)
        self.protocol_type.currentIndexChanged.connect(self._protocol_type_changed)
        self._update_protocol_type_fields()

    def _build_blocks_tab(self) -> None:
        layout = QHBoxLayout(self.blocks_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        left = QVBoxLayout()
        hint = QLabel("Blocks combine one electrode group, one protocol, and the fixed pre/stim/post phases written into the generated run package.")
        hint.setObjectName("MutedText")
        hint.setWordWrap(True)
        left.addWidget(hint)
        self.block_table = QTableWidget(0, 3)
        self.block_table.setHorizontalHeaderLabels(["Block", "Group", "Protocol"])
        self.block_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.block_table.itemSelectionChanged.connect(self._load_selected_block)
        left.addWidget(self.block_table, 1)
        layout.addLayout(left, 1)

        editor = QFrame()
        editor.setObjectName("Panel")
        edit_layout = QVBoxLayout(editor)
        edit_layout.setContentsMargins(10, 10, 10, 10)
        edit_layout.setSpacing(8)
        form = QFormLayout()
        self.block_name = QLineEdit()
        self.block_group = QComboBox()
        self.block_protocol = QComboBox()
        form.addRow("Block name", self.block_name)
        form.addRow("Electrode group", self.block_group)
        form.addRow("Protocol", self.block_protocol)
        self.phase_duration_fields: dict[str, QLineEdit] = {}
        self.phase_mode_combos: dict[str, QComboBox] = {}
        for phase_id in stimulus_builder.PHASES:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            duration = QLineEdit("300")
            mode = QComboBox()
            mode.addItems(["open_loop", "closed_loop", "manual"])
            self.phase_duration_fields[phase_id] = duration
            self.phase_mode_combos[phase_id] = mode
            row_layout.addWidget(duration)
            row_layout.addWidget(mode)
            form.addRow(phase_id, row)
        edit_layout.addLayout(form)
        buttons = QHBoxLayout()
        save = QPushButton("Add / Update")
        save.clicked.connect(self._save_block)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_block)
        buttons.addWidget(save)
        buttons.addWidget(remove)
        edit_layout.addLayout(buttons)
        edit_layout.addStretch(1)
        layout.addWidget(editor)

    def _build_preview_tab(self) -> None:
        layout = QVBoxLayout(self.preview_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        hint = QLabel("Preview shows the event timing pattern for the selected protocol before code generation.")
        hint.setObjectName("MutedText")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        controls = QHBoxLayout()
        self.preview_combo = QComboBox()
        controls.addWidget(self.preview_combo, 1)
        self.preview_window_label = QLabel("0.000 - 5.000 s / 5.000 s")
        self.preview_window_label.setObjectName("MutedText")
        self.preview_window_label.setMinimumWidth(190)
        self.preview_window_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.preview_window_label)
        reset_view = QPushButton("Reset")
        reset_view.clicked.connect(self._reset_preview_raster_view)
        controls.addWidget(reset_view)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._draw_preview)
        controls.addWidget(refresh)
        layout.addLayout(controls)
        plot_layout = QHBoxLayout()
        plot_layout.setSpacing(10)
        self.preview_canvas = FigureCanvas(Figure(figsize=(5.8, 4.2), tight_layout=True))
        self.preview_canvas.mpl_connect("scroll_event", self._preview_raster_scrolled)
        self.preview_canvas.mpl_connect("button_press_event", self._preview_raster_mouse_pressed)
        self.preview_canvas.mpl_connect("motion_notify_event", self._preview_raster_mouse_moved)
        self.preview_canvas.mpl_connect("button_release_event", self._preview_raster_mouse_released)
        self.preview_map_canvas = FigureCanvas(Figure(figsize=(3.8, 4.2), tight_layout=True))
        self.preview_map_canvas.mpl_connect("button_press_event", self._preview_map_clicked)
        self.preview_map_detail = QLabel("Click an electrode to inspect it.")
        self.preview_map_detail.setObjectName("MutedText")
        self.preview_map_detail.setWordWrap(True)
        map_panel = QVBoxLayout()
        map_panel.addWidget(self.preview_map_canvas, 1)
        map_panel.addWidget(self.preview_map_detail)
        plot_layout.addWidget(self.preview_canvas, 3)
        plot_layout.addLayout(map_panel, 2)
        layout.addLayout(plot_layout, 1)

    def _build_generate_tab(self) -> None:
        layout = QVBoxLayout(self.generate_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        hint = QLabel(
            "Generate writes a standalone MaxWell experiment package with config YAML, Python runner, C++ scaffold, "
            "and exported firing-rate sources for random stimulation."
        )
        hint.setObjectName("MutedText")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self.output_path = QLineEdit(str(self.output_dir))
        row_layout.addWidget(self.output_path, 1)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_output)
        row_layout.addWidget(browse)
        form.addRow("Output directory", row)
        layout.addLayout(form)
        generate = QPushButton("Generate Code Package")
        generate.setObjectName("PrimaryButton")
        generate.clicked.connect(self._generate)
        layout.addWidget(generate)
        self.generate_status = QLabel("Ready")
        self.generate_status.setObjectName("MutedText")
        self.generate_status.setWordWrap(True)
        layout.addWidget(self.generate_status)
        layout.addStretch(1)

    def _refresh_all(self) -> None:
        self._refresh_group_table()
        self._refresh_protocol_table()
        self._refresh_block_table()
        self._refresh_block_combos()
        self._refresh_preview_combo()
        self._refresh_source_combo()
        if self.protocols:
            self._fill_protocol_form(self.protocols[0])
        if self.groups:
            self._fill_group_form(self.groups[0])
        if self.blocks:
            self._fill_block_form(self.blocks[0])

    def _refresh_group_table(self) -> None:
        self.group_table.setRowCount(len(self.groups))
        for row, group in enumerate(self.groups):
            self.group_table.setItem(row, 0, QTableWidgetItem(group.name))
            self.group_table.setItem(row, 1, QTableWidgetItem(", ".join(str(item) for item in group.electrodes)))
        self.group_table.resizeColumnsToContents()

    def _refresh_protocol_table(self) -> None:
        self.protocol_table.setRowCount(len(self.protocols))
        for row, protocol in enumerate(self.protocols):
            source = self._source_label(self.protocol_source_paths.get(protocol.name, ""))
            for column, value in enumerate([protocol.name, protocol.type, source]):
                self.protocol_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.protocol_table.resizeColumnsToContents()

    def _refresh_block_table(self) -> None:
        self.block_table.setRowCount(len(self.blocks))
        for row, block in enumerate(self.blocks):
            for column, value in enumerate([block.name, block.electrode_group, block.protocol]):
                self.block_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.block_table.resizeColumnsToContents()

    def _refresh_block_combos(self) -> None:
        current_group = self.block_group.currentData()
        current_protocol = self.block_protocol.currentData()
        self.block_group.blockSignals(True)
        self.block_protocol.blockSignals(True)
        self.block_group.clear()
        self.block_protocol.clear()
        for group in self.groups:
            self.block_group.addItem(group.name, group.name)
        for protocol in self.protocols:
            self.block_protocol.addItem(protocol.name, protocol.name)
        self._set_combo_data(self.block_group, current_group)
        self._set_combo_data(self.block_protocol, current_protocol)
        self.block_group.blockSignals(False)
        self.block_protocol.blockSignals(False)

    def _refresh_preview_combo(self) -> None:
        current = self.preview_combo.currentData()
        self.preview_combo.blockSignals(True)
        self.preview_combo.clear()
        for protocol in self.protocols:
            self.preview_combo.addItem(protocol.name, protocol.name)
        self._set_combo_data(self.preview_combo, current)
        self.preview_combo.blockSignals(False)
        if self.protocols:
            self._draw_preview()

    def _refresh_source_combo(self) -> None:
        if not hasattr(self, "source_combo"):
            return
        current = self.source_combo.currentData()
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItem("Pipeline database: none", "")
        for record in self.records:
            path = str(record.get("path", ""))
            if path:
                self.source_combo.addItem(self._record_label(record), path)
        self._set_combo_data(self.source_combo, current)
        self.source_combo.blockSignals(False)

    def _source_selection_changed(self, *_args) -> None:
        name = ""
        if hasattr(self, "protocol_fields") and "name" in self.protocol_fields:
            name = self.protocol_fields["name"].text().strip()
        source_path = str(self.source_combo.currentData() or "")
        self._clear_preview_caches()
        if name:
            if source_path:
                self.protocol_source_paths[name] = source_path
            else:
                self.protocol_source_paths.pop(name, None)
            self._update_protocol_from_form_if_possible(name)
            self._sync_poisson_protocol_auto_group(name)
            self._refresh_protocol_table()
        if hasattr(self, "preview_combo") and str(self.preview_combo.currentData() or "") == name:
            self._draw_preview()

    def _selected_table_row(self, table: QTableWidget) -> int | None:
        rows = sorted({index.row() for index in table.selectedIndexes()})
        return rows[0] if rows else None

    def _load_selected_group(self) -> None:
        row = self._selected_table_row(self.group_table)
        if row is not None and 0 <= row < len(self.groups):
            self._fill_group_form(self.groups[row])

    def _load_selected_protocol(self) -> None:
        row = self._selected_table_row(self.protocol_table)
        if row is not None and 0 <= row < len(self.protocols):
            self._fill_protocol_form(self.protocols[row])

    def _load_selected_block(self) -> None:
        row = self._selected_table_row(self.block_table)
        if row is not None and 0 <= row < len(self.blocks):
            self._fill_block_form(self.blocks[row])

    def _fill_group_form(self, group) -> None:
        self.group_name.setText(group.name)
        self.group_electrodes.setText(", ".join(str(item) for item in group.electrodes))

    def _fill_protocol_form(self, protocol) -> None:
        self.protocol_fields["name"].setText(protocol.name)
        self._set_combo_text(self.protocol_type, protocol.type)
        for key, field in self.protocol_fields.items():
            if key == "name":
                continue
            field.setText(str(getattr(protocol, key)))
        self._set_combo_text(self.lambda_mode, protocol.lambda_mode)
        self.custom_points.setPlainText("\n".join(
            f"{point['time_ms']}, {point['amplitude_mv']}, {point.get('duration_us', protocol.pulse_width_us)}, {point.get('channel', protocol.channel)}"
            for point in protocol.custom_points
        ))
        self.protocol_notes.setPlainText(protocol.notes)
        self._set_combo_data(self.source_combo, self.protocol_source_paths.get(protocol.name, ""))
        self._update_protocol_type_fields()

    def _fill_block_form(self, block) -> None:
        self.block_name.setText(block.name)
        self._set_combo_data(self.block_group, block.electrode_group)
        self._set_combo_data(self.block_protocol, block.protocol)
        for phase in block.phases:
            if phase.id in self.phase_duration_fields:
                self.phase_duration_fields[phase.id].setText(str(phase.duration_s))
            if phase.id in self.phase_mode_combos:
                self._set_combo_text(self.phase_mode_combos[phase.id], phase.mode)

    def _save_group(self) -> None:
        try:
            group = stimulus_builder.ElectrodeGroup(
                self.group_name.text().strip(),
                stimulus_builder.parse_electrodes(self.group_electrodes.text()),
            )
            if not group.name:
                raise ValueError("Group name is required")
        except Exception as exc:
            _show_error_message(self, "Invalid group", str(exc))
            return
        row = self._selected_table_row(self.group_table)
        if row is None or row >= len(self.groups):
            self.groups.append(group)
        else:
            self.groups[row] = group
        self._clear_preview_caches()
        self._refresh_group_table()
        self._refresh_block_combos()

    def _save_protocol(self) -> None:
        try:
            protocol = self._protocol_from_form()
            if not protocol.name:
                raise ValueError("Protocol name is required")
            if protocol.type not in stimulus_builder.PROTOCOL_TYPES:
                raise ValueError("Unsupported protocol type")
        except Exception as exc:
            _show_error_message(self, "Invalid protocol", str(exc))
            return
        row = self._selected_table_row(self.protocol_table)
        if row is None or row >= len(self.protocols):
            self.protocols.append(protocol)
        else:
            old_name = self.protocols[row].name
            self.protocols[row] = protocol
            if old_name != protocol.name and old_name in self.protocol_source_paths:
                self.protocol_source_paths[protocol.name] = self.protocol_source_paths.pop(old_name)
            if old_name != protocol.name and old_name in self.poisson_auto_groups:
                self.poisson_auto_groups[protocol.name] = self.poisson_auto_groups.pop(old_name)
        source_path = str(self.source_combo.currentData() or "")
        if source_path:
            self.protocol_source_paths[protocol.name] = source_path
        else:
            self.protocol_source_paths.pop(protocol.name, None)
        self._clear_preview_caches()
        self._sync_poisson_protocol_auto_group(protocol.name)
        self._refresh_protocol_table()
        self._refresh_block_combos()
        self._refresh_preview_combo()
        self._set_combo_data(self.preview_combo, protocol.name)
        self._draw_preview()

    def _save_block(self) -> None:
        try:
            phases = [
                stimulus_builder.Phase(
                    phase_id,
                    int(self.phase_duration_fields[phase_id].text() or 300),
                    str(self.phase_mode_combos[phase_id].currentText() or "open_loop"),
                )
                for phase_id in stimulus_builder.PHASES
            ]
            block = stimulus_builder.ExperimentBlock(
                self.block_name.text().strip(),
                str(self.block_group.currentData() or ""),
                str(self.block_protocol.currentData() or ""),
                phases,
            )
            if not block.name:
                raise ValueError("Block name is required")
        except Exception as exc:
            _show_error_message(self, "Invalid block", str(exc))
            return
        row = self._selected_table_row(self.block_table)
        if row is None or row >= len(self.blocks):
            self.blocks.append(block)
        else:
            self.blocks[row] = block
        self._clear_preview_caches()
        self._sync_poisson_protocol_auto_group(block.protocol)
        self._refresh_block_table()

    def _remove_group(self) -> None:
        row = self._selected_table_row(self.group_table)
        if row is not None and 0 <= row < len(self.groups):
            removed_name = self.groups[row].name
            self.poisson_auto_groups = {key: value for key, value in self.poisson_auto_groups.items() if value != removed_name}
            del self.groups[row]
            self._clear_preview_caches()
            self._refresh_group_table()
            self._refresh_block_combos()

    def _remove_protocol(self) -> None:
        row = self._selected_table_row(self.protocol_table)
        if row is not None and 0 <= row < len(self.protocols):
            self.protocol_source_paths.pop(self.protocols[row].name, None)
            self._remove_poisson_auto_group(self.protocols[row].name)
            del self.protocols[row]
            self._clear_preview_caches()
            self._refresh_protocol_table()
            self._refresh_block_combos()
            self._refresh_preview_combo()

    def _remove_block(self) -> None:
        row = self._selected_table_row(self.block_table)
        if row is not None and 0 <= row < len(self.blocks):
            del self.blocks[row]
            self._clear_preview_caches()
            self._refresh_block_table()

    def _update_protocol_from_form_if_possible(self, protocol_name: str) -> None:
        try:
            protocol = self._protocol_from_form()
        except Exception:
            return
        if protocol.name != protocol_name:
            return
        for index, existing in enumerate(self.protocols):
            if existing.name == protocol_name:
                self.protocols[index] = protocol
                return

    def _sync_poisson_protocol_auto_group(self, protocol_name: str) -> None:
        protocol = next((item for item in self.protocols if item.name == protocol_name), None)
        if protocol is None or protocol.type != "poisson_random_electrodes":
            self._remove_poisson_auto_group(protocol_name)
            return
        related_blocks = [block for block in self.blocks if block.protocol == protocol.name]
        source_path = str(self.protocol_source_paths.get(protocol.name, "") or "")
        if source_path:
            protocol.spontaneous_data_path = source_path
        manual_group = self._manual_group_for_poisson_blocks(related_blocks)
        fallback = list(manual_group.electrodes) if manual_group is not None else self._default_electrodes()
        try:
            candidates = self._poisson_candidate_electrodes_for_ui(protocol, fallback)
        except Exception as exc:
            self.generate_status.setText(f"Poisson auto electrodes unavailable: {exc}")
            return
        group_name = self.poisson_auto_groups.get(protocol.name)
        if not group_name:
            base_group = manual_group.name if manual_group is not None else "poisson"
            group_name = self._unique_group_name(f"{base_group}_{protocol.name}_auto")
            self.poisson_auto_groups[protocol.name] = group_name
        auto_group = next((group for group in self.groups if group.name == group_name), None)
        if auto_group is None:
            auto_group = stimulus_builder.ElectrodeGroup(group_name, [])
            self.groups.append(auto_group)
        auto_group.electrodes = [int(value) for value in candidates]
        for block in related_blocks:
            block.electrode_group = group_name
        self._clear_preview_caches()
        self._refresh_group_table()
        self._refresh_block_table()
        self._refresh_block_combos()
        self.generate_status.setText(f"Poisson auto electrodes updated: {group_name} ({len(candidates)})")

    def _poisson_candidate_electrodes_for_ui(self, protocol, fallback: list[int]) -> list[int]:
        source_path = self._preview_source_path_for_protocol(protocol)
        source_record = self._record_by_path(source_path)
        if isinstance(source_record, dict):
            electrodes, rates = self._rate_table_from_record(source_record)
            rate_map = {int(electrode): float(rate) for electrode, rate in zip(electrodes, rates)}
            candidates = stimulus_builder._preview_candidate_electrodes(
                rate_map,
                int(getattr(protocol, "region_count", 32)),
                int(getattr(protocol, "max_candidate_electrodes", 32)),
            )
            if not candidates:
                raise ValueError(f"No poisson candidate electrodes could be selected for protocol {protocol.name}")
            return candidates
        return stimulus_builder.poisson_candidate_electrodes_for_protocol(protocol, fallback)

    def _remove_poisson_auto_group(self, protocol_name: str) -> None:
        group_name = self.poisson_auto_groups.pop(protocol_name, "")
        if not group_name:
            return
        self.groups = [group for group in self.groups if group.name != group_name]
        fallback = self.groups[0].name if self.groups else ""
        for block in self.blocks:
            if block.electrode_group == group_name:
                block.electrode_group = fallback

    def _manual_group_for_poisson_blocks(self, blocks: list) -> object | None:
        auto_group_names = set(self.poisson_auto_groups.values())
        group_lookup = {group.name: group for group in self.groups}
        for block in blocks:
            if block.electrode_group in auto_group_names:
                continue
            group = group_lookup.get(block.electrode_group)
            if group is not None:
                return group
        row = self._selected_table_row(self.group_table) if hasattr(self, "group_table") else None
        if row is not None and 0 <= row < len(self.groups):
            group = self.groups[row]
            if group.name not in auto_group_names:
                return group
        return next((group for group in self.groups if group.name not in auto_group_names), None)

    def _unique_group_name(self, base: str) -> str:
        existing = {group.name for group in self.groups}
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(base)).strip("_") or "poisson_auto"
        if cleaned not in existing:
            return cleaned
        index = 2
        while f"{cleaned}_{index}" in existing:
            index += 1
        return f"{cleaned}_{index}"

    def _protocol_from_form(self):
        return stimulus_builder.StimulusProtocol(
            name=self.protocol_fields["name"].text().strip(),
            type=str(self.protocol_type.currentText()).strip(),
            amplitude_mv=float(self.protocol_fields["amplitude_mv"].text() or 150),
            pulse_width_us=float(self.protocol_fields["pulse_width_us"].text() or 300),
            inter_phase_interval_us=float(self.protocol_fields["inter_phase_interval_us"].text() or 0),
            pulse_frequency_hz=float(self.protocol_fields["pulse_frequency_hz"].text() or 20),
            pulses_per_burst=int(self.protocol_fields["pulses_per_burst"].text() or 5),
            interpulse_interval_ms=float(self.protocol_fields["interpulse_interval_ms"].text() or 50),
            burst_count=int(self.protocol_fields["burst_count"].text() or 3),
            burst_interval_ms=float(self.protocol_fields["burst_interval_ms"].text() or 200),
            start_ms=float(self.protocol_fields["start_ms"].text() or 1500),
            channel=int(self.protocol_fields["channel"].text() or 0),
            custom_points=stimulus_builder.parse_custom_points(self.custom_points.toPlainText()),
            region_count=int(self.protocol_fields["region_count"].text() or 32),
            max_candidate_electrodes=int(self.protocol_fields["max_candidate_electrodes"].text() or 32),
            poisson_duration_s=float(self.protocol_fields["poisson_duration_s"].text() or 300),
            lambda_mode=str(self.lambda_mode.currentText() or "scale"),
            lambda_scale=float(self.protocol_fields["lambda_scale"].text() or 1.0),
            lambda_floor_hz=float(self.protocol_fields["lambda_floor_hz"].text() or 0.001),
            lambda_mean_hz=float(self.protocol_fields["lambda_mean_hz"].text() or 1.0),
            lambda_std_hz=float(self.protocol_fields["lambda_std_hz"].text() or 0.25),
            random_seed=int(self.protocol_fields["random_seed"].text() or 42),
            notes=self.protocol_notes.toPlainText().strip(),
        )

    def _protocol_type_changed(self, *_args) -> None:
        self._clear_preview_caches()
        self._update_protocol_type_fields()

    def _protocol_field_sets_for_type(self, protocol_type: str) -> tuple[set[str], set[str], bool, bool, bool]:
        protocol_type = str(protocol_type or "")
        visible = {"name"}
        advanced = set()
        show_custom = False
        show_source = False
        show_lambda = False
        if protocol_type == "single_pulse":
            visible.update({"amplitude_mv", "pulse_width_us", "start_ms", "channel"})
            advanced.add("inter_phase_interval_us")
        elif protocol_type == "individual_burst":
            visible.update({
                "amplitude_mv",
                "pulse_width_us",
                "pulse_frequency_hz",
                "pulses_per_burst",
                "interpulse_interval_ms",
                "start_ms",
                "channel",
            })
            advanced.add("inter_phase_interval_us")
        elif protocol_type == "sequence_with_burst":
            visible.update({
                "amplitude_mv",
                "pulse_width_us",
                "pulse_frequency_hz",
                "pulses_per_burst",
                "interpulse_interval_ms",
                "burst_count",
                "burst_interval_ms",
                "start_ms",
                "channel",
            })
            advanced.add("inter_phase_interval_us")
        elif protocol_type == "custom_sequence":
            visible.update({"pulse_width_us", "channel"})
            advanced.add("inter_phase_interval_us")
            show_custom = True
        elif protocol_type == "poisson_random_electrodes":
            visible.update({"amplitude_mv", "pulse_width_us", "pulses_per_burst", "poisson_duration_s"})
            advanced.update({
                "region_count",
                "max_candidate_electrodes",
                "lambda_scale",
                "lambda_floor_hz",
                "lambda_mean_hz",
                "lambda_std_hz",
                "random_seed",
            })
            show_source = True
            show_lambda = True
        return visible, advanced, show_custom, show_source, show_lambda

    def _update_protocol_type_fields(self) -> None:
        if not hasattr(self, "protocol_type"):
            return
        visible, advanced, show_custom, show_source, show_lambda = self._protocol_field_sets_for_type(self.protocol_type.currentText())
        lambda_mode = str(self.lambda_mode.currentText() or "scale")
        if show_lambda:
            if lambda_mode == "normal":
                advanced.discard("lambda_scale")
            else:
                advanced.discard("lambda_mean_hz")
                advanced.discard("lambda_std_hz")
        for key, field in self.protocol_fields.items():
            show = key in visible or key in advanced
            field.setVisible(show)
            label = self.protocol_field_labels.get(key)
            if label is not None:
                label.setVisible(show)
        self.lambda_mode.setVisible(show_lambda)
        self.lambda_mode_label.setVisible(show_lambda)
        self.source_box.setVisible(show_source)
        self.custom_points.setVisible(show_custom)
        self.custom_points_label.setVisible(show_custom)
        self._update_protocol_advanced_visibility()

    def _update_protocol_advanced_visibility(self) -> None:
        if not hasattr(self, "advanced_toggle"):
            return
        _visible, advanced, _show_custom, _show_source, _show_lambda = self._protocol_field_sets_for_type(self.protocol_type.currentText())
        if _show_lambda:
            lambda_mode = str(self.lambda_mode.currentText() or "scale")
            if lambda_mode == "normal":
                advanced.discard("lambda_scale")
            else:
                advanced.discard("lambda_mean_hz")
                advanced.discard("lambda_std_hz")
        has_advanced = bool(advanced)
        self.advanced_toggle.setVisible(has_advanced)
        self.advanced_toggle.setText("Hide advanced" if self.advanced_toggle.isChecked() and has_advanced else "Show advanced")
        self.advanced_box.setVisible(bool(has_advanced and self.advanced_toggle.isChecked()))

    def _preview_form_protocol(self) -> None:
        try:
            protocol = self._protocol_from_form()
        except Exception as exc:
            _show_error_message(self, "Invalid protocol", str(exc))
            return
        self._render_protocol_preview(protocol)
        self.tabs.setCurrentWidget(self.preview_tab)

    def _draw_preview(self) -> None:
        name = str(self.preview_combo.currentData() or self.preview_combo.currentText())
        protocol = next((item for item in self.protocols if item.name == name), None)
        if protocol is not None:
            self._render_protocol_preview(protocol)

    def _render_protocol_preview(self, protocol) -> None:
        series = self._preview_series_for_protocol(protocol)
        arrays = self._preview_raster_arrays_for_series(series)
        total_ms = self._preview_total_ms(protocol, arrays)
        view_key = (self._protocol_preview_signature(protocol), self._preview_source_path_for_protocol(protocol))
        if self._preview_view_key != view_key:
            self.preview_window_start_ms = 0.0
            self.preview_window_ms = min(5000.0, total_ms)
            self._preview_view_key = view_key
        self.preview_total_ms = total_ms
        self.preview_raster_protocol = protocol
        self.preview_raster_series = self._copy_preview_series(series)
        self.preview_raster_arrays = arrays
        self._clamp_preview_raster_window()
        self._draw_preview_raster_window()
        self._draw_preview_channel_map(protocol, series)

    def _draw_preview_raster_window(self) -> None:
        protocol = self.preview_raster_protocol
        series = self.preview_raster_series
        arrays = self.preview_raster_arrays
        figure = self.preview_canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        self.preview_raster_axis = ax
        start_ms = float(self.preview_window_start_ms)
        stop_ms = start_ms + float(self.preview_window_ms)
        offsets = list(range(len(series), 0, -1))
        visible_times = []
        display_sampled = False
        for _item, values in arrays:
            lo = int(np.searchsorted(values, start_ms, side="left"))
            hi = int(np.searchsorted(values, stop_ms, side="right"))
            window_values = values[lo:hi]
            if window_values.size > 5000:
                indices = np.linspace(0, window_values.size - 1, 5000, dtype=int)
                window_values = window_values[indices]
                display_sampled = True
            visible_times.append(window_values)
        has_events = any(np.asarray(item_times).size for item_times in visible_times)
        if has_events:
            ax.eventplot(visible_times, lineoffsets=offsets, linelengths=0.72, linewidths=1.5, colors="#0f766e")
        else:
            has_any_event = any(values.size for _item, values in arrays)
            message = "No preview events in this window" if has_any_event else (
                "No preview events\nSelect a pipeline spontaneous source or increase poisson rate/duration"
            )
            ax.text(
                0.5,
                0.5,
                message,
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#64748b",
            )
        protocol_name = getattr(protocol, "name", "protocol")
        sampled_text = " (sampled)" if display_sampled else ""
        ax.set_title(f"{protocol_name} raster preview{sampled_text}")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Stim channel")
        axis_pad_ms = min(50.0, max(1.0, float(self.preview_window_ms) * 0.01))
        axis_left = start_ms - axis_pad_ms if start_ms <= 0.0 else start_ms
        axis_right = stop_ms + axis_pad_ms if stop_ms >= float(self.preview_total_ms) else stop_ms
        ax.set_xlim(axis_left, axis_right)
        ax.set_yticks(offsets)
        ax.set_yticklabels([item["label"] for item in series])
        ax.grid(True, axis="x", color="#d7deea", linewidth=0.8)
        ax.set_ylim(0.5, max(len(series), 1) + 0.5)
        self._update_preview_window_label()
        self.preview_canvas.draw_idle()

    @staticmethod
    def _preview_raster_arrays_for_series(series: list[dict]) -> list[tuple[dict, np.ndarray]]:
        arrays = []
        for item in series:
            values = np.asarray(item.get("times_ms", []), dtype=float)
            values = values[np.isfinite(values)]
            values.sort()
            arrays.append((dict(item), values))
        return arrays

    def _preview_total_ms(self, protocol, arrays: list[tuple[dict, np.ndarray]]) -> float:
        values_max = max((float(values[-1]) for _item, values in arrays if values.size), default=0.0)
        protocol_type = str(getattr(protocol, "type", ""))
        if protocol_type == "poisson_random_electrodes":
            configured = max(100.0, float(getattr(protocol, "poisson_duration_s", 0.0)) * 1000.0)
        else:
            starts = [float(time_ms) for time_ms, _amp in stimulus_builder.pulse_starts_ms(protocol)]
            configured = max(starts, default=0.0) + max(1000.0, float(getattr(protocol, "pulse_width_us", 0.0)) / 1000.0)
        return max(100.0, values_max, configured, 5000.0)

    def _preview_generation_limit_ms(self, protocol) -> float:
        protocol_type = str(getattr(protocol, "type", ""))
        if protocol_type == "poisson_random_electrodes":
            return max(5000.0, float(getattr(protocol, "poisson_duration_s", 0.0)) * 1000.0)
        starts = [float(time_ms) for time_ms, _amp in stimulus_builder.pulse_starts_ms(protocol)]
        return max(5000.0, max(starts, default=0.0) + 1000.0)

    def _clamp_preview_raster_window(self) -> None:
        total = max(1.0, float(self.preview_total_ms))
        self.preview_window_ms = min(max(10.0, float(self.preview_window_ms)), total)
        max_start = max(0.0, total - self.preview_window_ms)
        self.preview_window_start_ms = min(max(0.0, float(self.preview_window_start_ms)), max_start)

    def _update_preview_window_label(self) -> None:
        start_s = float(self.preview_window_start_ms) / 1000.0
        stop_s = (float(self.preview_window_start_ms) + float(self.preview_window_ms)) / 1000.0
        total_s = float(self.preview_total_ms) / 1000.0
        self.preview_window_label.setText(f"{start_s:.3f} - {stop_s:.3f} s / {total_s:.3f} s")

    def _reset_preview_raster_view(self) -> None:
        self.preview_window_start_ms = 0.0
        self.preview_window_ms = min(5000.0, max(1.0, float(self.preview_total_ms)))
        self._clamp_preview_raster_window()
        self._draw_preview_raster_window()

    def _preview_raster_scrolled(self, event) -> None:
        if event is None or event.inaxes is not getattr(self, "preview_raster_axis", None):
            return
        if not self.preview_raster_arrays:
            return
        old_window = max(10.0, float(self.preview_window_ms))
        xdata = float(event.xdata) if event.xdata is not None and np.isfinite(event.xdata) else self.preview_window_start_ms + old_window * 0.5
        fraction = (xdata - self.preview_window_start_ms) / old_window
        fraction = min(1.0, max(0.0, fraction))
        direction = getattr(event, "step", 0)
        if direction == 0:
            direction = 1 if getattr(event, "button", "") == "up" else -1
        factor = 0.82 if direction > 0 else 1.22
        self.preview_window_ms = min(max(10.0, old_window * factor), max(10.0, float(self.preview_total_ms)))
        self.preview_window_start_ms = xdata - fraction * self.preview_window_ms
        self._clamp_preview_raster_window()
        self._draw_preview_raster_window()

    def _preview_raster_mouse_pressed(self, event) -> None:
        if event is None or event.inaxes is not getattr(self, "preview_raster_axis", None):
            return
        if getattr(event, "button", None) != 1:
            return
        self._preview_drag = {
            "x": float(event.x or 0.0),
            "start_ms": float(self.preview_window_start_ms),
            "moved": False,
        }

    def _preview_raster_mouse_moved(self, event) -> None:
        if not self._preview_drag or event is None:
            return
        ax = getattr(self, "preview_raster_axis", None)
        if ax is None:
            return
        width_px = max(1.0, float(ax.bbox.width))
        dx_px = float(event.x or 0.0) - float(self._preview_drag["x"])
        if abs(dx_px) > 2:
            self._preview_drag["moved"] = True
        self.preview_window_start_ms = float(self._preview_drag["start_ms"]) - dx_px / width_px * float(self.preview_window_ms)
        self._clamp_preview_raster_window()
        self._draw_preview_raster_window()

    def _preview_raster_mouse_released(self, event) -> None:
        self._preview_drag = None

    def _record_cache_key(self, record: dict | None) -> tuple:
        if not isinstance(record, dict):
            return ("missing",)
        data = record.get("raw_data")
        spike_count = 0
        channel_count = 0
        if isinstance(data, UnifiedMEAData):
            channel_count = len(data.spikes)
            spike_count = int(sum(np.asarray(times).size for times in data.spikes.values()))
        return (str(record.get("path", "")), id(data), channel_count, spike_count)

    def _preview_source_path_for_protocol(self, protocol) -> str:
        source_path = self.protocol_source_paths.get(getattr(protocol, "name", ""), "")
        form_name = self.protocol_fields.get("name").text().strip() if hasattr(self, "protocol_fields") and "name" in self.protocol_fields else ""
        if form_name == getattr(protocol, "name", "") and hasattr(self, "source_combo"):
            source_path = str(self.source_combo.currentData() or source_path)
        return str(source_path or "")

    def _protocol_preview_signature(self, protocol) -> tuple:
        custom_points = tuple(
            tuple(sorted((str(key), repr(value)) for key, value in dict(point).items()))
            for point in getattr(protocol, "custom_points", []) or []
        )
        fields = (
            "name",
            "type",
            "amplitude_mv",
            "pulse_width_us",
            "inter_phase_interval_us",
            "pulse_frequency_hz",
            "pulses_per_burst",
            "interpulse_interval_ms",
            "burst_count",
            "burst_interval_ms",
            "start_ms",
            "channel",
            "region_count",
            "max_candidate_electrodes",
            "poisson_duration_s",
            "lambda_mode",
            "lambda_scale",
            "lambda_floor_hz",
            "lambda_mean_hz",
            "lambda_std_hz",
            "random_seed",
            "spontaneous_data_path",
        )
        return tuple((field, repr(getattr(protocol, field, None))) for field in fields) + (("custom_points", custom_points),)

    @staticmethod
    def _copy_preview_series(series: list[dict]) -> list[dict]:
        copied = []
        for item in series:
            next_item = dict(item)
            next_item["times_ms"] = list(item.get("times_ms", []))
            copied.append(next_item)
        return copied

    def _preview_series_for_protocol(self, protocol) -> list[dict]:
        rates = None
        source_path = self._preview_source_path_for_protocol(protocol)
        source_record = self._record_by_path(source_path)
        cache_key = (
            "series",
            self._protocol_preview_signature(protocol),
            source_path,
            self._record_cache_key(source_record),
        )
        cached = self._preview_series_cache.get(cache_key)
        if cached is not None:
            return self._copy_preview_series(cached)
        if getattr(protocol, "type", "") == "poisson_random_electrodes":
            if isinstance(source_record, dict):
                electrodes, rate_values = self._rate_table_from_record(source_record)
                rates = {int(electrode): float(rate) for electrode, rate in zip(electrodes, rate_values)}
        series = stimulus_builder.preview_raster_series(
            protocol,
            preview_limit_ms=self._preview_generation_limit_ms(protocol),
            spontaneous_rates=rates,
        )
        if len(self._preview_series_cache) > 32:
            self._preview_series_cache.clear()
        self._preview_series_cache[cache_key] = self._copy_preview_series(series)
        return self._copy_preview_series(series)

    def _draw_preview_channel_map(self, protocol, series: list[dict]) -> None:
        figure = self.preview_map_canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        stimulation = self._stimulus_electrodes_for_preview(protocol, series)
        recording, metrics = self._recording_electrode_metrics_for_preview(protocol)
        selected = getattr(self, "preview_map_selected", None)
        state = draw_maxwell_channel_map(
            ax,
            self.channel_map,
            recording_electrodes=recording,
            stimulation_electrodes=stimulation,
            electrode_metrics=metrics,
            selected_electrode=None,
            title="Stim sites",
        )
        self.preview_map_selection_artist = None
        self.preview_map_state = state
        self.preview_map_protocol = protocol
        self.preview_map_series = list(series)

        missing = [
            electrode
            for electrode in stimulation
            if _resolve_channel_map_electrode(electrode, state.get("lookup", {}), state.get("positions", {})) is None
        ]
        if missing and state.get("positions"):
            ax.text(
                0.02,
                0.02,
                f"{len(missing)} unmapped",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="#b91c1c",
            )
        selected_electrode = self._update_preview_map_selection_artist(selected, draw=False)
        if selected_electrode:
            self.preview_map_detail.setText(self._preview_map_selection_text(str(selected_electrode)))
        elif stimulation:
            self.preview_map_detail.setText(f"Stim electrodes: {len(stimulation)} | Recording electrodes: {len(recording)}")
        else:
            self.preview_map_detail.setText(f"Recording electrodes: {len(recording)}")
        self.preview_map_canvas.draw_idle()

    def _update_preview_map_selection_artist(self, electrode, *, draw: bool = True) -> str | None:
        artist = getattr(self, "preview_map_selection_artist", None)
        if artist is not None:
            try:
                artist.remove()
            except ValueError:
                pass
            self.preview_map_selection_artist = None
        state = getattr(self, "preview_map_state", {}) or {}
        positions = state.get("positions", {})
        lookup = state.get("lookup", {})
        selected = _resolve_channel_map_electrode(electrode, lookup, positions)
        state["selected"] = selected if selected in positions else None
        if state["selected"] is None:
            if draw:
                self.preview_map_canvas.draw_idle()
            return None
        figure = self.preview_map_canvas.figure
        if not figure.axes:
            return None
        ax = figure.axes[0]
        x, y, _payload = positions[state["selected"]]
        self.preview_map_selection_artist = ax.scatter(
            [float(x)],
            [float(y)],
            s=92,
            facecolors="none",
            edgecolors="#111827",
            marker="o",
            linewidths=1.8,
            zorder=8,
        )
        if draw:
            self.preview_map_canvas.draw_idle()
        return str(state["selected"])

    def _stimulus_electrodes_for_preview(self, protocol, series: list[dict]) -> list[int]:
        protocol_type = str(getattr(protocol, "type", ""))
        electrodes: list[int] = []
        if protocol_type == "poisson_random_electrodes":
            for item in series:
                try:
                    electrodes.append(int(item.get("channel")))
                except (TypeError, ValueError):
                    continue
        elif protocol_type == "custom_sequence":
            for item in series:
                try:
                    electrodes.append(int(item.get("channel")))
                except (TypeError, ValueError):
                    continue
        else:
            group_names = [block.electrode_group for block in self.blocks if block.protocol == getattr(protocol, "name", "")]
            if not group_names and self.groups:
                group_names = [self.groups[0].name]
            group_lookup = {group.name: group for group in self.groups}
            for group_name in group_names:
                group = group_lookup.get(group_name)
                if group is not None:
                    electrodes.extend(int(value) for value in group.electrodes)
            if not electrodes:
                try:
                    electrodes.append(int(getattr(protocol, "channel", 0)))
                except (TypeError, ValueError):
                    pass
        seen = set()
        unique = []
        for electrode in electrodes:
            if electrode not in seen:
                unique.append(electrode)
                seen.add(electrode)
        return unique

    def _recording_electrode_metrics_for_preview(self, protocol) -> tuple[list[int], dict[int, dict]]:
        source_path = self._preview_source_path_for_protocol(protocol)
        records = []
        source_record = self._record_by_path(source_path)
        if isinstance(source_record, dict):
            records = [source_record]
        else:
            records = [record for record in self.records if isinstance(record.get("raw_data"), UnifiedMEAData)]
        cache_key = (
            "recording_metrics",
            source_path,
            id(self.channel_map),
            tuple(self._record_cache_key(record) for record in records),
        )
        cached = self._preview_recording_metrics_cache.get(cache_key)
        if cached is not None:
            recording, metrics = cached
            return list(recording), {int(key): dict(value) for key, value in metrics.items()}
        metrics: dict[int, dict] = {}
        for record in records:
            data = record.get("raw_data")
            if not isinstance(data, UnifiedMEAData):
                continue
            all_times = [np.asarray(times, dtype=float) for times in data.spikes.values() if np.asarray(times, dtype=float).size]
            duration_s = 1.0
            if all_times:
                finite_chunks = [values[np.isfinite(values)] for values in all_times if np.any(np.isfinite(values))]
                if finite_chunks:
                    finite = np.concatenate(finite_chunks)
                    if finite.size:
                        duration_s = max(float(np.nanmax(finite) - min(0.0, float(np.nanmin(finite)))), 1e-9)
            for index, (channel, times) in enumerate(_spike_series_from_unified(data), start=1):
                values = np.asarray(times, dtype=float)
                spike_count = int(np.count_nonzero(np.isfinite(values)))
                electrode = int(self._electrode_for_channel(channel, index))
                entry = metrics.setdefault(
                    electrode,
                    {
                        "electrode": electrode,
                        "channel": str(channel),
                        "spike_count": 0,
                        "firing_rate_hz": 0.0,
                        "source": Path(str(record.get("path", ""))).name,
                    },
                )
                entry["spike_count"] = int(entry.get("spike_count", 0)) + spike_count
                entry["firing_rate_hz"] = float(entry.get("firing_rate_hz", 0.0)) + spike_count / duration_s
        result = (sorted(metrics), metrics)
        if len(self._preview_recording_metrics_cache) > 16:
            self._preview_recording_metrics_cache.clear()
        self._preview_recording_metrics_cache[cache_key] = (list(result[0]), {int(key): dict(value) for key, value in metrics.items()})
        return result

    def _preview_map_clicked(self, event) -> None:
        if event is None or event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        state = getattr(self, "preview_map_state", {}) or {}
        points = list(state.get("point_lookup", []))
        if not points:
            return
        x = float(event.xdata)
        y = float(event.ydata)
        nearest = min(points, key=lambda item: (float(item["x"]) - x) ** 2 + (float(item["y"]) - y) ** 2)
        distance = ((float(nearest["x"]) - x) ** 2 + (float(nearest["y"]) - y) ** 2) ** 0.5
        if distance > 0.035:
            return
        self.preview_map_selected = str(nearest["electrode"])
        selected = self._update_preview_map_selection_artist(self.preview_map_selected)
        if selected:
            self.preview_map_detail.setText(self._preview_map_selection_text(selected))

    def _preview_map_selection_text(self, electrode: str) -> str:
        state = getattr(self, "preview_map_state", {}) or {}
        positions = state.get("positions", {})
        payload = {}
        if electrode in positions:
            payload = positions[electrode][2] if isinstance(positions[electrode][2], dict) else {}
        metrics = state.get("metrics", {}).get(electrode, {})
        channel = str(metrics.get("channel") or payload.get("channel") or "").strip() or "n/a"
        flags = []
        if electrode in state.get("recording", set()):
            flags.append("recording")
        if electrode in state.get("stimulation", set()):
            flags.append("stimulation")
        detail = [
            f"Electrode: {electrode}",
            f"Channel: {channel}",
            f"Role: {', '.join(flags) if flags else 'background'}",
        ]
        if metrics:
            detail.append(f"Firing rate: {float(metrics.get('firing_rate_hz', 0.0)):.3g} Hz")
            detail.append(f"Spikes: {int(metrics.get('spike_count', 0))}")
            source = str(metrics.get("source", "")).strip()
            if source:
                detail.append(f"Source: {source}")
        return " | ".join(detail)

    def _map_position_for_electrode(self, electrode: int):
        lookup, positions = _channel_map_positions(self.channel_map)
        candidates = [str(electrode), f"e{electrode}", f"chan{electrode}"]
        for candidate in candidates:
            pos = _position_for_channel(candidate, lookup)
            if pos is not None:
                return float(pos[0]), float(pos[1])
            direct = positions.get(candidate)
            if direct is not None:
                return float(direct[0]), float(direct[1])
        return None

    def _sync_info(self):
        values = {key: field.text().strip() for key, field in self.info_fields.items()}
        values["event_threshold"] = float(values["event_threshold"] or 8.5)
        values["amplifier_gain"] = int(values["amplifier_gain"] or 512)
        values["spike_step"] = int(values["spike_step"] or 10000)
        values["max_stims"] = int(values["max_stims"] or 10)
        for key, text in self.info_texts.items():
            values[key] = text.toPlainText().strip()
        return stimulus_builder.ExperimentInfo(**values)

    def _generate(self) -> None:
        try:
            info = self._sync_info()
            output = Path(self.output_path.text()).expanduser().resolve()
            protocols = copy.deepcopy(self.protocols)
            self._prepare_pipeline_rate_sources(output, protocols)
            result = stimulus_builder.build_package(output, info, self.groups, protocols, self.blocks)
        except Exception as exc:
            self.generate_status.setText("Generation failed")
            _show_error_message(self, "Generation failed", str(exc))
            return
        self.generate_status.setText(f"Generated: {result}")
        _show_info_message(self, "Stimulus Generation", f"Code package generated:\n{result}")

    def _prepare_pipeline_rate_sources(self, output: Path, protocols: list) -> None:
        rate_dir = output / "config" / "pipeline_rate_sources"
        for protocol in protocols:
            if protocol.type != "poisson_random_electrodes":
                continue
            source_path = self.protocol_source_paths.get(protocol.name, "")
            if not source_path:
                continue
            record = self._record_by_path(source_path)
            electrodes, rates = self._rate_table_from_record(record)
            rate_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", protocol.name).strip("_") or "protocol"
            rate_path = rate_dir / f"{safe_name}_rates.npz"
            np.savez(
                rate_path,
                electrodes=np.asarray(electrodes, dtype=np.int32),
                rates_hz=np.asarray(rates, dtype=float),
                firing_rate_hz=np.asarray(rates, dtype=float),
                source_path=str(source_path),
            )
            protocol.spontaneous_data_path = str(rate_path)

    def _rate_table_from_record(self, record: dict | None) -> tuple[list[int], list[float]]:
        if not isinstance(record, dict):
            raise ValueError("Selected spontaneous source is no longer in the pipeline database")
        cache_key = ("rates", id(self.channel_map), self._record_cache_key(record))
        cached = self._preview_rate_cache.get(cache_key)
        if cached is not None:
            return list(cached[0]), list(cached[1])
        data = record.get("raw_data")
        if not isinstance(data, UnifiedMEAData):
            raise ValueError("Spontaneous source must be a loaded spike-event dataset")
        rows = []
        all_times = [np.asarray(times, dtype=float) for times in data.spikes.values() if np.asarray(times, dtype=float).size]
        duration_s = 1.0
        if all_times:
            finite = np.concatenate([values[np.isfinite(values)] for values in all_times if np.any(np.isfinite(values))])
            if finite.size:
                duration_s = max(float(np.nanmax(finite) - min(0.0, float(np.nanmin(finite)))), 1e-9)
        for index, (channel, times) in enumerate(_spike_series_from_unified(data), start=1):
            values = np.asarray(times, dtype=float)
            electrode = self._electrode_for_channel(channel, index)
            rows.append((electrode, float(np.count_nonzero(np.isfinite(values))) / duration_s))
        if not rows:
            raise ValueError("Selected spontaneous source contains no spike trains")
        merged: dict[int, float] = {}
        for electrode, rate in rows:
            merged[int(electrode)] = merged.get(int(electrode), 0.0) + float(rate)
        ordered = sorted(merged.items())
        result = ([item[0] for item in ordered], [item[1] for item in ordered])
        if len(self._preview_rate_cache) > 16:
            self._preview_rate_cache.clear()
        self._preview_rate_cache[cache_key] = (list(result[0]), list(result[1]))
        return result

    def _channel_to_electrode_lookup(self) -> dict[str, int]:
        key = id(self.channel_map)
        if self._preview_electrode_lookup_key == key:
            return self._preview_electrode_lookup
        lookup: dict[str, int] = {}
        if isinstance(self.channel_map, ChannelMap):
            for electrode_key, payload in self.channel_map.electrodes.items():
                if not isinstance(payload, dict):
                    continue
                parsed = self._parse_electrode_int(payload.get("electrode"))
                if parsed is None:
                    parsed = self._parse_electrode_int(electrode_key)
                if parsed is None:
                    continue
                candidates = [electrode_key, payload.get("channel"), payload.get("source_channel")]
                aliases = payload.get("aliases", [])
                if isinstance(aliases, (list, tuple)):
                    candidates.extend(aliases)
                for candidate in candidates:
                    text = str(candidate).strip()
                    if not text:
                        continue
                    lookup.setdefault(text, parsed)
                    lookup.setdefault(_base_channel_from_raster_label(text), parsed)
                    lookup.setdefault(normalize_channel_name(text), parsed)
        self._preview_electrode_lookup_key = key
        self._preview_electrode_lookup = lookup
        return lookup

    def _electrode_for_channel(self, channel: str, fallback: int) -> int:
        base = _base_channel_from_raster_label(channel)
        lookup = self._channel_to_electrode_lookup()
        for candidate in (str(channel), base, normalize_channel_name(channel), normalize_channel_name(base)):
            if candidate in lookup:
                return int(lookup[candidate])
        parsed = self._parse_electrode_int(base)
        return parsed if parsed is not None else fallback

    def _default_electrodes(self) -> list[int]:
        electrodes = []
        if isinstance(self.channel_map, ChannelMap):
            for electrode_key, payload in self.channel_map.electrodes.items():
                if not isinstance(payload, dict):
                    continue
                if not payload.get("routed") and not payload.get("channel"):
                    continue
                parsed = self._parse_electrode_int(payload.get("electrode"))
                if parsed is None:
                    parsed = self._parse_electrode_int(electrode_key)
                if parsed is not None:
                    electrodes.append(parsed)
                if len(electrodes) >= 32:
                    break
        return electrodes or [7317]

    def _record_by_path(self, path: str) -> dict | None:
        for record in self.records:
            if str(record.get("path", "")) == str(path):
                return record
        return None

    def _record_label(self, record: dict) -> str:
        path = Path(str(record.get("path", "")))
        channels, spikes, _waveforms = _loaded_data_stats(record.get("raw_data"))
        activity = _loaded_data_activity_label(path, record.get("raw_data"))
        return f"{path.name} | {activity} | {channels} ch | {spikes} spikes"

    def _source_label(self, path: str) -> str:
        if not path:
            return "None"
        record = self._record_by_path(path)
        return self._record_label(record) if record else Path(path).name

    def _browse_cfg(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Select MaxWell cfg", "", "CFG files (*.cfg);;All files (*.*)")
        if path:
            self.info_fields["cfg_path"].setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory", self.output_path.text() or ".")
        if path:
            self.output_path.setText(path)

    def _set_combo_data(self, combo: QComboBox, value) -> None:
        for index in range(combo.count()):
            if str(combo.itemData(index)) == str(value):
                combo.setCurrentIndex(index)
                return
        if combo.count():
            combo.setCurrentIndex(0)

    def _set_combo_text(self, combo: QComboBox, value: str) -> None:
        index = combo.findText(str(value))
        if index >= 0:
            combo.setCurrentIndex(index)

    def _parse_electrode_int(self, value) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            match = re.search(r"(\d+)", str(value))
            return int(match.group(1)) if match else None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        _enable_standard_window_controls(self)
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
        self.file_database: list[dict] = []
        self.processed_database: list[dict] = []
        self.database_sort_column = None
        self.processed_sort_column = None
        self.pipeline_progress = None
        self.active_load_worker = None
        self.active_maxwell_waveform_worker = None
        self.maxwell_waveform_progress = None
        self.active_stimulus_worker = None
        self.active_multi_file_fa_worker = None
        self.stimulus_response_dialog = None
        self.stimulus_response_payload = None
        self.multi_file_fa_dialog = None
        self.multi_file_fa_payload = None
        self.generic_analysis_dialog = None
        self.generic_analysis_payload = None
        self.stimulus_generation_dialog = None
        self.active_processed_index = -1

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

        self.file_label = QLabel("No data files loaded")
        self.file_label.setWordWrap(True)
        self.file_label.setObjectName("MutedText")
        side_layout.addWidget(self.file_label)

        self.app_status_label = QLabel("Idle")
        self.app_status_label.setWordWrap(True)
        self.app_status_label.setObjectName("MutedText")
        side_layout.addWidget(self.app_status_label)

        self.open_button = QPushButton("Open Data Files")
        self.open_button.clicked.connect(self.open_data)
        self.preview_button = QPushButton("Raw Data Raster")
        self.preview_button.clicked.connect(self.preview_raw)
        self.preview_button.setEnabled(False)
        self.save_spike_train_button = QPushButton("Save File")
        self.save_spike_train_button.clicked.connect(self.save_spike_train)
        self.save_spike_train_button.setEnabled(False)
        self.channel_map_button = QPushButton("Channel Map")
        self.channel_map_button.clicked.connect(self.open_channel_map)
        self.sorting_button = QPushButton("Sorting")
        self.sorting_button.clicked.connect(self.open_sorting)
        self.stimulus_response_button = QPushButton("Stimulus Response Analysis")
        self.stimulus_response_button.clicked.connect(self.open_stimulus_response_analysis)
        self.multi_file_fa_button = QPushButton("Dynamics Analysis")
        self.multi_file_fa_button.clicked.connect(self.open_multi_file_factor_analysis)
        self.generic_analysis_button = QPushButton("Custom Analysis")
        self.generic_analysis_button.clicked.connect(self.open_generic_analysis)

        library_label = QLabel("Data Library")
        library_label.setObjectName("Header")
        side_layout.addWidget(library_label)
        for button in [
            self.open_button,
            self.preview_button,
            self.save_spike_train_button,
        ]:
            button.setMinimumHeight(40)
            side_layout.addWidget(button)

        analysis_label = QLabel("Analysis")
        analysis_label.setObjectName("Header")
        side_layout.addWidget(analysis_label)
        for button in [
            self.channel_map_button,
            self.sorting_button,
            self.stimulus_response_button,
            self.multi_file_fa_button,
            self.generic_analysis_button,
        ]:
            button.setMinimumHeight(40)
            side_layout.addWidget(button)

        side_layout.addStretch()

        content = QFrame()
        content.setObjectName("Content")
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(16)

        header = QLabel("Loaded file database")
        header.setObjectName("Header")
        header.setFont(QFont("Segoe UI", 20, QFont.Bold))
        content_layout.addWidget(header)

        self.database_table = QTableWidget(0, 7)
        self.database_table.setHorizontalHeaderLabels(["File", "Kind", "Label", "Channels", "Spikes", "Waveforms", "Folder"])
        self.database_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.database_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.database_table.setMinimumHeight(150)
        self.database_table.itemSelectionChanged.connect(self._database_selection_changed)
        table_header = self.database_table.horizontalHeader()
        table_header.setSectionsClickable(True)
        table_header.setSortIndicatorShown(False)
        table_header.sectionClicked.connect(self._database_header_clicked)
        content_layout.addWidget(self.database_table, 1)

        processed_header = QLabel("Processed data database")
        processed_header.setObjectName("Header")
        content_layout.addWidget(processed_header)

        self.processed_table = QTableWidget(0, 6)
        self.processed_table.setHorizontalHeaderLabels(["Name", "Source", "Type", "Samples", "Features", "Origin"])
        self.processed_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.processed_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.processed_table.setMinimumHeight(140)
        self.processed_table.itemSelectionChanged.connect(self._processed_selection_changed)
        processed_header_widget = self.processed_table.horizontalHeader()
        processed_header_widget.setSectionsClickable(True)
        processed_header_widget.setSortIndicatorShown(False)
        processed_header_widget.sectionClicked.connect(self._processed_header_clicked)
        content_layout.addWidget(self.processed_table, 1)

        self.data_preview = QTextEdit()
        self.data_preview.setReadOnly(True)
        self.data_preview.setMinimumHeight(220)
        self.data_preview.setPlaceholderText("Open data files to build a database, then select a raw or processed row to show a summary.")
        content_layout.addWidget(self.data_preview, 2)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Activity log")
        self.log.setMinimumHeight(150)
        content_layout.addWidget(self.log, 1)
        self._update_data_preview()
        self._set_app_status("Idle", "Open data files to build a database. Processed datasets are cached automatically as you preview and analyze data.")

        layout.addWidget(sidebar)
        layout.addWidget(content, 1)
        self.setCentralWidget(root)

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("File")
        open_action = QAction("Open Data Files", self)
        open_action.triggered.connect(self.open_data)
        file_menu.addAction(open_action)
        self.save_spike_train_action = QAction("Save File", self)
        self.save_spike_train_action.triggered.connect(self.save_spike_train)
        self.save_spike_train_action.setEnabled(False)
        file_menu.addAction(self.save_spike_train_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        tools_menu = self.menuBar().addMenu("Tools")
        channel_map_action = QAction("Channel Map", self)
        channel_map_action.triggered.connect(self.open_channel_map)
        tools_menu.addAction(channel_map_action)
        tools_menu.addSeparator()
        sorting_action = QAction("Sorting", self)
        sorting_action.triggered.connect(self.open_sorting)
        tools_menu.addAction(sorting_action)
        tools_menu.addSeparator()
        stimulus_action = QAction("Stimulus Response Analysis", self)
        stimulus_action.triggered.connect(self.open_stimulus_response_analysis)
        tools_menu.addAction(stimulus_action)
        stimulus_generation_action = QAction("Stimulus Generation", self)
        stimulus_generation_action.triggered.connect(self.open_stimulus_generation)
        tools_menu.addAction(stimulus_generation_action)
        multi_file_fa_action = QAction("Dynamics Analysis", self)
        multi_file_fa_action.triggered.connect(self.open_multi_file_factor_analysis)
        tools_menu.addAction(multi_file_fa_action)
        generic_action = QAction("Custom Analysis", self)
        generic_action.triggered.connect(self.open_generic_analysis)
        tools_menu.addAction(generic_action)

    def _start_progress(self, title: str, message: str, maximum: int = 0) -> QProgressDialog:
        return _create_progress_dialog(self, title, message, maximum)

    def _progress_step(self, dialog: QProgressDialog | None, message: str, value: int | None = None) -> None:
        _set_progress_dialog(dialog, message, value)

    def _finish_progress(self, dialog: QProgressDialog | None) -> None:
        _close_progress_dialog(dialog)

    def _set_app_status(self, title: str, detail: str | None = None) -> None:
        message = str(title).strip() or "Idle"
        if detail:
            message = f"{message}\n{str(detail).strip()}"
        if hasattr(self, "app_status_label"):
            self.app_status_label.setText(message)
            self.app_status_label.setToolTip(str(detail or title))

    def _update_analysis_action_states(self) -> None:
        has_data = self.raw_data is not None
        has_database = bool(self.file_database)
        has_processed = bool(self.processed_database)
        loading = self.active_load_worker is not None
        stimulus_busy = self.active_stimulus_worker is not None
        dynamics_busy = self.active_multi_file_fa_worker is not None
        waveform_busy = self.active_maxwell_waveform_worker is not None
        any_busy = loading or stimulus_busy or dynamics_busy or waveform_busy

        self.open_button.setEnabled(not any_busy)
        self.preview_button.setEnabled(has_data and not any_busy)
        self.save_spike_train_button.setEnabled(has_data and not any_busy)
        self.save_spike_train_action.setEnabled(has_data and not any_busy)
        self.channel_map_button.setEnabled(not any_busy)
        self.sorting_button.setEnabled(has_data and not any_busy)
        self.stimulus_response_button.setEnabled(has_database and not (loading or stimulus_busy or dynamics_busy))
        self.multi_file_fa_button.setEnabled(has_database and not (loading or stimulus_busy or dynamics_busy))
        self.generic_analysis_button.setEnabled(has_database and not any_busy)

    def open_stimulus_generation(self):
        if self.stimulus_generation_dialog is None:
            self.stimulus_generation_dialog = StimulusGenerationDialog(
                self.file_database,
                self,
                channel_map=self.channel_map,
            )
            self.stimulus_generation_dialog.setWindowModality(Qt.WindowModality.NonModal)
        else:
            self.stimulus_generation_dialog.refresh_pipeline_context(self.file_database, self.channel_map)
        self.stimulus_generation_dialog.show()
        self.stimulus_generation_dialog.raise_()
        self.stimulus_generation_dialog.activateWindow()

    def open_stimulus_response_analysis(self):
        if self.active_stimulus_worker is not None:
            _show_info_message(self, "Stimulus Response", "Stimulus response analysis is already running.")
            return
        if not self.file_database:
            _show_info_message(self, "Stimulus Response", "Load files into the database before stimulus response analysis.")
            return
        dialog = self._stimulus_response_analysis_dialog()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _stimulus_response_analysis_dialog(self):
        if self.stimulus_response_dialog is None:
            dialog = StimulusDatabaseAnalysisDialog(self.file_database, self)
            dialog.setWindowModality(Qt.WindowModality.NonModal)
            dialog.accepted.connect(self._start_stimulus_response_from_dialog)
            dialog.psth_button.clicked.connect(self._open_cached_stimulus_psth)
            dialog.activation_curve_button.clicked.connect(self._open_cached_stimulus_activation_curve)
            self.stimulus_response_dialog = dialog
        elif hasattr(self.stimulus_response_dialog, "_set_records"):
            self.stimulus_response_dialog._set_records(self.file_database)
        return self.stimulus_response_dialog

    def _start_stimulus_response_from_dialog(self):
        if self.active_stimulus_worker is not None:
            _show_info_message(self, "Stimulus Response", "Stimulus response analysis is already running.")
            return
        dialog = self.stimulus_response_dialog or self._stimulus_response_analysis_dialog()
        paths, pre_ms, response_ms, artifact_ms = dialog.values()
        if not paths:
            _show_info_message(self, "Stimulus Response", "Select at least one database file.")
            dialog.show()
            return
        self.pipeline_progress = self._start_progress("Stimulus response", "Starting stimulus response analysis...", 100)
        self._set_app_status("Stimulus response running", "Preparing stimulus-aligned rasters and summary payloads.")
        worker = StimulusResponseWorker(paths, pre_ms=pre_ms, response_ms=response_ms, artifact_ms=artifact_ms)
        self.pipeline_progress.canceled.connect(worker.cancel)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(lambda payload, worker=worker: self._stimulus_response_finished(payload, worker))
        worker.signals.failed.connect(lambda details, worker=worker: self._stimulus_response_failed(details, worker))
        worker.signals.canceled.connect(lambda message, worker=worker: self._stimulus_response_canceled(message, worker))
        self.active_stimulus_worker = worker
        self._update_analysis_action_states()
        self.thread_pool.start(worker)

    def _stimulus_response_finished(self, payload: dict, worker: StimulusResponseWorker):
        if worker._is_cancelled():
            self._stimulus_response_canceled("Stimulus response analysis cancelled", worker)
            return
        if self.active_stimulus_worker is worker:
            self.active_stimulus_worker = None
        self._finish_progress(self.pipeline_progress)
        self.pipeline_progress = None
        self._update_analysis_action_states()
        records = list(payload.get("records", []))
        errors = list(payload.get("errors", []))
        if not records:
            details = "\n".join(errors[:12]) if errors else "No stimulus response files could be analyzed."
            _show_warning_message(self, "Stimulus Response", details)
            self._log("Stimulus response analysis produced no usable files")
            self._set_app_status("Stimulus response produced no usable files", details.splitlines()[0] if details else "")
            self._return_to_stimulus_analysis_dialog()
            return
        self.stimulus_response_payload = payload
        self._stimulus_response_analysis_dialog().set_cached_payload(payload)
        self._open_stimulus_raster_window(payload)
        self._set_app_status("Stimulus response ready", f"{len(records)} files analyzed, {len(errors)} skipped.")
        self._log(f"Stimulus response analysis: {len(records)} files, {len(errors)} skipped")

    def _stimulus_response_failed(self, details: str, worker: StimulusResponseWorker):
        if self.active_stimulus_worker is worker:
            self.active_stimulus_worker = None
        self._finish_progress(self.pipeline_progress)
        self.pipeline_progress = None
        self._update_analysis_action_states()
        self._log(details)
        self._set_app_status("Stimulus response failed", details.splitlines()[-1] if details else "Unknown error")
        _show_error_message(self, "Stimulus Response failed", details.splitlines()[-1] if details else "Unknown error")
        self._return_to_stimulus_analysis_dialog()

    def _stimulus_response_canceled(self, message: str, worker: StimulusResponseWorker):
        if self.active_stimulus_worker is worker:
            self.active_stimulus_worker = None
        self._finish_progress(self.pipeline_progress)
        self.pipeline_progress = None
        self._update_analysis_action_states()
        self._log(message or "Stimulus response analysis cancelled")
        self._set_app_status("Stimulus response cancelled", message or "Stimulus response analysis cancelled")
        self._return_to_stimulus_analysis_dialog()

    def _open_cached_stimulus_psth(self):
        if not self.stimulus_response_payload:
            _show_info_message(self, "Stimulus Response", "Run analysis before opening PSTH.")
            return
        self._open_stimulus_psth_window(self.stimulus_response_payload)

    def _open_cached_stimulus_activation_curve(self):
        if not self.stimulus_response_payload:
            _show_info_message(self, "Stimulus Response", "Run analysis before opening activation curve.")
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
        window.finished.connect(lambda _result: self._return_to_stimulus_analysis_dialog())
        self._show_child(window)
        _defer_hide(dialog, 0)

    def _return_to_stimulus_analysis_dialog(self):
        if self.stimulus_response_dialog is None:
            return
        self.stimulus_response_dialog.show()
        self.stimulus_response_dialog.raise_()
        self.stimulus_response_dialog.activateWindow()

    def open_generic_analysis(self):
        if not self.file_database:
            _show_info_message(self, "Custom Analysis", "Load files into the database before custom analysis.")
            return
        dialog = self._generic_analysis_dialog()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _generic_analysis_dialog(self):
        if self.generic_analysis_dialog is None:
            dialog = GenericAnalysisDialog(self.file_database, self)
            dialog.setWindowModality(Qt.WindowModality.NonModal)
            dialog.accepted.connect(self._start_generic_analysis_from_dialog)
            self.generic_analysis_dialog = dialog
        elif hasattr(self.generic_analysis_dialog, "_set_records"):
            self.generic_analysis_dialog._set_records(self.file_database)
        return self.generic_analysis_dialog

    def _start_generic_analysis_from_dialog(self):
        dialog = self._generic_analysis_dialog()
        paths, parameters = dialog.values()
        if not paths:
            _show_info_message(self, "Custom Analysis", "Select at least one database file.")
            dialog.show()
            return
        selected_lookup = set(paths)
        records = []
        errors = []
        processed_records = []
        for record in self.file_database:
            path_text = str(record.get("path", ""))
            if path_text not in selected_lookup:
                continue
            try:
                processed = self._build_custom_analysis_record(record, parameters)
                self._upsert_processed_record(processed)
                processed_records.append(processed)
                records.append(
                    {
                        "path": str(processed.get("path", "")),
                        "file": str(processed.get("name") or Path(path_text).name),
                        "condition": str(processed.get("source_label") or "Custom"),
                        "matrix_description": str(processed.get("description", "")),
                        "processed": processed,
                    }
                )
            except Exception as exc:
                errors.append(f"{Path(path_text).name}: {exc}")
        payload = {"records": records, "errors": errors, "parameters": dict(parameters or {})}
        if not records:
            details = "\n".join(errors[:12]) if errors else "No files could be analyzed."
            _show_warning_message(self, "Custom Analysis", details)
            dialog.show()
            return
        self._refresh_processed_database_table()
        if self.generic_analysis_dialog is not None and hasattr(self.generic_analysis_dialog, "_set_records"):
            self.generic_analysis_dialog._set_records(self.file_database)
        self._update_analysis_action_states()
        self._set_app_status("Custom analysis ready", f"{len(processed_records)} processed dataset(s) added.")
        self.generic_analysis_payload = payload
        self._open_generic_analysis_window(payload)

    def _open_generic_analysis_window(self, payload: dict):
        window = GenericAnalysisWindow(payload, self)
        dialog = self._generic_analysis_dialog()
        window.finished.connect(lambda _result: self._return_to_generic_analysis_dialog())
        self._show_child(window)
        _defer_hide(dialog, 0)

    def _build_custom_analysis_record(self, record: dict, parameters: dict) -> dict:
        data = (record or {}).get("raw_data")
        if not isinstance(data, UnifiedMEAData):
            raise ValueError("Custom spike-vector analysis requires loaded spike-event data")
        path_text = str((record or {}).get("path", ""))
        spike_series = _spike_series_from_unified(data)
        selected_channels = _custom_channel_filter(spike_series, str(parameters.get("channels", "")))
        if not selected_channels:
            raise ValueError("No selected channels matched this file")
        windows = _parse_custom_time_windows(str(parameters.get("time_windows", "full")), data)
        analysis_type = str(parameters.get("analysis_type", "firing_rate_vector"))
        matrix, sample_labels, feature_labels, description = _custom_spike_vector_matrix(selected_channels, windows, analysis_type)
        x_values = _parse_optional_float_list(str(parameters.get("x_values", "")))
        if x_values.size and x_values.size != len(sample_labels):
            raise ValueError("x values must match the number of selected time windows")
        plot_payload = {
            "kind": "auto_xy",
            "plot_mode": str(parameters.get("plot_mode", "auto")),
            "x": x_values.tolist() if x_values.size else list(range(1, len(sample_labels) + 1)),
            "x_labels": list(sample_labels),
            "y_label": str(parameters.get("y_label") or ("Firing rate (Hz)" if analysis_type == "firing_rate_vector" else "Spike count")),
            "x_label": str(parameters.get("x_label") or "Time window"),
            "series_labels": list(feature_labels),
        }
        safe_stamp = int(time.time() * 1000)
        dataset_type = str(analysis_type)
        display_name = str(parameters.get("display_name", "") or "").strip() or f"{Path(path_text).name} | {dataset_type}"
        return {
            "path": f"{path_text}::custom::{dataset_type}::{safe_stamp}",
            "source_path": path_text,
            "origin_name": Path(path_text).name,
            "name": display_name,
            "source_label": _loaded_data_activity_label(path_text, data),
            "dataset_type": dataset_type,
            "dataset_group": "custom",
            "dataset_origin": "custom_analysis",
            "commit": f"custom | {dataset_type}",
            "commit_detail": description,
            "view_mode": "custom_vector",
            "description": description,
            "matrix": matrix,
            "sample_labels": list(sample_labels),
            "feature_labels": list(feature_labels),
            "parameters": dict(parameters or {}),
            "plot_payload": plot_payload,
        }

    def _return_to_generic_analysis_dialog(self):
        if self.generic_analysis_dialog is None:
            return
        self.generic_analysis_dialog.show()
        self.generic_analysis_dialog.raise_()
        self.generic_analysis_dialog.activateWindow()

    def _build_processed_records(self, raw_records, selected_paths, parameters: dict | None) -> tuple[list[dict], list[str]]:
        selected_lookup = {str(path) for path in (selected_paths or []) if path}
        normalized_parameters = dict(parameters or {})
        view_mode = str(normalized_parameters.get("view_mode", "auto"))
        dataset_type = str(normalized_parameters.get("dataset_type", view_mode) or view_mode)
        dataset_group = str(normalized_parameters.get("dataset_group", "generic") or "generic")
        dataset_origin = str(normalized_parameters.get("origin", "manual") or "manual")
        display_name = str(normalized_parameters.get("display_name", "") or "").strip()
        bin_ms = float(normalized_parameters.get("bin_ms", 10.0))
        burst_window_ms = float(normalized_parameters.get("burst_window_ms", 300.0))
        burst_threshold_z = float(normalized_parameters.get("burst_threshold_z", 4.0))
        array_axis = str(normalized_parameters.get("array_axis", "rows"))
        commit_title, commit_detail = _processed_dataset_commit(normalized_parameters)

        processed_records: list[dict] = []
        errors: list[str] = []
        for record in list(raw_records or []):
            path_text = str(record.get("path", ""))
            if selected_lookup and path_text not in selected_lookup:
                continue
            try:
                matrix, labels, description = _generic_analysis_matrix_from_record(
                    record,
                    view_mode=view_mode,
                    bin_ms=bin_ms,
                    burst_window_ms=burst_window_ms,
                    burst_threshold_z=burst_threshold_z,
                    array_axis=array_axis,
                )
                matrix = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
                if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
                    raise ValueError("No usable matrix could be constructed")
                processed_records.append(
                    {
                        "path": f"{path_text}::{dataset_type}",
                        "source_path": path_text,
                        "origin_name": Path(path_text).name,
                        "name": display_name or _processed_dataset_label(record, view_mode=view_mode),
                        "source_label": _loaded_data_activity_label(path_text, record.get("raw_data")),
                        "dataset_type": dataset_type,
                        "dataset_group": dataset_group,
                        "dataset_origin": dataset_origin,
                        "commit": commit_title,
                        "commit_detail": commit_detail,
                        "view_mode": view_mode,
                        "description": description,
                        "matrix": matrix,
                        "sample_labels": list(labels),
                        "parameters": normalized_parameters,
                    }
                )
            except Exception as exc:
                errors.append(f"{Path(path_text).name}: {exc}")
        return processed_records, errors

    def _ensure_processed_data_for_records(
        self,
        raw_records,
        *,
        parameter_sets: list[dict] | None = None,
        refresh_table: bool = True,
    ) -> tuple[int, list[str]]:
        added = 0
        errors: list[str] = []
        for record in list(raw_records or []):
            if not isinstance(record, dict):
                continue
            path_text = str(record.get("path", ""))
            if not path_text:
                continue
            record_parameter_sets = list(parameter_sets or _processed_dataset_presets_for_record(record))
            for params in record_parameter_sets:
                params = dict(params or {})
                view_mode = str(params.get("view_mode", "auto"))
                dataset_type = str(params.get("dataset_type", view_mode) or view_mode)
                processed_path = f"{path_text}::{dataset_type}"
                if any(str(existing.get("path", "")) == processed_path for existing in self.processed_database):
                    continue
                built, build_errors = self._build_processed_records([record], [path_text], params)
                errors.extend(build_errors)
                for processed_record in built:
                    self._upsert_processed_record(processed_record)
                    added += 1
        if refresh_table and added:
            self._refresh_processed_database_table()
            if self.generic_analysis_dialog is not None and hasattr(self.generic_analysis_dialog, "_set_records"):
                self.generic_analysis_dialog._set_records(self.processed_database)
            self._update_analysis_action_states()
        return added, errors

    def _ensure_processed_data_for_selected_records(self) -> None:
        selected_records = self._selected_database_records()
        if not selected_records:
            return
        added, errors = self._ensure_processed_data_for_records(selected_records)
        if added:
            self._log(f"Auto-cached {added} processed dataset(s)")
            self._set_app_status("Processed data cached", f"{len(self.processed_database)} processed dataset(s) ready.")
        if errors:
            self._log(f"Processed-data cache warnings: {'; '.join(errors[:4])}")

    def open_multi_file_factor_analysis(self):
        if self.active_multi_file_fa_worker is not None:
            _show_info_message(self, "Dynamics Analysis", "Dynamics analysis is already running.")
            return
        if not self.file_database:
            _show_info_message(self, "Dynamics Analysis", "Load files into the database before dynamics analysis.")
            return
        dialog = self._multi_file_fa_analysis_dialog()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _multi_file_fa_analysis_dialog(self):
        if self.multi_file_fa_dialog is None:
            dialog = FactorAnalysisDatabaseDialog(self.file_database, self)
            dialog.setWindowModality(Qt.WindowModality.NonModal)
            dialog.accepted.connect(self._start_multi_file_fa_from_dialog)
            dialog.open_result_button.clicked.connect(self._open_cached_multi_file_fa)
            self.multi_file_fa_dialog = dialog
        elif hasattr(self.multi_file_fa_dialog, "_set_records"):
            self.multi_file_fa_dialog._set_records(self.file_database)
        return self.multi_file_fa_dialog

    def _start_multi_file_fa_from_dialog(self):
        if self.active_multi_file_fa_worker is not None:
            _show_info_message(self, "Dynamics Analysis", "Dynamics analysis is already running.")
            return
        dialog = self._multi_file_fa_analysis_dialog()
        paths, parameters = dialog.values()
        if not paths:
            _show_info_message(self, "Dynamics Analysis", "Select at least one database file.")
            dialog.show()
            return
        requested_model = str(parameters.get("model_method", "fa") or "fa").strip().lower()
        selected_lookup = set(paths)
        selected_records = [record for record in self.file_database if str(record.get("path", "")) in selected_lookup]
        if len(selected_records) == 1:
            record = selected_records[0]
            data = record.get("raw_data")
            if not isinstance(data, UnifiedMEAData):
                _show_info_message(self, "Dynamics Analysis", "The selected file does not contain spike-event data.")
                dialog.show()
                return
            spike_series = _spike_series_from_unified(data)
            stim_times = np.asarray(getattr(data, "stim_times", []), dtype=float)
            stim_times = np.sort(stim_times[np.isfinite(stim_times)])
            artifact_ms = max(0.0, float(parameters.get("artifact_ms", 0.0)))
            if artifact_ms > 0.0 and stim_times.size:
                spike_series, _artifact_masks, _artifact_removed = _filter_spike_series_stim_tail(
                    spike_series,
                    stim_times,
                    artifact_ms,
                )
            if not spike_series:
                _show_info_message(self, "Dynamics Analysis", "The selected file does not contain readable spike trains.")
                dialog.show()
                return
            scope = str(parameters.get("analysis_scope", "burst"))
            if scope == "all_windows":
                intervals = _non_overlapping_spike_windows(spike_series, float(parameters.get("window_ms", 300.0)))
            else:
                intervals = _detect_burst_intervals(
                    spike_series,
                    bin_ms=float(parameters.get("time_bin_ms", 10.0)),
                    threshold_z=float(parameters.get("burst_threshold_z", 4.0)),
                    min_spikes=5,
                )
            if not intervals:
                _show_info_message(self, "Dynamics Analysis", "No usable burst or window intervals were found for the selected file.")
                dialog.show()
                return
            window = BurstTrajectoryWindow(
                spike_series,
                intervals,
                self,
                _maxwell_channel_map_from_unified(data) or self.channel_map,
                model_method=requested_model,
            )
            window.bin_ms.setValue(float(parameters.get("time_bin_ms", 10.0)))
            window.window_ms.setValue(int(round(float(parameters.get("window_ms", 300.0)))))
            scope_index = window.analysis_scope.findData(scope)
            if scope_index >= 0:
                window.analysis_scope.setCurrentIndex(scope_index)
            norm_index = window.normalize.findData(str(parameters.get("normalization", "channel_zscore")))
            if norm_index >= 0:
                window.normalize.setCurrentIndex(norm_index)
            window.latent_dim.setValue(int(parameters.get("latent_dim", 16)))
            if requested_model == "lds":
                linear_index = window.temporal_method.findData("linear")
                if linear_index >= 0:
                    window.temporal_method.setCurrentIndex(linear_index)
            window.filter_values["min_activity"] = float(parameters.get("min_total_activity", 1.0))
            window.filter_values["min_bursts"] = float(parameters.get("min_active_bursts", 1))
            window.filter_values["min_var"] = float(parameters.get("min_variance", 0.0))
            window.filter_values["max_channels"] = float(parameters.get("max_channels", 256))
            window._update_settings_summary()
            window._draw()
            window.finished.connect(lambda _result: self._return_to_multi_file_fa_dialog())
            self._show_child(window)
            _defer_hide(dialog, 0)
            return
        self.pipeline_progress = self._start_progress("Dynamics Analysis", "Starting dynamics analysis...", 100)
        self._set_app_status("Dynamics analysis running", "Preparing latent-state modeling across selected files.")
        worker = MultiFileFactorAnalysisWorker(paths, parameters)
        self.pipeline_progress.canceled.connect(worker.cancel)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(lambda payload, worker=worker: self._multi_file_fa_finished(payload, worker))
        worker.signals.failed.connect(lambda details, worker=worker: self._multi_file_fa_failed(details, worker))
        worker.signals.canceled.connect(lambda message, worker=worker: self._multi_file_fa_canceled(message, worker))
        self.active_multi_file_fa_worker = worker
        self._update_analysis_action_states()
        self.thread_pool.start(worker)

    def _multi_file_fa_finished(self, payload: dict, worker: MultiFileFactorAnalysisWorker):
        if worker._is_cancelled():
            self._multi_file_fa_canceled("Dynamics analysis cancelled", worker)
            return
        if self.active_multi_file_fa_worker is worker:
            self.active_multi_file_fa_worker = None
        self._finish_progress(self.pipeline_progress)
        self.pipeline_progress = None
        self._update_analysis_action_states()
        records = list(payload.get("records", []))
        errors = list(payload.get("errors", []))
        if not records:
            details = "\n".join(errors[:12]) if errors else "No files could be analyzed."
            _show_warning_message(self, "Dynamics Analysis", details)
            self._log("Dynamics analysis produced no usable files")
            self._set_app_status("Dynamics analysis produced no usable files", details.splitlines()[0] if details else "")
            self._return_to_multi_file_fa_dialog()
            return
        self.multi_file_fa_payload = payload
        self._multi_file_fa_analysis_dialog().set_cached_payload(payload)
        self._open_multi_file_fa_window(payload)
        self._set_app_status("Dynamics analysis ready", f"{len(records)} files analyzed, {len(errors)} skipped.")
        self._log(f"Dynamics analysis: {len(records)} files, {len(errors)} skipped")

    def _multi_file_fa_failed(self, details: str, worker: MultiFileFactorAnalysisWorker):
        if self.active_multi_file_fa_worker is worker:
            self.active_multi_file_fa_worker = None
        self._finish_progress(self.pipeline_progress)
        self.pipeline_progress = None
        self._update_analysis_action_states()
        self._log(details)
        self._set_app_status("Dynamics analysis failed", details.splitlines()[-1] if details else "Unknown error")
        _show_error_message(self, "Dynamics Analysis failed", details.splitlines()[-1] if details else "Unknown error")
        self._return_to_multi_file_fa_dialog()

    def _multi_file_fa_canceled(self, message: str, worker: MultiFileFactorAnalysisWorker):
        if self.active_multi_file_fa_worker is worker:
            self.active_multi_file_fa_worker = None
        self._finish_progress(self.pipeline_progress)
        self.pipeline_progress = None
        self._update_analysis_action_states()
        self._log(message or "Dynamics analysis cancelled")
        self._set_app_status("Dynamics analysis cancelled", message or "Dynamics analysis cancelled")
        self._return_to_multi_file_fa_dialog()

    def _open_cached_multi_file_fa(self):
        if not self.multi_file_fa_payload:
            _show_info_message(self, "Dynamics Analysis", "Run analysis before opening results.")
            return
        self._open_multi_file_fa_window(self.multi_file_fa_payload)

    def _open_multi_file_fa_window(self, payload: dict):
        window = MultiFileFactorAnalysisWindow(payload, self)
        self._show_multi_file_fa_result_window(window)

    def _show_multi_file_fa_result_window(self, window: QDialog):
        dialog = self._multi_file_fa_analysis_dialog()
        window.finished.connect(lambda _result: self._return_to_multi_file_fa_dialog())
        self._show_child(window)
        _defer_hide(dialog, 0)

    def _return_to_multi_file_fa_dialog(self):
        if self.multi_file_fa_dialog is None:
            return
        self.multi_file_fa_dialog.show()
        self.multi_file_fa_dialog.raise_()
        self.multi_file_fa_dialog.activateWindow()

    def open_data(self):
        dialog = DataFilesInputDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        paths = dialog.values()
        if not paths:
            return
        selected_wells_by_path = {}
        try:
            if len(paths) == 1 and Path(paths[0]).suffix.lower() == ".spk":
                wells = list_axion_spk_wells(paths[0])
                if wells:
                    selected_wells, accepted = self._select_wells(wells)
                    if not accepted:
                        return
                    selected_wells_by_path[str(Path(paths[0]))] = selected_wells
        except Exception as exc:
            _show_error_message(self, "Load failed", str(exc))
            return

        self.preview_button.setEnabled(False)
        self.save_spike_train_button.setEnabled(False)
        self.save_spike_train_action.setEnabled(False)
        self._set_app_status("Loading data", f"Preparing {len(paths)} file(s) for the database.")
        self.pipeline_progress = self._start_progress("Loading data", "Starting data load...", 100)
        worker = FileDatabaseLoadWorker(paths, selected_wells_by_path=selected_wells_by_path)
        self.pipeline_progress.canceled.connect(worker.cancel)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(lambda payload, worker=worker: self._data_load_finished(payload, worker))
        worker.signals.failed.connect(lambda details, worker=worker: self._data_load_failed(details, worker))
        worker.signals.canceled.connect(lambda message, worker=worker: self._data_load_canceled(message, worker))
        self.active_load_worker = worker
        self._update_analysis_action_states()
        self.thread_pool.start(worker)

    def _data_load_finished(self, payload: dict, worker):
        if worker._is_cancelled():
            self._data_load_canceled("Data loading cancelled", worker)
            return
        if self.active_load_worker is worker:
            self.active_load_worker = None
        self._finish_progress(self.pipeline_progress)
        self.pipeline_progress = None
        self._update_analysis_action_states()
        raw_records = payload.get("records")
        if raw_records is None and payload.get("path"):
            raw_records = [
                {
                    "path": str(payload.get("path", "")),
                    "raw_data": payload.get("raw_data"),
                    "data_kind": str(payload.get("data_kind", "")),
                }
            ]
        records = []
        raw_records = list(raw_records or [])
        for raw_record in raw_records:
            record = self._normalize_database_record(raw_record, allow_well_prompt=len(raw_records) == 1)
            if record is not None:
                records.append(record)
        errors = list(payload.get("errors", []))
        if not records:
            self._refresh_file_database_table()
            self._sync_active_file_controls()
            details = "\n".join(errors[:12]) if errors else "No readable files were loaded."
            self._set_app_status("Data load produced no usable files", details.splitlines()[0] if details else "")
            _show_warning_message(self, "Load failed", details)
            return

        first_new_index = 0
        for record in records:
            first_new_index = self._upsert_database_record(record)
            self._log_database_record(record)
        self.database_sort_column = None
        self._refresh_file_database_table()
        self.database_table.selectRow(first_new_index)
        self._set_active_database_index(first_new_index)
        if errors:
            self._log(f"Skipped {len(errors)} files: {'; '.join(errors[:4])}")
        self._log(f"Database loaded: {len(self.file_database)} files")
        self._set_app_status("Database ready", f"{len(self.file_database)} file(s) loaded into the workspace.")

    def _data_load_failed(self, details: str, worker):
        if self.active_load_worker is worker:
            self.active_load_worker = None
        self._finish_progress(self.pipeline_progress)
        self.pipeline_progress = None
        self._update_analysis_action_states()
        self.preview_button.setEnabled(self.raw_data is not None)
        can_save_file = self.raw_data is not None
        self.save_spike_train_button.setEnabled(can_save_file)
        self.save_spike_train_action.setEnabled(can_save_file)
        self._set_app_status("Data load failed", details.splitlines()[-1] if details else "Unknown error")
        _show_error_message(self, "Load failed", details.splitlines()[-1])

    def _data_load_canceled(self, message: str, worker):
        if self.active_load_worker is worker:
            self.active_load_worker = None
        self._finish_progress(self.pipeline_progress)
        self.pipeline_progress = None
        self._update_analysis_action_states()
        self.preview_button.setEnabled(self.raw_data is not None)
        can_save_file = self.raw_data is not None
        self.save_spike_train_button.setEnabled(can_save_file)
        self.save_spike_train_action.setEnabled(can_save_file)
        self._log(message or "Data loading cancelled")
        self._set_app_status("Data load cancelled", message or "Data loading cancelled")

    def _normalize_database_record(self, record: dict, *, allow_well_prompt: bool = False) -> dict | None:
        path = str(record.get("path", ""))
        raw_data = record.get("raw_data")
        data_kind = str(record.get("data_kind", ""))
        if isinstance(raw_data, UnifiedMEAData) and allow_well_prompt:
            filtered = self._maybe_select_loaded_well(raw_data)
            if filtered is None:
                return None
            raw_data = filtered
        return {"path": path, "raw_data": raw_data, "data_kind": data_kind}

    def _upsert_database_record(self, record: dict) -> int:
        path = str(record.get("path", ""))
        for index, existing in enumerate(self.file_database):
            if str(existing.get("path", "")) == path:
                record["_database_order"] = existing.get("_database_order", index)
                self.file_database[index] = record
                return index
        self._ensure_database_order()
        next_order = max((int(item.get("_database_order", index)) for index, item in enumerate(self.file_database)), default=-1) + 1
        record["_database_order"] = next_order
        self.file_database.append(record)
        return len(self.file_database) - 1

    def _database_record_stats(self, record: dict) -> tuple[int, int, int]:
        return _loaded_data_stats(record.get("raw_data"))

    def _ensure_database_order(self) -> None:
        for index, record in enumerate(self.file_database):
            if "_database_order" not in record:
                record["_database_order"] = index

    def _ensure_processed_order(self) -> None:
        for index, record in enumerate(self.processed_database):
            if "_processed_order" not in record:
                record["_processed_order"] = index

    def _refresh_file_database_table(self) -> None:
        self._ensure_database_order()
        selected_rows = self._selected_database_rows() if hasattr(self, "database_table") else []
        selected_paths = {str(self.file_database[row].get("path", "")) for row in selected_rows}
        self.database_table.blockSignals(True)
        self.database_table.setRowCount(len(self.file_database))
        for row, record in enumerate(self.file_database):
            path = Path(str(record.get("path", "")))
            channels, spikes, waveforms = self._database_record_stats(record)
            values = [
                path.name,
                _loaded_data_kind_label(record.get("raw_data"), str(record.get("data_kind", ""))),
                _loaded_data_activity_label(path, record.get("raw_data")),
                str(channels),
                str(spikes),
                str(waveforms),
                str(path.parent),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                self.database_table.setItem(row, column, item)
        self.database_table.resizeColumnsToContents()
        self.database_table.clearSelection()
        for row, record in enumerate(self.file_database):
            if str(record.get("path", "")) in selected_paths:
                self.database_table.selectRow(row)
        self.database_table.blockSignals(False)
        self._sync_active_file_controls()

    def _upsert_processed_record(self, record: dict) -> int:
        path = str(record.get("path", ""))
        for index, existing in enumerate(self.processed_database):
            if str(existing.get("path", "")) == path:
                record["_processed_order"] = existing.get("_processed_order", index)
                self.processed_database[index] = record
                return index
        self._ensure_processed_order()
        next_order = max((int(item.get("_processed_order", index)) for index, item in enumerate(self.processed_database)), default=-1) + 1
        record["_processed_order"] = next_order
        self.processed_database.append(record)
        return len(self.processed_database) - 1

    def _refresh_processed_database_table(self) -> None:
        self._ensure_processed_order()
        selected_rows = self._selected_processed_rows() if hasattr(self, "processed_table") else []
        selected_paths = {str(self.processed_database[row].get("path", "")) for row in selected_rows}
        self.processed_table.blockSignals(True)
        self.processed_table.setRowCount(len(self.processed_database))
        for row, record in enumerate(self.processed_database):
            matrix = np.asarray(record.get("matrix", []), dtype=float)
            origin = Path(str(record.get("source_path", ""))).name
            values = [
                str(record.get("name", "")),
                str(record.get("source_label", "")),
                str(record.get("dataset_type", "") or record.get("view_mode", "")),
                str(int(matrix.shape[0]) if matrix.ndim >= 1 else 0),
                str(int(matrix.shape[1]) if matrix.ndim == 2 else 0),
                origin,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, str(record.get("path", "")))
                self.processed_table.setItem(row, column, item)
        self.processed_table.resizeColumnsToContents()
        self.processed_table.clearSelection()
        for row, record in enumerate(self.processed_database):
            if str(record.get("path", "")) in selected_paths:
                self.processed_table.selectRow(row)
        self.processed_table.blockSignals(False)

    def _processed_header_clicked(self, column: int) -> None:
        if not self.processed_database:
            return
        if self.processed_sort_column == int(column):
            self.processed_sort_column = None
            self._reorder_processed_database(lambda record: self._processed_default_sort_key(record), "Processed database restored to default order")
            return
        self.processed_sort_column = int(column)
        label = self.processed_table.horizontalHeaderItem(int(column)).text()
        self._reorder_processed_database(
            lambda record: (self._processed_sort_value(record, int(column)), self._processed_default_sort_key(record)),
            f"Processed database sorted by {label}",
        )

    def _processed_default_sort_key(self, record: dict) -> tuple[int, str]:
        try:
            order = int(record.get("_processed_order", 0))
        except (TypeError, ValueError):
            order = 0
        return (order, str(record.get("path", "")).lower())

    def _processed_sort_value(self, record: dict, column: int):
        matrix = np.asarray(record.get("matrix", []), dtype=float)
        values = {
            0: str(record.get("name", "")).lower(),
            1: str(record.get("source_label", "")).lower(),
            2: str(record.get("dataset_type", "") or record.get("view_mode", "")).lower(),
            3: int(matrix.shape[0]) if matrix.ndim >= 1 else 0,
            4: int(matrix.shape[1]) if matrix.ndim == 2 else 0,
            5: str(Path(str(record.get("source_path", ""))).name).lower(),
        }
        return values.get(int(column), str(record.get("path", "")).lower())

    def _reorder_processed_database(self, key, message: str) -> None:
        selected_paths = {
            str(self.processed_database[row].get("path", ""))
            for row in self._selected_processed_rows()
            if 0 <= row < len(self.processed_database)
        }
        self.processed_table.clearSelection()
        self.processed_database.sort(key=key)
        self._refresh_processed_database_table()
        if selected_paths:
            self.processed_table.clearSelection()
            for row, record in enumerate(self.processed_database):
                if str(record.get("path", "")) in selected_paths:
                    self.processed_table.selectRow(row)
        self._log(message)

    def _selected_processed_rows(self) -> list[int]:
        rows = sorted({index.row() for index in self.processed_table.selectedIndexes()})
        return [row for row in rows if 0 <= row < len(self.processed_database)]

    def _selected_processed_records(self) -> list[dict]:
        rows = self._selected_processed_rows()
        if rows:
            return [self.processed_database[row] for row in rows]
        return list(self.processed_database)

    def _processed_selection_changed(self) -> None:
        rows = self._selected_processed_rows()
        if not rows:
            self.active_processed_index = -1
            self._update_data_preview()
            return
        self.active_processed_index = rows[0]
        self._update_data_preview()

    def _database_header_clicked(self, column: int) -> None:
        if not self.file_database:
            return
        if self.database_sort_column == int(column):
            self.database_sort_column = None
            self._reorder_file_database(lambda record: self._database_default_sort_key(record), "Database restored to default order")
            return
        self.database_sort_column = int(column)
        label = self.database_table.horizontalHeaderItem(int(column)).text()
        self._reorder_file_database(
            lambda record: (self._database_sort_value(record, int(column)), self._database_default_sort_key(record)),
            f"Database sorted by {label}",
        )

    def _database_default_sort_key(self, record: dict) -> tuple[int, str]:
        try:
            order = int(record.get("_database_order", 0))
        except (TypeError, ValueError):
            order = 0
        return (order, str(record.get("path", "")).lower())

    def _database_sort_value(self, record: dict, column: int):
        path = Path(str(record.get("path", "")))
        data = record.get("raw_data")
        channels, spikes, waveforms = self._database_record_stats(record)
        values = {
            0: str(path.name).lower(),
            1: _loaded_data_kind_label(data, str(record.get("data_kind", ""))).lower(),
            2: _loaded_data_activity_sort_key(_loaded_data_activity_label(path, data)),
            3: int(channels),
            4: int(spikes),
            5: int(waveforms),
            6: str(path.parent).lower(),
        }
        return values.get(int(column), str(path).lower())

    def _reorder_file_database(self, key, message: str) -> None:
        active_path = str(self.input_path or "")
        selected_paths = {
            str(self.file_database[row].get("path", ""))
            for row in self._selected_database_rows()
            if 0 <= row < len(self.file_database)
        }
        self.database_table.clearSelection()
        self.file_database.sort(key=key)
        self._refresh_file_database_table()
        if selected_paths:
            self.database_table.clearSelection()
            for row, record in enumerate(self.file_database):
                if str(record.get("path", "")) in selected_paths:
                    self.database_table.selectRow(row)
        if active_path:
            for row, record in enumerate(self.file_database):
                if str(record.get("path", "")) == active_path:
                    self.database_table.setCurrentCell(row, 0)
                    self._set_active_database_index(row)
                    break
        self._log(message)

    def _selected_database_rows(self) -> list[int]:
        rows = sorted({index.row() for index in self.database_table.selectedIndexes()})
        return [row for row in rows if 0 <= row < len(self.file_database)]

    def _selected_database_paths(self) -> list[str]:
        rows = self._selected_database_rows() if hasattr(self, "database_table") else []
        if rows:
            return [str(self.file_database[row].get("path", "")) for row in rows]
        return [str(record.get("path", "")) for record in self.file_database if record.get("path")]

    def _selected_database_records(self) -> list[dict]:
        rows = self._selected_database_rows()
        if rows:
            return [self.file_database[row] for row in rows]
        return list(self.file_database)

    def _database_selection_changed(self) -> None:
        rows = self._selected_database_rows()
        if not rows:
            self._sync_active_file_controls()
            return
        current = self.database_table.currentRow()
        self._set_active_database_index(current if current in rows else rows[0])

    def _set_active_database_index(self, index: int) -> None:
        if not (0 <= int(index) < len(self.file_database)):
            self.raw_data = None
            self.data_kind = ""
            self.input_path = ""
            self.file_label.setText("No data files loaded")
            self._update_data_preview()
            self._sync_active_file_controls()
            return
        record = self.file_database[int(index)]
        self.input_path = str(record.get("path", ""))
        self.raw_data = record.get("raw_data")
        self.data_kind = str(record.get("data_kind", ""))
        self.result = None
        self.file_label.setText(f"{int(index) + 1}/{len(self.file_database)}: {Path(self.input_path).name}")
        self._apply_source_channel_map()
        self._sync_active_file_controls()
        self._update_data_preview()
        self._validate_default_channel_map()

    def _sync_active_file_controls(self) -> None:
        has_data = self.raw_data is not None
        busy = any(
            worker is not None
            for worker in [
                self.active_load_worker,
                self.active_stimulus_worker,
                self.active_multi_file_fa_worker,
                self.active_maxwell_waveform_worker,
            ]
        )
        self.preview_button.setEnabled(has_data and not busy)
        self.save_spike_train_button.setEnabled(has_data and not busy)
        self.save_spike_train_action.setEnabled(has_data and not busy)
        self._update_analysis_action_states()

    def _log_database_record(self, record: dict) -> None:
        path = str(record.get("path", ""))
        data = record.get("raw_data")
        data_kind = str(record.get("data_kind", ""))
        self._log(f"Loaded {path}")
        if data_kind == "nev" and isinstance(data, UnifiedMEAData):
            spike_count = sum(values.size for values in data.spikes.values())
            source = data.meta.get("source", "spike file") if isinstance(data.meta, dict) else "spike file"
            selected = data.meta.get("selected_wells", []) if isinstance(data.meta, dict) else []
            self._log(f"Spike source: {source}")
            if selected:
                self._log(f"Selected wells: {', '.join(selected)}")
            self._log(f"Spike channels: {len(data.spikes)}")
            self._log(f"Spikes: {spike_count}")
            self._log("Spike-event files can be previewed and opened as results")
        elif data is not None:
            self._log(f"Raw data shape: {np.asarray(data).shape}")

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
            self.channel_map = _default_axion_channel_map(self.raw_data)
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
        if 0 <= int(self.active_processed_index) < len(self.processed_database):
            record = self.processed_database[int(self.active_processed_index)]
            matrix = np.asarray(record.get("matrix", []), dtype=float)
            params = dict(record.get("parameters", {}) or {})
            lines = [
                f"Processed dataset: {record.get('name', 'n/a')}",
                f"Source: {record.get('origin_name', 'n/a')}",
                f"Source label: {record.get('source_label', 'n/a')}",
                f"Dataset type: {record.get('dataset_type', record.get('view_mode', 'n/a'))}",
                f"Dataset group: {record.get('dataset_group', 'n/a')}",
                f"Generated by: {record.get('dataset_origin', 'n/a')}",
                f"Commit: {record.get('commit', 'n/a')}",
                f"Description: {record.get('description', 'n/a')}",
                f"Samples x features: {int(matrix.shape[0]) if matrix.ndim >= 1 else 0} x {int(matrix.shape[1]) if matrix.ndim == 2 else 0}",
            ]
            if params.get("analysis_kind") == "custom_basic":
                lines.extend(
                    [
                        f"Basic function: {params.get('analysis_type', 'n/a')}",
                        f"Time windows: {params.get('time_windows', 'n/a')}",
                        f"Channels: {params.get('channels') or 'all'}",
                    ]
                )
            else:
                for label, key in [
                    ("Normalization preset", "normalization"),
                    ("Similarity preset", "similarity"),
                    ("Reduction preset", "reduction"),
                    ("Clustering preset", "clustering"),
                ]:
                    if key in params:
                        lines.append(f"{label}: {params.get(key, 'n/a')}")
            commit_detail = str(record.get("commit_detail", "") or "").strip()
            if commit_detail:
                lines.append(f"Commit detail: {commit_detail}")
            labels = list(record.get("sample_labels", []))
            if labels:
                lines.append(f"Sample labels preview: {', '.join(labels[:6])}{' ...' if len(labels) > 6 else ''}")
            return "\n".join(lines)

        if self.raw_data is None:
            return (
                "No data loaded.\n\n"
                "Use Open Data Files to build a raw-file database. "
                "Custom analysis can turn selected files, channels, and time windows into processed datasets."
            )

        lines = []
        if self.input_path:
            lines.append(f"File: {self.input_path}")
        lines.append(f"Kind: {_loaded_data_kind_label(self.raw_data, self.data_kind)}")

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
                    if data.meta.get("waveforms_deferred") or (
                        data.meta.get("extract_waveforms") is False and not waveform_channels
                    ):
                        lines.append("Waveforms: deferred for faster loading; loaded on demand for sorting")
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
        selected_records = self._selected_database_records()
        if len(selected_records) > 1:
            self._save_multiple_database_records(selected_records)
            return
        if self.raw_data is None:
            _show_info_message(self, "Save File", "Load data before saving.")
            return
        self._save_single_record(self.raw_data, self.input_path)

    def _save_single_record(self, data, input_path: str = "") -> None:
        if data is None:
            _show_info_message(self, "Save File", "Load data before saving.")
            return

        save_mode = "Full file"
        if isinstance(data, UnifiedMEAData):
            save_mode, accepted = QInputDialog.getItem(
                self,
                "Save File",
                "Save mode:",
                ["Full file", "Spike train only"],
                0,
                False,
            )
            if not accepted:
                return

        default_name = "data.npz"
        if input_path:
            suffix = "spike_train" if save_mode == "Spike train only" else "data"
            default_name = f"{Path(input_path).stem}_{suffix}.npz"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save file",
            default_name,
            "NPZ data (*.npz);;All files (*)",
        )
        if not path:
            return
        if Path(path).suffix.lower() != ".npz":
            path = f"{path}.npz"

        progress = self._start_progress("Save File", "Saving data...", 0)
        try:
            if isinstance(data, UnifiedMEAData):
                if save_mode == "Spike train only":
                    saved = save_spike_train_npz(data, path)
                else:
                    saved = save_unified_npz(data, path, include_waveforms=True)
            else:
                saved = MEAWriter(path).save_data(data)
        except Exception as exc:
            _show_error_message(self, "Save File failed", str(exc))
            return
        finally:
            self._finish_progress(progress)

        self._log(f"Saved data: {saved}")
        _show_info_message(
            self,
            "Save File",
            f"Saved {save_mode.lower()}:\n{saved}",
        )

    def _save_multiple_database_records(self, records: list[dict]) -> None:
        valid_records = [record for record in list(records or []) if record.get("raw_data") is not None]
        if not valid_records:
            _show_info_message(self, "Save File", "Select one or more loaded database files first.")
            return

        contains_unified = any(isinstance(record.get("raw_data"), UnifiedMEAData) for record in valid_records)
        save_mode = "Full file"
        if contains_unified:
            save_mode, accepted = QInputDialog.getItem(
                self,
                "Save File",
                "Save mode for selected files:",
                ["Full file", "Spike train only"],
                0,
                False,
            )
            if not accepted:
                return

        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
            str(Path(self.input_path).parent) if self.input_path else str(Path.cwd()),
        )
        if not output_dir:
            return

        output_root = Path(output_dir)
        suffix = "spike_train" if save_mode == "Spike train only" else "data"
        saved_paths: list[str] = []
        failures: list[str] = []
        renamed_paths: list[str] = []
        used_paths: set[str] = set()
        progress = self._start_progress("Save File", "Saving selected files...", max(1, len(valid_records)))
        try:
            for index, record in enumerate(valid_records, start=1):
                raw_data = record.get("raw_data")
                input_path = str(record.get("path", ""))
                base_name = Path(input_path).stem or f"record_{index}"
                current_suffix = suffix
                if not isinstance(raw_data, UnifiedMEAData):
                    current_suffix = "array_data"
                target_path = _unique_output_path(output_root / f"{base_name}_{current_suffix}.npz", used_paths)
                if target_path.name != f"{base_name}_{current_suffix}.npz":
                    renamed_paths.append(f"{base_name}_{current_suffix}.npz -> {target_path.name}")
                self._progress_step(progress, f"Saving {target_path.name} ({index}/{len(valid_records)})...", index - 1)
                try:
                    if isinstance(raw_data, UnifiedMEAData):
                        if save_mode == "Spike train only":
                            saved = save_spike_train_npz(raw_data, str(target_path))
                        else:
                            saved = save_unified_npz(raw_data, str(target_path), include_waveforms=True)
                    else:
                        saved = MEAWriter(str(target_path)).save_data(raw_data)
                    saved_paths.append(str(saved))
                except Exception as exc:
                    failures.append(f"{Path(input_path).name or base_name}: {exc}")
            self._progress_step(progress, "Finishing save...", len(valid_records))
        finally:
            self._finish_progress(progress)

        if saved_paths:
            self._log(f"Saved {len(saved_paths)} file(s) to {output_dir}")
            summary = "\n".join(saved_paths[:8])
            if len(saved_paths) > 8:
                summary += f"\n... {len(saved_paths) - 8} more"
            if renamed_paths:
                summary += "\n\nRenamed to avoid conflicts:\n" + "\n".join(renamed_paths[:6])
                if len(renamed_paths) > 6:
                    summary += f"\n... {len(renamed_paths) - 6} more"
            _show_info_message(
                self,
                "Save File",
                f"Saved {len(saved_paths)} {save_mode.lower()} file(s):\n{summary}",
            )
        if failures:
            _show_warning_message(self, "Save File", "\n".join(failures[:12]))

    def preview_raw(self):
        if self.raw_data is None:
            return
        self._ensure_processed_data_for_selected_records()
        progress = self._start_progress("Preparing raster", "Preparing raster data...", 4)
        try:
            if self.data_kind == "nev":
                self._progress_step(progress, "Building raster rows...", 1)
                raster_series, has_units = _raster_series_from_unified(self.raw_data, include_noise=False)
                self._progress_step(progress, "Attaching waveform data...", 2)
                waveform_series = _raster_waveforms_from_unified(self.raw_data, include_noise=False)
                self._progress_step(progress, "Creating raster window...", 3)
                channel_groups = None
                if isinstance(self.raw_data.meta, dict) and self.raw_data.meta.get("source") == "axion_spk":
                    channel_groups = _axion_raster_well_groups(raster_series)
                window = SpikeRasterWindow(
                    "Raw Data Raster" if not has_units else "Raw Data Unit Raster",
                    raster_series,
                    waveform_series,
                    self.raw_data.sr,
                    self,
                    y_axis_label="Unit" if has_units else "Channel",
                    channel_map=self.channel_map,
                    stim_times=self.raw_data.stim_times,
                    channel_groups=channel_groups,
                )
            else:
                self._progress_step(progress, "Rendering array preview...", 2)
                figure = Visualizer().plot_timeseries(self.raw_data)
                window = PlotWindow("Raw Data Raster", figure, self)
            self._progress_step(progress, "Opening raster window...", 4)
        except Exception as exc:
            _show_error_message(self, "Preview failed", str(exc))
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
            _show_info_message(self, "Sorting", "Sorting currently requires loaded spike waveforms.")
            return
        if not self.raw_data.waveforms:
            if self._can_load_deferred_maxwell_waveforms():
                answer = QMessageBox.question(
                    self,
                    "Load Maxwell waveforms?",
                    "This Maxwell file was loaded in spike-only mode for speed. Sorting requires spike-aligned waveforms.\n\nLoad waveforms now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
                self._load_maxwell_waveforms_for_sorting()
                return
            _show_warning_message(self, "Sorting", "The loaded NEV file does not contain spike waveforms.")
            return
        progress = self._start_progress("Opening sorting", "Preparing sorting workspace...", 0)
        try:
            window = SortingWorkspaceWindow(self.raw_data, self)
        except Exception as exc:
            _show_error_message(self, "Sorting failed", str(exc))
            return
        finally:
            self._finish_progress(progress)
        self._show_child(window)

    def _can_load_deferred_maxwell_waveforms(self) -> bool:
        if not isinstance(self.raw_data, UnifiedMEAData) or not isinstance(self.raw_data.meta, dict):
            return False
        return (
            self.raw_data.meta.get("source") == "maxwell_h5"
            and bool(self.input_path)
            and Path(self.input_path).suffix.lower() in {".h5", ".hdf5"}
        )

    def _load_maxwell_waveforms_for_sorting(self):
        if self.active_maxwell_waveform_worker is not None:
            _show_info_message(self, "Sorting", "Maxwell waveforms are already loading.")
            return
        self.maxwell_waveform_progress = self._start_progress(
            "Loading Maxwell waveforms",
            "Reading spike-aligned waveforms for sorting...",
            100,
        )
        worker = MaxwellWaveformLoadWorker(self.input_path)
        self.maxwell_waveform_progress.canceled.connect(worker.cancel)
        worker.signals.progress.connect(self._on_maxwell_waveform_progress)
        worker.signals.finished.connect(lambda data, worker=worker: self._maxwell_waveforms_finished(data, worker))
        worker.signals.failed.connect(lambda details, worker=worker: self._maxwell_waveforms_failed(details, worker))
        worker.signals.canceled.connect(lambda message, worker=worker: self._maxwell_waveforms_canceled(message, worker))
        self.active_maxwell_waveform_worker = worker
        self.thread_pool.start(worker)

    def _on_maxwell_waveform_progress(self, value: int, message: str):
        if self.maxwell_waveform_progress is not None:
            self._progress_step(self.maxwell_waveform_progress, message, max(0, min(100, int(value))))

    def _maxwell_waveforms_finished(self, data: UnifiedMEAData, worker: MaxwellWaveformLoadWorker):
        if worker._is_cancelled():
            self._maxwell_waveforms_canceled("Waveform loading cancelled", worker)
            return
        if self.active_maxwell_waveform_worker is worker:
            self.active_maxwell_waveform_worker = None
        self._finish_progress(self.maxwell_waveform_progress)
        self.maxwell_waveform_progress = None
        if not isinstance(self.raw_data, UnifiedMEAData):
            return
        self.raw_data.waveforms = dict(data.waveforms)
        if not self.raw_data.sr and data.sr:
            self.raw_data.sr = data.sr
        if isinstance(self.raw_data.meta, dict) and isinstance(data.meta, dict):
            for key in (
                "raw_data",
                "waveform_unit",
                "waveform_window_ms",
                "waveform_extraction",
                "extract_waveforms",
                "waveforms_deferred",
                "waveform_channel_count",
            ):
                if key in data.meta:
                    self.raw_data.meta[key] = data.meta[key]
            self.raw_data.meta["waveforms_deferred"] = False
        self._update_data_preview()
        self._log(f"Loaded Maxwell waveforms for {len(self.raw_data.waveforms)} channels")
        self._refresh_file_database_table()
        if self.raw_data.waveforms:
            self.open_sorting()
        else:
            _show_warning_message(self, "Sorting", "No readable Maxwell waveforms were found in this file.")

    def _maxwell_waveforms_failed(self, details: str, worker: MaxwellWaveformLoadWorker):
        if self.active_maxwell_waveform_worker is worker:
            self.active_maxwell_waveform_worker = None
        self._finish_progress(self.maxwell_waveform_progress)
        self.maxwell_waveform_progress = None
        self._log(details)
        _show_error_message(self, "Waveform loading failed", details.splitlines()[-1])

    def _maxwell_waveforms_canceled(self, message: str, worker: MaxwellWaveformLoadWorker):
        if self.active_maxwell_waveform_worker is worker:
            self.active_maxwell_waveform_worker = None
        self._finish_progress(self.maxwell_waveform_progress)
        self.maxwell_waveform_progress = None
        self._log(message or "Waveform loading cancelled")

    def open_temporal_coupling(self):
        if self.data_kind != "nev" or not isinstance(self.raw_data, UnifiedMEAData):
            _show_info_message(self, "Temporal Coupling", "Temporal coupling analysis requires loaded sorted spike data.")
            return
        progress = self._start_progress("Temporal coupling", "Collecting sorted units...", 3)
        units = _unit_spike_trains_from_unified(self.raw_data, include_noise=False)
        self._progress_step(progress, "Checking available units...", 1)
        if len(units) < 2:
            self._finish_progress(progress)
            _show_info_message(
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
            _show_error_message(self, "Temporal Coupling failed", str(exc))
            return
        finally:
            self._finish_progress(progress)
        self._show_child(window)

    def run_pipeline(self):
        if not self.input_path or self.data_kind != "array":
            return
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
                channel_groups = _axion_raster_well_groups(raster_series) if source == "axion_spk" else None
                raster = SpikeRasterWindow(
                    f"{prefix} Spike Raster" if not has_units else f"{prefix} Unit Raster",
                    raster_series,
                    waveform_series,
                    self.raw_data.sr,
                    self,
                    y_axis_label="Unit" if has_units else "Channel",
                    channel_map=self.channel_map,
                    stim_times=self.raw_data.stim_times,
                    channel_groups=channel_groups,
                )
                self._progress_step(progress, "Opening result windows...", 5)
            except Exception as exc:
                _show_error_message(self, "Results failed", str(exc))
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
            _show_error_message(self, "Results failed", str(exc))
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
        self._log(f"Completed. Output: {result.output_path}")
        self._set_app_status("Pipeline complete", f"Output saved to {result.output_path}")
        self.open_results()

    def _on_failed(self, details: str):
        self._finish_progress(self.pipeline_progress)
        self.pipeline_progress = None
        self._log(details)
        self._set_app_status("Pipeline failed", details.splitlines()[-1] if details else "Unknown error")
        _show_error_message(self, "Pipeline failed", details.splitlines()[-1])

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
        stamp = time.strftime("%H:%M:%S")
        self.log.append(f"[{stamp}] {message}")

    def _show_child(self, window: QDialog):
        self.child_windows.append(window)
        window.finished.connect(lambda _: self._forget_child(window))
        window.show()
        window.raise_()
        window.activateWindow()

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
