"""Tests for GUI integration helpers."""

import os

import numpy as np
import pytest

from src.gui.app import (
    AutoSortingDialog,
    MainWindow,
    SettingsDialog,
    SortingResultsWindow,
    SortingWorkspaceWindow,
    TemporalCouplingWindow,
    _activity_heatmap_color,
    _available_channels_for_data,
    _burst_correlation_analysis,
    _burst_sequence_payload,
    _detect_burst_intervals,
    _format_time_tick,
    _cluster_color_map,
    _raster_series_from_unified,
    _raster_waveforms_from_unified,
    _spike_series_from_unified,
    _spike_trains_from_unified,
    _temporal_coupling_pairs,
    _unit_spike_trains_from_unified,
)
from src.gui.channel_map import ChannelMap
from src.mea_io import UnifiedMEAData
from src.pipeline import PipelineConfig


def test_activity_heatmap_color_uses_black_to_red_scale():
    low = _activity_heatmap_color(0.0)
    mid = _activity_heatmap_color(0.5)
    high = _activity_heatmap_color(1.0)

    assert (low.red(), low.green(), low.blue()) == (0, 0, 0)
    assert (high.red(), high.green(), high.blue()) == (220, 38, 38)
    assert mid.green() > mid.red()
    assert mid.green() > mid.blue()


def test_spike_trains_from_unified_uses_channel_number_order():
    data = UnifiedMEAData(
        spikes={
            "chan10": np.array([0.3]),
            "chan2": np.array([0.2]),
            "chan1": np.array([0.1]),
        }
    )

    trains = _spike_trains_from_unified(data)

    assert [train.tolist() for train in trains] == [[0.1], [0.2], [0.3]]


def test_spike_series_from_unified_preserves_channel_labels():
    data = UnifiedMEAData(
        spikes={
            "chan10": np.array([0.3]),
            "chan2": np.array([0.2]),
            "chan1": np.array([0.1]),
        }
    )

    series = _spike_series_from_unified(data)

    assert [(label, values.tolist()) for label, values in series] == [
        ("chan1", [0.1]),
        ("chan2", [0.2]),
        ("chan10", [0.3]),
    ]


def test_raster_series_uses_channel_rows_without_units():
    data = UnifiedMEAData(
        spikes={"chan2": np.array([0.2]), "chan1": np.array([0.1])},
        waveforms={"chan1": np.ones((1, 4)), "chan2": np.ones((1, 4)) * 2},
    )

    series, has_units = _raster_series_from_unified(data)
    waveforms = _raster_waveforms_from_unified(data)

    assert has_units is False
    assert [(label, values.tolist()) for label, values in series] == [
        ("chan1", [0.1]),
        ("chan2", [0.2]),
    ]
    assert sorted(waveforms) == ["chan1", "chan2"]


def test_raster_series_uses_unit_rows_when_sorting_labels_exist():
    data = UnifiedMEAData(
        spikes={"chan1": np.array([0.0, 0.1, 0.2, 0.3])},
        waveforms={"chan1": np.arange(16, dtype=float).reshape(4, 4)},
        sorting={"chan1": {"waveform_cluster_labels": np.array([0, 1, -1, 1])}},
    )

    series, has_units = _raster_series_from_unified(data)
    waveforms = _raster_waveforms_from_unified(data)

    assert has_units is True
    assert [(label, values.tolist()) for label, values in series] == [
        ("chan1 noise", [0.2]),
        ("chan1 unit 0", [0.0]),
        ("chan1 unit 1", [0.1, 0.3]),
    ]
    assert waveforms["chan1 noise"].tolist() == [[8.0, 9.0, 10.0, 11.0]]
    assert waveforms["chan1 unit 1"].tolist() == [
        [4.0, 5.0, 6.0, 7.0],
        [12.0, 13.0, 14.0, 15.0],
    ]


def test_raster_series_can_omit_noise_units():
    data = UnifiedMEAData(
        spikes={"chan1": np.array([0.0, 0.1, 0.2, 0.3])},
        waveforms={"chan1": np.arange(16, dtype=float).reshape(4, 4)},
        sorting={"chan1": {"waveform_cluster_labels": np.array([0, 1, -1, 1])}},
    )

    series, has_units = _raster_series_from_unified(data, include_noise=False)
    waveforms = _raster_waveforms_from_unified(data, include_noise=False)

    assert has_units is True
    assert [(label, values.tolist()) for label, values in series] == [
        ("chan1 unit 0", [0.0]),
        ("chan1 unit 1", [0.1, 0.3]),
    ]
    assert "chan1 noise" not in waveforms
    assert sorted(waveforms) == ["chan1 unit 0", "chan1 unit 1"]


def test_raster_series_does_not_split_raw_nev_source_unit_labels():
    data = UnifiedMEAData(
        spikes={"chan1": np.array([0.0, 0.1, 0.2, 0.3])},
        waveforms={"chan1": np.arange(16, dtype=float).reshape(4, 4)},
        meta={"source": "blackrock_nev"},
        sorting={"chan1": {"labels": np.array([0, 1, 0, 1])}},
    )

    series, has_units = _raster_series_from_unified(data)
    waveforms = _raster_waveforms_from_unified(data)

    assert has_units is False
    assert [(label, values.tolist()) for label, values in series] == [
        ("chan1", [0.0, 0.1, 0.2, 0.3]),
    ]
    assert waveforms["chan1"].shape == (4, 4)


def test_available_channels_for_nev_include_header_channels_without_spikes():
    data = UnifiedMEAData(
        spikes={"chan1": np.array([0.0])},
        meta={
            "source": "blackrock_nev",
            "electrode_labels": {1: "chan1", 2: "chan2", 65: "chan65"},
            "waveform_headers": {1: {}, 2: {}, 3: {}, 65: {}},
        },
    )

    channels = _available_channels_for_data(data, "nev")

    assert "chan1" in channels
    assert "chan2" in channels
    assert "chan3" in channels
    assert "chan65" not in channels


def test_format_time_tick_uses_three_decimal_places():
    assert _format_time_tick(377.9) == "377.900"
    assert _format_time_tick(0.01234) == "0.012"


def test_cluster_color_map_is_stable_by_cluster_label():
    labels = np.array([2, 0, -1, 2, 1, 0])

    assert _cluster_color_map(labels) == {-1: "#94a3b8", 0: "C0", 1: "C1", 2: "C2"}


def test_detect_burst_intervals_finds_population_rate_segments():
    baseline = np.linspace(0.0, 2.0, 20)
    burst = np.linspace(1.0, 1.06, 80)
    intervals = _detect_burst_intervals(
        [("chan1", np.sort(np.r_[baseline, burst]))],
        bin_ms=10,
        smooth_ms=30,
        threshold_z=2.0,
        min_duration_ms=20,
        min_spikes=10,
    )

    assert intervals
    assert any(start <= 1.02 and stop >= 1.04 for start, stop in intervals)


def test_detect_burst_intervals_uses_quiet_baseline_not_burst_inflated_mean():
    baseline = np.linspace(0.0, 10.0, 80)
    burst = np.linspace(4.0, 4.08, 120)
    intervals = _detect_burst_intervals(
        [("chan1", np.sort(np.r_[baseline, burst]))],
        bin_ms=10,
        smooth_ms=40,
        threshold_z=2.0,
        min_duration_ms=30,
        min_spikes=20,
    )

    assert intervals
    assert any(start <= 4.03 and stop >= 4.06 for start, stop in intervals)


def test_detect_burst_intervals_handles_zero_inflated_sparse_recordings():
    baseline = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    burst = np.linspace(5.0, 5.035, 8)
    intervals = _detect_burst_intervals(
        [("chan1", np.sort(np.r_[baseline, burst]))],
        bin_ms=10,
        smooth_ms=30,
        threshold_z=2.0,
        min_duration_ms=40,
        min_spikes=6,
    )

    assert intervals
    assert any(start <= 5.01 and stop >= 5.03 for start, stop in intervals)


def test_burst_correlation_analysis_orders_similar_burst_patterns():
    intervals = [(0.0, 0.1), (1.0, 1.1), (2.0, 2.1), (3.0, 3.1)]
    spike_series = [
        ("unit_a", np.array([0.01, 0.02, 1.01, 1.02])),
        ("unit_b", np.array([0.03, 1.03])),
        ("unit_c", np.array([2.01, 2.02, 3.01, 3.02])),
        ("unit_d", np.array([2.03, 3.03])),
    ]

    analysis = _burst_correlation_analysis(spike_series, intervals)
    correlation = analysis["correlation"]
    order = analysis["order"]

    assert correlation.shape == (4, 4)
    assert correlation[0, 1] > 0.9
    assert correlation[2, 3] > 0.9
    positions = {burst: index for index, burst in enumerate(order)}
    assert abs(positions[0] - positions[1]) == 1
    assert abs(positions[2] - positions[3]) == 1


def test_burst_correlation_analysis_uses_within_burst_timing():
    intervals = [(0.0, 0.04), (1.0, 1.04), (2.0, 2.04)]
    spike_series = [
        ("unit_a", np.array([0.005, 1.005, 2.025])),
        ("unit_b", np.array([0.025, 1.025, 2.005])),
    ]

    analysis = _burst_correlation_analysis(
        spike_series,
        intervals,
        time_bin_ms=10.0,
        window_ms=40.0,
        normalization="per_burst",
    )
    correlation = analysis["correlation"]

    assert analysis["activity"].shape == (3, 2, 4)
    assert correlation[0, 1] > 0.9
    assert correlation[0, 2] < 0.5


def test_burst_sequence_payload_saves_relative_spike_times_only():
    payload = _burst_sequence_payload(
        [
            ("unit_a", np.array([0.01, 0.02, 1.01])),
            ("unit_b", np.array([0.03, 1.04])),
        ],
        [(0.0, 0.05), (1.0, 1.05)],
    )

    assert payload["format"] == "mea_pipeline_burst_sequences_v1"
    assert payload["labels"].tolist() == ["unit_a", "unit_b"]
    assert payload["burst_intervals_s"].tolist() == [[0.0, 0.05], [1.0, 1.05]]
    assert payload["relative_spike_times_s"][0, 0].tolist() == [0.01, 0.02]
    assert payload["relative_spike_times_s"][1, 1].tolist() == [0.040000000000000036]
    assert "waveforms" not in payload


def test_temporal_coupling_detects_stable_positive_lag():
    reference = np.arange(0.1, 1.0, 0.1)
    target = reference + 0.007
    units = [
        {"id": "chan1 unit 0", "channel": "chan1", "unit": 0, "spikes": reference},
        {"id": "chan2 unit 1", "channel": "chan2", "unit": 1, "spikes": target},
    ]

    pairs = _temporal_coupling_pairs(units, window_ms=30, bin_ms=1, min_spikes=3)

    best = pairs[0]
    assert best["reference_id"] == "chan1 unit 0"
    assert best["target_id"] == "chan2 unit 1"
    assert best["peak_lag_ms"] == pytest.approx(6.5, abs=1.0)
    assert best["peak_count"] >= 8


def test_unit_spike_trains_from_unified_excludes_noise_units():
    data = UnifiedMEAData(
        spikes={"chan1": np.array([0.0, 0.1, 0.2])},
        sorting={"chan1": {"waveform_cluster_labels": np.array([0, -1, 1], dtype=np.int32)}},
    )

    units = _unit_spike_trains_from_unified(data, include_noise=False)

    assert [unit["id"] for unit in units] == ["chan1 unit 0", "chan1 unit 1"]
    assert [unit["spikes"].tolist() for unit in units] == [[0.0], [0.2]]


def test_spike_raster_window_duration_uses_grid_count():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import SpikeRasterWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = SpikeRasterWindow("Raster", [("chan1", np.array([0.0, 1.0]))])
    window.window_grids.setValue(25)
    window.grid_ms.setValue(40)

    assert window._window_ms() == 1000
    assert window.window_grids.buttonSymbols() == window.window_grids.ButtonSymbols.NoButtons
    assert window.grid_ms.buttonSymbols() == window.grid_ms.ButtonSymbols.NoButtons
    assert window.window_plus_button.width() == 28
    assert window.grid_plus_button.width() == 28
    assert window.window_grids.width() == 58
    assert window.grid_ms.width() == 66


def test_spike_raster_wheel_zoom_changes_grid_ms():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import SpikeRasterWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = SpikeRasterWindow("Raster", [("chan1", np.array([0.0, 10.0]))])
    window.grid_ms.setValue(100)
    window.window_grids.setValue(10)
    window.slider.setValue(1000)

    window._zoom_grid_at(0.5, 1)

    assert window.grid_ms.value() == 1
    assert window._window_ms() == 10

    window._zoom_grid_at(0.5, -1)

    assert window.grid_ms.value() == 101


def test_spike_raster_pan_to_absolute_time_updates_slider():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import SpikeRasterWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = SpikeRasterWindow("Raster", [("chan1", np.array([10.0, 20.0]))])
    window.grid_ms.setValue(100)
    window.window_grids.setValue(10)

    window._pan_to_absolute_ms(12_000)

    assert window.slider.value() == 2000


def test_spike_raster_select_channel_updates_waveform_canvas():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import SpikeRasterWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = SpikeRasterWindow(
        "Raster",
        [("chan1", np.array([0.0])), ("chan2", np.array([1.0]))],
        {"chan2": np.ones((3, 4))},
    )

    window._select_channel("chan2")

    assert window.canvas.selected_channel == "chan2"
    assert window.waveform_canvas.channel == "chan2"
    assert window.waveform_canvas.waveforms.shape == (3, 4)


def test_spike_raster_unit_labels_get_wider_axis_margin():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import SpikeRasterWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = SpikeRasterWindow(
        "Raster",
        [("chan100 unit 12", np.array([0.0])), ("chan2 noise", np.array([1.0]))],
        y_axis_label="Unit",
    )

    assert window.canvas.y_axis_label == "Unit"
    assert window.canvas.plot_left > 76
    assert window.canvas._row_center(24, 400, 0, 2) == 124.0
    assert window.canvas._row_center(24, 400, 1, 2) == 324.0


def test_spike_raster_vertical_scroll_limits_visible_rows():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import SpikeRasterWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    series = [(f"chan{index}", np.array([float(index)])) for index in range(1, 11)]
    window = SpikeRasterWindow("Raster", series)
    window.visible_rows.setValue(3)
    window.row_scroll.setValue(4)
    window.canvas.resize(880, 500)

    assert window.canvas.row_offset == 4
    assert window.canvas.visible_row_count == 3
    assert [label for label, _ in window.canvas._visible_spike_series()] == ["chan5", "chan6", "chan7"]
    assert window.row_scroll.maximum() == 7
    assert "Rows 5-7 / 10" in window.row_label.text()

    plot_height = window.canvas.height() - window.canvas.plot_top - window.canvas.plot_bottom
    first_row_y = window.canvas.plot_top + plot_height / 6

    assert window.canvas._channel_at_y(first_row_y) == "chan5"


def test_spike_raster_defaults_to_30_visible_rows():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import SpikeRasterWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    series = [(f"chan{index}", np.array([float(index)])) for index in range(1, 41)]
    window = SpikeRasterWindow("Raster", series)

    assert window.visible_rows.value() == 30
    assert window.canvas.visible_row_count == 30


def test_spike_raster_playback_advances_time_slider_and_heatmap_counts():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import SpikeRasterWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    channel_map = ChannelMap.new("test")
    channel_map.set_channel("A1", "chan1")
    channel_map.set_channel("A2", "chan2")
    window = SpikeRasterWindow(
        "Raster",
        [
            ("chan1", np.array([0.1, 0.2, 1.8])),
            ("chan2", np.array([0.15, 1.0, 1.08])),
        ],
        channel_map=channel_map,
    )
    window.window_grids.setValue(10)
    window.grid_ms.setValue(100)
    assert window.heatmap_ms.value() == 100
    window.heatmap_ms.setValue(40)
    window.slider.setValue(0)

    counts = window._window_channel_counts(0.0, 0.25)

    assert counts == {"chan1": 2, "chan2": 1}
    assert window.heatmap_ms.value() == 40

    window._update_view()

    assert window.heatmap_canvas.counts == {"chan1": 0, "chan2": 1}

    window.play_button.setChecked(True)
    window._toggle_playback()
    window._playback_step()

    assert window.slider.value() == 50
    assert window.rate_canvas.window_start == pytest.approx(window.min_time + 0.05)
    assert window.rate_canvas.window_duration == pytest.approx(window._window_ms() / 1000.0)
    assert window.rate_canvas.rates.size > 0
    centers_a, rates_a = window.rate_canvas._average_rate_trace(0.10, 0.50)
    centers_b, rates_b = window.rate_canvas._average_rate_trace(0.11, 0.51)
    common = np.intersect1d(centers_a, centers_b)
    assert common.size > 0
    for center in common:
        rate_a = rates_a[np.where(centers_a == center)[0][0]]
        rate_b = rates_b[np.where(centers_b == center)[0][0]]
        assert rate_a == pytest.approx(rate_b)
    assert window.play_button.text() == "Pause"

    window._stop_playback()

    assert window.play_button.text() == "Play"


def test_spike_raster_burst_bin_control_refreshes_bursts():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import SpikeRasterWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    baseline = np.linspace(0.0, 2.0, 20)
    burst = np.linspace(1.0, 1.06, 80)
    window = SpikeRasterWindow("Raster", [("chan1", np.sort(np.r_[baseline, burst]))])

    assert window.burst_bin_ms.value() == 10

    previous = list(window.burst_intervals)
    window.burst_bin_ms.setValue(20)

    assert window.burst_bin_ms.value() == 20
    assert window.canvas.burst_intervals == window.burst_intervals
    assert window.rate_canvas.burst_intervals == window.burst_intervals
    assert window.burst_intervals
    assert previous != [] or window.burst_intervals != []


def test_spike_raster_saves_burst_sequences_to_npy(tmp_path, monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
        from src.gui.app import SpikeRasterWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    path = tmp_path / "bursts.npy"
    window = SpikeRasterWindow(
        "Raster",
        [
            ("unit_a", np.array([0.01, 0.02, 1.01])),
            ("unit_b", np.array([0.03, 1.04])),
        ],
    )
    window.burst_intervals = [(0.0, 0.05), (1.0, 1.05)]

    monkeypatch.setattr(gui_app.QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(path), ""))
    monkeypatch.setattr(gui_app.QMessageBox, "information", lambda *args, **kwargs: None)

    assert window._save_bursts() is True
    payload = np.load(path, allow_pickle=True).item()
    assert payload["burst_count"] == 2
    assert payload["row_count"] == 2
    assert payload["relative_spike_times_s"][0, 1].tolist() == [0.03]
    assert "waveforms" not in payload


def test_spike_raster_window_opens_burst_correlation_dialog():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import SpikeRasterWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = SpikeRasterWindow(
        "Raster",
        [
            ("unit_a", np.array([0.01, 0.02, 1.01, 1.02])),
            ("unit_b", np.array([0.03, 1.03])),
            ("unit_c", np.array([2.01, 2.02, 3.01, 3.02])),
            ("unit_d", np.array([2.03, 3.03])),
        ],
    )
    window.burst_intervals = [(0.0, 0.1), (1.0, 1.1), (2.0, 2.1), (3.0, 3.1)]
    dialog = window._open_burst_correlation_window()

    assert not hasattr(window, "analysis_tabs")
    assert dialog.windowTitle() == "Burst Correlation"
    assert dialog.isMaximized()
    assert dialog.time_bin_ms.value() == 5.0
    assert dialog.window_ms.value() == 0
    assert dialog.normalize.currentData() == "per_burst"
    assert dialog.template_count.value() == 3
    assert dialog.global_block_threshold.value() == 0.45
    assert dialog.latency_block_threshold.value() == 0.45
    assert dialog.spatial_block_threshold.value() == 0.45
    assert dialog.dtw_block_threshold.value() == 0.45
    assert dialog.graph_block_threshold.value() == 0.45
    assert [dialog.method_combo.itemData(index) for index in range(dialog.method_combo.count())] == [
        "global_stats",
        "latency",
        "spatial",
        "template",
        "embedding",
        "dtw",
        "graph",
    ]
    for index in range(dialog.method_combo.count()):
        dialog.method_combo.setCurrentIndex(index)
        assert dialog.param_stack.currentWidget() is dialog.param_pages[dialog.method_combo.currentData()]
    dialog.method_combo.setCurrentIndex(dialog.method_combo.findData("template"))
    event = type("Event", (), {"inaxes": dialog.matrix_ax, "xdata": 1.0, "ydata": 0.0})()
    dialog._matrix_clicked(event)
    assert dialog.selected_pair == (int(dialog.current_order[0]), int(dialog.current_order[1]))
    assert len(dialog.sequence_canvas.figure.axes) >= 2
    diagonal_event = type("Event", (), {"inaxes": dialog.matrix_ax, "xdata": 0.0, "ydata": 0.0})()
    dialog._matrix_clicked(diagonal_event)
    assert dialog.selected_pair == (int(dialog.current_order[0]), int(dialog.current_order[0]))
    sequence_axes = dialog.sequence_canvas.figure.axes[:2]
    top_segments = [segment.tolist() for collection in sequence_axes[0].collections for segment in collection.get_segments()]
    bottom_segments = [segment.tolist() for collection in sequence_axes[1].collections for segment in collection.get_segments()]
    assert top_segments == bottom_segments
    first_burst = int(dialog.current_order[0])
    second_burst = int(dialog.current_order[1])
    third_burst = int(dialog.current_order[2])
    dialog.selected_pair = (first_burst, second_burst)
    dialog._draw_selected_burst_sequences()
    first_pair_top = [segment.tolist() for collection in dialog.sequence_canvas.figure.axes[0].collections for segment in collection.get_segments()]
    dialog.selected_pair = (first_burst, third_burst)
    dialog._draw_selected_burst_sequences()
    second_pair_top = [segment.tolist() for collection in dialog.sequence_canvas.figure.axes[0].collections for segment in collection.get_segments()]
    assert first_pair_top == second_pair_top
    assert dialog.sequence_canvas.figure.axes[0].get_ylim()[0] == pytest.approx(len(window.spike_series) - 0.5)
    assert "Bursts: 4" in dialog.summary.text()
    assert "Blocks:" in dialog.summary.text()
    assert "clusters" in dialog.summary.text()
    dialog.method_combo.setCurrentIndex(dialog.method_combo.findData("latency"))
    assert "threshold" in dialog.summary.text()
    assert dialog in window.analysis_windows


def test_spike_raster_opens_ibi_and_selected_unit_isi_dialogs():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import SpikeRasterWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = SpikeRasterWindow(
        "Raster",
        [
            ("chan1 unit 0", np.array([0.0, 0.1, 0.25, 0.6])),
            ("chan2 unit 1", np.array([0.05, 0.2, 0.5, 0.9])),
        ],
    )

    window.burst_intervals = [(0.0, 0.05), (0.4, 0.45), (1.0, 1.05)]
    ibi_dialog = window._open_ibi_window()
    isi_dialog = window._open_isi_window()

    assert ibi_dialog.bin_ms.value() == 300
    assert [isi_dialog.unit_combo.itemText(index) for index in range(isi_dialog.unit_combo.count())] == [
        "chan1 unit 0",
        "chan2 unit 1",
    ]
    assert isi_dialog.bin_ms.value() == 50

    isi_dialog.unit_combo.setCurrentText("chan2 unit 1")
    isi_dialog._draw()

    assert ibi_dialog.canvas.figure.axes[0].get_xlabel() == "IBI (ms)"
    assert isi_dialog.canvas.figure.axes[0].get_xlabel() == "ISI (ms)"
    assert isi_dialog.canvas.figure.axes[0].get_xlim() == (0.0, 1000.0)
    assert ibi_dialog in window.analysis_windows
    assert isi_dialog in window.analysis_windows


def test_spike_raster_label_stride_prevents_vertical_overlap():
    try:
        from src.gui.app import SpikeRasterCanvas
    except ImportError:
        pytest.skip("PySide6 is not available")

    assert SpikeRasterCanvas._label_stride_for_rows(60, 300, 15) == 3
    assert SpikeRasterCanvas._label_stride_for_rows(10, 300, 15) == 1


def test_spike_raster_waveforms_follow_visible_window():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import SpikeRasterWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    waveforms = np.arange(20, dtype=float).reshape(5, 4)
    window = SpikeRasterWindow(
        "Raster",
        [("chan1", np.array([0.0, 0.5, 1.0, 1.5, 2.0]))],
        {"chan1": waveforms},
        30000.0,
    )
    window.window_grids.setValue(10)
    window.grid_ms.setValue(100)
    window._pan_to_absolute_ms(500)
    window._select_channel("chan1")

    assert window.waveform_canvas.waveforms.tolist() == waveforms[1:4].tolist()
    assert window.waveform_canvas.sampling_rate == 30000.0


def test_auto_sorting_dialog_shows_method_specific_parameters():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    dialog = AutoSortingDialog()
    dialog.reduction_method.setCurrentIndex(dialog.reduction_method.findData("ica"))
    dialog.clustering_method.setCurrentIndex(dialog.clustering_method.findData("dbscan"))

    assert not dialog.ica_components.isHidden()
    assert dialog.pca_components.isHidden()
    assert not dialog.dbscan_eps.isHidden()
    assert dialog.max_clusters.isHidden()


def test_spinbox_hit_targets_are_fixed_across_dialogs():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QSpinBox, QDoubleSpinBox
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    widgets = [
        MainWindow(),
        SettingsDialog(PipelineConfig()),
        AutoSortingDialog(),
    ]

    total_spinboxes = 0
    for widget in widgets:
        spinboxes = [*widget.findChildren(QSpinBox), *widget.findChildren(QDoubleSpinBox)]
        total_spinboxes += len(spinboxes)
        for spinbox in spinboxes:
            assert spinbox.minimumHeight() >= 30
    assert total_spinboxes > 0


def test_main_window_replaces_pipeline_cards_with_data_preview():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QProgressBar
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert not window.findChildren(QProgressBar)
    assert not hasattr(window, "cards")
    assert "No data loaded" in window.data_preview.toPlainText()

    window.raw_data = UnifiedMEAData(
        spikes={"chan1": np.array([0.1, 0.2]), "chan2": np.array([0.3])},
        waveforms={"chan1": np.ones((2, 4))},
        sr=30000.0,
        meta={"source": "blackrock_nev"},
    )
    window.data_kind = "nev"
    window.input_path = "sample.nev"
    window._update_data_preview()

    preview = window.data_preview.toPlainText()
    assert "Loaded data preview" not in preview
    assert "Kind: nev" in preview
    assert "Total spikes: 3" in preview
    assert "chan1: 2 spikes" in preview


def test_sorting_results_window_constructs_with_embedding():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    data = UnifiedMEAData(
        spikes={"chan1": np.array([0.0, 0.1, 0.2])},
        waveforms={"chan1": np.ones((3, 4))},
        sr=30000.0,
        sorting={
            "_waveform_clustering": {
                "method": "waveform_clustering",
                "params": {"reduction_method": "pca", "clustering_method": "kmeans"},
                "summary": {"sorted_channels": 1, "total_clusters": 2, "total_spikes": 3},
            },
            "chan1": {
                "waveform_cluster_labels": np.array([0, 1, 1]),
                "embedding": np.array([[0.0, 0.0], [1.0, 1.0], [1.2, 1.1]]),
            },
        },
    )

    window = SortingResultsWindow(data)

    assert window.channel_combo.currentText() == "chan1"
    assert "Total clusters: 2" in window.summary_text.toPlainText()


def test_sorting_results_window_handles_zero_column_embedding():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    data = UnifiedMEAData(
        spikes={"chan1": np.array([0.0, 0.1])},
        waveforms={"chan1": np.ones((2, 4))},
        sorting={
            "chan1": {
                "waveform_cluster_labels": np.array([-1, -1], dtype=np.int32),
                "embedding": np.zeros((2, 0), dtype=np.float32),
            },
        },
    )

    window = SortingResultsWindow(data)

    assert window.channel_combo.currentText() == "chan1"


def test_sorting_workspace_lasso_assignment_updates_channel_labels():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    data = UnifiedMEAData(
        spikes={"chan1": np.array([0.0, 0.1, 0.2, 0.3])},
        waveforms={"chan1": np.array([
            [0.0, -0.1, -0.3, -0.1, 0.0, 0.1, 0.0, 0.0],
            [0.0, -0.2, -0.4, -0.1, 0.1, 0.1, 0.0, 0.0],
            [0.0, 0.2, 0.4, 0.2, 0.0, -0.1, 0.0, 0.0],
            [0.0, 0.1, 0.3, 0.1, -0.1, -0.1, 0.0, 0.0],
        ])},
        sr=30000.0,
    )
    window = SortingWorkspaceWindow(data)
    window.current_embedding = np.array([[0.0, 0.0], [0.1, 0.0], [2.0, 2.0], [2.1, 2.0]], dtype=np.float32)
    window.current_labels = np.zeros(4, dtype=np.int32)
    window.cluster_id.setValue(2)
    window.pending_assignment_label = 2

    window._finish_lasso([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])

    labels = data.sorting["chan1"]["waveform_cluster_labels"]
    assert labels.tolist() == [2, 2, 0, 0]
    assert window.dirty is True

    window._undo_manual_assignment()

    labels = data.sorting["chan1"]["waveform_cluster_labels"]
    assert labels.tolist() == [0, 0, 0, 0]
    assert window.dirty is True

    window.pending_assignment_label = -1
    window.lasso_mode = "noise"
    window._finish_lasso([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])

    labels = data.sorting["chan1"]["waveform_cluster_labels"]
    assert labels.tolist() == [-1, -1, 0, 0]
    assert window.lasso_mode == "noise"
    assert window.save_sorting_button.text() == "Save Sorting"

    window._toggle_lasso_mode("noise", -1)

    assert window.lasso_mode is None


def test_sorting_workspace_handles_zero_column_embedding():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    data = UnifiedMEAData(
        spikes={"chan1": np.array([0.0, 0.1])},
        waveforms={"chan1": np.ones((2, 8))},
        sorting={
            "chan1": {
                "waveform_cluster_labels": np.array([-1, -1], dtype=np.int32),
                "embedding": np.zeros((2, 0), dtype=np.float32),
            },
        },
    )

    window = SortingWorkspaceWindow(data)

    assert window._embedding_xy().shape == (2, 2)


def test_sorting_workspace_cluster_filter_limits_visible_labels():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    data = UnifiedMEAData(
        spikes={"chan1": np.array([0.0, 0.1, 0.2, 0.3])},
        waveforms={"chan1": np.ones((4, 8))},
        sorting={
            "chan1": {
                "waveform_cluster_labels": np.array([0, 1, -1, 1], dtype=np.int32),
                "embedding": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [1.2, 1.1]], dtype=np.float32),
            }
        },
    )
    window = SortingWorkspaceWindow(data)

    assert [window.cluster_filter.itemText(index) for index in range(window.cluster_filter.count())] == [
        "All clusters",
        "noise",
        "cluster 0",
        "cluster 1",
    ]

    window.cluster_filter.setCurrentIndex(window.cluster_filter.findData(1))

    assert window._active_cluster_filter() == 1
    assert window._visible_label_mask(window.current_labels).tolist() == [False, True, False, True]

    window.hide_noise.setChecked(True)
    window.cluster_filter.setCurrentIndex(window.cluster_filter.findData(-1))

    assert window._visible_label_mask(window.current_labels).tolist() == [False, False, False, False]


def test_sorting_workspace_runs_current_channel_sorting(monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    class FakeSignal:
        def connect(self, callback):
            pass

    class FakeSignals:
        progress = FakeSignal()
        finished = FakeSignal()
        failed = FakeSignal()

    captured = {}

    class FakeWorker:
        def __init__(self, data, config, channels=None):
            captured["channels"] = channels
            self.signals = FakeSignals()

    class FakeThreadPool:
        def start(self, worker):
            captured["started"] = worker

    app = QApplication.instance() or QApplication([])
    data = UnifiedMEAData(
        spikes={
            "chan1": np.array([0.0, 0.1]),
            "chan2": np.array([0.2, 0.3, 0.4]),
        },
        waveforms={
            "chan1": np.ones((2, 8)),
            "chan2": np.ones((3, 8)),
        },
    )
    window = SortingWorkspaceWindow(data)
    window.channel_combo.setCurrentText("chan2")
    window.thread_pool = FakeThreadPool()
    monkeypatch.setattr(gui_app, "SortingWorker", FakeWorker)

    window._run_channel_sorting()

    assert window.run_channel_button.text() == "Run Channel Sorting"
    assert captured["channels"] == ["chan2"]
    assert "started" in captured
    assert not window.run_auto_button.isEnabled()
    assert not window.run_channel_button.isEnabled()


def test_sorting_workspace_embedding_scroll_zooms_axes():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    class Event:
        button = "up"

    app = QApplication.instance() or QApplication([])
    data = UnifiedMEAData(
        spikes={"chan1": np.array([0.0, 0.1, 0.2, 0.3])},
        waveforms={"chan1": np.ones((4, 8))},
        sorting={
            "chan1": {
                "waveform_cluster_labels": np.array([0, 0, 1, 1], dtype=np.int32),
                "embedding": np.array([[0.0, 0.0], [0.2, 0.1], [2.0, 2.0], [2.2, 2.1]], dtype=np.float32),
            }
        },
    )
    window = SortingWorkspaceWindow(data)
    xlim_before = window.embedding_ax.get_xlim()
    ylim_before = window.embedding_ax.get_ylim()
    event = Event()
    event.inaxes = window.embedding_ax
    event.xdata = 1.0
    event.ydata = 1.0

    window._embedding_scroll_zoom(event)

    xlim_after = window.embedding_ax.get_xlim()
    ylim_after = window.embedding_ax.get_ylim()
    assert xlim_after[1] - xlim_after[0] < xlim_before[1] - xlim_before[0]
    assert ylim_after[1] - ylim_after[0] < ylim_before[1] - ylim_before[0]
    assert window.embedding_view_limits is not None


def test_temporal_coupling_window_constructs_pair_table():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    reference = np.arange(0.1, 1.0, 0.1)
    target = reference + 0.007
    spikes = np.concatenate([reference, target])
    labels = np.asarray([0] * reference.size + [1] * target.size, dtype=np.int32)
    order = np.argsort(spikes)
    data = UnifiedMEAData(
        spikes={"chan1": spikes[order]},
        waveforms={"chan1": np.ones((spikes.size, 8))},
        sorting={"chan1": {"waveform_cluster_labels": labels[order]}},
    )

    window = TemporalCouplingWindow(data)

    assert window.pair_table.rowCount() > 0
    assert window.pair_table.item(0, 0).text() == "chan1 unit 0"
    assert window.pair_table.item(0, 1).text() == "chan1 unit 1"


def test_temporal_coupling_window_sorts_pairs_by_selected_field():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    def pair(reference_id, target_id, match, strength):
        unit = {"id": reference_id, "spikes": np.array([0.1, 0.2, 0.3])}
        target = {"id": target_id, "spikes": np.array([0.11, 0.21, 0.31])}
        return {
            "reference": unit,
            "target": target,
            "reference_id": reference_id,
            "target_id": target_id,
            "deltas": np.array([0.01, 0.01, 0.01]),
            "hist": np.array([0, 3]),
            "edges": np.array([-0.001, 0.0, 0.001]),
            "peak_lag_ms": 10.0,
            "peak_count": 3,
            "z_score": strength,
            "lag_std_ms": 0.1,
            "matched_ratio": match,
            "strength": strength,
        }

    app = QApplication.instance() or QApplication([])
    reference = np.arange(0.1, 1.0, 0.1)
    target = reference + 0.007
    spikes = np.concatenate([reference, target])
    labels = np.asarray([0] * reference.size + [1] * target.size, dtype=np.int32)
    order = np.argsort(spikes)
    data = UnifiedMEAData(
        spikes={"chan1": spikes[order]},
        waveforms={"chan1": np.ones((spikes.size, 8))},
        sorting={"chan1": {"waveform_cluster_labels": labels[order]}},
    )
    window = TemporalCouplingWindow(data)
    window.all_results = [
        pair("chan2 unit 0", "chan3 unit 0", 0.20, 100.0),
        pair("chan1 unit 0", "chan4 unit 0", 0.80, 1.0),
    ]

    window._sort_by_table_header(6)

    assert window.sort_by.currentData() == "matched_ratio"
    assert window.pair_table.item(0, 0).text() == "chan1 unit 0"

    window.sort_order.setCurrentIndex(window.sort_order.findData("asc"))

    assert window.pair_table.item(0, 0).text() == "chan2 unit 0"

    window._sort_by_table_header(0)

    assert window.sort_by.currentData() == "reference_id"
    assert window.pair_table.item(0, 0).text() == "chan1 unit 0"


def test_sorting_workspace_right_click_exits_assignment_mode():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    class Event:
        button = 3

    app = QApplication.instance() or QApplication([])
    data = UnifiedMEAData(
        spikes={"chan1": np.array([0.0, 0.1])},
        waveforms={"chan1": np.ones((2, 8))},
    )
    window = SortingWorkspaceWindow(data)
    window.lasso_mode = "noise"
    window.noise_button.setChecked(True)

    window._lasso_button_press(Event())

    assert window.lasso_mode is None
    assert not window.noise_button.isChecked()


def test_sorting_workspace_save_clears_dirty_flag(tmp_path, monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    data = UnifiedMEAData(
        spikes={"chan1": np.array([0.0, 0.1])},
        waveforms={"chan1": np.ones((2, 8))},
        sorting={"chan1": {"waveform_cluster_labels": np.array([0, 1])}},
    )
    path = tmp_path / "sorted.npz"
    window = SortingWorkspaceWindow(data)
    window.dirty = True

    monkeypatch.setattr(gui_app.QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(path), ""))

    assert window._save_sorting() is True
    assert path.exists()
    assert window.dirty is False
    assert window.isMaximized()
    assert window.status.text() == f"Saved sorting: {path.name}"
    assert window.status.toolTip() == str(path)


def test_sorting_workspace_save_status_does_not_expand_layout(tmp_path, monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QSizePolicy
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    data = UnifiedMEAData(
        spikes={"chan1": np.array([0.0, 0.1])},
        waveforms={"chan1": np.ones((2, 8))},
        sorting={"chan1": {"waveform_cluster_labels": np.array([0, 1])}},
    )
    long_dir = tmp_path / ("very_long_directory_name_" * 8)
    long_dir.mkdir()
    path = long_dir / ("very_long_sorted_file_name_" * 6 + ".npz")
    window = SortingWorkspaceWindow(data)

    monkeypatch.setattr(gui_app.QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(path), ""))

    assert window._save_sorting() is True
    assert str(long_dir) not in window.status.text()
    assert window.status.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored


def test_spinbox_style_keeps_arrow_rules_visible():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    dialog = AutoSortingDialog()
    style = dialog.max_clusters.styleSheet()

    assert "up-arrow" in style
    assert "down-arrow" in style
    assert "height: 14px" in style
