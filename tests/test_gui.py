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
    StimulusActivationCurveWindow,
    StimulusPSTHWindow,
    StimulusResponseWindow,
    TemporalCouplingWindow,
    BurstDelayWindow,
    ElectrodeMapCanvas,
    _activity_heatmap_color,
    _available_channels_for_data,
    _burst_correlation_analysis,
    _burst_delay_aligned_pairs,
    _burst_delay_first_spike_matrix,
    _burst_delay_pair_values,
    _burst_sequence_payload,
    _source_interval_delay_values,
    _spike_train_delay_aligned_pairs,
    _detect_burst_intervals,
    _format_time_tick,
    _cluster_color_map,
    _raster_series_from_unified,
    _raster_waveforms_from_unified,
    _spike_series_from_unified,
    _spike_trains_from_unified,
    _extract_stimulus_parameters,
    _stimulus_parameter_label,
    _stimulus_response_group_records,
    _stimulus_response_record_from_data,
    _stimulus_response_supported_files,
    _default_maxwell_channel_map,
    StimulusResponseInputDialog,
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


def test_burst_delay_first_spike_matrix_uses_all_channels_by_default():
    spike_series = [(f"chan{index}", np.array([0.01 * index, 1.0])) for index in range(1, 7)]

    channels, _intervals, first_times = _burst_delay_first_spike_matrix(spike_series, [(0.0, 0.2)])

    assert channels == ["chan1", "chan2", "chan3", "chan4", "chan5", "chan6"]
    assert first_times.shape == (1, 6)

    limited_channels, _intervals, limited_first_times = _burst_delay_first_spike_matrix(
        spike_series,
        [(0.0, 0.2)],
        max_channels=3,
    )

    assert limited_channels == ["chan1", "chan2", "chan3"]
    assert limited_first_times.shape == (1, 3)


def test_burst_delay_first_spike_matrix_respects_burst_window():
    spike_series = [
        ("chan1", np.array([0.008, 0.018])),
        ("chan2", np.array([0.003, 0.020])),
    ]

    channels, intervals, first_times = _burst_delay_first_spike_matrix(
        spike_series,
        [(0.0, 0.030)],
        burst_window_ms=5.0,
    )

    assert channels == ["chan1", "chan2"]
    assert intervals == [(0.0, 0.005)]
    assert np.isnan(first_times[0, 0])
    assert first_times[0, 1] == pytest.approx(0.003)


def test_burst_delay_aligned_pairs_reports_five_bin_delay_average():
    delays_ms = np.asarray([8.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0, 11.0, 11.0, 12.0, 12.0])
    first_times = np.column_stack([np.zeros(delays_ms.size), delays_ms / 1000.0])

    results = _burst_delay_aligned_pairs(
        ["chan1", "chan2"],
        first_times,
        max_abs_delay_ms=20.0,
        min_abs_delay_ms=1.0,
        bin_ms=1.0,
        min_peak_count=5,
        min_peak_fraction=0.4,
        min_peak_to_background=1.0,
    )

    assert len(results) == 1
    assert results[0]["reference"] == "chan1"
    assert results[0]["target"] == "chan2"
    assert results[0]["peak_count"] == 5
    assert results[0]["delay_window_count"] == delays_ms.size
    assert results[0]["delay_ms"] == pytest.approx(float(np.mean(delays_ms)))


def test_burst_delay_min_delay_excludes_near_zero_pairs():
    first_times = np.array(
        [
            [0.000, 0.0005, 0.0040],
            [0.010, 0.0105, 0.0140],
            [0.020, 0.0205, 0.0240],
            [0.030, 0.0305, 0.0340],
            [0.040, 0.0405, 0.0440],
        ]
    )

    results = _burst_delay_aligned_pairs(
        ["chan1", "chan2", "chan3"],
        first_times,
        max_abs_delay_ms=10.0,
        min_abs_delay_ms=1.0,
        bin_ms=1.0,
        min_peak_count=5,
        min_peak_fraction=0.4,
        min_peak_to_background=1.0,
    )

    assert [(result["reference"], result["target"]) for result in results] == [("chan1", "chan3")]
    np.testing.assert_allclose(
        _burst_delay_pair_values(first_times, 0, 1, 10.0, min_abs_delay_ms=1.0),
        np.array([]),
    )


def test_source_interval_delay_values_use_first_target_per_source_interval():
    source = np.array([0.000, 0.010, 0.020])
    target = np.array([0.002, 0.003, 0.012, 0.018, 0.025])

    values = _source_interval_delay_values(source, target, max_abs_delay_ms=10.0, min_abs_delay_ms=1.0)

    np.testing.assert_allclose(values, np.array([2.0, 2.0]))


def test_source_interval_delay_values_skip_source_intervals_without_inserted_target():
    source = np.array([0.000, 0.010, 0.020, 0.030])
    target = np.array([0.012, 0.040])

    values = _source_interval_delay_values(
        source,
        target,
        max_abs_delay_ms=50.0,
        min_abs_delay_ms=1.0,
        intervals=[(0.0, 0.035)],
    )

    np.testing.assert_allclose(values, np.array([2.0]))


def test_source_interval_delay_values_include_single_source_spike_before_burst_end():
    source = np.array([0.010])
    target = np.array([0.014])

    values = _source_interval_delay_values(
        source,
        target,
        max_abs_delay_ms=20.0,
        min_abs_delay_ms=1.0,
        intervals=[(0.0, 0.020)],
    )

    np.testing.assert_allclose(values, np.array([4.0]))


def test_spike_train_delay_pairs_support_burst_and_all_spike_modes():
    source = np.array([0.000, 0.010, 0.020, 0.100, 0.110, 0.120])
    target = np.array([0.002, 0.012, 0.102, 0.112])
    channels = ["chan1", "chan2"]
    trains = [source, target]

    burst_values = _source_interval_delay_values(
        source,
        target,
        max_abs_delay_ms=10.0,
        min_abs_delay_ms=1.0,
        intervals=[(0.0, 0.03)],
    )
    all_values = _source_interval_delay_values(source, target, max_abs_delay_ms=10.0, min_abs_delay_ms=1.0)

    np.testing.assert_allclose(burst_values, np.array([2.0, 2.0]))
    np.testing.assert_allclose(all_values, np.array([2.0, 2.0, 2.0, 2.0]))

    results = _spike_train_delay_aligned_pairs(
        channels,
        trains,
        intervals=None,
        max_abs_delay_ms=10.0,
        min_abs_delay_ms=1.0,
        bin_ms=1.0,
        min_peak_count=4,
        min_peak_fraction=0.5,
        min_peak_to_background=1.0,
        mode="all_spikes",
    )

    assert len(results) == 1
    assert results[0]["reference"] == "chan1"
    assert results[0]["target"] == "chan2"
    assert results[0]["mode"] == "all_spikes"
    assert results[0]["delay_ms"] == pytest.approx(2.0)


def test_burst_delay_pair_selectors_filter_significant_subset(monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    monkeypatch.setattr(gui_app.QTimer, "singleShot", lambda *args, **kwargs: None)
    app = QApplication.instance() or QApplication([])
    window = BurstDelayWindow([("chan1", np.array([0.0]))], [], channel_map=ChannelMap.new("test"))
    window.channels = ["chan1", "chan2", "chan3", "chan4"]
    window.aligned_pairs = [
        {"reference_index": 0, "target_index": 1, "reference": "chan1", "target": "chan2", "delay_ms": 2.0},
        {"reference_index": 0, "target_index": 2, "reference": "chan1", "target": "chan3", "delay_ms": 3.0},
        {"reference_index": 3, "target_index": 2, "reference": "chan4", "target": "chan3", "delay_ms": -1.5},
    ]

    window._refresh_channel_combos("", "")

    def items(combo):
        return [combo.itemText(index) for index in range(combo.count())]

    assert items(window.reference_combo) == ["None", "chan1", "chan4"]
    assert items(window.target_combo) == ["None"]

    window.reference_combo.setCurrentText("chan1")

    assert items(window.reference_combo) == ["None", "chan1", "chan4"]
    assert items(window.target_combo) == ["None", "chan2", "chan3"]

    window.target_combo.setCurrentText("chan3")

    assert items(window.reference_combo) == ["None", "chan1", "chan4"]
    assert items(window.target_combo) == ["None", "chan2", "chan3"]

    window.reference_combo.setCurrentText("chan4")

    assert items(window.reference_combo) == ["None", "chan1", "chan4"]
    assert items(window.target_combo) == ["None", "chan3"]
    assert window.target_combo.currentText() == "None"


def test_burst_delay_manual_pair_uses_channel_map_picks_not_dropdowns(monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    monkeypatch.setattr(gui_app.QTimer, "singleShot", lambda *args, **kwargs: None)
    app = QApplication.instance() or QApplication([])
    window = BurstDelayWindow([("chan1", np.array([0.0]))], [], channel_map=ChannelMap.new("test"))
    window.channels = ["chan1", "chan2", "chan3", "chan4"]
    window.first_times = np.array([[0.001, 0.004, 0.007, 0.012], [0.002, 0.005, np.nan, 0.014]])
    window.aligned_pairs = [
        {"reference_index": 0, "target_index": 1, "reference": "chan1", "target": "chan2", "delay_ms": 3.0},
    ]
    window._refresh_channel_combos("", "")

    window.manual_pair_check.setChecked(True)
    window.map_canvas.figure.clear()
    window._map_ax = window.map_canvas.figure.add_subplot(111)
    window._map_channel_indices = np.array([0, 1, 2, 3], dtype=int)
    window._map_channel_xy = np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.6, 0.0]], dtype=float)

    def items(combo):
        return [combo.itemText(index) for index in range(combo.count())]

    assert items(window.reference_combo) == ["None", "chan1"]
    assert items(window.target_combo) == ["None"]

    class Click:
        def __init__(self, xdata, ydata, button):
            self.inaxes = window._map_ax
            self.xdata = xdata
            self.ydata = ydata
            self.x, self.y = window._map_ax.transData.transform((xdata, ydata))
            self.button = button

    window._map_clicked(Click(0.2, 0.0, 1))
    window.map_canvas.figure.clear()
    window._map_ax = window.map_canvas.figure.add_subplot(111)
    window._map_channel_indices = np.array([0, 1, 2, 3], dtype=int)
    window._map_channel_xy = np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.6, 0.0]], dtype=float)
    window._map_clicked(Click(0.6, 0.0, 3))
    selected = window._selected_pair_result()

    assert window._manual_reference_index == 1
    assert window._manual_target_index == 3
    assert selected["manual"] is True
    assert selected["reference"] == "chan2"
    assert selected["target"] == "chan4"
    np.testing.assert_allclose(window._selected_pair_delay_values(selected), np.array([8.0, 9.0]))


def test_burst_delay_map_clicks_are_manual_pair_by_default(monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    monkeypatch.setattr(gui_app.QTimer, "singleShot", lambda *args, **kwargs: None)
    app = QApplication.instance() or QApplication([])
    window = BurstDelayWindow([("chan1", np.array([0.0]))], [], channel_map=ChannelMap.new("test"))
    window.channels = ["chan1", "chan2"]
    window.first_times = np.array([[0.001, 0.004], [0.002, 0.005]])
    window.map_canvas.figure.clear()
    window._map_ax = window.map_canvas.figure.add_subplot(111)
    window._map_channel_indices = np.array([0, 1], dtype=int)
    window._map_channel_xy = np.array([[0.0, 0.0], [0.2, 0.0]], dtype=float)

    class Click:
        def __init__(self, xdata, ydata, button):
            self.inaxes = window._map_ax
            self.xdata = xdata
            self.ydata = ydata
            self.x, self.y = window._map_ax.transData.transform((xdata, ydata))
            self.button = button

    window._map_clicked(Click(0.0, 0.0, 1))
    window.map_canvas.figure.clear()
    window._map_ax = window.map_canvas.figure.add_subplot(111)
    window._map_channel_indices = np.array([0, 1], dtype=int)
    window._map_channel_xy = np.array([[0.0, 0.0], [0.2, 0.0]], dtype=float)
    window._map_clicked(Click(0.2, 0.0, 3))
    selected = window._selected_pair_result()

    assert window._manual_pair_active is True
    assert window._highlight_channel_index == 1
    assert selected["manual"] is True
    assert selected["reference"] == "chan1"
    assert selected["target"] == "chan2"


def test_burst_delay_map_canvas_connects_click_handler(monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    monkeypatch.setattr(gui_app.QTimer, "singleShot", lambda *args, **kwargs: None)
    app = QApplication.instance() or QApplication([])
    window = BurstDelayWindow([("chan1", np.array([0.0]))], [], channel_map=ChannelMap.new("test"))

    callbacks = window.map_canvas.callbacks.callbacks.get("button_press_event", {})

    assert callbacks


def test_burst_delay_parameter_changes_wait_for_analyze(monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    monkeypatch.setattr(gui_app.QTimer, "singleShot", lambda *args, **kwargs: None)
    app = QApplication.instance() or QApplication([])
    window = BurstDelayWindow([("chan1", np.array([0.0]))], [], channel_map=ChannelMap.new("test"))

    window.max_lag_ms.setValue(101)

    assert window.active_delay_worker is None
    assert "click Analyze" in window.summary.text()


def test_burst_delay_draws_source_target_waveforms(monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    monkeypatch.setattr(gui_app.QTimer, "singleShot", lambda *args, **kwargs: None)
    app = QApplication.instance() or QApplication([])
    source_waveforms = np.array([[0.0, -1.0, 0.2, 0.0], [0.1, -0.8, 0.1, 0.0]])
    target_waveforms = np.array([[0.0, -0.9, 0.3, 0.0], [0.0, -0.7, 0.2, 0.1]])
    window = BurstDelayWindow(
        [("chan1", np.array([0.0])), ("chan2", np.array([0.002]))],
        [],
        channel_map=ChannelMap.new("test"),
        waveform_series={"chan1 unit 0": source_waveforms, "chan2": target_waveforms},
        sampling_rate=30000,
    )
    window.channels = ["chan1", "chan2"]
    window.aligned_pairs = [
        {"reference_index": 0, "target_index": 1, "reference": "chan1", "target": "chan2", "delay_ms": 2.0}
    ]
    window._refresh_channel_combos("", "")
    window.reference_combo.setCurrentIndex(window.reference_combo.findData(0))
    window.target_combo.setCurrentIndex(window.target_combo.findData(1))

    window._draw_waveforms()

    axes = window.waveform_canvas.figure.axes
    assert len(axes) == 2
    assert "Source: chan1" == axes[0].get_title()
    assert "Target: chan2" == axes[1].get_title()
    assert axes[0].get_xlabel() == "Time (ms)"
    assert axes[0].get_lines()
    assert axes[1].get_lines()


def test_burst_delay_connected_components_use_undirected_significant_pairs(monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    monkeypatch.setattr(gui_app.QTimer, "singleShot", lambda *args, **kwargs: None)
    app = QApplication.instance() or QApplication([])
    window = BurstDelayWindow([("chan1", np.array([0.0]))], [], channel_map=ChannelMap.new("test"))
    window.channels = ["chan1", "chan2", "chan3", "chan4", "chan5"]
    window.aligned_pairs = [
        {"reference_index": 0, "target_index": 1, "reference": "chan1", "target": "chan2", "delay_ms": 1.0},
        {"reference_index": 2, "target_index": 1, "reference": "chan3", "target": "chan2", "delay_ms": 2.0},
        {"reference_index": 3, "target_index": 4, "reference": "chan4", "target": "chan5", "delay_ms": 3.0},
    ]

    assert window._delay_connected_components() == [[0, 1, 2], [3, 4]]


def test_burst_delay_propagation_paths_follow_source_to_sink_in_large_component(monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    monkeypatch.setattr(gui_app.QTimer, "singleShot", lambda *args, **kwargs: None)
    app = QApplication.instance() or QApplication([])
    channel_map = ChannelMap.new("test")
    electrodes = ["A1", "A2", "A3", "A4", "A5", "B5", "C5", "D5", "E5", "F5"]
    for index, electrode in enumerate(electrodes, start=1):
        channel_map.set_channel(electrode, f"chan{index}")
    window = BurstDelayWindow([("chan1", np.array([0.0]))], [], channel_map=channel_map)
    window.channels = [f"chan{index}" for index in range(1, 11)]
    path_edges = [(index, index + 1) for index in range(9)]
    branch_edges = [(0, 2), (0, 3), (4, 9), (6, 9), (7, 9)]
    window.aligned_pairs = []
    for reference_index, target_index in path_edges + branch_edges:
        window.aligned_pairs.append(
            {
                "reference_index": reference_index,
                "target_index": target_index,
                "reference": f"chan{reference_index + 1}",
                "target": f"chan{target_index + 1}",
                "delay_ms": float((target_index - reference_index) * 5.0),
                "peak_count": 30,
                "peak_fraction": 0.7,
                "peak_to_background": 8.0,
            }
        )

    components = window._delay_connected_components()
    paths = window._propagation_paths(components)

    assert paths
    node_paths = [[layer[0] for layer in path["layers"]] for path in paths]
    assert any(path_nodes[0] == 0 and path_nodes[-1] == 9 and len(path_nodes) >= 3 for path_nodes in node_paths)


def test_burst_delay_raster_can_anchor_source_away_from_zero(monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    monkeypatch.setattr(gui_app.QTimer, "singleShot", lambda *args, **kwargs: None)
    app = QApplication.instance() or QApplication([])
    window = BurstDelayWindow([("chan1", np.array([0.0]))], [], channel_map=ChannelMap.new("test"))
    window.channels = ["chan1", "chan2"]
    window.first_times = np.array([[0.010, 0.017], [0.030, 0.041], [np.nan, 0.050]])
    pair = {"reference_index": 0, "target_index": 1, "reference": "chan1", "target": "chan2", "delay_ms": 9.0}

    source_x, target_x, burst_rows, anchor = window._delay_raster_points(pair)

    assert anchor == 0.0
    np.testing.assert_allclose(source_x, np.array([10.0, 30.0]))
    np.testing.assert_allclose(target_x, np.array([17.0, 41.0]))
    np.testing.assert_array_equal(burst_rows, np.array([0, 1]))
    window.intervals = [(0.0, 0.200), (1.0, 1.150)]
    left, right, xlabel = window._delay_raster_xlim(anchor)
    assert (left, right) == pytest.approx((0.0, 200.0))
    assert xlabel == "Time from burst onset (ms)"

    window.burst_window_ms.blockSignals(True)
    window.burst_window_ms.setValue(50.0)
    window.burst_window_ms.blockSignals(False)
    left, right, xlabel = window._delay_raster_xlim(anchor)
    assert (left, right) == pytest.approx((0.0, 50.0))
    assert xlabel == "Time from burst onset, first 50 ms"

    window.raster_align_combo.setCurrentIndex(window.raster_align_combo.findData("source"))
    source_x, target_x, burst_rows, anchor = window._delay_raster_points(pair)

    assert anchor > 0.0
    np.testing.assert_allclose(source_x, np.full(2, anchor))
    np.testing.assert_allclose(target_x, anchor + np.array([7.0, 11.0]))
    np.testing.assert_array_equal(burst_rows, np.array([0, 1]))
    left, right, xlabel = window._delay_raster_xlim(anchor)
    assert (left, right) == pytest.approx((anchor - 50.0, anchor + 50.0))
    assert xlabel == "Time (ms, source anchored)"


def test_burst_delay_first_spike_probability_fit_uses_burst_count(monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    monkeypatch.setattr(gui_app.QTimer, "singleShot", lambda *args, **kwargs: None)
    app = QApplication.instance() or QApplication([])
    window = BurstDelayWindow([("chan1", np.array([0.0]))], [], channel_map=ChannelMap.new("test"))
    window.channels = ["chan1"]
    window.intervals = [(0.0, 0.050), (1.0, 1.050), (2.0, 2.050), (3.0, 3.050)]
    window.channel_trains = [np.array([0.005, 1.015, 2.080, 3.035])]
    window.bin_ms.blockSignals(True)
    window.bin_ms.setValue(10.0)
    window.bin_ms.blockSignals(False)

    fit = window._first_spike_probability_fit(0, 0.0, 50.0)

    assert fit["burst_count"] == 4
    assert fit["spike_count"] == 3
    assert float(np.sum(fit["observed_probability"])) == pytest.approx(0.75)
    assert np.all(fit["lower_probability"] <= fit["fit_probability"])
    assert np.all(fit["fit_probability"] <= fit["upper_probability"])


def test_burst_delay_first_spike_peak_colors_order_channels(monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    monkeypatch.setattr(gui_app.QTimer, "singleShot", lambda *args, **kwargs: None)
    app = QApplication.instance() or QApplication([])
    window = BurstDelayWindow([("chan1", np.array([0.0]))], [], channel_map=ChannelMap.new("test"))
    window.channels = ["chan1", "chan2"]
    window.intervals = [(0.0, 0.050), (1.0, 1.050), (2.0, 2.050)]
    window.channel_trains = [np.array([0.004, 1.005, 2.006]), np.array([0.034, 1.035, 2.036])]
    window.bin_ms.blockSignals(True)
    window.bin_ms.setValue(10.0)
    window.bin_ms.blockSignals(False)

    peak_times = window._first_spike_peak_times_by_channel()
    low = min(peak_times.values())
    high = max(peak_times.values())

    assert peak_times[0] < peak_times[1]
    assert window._first_spike_peak_color(peak_times[0], low, high) == "#001678"
    assert window._first_spike_peak_color(peak_times[1], low, high) == "#8b0000"


def test_burst_delay_raster_overlays_first_spike_probability_axis(monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    monkeypatch.setattr(gui_app.QTimer, "singleShot", lambda *args, **kwargs: None)
    app = QApplication.instance() or QApplication([])
    window = BurstDelayWindow([("chan1", np.array([0.0]))], [], channel_map=ChannelMap.new("test"))
    window.channels = ["chan1", "chan2"]
    window.intervals = [(0.0, 0.050), (1.0, 1.050), (2.0, 2.050)]
    window.first_times = np.array([[0.005, 0.010], [0.015, 0.020], [0.030, 0.038]])
    window.aligned_pairs = [
        {"reference_index": 0, "target_index": 1, "reference": "chan1", "target": "chan2", "delay_ms": 6.0}
    ]
    window._refresh_channel_combos("", "")
    window.reference_combo.setCurrentIndex(window.reference_combo.findData(0))
    window.target_combo.setCurrentIndex(window.target_combo.findData(1))

    window._draw_delay_raster()

    axes = window.delay_raster_canvas.figure.axes
    assert len(axes) == 2
    assert axes[1].get_ylabel() == "First-spike probability"
    assert axes[1].get_lines()


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


def test_stimulus_response_supported_files_recurses_and_deduplicates(tmp_path):
    first = tmp_path / "freq10Hz" / "amp_50uA" / "trial1.npz"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"placeholder")
    second = tmp_path / "freq20Hz" / "trial2.h5"
    second.parent.mkdir(parents=True)
    second.write_bytes(b"placeholder")
    ignored = tmp_path / "notes.txt"
    ignored.write_text("ignore")

    files = _stimulus_response_supported_files([tmp_path, first])

    assert files == [first, second]


def test_stimulus_parameter_extraction_handles_common_tokens():
    path = r"C:\data\condition_A\freq10Hz\amp_50uA\trial_width200us.npz"

    params = _extract_stimulus_parameters(path)
    label = _stimulus_parameter_label(params, path)

    assert params["frequency_hz"] == pytest.approx(10.0)
    assert params["amplitude_ua"] == pytest.approx(50.0)
    assert params["width_us"] == pytest.approx(200.0)
    assert "frequency_hz=10" in label
    assert "amplitude_ua=50" in label

    params = _extract_stimulus_parameters(r"C:\data\spont_pre\el=12\trial.npz")
    label = _stimulus_parameter_label(params, "trial.npz")
    assert params["stim_electrode"] == pytest.approx(12.0)
    assert params["activity"] == "spont"
    assert params["period"] == "pre"
    assert "stim_electrode=12" in label
    assert "activity=spont" in label

    params = _extract_stimulus_parameters(r"C:\data\multi-site02_after\trial.npz")
    label = _stimulus_parameter_label(params, "trial.npz")
    assert params["stim_mode"] == "multi-site"
    assert "site" not in params
    assert params["period"] == "after"
    assert "stim_mode=multi-site" in label


def test_stimulus_response_record_summarizes_spike_only_data():
    data = UnifiedMEAData(
        spikes={
            "chan1": np.array([0.095, 0.105, 0.115, 0.205, 0.215]),
            "chan2": np.array([0.106, 0.116, 0.206, 0.216]),
        },
        stim_times=np.array([0.100, 0.200]),
        waveforms={"chan1": np.ones((2, 4))},
    )

    record = _stimulus_response_record_from_data(
        "freq10Hz_amp50uA.npz",
        data,
        pre_ms=10.0,
        response_ms=30.0,
        bin_ms=10.0,
    )

    assert record["condition"] == "amplitude_ua=50, frequency_hz=10"
    assert record["stim_count"] == 2
    assert record["channel_count"] == 2
    assert record["response_spikes_per_stim"] == pytest.approx(4.0)
    assert record["baseline_rate_hz_per_channel"] > 0
    assert len(record["trial_spikes_ms"]) == 2
    np.testing.assert_allclose(record["trial_spikes_ms"][0], np.array([-5.0, 5.0, 6.0, 15.0, 16.0]))
    np.testing.assert_allclose(record["trial_spikes_ms"][1], np.array([5.0, 6.0, 15.0, 16.0]))
    assert record["channels"] == ["chan1", "chan2"]
    np.testing.assert_allclose(record["trial_channel_spikes_ms"][0][0], np.array([-5.0, 5.0, 15.0]))
    np.testing.assert_allclose(record["trial_channel_spikes_ms"][0][1], np.array([6.0, 16.0]))

    filtered = _stimulus_response_record_from_data(
        "freq10Hz_amp50uA.npz",
        data,
        pre_ms=10.0,
        response_ms=30.0,
        artifact_ms=6.0,
    )

    np.testing.assert_allclose(filtered["trial_spikes_ms"][0], np.array([15.0, 16.0]))
    np.testing.assert_allclose(filtered["trial_channel_spikes_ms"][0][0], np.array([15.0]))


def test_stimulus_response_group_records_averages_conditions():
    base = {
        "condition": "frequency_hz=10",
        "stim_count": 2,
        "response_rate_hz_per_channel": 10.0,
        "baseline_rate_hz_per_channel": 1.0,
        "mean_latency_ms": 5.0,
        "trial_spikes_ms": [np.array([-1.0, 2.0]), np.array([3.0])],
        "path": "a.npz",
    }
    other = dict(base)
    other["response_rate_hz_per_channel"] = 20.0
    other["trial_spikes_ms"] = [np.array([4.0, 5.0])]
    other["path"] = "b.npz"

    grouped = _stimulus_response_group_records([base, other])

    assert len(grouped) == 1
    assert grouped[0]["file_count"] == 2
    assert grouped[0]["response_rate_mean"] == pytest.approx(15.0)
    assert len(grouped[0]["trial_spikes_ms"]) == 3
    np.testing.assert_allclose(grouped[0]["trial_spikes_ms"][2], np.array([4.0, 5.0]))


def test_stimulus_response_window_uses_file_dropdowns_and_scrolls_trials():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    channel_map = ChannelMap.new("test")
    channel_map.set_channel("A1", "chan1")
    channel_map.set_channel("A2", "chan2")
    records = [
        {
            "condition": "frequency_hz=10",
            "file": "a.npz",
            "parameters": {"stim_electrode": 1.0},
            "stim_count": 100,
            "response_rate_hz_per_channel": 1.0,
            "baseline_rate_hz_per_channel": 0.1,
            "mean_latency_ms": 5.0,
            "channels": ["chan1", "chan2"],
            "trial_channel_spikes_ms": [
                [np.array([-5.0, 4.0]), np.array([6.0])]
                for _ in range(3)
            ],
            "trial_spikes_ms": [np.array([-5.0, 4.0, 6.0]) for _ in range(3)],
            "path": "a.npz",
        },
        {
            "condition": "stim_mode=multi-site",
            "file": "multi-site02_after.npz",
            "parameters": {"stim_mode": "multi-site", "period": "after"},
            "stim_count": 5,
            "response_rate_hz_per_channel": 2.0,
            "baseline_rate_hz_per_channel": 0.2,
            "mean_latency_ms": 6.0,
            "channels": ["chan1", "chan2"],
            "trial_channel_spikes_ms": [
                [np.array([6.0]), np.array([8.0])]
                for _ in range(2)
            ],
            "trial_spikes_ms": [np.array([6.0]) for _ in range(5)],
            "path": "b.npz",
        },
    ]
    window = StimulusResponseWindow(
        {"records": records, "errors": [], "pre_ms": 10.0, "response_ms": 30.0, "artifact_ms": 1.0},
        channel_map=channel_map,
    )

    assert window.left_file_combo.count() == 2
    assert "multi-site 1 sites" in window.right_file_combo.currentText()
    assert len(window.raster_canvas.figure.axes) == 2
    assert window.left_axis.get_ylabel() == "Channel"
    assert [label.get_text() for label in window.left_axis.get_yticklabels()] == ["chan1", "chan2"]
    window.display_window_ms.setValue(20.0)
    left, right = window.left_axis.get_xlim()
    assert right - left == pytest.approx(20.0)

    event = type("ScrollEvent", (), {"inaxes": window.left_axis, "step": -1})()
    window._raster_scrolled(event)

    assert window.trial_indices["left"] == 1
    assert window.trial_indices["right"] == 0
    window._finish_raster_lasso([(-6.0, 0.5), (-4.0, 0.5), (-4.0, 1.5), (-6.0, 1.5)], "left")
    assert "chan1" in window.highlight_channels
    assert window.channel_map_window is not None
    assert window.channel_map_window.windowModality() == Qt.WindowModality.NonModal
    assert "A1" in window.channel_map_window.stim_electrodes
    assert window.channel_map_window.canvas.minimumWidth() >= 1120
    assert "A1" in window.channel_map_window.canvas.stim_electrodes
    window._map_channels_selected(["chan2"])
    assert "chan2" in window.highlight_channels
    window.close()


def test_stimulus_response_local_response_order_groups_spatial_response_neighbors():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    channel_map = ChannelMap.new("test")
    channel_map.set_channel("A1", "chan1")
    channel_map.set_channel("A2", "chan2")
    channel_map.set_channel("H7", "chan4")
    channel_map.set_channel("H8", "chan3")
    record = {
        "condition": "stim",
        "file": "stim.npz",
        "parameters": {},
        "stim_count": 3,
        "response_rate_hz_per_channel": 1.0,
        "baseline_rate_hz_per_channel": 0.0,
        "mean_latency_ms": 8.0,
        "channels": ["chan1", "chan3", "chan2", "chan4"],
        "trial_channel_spikes_ms": [
            [np.array([5.0]), np.array([26.0]), np.array([6.0]), np.array([27.0])],
            [np.array([5.5]), np.array([25.0]), np.array([6.2]), np.array([28.0])],
            [np.array([4.8]), np.array([26.5]), np.array([6.1]), np.array([27.5])],
        ],
        "trial_spikes_ms": [np.array([5.0, 6.0, 26.0, 27.0])],
        "path": "stim.npz",
    }
    window = StimulusResponseWindow(
        {"records": [record], "errors": [], "pre_ms": 10.0, "response_ms": 40.0, "artifact_ms": 0.0},
        channel_map=channel_map,
    )

    ordered_channels, _trials = window._ordered_trial_channel_payload(record)
    assert abs(ordered_channels.index("chan1") - ordered_channels.index("chan2")) == 1
    assert abs(ordered_channels.index("chan3") - ordered_channels.index("chan4")) == 1

    window.row_order_combo.setCurrentIndex(window.row_order_combo.findData("electrode"))
    electrode_channels, _trials = window._ordered_trial_channel_payload(record)
    assert electrode_channels == record["channels"]
    window.close()


def test_stimulus_response_input_defaults_to_long_pre_and_after_window():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    dialog = StimulusResponseInputDialog()

    assert dialog.pre_ms.value() == pytest.approx(200.0)
    assert dialog.response_ms.value() == pytest.approx(1000.0)
    dialog.close()


def test_stimulus_psth_uses_cached_trial_channel_spikes():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    payload = {
        "records": [
            {
                "condition": "stim",
                "file": "a.npz",
                "channels": ["chan1", "chan2"],
                "trial_channel_spikes_ms": [
                    [np.array([1.0, 11.0]), np.array([2.0])],
                    [np.array([3.0]), np.array([4.0, 14.0])],
                ],
                "trial_spikes_ms": [np.array([1.0, 2.0, 11.0]), np.array([3.0, 4.0, 14.0])],
                "path": "a.npz",
            }
        ],
        "errors": [],
        "pre_ms": 0.0,
        "response_ms": 20.0,
        "artifact_ms": 0.0,
    }
    window = StimulusPSTHWindow(payload)
    window.bin_ms.setValue(10.0)

    centers, rates, trial_count, channel_count = window._psth_for_record(payload["records"][0], "")
    np.testing.assert_allclose(centers, np.array([5.0, 15.0]))
    np.testing.assert_allclose(rates, np.array([100.0, 50.0]))
    assert trial_count == 2
    assert channel_count == 2

    window.channel_combo.setCurrentIndex(window.channel_combo.findData("chan1"))
    centers, rates, trial_count, channel_count = window._psth_for_record(payload["records"][0], "chan1")
    np.testing.assert_allclose(rates, np.array([100.0, 50.0]))
    assert trial_count == 2
    assert channel_count == 1

    axes = window.canvas.figure.axes
    assert len(axes) == 2
    assert axes[0].get_ylim() == axes[1].get_ylim()
    window.close()


def test_stimulus_activation_curve_uses_baseline_corrected_evoked_strength():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    payload = {
        "records": [
            {
                "condition": "el1 amp10",
                "file": "el1_amp10uA.npz",
                "parameters": {"stim_electrode": 1.0, "amplitude_ua": 10.0},
                "channels": ["chan1", "chan2"],
                "trial_channel_spikes_ms": [
                    [np.array([-8.0, 2.0]), np.array([])],
                    [np.array([]), np.array([])],
                ],
                "trial_spikes_ms": [np.array([-8.0, 2.0]), np.array([])],
                "path": "el1_amp10uA.npz",
            },
            {
                "condition": "el1 amp20",
                "file": "el1_amp20uA.npz",
                "parameters": {"stim_electrode": 1.0, "amplitude_ua": 20.0},
                "channels": ["chan1", "chan2"],
                "trial_channel_spikes_ms": [
                    [np.array([2.0, 6.0]), np.array([3.0])],
                    [np.array([3.0, 7.0]), np.array([4.0])],
                ],
                "trial_spikes_ms": [np.array([2.0, 6.0]), np.array([3.0, 7.0])],
                "path": "el1_amp20uA.npz",
            },
            {
                "condition": "el2 amp10",
                "file": "el2_amp10uA.npz",
                "parameters": {"stim_electrode": 2.0, "amplitude_ua": 10.0},
                "channels": ["chan1"],
                "trial_channel_spikes_ms": [[np.array([5.0])]],
                "trial_spikes_ms": [np.array([5.0])],
                "path": "el2_amp10uA.npz",
            },
        ],
        "errors": [],
        "pre_ms": 10.0,
        "response_ms": 10.0,
        "artifact_ms": 0.0,
    }
    window = StimulusActivationCurveWindow(payload)

    curves = window.activation_curve_data(site="el1", channel="")
    assert list(curves) == ["el1"]
    np.testing.assert_allclose(curves["el1"]["amplitude"], np.array([10.0, 20.0]))
    np.testing.assert_allclose(curves["el1"]["mean"], np.array([0.05, 3.0]))
    np.testing.assert_allclose(curves["el1"]["sem"], np.array([0.0, 0.0]))

    channel_curves = window.activation_curve_data(site="el1", channel="chan1")
    np.testing.assert_allclose(channel_curves["el1"]["mean"], np.array([0.05, 2.0]))

    axes = window.canvas.figure.axes
    assert len(axes) == 1
    assert axes[0].get_ylabel() == "Evoked spikes / stimulus"
    assert axes[0].get_ylim()[1] > 3.0
    window.close()


def test_stimulus_response_results_return_to_cached_analysis_dialog():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    dialog = window._stimulus_response_analysis_dialog()
    dialog.show()
    payload = {
        "records": [
            {
                "condition": "stim",
                "file": "a.npz",
                "channels": ["chan1"],
                "trial_channel_spikes_ms": [[np.array([2.0])]],
                "trial_spikes_ms": [np.array([2.0])],
                "stim_count": 1,
                "response_rate_hz_per_channel": 1.0,
                "baseline_rate_hz_per_channel": 0.0,
                "mean_latency_ms": 2.0,
                "path": "a.npz",
            }
        ],
        "errors": [],
        "pre_ms": 0.0,
        "response_ms": 20.0,
        "artifact_ms": 0.0,
    }
    worker = type("Worker", (), {"_is_cancelled": lambda self: False})()

    window._stimulus_response_finished(payload, worker)
    app.processEvents()

    assert window.stimulus_response_payload is payload
    assert dialog.cached_payload is payload
    assert dialog.open_raster_button.isEnabled()
    assert dialog.psth_button.isEnabled()
    assert dialog.activation_curve_button.isEnabled()
    assert not dialog.isVisible()
    assert any(isinstance(child, StimulusResponseWindow) for child in window.child_windows)

    result_window = next(child for child in window.child_windows if isinstance(child, StimulusResponseWindow))
    result_window.close()
    app.processEvents()

    assert dialog.isVisible()
    window._open_cached_stimulus_psth()
    app.processEvents()
    assert any(isinstance(child, StimulusPSTHWindow) for child in window.child_windows)
    window._open_cached_stimulus_activation_curve()
    app.processEvents()
    assert any(isinstance(child, StimulusActivationCurveWindow) for child in window.child_windows)
    for child in list(window.child_windows):
        child.close()
    dialog.close()
    window.close()


def test_default_maxwell_map_is_full_template_and_recording_is_file_specific():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    channel_map = _default_maxwell_channel_map()
    assert channel_map is not None
    assert channel_map.name == "maxwell_map"
    assert len(channel_map.electrodes) >= 26400
    assert "e221" in channel_map.electrodes
    assert channel_map.channel_for("e221") == ""
    assert channel_map.electrodes["e221"].get("routed") is False

    canvas = ElectrodeMapCanvas(channel_map)
    payload = channel_map.electrodes["e221"]
    assert canvas._is_recording_electrode("e221", payload) is False

    canvas.set_available_channels(["well1_e221"])
    assert canvas._is_recording_electrode("e221", payload) is True


def test_stimulus_response_uses_maxwell_template_for_maxwell_style_channels():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = StimulusResponseWindow(
        {
            "records": [
                {
                    "condition": "stim",
                    "file": "well1_e221.npz",
                    "parameters": {"stim_electrode": 221.0},
                    "channels": ["well1_e221"],
                    "trial_channel_spikes_ms": [[np.array([2.0])]],
                    "trial_spikes_ms": [np.array([2.0])],
                    "stim_count": 1,
                    "response_rate_hz_per_channel": 1.0,
                    "baseline_rate_hz_per_channel": 0.0,
                    "mean_latency_ms": 2.0,
                    "path": "well1_e221.npz",
                }
            ],
            "errors": [],
            "pre_ms": 200.0,
            "response_ms": 1000.0,
            "artifact_ms": 1.0,
        },
        channel_map=ChannelMap.new("blackrock_like"),
    )

    assert window.channel_map is not None
    assert window.channel_map.name == "maxwell_map"
    assert "e221" in window.electrode_positions
    assert window._stim_electrodes_for_record(window.records[0]) == ["e221"]
    window._set_highlight_channels(["well1_e221"], open_map=True)
    assert window.channel_map_window is not None
    assert "e221" in window.channel_map_window.canvas.highlighted_electrodes
    window.close()


def test_stimulus_response_does_not_reuse_loaded_maxwell_file_map():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    loaded_file_map = _default_maxwell_channel_map()
    assert loaded_file_map is not None
    loaded_file_map.set_channel("e221", "well0_e221")
    loaded_file_map.electrodes["e221"]["routed"] = True

    window = StimulusResponseWindow(
        {
            "records": [
                {
                    "condition": "stim",
                    "file": "well1_el=222_after.npz",
                    "parameters": {"stim_electrode": 222.0},
                    "channels": ["well1_e222"],
                    "trial_channel_spikes_ms": [[np.array([3.0])]],
                    "trial_spikes_ms": [np.array([3.0])],
                    "stim_count": 1,
                    "response_rate_hz_per_channel": 1.0,
                    "baseline_rate_hz_per_channel": 0.0,
                    "mean_latency_ms": 3.0,
                    "path": "well1_el=222_after.npz",
                }
            ],
            "errors": [],
            "pre_ms": 200.0,
            "response_ms": 1000.0,
            "artifact_ms": 1.0,
        },
        channel_map=loaded_file_map,
    )

    assert window.channel_map is not loaded_file_map
    assert window.channel_map is not None
    assert window.channel_map.name == "maxwell_map"
    assert window.channel_map.channel_for("e221") == ""
    assert window.channel_map.electrodes["e221"].get("routed") is False
    assert "e222" in window.electrode_positions
    window._set_highlight_channels(["well1_e222"], open_map=True)
    assert window.channel_map_window is not None
    assert "e222" in window.channel_map_window.canvas.highlighted_electrodes
    assert "e221" not in window.channel_map_window.canvas.highlighted_electrodes
    window.close()


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


def test_spike_raster_actions_are_selected_from_dropdown(monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import SpikeRasterWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = SpikeRasterWindow("Raster", [("chan1", np.array([0.0, 1.0]))])
    called = []
    monkeypatch.setattr(window, "_open_burst_delay_window", lambda: called.append("burst_delay"))

    index = window.raster_action_combo.findData("burst_delay")
    window.raster_action_combo.setCurrentIndex(index)
    window._raster_action_selected(index)

    assert called == ["burst_delay"]
    assert window.raster_action_combo.currentIndex() == 0


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

    assert window.slider.value() == 0
    assert window._playback_time_ms == 50
    assert window.rate_canvas.playhead_time == pytest.approx(window.min_time + 0.05)
    assert window.rate_canvas.window_start == pytest.approx(window.min_time)
    assert window.rate_canvas.window_duration == pytest.approx(window._window_ms() / 1000.0)
    assert window.heatmap_canvas.counts == {"chan1": 0, "chan2": 1}
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


def test_spike_raster_heatmap_gif_frame_times_and_rgb_render():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import SpikeRasterWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    channel_map = ChannelMap.new("test")
    channel_map.set_channel("A1", "chan1")
    window = SpikeRasterWindow("Raster", [("chan1", np.array([0.1, 0.2, 0.3]))], channel_map=channel_map)

    frame_times = window._heatmap_gif_frame_times(0.1, 0.3, 100)
    np.testing.assert_allclose(frame_times, np.array([0.1, 0.2, 0.3]))

    rgb = window.heatmap_canvas.render_counts_rgb({"chan1": 2}, resolution=64, scale_max_count=2)

    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.uint8
    assert int(rgb.max()) > 0


def test_spike_raster_exports_heatmap_gif(tmp_path, monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PIL import Image
        from PySide6.QtWidgets import QApplication, QDialog
        import src.gui.app as gui_app
        from src.gui.app import SpikeRasterWindow
    except ImportError:
        pytest.skip("GIF export dependencies are not available")

    class FakeExportDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def values(self):
            return 0.1, 0.3, 100

    app = QApplication.instance() or QApplication([])
    channel_map = ChannelMap.new("test")
    channel_map.set_channel("A1", "chan1")
    window = SpikeRasterWindow("Raster", [("chan1", np.array([0.1, 0.2, 0.3]))], channel_map=channel_map)
    path = tmp_path / "heatmap.gif"

    monkeypatch.setattr(gui_app, "HeatmapGifExportDialog", FakeExportDialog)
    monkeypatch.setattr(gui_app.QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(path), ""))
    monkeypatch.setattr(gui_app.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(gui_app.QMessageBox, "critical", lambda *args, **kwargs: None)

    assert window._export_heatmap_gif() is True
    assert path.exists()
    with Image.open(path) as image:
        assert image.format == "GIF"
        assert image.n_frames >= 1


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
    assert window.stimulus_response_button.text() == "Stimulus Response Analysis"

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
