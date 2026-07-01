"""Tests for GUI integration helpers."""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt

from src.analysis import run_generic_matrix_analysis
from src.gui.app import (
    AutoSortingDialog,
    MainWindow,
    SettingsDialog,
    SortingResultsWindow,
    SortingWorkspaceWindow,
    StimulusActivationCurveWindow,
    MultiFileFactorAnalysisWindow,
    FactorAnalysisDatabaseDialog,
    GenericAnalysisDialog,
    GenericAnalysisWindow,
    StimulusPSTHWindow,
    StimulusDatabaseAnalysisDialog,
    StimulusGenerationDialog,
    StimulusResponseWindow,
    TemporalCouplingWindow,
    BurstDelayWindow,
    DataLoadWorker,
    DataFilesInputDialog,
    FileDatabaseLoadWorker,
    ElectrodeMapCanvas,
    _activity_heatmap_color,
    _available_channels_for_data,
    _burst_correlation_analysis,
    _burst_delay_aligned_pairs,
    _burst_delay_first_spike_matrix,
    _burst_delay_pair_values,
    _burst_sequence_payload,
    _burst_trajectory_analysis,
    _aligned_weight_similarity,
    _multi_file_factor_analysis_payload,
    _non_overlapping_spike_windows,
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
    _loaded_data_activity_label,
    _generic_analysis_matrix_from_record,
    _custom_channel_filter,
    _custom_plot_axis_mode,
    _custom_plot_dataset_y_options,
    _draw_custom_plot_figure,
    _custom_extract_processed_record,
    _custom_label_filter_indices,
    _custom_plot_x_axis,
    _custom_spike_vector_matrix,
    _parse_custom_time_windows,
    CustomDataSelectionDialog,
    CustomPlotDialog,
    CustomPlotWindow,
    _default_axion_channel_map,
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


def test_default_axion_channel_map_uses_64_by_6_layout_and_routes_channels():
    data = UnifiedMEAData(
        spikes={
            "A1_r1c1": np.array([0.1]),
            "B1_r2c3": np.array([0.2]),
        },
        meta={
            "source": "axion_spk",
            "wells": ["A1", "B1"],
            "channel_map": {
                "A1_r1c1": {"well": "A1", "electrode": "r1c1", "electrode_row": 1, "electrode_col": 1},
                "B1_r2c3": {"well": "B1", "electrode": "r2c3", "electrode_row": 2, "electrode_col": 3},
            },
        },
    )

    channel_map = _default_axion_channel_map(data)

    assert channel_map.name == "axion_map"
    assert channel_map.rows == 6
    assert channel_map.cols == 64
    assert channel_map.channel_for("A1_slot01") == "A1_r1c1"
    assert channel_map.channel_for("B1_slot11") == "B1_r2c3"
    assert channel_map.electrodes["A1_slot01"]["routed"] is True
    assert channel_map.electrodes["A1_slot02"]["routed"] is False
    a1 = channel_map.electrodes["A1_slot01"]
    a2 = channel_map.electrodes["A2_slot01"]
    b1 = channel_map.electrodes["B1_slot01"]
    assert float(a2["x_um"]) > float(a1["x_um"])
    assert float(b1["y_um"]) > float(a1["y_um"])
    assert int(a1["well_grid_row"]) == 0
    assert int(a2["well_grid_col"]) == 1
    assert int(b1["well_grid_row"]) == 1


def test_default_axion_channel_map_wells_arrange_in_three_by_two_grid():
    channel_map = _default_axion_channel_map(
        UnifiedMEAData(
            spikes={"A1_r1c1": np.array([0.1])},
            meta={"source": "axion_spk", "wells": ["A1", "A2", "A3", "B1", "B2", "B3"]},
        )
    )

    a1 = channel_map.electrodes["A1_slot01"]
    a2 = channel_map.electrodes["A2_slot01"]
    a3 = channel_map.electrodes["A3_slot01"]
    b1 = channel_map.electrodes["B1_slot01"]
    b2 = channel_map.electrodes["B2_slot01"]
    b3 = channel_map.electrodes["B3_slot01"]

    assert float(a2["x_um"]) > float(a1["x_um"])
    assert float(a3["x_um"]) > float(a2["x_um"])
    assert float(b1["y_um"]) > float(a1["y_um"])
    assert float(b2["x_um"]) > float(b1["x_um"])
    assert float(b3["x_um"]) > float(b2["x_um"])


def test_axion_heatmap_keeps_same_electrode_names_separate_across_wells():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
        from src.gui.app import ElectrodeHeatmapCanvas
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    channel_map = _default_axion_channel_map(
        UnifiedMEAData(
            spikes={"B1_r1c1": np.array([0.1])},
            meta={
                "source": "axion_spk",
                "wells": ["A1", "B1"],
                "channel_map": {
                    "B1_r1c1": {"well": "B1", "electrode": "r1c1", "electrode_row": 1, "electrode_col": 1},
                },
            },
        )
    )
    canvas = ElectrodeHeatmapCanvas(channel_map)
    b1_payload = channel_map.electrodes["B1_slot01"]
    a1_payload = channel_map.electrodes["A1_slot01"]

    assert canvas._count_for_entry({"B1_r1c1": 3}, b1_payload, "B1_slot01") == 3.0
    assert canvas._count_for_entry({"B1_r1c1": 3}, a1_payload, "A1_slot01") == 0.0


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


def test_burst_trajectory_analysis_returns_factor_analysis_latent_state_loadings_and_reconstruction():
    spike_series = [
        ("chan1", np.array([0.005, 0.018, 1.006, 1.019, 2.005, 2.018, 3.006, 3.020])),
        ("chan2", np.array([0.012, 0.028, 1.013, 1.029, 2.030, 2.041, 3.014, 3.031])),
        ("chan3", np.array([0.035, 1.034, 2.012, 3.040])),
        ("chan4", np.array([0.022, 1.022, 2.026, 3.024])),
        ("chan5", np.array([9.0])),
    ]
    intervals = [(0.0, 0.05), (1.0, 1.05), (2.0, 2.05), (3.0, 3.05)]

    analysis = _burst_trajectory_analysis(
        spike_series,
        intervals,
        time_bin_ms=10.0,
        window_ms=50.0,
        normalization="channel_zscore",
        latent_dim=3,
        cluster_count=2,
        min_total_activity=10.0,
        min_active_bursts=2,
        max_channels=4,
    )

    assert analysis["representation"] == "factor_analysis"
    assert analysis["state_projection"] == "Factor Analysis 3D"
    assert analysis["labels"] == ["chan1", "chan2", "chan3", "chan4", "chan5"]
    assert analysis["selected_labels"] == ["chan1", "chan2", "chan3", "chan4"]
    np.testing.assert_array_equal(analysis["selected_channel_indices"], np.array([0, 1, 2, 3]))
    assert "chan5" not in analysis["selected_labels"]
    assert analysis["observed_states"].shape == (4, 5, 4)
    assert analysis["reconstructed_states"].shape == (4, 5, 4)
    assert analysis["raw_observed_states"].shape == (4, 5, 4)
    assert analysis["raw_reconstructed_states"].shape == (4, 5, 4)
    assert np.max(analysis["raw_observed_states"]) > np.max(np.abs(analysis["observed_states"]))
    assert analysis["latent_states"].shape == (4, 5, 3)
    assert analysis["dispersion"].shape == (5,)
    assert analysis["reconstruction_rmse"].shape == (5,)
    assert np.isfinite(analysis["reconstruction_r2"])
    params = analysis["latent_params"]
    assert params["loadings"].shape == (3, 4)
    assert params["mean"].shape == (4,)
    assert params["noise_variance"].shape == (4,)
    assert analysis["channel_filter"]["total_activity"].shape == (5,)
    assert params["n_iter"] >= 0


def test_burst_trajectory_can_use_non_overlapping_all_data_windows():
    spike_series = [
        ("chan1", np.array([0.00, 0.01, 0.31, 0.32, 0.61, 0.62])),
        ("chan2", np.array([0.02, 0.33, 0.64])),
    ]

    windows = _non_overlapping_spike_windows(spike_series, 300.0)
    analysis = _burst_trajectory_analysis(
        spike_series,
        windows,
        time_bin_ms=100.0,
        window_ms=300.0,
        normalization="none",
        latent_dim=1,
        min_total_activity=0.0,
        min_active_bursts=1,
        analysis_scope="all_windows",
    )

    assert windows == [(0.0, 0.3), (0.3, 0.6), (0.6, 0.8999999999999999)]
    assert analysis["analysis_scope"] == "all_windows"
    assert analysis["raw_observed_states"].shape == (3, 3, 2)


def test_burst_trajectory_window_is_factor_analysis_only():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
        from src.gui.app import BurstTrajectoryWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = BurstTrajectoryWindow(
        [
            ("chan1", np.array([0.005, 0.018, 1.006, 1.019])),
            ("chan2", np.array([0.012, 0.028, 1.013, 1.029])),
            ("chan3", np.array([0.035, 1.034])),
        ],
        [(0.0, 0.05), (1.0, 1.05)],
    )

    assert hasattr(window, "raster_canvas")
    assert hasattr(window, "latent_canvas")
    assert hasattr(window, "psth_canvas")
    assert hasattr(window, "weight_canvas")
    assert hasattr(window, "display_value")
    assert hasattr(window, "rmse_order")
    assert hasattr(window, "reconstruction_metrics_button")
    assert hasattr(window, "weight_metrics_button")
    assert hasattr(window, "temporal_model_button")
    assert hasattr(window, "trajectory_analysis_button")
    assert hasattr(window, "normalized_time_button")
    assert hasattr(window, "temporal_method")
    assert hasattr(window, "history_bins")
    assert hasattr(window, "spatial_temporal_button")
    assert hasattr(window, "activity_similarity_weight")
    assert hasattr(window, "spatial_similarity_weight")
    assert hasattr(window, "region_membership_threshold")
    assert not hasattr(window, "export_latent_button")
    assert not hasattr(window, "latent_metrics_button")
    assert window.bin_ms.value() == 10.0
    assert window.window_ms.value() == 300
    assert [window.rmse_order.itemData(index) for index in range(window.rmse_order.count())] == ["desc", "asc"]
    assert [window.display_param.itemData(index) for index in range(window.display_param.count())] == ["burst"]
    assert not hasattr(window, "trajectory_canvas")
    assert not hasattr(window, "reducer")
    assert not hasattr(window, "representation")
    assert "Factor Analysis" in window.summary.text()
    assert "Channels:" in window.summary.text()
    assert window.raster_canvas.figure.axes[0].get_xlabel() == "Time from burst onset (ms)"
    raster_image = window.raster_canvas.figure.axes[0].images[0]
    assert raster_image.get_array().shape[0] == window.current["raw_observed_states"].shape[2] * 2
    assert raster_image.get_extent()[1] == pytest.approx(window.window_ms.value())
    assert "Latent state z(t)" in window.latent_canvas.figure.axes[0].get_title()
    assert "Raw observed vs reconstructed PSTH" in window.psth_canvas.figure.axes[0].get_title()
    assert window.psth_canvas.figure.axes[0].get_ylabel() == "Mean firing rate (Hz)"
    assert "Factor loading matrix W" in window.weight_canvas.figure.axes[0].get_title()

    window.temporal_method.setCurrentIndex(window.temporal_method.findData("knn"))
    window.history_bins.setValue(2)
    temporal_model = window._temporal_latent_model()
    assert temporal_model["history_bins"] == 2
    window._show_reconstruction_metrics()
    window._show_weight_metrics()
    window._show_temporal_model()
    window._show_trajectory_analysis()
    assert len(window.metric_windows) == 4
    assert all(metric_window.findChildren(FigureCanvas) for metric_window in window.metric_windows)
    recon_canvas = window.metric_windows[0].findChildren(FigureCanvas)[0]
    assert "Latent dim vs reconstruction" in recon_canvas.figure.axes[0].get_title()
    weight_canvas = window.metric_windows[1].findChildren(FigureCanvas)[0]
    assert "W subspace singular spectrum" in weight_canvas.figure.axes[0].get_title()
    temporal_canvas = window.metric_windows[2].findChildren(FigureCanvas)[0]
    assert "Raw data vs temporal reconstruction" in temporal_canvas.figure.axes[0].get_title()
    trajectory_canvas = window.metric_windows[3].findChildren(FigureCanvas)[0]
    titles = {axis.get_title() for axis in trajectory_canvas.figure.axes}
    assert "Trajectory-structure clusters" in titles
    assert "Trajectory distance matrix" in titles
    assert "Trajectory-PCA eigenvalues and explained variance" in titles


def test_burst_trajectory_window_shows_normalized_time_analysis():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import BurstTrajectoryWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    bursts = [(float(index), float(index) + 0.08 + 0.01 * (index % 3)) for index in range(6)]
    spike_series = [
        ("chan1", np.array([start + 0.010 for start, _stop in bursts] + [start + 0.020 for start, _stop in bursts])),
        ("chan2", np.array([start + 0.018 for start, _stop in bursts] + [start + 0.036 for start, _stop in bursts])),
        ("chan3", np.array([start + 0.046 for start, _stop in bursts])),
    ]
    window = BurstTrajectoryWindow(spike_series, bursts)

    try:
        result = window._normalized_time_analysis(resample_steps=24, max_k=3)
        assert result["trajectories"].shape[1] == 24
        assert result["population_trajectories"].shape == (len(bursts), 24)
        assert result["normalized_time"][0] == pytest.approx(0.0)
        assert result["normalized_time"][-1] == pytest.approx(1.0)
        assert "clusters" in result["cluster_stats"]
        assert result["silhouette_by_k"]

        window._show_normalized_time_analysis()
        assert window.metric_windows
        metric_canvas = window.metric_windows[-1].findChildren(FigureCanvas)[0]
        titles = {axis.get_title() for axis in metric_canvas.figure.axes}
        assert "Normalized-time burst classes" in titles
        assert "Latent dimensions over normalized burst time" in titles
        assert "Population activity over normalized burst time" in titles
    finally:
        for metric_window in list(window.metric_windows):
            metric_window.close()
        window.close()
        app.processEvents()


def test_burst_trajectory_spatial_temporal_regions_handles_module_time_weight_shapes():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import BurstTrajectoryWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    channel_map = ChannelMap(
        name="test",
        rows=2,
        cols=2,
        electrodes={
            "e1": {"channel": "chan1", "row": 0, "col": 0, "x_um": 0.0, "y_um": 0.0},
            "e2": {"channel": "chan2", "row": 0, "col": 1, "x_um": 20.0, "y_um": 0.0},
            "e3": {"channel": "chan3", "row": 1, "col": 0, "x_um": 0.0, "y_um": 20.0},
            "e4": {"channel": "chan4", "row": 1, "col": 1, "x_um": 20.0, "y_um": 20.0},
        },
    )
    window = BurstTrajectoryWindow(
        [
            ("chan1", np.array([0.005, 1.005])),
            ("chan2", np.array([0.015, 1.015])),
            ("chan3", np.array([0.025, 1.025])),
            ("chan4", np.array([0.035, 1.035])),
        ],
        [(0.0, 0.05), (1.0, 1.05)],
        channel_map=channel_map,
    )

    try:
        analysis = window._spatial_temporal_regions()
        assert isinstance(analysis, dict)
    finally:
        window.close()
        app.processEvents()


def test_burst_trajectory_window_supports_lds_main_views():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import BurstTrajectoryWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = BurstTrajectoryWindow(
        [
            ("chan1", np.array([0.005, 0.018, 1.006, 1.019, 2.006, 2.019])),
            ("chan2", np.array([0.012, 0.028, 1.013, 1.029, 2.013, 2.029])),
            ("chan3", np.array([0.035, 1.034, 2.034])),
        ],
        [(0.0, 0.05), (1.0, 1.05), (2.0, 2.05)],
        model_method="lds",
    )

    try:
        assert window.model_method == "lds"
        assert str((window.current or {}).get("model_method", "")).lower() == "lds"
        assert "LDS" in window.summary.text()
        assert "LDS rollout reconstruction" in window.raster_canvas.figure.axes[0].get_title()
        assert "Modeled latent state z(t)" in window.latent_canvas.figure.axes[0].get_title()
        assert "LDS rollout PSTH" in window.psth_canvas.figure.axes[0].get_title()
        assert "Factor loading matrix W" in window.weight_canvas.figure.axes[0].get_title()
    finally:
        window.close()
        app.processEvents()


def test_burst_trajectory_window_supports_pivae_main_views(monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as app_module
        from src.gui.app import BurstTrajectoryWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    def fake_pivae(raw_observed, latent_dim=16, time_bin_ms=10.0, cancel_check=None):
        raw = np.asarray(raw_observed, dtype=float)
        latent = np.repeat(np.mean(raw, axis=2, keepdims=True), min(int(latent_dim), 2), axis=2)
        params = {
            "method": "pi_vae",
            "latent_dim": latent.shape[2],
            "loadings": np.ones((latent.shape[2], raw.shape[2]), dtype=float),
            "mean": np.mean(raw.reshape((-1, raw.shape[2])), axis=0),
            "n_iter": 1,
        }
        return latent, raw.copy(), params

    monkeypatch.setattr(app_module, "_pivae_latent_states", fake_pivae)
    app = QApplication.instance() or QApplication([])
    window = BurstTrajectoryWindow(
        [
            ("chan1", np.array([0.005, 0.018, 1.006, 1.019, 2.006, 2.019])),
            ("chan2", np.array([0.012, 0.028, 1.013, 1.029, 2.013, 2.029])),
        ],
        [(0.0, 0.05), (1.0, 1.05), (2.0, 2.05)],
        model_method="pivae",
    )

    try:
        assert window.model_method == "pivae"
        assert str((window.current or {}).get("model_method", "")).lower() == "pivae"
        assert "pi-VAE" in window.summary.text()
        assert "pi-VAE reconstruction" in window.raster_canvas.figure.axes[0].get_title()
        assert "pi-VAE latent state z(t)" in window.latent_canvas.figure.axes[0].get_title()
        assert "pi-VAE reconstructed PSTH" in window.psth_canvas.figure.axes[0].get_title()
        assert "pi-VAE decoder loading" in window.weight_canvas.figure.axes[0].get_title()
    finally:
        window.close()
        app.processEvents()


def test_burst_trajectory_spatial_temporal_regions_report_validation_metrics():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import BurstTrajectoryWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    channel_map = ChannelMap(
        name="test",
        rows=2,
        cols=3,
        electrodes={
            "e1": {"channel": "chan1", "x_um": 0.0, "y_um": 0.0},
            "e2": {"channel": "chan2", "x_um": 20.0, "y_um": 0.0},
            "e3": {"channel": "chan3", "x_um": 40.0, "y_um": 0.0},
            "e4": {"channel": "chan4", "x_um": 0.0, "y_um": 20.0},
            "e5": {"channel": "chan5", "x_um": 20.0, "y_um": 20.0},
            "e6": {"channel": "chan6", "x_um": 40.0, "y_um": 20.0},
        },
    )
    bursts = [(float(index), float(index) + 0.12) for index in range(12)]
    spike_series = [
        ("chan1", np.array([start + 0.006 for start, _ in bursts] + [start + 0.012 for start, _ in bursts])),
        ("chan2", np.array([start + 0.007 for start, _ in bursts] + [start + 0.014 for start, _ in bursts])),
        ("chan3", np.array([start + 0.009 for start, _ in bursts] + [start + 0.016 for start, _ in bursts])),
        ("chan4", np.array([start + 0.045 for start, _ in bursts] + [start + 0.053 for start, _ in bursts])),
        ("chan5", np.array([start + 0.047 for start, _ in bursts] + [start + 0.056 for start, _ in bursts])),
        ("chan6", np.array([start + 0.050 for start, _ in bursts] + [start + 0.060 for start, _ in bursts])),
    ]
    window = BurstTrajectoryWindow(spike_series, bursts, channel_map=channel_map)

    try:
        window.activity_similarity_weight.setValue(0.65)
        window.spatial_similarity_weight.setValue(0.35)
        analysis = window._spatial_temporal_regions()
        assert analysis
        assert analysis["cluster_metrics"]
        assert len(analysis["modules"]) >= 2
        assert "similarity" in analysis
        assert analysis["activity_weight"] == pytest.approx(0.65, abs=1e-6)
        assert analysis["spatial_weight"] == pytest.approx(0.35, abs=1e-6)
        if analysis["edges"]:
            edge = analysis["edges"][0]
            assert "median_delay_ms" in edge
            assert "peak_delay_ms" in edge
            assert "peak_window_mean_ms" in edge
            assert "true_values" in edge
            assert "background_values" in edge
    finally:
        window.close()
        app.processEvents()


def test_burst_trajectory_spatial_temporal_filters_weakly_regional_channels():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import BurstTrajectoryWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    channel_map = ChannelMap(
        name="test",
        rows=2,
        cols=4,
        electrodes={
            "e1": {"channel": "chan1", "x_um": 0.0, "y_um": 0.0},
            "e2": {"channel": "chan2", "x_um": 20.0, "y_um": 0.0},
            "e3": {"channel": "chan3", "x_um": 40.0, "y_um": 0.0},
            "e4": {"channel": "chan4", "x_um": 60.0, "y_um": 0.0},
            "e5": {"channel": "chan5", "x_um": 0.0, "y_um": 20.0},
            "e6": {"channel": "chan6", "x_um": 20.0, "y_um": 20.0},
            "e7": {"channel": "chan7", "x_um": 40.0, "y_um": 20.0},
            "e8": {"channel": "chan8", "x_um": 60.0, "y_um": 20.0},
        },
    )
    bursts = [(float(index), float(index) + 0.12) for index in range(10)]
    spike_series = [
        ("chan1", np.array([start + 0.005 for start, _ in bursts] + [start + 0.010 for start, _ in bursts])),
        ("chan2", np.array([start + 0.006 for start, _ in bursts] + [start + 0.011 for start, _ in bursts])),
        ("chan3", np.array([start + 0.045 for start, _ in bursts] + [start + 0.053 for start, _ in bursts])),
        ("chan4", np.array([start + 0.047 for start, _ in bursts] + [start + 0.056 for start, _ in bursts])),
        ("chan5", np.array([start + 0.005 for start, _ in bursts[:5]] + [start + 0.048 for start, _ in bursts[5:]])),
        ("chan6", np.array([start + 0.006 for start, _ in bursts[:5]] + [start + 0.051 for start, _ in bursts[5:]])),
        ("chan7", np.array([start + 0.026 for start, _ in bursts])),
        ("chan8", np.array([start + 0.028 for start, _ in bursts])),
    ]
    window = BurstTrajectoryWindow(spike_series, bursts, channel_map=channel_map)

    try:
        window.region_membership_threshold.setValue(0.30)
        analysis = window._spatial_temporal_regions()
        assert analysis
        assert "channel_significant_mask" in analysis
        assert int(np.sum(analysis["channel_significant_mask"])) < len(analysis["channel_significant_mask"])
        assert int(analysis["retained_channels"]) == int(np.sum(analysis["channel_significant_mask"]))
    finally:
        window.close()
        app.processEvents()


def test_burst_trajectory_spatial_temporal_metric_figure_draws_without_channel_map_type_errors():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import BurstTrajectoryWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    channel_map = ChannelMap(
        name="test",
        rows=2,
        cols=2,
        electrodes={
            "e1": {"channel": "chan1", "row": 0, "col": 0, "x_um": 0.0, "y_um": 0.0},
            "e2": {"channel": "chan2", "row": 0, "col": 1, "x_um": 20.0, "y_um": 0.0},
            "e3": {"channel": "chan3", "row": 1, "col": 0, "x_um": 0.0, "y_um": 20.0},
            "e4": {"channel": "chan4", "row": 1, "col": 1, "x_um": 20.0, "y_um": 20.0},
        },
    )
    window = BurstTrajectoryWindow(
        [
            ("chan1", np.array([0.005, 1.005])),
            ("chan2", np.array([0.015, 1.015])),
            ("chan3", np.array([0.025, 1.025])),
            ("chan4", np.array([0.035, 1.035])),
        ],
        [(0.0, 0.05), (1.0, 1.05)],
        channel_map=channel_map,
    )

    try:
        window._show_spatial_temporal_analysis()
        assert window.metric_windows
        metric_window = window.metric_windows[-1]
        metric_canvas = metric_window.findChildren(FigureCanvas)[0]
        assert "Spatial-temporal regions and directed propagation" in metric_canvas.figure.axes[0].get_title()
    finally:
        for metric_window in list(window.metric_windows):
            metric_window.close()
        window.close()
        app.processEvents()


def test_aligned_weight_similarity_compares_rotated_factor_loadings():
    reference = np.array([[1.0, 0.2, -0.1], [0.0, 0.8, 0.3]])
    theta = np.pi / 5.0
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = rotation @ reference

    similarity, aligned = _aligned_weight_similarity([reference, rotated])

    assert similarity.shape == (2, 2)
    assert similarity[0, 1] == pytest.approx(1.0, abs=1e-6)
    assert aligned[1].shape == rotated.shape


def test_multi_file_factor_analysis_removes_stimulus_tail_before_fitting(monkeypatch, tmp_path):
    path = tmp_path / "el=1_amp=2.npz"
    path.write_bytes(b"placeholder")
    burst_offsets_a = np.array([0.0005, 0.010, 0.014, 0.018, 0.022, 0.026])
    burst_offsets_b = np.array([0.0015, 0.012, 0.016, 0.020, 0.024, 0.028])
    starts = np.array([0.0, 1.0, 2.0])
    data = UnifiedMEAData(
        spikes={
            "chan1": np.sort(np.concatenate([start + burst_offsets_a for start in starts])),
            "chan2": np.sort(np.concatenate([start + burst_offsets_b for start in starts])),
        },
        stim_times=np.array([0.0, 1.0, 2.0]),
        sr=20000.0,
        meta={"source": "maxwell_h5"},
    )
    import src.gui.app as app_module

    monkeypatch.setattr(app_module, "_stimulus_response_supported_files", lambda _paths: [path])
    monkeypatch.setattr(app_module, "_load_spike_only_data", lambda _path, cancel_check=None: data)
    monkeypatch.setattr(app_module, "_detect_burst_intervals", lambda *args, **kwargs: [(0.0, 0.04), (1.0, 1.04), (2.0, 2.04)])

    payload = _multi_file_factor_analysis_payload(
        [path],
        time_bin_ms=10.0,
        window_ms=40.0,
        latent_dim=1,
        min_total_activity=0.0,
        min_active_bursts=1,
        max_channels=2,
        burst_bin_ms=5.0,
        burst_smooth_ms=5.0,
        burst_threshold_z=1.0,
        artifact_ms=1.0,
    )
    unfiltered = _multi_file_factor_analysis_payload(
        [path],
        time_bin_ms=10.0,
        window_ms=40.0,
        latent_dim=1,
        min_total_activity=0.0,
        min_active_bursts=1,
        max_channels=2,
        burst_bin_ms=5.0,
        burst_smooth_ms=5.0,
        burst_threshold_z=1.0,
        artifact_ms=0.0,
    )

    assert payload["artifact_ms"] == 1.0
    assert payload["records"][0]["stim_count"] == 3
    assert payload["records"][0]["artifact_removed_spikes"] == 3
    raw = np.asarray(payload["records"][0]["analysis"]["raw_observed_states"], dtype=float)
    unfiltered_raw = np.asarray(unfiltered["records"][0]["analysis"]["raw_observed_states"], dtype=float)
    assert float(np.sum(unfiltered_raw) - np.sum(raw)) == pytest.approx(3.0 / 0.010)


def test_multi_file_factor_analysis_uses_global_channel_order_and_selection(monkeypatch, tmp_path):
    paths = [tmp_path / "segment_a.npz", tmp_path / "segment_b.npz"]
    for path in paths:
        path.write_bytes(b"placeholder")
    intervals = [(0.0, 0.04), (1.0, 1.04), (2.0, 2.04)]
    data_a = UnifiedMEAData(
        spikes={
            "chan2": np.array([0.012, 0.016, 1.012, 1.016, 2.012, 2.016]),
            "chan1": np.array([0.010, 0.014, 1.010, 1.014, 2.010, 2.014]),
        },
        sr=20000.0,
    )
    data_b = UnifiedMEAData(
        spikes={
            "chan3": np.array([9.0]),
            "chan1": np.array([0.011, 0.015, 1.011, 1.015, 2.011, 2.015]),
            "chan2": np.array([0.013, 0.017, 1.013, 1.017, 2.013, 2.017]),
        },
        sr=20000.0,
    )
    data_by_name = {paths[0].name: data_a, paths[1].name: data_b}
    import src.gui.app as app_module

    monkeypatch.setattr(app_module, "_stimulus_response_supported_files", lambda _paths: paths)
    monkeypatch.setattr(app_module, "_load_spike_only_data", lambda path, cancel_check=None: data_by_name[Path(path).name])
    monkeypatch.setattr(app_module, "_detect_burst_intervals", lambda *args, **kwargs: intervals)

    payload = _multi_file_factor_analysis_payload(
        paths,
        time_bin_ms=10.0,
        window_ms=40.0,
        analysis_scope="all_windows",
        latent_dim=1,
        min_total_activity=2.0,
        min_active_bursts=2,
        max_channels=2,
        artifact_ms=0.0,
    )

    assert payload["analysis_scope"] == "all_windows"
    assert payload["global_labels"] == ["chan1", "chan2", "chan3"]
    assert payload["global_selected_labels"] == ["chan1", "chan2"]
    assert len(payload["records"]) == 2
    for record in payload["records"]:
        assert record["analysis"]["selected_labels"] == payload["global_selected_labels"]
        assert record["analysis"]["latent_params"]["loadings"].shape[1] == 2


def test_multi_file_factor_analysis_supports_lds_model(monkeypatch, tmp_path):
    path = tmp_path / "segment_a.npz"
    path.write_bytes(b"placeholder")
    intervals = [(0.0, 0.04), (1.0, 1.04), (2.0, 2.04)]
    data = UnifiedMEAData(
        spikes={
            "chan1": np.array([0.010, 0.018, 1.010, 1.018, 2.010, 2.018]),
            "chan2": np.array([0.014, 0.022, 1.014, 1.022, 2.014, 2.022]),
        },
        sr=20000.0,
    )
    import src.gui.app as app_module

    monkeypatch.setattr(app_module, "_stimulus_response_supported_files", lambda _paths: [path])
    monkeypatch.setattr(app_module, "_load_spike_only_data", lambda _path, cancel_check=None: data)
    monkeypatch.setattr(app_module, "_detect_burst_intervals", lambda *args, **kwargs: intervals)

    payload = _multi_file_factor_analysis_payload(
        [path],
        time_bin_ms=10.0,
        window_ms=40.0,
        model_method="lds",
        latent_dim=1,
        min_total_activity=0.0,
        min_active_bursts=1,
        max_channels=2,
        artifact_ms=0.0,
    )

    assert payload["model_method"] == "lds"
    analysis = payload["records"][0]["analysis"]
    assert analysis["model_method"] == "lds"
    assert np.asarray(analysis["transition_matrix"], dtype=float).shape == (1, 1)
    assert np.asarray(analysis["model_latent_states"], dtype=float).shape == np.asarray(analysis["latent_states"], dtype=float).shape
    assert "rollout_r2" in analysis
    assert "one_step_r2" in analysis


def test_multi_file_factor_analysis_supports_pivae_model(monkeypatch, tmp_path):
    path = tmp_path / "segment_pivae.npz"
    path.write_bytes(b"placeholder")
    intervals = [(0.0, 0.04), (1.0, 1.04), (2.0, 2.04)]
    data = UnifiedMEAData(
        spikes={
            "chan1": np.array([0.010, 0.018, 1.010, 1.018, 2.010, 2.018]),
            "chan2": np.array([0.014, 0.022, 1.014, 1.022, 2.014, 2.022]),
        },
        sr=20000.0,
    )
    import src.gui.app as app_module

    def fake_pivae(raw_observed, latent_dim=16, time_bin_ms=10.0, cancel_check=None):
        raw = np.asarray(raw_observed, dtype=float)
        latent = np.repeat(np.mean(raw, axis=2, keepdims=True), int(latent_dim), axis=2)
        params = {
            "method": "pi_vae",
            "latent_dim": int(latent_dim),
            "loadings": np.ones((int(latent_dim), raw.shape[2]), dtype=float),
            "mean": np.mean(raw.reshape((-1, raw.shape[2])), axis=0),
            "n_iter": 1,
        }
        return latent, raw.copy(), params

    monkeypatch.setattr(app_module, "_stimulus_response_supported_files", lambda _paths: [path])
    monkeypatch.setattr(app_module, "_load_spike_only_data", lambda _path, cancel_check=None: data)
    monkeypatch.setattr(app_module, "_detect_burst_intervals", lambda *args, **kwargs: intervals)
    monkeypatch.setattr(app_module, "_pivae_latent_states", fake_pivae)

    payload = _multi_file_factor_analysis_payload(
        [path],
        time_bin_ms=10.0,
        window_ms=40.0,
        model_method="pivae",
        latent_dim=2,
        min_total_activity=0.0,
        min_active_bursts=1,
        max_channels=2,
        artifact_ms=0.0,
    )

    assert payload["model_method"] == "pivae"
    analysis = payload["records"][0]["analysis"]
    assert analysis["model_method"] == "pivae"
    assert analysis["latent_params"]["method"] == "pi_vae"
    assert "pi-VAE" in analysis["state_projection"]
    assert np.asarray(analysis["latent_states"], dtype=float).shape[-1] == 2
    assert np.asarray(analysis["raw_reconstructed_states"], dtype=float).shape == np.asarray(analysis["raw_observed_states"], dtype=float).shape


def test_multi_file_factor_analysis_window_visualizes_aligned_w_similarity():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    spike_series = [
        ("chan1", np.array([0.005, 0.018, 1.006, 1.019, 2.005, 2.018])),
        ("chan2", np.array([0.012, 0.028, 1.013, 1.029, 2.030, 2.041])),
        ("chan3", np.array([0.035, 1.034, 2.012])),
    ]
    intervals = [(0.0, 0.05), (1.0, 1.05), (2.0, 2.05)]
    analysis_a = _burst_trajectory_analysis(spike_series, intervals, time_bin_ms=10.0, window_ms=50.0, latent_dim=2, max_channels=3)
    analysis_b = _burst_trajectory_analysis(spike_series, intervals, time_bin_ms=10.0, window_ms=50.0, latent_dim=2, max_channels=3)
    weights = [
        np.asarray(analysis_a["latent_params"]["loadings"], dtype=float),
        np.asarray(analysis_b["latent_params"]["loadings"], dtype=float),
    ]
    similarity, aligned = _aligned_weight_similarity(weights)
    payload = {
        "records": [
            {
                "condition": "amp=1",
                "file": "a.npz",
                "analysis": analysis_a,
                "aligned_loadings": aligned[0],
                "burst_count": len(intervals),
                "selected_channel_count": len(analysis_a["selected_labels"]),
                "reconstruction_r2": analysis_a["reconstruction_r2"],
            },
            {
                "condition": "amp=2",
                "file": "b.npz",
                "analysis": analysis_b,
                "aligned_loadings": aligned[1],
                "burst_count": len(intervals),
                "selected_channel_count": len(analysis_b["selected_labels"]),
                "reconstruction_r2": analysis_b["reconstruction_r2"],
            },
        ],
        "errors": [],
        "w_similarity": similarity,
        "window_ms": 50.0,
        "model_method": "fa",
    }

    app = QApplication.instance() or QApplication([])
    window = MultiFileFactorAnalysisWindow(payload)

    assert "Model: FA" in window.summary.text()
    assert "Aligned W correlation" in window.similarity_canvas.figure.axes[0].get_title()
    assert "Selected aligned W" in window.weight_canvas.figure.axes[1].get_title()
    assert "Mean z(t)" in window.latent_canvas.figure.axes[0].get_title()
    assert "Reconstruction R2" in window.performance_canvas.figure.axes[0].get_title()


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


def test_loaded_data_activity_label_detects_stimulus_and_spontaneous_files():
    spontaneous = UnifiedMEAData(spikes={"chan1": np.array([0.1])}, sr=20000.0)
    stimulus = UnifiedMEAData(spikes={"chan1": np.array([0.1])}, stim_times=np.array([1.0]), sr=20000.0)

    assert _loaded_data_activity_label(r"C:\data\spont_pre\trial.npz", stimulus) == "Spontaneous"
    assert _loaded_data_activity_label("recording_without_events.npz", spontaneous) == "Spontaneous"
    assert _loaded_data_activity_label("recording_without_events.npz", stimulus) == "Stimulus"
    assert _loaded_data_activity_label(r"C:\data\el=12_after\trial.h5", spontaneous) == "Stimulus"


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
    from src.gui import visual_stimulus_package_builder as stimulus_builder

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
    assert not hasattr(dialog, "open_raster_button")
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


def test_multi_file_factor_analysis_results_return_to_cached_dialog():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    main_window = MainWindow()
    dialog = main_window._multi_file_fa_analysis_dialog()
    dialog.show()
    spike_series = [
        ("chan1", np.array([0.005, 0.018, 1.006, 1.019, 2.005, 2.018])),
        ("chan2", np.array([0.012, 0.028, 1.013, 1.029, 2.030, 2.041])),
    ]
    intervals = [(0.0, 0.05), (1.0, 1.05), (2.0, 2.05)]
    analysis = _burst_trajectory_analysis(spike_series, intervals, time_bin_ms=10.0, window_ms=50.0, latent_dim=1, max_channels=2)
    loadings = np.asarray(analysis["latent_params"]["loadings"], dtype=float)
    payload = {
        "records": [
            {
                "condition": "file a",
                "file": "a.npz",
                "analysis": analysis,
                "aligned_loadings": loadings,
                "burst_count": len(intervals),
                "selected_channel_count": len(analysis["selected_labels"]),
                "reconstruction_r2": analysis["reconstruction_r2"],
            }
        ],
        "errors": [],
        "w_similarity": np.eye(1),
        "window_ms": 50.0,
        "model_method": "fa",
    }
    worker = type("Worker", (), {"_is_cancelled": lambda self: False})()

    main_window._multi_file_fa_finished(payload, worker)
    app.processEvents()

    assert main_window.multi_file_fa_payload is payload
    assert dialog.cached_payload is payload
    assert dialog.open_result_button.isEnabled()
    assert not dialog.isVisible()
    assert any(isinstance(child, MultiFileFactorAnalysisWindow) for child in main_window.child_windows)

    result_window = next(child for child in main_window.child_windows if isinstance(child, MultiFileFactorAnalysisWindow))
    result_window.close()
    app.processEvents()

    assert dialog.isVisible()
    for child in list(main_window.child_windows):
        child.close()
    dialog.close()
    main_window.close()


def test_database_analysis_dialog_preserves_manual_subset_selection_on_refresh():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    records = [
        {"path": "a.nev", "raw_data": UnifiedMEAData(spikes={"chan1": np.array([0.1])}, sr=30000.0), "data_kind": "nev"},
        {"path": "b.nev", "raw_data": UnifiedMEAData(spikes={"chan1": np.array([0.2])}, sr=30000.0), "data_kind": "nev"},
        {"path": "c.nev", "raw_data": UnifiedMEAData(spikes={"chan1": np.array([0.3])}, sr=30000.0), "data_kind": "nev"},
    ]

    dialog = FactorAnalysisDatabaseDialog(records)
    dialog.table.clearSelection()
    dialog.table.selectRow(1)
    app.processEvents()

    assert [Path(path).name for path in dialog.values()[0]] == ["b.nev"]

    dialog._set_records(records)
    app.processEvents()

    assert [Path(path).name for path in dialog.values()[0]] == ["b.nev"]
    dialog.close()


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
    assert window.raster_action_combo.findData("burst_trajectory") == -1
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


def test_spike_raster_channel_selection_reuses_raster_cache():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from src.gui.app import SpikeRasterWindow
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    series = [(f"chan{index}", np.linspace(0.0, 2.0, 200)) for index in range(1, 8)]
    window = SpikeRasterWindow("Raster", series)
    window.canvas.resize(900, 500)

    window.canvas.grab()
    first_key = window.canvas._raster_cache_key
    first_cache = window.canvas._raster_cache

    window._select_channel("chan4")
    window.canvas.grab()

    assert first_key is not None
    assert window.canvas._raster_cache_key == first_key
    assert window.canvas._raster_cache is first_cache


def test_spike_raster_can_filter_axion_rows_by_well():
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
            ("A1_r1c1", np.array([0.0, 0.2])),
            ("A1_r1c2", np.array([0.1])),
            ("B1_r1c1", np.array([0.3])),
        ],
        channel_groups={
            "A1": ["A1_r1c1", "A1_r1c2"],
            "B1": ["B1_r1c1"],
        },
    )

    assert window.well_combo is not None
    assert [label for label, _ in window.spike_series] == ["A1_r1c1", "A1_r1c2", "B1_r1c1"]

    window.well_combo.setCurrentIndex(window.well_combo.findData("B1"))

    assert [label for label, _ in window.spike_series] == ["B1_r1c1"]
    assert window._window_channel_counts(0.0, 1.0) == {"B1_r1c1": 1}
    assert "all wells" not in window.raster_settings_summary.text().lower()


def test_main_window_uses_generated_axion_map_instead_of_previous_map():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    from src.gui import visual_stimulus_package_builder as stimulus_builder

    window = MainWindow()
    window.channel_map = ChannelMap.new("old_map")
    window.raw_data = UnifiedMEAData(
        spikes={"A1_r1c1": np.array([0.1])},
        meta={
            "source": "axion_spk",
            "wells": ["A1"],
            "channel_map": {
                "A1_r1c1": {"well": "A1", "electrode": "r1c1", "electrode_row": 1, "electrode_col": 1},
            },
        },
    )

    window._apply_source_channel_map()

    assert window.channel_map is not None
    assert window.channel_map.name == "axion_map"
    assert window.channel_map.rows == 6
    assert window.channel_map.cols == 64
    assert window.channel_map.channel_for("A1_slot01") == "A1_r1c1"


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


def test_spike_raster_window_channel_counts_cache_preserves_results():
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
            ("chan1 unit 0", np.array([0.10, 0.20])),
            ("chan1 unit 1", np.array([0.15])),
            ("chan2", np.array([0.12, 0.40])),
        ],
    )

    first = window._window_channel_counts(0.0, 0.25)
    second = window._window_channel_counts(0.0, 0.25)

    assert first == {"chan1": 3, "chan2": 1}
    assert second == first


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
        from PySide6.QtWidgets import QApplication, QProgressBar, QPushButton
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert not window.findChildren(QProgressBar)
    assert not hasattr(window, "cards")
    assert not hasattr(window, "run_button")
    assert not hasattr(window, "results_button")
    assert not hasattr(window, "settings_button")
    assert not hasattr(window, "temporal_button")
    button_texts = [button.text() for button in window.findChildren(QPushButton)]
    expected_button_order = [
        "Open Data Files",
        "Raw Data Raster",
        "Save File",
        "Channel Map",
        "Sorting",
        "Stimulus Response Analysis",
        "Dynamics Analysis",
    ]
    assert button_texts[: len(expected_button_order)] == expected_button_order
    assert "Run Full Pipeline" not in button_texts
    assert "Open Results" not in button_texts
    assert "Settings" not in button_texts
    assert "Temporal Coupling" not in button_texts
    assert "Sort by Label" not in button_texts
    menu_texts = [
        action.text()
        for menu_action in window.menuBar().actions()
        for action in (menu_action.menu().actions() if menu_action.menu() is not None else [])
    ]
    assert "Results" not in menu_texts
    assert "Settings" not in menu_texts
    assert "Temporal Coupling" not in menu_texts
    assert "No data loaded" in window.data_preview.toPlainText()
    assert window.database_table.columnCount() == 7
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
    assert "Kind: blackrock_nev" in preview
    assert "Total spikes: 3" in preview
    assert "chan1: 2 spikes" in preview


def test_main_and_dialog_windows_enable_maximize_controls():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    main_window = MainWindow()
    dialog = DataFilesInputDialog()

    assert bool(main_window.windowFlags() & Qt.WindowType.WindowMaximizeButtonHint)
    assert bool(dialog.windowFlags() & Qt.WindowType.WindowMaximizeButtonHint)

    dialog.close()
    main_window.close()


def test_main_window_file_database_selection_feeds_single_and_multi_file_actions(tmp_path):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    first_path = tmp_path / "first.nev"
    second_path = tmp_path / "el=12_after.raw.h5"
    first_path.write_bytes(b"placeholder")
    second_path.write_bytes(b"placeholder")
    first = UnifiedMEAData(
        spikes={"chan1": np.array([0.1, 0.2])},
        sr=30000.0,
        meta={"source": "blackrock_nev"},
    )
    second = UnifiedMEAData(
        spikes={"chan2": np.array([0.3, 0.4, 0.5])},
        stim_times=np.array([0.25]),
        sr=20000.0,
        meta={"source": "maxwell_h5", "waveforms_deferred": True},
    )
    window.file_database = [
        {"path": str(first_path), "raw_data": first, "data_kind": "nev"},
        {"path": str(second_path), "raw_data": second, "data_kind": "nev"},
    ]
    window._refresh_file_database_table()
    window.database_table.selectRow(1)
    app.processEvents()

    assert window.input_path == str(second_path)
    assert window.raw_data is second
    assert "el=12_after.raw.h5" in window.data_preview.toPlainText()
    assert "Kind: maxwell_h5" in window.data_preview.toPlainText()
    assert "deferred for faster loading" in window.data_preview.toPlainText()
    assert window.database_table.item(1, 1).text() == "maxwell_h5"
    assert window.database_table.item(0, 2).text() == "Spontaneous"
    assert window.database_table.item(1, 2).text() == "Stimulus"

    window._database_header_clicked(2)
    assert Path(window.file_database[0]["path"]).name == "el=12_after.raw.h5"
    assert window.database_table.item(0, 2).text() == "Stimulus"
    assert window.database_sort_column == 2

    window._database_header_clicked(2)
    assert Path(window.file_database[0]["path"]).name == "first.nev"
    assert window.database_sort_column is None

    window._database_header_clicked(2)
    assert Path(window.file_database[0]["path"]).name == "el=12_after.raw.h5"
    window._database_header_clicked(4)
    assert Path(window.file_database[0]["path"]).name == "first.nev"
    assert window.database_sort_column == 4

    window.database_table.selectAll()
    app.processEvents()
    window.open_stimulus_response_analysis()
    window.open_multi_file_factor_analysis()

    stimulus_paths = window.stimulus_response_dialog.values()[0]
    fa_paths = window.multi_file_fa_dialog.values()[0]
    assert sorted(Path(path).name for path in stimulus_paths) == ["el=12_after.raw.h5", "first.nev"]
    assert sorted(Path(path).name for path in fa_paths) == ["el=12_after.raw.h5", "first.nev"]
    stimulus_labels = [window.stimulus_response_dialog.table.item(row, 2).text() for row in range(2)]
    fa_labels = [window.multi_file_fa_dialog.table.item(row, 2).text() for row in range(2)]
    assert sorted(stimulus_labels) == ["Spontaneous", "Stimulus"]
    assert sorted(fa_labels) == ["Spontaneous", "Stimulus"]
    window.stimulus_response_dialog.close()
    window.multi_file_fa_dialog.close()
    window.close()


def test_dynamics_database_dialog_exposes_pivae_model():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    dialog = FactorAnalysisDatabaseDialog([])
    try:
        index = dialog.model_method.findData("pivae")
        assert index >= 0
        dialog.model_method.setCurrentIndex(index)
        _paths, parameters = dialog.values()
        assert parameters["model_method"] == "pivae"
    finally:
        dialog.close()
        app.processEvents()


def test_stimulus_database_dialog_sorts_by_label_column(tmp_path):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    spontaneous_path = tmp_path / "baseline.nev"
    stimulus_path = tmp_path / "el=12_after.raw.h5"
    spontaneous_path.write_bytes(b"placeholder")
    stimulus_path.write_bytes(b"placeholder")
    records = [
        {
            "path": str(spontaneous_path),
            "raw_data": UnifiedMEAData(spikes={"chan1": np.array([0.1])}, meta={"source": "blackrock_nev"}),
            "data_kind": "nev",
        },
        {
            "path": str(stimulus_path),
            "raw_data": UnifiedMEAData(
                spikes={"chan2": np.array([0.2, 0.3])},
                stim_times=np.array([0.15]),
                meta={"source": "maxwell_h5"},
            ),
            "data_kind": "nev",
        },
    ]

    dialog = StimulusDatabaseAnalysisDialog(records)
    dialog.table.clearSelection()
    dialog.table.selectRow(0)
    app.processEvents()

    dialog._database_header_clicked(2)
    assert Path(dialog.records[0]["path"]).name == "el=12_after.raw.h5"
    assert dialog.table.item(0, 2).text() == "Stimulus"
    assert dialog.database_sort_column == 2
    assert [Path(path).name for path in dialog.values()[0]] == ["baseline.nev"]

    dialog._database_header_clicked(2)
    assert Path(dialog.records[0]["path"]).name == "baseline.nev"
    assert dialog.database_sort_column is None
    dialog.close()


def test_stimulus_database_dialog_preserves_multi_selection_on_refresh(tmp_path):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import QItemSelectionModel
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    paths = [tmp_path / f"stim_{index}.nev" for index in range(3)]
    for path in paths:
        path.write_bytes(b"placeholder")
    records = [
        {
            "path": str(path),
            "raw_data": UnifiedMEAData(
                spikes={"chan1": np.array([0.01, 0.02])},
                stim_times=np.array([0.015]),
                meta={"source": "blackrock_nev"},
            ),
            "data_kind": "nev",
        }
        for path in paths
    ]

    dialog = StimulusDatabaseAnalysisDialog(records)
    selection_model = dialog.table.selectionModel()
    dialog.table.clearSelection()
    flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    selection_model.select(dialog.table.model().index(0, 0), flags)
    selection_model.select(dialog.table.model().index(2, 0), flags)
    app.processEvents()

    assert sorted(Path(path).name for path in dialog.values()[0]) == ["stim_0.nev", "stim_2.nev"]
    assert dialog.selected_count_label.text() == "Selected files: 2 / 3"

    dialog._set_records(records)
    app.processEvents()

    assert sorted(Path(path).name for path in dialog.values()[0]) == ["stim_0.nev", "stim_2.nev"]
    assert dialog.selected_count_label.text() == "Selected files: 2 / 3"
    dialog.close()


def test_main_window_stimulus_response_start_uses_all_selected_dialog_paths(monkeypatch, tmp_path):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import QItemSelectionModel
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    paths = [tmp_path / f"stim_start_{index}.nev" for index in range(3)]
    for path in paths:
        path.write_bytes(b"placeholder")
    window.file_database = [
        {
            "path": str(path),
            "raw_data": UnifiedMEAData(
                spikes={"chan1": np.array([0.01, 0.02])},
                stim_times=np.array([0.015]),
                meta={"source": "blackrock_nev"},
            ),
            "data_kind": "nev",
        }
        for path in paths
    ]
    dialog = window._stimulus_response_analysis_dialog()
    selection_model = dialog.table.selectionModel()
    dialog.table.clearSelection()
    flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    selection_model.select(dialog.table.model().index(0, 0), flags)
    selection_model.select(dialog.table.model().index(1, 0), flags)
    app.processEvents()

    captured = {}

    class FakeProgress:
        canceled = type("SignalStub", (), {"connect": lambda self, _callback: None})()

    class FakeWorker:
        def __init__(self, paths_arg, **kwargs):
            captured["paths"] = list(paths_arg)
            captured["kwargs"] = dict(kwargs)
            self.signals = type(
                "SignalsStub",
                (),
                {
                    "progress": type("SignalStub", (), {"connect": lambda self, _callback: None})(),
                    "finished": type("SignalStub", (), {"connect": lambda self, _callback: None})(),
                    "failed": type("SignalStub", (), {"connect": lambda self, _callback: None})(),
                    "canceled": type("SignalStub", (), {"connect": lambda self, _callback: None})(),
                },
            )()

        def cancel(self):
            pass

    monkeypatch.setattr(window, "_start_progress", lambda *args, **kwargs: FakeProgress())
    monkeypatch.setattr(window.thread_pool, "start", lambda worker: captured.setdefault("worker", worker))
    monkeypatch.setattr(gui_app, "StimulusResponseWorker", FakeWorker)

    window._start_stimulus_response_from_dialog()

    assert sorted(Path(path).name for path in captured["paths"]) == ["stim_start_0.nev", "stim_start_1.nev"]
    assert window.active_stimulus_worker is captured["worker"]
    dialog.close()
    window.close()


def test_data_files_input_dialog_adds_folders_and_removes_selected(tmp_path):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    folder = tmp_path / "dataset"
    folder.mkdir()
    h5_path = folder / "a.raw.h5"
    nev_path = folder / "b.nev"
    ignored = folder / "notes.md"
    h5_path.write_bytes(b"placeholder")
    nev_path.write_bytes(b"placeholder")
    ignored.write_text("ignore", encoding="utf-8")

    dialog = DataFilesInputDialog()
    dialog._add_paths([folder, h5_path])

    assert sorted(Path(path).name for path in dialog.values()) == ["a.raw.h5", "b.nev"]
    assert dialog.table.rowCount() == 2

    dialog.table.selectRow(0)
    dialog._remove_selected()

    assert len(dialog.values()) == 1
    assert Path(dialog.values()[0]).name in {"a.raw.h5", "b.nev"}
    dialog.close()


def test_generic_analysis_matrix_from_record_supports_unified_and_array_data():
    unified = UnifiedMEAData(
        spikes={
            "chan1": np.array([0.01, 0.03, 0.05]),
            "chan2": np.array([0.02, 0.04]),
        },
        sr=20000.0,
    )
    matrix, labels, description = _generic_analysis_matrix_from_record({"raw_data": unified}, bin_ms=10.0)
    assert matrix.shape[0] == 2
    assert labels == ["chan1", "chan2"]
    assert "Channel x time-bin spike-count matrix" in description

    time_matrix, time_labels, time_description = _generic_analysis_matrix_from_record(
        {"raw_data": unified},
        view_mode="time_channel",
        bin_ms=10.0,
    )
    assert time_matrix.shape[1] == 2
    assert time_labels
    assert "Time-bin x channel" in time_description

    burst_matrix, burst_labels, burst_description = _generic_analysis_matrix_from_record(
        {"raw_data": unified},
        view_mode="burst_flat",
        bin_ms=10.0,
        burst_window_ms=100.0,
        burst_threshold_z=0.5,
    )
    assert burst_matrix.ndim == 2
    assert burst_labels is not None
    assert "Burst x flattened" in burst_description

    array_matrix, array_labels, array_description = _generic_analysis_matrix_from_record({"raw_data": np.arange(12, dtype=float).reshape(3, 4)}, bin_ms=10.0)
    assert array_matrix.shape == (3, 4)
    assert array_labels == ["row 1", "row 2", "row 3"]
    assert "Array matrix" in array_description

    array_columns, array_column_labels, array_column_description = _generic_analysis_matrix_from_record(
        {"raw_data": np.arange(12, dtype=float).reshape(3, 4)},
        array_axis="columns",
        bin_ms=10.0,
    )
    assert array_columns.shape == (4, 3)
    assert array_column_labels == ["column 1", "column 2", "column 3", "column 4"]
    assert "samples=columns" in array_column_description


def test_custom_spike_vector_matrix_compares_time_window_rates():
    data = UnifiedMEAData(
        spikes={
            "chan1": np.array([0.1, 0.2, 1.2], dtype=float),
            "chan2": np.array([0.4, 1.1, 1.3, 1.4], dtype=float),
        },
        sr=20000.0,
    )
    windows = _parse_custom_time_windows("0-1, 1-2", data)
    selected = _custom_channel_filter(_spike_series_from_unified(data), "chan1, chan2")
    matrix, sample_labels, feature_labels, description = _custom_spike_vector_matrix(selected, windows, "firing_rate_vector")

    assert sample_labels == ["0-1s", "1-2s"]
    assert feature_labels == ["chan1", "chan2"]
    assert matrix.tolist() == [[2.0, 1.0], [1.0, 3.0]]
    assert "firing rate" in description


def test_custom_plot_auto_uses_feature_labels_for_single_window_vectors():
    record = {
        "name": "rates",
        "dataset_type": "firing_rate_vector",
        "matrix": np.array([[2.0, 4.0, 6.0]], dtype=float),
        "sample_labels": ["0-1s"],
        "feature_labels": ["chan1", "chan2", "chan10"],
    }
    assert _custom_plot_axis_mode(record, "auto") == "features"
    x, labels, x_label, transpose, matrix = _custom_plot_x_axis(record, {"x_mode": "auto"})
    assert x.tolist() == [1.0, 2.0, 3.0]
    assert labels == ["chan1", "chan2", "chan10"]
    assert x_label == "Channel / feature"
    assert transpose is True
    assert matrix.shape == (1, 3)


def test_custom_plot_auto_uses_sample_labels_for_multi_window_vectors():
    record = {
        "name": "rates",
        "matrix": np.array([[2.0, 4.0], [1.0, 3.0]], dtype=float),
        "sample_labels": ["0-1s", "1-2s"],
        "feature_labels": ["chan1", "chan2"],
    }
    assert _custom_plot_axis_mode(record, "auto") == "sample_labels"
    x, labels, x_label, transpose, _matrix = _custom_plot_x_axis(record, {"x_mode": "auto"})
    assert x.tolist() == [1.0, 2.0]
    assert labels == ["0-1s", "1-2s"]
    assert x_label == "Sample"
    assert transpose is False


def test_custom_plot_window_labels_channels_on_x_axis():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    record = {
        "name": "rates",
        "dataset_type": "firing_rate_vector",
        "matrix": np.array([[2.0, 4.0, 6.0]], dtype=float),
        "sample_labels": ["0-1s"],
        "feature_labels": ["chan1", "chan2", "chan10"],
    }
    window = CustomPlotWindow([record], {"x_mode": "auto", "plot_mode": "auto", "y_label": "Hz"})
    try:
        ax = window.canvas.figure.axes[0]
        assert ax.get_xlabel() == "Channel / feature"
        assert [tick.get_text() for tick in ax.get_xticklabels()[:3]] == ["chan1", "chan2", "chan10"]
    finally:
        window.close()
        app.processEvents()


def test_custom_plot_generated_x_axis_handles_long_sequences():
    record = {
        "matrix": np.arange(10, dtype=float).reshape(10, 1),
        "sample_labels": [f"w{index}" for index in range(10)],
        "feature_labels": ["chan1"],
    }
    x, labels, x_label, transpose, _matrix = _custom_plot_x_axis(
        record,
        {"x_mode": "generated", "x_start": 0.5, "x_step": 0.25},
    )
    assert x[:4].tolist() == [0.5, 0.75, 1.0, 1.25]
    assert labels == []
    assert x_label == "Generated x"
    assert transpose is False


def test_custom_plot_dialog_shows_x_controls_only_for_matching_modes():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    dialog = CustomPlotDialog([{"matrix": np.ones((3, 1), dtype=float)}])
    try:
        assert not hasattr(dialog, "table")
        assert not hasattr(dialog, "data_combo")
        assert dialog.y_data_combos[0].count() == 1
        assert str(dialog.y_data_combos[0].itemData(0)) == "record:0|dataset"
        assert dialog.y_data_combos[0].maximumWidth() <= 140
        assert dialog.x_mode.findData("sample_index") == -1
        assert not hasattr(dialog, "sample_filter")
        assert not hasattr(dialog, "save_extracted")
        assert not hasattr(dialog, "extracted_name")
        assert dialog.x_start.isHidden()
        assert dialog.x_step.isHidden()
        assert dialog.x_values.isHidden()
        dialog.x_mode.setCurrentIndex(dialog.x_mode.findData("generated"))
        dialog._update_x_controls()
        assert not dialog.x_start.isHidden()
        assert not dialog.x_step.isHidden()
        assert dialog.x_values.isHidden()
        dialog.x_mode.setCurrentIndex(dialog.x_mode.findData("manual"))
        dialog._update_x_controls()
        assert dialog.x_start.isHidden()
        assert dialog.x_step.isHidden()
        assert not dialog.x_values.isHidden()
    finally:
        dialog.close()
        app.processEvents()


def test_custom_plot_dialog_draws_result_inside_parameter_window():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    record = {
        "name": "rates with a deliberately long processed dataset name",
        "dataset_type": "firing_rate_vector",
        "matrix": np.array([[2.0, 4.0]], dtype=float),
        "sample_labels": ["0-1s"],
        "feature_labels": ["chan1", "chan2"],
    }
    dialog = CustomPlotDialog([record])
    try:
        assert "rates" in dialog.y_data_combos[0].currentText()
        assert dialog.y_data_combos[0].itemData(0, Qt.ItemDataRole.ToolTipRole) == record["name"]
        assert dialog.canvas.figure.axes == []
        dialog._draw_preview()
        assert dialog.canvas.figure.axes
        assert "Plotted datasets: 1" in dialog.summary.text()
    finally:
        dialog.close()
        app.processEvents()


def test_custom_plot_dialog_adds_multiple_y_data_series():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    records = [
        {
            "name": "rates A",
            "dataset_type": "firing_rate_vector",
            "matrix": np.array([[2.0], [3.0]], dtype=float),
            "sample_labels": ["0-1s", "1-2s"],
            "feature_labels": ["chan1"],
        },
        {
            "name": "rates B",
            "dataset_type": "firing_rate_vector",
            "matrix": np.array([[4.0], [5.0]], dtype=float),
            "sample_labels": ["0-1s", "1-2s"],
            "feature_labels": ["chan1"],
        },
    ]
    dialog = CustomPlotDialog(records)
    try:
        assert dialog.y_data_combos[0].count() == 2
        dialog._add_y_data_combo("record:1|dataset")
        dialog.y_data_combos[0].setCurrentIndex(dialog.y_data_combos[0].findData("record:0|dataset"))
        dialog.y_data_combos[1].setCurrentIndex(dialog.y_data_combos[1].findData("record:1|dataset"))
        dialog.plot_mode.setCurrentIndex(dialog.plot_mode.findData("line"))
        dialog._draw_preview()
        ax = dialog.canvas.figure.axes[0]
        assert len(ax.get_lines()) == 2
        assert ax.get_lines()[0].get_ydata().tolist() == [2.0, 3.0]
        assert ax.get_lines()[1].get_ydata().tolist() == [4.0, 5.0]
    finally:
        dialog.close()
        app.processEvents()


def test_custom_plot_dataset_y_options_do_not_expand_channels():
    records = [
        {
            "name": "rates",
            "dataset_type": "firing_rate_vector",
            "matrix": np.array([[2.0, 4.0], [3.0, 5.0]], dtype=float),
            "sample_labels": ["0-1s", "1-2s"],
            "feature_labels": ["chan1", "chan2"],
        }
    ]
    assert _custom_plot_dataset_y_options(records) == [("record:0|dataset", "rates")]


def test_custom_plot_multi_y_bar_offsets_do_not_overlap():
    figure = Figure(figsize=(5, 3), constrained_layout=True)
    record = {
        "name": "rates",
        "matrix": np.array([[2.0, 4.0], [3.0, 5.0]], dtype=float),
        "sample_labels": ["0-1s", "1-2s"],
        "feature_labels": ["chan1", "chan2"],
    }
    plotted = _draw_custom_plot_figure(
        figure,
        [record],
        {"x_mode": "sample_labels", "plot_mode": "bar", "y_data_keys": ["feature:0", "feature:1"], "plot_style": "multi_y"},
    )
    ax = figure.axes[0]
    assert plotted == 2
    centers = [patch.get_x() + patch.get_width() / 2.0 for patch in ax.patches[:4]]
    assert centers[0] != pytest.approx(centers[2])
    assert centers[1] != pytest.approx(centers[3])


def test_custom_plot_y_data_selects_dataset_and_channel():
    figure = Figure(figsize=(5, 3), constrained_layout=True)
    records = [
        {
            "name": "file A",
            "matrix": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            "sample_labels": ["0-1s", "1-2s"],
            "feature_labels": ["chan1", "chan2"],
        },
        {
            "name": "file B",
            "matrix": np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float),
            "sample_labels": ["0-1s", "1-2s"],
            "feature_labels": ["chan1", "chan2"],
        },
    ]
    plotted = _draw_custom_plot_figure(
        figure,
        records,
        {"x_mode": "sample_labels", "plot_mode": "line", "y_data_keys": ["record:0|dataset", "record:1|dataset"], "plot_style": "multi_y"},
    )
    lines = figure.axes[0].get_lines()
    assert plotted == 4
    assert lines[0].get_ydata().tolist() == [1.0, 3.0]
    assert lines[1].get_ydata().tolist() == [2.0, 4.0]
    assert lines[2].get_ydata().tolist() == [10.0, 30.0]
    assert lines[3].get_ydata().tolist() == [20.0, 40.0]


def test_custom_plot_generated_axis_defaults_to_one_step_one():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    record = {
        "matrix": np.array([[1.0], [2.0], [3.0]], dtype=float),
        "sample_labels": ["a", "b", "c"],
        "feature_labels": ["chan1"],
    }
    window = CustomPlotWindow([record], {"x_mode": "generated", "plot_mode": "bar", "y_label": "Hz"})
    try:
        ax = window.canvas.figure.axes[0]
        ticks = [float(value) for value in ax.get_xticks()[:3]]
        assert ticks == [1.0, 2.0, 3.0]
        assert ax.get_xlim()[0] == pytest.approx(1.0)
        assert ax.get_xlim()[1] == pytest.approx(3.0)
    finally:
        window.close()
        app.processEvents()


def test_custom_plot_generated_axis_pairs_feature_vector_values():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    record = {
        "matrix": np.array([[10.0, 20.0, 30.0]], dtype=float),
        "sample_labels": ["0-1s"],
        "feature_labels": ["chan1", "chan2", "chan3"],
    }
    x, labels, _x_label, transpose, matrix = _custom_plot_x_axis(record, {"x_mode": "generated"})
    assert x.tolist() == [1.0, 2.0, 3.0]
    assert labels == []
    assert transpose is True
    assert matrix[0, :].tolist() == [10.0, 20.0, 30.0]

    app = QApplication.instance() or QApplication([])
    window = CustomPlotWindow([record], {"x_mode": "generated", "plot_mode": "bar"})
    try:
        ax = window.canvas.figure.axes[0]
        bar_centers = [patch.get_x() + patch.get_width() / 2.0 for patch in ax.patches[:3]]
        heights = [patch.get_height() for patch in ax.patches[:3]]
        assert bar_centers == pytest.approx([1.0, 2.0, 3.0])
        assert heights == pytest.approx([10.0, 20.0, 30.0])
    finally:
        window.close()
        app.processEvents()


def test_custom_extract_processed_record_filters_same_channel_across_files():
    first = {
        "name": "file A",
        "path": "a::processed",
        "matrix": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        "sample_labels": ["0-1s", "1-2s"],
        "feature_labels": ["chan1", "chan2"],
    }
    second = {
        "name": "file B",
        "path": "b::processed",
        "matrix": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
        "sample_labels": ["0-1s", "1-2s"],
        "feature_labels": ["chan1", "chan2"],
    }
    params = {"feature_filter": "chan2", "sample_filter": "1-2s", "apply_filters": True}
    extracted = [_custom_extract_processed_record(record, params) for record in [first, second]]
    assert [record["feature_labels"] for record in extracted] == [["chan2"], ["chan2"]]
    assert [record["sample_labels"] for record in extracted] == [["1-2s"], ["1-2s"]]
    assert [np.asarray(record["matrix"]).tolist() for record in extracted] == [[[4.0]], [[8.0]]]


def test_custom_plot_window_compares_same_channel_from_multiple_processed_files():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    records = [
        {
            "name": "file A",
            "matrix": np.array([[2.0], [4.0]], dtype=float),
            "sample_labels": ["0-1s", "1-2s"],
            "feature_labels": ["chan2"],
        },
        {
            "name": "file B",
            "matrix": np.array([[3.0], [6.0]], dtype=float),
            "sample_labels": ["0-1s", "1-2s"],
            "feature_labels": ["chan2"],
        },
    ]
    window = CustomPlotWindow(records, {"x_mode": "sample_labels", "plot_mode": "line", "y_label": "Hz"})
    try:
        ax = window.canvas.figure.axes[0]
        lines = ax.get_lines()
        assert len(lines) == 2
        assert lines[0].get_ydata().tolist() == [2.0, 4.0]
        assert lines[1].get_ydata().tolist() == [3.0, 6.0]
        assert [tick.get_text() for tick in ax.get_xticklabels()[:2]] == ["0-1s", "1-2s"]
    finally:
        window.close()
        app.processEvents()


def test_custom_plot_window_handles_many_long_labels_without_tight_layout_warning():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    labels = [f"very_long_channel_label_{index:03d}" for index in range(80)]
    record = {
        "name": "long labels",
        "matrix": np.arange(80, dtype=float).reshape(1, 80),
        "sample_labels": ["0-1s"],
        "feature_labels": labels,
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Tight layout not applied.*", category=UserWarning)
        window = CustomPlotWindow([record], {"x_mode": "features", "plot_mode": "bar"})
        try:
            window.canvas.draw()
        finally:
            window.close()
            app.processEvents()


def test_main_window_custom_plot_saves_extracted_processed_data(monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.processed_database = [
        {
            "name": "file A rates",
            "path": "a::rates",
            "source_path": "a.nev",
            "source_label": "Spontaneous",
            "dataset_type": "firing_rate_vector",
            "matrix": np.array([[1.0, 2.0]], dtype=float),
            "sample_labels": ["0-1s"],
            "feature_labels": ["chan1", "chan2"],
            "description": "rates",
        },
        {
            "name": "file B rates",
            "path": "b::rates",
            "source_path": "b.nev",
            "source_label": "Spontaneous",
            "dataset_type": "firing_rate_vector",
            "matrix": np.array([[3.0, 4.0]], dtype=float),
            "sample_labels": ["0-1s"],
            "feature_labels": ["chan1", "chan2"],
            "description": "rates",
        },
    ]
    window._refresh_processed_database_table()
    params = {
        "x_mode": "auto",
        "x_values": "",
        "x_start": 1.0,
        "x_step": 1.0,
        "x_label": "",
        "y_label": "Hz",
        "plot_mode": "auto",
        "sample_filter": "",
        "feature_filter": "chan2",
        "save_extracted": True,
        "extracted_name": "",
        "apply_filters": True,
    }
    records = [_custom_extract_processed_record(record, params) for record in window.processed_database]
    try:
        saved = window._save_custom_extracted_records(records, params)
        extracted = [record for record in window.processed_database if record.get("dataset_group") == "custom_extracted"]
        assert saved == 2
        assert len(extracted) == 2
        assert all(record["feature_labels"] == ["chan2"] for record in extracted)
        assert [np.asarray(record["matrix"]).tolist() for record in extracted] == [[[2.0]], [[4.0]]]
    finally:
        window.close()
        app.processEvents()


def test_main_window_custom_plot_opens_embedded_plot_dialog(monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.processed_database = [
        {
            "name": "rates",
            "path": "rates::processed",
            "dataset_type": "firing_rate_vector",
            "matrix": np.array([[1.0, 2.0]], dtype=float),
            "sample_labels": ["0-1s"],
            "feature_labels": ["chan1", "chan2"],
        }
    ]
    shown = []
    monkeypatch.setattr(window, "_show_child", lambda child: shown.append(child))
    try:
        window._open_custom_plot_dialog()
        assert len(shown) == 1
        assert isinstance(shown[0], CustomPlotDialog)
        assert hasattr(shown[0], "canvas")
        assert not hasattr(shown[0], "data_combo")
        assert shown[0].y_data_combos
        assert not hasattr(shown[0], "table")
        assert not isinstance(shown[0], CustomPlotWindow)
    finally:
        for child in shown:
            child.close()
        window.close()
        app.processEvents()


def test_generic_analysis_window_constructs_from_payload():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    payload = {
        "records": [
            {
                "file": "sample.nev",
                "condition": "Spontaneous",
                "matrix_description": "Spike-count matrix",
                "analysis": run_generic_matrix_analysis(
                    np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=float),
                    sample_axis=0,
                    normalization="feature_zscore",
                    similarity="correlation",
                    reduction="pca",
                    reduction_dims=2,
                    clustering="kmeans",
                    cluster_count=2,
                ),
            }
        ],
        "errors": [],
        "parameters": {
            "normalization": "feature_zscore",
            "similarity": "correlation",
            "reduction": "pca",
            "clustering": "kmeans",
        },
    }
    window = GenericAnalysisWindow(payload)
    try:
        assert "Files: 1" in window.summary.text()
        assert window.canvas.figure.axes
        assert "Samples x features" in window.detail.toPlainText()
    finally:
        window.close()
        app.processEvents()


def test_main_window_can_open_custom_analysis_from_database(tmp_path, monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    path = tmp_path / "sample.nev"
    path.write_bytes(b"placeholder")
    window.file_database = [
        {
            "path": str(path),
            "raw_data": UnifiedMEAData(
                spikes={
                    "chan1": np.array([0.01, 0.03, 0.05]),
                    "chan2": np.array([0.02, 0.04]),
                },
                sr=20000.0,
            ),
            "data_kind": "nev",
        }
    ]
    window._refresh_file_database_table()
    window.database_table.selectRow(0)
    window._set_active_database_index(0)
    dialog = window._generic_analysis_dialog()
    dialog.action = "firing_rate_vector"

    assert dialog.values()[1]["action"] == "firing_rate_vector"
    dialog.close()
    window.close()


def test_main_window_custom_analysis_adds_processed_rate_dataset(tmp_path, monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    path = tmp_path / "sample.nev"
    path.write_bytes(b"placeholder")
    window.file_database = [
        {
            "path": str(path),
            "raw_data": UnifiedMEAData(
                spikes={
                    "chan1": np.array([0.1, 0.2, 1.2], dtype=float),
                    "chan2": np.array([0.4, 1.1, 1.3, 1.4], dtype=float),
                },
                sr=20000.0,
            ),
            "data_kind": "nev",
        }
    ]
    window._refresh_file_database_table()
    import src.gui.app as gui_app

    monkeypatch.setattr(gui_app, "_show_info_message", lambda *args, **kwargs: None)
    payload = window._run_custom_analysis_records(
        [str(path)],
        {
            "analysis_kind": "custom_basic",
            "analysis_type": "firing_rate_vector",
            "time_windows": "0-1, 1-2",
            "channels": "chan1, chan2",
            "display_name": "",
            "x_values": "",
            "x_label": "Time window",
            "y_label": "Firing rate (Hz)",
            "plot_mode": "auto",
        },
        open_result=False,
    )

    assert len(window.processed_database) == 2
    assert window.processed_database[0]["dataset_type"] == "custom_raw_segment"
    processed = window.processed_database[-1]
    assert processed["dataset_group"] == "custom"
    assert processed["name"] == "sample.nev | firing_rate_vector | 2 windows | 2 channels"
    assert np.asarray(processed["matrix"], dtype=float).tolist() == [[2.0, 1.0], [1.0, 3.0]]
    assert payload["records"]
    window.close()
    app.processEvents()


def test_custom_data_selection_dialog_uses_main_raster_callback(tmp_path):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    spikes = {f"chan{index}": np.array([0.01 * index, 1.0 + 0.01 * index], dtype=float) for index in range(1, 201)}
    data = UnifiedMEAData(spikes=spikes, sr=20000.0)
    path = tmp_path / "many_channels.nev"
    path.write_bytes(b"placeholder")
    channel_map = ChannelMap(
        name="dense",
        rows=20,
        cols=10,
        electrodes={
            f"e{index}": {
                "channel": f"chan{index}",
                "electrode": index,
                "x_um": float(index % 10),
                "y_um": float(index // 10),
                "aliases": [f"chan{index}", str(index)],
                "routed": True,
            }
            for index in range(1, 201)
        },
    )
    opened = []
    dialog = CustomDataSelectionDialog(
        [{"path": str(path), "raw_data": data, "data_kind": "nev"}],
        "firing_rate_vector",
        channel_map,
        open_main_raster_callback=lambda path_text: opened.append(path_text),
    )
    try:
        selected_paths, params = dialog.values()
        assert selected_paths == [str(path)]
        assert len(params["channels"].split(", ")) == 200
        assert len(dialog.channel_to_electrode_cache) == 200
        assert not hasattr(dialog, "raster")
        dialog._open_main_raster()
        assert opened == [str(path)]
    finally:
        dialog.close()
        app.processEvents()


def test_custom_data_selection_channel_map_right_click_and_box_remove(tmp_path):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    data = UnifiedMEAData(
        spikes={
            "chan1": np.array([0.1], dtype=float),
            "chan2": np.array([0.2], dtype=float),
            "chan3": np.array([0.3], dtype=float),
        },
        sr=20000.0,
    )
    path = tmp_path / "map_select.nev"
    path.write_bytes(b"placeholder")
    channel_map = ChannelMap(
        name="small",
        rows=1,
        cols=3,
        electrodes={
            "e1": {"channel": "chan1", "electrode": 1, "x_um": 0.0, "y_um": 0.0, "aliases": ["chan1"], "routed": True},
            "e2": {"channel": "chan2", "electrode": 2, "x_um": 10.0, "y_um": 0.0, "aliases": ["chan2"], "routed": True},
            "e3": {"channel": "chan3", "electrode": 3, "x_um": 20.0, "y_um": 0.0, "aliases": ["chan3"], "routed": True},
        },
    )
    dialog = CustomDataSelectionDialog(
        [{"path": str(path), "raw_data": data, "data_kind": "nev"}],
        "firing_rate_vector",
        channel_map,
    )

    class Event:
        def __init__(self, x, y, button):
            self.inaxes = dialog.map_ax
            self.xdata = x
            self.ydata = y
            self.button = button

    try:
        e1_x, e1_y, _payload = dialog.electrode_positions["e1"]
        e2_x, e2_y, _payload = dialog.electrode_positions["e2"]
        dialog._map_clicked(Event(e1_x, e1_y, 3))
        assert "chan1" not in dialog.selected_channels
        dialog._map_box_selected(Event(0.0, 0.0, 3), Event(0.6, 1.0, 3))
        assert "chan2" not in dialog.selected_channels
        dialog._map_box_selected(Event(0.0, 0.0, 1), Event(0.6, 1.0, 1))
        assert {"chan1", "chan2"} <= dialog.selected_channels
    finally:
        dialog.close()
        app.processEvents()


def test_custom_data_selection_channel_map_left_click_highlights_electrode(tmp_path):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    data = UnifiedMEAData(
        spikes={
            "chan1": np.array([0.1], dtype=float),
            "chan2": np.array([0.2], dtype=float),
        },
        sr=20000.0,
    )
    path = tmp_path / "map_highlight.nev"
    path.write_bytes(b"placeholder")
    channel_map = ChannelMap(
        name="highlight",
        rows=1,
        cols=2,
        electrodes={
            "e1": {"channel": "chan1", "electrode": 1, "x_um": 0.0, "y_um": 0.0, "aliases": ["chan1"], "routed": True},
            "e2": {"channel": "chan2", "electrode": 2, "x_um": 10.0, "y_um": 0.0, "aliases": ["chan2"], "routed": True},
        },
    )
    dialog = CustomDataSelectionDialog(
        [{"path": str(path), "raw_data": data, "data_kind": "nev"}],
        "firing_rate_vector",
        channel_map,
    )

    class Event:
        def __init__(self, x, y):
            self.inaxes = dialog.map_ax
            self.xdata = x
            self.ydata = y
            self.button = 1

    try:
        e1_x, e1_y, _payload = dialog.electrode_positions["e1"]
        before_collections = len(dialog.map_ax.collections)
        dialog._map_clicked(Event(e1_x, e1_y))
        assert dialog.map_selected_electrode == "e1"
        assert dialog.map_state.get("selected") == "e1"
        assert len(dialog.map_ax.collections) > before_collections
        selected_collection = dialog.map_ax.collections[-1]
        assert selected_collection.get_sizes()[0] >= 90
    finally:
        dialog.close()
        app.processEvents()


def test_custom_data_selection_channel_map_wheel_zoom_preserves_view(tmp_path):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    data = UnifiedMEAData(
        spikes={f"chan{index}": np.array([0.1 * index], dtype=float) for index in range(1, 5)},
        sr=20000.0,
    )
    path = tmp_path / "map_zoom.nev"
    path.write_bytes(b"placeholder")
    channel_map = ChannelMap(
        name="zoom",
        rows=2,
        cols=2,
        electrodes={
            f"e{index}": {
                "channel": f"chan{index}",
                "electrode": index,
                "x_um": float(index % 2),
                "y_um": float(index // 2),
                "aliases": [f"chan{index}"],
                "routed": True,
            }
            for index in range(1, 5)
        },
    )
    dialog = CustomDataSelectionDialog(
        [{"path": str(path), "raw_data": data, "data_kind": "nev"}],
        "firing_rate_vector",
        channel_map,
    )

    class Scroll:
        inaxes = None
        xdata = 0.5
        ydata = 0.5
        step = 1
        button = "up"

    try:
        event = Scroll()
        event.inaxes = dialog.map_ax
        old_xlim = dialog.map_ax.get_xlim()
        old_ylim = dialog.map_ax.get_ylim()
        dialog._map_scrolled(event)
        new_xlim = dialog.map_ax.get_xlim()
        new_ylim = dialog.map_ax.get_ylim()
        assert abs(new_xlim[1] - new_xlim[0]) < abs(old_xlim[1] - old_xlim[0])
        assert abs(new_ylim[1] - new_ylim[0]) < abs(old_ylim[1] - old_ylim[0])
        preserved = (tuple(new_xlim), tuple(new_ylim))
        dialog._refresh_map()
        assert dialog.map_ax.get_xlim() == pytest.approx(preserved[0])
        assert dialog.map_ax.get_ylim() == pytest.approx(preserved[1])
        dialog._reset_map_view()
        assert abs(dialog.map_ax.get_xlim()[1] - dialog.map_ax.get_xlim()[0]) > abs(preserved[0][1] - preserved[0][0])
    finally:
        dialog.close()
        app.processEvents()


def test_main_window_opens_custom_data_selection_without_modal_loop(tmp_path, monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    path = tmp_path / "sample.nev"
    path.write_bytes(b"placeholder")
    window.file_database = [
        {
            "path": str(path),
            "raw_data": UnifiedMEAData(
                spikes={f"chan{index}": np.array([0.1, 0.2], dtype=float) for index in range(1, 64)},
                sr=20000.0,
            ),
            "data_kind": "nev",
        }
    ]
    constructed = []
    shown = []

    def fake_show_child(child):
        shown.append(child)
        window.child_windows.append(child)
        child.finished.connect(lambda _result: window._forget_child(child))
        child.show()

    monkeypatch.setattr(window, "_show_child", fake_show_child)
    try:
        window._open_custom_data_selection("firing_rate_vector")
        assert shown
        constructed.append(shown[0])
        assert shown[0].open_main_raster_callback == window._open_main_raster_for_database_path
        assert shown[0].windowModality() == Qt.WindowModality.NonModal
        assert window.custom_data_selection_dialog is shown[0]
    finally:
        for dialog in constructed:
            dialog.close()
        window.close()
        app.processEvents()


def test_main_window_custom_data_selection_opens_main_raster_for_selected_path(tmp_path, monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    first = tmp_path / "first.nev"
    second = tmp_path / "second.nev"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    window.file_database = [
        {"path": str(first), "raw_data": UnifiedMEAData(spikes={"chan1": np.array([0.1])}, sr=20000.0), "data_kind": "nev"},
        {"path": str(second), "raw_data": UnifiedMEAData(spikes={"chan2": np.array([0.2])}, sr=20000.0), "data_kind": "nev"},
    ]
    window._refresh_file_database_table()
    opened = []
    monkeypatch.setattr(window, "preview_raw", lambda: opened.append(window.input_path))

    window._open_main_raster_for_database_path(str(second))

    assert opened == [str(second)]
    assert window.input_path == str(second)
    assert window.database_table.currentRow() == 1
    window.close()
    app.processEvents()


def test_main_window_tools_open_stimulus_generation_with_pipeline_source(tmp_path, monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    from src.gui import visual_stimulus_package_builder as stimulus_builder

    window = MainWindow()
    path = tmp_path / "spontaneous_source.nev"
    path.write_bytes(b"placeholder")
    data = UnifiedMEAData(
        spikes={
            "chan1": np.array([0.0, 0.1, 0.2], dtype=float),
            "chan2": np.array([0.05, 0.15], dtype=float),
        },
        sr=20000.0,
    )
    window.file_database = [{"path": str(path), "raw_data": data, "data_kind": "nev"}]
    window.channel_map = ChannelMap(
        name="test_maxwell",
        rows=1,
        cols=3,
        electrodes={
            "e1": {"channel": "chan1", "electrode": 1, "x_um": 0.0, "y_um": 0.0, "aliases": ["chan1", "1"], "routed": True},
            "e2": {"channel": "chan2", "electrode": 2, "x_um": 20.0, "y_um": 0.0, "aliases": ["chan2", "2"], "routed": True},
            "e7317": {"channel": "", "electrode": 7317, "x_um": 40.0, "y_um": 0.0, "aliases": ["7317"], "routed": True},
        },
    )

    messages = []
    import src.gui.app as gui_app

    monkeypatch.setattr(gui_app, "_show_info_message", lambda *args, **kwargs: messages.append(args))
    window.open_stimulus_generation()

    dialog = window.stimulus_generation_dialog
    assert isinstance(dialog, StimulusGenerationDialog)
    assert dialog.tabs.count() == 2
    assert dialog.tabs.tabText(0) == "Settings"
    assert dialog.tabs.tabText(1) == "Experiment"
    assert dialog.protocol_table.parent() is not dialog.protocols_tab
    assert dialog.group_table.parent() is not dialog.groups_tab
    assert dialog.block_table.parent() is not dialog.blocks_tab
    assert dialog.generate_form_widget.parent() is not dialog.generate_tab
    assert dialog.generate_status.parent() is not dialog.generate_tab
    assert dialog.width() <= 1400
    assert dialog.minimumSizeHint().height() < 780
    assert dialog.source_combo.findData(str(path)) >= 0
    assert dialog.protocol_type_label.text().endswith("*")
    assert "Pipeline spontaneous source *" == dialog.source_box.title()
    assert dialog.protocol_table.rowCount() == 0
    assert dialog.group_table.rowCount() == 0
    assert dialog.block_table.rowCount() == 0
    assert dialog.block_phase_table.rowCount() == 0
    assert dialog.protocols == []
    assert dialog.groups == []
    assert dialog.protocol_fields["name"].text() == "new_protocol"
    assert dialog.protocol_fields["amplitude_mv"].text() == "150.0"
    assert dialog.protocol_fields["pulse_width_us"].text() == "300.0"
    assert dialog.protocol_fields["start_ms"].text() == "1500.0"
    assert dialog.preview_group_name.text() == "new_group"
    assert "7317" in dialog.preview_group_electrodes.text()
    assert dialog.settings_workflow_scroll.minimumWidth() >= 620
    assert dialog.settings_workflow_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert dialog.protocol_table.minimumHeight() >= 146
    assert dialog.group_table.minimumHeight() >= 146
    assert dialog.block_table.minimumHeight() >= 146
    assert dialog.preview_canvas.minimumHeight() >= 210
    assert dialog.preview_map_canvas.minimumHeight() >= 300
    assert dialog.generate_button.text() == "Generate"
    assert dialog.generate_button.maximumWidth() <= 132
    assert dialog.settings_protocol_add_button.text() == "Add"
    assert dialog.settings_protocol_remove_button.text() == "Remove"
    assert not hasattr(dialog, "settings_protocol_update_button")
    assert dialog.settings_site_add_button.text() == "Add"
    assert dialog.settings_site_remove_button.text() == "Remove"
    assert not hasattr(dialog, "settings_site_update_button")
    assert dialog.settings_preview_button.text() == "Preview"
    assert dialog.settings_save_block_phase_button.text() == "Save block phase"
    assert dialog.block_phase_remove_button.text() == "Remove"
    assert dialog.block_add_button.text() == "Add"
    assert dialog.block_remove_button.text() == "Remove"
    assert not hasattr(dialog, "protocol_notes")
    assert not hasattr(dialog, "preview_block_name")
    assert dialog.protocol_fields["amplitude_mv"].maximumWidth() >= 220
    assert dialog.protocol_box.layout().horizontalSpacing() <= 4
    dialog.protocol_fields["name"].setText("settings_added")
    dialog.protocol_type.setCurrentText("single_pulse")
    dialog._save_protocol()
    assert any(protocol.name == "settings_added" for protocol in dialog.protocols)
    dialog.preview_group_name.setText("settings_site")
    dialog.preview_group_electrodes.setText("1, 2")
    dialog._save_group_from_settings()
    assert any(group.name == "settings_site" and group.electrodes == [1, 2] for group in dialog.groups)
    fixed = stimulus_builder.StimulusProtocol("feedback_single_150mV", "single_pulse", amplitude_mv=150.0)
    dialog.protocols.append(fixed)
    dialog.groups.append(stimulus_builder.ElectrodeGroup("group_A", [7317]))
    dialog.blocks.append(stimulus_builder.ExperimentBlock("group_A_150mV", "group_A", fixed.name))
    dialog._refresh_all()
    dialog._fill_protocol_form(fixed)
    assert dialog.source_box.isHidden()
    fixed_series = dialog._preview_series_for_protocol(fixed)
    assert 7317 in dialog._stimulus_electrodes_for_preview(fixed, fixed_series)
    dialog._render_protocol_preview(fixed)
    assert dialog.preview_raster_axis.get_xlim()[0] < 0.0
    assert dialog.preview_map_canvas.figure.axes
    state = dialog.preview_map_state
    assert "e1" in state["recording"]
    assert "e7317" in state["stimulation"]
    assert dialog.preview_map_canvas.figure.get_facecolor()[:3] == pytest.approx((1.0, 1.0, 1.0))
    detail = dialog._preview_map_selection_text("e1")
    assert "Electrode: e1" in detail
    assert "Firing rate:" in detail
    poisson = stimulus_builder.StimulusProtocol(
        "poisson_random_safe",
        "poisson_random_electrodes",
        amplitude_mv=150.0,
        pulse_width_us=200.0,
        pulses_per_burst=1,
        poisson_duration_s=300.0,
        lambda_mode="scale",
        random_seed=42,
    )
    dialog.protocols.append(poisson)
    dialog.protocol_source_paths[poisson.name] = str(path)
    dialog._refresh_all()
    dialog._fill_protocol_form(poisson)
    assert dialog.source_box.isVisible()
    preview_series = dialog._preview_series_for_protocol(poisson)
    assert any(item["times_ms"] for item in preview_series)
    assert max((max(item["times_ms"]) for item in preview_series if item["times_ms"]), default=0.0) > 5000.0
    dialog.protocol_fields["name"].setText("poisson_from_settings")
    dialog.protocol_type.setCurrentText("poisson_random_electrodes")
    dialog.preview_group_name.setText("poisson_manual")
    dialog.preview_group_electrodes.setText("1, 2")
    dialog.protocol_fields["region_count"].setText("1")
    dialog.protocol_fields["max_candidate_electrodes"].setText("2")
    dialog._set_combo_data(dialog.source_combo, str(path))
    dialog._save_protocol()
    auto_group = next(group for group in dialog.groups if group.name == "poisson_manual_poisson_from_settings_auto")
    assert set(auto_group.electrodes) == {1, 2}
    dialog.protocol_type.setCurrentText("single_pulse")
    dialog.protocol_fields["name"].setText("workflow_single")
    dialog.preview_group_name.setText("workflow_group")
    dialog.preview_group_electrodes.setText("1, 2")
    dialog._save_block_phase_from_settings()
    assert any(protocol.name == "workflow_single" for protocol in dialog.protocols)
    assert any(group.name == "workflow_group" and group.electrodes == [1, 2] for group in dialog.groups)
    assert any(phase.protocol == "workflow_single" and phase.electrode_group == "workflow_group" for phase in dialog.block_phases)
    assert dialog.block_phase_table.rowCount() >= 1
    dialog._set_combo_data(dialog.block_phase_combo, "workflow_single_workflow_group_phase")
    dialog.phase_duration_fields["01_pre_spont"].setText("111")
    dialog.phase_duration_fields["02_stim"].setText("222")
    dialog.phase_duration_fields["03_post_spont"].setText("333")
    dialog._save_block()
    workflow_block = next(block for block in dialog.blocks if block.protocol == "workflow_single" and block.electrode_group == "workflow_group")
    assert workflow_block.protocol == "workflow_single"
    assert workflow_block.electrode_group == "workflow_group"
    assert [phase.duration_s for phase in workflow_block.phases] == [111, 222, 333]
    before_protocols = len(dialog.protocols)
    before_groups = len(dialog.groups)
    before_blocks = len(dialog.blocks)
    dialog.protocol_fields["name"].setText("preview_only_single")
    dialog.protocol_type.setCurrentText("single_pulse")
    dialog.preview_group_name.setText("preview_only_group")
    dialog.preview_group_electrodes.setText("1, 2")
    dialog._preview_settings_workflow()
    assert len(dialog.protocols) == before_protocols
    assert len(dialog.groups) == before_groups
    assert len(dialog.blocks) == before_blocks
    assert {"e1", "e2"}.issubset(dialog.preview_map_state["stimulation"])
    original_count = len(dialog.blocks)
    dialog._set_combo_data(dialog.block_phase_combo, "workflow_single_workflow_group_phase")
    dialog._save_block()
    assert len(dialog.blocks) == original_count + 1
    added_block = dialog.blocks[-1]
    assert added_block.electrode_group == "workflow_group"
    dialog._select_block_by_name(added_block.name)
    dialog._remove_block()
    assert not any(block.name == added_block.name for block in dialog.blocks)
    output = tmp_path / "generated_stimulus_package"
    dialog.output_path.setText(str(output))
    dialog._generate()

    assert (output / "config" / "system.yaml").exists()
    assert (output / "config" / "stimulation.yaml").exists()
    rate_files = list((output / "config" / "pipeline_rate_sources").glob("*_rates.npz"))
    assert rate_files
    rates = np.load(rate_files[0])
    assert set(rates.files) >= {"electrodes", "rates_hz", "firing_rate_hz", "source_path"}
    assert str(rates["source_path"]) == str(path)
    dialog.close()
    window.close()
    app.processEvents()


def test_stimulus_generation_preview_raster_scrolls_and_zooms_without_regenerating(tmp_path, monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    from src.gui import visual_stimulus_package_builder as stimulus_builder

    path = tmp_path / "spontaneous_source.nev"
    path.write_bytes(b"placeholder")
    data = UnifiedMEAData(
        spikes={
            "chan1": np.linspace(0.0, 10.0, 80),
            "chan2": np.linspace(0.05, 10.05, 70),
        },
        sr=20000.0,
    )
    channel_map = ChannelMap(
        name="test_maxwell",
        rows=1,
        cols=2,
        electrodes={
            "e1": {"channel": "chan1", "electrode": 1, "x_um": 0.0, "y_um": 0.0, "aliases": ["chan1", "1"], "routed": True},
            "e2": {"channel": "chan2", "electrode": 2, "x_um": 20.0, "y_um": 0.0, "aliases": ["chan2", "2"], "routed": True},
        },
    )
    dialog = StimulusGenerationDialog(
        [{"path": str(path), "raw_data": data, "data_kind": "nev"}],
        channel_map=channel_map,
    )
    protocol = stimulus_builder.StimulusProtocol(
        "poisson_random_safe",
        "poisson_random_electrodes",
        poisson_duration_s=300.0,
        lambda_mode="scale",
        random_seed=42,
    )
    dialog.protocols = [protocol]
    dialog.protocol_source_paths[protocol.name] = str(path)
    dialog._refresh_all()
    dialog._render_protocol_preview(protocol)
    dialog.preview_canvas.draw()

    assert dialog.preview_total_ms == pytest.approx(300000.0)
    assert dialog.preview_window_ms == pytest.approx(5000.0)
    assert max((values[-1] for _item, values in dialog.preview_raster_arrays if values.size), default=0.0) > 5000.0

    regenerated = []
    redrew_map = []
    monkeypatch.setattr(dialog, "_preview_series_for_protocol", lambda *args, **kwargs: regenerated.append(args) or [])
    monkeypatch.setattr(dialog, "_draw_preview_channel_map", lambda *args, **kwargs: redrew_map.append(args))

    class ScrollEvent:
        inaxes = dialog.preview_raster_axis
        xdata = 2500.0
        step = -1
        button = "down"

    dialog._preview_raster_scrolled(ScrollEvent())
    assert dialog.preview_window_ms > 5000.0

    class PressEvent:
        inaxes = dialog.preview_raster_axis
        button = 1
        x = 240.0

    class MoveEvent:
        x = 120.0

    dialog._preview_raster_mouse_pressed(PressEvent())
    dialog._preview_raster_mouse_moved(MoveEvent())
    dialog._preview_raster_mouse_released(MoveEvent())

    assert dialog.preview_window_start_ms > 0.0
    assert regenerated == []
    assert redrew_map == []
    dialog._reset_preview_raster_view()
    assert dialog.preview_window_start_ms == pytest.approx(0.0)
    assert dialog.preview_window_ms == pytest.approx(5000.0)
    dialog.close()
    app.processEvents()


def test_stimulus_generation_individual_burst_preview_is_single_burst():
    from src.gui import visual_stimulus_package_builder as stimulus_builder

    protocol = stimulus_builder.StimulusProtocol(
        "three_bursts",
        "individual_burst",
        start_ms=0.0,
        pulses_per_burst=3,
        interpulse_interval_ms=50.0,
        burst_count=3,
        burst_interval_ms=1000.0,
    )

    series = stimulus_builder.preview_raster_series(protocol, preview_limit_ms=3000.0)

    assert series[0]["times_ms"] == [0.0, 50.0, 100.0]


def test_stimulus_generation_package_hardware_sequence_uses_scheduled_times(tmp_path):
    from src.gui import visual_stimulus_package_builder as stimulus_builder

    protocol = stimulus_builder.StimulusProtocol(
        "delayed_bursts",
        "sequence_with_burst",
        pulses_per_burst=2,
        interpulse_interval_ms=25.0,
        burst_count=2,
        burst_interval_ms=1000.0,
    )
    group = stimulus_builder.ElectrodeGroup("group_A", [1234])
    block = stimulus_builder.ExperimentBlock("block_A", "group_A", protocol.name)

    output_dir = stimulus_builder.build_package(
        tmp_path / "package",
        stimulus_builder.ExperimentInfo(),
        [group],
        [protocol],
        [block],
    )

    setup_text = (output_dir / "python" / "maxwell_setup.py").read_text(encoding="utf-8")
    stimulation_text = (output_dir / "config" / "stimulation.yaml").read_text(encoding="utf-8")

    assert "start_ms: 1500.0" in stimulation_text
    assert "for stim_ms in _scheduled_stim_times_ms(protocol):" in setup_text
    assert "seq.append(mx.DelaySamples(_samples_ms(stim_ms - current_ms)))" in setup_text


def test_stimulus_generation_package_configures_array_before_pre_spontaneous(tmp_path):
    from src.gui import visual_stimulus_package_builder as stimulus_builder

    protocol = stimulus_builder.StimulusProtocol("delayed_single", "single_pulse")
    group = stimulus_builder.ElectrodeGroup("group_A", [1234])
    block = stimulus_builder.ExperimentBlock("block_A", "group_A", protocol.name)

    output_dir = stimulus_builder.build_package(
        tmp_path / "package",
        stimulus_builder.ExperimentInfo(),
        [group],
        [protocol],
        [block],
    )

    runner_text = (output_dir / "python" / "experiment_runner.py").read_text(encoding="utf-8")
    configure_index = runner_text.index("configure_experiment_array(cfg_path, electrode_group, system_config)")
    phase_loop_index = runner_text.index('for phase in block.get("phases", []):')
    stim_phase_index = runner_text.index("def _run_stim_phase(")

    assert configure_index < phase_loop_index
    assert runner_text.find("configure_experiment_array", stim_phase_index) == -1


def test_stimulus_generation_package_writes_poisson_auto_electrode_group(tmp_path):
    from src.gui import visual_stimulus_package_builder as stimulus_builder

    rate_path = tmp_path / "rates.npz"
    np.savez(
        rate_path,
        electrodes=np.asarray([100, 101, 200, 201], dtype=np.int32),
        rates_hz=np.asarray([1.0, 9.0, 2.0, 8.0], dtype=float),
    )
    protocol = stimulus_builder.StimulusProtocol(
        "poisson_auto",
        "poisson_random_electrodes",
        spontaneous_data_path=str(rate_path),
        region_count=2,
        max_candidate_electrodes=4,
    )
    group = stimulus_builder.ElectrodeGroup("manual_group", [7317])
    block = stimulus_builder.ExperimentBlock("poisson_block", "manual_group", protocol.name)

    output_dir = stimulus_builder.build_package(
        tmp_path / "package",
        stimulus_builder.ExperimentInfo(),
        [group],
        [protocol],
        [block],
    )

    system_text = (output_dir / "config" / "system.yaml").read_text(encoding="utf-8")
    stimulation_text = (output_dir / "config" / "stimulation.yaml").read_text(encoding="utf-8")

    assert "electrode_group: manual_group_poisson_auto_auto" in system_text
    assert "name: manual_group_poisson_auto_auto" in stimulation_text
    assert "7317" in stimulation_text
    assert "101" in stimulation_text
    assert "201" in stimulation_text


def test_stimulus_generation_ui_updates_poisson_auto_electrode_group(tmp_path):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")
    from src.gui import visual_stimulus_package_builder as stimulus_builder

    app = QApplication.instance() or QApplication([])
    rate_path = tmp_path / "rates.npz"
    np.savez(
        rate_path,
        electrodes=np.asarray([100, 101, 200, 201], dtype=np.int32),
        rates_hz=np.asarray([1.0, 9.0, 2.0, 8.0], dtype=float),
    )
    dialog = StimulusGenerationDialog([], channel_map=None)
    protocol = stimulus_builder.StimulusProtocol(
        "poisson_ui",
        "poisson_random_electrodes",
        spontaneous_data_path=str(rate_path),
        region_count=2,
        max_candidate_electrodes=4,
    )
    dialog.protocols = [protocol]
    dialog.protocol_source_paths[protocol.name] = str(rate_path)
    dialog.groups = [stimulus_builder.ElectrodeGroup("manual_group", [7317])]
    dialog.blocks = [stimulus_builder.ExperimentBlock("poisson_block", "manual_group", protocol.name)]

    dialog._sync_poisson_protocol_auto_group(protocol.name)

    auto_group = next(group for group in dialog.groups if group.name == "manual_group_poisson_ui_auto")
    assert auto_group.electrodes == [101, 201]
    assert dialog.blocks[0].electrode_group == auto_group.name
    group_names = [dialog.group_table.item(row, 0).text() for row in range(dialog.group_table.rowCount())]
    assert auto_group.name in group_names
    dialog.close()
    app.processEvents()


def test_stimulus_generation_save_protocol_creates_poisson_group_without_block(tmp_path):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")
    from src.gui import visual_stimulus_package_builder as stimulus_builder

    app = QApplication.instance() or QApplication([])
    source_path = tmp_path / "spontaneous.nev"
    source_path.write_bytes(b"placeholder")
    data = UnifiedMEAData(
        spikes={
            "chan100": np.array([0.0, 0.5], dtype=float),
            "chan101": np.array([0.0, 0.1, 0.2, 0.3], dtype=float),
            "chan200": np.array([0.0], dtype=float),
            "chan201": np.array([0.0, 0.05, 0.1], dtype=float),
        },
        sr=20000.0,
    )
    dialog = StimulusGenerationDialog(
        [{"path": str(source_path), "raw_data": data, "data_kind": "nev"}],
        channel_map=None,
    )
    dialog.groups = [stimulus_builder.ElectrodeGroup("group_A", [7317])]
    dialog.blocks = [stimulus_builder.ExperimentBlock("ordinary_block", "group_A", "feedback_single_150mV")]
    dialog._refresh_all()

    dialog.protocol_fields["name"].setText("poisson_saved")
    dialog.protocol_type.setCurrentText("poisson_random_electrodes")
    dialog.protocol_fields["region_count"].setText("2")
    dialog.protocol_fields["max_candidate_electrodes"].setText("4")
    dialog._set_combo_data(dialog.source_combo, str(source_path))
    dialog._save_protocol()

    auto_group = next(group for group in dialog.groups if group.name == "group_A_poisson_saved_auto")
    assert auto_group.electrodes == [101, 201]
    assert dialog.blocks[0].electrode_group == "group_A"
    group_names = [dialog.group_table.item(row, 0).text() for row in range(dialog.group_table.rowCount())]
    assert auto_group.name in group_names
    dialog.close()
    app.processEvents()


def test_stimulus_generation_protocol_fields_follow_selected_type():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    dialog = StimulusGenerationDialog([], channel_map=None)

    dialog.protocol_type.setCurrentText("individual_burst")
    dialog._update_protocol_type_fields()
    assert not dialog.protocol_fields["pulses_per_burst"].isHidden()
    assert dialog.protocol_fields["burst_count"].isHidden()
    assert dialog.protocol_fields["burst_interval_ms"].isHidden()
    assert dialog.custom_points.isHidden()
    assert dialog.source_box.isHidden()

    dialog.protocol_type.setCurrentText("sequence_with_burst")
    dialog._update_protocol_type_fields()
    assert not dialog.protocol_fields["burst_count"].isHidden()
    assert not dialog.protocol_fields["burst_interval_ms"].isHidden()

    dialog.protocol_type.setCurrentText("custom_sequence")
    dialog._update_protocol_type_fields()
    assert not dialog.custom_points.isHidden()
    assert dialog.protocol_fields["amplitude_mv"].isHidden()
    assert dialog.protocol_fields["burst_count"].isHidden()

    dialog.protocol_type.setCurrentText("poisson_random_electrodes")
    dialog._update_protocol_type_fields()
    assert not dialog.source_box.isHidden()
    assert not dialog.lambda_mode.isHidden()
    assert not dialog.protocol_fields["poisson_duration_s"].isHidden()
    assert dialog.protocol_fields["burst_interval_ms"].isHidden()
    dialog.advanced_toggle.setChecked(True)
    dialog.lambda_mode.setCurrentText("scale")
    dialog._update_protocol_type_fields()
    assert not dialog.protocol_fields["lambda_scale"].isHidden()
    assert dialog.protocol_fields["lambda_mean_hz"].isHidden()
    assert dialog.protocol_fields["lambda_std_hz"].isHidden()
    dialog.lambda_mode.setCurrentText("normal")
    dialog._update_protocol_type_fields()
    assert dialog.protocol_fields["lambda_scale"].isHidden()
    assert not dialog.protocol_fields["lambda_mean_hz"].isHidden()
    assert not dialog.protocol_fields["lambda_std_hz"].isHidden()
    dialog.close()
    app.processEvents()


def test_stimulus_generation_poisson_lambda_scale_and_normal_modes():
    from src.gui import visual_stimulus_package_builder as stimulus_builder

    scale_protocol = stimulus_builder.StimulusProtocol(
        "scaled",
        "poisson_random_electrodes",
        lambda_mode="scale",
        lambda_scale=2.5,
        lambda_floor_hz=0.001,
    )
    assert stimulus_builder._preview_lambda_hz(4.0, scale_protocol, __import__("random").Random(1)) == pytest.approx(10.0)

    normal_protocol = stimulus_builder.StimulusProtocol(
        "normal",
        "poisson_random_electrodes",
        lambda_mode="normal",
        lambda_mean_hz=3.0,
        lambda_std_hz=0.0,
        lambda_floor_hz=0.001,
    )
    assert stimulus_builder._preview_lambda_hz(100.0, normal_protocol, __import__("random").Random(1)) == pytest.approx(3.0)

    payload = normal_protocol.to_yaml()["random_electrode_plan"]
    assert payload["lambda_mode"] == "normal"
    assert payload["lambda_mean_hz"] == pytest.approx(3.0)
    assert payload["lambda_std_hz"] == pytest.approx(0.0)
    assert "lambda_gaussian_cv" not in payload


def test_stimulus_generation_preview_map_click_uses_lightweight_selection(tmp_path, monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    from src.gui import visual_stimulus_package_builder as stimulus_builder

    path = tmp_path / "spontaneous_source.nev"
    path.write_bytes(b"placeholder")
    data = UnifiedMEAData(
        spikes={
            "chan1": np.array([0.0, 0.1, 0.2], dtype=float),
            "chan2": np.array([0.05, 0.15], dtype=float),
        },
        sr=20000.0,
    )
    channel_map = ChannelMap(
        name="test_maxwell",
        rows=1,
        cols=3,
        electrodes={
            "e1": {"channel": "chan1", "electrode": 1, "x_um": 0.0, "y_um": 0.0, "aliases": ["chan1", "1"], "routed": True},
            "e2": {"channel": "chan2", "electrode": 2, "x_um": 20.0, "y_um": 0.0, "aliases": ["chan2", "2"], "routed": True},
            "e7317": {"channel": "", "electrode": 7317, "x_um": 40.0, "y_um": 0.0, "aliases": ["7317"], "routed": True},
        },
    )
    dialog = StimulusGenerationDialog(
        [{"path": str(path), "raw_data": data, "data_kind": "nev"}],
        channel_map=channel_map,
    )
    protocol = stimulus_builder.StimulusProtocol("feedback_single_150mV", "single_pulse", amplitude_mv=150.0)
    dialog.protocols = [protocol]
    dialog.groups = [stimulus_builder.ElectrodeGroup("group_A", [1, 7317])]
    dialog.blocks = [stimulus_builder.ExperimentBlock("group_A_150mV", "group_A", protocol.name)]
    dialog._refresh_all()
    dialog._render_protocol_preview(protocol)
    state = dialog.preview_map_state
    x, y, _payload = state["positions"]["e1"]
    redraw_calls = []
    original_draw_map = dialog._draw_preview_channel_map
    monkeypatch.setattr(dialog, "_draw_preview_channel_map", lambda *args, **kwargs: redraw_calls.append(args))

    class Event:
        inaxes = dialog.preview_map_canvas.figure.axes[0]
        xdata = float(x)
        ydata = float(y)

    dialog._preview_map_clicked(Event())

    assert redraw_calls == []
    assert dialog.preview_map_selected == "e1"
    assert dialog.preview_map_selection_artist is not None
    assert "Electrode: e1" in dialog.preview_map_detail.text()

    class Scroll:
        inaxes = dialog.preview_map_canvas.figure.axes[0]
        xdata = float(x)
        ydata = float(y)
        step = 1
        button = "up"

    old_xlim = dialog.preview_map_canvas.figure.axes[0].get_xlim()
    dialog._preview_map_scrolled(Scroll())
    new_xlim = dialog.preview_map_canvas.figure.axes[0].get_xlim()
    assert abs(new_xlim[1] - new_xlim[0]) < abs(old_xlim[1] - old_xlim[0])
    assert redraw_calls == []
    preserved = tuple(new_xlim)
    original_draw_map(protocol, dialog.preview_map_series)
    assert dialog.preview_map_canvas.figure.axes[0].get_xlim() == pytest.approx(preserved)
    monkeypatch.setattr(dialog, "_draw_preview_channel_map", original_draw_map)
    dialog._reset_preview_map_view()
    assert abs(dialog.preview_map_canvas.figure.axes[0].get_xlim()[1] - dialog.preview_map_canvas.figure.axes[0].get_xlim()[0]) > abs(preserved[1] - preserved[0])
    dialog.close()
    app.processEvents()


def test_preview_raw_auto_caches_processed_datasets(tmp_path, monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    path = tmp_path / "processed_sample.nev"
    path.write_bytes(b"placeholder")
    raw_record = {
        "path": str(path),
        "raw_data": UnifiedMEAData(
            spikes={
                "chan1": np.array([0.01, 0.03, 0.05], dtype=float),
                "chan2": np.array([0.015, 0.035, 0.055], dtype=float),
            },
            sr=20000.0,
        ),
        "data_kind": "nev",
    }
    window.file_database = [raw_record]
    window._refresh_file_database_table()
    window.database_table.selectRow(0)
    window._set_active_database_index(0)

    shown = []

    def wrapped_show_child(child):
        shown.append(child)
        return child

    monkeypatch.setattr(window, "_show_child", wrapped_show_child)
    window.preview_raw()

    assert shown
    assert len(window.processed_database) >= 1
    assert any(record.get("dataset_type") == "channel_time" for record in window.processed_database)
    assert any(record.get("dataset_group") in {"raster", "burst", "array"} for record in window.processed_database)
    assert any(record.get("commit") for record in window.processed_database)
    window.processed_table.selectRow(0)
    window._processed_selection_changed()
    preview = window._data_preview_text()
    assert "Processed dataset:" in preview
    assert "Dataset type:" in preview
    assert "Commit:" in preview
    assert "Samples x features:" in preview

    for child in shown:
        if hasattr(child, "close"):
            child.close()
    window.close()
    app.processEvents()


def test_build_processed_records_preserve_dataset_metadata(tmp_path):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    path = tmp_path / "metadata_sample.nev"
    path.write_bytes(b"placeholder")
    raw_record = {
        "path": str(path),
        "raw_data": UnifiedMEAData(
            spikes={
                "chan1": np.array([0.01, 0.03, 0.05], dtype=float),
                "chan2": np.array([0.02, 0.04, 0.06], dtype=float),
            },
            sr=20000.0,
        ),
        "data_kind": "nev",
    }

    processed_records, errors = window._build_processed_records(
        [raw_record],
        [str(path)],
        {
            "dataset_type": "channel_time",
            "dataset_group": "raster",
            "origin": "auto-raster",
            "display_name": "Channel x time",
            "view_mode": "channel_time",
            "bin_ms": 10.0,
            "burst_window_ms": 300.0,
            "burst_threshold_z": 4.0,
            "array_axis": "rows",
        },
    )

    assert not errors
    assert len(processed_records) == 1
    processed = processed_records[0]
    assert processed["dataset_type"] == "channel_time"
    assert processed["dataset_group"] == "raster"
    assert processed["dataset_origin"] == "auto-raster"
    assert processed["name"] == "Channel x time"
    assert "channel-time spike counts" in processed["commit"]
    assert "bin=10 ms" in processed["commit_detail"]

    window.close()
    app.processEvents()


def test_main_window_save_file_saves_single_active_record(tmp_path, monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    data = UnifiedMEAData(spikes={"chan1": np.array([0.0, 0.1])}, sr=20000.0)
    window.raw_data = data
    window.input_path = str(tmp_path / "single.nev")
    path = tmp_path / "single_data.npz"

    monkeypatch.setattr(gui_app.QInputDialog, "getItem", lambda *args, **kwargs: ("Spike train only", True))
    monkeypatch.setattr(gui_app.QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(path), ""))
    monkeypatch.setattr(gui_app, "_show_info_message", lambda *args, **kwargs: None)

    window.save_spike_train()

    assert path.exists()
    window.close()
    window.deleteLater()
    app.closeAllWindows()
    app.processEvents()
    app.quit()
    app.processEvents()


def test_main_window_save_file_saves_multiple_selected_records(tmp_path, monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    first = tmp_path / "first.nev"
    second = tmp_path / "second.nev"
    first.write_bytes(b"placeholder")
    second.write_bytes(b"placeholder")
    window.file_database = [
        {"path": str(first), "raw_data": UnifiedMEAData(spikes={"a": np.array([0.0, 0.1])}, sr=20000.0), "data_kind": "nev"},
        {"path": str(second), "raw_data": UnifiedMEAData(spikes={"b": np.array([0.2, 0.3])}, sr=20000.0), "data_kind": "nev"},
    ]
    window._refresh_file_database_table()
    window._set_active_database_index(0)
    output_dir = tmp_path / "saved"
    output_dir.mkdir()

    monkeypatch.setattr(window, "_selected_database_records", lambda: list(window.file_database))
    monkeypatch.setattr(gui_app.QInputDialog, "getItem", lambda *args, **kwargs: ("Spike train only", True))
    monkeypatch.setattr(gui_app.QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(output_dir))
    monkeypatch.setattr(gui_app.QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(output_dir / "fallback.npz"), ""))
    monkeypatch.setattr(gui_app, "_show_info_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(gui_app, "_show_warning_message", lambda *args, **kwargs: None)

    window.save_spike_train()

    assert (output_dir / "first_spike_train.npz").exists()
    assert (output_dir / "second_spike_train.npz").exists()
    window.close()
    window.deleteLater()
    app.closeAllWindows()
    app.processEvents()
    app.quit()
    app.processEvents()


def test_main_window_save_file_multiple_records_avoids_name_collisions(tmp_path, monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    same_name_a = tmp_path / "group_a" / "same.nev"
    same_name_b = tmp_path / "group_b" / "same.nev"
    same_name_a.parent.mkdir()
    same_name_b.parent.mkdir()
    same_name_a.write_bytes(b"placeholder")
    same_name_b.write_bytes(b"placeholder")
    window.file_database = [
        {"path": str(same_name_a), "raw_data": UnifiedMEAData(spikes={"a": np.array([0.0, 0.1])}, sr=20000.0), "data_kind": "nev"},
        {"path": str(same_name_b), "raw_data": UnifiedMEAData(spikes={"b": np.array([0.2, 0.3])}, sr=20000.0), "data_kind": "nev"},
    ]
    output_dir = tmp_path / "collision_saved"
    output_dir.mkdir()

    monkeypatch.setattr(window, "_selected_database_records", lambda: list(window.file_database))
    monkeypatch.setattr(gui_app.QInputDialog, "getItem", lambda *args, **kwargs: ("Spike train only", True))
    monkeypatch.setattr(gui_app.QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(output_dir))
    monkeypatch.setattr(gui_app, "_show_info_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(gui_app, "_show_warning_message", lambda *args, **kwargs: None)

    window.save_spike_train()

    assert (output_dir / "same_spike_train.npz").exists()
    assert (output_dir / "same_spike_train_1.npz").exists()
    window.close()
    window.deleteLater()
    app.closeAllWindows()
    app.processEvents()
    app.quit()
    app.processEvents()


def test_main_window_save_file_multiple_records_handles_mixed_data_types(tmp_path, monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    nev_path = tmp_path / "mixed.nev"
    array_path = tmp_path / "array_source.npy"
    nev_path.write_bytes(b"placeholder")
    np.save(array_path, np.arange(6, dtype=float).reshape(2, 3))
    window.file_database = [
        {"path": str(nev_path), "raw_data": UnifiedMEAData(spikes={"a": np.array([0.0, 0.1])}, sr=20000.0), "data_kind": "nev"},
        {"path": str(array_path), "raw_data": np.arange(6, dtype=float).reshape(2, 3), "data_kind": "npy"},
    ]
    output_dir = tmp_path / "mixed_saved"
    output_dir.mkdir()

    monkeypatch.setattr(window, "_selected_database_records", lambda: list(window.file_database))
    monkeypatch.setattr(gui_app.QInputDialog, "getItem", lambda *args, **kwargs: ("Spike train only", True))
    monkeypatch.setattr(gui_app.QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(output_dir))
    monkeypatch.setattr(gui_app, "_show_info_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(gui_app, "_show_warning_message", lambda *args, **kwargs: None)

    window.save_spike_train()

    assert (output_dir / "mixed_spike_train.npz").exists()
    assert (output_dir / "array_source_array_data.npz").exists()
    window.close()
    window.deleteLater()
    app.closeAllWindows()
    app.processEvents()
    app.quit()
    app.processEvents()


def test_dynamics_analysis_single_selected_file_opens_burst_trajectory_window(monkeypatch, tmp_path):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    path = tmp_path / "single.nev"
    path.write_bytes(b"placeholder")
    data = UnifiedMEAData(
        spikes={
            "chan1": np.array([0.005, 0.020, 1.005, 1.020]),
            "chan2": np.array([0.012, 0.028, 1.012, 1.028]),
        },
        sr=30000.0,
    )
    window.file_database = [{"path": str(path), "raw_data": data, "data_kind": "nev"}]
    window._refresh_file_database_table()
    window.database_table.selectRow(0)
    window._set_active_database_index(0)
    dialog = window._multi_file_fa_analysis_dialog()
    dialog.table.selectRow(0)
    dialog.analysis_scope.setCurrentIndex(dialog.analysis_scope.findData("burst"))
    shown = []

    def fake_detect(*args, **kwargs):
        return [(0.0, 0.05), (1.0, 1.05)]

    def wrapped_show_child(child):
        shown.append(child)
        return child.show()

    monkeypatch.setattr(gui_app, "_detect_burst_intervals", fake_detect)
    monkeypatch.setattr(window, "_show_child", wrapped_show_child)
    window._start_multi_file_fa_from_dialog()

    assert shown
    child = shown[0]
    assert child.__class__.__name__ == "BurstTrajectoryWindow"
    assert child.bin_ms.value() == pytest.approx(dialog.bin_ms.value())
    assert child.window_ms.value() == int(round(dialog.window_ms.value()))
    child.close()
    dialog.close()
    window.close()


def test_dynamics_analysis_single_selected_file_removes_stimulus_tail(monkeypatch, tmp_path):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    path = tmp_path / "single_stimulus.nev"
    path.write_bytes(b"placeholder")
    starts = np.array([0.0, 1.0, 2.0])
    artifact_offsets = np.array([0.0005, 0.010, 0.014, 0.018, 0.022, 0.026])
    clean_offsets = np.array([0.0015, 0.012, 0.016, 0.020, 0.024, 0.028])
    data = UnifiedMEAData(
        spikes={
            "chan1": np.sort(np.concatenate([start + artifact_offsets for start in starts])),
            "chan2": np.sort(np.concatenate([start + clean_offsets for start in starts])),
        },
        stim_times=starts,
        sr=30000.0,
    )
    window.file_database = [{"path": str(path), "raw_data": data, "data_kind": "nev"}]
    window._refresh_file_database_table()
    window.database_table.selectRow(0)
    window._set_active_database_index(0)
    dialog = window._multi_file_fa_analysis_dialog()
    dialog.table.selectRow(0)
    dialog.analysis_scope.setCurrentIndex(dialog.analysis_scope.findData("burst"))
    dialog.artifact_ms.setValue(1.0)
    captured = {}
    shown = []

    def fake_detect(spike_series, *args, **kwargs):
        captured.update({str(label): np.asarray(times, dtype=float).copy() for label, times in spike_series})
        return [(0.0, 0.04), (1.0, 1.04), (2.0, 2.04)]

    def wrapped_show_child(child):
        shown.append(child)
        return child.show()

    monkeypatch.setattr(gui_app, "_detect_burst_intervals", fake_detect)
    monkeypatch.setattr(window, "_show_child", wrapped_show_child)
    window._start_multi_file_fa_from_dialog()

    assert shown
    assert captured["chan1"].size == starts.size * (artifact_offsets.size - 1)
    assert captured["chan2"].size == starts.size * clean_offsets.size
    removed_times = starts + artifact_offsets[0]
    for value in removed_times:
        assert not np.any(np.isclose(captured["chan1"], value, atol=1e-12))
    shown[0].close()
    dialog.close()
    window.close()


def test_dynamics_analysis_single_selected_file_lds_uses_burst_trajectory_window(monkeypatch, tmp_path):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        import src.gui.app as gui_app
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    path = tmp_path / "single_lds.nev"
    path.write_bytes(b"placeholder")
    data = UnifiedMEAData(
        spikes={
            "chan1": np.array([0.005, 0.020, 1.005, 1.020, 2.005, 2.020]),
            "chan2": np.array([0.012, 0.028, 1.012, 1.028, 2.012, 2.028]),
        },
        sr=30000.0,
    )
    window.file_database = [{"path": str(path), "raw_data": data, "data_kind": "nev"}]
    window._refresh_file_database_table()
    window.database_table.selectRow(0)
    window._set_active_database_index(0)
    dialog = window._multi_file_fa_analysis_dialog()
    dialog.table.selectRow(0)
    dialog.analysis_scope.setCurrentIndex(dialog.analysis_scope.findData("burst"))
    dialog.model_method.setCurrentIndex(dialog.model_method.findData("lds"))

    shown = []

    def fake_detect(*args, **kwargs):
        return [(0.0, 0.05), (1.0, 1.05), (2.0, 2.05)]

    def wrapped_show_child(child):
        shown.append(child)
        return child.show()

    monkeypatch.setattr(gui_app, "_detect_burst_intervals", fake_detect)
    monkeypatch.setattr(window, "_show_child", wrapped_show_child)
    window._start_multi_file_fa_from_dialog()

    assert shown
    child = shown[0]
    assert child.__class__.__name__ == "BurstTrajectoryWindow"
    assert getattr(child, "model_method", "") == "lds"
    assert str((child.current or {}).get("model_method", "")).lower() == "lds"
    assert "LDS" in child.summary.text()
    child.close()
    dialog.close()
    window.close()


def test_data_load_worker_reads_maxwell_h5_spikes_without_waveforms(monkeypatch, tmp_path):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    path = tmp_path / "large.raw.h5"
    path.write_bytes(b"placeholder")
    calls = []

    def fake_read_maxwell_h5(path_arg, *, cancel_check=None, extract_waveforms=True, **kwargs):
        calls.append((Path(path_arg), bool(extract_waveforms)))
        return UnifiedMEAData(
            spikes={"well0_e1": np.array([0.1, 0.2])},
            sr=20000.0,
            meta={"source": "maxwell_h5", "extract_waveforms": bool(extract_waveforms)},
        )

    import src.gui.app as app_module

    monkeypatch.setattr(app_module, "read_maxwell_h5", fake_read_maxwell_h5)
    worker = DataLoadWorker(str(path))
    finished = []
    failed = []
    worker.signals.finished.connect(finished.append)
    worker.signals.failed.connect(failed.append)

    worker.run()

    assert not failed
    assert calls == [(path, False)]
    assert len(finished) == 1
    assert finished[0]["data_kind"] == "nev"
    assert finished[0]["raw_data"].waveforms == {}


def test_file_database_worker_reads_multiple_maxwell_files_without_waveforms(monkeypatch, tmp_path):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    paths = [tmp_path / "a.raw.h5", tmp_path / "b.raw.h5"]
    for path in paths:
        path.write_bytes(b"placeholder")
    calls = []

    def fake_read_maxwell_h5(path_arg, *, cancel_check=None, extract_waveforms=True, **kwargs):
        calls.append((Path(path_arg).name, bool(extract_waveforms)))
        return UnifiedMEAData(
            spikes={"well0_e1": np.array([0.1])},
            sr=20000.0,
            meta={"source": "maxwell_h5", "extract_waveforms": bool(extract_waveforms)},
        )

    import src.gui.app as app_module

    monkeypatch.setattr(app_module, "read_maxwell_h5", fake_read_maxwell_h5)
    worker = FileDatabaseLoadWorker([str(path) for path in paths])
    finished = []
    failed = []
    worker.signals.finished.connect(finished.append)
    worker.signals.failed.connect(failed.append)

    worker.run()

    assert not failed
    assert calls == [("a.raw.h5", False), ("b.raw.h5", False)]
    assert len(finished) == 1
    assert len(finished[0]["records"]) == 2
    assert finished[0]["records"][0]["raw_data"].waveforms == {}


def test_sorting_on_fast_loaded_maxwell_triggers_deferred_waveform_load(monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QMessageBox
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.raw_data = UnifiedMEAData(
        spikes={"well0_e1": np.array([0.1, 0.2])},
        waveforms={},
        sr=20000.0,
        meta={"source": "maxwell_h5", "waveforms_deferred": True},
    )
    window.data_kind = "nev"
    window.input_path = "large.raw.h5"
    called = []
    monkeypatch.setattr(window, "_load_maxwell_waveforms_for_sorting", lambda: called.append(True))
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)

    window.open_sorting()

    assert called == [True]
    window.close()


def test_sorting_on_fast_loaded_maxwell_can_skip_deferred_waveforms(monkeypatch):
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QMessageBox
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.raw_data = UnifiedMEAData(
        spikes={"well0_e1": np.array([0.1, 0.2])},
        waveforms={},
        sr=20000.0,
        meta={"source": "maxwell_h5", "waveforms_deferred": True},
    )
    window.data_kind = "nev"
    window.input_path = "large.raw.h5"
    called = []
    monkeypatch.setattr(window, "_load_maxwell_waveforms_for_sorting", lambda: called.append(True))
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.No)

    window.open_sorting()

    assert called == []
    window.close()


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


def test_sorting_workspace_embedding_click_highlights_waveform_and_point():
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 is not available")

    app = QApplication.instance() or QApplication([])
    waveforms = np.array([
        [0.0, -0.1, -0.3, -0.1, 0.0, 0.1, 0.0, 0.0],
        [0.0, -0.2, -0.4, -0.1, 0.1, 0.1, 0.0, 0.0],
        [0.0, 0.2, 0.4, 0.2, 0.0, -0.1, 0.0, 0.0],
        [0.0, 0.1, 0.3, 0.1, -0.1, -0.1, 0.0, 0.0],
    ])
    data = UnifiedMEAData(
        spikes={"chan1": np.array([0.0, 0.1, 0.2, 0.3])},
        waveforms={"chan1": waveforms},
        sr=30000.0,
    )
    window = SortingWorkspaceWindow(data)
    window.current_embedding = np.array([[0.0, 0.0], [0.1, 0.0], [2.0, 2.0], [2.1, 2.0]], dtype=np.float32)
    window.current_labels = np.array([0, 0, 1, 1], dtype=np.int32)
    window._draw_all()
    app.processEvents()

    display_x, display_y = window.embedding_ax.transData.transform((2.0, 2.0))
    event = type(
        "ClickEvent",
        (),
        {
            "inaxes": window.embedding_ax,
            "button": 1,
            "x": display_x,
            "y": display_y,
            "xdata": 2.0,
            "ydata": 2.0,
        },
    )()

    window._embedding_clicked(event)

    assert window.selected_spike_index == 2
    assert "Selected spike 3" in window.status.text()
    waveform_ax = window.waveform_canvas.figure.axes[0]
    assert any(line.get_linewidth() >= 2.5 for line in waveform_ax.lines)
    embedding_ax = window.embedding_canvas.figure.axes[0]
    assert len(embedding_ax.collections) >= 3


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
