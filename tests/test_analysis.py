"""Tests for analysis module."""

import pytest
import numpy as np

from src.analysis import (
    Analyzer,
    compute_similarity_matrix,
    create_generic_analysis_figure,
    normalize_feature_matrix,
    prepare_feature_matrix,
    reduce_feature_matrix,
    run_generic_matrix_analysis,
    global_response_lines,
    load_channel_forecast_summary,
    spatial_response_metrics,
)


class TestAnalyzer:
    """Test Analyzer class."""
    
    def test_analyzer_initialization(self):
        """Test Analyzer initialization."""
        assert Analyzer() is not None
    
    def test_extract_features(self):
        """Test feature extraction."""
        data = np.arange(20, dtype=float).reshape(2, 10)
        features = Analyzer().extract_features(data)
        assert features["channel_count"] == 2
        assert features["sample_count"] == 10
        assert features["correlation"].shape == (2, 2)
        assert "generic_analysis" in features


def test_prepare_feature_matrix_flattens_nontrivial_sample_axis():
    data = np.arange(24, dtype=float).reshape(2, 3, 4)
    prepared = prepare_feature_matrix(data, sample_axis=1)
    assert prepared.matrix.shape == (3, 8)
    assert prepared.sample_count == 3
    assert prepared.feature_count == 8


def test_normalize_feature_matrix_supports_per_sample_and_feature_zscore():
    features = np.array([[1.0, 1.0, 2.0], [2.0, 0.0, 0.0]], dtype=float)
    per_sample, meta_a = normalize_feature_matrix(features, "per_sample_l1")
    assert meta_a["method"] == "per_sample_l1"
    np.testing.assert_allclose(np.sum(np.abs(per_sample), axis=1), np.array([1.0, 1.0]))

    zscore, meta_b = normalize_feature_matrix(features, "feature_zscore")
    assert meta_b["method"] == "feature_zscore"
    np.testing.assert_allclose(np.mean(zscore, axis=0), np.zeros(features.shape[1]), atol=1e-7)


def test_compute_similarity_matrix_and_reduction_work_for_generic_features():
    features = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.1, 0.0, 1.1, 1.0],
            [4.0, 4.0, 0.0, 0.0],
            [4.2, 4.1, 0.0, 0.1],
        ],
        dtype=float,
    )
    similarity = compute_similarity_matrix(features, method="correlation")
    assert similarity.shape == (4, 4)
    assert np.allclose(np.diag(similarity), 1.0)

    reduced = reduce_feature_matrix(features, method="pca", n_components=2, standardize=True)
    coords = np.asarray(reduced["coordinates"], dtype=float)
    assert coords.shape == (4, 2)


def test_run_generic_matrix_analysis_returns_visualizable_payload():
    data = np.array(
        [
            [[1.0, 0.0], [0.8, 0.2], [1.2, 0.1]],
            [[0.9, 0.1], [1.1, 0.0], [0.7, 0.3]],
            [[0.0, 1.0], [0.1, 0.9], [0.2, 1.1]],
            [[0.1, 1.1], [0.0, 1.0], [0.2, 0.8]],
        ],
        dtype=float,
    )
    result = run_generic_matrix_analysis(
        data,
        sample_axis=0,
        normalization="feature_zscore",
        similarity="correlation",
        reduction="pca",
        reduction_dims=2,
        clustering="kmeans",
        cluster_count=2,
    )
    assert result["normalized_matrix"].shape[0] == 4
    assert result["similarity_matrix"].shape == (4, 4)
    assert result["coordinates"].shape == (4, 2)
    assert result["groups"].shape == (4,)
    figure = create_generic_analysis_figure(result, title="Test generic analysis")
    assert len(figure.axes) >= 4


def test_channel_forecast_reader_reduces_npz_for_sample_viewing(tmp_path):
    samples, history, future, channels = 2, 3, 2, 4
    predicted = np.full((samples, future, channels), 4.0, dtype=np.float32)
    observed = np.full((samples, future, channels), 2.0, dtype=np.float32)
    context = np.arange(samples * history * channels, dtype=np.float32).reshape(samples, history, channels)
    path = tmp_path / "forecast.npz"
    np.savez(
        path,
        channel_rate_pred=predicted,
        target_channel_rate_raw=observed,
        context_channel_rate_raw=context,
        channel_xy=np.array([[[0, 0], [1, 0], [0, 1], [1, 1]]] * samples, dtype=np.float32),
        target_bin_valid_mask=np.array([[True, False], [True, True]]),
        target_start_time=np.array([1.25, 2.5], dtype=np.float32),
        target_stim_site_id=np.array(["13045+13479", "none"]),
        target_stim_features=np.array([[[1.0], [0.0]], [[0.0], [1.0]]], dtype=np.float32),
    )

    summary = load_channel_forecast_summary(path)
    assert summary.sample_count == samples
    assert summary.channel_count == channels
    np.testing.assert_allclose(summary.predicted_rate[0], np.full(channels, 4.0))
    np.testing.assert_allclose(summary.observed_rate[0], np.full(channels, 2.0))
    assert summary.sample_labels[0] == "1 | 1.250 s"
    assert summary.stimulation_electrodes[0] == (13045, 13479)
    assert summary.stimulus_at_forecast_start.tolist() == [True, False]
    assert summary.first_stimulus_bin.tolist() == [0, 1]
    times, observed_line, predicted_line = global_response_lines(summary, 0)
    np.testing.assert_allclose(times, [0, 5, 10, 15, 20])
    np.testing.assert_allclose(observed_line, [6, 22, 38, 8, 0])
    assert np.isnan(predicted_line[:history]).all()
    np.testing.assert_allclose(predicted_line[history:], [16, 0])


def test_spatial_response_metrics_use_coordinates_and_transport_distance():
    xy = np.array([[0, 0], [40, 0], [0, 40], [40, 40]], dtype=float)
    metrics = spatial_response_metrics(
        np.array([0, 0, 0, 4], dtype=float),
        np.array([0, 0, 0, 4], dtype=float),
        xy,
    )
    assert metrics["weighted_correlation"] == pytest.approx(1.0)
    assert metrics["wasserstein_distance"] == pytest.approx(0.0)
    assert metrics["centroid_distance"] == pytest.approx(0.0)

    shifted = spatial_response_metrics(
        np.array([4, 0, 0, 0], dtype=float),
        np.array([0, 0, 0, 4], dtype=float),
        xy,
    )
    assert shifted["centroid_distance"] == pytest.approx(np.sqrt(3200.0))
    assert shifted["wasserstein_distance"] > 0.0
