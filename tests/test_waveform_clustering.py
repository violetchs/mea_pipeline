"""Tests for NEV waveform clustering."""

import numpy as np

from src.mea_io import UnifiedMEAData
from src.sorting import WaveformClusteringConfig, cluster_nev_waveforms, waveform_embedding


def test_cluster_nev_waveforms_separates_synthetic_channel():
    rng = np.random.default_rng(11)
    base_a = np.r_[np.linspace(0, -1, 12), np.linspace(-1, 0.2, 12)]
    base_b = np.r_[np.linspace(0, 1, 12), np.linspace(1, -0.2, 12)]
    waveforms = np.vstack(
        [
            base_a + rng.normal(scale=0.03, size=24)
            for _ in range(40)
        ]
        + [
            base_b + rng.normal(scale=0.03, size=24)
            for _ in range(40)
        ]
    )
    data = UnifiedMEAData(
        spikes={"chan1": np.linspace(0, 1, waveforms.shape[0])},
        waveforms={"chan1": waveforms},
    )

    sorting = cluster_nev_waveforms(
        data,
        WaveformClusteringConfig(max_clusters=3, min_spikes=10, random_state=3),
    )

    labels = sorting["channels"]["chan1"]["labels"]
    assert labels.shape == (80,)
    assert np.unique(labels).size == 2
    assert sorting["summary"]["sorted_channels"] == 1


def test_cluster_nev_waveforms_supports_ica_and_gmm():
    rng = np.random.default_rng(13)
    waveforms = np.vstack(
        [
            rng.normal(loc=-0.6, scale=0.05, size=(30, 16)),
            rng.normal(loc=0.6, scale=0.05, size=(30, 16)),
        ]
    )
    data = UnifiedMEAData(
        spikes={"chan1": np.linspace(0, 1, waveforms.shape[0])},
        waveforms={"chan1": waveforms},
    )

    sorting = cluster_nev_waveforms(
        data,
        WaveformClusteringConfig(
            reduction_method="ica",
            clustering_method="gmm",
            max_clusters=3,
            min_spikes=10,
            ica_max_iter=1000,
            random_state=5,
        ),
    )

    payload = sorting["channels"]["chan1"]
    assert payload["embedding"].shape[0] == 60
    assert payload["labels"].shape == (60,)


def test_cluster_nev_waveforms_preserves_existing_noise_labels():
    rng = np.random.default_rng(17)
    waveforms = np.vstack(
        [
            rng.normal(loc=-0.8, scale=0.04, size=(30, 16)),
            rng.normal(loc=0.8, scale=0.04, size=(30, 16)),
            rng.normal(loc=0.0, scale=0.5, size=(6, 16)),
        ]
    )
    existing_labels = np.asarray([0] * 60 + [-1] * 6, dtype=np.int32)
    data = UnifiedMEAData(
        spikes={"chan1": np.linspace(0, 1, waveforms.shape[0])},
        waveforms={"chan1": waveforms},
        sorting={"chan1": {"waveform_cluster_labels": existing_labels}},
    )

    sorting = cluster_nev_waveforms(
        data,
        WaveformClusteringConfig(max_clusters=3, min_spikes=10, random_state=9),
    )

    payload = sorting["channels"]["chan1"]
    labels = payload["labels"]

    assert labels.shape == (66,)
    assert payload["embedding"].shape[0] == 66
    assert labels[-6:].tolist() == [-1] * 6
    assert -1 not in labels[:60]
    assert np.unique(labels[:60]).size == 2
    assert payload["noise_count"] == 6
    assert payload["clustered_spike_count"] == 60


def test_cluster_nev_waveforms_can_sort_selected_channel_only():
    rng = np.random.default_rng(23)
    waveforms_1 = rng.normal(loc=-0.3, scale=0.05, size=(30, 12))
    waveforms_2 = np.vstack(
        [
            rng.normal(loc=-0.8, scale=0.04, size=(25, 12)),
            rng.normal(loc=0.8, scale=0.04, size=(25, 12)),
        ]
    )
    data = UnifiedMEAData(
        spikes={
            "chan1": np.linspace(0, 1, waveforms_1.shape[0]),
            "chan2": np.linspace(0, 1, waveforms_2.shape[0]),
        },
        waveforms={
            "chan1": waveforms_1,
            "chan2": waveforms_2,
        },
    )

    sorting = cluster_nev_waveforms(
        data,
        WaveformClusteringConfig(max_clusters=3, min_spikes=10, random_state=12),
        channels=["chan2"],
    )

    assert list(sorting["channels"]) == ["chan2"]
    assert sorting["channels"]["chan2"]["labels"].shape == (50,)
    assert sorting["summary"]["sorted_channels"] == 1


def test_waveform_embedding_ica_returns_finite_fallback_for_degenerate_input():
    waveforms = np.ones((20, 12), dtype=float)
    waveforms[0, 0] = np.nan
    waveforms[1, 1] = np.inf

    embedding = waveform_embedding(
        waveforms,
        WaveformClusteringConfig(reduction_method="ica", ica_components=5, ica_max_iter=10),
    )

    assert embedding.shape[0] == 20
    assert np.isfinite(embedding).all()


def test_waveform_embedding_pca_is_stable_for_large_rank_deficient_input():
    rng = np.random.default_rng(31)
    base = rng.normal(size=(5000, 48))
    base[:, 24:] = base[:, :24] + rng.normal(scale=0.001, size=(5000, 24))
    config = WaveformClusteringConfig(reduction_method="pca", pca_components=8, random_state=23)

    first = waveform_embedding(base, config)
    second = waveform_embedding(base, config)

    assert first.shape == (5000, 8)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second)


def test_waveform_embedding_ica_is_stable_across_repeated_calls():
    rng = np.random.default_rng(29)
    base = rng.normal(size=(80, 24))
    base[:, 5:] = base[:, :19] * 0.8 + rng.normal(scale=0.01, size=(80, 19))
    config = WaveformClusteringConfig(reduction_method="ica", ica_components=6, ica_max_iter=80, random_state=19)

    for _ in range(8):
        embedding = waveform_embedding(base, config)
        assert embedding.shape[0] == 80
        assert embedding.shape[1] <= 6
        assert np.isfinite(embedding).all()
