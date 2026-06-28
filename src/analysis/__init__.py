"""Reusable analysis utilities and compatibility helpers."""

from __future__ import annotations

import numpy as np

from .figures import create_generic_analysis_figure
from .pivae import PiVAEAnalysisConfig, derive_pivae_covariates, fit_pivae_latent_states
from .primitives import (
    PreparedMatrix,
    assign_feature_clusters,
    compute_similarity_matrix,
    hierarchical_order_and_groups,
    normalize_feature_matrix,
    prepare_feature_matrix,
    reduce_feature_matrix,
    run_generic_matrix_analysis,
)

__all__ = [
    "Analyzer",
    "PreparedMatrix",
    "PiVAEAnalysisConfig",
    "assign_feature_clusters",
    "compute_similarity_matrix",
    "create_generic_analysis_figure",
    "derive_pivae_covariates",
    "fit_pivae_latent_states",
    "hierarchical_order_and_groups",
    "normalize_feature_matrix",
    "prepare_feature_matrix",
    "reduce_feature_matrix",
    "run_generic_matrix_analysis",
]


class Analyzer:
    """Simple compatibility wrapper around the reusable primitives."""

    def extract_features(self, data):
        prepared = prepare_feature_matrix(data, sample_axis=0)
        stats = self.compute_statistics(prepared.matrix)
        correlation = self.correlate_channels(prepared.matrix)
        generic = run_generic_matrix_analysis(
            prepared.matrix,
            sample_axis=0,
            normalization="feature_zscore",
            similarity="correlation",
            reduction="pca",
            reduction_dims=2,
            clustering="kmeans",
            cluster_count=min(3, max(1, prepared.sample_count)),
        )
        return {
            "statistics": stats,
            "correlation": correlation,
            "channel_count": int(prepared.sample_count),
            "sample_count": int(prepared.feature_count),
            "generic_analysis": generic,
        }

    def compute_statistics(self, data):
        array = np.asarray(data, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return {
            "mean": np.mean(array, axis=1),
            "std": np.std(array, axis=1),
            "min": np.min(array, axis=1),
            "max": np.max(array, axis=1),
            "rms": np.sqrt(np.mean(np.square(array), axis=1)),
        }

    def correlate_channels(self, data):
        array = np.asarray(data, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[0] < 2:
            return np.ones((array.shape[0], array.shape[0]))
        return np.nan_to_num(np.corrcoef(array))
