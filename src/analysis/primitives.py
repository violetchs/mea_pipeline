"""Reusable analysis primitives for flexible matrix-based workflows.

These helpers are intentionally generic: callers provide an array-like object,
choose how samples/features should be interpreted, then compose normalization,
similarity, reduction, clustering, and visualization around the returned
matrices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class PreparedMatrix:
    """Normalized view of an arbitrary array as a 2D samples x features matrix."""

    matrix: np.ndarray
    original_shape: tuple[int, ...]
    sample_axis: int
    sample_count: int
    feature_count: int


def prepare_feature_matrix(data, sample_axis: int = 0) -> PreparedMatrix:
    """Flatten arbitrary numeric data into a finite 2D samples x features matrix."""

    values = np.nan_to_num(np.asarray(data, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if values.ndim == 0:
        values = values.reshape((1, 1))
        sample_axis = 0
    elif values.ndim == 1:
        if sample_axis not in {0, -1}:
            sample_axis = 0
        values = values.reshape((-1, 1))
        sample_axis = 0
    else:
        sample_axis = int(sample_axis)
        if sample_axis < 0:
            sample_axis += values.ndim
        sample_axis = min(max(sample_axis, 0), values.ndim - 1)
        if sample_axis != 0:
            values = np.moveaxis(values, sample_axis, 0)
        values = values.reshape((values.shape[0], -1))
        sample_axis = 0
    return PreparedMatrix(
        matrix=np.asarray(values, dtype=float),
        original_shape=tuple(int(dim) for dim in np.asarray(data).shape) if np.asarray(data).ndim else tuple(),
        sample_axis=int(sample_axis),
        sample_count=int(values.shape[0]),
        feature_count=int(values.shape[1]) if values.ndim == 2 else 0,
    )


def normalize_feature_matrix(features: np.ndarray, method: str = "none") -> tuple[np.ndarray, dict]:
    """Apply a reusable normalization strategy to a samples x features matrix."""

    values = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    method = str(method or "none").strip().lower()
    metadata = {"method": method}
    if values.ndim != 2 or values.size == 0:
        return np.asarray(values, dtype=float), metadata
    if method in {"none", "raw"}:
        return values, metadata
    if method in {"per_sample_l1", "per_burst"}:
        totals = np.sum(np.abs(values), axis=1, keepdims=True)
        metadata["totals"] = totals
        return np.divide(values, totals, out=np.zeros_like(values), where=totals > 1e-12), metadata
    if method in {"feature_zscore", "unit_zscore"}:
        means = np.mean(values, axis=0, keepdims=True)
        stds = np.std(values, axis=0, keepdims=True)
        metadata["mean"] = means
        metadata["std"] = stds
        return np.divide(values - means, stds, out=np.zeros_like(values), where=stds > 1e-12), metadata
    if method == "global_zscore":
        mean = float(np.mean(values))
        std = float(np.std(values))
        metadata["mean"] = mean
        metadata["std"] = std
        if std <= 1e-12:
            return np.zeros_like(values), metadata
        return (values - mean) / std, metadata
    if method in {"sample_peak", "per_trace_peak"}:
        peaks = np.max(np.abs(values), axis=1, keepdims=True)
        metadata["peaks"] = peaks
        return np.divide(values, peaks, out=np.zeros_like(values), where=peaks > 1e-12), metadata
    if method == "log1p":
        return np.log1p(np.maximum(values, 0.0)), metadata
    if method == "robust_95":
        scale = float(np.nanpercentile(np.abs(values), 95.0)) if values.size else 0.0
        scale = max(scale, 1e-12)
        metadata["scale"] = scale
        return np.clip(values / scale, -1.0, 1.0), metadata
    return values, metadata


def compute_similarity_matrix(features: np.ndarray, method: str = "correlation") -> np.ndarray:
    """Compute a reusable sample-by-sample similarity matrix."""

    values = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    sample_count = int(values.shape[0]) if values.ndim == 2 else 0
    if sample_count == 0:
        return np.zeros((0, 0), dtype=float)
    if sample_count == 1:
        return np.ones((1, 1), dtype=float)
    method = str(method or "correlation").strip().lower()
    if method == "correlation":
        centered = values - np.mean(values, axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        normalized = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 1e-12)
        similarity = normalized @ normalized.T
        similarity = np.clip(similarity, -1.0, 1.0)
    elif method == "cosine":
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        normalized = np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)
        similarity = normalized @ normalized.T
        similarity = np.clip(similarity, -1.0, 1.0)
    elif method == "euclidean_affinity":
        squared_norms = np.sum(values ** 2, axis=1, keepdims=True)
        distance = np.sqrt(np.maximum(0.0, squared_norms + squared_norms.T - 2.0 * (values @ values.T)))
        positive = distance[distance > 1e-12]
        scale = float(np.median(positive)) if positive.size else 1.0
        scale = max(scale, 1e-12)
        similarity = np.exp(-0.5 * (distance / scale) ** 2)
    else:
        raise ValueError(f"Unsupported similarity method: {method}")
    similarity = np.nan_to_num(np.asarray(similarity, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    similarity = 0.5 * (similarity + similarity.T)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def hierarchical_order_and_groups(
    similarity: np.ndarray,
    threshold: float = 0.45,
    *,
    criterion: str = "distance",
    linkage_method: str = "average",
) -> tuple[np.ndarray, np.ndarray]:
    """Order samples and assign groups using hierarchical clustering."""

    values = np.nan_to_num(np.asarray(similarity, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    count = int(values.shape[0]) if values.ndim == 2 else 0
    if count == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    if count < 3:
        return np.arange(count, dtype=int), np.ones(count, dtype=int)
    distance = np.clip(1.0 - values, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    if condensed.size == 0 or np.allclose(condensed, 0.0):
        return np.arange(count, dtype=int), np.ones(count, dtype=int)
    tree = linkage(condensed, method=str(linkage_method or "average"))
    groups = fcluster(tree, t=max(0.0, float(threshold)), criterion=str(criterion or "distance")).astype(int)
    return leaves_list(tree).astype(int), groups


def assign_feature_clusters(
    features: np.ndarray,
    *,
    method: str = "kmeans",
    cluster_count: int = 3,
    similarity: np.ndarray | None = None,
    threshold: float = 0.45,
) -> np.ndarray:
    """Assign sample labels from generic features or a similarity matrix."""

    values = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    count = int(values.shape[0]) if values.ndim == 2 else 0
    if count == 0:
        return np.zeros(0, dtype=int)
    if count < 2:
        return np.ones(count, dtype=int)
    method = str(method or "kmeans").strip().lower()
    if method == "hierarchical":
        matrix = compute_similarity_matrix(values, "correlation") if similarity is None else similarity
        _, groups = hierarchical_order_and_groups(matrix, threshold=threshold, criterion="distance")
        return groups.astype(int)
    if method != "kmeans":
        raise ValueError(f"Unsupported clustering method: {method}")
    distinct_count = np.unique(np.round(values, decimals=12), axis=0).shape[0]
    effective_clusters = min(max(1, int(cluster_count)), count, distinct_count)
    if effective_clusters <= 1 or np.allclose(values, values[0]):
        return np.ones(count, dtype=int)
    return KMeans(n_clusters=effective_clusters, n_init=10, random_state=7).fit_predict(values).astype(int) + 1


def reduce_feature_matrix(
    features: np.ndarray,
    *,
    method: str = "pca",
    n_components: int = 2,
    standardize: bool = False,
) -> dict:
    """Reduce generic samples x features data and return coordinates plus metadata."""

    values = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    count = int(values.shape[0]) if values.ndim == 2 else 0
    if count == 0:
        return {"coordinates": np.zeros((0, max(1, int(n_components))), dtype=float), "method": str(method or "pca").lower()}
    if values.ndim != 2:
        values = values.reshape((count, -1))
    usable = values
    if standardize and usable.shape[0] >= 2 and usable.shape[1] >= 1:
        try:
            usable = StandardScaler().fit_transform(usable)
        except Exception:
            usable = values
    method = str(method or "pca").strip().lower()
    target_components = max(1, int(n_components))
    if method == "pca" or count < 3:
        components = min(target_components, usable.shape[0], usable.shape[1]) if usable.shape[1] else 1
        if components < 1:
            return {"coordinates": np.zeros((count, target_components), dtype=float), "method": "pca"}
        model = PCA(n_components=components, random_state=7)
        coordinates = model.fit_transform(usable)
        if coordinates.shape[1] < target_components:
            coordinates = np.column_stack([coordinates] + [np.zeros(count, dtype=float) for _ in range(target_components - coordinates.shape[1])])
        return {
            "coordinates": coordinates[:, :target_components],
            "method": "pca",
            "model": model,
            "explained_variance": np.asarray(getattr(model, "explained_variance_", np.zeros(0)), dtype=float),
            "explained_variance_ratio": np.asarray(getattr(model, "explained_variance_ratio_", np.zeros(0)), dtype=float),
            "prepared_features": usable,
        }
    if method == "tsne":
        perplexity = max(2, min(30, count - 1))
        try:
            coordinates = TSNE(
                n_components=target_components,
                perplexity=perplexity,
                random_state=7,
                init="pca",
                learning_rate="auto",
            ).fit_transform(usable)
        except Exception:
            return reduce_feature_matrix(usable, method="pca", n_components=target_components, standardize=False)
        return {"coordinates": coordinates, "method": "tsne", "prepared_features": usable}
    raise ValueError(f"Unsupported reduction method: {method}")


def run_generic_matrix_analysis(
    data,
    *,
    sample_axis: int = 0,
    normalization: str = "feature_zscore",
    similarity: str = "correlation",
    reduction: str = "pca",
    reduction_dims: int = 2,
    clustering: str = "kmeans",
    cluster_count: int = 3,
    grouping_threshold: float = 0.45,
    reduction_standardize: bool = False,
) -> dict:
    """End-to-end reusable analysis over arbitrary numeric arrays."""

    prepared = prepare_feature_matrix(data, sample_axis=sample_axis)
    normalized, normalization_meta = normalize_feature_matrix(prepared.matrix, method=normalization)
    similarity_matrix = compute_similarity_matrix(normalized, method=similarity)
    reduction_result = reduce_feature_matrix(
        normalized,
        method=reduction,
        n_components=reduction_dims,
        standardize=reduction_standardize,
    )
    coordinates = np.asarray(reduction_result.get("coordinates", np.zeros((prepared.sample_count, reduction_dims), dtype=float)), dtype=float)
    groups = assign_feature_clusters(
        coordinates if coordinates.ndim == 2 and coordinates.shape[1] > 0 else normalized,
        method=clustering,
        cluster_count=cluster_count,
        similarity=similarity_matrix,
        threshold=grouping_threshold,
    )
    order, hierarchical_groups = hierarchical_order_and_groups(similarity_matrix, threshold=grouping_threshold)
    return {
        "prepared": prepared,
        "normalized_matrix": normalized,
        "normalization": normalization_meta,
        "similarity_matrix": similarity_matrix,
        "reduction": reduction_result,
        "coordinates": coordinates,
        "groups": np.asarray(groups, dtype=int),
        "order": np.asarray(order, dtype=int),
        "hierarchical_groups": np.asarray(hierarchical_groups, dtype=int),
        "parameters": {
            "sample_axis": int(sample_axis),
            "normalization": str(normalization),
            "similarity": str(similarity),
            "reduction": str(reduction),
            "reduction_dims": int(reduction_dims),
            "clustering": str(clustering),
            "cluster_count": int(cluster_count),
            "grouping_threshold": float(grouping_threshold),
            "reduction_standardize": bool(reduction_standardize),
        },
    }


__all__ = [
    "PreparedMatrix",
    "assign_feature_clusters",
    "compute_similarity_matrix",
    "hierarchical_order_and_groups",
    "normalize_feature_matrix",
    "prepare_feature_matrix",
    "reduce_feature_matrix",
    "run_generic_matrix_analysis",
]
