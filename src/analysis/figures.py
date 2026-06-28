"""Reusable matplotlib figure builders for generic analysis modules."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure


def create_generic_analysis_figure(result: dict, *, title: str = "Generic analysis") -> Figure:
    """Visualize normalized data, similarity, embedding, and group sizes together."""

    figure = Figure(figsize=(11, 7), tight_layout=True)
    axes = figure.subplots(2, 2)

    normalized = np.asarray(result.get("normalized_matrix", []), dtype=float)
    similarity = np.asarray(result.get("similarity_matrix", []), dtype=float)
    coordinates = np.asarray(result.get("coordinates", []), dtype=float)
    groups = np.asarray(result.get("groups", []), dtype=int)
    order = np.asarray(result.get("order", []), dtype=int)
    reduction_meta = dict(result.get("reduction", {}) or {})
    parameters = dict(result.get("parameters", {}) or {})

    ax = axes[0, 0]
    if normalized.ndim == 2 and normalized.size:
        vmax = float(np.nanpercentile(np.abs(normalized), 98.0)) if normalized.size else 1.0
        vmax = max(vmax, 1e-9)
        image = ax.imshow(normalized, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title("Normalized feature matrix")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Sample")
        figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, "No normalized matrix", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes[0, 1]
    if similarity.ndim == 2 and similarity.size:
        display = similarity
        if order.size == similarity.shape[0]:
            display = similarity[np.ix_(order, order)]
        image = ax.imshow(display, aspect="auto", interpolation="nearest", cmap="viridis", vmin=np.min(display), vmax=np.max(display))
        ax.set_title("Similarity matrix")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Sample")
        figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, "No similarity matrix", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes[1, 0]
    if coordinates.ndim == 2 and coordinates.shape[0] > 0:
        x = coordinates[:, 0]
        y = coordinates[:, 1] if coordinates.shape[1] > 1 else np.zeros(coordinates.shape[0], dtype=float)
        if groups.size == coordinates.shape[0]:
            scatter = ax.scatter(x, y, c=groups, cmap="tab10", s=28, alpha=0.88, linewidths=0.2, edgecolors="white")
            figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.scatter(x, y, color="#2563eb", s=28, alpha=0.88)
        method = str(reduction_meta.get("method", parameters.get("reduction", "pca"))).upper()
        ax.set_title(f"{method} embedding")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
    else:
        ax.text(0.5, 0.5, "No embedding", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes[1, 1]
    if groups.size:
        labels, counts = np.unique(groups, return_counts=True)
        ax.bar(np.arange(labels.size), counts, color="#2563eb", alpha=0.82)
        ax.set_xticks(np.arange(labels.size))
        ax.set_xticklabels([str(int(label)) for label in labels])
        ax.set_title("Cluster / group sizes")
        ax.set_xlabel("Group")
        ax.set_ylabel("Samples")
    else:
        ax.text(0.5, 0.5, "No groups", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    figure.suptitle(title)
    return figure


__all__ = ["create_generic_analysis_figure"]
