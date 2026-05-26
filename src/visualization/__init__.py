"""Visualization module for plotting MEA data and analysis results."""

import numpy as np
from matplotlib.figure import Figure

__all__ = ["Visualizer"]


class Visualizer:
    """Visualize MEA data and results."""
    
    def __init__(self):
        """Initialize Visualizer."""
        pass
    
    def plot_results(self, data):
        """
        Plot analysis results.
        
        Args:
            data: Analysis results or processed data
        """
        if isinstance(data, dict) and "correlation" in data:
            return self.plot_heatmap(data["correlation"])
        return self.plot_timeseries(data)
    
    def plot_heatmap(self, data):
        """Plot data as heatmap."""
        fig = Figure(figsize=(7, 5), tight_layout=True)
        ax = fig.add_subplot(111)
        image = ax.imshow(np.asarray(data), cmap="viridis", aspect="auto")
        ax.set_title("Channel Correlation")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Channel")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        return fig
    
    def plot_timeseries(self, data):
        """Plot time series data."""
        array = np.asarray(data, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        fig = Figure(figsize=(8, 4), tight_layout=True)
        ax = fig.add_subplot(111)
        max_channels = min(array.shape[0], 8)
        for index in range(max_channels):
            ax.plot(array[index] + index * 4.0, linewidth=0.8)
        ax.set_title("MEA Time Series")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Channel offset")
        return fig
    
    def plot_raster(self, spike_times):
        """Plot spike raster diagram."""
        fig = Figure(figsize=(8, 4), tight_layout=True)
        ax = fig.add_subplot(111)
        for channel, spikes in enumerate(spike_times):
            ax.vlines(spikes, channel + 0.5, channel + 1.0, color="#2563eb", linewidth=0.8)
        ax.set_title("Spike Raster")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Channel")
        return fig
