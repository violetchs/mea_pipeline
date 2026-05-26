"""Tests for visualization module."""

import pytest
import numpy as np

from src.visualization import Visualizer


class TestVisualizer:
    """Test Visualizer class."""
    
    def test_visualizer_initialization(self):
        """Test Visualizer initialization."""
        assert Visualizer() is not None

    def test_plot_timeseries_returns_figure(self):
        figure = Visualizer().plot_timeseries(np.ones((2, 16)))
        assert len(figure.axes) == 1
