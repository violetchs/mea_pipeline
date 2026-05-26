"""Tests for analysis module."""

import pytest
import numpy as np

from src.analysis import Analyzer


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
