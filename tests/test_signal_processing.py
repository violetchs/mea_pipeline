"""Tests for signal processing module."""

import pytest
import numpy as np

from src.signal_processing import SignalProcessor


class TestSignalProcessor:
    """Test SignalProcessor class."""
    
    def test_processor_initialization(self):
        """Test SignalProcessor initialization."""
        processor = SignalProcessor(sampling_rate=20000)
        assert processor.sampling_rate == 20000
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        features = SignalProcessor().extract_features(np.ones((2, 8)))
        np.testing.assert_allclose(features["rms"], np.ones(2))
        np.testing.assert_allclose(features["peak_to_peak"], np.zeros(2))
