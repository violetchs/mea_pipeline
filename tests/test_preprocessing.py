"""Tests for preprocessing module."""

import pytest
import numpy as np

from src.preprocessing import Preprocessor


class TestPreprocessor:
    """Test Preprocessor class."""
    
    def test_preprocessor_initialization(self):
        """Test Preprocessor initialization."""
        preprocessor = Preprocessor()
        assert preprocessor.normalize_data is True
    
    def test_normalize(self):
        """Test data normalization."""
        data = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
        normalized = Preprocessor().normalize(data)
        np.testing.assert_allclose(normalized[0].mean(), 0.0, atol=1e-12)
        np.testing.assert_allclose(normalized[0].std(), 1.0, atol=1e-12)
        np.testing.assert_allclose(normalized[1], np.zeros(3))
