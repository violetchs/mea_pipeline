"""Conftest for test fixtures."""

import pytest
import numpy as np


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    rng = np.random.default_rng(7)
    return rng.normal(size=(4, 256))
