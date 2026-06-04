import re
import shutil
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_path(request):
    root = Path.cwd() / "test_tmp"
    root.mkdir(exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name).strip("._") or "test"
    path = root / name
    counter = 0
    while path.exists():
        counter += 1
        path = root / f"{name}_{counter}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def sample_data():
    return np.arange(24, dtype=float).reshape(4, 6)
