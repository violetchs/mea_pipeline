# MEA Data Processing Pipeline

## Overview

This project is a Python pipeline for microelectrode array (MEA) data loading,
preprocessing, signal filtering, feature extraction, analysis, visualization,
and desktop GUI operation.

The current workspace was checked on Windows with Python 3.13.11. The package
metadata and dependency lower bounds are aligned with that runtime.

## Project Layout

```text
src/
  mea_io/             Data loading and saving
  preprocessing/      Cleaning, outlier clipping, and normalization
  signal_processing/  Filtering, signal features, and spike detection
  analysis/           Statistics and channel correlation
  visualization/      Matplotlib figures
  gui/                PySide6 desktop GUI
tests/                Unit tests
notebooks/            Example analysis script
docs/                 Project documentation
data/                 Local data directory
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install as a package:

```powershell
python -m pip install .
```

Install directly from a public GitHub repository:

```powershell
python -m pip install -U git+https://github.com/<owner>/<repo>.git
```

Install a released wheel from GitHub Releases:

```powershell
python -m pip install -U https://github.com/<owner>/<repo>/releases/download/v0.1.0/mea_pipeline-0.1.0-py3-none-any.whl
```

After installation, launch the GUI with either command:

```powershell
mea-pipeline
mea-pipeline-gui
```

On Windows, `mea-pipeline-app` is also installed as a GUI-script entry point.

For editable development installation:

```powershell
python -m pip install -e ".[dev]"
```

## Usage

The reader and writer support `.npy`, `.npz`, `.csv`, `.txt`, and `.tsv` array
files. Native `.mea` files are not implemented in this copy of the project.
Blackrock `.nev` spike-event files are supported through
`mea_io.read_blackrock_nev` without extra vendor libraries. Axion `.spk`
reading is exposed through `MEAReader` and `src.mea_io.read_axion_spk`; it
requires MATLAB Engine for Python plus AxionFileLoader and preserves well IDs
in channel names such as `A1_r1c1`.

```python
from mea_io import MEAReader
from preprocessing import Preprocessor
from signal_processing import SignalProcessor
from analysis import Analyzer
from visualization import Visualizer

reader = MEAReader("data/sample.npy")
raw_data = reader.load_data()

preprocessed = Preprocessor().preprocess(raw_data)
processed = SignalProcessor().process(preprocessed)
features = Analyzer().extract_features(processed)

figure = Visualizer().plot_results(features)
```

Read a Blackrock NEV file:

```python
from mea_io import read_blackrock_nev

data = read_blackrock_nev("data/test/C13001.nev")
print(data.channels())
print(data.time_range())
```

Run the desktop GUI from the repository root without installing:

```powershell
python -m src.gui.app
```

Packaged installations do not include local `data/`, `.github/`, tests,
notebooks, or documentation folders. The installed package includes the default
channel-map config and Maxwell HDF5 compression plugin needed by the current
GUI and readers.

Run tests:

```powershell
python -m pytest
```

## Publishing

For public version updates, push this repository to GitHub and create a version
tag such as `v0.1.0`. The workflow in `.github/workflows/release.yml` builds the
wheel/source package and attaches them to a GitHub Release. See
`docs/PUBLISH.md` for the full release and user-installation workflow.

## Notes For This Workspace

- `git` is not available on the current PATH.
- The standalone `pytest` command is not available until dependencies are
  installed; use `python -m pytest` after installation.
- Generated Python bytecode, virtual environments, local data, processed output,
  and test caches are ignored by `.gitignore`.

## License

MIT License
