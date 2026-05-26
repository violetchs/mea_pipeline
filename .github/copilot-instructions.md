# MEA Data Processing Pipeline - Project Instructions

## Project Overview

This repository contains a Python pipeline for microelectrode array (MEA) data
processing. It covers data I/O, preprocessing, signal processing, analysis,
visualization, and a PySide6 desktop GUI.

## Runtime

- Current checked environment: Windows, Python 3.13.11.
- Use `python -m pip` and `python -m pytest` so commands resolve against the
  active interpreter.
- `git` is not available on the current PATH in this workspace.

## Project Structure

- `src/mea_io/` - data loading and saving
- `src/preprocessing/` - cleaning, outlier clipping, and normalization
- `src/signal_processing/` - filtering, features, and spike detection
- `src/analysis/` - statistics and channel correlation
- `src/visualization/` - matplotlib figures
- `src/gui/` - PySide6 desktop GUI
- `tests/` - unit tests
- `notebooks/` - example workflow scripts
- `docs/` - documentation

## Development Commands

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
```

Run the GUI from the repository root:

```powershell
python -m src.gui.app
```

## Notes

- Supported data formats are `.npy`, `.npz`, `.csv`, `.txt`, and `.tsv`.
- Native `.mea` parsing is not currently implemented.
- Keep generated data, processed output, bytecode caches, and virtual
  environments out of version control.
