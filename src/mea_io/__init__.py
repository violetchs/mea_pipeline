"""MEA I/O module for reading and writing MEA data."""

from pathlib import Path

import numpy as np
import pandas as pd

from .spike_readers import (
    AxionSpkReader,
    UnifiedMEAData,
    filter_unified_by_wells,
    list_axion_spk_wells,
    load_axion_cleaned_json,
    read_axion_spk,
    read_blackrock_nev,
    read_maxwell_h5,
    read_unified_npz,
    save_spike_train_npz,
    save_unified_npz,
)

__all__ = [
    "AxionSpkReader",
    "MEAReader",
    "MEAWriter",
    "UnifiedMEAData",
    "filter_unified_by_wells",
    "list_axion_spk_wells",
    "load_axion_cleaned_json",
    "read_axion_spk",
    "read_blackrock_nev",
    "read_maxwell_h5",
    "read_unified_npz",
    "save_spike_train_npz",
    "save_unified_npz",
]


class MEAReader:
    """Read MEA data from various file formats."""
    
    def __init__(self, filepath):
        """Initialize MEAReader with a file path."""
        self.filepath = filepath
    
    def load_data(self):
        """Load MEA data from file."""
        path = Path(self.filepath)
        if not path.exists():
            raise FileNotFoundError(f"MEA data file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.load(path, allow_pickle=False)
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if not archive.files:
                    raise ValueError(f"NPZ file contains no arrays: {path}")
                if any(key.startswith("spikes_") for key in archive.files):
                    return read_unified_npz(path)
                if "data" in archive.files:
                    return np.asarray(archive["data"])
                return np.asarray(archive[archive.files[0]])
        if suffix == ".spk":
            return read_axion_spk(path)
        if suffix in {".h5", ".hdf5"}:
            return read_maxwell_h5(path)
        if suffix in {".csv", ".txt", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else None
            frame = pd.read_csv(path, sep=sep, engine="python", header=None)
            return frame.to_numpy(dtype=float)

        raise ValueError(
            f"Unsupported file format '{suffix}'. Use .npy, .npz, .csv, .txt, .tsv, .spk, or .h5."
        )


class MEAWriter:
    """Write MEA data to file."""
    
    def __init__(self, filepath):
        """Initialize MEAWriter with a file path."""
        self.filepath = filepath
    
    def save_data(self, data):
        """Save MEA data to file."""
        path = Path(self.filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()

        if suffix == ".npy":
            np.save(path, data)
            return path
        if suffix == ".npz":
            np.savez_compressed(path, data=data)
            return path
        if suffix in {".csv", ".txt", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            np.savetxt(path, np.asarray(data), delimiter=delimiter)
            return path

        raise ValueError(
            f"Unsupported output format '{suffix}'. Use .npy, .npz, .csv, .txt, or .tsv."
        )
