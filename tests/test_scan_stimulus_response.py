"""Regression tests for scan sidecars and stimulus-response grouping."""

import json

import numpy as np

from src.gui.app import _stimulus_response_records_from_data
from src.mea_io.spike_readers import read_unified_npz


def test_scan_spike_npz_recovers_local_sites_from_source_sidecars(tmp_path):
    source_dir = tmp_path / "raw" / "02_stim"
    source_dir.mkdir(parents=True)
    source_h5 = source_dir / "scan.raw.h5"
    source_h5.write_bytes(b"")

    times = [1.0, 3.0, 5.0, 7.0]
    (source_dir / "segment_time_meta.json").write_text(
        json.dumps({"stim_times_sec": times, "stim_records": []}), encoding="utf-8"
    )
    (source_dir / "stim_plan.json").write_text(
        json.dumps(
            {
                "candidate_electrodes": [101, 202],
                "stimuli": [
                    {"time_sec": 0.1, "electrodes": [101]},
                    {"time_sec": 2.1, "electrodes": [202]},
                    {"time_sec": 4.1, "electrodes": [101]},
                    {"time_sec": 6.1, "electrodes": [202]},
                ],
            }
        ),
        encoding="utf-8",
    )

    npz_dir = tmp_path / "spike_data2"
    npz_dir.mkdir()
    npz_path = npz_dir / "scan.raw_spike_train.npz"
    meta = json.dumps({"file": str(source_h5), "source": "maxwell_h5"})
    np.savez_compressed(
        npz_path,
        stim_times=np.asarray(times, dtype=float),
        meta_json=np.asarray(meta),
        spikes_101=np.asarray([1.02, 3.02, 5.02, 7.02], dtype=float),
        spikes_202=np.asarray([1.03, 3.03, 5.03, 7.03], dtype=float),
    )

    data = read_unified_npz(npz_path)
    records = data.meta.get("stimulus_records", [])
    assert [record.get("electrodes") for record in records] == [[101], [202], [101], [202]]

    grouped = _stimulus_response_records_from_data(
        npz_path,
        data,
        pre_ms=100.0,
        response_ms=100.0,
    )
    assert {tuple(record["stimulus_group_key"]) for record in grouped} == {(101,), (202,)}
    assert {record["trial_count"] for record in grouped} == {2}

