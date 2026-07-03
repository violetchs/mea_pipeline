"""Readers for spike-based MEA data formats.

The Blackrock NEV reader is implemented locally so this project can read NEV
files without depending on the vendor-distributed ``brpylib.py`` module. Axion
SPK support is kept as an optional MATLAB-based reader because that format
requires Axion's MATLAB loader in typical workflows.
"""

from __future__ import annotations

import json
import mmap
import os
import re
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

AXIONBIO_SIGNATURE = b"AxionBio"
AXIONBIO_SPIKE_MARKER = b"\x06\x00\x00\x00Spikes"
AXIONBIO_TIMESTAMP_FREQUENCY_HZ = 50000.0


def _raise_if_cancelled(cancel_check=None) -> None:
    if cancel_check is not None and bool(cancel_check()):
        raise InterruptedError("Operation cancelled")


__all__ = [
    "AxionSpkReader",
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


@dataclass
class UnifiedMEAData:
    """In-memory representation shared by spike file readers.

    Attributes:
        spikes: Mapping of channel name to spike timestamps in seconds.
        waveforms: Mapping of channel name to waveform arrays shaped
            ``(n_spikes, n_samples)``. Waveforms are stored in microvolts when
            the source header provides enough scale information.
        sr: Waveform sampling rate in Hz, if known.
        stim_times: Stimulus/event timestamps in seconds.
        bad_intervals: User-curated bad time intervals.
        meta: Source-specific metadata.
        sorting: Optional spike sorting labels or embeddings per channel.
    """

    spikes: Dict[str, np.ndarray] = field(default_factory=dict)
    waveforms: Dict[str, np.ndarray] = field(default_factory=dict)
    sr: Optional[float] = None
    stim_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    bad_intervals: List[Tuple[float, float]] | np.ndarray = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    sorting: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def channels(self) -> List[str]:
        return list(self.spikes.keys())

    def time_range(self) -> Tuple[float, float]:
        tmin = np.inf
        tmax = -np.inf
        for timestamps in self.spikes.values():
            if timestamps.size:
                tmin = min(tmin, float(timestamps.min()))
                tmax = max(tmax, float(timestamps.max()))
        stim_times = np.asarray(self.stim_times, dtype=float)
        if stim_times.size:
            stim_times = stim_times[np.isfinite(stim_times)]
            if stim_times.size:
                tmin = min(tmin, float(stim_times.min()))
                tmax = max(tmax, float(stim_times.max()))

        if not np.isfinite(tmin):
            return 0.0, 0.0
        return float(tmin), float(tmax)


class AxionSpkReader:
    """Read Axion ``.spk`` files through MATLAB and AxionFileLoader.

    This reader intentionally imports ``matlab.engine`` only when SPK reading is
    requested. A missing MATLAB installation should not prevent NEV files from
    being read.
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.eng = None

    def start_matlab(self) -> None:
        try:
            import matlab.engine
        except ImportError as exc:
            raise ImportError(
                "Axion .spk reading requires MATLAB Engine for Python and the "
                "AxionFileLoader toolbox."
            ) from exc

        self.eng = matlab.engine.start_matlab()

    def read_spk(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not self.file_path.is_file():
            raise FileNotFoundError(f"SPK file not found: {self.file_path}")
        if self.eng is None:
            self.start_matlab()

        assert self.eng is not None
        spk_path = str(self.file_path)
        self.eng.cd(str(self.file_path.parent), nargout=0)
        self.eng.eval(f"f = AxisFile('{spk_path}');", nargout=0)
        self.eng.eval("Data = f.SpikeData.LoadData;", nargout=0)

        size = self.eng.eval("size(Data);", nargout=1)
        well_rows, well_cols, elec_cols, elec_rows = map(int, size._data)

        records: List[Dict[str, Any]] = []
        for well_row in range(1, well_rows + 1):
            for well_col in range(1, well_cols + 1):
                well_label = chr(ord("A") + well_row - 1) + str(well_col)
                for elec_col in range(1, elec_cols + 1):
                    for elec_row in range(1, elec_rows + 1):
                        electrode_label = f"r{elec_row}c{elec_col}"
                        index_str = f"{{{well_row},{well_col},{elec_col},{elec_row}}}"

                        try:
                            is_empty = self.eng.eval(f"isempty(Data{index_str})", nargout=1)
                        except Exception:
                            is_empty = True

                        if is_empty:
                            spike_times = None
                            spike_waveform = None
                        else:
                            try:
                                self.eng.eval(
                                    f"spikeTimes = [Data{index_str}(:).Start];",
                                    nargout=0,
                                )
                                self.eng.eval(
                                    f"spikeWaveform = Data{index_str}(:).GetVoltageVector;",
                                    nargout=0,
                                )
                                spike_times = np.asarray(self.eng.workspace["spikeTimes"])
                                spike_waveform = np.asarray(self.eng.workspace["spikeWaveform"])
                                spike_times = spike_times.reshape(-1)
                                spike_waveform = spike_waveform.T
                            except Exception:
                                spike_times = None
                                spike_waveform = None

                        records.append(
                            {
                                "well": well_label,
                                "electrode": electrode_label,
                                "data": {
                                    "spike_times": spike_times,
                                    "spike_waveform": spike_waveform,
                                },
                            }
                        )

        try:
            self.eng.eval("SE = f.StimulationEvents;", nargout=0)
            self.eng.eval("stimTimes = [SE(:).EventTime];", nargout=0)
            self.eng.eval("stimSamples = [SE(:).EventTimeSample];", nargout=0)
            stim_events = [
                {"EventTime": t, "EventTimeSample": s}
                for t, s in zip(
                    self.eng.workspace["stimTimes"],
                    self.eng.workspace["stimSamples"],
                )
            ]
        except Exception:
            stim_events = []

        return records, stim_events

    def close(self) -> None:
        if self.eng:
            self.eng.quit()
            self.eng = None


def _axion_channel_name(well: str, electrode: str) -> str:
    well = str(well or "").strip()
    electrode = str(electrode or "").strip()
    return f"{well}_{electrode}" if well and electrode else well or electrode


def _normalize_well_filter(wells: str | List[str] | Tuple[str, ...] | set[str] | None) -> set[str] | None:
    if wells is None:
        return None
    if isinstance(wells, str):
        values = [wells]
    else:
        values = list(wells)
    selected = {str(value).strip() for value in values if str(value).strip()}
    return selected or None


def _well_for_channel(data: UnifiedMEAData, channel: str) -> str:
    channel_map = data.meta.get("channel_map", {}) if isinstance(data.meta, dict) else {}
    payload = channel_map.get(channel, {}) if isinstance(channel_map, dict) else {}
    if isinstance(payload, dict) and payload.get("well"):
        return str(payload.get("well"))
    text = str(channel)
    return text.split("_", 1)[0] if "_" in text else ""


def filter_unified_by_wells(data: UnifiedMEAData, wells: str | List[str] | Tuple[str, ...] | set[str] | None) -> UnifiedMEAData:
    selected = _normalize_well_filter(wells)
    if not selected:
        return data

    keep_channels = [channel for channel in data.channels() if _well_for_channel(data, channel) in selected]
    channel_set = set(keep_channels)
    meta = dict(data.meta)
    raw_channel_map = meta.get("channel_map", {})
    if isinstance(raw_channel_map, dict):
        meta["channel_map"] = {channel: payload for channel, payload in raw_channel_map.items() if channel in channel_set}
    meta["selected_wells"] = sorted(selected)
    meta["wells"] = sorted({_well_for_channel(data, channel) for channel in keep_channels if _well_for_channel(data, channel)})
    meta["well_count"] = len(meta["wells"])
    meta["filtered_channel_count"] = len(keep_channels)
    meta["filtered_spike_count"] = int(sum(np.asarray(data.spikes[channel]).size for channel in keep_channels))
    return UnifiedMEAData(
        spikes={channel: data.spikes[channel] for channel in keep_channels if channel in data.spikes},
        waveforms={channel: data.waveforms[channel] for channel in keep_channels if channel in data.waveforms},
        sr=data.sr,
        stim_times=data.stim_times,
        bad_intervals=data.bad_intervals,
        meta=meta,
        sorting={channel: data.sorting[channel] for channel in keep_channels if channel in data.sorting}
        if isinstance(data.sorting, dict)
        else {},
    )


def _axion_electrode_position(electrode: str) -> Dict[str, int | str]:
    text = str(electrode or "").strip().lower()
    if text.startswith("r") and "c" in text:
        row_text, col_text = text[1:].split("c", 1)
        try:
            row = int(row_text)
            col = int(col_text)
            return {
                "electrode": f"r{row}c{col}",
                "electrode_row": row,
                "electrode_col": col,
                "mea_electrode": f"{chr(ord('A') + row - 1)}{col}" if 1 <= row <= 26 else "",
            }
        except ValueError:
            pass
    return {"electrode": str(electrode), "electrode_row": 0, "electrode_col": 0, "mea_electrode": ""}


def _axion_spike_payload(record: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray | None]:
    payload = record.get("data", {}) if isinstance(record, dict) else {}
    spike_times = payload.get("spike_times") if isinstance(payload, dict) else None
    spike_waveform = payload.get("spike_waveform") if isinstance(payload, dict) else None

    times = np.asarray([] if spike_times is None else spike_times, dtype=float).reshape(-1)
    times = times[np.isfinite(times)]

    if spike_waveform is None:
        return times, None
    waveforms = np.asarray(spike_waveform, dtype=float)
    if waveforms.ndim == 1:
        if times.size == 1:
            waveforms = waveforms.reshape(1, -1)
        else:
            waveforms = waveforms.reshape(-1, 1)
    if waveforms.ndim != 2:
        return times, None
    if times.size and waveforms.shape[0] != times.size and waveforms.shape[1] == times.size:
        waveforms = waveforms.T
    if times.size and waveforms.shape[0] != times.size:
        return times, None
    return times, waveforms


def _read_axionbio_metadata(path: Path) -> Dict[str, str]:
    with path.open("rb") as handle:
        raw = handle.read(256 * 1024)
    metadata: Dict[str, str] = {}
    for match in re.finditer(rb"[\x20-\x7E]{4,}", raw):
        text = match.group().decode("ascii", errors="replace")
        if "," not in text:
            continue
        for line in text.splitlines():
            if "," not in line:
                continue
            key, value = line.split(",", 1)
            key = key.strip()
            if key:
                metadata[key] = value.strip()
    return metadata


def _float_from_metadata(metadata: Dict[str, str], key: str) -> float | None:
    value = metadata.get(key)
    if not value:
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", value)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def _axionbio_timebase_from_header(raw: bytes) -> Tuple[float | None, float | None]:
    values: List[Tuple[int, float]] = []
    for offset in range(0, max(0, len(raw) - 8)):
        value = struct.unpack_from("<d", raw, offset)[0]
        if np.isfinite(value) and 1.0 <= value <= 200000.0:
            values.append((offset, float(value)))

    frequency_candidates = [(offset, value) for offset, value in values if 5000.0 <= value <= 200000.0]
    frequency_offset = frequency_candidates[0][0] if frequency_candidates else None
    frequency_hz = frequency_candidates[0][1] if frequency_candidates else None
    duration_candidates = [
        (offset, value)
        for offset, value in values
        if 1.0 <= value <= 24 * 60 * 60
        and (frequency_offset is None or offset > frequency_offset)
        and (frequency_hz is None or abs(value - frequency_hz) > max(1.0, frequency_hz * 0.01))
    ]
    duration_s = duration_candidates[0][1] if duration_candidates else None
    return frequency_hz, duration_s


def _find_axionbio_channel_entries(raw: bytes) -> Dict[int, Dict[str, Any]]:
    run_start = None
    run_count = 0
    best_start = None
    best_count = 0
    for offset in range(0, len(raw) - 8, 8):
        entry = raw[offset : offset + 8]
        valid = (
            entry[0] == 0
            and entry[1] == 0
            and 1 <= entry[2] <= 24
            and 1 <= entry[3] <= 16
            and 1 <= entry[4] <= 16
            and 1 <= entry[5] <= 16
            and 0 <= entry[6] <= 255
            and 0 <= entry[7] <= 255
        )
        if valid:
            if run_start is None:
                run_start = offset
                run_count = 0
            run_count += 1
        else:
            if run_start is not None and run_count > best_count:
                best_start = run_start
                best_count = run_count
            run_start = None
            run_count = 0
    if run_start is not None and run_count > best_count:
        best_start = run_start
        best_count = run_count

    if best_start is None or best_count < 64:
        return {}

    entries: Dict[int, Dict[str, Any]] = {}
    for index in range(best_count):
        entry = raw[best_start + index * 8 : best_start + (index + 1) * 8]
        well_col = int(entry[2])
        well_row = int(entry[3])
        electrode_col = int(entry[4])
        electrode_row = int(entry[5])
        well_index = int(entry[6])
        electrode_index = int(entry[7])
        code = (well_index << 8) | electrode_index
        well = f"{chr(ord('A') + well_row - 1)}{well_col}" if 1 <= well_row <= 26 else f"W{well_index}"
        electrode = f"r{electrode_row}c{electrode_col}"
        entries[code] = {
            "well": well,
            "well_row": well_row,
            "well_col": well_col,
            "well_index": well_index,
            "electrode": electrode,
            "electrode_row": electrode_row,
            "electrode_col": electrode_col,
            "electrode_index": electrode_index,
            "mea_electrode": f"{chr(ord('A') + electrode_row - 1)}{electrode_col}"
            if 1 <= electrode_row <= 26
            else "",
        }
    return entries


def _infer_axionbio_record_layout(mm: mmap.mmap, marker_offset: int) -> Tuple[int, int, int]:
    best: Tuple[int, int, int, int] | None = None
    for start_delta in range(len(AXIONBIO_SPIKE_MARKER), len(AXIONBIO_SPIKE_MARKER) + 16):
        data_offset = marker_offset + start_delta
        for waveform_samples in range(16, 96):
            record_bytes = 30 + waveform_samples * 2
            if data_offset + record_bytes * 8 > len(mm):
                continue
            score = 0
            checks = min(128, max(1, (len(mm) - data_offset) // record_bytes))
            for index in range(checks):
                offset = data_offset + index * record_bytes
                try:
                    code = struct.unpack_from("<H", mm, offset + 8)[0]
                    field_count = struct.unpack_from("<I", mm, offset + 10)[0]
                    threshold = struct.unpack_from("<d", mm, offset + 22)[0]
                except struct.error:
                    break
                if 0 <= code <= 0xFFFF and 0 < field_count < 64 and np.isfinite(threshold):
                    if field_count == 11 and 0.0 < threshold < 1000.0:
                        score += 3
                    else:
                        score += 1
            candidate = (score, data_offset, record_bytes, waveform_samples)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None or best[0] <= 0:
        raise ValueError("Could not infer AxionBio SPK spike record layout.")
    _, data_offset, record_bytes, waveform_samples = best
    return data_offset, record_bytes, waveform_samples


def _is_axionbio_spk(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(len(AXIONBIO_SIGNATURE)) == AXIONBIO_SIGNATURE


def list_axion_spk_wells(file_path: str | Path) -> List[str]:
    path = Path(file_path)
    if not _is_axionbio_spk(path):
        return []
    with path.open("rb") as handle:
        header_raw = handle.read(128 * 1024)
    entries = _find_axionbio_channel_entries(header_raw)
    wells = {str(payload.get("well")) for payload in entries.values() if payload.get("well")}
    return sorted(wells, key=lambda value: (value[:1], int(value[1:]) if value[1:].isdigit() else value))


def _read_axionbio_spk_native(path: Path, wells: str | List[str] | Tuple[str, ...] | set[str] | None = None) -> UnifiedMEAData:
    metadata = _read_axionbio_metadata(path)
    with path.open("rb") as handle:
        header_raw = handle.read(128 * 1024)
    channel_entries = _find_axionbio_channel_entries(header_raw)
    selected_wells = _normalize_well_filter(wells)

    pre_ms = _float_from_metadata(metadata, "Pre-Spike Duration")
    post_ms = _float_from_metadata(metadata, "Post-Spike Duration")
    voltage_scale = _float_from_metadata(metadata, "Voltage Scale")
    header_frequency_hz: float | None = None
    header_duration_s: float | None = None

    spikes: Dict[str, np.ndarray] = {}
    waveforms: Dict[str, np.ndarray] = {}
    channel_map: Dict[str, Dict[str, Any]] = {}
    wells: set[str] = set()
    electrodes: set[str] = set()
    duration_s = 0.0
    timestamp_offset_s = 0.0
    timestamp_frequency_hz = AXIONBIO_TIMESTAMP_FREQUENCY_HZ
    timestamp_frequency_source = "fallback_50000_hz"
    waveform_samples = 0
    record_count = 0
    valid_count = 0

    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        marker_offset = mm.find(AXIONBIO_SPIKE_MARKER)
        if marker_offset < 0:
            raise ValueError("AxionBio SPK file does not contain a Spikes section.")
        header_frequency_hz, header_duration_s = _axionbio_timebase_from_header(mm[:marker_offset])
        if header_frequency_hz is not None:
            timestamp_frequency_hz = float(header_frequency_hz)
            timestamp_frequency_source = "header_sampling_frequency"
        data_offset, record_bytes, waveform_samples = _infer_axionbio_record_layout(mm, marker_offset)
        record_count = max(0, (len(mm) - data_offset) // record_bytes)
        dtype = np.dtype(
            [
                ("sample", "<i8"),
                ("code", "<u2"),
                ("field_count", "<u4"),
                ("amplitude", "<f8"),
                ("threshold", "<f8"),
                ("waveform", "<i2", (waveform_samples,)),
            ]
        )
        records = np.memmap(path, dtype=dtype, mode="r", offset=data_offset, shape=(record_count,))
        valid = (
            (records["field_count"] == 11)
            & np.isfinite(records["amplitude"])
            & np.isfinite(records["threshold"])
            & (records["threshold"] > 0)
            & (records["threshold"] < 1000)
        )
        valid_indices = np.flatnonzero(valid)
        valid_count = int(valid_indices.size)
        if valid_count:
            valid_samples = records["sample"][valid_indices].astype(np.float64)
            min_sample = float(np.nanmin(valid_samples))
            max_sample = float(np.nanmax(valid_samples))
            if header_frequency_hz is not None:
                timestamp_offset_s = 0.0
                duration_s = max(
                    0.0 if header_duration_s is None else float(header_duration_s),
                    max_sample / timestamp_frequency_hz,
                    (max_sample - min_sample) / timestamp_frequency_hz,
                )
            else:
                raw_times = valid_samples / timestamp_frequency_hz
                timestamp_offset_s = float(np.nanmin(raw_times))
                duration_s = float(np.nanmax(raw_times) - timestamp_offset_s)

            codes = records["code"][valid_indices]
            for code in np.unique(codes):
                code_int = int(code)
                entry = channel_entries.get(code_int)
                if entry is None:
                    well_index = code_int >> 8
                    electrode_index = code_int & 0xFF
                    entry = {
                        "well": f"W{well_index}",
                        "well_index": well_index,
                        "well_row": 0,
                        "well_col": 0,
                        "electrode": f"e{electrode_index}",
                        "electrode_index": electrode_index,
                        "electrode_row": 0,
                        "electrode_col": 0,
                        "mea_electrode": "",
                    }
                if selected_wells is not None and str(entry.get("well", "")) not in selected_wells:
                    continue
                channel = _axion_channel_name(str(entry["well"]), str(entry["electrode"]))
                mask = codes == code
                indices = valid_indices[mask]
                times = records["sample"][indices].astype(np.float64) / timestamp_frequency_hz - timestamp_offset_s
                order = np.argsort(times)
                spikes[channel] = times[order]
                waveforms[channel] = np.asarray(records["waveform"][indices][order], dtype=np.int16)
                channel_map[channel] = {"channel": channel, "code": code_int, **entry}
                wells.add(str(entry["well"]))
                electrodes.add(str(entry["electrode"]))

    waveform_sampling_rate = header_frequency_hz
    if waveform_sampling_rate is None and pre_ms is not None and post_ms is not None and pre_ms + post_ms > 0:
        waveform_sampling_rate = float(waveform_samples) / ((pre_ms + post_ms) / 1000.0)

    meta = {
        "source": "axion_spk",
        "reader": "native_axionbio",
        "file": str(path),
        "metadata": metadata,
        "records_count": int(record_count),
        "valid_spike_count": int(valid_count),
        "wells": sorted(wells),
        "electrodes": sorted(electrodes),
        "channel_map": channel_map,
        "well_count": int(len(wells)),
        "electrode_count": int(len(electrodes)),
        "selected_wells": sorted(selected_wells) if selected_wells else [],
        "timestamp_offset_s": float(timestamp_offset_s),
        "timestamp_frequency_hz": float(timestamp_frequency_hz),
        "timestamp_frequency_source": timestamp_frequency_source,
        "header_duration_s": header_duration_s,
        "nominal_duration_s": header_duration_s,
        "duration_s": float(duration_s),
        "waveform_samples": int(waveform_samples),
        "waveform_unit": "adc_counts",
        "voltage_scale_v_per_count": voltage_scale,
        "voltage_scale_uv_per_count": None if voltage_scale is None else float(voltage_scale) * 1e6,
    }
    return UnifiedMEAData(spikes=spikes, waveforms=waveforms, sr=waveform_sampling_rate, meta=meta)


def read_axion_spk(
    file_path: str | Path,
    wells: str | List[str] | Tuple[str, ...] | set[str] | None = None,
) -> UnifiedMEAData:
    """Read an Axion ``.spk`` file into :class:`UnifiedMEAData`.

    The low-level Axion format is loaded through :class:`AxionSpkReader`, which
    uses MATLAB/AxionFileLoader when available. The conversion here preserves
    well identity by naming rows as ``"{well}_{electrode}"`` and stores a full
    channel-to-well/electrode mapping under ``meta["channel_map"]``.
    """

    path = Path(file_path)
    if _is_axionbio_spk(path):
        return _read_axionbio_spk_native(path, wells=wells)

    reader = AxionSpkReader(path)
    try:
        records, stim_events = reader.read_spk()
    finally:
        reader.close()

    spikes: Dict[str, np.ndarray] = {}
    waveforms: Dict[str, np.ndarray] = {}
    channel_map: Dict[str, Dict[str, Any]] = {}
    wells: set[str] = set()
    electrodes: set[str] = set()

    for record in records:
        well = str(record.get("well", "")).strip()
        electrode = str(record.get("electrode", "")).strip()
        channel = _axion_channel_name(well, electrode)
        if not channel:
            continue

        times, waveform = _axion_spike_payload(record)
        position = _axion_electrode_position(electrode)
        channel_map[channel] = {
            "channel": channel,
            "well": well,
            **position,
        }
        if well:
            wells.add(well)
        if electrode:
            electrodes.add(electrode)

        if times.size == 0:
            continue
        order = np.argsort(times)
        spikes[channel] = times[order]
        if waveform is not None:
            waveforms[channel] = waveform[order] if waveform.shape[0] == times.size else waveform

    timestamp_offset_s = min((float(values[0]) for values in spikes.values() if values.size), default=0.0)
    if timestamp_offset_s:
        for channel, values in list(spikes.items()):
            spikes[channel] = values - timestamp_offset_s

    duration_s = max((float(values[-1]) for values in spikes.values() if values.size), default=0.0)
    stim_times = []
    for event in stim_events:
        if isinstance(event, dict) and "EventTime" in event:
            try:
                stim_times.append(float(event["EventTime"]) - timestamp_offset_s)
            except (TypeError, ValueError):
                continue

    meta = {
        "source": "axion_spk",
        "file": str(path),
        "records_count": int(len(records)),
        "wells": sorted(wells),
        "electrodes": sorted(electrodes),
        "channel_map": channel_map,
        "well_count": int(len(wells)),
        "electrode_count": int(len(electrodes)),
        "timestamp_offset_s": float(timestamp_offset_s),
        "duration_s": float(duration_s),
        "stim_events": stim_events,
    }

    data = UnifiedMEAData(
        spikes=spikes,
        waveforms=waveforms,
        stim_times=np.asarray(stim_times, dtype=float),
        meta=meta,
    )
    return filter_unified_by_wells(data, wells)


def _cstring(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace").strip()


def _require_h5py():
    _ensure_local_hdf5_plugin_path()
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Maxwell .h5 reading requires h5py. Install it with `pip install h5py`.") from exc
    return h5py


def _ensure_local_hdf5_plugin_path() -> None:
    candidates = [
        Path(__file__).resolve().parents[2] / "tools" / "maxwell_hdf5_plugin",
        Path(__file__).resolve().parents[1] / "share" / "mea_pipeline" / "maxwell_hdf5_plugin",
        Path(sys.prefix) / "share" / "mea_pipeline" / "maxwell_hdf5_plugin",
    ]
    plugin_dir = next((candidate for candidate in candidates if candidate.is_dir()), None)
    if plugin_dir is None:
        return
    plugin_text = str(plugin_dir.resolve())
    current = os.environ.get("HDF5_PLUGIN_PATH", "")
    paths = [value for value in current.split(os.pathsep) if value]
    if plugin_text not in paths:
        os.environ["HDF5_PLUGIN_PATH"] = os.pathsep.join([plugin_text, *paths])


def _h5_scalar(group, key: str, default: Any = None) -> Any:
    if key not in group:
        return default
    try:
        value = group[key][()]
    except Exception:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\0")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace").strip("\0")
    return value.item() if hasattr(value, "item") else value


def _maxwell_group_sort_key(name: str) -> Tuple[str, int, str]:
    text = str(name)
    match = re.search(r"(\d+)$", text)
    return (text[: match.start()] if match else text, int(match.group(1)) if match else -1, text)


def _maxwell_well_label(group, fallback: str = "well0") -> str:
    raw = _h5_scalar(group, "well_id", None)
    if raw is None:
        return fallback
    try:
        return f"well{int(raw)}"
    except (TypeError, ValueError):
        text = str(raw).strip()
        return text or fallback


def _maxwell_data_groups(h5_file) -> List[Tuple[str, str, Any]]:
    groups: List[Tuple[str, str, Any]] = []
    if "data_store" in h5_file:
        store = h5_file["data_store"]
        for name in sorted(store.keys(), key=_maxwell_group_sort_key):
            obj = store[name]
            if "settings" in obj:
                groups.append((f"data_store/{name}", _maxwell_well_label(obj), obj))
    if groups:
        return groups

    if "wells" in h5_file:
        wells = h5_file["wells"]
        for well_name in sorted(wells.keys(), key=_maxwell_group_sort_key):
            well = wells[well_name]
            for rec_name in sorted(well.keys(), key=_maxwell_group_sort_key):
                obj = well[rec_name]
                if "settings" in obj:
                    groups.append((f"wells/{well_name}/{rec_name}", str(well_name), obj))
    return groups


def _maxwell_mapping_by_channel(group) -> Dict[int, Dict[str, Any]]:
    settings = group.get("settings")
    if settings is None or "mapping" not in settings:
        return {}
    mapping = np.asarray(settings["mapping"])
    result: Dict[int, Dict[str, Any]] = {}
    for row in mapping:
        try:
            channel = int(row["channel"])
        except Exception:
            continue
        if channel < 0:
            continue
        payload: Dict[str, Any] = {"source_channel": channel}
        for field_name in getattr(mapping.dtype, "names", ()) or ():
            value = row[field_name]
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, np.generic):
                value = value.item()
            payload[field_name] = value
        result[channel] = payload
    return result


def _maxwell_channel_name(well: str, channel_id: int, mapping: Dict[str, Any] | None) -> str:
    electrode = None if mapping is None else mapping.get("electrode")
    try:
        electrode_int = int(electrode)
    except (TypeError, ValueError):
        electrode_int = -1
    if electrode_int >= 0:
        return f"{well}_e{electrode_int}"
    return f"{well}_ch{int(channel_id)}"


def _maxwell_sampling_rate(group) -> float | None:
    settings = group.get("settings")
    if settings is None:
        return None
    value = _h5_scalar(settings, "sampling", None)
    try:
        sr = float(value)
    except (TypeError, ValueError):
        return None
    return sr if np.isfinite(sr) and sr > 0 else None


def _maxwell_raw_group_metadata(group, group_path: str, sr: float | None) -> Tuple[List[Dict[str, Any]], int | None]:
    raw_groups: List[Dict[str, Any]] = []
    first_frame: int | None = None
    if "groups" not in group:
        return raw_groups, first_frame

    for group_name in sorted(group["groups"].keys(), key=_maxwell_group_sort_key):
        raw_group = group["groups"][group_name]
        frame_nos_path = None
        if "frame_nos" in raw_group:
            frame_nos = raw_group["frame_nos"]
            frame_nos_path = f"{group_path}/groups/{group_name}/frame_nos"
            if frame_nos.shape and frame_nos.size:
                try:
                    value = int(frame_nos[0])
                    first_frame = value if first_frame is None else min(first_frame, value)
                except Exception:
                    pass

        if "raw" not in raw_group:
            continue
        raw = raw_group["raw"]
        payload: Dict[str, Any] = {
            "group": str(group_name),
            "path": f"{group_path}/groups/{group_name}/raw",
            "shape": tuple(int(value) for value in raw.shape),
            "dtype": str(raw.dtype),
            "frame_nos_path": frame_nos_path,
        }
        if sr and len(raw.shape) >= 2:
            payload["duration_s"] = float(raw.shape[1]) / float(sr)
        raw_groups.append(payload)
    return raw_groups, first_frame


def _maxwell_waveform_gain_uv(group) -> float | None:
    settings = group.get("settings")
    if settings is None:
        return None
    lsb = _h5_scalar(settings, "lsb", None)
    try:
        value = float(lsb)
    except (TypeError, ValueError):
        value = 0.0
    if np.isfinite(value) and value > 0:
        return value * 1e6

    gain = _h5_scalar(settings, "gain", None)
    try:
        gain_value = float(gain)
    except (TypeError, ValueError):
        gain_value = 0.0
    if np.isfinite(gain_value) and gain_value > 0:
        return 3.3 / (1024.0 * gain_value) * 1e6
    return None


def _maxwell_waveform_sample_counts(sr: float | None, window_ms: Tuple[float, float]) -> Tuple[int, int, int]:
    if sr is None or not np.isfinite(sr) or sr <= 0:
        return 0, 0, 0
    pre_ms, post_ms = window_ms
    pre = max(0, int(round(float(pre_ms) / 1000.0 * float(sr))))
    post = max(0, int(round(float(post_ms) / 1000.0 * float(sr))))
    return pre, post, pre + post + 1


def _maxwell_adc_zero_count(raw_dataset, group) -> float:
    dtype = np.dtype(raw_dataset.dtype)
    if not np.issubdtype(dtype, np.unsignedinteger):
        return 0.0

    settings = group.get("settings")
    if settings is not None:
        for key in ("adc_zero", "adc_zero_count", "zero_count", "adc_offset", "offset"):
            raw_value = _h5_scalar(settings, key, None)
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = 0.0
            if np.isfinite(value) and value:
                return value
        for key in ("adc_range", "adc_resolution", "bits"):
            raw_value = _h5_scalar(settings, key, None)
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = 0.0
            if np.isfinite(value) and value > 0:
                if value <= 32:
                    return float(2 ** int(value - 1))
                return value / 2.0

    return 0.0


def _maxwell_decode_raw_snippet(snippet: np.ndarray, zero_count: float) -> np.ndarray:
    snippet = np.asarray(snippet, dtype=np.float32)
    if np.isfinite(zero_count) and zero_count:
        snippet = snippet - np.float32(zero_count)
    return snippet


def _maxwell_extract_waveforms(
    group,
    group_path: str,
    channel_id: int,
    spike_frames: np.ndarray,
    sr: float | None,
    window_ms: Tuple[float, float],
    max_waveform_bytes: int,
) -> Tuple[np.ndarray | None, Dict[str, Any]]:
    pre, post, sample_count = _maxwell_waveform_sample_counts(sr, window_ms)
    info: Dict[str, Any] = {
        "data_group": group_path,
        "source_channel": int(channel_id),
        "pre_samples": int(pre),
        "post_samples": int(post),
        "sample_count": int(sample_count),
        "valid_count": 0,
        "skipped_count": int(len(spike_frames)),
        "error": "",
    }
    if sample_count <= 0 or spike_frames.size == 0 or "groups" not in group:
        return None, info

    estimated_bytes = int(spike_frames.size) * int(sample_count) * np.dtype(np.float32).itemsize
    if estimated_bytes > int(max_waveform_bytes):
        info["error"] = f"waveform extraction skipped: estimated {estimated_bytes} bytes exceeds limit"
        return None, info

    raw_group = None
    channel_row = -1
    for raw_group_name in sorted(group["groups"].keys(), key=_maxwell_group_sort_key):
        candidate = group["groups"][raw_group_name]
        if "raw" not in candidate or "channels" not in candidate or "frame_nos" not in candidate:
            continue
        try:
            channels = np.asarray(candidate["channels"])
        except Exception as exc:
            info["error"] = f"could not read raw channel list: {exc}"
            continue
        rows = np.flatnonzero(channels.astype(int) == int(channel_id))
        if rows.size:
            raw_group = candidate
            channel_row = int(rows[0])
            info["raw_group"] = str(raw_group_name)
            break

    if raw_group is None or channel_row < 0:
        info["error"] = "source channel is not present in any raw data group"
        return None, info

    try:
        frame_nos = np.asarray(raw_group["frame_nos"], dtype=np.int64)
    except Exception as exc:
        info["error"] = f"could not read raw frame numbers: {exc}"
        return None, info
    if frame_nos.size == 0:
        info["error"] = "raw frame_nos is empty"
        return None, info

    raw = raw_group["raw"]
    if raw.ndim != 2:
        info["error"] = f"unsupported raw matrix shape: {raw.shape}"
        return None, info

    gain_uv = _maxwell_waveform_gain_uv(group)
    zero_count = _maxwell_adc_zero_count(raw, group)
    waveforms = np.full((spike_frames.size, sample_count), np.nan, dtype=np.float32)
    valid = np.zeros(spike_frames.size, dtype=bool)
    first_frame = int(frame_nos[0])
    last_frame = int(frame_nos[-1])

    for index, frame in enumerate(np.asarray(spike_frames, dtype=np.int64)):
        start_frame = int(frame) - pre
        stop_frame = int(frame) + post
        if start_frame < first_frame or stop_frame > last_frame:
            continue
        start_index = int(start_frame - first_frame)
        stop_index = start_index + sample_count
        if start_index < 0 or stop_index > raw.shape[1]:
            continue
        if frame_nos[start_index] != start_frame or frame_nos[stop_index - 1] != stop_frame:
            located = int(np.searchsorted(frame_nos, int(frame)))
            start_index = located - pre
            stop_index = start_index + sample_count
            if start_index < 0 or stop_index > frame_nos.size:
                continue
            if frame_nos[start_index] != start_frame or frame_nos[stop_index - 1] != stop_frame:
                continue
        try:
            snippet = np.asarray(raw[channel_row, start_index:stop_index], dtype=np.float32)
        except Exception as exc:
            info["error"] = f"could not read raw waveform data: {exc}"
            return None, info
        if snippet.size != sample_count:
            continue
        snippet = _maxwell_decode_raw_snippet(snippet, zero_count)
        if gain_uv is not None:
            snippet = snippet * np.float32(gain_uv)
        waveforms[index] = snippet
        valid[index] = True

    info["valid_count"] = int(np.count_nonzero(valid))
    info["skipped_count"] = int(valid.size - np.count_nonzero(valid))
    info["unit"] = "uV" if gain_uv is not None else "adc_counts"
    info["adc_zero_count"] = float(zero_count)
    info["baseline_corrected"] = False
    return waveforms if np.any(valid) else None, info


def _maxwell_extract_group_waveforms(
    group,
    group_path: str,
    requests: Dict[str, Tuple[int, np.ndarray]],
    sr: float | None,
    window_ms: Tuple[float, float],
    max_waveform_bytes: int,
    cancel_check=None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    pre, post, sample_count = _maxwell_waveform_sample_counts(sr, window_ms)
    waveforms_by_channel: Dict[str, np.ndarray] = {}
    info_by_channel: Dict[str, Dict[str, Any]] = {}
    total_spikes = int(sum(np.asarray(frames).size for _, frames in requests.values()))
    estimated_bytes = total_spikes * int(sample_count) * np.dtype(np.float32).itemsize

    for channel_name, (channel_id, frames) in requests.items():
        info_by_channel[channel_name] = {
            "data_group": group_path,
            "source_channel": int(channel_id),
            "pre_samples": int(pre),
            "post_samples": int(post),
            "sample_count": int(sample_count),
            "valid_count": 0,
            "skipped_count": int(np.asarray(frames).size),
            "error": "",
        }

    if not requests or sample_count <= 0 or "groups" not in group:
        return waveforms_by_channel, info_by_channel
    if estimated_bytes > int(max_waveform_bytes):
        error = f"waveform extraction skipped: estimated {estimated_bytes} bytes exceeds limit"
        for info in info_by_channel.values():
            info["error"] = error
        return waveforms_by_channel, info_by_channel

    pending = set(requests)
    for raw_group_name in sorted(group["groups"].keys(), key=_maxwell_group_sort_key):
        _raise_if_cancelled(cancel_check)
        raw_group = group["groups"][raw_group_name]
        if "raw" not in raw_group or "channels" not in raw_group or "frame_nos" not in raw_group:
            continue
        try:
            raw_channels = np.asarray(raw_group["channels"], dtype=np.int64)
            frame_nos = np.asarray(raw_group["frame_nos"], dtype=np.int64)
        except Exception as exc:
            for channel_name in pending:
                info_by_channel[channel_name]["error"] = f"could not read raw channel/frame metadata: {exc}"
            continue
        if frame_nos.size == 0:
            continue

        row_by_channel = {int(channel): int(index) for index, channel in enumerate(raw_channels)}
        active_names = [
            channel_name
            for channel_name in list(pending)
            if int(requests[channel_name][0]) in row_by_channel
        ]
        if not active_names:
            continue

        raw = raw_group["raw"]
        if raw.ndim != 2:
            for channel_name in active_names:
                info_by_channel[channel_name]["error"] = f"unsupported raw matrix shape: {raw.shape}"
            pending.difference_update(active_names)
            continue

        gain_uv = _maxwell_waveform_gain_uv(group)
        zero_count = _maxwell_adc_zero_count(raw, group)
        unit = "uV" if gain_uv is not None else "adc_counts"
        first_frame = int(frame_nos[0])
        last_frame = int(frame_nos[-1])
        contiguous_frames = bool(frame_nos.size == 1 or np.all(np.diff(frame_nos) == 1))

        flat_records: List[Tuple[int, str, int, int]] = []
        for channel_name in active_names:
            channel_id, frames = requests[channel_name]
            frames = np.asarray(frames, dtype=np.int64)
            waveforms_by_channel[channel_name] = np.full((frames.size, sample_count), np.nan, dtype=np.float32)
            info = info_by_channel[channel_name]
            info["raw_group"] = str(raw_group_name)
            info["unit"] = unit
            info["adc_zero_count"] = float(zero_count)
            info["baseline_corrected"] = False
            row = row_by_channel[int(channel_id)]
            for spike_index, frame in enumerate(frames):
                if spike_index % 2048 == 0:
                    _raise_if_cancelled(cancel_check)
                start_frame = int(frame) - pre
                stop_frame = int(frame) + post
                if start_frame < first_frame or stop_frame > last_frame:
                    continue
                if contiguous_frames:
                    start_index = start_frame - first_frame
                else:
                    center_index = int(np.searchsorted(frame_nos, int(frame)))
                    start_index = center_index - pre
                    stop_index = start_index + sample_count
                    if start_index < 0 or stop_index > frame_nos.size:
                        continue
                    if frame_nos[start_index] != start_frame or frame_nos[stop_index - 1] != stop_frame:
                        continue
                flat_records.append((start_index, channel_name, spike_index, row))

        if flat_records:
            flat_records.sort(key=lambda item: item[0])
            chunk_samples = raw.chunks[1] if raw.chunks and len(raw.chunks) > 1 and raw.chunks[1] else sample_count
            block_samples = max(int(chunk_samples), sample_count, int(float(sr or 20000.0) * 0.5))
            cursor = 0
            while cursor < len(flat_records):
                _raise_if_cancelled(cancel_check)
                first_start = int(flat_records[cursor][0])
                block_start = max(0, (first_start // int(chunk_samples)) * int(chunk_samples))
                block_stop = min(raw.shape[1], block_start + block_samples)
                end = cursor
                while end < len(flat_records) and flat_records[end][0] + sample_count <= block_stop:
                    end += 1
                if end == cursor:
                    block_stop = min(raw.shape[1], flat_records[cursor][0] + sample_count)
                    end += 1
                try:
                    block = np.asarray(raw[:, block_start:block_stop], dtype=np.float32)
                except Exception as exc:
                    for channel_name in active_names:
                        info_by_channel[channel_name]["error"] = f"could not read raw waveform data: {exc}"
                    break

                for start_index, channel_name, spike_index, row in flat_records[cursor:end]:
                    local_start = int(start_index) - int(block_start)
                    local_stop = local_start + sample_count
                    snippet = block[int(row), local_start:local_stop]
                    if snippet.size != sample_count:
                        continue
                    snippet = _maxwell_decode_raw_snippet(snippet, zero_count)
                    if gain_uv is not None:
                        snippet = snippet * np.float32(gain_uv)
                    waveforms_by_channel[channel_name][spike_index] = snippet
                cursor = end

        for channel_name in active_names:
            values = waveforms_by_channel.get(channel_name)
            valid_count = int(np.count_nonzero(np.all(np.isfinite(values), axis=1))) if values is not None else 0
            info_by_channel[channel_name]["valid_count"] = valid_count
            info_by_channel[channel_name]["skipped_count"] = int(requests[channel_name][1].size - valid_count)
            if values is not None and valid_count == 0:
                waveforms_by_channel.pop(channel_name, None)
            pending.discard(channel_name)

    for channel_name in pending:
        if not info_by_channel[channel_name]["error"]:
            info_by_channel[channel_name]["error"] = "source channel is not present in any raw data group"
    return waveforms_by_channel, info_by_channel


def _maxwell_segment_duration_s(
    group,
    sr: float | None,
    spike_frames: np.ndarray,
    raw_groups: List[Dict[str, Any]],
) -> float:
    candidates: List[float] = []
    for payload in raw_groups:
        duration = payload.get("duration_s")
        if isinstance(duration, (int, float)) and np.isfinite(duration) and duration > 0:
            candidates.append(float(duration))

    if sr and spike_frames.size:
        frame_span = float(np.nanmax(spike_frames) - np.nanmin(spike_frames)) / float(sr)
        if np.isfinite(frame_span) and frame_span > 0:
            candidates.append(frame_span)

    start = _h5_scalar(group, "start_time", None)
    stop = _h5_scalar(group, "stop_time", None)
    try:
        delta = float(stop) - float(start)
    except (TypeError, ValueError):
        delta = 0.0
    if delta > 0:
        ref = max(candidates) if candidates else 0.0
        timebase_candidates: List[float] = []
        if sr:
            timebase_candidates.append(delta / float(sr))
        timebase_candidates.append(delta / 1000.0)
        valid = [value for value in timebase_candidates if np.isfinite(value) and 0 < value < 7 * 24 * 3600]
        if ref > 0 and valid:
            candidates.append(min(valid, key=lambda value: abs(value - ref)))
        elif valid:
            candidates.append(max(valid))

    return max(candidates, default=0.0)


def _maxwell_parse_event_message(message: Any) -> Dict[str, Any]:
    if message is None:
        return {}
    if isinstance(message, bytes):
        text = message.decode("utf-8", errors="replace").strip()
    else:
        text = str(message).strip()
    if not text:
        return {}

    parsed: Dict[str, Any] = {"stim_message": text}
    try:
        decoded = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return parsed
    if not isinstance(decoded, dict):
        parsed["stim_payload"] = decoded
        return parsed

    parsed["stim_payload"] = decoded
    search_text = " ".join(f"{key}={value}" for key, value in decoded.items())
    amp_match = re.search(r"amp_?mV\s*=\s*([-+]?\d+(?:\.\d+)?)", search_text, flags=re.IGNORECASE)
    phase_match = re.search(r"phase_?us\s*=\s*([-+]?\d+(?:\.\d+)?)", search_text, flags=re.IGNORECASE)
    pulse_match = re.search(r"pulse\s*=\s*([^\s,;]+)", search_text, flags=re.IGNORECASE)
    if amp_match:
        parsed["stim_amplitude_mV"] = float(amp_match.group(1))
    if phase_match:
        parsed["stim_phase_us"] = float(phase_match.group(1))
    if pulse_match:
        parsed["stim_pulse"] = pulse_match.group(1).replace("\\/", "/")

    label_parts = []
    if "stim_amplitude_mV" in parsed:
        label_parts.append(f"{parsed['stim_amplitude_mV']:g} mV")
    if "stim_phase_us" in parsed:
        label_parts.append(f"{parsed['stim_phase_us']:g} us")
    if "stim_pulse" in parsed:
        label_parts.append(f"pulse {parsed['stim_pulse']}")
    if label_parts:
        parsed["stim_label"] = ", ".join(label_parts)
    return parsed


def _maxwell_event_times(group, sr: float | None, frame_origin: int | None, segment_offset_s: float) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if sr is None or frame_origin is None or "events" not in group:
        return np.array([], dtype=float), []
    events = np.asarray(group["events"])
    if events.size == 0 or "frameno" not in (events.dtype.names or ()):
        return np.array([], dtype=float), []

    times = (events["frameno"].astype(float) - float(frame_origin)) / float(sr) + float(segment_offset_s)
    records: List[Dict[str, Any]] = []
    for index, event in enumerate(events):
        payload: Dict[str, Any] = {"time_s": float(times[index])}
        for field_name in events.dtype.names or ():
            value = event[field_name]
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except Exception:
                    pass
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            payload[field_name] = value
        payload.update(_maxwell_parse_event_message(payload.get("eventmessage")))
        payload.setdefault("event_source", "maxwell_h5")
        records.append(payload)
    return times[np.isfinite(times)], records


def _stim_artifact_keep_mask(times: np.ndarray, stim_times: np.ndarray, window_s: float) -> np.ndarray:
    values = np.asarray(times, dtype=float)
    if values.size == 0:
        return np.ones(values.shape, dtype=bool)
    events = np.asarray(stim_times, dtype=float)
    events = events[np.isfinite(events)]
    if events.size == 0 or window_s <= 0:
        return np.ones(values.shape, dtype=bool)
    events.sort()

    indices = np.searchsorted(events, values)
    artifact = np.zeros(values.shape, dtype=bool)
    right = indices < events.size
    if np.any(right):
        artifact[right] |= np.abs(events[indices[right]] - values[right]) <= window_s
    left = indices > 0
    if np.any(left):
        artifact[left] |= np.abs(values[left] - events[indices[left] - 1]) <= window_s
    return ~artifact


def read_maxwell_h5(
    file_path: str | Path,
    *,
    extract_waveforms: bool = True,
    waveform_window_ms: Tuple[float, float] = (1.0, 2.0),
    max_waveform_bytes: int = 512 * 1024 * 1024,
    stim_artifact_window_ms: float = 0.0,
    cancel_check=None,
) -> UnifiedMEAData:
    """Read Maxwell Biosystems ``.raw.h5`` files into :class:`UnifiedMEAData`.

    The reader follows Maxwell/Neo-style HDF5 layouts and extracts spike events,
    sampling rate, well/recording identifiers, and channel-electrode coordinates.
    When continuous raw datasets are present, spike-aligned waveform snippets
    are extracted using ``waveform_window_ms`` as ``(pre_ms, post_ms)``. If raw
    data cannot be read, for example because Maxwell's HDF5 compression plugin
    is missing, spike loading still succeeds and the extraction failure is kept
    in metadata. Stimulation artifacts are preserved by default; pass a positive
    ``stim_artifact_window_ms`` to remove spike timestamps within that window
    before or after each stimulation marker at read time.
    """

    h5py = _require_h5py()
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Maxwell H5 file not found: {path}")
    _raise_if_cancelled(cancel_check)

    spikes_acc: Dict[str, List[np.ndarray]] = defaultdict(list)
    channel_map: Dict[str, Dict[str, Any]] = {}
    event_times: List[np.ndarray] = []
    event_records: List[Dict[str, Any]] = []
    raw_data: List[Dict[str, Any]] = []
    data_groups_meta: List[Dict[str, Any]] = []
    wells: set[str] = set()
    recordings: set[str] = set()
    sampling_rates: List[float] = []
    segment_offsets: Dict[str, float] = defaultdict(float)
    total_spikes = 0

    with h5py.File(path, "r") as h5_file:
        version = _h5_scalar(h5_file, "version", "")
        data_groups = _maxwell_data_groups(h5_file)
        if not data_groups:
            raise ValueError("Unsupported Maxwell H5 layout: no readable data_store or wells recordings found.")

        waveforms_acc: Dict[str, List[np.ndarray]] = defaultdict(list)
        waveform_extraction: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for group_path, fallback_well, group in data_groups:
            _raise_if_cancelled(cancel_check)
            sr = _maxwell_sampling_rate(group)
            if sr:
                sampling_rates.append(float(sr))
            well = _maxwell_well_label(group, fallback_well)
            recording_id = _h5_scalar(group, "recording_id", group_path.rsplit("/", 1)[-1])
            recording = f"rec{recording_id}" if isinstance(recording_id, (int, np.integer)) else str(recording_id)
            wells.add(well)
            recordings.add(recording)

            mapping_by_channel = _maxwell_mapping_by_channel(group)
            group_raw_meta, raw_first_frame = _maxwell_raw_group_metadata(group, group_path, sr)
            raw_data.extend(group_raw_meta)

            spike_frames = np.array([], dtype=float)
            spike_data = None
            if "spikes" in group:
                spike_data = np.asarray(group["spikes"])
                if spike_data.size and "frameno" in (spike_data.dtype.names or ()):
                    spike_frames = spike_data["frameno"].astype(float)

            frame_origin = raw_first_frame
            if spike_frames.size:
                spike_min = int(np.nanmin(spike_frames))
                frame_origin = spike_min if frame_origin is None else min(frame_origin, spike_min)

            segment_offset_s = float(segment_offsets[well])
            segment_duration_s = _maxwell_segment_duration_s(group, sr, spike_frames, group_raw_meta)

            group_waveform_requests: Dict[str, Tuple[int, np.ndarray]] = {}
            if spike_data is not None and spike_data.size and sr and frame_origin is not None:
                names = spike_data.dtype.names or ()
                if "channel" not in names:
                    raise ValueError(f"Maxwell spikes dataset lacks a channel field: {group_path}/spikes")
                times = (spike_data["frameno"].astype(float) - float(frame_origin)) / float(sr) + segment_offset_s
                valid_times = np.isfinite(times)
                for channel_id in np.unique(spike_data["channel"]):
                    _raise_if_cancelled(cancel_check)
                    channel_int = int(channel_id)
                    mask = (spike_data["channel"] == channel_id) & valid_times
                    if not np.any(mask):
                        continue
                    mapping = mapping_by_channel.get(channel_int, {"source_channel": channel_int})
                    channel_name = _maxwell_channel_name(well, channel_int, mapping)
                    values = np.asarray(times[mask], dtype=float)
                    order = np.argsort(values)
                    values = values[order]
                    spikes_acc[channel_name].append(values)
                    total_spikes += int(values.size)

                    if extract_waveforms and group_raw_meta:
                        frames = np.asarray(spike_data["frameno"][mask], dtype=np.int64)[order]
                        group_waveform_requests[channel_name] = (channel_int, frames)

                    payload = dict(mapping)
                    payload.update(
                        {
                            "channel": channel_name,
                            "well": well,
                            "recording": recording,
                            "source_channel": channel_int,
                            "data_group": group_path,
                        }
                    )
                    if "x" in payload:
                        payload["x_um"] = float(payload["x"])
                        if "y" in payload:
                            payload["y_um"] = float(payload["y"])
                        channel_map.setdefault(channel_name, payload)

            if extract_waveforms and group_raw_meta and group_waveform_requests:
                group_waveforms, group_extraction = _maxwell_extract_group_waveforms(
                    group,
                    group_path,
                    group_waveform_requests,
                    sr,
                    waveform_window_ms,
                    max_waveform_bytes,
                    cancel_check=cancel_check,
                )
                for channel_name, extraction_info in group_extraction.items():
                    waveform_extraction[channel_name].append(extraction_info)
                for channel_name, channel_waveforms in group_waveforms.items():
                    if channel_waveforms is not None:
                        waveforms_acc[channel_name].append(channel_waveforms)

            group_event_times, group_event_records = _maxwell_event_times(group, sr, frame_origin, segment_offset_s)
            if group_event_times.size:
                event_times.append(group_event_times)
            event_records.extend(group_event_records)
            data_groups_meta.append(
                {
                    "path": group_path,
                    "well": well,
                    "recording": recording,
                    "sampling_rate_hz": sr,
                    "start_time": _h5_scalar(group, "start_time", None),
                    "stop_time": _h5_scalar(group, "stop_time", None),
                    "frame_origin": frame_origin,
                    "time_offset_s": segment_offset_s,
                    "duration_s": float(segment_duration_s),
                    "spike_count": int(spike_data.size) if spike_data is not None else 0,
                }
            )
            segment_offsets[well] += max(0.0, float(segment_duration_s))

    stim_times = np.concatenate(event_times) if event_times else np.array([], dtype=float)
    stim_times = np.asarray(sorted(set(float(value) for value in stim_times if np.isfinite(value))), dtype=float)
    if stim_times.size == 0:
        stim_times = _load_maxwell_stim_sidecar_times(path)
    artifact_window_s = max(0.0, float(stim_artifact_window_ms) / 1000.0)
    spike_filters: Dict[str, Dict[str, np.ndarray]] = {}
    artifacts_removed_by_channel: Dict[str, int] = {}

    spikes = {}
    for channel, chunks in spikes_acc.items():
        _raise_if_cancelled(cancel_check)
        merged_raw = np.concatenate(chunks) if chunks else np.array([], dtype=float)
        finite_mask = np.isfinite(merged_raw)
        finite = merged_raw[finite_mask]
        order = np.argsort(finite)
        merged = finite[order]
        keep = _stim_artifact_keep_mask(merged, stim_times, artifact_window_s)
        removed = int(keep.size - np.count_nonzero(keep))
        if removed:
            artifacts_removed_by_channel[channel] = removed
        spike_filters[channel] = {
            "finite_mask": finite_mask,
            "order": order,
            "keep": keep,
        }
        spikes[channel] = merged[keep]

    waveforms: Dict[str, np.ndarray] = {}
    for channel, chunks in waveforms_acc.items():
        _raise_if_cancelled(cancel_check)
        if not chunks:
            continue
        merged_waveforms = np.vstack(chunks).astype(np.float32, copy=False)
        filter_payload = spike_filters.get(channel)
        if filter_payload is not None and merged_waveforms.shape[0] == filter_payload["finite_mask"].size:
            merged_waveforms = merged_waveforms[filter_payload["finite_mask"]]
            merged_waveforms = merged_waveforms[filter_payload["order"]]
            merged_waveforms = merged_waveforms[filter_payload["keep"]]
        if np.any(np.isfinite(merged_waveforms)):
            waveforms[channel] = merged_waveforms

    duration_s = max(segment_offsets.values(), default=0.0)
    for values in spikes.values():
        if values.size:
            duration_s = max(duration_s, float(values[-1]))
    if stim_times.size:
        duration_s = max(duration_s, float(stim_times[-1]))

    sr = float(np.median(sampling_rates)) if sampling_rates else None
    waveform_units = {
        str(info.get("unit"))
        for infos in waveform_extraction.values()
        for info in infos
        if info.get("valid_count", 0) and info.get("unit")
    }
    waveform_unit = "not_loaded"
    if waveform_units:
        waveform_unit = "uV" if "uV" in waveform_units else sorted(waveform_units)[0]

    meta = {
        "source": "maxwell_h5",
        "file": str(path),
        "version": str(version),
        "wells": sorted(wells),
        "recordings": sorted(recordings),
        "well_count": int(len(wells)),
        "recording_count": int(len(recordings)),
        "channel_map": channel_map,
        "raw_data": raw_data,
        "data_groups": data_groups_meta,
        "spike_count": int(sum(values.size for values in spikes.values())),
        "raw_spike_count": int(total_spikes),
        "event_count": int(stim_times.size),
        "event_records": event_records,
        "duration_s": float(duration_s),
        "reader": "native_maxwell_h5",
        "stim_artifact_window_ms": float(stim_artifact_window_ms),
        "stim_artifact_removed_count": int(sum(artifacts_removed_by_channel.values())),
        "stim_artifact_removed_by_channel": artifacts_removed_by_channel,
        "waveform_unit": waveform_unit,
        "waveform_window_ms": [float(waveform_window_ms[0]), float(waveform_window_ms[1])],
        "waveform_extraction": dict(waveform_extraction),
        "extract_waveforms": bool(extract_waveforms),
        "waveforms_deferred": bool(not extract_waveforms),
        "waveform_channel_count": int(len(waveforms)),
    }

    return UnifiedMEAData(
        spikes=spikes,
        waveforms=waveforms,
        sr=sr,
        stim_times=stim_times,
        meta=meta,
    )


def _load_maxwell_stim_sidecar_times(path: Path) -> np.ndarray:
    candidates = [
        path.with_name("stim_times.txt"),
        path.with_name("segment_time_meta.json"),
        path.parent / "stim_times.txt",
        path.parent / "segment_time_meta.json",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            if candidate.suffix.lower() == ".txt":
                values = []
                for raw_line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        values.append(float(line))
                    except ValueError:
                        continue
                if values:
                    return np.asarray(sorted(set(float(value) for value in values if np.isfinite(value))), dtype=float)
            elif candidate.suffix.lower() == ".json":
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                values = payload.get("stim_times_sec", [])
                if isinstance(values, list) and values:
                    return np.asarray(sorted(set(float(value) for value in values if np.isfinite(value))), dtype=float)
        except Exception:
            continue
    return np.array([], dtype=float)


def _nev_type_label(file_type_id: bytes) -> str:
    text = file_type_id.decode("ascii", errors="replace").strip("\0")
    hex_text = file_type_id.hex(" ")
    return f"{text!r} (hex: {hex_text})"


def _parse_nev_basic_header(raw: bytes) -> Dict[str, Any]:
    if len(raw) < 336:
        raise ValueError("File is too small to be a Blackrock NEV file.")
    file_type_id = raw[:8]
    if file_type_id not in {b"NEURALEV", b"BREVENTS"}:
        raise ValueError(
            "Unsupported NEV file type "
            f"{_nev_type_label(file_type_id)}. Expected NEURALEV or BREVENTS header."
        )

    major, minor, flags, header_bytes, packet_bytes, ts_resolution, sample_resolution = (
        struct.unpack_from("<BBHIIII", raw, 8)
    )
    time_origin = struct.unpack_from("<8H", raw, 28)
    extended_header_count = struct.unpack_from("<I", raw, 332)[0]

    return {
        "file_type_id": file_type_id.decode("ascii", errors="replace"),
        "file_spec": f"{major}.{minor}",
        "additional_flags": flags,
        "bytes_in_headers": header_bytes,
        "bytes_in_data_packets": packet_bytes,
        "timestamp_resolution": ts_resolution,
        "sample_resolution": sample_resolution,
        "time_origin": {
            "year": time_origin[0],
            "month": time_origin[1],
            "day_of_week": time_origin[2],
            "day": time_origin[3],
            "hour": time_origin[4],
            "minute": time_origin[5],
            "second": time_origin[6],
            "millisecond": time_origin[7],
        },
        "creating_application": _cstring(raw[44:76]),
        "comment": _cstring(raw[76:332]),
        "extended_header_count": extended_header_count,
    }


def _parse_nev_extended_headers(raw: bytes, count: int) -> Dict[str, Any]:
    labels: Dict[int, str] = {}
    waveform_meta: Dict[int, Dict[str, Any]] = {}
    filters: Dict[int, Dict[str, Any]] = {}

    for index in range(count):
        start = 336 + index * 32
        chunk = raw[start : start + 32]
        if len(chunk) < 32:
            raise ValueError("NEV extended header is truncated.")

        packet_id = _cstring(chunk[:8])
        if packet_id == "NEUEVWAV":
            electrode_id = struct.unpack_from("<H", chunk, 8)[0]
            waveform_meta[electrode_id] = {
                "physical_connector": chunk[10],
                "connector_pin": chunk[11],
                "digitization_factor": struct.unpack_from("<H", chunk, 12)[0],
                "energy_threshold": struct.unpack_from("<H", chunk, 14)[0],
                "high_threshold": struct.unpack_from("<h", chunk, 16)[0],
                "low_threshold": struct.unpack_from("<h", chunk, 18)[0],
                "sorted_unit_count": chunk[20],
                "bytes_per_waveform_sample": chunk[21],
                "spike_width_samples": struct.unpack_from("<H", chunk, 22)[0],
            }
        elif packet_id == "NEUEVLBL":
            electrode_id = struct.unpack_from("<H", chunk, 8)[0]
            label = _cstring(chunk[10:26])
            if label:
                labels[electrode_id] = label
        elif packet_id == "NEUEVFLT":
            electrode_id = struct.unpack_from("<H", chunk, 8)[0]
            filters[electrode_id] = {
                "high_freq_corner": struct.unpack_from("<I", chunk, 10)[0],
                "high_freq_order": struct.unpack_from("<I", chunk, 14)[0],
                "high_filter_type": struct.unpack_from("<H", chunk, 18)[0],
                "low_freq_corner": struct.unpack_from("<I", chunk, 20)[0],
                "low_freq_order": struct.unpack_from("<I", chunk, 24)[0],
                "low_filter_type": struct.unpack_from("<H", chunk, 28)[0],
            }

    return {
        "labels": labels,
        "waveforms": waveform_meta,
        "filters": filters,
    }


def _scale_waveform(raw_waveform: np.ndarray, channel_meta: Dict[str, Any] | None) -> np.ndarray:
    if not channel_meta:
        return raw_waveform.astype(float)

    digitization_factor = channel_meta.get("digitization_factor")
    if not digitization_factor:
        return raw_waveform.astype(float)

    # NEUEVWAV stores the digitization factor in nV per ADC count. Return uV.
    return raw_waveform.astype(float) * (float(digitization_factor) / 1000.0)


def _waveform_dtype(bytes_per_sample: int) -> str:
    if bytes_per_sample == 1:
        return "i1"
    if bytes_per_sample == 2:
        return "<i2"
    if bytes_per_sample == 4:
        return "<i4"
    raise ValueError(f"Unsupported NEV waveform sample width: {bytes_per_sample} bytes")


def _extract_nev_waveform(
    raw: bytes,
    packet_offset: int,
    packet_bytes: int,
    channel_meta: Dict[str, Any] | None,
) -> np.ndarray:
    payload_offset = packet_offset + 8
    payload_bytes = packet_bytes - 8
    if payload_bytes <= 0:
        return np.asarray([], dtype=np.int16)

    bytes_per_sample = int((channel_meta or {}).get("bytes_per_waveform_sample") or 2)
    dtype = _waveform_dtype(bytes_per_sample)
    sample_count = payload_bytes // bytes_per_sample
    waveform_offset = payload_offset

    meta_samples = int((channel_meta or {}).get("spike_width_samples") or 0)
    meta_bytes = meta_samples * bytes_per_sample
    if meta_samples > 0 and meta_bytes <= payload_bytes:
        sample_count = meta_samples
        # BREVENTS/NEV 3.0 files can include extra per-packet fields before the waveform.
        waveform_offset = payload_offset + payload_bytes - meta_bytes

    return np.frombuffer(
        raw,
        dtype=dtype,
        count=sample_count,
        offset=waveform_offset,
    ).copy()


def _parse_nev_data_packet(
    raw: bytes,
    packet_offset: int,
    packet_bytes: int,
    waveform_meta: Dict[int, Dict[str, Any]],
) -> tuple[int, int, int, bool]:
    timestamp = struct.unpack_from("<I", raw, packet_offset)[0]
    packet_id = struct.unpack_from("<H", raw, packet_offset + 4)[0]
    unit = raw[packet_offset + 6] if packet_bytes > 6 else 0

    if 1 <= packet_id <= 2048:
        return timestamp, packet_id, int(unit), True

    # Some BREVENTS/NEV 3.0 files store a zero word at +4 and place the
    # electrode id in the first extra word before the waveform payload.
    if packet_id == 0 and packet_bytes >= 12:
        alternate_packet_id = struct.unpack_from("<H", raw, packet_offset + 8)[0]
        if 1 <= alternate_packet_id <= 2048 and alternate_packet_id in waveform_meta:
            alternate_unit = raw[packet_offset + 10]
            return timestamp, alternate_packet_id, int(alternate_unit), True

    return timestamp, packet_id, int(unit), False


def _parse_nev_event_record(
    raw: bytes,
    packet_offset: int,
    packet_bytes: int,
    packet_id: int,
    timestamp_s: float,
) -> Dict[str, Any]:
    code = struct.unpack_from("<H", raw, packet_offset + 8)[0] if packet_bytes >= 10 else None
    value = struct.unpack_from("<H", raw, packet_offset + 10)[0] if packet_bytes >= 12 else None
    record = {
        "time_s": float(timestamp_s),
        "packet_id": int(packet_id),
        "code": None if code is None else int(code),
        "value": None if value is None else int(value),
    }
    if packet_bytes > 16:
        comment = raw[packet_offset + 16 : packet_offset + packet_bytes].split(b"\0", 1)[0]
        try:
            text = comment.decode("utf-8", errors="replace").strip()
        except Exception:
            text = ""
        if text:
            record["comment"] = text
            parts = [part.strip() for part in text.split(",")]
            if parts and parts[0] in {"stim_request", "stim_corrected"}:
                record["stim_kind"] = parts[0]
                if len(parts) > 1:
                    record["stim_electrode"] = parts[1]
                if len(parts) > 2:
                    record["stim_request_id"] = parts[2]
                if parts[0] == "stim_corrected" and len(parts) > 3:
                    try:
                        record["stim_corrected_time_s"] = float(parts[3])
                    except ValueError:
                        pass
    return record


def _nev_stim_event_records(event_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stim_records = [event for event in event_records if event.get("code") == 0xFFFF]
    if stim_records:
        return stim_records
    return [event for event in event_records if event.get("code") != 0xFFF9]


def _nev_stim_marker_times(stim_records: List[Dict[str, Any]]) -> np.ndarray:
    corrected_by_request: Dict[str, float] = {}
    request_by_id: Dict[str, float] = {}
    fallback: List[float] = []

    for index, event in enumerate(stim_records):
        event_time = event.get("time_s")
        try:
            event_time_s = float(event_time)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(event_time_s):
            continue

        request_id = str(event.get("stim_request_id") or f"event-{index}")
        stim_kind = event.get("stim_kind")
        if stim_kind == "stim_corrected":
            corrected_time = event.get("stim_corrected_time_s")
            try:
                corrected_time_s = float(corrected_time)
            except (TypeError, ValueError):
                corrected_time_s = event_time_s
            if np.isfinite(corrected_time_s):
                corrected_by_request.setdefault(request_id, corrected_time_s)
            continue
        if stim_kind == "stim_request":
            request_by_id.setdefault(request_id, event_time_s)
            continue
        fallback.append(event_time_s)

    if corrected_by_request:
        values = corrected_by_request.values()
    elif request_by_id:
        values = request_by_id.values()
    else:
        values = fallback
    return np.asarray(sorted(set(float(value) for value in values if np.isfinite(float(value)))), dtype=float)


def _nearest_nominal_recording_duration(duration_s: float) -> float | None:
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        return None
    common = np.asarray(
        [10, 15, 20, 25, 30, 60, 90, 120, 150, 180, 240, 300, 433, 450, 600, 900, 1200],
        dtype=float,
    )
    index = int(np.argmin(np.abs(common - duration_s)))
    candidate = float(common[index])
    tolerance = max(0.25, candidate * 0.01)
    if abs(candidate - duration_s) <= tolerance:
        return candidate
    rounded = round(duration_s / 10.0) * 10.0
    if duration_s >= 30.0 and abs(rounded - duration_s) <= max(0.5, rounded * 0.002):
        return float(rounded)
    return None


def _nev_recording_timebase(
    event_records: List[Dict[str, Any]],
    first_spike_s: float | None,
    last_spike_s: float | None,
) -> Tuple[float, str, float | None]:
    sync_events = [event for event in event_records if event.get("code") == 0xFFF9]
    starts = sorted(float(event["time_s"]) for event in sync_events if event.get("value") == 0)
    stops = sorted(float(event["time_s"]) for event in sync_events if event.get("value") == 1)

    if first_spike_s is not None and stops:
        stop_after_first = next((stop for stop in stops if stop >= first_spike_s), stops[-1])
        zero_starts = [start for start in starts if abs(start) < 1e-9]
        positive_starts = [start for start in starts if start > 0.0 and start <= first_spike_s + 1e-6]
        if zero_starts and not positive_starts and first_spike_s > 5.0 and stop_after_first > first_spike_s:
            nominal_duration = _nearest_nominal_recording_duration(stop_after_first - first_spike_s)
            if nominal_duration is not None:
                inferred_start = stop_after_first - nominal_duration
                if inferred_start > 0.0 and inferred_start <= first_spike_s + 0.25:
                    return min(inferred_start, first_spike_s), "inferred_recording_start_from_stop_event", stop_after_first

    if starts:
        if first_spike_s is None:
            start = starts[0]
        else:
            candidates = [start for start in starts if start <= first_spike_s + 1e-6]
            start = candidates[-1] if candidates else starts[0]
        stop = next((value for value in stops if value >= start), None)
        return start, "recording_start_event", stop

    if event_records:
        start = min(float(event["time_s"]) for event in event_records)
        stop = max(float(event["time_s"]) for event in event_records)
        return start, "first_event", stop

    if first_spike_s is not None:
        return first_spike_s, "first_neural_spike", last_spike_s

    return 0.0, "none", None


def read_blackrock_nev(file_path: str | Path, cancel_check=None) -> UnifiedMEAData:
    """Read a Blackrock ``.nev`` file into :class:`UnifiedMEAData`.

    The parser supports neural event packets with optional waveforms. Digital
    event/comment packet timestamps are captured as ``stim_times`` when present,
    but this project currently does not interpret their payload fields.
    """

    path = Path(file_path)
    _raise_if_cancelled(cancel_check)
    raw = path.read_bytes()
    _raise_if_cancelled(cancel_check)
    basic = _parse_nev_basic_header(raw[:336])

    header_bytes = int(basic["bytes_in_headers"])
    packet_bytes = int(basic["bytes_in_data_packets"])
    ts_resolution = float(basic["timestamp_resolution"])
    sample_resolution = float(basic["sample_resolution"])

    if packet_bytes < 8:
        raise ValueError(f"Invalid NEV packet size: {packet_bytes}")
    if header_bytes > len(raw):
        raise ValueError("NEV header length exceeds file size.")

    ext_raw = raw[:header_bytes]
    extended = _parse_nev_extended_headers(ext_raw, int(basic["extended_header_count"]))
    labels: Dict[int, str] = extended["labels"]
    waveform_meta: Dict[int, Dict[str, Any]] = extended["waveforms"]

    payload = raw[header_bytes:]
    if len(payload) % packet_bytes != 0:
        raise ValueError("NEV data packet section is not aligned to packet size.")

    grouped_times: Dict[str, List[float]] = defaultdict(list)
    grouped_wfs: Dict[str, List[np.ndarray]] = defaultdict(list)
    grouped_units: Dict[str, List[int]] = defaultdict(list)
    event_times: List[float] = []
    event_records: List[Dict[str, Any]] = []

    packet_count = len(payload) // packet_bytes

    for packet_index in range(packet_count):
        if packet_index % 8192 == 0:
            _raise_if_cancelled(cancel_check)
        offset = header_bytes + packet_index * packet_bytes
        timestamp, packet_id, unit, is_neural = _parse_nev_data_packet(
            raw,
            offset,
            packet_bytes,
            waveform_meta,
        )
        timestamp_s = float(timestamp) / ts_resolution

        if is_neural:
            channel_name = labels.get(packet_id, f"chan{packet_id}")
            grouped_times[channel_name].append(timestamp_s)
            grouped_units[channel_name].append(int(unit))

            channel_meta = waveform_meta.get(packet_id)
            waveform = _extract_nev_waveform(raw, offset, packet_bytes, channel_meta)
            grouped_wfs[channel_name].append(_scale_waveform(waveform, channel_meta))
        elif packet_id == 0 or packet_id >= 0xFF00:
            event_times.append(timestamp_s)
            event_records.append(_parse_nev_event_record(raw, offset, packet_bytes, packet_id, timestamp_s))

    spikes: Dict[str, np.ndarray] = {}
    waveforms: Dict[str, np.ndarray] = {}
    sorting: Dict[str, Dict[str, Any]] = {}

    for channel_name, timestamps in grouped_times.items():
        _raise_if_cancelled(cancel_check)
        times = np.asarray(timestamps, dtype=float)
        order = np.argsort(times)
        spikes[channel_name] = times[order]

        channel_waveforms = np.vstack(grouped_wfs[channel_name])
        waveforms[channel_name] = channel_waveforms[order]
        sorting[channel_name] = {
            "labels": np.asarray(grouped_units[channel_name], dtype=np.int32)[order],
        }

    first_spike_s = min((float(values[0]) for values in spikes.values() if values.size), default=None)
    last_spike_s = max((float(values[-1]) for values in spikes.values() if values.size), default=None)
    timestamp_offset_s, timestamp_offset_source, recording_stop_s = _nev_recording_timebase(
        event_records,
        first_spike_s,
        last_spike_s,
    )
    if timestamp_offset_s:
        for channel_name, values in list(spikes.items()):
            spikes[channel_name] = values - timestamp_offset_s
        event_times = [float(value) - timestamp_offset_s for value in event_times]
        for event in event_records:
            event["time_s"] = float(event["time_s"]) - timestamp_offset_s
            if "stim_corrected_time_s" in event:
                event["stim_corrected_time_s"] = float(event["stim_corrected_time_s"]) - timestamp_offset_s

    stim_event_records = _nev_stim_event_records(event_records)
    stim_times = _nev_stim_marker_times(stim_event_records)

    duration_s = 0.0
    for values in spikes.values():
        if values.size:
            duration_s = max(duration_s, float(values[-1]))
    if stim_times.size:
        duration_s = max(duration_s, float(stim_times[-1]))
    if recording_stop_s is not None:
        duration_s = max(duration_s, float(recording_stop_s) - float(timestamp_offset_s))

    meta = {
        "source": "blackrock_nev",
        "file": str(path),
        "basic_header": basic,
        "electrode_labels": labels,
        "waveform_headers": waveform_meta,
        "filter_headers": extended["filters"],
        "packet_count": packet_count,
        "neural_packet_count": int(sum(len(v) for v in spikes.values())),
        "event_packet_count": int(len(event_times)),
        "event_records": event_records,
        "stim_event_count": int(stim_times.size),
        "stim_event_records": stim_event_records,
        "timestamp_offset_s": float(timestamp_offset_s),
        "timestamp_offset_source": timestamp_offset_source,
        "recording_stop_s": None if recording_stop_s is None else float(recording_stop_s) - float(timestamp_offset_s),
        "duration_s": float(duration_s),
    }

    return UnifiedMEAData(
        spikes=spikes,
        waveforms=waveforms,
        sr=sample_resolution,
        stim_times=stim_times,
        meta=meta,
        sorting=sorting,
    )


def read_unified_npz(path: str | Path) -> UnifiedMEAData:
    """Read a unified NPZ file produced by downstream curation workflows."""

    with np.load(path, allow_pickle=True) as npz:
        stim_times = np.asarray(npz.get("stim_times", np.array([], dtype=float)), dtype=float)

        sr = None
        if "sr" in npz:
            try:
                sr = float(npz["sr"])
            except Exception:
                sr = None

        bad_intervals = np.asarray(
            npz.get("bad_intervals", np.zeros((0, 2), dtype=float)),
            dtype=float,
        )
        if bad_intervals.size == 0:
            bad_intervals = np.zeros((0, 2), dtype=float)
        if bad_intervals.ndim == 1:
            bad_intervals = (
                bad_intervals.reshape(-1, 2)
                if bad_intervals.size % 2 == 0
                else np.zeros((0, 2), dtype=float)
            )

        meta: Dict[str, Any] = {}
        if "meta_json" in npz:
            raw_meta = npz["meta_json"]
            if isinstance(raw_meta, np.ndarray):
                try:
                    raw_meta = raw_meta.item()
                except Exception:
                    raw_meta = raw_meta.flat[0]
            if isinstance(raw_meta, bytes):
                raw_meta = raw_meta.decode("utf-8", "ignore")
            if isinstance(raw_meta, (str, np.str_)):
                try:
                    meta = json.loads(str(raw_meta))
                except Exception:
                    meta = {"meta_json_raw": str(raw_meta)}

        spikes: Dict[str, np.ndarray] = {}
        waveforms: Dict[str, np.ndarray] = {}
        sorting: Dict[str, Dict[str, Any]] = {}
        if "sorting_meta_json" in npz:
            raw_sorting = npz["sorting_meta_json"]
            if isinstance(raw_sorting, np.ndarray):
                try:
                    raw_sorting = raw_sorting.item()
                except Exception:
                    raw_sorting = raw_sorting.flat[0]
            if isinstance(raw_sorting, bytes):
                raw_sorting = raw_sorting.decode("utf-8", "ignore")
            if isinstance(raw_sorting, (str, np.str_)):
                try:
                    loaded_sorting_meta = json.loads(str(raw_sorting))
                    if isinstance(loaded_sorting_meta, dict):
                        sorting.update(loaded_sorting_meta)
                except Exception:
                    pass

        for key in npz.files:
            if key.startswith("spikes_"):
                channel = key[len("spikes_") :]
                spikes[channel] = np.asarray(npz[key], dtype=float)
            elif key.startswith("waveforms_"):
                channel = key[len("waveforms_") :]
                waveform_array = np.asarray(npz[key])
                if waveform_array.ndim == 1:
                    waveform_array = waveform_array.reshape(-1, 1)
                waveforms[channel] = waveform_array

        for key in npz.files:
            if key.startswith("labels_"):
                channel = key[len("labels_") :]
                sorting.setdefault(channel, {})["labels"] = np.asarray(npz[key], dtype=np.int32)
            elif key.startswith("waveform_cluster_labels_"):
                channel = key[len("waveform_cluster_labels_") :]
                sorting.setdefault(channel, {})["waveform_cluster_labels"] = np.asarray(npz[key], dtype=np.int32)
            elif key.startswith("embed_"):
                channel = key[len("embed_") :]
                sorting.setdefault(channel, {})["embedding"] = np.asarray(npz[key], dtype=np.float32)

    return UnifiedMEAData(
        spikes=spikes,
        waveforms=waveforms,
        sr=sr,
        stim_times=stim_times,
        bad_intervals=bad_intervals,
        meta=meta,
        sorting={channel: obj for channel, obj in sorting.items() if obj},
    )


def save_unified_npz(data: UnifiedMEAData, path: str | Path, *, include_waveforms: bool = True) -> Path:
    """Save unified MEA data and sorting results to a compressed NPZ file.

    The file preserves per-channel spikes, waveforms and sorting labels. It
    also stores unit-level views as ``unit_spikes_{channel}_{unit}`` and
    ``unit_waveforms_{channel}_{unit}``, including noise unit ``-1`` when
    present. Use :func:`save_spike_train_npz` for a lightweight file that omits
    waveform arrays.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, Any] = {
        "stim_times": np.asarray(data.stim_times, dtype=float),
        "bad_intervals": np.asarray(data.bad_intervals, dtype=float).reshape(-1, 2)
        if np.asarray(data.bad_intervals).size
        else np.zeros((0, 2), dtype=float),
        "meta_json": json.dumps(_json_safe(data.meta), ensure_ascii=False),
        "sorting_meta_json": json.dumps(
            _json_safe({key: value for key, value in data.sorting.items() if str(key).startswith("_")}),
            ensure_ascii=False,
        ),
        "save_options_json": json.dumps(
            {
                "format": "unified_mea_npz" if include_waveforms else "unified_spike_train_npz",
                "include_waveforms": bool(include_waveforms),
            },
            ensure_ascii=False,
        ),
    }
    if data.sr is not None:
        arrays["sr"] = np.asarray(float(data.sr), dtype=float)

    for channel in data.channels():
        safe_channel = _safe_key(channel)
        spikes = np.asarray(data.spikes[channel], dtype=float)
        arrays[f"spikes_{safe_channel}"] = spikes
        waveforms = None
        if include_waveforms:
            waveforms = np.asarray(data.waveforms.get(channel, np.zeros((0, 0), dtype=float)))
            arrays[f"waveforms_{safe_channel}"] = waveforms

        sorting = data.sorting.get(channel, {}) if isinstance(data.sorting, dict) else {}
        labels = _labels_for_channel(sorting, spikes.size)
        if labels is not None:
            arrays[f"labels_{safe_channel}"] = labels
            arrays[f"waveform_cluster_labels_{safe_channel}"] = labels
            for unit in sorted(int(value) for value in np.unique(labels)):
                mask = labels == unit
                unit_key = _safe_unit_key(unit)
                arrays[f"unit_spikes_{safe_channel}_{unit_key}"] = spikes[mask]
                if include_waveforms and waveforms is not None and waveforms.ndim == 2 and waveforms.shape[0] == spikes.size:
                    arrays[f"unit_waveforms_{safe_channel}_{unit_key}"] = waveforms[mask]
        embedding = sorting.get("embedding")
        if embedding is not None:
            arrays[f"embed_{safe_channel}"] = np.asarray(embedding, dtype=np.float32)

    np.savez_compressed(output, **arrays)
    return output


def save_spike_train_npz(data: UnifiedMEAData, path: str | Path) -> Path:
    """Save metadata, sorting results and spike trains without waveforms."""

    return save_unified_npz(data, path, include_waveforms=False)


def _labels_for_channel(sorting: Dict[str, Any], expected_size: int) -> Optional[np.ndarray]:
    labels = sorting.get("waveform_cluster_labels")
    if labels is None:
        labels = sorting.get("labels")
    if labels is None:
        return None
    labels = np.asarray(labels, dtype=np.int32)
    if labels.size != expected_size:
        return None
    return labels


def _safe_key(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))


def _safe_unit_key(unit: int) -> str:
    return f"noise{abs(unit)}" if unit < 0 else f"unit{unit}"


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def load_axion_cleaned_json(path: str | Path) -> List[Dict[str, Any]]:
    """Load an Axion curation JSON export into list-of-record dictionaries."""

    with Path(path).open("r", encoding="utf-8") as file:
        raw_list = json.load(file)

    output: List[Dict[str, Any]] = []
    for record in raw_list:
        data = record.get("data", {})
        spike_times = np.asarray(data.get("spike_times", []), dtype=float)
        spike_waveform = np.asarray(data.get("spike_waveform", []), dtype=float)
        if spike_waveform.ndim == 1:
            spike_waveform = spike_waveform[None, :]

        output.append(
            {
                "well": record.get("well", ""),
                "electrode": record.get("electrode", ""),
                "data": {
                    "spike_times": spike_times,
                    "spike_waveform": spike_waveform,
                },
            }
        )

    return output
