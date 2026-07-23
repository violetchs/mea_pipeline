from __future__ import annotations

import os
import sys
import time
import hashlib
import json
import random
from pathlib import Path
from typing import Any


_MX = None


def _mx() -> Any:
    global _MX
    if _MX is not None:
        return _MX
    api_path = os.environ.get("MAXLAB_PYTHON_PATH")
    if api_path and api_path not in sys.path:
        sys.path.insert(0, api_path)
    try:
        import maxlab as mx  # type: ignore
    except ImportError as exc:
        raise RuntimeError("maxlab is required for real runs. Install MaxWell API or set MAXLAB_PYTHON_PATH.") from exc
    _MX = mx
    return mx


def initialize_maxlab() -> None:
    mx = _mx()
    mx.initialize()
    response = mx.send(mx.Core().enable_stimulation_power(True))
    if response != "Ok":
        raise RuntimeError(f"MaxLab initialization failed: {response}")
    time.sleep(getattr(mx.Timing, "waitInit", 0))


def _event(properties: str, event_id: int) -> Any:
    mx = _mx()
    if len(properties.split()) % 2 != 0:
        raise ValueError("mx.Event properties must be key-value pairs")
    return mx.Event(0, 1, event_id, properties)


def _half_bits(amplitude_mv: float) -> int:
    mx = _mx()
    half_bits = int(round(abs(float(amplitude_mv)) / float(mx.query_DAC_lsb_mV())))
    return max(1, min(511, half_bits))


def _samples_us(us: float) -> int:
    return max(1, int(round(float(us) / 50.0)))


def _samples_ms(ms: float) -> int:
    return _samples_us(float(ms) * 1000.0)


def build_stim_sequence(protocol: dict[str, Any], electrode_group_name: str) -> Any:
    mx = _mx()
    seq = mx.Sequence(initial_delay=100, persistent=False)
    name = str(protocol.get("name", "stim"))
    width_us = float(protocol.get("pulse_width_us", 300.0))
    ipi_us = float(protocol.get("inter_phase_interval_us", 0.0))
    channel = int(protocol.get("channel", 0))
    event_id = 1

    def pulse(amplitude_mv: float, event_index: int, dac_channel: int = channel, duration_us: float = width_us) -> float:
        bits = _half_bits(amplitude_mv)
        seq.append(_event(f"type stim name {name} amplitude_mv {amplitude_mv}", event_index))
        seq.append(mx.DAC(dac_channel, 512 - bits))
        seq.append(mx.DelaySamples(_samples_us(duration_us)))
        if ipi_us > 0:
            seq.append(mx.DelaySamples(_samples_us(ipi_us)))
        seq.append(mx.DAC(dac_channel, 512 + bits))
        seq.append(mx.DelaySamples(_samples_us(duration_us)))
        seq.append(mx.DAC(dac_channel, 512))
        return (float(duration_us) * 2.0 + max(0.0, ipi_us)) / 1000.0

    ptype = protocol.get("type")
    if ptype in {"single_pulse", "individual_burst", "sequence_with_burst", "sequence_with_poisson_burst"}:
        current_ms = 0.0
        for stim_ms in _scheduled_stim_times_ms(protocol):
            if stim_ms > current_ms:
                seq.append(mx.DelaySamples(_samples_ms(stim_ms - current_ms)))
            current_ms = max(current_ms, stim_ms) + pulse(float(protocol.get("amplitude_mv", 150.0)), event_id)
            event_id += 1
    elif ptype == "custom_sequence":
        current_ms = 0.0
        for point in sorted(protocol.get("custom_points", []), key=lambda item: float(item["time_ms"])):
            point_ms = float(point["time_ms"])
            if point_ms > current_ms:
                seq.append(mx.DelaySamples(_samples_ms(point_ms - current_ms)))
            pulse_width_ms = pulse(float(point["amplitude_mv"]), event_id, int(point.get("channel", channel)), float(point.get("duration_us", width_us)))
            event_id += 1
            current_ms = max(current_ms, point_ms) + pulse_width_ms
    else:
        raise ValueError(f"Unsupported protocol type: {ptype}")
    return seq


def build_poisson_random_sequence(
    protocol: dict[str, Any],
    plan_rows: list[dict[str, Any]],
    stim_unit_by_electrode: dict[int, int],
) -> Any:
    mx = _mx()
    seq = mx.Sequence(initial_delay=100, persistent=False)
    name = str(protocol.get("name", "poisson_random"))
    width_us_default = float(protocol.get("pulse_width_us", 300.0))
    ipi_us = float(protocol.get("inter_phase_interval_us", 0.0) or 0.0)
    dac_channel = 0
    current_ms = 0.0
    event_id = 1
    connected_stim_unit: int | None = None

    def pulse(amplitude_mv: float, event_index: int, duration_us: float) -> float:
        bits = _half_bits(amplitude_mv)
        seq.append(_event(f"type stim name {name} amplitude_mv {amplitude_mv}", event_index))
        seq.append(mx.DAC(dac_channel, 512 - bits))
        seq.append(mx.DelaySamples(_samples_us(duration_us)))
        if ipi_us > 0:
            seq.append(mx.DelaySamples(_samples_us(ipi_us)))
        seq.append(mx.DAC(dac_channel, 512 + bits))
        seq.append(mx.DelaySamples(_samples_us(duration_us)))
        seq.append(mx.DAC(dac_channel, 512))
        return (float(duration_us) * 2.0 + max(0.0, ipi_us)) / 1000.0

    for row in sorted(plan_rows, key=lambda item: (float(item["time_sec"]), int(item["electrode"]))):
        electrode = int(row["electrode"])
        if electrode not in stim_unit_by_electrode:
            raise RuntimeError(f"No stimulation unit configured for poisson electrode {electrode}")
        stim_unit = int(stim_unit_by_electrode[electrode])
        point_ms = float(row["time_sec"]) * 1000.0
        if point_ms > current_ms:
            seq.append(mx.DelaySamples(_samples_ms(point_ms - current_ms)))
        if connected_stim_unit != stim_unit:
            if connected_stim_unit is not None:
                seq.append(mx.StimulationUnit(connected_stim_unit).connect(False))
            seq.append(mx.StimulationUnit(stim_unit).connect(True))
            connected_stim_unit = stim_unit
        duration_ms = pulse(
            float(row.get("amplitude_mv", protocol.get("amplitude_mv", 150.0))),
            event_id,
            float(row.get("pulse_width_us", width_us_default)),
        )
        event_id += 1
        current_ms = max(current_ms, point_ms) + duration_ms
    if connected_stim_unit is not None:
        seq.append(mx.StimulationUnit(connected_stim_unit).connect(False))
    return seq


def resolve_experiment_array(
    cfg_path: Path,
    electrode_group: dict[str, Any],
    system_config: dict[str, Any],
    source_by_electrode: dict[int, int] | None = None,
    source_by_stim_unit: dict[int, int] | None = None,
    *,
    allow_missing_electrodes: bool = False,
    probe_only: bool = False,
    return_stim_units: bool = False,
    initial_connect: bool = True,
) -> Any:
    mx = _mx()
    electrodes = [int(item) for item in electrode_group.get("electrodes", [])]
    if not electrodes:
        raise ValueError("No electrodes configured")
    array = mx.Array("stimulation")
    array.reset()
    array.clear_selected_electrodes()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"cfg_path does not exist: {cfg_path}")
    array.load_config(str(cfg_path))
    stim_units = []
    stim_unit_sources: dict[int, int] = {}
    stim_unit_by_electrode: dict[int, int] = {}
    connected_electrodes: list[int] = []
    skipped_electrodes: list[int] = []
    for electrode in electrodes:
        try:
            array.connect_electrode_to_stimulation(electrode)
            stim_unit = array.query_stimulation_at_electrode(electrode)
            if len(stim_unit) == 0:
                raise RuntimeError(f"No stimulation unit can connect to electrode {electrode}")
        except Exception:
            if allow_missing_electrodes:
                skipped_electrodes.append(electrode)
                continue
            raise
        stim_unit_int = int(stim_unit)
        stim_units.append(stim_unit_int)
        connected_electrodes.append(electrode)
        stim_unit_by_electrode[electrode] = stim_unit_int
        if source_by_stim_unit is not None and stim_unit_int in source_by_stim_unit:
            source_id = int(source_by_stim_unit[stim_unit_int])
        else:
            source_id = 0 if source_by_electrode is None else int(source_by_electrode.get(electrode, 0))
        if stim_unit_int in stim_unit_sources and stim_unit_sources[stim_unit_int] != source_id:
            raise RuntimeError(f"Stimulation unit {stim_unit_int} mapped to conflicting DAC sources")
        stim_unit_sources[stim_unit_int] = source_id
    if not connected_electrodes and allow_missing_electrodes and probe_only:
        if return_stim_units:
            return array, connected_electrodes, skipped_electrodes, stim_unit_by_electrode
        return array, connected_electrodes, skipped_electrodes
    if not connected_electrodes:
        raise RuntimeError("No stimulation unit can connect to any configured electrode")
    if probe_only:
        if return_stim_units:
            return array, connected_electrodes, skipped_electrodes, stim_unit_by_electrode
        return array, connected_electrodes, skipped_electrodes
    mx.activate([0])
    for stim_unit in sorted(set(stim_units)):
        mx.send(
            mx.StimulationUnit(stim_unit)
            .power_up(True)
            .connect(bool(initial_connect))
            .set_voltage_mode()
            .dac_source(stim_unit_sources.get(stim_unit, 0))
        )
    array.download([0])
    time.sleep(getattr(mx.Timing, "waitAfterDownload", 0))
    mx.offset()
    if return_stim_units:
        return array, connected_electrodes, skipped_electrodes, stim_unit_by_electrode
    return array, connected_electrodes, skipped_electrodes


def probe_stimulation_electrodes(
    cfg_path: Path,
    electrode_group: dict[str, Any],
    system_config: dict[str, Any],
) -> tuple[list[int], list[int], dict[int, int]]:
    _array, connected, skipped, stim_units = resolve_experiment_array(
        cfg_path,
        electrode_group,
        system_config,
        allow_missing_electrodes=True,
        probe_only=True,
        return_stim_units=True,
    )
    return connected, skipped, stim_units


def configure_experiment_array(
    cfg_path: Path,
    electrode_group: dict[str, Any],
    system_config: dict[str, Any],
    source_by_electrode: dict[int, int] | None = None,
    source_by_stim_unit: dict[int, int] | None = None,
) -> Any:
    array, _connected_electrodes, _skipped_electrodes = resolve_experiment_array(
        cfg_path,
        electrode_group,
        system_config,
        source_by_electrode=source_by_electrode,
        source_by_stim_unit=source_by_stim_unit,
    )
    return array


def configure_poisson_experiment_array(
    cfg_path: Path,
    electrode_group: dict[str, Any],
    system_config: dict[str, Any],
) -> tuple[Any, dict[int, int]]:
    array, _connected_electrodes, _skipped_electrodes, stim_unit_by_electrode = resolve_experiment_array(
        cfg_path,
        electrode_group,
        system_config,
        source_by_electrode={},
        source_by_stim_unit={},
        return_stim_units=True,
        initial_connect=False,
    )
    return array, stim_unit_by_electrode


def create_experiment_saving(run_dir: Path, file_name: str) -> Any:
    mx = _mx()
    saving = mx.Saving()
    saving.open_directory(str(run_dir))
    saving.start_file(file_name)
    saving.group_define(0, "all_channels", list(range(1024)))
    saving.start_recording([0])
    return saving


def get_stim_times_for_protocol(protocol: dict[str, Any], duration_s: int) -> list[float]:
    times_ms = _scheduled_stim_times_ms(protocol)
    return sorted(round(ms / 1000.0, 6) for ms in times_ms if 0 <= ms / 1000.0 <= float(duration_s))


def _scheduled_stim_times_ms(protocol: dict[str, Any]) -> list[float]:
    ptype = protocol.get("type")
    start_ms = float(protocol.get("start_ms", 0.0))
    times_ms = []
    if ptype == "single_pulse":
        times_ms.append(start_ms)
    elif ptype == "individual_burst":
        if _randomize_burst_pulse_intervals(protocol):
            rng = random.Random(_burst_pulse_interval_seed(protocol))
            for burst_start in _burst_starts_ms(protocol, poisson=False):
                for offset_ms in _burst_pulse_offsets_ms(protocol, rng):
                    times_ms.append(burst_start + offset_ms)
        else:
            interval = _pulse_interval_ms(protocol)
            times_ms.extend(start_ms + i * interval for i in range(int(protocol.get("pulses_per_burst", 5))))
    elif ptype in {"sequence_with_burst", "sequence_with_poisson_burst"}:
        interval = _pulse_interval_ms(protocol)
        burst_starts = _burst_starts_ms(protocol, poisson=(ptype == "sequence_with_poisson_burst"))
        if ptype == "sequence_with_burst" and _randomize_burst_pulse_intervals(protocol):
            rng = random.Random(_burst_pulse_interval_seed(protocol))
            for burst_start in burst_starts:
                for offset_ms in _burst_pulse_offsets_ms(protocol, rng):
                    times_ms.append(burst_start + offset_ms)
            return sorted(ms for ms in times_ms if ms >= 0.0)
        for burst_start in burst_starts:
            for pulse_index in range(int(protocol.get("pulses_per_burst", 5))):
                times_ms.append(burst_start + pulse_index * interval)
    elif ptype == "custom_sequence":
        times_ms.extend(float(point["time_ms"]) for point in protocol.get("custom_points", []))
    else:
        raise ValueError(f"Unsupported protocol type: {ptype}")
    return sorted(ms for ms in times_ms if ms >= 0.0)


def _pulse_interval_ms(protocol: dict[str, Any]) -> float:
    return 1000.0 / max(float(protocol.get("pulse_frequency_hz", 20.0)), 0.001)


def _randomize_burst_pulse_intervals(protocol: dict[str, Any]) -> bool:
    value = protocol.get("randomize_burst_pulse_intervals", False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _burst_pulse_interval_seed(protocol: dict[str, Any]) -> int:
    seed_payload = json.dumps(
        {
            "name": protocol.get("name", ""),
            "type": protocol.get("type", ""),
            "start_ms": protocol.get("start_ms", 0.0),
            "burst_count": protocol.get("burst_count", 3),
            "burst_frequency_hz": protocol.get("burst_frequency_hz", 5.0),
            "pulses_per_burst": protocol.get("pulses_per_burst", 5),
            "pulse_frequency_hz": protocol.get("pulse_frequency_hz", 20.0),
            "randomize_burst_pulse_intervals": protocol.get("randomize_burst_pulse_intervals", False),
            "burst_pulse_interval_min_ms": protocol.get("burst_pulse_interval_min_ms", 10.0),
            "burst_pulse_interval_max_ms": protocol.get("burst_pulse_interval_max_ms", 100.0),
            "random_seed": protocol.get("random_seed", 42),
            "amplitude_mv": protocol.get("amplitude_mv", 150.0),
            "pulse_width_us": protocol.get("pulse_width_us", 300.0),
        },
        sort_keys=True,
    )
    return int(hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()[:16], 16)


def _burst_pulse_offsets_ms(protocol: dict[str, Any], rng=None) -> list[float]:
    count = max(1, int(protocol.get("pulses_per_burst", 5)))
    if not _randomize_burst_pulse_intervals(protocol):
        interval = _pulse_interval_ms(protocol)
        return [index * interval for index in range(count)]
    min_ms = max(0.0, float(protocol.get("burst_pulse_interval_min_ms", 10.0)))
    max_ms = max(min_ms, float(protocol.get("burst_pulse_interval_max_ms", 100.0)))
    if rng is None:
        rng = random.Random(_burst_pulse_interval_seed(protocol))
    offsets = [0.0]
    current_ms = 0.0
    for _index in range(1, count):
        current_ms += rng.uniform(min_ms, max_ms)
        offsets.append(current_ms)
    return offsets


def _burst_interval_ms(protocol: dict[str, Any]) -> float:
    return 1000.0 / max(float(protocol.get("burst_frequency_hz", 5.0)), 0.001)


def _burst_starts_ms(protocol: dict[str, Any], *, poisson: bool) -> list[float]:
    burst_count = max(0, int(protocol.get("burst_count", 3)))
    start_ms = float(protocol.get("start_ms", 0.0))
    if burst_count <= 0:
        return []
    starts = [start_ms]
    if burst_count == 1:
        return starts
    if poisson:
        seed_payload = json.dumps(
            {
                "name": protocol.get("name", ""),
                "type": protocol.get("type", ""),
                "start_ms": start_ms,
                "burst_count": burst_count,
                "burst_frequency_hz": float(protocol.get("burst_frequency_hz", 5.0)),
                "pulses_per_burst": int(protocol.get("pulses_per_burst", 5)),
                "pulse_frequency_hz": float(protocol.get("pulse_frequency_hz", 20.0)),
                "amplitude_mv": float(protocol.get("amplitude_mv", 150.0)),
                "pulse_width_us": float(protocol.get("pulse_width_us", 300.0)),
            },
            sort_keys=True,
        )
        seed = int(hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        mean_s = _burst_interval_ms(protocol) / 1000.0
        current_ms = start_ms
        for _index in range(1, burst_count):
            current_ms += rng.expovariate(1.0 / mean_s) * 1000.0
            starts.append(current_ms)
        return starts
    burst_interval = _burst_interval_ms(protocol)
    return [start_ms + burst_index * burst_interval for burst_index in range(burst_count)]
