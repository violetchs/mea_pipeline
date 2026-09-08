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
MAX_STIMULATION_UNITS_PER_ROUTE = 32


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


def _hardware_dac_config(system_config: dict[str, Any] | None = None) -> tuple[list[int], int, bool]:
    config = {}
    if isinstance(system_config, dict):
        maxwell_config = system_config.get("maxwell", {})
        if isinstance(maxwell_config, dict):
            config = maxwell_config.get("hardware_dac", {}) or {}
    signal_dacs = []
    for value in config.get("signal_dacs", [0, 1]):
        try:
            dac = int(value)
        except (TypeError, ValueError):
            continue
        if dac >= 0 and dac not in signal_dacs:
            signal_dacs.append(dac)
    if not signal_dacs:
        signal_dacs = [0, 1]
    try:
        neutral_dac = int(config.get("neutral_dac", 2))
    except (TypeError, ValueError):
        neutral_dac = 2
    if neutral_dac in signal_dacs:
        raise ValueError("neutral_dac must not be one of signal_dacs")
    return signal_dacs, neutral_dac, bool(config.get("sync_dual_dac", True))


def _default_stim_unit_dac_sources(
    stim_unit_by_electrode: dict[int, int],
    system_config: dict[str, Any] | None = None,
) -> dict[int, int]:
    signal_dacs, _neutral_dac, _sync_dual_dac = _hardware_dac_config(system_config)
    units = sorted({int(value) for value in stim_unit_by_electrode.values()})
    return {
        unit: int(signal_dacs[index % len(signal_dacs)])
        for index, unit in enumerate(units)
    }


def _append_dac_codes(
    seq: Any,
    sources: list[int] | set[int] | tuple[int, ...],
    code: int,
    *,
    sync_dual_dac: bool,
) -> None:
    mx = _mx()
    wanted = sorted({int(value) for value in sources if int(value) >= 0})
    if not wanted:
        return
    code_int = int(code)
    if sync_dual_dac and {0, 1}.issubset(wanted):
        try:
            seq.append(mx.chip.DAC(dac_no=0, dac_code=code_int, dac_code_second=code_int))
        except (AttributeError, TypeError):
            try:
                seq.append(mx.DAC(0, code_int, code_int))
            except TypeError:
                seq.append(mx.DAC(0, code_int))
                seq.append(mx.DAC(1, code_int))
        for source in wanted:
            if source not in {0, 1}:
                seq.append(mx.DAC(source, code_int))
        return
    for source in wanted:
        seq.append(mx.DAC(source, code_int))


def _active_dac_sources(
    row_electrodes: list[int],
    stim_unit_by_electrode: dict[int, int],
    stim_unit_to_dac: dict[int, int],
) -> list[int]:
    return sorted(
        {
            int(stim_unit_to_dac[int(stim_unit_by_electrode[electrode])])
            for electrode in row_electrodes
            if electrode in stim_unit_by_electrode
            and int(stim_unit_by_electrode[electrode]) in stim_unit_to_dac
        }
    )


def _event_level_switch_enabled(protocol: dict[str, Any]) -> bool:
    switch_cfg = protocol.get("site_switch", {}) if isinstance(protocol, dict) else {}
    if isinstance(switch_cfg, dict) and bool(switch_cfg.get("enabled", False)):
        return True
    return str(protocol.get("type", "")) == "electrode_pool_sequence"


def _event_unit_dac_sources(
    row_electrodes: list[int],
    stim_unit_by_electrode: dict[int, int],
    signal_dacs: list[int],
) -> dict[int, int]:
    sources = [int(value) for value in signal_dacs if int(value) >= 0]
    if not sources:
        sources = [0, 1]
    unit_sources: dict[int, int] = {}
    for electrode in sorted({int(value) for value in row_electrodes}):
        if electrode not in stim_unit_by_electrode:
            continue
        stim_unit = int(stim_unit_by_electrode[electrode])
        if stim_unit not in unit_sources:
            unit_sources[stim_unit] = sources[len(unit_sources) % len(sources)]
    return unit_sources


def _route_signature(
    target_stim_units: list[int],
    event_unit_to_dac: dict[int, int] | None = None,
    stim_unit_to_dac: dict[int, int] | None = None,
    *,
    event_level_switch: bool = False,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    units = tuple(sorted({int(value) for value in target_stim_units}))
    if event_level_switch:
        dac_items = tuple(sorted((int(unit), int(source)) for unit, source in (event_unit_to_dac or {}).items()))
    else:
        dac_items = tuple(
            sorted(
                (int(unit), int(source))
                for unit, source in (stim_unit_to_dac or {}).items()
                if int(unit) in units
            )
        )
    return units, dac_items


def build_stim_sequence(
    protocol: dict[str, Any],
    electrode_group_name: str,
    *,
    stim_unit_by_electrode: dict[int, int] | None = None,
    stim_unit_to_dac: dict[int, int] | None = None,
    electrode_group_electrodes: list[int] | None = None,
    system_config: dict[str, Any] | None = None,
) -> Any:
    mx = _mx()
    seq = mx.Sequence(initial_delay=100, persistent=False)
    name = str(protocol.get("name", "stim"))
    width_us = float(protocol.get("pulse_width_us", 300.0))
    ipi_us = float(protocol.get("inter_phase_interval_us", 0.0))
    signal_dacs, neutral_dac, sync_dual_dac = _hardware_dac_config(system_config)
    unit_map = {int(key): int(value) for key, value in (stim_unit_by_electrode or {}).items()}
    unit_to_dac = {int(key): int(value) for key, value in (stim_unit_to_dac or {}).items()}
    if unit_map and not unit_to_dac:
        unit_to_dac = _default_stim_unit_dac_sources(unit_map, system_config)
    target_electrodes = [int(value) for value in (electrode_group_electrodes or [])]
    active_dacs = _active_dac_sources(target_electrodes, unit_map, unit_to_dac) or list(signal_dacs)
    hold_dacs = sorted(set(signal_dacs) | {int(neutral_dac)})
    event_id = 1

    def pulse(
        amplitude_mv: float,
        event_index: int,
        duration_us: float = width_us,
        pulse_dacs: list[int] | None = None,
    ) -> float:
        bits = _half_bits(amplitude_mv)
        current_dacs = sorted({int(value) for value in (pulse_dacs or active_dacs)})
        seq.append(
            _event(
                f"type stim name {name} amplitude_mv {amplitude_mv} "
                f"electrode_group {electrode_group_name} "
                f"dac_sources {','.join(str(value) for value in current_dacs)}",
                event_index,
            )
        )
        _append_dac_codes(seq, hold_dacs, 512, sync_dual_dac=sync_dual_dac)
        _append_dac_codes(seq, current_dacs, 512 - bits, sync_dual_dac=sync_dual_dac)
        seq.append(mx.DelaySamples(_samples_us(duration_us)))
        if ipi_us > 0:
            seq.append(mx.DelaySamples(_samples_us(ipi_us)))
        _append_dac_codes(seq, current_dacs, 512 + bits, sync_dual_dac=sync_dual_dac)
        seq.append(mx.DelaySamples(_samples_us(duration_us)))
        _append_dac_codes(seq, hold_dacs, 512, sync_dual_dac=sync_dual_dac)
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
            point_dacs = (
                [int(point["channel"])]
                if "channel" in point and str(point.get("channel", "")).strip()
                else None
            )
            pulse_width_ms = pulse(
                float(point["amplitude_mv"]),
                event_id,
                float(point.get("duration_us", width_us)),
                point_dacs,
            )
            event_id += 1
            current_ms = max(current_ms, point_ms) + pulse_width_ms
    else:
        raise ValueError(f"Unsupported protocol type: {ptype}")
    return seq


def build_poisson_random_sequence(
    protocol: dict[str, Any],
    plan_rows: list[dict[str, Any]],
    stim_unit_by_electrode: dict[int, int],
    stim_unit_to_dac: dict[int, int] | None = None,
    system_config: dict[str, Any] | None = None,
) -> Any:
    mx = _mx()
    seq = mx.Sequence(initial_delay=100, persistent=False)
    name = str(protocol.get("name", "poisson_random"))
    width_us_default = float(protocol.get("pulse_width_us", 300.0))
    ipi_us = float(protocol.get("inter_phase_interval_us", 0.0) or 0.0)
    signal_dacs, neutral_dac, sync_dual_dac = _hardware_dac_config(system_config)
    normalized_unit_map = {int(key): int(value) for key, value in stim_unit_by_electrode.items()}
    normalized_unit_to_dac = {
        int(key): int(value)
        for key, value in (stim_unit_to_dac or {}).items()
    }
    if not normalized_unit_to_dac:
        normalized_unit_to_dac = _default_stim_unit_dac_sources(normalized_unit_map, system_config)
    hold_dacs = sorted(set(signal_dacs) | {int(neutral_dac)})
    current_ms = 0.0
    event_id = 1
    connected_stim_unit: set[int] | None = None
    connected_route_signature: tuple[tuple[int, ...], tuple[tuple[int, int], ...]] | None = None
    connect_settle_ms = _plan_connect_settle_ms(protocol)
    event_level_switch = _event_level_switch_enabled(protocol)

    def pulse(
        amplitude_mv: float,
        event_index: int,
        duration_us: float,
        active_dacs: list[int],
        target_units: list[int],
    ) -> float:
        bits = _half_bits(amplitude_mv)
        seq.append(
            _event(
                f"type stim name {name} amplitude_mv {amplitude_mv} "
                f"stim_units {','.join(str(value) for value in target_units)} "
                f"dac_sources {','.join(str(value) for value in active_dacs)}",
                event_index,
            )
        )
        _append_dac_codes(seq, hold_dacs, 512, sync_dual_dac=sync_dual_dac)
        _append_dac_codes(seq, active_dacs, 512 - bits, sync_dual_dac=sync_dual_dac)
        seq.append(mx.DelaySamples(_samples_us(duration_us)))
        if ipi_us > 0:
            seq.append(mx.DelaySamples(_samples_us(ipi_us)))
        _append_dac_codes(seq, active_dacs, 512 + bits, sync_dual_dac=sync_dual_dac)
        seq.append(mx.DelaySamples(_samples_us(duration_us)))
        _append_dac_codes(seq, hold_dacs, 512, sync_dual_dac=sync_dual_dac)
        return (float(duration_us) * 2.0 + max(0.0, ipi_us)) / 1000.0

    for row in sorted(plan_rows, key=lambda item: (float(item["time_sec"]), int(item.get("electrode", 0)))):
        row_electrodes = _row_electrodes(row)
        if not row_electrodes:
            continue
        missing = [electrode for electrode in row_electrodes if electrode not in stim_unit_by_electrode]
        if missing:
            raise RuntimeError(f"No stimulation unit configured for stimulation electrode(s): {','.join(str(item) for item in missing)}")
        target_stim_units = {
            int(normalized_unit_map[electrode])
            for electrode in row_electrodes
        }
        target_unit_list = sorted(target_stim_units)
        event_unit_to_dac = (
            _event_unit_dac_sources(row_electrodes, normalized_unit_map, signal_dacs)
            if event_level_switch
            else {
                int(unit): int(normalized_unit_to_dac[unit])
                for unit in target_unit_list
                if unit in normalized_unit_to_dac
            }
        )
        route_signature = _route_signature(
            target_unit_list,
            event_unit_to_dac if event_level_switch else None,
            normalized_unit_to_dac,
            event_level_switch=event_level_switch,
        )
        active_dacs = sorted(set(event_unit_to_dac.values()))
        if not active_dacs:
            raise RuntimeError(
                "No DAC source configured for stimulation electrodes: "
                + ",".join(str(item) for item in row_electrodes)
            )
        point_ms = float(row["time_sec"]) * 1000.0
        current_units = set() if connected_stim_unit is None else set(connected_stim_unit)
        route_changed = (
            connected_route_signature is None
            or connected_route_signature != route_signature
            or (not event_level_switch and current_units != target_stim_units)
        )
        switch_needed = bool(route_changed)
        if route_changed and connect_settle_ms > 0.0:
            switch_ms = max(current_ms, point_ms - connect_settle_ms)
            if switch_ms > current_ms:
                seq.append(mx.DelaySamples(_samples_ms(switch_ms - current_ms)))
                current_ms = switch_ms
        elif point_ms > current_ms:
            seq.append(mx.DelaySamples(_samples_ms(point_ms - current_ms)))
            current_ms = point_ms
        if switch_needed:
            if event_level_switch:
                _append_dac_codes(seq, hold_dacs, 512, sync_dual_dac=sync_dual_dac)
            for stim_unit in sorted(current_units - target_stim_units):
                seq.append(mx.StimulationUnit(stim_unit).connect(False))
            units_to_activate = (
                sorted(target_stim_units)
                if event_level_switch
                else sorted(target_stim_units - current_units)
            )
            for stim_unit in units_to_activate:
                source = int(
                    event_unit_to_dac.get(
                        stim_unit,
                        normalized_unit_to_dac.get(stim_unit, signal_dacs[0]),
                    )
                )
                if event_level_switch:
                    seq.append(
                        mx.StimulationUnit(stim_unit)
                        .power_up(True)
                        .connect(True)
                        .set_voltage_mode()
                        .dac_source(source)
                    )
                else:
                    seq.append(mx.StimulationUnit(stim_unit).connect(True))
            connected_stim_unit = set(target_stim_units)
            connected_route_signature = route_signature
        if point_ms > current_ms:
            seq.append(mx.DelaySamples(_samples_ms(point_ms - current_ms)))
            current_ms = point_ms
        duration_ms = pulse(
            float(row.get("amplitude_mv", protocol.get("amplitude_mv", 150.0))),
            event_id,
            float(row.get("pulse_width_us", width_us_default)),
            active_dacs,
            target_unit_list,
        )
        event_id += 1
        current_ms = max(current_ms, point_ms) + duration_ms
    if connected_stim_unit is not None:
        if event_level_switch:
            _append_dac_codes(seq, hold_dacs, 512, sync_dual_dac=sync_dual_dac)
        for stim_unit in sorted(connected_stim_unit):
            seq.append(mx.StimulationUnit(stim_unit).connect(False))
    return seq


def _plan_connect_settle_ms(protocol: dict[str, Any]) -> float:
    switch_cfg = protocol.get("site_switch", {}) if isinstance(protocol, dict) else {}
    plan_cfg = protocol.get("electrode_pool_sequence", {}) if isinstance(protocol, dict) else {}
    random_cfg = protocol.get("random_electrode_plan", {}) if isinstance(protocol, dict) else {}
    raw_value = (
        switch_cfg.get("connect_settle_ms")
        if isinstance(switch_cfg, dict) and "connect_settle_ms" in switch_cfg
        else plan_cfg.get("connect_settle_ms")
        if isinstance(plan_cfg, dict) and "connect_settle_ms" in plan_cfg
        else random_cfg.get("connect_settle_ms")
        if isinstance(random_cfg, dict) and "connect_settle_ms" in random_cfg
        else None
    )
    if raw_value is None:
        if isinstance(switch_cfg, dict) and switch_cfg.get("enabled", False):
            raw_value = 3.0
        elif str(protocol.get("type", "")) == "electrode_pool_sequence":
            raw_value = 3.0
        else:
            raw_value = 0.0
    try:
        return max(0.0, float(raw_value))
    except (TypeError, ValueError):
        return 3.0


def _row_electrodes(row: dict[str, Any]) -> list[int]:
    values = row.get("electrodes")
    if isinstance(values, str):
        parsed = [token for token in values.replace(";", ",").split(",") if token.strip()]
        return [int(float(token)) for token in parsed]
    if isinstance(values, (list, tuple)):
        return [int(value) for value in values]
    if "electrode" in row:
        return [int(row["electrode"])]
    return []


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
    return_hardware_mapping: bool = False,
    initial_connect: bool = True,
) -> Any:
    mx = _mx()
    electrodes = [int(item) for item in electrode_group.get("electrodes", [])]
    if not electrodes:
        raise ValueError("No electrodes configured")
    unique_electrodes = sorted(set(electrodes))
    if len(unique_electrodes) > MAX_STIMULATION_UNITS_PER_ROUTE:
        raise RuntimeError(
            f"Configured stimulation pool contains {len(unique_electrodes)} electrodes, "
            f"but MaxOne supports at most {MAX_STIMULATION_UNITS_PER_ROUTE} stimulation units per route. "
            "Split the experiment into smaller blocks or reduce the event-group union."
        )
    array = mx.Array("stimulation")
    array.reset()
    array.clear_selected_electrodes()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"cfg_path does not exist: {cfg_path}")
    array.load_config(str(cfg_path))
    stim_units = []
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
    explicit_unit_sources = {
        int(key): int(value)
        for key, value in (source_by_stim_unit or {}).items()
    }
    if explicit_unit_sources:
        stim_unit_sources = dict(explicit_unit_sources)
        if source_by_electrode:
            for electrode, stim_unit in stim_unit_by_electrode.items():
                stim_unit_sources.setdefault(
                    int(stim_unit),
                    int(source_by_electrode.get(electrode, 0)),
                )
    elif source_by_electrode:
        stim_unit_sources = {}
        for electrode, stim_unit in stim_unit_by_electrode.items():
            source_id = int(source_by_electrode.get(electrode, 0))
            existing = stim_unit_sources.get(int(stim_unit))
            if existing is not None and existing != source_id:
                raise RuntimeError(f"Stimulation unit {stim_unit} mapped to conflicting DAC sources")
            stim_unit_sources[int(stim_unit)] = source_id
    else:
        stim_unit_sources = _default_stim_unit_dac_sources(
            stim_unit_by_electrode,
            system_config,
        )
    if not connected_electrodes and allow_missing_electrodes and probe_only:
        if return_hardware_mapping:
            return array, connected_electrodes, skipped_electrodes, stim_unit_by_electrode, stim_unit_sources
        if return_stim_units:
            return array, connected_electrodes, skipped_electrodes, stim_unit_by_electrode
        return array, connected_electrodes, skipped_electrodes
    if not connected_electrodes:
        raise RuntimeError("No stimulation unit can connect to any configured electrode")
    if probe_only:
        if return_hardware_mapping:
            return array, connected_electrodes, skipped_electrodes, stim_unit_by_electrode, stim_unit_sources
        if return_stim_units:
            return array, connected_electrodes, skipped_electrodes, stim_unit_by_electrode
        return array, connected_electrodes, skipped_electrodes
    mx.activate([0])
    _signal_dacs, neutral_dac, _sync_dual_dac = _hardware_dac_config(system_config)
    for stim_unit in sorted(set(stim_units)):
        initial_source = (
            int(neutral_dac)
            if not initial_connect
            else int(stim_unit_sources.get(stim_unit, 0))
        )
        mx.send(
            mx.StimulationUnit(stim_unit)
            .power_up(True)
            .connect(bool(initial_connect))
            .set_voltage_mode()
            .dac_source(initial_source)
        )
    array.download([0])
    time.sleep(getattr(mx.Timing, "waitAfterDownload", 0))
    mx.offset()
    if return_hardware_mapping:
        return array, connected_electrodes, skipped_electrodes, stim_unit_by_electrode, stim_unit_sources
    if return_stim_units:
        return array, connected_electrodes, skipped_electrodes, stim_unit_by_electrode
    return array, connected_electrodes, skipped_electrodes


def probe_stimulation_electrodes(
    cfg_path: Path,
    electrode_group: dict[str, Any],
    system_config: dict[str, Any],
) -> tuple[list[int], list[int], dict[int, int]]:
    electrodes = [int(item) for item in electrode_group.get("electrodes", [])]
    if len(set(electrodes)) <= MAX_STIMULATION_UNITS_PER_ROUTE:
        _array, connected, skipped, stim_units = resolve_experiment_array(
            cfg_path,
            electrode_group,
            system_config,
            allow_missing_electrodes=True,
            probe_only=True,
            return_stim_units=True,
        )
        return connected, skipped, stim_units

    connected_all: list[int] = []
    skipped_all: list[int] = []
    stim_units_all: dict[int, int] = {}
    unique_electrodes = sorted(set(electrodes))
    for offset in range(0, len(unique_electrodes), MAX_STIMULATION_UNITS_PER_ROUTE):
        probe_group = dict(electrode_group)
        probe_group["electrodes"] = unique_electrodes[offset : offset + MAX_STIMULATION_UNITS_PER_ROUTE]
        _array, connected, skipped, stim_units = resolve_experiment_array(
            cfg_path,
            probe_group,
            system_config,
            allow_missing_electrodes=True,
            probe_only=True,
            return_stim_units=True,
        )
        connected_all.extend(int(item) for item in connected)
        skipped_all.extend(int(item) for item in skipped)
        stim_units_all.update({int(electrode): int(stim_unit) for electrode, stim_unit in stim_units.items()})
    return connected_all, skipped_all, stim_units_all


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


def configure_experiment_array_with_mapping(
    cfg_path: Path,
    electrode_group: dict[str, Any],
    system_config: dict[str, Any],
    *,
    initial_connect: bool = True,
) -> tuple[Any, dict[int, int], dict[int, int]]:
    (
        array,
        _connected_electrodes,
        _skipped_electrodes,
        stim_unit_by_electrode,
        stim_unit_to_dac,
    ) = resolve_experiment_array(
        cfg_path,
        electrode_group,
        system_config,
        return_hardware_mapping=True,
        initial_connect=initial_connect,
    )
    return array, stim_unit_by_electrode, stim_unit_to_dac


def configure_poisson_experiment_array(
    cfg_path: Path,
    electrode_group: dict[str, Any],
    system_config: dict[str, Any],
) -> tuple[Any, dict[int, int], dict[int, int]]:
    (
        array,
        _connected_electrodes,
        _skipped_electrodes,
        stim_unit_by_electrode,
        stim_unit_to_dac,
    ) = resolve_experiment_array(
        cfg_path,
        electrode_group,
        system_config,
        return_hardware_mapping=True,
        initial_connect=False,
    )
    return array, stim_unit_by_electrode, stim_unit_to_dac


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
    elif ptype == "electrode_pool_sequence":
        cfg = protocol.get("electrode_pool_sequence", {}) or {}
        explicit = cfg.get("event_groups") or []
        if explicit:
            count = len(_expand_pool_event_groups(
                explicit,
                mode=str(cfg.get("selection_mode", "balanced_random_groups") or "balanced_random_groups"),
                repeats=max(1, int(cfg.get("event_count", 1) or 1)),
                random_seed=int(cfg.get("random_seed", protocol.get("random_seed", 42))),
            ))
        else:
            count = max(0, int(cfg.get("event_count", 10)))
        interval_ms = max(0.0, float(cfg.get("event_interval_ms", 1000.0)))
        times_ms.extend(start_ms + index * interval_ms for index in range(count))
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
