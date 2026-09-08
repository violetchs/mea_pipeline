from __future__ import annotations

import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from python.maxwell_setup import (
    build_poisson_random_sequence,
    build_stim_sequence,
    configure_poisson_experiment_array,
    configure_experiment_array,
    configure_experiment_array_with_mapping,
    create_experiment_saving,
    get_stim_times_for_protocol,
    initialize_maxlab,
    probe_stimulation_electrodes,
)
from python.random_stim_plan import build_poisson_random_plan
from python.random_stim_plan import build_electrode_pool_sequence_plan
from python.random_stim_plan import build_site_switch_plan
from python.random_stim_plan import electrode_pool_event_groups
from python.random_stim_plan import site_switch_event_groups_for_count
from python.random_stim_plan import poisson_rates_for_electrodes
from python.random_stim_plan import select_poisson_candidate_electrodes
from python.utils.time_log import ExternalTimeLog, SegmentStimLog

PLAN_PROTOCOL_TYPES = {"poisson_random_electrodes", "electrode_pool_sequence"}
MAX_STIMULATION_UNITS_PER_ROUTE = 32


def _protocol_uses_plan(protocol: dict[str, Any]) -> bool:
    if protocol.get("type") in PLAN_PROTOCOL_TYPES:
        return True
    return _protocol_requires_dynamic_site_switch(protocol)


def _protocol_routing_mode(protocol: dict[str, Any]) -> str:
    if _protocol_uses_plan(protocol):
        return "dynamic_connect_switch"
    return "static_route"


def _hardware_dac_config(system_config: dict[str, Any]) -> tuple[list[int], int, bool]:
    maxwell_config = system_config.get("maxwell", {}) if isinstance(system_config, dict) else {}
    hardware_config = maxwell_config.get("hardware_dac", {}) if isinstance(maxwell_config, dict) else {}
    signal_dacs: list[int] = []
    for value in hardware_config.get("signal_dacs", [0, 1]):
        try:
            dac = int(value)
        except (TypeError, ValueError):
            continue
        if dac >= 0 and dac not in signal_dacs:
            signal_dacs.append(dac)
    if not signal_dacs:
        signal_dacs = [0, 1]
    neutral_dac = int(hardware_config.get("neutral_dac", 2))
    if neutral_dac in signal_dacs:
        raise ValueError("maxwell.hardware_dac.neutral_dac must not be a signal DAC")
    return signal_dacs, neutral_dac, bool(hardware_config.get("sync_dual_dac", True))


def _hardware_system_config(
    system_config: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    result = dict(system_config)
    maxwell_config = dict(system_config.get("maxwell", {}) or {})
    hardware_config = dict(maxwell_config.get("hardware_dac", {}) or {})
    protocol_hardware = protocol.get("hardware_dac", {}) if isinstance(protocol, dict) else {}
    if isinstance(protocol_hardware, dict):
        hardware_config.update(protocol_hardware)
    maxwell_config["hardware_dac"] = hardware_config
    result["maxwell"] = maxwell_config
    return result


def _hardware_mapping_payload(
    system_config: dict[str, Any],
    stim_unit_by_electrode: dict[int, int],
    stim_unit_to_dac: dict[int, int],
    *,
    routing_mode: str,
    connect_settle_ms: float,
    initial_connect: bool,
) -> dict[str, Any]:
    signal_dacs, neutral_dac, sync_dual_dac = _hardware_dac_config(system_config)
    return {
        "routing_mode": routing_mode,
        "initial_connect": bool(initial_connect),
        "connect_settle_ms": float(connect_settle_ms),
        "signal_dacs": [int(value) for value in signal_dacs],
        "neutral_dac": int(neutral_dac),
        "sync_dual_dac": bool(sync_dual_dac),
        "dac_allocation": "round_robin_stimulation_units",
        "electrode_to_stim_unit": {
            str(int(electrode)): int(stim_unit)
            for electrode, stim_unit in sorted(stim_unit_by_electrode.items())
        },
        "stim_unit_to_dac_source": {
            str(int(stim_unit)): int(dac_source)
            for stim_unit, dac_source in sorted(stim_unit_to_dac.items())
        },
    }


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


def _route_signature(
    target_stim_units: list[int],
    event_unit_to_dac: dict[int, int] | None = None,
    stim_unit_to_dac: dict[int, int] | None = None,
    *,
    event_level_switch: bool = False,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Return the routing state used to decide whether hardware must switch."""
    units = tuple(sorted({int(value) for value in target_stim_units}))
    unit_to_dac = event_unit_to_dac if event_level_switch else stim_unit_to_dac
    dac_items = tuple(
        sorted(
            (int(unit), int(source))
            for unit, source in (unit_to_dac or {}).items()
            if int(unit) in units
        )
    )
    return units, dac_items


def _hardware_route_records(
    stim_times: list[float],
    plan_rows: list[dict[str, Any]],
    electrode_group: dict[str, Any],
    stim_unit_by_electrode: dict[int, int],
    stim_unit_to_dac: dict[int, int],
    connect_settle_ms: float,
    *,
    initial_connect: bool,
    event_level_switch: bool = False,
    signal_dacs: list[int] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    previous_units: set[int] = set()
    previous_route_signature: tuple[tuple[int, ...], tuple[tuple[int, int], ...]] | None = None
    active_signal_dacs = [int(value) for value in (signal_dacs or [0, 1]) if int(value) >= 0] or [0, 1]
    for index, stim_time in enumerate(stim_times, start=1):
        row = plan_rows[index - 1] if index <= len(plan_rows) else {}
        row_electrodes = _row_electrodes(row)
        if not row_electrodes:
            row_electrodes = [int(value) for value in electrode_group.get("electrodes", [])]
        target_units = sorted(
            {
                int(stim_unit_by_electrode[electrode])
                for electrode in row_electrodes
                if electrode in stim_unit_by_electrode
            }
        )
        if event_level_switch:
            event_unit_to_dac: dict[int, int] = {}
            for electrode in sorted({int(value) for value in row_electrodes}):
                stim_unit = stim_unit_by_electrode.get(electrode)
                if stim_unit is None:
                    continue
                stim_unit_int = int(stim_unit)
                if stim_unit_int not in event_unit_to_dac:
                    event_unit_to_dac[stim_unit_int] = active_signal_dacs[
                        len(event_unit_to_dac) % len(active_signal_dacs)
                    ]
            active_dacs = sorted(set(event_unit_to_dac.values()))
            event_dac_by_stim_unit = event_unit_to_dac
        else:
            active_dacs = sorted(
                {
                    int(stim_unit_to_dac[stim_unit])
                    for stim_unit in target_units
                    if stim_unit in stim_unit_to_dac
                }
            )
            event_dac_by_stim_unit = {
                int(stim_unit): int(stim_unit_to_dac[stim_unit])
                for stim_unit in target_units
                if stim_unit in stim_unit_to_dac
            }
        route_signature = (
            tuple(sorted({int(value) for value in target_units})),
            tuple(sorted((int(unit), int(source)) for unit, source in (event_dac_by_stim_unit if event_level_switch else event_dac_by_stim_unit).items()))
            if event_level_switch
            else tuple(
                sorted(
                    (int(unit), int(source))
                    for unit, source in stim_unit_to_dac.items()
                    if int(unit) in target_units
                )
            ),
        )
        route_switched = bool(target_units) and (
            event_level_switch
            or previous_route_signature is None
            or route_signature != previous_route_signature
        )
        if initial_connect and index == 1:
            route_switched = False
        route_time = (
            max(0.0, float(stim_time) - max(0.0, float(connect_settle_ms)) / 1000.0)
            if route_switched
            else None
        )
        records.append(
            {
                "stim_index": int(index),
                "pulse_time_s": round(float(stim_time), 6),
                "route_switch": bool(route_switched),
                "connect_time_s": None if route_time is None else round(route_time, 6),
                "settle_ms": float(connect_settle_ms) if route_switched else 0.0,
                "previous_stim_units": sorted(previous_units),
                "target_stim_units": target_units,
                "active_dac_sources": active_dacs,
                "stim_unit_to_event_dac": event_dac_by_stim_unit,
                "electrodes": [int(value) for value in row_electrodes],
            }
        )
        previous_units = set(target_units)
        previous_route_signature = route_signature
    return records


def _protocol_requires_dynamic_site_switch(protocol: dict[str, Any]) -> bool:
    switch_cfg = protocol.get("site_switch", {}) or {}
    if not switch_cfg.get("enabled", False):
        return False
    explicit = switch_cfg.get("event_groups") or []
    groups: list[tuple[int, ...]] = []
    for raw_group in explicit:
        values = tuple(_event_group_values(raw_group))
        if values:
            groups.append(values)
    return len(set(groups)) > 1


def run_experiment(system_config: dict[str, Any], stimulation_config: dict[str, Any], run_dir: Path, dry_run: bool = False) -> None:
    raw_root = run_dir / "raw_data"
    raw_root.mkdir(parents=True, exist_ok=True)
    audit = ExternalTimeLog(run_dir=run_dir)
    audit.mark_event("experiment_start", "", "", "")

    groups = {item["name"]: item for item in stimulation_config.get("electrode_groups", [])}
    protocols = {item["name"]: item for item in stimulation_config.get("protocols", [])}
    _hydrate_site_switch_event_centers(protocols, groups)
    blocks = system_config.get("experiment", {}).get("blocks", [])
    if not blocks:
        raise ValueError("No experiment.blocks configured")
    recording_prefix = system_config.get("experiment", {}).get("recording_name_prefix") or system_config.get("experiment", {}).get("name", "recording")
    cfg_path = Path(system_config.get("electrode_map", {}).get("cfg_path", ""))
    logging.info("Experiment start: run_dir=%s dry_run=%s blocks=%d cfg=%s", run_dir, dry_run, len(blocks), cfg_path)
    audit.mark_event(
        "experiment_config_loaded",
        "",
        "",
        "",
        extra={"run_dir": str(run_dir), "dry_run": dry_run, "block_count": len(blocks), "cfg_path": str(cfg_path)},
    )

    if dry_run:
        logging.info("Dry run enabled: hardware calls skipped")
        audit.mark_event("hardware_skipped_dry_run", "", "", "")
    else:
        try:
            audit.mark_event("cfg_preflight_start", "", "", "", extra={"cfg_path": str(cfg_path)})
            preflight = _validate_cfg_stimulation_sites(cfg_path, blocks, groups, protocols)
            logging.info(
                "CFG preflight OK: cfg electrodes=%d requested stimulation electrodes=%d",
                preflight["cfg_electrode_count"],
                preflight["requested_electrode_count"],
            )
            audit.mark_event("cfg_preflight_ok", "", "", "", extra=preflight)
            audit.mark_event("hardware_initialize_start", "", "", "")
            initialize_maxlab()
            audit.mark_event("hardware_initialize_done", "", "", "")
            logging.info("MaxLab initialized")
        except Exception as exc:
            logging.exception("Experiment startup failed")
            audit.mark_event("experiment_startup_failed", "", "", "", extra={"error": str(exc)})
            audit.mark_event("experiment_end", "", "", "")
            audit.save()
            raise

    try:
        for block in blocks:
            block_name = block["name"]
            block_dir = raw_root / block_name
            block_dir.mkdir(parents=True, exist_ok=True)
            (block_dir / "block_meta.yaml").write_text(
                f"name: {block_name}\nelectrode_group: {block['electrode_group']}\nprotocol: {block['protocol']}\n",
                encoding="utf-8",
            )
            protocol = protocols.get(block.get("protocol", ""), {})
            electrode_group = groups.get(block.get("electrode_group", ""), {"name": "", "electrodes": []})
            is_record_only = _block_has_record_only_stim(block)
            is_rest_only = _block_is_rest_only(block)
            logging.info(
                "Block start: %s protocol=%s group=%s record_only=%s rest_only=%s",
                block_name,
                block.get("protocol", ""),
                block.get("electrode_group", ""),
                is_record_only,
                is_rest_only,
            )
            audit.mark_event(
                "block_start",
                block_name,
                "",
                "",
                extra={
                    "protocol": block.get("protocol", ""),
                    "electrode_group": block.get("electrode_group", ""),
                    "record_only": is_record_only,
                    "rest_only": is_rest_only,
                },
            )
            if protocol and electrode_group:
                electrode_group = _effective_electrode_group(protocol, electrode_group)
                hardware_system_config = _hardware_system_config(system_config, protocol)
                prepared_stim_unit_by_electrode: dict[int, int] = {}
                prepared_stim_unit_to_dac: dict[int, int] = {}
                if not dry_run and not _protocol_uses_plan(protocol) and not is_record_only and not is_rest_only:
                    electrode_group, replacements, unresolved_electrodes = _replace_unconnectable_stimulation_electrodes(
                        cfg_path,
                        electrode_group,
                        system_config,
                        protocol,
                    )
                    if replacements:
                        replacement_text = ",".join(f"{source}->{target}" for source, target in replacements.items())
                        logging.warning("Replaced unconnectable stimulation electrodes: %s", replacement_text)
                        audit.mark_event(
                            "stimulation_electrode_replacement",
                            block_name,
                            "",
                            "",
                            extra={"replacements": replacement_text},
                        )
                    if unresolved_electrodes:
                        raise RuntimeError(
                            "No connectable replacement found for stimulation electrode(s): "
                            + ",".join(str(item) for item in unresolved_electrodes)
                        )
                    audit.mark_event(
                        "array_configure_start",
                        block_name,
                        "",
                        "",
                        extra={
                            "protocol": protocol.get("name", ""),
                            "protocol_type": protocol.get("type", ""),
                            "electrode_group": electrode_group.get("name", ""),
                            "electrode_count": len(electrode_group.get("electrodes", [])),
                        },
                    )
                    (
                        _array,
                        prepared_stim_unit_by_electrode,
                        prepared_stim_unit_to_dac,
                    ) = configure_experiment_array_with_mapping(
                        cfg_path,
                        electrode_group,
                        hardware_system_config,
                        initial_connect=True,
                    )
                    audit.mark_event(
                        "array_configure_done",
                        block_name,
                        "",
                        "",
                        extra={
                            "stim_unit_count": len(set(prepared_stim_unit_by_electrode.values())),
                            "stim_unit_to_dac_source": prepared_stim_unit_to_dac,
                        },
                    )
                    logging.info("Array configured: block=%s electrodes=%d", block_name, len(electrode_group.get("electrodes", [])))
            else:
                hardware_system_config = system_config
                prepared_stim_unit_by_electrode = {}
                prepared_stim_unit_to_dac = {}

            for phase in block.get("phases", []):
                phase_id = phase["id"]
                phase_dir = block_dir / phase_id
                phase_dir.mkdir(parents=True, exist_ok=True)
                duration_s = int(phase.get("duration_s", 300))
                segment_name = f"{recording_prefix}_{block_name}_{phase_id}"
                segment_start = time.time()
                logging.info("Phase start: block=%s phase=%s duration_s=%d segment=%s", block_name, phase_id, duration_s, segment_name)
                if not _phase_is_rest_only(phase):
                    audit.mark_event("record_start", block_name, phase_id, segment_name, epoch_sec=segment_start)

                if _phase_is_rest_only(phase):
                    audit.mark_event("rest_start", block_name, phase_id, segment_name, epoch_sec=segment_start)
                    logging.info("Rest-only phase: block=%s phase=%s duration_s=%d", block_name, phase_id, duration_s)
                    if not dry_run and duration_s > 0:
                        time.sleep(duration_s)
                    audit.mark_event("rest_end", block_name, phase_id, segment_name)
                elif phase_id == "02_stim" and not _phase_is_record_only(phase):
                    _run_stim_phase(
                        cfg_path=cfg_path,
                        system_config=system_config,
                        phase_dir=phase_dir,
                        block_name=block_name,
                        phase_id=phase_id,
                        segment_name=segment_name,
                        duration_s=duration_s,
                        protocol=protocol,
                        electrode_group=electrode_group,
                        audit=audit,
                        segment_start=segment_start,
                        dry_run=dry_run,
                        prepared_stim_unit_by_electrode=prepared_stim_unit_by_electrode,
                        prepared_stim_unit_to_dac=prepared_stim_unit_to_dac,
                        hardware_system_config=hardware_system_config,
                    )
                elif not dry_run:
                    audit.mark_event("recording_file_start", block_name, phase_id, segment_name, extra={"phase_mode": phase.get("mode", "")})
                    saving = create_experiment_saving(phase_dir, segment_name)
                    time.sleep(duration_s)
                    saving.stop_recording()
                    saving.stop_file()
                    audit.mark_event("recording_file_stop", block_name, phase_id, segment_name)
                    logging.info("Recording-only phase saved: block=%s phase=%s", block_name, phase_id)
                if phase_id == "02_stim" and _phase_is_record_only(phase):
                    logging.info("Record-only stim phase: block=%s phase=%s pulse_count=0", block_name, phase_id)
                    segment_log = SegmentStimLog(block_name, phase_id, segment_name, segment_start)
                    segment_log.record_end_epoch = time.time()
                    segment_log.save_txt(phase_dir / "stim_times.txt")
                    segment_log.save_json(phase_dir / "segment_time_meta.json")
                    audit.mark_event("stim_log_written", block_name, phase_id, segment_name, extra={"pulse_count": 0})

                if _phase_is_rest_only(phase):
                    audit.mark_event("phase_end", block_name, phase_id, segment_name)
                else:
                    audit.mark_event("record_end", block_name, phase_id, segment_name)
                logging.info("Phase end: block=%s phase=%s", block_name, phase_id)
            audit.mark_event("block_end", block_name, "", "")
            logging.info("Block end: %s", block_name)
    finally:
        audit.mark_event("experiment_end", "", "", "")
        audit.save()
        logging.info("Experiment end: %s", run_dir)


def _phase_is_record_only(phase: dict[str, Any]) -> bool:
    return str(phase.get("mode", "")).strip().lower() in {"record_only", "recording_only", "no_stim", "none"}


def _phase_is_rest_only(phase: dict[str, Any]) -> bool:
    return str(phase.get("mode", "")).strip().lower() in {"rest_only", "rest", "recovery", "idle", "no_record"}


def _block_has_record_only_stim(block: dict[str, Any]) -> bool:
    for phase in block.get("phases", []) or []:
        if str(phase.get("id", "")) == "02_stim":
            return _phase_is_record_only(phase)
    return False


def _block_is_rest_only(block: dict[str, Any]) -> bool:
    phases = list(block.get("phases", []) or [])
    return bool(phases) and all(_phase_is_rest_only(phase) for phase in phases)


def _validate_cfg_stimulation_sites(
    cfg_path: Path,
    blocks: list[dict[str, Any]],
    groups: dict[str, dict[str, Any]],
    protocols: dict[str, dict[str, Any]],
) -> dict[str, int]:
    cfg_electrodes = _cfg_recording_electrodes(cfg_path)
    requested: list[int] = []
    missing_references: list[str] = []
    for block in blocks:
        if _block_has_record_only_stim(block) or _block_is_rest_only(block):
            continue
        block_name = str(block.get("name", ""))
        protocol_name = str(block.get("protocol", ""))
        group_name = str(block.get("electrode_group", ""))
        protocol = protocols.get(protocol_name)
        electrode_group = groups.get(group_name)
        if protocol is None:
            missing_references.append(f"{block_name}: protocol {protocol_name!r}")
            continue
        if electrode_group is None:
            missing_references.append(f"{block_name}: electrode_group {group_name!r}")
            continue
        effective_group = _effective_electrode_group(protocol, electrode_group)
        _validate_stimulation_unit_pool_size(protocol, effective_group, context=f"block {block_name}")
        requested.extend(int(item) for item in effective_group.get("electrodes", []))
    if missing_references:
        raise RuntimeError("Invalid block references before experiment run: " + "; ".join(missing_references))
    missing = sorted({electrode for electrode in requested if electrode not in cfg_electrodes})
    if missing:
        raise RuntimeError(
            "Stimulation electrodes are not present in the cfg recording map: "
            + ",".join(str(electrode) for electrode in missing)
        )
    return {
        "cfg_electrode_count": len(cfg_electrodes),
        "requested_electrode_count": len(set(requested)),
    }


def _hydrate_site_switch_event_centers(protocols: dict[str, dict[str, Any]], groups: dict[str, dict[str, Any]]) -> None:
    centers_by_values: dict[tuple[int, ...], int] = {}
    for group in groups.values():
        values = tuple(_event_group_values(group.get("electrodes", [])))
        center = _group_center_electrode(group)
        if values and center is not None:
            centers_by_values[values] = int(center)
            centers_by_values[tuple(sorted(values))] = int(center)
    for protocol in protocols.values():
        switch_cfg = protocol.get("site_switch", {}) if isinstance(protocol, dict) else {}
        if not isinstance(switch_cfg, dict) or not switch_cfg.get("enabled"):
            continue
        event_groups = switch_cfg.get("event_groups") or []
        if not event_groups:
            continue
        raw_centers = list(switch_cfg.get("event_group_centers") or [])
        centers: list[int | None] = []
        changed = len(raw_centers) != len(event_groups)
        for index, raw_group in enumerate(event_groups):
            values = _event_group_values(raw_group)
            center = _normalize_site_switch_center(raw_centers[index] if index < len(raw_centers) else None)
            if center is None:
                matched_center = centers_by_values.get(
                    tuple(values),
                    centers_by_values.get(tuple(sorted(values))),
                )
                if matched_center is not None:
                    center = int(matched_center)
                    changed = True
            centers.append(center)
        if changed and any(center is not None for center in centers):
            switch_cfg["event_group_centers"] = centers


def _normalize_site_switch_center(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _event_group_values(raw_group: Any) -> list[int]:
    if raw_group is None:
        return []
    if isinstance(raw_group, str):
        return [int(float(token)) for token in re.split(r"[\s,;]+", raw_group.strip("[]()")) if token.strip()]
    if isinstance(raw_group, dict):
        for key in ("electrodes", "group", "values"):
            if key in raw_group:
                return _event_group_values(raw_group.get(key))
        return []
    if isinstance(raw_group, (list, tuple, set)):
        values: list[int] = []
        for value in raw_group:
            if isinstance(value, (list, tuple, set, dict)):
                values.extend(_event_group_values(value))
            elif isinstance(value, str) and ("," in value or ";" in value or value.strip().startswith("[")):
                values.extend(_event_group_values(value))
            else:
                values.append(int(value))
        return _unique_ints(values)
    return [int(raw_group)]


def _cfg_recording_electrodes(cfg_path: Path) -> set[int]:
    if not cfg_path.is_file():
        raise FileNotFoundError(f"cfg_path does not exist: {cfg_path}")
    text = cfg_path.read_text(encoding="utf-8", errors="replace")
    electrodes = {int(match) for match in re.findall(r"\b\d+\((\d+)\)", text)}
    if not electrodes:
        raise RuntimeError(f"No recording electrodes could be parsed from cfg_path: {cfg_path}")
    return electrodes


def _run_stim_phase(
    cfg_path: Path,
    system_config: dict[str, Any],
    phase_dir: Path,
    block_name: str,
    phase_id: str,
    segment_name: str,
    duration_s: int,
    protocol: dict[str, Any],
    electrode_group: dict[str, Any],
    audit: ExternalTimeLog,
    segment_start: float,
    dry_run: bool,
    prepared_stim_unit_by_electrode: dict[int, int] | None = None,
    prepared_stim_unit_to_dac: dict[int, int] | None = None,
    hardware_system_config: dict[str, Any] | None = None,
) -> None:
    protocol_type = str(protocol.get("type", ""))
    protocol_name = str(protocol.get("name", ""))
    routing_mode = _protocol_routing_mode(protocol)
    source_group_name = str(electrode_group.get("name", ""))
    source_electrode_count = len(electrode_group.get("electrodes", []))
    logging.info(
        "Stim phase prepare: block=%s protocol=%s type=%s routing=%s group=%s electrodes=%d duration_s=%d",
        block_name,
        protocol_name,
        protocol_type,
        routing_mode,
        source_group_name,
        source_electrode_count,
        duration_s,
    )
    audit.mark_event(
        "stim_phase_prepare",
        block_name,
        phase_id,
        segment_name,
        extra={
            "protocol": protocol_name,
            "protocol_type": protocol_type,
            "routing_mode": routing_mode,
            "electrode_group": source_group_name,
            "electrode_count": source_electrode_count,
            "duration_s": duration_s,
        },
    )
    filtered_group = dict(electrode_group)
    filtered_group["electrodes"] = [int(item) for item in electrode_group.get("electrodes", [])]
    if protocol.get("type") == "poisson_random_electrodes":
        filtered_group = dict(electrode_group)
        replacements: dict[int, int] = {}
        unresolved_electrodes: list[int] = []
        if not dry_run:
            audit.mark_event("poisson_electrode_probe_start", block_name, phase_id, segment_name)
            filtered_group, replacements, unresolved_electrodes = _replace_unconnectable_poisson_electrodes(
                cfg_path,
                electrode_group,
                system_config,
                protocol,
            )
            audit.mark_event(
                "poisson_electrode_probe_done",
                block_name,
                phase_id,
                segment_name,
                extra={
                    "source_electrode_count": source_electrode_count,
                    "effective_electrode_count": len(filtered_group.get("electrodes", [])),
                    "replacement_count": len(replacements),
                    "unresolved_count": len(unresolved_electrodes),
                },
            )
            if replacements:
                replacement_text = ",".join(f"{source}->{target}" for source, target in replacements.items())
                logging.warning("Replaced unconnectable poisson electrodes within expanded neighborhood: %s", replacement_text)
                audit.mark_event(
                    "poisson_electrode_replacement",
                    block_name,
                    phase_id,
                    segment_name,
                    extra={"replacements": replacement_text},
                )
            if unresolved_electrodes:
                logging.warning(
                    "Skipped %d poisson electrodes with no connectable expanded-neighborhood replacement: %s",
                    len(unresolved_electrodes),
                    ",".join(str(item) for item in unresolved_electrodes),
                )
            if not filtered_group.get("electrodes"):
                raise RuntimeError("No stimulation unit can connect to any poisson candidate electrode")
        else:
            filtered_group["electrodes"] = [int(item) for item in electrode_group.get("electrodes", [])]
        plan_rows = build_poisson_random_plan(
            protocol=protocol,
            phase_dir=phase_dir,
            duration_s=duration_s,
            fallback_electrodes=[int(item) for item in filtered_group.get("electrodes", [])],
        )
        stim_times = [float(row["time_sec"]) for row in plan_rows]
    elif protocol.get("type") == "electrode_pool_sequence":
        filtered_group = dict(electrode_group)
        filtered_group["electrodes"] = [int(item) for item in electrode_group.get("electrodes", [])]
        if not filtered_group["electrodes"]:
            raise ValueError("Electrode pool sequence needs a non-empty site group")
        plan_rows = build_electrode_pool_sequence_plan(
            protocol=protocol,
            phase_dir=phase_dir,
            duration_s=duration_s,
            fallback_electrodes=filtered_group["electrodes"],
        )
        stim_times = [float(row["time_sec"]) for row in plan_rows]
    elif _protocol_uses_plan(protocol):
        filtered_group = dict(electrode_group)
        filtered_group["electrodes"] = [int(item) for item in electrode_group.get("electrodes", [])]
        if not filtered_group["electrodes"]:
            raise ValueError("Site switching needs a non-empty site group")
        base_stim_times = get_stim_times_for_protocol(protocol, duration_s)
        plan_rows = build_site_switch_plan(
            protocol=protocol,
            phase_dir=phase_dir,
            duration_s=duration_s,
            fallback_electrodes=filtered_group["electrodes"],
            stim_times_sec=base_stim_times,
        )
        stim_times = [float(row["time_sec"]) for row in plan_rows]
    else:
        plan_rows = []
        stim_times = get_stim_times_for_protocol(protocol, duration_s)
    if not dry_run and _protocol_uses_plan(protocol) and protocol.get("type") != "poisson_random_electrodes":
        plan_electrodes = sorted(_planned_electrodes(plan_rows, filtered_group))
        electrode_center_lookup = _planned_electrode_centers(plan_rows, filtered_group)
        probe_group = dict(filtered_group)
        probe_group["electrodes"] = plan_electrodes
        filtered_group, replacements, unresolved_electrodes = _replace_unconnectable_stimulation_electrodes(
            cfg_path,
            probe_group,
            system_config,
            protocol,
            electrode_center_lookup=electrode_center_lookup,
        )
        if replacements:
            plan_rows = _apply_electrode_replacements_to_plan_rows(plan_rows, replacements)
            _rewrite_replaced_stim_plan_files(phase_dir, plan_rows, filtered_group, replacements)
            replacement_text = ",".join(f"{source}->{target}" for source, target in replacements.items())
            logging.warning("Replaced unconnectable stimulation electrodes in plan: %s", replacement_text)
            audit.mark_event(
                "stimulation_electrode_replacement",
                block_name,
                phase_id,
                segment_name,
                extra={"replacements": replacement_text},
            )
        if unresolved_electrodes:
            raise RuntimeError(
                "No connectable replacement found for stimulation electrode(s): "
                + ",".join(str(item) for item in unresolved_electrodes)
            )
    planned_electrodes = _planned_electrodes(plan_rows, filtered_group)
    _validate_stimulation_unit_pool_size(
        protocol,
        {"name": filtered_group.get("name", ""), "electrodes": sorted(planned_electrodes)},
        context=f"block {block_name} phase {phase_id}",
    )
    audit.mark_event(
        "stim_plan_ready",
        block_name,
        phase_id,
        segment_name,
        extra={
            "stim_count": len(stim_times),
            "planned_electrode_count": len(planned_electrodes),
            "routing_mode": routing_mode,
            "first_stim_time_sec": min(stim_times) if stim_times else "",
            "last_stim_time_sec": max(stim_times) if stim_times else "",
        },
    )
    logging.info(
        "Stim plan ready: block=%s protocol=%s routing=%s pulses=%d electrodes=%d first=%s last=%s",
        block_name,
        protocol_name,
        routing_mode,
        len(stim_times),
        len(planned_electrodes),
        min(stim_times) if stim_times else "",
        max(stim_times) if stim_times else "",
    )
    sequence_start_epoch = segment_start
    sequence_start_offset = 0.0
    stim_unit_by_electrode = {
        int(key): int(value)
        for key, value in (prepared_stim_unit_by_electrode or {}).items()
    }
    stim_unit_to_dac = {
        int(key): int(value)
        for key, value in (prepared_stim_unit_to_dac or {}).items()
    }
    effective_hardware_config = hardware_system_config or _hardware_system_config(system_config, protocol)
    connect_settle_ms = _plan_connect_settle_ms(protocol)
    if dry_run:
        segment_log = SegmentStimLog(block_name, phase_id, segment_name, segment_start)
        sequence_start_offset = _recording_settle_s(system_config, protocol)
        sequence_start_epoch = segment_start + sequence_start_offset
        segment_log.set_hardware_mapping(
            _hardware_mapping_payload(
                effective_hardware_config,
                stim_unit_by_electrode,
                stim_unit_to_dac,
                routing_mode=routing_mode,
                connect_settle_ms=connect_settle_ms,
                initial_connect=not _protocol_uses_plan(protocol),
            )
        )
        logging.info("Dry run: %s %s pulses=%d", block_name, protocol.get("name"), len(stim_times))
        audit.mark_event("stim_dry_run_complete", block_name, phase_id, segment_name, extra={"stim_count": len(stim_times)})
    else:
        if _protocol_uses_plan(protocol):
            audit.mark_event(
                "array_configure_start",
                block_name,
                phase_id,
                segment_name,
                extra={
                    "protocol": protocol_name,
                    "protocol_type": protocol_type,
                    "routing_mode": routing_mode,
                    "electrode_count": len(filtered_group.get("electrodes", [])),
                    "initial_connect": False,
                },
            )
            (
                _array,
                stim_unit_by_electrode,
                stim_unit_to_dac,
            ) = configure_poisson_experiment_array(
                cfg_path,
                filtered_group,
                effective_hardware_config,
            )
            audit.mark_event(
                "array_configure_done",
                block_name,
                phase_id,
                segment_name,
                extra={
                    "stim_unit_count": len(set(stim_unit_by_electrode.values())),
                    "stim_unit_to_dac_source": stim_unit_to_dac,
                },
            )
            logging.info("Plan-based array configured: block=%s stim_units=%d", block_name, len(set(stim_unit_by_electrode.values())))
        audit.mark_event("recording_file_start", block_name, phase_id, segment_name, extra={"phase_mode": "stimulation"})
        saving = create_experiment_saving(phase_dir, segment_name)
        record_start_epoch = time.time()
        audit.mark_event("recording_file_started", block_name, phase_id, segment_name, epoch_sec=record_start_epoch)
        logging.info("Recording started: block=%s phase=%s segment=%s", block_name, phase_id, segment_name)
        segment_log = SegmentStimLog(block_name, phase_id, segment_name, record_start_epoch)
        segment_log.set_hardware_mapping(
            _hardware_mapping_payload(
                effective_hardware_config,
                stim_unit_by_electrode,
                stim_unit_to_dac,
                routing_mode=routing_mode,
                connect_settle_ms=connect_settle_ms,
                initial_connect=not _protocol_uses_plan(protocol),
            )
        )
        recording_settle_s = _recording_settle_s(system_config, protocol)
        if recording_settle_s > 0:
            audit.mark_event(
                "recording_settle_start",
                block_name,
                phase_id,
                segment_name,
                extra={"recording_settle_s": recording_settle_s},
            )
            time.sleep(recording_settle_s)
            audit.mark_event("recording_settle_done", block_name, phase_id, segment_name, extra={"recording_settle_s": recording_settle_s})
        if _protocol_uses_plan(protocol):
            audit.mark_event("stim_sequence_build_start", block_name, phase_id, segment_name, extra={"protocol_type": protocol_type})
            sequence = build_poisson_random_sequence(
                protocol,
                plan_rows,
                stim_unit_by_electrode,
                stim_unit_to_dac,
                effective_hardware_config,
            )
            audit.mark_event("stim_sequence_build_done", block_name, phase_id, segment_name, extra={"stim_count": len(stim_times)})
            sequence_start_epoch = time.time()
            sequence_start_offset = max(0.0, sequence_start_epoch - record_start_epoch)
            audit.mark_event(
                "stim_sequence_send",
                block_name,
                phase_id,
                segment_name,
                epoch_sec=sequence_start_epoch,
                extra={"stim_count": len(stim_times)},
            )
            sequence.send()
            audit.mark_event("stim_sequence_send_done", block_name, phase_id, segment_name, extra={"stim_count": len(stim_times)})
            logging.info("Stim sequence sent: block=%s protocol=%s pulses=%d", block_name, protocol_name, len(stim_times))
        else:
            audit.mark_event("stim_sequence_build_start", block_name, phase_id, segment_name, extra={"protocol_type": protocol_type})
            sequence = build_stim_sequence(
                protocol,
                filtered_group.get("name", ""),
                stim_unit_by_electrode=stim_unit_by_electrode,
                stim_unit_to_dac=stim_unit_to_dac,
                electrode_group_electrodes=[int(value) for value in filtered_group.get("electrodes", [])],
                system_config=effective_hardware_config,
            )
            audit.mark_event("stim_sequence_build_done", block_name, phase_id, segment_name, extra={"stim_count": len(stim_times)})
            sequence_start_epoch = time.time()
            sequence_start_offset = max(0.0, sequence_start_epoch - record_start_epoch)
            audit.mark_event(
                "stim_sequence_send",
                block_name,
                phase_id,
                segment_name,
                epoch_sec=sequence_start_epoch,
                extra={"stim_count": len(stim_times)},
            )
            sequence.send()
            audit.mark_event("stim_sequence_send_done", block_name, phase_id, segment_name, extra={"stim_count": len(stim_times)})
            logging.info("Stim sequence sent: block=%s protocol=%s pulses=%d", block_name, protocol_name, len(stim_times))
        elapsed_s = time.time() - record_start_epoch
        target_recording_s = duration_s + recording_settle_s
        if elapsed_s < target_recording_s:
            audit.mark_event(
                "recording_hold_start",
                block_name,
                phase_id,
                segment_name,
                extra={"remaining_s": round(target_recording_s - elapsed_s, 6), "target_recording_s": target_recording_s},
            )
            time.sleep(target_recording_s - elapsed_s)
            audit.mark_event("recording_hold_done", block_name, phase_id, segment_name, extra={"target_recording_s": target_recording_s})
        saving.stop_recording()
        saving.stop_file()
        audit.mark_event("recording_file_stop", block_name, phase_id, segment_name)
        logging.info("Recording stopped: block=%s phase=%s", block_name, phase_id)

    route_records = _hardware_route_records(
        [float(value) for value in stim_times],
        plan_rows,
        filtered_group,
        stim_unit_by_electrode,
        stim_unit_to_dac,
        connect_settle_ms,
        initial_connect=not _protocol_uses_plan(protocol),
        event_level_switch=(
            _protocol_requires_dynamic_site_switch(protocol)
            or protocol.get("type") == "electrode_pool_sequence"
        ),
        signal_dacs=_hardware_dac_config(effective_hardware_config)[0],
    )
    for index, stim_time in enumerate(stim_times, start=1):
        plan_row = plan_rows[index - 1] if plan_rows else {}
        epoch_sec = sequence_start_epoch + stim_time
        route_record = route_records[index - 1] if index <= len(route_records) else {}
        stim_extra = {
            "stim_index": index,
            "stim_time_sec": stim_time + sequence_start_offset,
            "plan_time_sec": stim_time,
            "amplitude_mv": plan_row.get("amplitude_mv", protocol.get("amplitude_mv", "")),
            "electrodes": _plan_row_electrodes_text(plan_row, electrode_group),
            "lambda_hz": plan_row.get("lambda_hz", ""),
            "firing_rate_hz": plan_row.get("firing_rate_hz", ""),
            "pulses_per_stimulus": plan_row.get("pulses_per_stimulus", protocol.get("pulses_per_burst", "")),
            "target_stim_units": route_record.get("target_stim_units", []),
            "active_dac_sources": route_record.get("active_dac_sources", []),
            "stim_unit_to_event_dac": route_record.get("stim_unit_to_event_dac", {}),
            "route_switch": route_record.get("route_switch", False),
            "connect_time_s": route_record.get("connect_time_s"),
            "settle_ms": route_record.get("settle_ms", 0.0),
        }
        route_record["record_time_s"] = round(float(stim_extra["stim_time_sec"]), 6)
        segment_log.add_route_record(route_record)
        segment_log.add_stim(epoch_sec, index, extra=stim_extra)
        audit.mark_event(
            "stim_send",
            block_name,
            phase_id,
            segment_name,
            epoch_sec=epoch_sec,
            extra=stim_extra,
        )
    segment_log.record_end_epoch = time.time()
    segment_log.save_txt(phase_dir / "stim_times.txt")
    segment_log.save_json(phase_dir / "segment_time_meta.json")
    audit.mark_event(
        "stim_log_written",
        block_name,
        phase_id,
        segment_name,
        extra={
            "pulse_count": len(stim_times),
            "stim_times_path": str(phase_dir / "stim_times.txt"),
            "segment_meta_path": str(phase_dir / "segment_time_meta.json"),
        },
    )
    logging.info("Stim logs written: block=%s phase=%s pulses=%d", block_name, phase_id, len(stim_times))


def _planned_electrodes(plan_rows: list[dict[str, Any]], electrode_group: dict[str, Any]) -> set[int]:
    if not plan_rows:
        return {int(item) for item in electrode_group.get("electrodes", [])}
    electrodes: set[int] = set()
    for row in plan_rows:
        if "electrodes" in row:
            values = row.get("electrodes")
            if isinstance(values, str):
                electrodes.update(int(float(token)) for token in values.replace(";", ",").split(",") if token.strip())
            elif isinstance(values, (list, tuple)):
                electrodes.update(int(value) for value in values)
        elif "electrode" in row:
            electrodes.add(int(row["electrode"]))
    return electrodes


def _planned_electrode_centers(plan_rows: list[dict[str, Any]], electrode_group: dict[str, Any]) -> dict[int, int]:
    lookup: dict[int, int] = {}
    fallback_center = _group_center_electrode(electrode_group)
    for row in plan_rows:
        center = row.get("center_electrode", fallback_center)
        try:
            center_int = int(center)
        except (TypeError, ValueError):
            continue
        for electrode in _planned_electrodes([row], electrode_group):
            lookup[int(electrode)] = center_int
    return lookup


def _plan_row_electrodes_text(plan_row: dict[str, Any], electrode_group: dict[str, Any]) -> str:
    if "electrodes" in plan_row:
        values = plan_row.get("electrodes")
        if isinstance(values, (list, tuple)):
            return ",".join(str(int(item)) for item in values)
        return str(values)
    if "electrode" in plan_row:
        return str(plan_row.get("electrode"))
    return ",".join(str(item) for item in electrode_group.get("electrodes", []))


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


def _recording_settle_s(system_config: dict[str, Any], protocol: dict[str, Any]) -> float:
    maxwell_cfg = system_config.get("maxwell", {}) if isinstance(system_config, dict) else {}
    protocol_cfg = protocol.get("random_electrode_plan", {}) if isinstance(protocol, dict) else {}
    raw_value = protocol_cfg.get("recording_settle_s", maxwell_cfg.get("recording_settle_s", 2.0))
    try:
        return max(0.0, float(raw_value))
    except (TypeError, ValueError):
        return 2.0


def _effective_electrode_group(protocol: dict[str, Any], electrode_group: dict[str, Any]) -> dict[str, Any]:
    fallback = [int(item) for item in electrode_group.get("electrodes", [])]
    if protocol.get("type") == "poisson_random_electrodes":
        candidate_electrodes = select_poisson_candidate_electrodes(protocol, fallback)
        source = "poisson_random_electrodes"
    elif protocol.get("type") == "electrode_pool_sequence":
        event_groups = electrode_pool_event_groups(protocol, fallback)
        candidate_electrodes = _unique_ints([electrode for group in event_groups for electrode in group]) or fallback
        source = "electrode_pool_sequence"
    elif _protocol_uses_plan(protocol):
        stim_count = len(get_stim_times_for_protocol(protocol, 24 * 60 * 60))
        event_groups = site_switch_event_groups_for_count(protocol, fallback, stim_count)
        candidate_electrodes = _unique_ints([electrode for group in event_groups for electrode in group]) or fallback
        source = "site_switch"
    else:
        return electrode_group
    effective = dict(electrode_group)
    effective["electrodes"] = candidate_electrodes
    effective["candidate_source"] = source
    return effective


def _validate_stimulation_unit_pool_size(protocol: dict[str, Any], electrode_group: dict[str, Any], *, context: str) -> None:
    electrodes = _unique_ints([int(item) for item in electrode_group.get("electrodes", [])])
    if len(electrodes) <= MAX_STIMULATION_UNITS_PER_ROUTE:
        return
    routing = "dynamic stimulation switching" if _protocol_uses_plan(protocol) else "static stimulation routing"
    raise RuntimeError(
        f"{context}: {routing} requested {len(electrodes)} electrodes, "
        f"but MaxOne supports at most {MAX_STIMULATION_UNITS_PER_ROUTE} stimulation units in one routed pool. "
        "Split the experiment into smaller blocks or reduce the event-group union."
    )


def _apply_electrode_replacements_to_plan_rows(
    plan_rows: list[dict[str, Any]],
    replacements: dict[int, int],
) -> list[dict[str, Any]]:
    if not replacements:
        return list(plan_rows)
    updated: list[dict[str, Any]] = []
    for row in plan_rows:
        next_row = dict(row)
        if "electrodes" in next_row:
            values = next_row.get("electrodes")
            if isinstance(values, str):
                electrodes = [int(float(token)) for token in values.replace(";", ",").split(",") if token.strip()]
                next_row["electrodes"] = ",".join(str(replacements.get(electrode, electrode)) for electrode in electrodes)
            elif isinstance(values, (list, tuple)):
                next_row["electrodes"] = [int(replacements.get(int(electrode), int(electrode))) for electrode in values]
        if "electrode" in next_row:
            electrode = int(next_row["electrode"])
            next_row["electrode"] = int(replacements.get(electrode, electrode))
        updated.append(next_row)
    return updated


def _rewrite_replaced_stim_plan_files(
    phase_dir: Path,
    plan_rows: list[dict[str, Any]],
    electrode_group: dict[str, Any],
    replacements: dict[int, int],
) -> None:
    if not replacements:
        return
    csv_path = phase_dir / "stim_plan.csv"
    json_path = phase_dir / "stim_plan.json"
    fieldnames = [
        "time_sec",
        "electrode",
        "electrodes",
        "firing_rate_hz",
        "lambda_hz",
        "amplitude_mv",
        "pulse_width_us",
        "pulses_per_stimulus",
        "center_electrode",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in plan_rows:
            csv_row = {key: row.get(key, "") for key in fieldnames}
            if isinstance(csv_row.get("electrodes"), (list, tuple)):
                csv_row["electrodes"] = ",".join(str(int(item)) for item in csv_row["electrodes"])
            writer.writerow(csv_row)

    payload: dict[str, Any] = {}
    if json_path.exists():
        try:
            loaded = json.loads(json_path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            payload = {}
    payload["candidate_electrodes"] = [int(item) for item in electrode_group.get("electrodes", [])]
    config = payload.get("config", {})
    if not isinstance(config, dict):
        config = {}
    config["stimulation_electrode_replacements"] = {str(source): int(target) for source, target in replacements.items()}
    payload["config"] = config
    payload["stimuli"] = plan_rows
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _replace_unconnectable_poisson_electrodes(
    cfg_path: Path,
    electrode_group: dict[str, Any],
    system_config: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[int, int], list[int]]:
    original_electrodes = [int(item) for item in electrode_group.get("electrodes", [])]
    rates, random_cfg = poisson_rates_for_electrodes(protocol, original_electrodes, restrict_to_fallback=False)
    floor = float(random_cfg.get("lambda_floor_hz", 0.001))
    return _replace_unconnectable_stimulation_electrodes(
        cfg_path,
        electrode_group,
                effective_hardware_config,
        protocol,
        rates=rates,
        floor=floor,
        max_radius=int(random_cfg.get("replacement_max_radius", 10) or 10),
    )


def _replace_unconnectable_stimulation_electrodes(
    cfg_path: Path,
    electrode_group: dict[str, Any],
    system_config: dict[str, Any],
    protocol: dict[str, Any] | None = None,
    *,
    rates: dict[int, float] | None = None,
    floor: float = 1.0,
    max_radius: int | None = None,
    electrode_center_lookup: dict[int, int] | None = None,
) -> tuple[dict[str, Any], dict[int, int], list[int]]:
    original_electrodes = _unique_ints([int(item) for item in electrode_group.get("electrodes", [])])
    rate_map = rates or {electrode: float(floor) for electrode in original_electrodes}
    maxwell_cfg = system_config.get("maxwell", {}) if isinstance(system_config, dict) else {}
    protocol_cfg = protocol.get("random_electrode_plan", {}) if isinstance(protocol, dict) else {}
    if max_radius is None:
        max_radius = int(protocol_cfg.get("replacement_max_radius", maxwell_cfg.get("replacement_max_radius", 10)) or 10)
    max_radius = max(2, int(max_radius))
    connected_electrodes, missing_electrodes, stim_unit_by_electrode = probe_stimulation_electrodes(
        cfg_path,
        electrode_group,
        system_config,
    )
    primary_electrodes, stim_unit_conflicts = _dedupe_by_stimulation_unit(
        connected_electrodes,
        stim_unit_by_electrode,
        rate_map,
        floor,
    )
    unresolved_targets = _unique_ints([*missing_electrodes, *stim_unit_conflicts])
    if not unresolved_targets:
        filtered = dict(electrode_group)
        filtered["electrodes"] = primary_electrodes
        return filtered, {}, []

    search_radii = _replacement_search_radii(max_radius)
    used = set(primary_electrodes)
    used_stim_units = {int(stim_unit_by_electrode[electrode]) for electrode in primary_electrodes if electrode in stim_unit_by_electrode}
    missing_set = set(unresolved_targets)
    cfg_electrodes = _cfg_recording_electrodes(cfg_path)
    center_electrode = _group_center_electrode(electrode_group)
    center_lookup = {int(key): int(value) for key, value in (electrode_center_lookup or {}).items()}
    center_pool = sorted(set(center_lookup.values()) | ({center_electrode} if center_electrode is not None else set()))
    if center_pool:
        neighbor_pool = sorted(
            (candidate for candidate in cfg_electrodes if candidate not in missing_set),
            key=lambda candidate: (
                min(_electrode_grid_distance(center, candidate) for center in center_pool),
                candidate,
            ),
        )
    else:
        neighbor_pool = sorted(
            {
                candidate
                for electrode in unresolved_targets
                for candidate in _electrode_neighbors(electrode, max_radius)
                if candidate not in missing_set and candidate in cfg_electrodes
            }
        )
    probe_group = dict(electrode_group)
    probe_group["electrodes"] = neighbor_pool
    if neighbor_pool:
        connectable_neighbors, _skipped_neighbors, neighbor_stim_units = probe_stimulation_electrodes(
            cfg_path,
            probe_group,
            system_config,
        )
        connectable_neighbor_set = set(connectable_neighbors)
    else:
        connectable_neighbor_set = set()
        neighbor_stim_units = {}

    replacements: dict[int, int] = {}
    unresolved: list[int] = []
    for electrode in unresolved_targets:
        replacement = None
        target_center = center_lookup.get(int(electrode), center_electrode)
        for radius in ([None] if target_center is not None else search_radii):
            if target_center is not None:
                radius_candidates = neighbor_pool
            else:
                radius_candidates = _electrode_neighbors(electrode, radius)
            ranked = sorted(
                (
                    candidate
                    for candidate in radius_candidates
                    if (
                        candidate in connectable_neighbor_set
                        and candidate not in used
                        and int(neighbor_stim_units.get(candidate, -1)) not in used_stim_units
                    )
                ),
                key=lambda candidate: (
                    _electrode_grid_distance(target_center if target_center is not None else electrode, candidate),
                    -float(rate_map.get(candidate, floor)),
                    candidate,
                ),
            )
            if ranked:
                replacement = ranked[0]
                break
        if replacement is None:
            unresolved.append(electrode)
            continue
        replacements[electrode] = replacement
        used.add(replacement)
        used_stim_units.add(int(neighbor_stim_units[replacement]))

    resolved_electrodes: list[int] = []
    for electrode in original_electrodes:
        if electrode in primary_electrodes:
            resolved_electrodes.append(electrode)
        elif electrode in replacements:
            resolved_electrodes.append(replacements[electrode])
    filtered = dict(electrode_group)
    filtered["electrodes"] = _unique_ints(resolved_electrodes)
    if not filtered["electrodes"]:
        return filtered, replacements, unresolved
    final_connected, _final_missing, final_stim_units = probe_stimulation_electrodes(
        cfg_path,
        filtered,
        system_config,
    )
    final_primary, final_conflicts = _dedupe_by_stimulation_unit(
        final_connected,
        final_stim_units,
        rate_map,
        floor,
    )
    if final_conflicts:
        unresolved.extend(final_conflicts)
        filtered["electrodes"] = final_primary
    return filtered, replacements, unresolved


def _group_center_electrode(electrode_group: dict[str, Any]) -> int | None:
    try:
        value = electrode_group.get("center_electrode")
        if value is None or str(value).strip() == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _dedupe_by_stimulation_unit(
    electrodes: list[int],
    stim_unit_by_electrode: dict[int, int],
    rates: dict[int, float],
    floor: float,
) -> tuple[list[int], list[int]]:
    best_by_unit: dict[int, int] = {}
    conflicts: list[int] = []
    for electrode in electrodes:
        stim_unit = int(stim_unit_by_electrode.get(electrode, -1))
        if stim_unit < 0:
            conflicts.append(electrode)
            continue
        current = best_by_unit.get(stim_unit)
        if current is None:
            best_by_unit[stim_unit] = electrode
            continue
        current_key = (float(rates.get(current, floor)), -current)
        candidate_key = (float(rates.get(electrode, floor)), -electrode)
        if candidate_key > current_key:
            conflicts.append(current)
            best_by_unit[stim_unit] = electrode
        else:
            conflicts.append(electrode)
    winners = set(best_by_unit.values())
    return [electrode for electrode in electrodes if electrode in winners], _unique_ints(conflicts)


def _replacement_search_radii(max_radius: int) -> list[int]:
    radii = [2]
    radius = 4
    while radius <= max_radius:
        radii.append(radius)
        radius += 2
    if radii[-1] != max_radius:
        radii.append(max_radius)
    return sorted(set(max(2, int(value)) for value in radii))


def _electrode_neighbors(electrode: int, radius: int) -> list[int]:
    maxwell_rows = 120
    maxwell_cols = 220
    radius = max(1, int(radius))
    row = int(electrode) // 220
    col = int(electrode) % 220
    candidates: list[int] = []
    for neighbor_row in range(max(0, row - radius), min(maxwell_rows - 1, row + radius) + 1):
        for neighbor_col in range(max(0, col - radius), min(maxwell_cols - 1, col + radius) + 1):
            candidate = neighbor_row * maxwell_cols + neighbor_col
            if candidate != int(electrode):
                candidates.append(candidate)
    return candidates


def _electrode_grid_distance(left: int, right: int) -> int:
    left_row, left_col = int(left) // 220, int(left) % 220
    right_row, right_col = int(right) // 220, int(right) % 220
    return abs(left_row - right_row) + abs(left_col - right_col)


def _unique_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        item = int(value)
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
