from __future__ import annotations

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
    create_experiment_saving,
    get_stim_times_for_protocol,
    initialize_maxlab,
    probe_stimulation_electrodes,
)
from python.random_stim_plan import build_poisson_random_plan
from python.random_stim_plan import poisson_rates_for_electrodes
from python.random_stim_plan import select_poisson_candidate_electrodes
from python.utils.time_log import ExternalTimeLog, SegmentStimLog


def run_experiment(system_config: dict[str, Any], stimulation_config: dict[str, Any], run_dir: Path, dry_run: bool = False) -> None:
    raw_root = run_dir / "raw_data"
    raw_root.mkdir(parents=True, exist_ok=True)
    audit = ExternalTimeLog(run_dir=run_dir)
    audit.mark_event("experiment_start", "", "", "")

    groups = {item["name"]: item for item in stimulation_config.get("electrode_groups", [])}
    protocols = {item["name"]: item for item in stimulation_config.get("protocols", [])}
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
            logging.info(
                "Block start: %s protocol=%s group=%s record_only=%s",
                block_name,
                block.get("protocol", ""),
                block.get("electrode_group", ""),
                is_record_only,
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
                },
            )
            if protocol and electrode_group:
                electrode_group = _effective_electrode_group(protocol, electrode_group)
                if not dry_run and protocol.get("type") != "poisson_random_electrodes" and not is_record_only:
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
                    configure_experiment_array(cfg_path, electrode_group, system_config)
                    audit.mark_event("array_configure_done", block_name, "", "")
                    logging.info("Array configured: block=%s electrodes=%d", block_name, len(electrode_group.get("electrodes", [])))

            for phase in block.get("phases", []):
                phase_id = phase["id"]
                phase_dir = block_dir / phase_id
                phase_dir.mkdir(parents=True, exist_ok=True)
                duration_s = int(phase.get("duration_s", 300))
                segment_name = f"{recording_prefix}_{block_name}_{phase_id}"
                segment_start = time.time()
                logging.info("Phase start: block=%s phase=%s duration_s=%d segment=%s", block_name, phase_id, duration_s, segment_name)
                audit.mark_event("record_start", block_name, phase_id, segment_name, epoch_sec=segment_start)

                if phase_id == "02_stim" and not _phase_is_record_only(phase):
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


def _block_has_record_only_stim(block: dict[str, Any]) -> bool:
    for phase in block.get("phases", []) or []:
        if str(phase.get("id", "")) == "02_stim":
            return _phase_is_record_only(phase)
    return False


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
        if _block_has_record_only_stim(block):
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
) -> None:
    protocol_type = str(protocol.get("type", ""))
    protocol_name = str(protocol.get("name", ""))
    source_group_name = str(electrode_group.get("name", ""))
    source_electrode_count = len(electrode_group.get("electrodes", []))
    logging.info(
        "Stim phase prepare: block=%s protocol=%s type=%s group=%s electrodes=%d duration_s=%d",
        block_name,
        protocol_name,
        protocol_type,
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
            "electrode_group": source_group_name,
            "electrode_count": source_electrode_count,
            "duration_s": duration_s,
        },
    )
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
    else:
        plan_rows = []
        stim_times = get_stim_times_for_protocol(protocol, duration_s)
    planned_electrodes = _planned_electrodes(plan_rows, electrode_group)
    audit.mark_event(
        "stim_plan_ready",
        block_name,
        phase_id,
        segment_name,
        extra={
            "stim_count": len(stim_times),
            "planned_electrode_count": len(planned_electrodes),
            "first_stim_time_sec": min(stim_times) if stim_times else "",
            "last_stim_time_sec": max(stim_times) if stim_times else "",
        },
    )
    logging.info(
        "Stim plan ready: block=%s protocol=%s pulses=%d electrodes=%d first=%s last=%s",
        block_name,
        protocol_name,
        len(stim_times),
        len(planned_electrodes),
        min(stim_times) if stim_times else "",
        max(stim_times) if stim_times else "",
    )
    sequence_start_epoch = segment_start
    sequence_start_offset = 0.0
    if dry_run:
        segment_log = SegmentStimLog(block_name, phase_id, segment_name, segment_start)
        sequence_start_offset = _recording_settle_s(system_config, protocol)
        sequence_start_epoch = segment_start + sequence_start_offset
        logging.info("Dry run: %s %s pulses=%d", block_name, protocol.get("name"), len(stim_times))
        audit.mark_event("stim_dry_run_complete", block_name, phase_id, segment_name, extra={"stim_count": len(stim_times)})
    else:
        if protocol.get("type") == "poisson_random_electrodes":
            audit.mark_event(
                "array_configure_start",
                block_name,
                phase_id,
                segment_name,
                extra={
                    "protocol": protocol_name,
                    "protocol_type": protocol_type,
                    "electrode_count": len(filtered_group.get("electrodes", [])),
                    "initial_connect": False,
                },
            )
            _array, stim_unit_by_electrode = configure_poisson_experiment_array(
                cfg_path,
                filtered_group,
                system_config,
            )
            audit.mark_event(
                "array_configure_done",
                block_name,
                phase_id,
                segment_name,
                extra={"stim_unit_count": len(set(stim_unit_by_electrode.values()))},
            )
            logging.info("Poisson array configured: block=%s stim_units=%d", block_name, len(set(stim_unit_by_electrode.values())))
        audit.mark_event("recording_file_start", block_name, phase_id, segment_name, extra={"phase_mode": "stimulation"})
        saving = create_experiment_saving(phase_dir, segment_name)
        record_start_epoch = time.time()
        audit.mark_event("recording_file_started", block_name, phase_id, segment_name, epoch_sec=record_start_epoch)
        logging.info("Recording started: block=%s phase=%s segment=%s", block_name, phase_id, segment_name)
        segment_log = SegmentStimLog(block_name, phase_id, segment_name, record_start_epoch)
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
        if protocol.get("type") == "poisson_random_electrodes":
            audit.mark_event("stim_sequence_build_start", block_name, phase_id, segment_name, extra={"protocol_type": protocol_type})
            sequence = build_poisson_random_sequence(protocol, plan_rows, stim_unit_by_electrode)
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
            sequence = build_stim_sequence(protocol, electrode_group.get("name", ""))
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

    for index, stim_time in enumerate(stim_times, start=1):
        plan_row = plan_rows[index - 1] if plan_rows else {}
        epoch_sec = sequence_start_epoch + stim_time
        segment_log.add_stim(epoch_sec, index)
        audit.mark_event(
            "stim_send",
            block_name,
            phase_id,
            segment_name,
            epoch_sec=epoch_sec,
            extra={
                "stim_index": index,
                "stim_time_sec": stim_time + sequence_start_offset,
                "plan_time_sec": stim_time,
                "amplitude_mv": plan_row.get("amplitude_mv", protocol.get("amplitude_mv", "")),
                "electrodes": str(plan_row.get("electrode", ",".join(str(item) for item in electrode_group.get("electrodes", [])))),
                "lambda_hz": plan_row.get("lambda_hz", ""),
                "firing_rate_hz": plan_row.get("firing_rate_hz", ""),
                "pulses_per_stimulus": plan_row.get("pulses_per_stimulus", protocol.get("pulses_per_burst", "")),
            },
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


def _recording_settle_s(system_config: dict[str, Any], protocol: dict[str, Any]) -> float:
    maxwell_cfg = system_config.get("maxwell", {}) if isinstance(system_config, dict) else {}
    protocol_cfg = protocol.get("random_electrode_plan", {}) if isinstance(protocol, dict) else {}
    raw_value = protocol_cfg.get("recording_settle_s", maxwell_cfg.get("recording_settle_s", 2.0))
    try:
        return max(0.0, float(raw_value))
    except (TypeError, ValueError):
        return 2.0


def _effective_electrode_group(protocol: dict[str, Any], electrode_group: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("type") != "poisson_random_electrodes":
        return electrode_group
    fallback = [int(item) for item in electrode_group.get("electrodes", [])]
    candidate_electrodes = select_poisson_candidate_electrodes(protocol, fallback)
    effective = dict(electrode_group)
    effective["electrodes"] = candidate_electrodes
    effective["candidate_source"] = "poisson_random_electrodes"
    return effective


def _replace_unconnectable_poisson_electrodes(
    cfg_path: Path,
    electrode_group: dict[str, Any],
    system_config: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[int, int], list[int]]:
    original_electrodes = [int(item) for item in electrode_group.get("electrodes", [])]
    rates, random_cfg = poisson_rates_for_electrodes(protocol, original_electrodes, restrict_to_fallback=False)
    floor = float(random_cfg.get("lambda_floor_hz", 0.001))
    connected_electrodes, missing_electrodes, stim_unit_by_electrode = probe_stimulation_electrodes(
        cfg_path,
        electrode_group,
        system_config,
    )
    primary_electrodes, stim_unit_conflicts = _dedupe_by_stimulation_unit(
        connected_electrodes,
        stim_unit_by_electrode,
        rates,
        floor,
    )
    unresolved_targets = _unique_ints([*missing_electrodes, *stim_unit_conflicts])
    if not unresolved_targets:
        filtered = dict(electrode_group)
        filtered["electrodes"] = primary_electrodes
        return filtered, {}, []

    max_radius = max(2, int(random_cfg.get("replacement_max_radius", 10) or 10))
    search_radii = _replacement_search_radii(max_radius)
    used = set(primary_electrodes)
    used_stim_units = {int(stim_unit_by_electrode[electrode]) for electrode in primary_electrodes if electrode in stim_unit_by_electrode}
    missing_set = set(unresolved_targets)
    neighbor_pool = sorted(
        {
            candidate
            for electrode in unresolved_targets
            for candidate in _electrode_neighbors(electrode, max_radius)
            if candidate not in missing_set
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
        for radius in search_radii:
            ranked = sorted(
                (
                    candidate
                    for candidate in _electrode_neighbors(electrode, radius)
                    if (
                        candidate in connectable_neighbor_set
                        and candidate not in used
                        and int(neighbor_stim_units.get(candidate, -1)) not in used_stim_units
                    )
                ),
                key=lambda candidate: (
                    -float(rates.get(candidate, floor)),
                    _electrode_grid_distance(electrode, candidate),
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
    final_connected, _final_missing, final_stim_units = probe_stimulation_electrodes(
        cfg_path,
        filtered,
        system_config,
    )
    final_primary, final_conflicts = _dedupe_by_stimulation_unit(
        final_connected,
        final_stim_units,
        rates,
        floor,
    )
    if final_conflicts:
        unresolved.extend(final_conflicts)
        filtered["electrodes"] = final_primary
    return filtered, replacements, unresolved


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
