from __future__ import annotations

import logging
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

    if dry_run:
        logging.info("Dry run enabled: hardware calls skipped")
    else:
        initialize_maxlab()

    try:
        for block in blocks:
            block_name = block["name"]
            block_dir = raw_root / block_name
            block_dir.mkdir(parents=True, exist_ok=True)
            (block_dir / "block_meta.yaml").write_text(
                f"name: {block_name}\nelectrode_group: {block['electrode_group']}\nprotocol: {block['protocol']}\n",
                encoding="utf-8",
            )
            protocol = protocols[block["protocol"]]
            electrode_group = groups[block["electrode_group"]]
            electrode_group = _effective_electrode_group(protocol, electrode_group)
            if not dry_run and protocol.get("type") != "poisson_random_electrodes":
                configure_experiment_array(cfg_path, electrode_group, system_config)

            for phase in block.get("phases", []):
                phase_id = phase["id"]
                phase_dir = block_dir / phase_id
                phase_dir.mkdir(parents=True, exist_ok=True)
                duration_s = int(phase.get("duration_s", 300))
                segment_name = f"{recording_prefix}_{block_name}_{phase_id}"
                segment_start = time.time()
                audit.mark_event("record_start", block_name, phase_id, segment_name, epoch_sec=segment_start)

                if phase_id == "02_stim":
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
                    saving = create_experiment_saving(phase_dir, segment_name)
                    time.sleep(duration_s)
                    saving.stop_recording()
                    saving.stop_file()

                audit.mark_event("record_end", block_name, phase_id, segment_name)
    finally:
        audit.mark_event("experiment_end", "", "", "")
        audit.save()


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
    if protocol.get("type") == "poisson_random_electrodes":
        filtered_group = dict(electrode_group)
        replacements: dict[int, int] = {}
        unresolved_electrodes: list[int] = []
        if not dry_run:
            filtered_group, replacements, unresolved_electrodes = _replace_unconnectable_poisson_electrodes(
                cfg_path,
                electrode_group,
                system_config,
                protocol,
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
    sequence_start_epoch = segment_start
    sequence_start_offset = 0.0
    if dry_run:
        segment_log = SegmentStimLog(block_name, phase_id, segment_name, segment_start)
        sequence_start_offset = _recording_settle_s(system_config, protocol)
        sequence_start_epoch = segment_start + sequence_start_offset
        logging.info("Dry run: %s %s pulses=%d", block_name, protocol.get("name"), len(stim_times))
    else:
        if protocol.get("type") == "poisson_random_electrodes":
            _array, stim_unit_by_electrode = configure_poisson_experiment_array(
                cfg_path,
                filtered_group,
                system_config,
            )
        saving = create_experiment_saving(phase_dir, segment_name)
        record_start_epoch = time.time()
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
        if protocol.get("type") == "poisson_random_electrodes":
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
            build_poisson_random_sequence(protocol, plan_rows, stim_unit_by_electrode).send()
        else:
            sequence = build_stim_sequence(protocol, electrode_group.get("name", ""))
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
        elapsed_s = time.time() - record_start_epoch
        target_recording_s = duration_s + recording_settle_s
        if elapsed_s < target_recording_s:
            time.sleep(target_recording_s - elapsed_s)
        saving.stop_recording()
        saving.stop_file()

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
