
from __future__ import annotations

import ast
import csv
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np


def _unique_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        item = int(value)
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _normalize_event_group(raw_group: Any) -> list[int]:
    if raw_group is None:
        return []
    if isinstance(raw_group, str):
        text = raw_group.strip()
        if not text:
            return []
        parsed: Any | None = None
        if text.startswith("[") or text.startswith("("):
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None
        if parsed is not None:
            return _normalize_event_group(parsed)
        return _unique_ints([int(value) for value in re.split(r"[\s,;]+", text.strip("[]()")) if str(value).strip()])
    if isinstance(raw_group, dict):
        for key in ("electrodes", "group", "values"):
            if key in raw_group:
                return _normalize_event_group(raw_group.get(key))
        return []
    if isinstance(raw_group, (list, tuple, set)):
        values: list[int] = []
        for value in raw_group:
            if isinstance(value, (list, tuple, set, dict)):
                values.extend(_normalize_event_group(value))
            elif isinstance(value, str) and (value.strip().startswith("[") or "," in value or ";" in value):
                values.extend(_normalize_event_group(value))
            else:
                values.append(int(value))
        return _unique_ints(values)
    return _unique_ints([int(raw_group)])


def build_poisson_random_plan(
    protocol: dict[str, Any],
    phase_dir: Path,
    duration_s: int,
    fallback_electrodes: list[int],
) -> list[dict[str, Any]]:
    rates, random_cfg = _poisson_rates_and_config(protocol, fallback_electrodes)
    candidate_electrodes = select_candidate_electrodes(
        rates,
        region_count=int(random_cfg.get("region_count", 32)),
        max_candidates=int(random_cfg.get("max_candidate_electrodes", 32)),
    )
    if not candidate_electrodes:
        raise ValueError("No candidate stimulation electrodes could be selected")

    plan_duration_s = min(float(random_cfg.get("duration_s", duration_s)), float(duration_s))
    rng = random.Random(int(random_cfg.get("random_seed", 42)))
    rows: list[dict[str, Any]] = []
    for electrode in candidate_electrodes:
        firing_rate = max(float(rates.get(electrode, 0.0)), 0.0)
        lambda_hz = choose_lambda_hz(firing_rate, random_cfg, rng)
        if lambda_hz <= 0:
            continue
        current = rng.expovariate(lambda_hz)
        while current <= plan_duration_s:
            rows.append(
                {
                    "time_sec": round(current, 6),
                    "electrode": int(electrode),
                    "firing_rate_hz": round(firing_rate, 6),
                    "lambda_hz": round(lambda_hz, 6),
                    "amplitude_mv": float(protocol.get("amplitude_mv", 150.0)),
                    "pulse_width_us": float(protocol.get("pulse_width_us", 300.0)),
                    "pulses_per_stimulus": 1,
                }
            )
            current += rng.expovariate(lambda_hz)

    rows.sort(key=lambda item: (item["time_sec"], item["electrode"]))
    rows, skipped_overlaps, min_interval_s = enforce_common_dac_spacing(
        rows,
        inter_phase_interval_us=float(protocol.get("inter_phase_interval_us", 0.0) or 0.0),
        duration_s=plan_duration_s,
    )
    if skipped_overlaps:
        random_cfg = dict(random_cfg)
        random_cfg["common_dac_skipped_overlaps"] = int(skipped_overlaps)
        random_cfg["common_dac_min_interval_sec"] = round(float(min_interval_s), 6)
    if not rows:
        raise ValueError(
            "Poisson random stimulation generated 0 pulses. "
            "Check spontaneous_data_path, lambda_scale/lambda_floor_hz, and stim phase duration."
        )
    save_plan(phase_dir, rows, candidate_electrodes, random_cfg)
    return rows


def enforce_common_dac_spacing(
    rows: list[dict[str, Any]],
    *,
    inter_phase_interval_us: float = 0.0,
    duration_s: float,
) -> tuple[list[dict[str, Any]], int, float]:
    filtered: list[dict[str, Any]] = []
    next_available_s = 0.0
    skipped = 0
    max_interval_s = 0.0
    for row in rows:
        interval_s = (2.0 * float(row.get("pulse_width_us", 300.0)) + max(0.0, float(inter_phase_interval_us))) / 1_000_000.0
        max_interval_s = max(max_interval_s, interval_s)
        time_s = float(row["time_sec"])
        if time_s + 1e-9 < next_available_s:
            skipped += 1
            continue
        if time_s > float(duration_s):
            skipped += 1
            continue
        item = dict(row)
        item["time_sec"] = round(time_s, 6)
        item["pulses_per_stimulus"] = 1
        filtered.append(item)
        next_available_s = time_s + interval_s
    return filtered, skipped, max_interval_s


def build_electrode_pool_sequence_plan(
    protocol: dict[str, Any],
    phase_dir: Path,
    duration_s: int,
    fallback_electrodes: list[int],
) -> list[dict[str, Any]]:
    pool_cfg = protocol.get("electrode_pool_sequence", {}) or {}
    pool = _unique_ints([int(electrode) for electrode in fallback_electrodes])
    if not pool:
        raise ValueError("Electrode pool sequence needs a non-empty site group")
    event_groups = electrode_pool_event_groups(protocol, pool)
    if not event_groups:
        raise ValueError("Electrode pool sequence generated 0 event groups")
    start_s = max(0.0, float(protocol.get("start_ms", 0.0)) / 1000.0)
    interval_s = max(0.0, float(pool_cfg.get("event_interval_ms", 1000.0)) / 1000.0)
    rows: list[dict[str, Any]] = []
    for index, electrodes in enumerate(event_groups):
        time_s = start_s + index * interval_s
        if time_s > float(duration_s):
            break
        targets = _unique_ints([int(value) for value in electrodes])
        if not targets:
            continue
        rows.append(
            {
                "time_sec": round(time_s, 6),
                "electrode": int(targets[0]),
                "electrodes": targets,
                "firing_rate_hz": "",
                "lambda_hz": "",
                "amplitude_mv": float(protocol.get("amplitude_mv", 150.0)),
                "pulse_width_us": float(protocol.get("pulse_width_us", 300.0)),
                "pulses_per_stimulus": 1,
            }
        )
    rows, skipped_overlaps, min_interval_s = enforce_common_dac_spacing(
        rows,
        inter_phase_interval_us=float(protocol.get("inter_phase_interval_us", 0.0) or 0.0),
        duration_s=float(duration_s),
    )
    output_cfg = dict(pool_cfg)
    output_cfg["pool_electrodes"] = pool
    if skipped_overlaps:
        output_cfg["common_dac_skipped_overlaps"] = int(skipped_overlaps)
        output_cfg["common_dac_min_interval_sec"] = round(float(min_interval_s), 6)
    if not rows:
        raise ValueError("Electrode pool sequence generated 0 pulses. Check start, interval, event count, and stim duration.")
    save_plan(phase_dir, rows, pool, output_cfg)
    return rows


def build_site_switch_plan(
    protocol: dict[str, Any],
    phase_dir: Path,
    duration_s: int,
    fallback_electrodes: list[int],
    stim_times_sec: list[float],
) -> list[dict[str, Any]]:
    switch_cfg = protocol.get("site_switch", {}) or {}
    pool = _unique_ints([int(electrode) for electrode in fallback_electrodes])
    if not pool:
        raise ValueError("Site switching needs a non-empty site group")
    valid_times = [float(value) for value in stim_times_sec if 0.0 <= float(value) <= float(duration_s)]
    event_groups = site_switch_event_groups_for_count(protocol, pool, len(valid_times))
    event_centers = site_switch_event_centers_for_count(protocol, pool, len(valid_times), event_groups)
    if not event_groups and valid_times:
        raise ValueError("Site switching generated 0 event groups")
    rows: list[dict[str, Any]] = []
    for index, time_s in enumerate(valid_times):
        targets = _unique_ints([int(value) for value in (event_groups[index] if index < len(event_groups) else [])])
        if not targets:
            continue
        center = event_centers[index] if index < len(event_centers) else None
        row = {
            "time_sec": round(time_s, 6),
            "electrode": int(targets[0]),
            "electrodes": targets,
            "firing_rate_hz": "",
            "lambda_hz": "",
            "amplitude_mv": float(protocol.get("amplitude_mv", 150.0)),
            "pulse_width_us": float(protocol.get("pulse_width_us", 300.0)),
            "pulses_per_stimulus": 1,
        }
        if center is not None:
            row["center_electrode"] = int(center)
        rows.append(
            row
        )
    rows, skipped_overlaps, min_interval_s = enforce_common_dac_spacing(
        rows,
        inter_phase_interval_us=float(protocol.get("inter_phase_interval_us", 0.0) or 0.0),
        duration_s=float(duration_s),
    )
    output_cfg = dict(switch_cfg)
    output_cfg["pool_electrodes"] = pool
    output_cfg["source_protocol_type"] = protocol.get("type", "")
    if skipped_overlaps:
        output_cfg["common_dac_skipped_overlaps"] = int(skipped_overlaps)
        output_cfg["common_dac_min_interval_sec"] = round(float(min_interval_s), 6)
    if not rows:
        raise ValueError("Site switching generated 0 pulses. Check protocol timing, site groups, and stim duration.")
    save_plan(phase_dir, rows, pool, output_cfg)
    return rows


def electrode_pool_event_groups(protocol: dict[str, Any], electrode_pool: list[int]) -> list[list[int]]:
    pool = _unique_ints([int(value) for value in electrode_pool])
    pool_cfg = protocol.get("electrode_pool_sequence", {}) or {}
    explicit = pool_cfg.get("event_groups") or []
    if explicit:
        base_groups: list[list[int]] = []
        for raw_group in explicit:
            values = _normalize_event_group(raw_group)
            if values:
                base_groups.append(values)
        return _expand_pool_event_groups(
            base_groups,
            mode=str(pool_cfg.get("selection_mode", "balanced_random_groups") or "balanced_random_groups"),
            repeats=max(1, int(pool_cfg.get("event_count", 1) or 1)),
            random_seed=int(pool_cfg.get("random_seed", protocol.get("random_seed", 42))),
        )
    if not pool:
        return []
    event_count = max(0, int(pool_cfg.get("event_count", 10)))
    per_event = max(1, min(int(pool_cfg.get("electrodes_per_event", 1)), len(pool)))
    mode = str(pool_cfg.get("selection_mode", "random") or "random").strip().lower()
    rng = random.Random(int(pool_cfg.get("random_seed", protocol.get("random_seed", 42))))
    groups: list[list[int]] = []
    for index in range(event_count):
        if mode == "all":
            selected = list(pool)
        elif mode == "cycle":
            selected = [pool[(index * per_event + offset) % len(pool)] for offset in range(per_event)]
        else:
            selected = rng.sample(pool, per_event)
        groups.append(_unique_ints(selected))
    return groups


def site_switch_event_groups_for_count(protocol: dict[str, Any], electrode_pool: list[int], event_count: int) -> list[list[int]]:
    pool = _unique_ints([int(value) for value in electrode_pool])
    target_count = max(0, int(event_count))
    if target_count <= 0:
        return []
    pulses_per_event = _site_switch_pulses_per_event(protocol)
    switch_count = int(math.ceil(target_count / max(pulses_per_event, 1)))
    switch_cfg = protocol.get("site_switch", {}) or {}
    explicit = switch_cfg.get("event_groups") or []
    if explicit:
        base_groups: list[list[int]] = []
        for raw_group in explicit:
            values = _normalize_event_group(raw_group)
            if values:
                base_groups.append(values)
        groups = _balanced_site_switch_groups(
            base_groups,
            mode=str(switch_cfg.get("selection_mode", "balanced_random_groups") or "balanced_random_groups"),
            target_count=switch_count,
            random_seed=int(switch_cfg.get("random_seed", protocol.get("random_seed", 42))),
        )
        return _expand_site_switch_groups_to_pulses(groups, target_count, pulses_per_event)
    if not pool:
        return []
    groups = _balanced_site_switch_groups(
        [pool],
        mode=str(switch_cfg.get("selection_mode", "balanced_random_groups") or "balanced_random_groups"),
        target_count=switch_count,
        random_seed=int(switch_cfg.get("random_seed", protocol.get("random_seed", 42))),
    )
    return _expand_site_switch_groups_to_pulses(groups, target_count, pulses_per_event)


def site_switch_event_centers_for_count(
    protocol: dict[str, Any],
    electrode_pool: list[int],
    event_count: int,
    event_groups: list[list[int]] | None = None,
) -> list[int | None]:
    switch_cfg = protocol.get("site_switch", {}) or {}
    explicit = switch_cfg.get("event_groups") or []
    if not explicit:
        return [None] * max(0, int(event_count))
    raw_centers = list(switch_cfg.get("event_group_centers") or [])
    center_by_group: dict[tuple[int, ...], int | None] = {}
    for index, raw_group in enumerate(explicit):
        values = tuple(_normalize_event_group(raw_group))
        if not values:
            continue
        center_by_group[values] = _normalize_event_center(raw_centers[index] if index < len(raw_centers) else None)
    groups = event_groups if event_groups is not None else site_switch_event_groups_for_count(protocol, electrode_pool, event_count)
    return [center_by_group.get(tuple(_normalize_event_group(group))) for group in groups]


def _normalize_event_center(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _site_switch_pulses_per_event(protocol: dict[str, Any]) -> int:
    protocol_type = str(protocol.get("type", ""))
    if protocol_type in {"individual_burst", "sequence_with_burst", "sequence_with_poisson_burst"}:
        return max(1, int(protocol.get("pulses_per_burst", 1) or 1))
    return 1


def _expand_site_switch_groups_to_pulses(groups: list[list[int]], target_count: int, pulses_per_event: int) -> list[list[int]]:
    if not groups or target_count <= 0:
        return []
    expanded: list[list[int]] = []
    repeat_count = max(1, int(pulses_per_event))
    for group in groups:
        for _pulse_index in range(repeat_count):
            expanded.append(list(group))
            if len(expanded) >= target_count:
                return expanded
    return _fit_site_switch_group_count(expanded, target_count)


def _balanced_site_switch_groups(
    base_groups: list[list[int]],
    *,
    mode: str,
    target_count: int,
    random_seed: int,
) -> list[list[int]]:
    groups = [_normalize_event_group(group) for group in base_groups if group]
    groups = [group for group in groups if group]
    target = max(0, int(target_count))
    if not groups or target <= 0:
        return []
    mode = str(mode or "balanced_random_groups").strip().lower()
    quotas = [target // len(groups)] * len(groups)
    for index in range(target % len(groups)):
        quotas[index] += 1
    if mode in {"balanced_random_groups", "random_groups", "random", "balanced_random"}:
        expanded = [
            list(group)
            for group, quota in zip(groups, quotas)
            for _repeat in range(quota)
        ]
        rng = random.Random(int(random_seed))
        rng.shuffle(expanded)
        return expanded
    ordered: list[list[int]] = []
    used = [0] * len(groups)
    while len(ordered) < target:
        progressed = False
        for index, group in enumerate(groups):
            if used[index] >= quotas[index]:
                continue
            ordered.append(list(group))
            used[index] += 1
            progressed = True
            if len(ordered) >= target:
                break
        if not progressed:
            break
    return ordered


def _fit_site_switch_group_count(groups: list[list[int]], target_count: int) -> list[list[int]]:
    if not groups or target_count <= 0:
        return []
    if len(groups) >= target_count:
        return [list(group) for group in groups[:target_count]]
    fitted = [list(group) for group in groups]
    index = 0
    while len(fitted) < target_count:
        fitted.append(list(groups[index % len(groups)]))
        index += 1
    return fitted


def _expand_pool_event_groups(base_groups: list[list[int]], *, mode: str, repeats: int, random_seed: int) -> list[list[int]]:
    groups = [_normalize_event_group(group) for group in base_groups if group]
    groups = [group for group in groups if group]
    if not groups:
        return []
    mode = str(mode or "balanced_random_groups").strip().lower()
    if mode in {"explicit", "as_list", "once"}:
        return [list(group) for group in groups]
    if mode in {"sequence_groups", "group_sequence", "cycle", "ordered", "sequential"}:
        return [list(group) for _repeat in range(max(1, int(repeats))) for group in groups]
    if mode in {"balanced_random_groups", "random_groups", "random", "balanced_random"}:
        expanded = [list(group) for group in groups for _repeat in range(max(1, int(repeats)))]
        rng = random.Random(int(random_seed))
        rng.shuffle(expanded)
        return expanded
    return [list(group) for _repeat in range(max(1, int(repeats))) for group in groups]


def select_poisson_candidate_electrodes(protocol: dict[str, Any], fallback_electrodes: list[int]) -> list[int]:
    rates, random_cfg = _poisson_rates_and_config(protocol, fallback_electrodes)
    candidate_electrodes = select_candidate_electrodes(
        rates,
        region_count=int(random_cfg.get("region_count", 32)),
        max_candidates=int(random_cfg.get("max_candidate_electrodes", 32)),
    )
    if not candidate_electrodes:
        raise ValueError("No candidate stimulation electrodes could be selected")
    return candidate_electrodes


def _poisson_rates_and_config(protocol: dict[str, Any], fallback_electrodes: list[int]) -> tuple[dict[int, float], dict[str, Any]]:
    return poisson_rates_for_electrodes(protocol, fallback_electrodes, restrict_to_fallback=True)


def poisson_rates_for_electrodes(
    protocol: dict[str, Any],
    fallback_electrodes: list[int] | None = None,
    *,
    restrict_to_fallback: bool = True,
) -> tuple[dict[int, float], dict[str, Any]]:
    random_cfg = protocol.get("random_electrode_plan", {})
    raw_source_path = str(random_cfg.get("spontaneous_data_path", "") or "").strip()
    source_path = _resolve_rate_source_path(raw_source_path)
    fallback = [int(electrode) for electrode in (fallback_electrodes or [])]

    if source_path.exists():
        rates = load_spontaneous_rates(source_path)
        if fallback and restrict_to_fallback:
            floor = float(random_cfg.get("lambda_floor_hz", 0.001))
            rates = {electrode: float(rates.get(electrode, floor)) for electrode in fallback}
        elif fallback:
            floor = float(random_cfg.get("lambda_floor_hz", 0.001))
            rates = {int(electrode): float(rate) for electrode, rate in rates.items()}
            for electrode in fallback:
                rates.setdefault(electrode, floor)
    elif raw_source_path:
        raise FileNotFoundError(
            f"Poisson spontaneous rate source not found: {raw_source_path}. "
            "Regenerate the package so config uses config/pipeline_rate_sources/*.npz, "
            "or copy the rate source to the configured path."
        )
    else:
        rates = {electrode: float(random_cfg.get("lambda_floor_hz", 0.001)) for electrode in fallback}
    return rates, random_cfg


def _resolve_rate_source_path(path_text: str) -> Path:
    source_path = Path(path_text)
    if not path_text:
        return source_path
    if source_path.exists():
        return source_path
    candidates = []
    if not source_path.is_absolute():
        candidates.append(Path.cwd() / source_path)
        candidates.append(Path(__file__).resolve().parents[1] / source_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return source_path


def load_spontaneous_rates(path: Path) -> dict[int, float]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        return _load_rate_table(path)
    if suffix == ".npz":
        return _load_npz_rates(path)
    raise ValueError(f"Unsupported spontaneous data format: {path}")


def _load_rate_table(path: Path) -> dict[int, float]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", newline="", encoding="utf-8") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        if "," not in sample and "\t" not in sample:
            delimiter = None
        if delimiter is None:
            rows = []
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                tokens = line.replace(",", " ").split()
                if len(tokens) >= 2:
                    rows.append({"electrode": tokens[0], "firing_rate_hz": tokens[1]})
        else:
            rows = list(csv.DictReader(handle, delimiter=delimiter))

    rates: dict[int, float] = {}
    for row in rows:
        electrode_key = _first_existing(row, ["electrode", "electrode_id", "channel", "channel_id"])
        rate_key = _first_existing(row, ["firing_rate_hz", "rate_hz", "rate", "spikes_per_sec"])
        if electrode_key is None or rate_key is None:
            raise ValueError("Rate table needs electrode and firing_rate_hz columns")
        rates[int(float(row[electrode_key]))] = float(row[rate_key])
    return rates


def _load_npz_rates(path: Path) -> dict[int, float]:
    data = np.load(path, allow_pickle=True)
    if "electrodes" in data and "rates_hz" in data:
        return {int(e): float(r) for e, r in zip(data["electrodes"], data["rates_hz"])}
    if "electrodes" in data and "firing_rate_hz" in data:
        return {int(e): float(r) for e, r in zip(data["electrodes"], data["firing_rate_hz"])}
    if {"spike_times", "spike_electrodes", "duration_s"}.issubset(set(data.files)):
        duration_s = float(np.asarray(data["duration_s"]).reshape(-1)[0])
        counts: dict[int, int] = {}
        for electrode in data["spike_electrodes"]:
            counts[int(electrode)] = counts.get(int(electrode), 0) + 1
        return {electrode: count / max(duration_s, 1e-9) for electrode, count in counts.items()}
    raise ValueError("NPZ needs electrodes+rates_hz or spike_times+spike_electrodes+duration_s")


def _first_existing(row: dict[str, Any], candidates: list[str]) -> str | None:
    lowered = {key.strip().lstrip("\ufeff").lower(): key for key in row}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def select_candidate_electrodes(rates: dict[int, float], region_count: int, max_candidates: int) -> list[int]:
    if region_count <= 0:
        region_count = 32
    sorted_items = sorted(rates.items(), key=lambda item: item[0])
    if not sorted_items:
        return []
    if len(sorted_items) <= max(1, min(max_candidates, 32)) and int(region_count) >= len(sorted_items):
        return [electrode for electrode, _rate in sorted_items]

    candidates: list[int] = []
    for region_start in range(0, len(sorted_items), region_count):
        region = sorted_items[region_start:region_start + region_count]
        best_electrode, _rate = max(region, key=lambda item: item[1])
        candidates.append(best_electrode)
    return candidates[:max(1, min(max_candidates, 32))]


def choose_lambda_hz(firing_rate_hz: float, cfg: dict[str, Any], rng: random.Random) -> float:
    mode = str(cfg.get("lambda_mode", "scale"))
    floor = float(cfg.get("lambda_floor_hz", 0.001))
    scale = float(cfg.get("lambda_scale", 1.0))
    base = max(float(firing_rate_hz), 0.0)
    if mode in {"scale", "equal", "greater", "less"}:
        if mode == "equal":
            scale = 1.0
        elif mode == "greater":
            scale = max(scale, 1.0)
        elif mode == "less":
            scale = 1.0 / max(scale, 1e-9)
        value = base * scale
    elif mode in {"normal", "gaussian"}:
        mean = max(float(cfg.get("lambda_mean_hz", floor)), 0.0)
        if mode == "gaussian" and "lambda_std_hz" not in cfg:
            sigma = max(base * float(cfg.get("lambda_gaussian_cv", 0.25)), floor)
        else:
            sigma = max(float(cfg.get("lambda_std_hz", floor)), 0.0)
        value = rng.gauss(mean, sigma)
    else:
        raise ValueError(f"Unknown lambda_mode: {mode}")
    return max(float(value), floor)


def save_plan(phase_dir: Path, rows: list[dict[str, Any]], candidate_electrodes: list[int], cfg: dict[str, Any]) -> None:
    phase_dir.mkdir(parents=True, exist_ok=True)
    csv_path = phase_dir / "stim_plan.csv"
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
        for row in rows:
            csv_row = {key: row.get(key, "") for key in fieldnames}
            if isinstance(csv_row.get("electrodes"), (list, tuple)):
                csv_row["electrodes"] = ",".join(str(int(value)) for value in csv_row["electrodes"])
            writer.writerow(csv_row)
    (phase_dir / "stim_plan.json").write_text(
        json.dumps(
            {
                "candidate_electrodes": candidate_electrodes,
                "random_config": cfg,
                "stimuli": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
