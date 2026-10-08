from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import math
import random
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROTOCOL_TYPES = (
    "single_pulse",
    "individual_burst",
    "sequence_with_burst",
    "random",
    # Kept for loading older generated packages. New protocols use `random`
    # with an explicit inter-burst distribution.
    "sequence_with_poisson_burst",
    "custom_sequence",
    "poisson_random_electrodes",
    "electrode_pool_sequence",
)
PROTOCOL_TYPE_GROUPS = (
    ("Basic", ("single_pulse",)),
    ("Burst", ("individual_burst", "sequence_with_burst", "random")),
    ("Data-driven", ("electrode_pool_sequence",)),
    ("Custom", ("custom_sequence",)),
)
PROTOCOL_TYPE_LABELS = {
    "single_pulse": "Single pulse",
    "individual_burst": "Individual burst",
    "sequence_with_burst": "Sequence burst",
    "random": "Random burst",
    "sequence_with_poisson_burst": "Legacy: Poisson burst sequence",
    "poisson_random_electrodes": "Legacy: Poisson random electrodes",
    "electrode_pool_sequence": "Pulse switch electrode pool",
    "custom_sequence": "Custom sequence",
}
PHASES = ("01_pre_spont", "02_stim", "03_post_spont")
MAX_DEFAULT_NAME_LENGTH = 20
# MaxOne supports at most 32 stimulation units in one routed stimulation
# array. Keep this available to the GUI-side package validation as well as the
# generated runtime templates.
MAX_STIMULATION_UNITS_PER_ROUTE = 32
_TRAILING_NUMBER_SUFFIX_RE = re.compile(r"(?:_\d+)+$")
_PROTOCOL_DEFAULT_NAMES = {
    "single_pulse": "single_pulse",
    "individual_burst": "ind_burst",
    "sequence_with_burst": "seq_burst",
    "random": "random",
    "sequence_with_poisson_burst": "poisson_burst",
    "custom_sequence": "custom_seq",
    "poisson_random_electrodes": "poisson_rand",
    "electrode_pool_sequence": "pool_switch",
}


def short_default_name(
    value: str,
    *,
    fallback: str = "item",
    max_len: int = MAX_DEFAULT_NAME_LENGTH,
    strip_numeric_suffix: bool = False,
) -> str:
    limit = max(1, int(max_len or MAX_DEFAULT_NAME_LENGTH))
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "")).strip("_").lower()
    cleaned = re.sub(r"_+", "_", cleaned)
    if strip_numeric_suffix:
        cleaned = _TRAILING_NUMBER_SUFFIX_RE.sub("", cleaned).strip("_")
    if not cleaned:
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(fallback or "item")).strip("_").lower() or "item"
    if len(cleaned) > limit:
        cleaned = cleaned[:limit].rstrip("_")
    return cleaned or "item"[:limit]


def unique_short_name(
    base: str,
    existing_names: set[str] | list[str] | tuple[str, ...],
    *,
    fallback: str = "item",
    max_len: int = MAX_DEFAULT_NAME_LENGTH,
) -> str:
    limit = max(1, int(max_len or MAX_DEFAULT_NAME_LENGTH))
    cleaned = short_default_name(base, fallback=fallback, max_len=limit, strip_numeric_suffix=True)
    existing_keys = {str(name or "").strip().lower() for name in existing_names}
    if cleaned.lower() not in existing_keys:
        return cleaned
    index = 2
    while True:
        suffix = f"_{index}"
        stem = short_default_name(
            cleaned,
            fallback=fallback,
            max_len=max(1, limit - len(suffix)),
            strip_numeric_suffix=False,
        )
        candidate = f"{stem}{suffix}"
        if candidate.lower() not in existing_keys:
            return candidate
        index += 1


def _default_maxwell_cfg_path() -> str:
    now = dt.datetime.now()
    return f"/home/maxwell/configs/{now:%y%m%d}/{now:%Hh%Mm%Ss}.cfg"


@dataclass
class ExperimentInfo:
    name: str = "maxwell_experiment"
    culture_id: str = ""
    div: str = ""
    date: str = field(default_factory=lambda: dt.date.today().isoformat())
    recording_prefix: str = "recording"
    scientific_question: str = ""
    closed_loop_logic: str = ""
    expected_output: str = ""
    cfg_path: str = field(default_factory=_default_maxwell_cfg_path)
    data_root: str = "./data"
    device: str = "maxone"
    event_threshold: float = 8.5
    amplifier_gain: int = 512
    recording_settle_s: float = 2.0
    cpp_runner: str = "closed_loop_runner"
    spike_step: int = 10000
    max_stims: int = 10
    sequence_name: str = "spike_10k_closed_loop"


@dataclass
class ElectrodeGroup:
    name: str
    electrodes: list[int]
    center_electrode: int | None = None
    multi_electrode: bool = False
    electrode_count: int = 1
    site_switch_enabled: bool = False
    pool_event_count: int = 10
    pool_event_interval_ms: float = 1000.0
    pool_electrodes_per_event: int = 1
    pool_selection_mode: str = "balanced_random_groups"
    pool_event_groups: list[list[int]] = field(default_factory=list)

    def to_yaml(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "electrodes": self.electrodes}
        if self.center_electrode is not None:
            data["center_electrode"] = int(self.center_electrode)
        data["multi_electrode"] = bool(self.multi_electrode)
        data["electrode_count"] = max(1, int(self.electrode_count))
        if self.site_switch_enabled:
            data["site_switch"] = {
                "enabled": True,
                "selection_mode": self.pool_selection_mode,
            }
            if self.pool_event_groups:
                data["site_switch"]["event_groups"] = self.pool_event_groups
        return data


@dataclass
class StimulusProtocol:
    name: str
    type: str = "single_pulse"
    amplitude_mv: float = 150.0
    pulse_width_us: float = 200.0
    inter_phase_interval_us: float = 0.0
    pulse_frequency_hz: float = 20.0
    pulses_per_burst: int = 5
    interpulse_interval_ms: float = 0.0
    randomize_burst_pulse_intervals: bool = False
    burst_pulse_interval_min_ms: float = 10.0
    burst_pulse_interval_max_ms: float = 100.0
    burst_count: int = 3
    burst_frequency_hz: float = 5.0
    burst_interval_ms: float = 200.0
    start_ms: float = 1500.0
    channel: int = 0
    custom_points: list[dict[str, float]] = field(default_factory=list)
    spontaneous_data_path: str = ""
    candidate_source: str = "spontaneous_data"
    region_count: int = 32
    max_candidate_electrodes: int = 32
    poisson_duration_s: float = 300.0
    lambda_mode: str = "scale"
    lambda_scale: float = 1.0
    lambda_floor_hz: float = 0.001
    lambda_mean_hz: float = 1.0
    lambda_std_hz: float = 0.25
    random_seed: int = 42
    random_seed_mode: str = "auto_on_save"
    connect_settle_ms: float = 3.0
    signal_dacs: list[int] = field(default_factory=lambda: [0, 1])
    neutral_dac: int = 2
    sync_dual_dac: bool = True
    poisson_candidate_electrodes: list[int] = field(default_factory=list)
    site_switch_enabled: bool = False
    pool_event_count: int = 10
    pool_event_interval_ms: float = 1000.0
    pool_electrodes_per_event: int = 1
    pool_selection_mode: str = "balanced_random_groups"
    pool_event_groups: list[list[int]] = field(default_factory=list)
    pool_event_group_centers: list[int | None] = field(default_factory=list)
    pool_event_group_counts: list[int] = field(default_factory=list)
    random_distribution: str = "poisson"
    random_lambda_hz: float = 5.0
    random_interval_min_ms: float = 100.0
    random_interval_max_ms: float = 1000.0
    random_duration_s: float = 60.0
    scan_mode: str = "off"
    scan_local_electrodes: int = 1
    scan_band_width: str = "full"
    scan_selection_mode: str = "random"
    notes: str = ""

    def to_yaml(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "amplitude_mv": self.amplitude_mv,
            "pulse_width_us": self.pulse_width_us,
            "inter_phase_interval_us": 0.0,
            "execution_mode": "d21_host_per_pulse",
            "tail_wait_sec": 0.5,
            "hardware_dac": {
                "signal_dacs": [int(value) for value in self.signal_dacs] or [0, 1],
                "neutral_dac": int(self.neutral_dac),
                "sync_dual_dac": bool(self.sync_dual_dac),
                "allocation": "round_robin_electrode_order",
            },
        }
        if str(self.spontaneous_data_path or "").strip():
            data["stimulation_rate_filter"] = {
                "source_path": str(self.spontaneous_data_path),
                "threshold_mode": "median",
            }
        if self.type != "poisson_random_electrodes":
            data.update(
                {
                    "pulse_frequency_hz": self.pulse_frequency_hz,
                    "pulses_per_burst": self.pulses_per_burst,
                    "randomize_burst_pulse_intervals": bool(self.randomize_burst_pulse_intervals),
                    "burst_pulse_interval_min_ms": self.burst_pulse_interval_min_ms,
                    "burst_pulse_interval_max_ms": self.burst_pulse_interval_max_ms,
                    "burst_count": self.burst_count,
                    "burst_frequency_hz": self.burst_frequency_hz,
                    "start_ms": self.start_ms,
                    "channel": self.channel,
                    "random_seed": self.random_seed,
                }
            )
        if self.type == "random":
            data["random"] = {
                "distribution": str(self.random_distribution or "poisson").strip().lower(),
                "lambda_hz": float(self.random_lambda_hz),
                "interval_min_ms": float(self.random_interval_min_ms),
                "interval_max_ms": float(self.random_interval_max_ms),
                "duration_s": float(self.random_duration_s),
                "random_seed": int(self.random_seed),
            }
            data["duration_s"] = float(self.random_duration_s)
            data["total_duration_s"] = protocol_total_duration_s(self)
        if self.site_switch_enabled and self.type != "poisson_random_electrodes":
            data["site_switch"] = {
                "enabled": True,
                "selection_mode": self.pool_selection_mode,
                "random_seed": self.random_seed,
                "connect_settle_ms": self.connect_settle_ms,
            }
            if self.pool_event_groups:
                data["site_switch"]["event_groups"] = self.pool_event_groups
                if self.pool_event_group_centers:
                    data["site_switch"]["event_group_centers"] = self.pool_event_group_centers
                if self.pool_event_group_counts:
                    data["site_switch"]["event_group_counts"] = self.pool_event_group_counts
        if self.type == "electrode_pool_sequence":
            data["electrode_pool_sequence"] = {
                "event_count": self.pool_event_count,
                "event_interval_ms": self.pool_event_interval_ms,
                "electrodes_per_event": self.pool_electrodes_per_event,
                "selection_mode": self.pool_selection_mode,
                "random_seed": self.random_seed,
                "connect_settle_ms": self.connect_settle_ms,
            }
            if self.pool_event_groups:
                data["electrode_pool_sequence"]["event_groups"] = self.pool_event_groups
                if self.pool_event_group_centers:
                    data["electrode_pool_sequence"]["event_group_centers"] = self.pool_event_group_centers
                if self.pool_event_group_counts:
                    data["electrode_pool_sequence"]["event_group_counts"] = self.pool_event_group_counts
        legacy_scan_mode = str(self.scan_mode or "off").strip().lower()
        configured_band = str(self.scan_band_width or "full").strip().lower()
        scan_mode = configured_band if configured_band not in {"", "full"} or legacy_scan_mode in {"off", "none", "0"} else legacy_scan_mode
        if scan_mode in {"full", "all"} and legacy_scan_mode in {"off", "none", "0"}:
            scan_mode = "off"
        if scan_mode not in {"", "off", "none", "0"}:
            data["scan"] = {
                "mode": scan_mode,
                "band_width": scan_mode,
                "local_electrodes": max(1, min(3, int(self.scan_local_electrodes))),
                "selection_mode": "scan_band_sequence",
                "random_seed": int(self.random_seed),
            }
            data["site_switch"] = {
                "enabled": True,
                "selection_mode": "scan_band_sequence",
                "random_seed": int(self.random_seed),
                "connect_settle_ms": float(self.connect_settle_ms),
            }
        if self.type == "custom_sequence":
            data["custom_points"] = self.custom_points
        if self.type == "poisson_random_electrodes":
            data["random_electrode_plan"] = {
                "spontaneous_data_path": self.spontaneous_data_path,
                "candidate_source": self.candidate_source,
                "region_count": self.region_count,
                "max_candidate_electrodes": self.max_candidate_electrodes,
                "duration_s": self.poisson_duration_s,
                "lambda_mode": self.lambda_mode,
                "lambda_scale": self.lambda_scale,
                "lambda_floor_hz": self.lambda_floor_hz,
                "lambda_mean_hz": self.lambda_mean_hz,
                "lambda_std_hz": self.lambda_std_hz,
                "random_seed": self.random_seed,
                "connect_settle_ms": self.connect_settle_ms,
            }
            if self.poisson_candidate_electrodes:
                data["random_electrode_plan"]["candidate_electrodes"] = self.poisson_candidate_electrodes
        if self.notes:
            data["notes"] = self.notes
        return data


@dataclass
class Phase:
    id: str
    duration_s: int = 300
    mode: str = "open_loop"

    def to_yaml(self) -> dict[str, Any]:
        return {"id": self.id, "duration_s": self.duration_s, "mode": self.mode}


@dataclass
class ExperimentBlock:
    name: str
    electrode_group: str
    protocol: str
    phases: list[Phase] = field(default_factory=lambda: [
        Phase("01_pre_spont", 300, "open_loop"),
        Phase("02_stim", 300, "open_loop"),
        Phase("03_post_spont", 300, "open_loop"),
    ])

    def to_yaml(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "electrode_group": self.electrode_group,
            "protocol": self.protocol,
            "phases": [phase.to_yaml() for phase in self.phases],
        }


def parse_electrodes(text: str) -> list[int]:
    values: list[int] = []
    for token in re.split(r"[\s,;]+", text.strip()):
        if token:
            values.append(int(token))
    return values


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


def parse_electrode_event_groups(text: str) -> list[list[int]]:
    groups: list[list[int]] = []
    for raw_group in re.split(r"[|\n]+", text.strip()):
        raw_group = raw_group.strip()
        if not raw_group:
            continue
        values = parse_electrodes(raw_group)
        if values:
            groups.append(values)
    return groups


def parse_custom_points(text: str) -> list[dict[str, float]]:
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        raw_points = json.loads(text)
        return [_normalize_custom_point(item) for item in raw_points]
    points = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            points.append(_normalize_custom_point(json.loads(line)))
            continue
        tokens = [token for token in re.split(r"[\s,;]+", line) if token]
        if len(tokens) < 2:
            raise ValueError(f"Custom point needs at least time_ms and amplitude_mv: {raw_line}")
        point = {"time_ms": float(tokens[0]), "amplitude_mv": float(tokens[1])}
        if len(tokens) >= 3:
            point["duration_us"] = float(tokens[2])
        if len(tokens) >= 4:
            point["channel"] = float(tokens[3])
        points.append(point)
    return sorted(points, key=lambda item: item["time_ms"])


def _normalize_custom_point(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        if "time_ms" not in value or "amplitude_mv" not in value:
            raise ValueError("Custom point dict must contain time_ms and amplitude_mv")
        point = {"time_ms": float(value["time_ms"]), "amplitude_mv": float(value["amplitude_mv"])}
        if "duration_us" in value:
            point["duration_us"] = float(value["duration_us"])
        if "channel" in value:
            point["channel"] = float(value["channel"])
        return point
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        point = {"time_ms": float(value[0]), "amplitude_mv": float(value[1])}
        if len(value) >= 3:
            point["duration_us"] = float(value[2])
        if len(value) >= 4:
            point["channel"] = float(value[3])
        return point
    raise ValueError(f"Unsupported custom point: {value!r}")


def preview_points(protocol: StimulusProtocol) -> list[tuple[float, float]]:
    width_ms = max(protocol.pulse_width_us / 1000.0, 0.001)
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for start_ms, amplitude in pulse_starts_ms(protocol):
        points.extend(
            [
                (max(start_ms - 0.001, 0.0), 0.0),
                (start_ms, amplitude),
                (start_ms + width_ms, amplitude),
                (start_ms + width_ms + 0.001, 0.0),
            ]
        )
    return sorted(points, key=lambda item: item[0]) or [(0.0, 0.0)]


def preview_raster_series(
    protocol: StimulusProtocol,
    preview_limit_ms: float = 5000.0,
    spontaneous_rates: dict[int, float] | None = None,
    electrode_pool: list[int] | None = None,
) -> list[dict[str, Any]]:
    if protocol.type in {"single_pulse", "individual_burst", "sequence_with_burst", "random", "sequence_with_poisson_burst"}:
        if _protocol_uses_site_switch(protocol):
            return _site_switch_preview_raster_series(protocol, preview_limit_ms, electrode_pool)
        return [{
            "label": f"ch{protocol.channel}",
            "channel": protocol.channel,
            "times_ms": [time_ms for time_ms, _amp in pulse_starts_ms(protocol) if 0.0 <= time_ms <= preview_limit_ms],
        }]

    if protocol.type == "custom_sequence":
        if _protocol_uses_site_switch(protocol):
            return _site_switch_preview_raster_series(protocol, preview_limit_ms, electrode_pool)
        grouped: dict[int, list[float]] = {}
        for point in sorted(protocol.custom_points, key=lambda item: item["time_ms"]):
            channel = int(point.get("channel", protocol.channel))
            grouped.setdefault(channel, []).append(float(point["time_ms"]))
        return [
            {"label": f"ch{channel}", "channel": channel, "times_ms": [time_ms for time_ms in times if 0.0 <= time_ms <= preview_limit_ms]}
            for channel, times in sorted(grouped.items())
        ]

    if protocol.type == "poisson_random_electrodes":
        rates = dict(spontaneous_rates or {})
        if not rates and protocol.spontaneous_data_path.strip():
            rates = _preview_spontaneous_rates(protocol)
        if rates:
            candidate_electrodes = (
                _unique_ints([int(value) for value in protocol.poisson_candidate_electrodes])
                if protocol.poisson_candidate_electrodes
                else _preview_candidate_electrodes(rates, protocol.region_count, protocol.max_candidate_electrodes)
            )
        else:
            candidate_electrodes = list(range(min(max(protocol.max_candidate_electrodes, 1), 32)))
            rates = {electrode: max(protocol.lambda_floor_hz, 0.001) for electrode in candidate_electrodes}
        rng = random.Random(protocol.random_seed)
        duration_s = max(0.1, min(protocol.poisson_duration_s, preview_limit_ms / 1000.0))
        series: list[dict[str, Any]] = []
        for electrode in candidate_electrodes:
            lambda_hz = _preview_lambda_hz(rates.get(electrode, protocol.lambda_floor_hz), protocol, rng)
            if lambda_hz <= 0:
                continue
            times_ms: list[float] = []
            current_s = rng.expovariate(lambda_hz)
            while current_s <= duration_s:
                times_ms.append(round(current_s * 1000.0, 3))
                current_s += rng.expovariate(lambda_hz)
            series.append({"label": str(electrode), "channel": int(electrode), "times_ms": times_ms})
        return series

    if protocol.type == "electrode_pool_sequence":
        pool = [int(value) for value in (electrode_pool or [])]
        if not pool:
            pool = [int(protocol.channel)]
        event_groups = electrode_pool_event_groups(protocol, pool)
        grouped: dict[int, list[float]] = {}
        starts = [time_ms for time_ms, _amp in pulse_starts_ms(protocol)]
        for index, time_ms in enumerate(starts):
            if not 0.0 <= time_ms <= preview_limit_ms:
                continue
            targets = event_groups[index] if index < len(event_groups) else []
            for electrode in targets:
                grouped.setdefault(int(electrode), []).append(float(time_ms))
        return [
            {"label": str(electrode), "channel": int(electrode), "times_ms": times}
            for electrode, times in sorted(grouped.items())
        ]

    return [{"label": f"ch{protocol.channel}", "channel": protocol.channel, "times_ms": []}]


def _protocol_uses_site_switch(protocol: StimulusProtocol) -> bool:
    return bool(getattr(protocol, "site_switch_enabled", False)) and str(getattr(protocol, "type", "")) != "poisson_random_electrodes"


def _site_switch_preview_raster_series(
    protocol: StimulusProtocol,
    preview_limit_ms: float,
    electrode_pool: list[int] | None,
) -> list[dict[str, Any]]:
    pool = [int(value) for value in (electrode_pool or [])]
    if not pool:
        pool = [int(protocol.channel)]
    starts = [time_ms for time_ms, _amp in pulse_starts_ms(protocol)]
    event_groups = site_switch_event_groups_for_count(protocol, pool, len(starts))
    grouped: dict[int, list[float]] = {}
    for index, time_ms in enumerate(starts):
        if not 0.0 <= time_ms <= preview_limit_ms:
            continue
        targets = event_groups[index] if index < len(event_groups) else []
        for electrode in targets:
            grouped.setdefault(int(electrode), []).append(float(time_ms))
    return [
        {"label": str(electrode), "channel": int(electrode), "times_ms": times}
        for electrode, times in sorted(grouped.items())
    ]


def pulse_starts_ms(protocol: StimulusProtocol) -> list[tuple[float, float]]:
    if protocol.type == "single_pulse":
        return [(protocol.start_ms, protocol.amplitude_mv)]
    if protocol.type == "individual_burst":
        if _randomize_burst_pulse_intervals(protocol):
            return _burst_pulse_starts(protocol, poisson=False)
        interval = _pulse_interval_ms(protocol)
        return [(protocol.start_ms + i * interval, protocol.amplitude_mv) for i in range(protocol.pulses_per_burst)]
    if protocol.type == "sequence_with_burst":
        return _burst_pulse_starts(protocol, poisson=False)
    if protocol.type == "sequence_with_poisson_burst":
        return _burst_pulse_starts(protocol, poisson=True)
    if protocol.type == "random":
        return _random_burst_pulse_starts(protocol)
    if protocol.type == "custom_sequence":
        return [(point["time_ms"], point["amplitude_mv"]) for point in protocol.custom_points]
    if protocol.type == "poisson_random_electrodes":
        # Lightweight preview only: the generated package builds the real plan
        # from spontaneous data at run time.
        interval_ms = 1000.0 / max(protocol.lambda_floor_hz, 0.001)
        preview_count = min(max(protocol.max_candidate_electrodes, 1), 32)
        return [
            (protocol.start_ms + index * interval_ms, protocol.amplitude_mv)
            for index in range(preview_count)
        ]
    if protocol.type == "electrode_pool_sequence":
        if protocol.pool_event_groups:
            count = len(_expand_pool_event_groups(
                protocol.pool_event_groups,
                mode=str(protocol.pool_selection_mode or "balanced_random_groups"),
                repeats=max(1, int(protocol.pool_event_count or 1)),
                random_seed=int(protocol.random_seed),
            ))
        else:
            count = max(0, int(protocol.pool_event_count))
        interval_ms = max(0.0, float(protocol.pool_event_interval_ms))
        return [
            (float(protocol.start_ms) + index * interval_ms, protocol.amplitude_mv)
            for index in range(count)
        ]
    return []


def electrode_pool_event_groups(protocol: StimulusProtocol, electrode_pool: list[int]) -> list[list[int]]:
    pool = _unique_ints([int(value) for value in electrode_pool])
    if protocol.pool_event_groups:
        base_groups = []
        for group in protocol.pool_event_groups:
            values = _normalize_event_group(group)
            if values:
                base_groups.append(values)
        return _expand_pool_event_groups(
            base_groups,
            mode=str(protocol.pool_selection_mode or "balanced_random_groups"),
            repeats=max(1, int(protocol.pool_event_count or 1)),
            random_seed=int(protocol.random_seed),
        )
    if not pool:
        return []
    event_count = max(0, int(protocol.pool_event_count))
    per_event = max(1, min(int(protocol.pool_electrodes_per_event), len(pool)))
    mode = str(protocol.pool_selection_mode or "random").strip().lower()
    rng = random.Random(int(protocol.random_seed))
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


def site_switch_event_groups_for_count(protocol: StimulusProtocol, electrode_pool: list[int], event_count: int) -> list[list[int]]:
    pool = _unique_ints([int(value) for value in electrode_pool])
    target_count = max(0, int(event_count))
    if target_count <= 0:
        return []
    pulses_per_event = _site_switch_pulses_per_event(protocol)
    switch_count = int(math.ceil(target_count / max(pulses_per_event, 1)))
    if protocol.pool_event_groups:
        base_groups = [_normalize_event_group(group) for group in protocol.pool_event_groups if group]
        base_groups = [group for group in base_groups if group]
        groups = _balanced_site_switch_groups(
            base_groups,
            mode=str(protocol.pool_selection_mode or "balanced_random_groups"),
            target_count=switch_count,
            random_seed=int(protocol.random_seed),
        )
        return _expand_site_switch_groups_to_pulses(groups, target_count, pulses_per_event)
    if not pool:
        return []
    groups = _balanced_site_switch_groups(
        [pool],
        mode=str(protocol.pool_selection_mode or "balanced_random_groups"),
        target_count=switch_count,
        random_seed=int(protocol.random_seed),
    )
    return _expand_site_switch_groups_to_pulses(groups, target_count, pulses_per_event)


def _site_switch_pulses_per_event(protocol: StimulusProtocol) -> int:
    protocol_type = str(getattr(protocol, "type", ""))
    if protocol_type in {"individual_burst", "sequence_with_burst", "random", "sequence_with_poisson_burst"}:
        return max(1, int(getattr(protocol, "pulses_per_burst", 1) or 1))
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
    if mode in {"scan_band_sequence", "band_sequence"}:
        return [
            list(group)
            for group, quota in zip(groups, quotas)
            for _repeat in range(quota)
        ]
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


def _preview_spontaneous_rates(protocol: StimulusProtocol) -> dict[int, float]:
    path = Path(protocol.spontaneous_data_path).expanduser()
    if not path.exists():
        return {}
    suffix = path.suffix.lower()
    if suffix == ".npz":
        import numpy as np

        data = np.load(path, allow_pickle=True)
        if "electrodes" in data and "rates_hz" in data:
            return {int(e): float(r) for e, r in zip(data["electrodes"], data["rates_hz"])}
        if "electrodes" in data and "firing_rate_hz" in data:
            return {int(e): float(r) for e, r in zip(data["electrodes"], data["firing_rate_hz"])}
        return {}

    rows = path.read_text(encoding="utf-8-sig").splitlines()
    if not rows:
        return {}
    header = [token.strip().lower() for token in re.split(r"[\t,]+", rows[0]) if token.strip()]
    try:
        electrode_index = header.index("electrode")
    except ValueError:
        try:
            electrode_index = header.index("electrode_id")
        except ValueError:
            return {}
    rate_index = -1
    for candidate in ("firing_rate_hz", "rate_hz", "rate", "spikes_per_sec"):
        if candidate in header:
            rate_index = header.index(candidate)
            break
    if rate_index < 0:
        return {}

    rates: dict[int, float] = {}
    for row in rows[1:]:
        if not row.strip() or row.lstrip().startswith("#"):
            continue
        values = [token.strip() for token in re.split(r"[\t,]+", row)]
        if len(values) <= max(electrode_index, rate_index):
            continue
        rates[int(float(values[electrode_index]))] = float(values[rate_index])
    return rates


def _preview_candidate_electrodes(
    rates: dict[int, float],
    region_count: int,
    max_candidates: int,
    positions: dict[int, tuple[float, float]] | None = None,
) -> list[int]:
    position_map = positions if isinstance(positions, dict) else {}

    def spatial_key(item):
        electrode = int(item[0])
        position = position_map.get(electrode)
        if position is not None:
            try:
                x, y = float(position[0]), float(position[1])
                if math.isfinite(x) and math.isfinite(y):
                    return (y, x, electrode)
            except (TypeError, ValueError, IndexError):
                pass
        return (electrode // 220, electrode % 220, electrode)

    sorted_items = sorted(rates.items(), key=spatial_key)
    if not sorted_items:
        return []
    step = max(1, int(region_count))
    if len(sorted_items) <= max(1, int(max_candidates)) and step >= len(sorted_items):
        return [electrode for electrode, _rate in sorted_items]
    candidates: list[int] = []
    for start in range(0, len(sorted_items), step):
        region = sorted_items[start:start + step]
        electrode, _rate = max(region, key=lambda item: item[1])
        candidates.append(electrode)
    return candidates[:max(1, min(max_candidates, 32))]


def _protocol_seed(protocol: StimulusProtocol) -> int:
    payload = json.dumps(
        {
            "name": protocol.name,
            "type": protocol.type,
            "start_ms": protocol.start_ms,
            "burst_count": protocol.burst_count,
            "burst_frequency_hz": protocol.burst_frequency_hz,
            "random_distribution": getattr(protocol, "random_distribution", "poisson"),
            "random_lambda_hz": getattr(protocol, "random_lambda_hz", 5.0),
            "random_interval_min_ms": getattr(protocol, "random_interval_min_ms", 100.0),
            "random_interval_max_ms": getattr(protocol, "random_interval_max_ms", 1000.0),
            "random_duration_s": getattr(protocol, "random_duration_s", 60.0),
            "pulses_per_burst": protocol.pulses_per_burst,
            "pulse_frequency_hz": protocol.pulse_frequency_hz,
            "randomize_burst_pulse_intervals": bool(protocol.randomize_burst_pulse_intervals),
            "burst_pulse_interval_min_ms": protocol.burst_pulse_interval_min_ms,
            "burst_pulse_interval_max_ms": protocol.burst_pulse_interval_max_ms,
            "random_seed": protocol.random_seed,
            "amplitude_mv": protocol.amplitude_mv,
            "pulse_width_us": protocol.pulse_width_us,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _burst_starts_ms(protocol: StimulusProtocol, *, poisson: bool) -> list[float]:
    burst_count = max(0, int(protocol.burst_count))
    if burst_count <= 0:
        return []
    starts = [float(protocol.start_ms)]
    if burst_count == 1:
        return starts
    if poisson:
        rng = random.Random(_protocol_seed(protocol))
        mean_ms = _burst_interval_ms(protocol)
        mean_s = mean_ms / 1000.0
        current_ms = float(protocol.start_ms)
        for _index in range(1, burst_count):
            current_ms += rng.expovariate(1.0 / mean_s) * 1000.0
            starts.append(current_ms)
        return starts
    burst_interval = _burst_interval_ms(protocol)
    return [float(protocol.start_ms) + burst_index * burst_interval for burst_index in range(burst_count)]


def _burst_pulse_starts(protocol: StimulusProtocol, *, poisson: bool) -> list[tuple[float, float]]:
    starts = _burst_starts_ms(protocol, poisson=poisson)
    if _randomize_burst_pulse_intervals(protocol):
        rng = random.Random(_protocol_seed(protocol))
        return [
            (burst_start + offset_ms, protocol.amplitude_mv)
            for burst_start in starts
            for offset_ms in _burst_pulse_offsets_ms(protocol, rng)
        ]
    offsets = _burst_pulse_offsets_ms(protocol)
    return [
        (burst_start + offset_ms, protocol.amplitude_mv)
        for burst_start in starts
        for offset_ms in offsets
    ]


def _random_burst_starts_ms(protocol: StimulusProtocol) -> list[float]:
    """Generate burst starts for the user-facing random burst protocol."""
    start_ms = float(getattr(protocol, "start_ms", 0.0))
    duration_s = max(0.0, float(getattr(protocol, "random_duration_s", 0.0)))
    if duration_s <= 0.0:
        return [start_ms]
    stop_ms = start_ms + duration_s * 1000.0
    distribution = str(getattr(protocol, "random_distribution", "poisson") or "poisson").strip().lower()
    minimum = max(0.0, float(getattr(protocol, "random_interval_min_ms", 100.0)))
    rng = random.Random(_protocol_seed(protocol))
    starts = [start_ms]
    current_ms = start_ms
    for _index in range(200_000):
        if distribution in {"poisson", "exponential"}:
            lambda_hz = max(float(getattr(protocol, "random_lambda_hz", 5.0)), 1e-9)
            interval_ms = minimum + rng.expovariate(lambda_hz) * 1000.0
        elif distribution in {"uniform", "flat"}:
            maximum = max(minimum, float(getattr(protocol, "random_interval_max_ms", 1000.0)))
            interval_ms = rng.uniform(minimum, maximum)
        else:
            raise ValueError(f"Unsupported random burst distribution: {distribution}")
        current_ms += max(0.001, interval_ms)
        if current_ms > stop_ms:
            break
        starts.append(current_ms)
    return starts


def _random_burst_pulse_starts(protocol: StimulusProtocol) -> list[tuple[float, float]]:
    starts = _random_burst_starts_ms(protocol)
    if _randomize_burst_pulse_intervals(protocol):
        rng = random.Random(_protocol_seed(protocol))
        offsets_by_burst = [_burst_pulse_offsets_ms(protocol, rng) for _ in starts]
    else:
        offsets = _burst_pulse_offsets_ms(protocol)
        offsets_by_burst = [offsets for _ in starts]
    return [
        (burst_start + offset_ms, protocol.amplitude_mv)
        for burst_start, offsets in zip(starts, offsets_by_burst)
        for offset_ms in offsets
    ]


def protocol_total_duration_s(protocol: StimulusProtocol) -> float:
    """Return the complete protocol duration including the final pulse width."""
    starts = [float(time_ms) for time_ms, _amplitude in pulse_starts_ms(protocol)]
    if not starts:
        return max(0.0, float(getattr(protocol, "start_ms", 0.0))) / 1000.0
    width_ms = max(0.0, float(getattr(protocol, "pulse_width_us", 0.0))) / 1000.0
    return max(starts) / 1000.0 + width_ms / 1000.0


def scan_stimulation_duration_s(protocol: StimulusProtocol, extra_s: float = 10.0) -> int:
    """Return a recording-safe duration for a scan stimulation phase."""
    starts = [float(time_ms) for time_ms, _amplitude in pulse_starts_ms(protocol)]
    if not starts:
        return max(1, int(math.ceil(float(extra_s))))
    duration_s = protocol_total_duration_s(protocol)
    if protocol.type in {"individual_burst", "sequence_with_burst", "sequence_with_poisson_burst"}:
        # The scan phase covers the full event cadence. Pulse width is already
        # contained in the hardware command and should not round a 600 s plan
        # up to 601 s when the requested cadence is 300 events x 2 s.
        duration_s = max(starts) / 1000.0 + _burst_interval_ms(protocol) / 1000.0
    elif protocol.type == "electrode_pool_sequence":
        duration_s += max(0.0, float(protocol.pool_event_interval_ms)) / 1000.0
    return max(1, int(math.ceil(duration_s + max(0.0, float(extra_s)))))


def _preview_lambda_hz(firing_rate_hz: float, protocol: StimulusProtocol, rng: random.Random) -> float:
    floor = max(protocol.lambda_floor_hz, 0.001)
    base = max(float(firing_rate_hz), 0.0)
    mode = str(protocol.lambda_mode or "scale")
    if mode in {"scale", "equal", "greater", "less"}:
        if mode == "equal":
            scale = 1.0
        elif mode == "greater":
            scale = max(float(protocol.lambda_scale), 1.0)
        elif mode == "less":
            scale = 1.0 / max(float(protocol.lambda_scale), 1e-9)
        else:
            scale = float(protocol.lambda_scale)
        value = base * scale
    elif mode in {"normal", "gaussian"}:
        mean = max(float(getattr(protocol, "lambda_mean_hz", floor)), 0.0)
        if mode == "gaussian" and not hasattr(protocol, "lambda_std_hz"):
            sigma = max(base * float(getattr(protocol, "lambda_gaussian_cv", 0.25)), floor)
        else:
            sigma = max(float(getattr(protocol, "lambda_std_hz", floor)), 0.0)
        value = rng.gauss(mean, sigma)
    else:
        value = base * float(protocol.lambda_scale)
    return max(float(value), floor)


def _pulse_interval_ms(protocol: StimulusProtocol) -> float:
    return 1000.0 / max(protocol.pulse_frequency_hz, 0.001)


def _burst_interval_ms(protocol: StimulusProtocol) -> float:
    return 1000.0 / max(float(getattr(protocol, "burst_frequency_hz", 5.0)), 0.001)


def _default_protocol_name(protocol_type: str) -> str:
    protocol_type = str(protocol_type or "protocol")
    return short_default_name(
        _PROTOCOL_DEFAULT_NAMES.get(protocol_type, protocol_type),
        fallback="protocol",
    )


def _protocol_seed(protocol: StimulusProtocol) -> int:
    payload = json.dumps(
        {
            "name": protocol.name,
            "type": protocol.type,
            "start_ms": protocol.start_ms,
            "burst_count": protocol.burst_count,
            "burst_frequency_hz": protocol.burst_frequency_hz,
            "random_distribution": getattr(protocol, "random_distribution", "poisson"),
            "random_lambda_hz": getattr(protocol, "random_lambda_hz", 5.0),
            "random_interval_min_ms": getattr(protocol, "random_interval_min_ms", 100.0),
            "random_interval_max_ms": getattr(protocol, "random_interval_max_ms", 1000.0),
            "random_duration_s": getattr(protocol, "random_duration_s", 60.0),
            "pulses_per_burst": protocol.pulses_per_burst,
            "pulse_frequency_hz": protocol.pulse_frequency_hz,
            "randomize_burst_pulse_intervals": bool(protocol.randomize_burst_pulse_intervals),
            "burst_pulse_interval_min_ms": protocol.burst_pulse_interval_min_ms,
            "burst_pulse_interval_max_ms": protocol.burst_pulse_interval_max_ms,
            "random_seed": protocol.random_seed,
            "amplitude_mv": protocol.amplitude_mv,
            "pulse_width_us": protocol.pulse_width_us,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _burst_starts_ms(protocol: StimulusProtocol, *, poisson: bool) -> list[float]:
    burst_count = max(0, int(protocol.burst_count))
    if burst_count <= 0:
        return []
    starts = [float(protocol.start_ms)]
    if burst_count == 1:
        return starts
    if poisson:
        rng = random.Random(_protocol_seed(protocol))
        mean_s = _burst_interval_ms(protocol) / 1000.0
        current_ms = float(protocol.start_ms)
        for _index in range(1, burst_count):
            current_ms += rng.expovariate(1.0 / mean_s) * 1000.0
            starts.append(current_ms)
        return starts
    burst_interval = _burst_interval_ms(protocol)
    return [float(protocol.start_ms) + burst_index * burst_interval for burst_index in range(burst_count)]


def _burst_pulse_starts(protocol: StimulusProtocol, *, poisson: bool) -> list[tuple[float, float]]:
    starts = _burst_starts_ms(protocol, poisson=poisson)
    if _randomize_burst_pulse_intervals(protocol):
        rng = random.Random(_protocol_seed(protocol))
        return [
            (burst_start + offset_ms, protocol.amplitude_mv)
            for burst_start in starts
            for offset_ms in _burst_pulse_offsets_ms(protocol, rng)
        ]
    offsets = _burst_pulse_offsets_ms(protocol)
    return [
        (burst_start + offset_ms, protocol.amplitude_mv)
        for burst_start in starts
        for offset_ms in offsets
    ]


def _randomize_burst_pulse_intervals(protocol: StimulusProtocol) -> bool:
    return bool(getattr(protocol, "randomize_burst_pulse_intervals", False))


def _burst_pulse_offsets_ms(protocol: StimulusProtocol, rng: random.Random | None = None) -> list[float]:
    count = max(1, int(protocol.pulses_per_burst))
    if not _randomize_burst_pulse_intervals(protocol):
        interval = _pulse_interval_ms(protocol)
        return [pulse_index * interval for pulse_index in range(count)]
    min_ms = max(0.0, float(getattr(protocol, "burst_pulse_interval_min_ms", 10.0)))
    max_ms = max(min_ms, float(getattr(protocol, "burst_pulse_interval_max_ms", 100.0)))
    if rng is None:
        rng = random.Random(_protocol_seed(protocol))
    offsets = [0.0]
    current_ms = 0.0
    for _pulse_index in range(1, count):
        current_ms += rng.uniform(min_ms, max_ms)
        offsets.append(current_ms)
    return offsets


def distribution_preview_specs(
    protocol: StimulusProtocol,
    spontaneous_rates: dict[int, float] | None = None,
    electrode_pool: list[int] | None = None,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    protocol_type = str(getattr(protocol, "type", ""))

    if protocol_type in {"individual_burst", "sequence_with_burst", "random", "sequence_with_poisson_burst"}:
        if _randomize_burst_pulse_intervals(protocol):
            rng = random.Random(_protocol_seed(protocol))
            intervals_ms: list[float] = []
            if protocol_type == "random":
                burst_starts = _random_burst_starts_ms(protocol)
            else:
                burst_starts = _burst_starts_ms(protocol, poisson=(protocol_type == "sequence_with_poisson_burst"))
            for _start_ms in burst_starts:
                offsets = _burst_pulse_offsets_ms(protocol, rng)
                intervals_ms.extend(
                    max(0.0, float(next_offset) - float(previous_offset))
                    for previous_offset, next_offset in zip(offsets, offsets[1:])
                )
            specs.append(
                {
                    "title": "Intra-burst pulse interval",
                    "x_label": "Interval (ms)",
                    "y_label": "Density",
                    "actual": intervals_ms,
                    "actual_label": "Generated intervals",
                    "expected": {
                        "kind": "uniform",
                        "min": max(0.0, float(getattr(protocol, "burst_pulse_interval_min_ms", 10.0))),
                        "max": max(
                            max(0.0, float(getattr(protocol, "burst_pulse_interval_min_ms", 10.0))),
                            float(getattr(protocol, "burst_pulse_interval_max_ms", 100.0)),
                        ),
                        "label": "Expected uniform",
                    },
                }
            )

    if protocol_type == "sequence_with_poisson_burst":
        starts = _burst_starts_ms(protocol, poisson=True)
        intervals_ms = [
            max(0.0, float(next_start) - float(previous_start))
            for previous_start, next_start in zip(starts, starts[1:])
        ]
        specs.append(
            {
                "title": "Burst interval",
                "x_label": "Interval (ms)",
                "y_label": "Density",
                "actual": intervals_ms,
                "actual_label": "Generated burst intervals",
                "expected": {
                    "kind": "exponential",
                    "mean": _burst_interval_ms(protocol),
                    "label": "Expected Poisson interval",
                },
            }
        )

    if protocol_type == "random":
        starts = _random_burst_starts_ms(protocol)
        intervals_ms = [
            max(0.0, float(next_start) - float(previous_start))
            for previous_start, next_start in zip(starts, starts[1:])
        ]
        distribution = str(getattr(protocol, "random_distribution", "poisson") or "poisson").strip().lower()
        if distribution in {"poisson", "exponential"}:
            minimum = max(0.0, float(getattr(protocol, "random_interval_min_ms", 100.0)))
            expected = {
                "kind": "exponential",
                "mean": 1000.0 / max(float(getattr(protocol, "random_lambda_hz", 5.0)), 1e-9),
                "min": minimum,
                "label": "Expected Poisson interval",
            }
        else:
            minimum = max(0.0, float(getattr(protocol, "random_interval_min_ms", 100.0)))
            maximum = max(minimum, float(getattr(protocol, "random_interval_max_ms", 1000.0)))
            expected = {
                "kind": "uniform",
                "min": minimum,
                "max": maximum,
                "label": "Expected uniform interval",
            }
        specs.append(
            {
                "title": "Random burst interval",
                "x_label": "Interval (ms)",
                "y_label": "Density",
                "actual": intervals_ms,
                "actual_label": "Generated burst intervals",
                "expected": expected,
            }
        )

    if protocol_type == "poisson_random_electrodes" and str(getattr(protocol, "lambda_mode", "scale")) in {"normal", "gaussian"}:
        rates = dict(spontaneous_rates or {})
        if not rates and str(getattr(protocol, "spontaneous_data_path", "")).strip():
            rates = _preview_spontaneous_rates(protocol)
        if rates:
            candidate_electrodes = (
                _unique_ints([int(value) for value in protocol.poisson_candidate_electrodes])
                if protocol.poisson_candidate_electrodes
                else _preview_candidate_electrodes(
                    rates,
                    int(getattr(protocol, "region_count", 32)),
                    int(getattr(protocol, "max_candidate_electrodes", 32)),
                )
            )
        else:
            candidate_electrodes = list(range(min(max(int(getattr(protocol, "max_candidate_electrodes", 32)), 1), 32)))
            rates = {
                electrode: max(float(getattr(protocol, "lambda_floor_hz", 0.001)), 0.001)
                for electrode in candidate_electrodes
            }
        rng = random.Random(int(getattr(protocol, "random_seed", 42)))
        lambdas: list[float] = []
        for electrode in candidate_electrodes:
            lambda_hz = _preview_lambda_hz(
                rates.get(electrode, float(getattr(protocol, "lambda_floor_hz", 0.001))),
                protocol,
                rng,
            )
            lambdas.append(lambda_hz)
        specs.append(
            {
                "title": "Poisson lambda per electrode",
                "x_label": "Lambda (Hz)",
                "y_label": "Density",
                "actual": lambdas,
                "actual_label": "Generated lambdas",
                "expected": {
                    "kind": "normal",
                    "mean": max(float(getattr(protocol, "lambda_mean_hz", 1.0)), 0.0),
                    "std": max(float(getattr(protocol, "lambda_std_hz", 0.25)), 0.0),
                    "floor": max(float(getattr(protocol, "lambda_floor_hz", 0.001)), 0.001),
                    "label": "Expected normal",
                },
            }
        )

    return specs


def build_package(
    output_dir: Path,
    info: ExperimentInfo,
    groups: list[ElectrodeGroup],
    protocols: list[StimulusProtocol],
    blocks: list[ExperimentBlock],
) -> Path:
    output_dir = output_dir.expanduser().resolve()
    _validate(info, groups, protocols, blocks)
    groups, blocks = _with_plan_electrode_groups(groups, protocols, blocks)
    protocol_lookup = {protocol.name: protocol for protocol in protocols}
    adjusted_blocks: list[ExperimentBlock] = []
    for block in blocks:
        protocol = protocol_lookup.get(block.protocol)
        phases = [Phase(phase.id, phase.duration_s, phase.mode) for phase in block.phases]
        if protocol is not None and str(getattr(protocol, "scan_mode", "off") or "off").lower() not in {"", "off", "none", "0"}:
            phases = [
                Phase(
                    phase.id,
                    scan_stimulation_duration_s(protocol) if phase.id == "02_stim" else phase.duration_s,
                    phase.mode,
                )
                for phase in phases
            ]
        adjusted_blocks.append(ExperimentBlock(block.name, block.electrode_group, block.protocol, phases))
    blocks = adjusted_blocks

    for rel in ["config", "python/utils", "scripts", "data"]:
        (output_dir / rel).mkdir(parents=True, exist_ok=True)

    _write(output_dir / "README.md", _readme(info, groups, protocols, blocks))
    _write(output_dir / "requirements.txt", "pyyaml>=6.0\nnumpy>=1.24\nh5py>=3.0\n")
    _write(output_dir / ".gitignore", "__pycache__/\n*.pyc\ndata/\ncpp/build/\n*.h5\n")
    _write(output_dir / "setup.py", _setup_py(info))
    _write(output_dir / "main.py", GENERATED_MAIN)
    _write(output_dir / "python/__init__.py", "# Generated experiment package.\n")
    _write(output_dir / "python/experiment_runner.py", GENERATED_EXPERIMENT_RUNNER)
    _write(output_dir / "python/maxwell_setup.py", GENERATED_MAXWELL_SETUP)
    _write(output_dir / "python/random_stim_plan.py", GENERATED_RANDOM_STIM_PLAN)
    _write(output_dir / "python/utils/__init__.py", "# Generated utility package.\n")
    _write(output_dir / "python/utils/time_log.py", GENERATED_TIME_LOG)
    _write(output_dir / "config/system.yaml", _dump_yaml(_system_yaml(info, blocks)))
    _write(output_dir / "config/stimulation.yaml", _dump_yaml(_stimulation_yaml(groups, protocols)))
    return output_dir


def _validate(
    info: ExperimentInfo,
    groups: list[ElectrodeGroup],
    protocols: list[StimulusProtocol],
    blocks: list[ExperimentBlock],
) -> None:
    if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", info.name):
        raise ValueError("Experiment name must be English snake_case, e.g. closed_loop_spike_threshold")
    all_non_stim = bool(blocks) and all(
        _block_is_record_only(block) or _block_is_rest_only(block)
        for block in blocks
    )
    if not groups and not all_non_stim:
        raise ValueError("At least one electrode group is required")
    if not protocols and not all_non_stim:
        raise ValueError("At least one stimulation protocol is required")
    if not blocks:
        raise ValueError("At least one block is required")
    group_names = {group.name for group in groups}
    protocol_names = {protocol.name for protocol in protocols}
    for group in groups:
        if group.multi_electrode and group.center_electrode is None and not all_non_stim:
            raise ValueError(f"Multi-electrode group {group.name} needs a center electrode")
        if int(group.electrode_count) < 1 or int(group.electrode_count) > MAX_STIMULATION_UNITS_PER_ROUTE:
            raise ValueError(
                f"Electrode group {group.name} electrode count must be between 1 and {MAX_STIMULATION_UNITS_PER_ROUTE}"
            )
        if not group.multi_electrode and int(group.electrode_count) != 1:
            raise ValueError(f"Electrode group {group.name} must use electrode count 1 when multi-electrode stimulation is off")
    for protocol in protocols:
        if protocol.type not in PROTOCOL_TYPES:
            raise ValueError(f"Unsupported protocol type: {protocol.type}")
        if protocol.type == "custom_sequence" and not protocol.custom_points:
            raise ValueError(f"Custom protocol {protocol.name} needs at least one point")
        if protocol.type == "random":
            distribution = str(getattr(protocol, "random_distribution", "poisson") or "poisson").strip().lower()
            if distribution not in {"poisson", "exponential", "uniform", "flat"}:
                raise ValueError(f"Protocol {protocol.name} needs Poisson or uniform random burst distribution")
            if float(getattr(protocol, "random_duration_s", 0.0)) <= 0.0:
                raise ValueError(f"Protocol {protocol.name} needs random duration > 0 s")
            minimum = float(getattr(protocol, "random_interval_min_ms", 100.0))
            if minimum < 0.0:
                raise ValueError(f"Protocol {protocol.name} needs interval min >= 0 ms")
            if distribution in {"poisson", "exponential"}:
                if float(getattr(protocol, "random_lambda_hz", 0.0)) <= 0.0:
                    raise ValueError(f"Protocol {protocol.name} needs lambda > 0 Hz")
            else:
                maximum = float(getattr(protocol, "random_interval_max_ms", 0.0))
                if minimum < 0.0 or maximum < minimum or maximum <= 0.0:
                    raise ValueError(f"Protocol {protocol.name} needs 0 <= interval min <= interval max and max > 0 ms")
        if protocol.randomize_burst_pulse_intervals:
            if protocol.burst_pulse_interval_min_ms < 0:
                raise ValueError(f"Protocol {protocol.name} needs burst pulse interval min >= 0 ms")
            if protocol.burst_pulse_interval_max_ms < protocol.burst_pulse_interval_min_ms:
                raise ValueError(f"Protocol {protocol.name} needs burst pulse interval max >= min")
        if protocol.type == "poisson_random_electrodes" and protocol.max_candidate_electrodes > 32:
            raise ValueError("MaxOne-safe random stimulation supports at most 32 candidate electrodes")
        if protocol.type == "electrode_pool_sequence":
            if protocol.pool_event_count <= 0 and not protocol.pool_event_groups:
                raise ValueError(f"Electrode pool protocol {protocol.name} needs at least one event")
            if protocol.pool_electrodes_per_event <= 0 and not protocol.pool_event_groups:
                raise ValueError(f"Electrode pool protocol {protocol.name} needs electrodes_per_event > 0")
            if protocol.pool_event_interval_ms < 0:
                raise ValueError(f"Electrode pool protocol {protocol.name} needs event_interval_ms >= 0")
        if _protocol_uses_site_switch_config(protocol):
            if not protocol.pool_event_groups:
                raise ValueError(f"Protocol {protocol.name} needs at least one site-switch event group")
    for block in blocks:
        if _block_is_record_only(block) or _block_is_rest_only(block):
            continue
        if block.electrode_group not in group_names:
            raise ValueError(f"Block {block.name} references missing group {block.electrode_group}")
        if block.protocol not in protocol_names:
            raise ValueError(f"Block {block.name} references missing protocol {block.protocol}")
        if [phase.id for phase in block.phases] != list(PHASES):
            raise ValueError(f"Block {block.name} must contain fixed phases: {', '.join(PHASES)}")


def _block_is_record_only(block: ExperimentBlock) -> bool:
    for phase in getattr(block, "phases", []) or []:
        if phase.id == "02_stim":
            return str(phase.mode or "").strip().lower() in {"record_only", "recording_only", "no_stim", "none"}
    return False


def _phase_is_rest_only(phase: Phase) -> bool:
    return str(phase.mode or "").strip().lower() in {
        "rest_only",
        "rest",
        "recovery",
        "idle",
        "no_record",
    }


def _block_is_rest_only(block: ExperimentBlock) -> bool:
    phases = list(getattr(block, "phases", []) or [])
    return bool(phases) and all(_phase_is_rest_only(phase) for phase in phases)


def _protocol_uses_site_switch_config(protocol: StimulusProtocol) -> bool:
    return bool(getattr(protocol, "site_switch_enabled", False)) and protocol.type != "poisson_random_electrodes"


def _with_plan_electrode_groups(
    groups: list[ElectrodeGroup],
    protocols: list[StimulusProtocol],
    blocks: list[ExperimentBlock],
) -> tuple[list[ElectrodeGroup], list[ExperimentBlock]]:
    protocol_lookup = {protocol.name: protocol for protocol in protocols}
    group_lookup = {group.name: group for group in groups}
    resolved_groups = [
        ElectrodeGroup(
            group.name,
            list(group.electrodes),
            center_electrode=group.center_electrode,
            multi_electrode=bool(group.multi_electrode),
            electrode_count=max(1, int(group.electrode_count)),
        )
        for group in groups
    ]
    resolved_group_lookup = {group.name: group for group in resolved_groups}
    resolved_blocks: list[ExperimentBlock] = []
    existing_names = {group.name for group in resolved_groups}

    for block in blocks:
        if _block_is_record_only(block) or _block_is_rest_only(block):
            resolved_blocks.append(
                ExperimentBlock(
                    block.name,
                    block.electrode_group,
                    block.protocol,
                    [Phase(phase.id, phase.duration_s, phase.mode) for phase in block.phases],
                )
            )
            continue
        protocol = protocol_lookup.get(block.protocol)
        group = group_lookup.get(block.electrode_group)
        target_group = block.electrode_group
        if protocol is not None and group is not None and (
            protocol.type == "poisson_random_electrodes"
            or protocol.type == "electrode_pool_sequence"
            or _protocol_uses_site_switch_config(protocol)
        ):
            if protocol.type == "poisson_random_electrodes":
                candidates = poisson_candidate_electrodes_for_protocol(protocol, list(group.electrodes))
            else:
                candidates = electrode_pool_candidate_electrodes_for_protocol(protocol, list(group.electrodes))
            group_label = str(block.electrode_group)
            if not group_label.endswith("_auto") and not group_label.endswith(f"_{protocol.name}_auto"):
                target_group = _unique_group_name(f"{block.electrode_group}_auto", existing_names)
            else:
                target_group = block.electrode_group
            existing_names.add(target_group)
            target_electrodes = [int(value) for value in candidates]
            if target_group in resolved_group_lookup:
                resolved_group_lookup[target_group].electrodes = target_electrodes
            else:
                auto_group = ElectrodeGroup(
                    target_group,
                    target_electrodes,
                    center_electrode=group.center_electrode,
                    multi_electrode=bool(group.multi_electrode),
                    electrode_count=max(1, int(group.electrode_count)),
                )
                resolved_groups.append(auto_group)
                resolved_group_lookup[target_group] = auto_group
        resolved_blocks.append(
            ExperimentBlock(
                block.name,
                target_group,
                block.protocol,
                [Phase(phase.id, phase.duration_s, phase.mode) for phase in block.phases],
            )
        )
    return resolved_groups, resolved_blocks


def _unique_group_name(base: str, existing_names: set[str]) -> str:
    return unique_short_name(base, existing_names, fallback="auto_group")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _system_yaml(info: ExperimentInfo, blocks: list[ExperimentBlock]) -> dict[str, Any]:
    return {
        "culture": {"id": info.culture_id, "div": info.div},
        "experiment": {
            "date": info.date,
            "name": info.name,
            "recording_name_prefix": info.recording_prefix or info.name,
            "scientific_question": info.scientific_question,
            "closed_loop_logic": info.closed_loop_logic,
            "expected_output": info.expected_output,
            "blocks": [block.to_yaml() for block in blocks],
        },
        "electrode_map": {"cfg_path": info.cfg_path},
        "data": {"root": info.data_root},
        "maxwell": {
            "device": info.device,
            "event_threshold": info.event_threshold,
            "amplifier_gain": info.amplifier_gain,
            "recording_settle_s": info.recording_settle_s,
            "hardware_dac": {
                "signal_dacs": [0, 1],
                "neutral_dac": 2,
                "sync_dual_dac": True,
                "allocation": "round_robin_electrode_order",
            },
        },
        "burst_detection": {"bin_ms": 10, "smooth_sigma_ms": 300, "k_rms": 1.2},
    }


def _stimulation_yaml(groups: list[ElectrodeGroup], protocols: list[StimulusProtocol]) -> dict[str, Any]:
    return {"electrode_groups": [group.to_yaml() for group in groups], "protocols": [protocol.to_yaml() for protocol in protocols]}


def _dump_yaml(value: Any, indent: int = 0) -> str:
    prefix = "  " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_dump_yaml(item, indent + 1).rstrip())
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        if not value:
            return f"{prefix}[]\n"
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.append(_dump_yaml(item, indent + 1).rstrip())
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    return f"{prefix}{_yaml_scalar(value)}\n"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "":
        return '""'
    if re.match(r"^[A-Za-z0-9_./:+-]+$", text) and text.lower() not in {"true", "false", "null"}:
        return text
    return json.dumps(text, ensure_ascii=False)


def _readme(
    info: ExperimentInfo,
    groups: list[ElectrodeGroup],
    protocols: list[StimulusProtocol],
    blocks: list[ExperimentBlock],
) -> str:
    group_lines = "\n".join(f"- `{group.name}`: {group.electrodes}" for group in groups)
    protocol_lines = "\n".join(
        (
            f"- `{protocol.name}`: `{protocol.type}`, amplitude={protocol.amplitude_mv} mV, "
            f"width={protocol.pulse_width_us} us"
            + (
                f", distribution={protocol.random_distribution}, "
                f"total_duration_s={protocol_total_duration_s(protocol):.3f}"
                if protocol.type == "random"
                else ""
            )
        )
        for protocol in protocols
    )
    rate_source_lines = "\n".join(
        f"- `{protocol.name}`: `config/pipeline_rate_sources/*_rates.npz` (strictly above the finite-channel median)"
        for protocol in protocols
        if str(protocol.spontaneous_data_path or "").strip()
    ) or "- No spontaneous firing-rate source selected."
    block_lines = "\n".join(
        f"- `{block.name}`: group=`{block.electrode_group}`, protocol=`{block.protocol}`"
        for block in blocks
    )
    return f"""# {info.name}

Generated MaxWell experiment package.

## Experiment Information

- Culture ID: {info.culture_id}
- DIV: {info.div}
- Date: {info.date}
- Recording prefix: {info.recording_prefix}
- Scientific question: {info.scientific_question}
- Closed-loop logic: {info.closed_loop_logic}
- Expected output: {info.expected_output}

## Electrode Groups

{group_lines}

## Stimulation Protocols

{protocol_lines}

## Stimulation Electrode Activity Filter

{rate_source_lines}

When a spontaneous source is selected, the GUI writes one NPZ rate table per
source into `config/pipeline_rate_sources/`. Runtime electrode selection keeps
the existing CFG/unit and spatial constraints, then accepts only candidates
whose firing rate is strictly above the finite-channel median. If a center or
replacement electrode fails that activity check, it is re-selected using the
same spatial search and unit-uniqueness validation.

## Blocks

{block_lines}

## Run

```bash
pip install -r requirements.txt
python main.py --config-dir config --dry-run
python main.py --config-dir config
```

Use `--dry-run` to validate config and generate the required run directory, block folders, `stim_times.txt`, `segment_time_meta.json`, and run-level audit tables without MaxWell hardware.

For real hardware runs, install the MaxWell Python API or set `MAXLAB_PYTHON_PATH` to a folder containing `maxlab`, for example the workspace `api_utils/api_utils` folder.

## Output Structure

Each `main.py` execution creates `data/{{YYYYMMDD_HHMMSS}}_data/`. Every block has fixed `01_pre_spont`, `02_stim`, and `03_post_spont` phases. Stimulating blocks record spontaneous/stimulation data as configured. A `rest_only` block uses the three durations as quiet recovery waits and does not start recording, configure stimulation, or write a stimulation segment file.

Stimulation uses signal DACs 0 and 1 by default, with DAC 2 held at neutral. Stimulation units are assigned to signal DACs in electrode order. Each pulse is sent separately from the host, following the D21 connection and waveform flow. For switching plans, DACs are zeroed and every active unit is configured before waiting for the next pulse, including repeated groups. `connect_settle_ms` is retained as plan metadata and does not control hardware switching; logs contain measured host connection and send times. Pulses have two contiguous phases, with no inter-phase gap. Hardware event timestamps remain the source for precise acquisition alignment. Each stimulation phase writes `hardware_mapping.json`, `hardware_route_log.json`, and includes the same mapping in `segment_time_meta.json`; these files record electrode-to-unit, unit-to-DAC, route switches, connect timing, and settling time for each pulse.
"""


def _setup_py(info: ExperimentInfo) -> str:
    return f"""from setuptools import find_packages, setup

setup(
    name={info.name!r},
    version="0.1.0",
    packages=find_packages(include=["python", "python.*"]),
    install_requires=["pyyaml>=6.0", "numpy>=1.24", "h5py>=3.0"],
    python_requires=">=3.10",
)
"""


GENERATED_MAIN = r'''
from __future__ import annotations

import argparse
import importlib.metadata
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from python.experiment_runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generated MaxWell experiment package.")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def check_requirements(path: Path) -> list[str]:
    missing = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        package = line.split(">=", 1)[0].split("==", 1)[0].strip()
        import_name = "yaml" if package == "pyyaml" else package.replace("-", "_")
        try:
            __import__(import_name)
        except ImportError:
            try:
                importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                missing.append(package)
    return missing


def configure_logging(run_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(run_dir / "log.txt", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = base_dir / config_dir
    system_path = config_dir / "system.yaml"
    stimulation_path = config_dir / "stimulation.yaml"
    if not system_path.exists() or not stimulation_path.exists():
        print(f"Missing config files in {config_dir}")
        return 1

    missing = check_requirements(base_dir / "requirements.txt")
    if missing:
        print("Missing requirements: " + ", ".join(missing))
        print("Install with: pip install -r requirements.txt")
    else:
        print("Environment check OK")

    system_config = load_yaml(system_path)
    stimulation_config = load_yaml(stimulation_path)
    data_root = Path(system_config.get("data", {}).get("root", "./data"))
    if not data_root.is_absolute():
        data_root = base_dir / data_root
    run_dir = data_root / time.strftime("%Y%m%d_%H%M%S_data")
    run_dir.mkdir(parents=True, exist_ok=False)
    configure_logging(run_dir)
    logging.info("Run directory created: %s", run_dir)

    try:
        cfg_path = Path(system_config.get("electrode_map", {}).get("cfg_path", ""))
        if not cfg_path.is_file():
            raise FileNotFoundError(f"cfg_path does not exist: {cfg_path}")
        cfg_copy_path = run_dir / time.strftime("%Hh%Mm%Ss.cfg")
        shutil.copy(cfg_path, cfg_copy_path)
        logging.info("CFG copied: %s -> %s", cfg_path, cfg_copy_path)
        snapshot = run_dir / "config_snapshot"
        snapshot.mkdir(exist_ok=True)
        shutil.copy(system_path, snapshot / "system.yaml")
        shutil.copy(stimulation_path, snapshot / "stimulation.yaml")
        logging.info("Config snapshot saved: %s", snapshot)
        run_experiment(system_config, stimulation_config, run_dir, dry_run=args.dry_run)
        print(run_dir.resolve())
        return 0
    except Exception as exc:
        logging.exception("Experiment failed")
        print(f"Experiment failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''.lstrip()


GENERATED_EXPERIMENT_RUNNER = r'''
from __future__ import annotations

import csv
import json
import logging
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

from python.maxwell_setup import (
    build_poisson_random_sequence,
    build_stim_sequence,
    configure_poisson_experiment_array,
    resolve_experiment_array,
    _default_stim_unit_dac_sources,
    configure_experiment_array,
    configure_experiment_array_with_mapping,
    create_experiment_saving,
    get_stim_times_for_protocol,
    enable_stimulation_power,
    initialize_maxlab,
    prepare_recording_only,
    record_channels_excluding,
    probe_stimulation_electrodes,
    scan_cfg_stimulation_units,
)
from python.random_stim_plan import build_poisson_random_plan
from python.random_stim_plan import build_electrode_pool_sequence_plan
from python.random_stim_plan import build_site_switch_plan
from python.random_stim_plan import electrode_pool_event_groups
from python.random_stim_plan import site_switch_event_groups_for_count
from python.random_stim_plan import poisson_rates_for_electrodes
from python.random_stim_plan import select_poisson_candidate_electrodes
from python.random_stim_plan import load_spontaneous_rates
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
        "execution_mode": "d21_host_per_pulse",
        "connect_timing": "switch_before_host_deadline_wait",
        "connect_settle_ms": float(connect_settle_ms),
        "signal_dacs": [int(value) for value in signal_dacs],
        "neutral_dac": int(neutral_dac),
        "sync_dual_dac": bool(sync_dual_dac),
        "dac_allocation": "round_robin_electrode_order",
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
        route_signature = _route_signature(
            target_units,
            event_dac_by_stim_unit if event_level_switch else None,
            stim_unit_to_dac,
            event_level_switch=event_level_switch,
        )
        route_switched = bool(target_units) and (
            previous_route_signature is None
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
    # CFG-driven scan plans are resolved to one local spatial site per event
    # at runtime, so they must use the host-side dynamic switching path even
    # before explicit event_groups have been materialized.
    if _scan_mode_enabled(protocol):
        return True
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
    if not dry_run:
        cfg_electrodes = _cfg_recording_electrodes(cfg_path)
        id_mapping = _normalize_stimulation_ids_for_cfg(groups, protocols, cfg_electrodes)
        if id_mapping:
            logging.warning("Normalized stimulation electrode IDs to CFG namespace: %s", ",".join(f"{source}->{target}" for source, target in sorted(id_mapping.items())))
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
        runtime_stim_unit_by_electrode: dict[int, int] = {}
        runtime_stim_unit_diagnostics: list[dict[str, Any]] = []
    else:
        try:
            audit.mark_event("hardware_initialize_start", "", "", "")
            requires_unit_scan = any(
                bool(groups.get(str(block.get("electrode_group", "")), {}).get("multi_electrode", False))
                or _group_center_electrode(groups.get(str(block.get("electrode_group", "")), {})) is not None
                or _protocol_uses_plan(protocols.get(str(block.get("protocol", "")), {}))
                or _protocol_has_stimulation_rate_source(protocols.get(str(block.get("protocol", "")), {}))
                or _scan_mode_enabled(protocols.get(str(block.get("protocol", "")), {}))
                for block in blocks
                if not _block_has_record_only_stim(block) and not _block_is_rest_only(block)
            )
            # The Maxwell probe utility resolves CFG electrode units before
            # stimulation power is enabled. Match that order here because
            # enabling power can change the temporary stimulation-array state.
            initialize_maxlab(system_config, power_up_stimulation=not requires_unit_scan)
            audit.mark_event("hardware_initialize_done", "", "", "")
            logging.info("MaxLab initialized")
            if requires_unit_scan:
                runtime_stim_unit_by_electrode, unresolved_cfg_units, runtime_stim_unit_diagnostics = scan_cfg_stimulation_units(
                    cfg_path,
                    cfg_electrodes,
                    system_config,
                )
            else:
                runtime_stim_unit_by_electrode, unresolved_cfg_units, runtime_stim_unit_diagnostics = {}, [], []
            if requires_unit_scan:
                enable_stimulation_power()
                audit.mark_event("stimulation_power_enabled", "", "", "")
            (run_dir / "stimulation_unit_scan.json").write_text(
                json.dumps(
                    {
                        "cfg_path": str(cfg_path),
                        "cfg_recording_electrode_count": len(cfg_electrodes),
                        "mapped_count": len(runtime_stim_unit_by_electrode),
                        "unresolved_count": len(unresolved_cfg_units),
                        "results": runtime_stim_unit_diagnostics,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            status_counts: dict[str, int] = {}
            for detail in runtime_stim_unit_diagnostics:
                status = str(detail.get("status", "unknown"))
                status_counts[status] = status_counts.get(status, 0) + 1
            logging.info(
                "CFG stimulation-unit scan complete: mapped=%d unresolved=%d statuses=%s",
                len(runtime_stim_unit_by_electrode),
                len(unresolved_cfg_units),
                status_counts,
            )
            audit.mark_event(
                "cfg_stimulation_unit_scan_done",
                "",
                "",
                "",
                extra={
                    "mapped_count": len(runtime_stim_unit_by_electrode),
                    "unresolved_count": len(unresolved_cfg_units),
                    "status_counts": status_counts,
                    "diagnostics_path": str(run_dir / "stimulation_unit_scan.json"),
                },
            )
            # Scan-mode site selection depends on the CFG stimulation-unit map,
            # so validate requested sites only after that map has been loaded.
            audit.mark_event("cfg_preflight_start", "", "", "", extra={"cfg_path": str(cfg_path)})
            preflight = _validate_cfg_stimulation_sites(
                cfg_path,
                blocks,
                groups,
                protocols,
                cfg_electrodes=cfg_electrodes,
                stim_unit_by_electrode=runtime_stim_unit_by_electrode,
            )
            logging.info(
                "CFG preflight OK: cfg electrodes=%d requested stimulation electrodes=%d",
                preflight["cfg_electrode_count"],
                preflight["requested_electrode_count"],
            )
            audit.mark_event("cfg_preflight_ok", "", "", "", extra=preflight)
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
                reserved_units: set[int] = set()
                stimulation_rates, stimulation_rate_threshold, rate_filter_info = _stimulation_rate_filter(
                    protocol,
                    system_config,
                )
                if rate_filter_info.get("enabled"):
                    logging.info(
                        "Stimulation rate filter: block=%s threshold=%.6g Hz source=%s rates=%d",
                        block_name,
                        float(rate_filter_info.get("threshold_hz", 0.0)),
                        rate_filter_info.get("source_path", ""),
                        int(rate_filter_info.get("rate_count", 0)),
                    )
                    audit.mark_event(
                        "stimulation_rate_filter_ready",
                        block_name,
                        "",
                        "",
                        extra=rate_filter_info,
                    )
                electrode_group = _effective_electrode_group(
                    protocol,
                    electrode_group,
                    cfg_electrodes=cfg_electrodes if not dry_run else None,
                    stim_unit_by_electrode=runtime_stim_unit_by_electrode if not dry_run else None,
                    reserved_units=reserved_units,
                    recording_rates=stimulation_rates,
                    rate_threshold=stimulation_rate_threshold,
                    cfg_path=cfg_path if not dry_run else None,
                    system_config=system_config if not dry_run else None,
                )
                hardware_system_config = _hardware_system_config(system_config, protocol)
                prepared_stim_unit_by_electrode: dict[int, int] = {}
                prepared_stim_unit_to_dac: dict[int, int] = {}
                if not dry_run and not _protocol_uses_plan(protocol) and not is_record_only and not is_rest_only:
                    electrode_group, replacements, unresolved_electrodes = _replace_unconnectable_stimulation_electrodes(
                        cfg_path,
                        electrode_group,
                        system_config,
                        protocol,
                        rates=stimulation_rates,
                        rate_threshold=stimulation_rate_threshold,
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
                        stimulation_rates=stimulation_rates,
                        stimulation_rate_threshold=stimulation_rate_threshold,
                    )
                elif not dry_run:
                    prepare_recording_only(cfg_path, system_config)
                    audit.mark_event("recording_file_start", block_name, phase_id, segment_name, extra={"phase_mode": phase.get("mode", "")})
                    saving = create_experiment_saving(phase_dir, segment_name, record_channels_excluding(cfg_path, []))
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
    *,
    cfg_electrodes: set[int] | None = None,
    stim_unit_by_electrode: dict[int, int] | None = None,
) -> dict[str, int]:
    cfg_electrodes = cfg_electrodes if cfg_electrodes is not None else _cfg_recording_electrodes(cfg_path)
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
        effective_group = _effective_electrode_group(
            protocol,
            electrode_group,
            cfg_electrodes=sorted(cfg_electrodes),
            stim_unit_by_electrode=stim_unit_by_electrode,
            cfg_path=cfg_path,
        )
        if _scan_mode_enabled(protocol):
            route_groups = [
                {"name": f"{block_name}:scan_route_{index + 1}", "electrodes": list(route)}
                for index, route in enumerate(effective_group.get("scan_route_groups", []) or [])
                if route
            ]
            if not route_groups:
                raise RuntimeError(f"block {block_name}: scan mode did not produce any route groups")
            for route_group in route_groups:
                _validate_stimulation_unit_pool_size(protocol, route_group, context=f"block {block_name}")
        else:
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


def _cfg_electrode_id(value: Any, cfg_electrodes: set[int]) -> tuple[int, int | None]:
    """Keep the electrode ID from the H5/CFG mapping unchanged."""
    return int(value), None


def _normalize_stimulation_ids_for_cfg(
    groups: dict[str, dict[str, Any]],
    protocols: dict[str, dict[str, Any]],
    cfg_electrodes: set[int],
) -> dict[int, int]:
    """Normalize configured stimulation references before planning/routing."""
    mapping: dict[int, int] = {}

    def normalize_values(values: Any) -> list[int]:
        normalized = []
        for value in _event_group_values(values):
            target, source = _cfg_electrode_id(value, cfg_electrodes)
            normalized.append(target)
            if source is not None:
                mapping[source] = target
        return _unique_ints(normalized)

    for group in groups.values():
        if not isinstance(group, dict):
            continue
        if "electrodes" in group:
            group["electrodes"] = normalize_values(group.get("electrodes"))
        if group.get("center_electrode") is not None:
            center, source = _cfg_electrode_id(group["center_electrode"], cfg_electrodes)
            group["center_electrode"] = center
            if source is not None:
                mapping[source] = center

    for protocol in protocols.values():
        if not isinstance(protocol, dict):
            continue
        switch_cfg = protocol.get("site_switch")
        if isinstance(switch_cfg, dict):
            switch_cfg["event_groups"] = [normalize_values(raw) for raw in switch_cfg.get("event_groups") or []]
            centers = []
            for raw_center in switch_cfg.get("event_group_centers") or []:
                center = _normalize_site_switch_center(raw_center)
                if center is None:
                    centers.append(None)
                    continue
                normalized, source = _cfg_electrode_id(center, cfg_electrodes)
                centers.append(normalized)
                if source is not None:
                    mapping[source] = normalized
            if centers:
                switch_cfg["event_group_centers"] = centers
        pool_cfg = protocol.get("electrode_pool_sequence")
        if isinstance(pool_cfg, dict) and "event_groups" in pool_cfg:
            pool_cfg["event_groups"] = [normalize_values(raw) for raw in pool_cfg.get("event_groups") or []]
    return mapping


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


def _resolve_runtime_rate_source(path_text: str) -> Path:
    source = Path(str(path_text or "")).expanduser()
    if source.is_file():
        return source
    candidates = []
    if not source.is_absolute():
        candidates.extend((Path.cwd() / source, Path(__file__).resolve().parents[1] / source))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return source


def _stimulation_rate_filter(
    protocol: dict[str, Any],
    system_config: dict[str, Any] | None = None,
) -> tuple[dict[int, float], float | None, dict[str, Any]]:
    """Load recording rates and derive the minimum stimulation rate.

    Generated packages store a rate table from the selected spontaneous
    recording.  The default policy is deliberately explicit: a candidate must
    have a rate strictly above the finite-channel median.  Older packages that
    do not contain a rate source keep the historical unit/distance-only logic.
    """
    config = protocol.get("stimulation_rate_filter", {}) if isinstance(protocol, dict) else {}
    if not isinstance(config, dict):
        config = {}
    random_config = protocol.get("random_electrode_plan", {}) if isinstance(protocol, dict) else {}
    if not isinstance(random_config, dict):
        random_config = {}
    source_text = str(
        config.get("source_path")
        or protocol.get("recording_rate_source", "")
        or random_config.get("spontaneous_data_path", "")
        or ""
    ).strip()
    if not source_text:
        return {}, None, {"enabled": False, "reason": "no_rate_source"}
    source_path = _resolve_runtime_rate_source(source_text)
    if not source_path.is_file():
        raise FileNotFoundError(f"Stimulation rate source not found: {source_text}")
    rates = {
        int(electrode): float(rate)
        for electrode, rate in load_spontaneous_rates(source_path).items()
        if math.isfinite(float(rate))
    }
    if not rates:
        raise RuntimeError(f"Stimulation rate source contains no finite channel rates: {source_path}")
    mode = str(config.get("threshold_mode", "median") or "median").strip().lower()
    explicit = config.get("threshold_hz")
    if mode in {"none", "off", "disabled"}:
        threshold = None
    elif explicit is not None:
        threshold = float(explicit)
    elif mode == "median":
        threshold = float(statistics.median(rates.values()))
    else:
        raise ValueError(f"Unsupported stimulation rate threshold mode: {mode}")
    return rates, threshold, {
        "enabled": threshold is not None,
        "source_path": str(source_path),
        "threshold_mode": mode,
        "threshold_hz": threshold,
        "rate_count": len(rates),
    }


def _protocol_has_stimulation_rate_source(protocol: dict[str, Any]) -> bool:
    if not isinstance(protocol, dict):
        return False
    config = protocol.get("stimulation_rate_filter", {})
    random_config = protocol.get("random_electrode_plan", {})
    return bool(
        isinstance(config, dict)
        and str(config.get("source_path", "") or "").strip()
    ) or bool(
        str(protocol.get("recording_rate_source", "") or "").strip()
    ) or bool(
        isinstance(random_config, dict)
        and str(random_config.get("spontaneous_data_path", "") or "").strip()
    )


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
    stimulation_rates: dict[int, float] | None = None,
    stimulation_rate_threshold: float | None = None,
) -> None:
    # Runtime candidate selection is the executable stimulation definition.
    # Keep the original protocol for metadata, but feed this resolved copy to
    # every plan builder so center-only event groups are never routed by
    # accident.
    plan_protocol = _protocol_with_runtime_plan_groups(protocol, electrode_group)
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
                rates=stimulation_rates,
                rate_threshold=stimulation_rate_threshold,
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
        plan_protocol = dict(plan_protocol)
        random_cfg = dict(plan_protocol.get("random_electrode_plan", {}) or {})
        random_cfg["candidate_electrodes"] = [int(item) for item in filtered_group.get("electrodes", [])]
        plan_protocol["random_electrode_plan"] = random_cfg
        plan_rows = build_poisson_random_plan(
            protocol=plan_protocol,
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
            protocol=plan_protocol,
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
            protocol=plan_protocol,
            phase_dir=phase_dir,
            duration_s=duration_s,
            fallback_electrodes=filtered_group["electrodes"],
            stim_times_sec=base_stim_times,
        )
        stim_times = [float(row["time_sec"]) for row in plan_rows]
    else:
        plan_rows = []
        stim_times = get_stim_times_for_protocol(protocol, duration_s)
    if not dry_run and _protocol_uses_plan(protocol) and protocol.get("type") != "poisson_random_electrodes" and not _scan_mode_enabled(protocol):
        # Preserve the exact order that passed cumulative unit validation.
        # Maxwell may revise unit allocation when connection order changes.
        plan_electrodes = _unique_ints([
            *[int(item) for item in filtered_group.get("electrodes", [])],
            *[int(item) for row in plan_rows for item in _planned_electrodes([row], filtered_group)],
        ])
        electrode_center_lookup = _planned_electrode_centers(plan_rows, filtered_group)
        probe_group = dict(filtered_group)
        probe_group["electrodes"] = plan_electrodes
        filtered_group, replacements, unresolved_electrodes = _replace_unconnectable_stimulation_electrodes(
            cfg_path,
            probe_group,
            system_config,
            protocol,
            rates=stimulation_rates,
            rate_threshold=stimulation_rate_threshold,
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
    if not _scan_mode_enabled(protocol):
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
    connect_settle_ms = 0.0  # D21 switches on the host before waiting for the pulse.
    sequence = None
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
            if _scan_mode_enabled(protocol):
                scan_map = {
                    int(key): int(value)
                    for key, value in (filtered_group.get("scan_stim_unit_by_electrode", {}) or {}).items()
                }
                scan_route_groups = [
                    [int(value) for value in group]
                    for group in (filtered_group.get("scan_route_groups", []) or [])
                    if group
                ]
                if not scan_route_groups:
                    raise RuntimeError("Scan mode did not produce any hardware route groups")

                def configure_scan_route_hardware(route_index: int) -> None:
                    candidates = list(scan_route_groups[route_index])
                    for _attempt in range(len(candidates) + 1):
                        configure_group = dict(filtered_group)
                        configure_group["electrodes"] = candidates
                        (
                            _array,
                            _connected,
                            _skipped,
                            configured_units,
                            configured_unit_to_dac,
                        ) = resolve_experiment_array(
                            cfg_path,
                            configure_group,
                            effective_hardware_config,
                            return_hardware_mapping=True,
                            initial_connect=False,
                            require_unique_units=False,
                        )
                        unit_to_electrodes: dict[int, list[int]] = {}
                        for electrode in candidates:
                            if electrode in configured_units:
                                unit_to_electrodes.setdefault(int(configured_units[electrode]), []).append(electrode)
                        conflicts = {
                            unit: values for unit, values in unit_to_electrodes.items() if len(values) > 1
                        }
                        if conflicts:
                            keep: list[int] = []
                            seen_units: set[int] = set()
                            for electrode in candidates:
                                unit = configured_units.get(electrode)
                                if unit is None or int(unit) in seen_units:
                                    continue
                                keep.append(electrode)
                                seen_units.add(int(unit))
                            if len(keep) == len(candidates):
                                raise RuntimeError(
                                    "Scan route hardware returned unresolved stimulation-unit conflicts: "
                                    + "; ".join(f"unit {unit}: {values}" for unit, values in sorted(conflicts.items()))
                                )
                            candidates = keep
                            continue
                        scan_route_groups[route_index] = candidates
                        stim_unit_by_electrode.clear()
                        stim_unit_by_electrode.update({int(key): int(value) for key, value in configured_units.items()})
                        stim_unit_to_dac.clear()
                        stim_unit_to_dac.update({int(key): int(value) for key, value in configured_unit_to_dac.items()})
                        return
                    raise RuntimeError(f"Could not resolve a unique stimulation-unit route for scan band {route_index + 1}")

                configure_scan_route_hardware(0)
            else:
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
        else:
            _array, stim_unit_by_electrode, stim_unit_to_dac = configure_experiment_array_with_mapping(
                cfg_path, filtered_group, effective_hardware_config, initial_connect=True,
            )
        audit.mark_event("recording_file_start", block_name, phase_id, segment_name, extra={"phase_mode": "stimulation"})
        saving = create_experiment_saving(phase_dir, segment_name, record_channels_excluding(cfg_path, filtered_group.get("electrodes", [])))
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
            scan_route_groups = [
                [int(value) for value in group]
                for group in (filtered_group.get("scan_route_groups", []) or [])
                if group
            ]
            def configure_scan_route(event_electrodes, previous_route_key=None):
                if not scan_route_groups:
                    return previous_route_key
                event_set = {int(value) for value in event_electrodes}
                route_index = next(
                    (index for index, group in enumerate(scan_route_groups) if event_set.intersection(group)),
                    0,
                )
                if previous_route_key == route_index:
                    return previous_route_key
                configure_scan_route_hardware(route_index)
                audit.mark_event(
                    "scan_route_configure",
                    block_name,
                    phase_id,
                    segment_name,
                    extra={"route_index": route_index, "electrode_count": len(scan_route_groups[route_index])},
                )
                return route_index

            sequence = build_poisson_random_sequence(
                protocol,
                plan_rows,
                stim_unit_by_electrode,
                stim_unit_to_dac,
                effective_hardware_config,
                route_configurator=configure_scan_route if scan_route_groups else None,
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
    if sequence is not None:
        route_records = sequence.route_records
    for index, stim_time in enumerate(stim_times, start=1):
        plan_row = plan_rows[index - 1] if plan_rows else {}
        epoch_sec = (sequence.sent_epochs[index - 1] if sequence is not None
                     else sequence_start_epoch + stim_time)
        route_record = route_records[index - 1] if index <= len(route_records) else {}
        stim_extra = {
            "stim_index": index,
            "stim_time_sec": epoch_sec - segment_log.record_start_epoch,
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


def _runtime_group_electrodes(
    electrode_group: dict[str, Any],
    cfg_electrodes: list[int],
    stim_unit_by_electrode: dict[int, int] | None,
    reserved_units: set[int] | None = None,
    recording_rates: dict[int, float] | None = None,
    rate_threshold: float | None = None,
    cfg_path: Path | None = None,
    system_config: dict[str, Any] | None = None,
    selected_pool: list[int] | None = None,
) -> list[int] | None:
    # Keep this local guard for packages generated by older app versions that
    # may not have emitted the module-level constant.
    max_route_units = int(globals().get("MAX_STIMULATION_UNITS_PER_ROUTE", 32))
    if "center_electrode" not in electrode_group:
        return None
    center = _group_center_electrode(electrode_group)
    if center is None:
        raise RuntimeError(f"Stimulation group {electrode_group.get('name', '')!r} has no center_electrode")
    count = max(1, int(electrode_group.get("electrode_count", 1) or 1))
    if not bool(electrode_group.get("multi_electrode", False)):
        count = 1
    if count > max_route_units:
        raise RuntimeError(f"Stimulation group {electrode_group.get('name', '')!r} requests {count} electrodes; MaxOne supports at most {max_route_units} per route")
    if stim_unit_by_electrode is None:
        return [center]
    unit_map = {int(key): int(value) for key, value in stim_unit_by_electrode.items()}
    reserved = set(reserved_units or set())
    candidates = sorted(
        (
            int(electrode)
            for electrode in cfg_electrodes
            if int(electrode) in unit_map
            and (
                recording_rates is None
                or rate_threshold is None
                or float(recording_rates.get(int(electrode), float("-inf"))) > float(rate_threshold)
            )
        ),
        key=lambda electrode: (_electrode_grid_distance(center, electrode), electrode),
    )
    # The configured center is a spatial anchor, not a guarantee that the
    # hardware exposes a stimulation unit for that electrode. If it is absent
    # or already reserved, the nearest valid candidate becomes the first
    # actual stimulation electrode.
    selected: list[int] = []
    used_units: set[int] = set()
    joint_pool = selected_pool if selected_pool is not None else []
    final_joint_units: dict[int, int] = {}
    for electrode in candidates:
        if electrode in joint_pool or electrode in selected:
            continue
        unit = unit_map[electrode]
        if cfg_path is None and (unit in used_units or unit in reserved):
            continue
        if cfg_path is not None and system_config is not None:
            trial_electrodes = _unique_ints([*joint_pool, *selected, electrode])
            trial_group = {"name": electrode_group.get("name", ""), "electrodes": trial_electrodes}
            trial_connected, trial_missing, trial_units = probe_stimulation_electrodes(
                cfg_path,
                trial_group,
                system_config,
            )
            if trial_missing or trial_connected != trial_electrodes:
                continue
            if len(set(trial_units.values())) != len(trial_electrodes):
                continue
            final_joint_units = {int(key): int(value) for key, value in trial_units.items()}
        selected.append(electrode)
        used_units.add(final_joint_units.get(electrode, unit))
        if len(selected) >= count:
            break
    if len(selected) != count:
        rate_detail = ""
        if recording_rates is not None and rate_threshold is not None:
            rate_detail = f" and firing rate > {float(rate_threshold):.6g} Hz"
        raise RuntimeError(
            f"Stimulation group {electrode_group.get('name', '')!r} could only find {len(selected)} electrodes with unique units{rate_detail} near center {center}; requested {count}"
        )
    if reserved_units is not None:
        if final_joint_units:
            reserved_units.clear()
            reserved_units.update(final_joint_units.values())
        else:
            reserved_units.update(used_units)
    if selected_pool is not None:
        selected_pool.extend(electrode for electrode in selected if electrode not in selected_pool)
    return selected


def _scan_mode_factor(scan_cfg: dict[str, Any]) -> int:
    cfg = scan_cfg or {}
    raw = str(cfg.get("band_width", cfg.get("mode", "off")) or "off").strip().lower().replace("×", "x")
    if raw in {"", "off", "none", "0"}:
        return 0
    if raw in {"all", "full", "1", "1x"}:
        return 1
    if raw in {"2", "3", "4"}:
        return int(raw)
    match = re.fullmatch(r"([1-4])x", raw)
    if match:
        return max(1, min(4, int(match.group(1))))
    raise ValueError(f"Unsupported scan band width {raw!r}; use full, 2, 3, 4, or off")


def _scan_spatial_event_groups(
    cfg_electrodes: list[int],
    stim_unit_by_electrode: dict[int, int],
    scan_cfg: dict[str, Any],
) -> list[list[int]]:
    """Build local multi-electrode sites over equal-width spatial bands."""
    factor = _scan_mode_factor(scan_cfg)
    if factor <= 0:
        return []
    local_count = max(1, min(3, int(scan_cfg.get("local_electrodes", scan_cfg.get("electrode_count", 1)) or 1)))
    rows = 5
    columns = {1: 6, 2: 3, 3: 2}[local_count]
    candidates = sorted({int(value) for value in cfg_electrodes if int(value) in stim_unit_by_electrode})
    if not candidates:
        raise RuntimeError("Scan mode found no CFG electrodes with stimulation units")
    row_values = [int(value) // 220 for value in candidates]
    col_values = [int(value) % 220 for value in candidates]
    row_min, row_max = min(row_values), max(row_values)
    col_min, col_max = min(col_values), max(col_values)
    row_span = max(1.0, float(row_max - row_min + 1))
    col_span = max(1.0, float(col_max - col_min + 1))
    used_electrodes: set[int] = set()
    groups: list[list[int]] = []
    for partition in range(factor):
        # All sites in one band share a routed stimulation pool. Units may be
        # reused only after the next band is configured.
        used_partition_units: set[int] = set()
        part_start = row_min + row_span * partition / factor
        part_stop = row_min + row_span * (partition + 1) / factor
        part_candidates = [value for value in candidates if part_start <= value // 220 < part_stop or (partition == factor - 1 and value // 220 <= row_max)]
        if not part_candidates:
            continue
        part_height = max(1.0, part_stop - part_start)
        for row_index in range(rows):
            target_row = part_start + (row_index + 0.5) * part_height / rows
            for col_index in range(columns):
                target_col = col_min + (col_index + 0.5) * col_span / columns
                ranked = sorted(
                    part_candidates,
                    key=lambda value: (
                        (value // 220 - target_row) ** 2 + (value % 220 - target_col) ** 2,
                        int(stim_unit_by_electrode[value]),
                        value,
                    ),
                )
                selected: list[int] = []
                selected_units: set[int] = set()
                for value in ranked:
                    unit = int(stim_unit_by_electrode[value])
                    if value in used_electrodes or unit in used_partition_units or unit in selected_units:
                        continue
                    selected.append(int(value))
                    selected_units.add(unit)
                    if len(selected) >= local_count:
                        break
                if len(selected) < local_count:
                    continue
                used_electrodes.update(selected)
                used_partition_units.update(selected_units)
                groups.append(selected)
    if len(groups) < rows * columns * factor:
        logging.warning("Scan mode selected %d/%d spatial sites because CFG candidates were exhausted", len(groups), rows * columns * factor)
    return groups


def _scan_route_groups(scan_groups: list[list[int]], max_units: int = 30) -> list[list[int]]:
    """Pack complete local sites into hardware-sized route groups."""
    limit = max(1, int(max_units))
    routes: list[list[int]] = []
    current: list[int] = []
    current_set: set[int] = set()
    for raw_group in scan_groups:
        group = _unique_ints([int(value) for value in raw_group])
        if not group:
            continue
        group_set = set(group)
        if len(group_set) > limit:
            raise RuntimeError(f"Scan local site contains {len(group_set)} electrodes, exceeding route limit {limit}")
        if current and len(current_set | group_set) > limit:
            routes.append(current)
            current = []
            current_set = set()
        current.extend(value for value in group if value not in current_set)
        current_set.update(group_set)
    if current:
        routes.append(current)
    return routes


def _scan_mode_enabled(protocol: dict[str, Any]) -> bool:
    return _scan_mode_factor(protocol.get("scan", {}) if isinstance(protocol, dict) else {}) > 0


def _effective_electrode_group(
    protocol: dict[str, Any],
    electrode_group: dict[str, Any],
    *,
    cfg_electrodes: list[int] | None = None,
    stim_unit_by_electrode: dict[int, int] | None = None,
    reserved_units: set[int] | None = None,
    recording_rates: dict[int, float] | None = None,
    rate_threshold: float | None = None,
    cfg_path: Path | None = None,
    system_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fallback = [int(item) for item in electrode_group.get("electrodes", [])]
    scan_cfg = protocol.get("scan", {}) if isinstance(protocol, dict) else {}
    if isinstance(scan_cfg, dict) and _scan_mode_factor(scan_cfg) > 0:
        if cfg_electrodes is None or stim_unit_by_electrode is None:
            raise RuntimeError("Scan mode requires a loaded CFG and stimulation-unit scan")
        scan_groups = _scan_spatial_event_groups(cfg_electrodes, stim_unit_by_electrode, scan_cfg)
        if not scan_groups:
            raise RuntimeError("Scan mode could not select any stimulation sites")
        selection_protocol = dict(protocol)
        switch_cfg = dict(selection_protocol.get("site_switch", {}) or {})
        switch_cfg.update({
            "enabled": True,
            "selection_mode": str(scan_cfg.get("selection_mode", "random") or "random"),
            "random_seed": int(scan_cfg.get("random_seed", protocol.get("random_seed", 42))),
            "event_groups": scan_groups,
            "event_group_centers": [group[0] for group in scan_groups],
            "event_group_counts": [len(group) for group in scan_groups],
        })
        selection_protocol["site_switch"] = switch_cfg
        effective = dict(electrode_group)
        effective["electrodes"] = _unique_ints([item for group in scan_groups for item in group])
        effective["candidate_source"] = "cfg_spatial_scan"
        effective["resolved_plan_key"] = "site_switch"
        effective["resolved_plan_event_groups"] = scan_groups
        effective["resolved_plan_event_centers"] = [group[0] for group in scan_groups]
        effective["scan_route_groups"] = _scan_route_groups(scan_groups)
        effective["scan_stim_unit_by_electrode"] = {int(key): int(value) for key, value in stim_unit_by_electrode.items()}
        return effective
    runtime_electrodes = _runtime_group_electrodes(
        electrode_group,
        cfg_electrodes or [],
        stim_unit_by_electrode,
        reserved_units,
        recording_rates,
        rate_threshold,
    )
    if runtime_electrodes is not None:
        fallback = runtime_electrodes
        if not _protocol_uses_plan(protocol):
            effective = dict(electrode_group)
            effective["electrodes"] = runtime_electrodes
            effective["resolved_from_cfg_units"] = stim_unit_by_electrode is not None
            return effective
    selection_protocol = protocol
    resolved_plan_key: str | None = None
    resolved_plan_groups: list[list[int]] | None = None
    resolved_plan_centers: list[int | None] | None = None
    if cfg_electrodes is not None and stim_unit_by_electrode is not None and _protocol_uses_plan(protocol):
        selection_protocol = dict(protocol)
        for switch_key in ("site_switch", "electrode_pool_sequence"):
            switch_cfg = protocol.get(switch_key)
            if not isinstance(switch_cfg, dict) or not switch_cfg.get("event_groups"):
                continue
            groups_cfg = dict(switch_cfg)
            centers = list(groups_cfg.get("event_group_centers") or [])
            counts = list(groups_cfg.get("event_group_counts") or [])
            resolved_groups: list[list[int]] = []
            plan_reserved_units: set[int] = set()
            plan_selected_pool: list[int] = []
            for index, raw_group in enumerate(groups_cfg.get("event_groups") or []):
                center = _normalize_site_switch_center(centers[index] if index < len(centers) else None)
                count = max(1, int(counts[index])) if index < len(counts) else 1
                if center is None:
                    resolved_groups.append(_unique_ints(_event_group_values(raw_group)))
                    continue
                selected = _runtime_group_electrodes(
                    {
                        "name": f"{electrode_group.get('name', '')}:{index}",
                        "center_electrode": center,
                        "multi_electrode": count > 1,
                        "electrode_count": count,
                    },
                    cfg_electrodes,
                    stim_unit_by_electrode,
                    plan_reserved_units,
                    recording_rates,
                    rate_threshold,
                    cfg_path,
                    system_config,
                    plan_selected_pool,
                )
                resolved_groups.append(selected or [center])
            groups_cfg["event_groups"] = resolved_groups
            selection_protocol[switch_key] = groups_cfg
            resolved_plan_key = switch_key
            resolved_plan_groups = resolved_groups
            resolved_plan_centers = [
                _normalize_site_switch_center(centers[index] if index < len(centers) else None)
                for index in range(len(resolved_groups))
            ]
    if protocol.get("type") == "poisson_random_electrodes":
        candidate_electrodes = select_poisson_candidate_electrodes(protocol, fallback)
        source = "poisson_random_electrodes"
    elif protocol.get("type") == "electrode_pool_sequence":
        event_groups = electrode_pool_event_groups(selection_protocol, fallback)
        candidate_electrodes = _unique_ints([electrode for group in event_groups for electrode in group]) or fallback
        source = "electrode_pool_sequence"
    elif _protocol_uses_plan(protocol):
        stim_count = len(get_stim_times_for_protocol(protocol, 24 * 60 * 60))
        event_groups = site_switch_event_groups_for_count(selection_protocol, fallback, stim_count)
        candidate_electrodes = _unique_ints([electrode for group in event_groups for electrode in group]) or fallback
        source = "site_switch"
    else:
        return electrode_group
    effective = dict(electrode_group)
    effective["electrodes"] = candidate_electrodes
    effective["candidate_source"] = source
    if resolved_plan_key is not None and resolved_plan_groups is not None:
        # These are the actual CFG/unit-resolved electrodes to be used by the
        # pulse plan. Keep them on the effective group so plan generation does
        # not fall back to center-only event groups later in the run.
        effective["resolved_plan_key"] = resolved_plan_key
        effective["resolved_plan_event_groups"] = resolved_plan_groups
        effective["resolved_plan_event_centers"] = resolved_plan_centers or []
    return effective


def _protocol_with_runtime_plan_groups(
    protocol: dict[str, Any],
    electrode_group: dict[str, Any],
) -> dict[str, Any]:
    key = electrode_group.get("resolved_plan_key")
    groups = electrode_group.get("resolved_plan_event_groups")
    if not isinstance(key, str) or not isinstance(groups, list):
        return protocol
    updated = dict(protocol)
    config = dict(updated.get(key, {}) or {})
    config["event_groups"] = [list(map(int, group)) for group in groups]
    centers = electrode_group.get("resolved_plan_event_centers")
    if isinstance(centers, list):
        config["event_group_centers"] = [
            None if center is None else int(center) for center in centers
        ]
    updated[key] = config
    if key == "site_switch":
        updated["site_switch"] = config
    return updated


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
    *,
    rates: dict[int, float] | None = None,
    rate_threshold: float | None = None,
) -> tuple[dict[str, Any], dict[int, int], list[int]]:
    original_electrodes = [int(item) for item in electrode_group.get("electrodes", [])]
    random_cfg = protocol.get("random_electrode_plan", {})
    rate_map = {
        int(electrode): float(rate)
        for electrode, rate in (rates or {}).items()
        if math.isfinite(float(rate))
    }
    if not rate_map:
        rate_map, random_cfg = poisson_rates_for_electrodes(
            protocol,
            original_electrodes,
            restrict_to_fallback=False,
        )
    rate_filter_cfg = protocol.get("stimulation_rate_filter", {})
    rate_filter_disabled = (
        isinstance(rate_filter_cfg, dict)
        and str(rate_filter_cfg.get("threshold_mode", "")).strip().lower()
        in {"none", "off", "disabled"}
    )
    if rate_threshold is None and rate_map and not rate_filter_disabled:
        rate_threshold = float(statistics.median(rate_map.values()))
    floor = float(random_cfg.get("lambda_floor_hz", 0.001))
    return _replace_unconnectable_stimulation_electrodes(
        cfg_path,
        electrode_group,
        system_config,
        protocol,
        rates=rate_map,
        floor=floor,
        rate_threshold=rate_threshold,
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
    rate_threshold: float | None = None,
    max_radius: int | None = None,
    electrode_center_lookup: dict[int, int] | None = None,
) -> tuple[dict[str, Any], dict[int, int], list[int]]:
    original_electrodes = _unique_ints([int(item) for item in electrode_group.get("electrodes", [])])
    rate_map = {
        int(electrode): float(rate)
        for electrode, rate in (rates or {}).items()
        if math.isfinite(float(rate))
    }
    rate_filter_cfg = protocol.get("stimulation_rate_filter", {}) if isinstance(protocol, dict) else {}
    rate_filter_disabled = (
        isinstance(rate_filter_cfg, dict)
        and str(rate_filter_cfg.get("threshold_mode", "")).strip().lower()
        in {"none", "off", "disabled"}
    )
    if rate_threshold is None and rate_map and not rate_filter_disabled:
        rate_threshold = float(statistics.median(rate_map.values()))

    def _rate_ok(electrode: int) -> bool:
        if rate_threshold is None:
            return True
        rate = rate_map.get(int(electrode))
        return rate is not None and math.isfinite(float(rate)) and float(rate) > float(rate_threshold)

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
    low_rate_electrodes = [electrode for electrode in primary_electrodes if not _rate_ok(electrode)]
    if low_rate_electrodes:
        logging.warning(
            "Re-selecting stimulation electrodes below firing-rate threshold %.6g Hz: %s",
            float(rate_threshold),
            ",".join(str(item) for item in low_rate_electrodes),
        )
        low_rate_set = set(low_rate_electrodes)
        primary_electrodes = [electrode for electrode in primary_electrodes if electrode not in low_rate_set]
    unresolved_targets = _unique_ints([
        *missing_electrodes,
        *stim_unit_conflicts,
        *low_rate_electrodes,
    ])
    if not unresolved_targets:
        filtered = dict(electrode_group)
        filtered["electrodes"] = primary_electrodes
        if rate_threshold is not None:
            filtered["stimulation_rate_threshold_hz"] = float(rate_threshold)
        return filtered, {}, []

    search_radii = _replacement_search_radii(max_radius)
    used = set(primary_electrodes)
    missing_set = set(unresolved_targets)
    cfg_electrodes = _cfg_recording_electrodes(cfg_path)
    center_electrode = _group_center_electrode(electrode_group)
    center_lookup = {int(key): int(value) for key, value in (electrode_center_lookup or {}).items()}
    try:
        candidate_limit = max(1, int(maxwell_cfg.get("replacement_max_candidates", 128) or 128))
    except (TypeError, ValueError):
        candidate_limit = 128

    replacements: dict[int, int] = {}
    unresolved: list[int] = []
    for electrode in unresolved_targets:
        replacement = None
        target_center = center_lookup.get(int(electrode), center_electrode)
        for radius in ([None] if target_center is not None else search_radii):
            if target_center is not None:
                radius_candidates = cfg_electrodes
            else:
                radius_candidates = _electrode_neighbors(electrode, radius)
            ranked = sorted(
                (
                    candidate
                    for candidate in radius_candidates
                    if (
                        candidate in cfg_electrodes
                        and candidate not in used
                        and candidate not in missing_set
                        and _rate_ok(candidate)
                    )
                ),
                key=lambda candidate: (
                    _electrode_grid_distance(target_center if target_center is not None else electrode, candidate),
                    -float(rate_map.get(candidate, floor)),
                    candidate,
                ),
            )[:candidate_limit]
            for candidate in ranked:
                # Stimulation-unit allocation is stateful. Validate the
                # candidate together with every electrode already selected;
                # a unit observed in a separate probe Array is not reusable
                # evidence for the final route.
                trial_electrodes = _unique_ints([*primary_electrodes, *replacements.values(), candidate])
                trial_group = dict(electrode_group)
                trial_group["electrodes"] = trial_electrodes
                trial_connected, trial_missing, trial_units = probe_stimulation_electrodes(
                    cfg_path,
                    trial_group,
                    system_config,
                )
                if trial_missing or set(trial_connected) != set(trial_electrodes):
                    continue
                if len(set(trial_units.values())) != len(trial_electrodes):
                    continue
                replacement = candidate
                break
            if replacement is not None:
                break
        if replacement is None:
            unresolved.append(electrode)
            continue
        replacements[electrode] = replacement
        used.add(replacement)

    # Preserve the exact connection order used by the successful incremental
    # trials; changing it can cause Maxwell to choose a different allocation.
    resolved_electrodes = _unique_ints([*primary_electrodes, *replacements.values()])
    filtered = dict(electrode_group)
    filtered["electrodes"] = _unique_ints(resolved_electrodes)
    if rate_threshold is not None:
        filtered["stimulation_rate_threshold_hz"] = float(rate_threshold)
    if not filtered["electrodes"]:
        return filtered, replacements, _unique_ints(unresolved)
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
    final_low_rate = [electrode for electrode in final_primary if not _rate_ok(electrode)]
    if final_conflicts or final_low_rate:
        unresolved.extend(final_conflicts)
        unresolved.extend(final_low_rate)
        filtered["electrodes"] = [
            electrode for electrode in final_primary if _rate_ok(electrode)
        ]
    else:
        filtered["electrodes"] = final_primary
    return filtered, replacements, _unique_ints(unresolved)


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
'''.lstrip()


GENERATED_TIME_LOG = r'''
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SegmentStimLog:
    block: str
    phase: str
    segment_name: str
    record_start_epoch: float
    record_end_epoch: float | None = None
    stim_times_sec: list[float] = field(default_factory=list)
    stim_records: list[dict[str, Any]] = field(default_factory=list)
    hardware_mapping: dict[str, Any] = field(default_factory=dict)
    route_records: list[dict[str, Any]] = field(default_factory=list)

    def set_hardware_mapping(self, mapping: dict[str, Any] | None) -> None:
        self.hardware_mapping = dict(mapping or {})

    def add_route_record(self, record: dict[str, Any]) -> None:
        self.route_records.append(dict(record))

    def add_stim(self, epoch_sec: float, stim_index: int, extra: dict[str, Any] | None = None) -> None:
        stim_time_sec = round(float(epoch_sec - self.record_start_epoch), 6)
        self.stim_times_sec.append(stim_time_sec)
        record = {
            "stim_index": int(stim_index),
            "time_s": stim_time_sec,
            "epoch_sec": float(epoch_sec),
        }
        if extra:
            record.update(extra)
        self.stim_records.append(record)

    def save_txt(self, path: Path) -> None:
        lines = [
            "# stim_times.txt - times relative to THIS segment start (seconds)",
            f"# block: {self.block}",
            f"# phase: {self.phase}",
            f"# segment_name: {self.segment_name}",
            f"# record_start_epoch: {self.record_start_epoch}",
            f"# pulse_count: {len(self.stim_times_sec)}",
            f"# generated_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        ]
        sorted_records = sorted(self.stim_records, key=lambda item: float(item.get("time_s", 0.0)))
        if sorted_records:
            for record in sorted_records:
                electrodes = str(record.get("electrodes", "") or "")
                plan_time = record.get("plan_time_sec", "")
                lines.append(f"{float(record.get('time_s', 0.0)):.6f}\telectrodes={electrodes}\tplan_time_sec={plan_time}")
        else:
            lines.extend(f"{value:.6f}" for value in sorted(self.stim_times_sec))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def save_json(self, path: Path) -> None:
        data = {
            "block": self.block,
            "phase": self.phase,
            "segment_name": self.segment_name,
            "h5_basename": f"{self.segment_name}.raw.h5",
            "record_start_epoch": self.record_start_epoch,
            "record_end_epoch": self.record_end_epoch,
            "stim_times_sec": sorted(self.stim_times_sec),
            "stim_records": sorted(self.stim_records, key=lambda item: float(item.get("time_s", 0.0))),
            "pulse_count": len(self.stim_times_sec),
            "hardware_mapping": self.hardware_mapping,
            "route_records": sorted(
                self.route_records,
                key=lambda item: float(item.get("pulse_time_s", item.get("time_s", 0.0))),
            ),
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        if self.hardware_mapping:
            (path.parent / "hardware_mapping.json").write_text(
                json.dumps(self.hardware_mapping, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        if self.route_records:
            (path.parent / "hardware_route_log.json").write_text(
                json.dumps(self.route_records, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )


@dataclass
class ExternalTimeLog:
    run_dir: Path
    events: list[dict[str, Any]] = field(default_factory=list)
    time_origin_epoch: float = field(default_factory=time.time)

    def mark_event(
        self,
        event_type: str,
        block: str,
        phase: str,
        segment_name: str,
        epoch_sec: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        now = time.time() if epoch_sec is None else float(epoch_sec)
        event = {
            "event_type": event_type,
            "block": block,
            "phase": phase,
            "segment_name": segment_name,
            "wall_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "wall_time_local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            "epoch_sec": now,
            "offset_sec": now - self.time_origin_epoch,
        }
        if extra:
            event.update(extra)
        self.events.append(event)

    def save(self) -> None:
        if not self.events:
            return
        fieldnames: list[str] = []
        for event in self.events:
            for key in event:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (self.run_dir / "external_time_table.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.events)
        (self.run_dir / "external_time_table.json").write_text(json.dumps(self.events, indent=2, ensure_ascii=False), encoding="utf-8")
'''.lstrip()


GENERATED_RANDOM_STIM_PLAN = r'''
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
    explicit = _unique_ints([int(value) for value in random_cfg.get("candidate_electrodes", []) or []])
    candidate_electrodes = explicit or select_candidate_electrodes(
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
                    "pulse_width_us": float(protocol.get("pulse_width_us", 200.0)),
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
                "pulse_width_us": float(protocol.get("pulse_width_us", 200.0)),
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
            "pulse_width_us": float(protocol.get("pulse_width_us", 200.0)),
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
    if protocol_type in {"individual_burst", "sequence_with_burst", "random", "sequence_with_poisson_burst"}:
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
    if mode in {"scan_band_sequence", "band_sequence"}:
        return [
            list(group)
            for group, quota in zip(groups, quotas)
            for _repeat in range(quota)
        ]
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
        interval_s = (2.0 * float(row.get("pulse_width_us", 200.0)) + max(0.0, float(inter_phase_interval_us))) / 1_000_000.0
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


def select_poisson_candidate_electrodes(protocol: dict[str, Any], fallback_electrodes: list[int]) -> list[int]:
    random_cfg = protocol.get("random_electrode_plan", {}) if isinstance(protocol, dict) else {}
    explicit = _unique_ints([int(value) for value in random_cfg.get("candidate_electrodes", []) or []])
    if explicit:
        return explicit
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


def select_candidate_electrodes(
    rates: dict[int, float],
    region_count: int,
    max_candidates: int,
    electrode_order: list[int] | None = None,
) -> list[int]:
    if region_count <= 0:
        region_count = 32
    order_lookup = {int(electrode): index for index, electrode in enumerate(_unique_ints([int(value) for value in electrode_order or []]))}
    sorted_items = sorted(
        rates.items(),
        key=lambda item: (
            order_lookup.get(int(item[0]), len(order_lookup)),
            int(item[0]),
        ),
    )
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
'''


def poisson_candidate_electrodes_for_protocol(protocol: StimulusProtocol, fallback_electrodes: list[int]) -> list[int]:
    if protocol.type != "poisson_random_electrodes":
        return list(fallback_electrodes)
    explicit = _unique_ints([int(value) for value in getattr(protocol, "poisson_candidate_electrodes", []) or []])
    if explicit:
        return explicit
    rates = _preview_spontaneous_rates(protocol)
    fallback = [int(electrode) for electrode in fallback_electrodes]
    if not rates:
        rates = {int(electrode): float(protocol.lambda_floor_hz) for electrode in fallback}
    candidates = _preview_candidate_electrodes(rates, protocol.region_count, protocol.max_candidate_electrodes)
    if not candidates:
        raise ValueError(f"No poisson candidate electrodes could be selected for protocol {protocol.name}")
    return candidates


def electrode_pool_candidate_electrodes_for_protocol(protocol: StimulusProtocol, fallback_electrodes: list[int]) -> list[int]:
    fallback = _unique_ints([int(electrode) for electrode in fallback_electrodes])
    if protocol.type != "electrode_pool_sequence" and not _protocol_uses_site_switch_config(protocol):
        return fallback
    if not protocol.pool_event_groups:
        return fallback
    if _protocol_uses_site_switch_config(protocol):
        event_count = len(pulse_starts_ms(protocol))
        event_groups = site_switch_event_groups_for_count(protocol, fallback, event_count)
    else:
        event_groups = electrode_pool_event_groups(protocol, fallback)
    union: list[int] = []
    seen: set[int] = set()
    for group in event_groups:
        for electrode in group:
            value = int(electrode)
            if value not in seen:
                seen.add(value)
                union.append(value)
    if not union:
        raise ValueError(f"Electrode pool protocol {protocol.name} generated an empty auto site group")
    return union


GENERATED_MAXWELL_SETUP = r'''
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


def enable_stimulation_power() -> None:
    mx = _mx()
    response = mx.send(mx.Core().enable_stimulation_power(True))
    if response != "Ok":
        raise RuntimeError(f"MaxLab stimulation power enable failed: {response}")


def initialize_maxlab(
    system_config: dict[str, Any] | None = None,
    *,
    power_up_stimulation: bool = True,
) -> None:
    mx = _mx()
    mx.initialize()
    time.sleep(mx.Timing.waitInit)
    if power_up_stimulation:
        enable_stimulation_power()
    config = (system_config or {}).get("maxwell", {})
    mx.send(mx.Amplifier().set_gain(int(config.get("amplifier_gain", 512))))
    mx.set_event_threshold(float(config.get("event_threshold", 5.5)))


def _event(properties: str, event_id: int) -> Any:
    mx = _mx()
    tokens = str(properties).split()
    # Maxwell accepts only key-value pairs. Keep diagnostics best-effort so a
    # malformed event label never aborts an otherwise valid stimulation.
    if len(tokens) % 2:
        tokens.append("none")
    return mx.Event(0, 1, event_id, " ".join(tokens))


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
    units = list(dict.fromkeys(int(value) for value in stim_unit_by_electrode.values()))
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


class _HostPulseSequence:
    """D21: switch immediately, wait for the deadline, send one pulse."""

    def __init__(self, protocol, rows, unit_map, unit_sources, system_config, *, switch, route_configurator=None):
        self.protocol = protocol
        self.rows = sorted(rows, key=lambda row: float(row["time_sec"]))
        self.unit_map = unit_map
        self.unit_sources = unit_sources
        self.system_config = system_config or {}
        self.switch = switch
        self.route_configurator = route_configurator
        self.sent_epochs = []
        self.route_records = []

    def send(self):
        mx = _mx()
        signal, neutral, sync = _hardware_dac_config(self.system_config)
        hold = sorted(set(signal) | {neutral})
        previous = set()
        previous_sent = None
        previous_plan = 0.0
        configured_route_key = None
        origin = time.time()
        sample_us = float(self.system_config.get("maxwell", {}).get("sample_us", 50.0))
        if sample_us <= 0:
            raise ValueError("sample_us must be positive")
        self.sent_epochs = []
        self.route_records = []

        def zero():
            seq = mx.Sequence(initial_delay=0, persistent=False)
            _append_dac_codes(seq, hold, 512, sync_dual_dac=sync)
            seq.send()

        try:
            for index, row in enumerate(self.rows, 1):
                electrodes = _row_electrodes(row)
                connected_at = None
                if self.switch:
                    if self.route_configurator is not None:
                        configured_route_key = self.route_configurator(electrodes, configured_route_key)
                    electrodes = [electrode for electrode in electrodes if electrode in self.unit_map]
                units = {int(self.unit_map[e]) for e in electrodes}
                sources = (_event_unit_dac_sources(electrodes, self.unit_map, signal)
                           if self.switch else self.unit_sources)
                active = sorted({sources[u] for u in units})
                if row.get("channel") is not None:
                    active = [int(row["channel"])]
                if self.switch:
                    zero()
                    for unit in sorted(previous - units):
                        mx.send(mx.StimulationUnit(unit).connect(False))
                    for electrode in sorted(set(electrodes)):
                        unit = int(self.unit_map[electrode])
                        mx.send(mx.StimulationUnit(unit).power_up(True).connect(True)
                                .set_voltage_mode().dac_source(sources[unit]))
                    connected_at = time.time()
                planned = float(row["time_sec"])
                deadline = (origin + planned if previous_sent is None else
                            previous_sent + max(0.0, planned - previous_plan))
                time.sleep(max(0.0, deadline - time.time()))
                amplitude = float(row.get("amplitude_mv", self.protocol.get("amplitude_mv", 150.0)))
                lsb = float(mx.query_DAC_lsb_mV())
                if lsb <= 0:
                    raise ValueError("DAC LSB must be positive")
                bits = int(round(abs(amplitude) / lsb))
                if not 1 <= bits <= 511:
                    raise ValueError(f"Stimulation amplitude out of DAC range: {amplitude}")
                width = float(row.get("pulse_width_us", self.protocol.get("pulse_width_us", 200.0)))
                phase_samples = max(1, int(round(width / sample_us)))
                pulse = mx.Sequence(initial_delay=100, persistent=False)
                electrode_text = "-".join(map(str, electrodes)) or "none"
                pulse.append(_event(f"type stim mode {'pool_fast_switch' if self.switch else 'single_connect'} "
                                    f"pulse {index}/{len(self.rows)} electrodes {electrode_text}", index))
                _append_dac_codes(pulse, hold, 512, sync_dual_dac=sync)
                inactive = sorted(set(hold) - set(active))
                for code in (512 - bits, 512 + bits):
                    _append_dac_codes(pulse, active, code, sync_dual_dac=sync)
                    _append_dac_codes(pulse, inactive, 512, sync_dual_dac=sync)
                    pulse.append(mx.DelaySamples(phase_samples))
                _append_dac_codes(pulse, hold, 512, sync_dual_dac=sync)
                sent = time.time()
                pulse.send()
                self.sent_epochs.append(sent)
                self.route_records.append({
                    "stim_index": index, "pulse_time_s": sent - origin,
                    "route_switch": self.switch, "connect_epoch": connected_at,
                    "connect_time_s": None if connected_at is None else connected_at - origin,
                    "settle_ms": 0.0 if connected_at is None else (sent - connected_at) * 1000.0,
                    "previous_stim_units": sorted(previous), "target_stim_units": sorted(units),
                    "active_dac_sources": active,
                    "stim_unit_to_event_dac": {u: sources[u] for u in units},
                    "electrodes": electrodes, "timing_source": "host_send",
                })
                previous = units
                previous_sent, previous_plan = sent, planned
        finally:
            if self.switch:
                zero()
                for unit in sorted(set(self.unit_map.values())):
                    mx.send(mx.StimulationUnit(unit).connect(False))
        time.sleep(float(self.protocol.get("tail_wait_sec", 0.5)))


def build_stim_sequence(
    protocol, electrode_group_name, *, stim_unit_by_electrode=None,
    stim_unit_to_dac=None, electrode_group_electrodes=None, system_config=None,
):
    if protocol.get("type") == "custom_sequence":
        rows = [dict(point, time_sec=float(point["time_ms"]) / 1000.0,
                     pulse_width_us=point.get("duration_us", protocol.get("pulse_width_us", 200.0)))
                for point in protocol.get("custom_points", [])]
    else:
        rows = [{"time_sec": value / 1000.0} for value in _scheduled_stim_times_ms(protocol)]
    for row in rows:
        row["electrodes"] = list(electrode_group_electrodes or [])
    unit_map = stim_unit_by_electrode or {}
    sources = stim_unit_to_dac or _default_stim_unit_dac_sources(unit_map, system_config)
    return _HostPulseSequence(protocol, rows, unit_map, sources, system_config, switch=False)


def build_poisson_random_sequence(
    protocol, plan_rows, stim_unit_by_electrode, stim_unit_to_dac=None, system_config=None,
    route_configurator=None,
):
    return _HostPulseSequence(protocol, plan_rows, stim_unit_by_electrode,
                              stim_unit_to_dac or {}, system_config, switch=True,
                              route_configurator=route_configurator)


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
    require_unique_units: bool = True,
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
    stim_unit_by_electrode: dict[int, int] = {}
    connected_electrodes: list[int] = []
    skipped_electrodes: list[int] = []
    # Maxwell can revise an earlier allocation when a later electrode is
    # connected. Query only after every connection request has completed so
    # the returned mapping describes the final shared Array state.
    for electrode in electrodes:
        try:
            array.connect_electrode_to_stimulation(electrode)
        except Exception:
            if allow_missing_electrodes:
                skipped_electrodes.append(electrode)
                continue
            raise
        connected_electrodes.append(electrode)
    queried_electrodes: list[int] = []
    for electrode in connected_electrodes:
        try:
            stim_unit = array.query_stimulation_at_electrode(electrode)
            if stim_unit is None or (isinstance(stim_unit, str) and not stim_unit.strip()):
                raise RuntimeError(f"No stimulation unit can connect to electrode {electrode}")
            stim_unit_by_electrode[electrode] = int(stim_unit)
            queried_electrodes.append(electrode)
        except Exception:
            if allow_missing_electrodes:
                skipped_electrodes.append(electrode)
                continue
            raise
    connected_electrodes = queried_electrodes
    if require_unique_units:
        unit_to_electrodes: dict[int, list[int]] = {}
        for electrode, stim_unit in stim_unit_by_electrode.items():
            unit_to_electrodes.setdefault(int(stim_unit), []).append(int(electrode))
        conflicts = {unit: mapped for unit, mapped in unit_to_electrodes.items() if len(mapped) > 1}
        if conflicts:
            details = "; ".join(f"unit {unit}: {mapped}" for unit, mapped in sorted(conflicts.items()))
            raise RuntimeError(
                "Each stimulation electrode must have a unique stimulation unit; "
                f"allocation conflicts detected ({details})"
            )
    stim_units = list(stim_unit_by_electrode.values())
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
    array.download()
    time.sleep(mx.Timing.waitAfterDownload)
    mx.offset()
    device = str(system_config.get("maxwell", {}).get("device", "maxone")).lower()
    offset_wait = mx.Timing.waitInMX2Offset if device == "maxtwo" else mx.Timing.waitInMX1Offset
    time.sleep(offset_wait + getattr(mx.Timing, "waitAfterOffset", 0.0))
    signal, neutral, sync = _hardware_dac_config(system_config)
    zero = mx.Sequence(initial_delay=0, persistent=False)
    _append_dac_codes(zero, sorted(set(signal) | {neutral}), 512, sync_dual_dac=sync)
    zero.send()
    _signal_dacs, neutral_dac, _sync_dual_dac = _hardware_dac_config(system_config)
    for stim_unit in dict.fromkeys(stim_units):
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
    mx.clear_events()
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
            require_unique_units=False,
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
            require_unique_units=False,
        )
        connected_all.extend(int(item) for item in connected)
        skipped_all.extend(int(item) for item in skipped)
        stim_units_all.update({int(electrode): int(stim_unit) for electrode, stim_unit in stim_units.items()})
    return connected_all, skipped_all, stim_units_all


def probe_stimulation_electrode_detail(
    cfg_path: Path,
    electrode: int,
    system_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the uncollapsed reason for one electrode's unit allocation."""
    detail: dict[str, Any] = {
        "electrode": int(electrode),
        "cfg_present": bool(cfg_path.is_file()),
        "cfg_loaded": False,
        "connect_ok": False,
        "query_ok": False,
        "status": "unknown",
        "stim_unit": None,
        "query_raw_repr": None,
        "reset_error": None,
        "load_error": None,
        "connect_error": None,
        "query_error": None,
    }
    if not detail["cfg_present"]:
        detail["status"] = "cfg_missing"
        return detail
    try:
        mx = _mx()
        array = mx.Array("stimulation")
        try:
            array.reset()
            array.clear_selected_electrodes()
        except Exception as exc:
            detail["reset_error"] = f"{type(exc).__name__}: {exc}"
            detail["status"] = "reset_failed"
            return detail
        try:
            array.load_config(str(cfg_path))
            detail["cfg_loaded"] = True
        except Exception as exc:
            detail["load_error"] = f"{type(exc).__name__}: {exc}"
            detail["status"] = "cfg_load_failed"
            return detail
        try:
            array.connect_electrode_to_stimulation(int(electrode))
            detail["connect_ok"] = True
        except Exception as exc:
            detail["connect_error"] = f"{type(exc).__name__}: {exc}"
            detail["status"] = "connect_failed"
            return detail
        try:
            raw_unit = array.query_stimulation_at_electrode(int(electrode))
            detail["query_raw_repr"] = repr(raw_unit)
            if raw_unit is None or (isinstance(raw_unit, str) and not raw_unit.strip()):
                detail["status"] = "no_stimulation_unit"
                return detail
            detail["stim_unit"] = int(raw_unit)
            detail["query_ok"] = True
            detail["status"] = "assigned_unit"
            return detail
        except Exception as exc:
            detail["query_error"] = f"{type(exc).__name__}: {exc}"
            detail["status"] = "query_failed"
            return detail
    except Exception as exc:
        detail["status"] = "probe_failed"
        detail["probe_error"] = f"{type(exc).__name__}: {exc}"
        return detail


def scan_cfg_stimulation_units(
    cfg_path: Path,
    cfg_electrodes: list[int],
    system_config: dict[str, Any],
) -> tuple[dict[int, int], list[int], list[dict[str, Any]]]:
    """Probe CFG electrodes one at a time and return the hardware unit map.

    Maxwell allocates stimulation units from the loaded CFG rather than from
    the numeric electrode ID.  Probing one electrode per temporary Array is
    deliberately conservative: it prevents a multi-electrode probe from
    creating duplicate-unit false positives while selecting a stimulation
    group.
    """
    unit_map: dict[int, int] = {}
    unresolved: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    for electrode in sorted({int(value) for value in cfg_electrodes}):
        detail = probe_stimulation_electrode_detail(cfg_path, electrode, system_config)
        if detail.get("status") == "assigned_unit" and detail.get("stim_unit") is not None:
            unit_map[int(electrode)] = int(detail["stim_unit"])
        else:
            unresolved.append(int(electrode))
        diagnostics.append(detail)
    return unit_map, unresolved, diagnostics


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


def record_channels_excluding(cfg_path: Path, electrodes: list[int]) -> list[int]:
    import re
    excluded = set(map(int, electrodes))
    text = cfg_path.read_text(encoding="utf-8", errors="replace")
    channels = sorted({int(ch) for ch, el in re.findall(r"(\d+)\((\d+)\)", text)
                       if int(el) not in excluded})
    if not channels:
        raise ValueError("No recording channels remain in the CFG mapping")
    return channels


def prepare_recording_only(cfg_path: Path, system_config: dict[str, Any]) -> None:
    mx = _mx()
    array = mx.Array("stimulation")
    array.load_config(str(cfg_path))
    mx.activate([0])
    array.download()
    time.sleep(mx.Timing.waitAfterDownload)
    mx.offset()
    device = str(system_config.get("maxwell", {}).get("device", "maxone")).lower()
    offset_wait = mx.Timing.waitInMX2Offset if device == "maxtwo" else mx.Timing.waitInMX1Offset
    time.sleep(offset_wait + getattr(mx.Timing, "waitAfterOffset", 0.0))
    mx.clear_events()


def create_experiment_saving(run_dir: Path, file_name: str, record_channels: list[int]) -> Any:
    mx = _mx()
    saving = mx.Saving()
    saving.open_directory(str(run_dir))
    saving.group_delete_all()
    saving.group_define(0, "exp", record_channels)
    saving.start_file(file_name)
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
    elif ptype == "random":
        burst_starts = _random_burst_starts_ms(protocol)
        if _randomize_burst_pulse_intervals(protocol):
            rng = random.Random(_burst_pulse_interval_seed(protocol))
            for burst_start in burst_starts:
                for offset_ms in _burst_pulse_offsets_ms(protocol, rng):
                    times_ms.append(burst_start + offset_ms)
        else:
            interval = _pulse_interval_ms(protocol)
            for burst_start in burst_starts:
                for pulse_index in range(int(protocol.get("pulses_per_burst", 5))):
                    times_ms.append(burst_start + pulse_index * interval)
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
    random_cfg = protocol.get("random", {}) or {}
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
            "pulse_width_us": protocol.get("pulse_width_us", 200.0),
            "random_distribution": random_cfg.get("distribution", protocol.get("random_distribution", "poisson")),
            "random_lambda_hz": random_cfg.get("lambda_hz", protocol.get("random_lambda_hz", 5.0)),
            "random_interval_min_ms": random_cfg.get("interval_min_ms", protocol.get("random_interval_min_ms", 100.0)),
            "random_interval_max_ms": random_cfg.get("interval_max_ms", protocol.get("random_interval_max_ms", 1000.0)),
            "random_duration_s": random_cfg.get("duration_s", protocol.get("random_duration_s", 60.0)),
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


def _random_burst_starts_ms(protocol: dict[str, Any]) -> list[float]:
    random_cfg = protocol.get("random", {}) or {}
    start_ms = float(protocol.get("start_ms", 0.0))
    duration_s = max(0.0, float(random_cfg.get("duration_s", protocol.get("duration_s", 60.0))))
    if duration_s <= 0.0:
        return [start_ms]
    stop_ms = start_ms + duration_s * 1000.0
    distribution = str(random_cfg.get("distribution", protocol.get("random_distribution", "poisson")) or "poisson").strip().lower()
    lambda_hz = max(float(random_cfg.get("lambda_hz", protocol.get("random_lambda_hz", 5.0))), 1e-9)
    minimum = max(0.0, float(random_cfg.get("interval_min_ms", protocol.get("random_interval_min_ms", 100.0))))
    maximum = max(minimum, float(random_cfg.get("interval_max_ms", protocol.get("random_interval_max_ms", 1000.0))))
    seed_payload = json.dumps(
        {
            "name": protocol.get("name", ""),
            "type": protocol.get("type", ""),
            "start_ms": start_ms,
            "burst_count": int(protocol.get("burst_count", 3)),
            "burst_frequency_hz": float(protocol.get("burst_frequency_hz", 5.0)),
            "random_distribution": distribution,
            "random_lambda_hz": lambda_hz,
            "random_interval_min_ms": minimum,
            "random_interval_max_ms": maximum,
            "random_duration_s": duration_s,
            "pulses_per_burst": int(protocol.get("pulses_per_burst", 5)),
            "pulse_frequency_hz": float(protocol.get("pulse_frequency_hz", 20.0)),
            "randomize_burst_pulse_intervals": protocol.get("randomize_burst_pulse_intervals", False),
            "burst_pulse_interval_min_ms": float(protocol.get("burst_pulse_interval_min_ms", 10.0)),
            "burst_pulse_interval_max_ms": float(protocol.get("burst_pulse_interval_max_ms", 100.0)),
            "random_seed": protocol.get("random_seed", random_cfg.get("random_seed", 42)),
            "amplitude_mv": float(protocol.get("amplitude_mv", 150.0)),
            "pulse_width_us": float(protocol.get("pulse_width_us", 200.0)),
        },
        sort_keys=True,
    )
    seed = int(hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    starts = [start_ms]
    current_ms = start_ms
    for _index in range(200_000):
        if distribution in {"poisson", "exponential"}:
            interval_ms = minimum + rng.expovariate(lambda_hz) * 1000.0
        elif distribution in {"uniform", "flat"}:
            interval_ms = rng.uniform(minimum, maximum)
        else:
            raise ValueError(f"Unsupported random burst distribution: {distribution}")
        current_ms += max(0.001, interval_ms)
        if current_ms > stop_ms:
            break
        starts.append(current_ms)
    return starts


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
                "pulse_width_us": float(protocol.get("pulse_width_us", 200.0)),
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
'''.lstrip()
