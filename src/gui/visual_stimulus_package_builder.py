from __future__ import annotations

import datetime as dt
import hashlib
import json
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
    "sequence_with_poisson_burst",
    "custom_sequence",
    "poisson_random_electrodes",
)
PHASES = ("01_pre_spont", "02_stim", "03_post_spont")


@dataclass
class ExperimentInfo:
    name: str = "new_maxwell_experiment"
    culture_id: str = ""
    div: str = ""
    date: str = field(default_factory=lambda: dt.date.today().isoformat())
    recording_prefix: str = "recording"
    scientific_question: str = ""
    closed_loop_logic: str = ""
    expected_output: str = ""
    cfg_path: str = "./config/system.cfg"
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

    def to_yaml(self) -> dict[str, Any]:
        return {"name": self.name, "electrodes": self.electrodes}


@dataclass
class StimulusProtocol:
    name: str
    type: str = "single_pulse"
    amplitude_mv: float = 150.0
    pulse_width_us: float = 300.0
    inter_phase_interval_us: float = 0.0
    pulse_frequency_hz: float = 20.0
    pulses_per_burst: int = 5
    interpulse_interval_ms: float = 50.0
    burst_count: int = 3
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
    poisson_candidate_electrodes: list[int] = field(default_factory=list)
    notes: str = ""

    def to_yaml(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "amplitude_mv": self.amplitude_mv,
            "pulse_width_us": self.pulse_width_us,
            "inter_phase_interval_us": self.inter_phase_interval_us,
        }
        if self.type != "poisson_random_electrodes":
            data.update(
                {
                    "pulse_frequency_hz": self.pulse_frequency_hz,
                    "pulses_per_burst": self.pulses_per_burst,
                    "interpulse_interval_ms": self.interpulse_interval_ms,
                    "burst_count": self.burst_count,
                    "burst_interval_ms": self.burst_interval_ms,
                    "start_ms": self.start_ms,
                    "channel": self.channel,
                }
            )
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
) -> list[dict[str, Any]]:
    if protocol.type in {"single_pulse", "individual_burst", "sequence_with_burst", "sequence_with_poisson_burst"}:
        return [{
            "label": f"ch{protocol.channel}",
            "channel": protocol.channel,
            "times_ms": [time_ms for time_ms, _amp in pulse_starts_ms(protocol) if 0.0 <= time_ms <= preview_limit_ms],
        }]

    if protocol.type == "custom_sequence":
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
            candidate_electrodes = _preview_candidate_electrodes(rates, protocol.region_count, protocol.max_candidate_electrodes)
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

    return [{"label": f"ch{protocol.channel}", "channel": protocol.channel, "times_ms": []}]


def pulse_starts_ms(protocol: StimulusProtocol) -> list[tuple[float, float]]:
    if protocol.type == "single_pulse":
        return [(protocol.start_ms, protocol.amplitude_mv)]
    if protocol.type == "individual_burst":
        interval = _pulse_interval_ms(protocol)
        return [(protocol.start_ms + i * interval, protocol.amplitude_mv) for i in range(protocol.pulses_per_burst)]
    if protocol.type == "sequence_with_burst":
        return _burst_pulse_starts(protocol, poisson=False)
    if protocol.type == "sequence_with_poisson_burst":
        return _burst_pulse_starts(protocol, poisson=True)
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
    return []


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


def _preview_candidate_electrodes(rates: dict[int, float], region_count: int, max_candidates: int) -> list[int]:
    sorted_items = sorted(rates.items(), key=lambda item: item[0])
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
            "burst_interval_ms": protocol.burst_interval_ms,
            "pulses_per_burst": protocol.pulses_per_burst,
            "pulse_frequency_hz": protocol.pulse_frequency_hz,
            "interpulse_interval_ms": protocol.interpulse_interval_ms,
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
        mean_ms = max(float(protocol.burst_interval_ms), 0.001)
        mean_s = mean_ms / 1000.0
        current_ms = float(protocol.start_ms)
        for _index in range(1, burst_count):
            current_ms += rng.expovariate(1.0 / mean_s) * 1000.0
            starts.append(current_ms)
        return starts
    burst_interval = max(float(protocol.burst_interval_ms), 0.0)
    return [float(protocol.start_ms) + burst_index * burst_interval for burst_index in range(burst_count)]


def _burst_pulse_starts(protocol: StimulusProtocol, *, poisson: bool) -> list[tuple[float, float]]:
    starts = _burst_starts_ms(protocol, poisson=poisson)
    pulse_interval = _pulse_interval_ms(protocol)
    return [
        (burst_start + pulse_index * pulse_interval, protocol.amplitude_mv)
        for burst_start in starts
        for pulse_index in range(max(1, int(protocol.pulses_per_burst)))
    ]


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
    if protocol.interpulse_interval_ms > 0:
        return protocol.interpulse_interval_ms
    return 1000.0 / max(protocol.pulse_frequency_hz, 0.001)


def _default_protocol_name(protocol_type: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(protocol_type or "protocol")).strip("_") or "protocol"
    return cleaned


def _protocol_seed(protocol: StimulusProtocol) -> int:
    payload = json.dumps(
        {
            "name": protocol.name,
            "type": protocol.type,
            "start_ms": protocol.start_ms,
            "burst_count": protocol.burst_count,
            "burst_interval_ms": protocol.burst_interval_ms,
            "pulses_per_burst": protocol.pulses_per_burst,
            "pulse_frequency_hz": protocol.pulse_frequency_hz,
            "interpulse_interval_ms": protocol.interpulse_interval_ms,
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
        mean_s = max(float(protocol.burst_interval_ms), 0.001) / 1000.0
        current_ms = float(protocol.start_ms)
        for _index in range(1, burst_count):
            current_ms += rng.expovariate(1.0 / mean_s) * 1000.0
            starts.append(current_ms)
        return starts
    burst_interval = max(float(protocol.burst_interval_ms), 0.0)
    return [float(protocol.start_ms) + burst_index * burst_interval for burst_index in range(burst_count)]


def _burst_pulse_starts(protocol: StimulusProtocol, *, poisson: bool) -> list[tuple[float, float]]:
    starts = _burst_starts_ms(protocol, poisson=poisson)
    pulse_interval = _pulse_interval_ms(protocol)
    return [
        (burst_start + pulse_index * pulse_interval, protocol.amplitude_mv)
        for burst_start in starts
        for pulse_index in range(max(1, int(protocol.pulses_per_burst)))
    ]


def build_package(
    output_dir: Path,
    info: ExperimentInfo,
    groups: list[ElectrodeGroup],
    protocols: list[StimulusProtocol],
    blocks: list[ExperimentBlock],
) -> Path:
    output_dir = output_dir.expanduser().resolve()
    _validate(info, groups, protocols, blocks)
    groups, blocks = _with_poisson_electrode_groups(groups, protocols, blocks)

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
    if not groups:
        raise ValueError("At least one electrode group is required")
    if not protocols:
        raise ValueError("At least one stimulation protocol is required")
    if not blocks:
        raise ValueError("At least one block is required")
    group_names = {group.name for group in groups}
    protocol_names = {protocol.name for protocol in protocols}
    for protocol in protocols:
        if protocol.type not in PROTOCOL_TYPES:
            raise ValueError(f"Unsupported protocol type: {protocol.type}")
        if protocol.type == "custom_sequence" and not protocol.custom_points:
            raise ValueError(f"Custom protocol {protocol.name} needs at least one point")
        if protocol.type == "poisson_random_electrodes" and protocol.max_candidate_electrodes > 32:
            raise ValueError("MaxOne-safe random stimulation supports at most 32 candidate electrodes")
    for block in blocks:
        if block.electrode_group not in group_names:
            raise ValueError(f"Block {block.name} references missing group {block.electrode_group}")
        if block.protocol not in protocol_names:
            raise ValueError(f"Block {block.name} references missing protocol {block.protocol}")
        if [phase.id for phase in block.phases] != list(PHASES):
            raise ValueError(f"Block {block.name} must contain fixed phases: {', '.join(PHASES)}")


def _with_poisson_electrode_groups(
    groups: list[ElectrodeGroup],
    protocols: list[StimulusProtocol],
    blocks: list[ExperimentBlock],
) -> tuple[list[ElectrodeGroup], list[ExperimentBlock]]:
    protocol_lookup = {protocol.name: protocol for protocol in protocols}
    group_lookup = {group.name: group for group in groups}
    resolved_groups = [ElectrodeGroup(group.name, list(group.electrodes)) for group in groups]
    resolved_group_lookup = {group.name: group for group in resolved_groups}
    resolved_blocks: list[ExperimentBlock] = []
    existing_names = {group.name for group in resolved_groups}

    for block in blocks:
        protocol = protocol_lookup.get(block.protocol)
        group = group_lookup.get(block.electrode_group)
        target_group = block.electrode_group
        if protocol is not None and group is not None and protocol.type == "poisson_random_electrodes":
            candidates = poisson_candidate_electrodes_for_protocol(protocol, list(group.electrodes))
            if not str(block.electrode_group).endswith(f"_{protocol.name}_auto"):
                target_group = _unique_group_name(f"{block.electrode_group}_{protocol.name}_auto", existing_names)
            else:
                target_group = block.electrode_group
            existing_names.add(target_group)
            target_electrodes = [int(value) for value in candidates]
            if target_group in resolved_group_lookup:
                resolved_group_lookup[target_group].electrodes = target_electrodes
            else:
                auto_group = ElectrodeGroup(target_group, target_electrodes)
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
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(base)).strip("_") or "poisson_auto"
    if cleaned not in existing_names:
        return cleaned
    index = 2
    while f"{cleaned}_{index}" in existing_names:
        index += 1
    return f"{cleaned}_{index}"


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
        f"- `{protocol.name}`: `{protocol.type}`, amplitude={protocol.amplitude_mv} mV, width={protocol.pulse_width_us} us"
        for protocol in protocols
    )
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

Each `main.py` execution creates `data/{{YYYYMMDD_HHMMSS}}_data/`. Every block has fixed `01_pre_spont`, `02_stim`, and `03_post_spont` phases. Each `02_stim` phase writes `stim_times.txt` with times in seconds relative to that segment start.
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

    try:
        cfg_path = Path(system_config.get("electrode_map", {}).get("cfg_path", ""))
        if cfg_path.exists():
            shutil.copy(cfg_path, run_dir / time.strftime("%Hh%Mm%Ss.cfg"))
        elif not args.dry_run:
            logging.warning("cfg_path does not exist: %s", cfg_path)
        snapshot = run_dir / "config_snapshot"
        snapshot.mkdir(exist_ok=True)
        shutil.copy(system_path, snapshot / "system.yaml")
        shutil.copy(stimulation_path, snapshot / "stimulation.yaml")
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

    def add_stim(self, epoch_sec: float, stim_index: int, extra: dict[str, Any] | None = None) -> None:
        self.stim_times_sec.append(round(float(epoch_sec - self.record_start_epoch), 6))

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
            "pulse_count": len(self.stim_times_sec),
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np


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
        "firing_rate_hz",
        "lambda_hz",
        "amplitude_mv",
        "pulse_width_us",
        "pulses_per_stimulus",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
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
    rates = _preview_spontaneous_rates(protocol)
    fallback = [int(electrode) for electrode in fallback_electrodes]
    if not rates:
        rates = {int(electrode): float(protocol.lambda_floor_hz) for electrode in fallback}
    candidates = _preview_candidate_electrodes(rates, protocol.region_count, protocol.max_candidate_electrodes)
    if not candidates:
        raise ValueError(f"No poisson candidate electrodes could be selected for protocol {protocol.name}")
    return candidates


GENERATED_MAXWELL_SETUP = r'''
from __future__ import annotations

import os
import sys
import time
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
    if cfg_path.exists():
        array.load_config(str(cfg_path))
    else:
        array.select_electrodes(electrodes)
        array.select_stimulation_electrodes(electrodes)
        array.route()
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
        interval = _pulse_interval_ms(protocol)
        times_ms.extend(start_ms + i * interval for i in range(int(protocol.get("pulses_per_burst", 5))))
    elif ptype in {"sequence_with_burst", "sequence_with_poisson_burst"}:
        interval = _pulse_interval_ms(protocol)
        burst_starts = _burst_starts_ms(protocol, poisson=(ptype == "sequence_with_poisson_burst"))
        for burst_start in burst_starts:
            for pulse_index in range(int(protocol.get("pulses_per_burst", 5))):
                times_ms.append(burst_start + pulse_index * interval)
    elif ptype == "custom_sequence":
        times_ms.extend(float(point["time_ms"]) for point in protocol.get("custom_points", []))
    else:
        raise ValueError(f"Unsupported protocol type: {ptype}")
    return sorted(ms for ms in times_ms if ms >= 0.0)


def _pulse_interval_ms(protocol: dict[str, Any]) -> float:
    interval = float(protocol.get("interpulse_interval_ms", 0.0) or 0.0)
    if interval > 0:
        return interval
    return 1000.0 / max(float(protocol.get("pulse_frequency_hz", 20.0)), 0.001)


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
                "burst_interval_ms": float(protocol.get("burst_interval_ms", 200.0)),
                "pulses_per_burst": int(protocol.get("pulses_per_burst", 5)),
                "pulse_frequency_hz": float(protocol.get("pulse_frequency_hz", 20.0)),
                "interpulse_interval_ms": float(protocol.get("interpulse_interval_ms", 0.0) or 0.0),
                "amplitude_mv": float(protocol.get("amplitude_mv", 150.0)),
                "pulse_width_us": float(protocol.get("pulse_width_us", 300.0)),
            },
            sort_keys=True,
        )
        seed = int(hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        mean_s = max(float(protocol.get("burst_interval_ms", 200.0)), 0.001) / 1000.0
        current_ms = start_ms
        for _index in range(1, burst_count):
            current_ms += rng.expovariate(1.0 / mean_s) * 1000.0
            starts.append(current_ms)
        return starts
    burst_interval = max(float(protocol.get("burst_interval_ms", 200.0)), 0.0)
    return [start_ms + burst_index * burst_interval for burst_index in range(burst_count)]
'''.lstrip()
