from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


_MX = None


def _candidate_maxlab_python_paths() -> list[Path]:
    paths: list[Path] = []
    env_path = os.environ.get("MAXLAB_PYTHON_PATH", "")
    for item in env_path.split(os.pathsep):
        if item:
            paths.append(Path(item).expanduser())

    maxlab_root = Path(os.environ.get("MAXLAB_ROOT", Path.home() / "MaxLab")).expanduser()
    paths.extend(
        [
            maxlab_root,
            maxlab_root / "python",
            maxlab_root / "lib",
            maxlab_root / "share" / "libmaxlab",
            maxlab_root / "share" / "libmaxlab" / "python",
            maxlab_root / "share" / "libmaxlab" / "maxlab_lib",
        ]
    )
    if maxlab_root.exists():
        for pattern in ("maxlab.py", "maxlab/__init__.py"):
            for candidate in maxlab_root.rglob(pattern):
                paths.append(candidate.parent if candidate.name == "maxlab.py" else candidate.parent.parent)

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        text = str(path)
        if text and text not in seen:
            seen.add(text)
            unique.append(path)
    return unique


def _mx() -> Any:
    global _MX
    if _MX is not None:
        return _MX
    searched: list[str] = []
    for api_path in _candidate_maxlab_python_paths():
        searched.append(str(api_path))
        if api_path.exists() and str(api_path) not in sys.path:
            sys.path.insert(0, str(api_path))
    try:
        import maxlab as mx  # type: ignore
    except ImportError as exc:
        searched_text = "\n  ".join(searched[:32]) if searched else "(no candidate paths)"
        raise RuntimeError(
            "maxlab is required. Install MaxWell API or set MAXLAB_PYTHON_PATH to the directory that contains "
            f"maxlab.py or the maxlab package. Searched:\n  {searched_text}"
        ) from exc
    _MX = mx
    return mx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MaxWell hardware for GUI-driven closed-loop stimulation.")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--cfg", default="")
    parser.add_argument("--rules", default="")
    parser.add_argument("--rules-file", default="")
    parser.add_argument("--amplitude-mv", type=float, default=150.0)
    parser.add_argument("--pulse-width-us", type=float, default=300.0)
    parser.add_argument("--inter-phase-interval-us", type=float, default=0.0)
    parser.add_argument("--dac", type=int, default=0)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--recording-name", default="closed_loop")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--runner", default="")
    parser.add_argument("--run-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_rules(args: argparse.Namespace, system_config: dict[str, Any]) -> list[dict[str, Any]]:
    if args.rules:
        rules = json.loads(args.rules)
    elif args.rules_file:
        rules = json.loads(Path(args.rules_file).read_text(encoding="utf-8"))
    else:
        closed_loop = system_config.get("closed_loop", {}) if isinstance(system_config, dict) else {}
        rules = closed_loop.get("rules", [])
        if not rules:
            rules = [
                {
                    "start_s": 0,
                    "stop_s": None,
                    "detect_spec": str(closed_loop.get("detect_spec", "auto:top:8:5")),
                    "detect_channels": closed_loop.get("detect_channels", []),
                    "threshold": int(closed_loop.get("threshold", 12)),
                    "stim_electrodes": closed_loop.get("stim_electrodes", []),
                    "sequence": str(closed_loop.get("sequence_name", "closed_loop")),
                }
            ]
    if not isinstance(rules, list):
        raise ValueError("Closed-loop rules must be a JSON list")
    return [dict(item) for item in rules]


def _half_bits(amplitude_mv: float) -> int:
    mx = _mx()
    bits = int(round(abs(float(amplitude_mv)) / float(mx.query_DAC_lsb_mV())))
    return max(1, min(511, bits))


def _samples_us(us: float) -> int:
    return max(1, int(round(float(us) / 50.0)))


def wait_after_offset(device: str) -> None:
    mx = _mx()
    extra = getattr(mx.Timing, "waitAfterOffset", 0.0)
    if str(device).lower() == "maxtwo":
        time.sleep(getattr(mx.Timing, "waitInMX2Offset", 0.0) + extra)
    else:
        time.sleep(getattr(mx.Timing, "waitInMX1Offset", 0.0) + extra)


def stim_electrodes_from_rules(rules: list[dict[str, Any]]) -> list[int]:
    electrodes: list[int] = []
    for rule in rules:
        for value in rule.get("stim_electrodes", []) or []:
            try:
                electrode = int(value)
            except (TypeError, ValueError):
                continue
            if electrode not in electrodes:
                electrodes.append(electrode)
    return electrodes


def sequences_from_rules(rules: list[dict[str, Any]], stim_unit_by_electrode: dict[int, int]) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    for rule in rules:
        sequence = str(rule.get("sequence") or "closed_loop")
        units = mapping.setdefault(sequence, [])
        for value in rule.get("stim_electrodes", []) or []:
            try:
                electrode = int(value)
            except (TypeError, ValueError):
                continue
            unit = stim_unit_by_electrode.get(electrode)
            if unit is not None and unit not in units:
                units.append(unit)
    return mapping


def reset_sequence(name: str) -> None:
    mx = _mx()
    seq = mx.Sequence(str(name), persistent=False)
    del seq


def append_biphasic_pulse(seq: Any, amplitude_mv: float, pulse_width_us: float, ipi_us: float, dac: int) -> None:
    mx = _mx()
    bits = _half_bits(amplitude_mv)
    seq.append(mx.DAC(int(dac), 512 - bits))
    seq.append(mx.DelaySamples(_samples_us(pulse_width_us)))
    if ipi_us > 0:
        seq.append(mx.DelaySamples(_samples_us(ipi_us)))
    seq.append(mx.DAC(int(dac), 512 + bits))
    seq.append(mx.DelaySamples(_samples_us(pulse_width_us)))
    seq.append(mx.DAC(int(dac), 512))


def prepare_hardware(
    cfg_path: Path,
    rules: list[dict[str, Any]],
    system_config: dict[str, Any],
    *,
    amplitude_mv: float,
    pulse_width_us: float,
    ipi_us: float,
    dac: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    electrodes = stim_electrodes_from_rules(rules)
    if not electrodes:
        raise ValueError("No stim_electrodes in closed-loop rules; hardware setup cannot create response sequences")
    if dry_run:
        return {"cfg_path": str(cfg_path), "stim_electrodes": electrodes, "sequence_units": {}}

    mx = _mx()
    for rule in rules:
        reset_sequence(str(rule.get("sequence") or "closed_loop"))

    mx.initialize()
    time.sleep(getattr(mx.Timing, "waitInit", 0.0))
    response = mx.send(mx.Core().enable_stimulation_power(True))
    if response != "Ok":
        raise RuntimeError(f"Failed to enable stimulation power: {response}")
    maxwell_cfg = system_config.get("maxwell", {}) if isinstance(system_config, dict) else {}
    mx.send(mx.Amplifier().set_gain(int(maxwell_cfg.get("amplifier_gain", 512))))
    mx.set_event_threshold(float(maxwell_cfg.get("event_threshold", 8.5)))

    array = mx.Array("closed_loop")
    array.reset()
    array.clear_selected_electrodes()
    if cfg_path.exists():
        array.load_config(str(cfg_path))
    else:
        raise FileNotFoundError(f"cfg_path does not exist: {cfg_path}")

    stim_unit_by_electrode: dict[int, int] = {}
    for electrode in electrodes:
        array.connect_electrode_to_stimulation(electrode)
        stim_unit = array.query_stimulation_at_electrode(electrode)
        if not stim_unit:
            raise RuntimeError(f"No stimulation unit can connect to electrode {electrode}")
        stim_unit_by_electrode[electrode] = int(stim_unit)

    mx.activate([0])
    array.download([0])
    time.sleep(getattr(mx.Timing, "waitAfterDownload", 0.0))
    mx.offset()
    wait_after_offset(str(maxwell_cfg.get("device", "maxone")))

    for stim_unit in sorted(set(stim_unit_by_electrode.values())):
        mx.send(mx.StimulationUnit(stim_unit).power_up(True).connect(False).set_voltage_mode().dac_source(int(dac)))

    sequence_units = sequences_from_rules(rules, stim_unit_by_electrode)
    for event_id, (sequence_name, units) in enumerate(sequence_units.items(), start=1):
        seq = mx.Sequence(sequence_name, persistent=True)
        seq.append(mx.Event(0, 1, event_id, f"type stim name {sequence_name}"))
        for unit in units:
            seq.append(mx.StimulationUnit(unit).connect(True))
        append_biphasic_pulse(seq, amplitude_mv, pulse_width_us, ipi_us, dac)
        for unit in units:
            seq.append(mx.StimulationUnit(unit).connect(False))

    mx.clear_events()
    return {
        "cfg_path": str(cfg_path),
        "stim_electrodes": electrodes,
        "stim_unit_by_electrode": stim_unit_by_electrode,
        "sequence_units": sequence_units,
    }


def start_saving(data_dir: Path, recording_name: str) -> Any:
    mx = _mx()
    saver = mx.Saving()
    saver.open_directory(str(data_dir))
    saver.group_delete_all()
    saver.group_define(0, "all_channels", list(range(1024)))
    time.sleep(0.5)
    saver.start_file(recording_name)
    saver.start_recording([0])
    return saver


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parents[1]
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = base_dir / config_dir
    system_config = load_yaml(config_dir / "system.yaml")
    cfg_text = str(args.cfg or system_config.get("electrode_map", {}).get("cfg_path", "") or "").strip()
    cfg_path = Path(cfg_text)
    rules = load_rules(args, system_config)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and (not cfg_text or not cfg_path.exists()):
        raise FileNotFoundError(f"cfg_path does not exist: {cfg_text or '<empty>'}")

    meta = prepare_hardware(
        cfg_path,
        rules,
        system_config,
        amplitude_mv=args.amplitude_mv,
        pulse_width_us=args.pulse_width_us,
        ipi_us=args.inter_phase_interval_us,
        dac=args.dac,
        dry_run=args.dry_run,
    )
    setup_meta = data_dir / "closed_loop_setup_meta.json"
    setup_meta.write_text(json.dumps({"rules": rules, "setup": meta}, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Closed-loop hardware prepared. Meta: {setup_meta}")

    if args.dry_run or not args.runner:
        return 0

    saver = None
    if args.record:
        saver = start_saving(data_dir, args.recording_name)
        print(f"Recording started: {args.recording_name}")
    runner_cmd = [
        args.runner,
        "--rules",
        json.dumps(rules, ensure_ascii=True),
        "--blank-ms",
        str(int(system_config.get("closed_loop", {}).get("blank_ms", 500))),
    ]
    if args.run_seconds > 0:
        runner_cmd += ["--run-seconds", str(args.run_seconds)]
    try:
        proc = subprocess.Popen(runner_cmd, cwd=str(base_dir))
        ret = proc.wait()
    finally:
        if saver is not None:
            saver.stop_recording()
            time.sleep(getattr(_mx().Timing, "waitAfterRecording", 0.0))
            saver.stop_file()
            saver.group_delete_all()
            print("Recording stopped")
    return int(ret)


if __name__ == "__main__":
    raise SystemExit(main())
