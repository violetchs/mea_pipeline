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
