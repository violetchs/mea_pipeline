"""Runtime runner for GUI-generated custom analysis modules."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("mea_agent_generated_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    analyze = getattr(module, "analyze", None)
    if not callable(analyze):
        raise RuntimeError("Generated module must define analyze(context: dict) -> dict")
    return module


def _matrix_to_file(record: dict, output_dir: Path, index: int) -> dict:
    payload = dict(record)
    output_root = output_dir.resolve()
    matrix_path = payload.get("matrix_path")
    matrix = payload.get("matrix")
    if matrix_path:
        path = Path(str(matrix_path))
        if not path.is_absolute():
            path = output_dir / path
        path = path.resolve()
        if not _is_relative_to(path, output_root):
            raise RuntimeError(f"processed_records[{index}] matrix_path must be inside output_dir")
        values = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    elif matrix is not None:
        values = np.asarray(matrix, dtype=float)
        path = (output_dir / f"processed_{index:03d}.npy").resolve()
    else:
        raise RuntimeError(f"processed_records[{index}] missing matrix or matrix_path")

    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise RuntimeError(f"processed_records[{index}] matrix must be 1D or 2D")
    if not np.isfinite(values).all():
        raise RuntimeError(f"processed_records[{index}] matrix contains NaN or infinite values")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values, allow_pickle=False)
    payload.pop("matrix", None)
    payload["matrix_path"] = str(path)
    payload["shape"] = [int(values.shape[0]), int(values.shape[1])]
    return payload


def _normalize_result(result: object, output_dir: Path) -> dict:
    if not isinstance(result, dict):
        raise RuntimeError("analyze(context) must return a dict")
    records = result.get("processed_records", [])
    if records is None:
        records = []
    if not isinstance(records, list):
        raise RuntimeError("processed_records must be a list")
    normalized = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RuntimeError(f"processed_records[{index}] must be a dict")
        normalized.append(_matrix_to_file(record, output_dir, index))
    figures = result.get("figures", [])
    if figures is None:
        figures = []
    elif isinstance(figures, (str, dict)):
        figures = [figures]
    elif not isinstance(figures, list):
        raise RuntimeError("figures must be a list, dict, string, or null")
    return {
        "processed_records": normalized,
        "figures": figures,
        "summary": str(result.get("summary", "") or ""),
    }


def run(module_path: Path, input_manifest_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(input_manifest_path.read_text(encoding="utf-8-sig"))
    context = {
        "input_manifest_path": str(input_manifest_path),
        "output_dir": str(output_dir),
        "selected_raw_records": list(manifest.get("selected_raw_records", []) or []),
        "selected_processed_records": list(manifest.get("selected_processed_records", []) or []),
        "parameters": dict(manifest.get("parameters", {}) or {}),
    }
    module = _load_module(module_path)
    return _normalize_result(module.analyze(context), output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a GUI-generated MEA custom analysis module.")
    parser.add_argument("--module", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    module_path = Path(args.module).resolve()
    input_manifest_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    result_path = output_dir / "result.json"
    try:
        result = run(module_path, input_manifest_path, output_dir)
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"ok": True, "result_path": str(result_path)}, ensure_ascii=False))
        return 0
    except Exception:
        details = traceback.format_exc()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "error.txt").write_text(details, encoding="utf-8")
        print(details, file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - manual subprocess entrypoint.
    raise SystemExit(main())
