"""GUI for runtime-generated custom analysis modules."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import ast
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import numpy as np
import matplotlib.image as mpimg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

_FONT_FALLBACKS = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["font.sans-serif"] = _FONT_FALLBACKS + [
    font for font in mpl.rcParams.get("font.sans-serif", []) if font not in _FONT_FALLBACKS
]
mpl.rcParams["axes.unicode_minus"] = False

try:
    from PySide6.QtCore import QProcess, QTimer
    from PySide6.QtGui import QColor, QFont
    from PySide6.QtWidgets import (
        QComboBox,
        QApplication,
        QDialog,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - GUI import guard.
    raise SystemExit("PySide6 is required for Agent Custom Code.") from exc


RawProvider = Callable[[], list[dict]]
ProcessedProvider = Callable[[], list[dict]]
SaveProcessedCallback = Callable[[list[dict]], int]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_module_name() -> str:
    package = str(__package__ or "")
    if package.startswith("src.gui"):
        return "src.gui.agent_runtime"
    return "gui.agent_runtime"


def _agent_root() -> Path:
    return _repo_root() / ".agents"


def _agent_config_path() -> Path:
    return _agent_root() / "agent_config.json"


def _resolve_agent_executable(command: str) -> str:
    command = str(command or "").strip().strip('"')
    if not command:
        return ""
    if any(separator in command for separator in (os.path.sep, "/", "\\")):
        path = Path(command)
        return str(path) if path.exists() else ""
    return shutil.which(command) or ""


def _default_codex_command_value() -> str:
    if os.name == "nt":
        cmd = shutil.which("codex.cmd")
        if cmd:
            return cmd
    return shutil.which("codex") or "codex"


def _default_agent_backend_value() -> str:
    if _resolve_agent_executable(_default_codex_command_value()):
        return "codex"
    if _resolve_agent_executable("claude"):
        return "claude"
    return "custom"


def _default_agent_command_value(backend: str) -> str:
    if backend == "codex":
        return _default_codex_command_value()
    if backend == "claude":
        return shutil.which("claude") or "claude"
    return ""


def _default_agent_args_template_value(backend: str) -> str:
    if backend == "codex":
        return "exec --cd {module_dir} --sandbox workspace-write {prompt}"
    if backend == "claude":
        return "-p {prompt}"
    return "{prompt}"


def _load_agent_config_file() -> dict:
    path = _agent_config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _save_agent_config_file(config: dict) -> None:
    _agent_root().mkdir(parents=True, exist_ok=True)
    _agent_config_path().write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def _agent_setup_text_for_backend(backend: str) -> str:
    if backend == "codex":
        return (
            "Codex CLI setup:\n"
            "1. Install the Codex CLI on this computer.\n"
            "2. Authenticate/login in a terminal using the Codex CLI command.\n"
            "3. Confirm `codex --version` or `codex.cmd --version` works.\n"
            "4. Put the command on PATH, or use Browse to select the executable.\n"
            "5. Click Retest."
        )
    if backend == "claude":
        return (
            "Claude Code setup:\n"
            "1. Install Claude Code on this computer.\n"
            "2. Authenticate/login in a terminal using the Claude command.\n"
            "3. Confirm `claude --version` works.\n"
            "4. Put the command on PATH, or use Browse to select the executable.\n"
            "5. If your Claude command needs special non-interactive arguments, choose Custom command."
        )
    return (
        "Custom agent setup:\n"
        "1. Install an agent CLI that can run non-interactively and edit files in the module directory.\n"
        "2. Set Command to the executable path.\n"
        "3. Set Args template using placeholders: {module_dir}, {prompt}, {task_file}.\n"
        "4. The agent must read AGENTS.md/AGENT_TASK.md and write module.py.\n"
        "5. Confirm the command works from a terminal, then click Retest."
    )


def _agent_process_command(
    *,
    backend: str,
    command: str,
    args_template: str,
    module_dir: Path,
    prompt: str,
    task_path: Path,
) -> tuple[str, list[str], Path]:
    executable = _resolve_agent_executable(command) or command
    if backend == "codex":
        return executable, ["exec", "--cd", str(module_dir), "--sandbox", "workspace-write", prompt], module_dir
    if backend == "claude":
        return executable, ["-p", prompt], module_dir
    template = args_template.strip() or "{prompt}"
    expanded = template.format(module_dir=str(module_dir), prompt=prompt, task_file=str(task_path))
    args = shlex.split(expanded, posix=(os.name != "nt"))
    return executable, args, module_dir


def _looks_like_code_detail_line(line: str) -> bool:
    stripped = str(line or "").strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if stripped.startswith(("{", "}", "[", "]", "(", ")")):
        return True
    if stripped.startswith(("@", "#!")):
        return True
    if stripped in {");", "];", "};"}:
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_\.]*\s*=\s*.+", stripped):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_\.]*\(.+\)$", stripped):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_\.]*\[[^\]]+\]\s*=", stripped):
        return True
    if re.match(r"^[\"'][^\"']+[\"']\s*:\s*", stripped):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*:\s*(dict|list|str|int|float|bool|Path|np\.)", stripped):
        return True
    code_markers = (
        "np.",
        "pd.",
        "plt.",
        "json.",
        "Path(",
        "Path.",
        "os.",
        "sys.",
        "shutil.",
        "subprocess.",
        "QProcess",
        "QTimer",
        "Figure(",
        "npz",
    )
    if any(marker.lower() in lower for marker in code_markers):
        return True
    shell_code_markers = (
        "apply_patch",
        "<<'patch'",
        "<<\"patch\"",
        "cat >",
        "python - <<",
        "python -c ",
    )
    if any(marker in lower for marker in shell_code_markers):
        return True
    return False


def _agent_log_lines(text: str, *, stream: str = "") -> list[str]:
    lines: list[str] = []
    skip_user_task_body = False
    skip_fenced_block = False
    skip_patch_block = False
    for raw_line in str(text or "").splitlines():
        raw_line = _repair_mojibake_text(raw_line)
        line = raw_line.strip()
        lower = line.lower()
        if skip_fenced_block:
            if line.startswith("```"):
                skip_fenced_block = False
            continue
        if skip_patch_block:
            if lower.startswith("*** end patch"):
                skip_patch_block = False
            continue
        if skip_user_task_body:
            if not line:
                skip_user_task_body = False
            continue
        if not line:
            continue
        if line.startswith(("stderr:", "stdout:")):
            line = line.split(":", 1)[1].strip()
        lower = line.lower()
        if not line:
            continue
        if _looks_like_unfixed_mojibake(line):
            continue
        if lower in {"codex", "user", "mcp startup: no servers"}:
            continue
        if lower == "thinking":
            lines.append("Agent thinking...")
            continue
        if lower == "exec":
            lines.append("Agent running a command...")
            continue
        if lower == "tokens used":
            continue
        metadata_prefixes = (
            "openai codex ",
            "--------",
            "workdir:",
            "model:",
            "provider:",
            "approval:",
            "sandbox:",
            "reasoning effort:",
            "reasoning summaries:",
            "session id:",
            "mcp startup:",
        )
        if lower.startswith(metadata_prefixes):
            continue
        if lower.startswith("## user task"):
            skip_user_task_body = True
            continue
        if line.startswith("```"):
            skip_fenced_block = True
            continue
        if lower.startswith("*** begin patch"):
            skip_patch_block = True
            continue
        patch_prefixes = (
            "diff --git ",
            "index ",
            "@@",
            "+++ ",
            "--- ",
            "*** update file:",
            "*** add file:",
            "*** delete file:",
            "*** end patch",
        )
        if lower.startswith(patch_prefixes):
            continue
        if _looks_like_code_detail_line(line):
            continue
        if line.startswith(("+", "-")) and not line.startswith(("+ ", "- ")):
            continue
        if line.startswith((">", "<")):
            continue
        code_prefixes = (
            "def ",
            "class ",
            "import ",
            "from ",
            "return ",
            "raise ",
            "if ",
            "elif ",
            "else:",
            "for ",
            "while ",
            "with ",
            "try:",
            "except ",
            "finally:",
        )
        if lower.startswith(code_prefixes):
            continue
        prompt_prefixes = (
            "task - ",
            "you are generating ",
            "you already have ",
            "# mandatory agent instructions",
            "## user task",
            "context: you are generating ",
            "input data paths are provided ",
            "raw spike input package ",
            "processed input package ",
            "when multiple raw ",
            "for cross-file channel ",
            "if the task asks ",
            "return {'processed_records'",
            "each processed record ",
            "required final state:",
        )
        if lower.startswith(prompt_prefixes):
            continue
        if stream == "stderr" and lower.startswith(("traceback", "error:", "failed", "runtimeerror")):
            lines.append(f"Error: {line[:320]}")
            continue
        if len(line) > 320:
            line = line[:317] + "..."
        lines.append(line)
    return lines


def _decode_process_output(data) -> str:
    raw = bytes(data)
    if not raw:
        return ""
    for encoding in ("utf-8", "gb18030", "cp936"):
        try:
            return _repair_mojibake_text(raw.decode(encoding))
        except UnicodeError:
            continue
    return _repair_mojibake_text(raw.decode("utf-8", errors="replace"))


def _repair_mojibake_text(text: str) -> str:
    value = str(text or "")
    if not value:
        return value
    suspect_markers = ("瀵", "鎹", "鍔", "瀛", "骞", "绾", "缁", "撴", "鍥", "紝", "愶")
    if not any(marker in value for marker in suspect_markers):
        return value
    for encoding in ("gb18030", "gbk", "cp936"):
        try:
            fixed = value.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
        if _looks_more_readable_chinese(fixed, value):
            return fixed
    return value


def _looks_like_unfixed_mojibake(text: str) -> bool:
    value = str(text or "")
    if not value:
        return False
    bad_tokens = ("瀵", "鎹", "鍔", "瀛", "骞", "缁", "撴", "愶", "紝")
    bad_count = sum(value.count(token) for token in bad_tokens)
    if bad_count < 2:
        return False
    useful_tokens = ("数据", "分析", "结果", "通道", "发放", "刺激")
    return not any(token in value for token in useful_tokens)


def _looks_more_readable_chinese(candidate: str, original: str) -> bool:
    useful_tokens = ("数据", "分析", "动力学", "结果", "图", "通道", "发放", "刺激", "文件")
    if any(token in candidate for token in useful_tokens):
        return True
    bad_tokens = ("瀵", "鎹", "鍔", "瀛", "骞", "缁", "撴", "愶")
    return sum(candidate.count(token) for token in bad_tokens) < sum(original.count(token) for token in bad_tokens)


def _command_for_log(program: str, args: list[str]) -> str:
    display = [str(program)]
    for arg in list(args or []):
        value = str(arg)
        if "\n" in value or len(value) > 160 or value.lower().startswith(("task:", "task - ")):
            display.append("<prompt>")
            continue
        if any(char.isspace() for char in value):
            display.append(f'"{value}"')
        else:
            display.append(value)
    return " ".join(display)


def _file_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    if not root.exists():
        return snapshot
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = str(path)
        snapshot[relative] = (int(stat.st_mtime_ns), int(stat.st_size))
    return snapshot


def _changed_files(root: Path, before: dict[str, tuple[int, int]]) -> list[str]:
    after = _file_snapshot(root)
    changed = [name for name, state in after.items() if before.get(name) != state]
    return sorted(changed)


def _read_python_text_without_bom(path: Path) -> tuple[str, bool]:
    raw = path.read_bytes()
    had_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    if had_bom:
        path.write_text(text, encoding="utf-8")
    return text, had_bom


def _compact_module_summary(text: str, *, max_len: int = 360) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    value = re.sub(r"^(summary|module summary|功能摘要)\s*[:：]\s*", "", value, flags=re.IGNORECASE).strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3].rstrip() + "..."


def _summary_from_python_code(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    analyze_doc = ""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "analyze":
            analyze_doc = ast.get_docstring(node) or ""
            break
    module_doc = ast.get_docstring(tree) or ""
    for candidate in (analyze_doc, module_doc):
        summary = _compact_module_summary(candidate)
        if summary and "Agent analysis code has not been generated yet" not in summary:
            return summary
    for line in str(code or "").splitlines()[:120]:
        match = re.match(r"\s*#\s*(?:summary|module summary|功能摘要)\s*[:：]\s*(.+)$", line, flags=re.IGNORECASE)
        if match:
            return _compact_module_summary(match.group(1))
    return ""


def _summary_from_readme(readme_path: Path) -> str:
    if not readme_path.exists():
        return ""
    try:
        lines = readme_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError:
        return ""
    capture = False
    captured: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^#+\s*(summary|module summary|功能摘要)\b", stripped, flags=re.IGNORECASE):
            capture = True
            inline = re.sub(r"^#+\s*(summary|module summary|功能摘要)\s*[:：]?\s*", "", stripped, flags=re.IGNORECASE)
            if inline:
                captured.append(inline)
            continue
        if capture:
            if stripped.startswith("#") and captured:
                break
            if stripped:
                captured.append(stripped)
            elif captured:
                break
    if captured:
        return _compact_module_summary(" ".join(captured))
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.lower().startswith(("task:", "see agent_task", "see agents.md")):
            continue
        return _compact_module_summary(stripped)
    return ""


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _slugify(text: str, fallback: str = "agent_module") -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", str(text or "").strip()).strip("_").lower()
    return slug[:64] or fallback


def _module_template(module_name: str) -> str:
    safe_name = module_name.replace('"', "'")
    return f'''"""Generated MEA custom analysis module: {safe_name}.

The GUI runs this file in a separate Python process.

Agent contract:
- Implement analyze(context: dict) -> dict.
- Read only files listed in context or files you create under context["output_dir"].
- Return a dict with processed_records, figures, and summary.
- Each processed record must include name, dataset_type, matrix or matrix_path,
  sample_labels, feature_labels, and description.

Input helpers below are generic conveniences only. They do not define the
analysis. The agent should replace or extend this file according to the user's
task.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_raw_spike_record(record: dict) -> dict:
    """Load one raw spike input exported by the GUI."""
    data_path = Path(record["data_path"])
    with np.load(data_path, allow_pickle=True) as data:
        metadata_json = str(data["metadata_json"][0]) if "metadata_json" in data else "{{}}"
        try:
            metadata = json.loads(metadata_json)
        except Exception:
            metadata = {{}}
        return {{
            "channels": [str(item) for item in data["channels"].tolist()],
            "spike_times": data["spike_times"],
            "stim_times": np.asarray(data["stim_times"], dtype=float) if "stim_times" in data else np.asarray([], dtype=float),
            "sampling_rate": float(np.asarray(data["sampling_rate"]).reshape(-1)[0]) if "sampling_rate" in data else 0.0,
            "metadata": metadata,
        }}


def load_processed_record(record: dict) -> dict:
    """Load one processed input exported by the GUI."""
    data_path = Path(record["data_path"])
    with np.load(data_path, allow_pickle=True) as data:
        metadata_json = str(data["metadata_json"][0]) if "metadata_json" in data else "{{}}"
        try:
            metadata = json.loads(metadata_json)
        except Exception:
            metadata = {{}}
        return {{
            "matrix": np.asarray(data["matrix"], dtype=float),
            "sample_labels": [str(item) for item in data["sample_labels"].tolist()] if "sample_labels" in data else [],
            "feature_labels": [str(item) for item in data["feature_labels"].tolist()] if "feature_labels" in data else [],
            "metadata": metadata,
        }}


def analyze(context: dict) -> dict:
    raise RuntimeError(
        "Agent analysis code has not been generated yet. "
        "Use Generate Module or Modify Selected with a concrete task before running this module."
    )
'''
    return f'''"""Generated MEA custom analysis module: {safe_name}.

The GUI runs this file in a separate Python process. Implement analyze(context)
and return processed_records with matrix or matrix_path fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load_processed_matrix(record: dict) -> np.ndarray:
    data_path = Path(record["data_path"])
    with np.load(data_path, allow_pickle=True) as data:
        return np.asarray(data["matrix"], dtype=float)


def _short_record_name(record: dict, fallback: str) -> str:
    name = str(record.get("name") or Path(str(record.get("source_path", ""))).name or fallback)
    return name[:80]


def _raw_channel_rates(record: dict) -> tuple[list[str], np.ndarray]:
    data_path = Path(record["data_path"])
    with np.load(data_path, allow_pickle=True) as data:
        channels = [str(item) for item in data["channels"].tolist()]
        spike_times = data["spike_times"]
    rates = []
    for times in spike_times:
        values = np.asarray(times, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size > 1:
            duration = max(1e-9, float(finite.max() - finite.min()))
        else:
            duration = 1.0
        rates.append(float(finite.size) / duration)
    return channels, np.asarray(rates, dtype=float)


def _wants_one_plot_per_channel(task: str) -> bool:
    text = str(task or "").lower().replace(" ", "")
    if any(token in text for token in ("每个通道", "每一通道", "每个电极", "每一电极", "eachchannel", "perchannel")):
        if any(token in text for token in ("图", "plot", "figure", "result", "结果")):
            return True
    keywords = [
        "每个通道一张图",
        "每个通道一个图",
        "每个电极一张图",
        "每个电极一个图",
        "每个channel一张图",
        "per-channelplot",
        "oneplotperchannel",
        "onefigureperchannel",
        "eachchannelplot",
    ]
    return any(keyword in text for keyword in keywords)


def _raw_firing_rate_matrix(raw_records: list[dict]) -> tuple[list[str], list[str], np.ndarray]:
    per_file = []
    channel_order: list[str] = []
    seen = set()
    for index, record in enumerate(raw_records):
        channels, rates = _raw_channel_rates(record)
        rate_lookup = {{channel: float(rate) for channel, rate in zip(channels, rates)}}
        for channel in channels:
            if channel not in seen:
                seen.add(channel)
                channel_order.append(channel)
        per_file.append((_short_record_name(record, f"raw {{index + 1}}"), rate_lookup))

    matrix = np.full((len(channel_order), len(per_file)), np.nan, dtype=float)
    for column, (_name, rate_lookup) in enumerate(per_file):
        for row, channel in enumerate(channel_order):
            if channel in rate_lookup:
                matrix[row, column] = rate_lookup[channel]
    matrix = np.nan_to_num(matrix, nan=0.0)
    file_labels = [name for name, _lookup in per_file]
    return channel_order, file_labels, matrix


def _compare_raw_firing_rates(raw_records: list[dict], output_dir: Path) -> dict:
    channel_order, file_labels, matrix = _raw_firing_rate_matrix(raw_records)
    matrix_path = output_dir / "raw_channel_firing_rate_comparison.npy"
    np.save(matrix_path, matrix, allow_pickle=False)
    return {{
        "name": "{safe_name} channel firing rate comparison",
        "dataset_type": "firing_rate_hz",
        "matrix_path": str(matrix_path),
        "sample_labels": channel_order,
        "feature_labels": file_labels,
        "value_label": "firing_rate_hz",
        "x_axis_label": "Channel",
        "description": "Per-channel firing-rate comparison across selected raw files.",
    }}


def _compare_raw_firing_rates_per_channel(raw_records: list[dict], output_dir: Path) -> list[dict]:
    channel_order, file_labels, matrix = _raw_firing_rate_matrix(raw_records)
    records = []
    for row, channel in enumerate(channel_order):
        channel_matrix = matrix[row : row + 1, :]
        safe_channel = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(channel))[:60] or str(row)
        matrix_path = output_dir / f"channel_{{row:04d}}_{{safe_channel}}_firing_rate.npy"
        np.save(matrix_path, channel_matrix, allow_pickle=False)
        records.append({{
            "name": "{safe_name} channel " + str(channel) + " firing rate",
            "dataset_type": "firing_rate_hz",
            "matrix_path": str(matrix_path),
            "sample_labels": [str(channel)],
            "feature_labels": file_labels,
            "value_label": "firing_rate_hz",
            "x_axis_label": "File",
            "description": "One-channel firing-rate comparison across selected raw files.",
        }})
    return records


def analyze(context: dict) -> dict:
    output_dir = Path(context["output_dir"])
    processed = list(context.get("selected_processed_records", []))
    raw = list(context.get("selected_raw_records", []))
    task = str((context.get("parameters", {{}}) or {{}}).get("task", ""))

    if len(raw) >= 1:
        if _wants_one_plot_per_channel(task):
            records = _compare_raw_firing_rates_per_channel(raw, output_dir)
            summary = f"Computed one per-channel firing-rate result for each of {{len(records)}} channel(s) across {{len(raw)}} raw file(s)."
        else:
            records = [_compare_raw_firing_rates(raw, output_dir)]
            summary = f"Computed per-channel firing rates for {{len(raw)}} selected raw file(s)."
    elif processed:
        matrices = [_load_processed_matrix(record) for record in processed]
        matrix = np.vstack([np.nanmean(values, axis=0).reshape(1, -1) for values in matrices])
        result = np.nanmean(matrix, axis=0).reshape(1, -1)
        feature_labels = list(processed[0].get("feature_labels", []))
        sample_labels = [_short_record_name(record, f"processed {{index + 1}}") for index, record in enumerate(processed)]
        matrix_path = output_dir / "processed_feature_mean_comparison.npy"
        np.save(matrix_path, matrix, allow_pickle=False)
        record = {{
            "name": "{safe_name} processed feature comparison",
            "dataset_type": "agent_custom",
            "matrix_path": str(matrix_path),
            "sample_labels": sample_labels,
            "feature_labels": feature_labels,
            "description": f"Computed feature-wise means for {{len(processed)}} selected processed dataset(s).",
        }}
        summary = record["description"]
        records = [record]
    else:
        matrix_path = output_dir / "agent_result.npy"
        np.save(matrix_path, np.zeros((0, 1), dtype=float), allow_pickle=False)
        record = {{
            "name": "{safe_name} result",
            "dataset_type": "agent_custom",
            "matrix_path": str(matrix_path),
            "sample_labels": [],
            "feature_labels": ["value"],
            "description": "No selected input data.",
        }}
        summary = "No selected input data."
        records = [record]

    return {{
        "processed_records": records,
        "figures": [],
        "summary": summary,
    }}
'''


class AgentEnvironmentCheckDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Agent Environment Check")
        self.resize(920, 620)
        self.root = _agent_root()
        self.check_root = self.root / "env_check"
        self.current_process: QProcess | None = None
        self.probe_dir: Path | None = None
        self._probe_file_snapshot: dict[str, tuple[int, int]] = {}
        self.environment_ready = False
        self._loading_config = False
        self._build_ui()
        self._load_config_to_ui()
        QTimer.singleShot(0, self._show_cached_success_or_start_check)

    def _build_ui(self) -> None:
        self.backend = QComboBox()
        self.backend.addItem("Codex CLI", "codex")
        self.backend.addItem("Claude Code", "claude")
        self.backend.addItem("Custom command", "custom")
        self.backend.currentIndexChanged.connect(self._backend_changed)
        self.command = QLineEdit()
        self.command.textEdited.connect(self._invalidate_cached_ready)
        self.args_template = QLineEdit()
        self.args_template.textEdited.connect(self._invalidate_cached_ready)
        self.command_browse = QPushButton("Browse")
        self.command_browse.clicked.connect(self._browse_command)
        self.retest_button = QPushButton("Retest")
        self.retest_button.clicked.connect(self._start_check)
        self.continue_button = QPushButton("Continue")
        self.continue_button.clicked.connect(self.accept)
        self.continue_button.setEnabled(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.guide_button = QPushButton("Setup Guide")
        self.guide_button.clicked.connect(self._show_setup_guide)
        self.steps = QTableWidget(0, 3)
        self.steps.setHorizontalHeaderLabels(["Step", "Status", "Details"])
        self.steps.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.steps.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 8))
        self.log.document().setMaximumBlockCount(2000)

        layout = QVBoxLayout(self)
        intro = QLabel("Agent Custom Code requires a local agent CLI. The checker runs sequentially and stops at the first failed step.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        grid = QGridLayout()
        grid.addWidget(QLabel("Backend"), 0, 0)
        grid.addWidget(self.backend, 0, 1, 1, 2)
        grid.addWidget(QLabel("Command"), 1, 0)
        grid.addWidget(self.command, 1, 1)
        grid.addWidget(self.command_browse, 1, 2)
        grid.addWidget(QLabel("Args"), 2, 0)
        grid.addWidget(self.args_template, 2, 1, 1, 2)
        layout.addLayout(grid)
        layout.addWidget(self.steps, 2)
        layout.addWidget(QLabel("Check log"))
        layout.addWidget(self.log, 1)
        buttons = QHBoxLayout()
        buttons.addWidget(self.retest_button)
        buttons.addWidget(self.guide_button)
        buttons.addStretch(1)
        buttons.addWidget(self.continue_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)

    def _load_config_to_ui(self) -> None:
        self._loading_config = True
        config = _load_agent_config_file()
        backend = str(config.get("backend") or _default_agent_backend_value())
        for index in range(self.backend.count()):
            if self.backend.itemData(index) == backend:
                self.backend.setCurrentIndex(index)
                break
        self.command.setText(str(config.get("command") or _default_agent_command_value(backend)))
        self.args_template.setText(str(config.get("args_template") or _default_agent_args_template_value(backend)))
        self._backend_changed()
        if config.get("args_template"):
            self.args_template.setText(str(config.get("args_template")))
        self._loading_config = False

    def _backend_changed(self) -> None:
        backend = str(self.backend.currentData() or "custom")
        current = self.command.text().strip()
        if not current or current in {"codex", "codex.cmd", "claude"}:
            self.command.setText(_default_agent_command_value(backend))
        if not self.args_template.text().strip() or backend != "custom":
            self.args_template.setText(_default_agent_args_template_value(backend))
        self.args_template.setEnabled(backend == "custom")
        if not self._loading_config:
            self._invalidate_cached_ready()

    def _invalidate_cached_ready(self) -> None:
        if self._loading_config:
            return
        self.environment_ready = False
        self.continue_button.setEnabled(False)

    def _browse_command(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Select Agent Executable")
        if path:
            self.command.setText(path)
            self._invalidate_cached_ready()

    def _show_setup_guide(self) -> None:
        QMessageBox.information(self, "Agent Setup Guide", _agent_setup_text_for_backend(str(self.backend.currentData() or "custom")))

    def _config(self) -> dict:
        return {
            "backend": str(self.backend.currentData() or "custom"),
            "command": self.command.text().strip(),
            "args_template": self.args_template.text().strip(),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _set_step(self, row: int, label: str, status: str, detail: str, color: str = "#6b7280") -> None:
        if self.steps.rowCount() <= row:
            self.steps.setRowCount(row + 1)
        for column, value in enumerate([label, status, detail]):
            item = QTableWidgetItem(value)
            if column == 1:
                item.setForeground(QColor(color))
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.steps.setItem(row, column, item)
        self.steps.resizeColumnsToContents()

    def _pass_step(self, row: int, label: str, detail: str) -> None:
        self._set_step(row, label, "✓", detail, "#16a34a")

    def _fail_step(self, row: int, label: str, detail: str) -> None:
        self._set_step(row, label, "×", detail, "#dc2626")
        self.log.append(detail)
        self.retest_button.setEnabled(True)
        self.continue_button.setEnabled(False)

    def _start_check(self) -> None:
        if self.current_process is not None:
            return
        self.environment_ready = False
        self.steps.setRowCount(0)
        self.log.clear()
        self.continue_button.setEnabled(False)
        self.retest_button.setEnabled(False)
        config = self._config()
        config["environment_ready"] = False
        config["checked_at"] = ""
        _save_agent_config_file(config)
        self._run_step_1()

    def _show_cached_success_or_start_check(self) -> None:
        config = _load_agent_config_file()
        if not config.get("environment_ready"):
            self._start_check()
            return
        backend = str(config.get("backend") or "")
        command = str(config.get("command") or "")
        args_template = str(config.get("args_template") or "")
        if backend != str(self.backend.currentData() or ""):
            self._start_check()
            return
        if command != self.command.text().strip():
            self._start_check()
            return
        if args_template != self.args_template.text().strip():
            self._start_check()
            return
        checked_at = str(config.get("checked_at") or config.get("updated_at") or "previous session")
        self.steps.setRowCount(0)
        self.log.clear()
        self._pass_step(0, "1. Find agent command", command or "<configured>")
        self._pass_step(1, "2. Check version/login", f"Previously passed at {checked_at}.")
        self._pass_step(2, "3. Write probe task files", "Previously passed.")
        self._pass_step(3, "4. Real agent write test", "Previously passed. Click Retest to run the full probe again.")
        self.log.append(f"Agent environment was verified at {checked_at}.")
        self.log.append("Using cached verification. Click Retest to check the environment again.")
        self.environment_ready = True
        self.retest_button.setEnabled(True)
        self.continue_button.setEnabled(True)

    def _run_step_1(self) -> None:
        config = self._config()
        executable = _resolve_agent_executable(config["command"])
        if not executable:
            self._fail_step(0, "1. Find agent command", f"Command not found: {config['command']}\n\n{_agent_setup_text_for_backend(config['backend'])}")
            return
        self._pass_step(0, "1. Find agent command", executable)
        self._run_step_2(executable)

    def _run_step_2(self, executable: str) -> None:
        config = self._config()
        if config["backend"] == "custom":
            self._pass_step(1, "2. Check version/login", "Custom backend: executable exists; generic version/login check skipped.")
            self._run_step_3()
            return
        try:
            completed = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=6, encoding="utf-8", errors="replace")
        except Exception as exc:
            self._fail_step(1, "2. Check version/login", f"Version check could not run:\n{exc}\n\n{_agent_setup_text_for_backend(config['backend'])}")
            return
        output = (completed.stdout or completed.stderr or "").strip()
        if completed.returncode != 0:
            self._fail_step(1, "2. Check version/login", f"Version check failed with exit code {completed.returncode}:\n{output}")
            return
        expected = "codex" if config["backend"] == "codex" else "claude"
        if expected not in output.lower():
            self._fail_step(1, "2. Check version/login", f"Unexpected agent command output:\n{output}")
            return
        self._pass_step(1, "2. Check version/login", output)
        self._run_step_3()

    def _run_step_3(self) -> None:
        self.check_root.mkdir(parents=True, exist_ok=True)
        self.probe_dir = self.check_root / time.strftime("%Y%m%d_%H%M%S")
        self.probe_dir.mkdir(parents=True, exist_ok=True)
        (self.probe_dir / "probe_task.py").write_text(
            '"""Agent environment probe file."""\n\nPROBE_RESULT = "PROBE_PENDING"\n',
            encoding="utf-8",
        )
        task = (
            "# Mandatory Agent Instructions\n\n"
            "This is an environment probe, not a real analysis module.\n"
            "Edit probe_task.py now. Do not ask questions.\n"
            "Change PROBE_RESULT to AGENT_PROBE_SUCCESS.\n"
            "Create AGENT_PROBE_RESULT.txt containing exactly AGENT_PROBE_SUCCESS.\n"
        )
        (self.probe_dir / "AGENTS.md").write_text(task, encoding="utf-8")
        (self.probe_dir / "AGENT_TASK.md").write_text(task, encoding="utf-8")
        self._pass_step(2, "3. Write probe task files", str(self.probe_dir))
        self._run_step_4()

    def _run_step_4(self) -> None:
        if self.probe_dir is None:
            self._fail_step(3, "4. Real agent write test", "Probe directory was not created.")
            return
        config = self._config()
        prompt = (
            "TASK - write files now, do not ask clarification questions: "
            "read AGENTS.md and AGENT_TASK.md, edit probe_task.py so it contains "
            "PROBE_RESULT = 'AGENT_PROBE_SUCCESS', and create AGENT_PROBE_RESULT.txt "
            "containing AGENT_PROBE_SUCCESS."
        )
        program, args, workdir = _agent_process_command(
            backend=config["backend"],
            command=config["command"],
            args_template=config["args_template"],
            module_dir=self.probe_dir,
            prompt=prompt,
            task_path=self.probe_dir / "AGENT_TASK.md",
        )
        self._set_step(3, "4. Real agent write test", "...", f"Running: {_command_for_log(program, args)}")
        process = QProcess(self)
        process.setProgram(program)
        process.setArguments(args)
        process.setWorkingDirectory(str(workdir))
        process.readyReadStandardOutput.connect(self._read_probe_stdout)
        process.readyReadStandardError.connect(self._read_probe_stderr)
        process.finished.connect(self._probe_finished)
        process.errorOccurred.connect(self._probe_error)
        self.current_process = process
        self._probe_file_snapshot = _file_snapshot(self.probe_dir)
        self.log.append(f"$ {_command_for_log(program, args)}")
        process.start()
        QTimer.singleShot(180000, lambda process=process: self._probe_timeout(process))

    def _read_probe_stdout(self) -> None:
        if self.current_process is None:
            return
        text = _decode_process_output(self.current_process.readAllStandardOutput())
        for line in _agent_log_lines(text, stream="stdout"):
            self.log.append(line)

    def _read_probe_stderr(self) -> None:
        if self.current_process is None:
            return
        text = _decode_process_output(self.current_process.readAllStandardError())
        for line in _agent_log_lines(text, stream="stderr"):
            self.log.append(line)

    def _probe_timeout(self, process: QProcess) -> None:
        if process is self.current_process and process.state() != QProcess.ProcessState.NotRunning:
            self.log.append("Probe timed out.")
            process.kill()

    def _probe_error(self, error: QProcess.ProcessError) -> None:
        self.current_process = None
        self._fail_step(3, "4. Real agent write test", f"Agent process error: {error.name}")

    def _probe_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self.current_process = None
        if self.probe_dir is None:
            self._fail_step(3, "4. Real agent write test", "Probe directory missing after process finished.")
            return
        probe_path = self.probe_dir / "probe_task.py"
        marker_path = self.probe_dir / "AGENT_PROBE_RESULT.txt"
        code = probe_path.read_text(encoding="utf-8", errors="replace") if probe_path.exists() else ""
        marker = marker_path.read_text(encoding="utf-8", errors="replace") if marker_path.exists() else ""
        changed = _changed_files(self.probe_dir, self._probe_file_snapshot)
        if changed:
            self.log.append("Modified files: " + ", ".join(changed))
        if exit_code != 0:
            self._fail_step(3, "4. Real agent write test", f"Agent exited with code {exit_code}, status {exit_status.name}.")
            return
        if "AGENT_PROBE_SUCCESS" not in code or "PROBE_PENDING" in code:
            self._fail_step(3, "4. Real agent write test", "Agent ran but did not update probe_task.py.")
            return
        if "AGENT_PROBE_SUCCESS" not in marker:
            self._fail_step(3, "4. Real agent write test", "Agent ran but did not create AGENT_PROBE_RESULT.txt.")
            return
        self._pass_step(3, "4. Real agent write test", "Agent read the task, edited probe_task.py, and wrote the probe marker.")
        config = self._config()
        config["environment_ready"] = True
        config["checked_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _save_agent_config_file(config)
        self.environment_ready = True
        self.retest_button.setEnabled(True)
        self.continue_button.setEnabled(True)

    def accept(self) -> None:
        if not self.environment_ready:
            QMessageBox.warning(self, "Agent Environment", "Complete the environment check first. All steps must be green before continuing.")
            return
        super().accept()

    def reject(self) -> None:
        if self.current_process is not None and self.current_process.state() != QProcess.ProcessState.NotRunning:
            self.current_process.terminate()
            QTimer.singleShot(1500, self.current_process.kill)
        super().reject()


class AgentCustomCodeDialog(QDialog):
    def __init__(
        self,
        *,
        raw_provider: RawProvider,
        selected_raw_provider: RawProvider,
        processed_provider: ProcessedProvider,
        selected_processed_provider: ProcessedProvider,
        save_processed_callback: SaveProcessedCallback,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Agent Custom Code")
        self.resize(1420, 840)
        self.raw_provider = raw_provider
        self.selected_raw_provider = selected_raw_provider
        self.processed_provider = processed_provider
        self.selected_processed_provider = selected_processed_provider
        self.save_processed_callback = save_processed_callback
        self.root = _repo_root() / ".agents"
        self.modules_root = self.root / "generated_modules"
        self.runs_root = self.root / "runs"
        self.current_process: QProcess | None = None
        self.current_mode = ""
        self.current_run_dir: Path | None = None
        self.current_output_dir: Path | None = None
        self.current_result_folder_name = ""
        self.current_result_folder_id = ""
        self.current_input_signature = ""
        self.current_input_label = ""
        self.current_module_dir: Path | None = None
        self._process_snapshot_root: Path | None = None
        self._process_file_snapshot: dict[str, tuple[int, int]] = {}
        self._preview_drag_state: dict | None = None
        self.last_result: dict | None = None
        self._result_items: list[dict] = []
        self._result_folders: dict[str, dict] = {
            "root": {"id": "root", "name": "Results", "parent_id": "", "path": "", "created_at": ""}
        }
        self._result_current_folder_id = "root"
        self._result_next_folder_id = 1
        self._result_next_id = 1
        self._visible_result_entries: list[dict] = []
        self._temporary_run_dirs: set[Path] = set()
        self._active_timing: dict[str, float] = {}
        self._raw_records_cache: list[dict] = []
        self._processed_records_cache: list[dict] = []
        self._build_ui()
        self._ensure_roots()
        self._refresh_database_tables()
        self._refresh_modules()

    def _build_ui(self) -> None:
        self.raw_table = QTableWidget(0, 4)
        self.raw_table.setHorizontalHeaderLabels(["Raw file", "Kind", "Channels", "Stim"])
        self.raw_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.raw_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.raw_table.setMinimumHeight(190)

        self.processed_table = QTableWidget(0, 4)
        self.processed_table.setHorizontalHeaderLabels(["Processed data", "Type", "Samples", "Features"])
        self.processed_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.processed_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.processed_table.setMinimumHeight(190)

        self.refresh_data_button = QPushButton("Refresh Data")
        self.refresh_data_button.clicked.connect(self._refresh_database_tables)

        self.task_text = QTextEdit()
        self.task_text.setPlaceholderText("Describe what the agent should analyze with the selected database rows.")
        self.task_text.setMinimumHeight(90)
        self.task_text.setMaximumHeight(130)
        self.test_timeout = QLineEdit("600")
        self.test_timeout.setVisible(False)


        self.generate_button = QPushButton("Generate Module")
        self.generate_button.clicked.connect(self._generate_module)
        self.modify_button = QPushButton("Modify Selected")
        self.modify_button.clicked.connect(self._modify_selected_module)
        self.run_button = QPushButton("Run Module")
        self.run_button.clicked.connect(self._run_selected_module)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_process)
        self.stop_button.setEnabled(False)

        self.module_table = QTableWidget(0, 4)
        self.module_table.setHorizontalHeaderLabels(["Name", "ID", "Updated", "Summary"])
        self.module_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.module_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.module_table.setMaximumHeight(210)
        self.module_table.itemSelectionChanged.connect(self._module_selection_changed)
        self.module_summary_label = QLabel("Select a module to view the full summary.")
        self.module_summary_label.setObjectName("MutedText")
        self.module_summary_label.setWordWrap(True)
        self.module_summary_label.setMinimumHeight(42)
        self.module_summary_label.setMaximumHeight(86)
        self.rename_button = QPushButton("Rename")
        self.rename_button.clicked.connect(self._rename_selected_module)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self._delete_selected_module)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 8))
        self.log.setMinimumHeight(150)
        self.log.document().setMaximumBlockCount(2000)
        self.run_progress = QProgressBar()
        self.run_progress.setRange(0, 100)
        self.run_progress.setValue(0)
        self.run_progress.setFormat("Idle")
        self.run_progress.setMaximumHeight(18)
        self.output_table = QTableWidget(0, 5)
        self.output_table.setHorizontalHeaderLabels(["Name", "Kind", "Samples", "Features", "Description"])
        self.output_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.output_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.output_table.setMinimumHeight(150)
        self.output_table.itemSelectionChanged.connect(self._result_selection_changed)
        self.output_table.itemDoubleClicked.connect(self._open_selected_result_folder)
        self.result_path_label = QLabel("Results")
        self.result_path_label.setObjectName("MutedText")
        self.result_path_label.setWordWrap(True)
        self.up_folder_button = QPushButton("Up")
        self.up_folder_button.clicked.connect(self._go_result_folder_up)
        self.new_folder_button = QPushButton("New Folder")
        self.new_folder_button.clicked.connect(self._new_result_folder_action)
        self.move_results_button = QPushButton("Move To")
        self.move_results_button.clicked.connect(self._move_selected_results)
        self.rename_result_button = QPushButton("Rename")
        self.rename_result_button.clicked.connect(self._rename_selected_result_entry)
        self.save_results_button = QPushButton("Save Results")
        self.save_results_button.clicked.connect(self._save_last_results)
        self.save_results_button.setEnabled(False)
        self.delete_results_button = QPushButton("Delete")
        self.delete_results_button.clicked.connect(self._delete_selected_results)
        self.preview_canvas = FigureCanvas(Figure(figsize=(8.8, 6.2), tight_layout=True))
        self.preview_canvas.setMinimumSize(640, 420)
        self.preview_canvas.mpl_connect("scroll_event", self._zoom_preview_event)
        self.preview_canvas.mpl_connect("button_press_event", self._preview_press_event)
        self.preview_canvas.mpl_connect("motion_notify_event", self._preview_motion_event)
        self.preview_canvas.mpl_connect("button_release_event", self._preview_release_event)
        self._draw_empty_preview(self.preview_canvas, "Result preview")

        left = QFrame()
        left.setObjectName("Panel")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)
        data_header = QHBoxLayout()
        data_header.addWidget(QLabel("Input database"))
        data_header.addStretch(1)
        data_header.addWidget(self.refresh_data_button)
        left_layout.addLayout(data_header)
        left_layout.addWidget(QLabel("Raw files"))
        left_layout.addWidget(self.raw_table, 2)
        left_layout.addWidget(QLabel("Task"))
        left_layout.addWidget(self.task_text)
        left_layout.addWidget(QLabel("Generated modules"))
        left_layout.addWidget(self.module_table, 2)
        left_layout.addWidget(self.module_summary_label)
        module_buttons = QGridLayout()
        for index, button in enumerate(
            [
                self.generate_button,
                self.modify_button,
                self.rename_button,
                self.delete_button,
                self.run_button,
                self.stop_button,
            ]
        ):
            module_buttons.addWidget(button, index // 3, index % 3)
        left_layout.addLayout(module_buttons)

        log_panel = QFrame()
        log_panel.setObjectName("Panel")
        log_panel.setMaximumHeight(260)
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(8, 8, 8, 8)
        log_layout.setSpacing(4)
        log_layout.addWidget(QLabel("Run log"))
        log_layout.addWidget(self.log, 1)
        log_layout.addWidget(self.run_progress)

        result_panel = QFrame()
        result_panel.setObjectName("Panel")
        result_panel.setMaximumHeight(260)
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(8, 8, 8, 8)
        result_layout.setSpacing(4)
        result_header = QHBoxLayout()
        result_header.addWidget(QLabel("Result list"))
        result_header.addStretch(1)
        result_header.addWidget(self.up_folder_button)
        result_layout.addLayout(result_header)
        result_layout.addWidget(self.result_path_label)
        result_layout.addWidget(self.output_table, 1)
        result_buttons = QHBoxLayout()
        result_buttons.setContentsMargins(0, 0, 0, 0)
        result_buttons.setSpacing(6)
        result_buttons.addWidget(self.new_folder_button)
        result_buttons.addWidget(self.move_results_button)
        result_buttons.addWidget(self.rename_result_button)
        result_buttons.addWidget(self.save_results_button)
        result_buttons.addWidget(self.delete_results_button)
        result_layout.addLayout(result_buttons)

        workspace = QWidget()
        workspace_layout = QGridLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(6)
        workspace_layout.addWidget(log_panel, 0, 0)
        workspace_layout.addWidget(result_panel, 0, 1)
        workspace_layout.addWidget(self.preview_canvas, 1, 0, 1, 2)
        workspace_layout.setColumnStretch(0, 3)
        workspace_layout.setColumnStretch(1, 5)
        workspace_layout.setRowStretch(0, 1)
        workspace_layout.setRowStretch(1, 5)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(left, 3)
        layout.addWidget(workspace, 8)

    def _ensure_roots(self) -> None:
        self.modules_root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def refresh_context(self) -> None:
        self._refresh_database_tables()
        self._refresh_modules()

    def closeEvent(self, event) -> None:
        if self.current_process is not None and self.current_process.state() != QProcess.ProcessState.NotRunning:
            self.current_process.terminate()
            if not self.current_process.waitForFinished(1500):
                self.current_process.kill()
                self.current_process.waitForFinished(500)
        self.current_process = None
        self._cleanup_unsaved_results()
        super().closeEvent(event)

    def _refresh_database_tables(self) -> None:
        raw_selection = self._selected_record_keys(self._raw_records_cache, self.raw_table, key="path")
        processed_selection = self._selected_record_keys(self._processed_records_cache, self.processed_table, key="path")
        self._raw_records_cache = list(self.raw_provider() or [])
        self._processed_records_cache = list(self.processed_provider() or [])
        self._fill_raw_table()
        self._fill_processed_table()
        self._restore_or_apply_provider_selection(
            self.raw_table,
            self._raw_records_cache,
            raw_selection,
            list(self.selected_raw_provider() or []),
            key="path",
        )
        self._restore_or_apply_provider_selection(
            self.processed_table,
            self._processed_records_cache,
            processed_selection,
            list(self.selected_processed_provider() or []),
            key="path",
        )

    def _record_key(self, record: dict, key: str) -> str:
        value = record.get(key)
        if value is None and key == "path":
            value = record.get("source_path", record.get("name", ""))
        return str(value or "").strip()

    def _selected_record_keys(self, records: list[dict], table: QTableWidget, *, key: str) -> set[str]:
        rows = sorted({index.row() for index in table.selectedIndexes()})
        return {
            self._record_key(records[row], key)
            for row in rows
            if 0 <= row < len(records) and self._record_key(records[row], key)
        }

    def _restore_or_apply_provider_selection(
        self,
        table: QTableWidget,
        records: list[dict],
        previous_keys: set[str],
        provider_records: list[dict],
        *,
        key: str,
    ) -> None:
        target_keys = set(previous_keys)
        if not target_keys:
            target_keys = {self._record_key(record, key) for record in provider_records if self._record_key(record, key)}
        if not target_keys:
            return
        table.blockSignals(True)
        try:
            table.clearSelection()
            for row, record in enumerate(records):
                if self._record_key(record, key) in target_keys:
                    table.selectRow(row)
        finally:
            table.blockSignals(False)

    def _fill_raw_table(self) -> None:
        self.raw_table.setRowCount(len(self._raw_records_cache))
        for row, record in enumerate(self._raw_records_cache):
            data = record.get("raw_data")
            name = Path(str(record.get("path", record.get("name", f"raw_{row}")))).name
            kind = str(record.get("data_kind", "raw") or "raw")
            channels = ""
            stim_count = ""
            if hasattr(data, "spikes"):
                channels = str(len(getattr(data, "spikes", {}) or {}))
                stim_times = getattr(data, "stim_times", [])
                stim_count = str(len(stim_times)) if stim_times is not None else "0"
            values = [name, kind, channels, stim_count]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                self.raw_table.setItem(row, column, item)
        self.raw_table.resizeColumnsToContents()

    def _fill_processed_table(self) -> None:
        self.processed_table.setRowCount(len(self._processed_records_cache))
        for row, record in enumerate(self._processed_records_cache):
            matrix = np.asarray(record.get("matrix", []))
            shape = list(matrix.shape)
            samples = str(shape[0]) if len(shape) >= 1 else ""
            features = str(shape[1]) if len(shape) >= 2 else ""
            values = [
                str(record.get("name", f"processed_{row}") or f"processed_{row}"),
                str(record.get("dataset_type", "") or ""),
                samples,
                features,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.processed_table.setItem(row, column, item)
        self.processed_table.resizeColumnsToContents()

    def _selected_raw_records_for_scope(self) -> list[dict]:
        rows = sorted({index.row() for index in self.raw_table.selectedIndexes()})
        return [self._raw_records_cache[row] for row in rows if 0 <= row < len(self._raw_records_cache)]

    def _selected_processed_records_for_scope(self) -> list[dict]:
        rows = sorted({index.row() for index in self.processed_table.selectedIndexes()})
        return [self._processed_records_cache[row] for row in rows if 0 <= row < len(self._processed_records_cache)]

    def _record_signature_key(self, record: dict, *, fallback: str) -> str:
        for key in ("path", "source_path", "name"):
            value = str(record.get(key, "") or "").strip()
            if value:
                return value
        return fallback

    def _input_signature_for_records(self, raw_records: list[dict], processed_records: list[dict]) -> tuple[str, str]:
        raw_keys = [
            self._record_signature_key(record, fallback=f"raw_{index}")
            for index, record in enumerate(raw_records)
        ]
        processed_keys = [
            self._record_signature_key(record, fallback=f"processed_{index}")
            for index, record in enumerate(processed_records)
        ]
        payload = {
            "raw": sorted(raw_keys),
            "processed": sorted(processed_keys),
        }
        signature = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        label_parts = []
        if raw_keys:
            label_parts.append(f"{len(raw_keys)} raw")
        if processed_keys:
            label_parts.append(f"{len(processed_keys)} processed")
        if not label_parts:
            label_parts.append("no selected input")
        preview_names = [Path(item).name for item in [*raw_keys, *processed_keys][:3]]
        label = ", ".join(label_parts)
        if preview_names:
            label = f"{label}: {', '.join(preview_names)}"
            if len(raw_keys) + len(processed_keys) > 3:
                label += "..."
        return signature, label

    def _default_module_name(self) -> str:
        first_line = ""
        for line in self.task_text.toPlainText().splitlines():
            first_line = line.strip()
            if first_line:
                break
        return first_line[:48] or "Agent custom analysis"

    def _selected_module_dir(self) -> Path | None:
        row = self.module_table.currentRow()
        if row < 0:
            return None
        item = self.module_table.item(row, 1)
        if item is None:
            return None
        path = self.modules_root / item.text().strip()
        return path if path.exists() else None

    def _manifest_for(self, module_dir: Path) -> dict:
        path = module_dir / "manifest.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _refresh_modules(self) -> None:
        self._ensure_roots()
        module_dirs = sorted([path for path in self.modules_root.iterdir() if path.is_dir()], key=lambda path: path.name)
        current_id = ""
        selected = self._selected_module_dir()
        if selected is not None:
            current_id = selected.name
        self.module_table.setRowCount(len(module_dirs))
        for row, module_dir in enumerate(module_dirs):
            manifest = self._manifest_for(module_dir)
            updated = manifest.get("updated_at") or manifest.get("created_at") or ""
            full_summary = str(manifest.get("summary", "") or manifest.get("task", "") or "")
            summary = full_summary[:120] + ("..." if len(full_summary) > 120 else "")
            values = [str(manifest.get("name") or module_dir.name), module_dir.name, str(updated), summary]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(full_summary if column == 3 and full_summary else value)
                self.module_table.setItem(row, column, item)
            if module_dir.name == current_id:
                self.module_table.selectRow(row)
        self.module_table.resizeColumnsToContents()
        if self.module_table.rowCount() and self.module_table.currentRow() < 0:
            self.module_table.selectRow(0)

    def _module_selection_changed(self) -> None:
        module_dir = self._selected_module_dir()
        if module_dir is None:
            return
        manifest = self._manifest_for(module_dir)
        summary = str(manifest.get("summary", "") or manifest.get("task", "") or "No summary.")
        self.module_summary_label.setText(summary)
        self.module_summary_label.setToolTip(summary)
        self.log.append(f"Selected module: {manifest.get('name') or module_dir.name}")

    def _new_module_dir(self, name: str) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = _slugify(name, "agent_module")
        candidate = self.modules_root / f"{stamp}_{base}"
        suffix = 2
        while candidate.exists():
            candidate = self.modules_root / f"{stamp}_{base}_{suffix}"
            suffix += 1
        return candidate

    def _write_module_files(self, module_dir: Path, name: str, task: str, *, generated_by: str) -> None:
        module_dir.mkdir(parents=True, exist_ok=True)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest = {
            "id": module_dir.name,
            "name": name,
            "task": task,
            "created_at": now,
            "updated_at": now,
            "generated_by": generated_by,
            "entrypoint": "module.py",
            "summary": task[:200],
        }
        (module_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        (module_dir / "module.py").write_text(_module_template(name), encoding="utf-8")
        readme = f"# {name}\n\nTask:\n\n{task or 'Agent custom analysis module.'}\n"
        (module_dir / "README.md").write_text(readme, encoding="utf-8")

    def _create_template_module(self) -> Path:
        name = self._default_module_name()
        task = self.task_text.toPlainText().strip() or "Agent custom analysis module."
        module_dir = self._new_module_dir(name)
        self._write_module_files(module_dir, name, task, generated_by="shell")
        self.log.append(f"Agent module shell created: {module_dir}")
        self._refresh_modules()
        self._select_module_id(module_dir.name)
        return module_dir

    def _select_module_id(self, module_id: str) -> None:
        for row in range(self.module_table.rowCount()):
            item = self.module_table.item(row, 1)
            if item is not None and item.text() == module_id:
                self.module_table.selectRow(row)
                return

    def _generate_module(self) -> None:
        if self.current_process is not None:
            self.log.append("Another agent command is already running.")
            return
        module_dir = self._create_template_module()
        self._start_codex_module_update(module_dir, modify_existing=False)

    def _modify_selected_module(self) -> None:
        if self.current_process is not None:
            self.log.append("Another agent command is already running.")
            return
        module_dir = self._selected_module_dir()
        if module_dir is None:
            QMessageBox.information(self, "Agent Custom Code", "Select a saved module to modify.")
            return
        self._start_codex_module_update(module_dir, modify_existing=True)

    def _start_codex_module_update(self, module_dir: Path, *, modify_existing: bool) -> None:
        task = self.task_text.toPlainText().strip()
        if modify_existing and not task:
            task = str(self._manifest_for(module_dir).get("task", "") or "").strip()
        task_document = self._agent_prompt(module_dir, task, modify_existing=modify_existing)
        task_path = module_dir / "AGENT_TASK.md"
        task_path.write_text(task_document, encoding="utf-8")
        agents_path = module_dir / "AGENTS.md"
        agents_path.write_text(task_document, encoding="utf-8")
        self._update_module_task_metadata(module_dir, task)
        task_line = " ".join((task or self._manifest_for(module_dir).get("task", "") or "").split())
        if not task_line:
            task_line = "Generate a custom analysis module from the selected input data."
        task_line = task_line[:500]
        prompt = (
            f"Task: {task_line}. "
            "Read AGENTS.md and AGENT_TASK.md in this directory and implement module.py now. "
            "Do not ask clarification questions. Edit files before finishing."
        )
        config = _load_agent_config_file()
        backend = str(config.get("backend") or _default_agent_backend_value())
        command = str(config.get("command") or _default_agent_command_value(backend))
        args_template = str(config.get("args_template") or _default_agent_args_template_value(backend))
        if not _resolve_agent_executable(command):
            self.log.append("Agent command not found; run the preflight environment check again.")
            return
        program, arguments, working_dir = _agent_process_command(
            backend=backend,
            command=command,
            args_template=args_template,
            module_dir=module_dir,
            prompt=prompt,
            task_path=task_path,
        )
        process = QProcess(self)
        process.setProgram(program)
        process.setArguments(arguments)
        process.setWorkingDirectory(str(working_dir))
        process.readyReadStandardOutput.connect(self._read_process_stdout)
        process.readyReadStandardError.connect(self._read_process_stderr)
        process.finished.connect(self._process_finished)
        process.errorOccurred.connect(self._process_error)
        self.current_process = process
        self.current_mode = "generate"
        self.current_module_dir = module_dir
        self._process_snapshot_root = module_dir
        self._process_file_snapshot = _file_snapshot(module_dir)
        self._active_timing = {"started_at": time.perf_counter()}
        self._set_busy(True)
        self._set_run_progress_busy("Generating module...")
        self.log.append("Agent task prepared.")
        self.log.append(f"$ {_command_for_log(program, arguments)}")
        process.start()

    def _update_module_task_metadata(self, module_dir: Path, task: str) -> None:
        manifest = self._manifest_for(module_dir)
        if not manifest:
            return
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        if task.strip():
            manifest["task"] = task.strip()
        manifest["updated_at"] = now
        (module_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        readme = f"# {manifest.get('name') or module_dir.name}\n\nTask:\n\n{manifest.get('task') or 'Agent custom analysis module.'}\n\nSee AGENT_TASK.md for the full generation contract.\n"
        (module_dir / "README.md").write_text(readme, encoding="utf-8")

    def _agent_prompt(self, module_dir: Path, task: str, *, modify_existing: bool = False) -> str:
        selected_raw = self._selected_raw_records_for_scope()
        selected_processed = self._selected_processed_records_for_scope()
        input_summary = self._agent_input_summary(selected_raw, selected_processed)
        manifest = self._manifest_for(module_dir)
        existing_summary = str(manifest.get("summary", "") or manifest.get("task", "") or "").strip()
        mode_text = (
            "Modify the existing module.py in place according to the user task. Preserve existing functionality by default."
            if modify_existing
            else "Generate module.py according to the user task. The initial file is only an interface shell; replace the placeholder analyze(context) implementation."
        )
        implementation_text = (
            "The current module.py is the baseline implementation. Extend it carefully and keep old behavior compatible unless explicit removal is requested."
            if modify_existing
            else "The initial module.py shell is not an analysis template. Treat the user task as the source of truth and write the analysis code yourself."
        )
        task_text = task.strip() or "Generate a custom analysis module from the selected input data."
        return (
            "# Mandatory Agent Instructions\n\n"
            "You already have the complete user task below. Do not infer a different task from filenames, scaffold code, or directory contents.\n"
            "Do not ask clarification questions. Implement the task directly by editing module.py.\n\n"
            "## User Task\n"
            f"{task_text}\n\n"
            "You are running inside Codex CLI non-interactively. Do not reply with questions or instructions for the user.\n"
            "You must edit module.py in the current module directory before finishing.\n"
            "If details are ambiguous, make a reasonable implementation from the selected input summary and document assumptions in README.md.\n\n"
            "Context: you are generating a runtime custom data-processing module for the MEA Pipeline GUI.\n"
            f"Only edit files inside this directory: {module_dir}\n"
            "Do not edit repository source files, do not use git, do not control hardware.\n"
            "Do not open GUI windows, do not call plt.show(), and do not require interactive input.\n"
            "The module must define analyze(context: dict) -> dict in module.py.\n"
            f"{implementation_text}\n\n"
            f"Existing module summary: {existing_summary or 'No existing implementation summary.'}\n"
            "Modification semantics: if this is a modification, treat requests such as add, also support, extend, include, or 增加/添加 as additive changes.\n"
            "Do not delete, replace, or simplify existing analyses, outputs, helper functions, parameters, result formats, or figures unless the user explicitly asks to remove or replace them.\n"
            "Read the current module.py before editing it, keep compatible behavior for old tasks, and add the new behavior alongside the existing behavior.\n\n"
            "After implementing, update the module description so it accurately describes the current module behavior, not just the user request.\n"
            "Write a concise implementation summary in manifest.json['summary']; preserve manifest id/name/created_at/generated_by/entrypoint when editing it.\n"
            "Also add or update a module-level or analyze() docstring in module.py with the same current-functionality summary, and add a README.md Summary section.\n\n"
            "Input data paths are provided in context['selected_raw_records'] and context['selected_processed_records'].\n"
            "Write every generated output file or subfolder under context['output_dir']; the GUI treats that directory as one result-history folder for this module run.\n"
            "Do not write generated result files outside context['output_dir'].\n"
            "Raw spike input package (.npz) keys: channels, spike_times, stim_times, sampling_rate, metadata_json. spike_times is an object array aligned to channels.\n"
            "Processed input package (.npz) keys: matrix, sample_labels, feature_labels, metadata_json.\n"
            "When multiple raw or processed records are selected, iterate over every selected record. Do not use only the first record unless the user explicitly asks for that.\n"
            "For cross-file channel comparisons, align rows by channel/electrode label, use sample_labels for channels/electrodes, and use feature_labels for file/condition names.\n"
            "If the task asks to compare files, preserve per-file values as separate columns or separate processed_records; do not collapse them into one global summary.\n"
            "If the task asks for one plot/result per channel/electrode, return one processed_record per channel/electrode. Each such record should have a 1 x N matrix, feature_labels as file/condition names, sample_labels containing the channel/electrode label, and value_label for the measured value.\n"
            "Performance requirements: load each input .npz file at most once, cache arrays in local variables, prefer numpy vectorized operations over Python loops where practical, avoid repeated per-channel file I/O, and avoid generating many high-resolution figures unless the user explicitly asks for them.\n"
            "For large channel/file comparisons, return numeric processed_records as the primary output and create only lightweight preview figures needed by the task. Use matplotlib Agg/non-interactive saving if figures are required.\n"
            "Keep runtime memory bounded: do not duplicate large matrices unnecessarily, convert labels to compact Python lists only for output metadata, and close matplotlib figures after saving them.\n"
            "Do not return NaN or infinite values in output matrices. If a statistic is undefined, replace it with 0.0 and mention the replacement in summary.\n"
            "Return {'processed_records': [...], 'figures': [], 'summary': '...'}.\n"
            "Each processed record must include name, dataset_type, matrix or matrix_path, sample_labels, feature_labels, description. Optional value_label and x_axis_label improve GUI plotting.\n\n"
            f"{mode_text}\n"
            f"The user currently selected {len(selected_raw)} raw record(s) and {len(selected_processed)} processed record(s) as intended inputs.\n\n"
            f"Selected input summary:\n{input_summary}\n\n"
            "Required final state: module.py must contain a concrete analyze(context) implementation and must not leave the placeholder RuntimeError.\n"
        )

    def _agent_input_summary(self, raw_records: list[dict], processed_records: list[dict]) -> str:
        lines = []
        for index, record in enumerate(raw_records[:12], start=1):
            data = record.get("raw_data")
            name = Path(str(record.get("path", record.get("name", f"raw {index}")))).name
            channels = len(getattr(data, "spikes", {}) or {}) if hasattr(data, "spikes") else "unknown"
            stim_values = getattr(data, "stim_times", None) if hasattr(data, "stim_times") else None
            stim_count = len(stim_values) if stim_values is not None else 0
            lines.append(
                f"- raw[{index}]: name={name!r}, kind={str(record.get('data_kind', 'raw'))!r}, channels={channels}, stim_times={stim_count}"
            )
        if len(raw_records) > 12:
            lines.append(f"- ... {len(raw_records) - 12} more raw record(s)")
        for index, record in enumerate(processed_records[:12], start=1):
            matrix = np.asarray(record.get("matrix", []))
            sample_labels = list(record.get("sample_labels", []) or [])
            feature_labels = list(record.get("feature_labels", []) or [])
            lines.append(
                f"- processed[{index}]: name={str(record.get('name', f'processed {index}'))!r}, type={str(record.get('dataset_type', ''))!r}, shape={tuple(matrix.shape)}, sample_labels={sample_labels[:5]}, feature_labels={feature_labels[:5]}"
            )
        if len(processed_records) > 12:
            lines.append(f"- ... {len(processed_records) - 12} more processed record(s)")
        return "\n".join(lines) if lines else "- no selected input records"

    def _set_busy(self, busy: bool) -> None:
        for button in [
            self.generate_button,
            self.modify_button,
            self.run_button,
            self.delete_button,
            self.rename_button,
            self.refresh_data_button,
            self.delete_results_button,
            self.save_results_button,
            self.up_folder_button,
            self.new_folder_button,
            self.move_results_button,
            self.rename_result_button,
        ]:
            button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        if not busy:
            self._sync_result_buttons()

    def _set_run_progress_idle(self) -> None:
        self.run_progress.setRange(0, 100)
        self.run_progress.setValue(0)
        self.run_progress.setFormat("Idle")

    def _set_run_progress_busy(self, label: str) -> None:
        self.run_progress.setRange(0, 0)
        self.run_progress.setFormat(label)

    def _set_run_progress_done(self, label: str, *, ok: bool = True) -> None:
        self.run_progress.setRange(0, 100)
        self.run_progress.setValue(100 if ok else 0)
        self.run_progress.setFormat(label)

    def _stop_process(self) -> None:
        if self.current_process is None:
            return
        self.log.append("Stopping process...")
        self._set_run_progress_busy("Stopping...")
        self.current_process.terminate()
        QTimer.singleShot(2000, self._kill_process_if_needed)

    def _kill_process_if_needed(self) -> None:
        if self.current_process is not None and self.current_process.state() != QProcess.ProcessState.NotRunning:
            self.current_process.kill()

    def _read_process_stdout(self) -> None:
        if self.current_process is None:
            return
        text = _decode_process_output(self.current_process.readAllStandardOutput())
        for line in _agent_log_lines(text, stream="stdout"):
            self.log.append(line)

    def _read_process_stderr(self) -> None:
        if self.current_process is None:
            return
        text = _decode_process_output(self.current_process.readAllStandardError())
        for line in _agent_log_lines(text, stream="stderr"):
            self.log.append(line)

    def _process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        mode = self.current_mode
        module_dir = self.current_module_dir
        snapshot_root = self._process_snapshot_root
        snapshot = dict(self._process_file_snapshot)
        finished_at = time.perf_counter()
        timing = dict(self._active_timing)
        self.log.append(f"Process finished: code={exit_code}, status={exit_status.name}")
        self.current_process = None
        if snapshot_root is not None:
            changed = _changed_files(snapshot_root, snapshot)
            if changed:
                label = "Output files" if mode == "run" else "Modified files"
                self.log.append(f"{label}: " + ", ".join(changed[:24]))
                if len(changed) > 24:
                    self.log.append(f"... {len(changed) - 24} more file(s)")
        if mode == "generate":
            self._refresh_modules()
            self._module_selection_changed()
            self._check_generated_module(module_dir, exit_code)
        elif mode == "run":
            result_load_started_at = time.perf_counter()
            self._load_run_result(exit_code, module_dir)
            result_load_s = time.perf_counter() - result_load_started_at
            input_s = float(timing.get("input_export_s", 0.0) or 0.0)
            process_started_at = float(timing.get("process_started_at", 0.0) or 0.0)
            module_s = max(0.0, finished_at - process_started_at) if process_started_at else 0.0
            total_started_at = float(timing.get("started_at", 0.0) or 0.0)
            total_s = max(0.0, time.perf_counter() - total_started_at) if total_started_at else input_s + module_s + result_load_s
            self.log.append(
                f"Timing: input export {input_s:.2f}s, module run {module_s:.2f}s, result load {result_load_s:.2f}s, total {total_s:.2f}s."
            )
        if mode == "run":
            self._set_run_progress_done("Run completed" if exit_code == 0 else "Run failed", ok=(exit_code == 0))
        elif mode == "generate":
            started_at = float(timing.get("started_at", 0.0) or 0.0)
            if started_at:
                self.log.append(f"Timing: generation {max(0.0, finished_at - started_at):.2f}s.")
            self._set_run_progress_done("Generation completed" if exit_code == 0 else "Generation failed", ok=(exit_code == 0))
        self.current_mode = ""
        self.current_module_dir = None
        self.current_run_dir = None
        self.current_output_dir = None
        self.current_result_folder_name = ""
        self.current_result_folder_id = ""
        self.current_input_signature = ""
        self.current_input_label = ""
        self._process_snapshot_root = None
        self._process_file_snapshot = {}
        self._active_timing = {}
        self._set_busy(False)
        self._sync_result_buttons()

    def _process_error(self, error: QProcess.ProcessError) -> None:
        self.log.append(f"Process error: {error.name}")
        if self.current_result_folder_id in self._result_folders:
            folder = self._result_folders[self.current_result_folder_id]
            folder["status"] = "failed"
            folder["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._refresh_result_table()
            self._select_result_folder_id(self.current_result_folder_id)
        self.current_process = None
        self.current_mode = ""
        self.current_module_dir = None
        self.current_run_dir = None
        self.current_output_dir = None
        self.current_result_folder_name = ""
        self.current_result_folder_id = ""
        self.current_input_signature = ""
        self.current_input_label = ""
        self._process_snapshot_root = None
        self._process_file_snapshot = {}
        self._active_timing = {}
        self._set_run_progress_done("Process error", ok=False)
        self._set_busy(False)
        self._sync_result_buttons()

    def _check_generated_module(self, module_dir: Path | None, exit_code: int) -> None:
        if module_dir is None:
            return
        task_path = module_dir / "AGENT_TASK.md"
        if not task_path.exists():
            self.log.append("Agent task document is missing; generation prompt may not have been written.")
        agents_path = module_dir / "AGENTS.md"
        if not agents_path.exists():
            self.log.append("Agent instructions document is missing; Codex may not have loaded the full task.")
        code_path = module_dir / "module.py"
        if not code_path.exists():
            self.log.append("Agent generation failed: module.py is missing.")
            return
        try:
            code, removed_bom = _read_python_text_without_bom(code_path)
        except Exception as exc:
            self.log.append(f"Could not inspect generated module.py: {exc}")
            return
        if removed_bom:
            self.log.append("Removed UTF-8 BOM from generated module.py.")
        placeholder = "Agent analysis code has not been generated yet" in code
        has_analyze = "def analyze(" in code
        if exit_code != 0:
            self.log.append("Agent generation process failed; module.py may still be the placeholder shell.")
        elif placeholder or not has_analyze:
            self.log.append(
                "Agent did not generate executable analysis code. Check that Task is filled and input data is selected, then click Modify Selected."
            )
        else:
            try:
                compile(code, str(code_path), "exec")
            except SyntaxError as exc:
                self.log.append(f"Agent generated module.py but it has a syntax error: line {exc.lineno}: {exc.msg}")
                return
            summary = self._refresh_module_summary_from_generated_files(module_dir, code)
            if summary:
                self.log.append(f"Module summary updated: {summary}")
            self.log.append("Agent generated module.py successfully.")
            self.task_text.clear()

    def _refresh_module_summary_from_generated_files(self, module_dir: Path, code: str) -> str:
        manifest = self._manifest_for(module_dir)
        if not manifest:
            return ""
        task = _compact_module_summary(str(manifest.get("task", "") or ""))
        current_summary = _compact_module_summary(str(manifest.get("summary", "") or ""))
        code_summary = _summary_from_python_code(code)
        readme_summary = _summary_from_readme(module_dir / "README.md")
        summary = code_summary or readme_summary or current_summary or task
        summary = _compact_module_summary(summary)
        if not summary:
            return ""
        manifest["summary"] = summary
        manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (module_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        readme_path = module_dir / "README.md"
        existing = readme_path.read_text(encoding="utf-8-sig", errors="replace") if readme_path.exists() else ""
        if not re.search(r"^#+\s*(summary|module summary|功能摘要)\b", existing, flags=re.IGNORECASE | re.MULTILINE):
            title = manifest.get("name") or module_dir.name
            readme_path.write_text(
                f"# {title}\n\n## Summary\n\n{summary}\n\n## Task\n\n{manifest.get('task') or 'Agent custom analysis module.'}\n",
                encoding="utf-8",
            )
        self._refresh_modules()
        self._select_module_id(module_dir.name)
        self._module_selection_changed()
        return summary

    def _run_selected_module(self) -> None:
        if self.current_process is not None:
            self.log.append("Another process is already running.")
            return
        module_dir = self._selected_module_dir()
        if module_dir is None:
            QMessageBox.information(self, "Agent Custom Code", "Select or create a module first.")
            return
        run_dir = self.runs_root / time.strftime("%Y%m%d_%H%M%S")
        suffix = 2
        while run_dir.exists():
            run_dir = self.runs_root / f"{time.strftime('%Y%m%d_%H%M%S')}_{suffix}"
            suffix += 1
        input_dir = run_dir / "input"
        selected_raw = self._selected_raw_records_for_scope()
        selected_processed = self._selected_processed_records_for_scope()
        input_signature, input_label = self._input_signature_for_records(selected_raw, selected_processed)
        result_parent_id = self._result_current_folder_id if self._result_current_folder_id in self._result_folders else "root"
        existing_folder_id = self._find_result_folder_for_run(module_dir, input_signature)
        if existing_folder_id:
            existing_folder = self._result_folders.get(existing_folder_id, {})
            result_parent_id = str(existing_folder.get("parent_id", "") or "root")
            result_folder_name = str(existing_folder.get("name", "") or self._module_result_folder_name(module_dir))
        else:
            result_folder_name = self._unique_result_child_name(
                result_parent_id,
                self._module_result_folder_name(module_dir),
            )
        output_dir = run_dir / "output" / _slugify(result_folder_name, "result")
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        result_folder_id = self._prepare_result_folder_for_run(
            module_dir,
            parent_id=result_parent_id,
            output_dir=output_dir,
            run_id=run_dir.name,
            input_signature=input_signature,
            input_label=input_label,
        )
        self._result_current_folder_id = result_parent_id
        self._refresh_result_table()
        self._select_result_folder_id(result_folder_id)
        self._draw_empty_preview(self.preview_canvas, f"Folder: {result_folder_name}\nRunning...\n{input_label}")
        run_started_at = time.perf_counter()
        self._set_run_progress_busy("Preparing input...")
        QApplication.processEvents()
        input_started_at = time.perf_counter()
        manifest_path = self._export_input_manifest(input_dir)
        input_export_s = time.perf_counter() - input_started_at
        process = QProcess(self)
        process.setProgram(sys.executable)
        runtime_args = [
            "-m",
            _runtime_module_name(),
            "--module",
            str(module_dir / "module.py"),
            "--input",
            str(manifest_path),
            "--output",
            str(output_dir),
        ]
        process.setArguments(runtime_args)
        process.setWorkingDirectory(str(_repo_root()))
        process.readyReadStandardOutput.connect(self._read_process_stdout)
        process.readyReadStandardError.connect(self._read_process_stderr)
        process.finished.connect(self._process_finished)
        process.errorOccurred.connect(self._process_error)
        self.current_process = process
        self.current_mode = "run"
        self.current_module_dir = module_dir
        self.current_run_dir = run_dir
        self.current_output_dir = output_dir
        self.current_result_folder_name = result_folder_name
        self.current_result_folder_id = result_folder_id
        self.current_input_signature = input_signature
        self.current_input_label = input_label
        self._temporary_run_dirs.add(run_dir)
        self._process_snapshot_root = run_dir
        self._process_file_snapshot = _file_snapshot(run_dir)
        self._active_timing = {
            "started_at": run_started_at,
            "input_export_s": input_export_s,
            "process_started_at": time.perf_counter(),
        }
        self.last_result = None
        self._sync_result_buttons()
        self._set_busy(True)
        self._set_run_progress_busy("Running module...")
        timeout_s = max(1, self._timeout_seconds())
        QTimer.singleShot(timeout_s * 1000, lambda process=process: self._timeout_process(process))
        self.log.append(f"$ {_command_for_log(sys.executable, runtime_args)}")
        process.start()

    def _timeout_seconds(self) -> int:
        try:
            return int(float(self.test_timeout.text().strip()))
        except ValueError:
            return 600

    def _timeout_process(self, process: QProcess | None = None) -> None:
        if process is not None and process is not self.current_process:
            return
        if self.current_process is not None and self.current_mode == "run":
            self.log.append("Run timed out.")
            self._stop_process()

    def _export_input_manifest(self, input_dir: Path) -> Path:
        raw_entries = []
        for index, record in enumerate(self._selected_raw_records_for_scope()):
            raw_entries.append(self._export_raw_record(record, input_dir, index))
        processed_entries = []
        for index, record in enumerate(self._selected_processed_records_for_scope()):
            processed_entries.append(self._export_processed_record(record, input_dir, index))
        manifest = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "selected_raw_records": raw_entries,
            "selected_processed_records": processed_entries,
            "parameters": {
                "task": self.task_text.toPlainText().strip(),
                "input_scope": "selected_database_rows",
            },
        }
        path = input_dir / "input_manifest.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _export_raw_record(self, record: dict, input_dir: Path, index: int) -> dict:
        data = record.get("raw_data")
        label = Path(str(record.get("path", f"raw_{index}"))).name
        data_path = input_dir / f"raw_{index:03d}.npz"
        if hasattr(data, "spikes"):
            channel_keys = sorted(list(data.spikes), key=lambda value: str(value))
            channels = [str(channel) for channel in channel_keys]
            spike_times = np.asarray([np.asarray(data.spikes[channel], dtype=float) for channel in channel_keys], dtype=object)
            stim_times = np.asarray(getattr(data, "stim_times", []), dtype=float)
            sampling_rate = float(getattr(data, "sr", 0.0) or 0.0)
            meta_json = json.dumps(_json_safe(getattr(data, "meta", {}) or {}), ensure_ascii=False)
            np.savez(
                data_path,
                channels=np.asarray(channels, dtype=object),
                spike_times=spike_times,
                stim_times=stim_times,
                sampling_rate=np.asarray([sampling_rate], dtype=float),
                metadata_json=np.asarray([meta_json], dtype=object),
            )
            kind = "spike"
        else:
            values = np.asarray(data, dtype=float)
            np.savez(data_path, matrix=values)
            kind = "array"
        return {
            "name": label,
            "source_path": str(record.get("path", "")),
            "data_kind": str(record.get("data_kind", kind)),
            "kind": kind,
            "data_path": str(data_path),
        }

    def _export_processed_record(self, record: dict, input_dir: Path, index: int) -> dict:
        data_path = input_dir / f"processed_{index:03d}.npz"
        matrix = np.asarray(record.get("matrix", []), dtype=float)
        sample_labels = [str(item) for item in list(record.get("sample_labels", []))]
        feature_labels = [str(item) for item in list(record.get("feature_labels", []))]
        metadata_json = json.dumps(_json_safe(record), ensure_ascii=False)
        np.savez(
            data_path,
            matrix=matrix,
            sample_labels=np.asarray(sample_labels, dtype=object),
            feature_labels=np.asarray(feature_labels, dtype=object),
            metadata_json=np.asarray([metadata_json], dtype=object),
        )
        return {
            "name": str(record.get("name", f"processed_{index}")),
            "source_path": str(record.get("source_path", record.get("path", ""))),
            "dataset_type": str(record.get("dataset_type", "")),
            "data_path": str(data_path),
            "sample_labels": sample_labels,
            "feature_labels": feature_labels,
        }

    def _module_result_folder_name(self, module_dir: Path) -> str:
        manifest = self._manifest_for(module_dir)
        name = str(manifest.get("name") or module_dir.name or "agent result").strip()
        return name or "agent result"

    def _result_folder_path_parts(self, folder_id: str | None = None) -> list[str]:
        folder_id = str(folder_id or self._result_current_folder_id or "root")
        parts = []
        seen = set()
        while folder_id and folder_id != "root" and folder_id not in seen:
            seen.add(folder_id)
            folder = self._result_folders.get(folder_id)
            if not folder:
                break
            parts.append(str(folder.get("name") or folder_id))
            folder_id = str(folder.get("parent_id") or "root")
        return ["Results", *reversed(parts)]

    def _result_folder_path_text(self, folder_id: str | None = None) -> str:
        return " / ".join(self._result_folder_path_parts(folder_id))

    def _result_child_names(self, parent_id: str, *, exclude_id: str = "") -> set[str]:
        names = set()
        for folder_id, folder in self._result_folders.items():
            if folder_id == exclude_id:
                continue
            if str(folder.get("parent_id", "")) == parent_id:
                names.add(str(folder.get("name", "")).strip().lower())
        for item in self._result_items:
            if str(item.get("id", "")) == exclude_id:
                continue
            if str(item.get("folder_id", "root")) == parent_id:
                names.add(self._result_item_display_name(item).strip().lower())
        return names

    def _unique_result_child_name(self, parent_id: str, base_name: str) -> str:
        base = str(base_name or "result").strip() or "result"
        existing = self._result_child_names(parent_id)
        if base.lower() not in existing:
            return base
        suffix = 2
        while f"{base} {suffix}".lower() in existing:
            suffix += 1
        return f"{base} {suffix}"

    def _create_result_folder(
        self,
        name: str,
        *,
        parent_id: str | None = None,
        path: Path | None = None,
        run_id: str = "",
        module_manifest: dict | None = None,
        module_id: str = "",
        input_signature: str = "",
        input_label: str = "",
        status: str = "",
    ) -> str:
        parent = str(parent_id or self._result_current_folder_id or "root")
        folder_id = f"folder_{self._result_next_folder_id}"
        self._result_next_folder_id += 1
        self._result_folders[folder_id] = {
            "id": folder_id,
            "name": str(name or folder_id),
            "parent_id": parent,
            "path": str(path or ""),
            "run_id": str(run_id or ""),
            "module_manifest": dict(module_manifest or {}),
            "module_id": str(module_id or ""),
            "input_signature": str(input_signature or ""),
            "input_label": str(input_label or ""),
            "status": str(status or ""),
            "saved": False,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return folder_id

    def _find_result_folder_for_run(self, module_dir: Path, input_signature: str) -> str:
        module_id = module_dir.name
        for folder_id, folder in self._result_folders.items():
            if folder_id == "root":
                continue
            if str(folder.get("module_id", "")) == module_id and str(folder.get("input_signature", "")) == input_signature:
                return folder_id
        return ""

    def _prepare_result_folder_for_run(
        self,
        module_dir: Path,
        *,
        parent_id: str,
        output_dir: Path,
        run_id: str,
        input_signature: str,
        input_label: str,
    ) -> str:
        module_manifest = self._manifest_for(module_dir)
        folder_id = self._find_result_folder_for_run(module_dir, input_signature)
        if folder_id:
            folder = self._result_folders[folder_id]
            old_path = Path(str(folder.get("path", "")))
            if old_path.exists() and self._is_within_runs_root(old_path):
                old_run_dir = self._run_dir_for_output_path(old_path)
                if old_run_dir is not None and old_run_dir.exists() and self._is_within_runs_root(old_run_dir):
                    shutil.rmtree(old_run_dir, ignore_errors=True)
                    self._temporary_run_dirs.discard(old_run_dir)
            self._result_items = [item for item in self._result_items if str(item.get("folder_id", "root")) != folder_id]
            folder.update(
                {
                    "path": str(output_dir),
                    "run_id": str(run_id),
                    "module_manifest": dict(module_manifest or {}),
                    "input_label": input_label,
                    "status": "running",
                    "saved": False,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        else:
            base_name = self._module_result_folder_name(module_dir)
            folder_name = self._unique_result_child_name(parent_id, base_name)
            folder_id = self._create_result_folder(
                folder_name,
                parent_id=parent_id,
                path=output_dir,
                run_id=run_id,
                module_manifest=module_manifest,
                module_id=module_dir.name,
                input_signature=input_signature,
                input_label=input_label,
                status="running",
            )
        return folder_id

    def _select_result_folder_id(self, folder_id: str) -> None:
        target = str(folder_id or "")
        for row, entry in enumerate(self._visible_result_entries):
            if entry.get("entry_type") == "folder" and str(entry.get("folder_id", "")) == target:
                self.output_table.clearSelection()
                self.output_table.selectRow(row)
                return

    def _visible_result_entries_for_current_folder(self) -> list[dict]:
        current = str(self._result_current_folder_id or "root")
        entries = []
        folders = [
            folder
            for folder_id, folder in self._result_folders.items()
            if folder_id != "root" and str(folder.get("parent_id", "")) == current
        ]
        folders.sort(key=lambda item: str(item.get("name", "")).lower())
        for folder in folders:
            entries.append({"entry_type": "folder", "folder_id": str(folder.get("id", "")), "folder": folder})
        items = [item for item in self._result_items if str(item.get("folder_id", "root")) == current]
        items.sort(key=lambda item: (str(item.get("run_id", "")), self._result_item_display_name(item).lower()))
        for item in items:
            entries.append({"entry_type": "item", "item_id": str(item.get("id", "")), "item": item})
        return entries

    def _result_item_index(self, item_id: str) -> int:
        for index, item in enumerate(self._result_items):
            if str(item.get("id", "")) == str(item_id):
                return index
        return -1

    def _selected_result_entries(self) -> list[dict]:
        rows = sorted({index.row() for index in self.output_table.selectedIndexes()})
        return [self._visible_result_entries[row] for row in rows if 0 <= row < len(self._visible_result_entries)]

    def _selected_result_item_indexes(self) -> list[tuple[int, dict]]:
        selected = []
        for entry in self._selected_result_entries():
            if entry.get("entry_type") != "item":
                continue
            index = self._result_item_index(str(entry.get("item_id", "")))
            if 0 <= index < len(self._result_items):
                selected.append((index, self._result_items[index]))
        return selected

    def _result_folder_descendant_ids(self, folder_id: str) -> set[str]:
        root_id = str(folder_id or "")
        ids = {root_id} if root_id else set()
        changed = True
        while changed:
            changed = False
            for child_id, folder in self._result_folders.items():
                if child_id in ids:
                    continue
                if str(folder.get("parent_id", "")) in ids:
                    ids.add(child_id)
                    changed = True
        return ids

    def _selected_folder_result_item_indexes(self) -> list[tuple[int, dict]]:
        selected_folder_ids = {
            str(entry.get("folder_id", ""))
            for entry in self._selected_result_entries()
            if entry.get("entry_type") == "folder"
        }
        if not selected_folder_ids:
            return []
        folder_ids: set[str] = set()
        for folder_id in selected_folder_ids:
            folder_ids.update(self._result_folder_descendant_ids(folder_id))
        return [
            (index, item)
            for index, item in enumerate(self._result_items)
            if str(item.get("folder_id", "root")) in folder_ids
        ]

    def _open_selected_result_folder(self) -> None:
        entries = self._selected_result_entries()
        if len(entries) != 1 or entries[0].get("entry_type") != "folder":
            return
        folder_id = str(entries[0].get("folder_id", ""))
        if folder_id in self._result_folders:
            self._result_current_folder_id = folder_id
            self._refresh_result_table()
            self._result_selection_changed()

    def _go_result_folder_up(self) -> None:
        current = self._result_folders.get(str(self._result_current_folder_id), {})
        parent_id = str(current.get("parent_id") or "root")
        self._result_current_folder_id = parent_id if parent_id in self._result_folders else "root"
        self._refresh_result_table()
        self._result_selection_changed()

    def _new_result_folder_action(self) -> None:
        name, accepted = QInputDialog.getText(self, "New result folder", "Folder name:", text="New folder")
        if not accepted:
            return
        name = name.strip()
        if not name:
            return
        if name.lower() in self._result_child_names(self._result_current_folder_id):
            QMessageBox.warning(self, "Duplicate name", f'"{name}" already exists in this result folder.')
            return
        self._create_result_folder(name, parent_id=self._result_current_folder_id)
        self._refresh_result_table()

    def _all_result_folder_choices(self) -> list[tuple[str, str]]:
        choices = [("Results", "root")]
        folder_ids = [folder_id for folder_id in self._result_folders if folder_id != "root"]
        folder_ids.sort(key=lambda value: self._result_folder_path_text(value).lower())
        for folder_id in folder_ids:
            choices.append((self._result_folder_path_text(folder_id), folder_id))
        return choices

    def _move_selected_results(self) -> None:
        selected = self._selected_result_item_indexes()
        if not selected:
            QMessageBox.information(self, "Agent Custom Code", "Select result files to move.")
            return
        choices = self._all_result_folder_choices()
        labels = [label for label, _folder_id in choices]
        current_label = self._result_folder_path_text(self._result_current_folder_id)
        index = max(0, labels.index(current_label) if current_label in labels else 0)
        label, accepted = QInputDialog.getItem(self, "Move results", "Destination folder:", labels, index, False)
        if not accepted:
            return
        destination = dict(choices).get(label, "root")
        selected_ids = {str(item.get("id", "")) for _index, item in selected}
        existing = set()
        for folder_id, folder in self._result_folders.items():
            if str(folder.get("parent_id", "")) == destination:
                existing.add(str(folder.get("name", "")).strip().lower())
        for item in self._result_items:
            if str(item.get("id", "")) in selected_ids:
                continue
            if str(item.get("folder_id", "root")) == destination:
                existing.add(self._result_item_display_name(item).strip().lower())
        moving_names = []
        for _index, item in selected:
            name = self._result_item_display_name(item).strip()
            key = name.lower()
            if key in existing or key in moving_names:
                QMessageBox.warning(self, "Duplicate name", f'"{name}" already exists in the destination result folder.')
                return
            moving_names.append(key)
        for _index, item in selected:
            item["folder_id"] = destination
        self._refresh_result_table()
        self._result_selection_changed()

    def _rename_selected_result_entry(self) -> None:
        entries = self._selected_result_entries()
        if len(entries) != 1:
            QMessageBox.information(self, "Agent Custom Code", "Select one result or folder to rename.")
            return
        entry = entries[0]
        if entry.get("entry_type") == "folder":
            folder = entry.get("folder", {})
            entry_id = str(folder.get("id", ""))
            parent_id = str(folder.get("parent_id", "root"))
            current = str(folder.get("name", "") or entry_id)
        else:
            item = entry.get("item", {})
            entry_id = str(item.get("id", ""))
            parent_id = str(item.get("folder_id", "root"))
            current = self._result_item_display_name(item)
        name, accepted = QInputDialog.getText(self, "Rename result entry", "Name:", text=current)
        if not accepted:
            return
        name = name.strip()
        if not name:
            return
        if name.lower() in self._result_child_names(parent_id, exclude_id=entry_id):
            QMessageBox.warning(self, "Duplicate name", f'"{name}" already exists in this result folder.')
            return
        if entry.get("entry_type") == "folder":
            self._result_folders[entry_id]["name"] = name
        else:
            self._rename_result_item(entry.get("item", {}), name)
        self._refresh_result_table()

    def _rename_result_item(self, item: dict, name: str) -> None:
        if item.get("kind") == "figure":
            figure = item.get("figure")
            if isinstance(figure, dict):
                figure["name"] = name
            else:
                item["figure"] = {"name": name, "path": str(figure)}
            return
        record = item.get("record")
        if isinstance(record, dict):
            record["name"] = name

    def _is_within_runs_root(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
            runs_root = self.runs_root.resolve()
        except OSError:
            return False
        return resolved == runs_root or runs_root in resolved.parents

    def _run_dir_for_output_path(self, path: Path) -> Path | None:
        try:
            resolved = path.resolve()
            runs_root = self.runs_root.resolve()
        except OSError:
            return None
        for candidate in [resolved, *resolved.parents]:
            if candidate.parent == runs_root:
                return candidate
        return None

    def _delete_result_folder_recursive(self, folder_id: str) -> None:
        folder = self._result_folders.get(folder_id)
        if not folder or folder_id == "root":
            return
        for child_id, child in list(self._result_folders.items()):
            if str(child.get("parent_id", "")) == folder_id:
                self._delete_result_folder_recursive(child_id)
        self._result_items = [item for item in self._result_items if str(item.get("folder_id", "root")) != folder_id]
        folder_path = Path(str(folder.get("path", "")))
        if folder_path.exists() and self._is_within_runs_root(folder_path):
            shutil.rmtree(folder_path, ignore_errors=True)
        self._result_folders.pop(folder_id, None)

    def _delete_result_folder_output(self, folder_id: str) -> None:
        folder = self._result_folders.get(folder_id)
        if not folder or folder_id == "root":
            return
        folder_path = Path(str(folder.get("path", "")))
        if folder_path.exists() and self._is_within_runs_root(folder_path):
            run_dir = self._run_dir_for_output_path(folder_path)
            if run_dir is not None and run_dir.exists() and self._is_within_runs_root(run_dir):
                shutil.rmtree(run_dir, ignore_errors=True)
                self._temporary_run_dirs.discard(run_dir)
            else:
                shutil.rmtree(folder_path, ignore_errors=True)

    def _mark_result_folder_saved(self, folder_id: str) -> None:
        folder = self._result_folders.get(str(folder_id or ""))
        if folder:
            folder["saved"] = True
            folder_path_text = str(folder.get("path", "")).strip()
            if folder_path_text:
                folder_path = Path(folder_path_text)
                run_dir = self._run_dir_for_output_path(folder_path)
                if run_dir is not None:
                    self._temporary_run_dirs.discard(run_dir)

    def _cleanup_unsaved_results(self) -> None:
        unsaved_folders = [
            folder_id
            for folder_id, folder in list(self._result_folders.items())
            if folder_id != "root" and str(folder.get("path", "")).strip() and not bool(folder.get("saved"))
        ]
        for folder_id in unsaved_folders:
            self._delete_result_folder_output(folder_id)
            self._delete_result_folder_recursive(folder_id)
        for run_dir in list(self._temporary_run_dirs):
            if run_dir.exists() and self._is_within_runs_root(run_dir):
                shutil.rmtree(run_dir, ignore_errors=True)
            self._temporary_run_dirs.discard(run_dir)
        self._result_items = [item for item in self._result_items if bool(item.get("saved"))]
        if self._result_current_folder_id not in self._result_folders:
            self._result_current_folder_id = "root"
        self._refresh_result_table()

    def _find_run_result_path(self, output_dir: Path) -> Path | None:
        candidates = [output_dir / "result.json"]
        if self.current_run_dir is not None:
            candidates.append(self.current_run_dir / "output" / "result.json")
            try:
                candidates.extend(sorted((self.current_run_dir / "output").rglob("result.json")))
            except OSError:
                pass
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _is_path_recorded(self, path: Path, output_dir: Path, records: list[dict], figures: list) -> bool:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        for record in records:
            matrix_path = str(record.get("matrix_path", "") or "")
            if not matrix_path:
                continue
            candidate = Path(matrix_path)
            if not candidate.is_absolute():
                candidate = output_dir / candidate
            try:
                if candidate.resolve() == resolved:
                    return True
            except OSError:
                if candidate == resolved:
                    return True
        for figure in figures:
            if isinstance(figure, dict):
                figure_path = str(figure.get("path", "") or "")
            else:
                figure_path = str(figure or "")
            if not figure_path:
                continue
            candidate = Path(figure_path)
            if not candidate.is_absolute():
                candidate = output_dir / candidate
            try:
                if candidate.resolve() == resolved:
                    return True
            except OSError:
                if candidate == resolved:
                    return True
        return False

    def _discover_output_files(self, output_dir: Path, records: list[dict], figures: list) -> tuple[list[dict], list[dict], list[dict]]:
        discovered_figures: list[dict] = []
        discovered_records: list[dict] = []
        discovered_files: list[dict] = []
        image_suffixes = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
        matrix_suffixes = {".npy"}
        try:
            files = sorted(path for path in output_dir.rglob("*") if path.is_file())
        except OSError:
            return discovered_records, discovered_figures, discovered_files
        for path in files:
            if path.name in {"result.json", "error.txt"}:
                continue
            if self._is_path_recorded(path, output_dir, records, figures):
                continue
            suffix = path.suffix.lower()
            try:
                relative = path.relative_to(output_dir).as_posix()
            except ValueError:
                relative = str(path)
            if suffix in image_suffixes:
                discovered_figures.append({"name": path.stem, "path": relative})
            elif suffix in matrix_suffixes:
                discovered_records.append(
                    {
                        "name": path.stem,
                        "dataset_type": "agent_output_file",
                        "matrix_path": str(path),
                        "sample_labels": [],
                        "feature_labels": [],
                        "description": f"Discovered output matrix file: {relative}",
                    }
                )
            else:
                discovered_files.append({"name": path.name, "path": relative, "description": f"Output file: {relative}"})
        return discovered_records, discovered_figures, discovered_files

    def _load_run_result(self, exit_code: int, module_dir: Path | None) -> None:
        if self.current_run_dir is None:
            return
        output_dir = self.current_output_dir or (self.current_run_dir / "output")
        result_path = self._find_run_result_path(output_dir)
        if result_path is not None:
            output_dir = result_path.parent
            self.current_output_dir = output_dir
        if exit_code != 0 or result_path is None:
            if self.current_result_folder_id in self._result_folders:
                folder = self._result_folders[self.current_result_folder_id]
                folder["status"] = "failed"
                folder["path"] = str(output_dir)
                folder["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self._refresh_result_table()
                self._select_result_folder_id(self.current_result_folder_id)
            error_path = output_dir / "error.txt"
            if error_path.exists():
                error_text = error_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                emitted = False
                for line in _agent_log_lines(error_text, stream="stderr"):
                    self.log.append(line)
                    emitted = True
                if not emitted:
                    self.log.append(f"Module run failed. Details saved in {error_path.name}.")
            self._sync_result_buttons()
            return
        try:
            self.last_result = json.loads(result_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            self.log.append(f"Could not read result.json: {exc}")
            self._sync_result_buttons()
            return
        records = list(self.last_result.get("processed_records", []) or [])
        figures = list(self.last_result.get("figures", []) or [])
        discovered_records, discovered_figures, discovered_files = self._discover_output_files(output_dir, records, figures)
        if discovered_records or discovered_figures or discovered_files:
            records.extend(discovered_records)
            figures.extend(discovered_figures)
            self.log.append(
                f"Discovered {len(discovered_records)} matrix file(s), {len(discovered_figures)} figure file(s), and {len(discovered_files)} other file(s) in output folder."
            )
        module_manifest = self._manifest_for(module_dir) if module_dir is not None else {}
        run_id = self.current_run_dir.name
        run_folder_id = self.current_result_folder_id if self.current_result_folder_id in self._result_folders else ""
        parent_id = "root"
        if run_folder_id:
            folder = self._result_folders[run_folder_id]
            parent_id = str(folder.get("parent_id", "") or "root")
            folder.update(
                {
                    "path": str(output_dir),
                    "run_id": run_id,
                    "module_manifest": dict(module_manifest or {}),
                    "status": "complete",
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            self._result_items = [item for item in self._result_items if str(item.get("folder_id", "root")) != run_folder_id]
            folder_name = str(folder.get("name", "") or self.current_result_folder_name or run_id)
        else:
            parent_id = self._result_current_folder_id if self._result_current_folder_id in self._result_folders else "root"
            default_folder_name = self._module_result_folder_name(module_dir) if module_dir is not None else run_id
            requested_folder_name = self.current_result_folder_name or default_folder_name
            folder_name = self._unique_result_child_name(parent_id, requested_folder_name)
            run_folder_id = self._create_result_folder(
                folder_name,
                parent_id=parent_id,
                path=output_dir,
                run_id=run_id,
                module_manifest=module_manifest,
                module_id=module_dir.name if module_dir is not None else "",
                input_signature=self.current_input_signature,
                input_label=self.current_input_label,
                status="complete",
            )
        new_rows = []
        for record in records:
            new_rows.append(
                self._new_result_item(
                    "processed",
                    record=record,
                    run_dir=self.current_run_dir,
                    run_id=run_id,
                    folder_id=run_folder_id,
                    module_dir=module_dir,
                    module_manifest=module_manifest,
                )
            )
        for figure in figures:
            new_rows.append(
                self._new_result_item(
                    "figure",
                    figure=figure,
                    run_dir=self.current_run_dir,
                    run_id=run_id,
                    folder_id=run_folder_id,
                    module_dir=module_dir,
                    module_manifest=module_manifest,
                )
            )
        for file_info in discovered_files:
            new_rows.append(
                self._new_result_item(
                    "file",
                    figure=file_info,
                    run_dir=self.current_run_dir,
                    run_id=run_id,
                    folder_id=run_folder_id,
                    module_dir=module_dir,
                    module_manifest=module_manifest,
                )
            )
        self._result_items.extend(new_rows)
        self._result_current_folder_id = parent_id
        self._refresh_result_table()
        self._select_result_folder_id(run_folder_id)
        self.log.append(str(self.last_result.get("summary", "") or "Run completed."))
        self._sync_result_buttons()
        if new_rows:
            self._result_selection_changed()
        else:
            self._draw_empty_preview(self.preview_canvas, f"Folder: {folder_name}\nNo processed records or figures were returned.")

    def _new_result_item(
        self,
        kind: str,
        *,
        record: dict | None = None,
        figure=None,
        run_dir: Path,
        run_id: str,
        folder_id: str,
        module_dir: Path | None,
        module_manifest: dict,
    ) -> dict:
        item = {
            "id": f"result_{self._result_next_id}",
            "kind": kind,
            "run_dir": run_dir,
            "output_dir": self.current_output_dir or (Path(run_dir) / "output"),
            "run_id": run_id,
            "folder_id": folder_id,
            "module_dir": module_dir,
            "module_manifest": dict(module_manifest or {}),
            "saved": False,
        }
        self._result_next_id += 1
        if kind == "processed":
            item["record"] = dict(record or {})
        elif kind == "figure":
            item["figure"] = figure
        else:
            item["file"] = dict(figure or {})
        return item

    def _refresh_result_table(self) -> None:
        self._visible_result_entries = self._visible_result_entries_for_current_folder()
        self.result_path_label.setText(self._result_folder_path_text(self._result_current_folder_id))
        self.up_folder_button.setEnabled(self._result_current_folder_id != "root")
        self.output_table.setRowCount(len(self._visible_result_entries))
        for row, entry in enumerate(self._visible_result_entries):
            values = self._result_entry_table_values(entry, row)
            for column, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                table_item.setToolTip(value)
                self.output_table.setItem(row, column, table_item)
        self.output_table.resizeColumnsToContents()
        self._sync_result_buttons()

    def _result_entry_table_values(self, entry: dict, row: int) -> list[str]:
        if entry.get("entry_type") == "folder":
            folder = entry.get("folder", {})
            path = str(folder.get("path", "") or "")
            status = str(folder.get("status", "") or "")
            input_label = str(folder.get("input_label", "") or "")
            description_parts = []
            if status:
                description_parts.append(status)
            if input_label:
                description_parts.append(input_label)
            if path:
                description_parts.append(f"Run output: {path}")
            description = " | ".join(description_parts) if description_parts else "Folder"
            child_count = self._result_folder_child_count(str(folder.get("id", "")))
            return [f"[folder] {folder.get('name', f'folder {row + 1}')}", "folder", status, str(child_count), description]
        return self._result_item_table_values(entry.get("item", {}), row)

    def _result_folder_child_count(self, folder_id: str) -> int:
        folder_count = sum(
            1
            for child_id, folder in self._result_folders.items()
            if child_id != "root" and str(folder.get("parent_id", "")) == folder_id
        )
        item_count = sum(1 for item in self._result_items if str(item.get("folder_id", "root")) == folder_id)
        return folder_count + item_count

    def _result_item_display_name(self, item: dict) -> str:
        if item.get("kind") == "figure":
            figure_info = item.get("figure")
            if isinstance(figure_info, dict):
                figure_path = str(figure_info.get("path", ""))
                return str(figure_info.get("name") or Path(figure_path).name or "figure")
            else:
                return Path(str(figure_info)).name or "figure"
        if item.get("kind") == "file":
            file_info = item.get("file", {})
            file_path = str(file_info.get("path", ""))
            return str(file_info.get("name") or Path(file_path).name or "file")
        record = item.get("record", {})
        return str(record.get("name", "") or item.get("id", "result"))

    def _result_item_table_values(self, item: dict, row: int) -> list[str]:
        if item.get("kind") == "figure":
            figure_info = item.get("figure")
            if isinstance(figure_info, dict):
                figure_path = str(figure_info.get("path", ""))
                figure_name = str(figure_info.get("name") or Path(figure_path).name or f"figure {row + 1}")
                description = str(figure_info.get("description", ""))
            else:
                figure_path = str(figure_info)
                figure_name = Path(figure_path).name or f"figure {row + 1}"
                description = figure_path
            return [figure_name, "figure", "", "", description]
        if item.get("kind") == "file":
            file_info = item.get("file", {})
            file_path = str(file_info.get("path", ""))
            file_name = str(file_info.get("name") or Path(file_path).name or f"file {row + 1}")
            description = str(file_info.get("description", "") or file_path)
            return [file_name, "file", "", "", description]
        record = item.get("record", {})
        shape = list(record.get("shape", []) or [])
        samples = str(shape[0]) if len(shape) >= 1 else ""
        features = str(shape[1]) if len(shape) >= 2 else ""
        status = "saved" if item.get("saved") else str(record.get("description", ""))
        return [
            str(record.get("name", "")),
            str(record.get("dataset_type", "")),
            samples,
            features,
            status,
        ]

    def _result_selection_changed(self) -> None:
        if not self._visible_result_entries:
            self._draw_empty_preview(self.preview_canvas, "Result preview")
            return
        rows = sorted({index.row() for index in self.output_table.selectedIndexes()})
        if not rows and self.output_table.rowCount():
            rows = [0]
        if rows:
            self._draw_result_entry_preview(self.preview_canvas, rows[0])
        else:
            self._draw_empty_preview(self.preview_canvas, "Result preview")
        self._sync_result_buttons()

    def _zoom_preview_event(self, event) -> None:
        ax = getattr(event, "inaxes", None)
        if ax is None:
            return
        xdata = getattr(event, "xdata", None)
        ydata = getattr(event, "ydata", None)
        x_left, x_right = ax.get_xlim()
        y_bottom, y_top = ax.get_ylim()
        if xdata is None:
            xdata = (x_left + x_right) / 2.0
        if ydata is None:
            ydata = (y_bottom + y_top) / 2.0
        button = str(getattr(event, "button", "")).lower()
        if button not in {"up", "down"}:
            return
        scale = 0.88 if button == "up" else 1.14
        x_span = float(x_right) - float(x_left)
        y_span = float(y_top) - float(y_bottom)
        if abs(x_span) <= 1e-12 or abs(y_span) <= 1e-12:
            return
        x_anchor = (float(xdata) - float(x_left)) / x_span
        y_anchor = (float(ydata) - float(y_bottom)) / y_span
        new_x_span = x_span * scale
        new_y_span = y_span * scale
        new_x_left = float(xdata) - x_anchor * new_x_span
        new_y_bottom = float(ydata) - y_anchor * new_y_span
        ax.set_xlim(new_x_left, new_x_left + new_x_span)
        ax.set_ylim(new_y_bottom, new_y_bottom + new_y_span)
        self.preview_canvas.draw_idle()

    def _preview_press_event(self, event) -> None:
        ax = getattr(event, "inaxes", None)
        if ax is None or getattr(event, "button", None) != 1:
            self._preview_drag_state = None
            return
        self._preview_drag_state = {
            "ax": ax,
            "x": float(getattr(event, "x", 0.0)),
            "y": float(getattr(event, "y", 0.0)),
            "xlim": tuple(ax.get_xlim()),
            "ylim": tuple(ax.get_ylim()),
            "width": max(1.0, float(ax.bbox.width)),
            "height": max(1.0, float(ax.bbox.height)),
        }

    def _preview_motion_event(self, event) -> None:
        state = self._preview_drag_state
        if not state:
            return
        ax = state.get("ax")
        if ax is None:
            return
        dx_pixels = float(getattr(event, "x", state["x"])) - float(state["x"])
        dy_pixels = float(getattr(event, "y", state["y"])) - float(state["y"])
        x_left, x_right = state["xlim"]
        y_bottom, y_top = state["ylim"]
        x_span = float(x_right) - float(x_left)
        y_span = float(y_top) - float(y_bottom)
        dx_data = -dx_pixels / float(state["width"]) * x_span
        dy_data = -dy_pixels / float(state["height"]) * y_span
        ax.set_xlim(float(x_left) + dx_data, float(x_right) + dx_data)
        ax.set_ylim(float(y_bottom) + dy_data, float(y_top) + dy_data)
        self.preview_canvas.draw_idle()

    def _preview_release_event(self, _event) -> None:
        self._preview_drag_state = None

    def _draw_empty_preview(self, canvas: FigureCanvas, label: str) -> None:
        self._preview_drag_state = None
        figure = canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        ax.text(0.5, 0.5, label, ha="center", va="center", color="#777777", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        canvas.draw_idle()

    def _draw_result_entry_preview(self, canvas: FigureCanvas, visible_row: int) -> None:
        self._preview_drag_state = None
        if not (0 <= visible_row < len(self._visible_result_entries)):
            self._draw_empty_preview(canvas, "No result")
            return
        entry = self._visible_result_entries[visible_row]
        if entry.get("entry_type") == "folder":
            folder = entry.get("folder", {})
            self._draw_empty_preview(canvas, f"Folder: {folder.get('name', 'folder')}\nDouble-click to open")
            return
        item = entry.get("item", {})
        if item.get("kind") == "figure":
            self._draw_figure_file_preview(canvas, item.get("figure"), item.get("output_dir") or item.get("run_dir"))
            return
        if item.get("kind") == "file":
            self._draw_file_preview(canvas, item.get("file"), item.get("output_dir") or item.get("run_dir"))
            return
        record = item.get("record", {})
        matrix_path = Path(str(record.get("matrix_path", "")))
        figure = canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        if not matrix_path.exists():
            ax.text(0.5, 0.5, "Missing matrix", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            canvas.draw_idle()
            return
        try:
            matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=float)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Preview failed\n{exc}", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            canvas.draw_idle()
            return
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        title = str(record.get("name", f"result {visible_row + 1}") or f"result {visible_row + 1}")
        sample_labels = [str(item) for item in list(record.get("sample_labels", []) or [])]
        feature_labels = [str(item) for item in list(record.get("feature_labels", []) or [])]
        ax.set_title(title[:70], fontsize=9)
        if matrix.size == 0:
            ax.text(0.5, 0.5, "Empty matrix", ha="center", va="center", transform=ax.transAxes)
        elif matrix.shape[1] == 1:
            values = matrix.reshape(-1)
            ax.plot(np.arange(1, values.size + 1), values, linewidth=1.4)
            self._apply_axis_labels(
                ax,
                x_labels=sample_labels,
                x_default="Sample",
                y_label=feature_labels[0] if feature_labels else self._record_value_label(record),
                value_count=values.size,
            )
        elif matrix.shape[0] == 1:
            values = matrix.reshape(-1)
            ax.plot(np.arange(1, values.size + 1), values, linewidth=1.4)
            self._apply_axis_labels(
                ax,
                x_labels=feature_labels,
                x_default=str(record.get("x_axis_label", "") or "Feature"),
                y_label=self._record_value_label(record),
                value_count=values.size,
            )
        elif 1 < matrix.shape[1] <= 6:
            x = np.arange(matrix.shape[0])
            for column in range(matrix.shape[1]):
                label = feature_labels[column] if column < len(feature_labels) else f"series {column + 1}"
                ax.plot(x, matrix[:, column], linewidth=1.2, label=label)
            ax.legend(loc="best", fontsize=7)
            self._apply_axis_labels(
                ax,
                x_labels=sample_labels,
                x_default=str(record.get("x_axis_label", "") or "Sample"),
                y_label=self._record_value_label(record),
                value_count=matrix.shape[0],
                zero_based=True,
            )
        elif matrix.shape[0] <= 6:
            x = np.arange(matrix.shape[1])
            for row in range(matrix.shape[0]):
                label = sample_labels[row] if row < len(sample_labels) else str(row + 1)
                ax.plot(x, matrix[row], linewidth=1.1, label=label)
            ax.legend(loc="best", fontsize=7)
            self._apply_axis_labels(
                ax,
                x_labels=feature_labels,
                x_default=str(record.get("x_axis_label", "") or "Feature"),
                y_label=self._record_value_label(record),
                value_count=matrix.shape[1],
                zero_based=True,
            )
        else:
            image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
            figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel("Feature" if not feature_labels else "Feature label")
            ax.set_ylabel("Sample" if not sample_labels else "Sample label")
            if feature_labels and len(feature_labels) <= 12:
                ax.set_xticks(np.arange(len(feature_labels)))
                ax.set_xticklabels(feature_labels, rotation=45, ha="right", fontsize=7)
            if sample_labels and len(sample_labels) <= 18:
                ax.set_yticks(np.arange(len(sample_labels)))
                ax.set_yticklabels(sample_labels, fontsize=7)
        canvas.draw_idle()

    def _record_value_label(self, record: dict) -> str:
        value_label = str(record.get("value_label", "") or "").strip()
        if value_label:
            return value_label
        dataset_type = str(record.get("dataset_type", "") or "").strip()
        if dataset_type:
            return dataset_type
        return "Value"

    def _apply_axis_labels(
        self,
        ax,
        *,
        x_labels: list[str],
        x_default: str,
        y_label: str,
        value_count: int,
        zero_based: bool = False,
    ) -> None:
        ax.set_xlabel(x_default if not x_labels else self._label_axis_name(x_labels, x_default))
        ax.set_ylabel(y_label or "Value")
        if not x_labels or len(x_labels) != int(value_count):
            return
        if len(x_labels) > 24:
            return
        ticks = np.arange(value_count) if zero_based else np.arange(1, value_count + 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)

    def _label_axis_name(self, labels: list[str], default: str) -> str:
        lowered = " ".join(labels[: min(8, len(labels))]).lower()
        if any(token in lowered for token in ("ch", "channel", "electrode")):
            return "Channel"
        if any(token in lowered for token in ("time", "sec", "ms")):
            return "Time"
        return default

    def _draw_figure_file_preview(self, canvas: FigureCanvas, figure_info, output_dir: Path | None = None) -> None:
        if isinstance(figure_info, dict):
            path_text = str(figure_info.get("path", ""))
            title = str(figure_info.get("name") or Path(path_text).name or "Figure")
        else:
            path_text = str(figure_info)
            title = Path(path_text).name or "Figure"
        figure_path = Path(path_text)
        if not figure_path.is_absolute() and output_dir is not None:
            figure_path = Path(output_dir) / figure_path
        figure = canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        if not figure_path.exists():
            ax.text(0.5, 0.5, "Missing figure", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            canvas.draw_idle()
            return
        if figure_path.suffix.lower() in {".pdf", ".svg"}:
            size_kb = figure_path.stat().st_size / 1024.0
            ax.text(
                0.5,
                0.5,
                f"{title}\n{figure_path.name}\n{size_kb:.1f} KB\nPreview opens for bitmap images only.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            canvas.draw_idle()
            return
        try:
            image = mpimg.imread(figure_path)
            ax.imshow(image)
            ax.set_title(title[:70], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        except Exception as exc:
            ax.text(0.5, 0.5, f"Figure preview failed\n{exc}", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        canvas.draw_idle()

    def _draw_file_preview(self, canvas: FigureCanvas, file_info, output_dir: Path | None = None) -> None:
        if isinstance(file_info, dict):
            path_text = str(file_info.get("path", ""))
            title = str(file_info.get("name") or Path(path_text).name or "File")
            description = str(file_info.get("description", "") or "")
        else:
            path_text = str(file_info)
            title = Path(path_text).name or "File"
            description = path_text
        file_path = Path(path_text)
        if not file_path.is_absolute() and output_dir is not None:
            file_path = Path(output_dir) / file_path
        figure = canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        if file_path.exists():
            size_text = f"{file_path.stat().st_size / 1024.0:.1f} KB"
            path_label = str(file_path)
        else:
            size_text = "missing"
            path_label = path_text
        ax.text(
            0.5,
            0.5,
            f"{title}\n{size_text}\n{description}\n{path_label}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            wrap=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        canvas.draw_idle()

    def _save_last_results(self) -> None:
        if not self._result_items:
            return
        selected_items = self._selected_result_item_indexes()
        if not selected_items:
            selected_items = self._selected_folder_result_item_indexes()
        if selected_items:
            rows = [index for index, _item in selected_items]
        else:
            rows = [index for index, item in enumerate(self._result_items) if item.get("kind") == "processed" and not item.get("saved")]
        records = []
        saved_row_indexes = []
        for index in rows:
            if not (0 <= index < len(self._result_items)):
                continue
            item = self._result_items[index]
            if item.get("kind") != "processed":
                continue
            if item.get("saved"):
                continue
            record = dict(item.get("record", {}) or {})
            matrix_path = Path(str(record.get("matrix_path", "")))
            if not matrix_path.exists():
                self.log.append(f"Missing matrix file: {matrix_path}")
                continue
            matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=float)
            if matrix.ndim == 1:
                matrix = matrix.reshape(-1, 1)
            module_dir = item.get("module_dir")
            module_manifest = dict(item.get("module_manifest", {}) or {})
            run_id = str(item.get("run_id", ""))
            name = str(record.get("name") or f"agent result {index + 1}")
            saved = {
                "name": name,
                "path": f"agent::{module_manifest.get('id', 'module')}::{run_id}::{index}",
                "source_path": str(module_dir or ""),
                "origin_name": str(module_manifest.get("name", "Agent module")),
                "source_label": str(module_manifest.get("name", "Agent module")),
                "dataset_type": str(record.get("dataset_type", "agent_custom")),
                "dataset_group": "agent_generated",
                "dataset_origin": "agent_custom_code",
                "matrix": matrix,
                "sample_labels": [str(item) for item in list(record.get("sample_labels", []))],
                "feature_labels": [str(item) for item in list(record.get("feature_labels", []))],
                "description": str(record.get("description", "")),
                "commit": "agent custom code",
                "commit_detail": str(module_manifest.get("task", "")),
                "parameters": {
                    "analysis_kind": "agent_custom_code",
                    "module_id": str(module_manifest.get("id", "")),
                    "run_id": run_id,
                },
            }
            records.append(saved)
            saved_row_indexes.append(index)
        saved_count = self.save_processed_callback(records)
        self.log.append(f"Saved {saved_count} processed record(s).")
        if saved_count == len(records):
            for index in saved_row_indexes[:saved_count]:
                if 0 <= index < len(self._result_items):
                    self._result_items[index]["saved"] = True
                    self._mark_result_folder_saved(str(self._result_items[index].get("folder_id", "")))
            self._refresh_result_table()
            self._sync_result_buttons()
        elif saved_count:
            self.log.append(
                "Only part of the selected results were saved. Unsaved rows remain marked as unsaved to avoid hiding duplicate-name failures."
            )
            self._sync_result_buttons()

    def _selected_result_rows(self) -> list[int]:
        rows = sorted({index.row() for index in self.output_table.selectedIndexes()})
        indexes = []
        for row in rows:
            if not (0 <= row < len(self._visible_result_entries)):
                continue
            entry = self._visible_result_entries[row]
            if entry.get("entry_type") != "item":
                continue
            index = self._result_item_index(str(entry.get("item_id", "")))
            if 0 <= index < len(self._result_items):
                indexes.append(index)
        return indexes

    def _sync_result_buttons(self) -> None:
        has_unsaved_processed = any(
            item.get("kind") == "processed" and not item.get("saved") for item in self._result_items
        )
        self.save_results_button.setEnabled(self.current_process is None and has_unsaved_processed)
        has_visible_selection = bool(self._selected_result_entries()) if hasattr(self, "output_table") else False
        self.delete_results_button.setEnabled(self.current_process is None and bool(self._visible_result_entries))
        self.rename_result_button.setEnabled(self.current_process is None and has_visible_selection)
        self.move_results_button.setEnabled(self.current_process is None and bool(self._selected_result_item_indexes()))
        self.up_folder_button.setEnabled(self.current_process is None and self._result_current_folder_id != "root")
        self.new_folder_button.setEnabled(self.current_process is None)

    def _delete_selected_results(self) -> None:
        entries = self._selected_result_entries()
        if not entries:
            QMessageBox.information(self, "Agent Custom Code", "Select result rows or folders to delete.")
            return
        folder_count = sum(1 for entry in entries if entry.get("entry_type") == "folder")
        item_count = sum(1 for entry in entries if entry.get("entry_type") == "item")
        answer = QMessageBox.question(
            self,
            "Delete result history",
            f"Delete {item_count} result file(s) and {folder_count} folder(s)?\nFolders are removed recursively.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        item_ids = {str(entry.get("item_id", "")) for entry in entries if entry.get("entry_type") == "item"}
        if item_ids:
            self._result_items = [item for item in self._result_items if str(item.get("id", "")) not in item_ids]
        for entry in entries:
            if entry.get("entry_type") == "folder":
                self._delete_result_folder_recursive(str(entry.get("folder_id", "")))
        self._refresh_result_table()
        self._sync_result_buttons()
        self._result_selection_changed()

    def _rename_selected_module(self) -> None:
        module_dir = self._selected_module_dir()
        if module_dir is None:
            return
        manifest = self._manifest_for(module_dir)
        current = str(manifest.get("name") or module_dir.name)
        name, accepted = QInputDialog.getText(self, "Rename module", "Module name:", text=current)
        if not accepted or not name.strip():
            return
        manifest["name"] = name.strip()
        manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (module_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        self._refresh_modules()
        self._select_module_id(module_dir.name)

    def _delete_selected_module(self) -> None:
        module_dir = self._selected_module_dir()
        if module_dir is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete Agent Module",
            f"Delete generated module?\n{module_dir}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        shutil.rmtree(module_dir, ignore_errors=True)
        self._refresh_modules()
