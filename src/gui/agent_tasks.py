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
        QDialog,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMessageBox,
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


def _agent_log_lines(text: str, *, stream: str = "") -> list[str]:
    lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("stderr:", "stdout:")):
            line = line.split(":", 1)[1].strip()
        lower = line.lower()
        if not line:
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
        patch_prefixes = (
            "diff --git ",
            "index ",
            "@@",
            "+++ ",
            "--- ",
            "*** begin patch",
            "*** update file:",
            "*** add file:",
            "*** delete file:",
            "*** end patch",
            "```",
        )
        if lower.startswith(patch_prefixes):
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
        if lower.startswith(("task - ", "you are generating ", "you already have ", "# mandatory agent instructions")):
            continue
        if stream == "stderr" and lower.startswith(("traceback", "error:", "failed", "runtimeerror")):
            lines.append(f"Error: {line[:320]}")
            continue
        if len(line) > 320:
            line = line[:317] + "..."
        lines.append(line)
    return lines


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
        text = bytes(self.current_process.readAllStandardOutput()).decode(errors="replace")
        for line in _agent_log_lines(text, stream="stdout"):
            self.log.append(line)

    def _read_probe_stderr(self) -> None:
        if self.current_process is None:
            return
        text = bytes(self.current_process.readAllStandardError()).decode(errors="replace")
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
        self.current_module_dir: Path | None = None
        self._process_snapshot_root: Path | None = None
        self._process_file_snapshot: dict[str, tuple[int, int]] = {}
        self._preview_drag_state: dict | None = None
        self.last_result: dict | None = None
        self._result_items: list[dict] = []
        self._result_next_id = 1
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
        self.rename_button = QPushButton("Rename")
        self.rename_button.clicked.connect(self._rename_selected_module)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self._delete_selected_module)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 8))
        self.log.setMinimumHeight(150)
        self.log.document().setMaximumBlockCount(2000)
        self.output_table = QTableWidget(0, 5)
        self.output_table.setHorizontalHeaderLabels(["Name", "Type", "Samples", "Features", "Description"])
        self.output_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.output_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.output_table.setMinimumHeight(150)
        self.output_table.itemSelectionChanged.connect(self._result_selection_changed)
        self.save_results_button = QPushButton("Save Results")
        self.save_results_button.clicked.connect(self._save_last_results)
        self.save_results_button.setEnabled(False)
        self.delete_results_button = QPushButton("Delete Result")
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

        result_panel = QFrame()
        result_panel.setObjectName("Panel")
        result_panel.setMaximumHeight(260)
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(8, 8, 8, 8)
        result_layout.setSpacing(4)
        result_layout.addWidget(QLabel("Result library"))
        result_layout.addWidget(self.output_table, 1)
        result_buttons = QHBoxLayout()
        result_buttons.setContentsMargins(0, 0, 0, 0)
        result_buttons.setSpacing(6)
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
            summary = str(manifest.get("summary", "") or "")[:120]
            values = [str(manifest.get("name") or module_dir.name), module_dir.name, str(updated), summary]
            for column, value in enumerate(values):
                self.module_table.setItem(row, column, QTableWidgetItem(value))
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
        self._set_busy(True)
        self.log.append(f"Agent task written: {task_path}")
        self.log.append(f"Agent instructions written: {agents_path}")
        self.log.append(f"$ {_command_for_log(program, arguments)}")
        process.start()

    def _update_module_task_metadata(self, module_dir: Path, task: str) -> None:
        manifest = self._manifest_for(module_dir)
        if not manifest:
            return
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        if task.strip():
            manifest["task"] = task.strip()
            manifest["summary"] = task.strip()[:200]
        manifest["updated_at"] = now
        (module_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        readme = f"# {manifest.get('name') or module_dir.name}\n\nTask:\n\n{manifest.get('task') or 'Agent custom analysis module.'}\n\nSee AGENT_TASK.md for the full generation contract.\n"
        (module_dir / "README.md").write_text(readme, encoding="utf-8")

    def _agent_prompt(self, module_dir: Path, task: str, *, modify_existing: bool = False) -> str:
        selected_raw = self._selected_raw_records_for_scope()
        selected_processed = self._selected_processed_records_for_scope()
        input_summary = self._agent_input_summary(selected_raw, selected_processed)
        mode_text = (
            "Modify the existing module.py in place according to the user task."
            if modify_existing
            else "Generate module.py according to the user task. The initial file is only an interface shell; replace the placeholder analyze(context) implementation."
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
            "The initial module.py shell is not an analysis template. Treat the user task as the source of truth and write the analysis code yourself.\n\n"
            "Input data paths are provided in context['selected_raw_records'] and context['selected_processed_records'].\n"
            "Raw spike input package (.npz) keys: channels, spike_times, stim_times, sampling_rate, metadata_json. spike_times is an object array aligned to channels.\n"
            "Processed input package (.npz) keys: matrix, sample_labels, feature_labels, metadata_json.\n"
            "When multiple raw or processed records are selected, iterate over every selected record. Do not use only the first record unless the user explicitly asks for that.\n"
            "For cross-file channel comparisons, align rows by channel/electrode label, use sample_labels for channels/electrodes, and use feature_labels for file/condition names.\n"
            "If the task asks to compare files, preserve per-file values as separate columns or separate processed_records; do not collapse them into one global summary.\n"
            "If the task asks for one plot/result per channel/electrode, return one processed_record per channel/electrode. Each such record should have a 1 x N matrix, feature_labels as file/condition names, sample_labels containing the channel/electrode label, and value_label for the measured value.\n"
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
        ]:
            button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)

    def _stop_process(self) -> None:
        if self.current_process is None:
            return
        self.log.append("Stopping process...")
        self.current_process.terminate()
        QTimer.singleShot(2000, self._kill_process_if_needed)

    def _kill_process_if_needed(self) -> None:
        if self.current_process is not None and self.current_process.state() != QProcess.ProcessState.NotRunning:
            self.current_process.kill()

    def _read_process_stdout(self) -> None:
        if self.current_process is None:
            return
        text = bytes(self.current_process.readAllStandardOutput()).decode(errors="replace")
        for line in _agent_log_lines(text, stream="stdout"):
            self.log.append(line)

    def _read_process_stderr(self) -> None:
        if self.current_process is None:
            return
        text = bytes(self.current_process.readAllStandardError()).decode(errors="replace")
        for line in _agent_log_lines(text, stream="stderr"):
            self.log.append(line)

    def _process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        mode = self.current_mode
        module_dir = self.current_module_dir
        snapshot_root = self._process_snapshot_root
        snapshot = dict(self._process_file_snapshot)
        self.log.append(f"Process finished: code={exit_code}, status={exit_status.name}")
        self.current_process = None
        self.current_mode = ""
        self.current_module_dir = None
        self._process_snapshot_root = None
        self._process_file_snapshot = {}
        self._set_busy(False)
        self._sync_result_buttons()
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
            self._load_run_result(exit_code, module_dir)

    def _process_error(self, error: QProcess.ProcessError) -> None:
        self.log.append(f"Process error: {error.name}")
        self.current_process = None
        self.current_mode = ""
        self.current_module_dir = None
        self._process_snapshot_root = None
        self._process_file_snapshot = {}
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
            code = code_path.read_text(encoding="utf-8")
        except Exception as exc:
            self.log.append(f"Could not inspect generated module.py: {exc}")
            return
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
            self.log.append("Agent generated module.py successfully.")

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
        output_dir = run_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self._export_input_manifest(input_dir)
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
        self._process_snapshot_root = run_dir
        self._process_file_snapshot = _file_snapshot(run_dir)
        self.last_result = None
        self._sync_result_buttons()
        self._set_busy(True)
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

    def _load_run_result(self, exit_code: int, module_dir: Path | None) -> None:
        if self.current_run_dir is None:
            return
        result_path = self.current_run_dir / "output" / "result.json"
        if exit_code != 0 or not result_path.exists():
            error_path = self.current_run_dir / "output" / "error.txt"
            if error_path.exists():
                self.log.append(error_path.read_text(encoding="utf-8")[-4000:])
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
        module_manifest = self._manifest_for(module_dir) if module_dir is not None else {}
        run_id = self.current_run_dir.name
        new_rows = []
        for record in records:
            new_rows.append(
                self._new_result_item(
                    "processed",
                    record=record,
                    run_dir=self.current_run_dir,
                    run_id=run_id,
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
                    module_dir=module_dir,
                    module_manifest=module_manifest,
                )
            )
        start_row = len(self._result_items)
        self._result_items.extend(new_rows)
        self._refresh_result_table()
        self.log.append(str(self.last_result.get("summary", "") or "Run completed."))
        self._sync_result_buttons()
        if new_rows:
            self.output_table.clearSelection()
            self.output_table.selectRow(start_row)
            if len(new_rows) > 1:
                self.output_table.selectRow(start_row + 1)
            self._result_selection_changed()

    def _new_result_item(
        self,
        kind: str,
        *,
        record: dict | None = None,
        figure=None,
        run_dir: Path,
        run_id: str,
        module_dir: Path | None,
        module_manifest: dict,
    ) -> dict:
        item = {
            "id": f"result_{self._result_next_id}",
            "kind": kind,
            "run_dir": run_dir,
            "run_id": run_id,
            "module_dir": module_dir,
            "module_manifest": dict(module_manifest or {}),
            "saved": False,
        }
        self._result_next_id += 1
        if kind == "processed":
            item["record"] = dict(record or {})
        else:
            item["figure"] = figure
        return item

    def _refresh_result_table(self) -> None:
        self.output_table.setRowCount(len(self._result_items))
        for row, item in enumerate(self._result_items):
            values = self._result_item_table_values(item, row)
            for column, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                table_item.setToolTip(value)
                self.output_table.setItem(row, column, table_item)
        self.output_table.resizeColumnsToContents()

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
        if not self._result_items:
            self._draw_empty_preview(self.preview_canvas, "Result preview")
            return
        rows = sorted({index.row() for index in self.output_table.selectedIndexes()})
        if not rows and self.output_table.rowCount():
            rows = [0]
        if rows:
            self._draw_record_preview(self.preview_canvas, rows[0])
        else:
            self._draw_empty_preview(self.preview_canvas, "Result preview")

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
        scale = 0.8 if button == "up" else 1.25
        new_x_left = xdata - (xdata - x_left) * scale
        new_x_right = xdata + (x_right - xdata) * scale
        new_y_bottom = ydata - (ydata - y_bottom) * scale
        new_y_top = ydata + (y_top - ydata) * scale
        if abs(new_x_right - new_x_left) > 1e-9:
            ax.set_xlim(new_x_left, new_x_right)
        if abs(new_y_top - new_y_bottom) > 1e-9:
            ax.set_ylim(new_y_bottom, new_y_top)
        self.preview_canvas.draw_idle()

    def _preview_press_event(self, event) -> None:
        ax = getattr(event, "inaxes", None)
        if ax is None or getattr(event, "button", None) != 1:
            self._preview_drag_state = None
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            self._preview_drag_state = None
            return
        self._preview_drag_state = {
            "ax": ax,
            "x": float(event.xdata),
            "y": float(event.ydata),
            "xlim": tuple(ax.get_xlim()),
            "ylim": tuple(ax.get_ylim()),
        }

    def _preview_motion_event(self, event) -> None:
        state = self._preview_drag_state
        if not state:
            return
        ax = state.get("ax")
        if ax is None:
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return
        dx = float(event.xdata) - float(state["x"])
        dy = float(event.ydata) - float(state["y"])
        x_left, x_right = state["xlim"]
        y_bottom, y_top = state["ylim"]
        ax.set_xlim(float(x_left) - dx, float(x_right) - dx)
        ax.set_ylim(float(y_bottom) - dy, float(y_top) - dy)
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

    def _draw_record_preview(self, canvas: FigureCanvas, record_index: int) -> None:
        self._preview_drag_state = None
        if not (0 <= record_index < len(self._result_items)):
            self._draw_empty_preview(canvas, "No result")
            return
        item = self._result_items[record_index]
        if item.get("kind") == "figure":
            self._draw_figure_file_preview(canvas, item.get("figure"), item.get("run_dir"))
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
        title = str(record.get("name", f"result {record_index + 1}") or f"result {record_index + 1}")
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

    def _draw_figure_file_preview(self, canvas: FigureCanvas, figure_info, run_dir: Path | None = None) -> None:
        if isinstance(figure_info, dict):
            path_text = str(figure_info.get("path", ""))
            title = str(figure_info.get("name") or Path(path_text).name or "Figure")
        else:
            path_text = str(figure_info)
            title = Path(path_text).name or "Figure"
        figure_path = Path(path_text)
        if not figure_path.is_absolute() and run_dir is not None:
            figure_path = Path(run_dir) / "output" / figure_path
        figure = canvas.figure
        figure.clear()
        ax = figure.add_subplot(111)
        if not figure_path.exists():
            ax.text(0.5, 0.5, "Missing figure", ha="center", va="center", transform=ax.transAxes)
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

    def _save_last_results(self) -> None:
        if not self._result_items:
            return
        rows = self._selected_result_rows()
        if not rows:
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
            self._refresh_result_table()
            self._sync_result_buttons()
        elif saved_count:
            self.log.append(
                "Only part of the selected results were saved. Unsaved rows remain marked as unsaved to avoid hiding duplicate-name failures."
            )
            self._sync_result_buttons()

    def _selected_result_rows(self) -> list[int]:
        rows = sorted({index.row() for index in self.output_table.selectedIndexes()})
        return [row for row in rows if 0 <= row < len(self._result_items)]

    def _sync_result_buttons(self) -> None:
        has_unsaved_processed = any(
            item.get("kind") == "processed" and not item.get("saved") for item in self._result_items
        )
        self.save_results_button.setEnabled(self.current_process is None and has_unsaved_processed)
        self.delete_results_button.setEnabled(self.current_process is None and bool(self._result_items))

    def _delete_selected_results(self) -> None:
        rows = self._selected_result_rows()
        if not rows:
            QMessageBox.information(self, "Agent Custom Code", "Select result rows to delete.")
            return
        for row in sorted(rows, reverse=True):
            if 0 <= row < len(self._result_items):
                self._result_items.pop(row)
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
