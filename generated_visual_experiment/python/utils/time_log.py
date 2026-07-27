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
