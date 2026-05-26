"""Channel-to-electrode map persistence and validation."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


MAP_ROWS = 8
MAP_COLS = 8


def _default_map_store() -> Path:
    override = os.environ.get("MEA_PIPELINE_MAP_STORE")
    if override:
        return Path(override)
    source_tree_store = Path(__file__).resolve().parents[2] / "config" / "channel_maps.json"
    if source_tree_store.exists():
        return source_tree_store
    return Path.home() / ".mea_pipeline" / "channel_maps.json"


def _packaged_map_store() -> Path | None:
    candidates = [
        Path(sys.prefix) / "share" / "mea_pipeline" / "config" / "channel_maps.json",
        Path(__file__).resolve().parents[1] / "share" / "mea_pipeline" / "config" / "channel_maps.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


DEFAULT_MAP_STORE = _default_map_store()


def electrode_id(row: int, col: int) -> str:
    return f"{chr(ord('A') + row)}{col + 1}"


def empty_electrodes() -> Dict[str, Dict[str, object]]:
    return {
        electrode_id(row, col): {"channel": "", "reference": False}
        for row in range(MAP_ROWS)
        for col in range(MAP_COLS)
    }


@dataclass
class ChannelMap:
    name: str = "Untitled"
    rows: int = MAP_ROWS
    cols: int = MAP_COLS
    electrodes: Dict[str, Dict[str, object]] = field(default_factory=empty_electrodes)

    @classmethod
    def new(cls, name: str = "Untitled") -> "ChannelMap":
        return cls(name=name, electrodes=empty_electrodes())

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ChannelMap":
        name = str(payload.get("name") or "Untitled")
        rows = int(payload.get("rows") or MAP_ROWS)
        cols = int(payload.get("cols") or MAP_COLS)
        raw_electrodes = payload.get("electrodes")
        electrodes = empty_electrodes() if rows == MAP_ROWS and cols == MAP_COLS else {}

        if isinstance(raw_electrodes, dict):
            for key, value in raw_electrodes.items():
                if (rows == MAP_ROWS and cols == MAP_COLS and key not in electrodes) or not isinstance(value, dict):
                    continue
                entry = dict(value)
                electrodes[key] = {
                    **entry,
                    "channel": str(value.get("channel") or "").strip(),
                    "reference": bool(value.get("reference")),
                }

        return cls(name=name, rows=rows, cols=cols, electrodes=electrodes)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "rows": self.rows,
            "cols": self.cols,
            "electrodes": self.electrodes,
        }

    def channel_for(self, electrode: str) -> str:
        return str(self.electrodes.get(electrode, {}).get("channel") or "").strip()

    def is_reference(self, electrode: str) -> bool:
        return bool(self.electrodes.get(electrode, {}).get("reference"))

    def set_channel(self, electrode: str, channel: str) -> None:
        if electrode not in self.electrodes:
            raise KeyError(f"Unknown electrode: {electrode}")
        self.electrodes[electrode]["channel"] = channel.strip()

    def set_reference(self, electrode: str, enabled: bool) -> None:
        if electrode not in self.electrodes:
            raise KeyError(f"Unknown electrode: {electrode}")
        self.electrodes[electrode]["reference"] = bool(enabled)

    def mapped_channels(self) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for electrode, payload in self.electrodes.items():
            channel = str(payload.get("channel") or "").strip()
            if channel:
                result[electrode] = channel
        return result

    def reference_electrodes(self) -> List[str]:
        return [electrode for electrode in self.electrodes if self.is_reference(electrode)]


def load_map_store(path: Path = DEFAULT_MAP_STORE) -> Dict[str, object]:
    if not path.exists():
        packaged = _packaged_map_store()
        if packaged is not None and packaged != path:
            return load_map_store(packaged)
        return {"default": "", "maps": {}}

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        return {"default": "", "maps": {}}
    maps = payload.get("maps")
    if not isinstance(maps, dict):
        maps = {}
    return {"default": str(payload.get("default") or ""), "maps": maps}


def save_map_store(payload: Dict[str, object], path: Path = DEFAULT_MAP_STORE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def list_channel_maps(path: Path = DEFAULT_MAP_STORE) -> List[str]:
    store = load_map_store(path)
    maps = store.get("maps", {})
    if not isinstance(maps, dict):
        return []
    return sorted(str(name) for name in maps.keys())


def save_channel_map(
    channel_map: ChannelMap,
    *,
    make_default: bool = False,
    path: Path = DEFAULT_MAP_STORE,
) -> None:
    store = load_map_store(path)
    maps = store.setdefault("maps", {})
    if not isinstance(maps, dict):
        maps = {}
        store["maps"] = maps
    maps[channel_map.name] = channel_map.to_dict()
    if make_default:
        store["default"] = channel_map.name
    save_map_store(store, path)


def load_channel_map(name: str, path: Path = DEFAULT_MAP_STORE) -> Optional[ChannelMap]:
    store = load_map_store(path)
    maps = store.get("maps", {})
    if not isinstance(maps, dict) or name not in maps:
        return None
    payload = maps[name]
    if not isinstance(payload, dict):
        return None
    return ChannelMap.from_dict(payload)


def default_channel_map(path: Path = DEFAULT_MAP_STORE) -> Optional[ChannelMap]:
    store = load_map_store(path)
    default_name = str(store.get("default") or "")
    if not default_name:
        return None
    return load_channel_map(default_name, path)


def set_default_channel_map(name: str, path: Path = DEFAULT_MAP_STORE) -> None:
    store = load_map_store(path)
    maps = store.get("maps", {})
    if not isinstance(maps, dict) or name not in maps:
        raise KeyError(f"Unknown channel map: {name}")
    store["default"] = name
    save_map_store(store, path)


def normalize_channel_name(channel: object) -> str:
    value = str(channel or "").strip()
    compact = re.sub(r"\s+", "", value).lower()
    axion_match = re.fullmatch(r"[a-z]\d+_(r\d+c\d+)", compact)
    if axion_match:
        return axion_match.group(1)
    match = re.fullmatch(r"(?:channel|chan|ch)?0*(\d+)", compact)
    if match:
        return f"chan{int(match.group(1))}"
    return compact


def validate_channel_map(
    channel_map: ChannelMap,
    available_channels: Iterable[str] | None = None,
) -> Dict[str, object]:
    mapped = channel_map.mapped_channels()
    channel_to_electrodes: Dict[str, List[str]] = {}
    canonical_to_mapped_channels: Dict[str, List[str]] = {}
    for electrode, channel in mapped.items():
        canonical = normalize_channel_name(channel)
        channel_to_electrodes.setdefault(canonical, []).append(electrode)
        canonical_to_mapped_channels.setdefault(canonical, [])
        if channel not in canonical_to_mapped_channels[canonical]:
            canonical_to_mapped_channels[canonical].append(channel)

    duplicates = {
        channel: electrodes
        for channel, electrodes in channel_to_electrodes.items()
        if len(electrodes) > 1
    }

    canonical_to_available_channels: Dict[str, List[str]] = {}
    for channel in available_channels or []:
        text = str(channel)
        canonical_to_available_channels.setdefault(normalize_channel_name(text), [])
        if text not in canonical_to_available_channels[normalize_channel_name(text)]:
            canonical_to_available_channels[normalize_channel_name(text)].append(text)

    available_set = set(canonical_to_available_channels)
    mapped_set = set(channel_to_electrodes)
    unknown_channels = (
        sorted(
            channel
            for canonical in mapped_set - available_set
            for channel in canonical_to_mapped_channels.get(canonical, [canonical])
        )
        if available_set
        else []
    )
    unmapped_channels = (
        sorted(
            channel
            for canonical in available_set - mapped_set
            for channel in canonical_to_available_channels.get(canonical, [canonical])
        )
        if available_set
        else []
    )
    empty_electrode_count = sum(1 for electrode in channel_map.electrodes if not channel_map.channel_for(electrode))

    return {
        "mapped_count": len(mapped),
        "empty_electrode_count": empty_electrode_count,
        "reference_electrodes": channel_map.reference_electrodes(),
        "duplicates": duplicates,
        "unknown_channels": unknown_channels,
        "unmapped_channels": unmapped_channels,
        "is_valid": not duplicates and not unknown_channels,
    }
