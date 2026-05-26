"""Tests for channel map persistence and validation."""

from src.gui.channel_map import (
    ChannelMap,
    default_channel_map,
    electrode_id,
    save_channel_map,
    validate_channel_map,
)


def test_new_channel_map_starts_empty():
    channel_map = ChannelMap.new("blank")

    assert len(channel_map.electrodes) == 64
    assert channel_map.channel_for("A1") == ""
    assert channel_map.is_reference("A1") is False


def test_save_and_load_default_channel_map(tmp_path):
    path = tmp_path / "maps.json"
    channel_map = ChannelMap.new("default-map")
    channel_map.set_channel("A1", "chan1")
    channel_map.set_reference("H8", True)

    save_channel_map(channel_map, make_default=True, path=path)
    loaded = default_channel_map(path)

    assert loaded is not None
    assert loaded.name == "default-map"
    assert loaded.channel_for("A1") == "chan1"
    assert loaded.is_reference("H8") is True


def test_validate_channel_map_reports_duplicates_and_unknowns():
    channel_map = ChannelMap.new("check")
    channel_map.set_channel("A1", "chan1")
    channel_map.set_channel("A2", "chan1")
    channel_map.set_channel(electrode_id(0, 2), "chan99")

    report = validate_channel_map(channel_map, ["chan1", "chan2"])

    assert report["is_valid"] is False
    assert report["duplicates"] == {"chan1": ["A1", "A2"]}
    assert report["unknown_channels"] == ["chan99"]
    assert report["unmapped_channels"] == ["chan2"]


def test_validate_channel_map_matches_numeric_channel_aliases():
    channel_map = ChannelMap.new("aliases")
    channel_map.set_channel("A1", "1")
    channel_map.set_channel("A2", "chan02")
    channel_map.set_channel("A3", "Channel3")

    report = validate_channel_map(channel_map, ["chan1", "chan2", "chan3", "chan4"])

    assert report["is_valid"] is True
    assert report["unknown_channels"] == []
    assert report["unmapped_channels"] == ["chan4"]
