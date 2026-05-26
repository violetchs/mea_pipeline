"""Tests for MEA I/O module."""

import pytest
import numpy as np
from pathlib import Path
import struct

from src.mea_io import (
    MEAReader,
    MEAWriter,
    UnifiedMEAData,
    filter_unified_by_wells,
    list_axion_spk_wells,
    read_axion_spk,
    read_blackrock_nev,
    read_maxwell_h5,
    read_unified_npz,
    save_unified_npz,
)


class TestMEAReader:
    """Test MEAReader class."""
    
    def test_reader_initialization(self):
        """Test MEAReader initialization."""
        reader = MEAReader("data.npy")
        assert reader.filepath == "data.npy"

    def test_read_write_npy(self, tmp_path, sample_data):
        path = tmp_path / "sample.npy"
        MEAWriter(path).save_data(sample_data)
        loaded = MEAReader(path).load_data()
        np.testing.assert_allclose(loaded, sample_data)

    def test_reader_loads_axion_spk_as_unified_data(self, tmp_path, monkeypatch):
        import src.mea_io.spike_readers as spike_readers

        class FakeAxionSpkReader:
            def __init__(self, file_path):
                self.file_path = file_path

            def read_spk(self):
                return [
                    {
                        "well": "A1",
                        "electrode": "r1c1",
                        "data": {
                            "spike_times": np.array([0.2, 0.1]),
                            "spike_waveform": np.array([[1.0, 2.0], [3.0, 4.0]]),
                        },
                    },
                    {
                        "well": "A2",
                        "electrode": "r1c1",
                        "data": {
                            "spike_times": np.array([1.0]),
                            "spike_waveform": np.array([[5.0, 6.0]]),
                        },
                    },
                ], [{"EventTime": 0.15, "EventTimeSample": 15}]

            def close(self):
                pass

        monkeypatch.setattr(spike_readers, "AxionSpkReader", FakeAxionSpkReader)
        path = tmp_path / "sample.spk"
        path.write_bytes(b"FakeSpk")

        data = MEAReader(path).load_data()

        assert isinstance(data, UnifiedMEAData)
        assert data.meta["source"] == "axion_spk"
        assert sorted(data.channels()) == ["A1_r1c1", "A2_r1c1"]
        np.testing.assert_allclose(data.spikes["A1_r1c1"], np.array([0.0, 0.1]))
        np.testing.assert_allclose(data.spikes["A2_r1c1"], np.array([0.9]))
        np.testing.assert_allclose(data.waveforms["A1_r1c1"], np.array([[3.0, 4.0], [1.0, 2.0]]))
        assert data.meta["channel_map"]["A1_r1c1"]["well"] == "A1"
        assert data.meta["channel_map"]["A2_r1c1"]["mea_electrode"] == "A1"
        np.testing.assert_allclose(data.stim_times, np.array([0.05]))

    def test_reader_loads_axionbio_spk_natively(self, tmp_path):
        from src.mea_io.spike_readers import AXIONBIO_SPIKE_MARKER

        path = tmp_path / "native.spk"
        raw = bytearray(b"AxionBio" + b"\x00" * 4096)
        metadata = (
            b"Pre-Spike Duration,0.84 ms\n"
            b"Post-Spike Duration,2.16 ms\n"
            b"Voltage Scale,-5.0E-08 V/sample\n"
        )
        raw[64 : 64 + len(metadata)] = metadata

        table_offset = 512
        for index in range(64):
            row = index // 8 + 1
            col = index % 8 + 1
            raw[table_offset + index * 8 : table_offset + (index + 1) * 8] = bytes(
                [0, 0, 1, 1, col, row, 1, index]
            )

        marker_offset = 1600
        raw[marker_offset : marker_offset + len(AXIONBIO_SPIKE_MARKER)] = AXIONBIO_SPIKE_MARKER
        data_offset = marker_offset + len(AXIONBIO_SPIKE_MARKER) + 4
        waveform = np.arange(38, dtype="<i2")
        records = [
            struct.pack("<qHIdd", 50000 * (index + 1), 0x0100, 11, 1.0e-6, 6.0)
            + (waveform + index).tobytes()
            for index in range(8)
        ]
        payload = b"".join(records)
        raw[data_offset : data_offset + len(payload)] = payload
        path.write_bytes(bytes(raw[: data_offset + len(payload)]))

        data = MEAReader(path).load_data()

        assert data.meta["reader"] == "native_axionbio"
        assert data.meta["valid_spike_count"] == 8
        assert list(data.spikes) == ["A1_r1c1"]
        np.testing.assert_allclose(data.spikes["A1_r1c1"][:2], np.array([0.0, 1.0]))
        assert data.waveforms["A1_r1c1"].shape == (8, 38)
        assert data.waveforms["A1_r1c1"].dtype == np.int16
        assert data.meta["channel_map"]["A1_r1c1"]["well_index"] == 1
        assert data.meta["channel_map"]["A1_r1c1"]["electrode_index"] == 0
        assert data.meta["waveform_unit"] == "adc_counts"
        assert data.sr == pytest.approx(38 / 0.003)

    def test_axionbio_spk_lists_and_filters_wells(self, tmp_path):
        from src.mea_io.spike_readers import AXIONBIO_SPIKE_MARKER

        path = tmp_path / "native_wells.spk"
        raw = bytearray(b"AxionBio" + b"\x00" * 8192)
        struct.pack_into("<d", raw, 0x127A, 12500.0)
        struct.pack_into("<d", raw, 0x12D6, 4.0)

        table_offset = 512
        for index in range(64):
            row = index // 8 + 1
            col = index % 8 + 1
            raw[table_offset + index * 8 : table_offset + (index + 1) * 8] = bytes(
                [0, 0, 1, 1, col, row, 1, index]
            )
        for index in range(64):
            row = index // 8 + 1
            col = index % 8 + 1
            offset = table_offset + (64 + index) * 8
            raw[offset : offset + 8] = bytes([0, 0, 1, 2, col, row, 2, index])

        marker_offset = 6000
        raw[marker_offset : marker_offset + len(AXIONBIO_SPIKE_MARKER)] = AXIONBIO_SPIKE_MARKER
        data_offset = marker_offset + len(AXIONBIO_SPIKE_MARKER) + 4
        waveform = np.arange(38, dtype="<i2")
        records = [
            struct.pack("<qHIdd", 12500 * (index + 1), 0x0100 if index % 2 == 0 else 0x0200, 11, 1.0e-6, 6.0)
            + waveform.tobytes()
            for index in range(8)
        ]
        payload = b"".join(records)
        raw[data_offset : data_offset + len(payload)] = payload
        path.write_bytes(bytes(raw[: data_offset + len(payload)]))

        assert list_axion_spk_wells(path) == ["A1", "B1"]
        data = read_axion_spk(path, wells=["B1"])

        assert data.meta["selected_wells"] == ["B1"]
        assert data.meta["wells"] == ["B1"]
        assert list(data.spikes) == ["B1_r1c1"]
        assert data.waveforms["B1_r1c1"].shape == (4, 38)

    def test_filter_unified_by_wells_keeps_matching_channels(self):
        data = UnifiedMEAData(
            spikes={"A1_r1c1": np.array([0.1]), "B1_r1c1": np.array([0.2])},
            waveforms={
                "A1_r1c1": np.ones((1, 2)),
                "B1_r1c1": np.ones((1, 2)) * 2,
            },
            meta={
                "source": "axion_spk",
                "wells": ["A1", "B1"],
                "channel_map": {
                    "A1_r1c1": {"well": "A1", "electrode": "r1c1"},
                    "B1_r1c1": {"well": "B1", "electrode": "r1c1"},
                },
            },
        )

        filtered = filter_unified_by_wells(data, ["A1"])

        assert filtered.channels() == ["A1_r1c1"]
        assert filtered.meta["selected_wells"] == ["A1"]
        assert filtered.meta["filtered_spike_count"] == 1

    def test_axionbio_spk_uses_header_sampling_frequency_timebase(self, tmp_path):
        from src.mea_io.spike_readers import AXIONBIO_SPIKE_MARKER

        path = tmp_path / "native_duration.spk"
        raw = bytearray(b"AxionBio" + b"\x00" * 8192)
        metadata = b"Pre-Spike Duration,0.84 ms\nPost-Spike Duration,2.16 ms\n"
        raw[64 : 64 + len(metadata)] = metadata
        struct.pack_into("<d", raw, 0x127A, 100000.0)
        struct.pack_into("<d", raw, 0x12D6, 4.0)

        table_offset = 512
        for index in range(64):
            row = index // 8 + 1
            col = index % 8 + 1
            raw[table_offset + index * 8 : table_offset + (index + 1) * 8] = bytes(
                [0, 0, 1, 1, col, row, 1, index]
            )

        marker_offset = 6000
        raw[marker_offset : marker_offset + len(AXIONBIO_SPIKE_MARKER)] = AXIONBIO_SPIKE_MARKER
        data_offset = marker_offset + len(AXIONBIO_SPIKE_MARKER) + 4
        waveform = np.arange(38, dtype="<i2")
        records = [
            struct.pack("<qHIdd", 50000 * (index + 1), 0x0100, 11, 1.0e-6, 6.0)
            + waveform.tobytes()
            for index in range(8)
        ]
        payload = b"".join(records)
        raw[data_offset : data_offset + len(payload)] = payload
        path.write_bytes(bytes(raw[: data_offset + len(payload)]))

        data = read_axion_spk(path)

        assert data.meta["duration_s"] == pytest.approx(4.0)
        assert data.meta["nominal_duration_s"] == pytest.approx(4.0)
        assert data.meta["timestamp_frequency_hz"] == pytest.approx(100000.0)
        assert data.meta["timestamp_frequency_source"] == "header_sampling_frequency"
        np.testing.assert_allclose(data.spikes["A1_r1c1"][:2], np.array([0.5, 1.0]))


class TestMEAWriter:
    """Test MEAWriter class."""
    
    def test_writer_initialization(self):
        """Test MEAWriter initialization."""
        writer = MEAWriter("out.npy")
        assert writer.filepath == "out.npy"


class TestMaxwellH5Reader:
    """Test Maxwell Biosystems H5 spike-event reading."""

    def test_read_maxwell_h5_spikes_and_mapping(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        path = tmp_path / "data.raw.h5"

        mapping_dtype = np.dtype(
            [
                ("channel", "<i4"),
                ("electrode", "<i4"),
                ("x", "<f8"),
                ("y", "<f8"),
            ]
        )
        spike_dtype = np.dtype(
            [
                ("frameno", "<i8"),
                ("channel", "<i4"),
                ("amplitude", "<f4"),
            ]
        )

        with h5py.File(path, "w") as h5:
            h5.create_dataset("version", data=np.asarray([b"20190530"]))
            group = h5.create_group("data_store/data0000")
            group.create_dataset("well_id", data=np.asarray([0], dtype=np.int32))
            group.create_dataset("recording_id", data=np.asarray([0], dtype=np.int32))
            group.create_dataset("start_time", data=np.asarray([1000], dtype=np.int64))
            group.create_dataset("stop_time", data=np.asarray([1100], dtype=np.int64))
            settings = group.create_group("settings")
            settings.create_dataset("sampling", data=np.asarray([20000.0]))
            settings.create_dataset(
                "mapping",
                data=np.asarray(
                    [(1, 11, 17.5, 27.5), (2, 12, 87.5, 97.5)],
                    dtype=mapping_dtype,
                ),
            )
            exp = group.create_group("groups/exp")
            exp.create_dataset("frame_nos", data=np.arange(1000, 3000, dtype=np.uint64))
            exp.create_dataset("channels", data=np.asarray([1, 2], dtype=np.uint16))
            exp.create_dataset(
                "raw",
                data=np.vstack(
                    [
                        np.arange(2000, dtype=np.uint16),
                        np.arange(2000, dtype=np.uint16) + 1000,
                    ]
                ),
            )
            group.create_dataset(
                "spikes",
                data=np.asarray(
                    [(1100, 1, -15.0), (1300, 1, -12.0), (1500, 2, -20.0)],
                    dtype=spike_dtype,
                ),
            )

        data = read_maxwell_h5(path)

        assert data.meta["source"] == "maxwell_h5"
        assert data.meta["reader"] == "native_maxwell_h5"
        assert data.sr == pytest.approx(20000.0)
        assert sorted(data.channels()) == ["well0_e11", "well0_e12"]
        np.testing.assert_allclose(data.spikes["well0_e11"], np.array([0.005, 0.015]))
        np.testing.assert_allclose(data.spikes["well0_e12"], np.array([0.025]))
        assert data.meta["duration_s"] == pytest.approx(0.1)
        assert data.meta["channel_map"]["well0_e11"]["source_channel"] == 1
        assert data.meta["channel_map"]["well0_e11"]["electrode"] == 11
        assert data.meta["channel_map"]["well0_e11"]["x_um"] == pytest.approx(17.5)
        assert data.meta["raw_data"][0]["path"] == "data_store/data0000/groups/exp/raw"
        assert data.meta["waveform_unit"] == "adc_counts"
        assert data.waveforms["well0_e11"].shape == (2, 61)
        np.testing.assert_allclose(data.waveforms["well0_e11"][0], np.arange(80, 141, dtype=float))
        np.testing.assert_allclose(data.waveforms["well0_e12"][0], np.arange(1480, 1541, dtype=float))

    def test_mea_reader_loads_maxwell_h5(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        path = tmp_path / "reader.raw.h5"
        mapping_dtype = np.dtype([("channel", "<i4"), ("electrode", "<i4"), ("x", "<f8"), ("y", "<f8")])
        spike_dtype = np.dtype([("frameno", "<i8"), ("channel", "<i4"), ("amplitude", "<f4")])
        with h5py.File(path, "w") as h5:
            group = h5.create_group("data_store/data0000")
            group.create_dataset("well_id", data=np.asarray([0], dtype=np.int32))
            settings = group.create_group("settings")
            settings.create_dataset("sampling", data=np.asarray([10000.0]))
            settings.create_dataset("mapping", data=np.asarray([(7, 77, 1.0, 2.0)], dtype=mapping_dtype))
            group.create_dataset("spikes", data=np.asarray([(10, 7, 1.0)], dtype=spike_dtype))

        data = MEAReader(path).load_data()

        assert isinstance(data, UnifiedMEAData)
        assert data.meta["source"] == "maxwell_h5"
        np.testing.assert_allclose(data.spikes["well0_e77"], np.array([0.0]))

    def test_read_maxwell_h5_local_sample(self):
        path = Path("data/CRR/260520/26048/ActivityScan/000003/data.raw.h5")
        if not path.exists():
            pytest.skip("Local Maxwell fixture is not available")

        data = read_maxwell_h5(path)

        assert data.meta["source"] == "maxwell_h5"
        assert data.sr == pytest.approx(20000.0)
        assert data.meta["spike_count"] == 4329
        assert len(data.spikes) > 0
        assert data.meta["wells"] == ["well0"]


class TestBlackrockNevReader:
    """Test Blackrock NEV spike-event reading."""

    def test_read_blackrock_nev_accepts_brevents_header(self, tmp_path):
        path = tmp_path / "legacy.nev"
        raw = bytearray(336)
        raw[:8] = b"BREVENTS"
        struct.pack_into("<BBHIIII", raw, 8, 3, 0, 1, 336, 8, 30000, 30000)
        struct.pack_into("<8H", raw, 28, 2026, 5, 1, 18, 12, 0, 0, 0)
        struct.pack_into("<I", raw, 332, 0)
        path.write_bytes(raw)

        data = read_blackrock_nev(path)

        assert data.meta["basic_header"]["file_type_id"] == "BREVENTS"
        assert data.meta["basic_header"]["bytes_in_data_packets"] == 8
        assert data.sr == 30000.0
        assert data.meta["packet_count"] == 0

    def test_read_blackrock_nev_uses_waveform_metadata_width(self, tmp_path):
        path = tmp_path / "legacy_with_waveform.nev"
        header_bytes = 368
        packet_bytes = 108
        raw = bytearray(header_bytes + packet_bytes)
        raw[:8] = b"BREVENTS"
        struct.pack_into("<BBHIIII", raw, 8, 3, 0, 1, header_bytes, packet_bytes, 30000, 30000)
        struct.pack_into("<8H", raw, 28, 2026, 5, 1, 18, 12, 0, 0, 0)
        struct.pack_into("<I", raw, 332, 1)

        raw[336:344] = b"NEUEVWAV"
        struct.pack_into("<H", raw, 344, 1)
        raw[346] = 1
        raw[347] = 1
        struct.pack_into("<H", raw, 348, 1000)
        raw[357] = 2
        struct.pack_into("<H", raw, 358, 48)

        packet_offset = header_bytes
        struct.pack_into("<IHBB", raw, packet_offset, 30000, 1, 0, 0)
        raw[packet_offset + 8 : packet_offset + 12] = b"\x11\x22\x33\x44"
        waveform = np.arange(48, dtype="<i2")
        raw[packet_offset + 12 : packet_offset + packet_bytes] = waveform.tobytes()
        path.write_bytes(raw)

        data = read_blackrock_nev(path)

        assert data.waveforms["chan1"].shape == (1, 48)
        np.testing.assert_allclose(data.waveforms["chan1"][0], waveform.astype(float))

    def test_read_blackrock_nev_accepts_brevents_alternate_packet_layout(self, tmp_path):
        path = tmp_path / "legacy_alternate_packet.nev"
        header_bytes = 368
        packet_bytes = 108
        raw = bytearray(header_bytes + packet_bytes)
        raw[:8] = b"BREVENTS"
        struct.pack_into("<BBHIIII", raw, 8, 3, 0, 1, header_bytes, packet_bytes, 30000, 30000)
        struct.pack_into("<8H", raw, 28, 2026, 5, 1, 18, 12, 0, 0, 0)
        struct.pack_into("<I", raw, 332, 1)

        raw[336:344] = b"NEUEVWAV"
        struct.pack_into("<H", raw, 344, 32)
        raw[357] = 2
        struct.pack_into("<H", raw, 358, 48)

        packet_offset = header_bytes
        struct.pack_into("<IHHH", raw, packet_offset, 60000, 0, 0, 32)
        raw[packet_offset + 10] = 3
        waveform = np.arange(48, dtype="<i2")
        raw[packet_offset + 12 : packet_offset + packet_bytes] = waveform.tobytes()
        path.write_bytes(raw)

        data = read_blackrock_nev(path)

        assert list(data.spikes) == ["chan32"]
        np.testing.assert_allclose(data.spikes["chan32"], np.array([0.0]))
        assert data.meta["timestamp_offset_s"] == 2.0
        assert data.meta["duration_s"] == 0.0
        np.testing.assert_array_equal(data.sorting["chan32"]["labels"], np.array([3], dtype=np.int32))
        assert data.waveforms["chan32"].shape == (1, 48)

    def test_read_blackrock_nev_keeps_generic_event_timebase(self, tmp_path):
        path = tmp_path / "legacy_event_before_spike.nev"
        header_bytes = 368
        packet_bytes = 108
        raw = bytearray(header_bytes + packet_bytes * 2)
        raw[:8] = b"BREVENTS"
        struct.pack_into("<BBHIIII", raw, 8, 3, 0, 1, header_bytes, packet_bytes, 30000, 30000)
        struct.pack_into("<8H", raw, 28, 2026, 5, 1, 18, 12, 0, 0, 0)
        struct.pack_into("<I", raw, 332, 1)

        raw[336:344] = b"NEUEVWAV"
        struct.pack_into("<H", raw, 344, 1)
        raw[357] = 2
        struct.pack_into("<H", raw, 358, 48)

        event_offset = header_bytes
        struct.pack_into("<IHBB", raw, event_offset, 0, 0, 0, 0)

        spike_offset = header_bytes + packet_bytes
        struct.pack_into("<IHBB", raw, spike_offset, 90000, 1, 0, 0)
        waveform = np.arange(48, dtype="<i2")
        raw[spike_offset + 12 : spike_offset + packet_bytes] = waveform.tobytes()
        path.write_bytes(raw)

        data = read_blackrock_nev(path)

        np.testing.assert_allclose(data.spikes["chan1"], np.array([3.0]))
        np.testing.assert_allclose(data.stim_times, np.array([0.0]))
        assert data.meta["timestamp_offset_s"] == 0.0
        assert data.meta["timestamp_offset_source"] == "first_event"

    def test_read_blackrock_nev_keeps_recording_start_before_first_spike(self, tmp_path):
        path = tmp_path / "legacy_recording_start.nev"
        header_bytes = 368
        packet_bytes = 108
        raw = bytearray(header_bytes + packet_bytes * 3)
        raw[:8] = b"BREVENTS"
        struct.pack_into("<BBHIIII", raw, 8, 3, 0, 1, header_bytes, packet_bytes, 30000, 30000)
        struct.pack_into("<8H", raw, 28, 2026, 5, 1, 18, 12, 0, 0, 0)
        struct.pack_into("<I", raw, 332, 1)

        raw[336:344] = b"NEUEVWAV"
        struct.pack_into("<H", raw, 344, 1)
        raw[357] = 2
        struct.pack_into("<H", raw, 358, 48)

        start_offset = header_bytes
        struct.pack_into("<IHHH", raw, start_offset, 0, 0, 0, 0xFFF9)
        struct.pack_into("<H", raw, start_offset + 10, 0)

        spike_offset = header_bytes + packet_bytes
        struct.pack_into("<IHBB", raw, spike_offset, 90000, 1, 0, 0)

        stop_offset = header_bytes + packet_bytes * 2
        struct.pack_into("<IHHH", raw, stop_offset, 18000000, 0, 0, 0xFFF9)
        struct.pack_into("<H", raw, stop_offset + 10, 1)
        path.write_bytes(raw)

        data = read_blackrock_nev(path)

        np.testing.assert_allclose(data.spikes["chan1"], np.array([3.0]))
        assert data.meta["timestamp_offset_s"] == 0.0
        assert data.meta["timestamp_offset_source"] == "recording_start_event"
        assert data.meta["duration_s"] == 600.0

    def test_read_blackrock_nev_infers_corrupt_zero_start_from_stop_event(self, tmp_path):
        path = tmp_path / "legacy_corrupt_start.nev"
        header_bytes = 368
        packet_bytes = 108
        raw = bytearray(header_bytes + packet_bytes * 3)
        raw[:8] = b"BREVENTS"
        struct.pack_into("<BBHIIII", raw, 8, 3, 0, 1, header_bytes, packet_bytes, 30000, 30000)
        struct.pack_into("<8H", raw, 28, 2026, 5, 1, 18, 12, 0, 0, 0)
        struct.pack_into("<I", raw, 332, 1)

        raw[336:344] = b"NEUEVWAV"
        struct.pack_into("<H", raw, 344, 1)
        raw[357] = 2
        struct.pack_into("<H", raw, 358, 48)

        start_offset = header_bytes
        struct.pack_into("<IHHH", raw, start_offset, 0, 0, 0, 0xFFF9)
        struct.pack_into("<H", raw, start_offset + 10, 0)

        spike_offset = header_bytes + packet_bytes
        struct.pack_into("<IHBB", raw, spike_offset, 3000000, 1, 0, 0)

        stop_offset = header_bytes + packet_bytes * 2
        struct.pack_into("<IHHH", raw, stop_offset, 21000000, 0, 0, 0xFFF9)
        struct.pack_into("<H", raw, stop_offset + 10, 1)
        path.write_bytes(raw)

        data = read_blackrock_nev(path)

        np.testing.assert_allclose(data.spikes["chan1"], np.array([0.0]))
        assert data.meta["timestamp_offset_s"] == 100.0
        assert data.meta["timestamp_offset_source"] == "inferred_recording_start_from_stop_event"
        assert data.meta["duration_s"] == 600.0

    def test_read_blackrock_nev_reports_actual_unsupported_type(self, tmp_path):
        path = tmp_path / "not_nev.nev"
        raw = bytearray(336)
        raw[:8] = b"TYPEBAD!"
        path.write_bytes(raw)

        with pytest.raises(ValueError, match="TYPEBAD!"):
            read_blackrock_nev(path)

    def test_read_blackrock_nev_sample(self):
        path = Path("data/test/C13001.nev")
        if not path.exists():
            pytest.skip("Local NEV fixture is not available")

        data = read_blackrock_nev(path)

        assert data.meta["source"] == "blackrock_nev"
        assert data.meta["packet_count"] == 17577
        assert data.meta["basic_header"]["bytes_in_data_packets"] == 104
        assert data.sr == 30000.0
        assert len(data.spikes) == 33
        assert sum(values.size for values in data.spikes.values()) == 17577
        assert data.stim_times.size == 0

        first_channel = sorted(data.channels())[0]
        assert first_channel == "chan1"
        assert data.waveforms[first_channel].shape[1] == 48
        assert data.spikes[first_channel].shape[0] == data.waveforms[first_channel].shape[0]


class TestUnifiedSortingSave:
    def test_mea_reader_detects_unified_npz(self, tmp_path):
        data = UnifiedMEAData(
            spikes={"chan1": np.array([0.1, 0.2])},
            waveforms={"chan1": np.ones((2, 4))},
            sr=30000.0,
            sorting={"chan1": {"waveform_cluster_labels": np.array([0, 1], dtype=np.int32)}},
        )
        path = tmp_path / "sorted.npz"

        save_unified_npz(data, path)
        loaded = MEAReader(path).load_data()

        assert isinstance(loaded, UnifiedMEAData)
        np.testing.assert_allclose(loaded.spikes["chan1"], np.array([0.1, 0.2]))
        np.testing.assert_array_equal(loaded.sorting["chan1"]["waveform_cluster_labels"], np.array([0, 1]))

    def test_save_unified_npz_preserves_channel_units(self, tmp_path):
        data = UnifiedMEAData(
            spikes={"chan1": np.array([0.1, 0.2, 0.3, 0.4])},
            waveforms={"chan1": np.arange(16, dtype=float).reshape(4, 4)},
            sr=30000.0,
            sorting={
                "_waveform_clustering": {"method": "waveform_clustering"},
                "chan1": {
                    "waveform_cluster_labels": np.array([0, 1, 1, -1], dtype=np.int32),
                    "embedding": np.ones((4, 2), dtype=np.float32),
                },
            },
        )
        path = tmp_path / "sorted.npz"

        save_unified_npz(data, path)
        loaded = read_unified_npz(path)

        np.testing.assert_allclose(loaded.spikes["chan1"], data.spikes["chan1"])
        np.testing.assert_array_equal(
            loaded.sorting["chan1"]["waveform_cluster_labels"],
            np.array([0, 1, 1, -1], dtype=np.int32),
        )
        with np.load(path) as npz:
            assert "unit_spikes_chan1_unit0" in npz.files
            assert "unit_spikes_chan1_unit1" in npz.files
            assert "unit_spikes_chan1_noise1" in npz.files
            np.testing.assert_allclose(npz["unit_spikes_chan1_unit1"], np.array([0.2, 0.3]))
