"""Comprehensive integration tests for CRC32C checksum functionality.

Tests verify that MDIO's CRC32C checksum generation matches the Google crc32c library
when ingesting SEG-Y files. Uses existing synthetic test data for consistency.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

import google_crc32c
import pytest

from mdio import segy_to_mdio
from mdio.builder.template_registry import TemplateRegistry
from mdio.segy.parsers import parse_headers
from mdio.segy._workers import info_worker
from mdio.segy.geometry import StreamerShotGeometryType

from tests.integration.conftest import get_segy_mock_4d_spec
from tests.integration.test_segy_import_export_masked import (
    COCA_3D_CONF,
    GATHER_2D_CONF,
    GATHER_3D_CONF,
    STACK_2D_CONF,
    STACK_3D_CONF,
    STREAMER_2D_CONF,
    STREAMER_3D_CONF,
    MaskedExportConfig,
    _segy_spec_mock_nd_segy,
    mock_nd_segy,
)

if TYPE_CHECKING:
    from pathlib import Path


def get_expected_crc32c(segy_path: Path) -> int:
    """Calculate the expected CRC32C checksum using Google crc32c library.

    This calculates CRC32C over the entire file (headers + trace data),
    matching how MDIO calculates the checksum.

    Args:
        segy_path: Path to the SEG-Y file

    Returns:
        CRC32C checksum as integer
    """
    crc = google_crc32c.Checksum()

    with segy_path.open("rb") as f:
        while True:
            chunk = f.read(8192)  # Read in chunks for memory efficiency
            if not chunk:
                break
            crc.update(chunk)

    return int.from_bytes(crc.digest(), byteorder="big")


class TestCRC32CChecksum:
    """Comprehensive integration tests for CRC32C checksum functionality."""

    @pytest.mark.parametrize("chan_header_type", [StreamerShotGeometryType.A, StreamerShotGeometryType.B])
    def test_ingestion_stores_correct_crc32c_checksum(
        self, segy_mock_4d_shots, zarr_tmp, chan_header_type, monkeypatch
    ):
        """Test that full ingestion stores the correct CRC32C checksum in MDIO metadata."""
        # Enable checksum calculation via environment variable
        monkeypatch.setenv("MDIO__IMPORT__RAW_HEADERS", "true")

        # Get synthetic SEG-Y file from existing fixtures
        segy_path = segy_mock_4d_shots[chan_header_type]

        # Calculate expected checksum using Google crc32c
        expected_crc32c = get_expected_crc32c(segy_path)

        # Import to MDIO format with checksum calculation enabled
        output_path = zarr_tmp / f"test_crc32c_{chan_header_type.value}.mdio"

        # Get the SEG-Y spec for the mock data
        segy_spec = get_segy_mock_4d_spec()

        segy_to_mdio(
            segy_spec=segy_spec,
            mdio_template=TemplateRegistry().get("PreStackShotGathers3DTime"),
            input_path=segy_path,
            output_path=output_path,
            overwrite=True,
        )

        # Verify checksum is stored in the output MDIO dataset
        import zarr
        store = zarr.open(str(output_path), mode="r")
        assert "segy_input_crc32c" in store.attrs
        stored_crc32c = store.attrs["segy_input_crc32c"]

        # Assert stored checksum matches Google crc32c library result
        assert stored_crc32c == expected_crc32c, (
            f"Stored CRC32C mismatch for {chan_header_type} geometry. "
            f"Expected: 0x{expected_crc32c:08x}, Stored: 0x{stored_crc32c:08x}"
        )

        # Verify other checksum metadata
        assert store.attrs.get("crc32c_algorithm") == "CRC32C"
        assert store.attrs.get("checksum_scope") == "full_file"
        assert store.attrs.get("checksum_library") == "google-crc32c"


    def test_ingestion_stores_correct_checksum(self, segy_mock_4d_shots, zarr_tmp, monkeypatch):
        """Test that the full ingestion process stores the correct CRC32C checksum."""
        # Enable checksum calculation via environment variable
        monkeypatch.setenv("MDIO__IMPORT__RAW_HEADERS", "true")

        # Use one of the synthetic files
        segy_path = segy_mock_4d_shots[StreamerShotGeometryType.A]

        # Calculate expected checksum
        expected_crc32c = get_expected_crc32c(segy_path)

        # Import to MDIO format
        output_path = zarr_tmp / "test_output.mdio"

        # Get the SEG-Y spec for the mock data
        segy_spec = get_segy_mock_4d_spec()

        segy_to_mdio(
            segy_spec=segy_spec,
            mdio_template=TemplateRegistry().get("PreStackShotGathers3DTime"),
            input_path=segy_path,
            output_path=output_path,
            overwrite=True,
        )

        # Verify checksum is stored in the output
        import zarr
        store = zarr.open(str(output_path), mode="r")
        assert "segy_input_crc32c" in store.attrs
        stored_crc32c = store.attrs["segy_input_crc32c"]

        assert stored_crc32c == expected_crc32c, (
            f"Stored checksum doesn't match expected. "
            f"Expected: 0x{expected_crc32c:08x}, Stored: 0x{stored_crc32c:08x}"
        )

        # Verify other checksum metadata
        assert store.attrs.get("crc32c_algorithm") == "CRC32C"
        assert store.attrs.get("checksum_scope") == "full_file"
        assert store.attrs.get("checksum_library") == "google-crc32c"

    def test_corrupted_data_detects_changes(self, segy_mock_4d_shots, tmp_path):
        """Test that checksums detect data corruption."""
        # Use one of the synthetic files
        segy_path = segy_mock_4d_shots[StreamerShotGeometryType.A]

        # Calculate original checksum
        original_crc = get_expected_crc32c(segy_path)

        # Create a corrupted copy of the file
        corrupted_path = tmp_path / "corrupted.sgy"
        with segy_path.open("rb") as src, corrupted_path.open("wb") as dst:
            dst.write(src.read())

        # Corrupt the file by modifying some bytes in the trace data
        with corrupted_path.open("r+b") as f:
            # Seek to trace data and modify some bytes
            f.seek(3600 + 100)  # Skip headers and some trace data
            f.write(b'\xFF\xFF\xFF\xFF')  # Write some different bytes

        # Calculate new checksum
        corrupted_crc = get_expected_crc32c(corrupted_path)

        # Checksums should be different
        assert corrupted_crc != original_crc, (
            f"Checksum didn't detect corruption. "
            f"Original: 0x{original_crc:08x}, Corrupted: 0x{corrupted_crc:08x}"
        )

        # Also verify MDIO detects the change
        segy_file_kwargs = {
            "url": str(corrupted_path),
            "spec": None,
            "settings": None,
            "header_overrides": None,
        }

        segy_file_info = info_worker(segy_file_kwargs)
        _, mdio_crc = parse_headers(
            segy_file_kwargs=segy_file_kwargs,
            num_traces=segy_file_info.num_traces,
            calculate_checksum=True,
        )

        assert mdio_crc == corrupted_crc, (
            f"MDIO checksum doesn't match corrupted file checksum. "
            f"MDIO: 0x{mdio_crc:08x}, Expected: 0x{corrupted_crc:08x}"
        )

    @pytest.mark.parametrize(
        "test_conf",
        [
            STACK_2D_CONF,
            STACK_3D_CONF,
            GATHER_2D_CONF,
            GATHER_3D_CONF,
            STREAMER_2D_CONF,
            STREAMER_3D_CONF,
            COCA_3D_CONF,
        ],
        ids=["2d_stack", "3d_stack", "2d_gather", "3d_gather", "2d_streamer", "3d_streamer", "3d_coca"],
    )
    def test_ingestion_stores_correct_crc32c_for_masked_export_synthetics(
        self, test_conf: MaskedExportConfig, tmp_path, monkeypatch
    ):
        """Test that CRC32C checksums are correctly stored for synthetic files from masked export tests."""
        # Enable checksum calculation via environment variable
        monkeypatch.setenv("MDIO__IMPORT__RAW_HEADERS", "true")

        grid_conf, segy_factory_conf, segy_to_mdio_conf, _ = test_conf

        # Generate synthetic SEG-Y file
        segy_path = tmp_path / f"{grid_conf.name}.sgy"
        segy_spec = mock_nd_segy(str(segy_path), grid_conf, segy_factory_conf)

        # Calculate expected checksum using Google crc32c
        expected_crc32c = get_expected_crc32c(segy_path)

        # Import to MDIO format with checksum calculation enabled
        output_path = tmp_path / f"{grid_conf.name}.mdio"
        template_name = segy_to_mdio_conf.template

        segy_to_mdio(
            segy_spec=segy_spec,
            mdio_template=TemplateRegistry().get(template_name),
            input_path=segy_path,
            output_path=output_path,
            overwrite=True,
        )

        # Verify checksum is stored in the output MDIO dataset
        import zarr
        store = zarr.open(str(output_path), mode="r")
        assert "segy_input_crc32c" in store.attrs
        stored_crc32c = store.attrs["segy_input_crc32c"]

        # Assert stored checksum matches Google crc32c library result
        assert stored_crc32c == expected_crc32c, (
            f"Stored CRC32C mismatch for {grid_conf.name} configuration. "
            f"Expected: 0x{expected_crc32c:08x}, Stored: 0x{stored_crc32c:08x}"
        )

        # Verify other checksum metadata
        assert store.attrs.get("crc32c_algorithm") == "CRC32C"
        assert store.attrs.get("checksum_scope") == "full_file"
        assert store.attrs.get("checksum_library") == "google-crc32c"
