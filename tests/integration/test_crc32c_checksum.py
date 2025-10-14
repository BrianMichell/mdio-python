"""Comprehensive integration tests for CRC32C checksum functionality.

Tests verify that MDIO's CRC32C checksum generation matches the Google crc32c library
when ingesting SEG-Y files. Uses existing synthetic test data for consistency.
"""

from pathlib import Path

import pytest
import zarr
from tests.integration.test_segy_import_export_masked import COCA_3D_CONF
from tests.integration.test_segy_import_export_masked import GATHER_2D_CONF
from tests.integration.test_segy_import_export_masked import GATHER_3D_CONF
from tests.integration.test_segy_import_export_masked import STACK_2D_CONF
from tests.integration.test_segy_import_export_masked import STACK_3D_CONF
from tests.integration.test_segy_import_export_masked import STREAMER_2D_CONF
from tests.integration.test_segy_import_export_masked import STREAMER_3D_CONF
from tests.integration.test_segy_import_export_masked import MaskedExportConfig
from tests.integration.test_segy_import_export_masked import mock_nd_segy

from mdio import segy_to_mdio
from mdio.builder.template_registry import TemplateRegistry
from mdio.segy._workers import info_worker
from mdio.segy.checksum import is_checksum_available
from mdio.segy.checksum import parse_headers

# Skip all tests in this module if checksum libraries are not available
pytestmark = pytest.mark.skipif(
    not is_checksum_available(), reason="CRC32C checksum libraries (google-crc32c, crc32c_dist_rs) not installed"
)


def get_expected_crc32c(segy_path: Path) -> int:
    """Calculate the expected CRC32C checksum using Google crc32c library.

    This calculates CRC32C over the entire file (headers + trace data),
    matching how MDIO calculates the checksum.

    NOTE: This is TEST-ONLY code. Never use this in production as it reads
    the entire file into memory which has significant performance and cost
    penalties, especially for cloud storage.

    Args:
        segy_path: Path to the SEG-Y file

    Returns:
        CRC32C checksum as integer
    """
    # Import here to keep it test-only
    import google_crc32c  # noqa: PLC0415

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
    def test_ingestion_stores_correct_crc32c_for_all_configurations(
        self, test_conf: MaskedExportConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that CRC32C checksums are correctly stored for synthetic files from masked export tests."""
        # Enable checksum calculation via environment variable
        monkeypatch.setenv("MDIO__IMPORT__RAW_HEADERS", "true")
        monkeypatch.setenv("MDIO__IMPORT__DO_CRC32C", "true")

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

    def test_corrupted_data_detects_changes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that checksums detect data corruption using synthetic streamer data."""
        # Enable checksum calculation via environment variable
        monkeypatch.setenv("MDIO__IMPORT__RAW_HEADERS", "true")
        monkeypatch.setenv("MDIO__IMPORT__DO_CRC32C", "true")

        # Use streamer configuration for this test
        test_conf = STREAMER_3D_CONF
        grid_conf, segy_factory_conf, segy_to_mdio_conf, _ = test_conf

        # Generate synthetic SEG-Y file
        segy_path = tmp_path / "corruption_test.sgy"
        segy_spec = mock_nd_segy(str(segy_path), grid_conf, segy_factory_conf)

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
            f.write(b"\xff\xff\xff\xff")  # Write some different bytes

        # Calculate new checksum
        corrupted_crc = get_expected_crc32c(corrupted_path)

        # Checksums should be different
        assert corrupted_crc != original_crc, (
            f"Checksum didn't detect corruption. Original: 0x{original_crc:08x}, Corrupted: 0x{corrupted_crc:08x}"
        )

        # Also verify MDIO detects the change
        segy_file_kwargs = {
            "url": str(corrupted_path),
            "spec": segy_spec,
            "settings": None,
            "header_overrides": None,
        }

        segy_file_info = info_worker(segy_file_kwargs)
        _, mdio_crc = parse_headers(
            segy_file_kwargs=segy_file_kwargs,
            num_traces=segy_file_info.num_traces,
        )

        assert mdio_crc == corrupted_crc, (
            f"MDIO checksum doesn't match corrupted file checksum. "
            f"MDIO: 0x{mdio_crc:08x}, Expected: 0x{corrupted_crc:08x}"
        )
