"""Checksum utilities for SEG-Y ingestion.

This module provides optional CRC32C checksum functionality for SEG-Y ingestion.
All checksum-related imports and logic are isolated here to maintain loose coupling
with the core functionality.

Checksum calculation is controlled by the MDIO__IMPORT__DO_CRC32C environment variable
and requires optional dependencies (google-crc32c, crc32c_dist_rs).

IMPORTANT: This module uses distributed CRC32C calculation to avoid reading entire files
into memory. Never add functions that read entire files - this has severe performance
and cost implications, especially for cloud storage.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from segy.arrays import HeaderArray

logger = logging.getLogger(__name__)

# Try to import CRC32C libraries, but gracefully handle if they're not installed
_CHECKSUM_AVAILABLE = False
_IMPORT_ERROR_MSG = ""

try:
    import google_crc32c
    from crc32c_dist_rs import DistributedCRC32C

    _CHECKSUM_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERROR_MSG = (
        f"CRC32C checksum libraries not available: {e}. "
        "Install with: pip install multidimio[checksum] or pip install google-crc32c crc32c_dist_rs"
    )
    # Define placeholder types for type checking
    google_crc32c = None  # type: ignore[assignment]
    DistributedCRC32C = None  # type: ignore[assignment,misc]


def is_checksum_available() -> bool:
    """Check if checksum libraries are available.

    Returns:
        True if google-crc32c and crc32c_dist_rs are installed, False otherwise.
    """
    return _CHECKSUM_AVAILABLE


def should_calculate_checksum() -> bool:
    """Determine if checksum calculation should be performed.

    Checks both the environment variable and library availability.

    Returns:
        True if MDIO__IMPORT__DO_CRC32C is enabled and libraries are available.
    """
    env_enabled = os.getenv("MDIO__IMPORT__DO_CRC32C", "false").lower() in ("1", "true", "yes", "on")

    if env_enabled and not _CHECKSUM_AVAILABLE:
        logger.warning(
            "MDIO__IMPORT__DO_CRC32C is enabled but checksum libraries are not available. "
            "Checksum calculation will be skipped. %s",
            _IMPORT_ERROR_MSG,
        )
        return False

    return env_enabled


def require_checksum_libraries() -> None:
    """Raise an error if checksum libraries are not available.

    Raises:
        ImportError: If checksum libraries are not installed.
    """
    if not _CHECKSUM_AVAILABLE:
        raise ImportError(_IMPORT_ERROR_MSG)


def calculate_bytes_crc32c(data: bytes) -> int:
    """Calculate CRC32C checksum for a byte array.

    Args:
        data: Byte array to checksum.

    Returns:
        CRC32C checksum as integer.

    Raises:
        ImportError: If checksum libraries are not available.
    """
    require_checksum_libraries()

    crc = google_crc32c.Checksum(data)
    return int.from_bytes(crc.digest(), byteorder="big")


def create_distributed_crc32c(initial_bytes: bytes, total_length: int) -> Any:
    """Create a distributed CRC32C combiner instance.

    Args:
        initial_bytes: Initial bytes (e.g., file header).
        total_length: Total expected file length in bytes.

    Returns:
        DistributedCRC32C instance.

    Raises:
        ImportError: If checksum libraries are not available.
    """
    require_checksum_libraries()

    return DistributedCRC32C(initial_bytes, total_length)


def finalize_distributed_checksum(
    results: list[tuple[HeaderArray, tuple[int, int, int]]], combiner: Any
) -> tuple[HeaderArray, int]:
    """Finalize a distributed CRC32C checksum from scan results.

    Args:
        results: List of (HeaderArray, (byte_offset, partial_crc, byte_length)) tuples.
        combiner: DistributedCRC32C instance.

    Returns:
        Tuple of (concatenated HeaderArray, final CRC32C checksum).

    Raises:
        ValueError: If checksum finalization fails.
        ImportError: If checksum libraries are not available.
    """
    require_checksum_libraries()

    import numpy as np

    headers: list[HeaderArray] = []
    for result in results:
        headers.append(result[0])
        byte_offset, partial_crc, byte_length = result[1]
        combiner.add_fragment(byte_offset, byte_length, partial_crc)

    combined_crc = combiner.try_finalize()
    if combined_crc is None:
        msg = "Failed to finalize CRC32C - file may not be fully covered"
        raise ValueError(msg)

    # Merge headers and return with checksum
    return np.concatenate(headers), combined_crc


__all__ = [
    "is_checksum_available",
    "should_calculate_checksum",
    "require_checksum_libraries",
    "calculate_bytes_crc32c",
    "create_distributed_crc32c",
    "finalize_distributed_checksum",
]

