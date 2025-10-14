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
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from math import ceil
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from psutil import cpu_count
from segy import SegyFile
from segy.arrays import HeaderArray
from tqdm.auto import tqdm
from upath import UPath

from mdio.segy._raw_trace_wrapper import SegyFileRawTraceWrapper

if TYPE_CHECKING:
    from mdio.segy._workers import SegyFileArguments

    try:
        from crc32c_dist_rs import DistributedCRC32C
    except ImportError:
        DistributedCRC32C = Any  # type: ignore[assignment,misc]

default_cpus = cpu_count(logical=True)

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


def header_scan_worker(
    segy_file_kwargs: SegyFileArguments,
    trace_range: tuple[int, int],
    subset: list[str] | None = None,
) -> HeaderArray | tuple[HeaderArray, tuple[int, int, int]]:
    """Header scan worker with optional checksum calculation.

    If SegyFile is not open, it can either accept a path string or a handle that was opened in
    a different context manager.

    Args:
        segy_file_kwargs: Arguments to open SegyFile instance.
        trace_range: Tuple consisting of the trace ranges to read.
        subset: List of header names to filter and keep.
        calculate_checksum: If True, also calculate CRC32C for this trace range.

    Returns:
        HeaderArray if calculate_checksum is False, otherwise tuple of (HeaderArray, checksum_info)
        where checksum_info is (byte_offset, crc32c, byte_length).
    """
    print("Using header_scan_worker from checksum.py")
    segy_file = SegyFile(**segy_file_kwargs)

    start_trace, end_trace = trace_range
    trace_indices = list(range(start_trace, end_trace))

    # Read full trace data (header + samples) ONCE using the raw trace wrapper
    # This avoids duplicate I/O - we get both headers and raw bytes in a single read
    traces = SegyFileRawTraceWrapper(segy_file, trace_indices)

    # Extract headers from the data we already read (no additional I/O)
    trace_header = traces.header

    if subset is not None:
        # struct field selection needs a list, not a tuple; a subset is a tuple from the template.
        trace_header = trace_header[list(subset)]

    # Get non-void fields from dtype and copy to new array for memory efficiency
    fields = trace_header.dtype.fields
    non_void_fields = [(name, dtype) for name, (dtype, _) in fields.items()]
    new_dtype = np.dtype(non_void_fields)

    # Copy to non-padded memory, ndmin is to handle the case where there is 1 trace in block
    # (singleton) so we can concat and assign stuff later.
    trace_header = np.array(trace_header, dtype=new_dtype, ndmin=1)

    # Calculate checksum from the raw bytes ALREADY IN MEMORY (NO ADDITIONAL I/O!)
    raw_bytes = traces.trace_buffer_array.tobytes()
    partial_crc32c = calculate_bytes_crc32c(raw_bytes)

    # Calculate byte offset and length
    trace_header_size = segy_file.spec.trace.header.itemsize
    # sample_size = segy_file.spec.trace.sample.itemsize
    sample_size = 4  # This will always be a 4-byte float
    num_samples = len(segy_file.sample_labels)
    trace_size = trace_header_size + (num_samples * sample_size)

    byte_offset = 3600 + start_trace * trace_size
    byte_length = len(raw_bytes)

    checksum_info = (byte_offset, partial_crc32c, byte_length)

    return HeaderArray(trace_header), checksum_info


def parse_headers(  # noqa: PLR0913
    segy_file_kwargs: SegyFileArguments,
    num_traces: int,
    subset: list[str] | None = None,
    block_size: int = 10000,
    progress_bar: bool = True,
) -> HeaderArray | tuple[HeaderArray, int]:
    """Read and parse given `byte_locations` from SEG-Y file.

    Args:
        segy_file_kwargs: SEG-Y file arguments.
        num_traces: Total number of traces in the SEG-Y file.
        subset: List of header names to filter and keep.
        block_size: Number of traces to read for each block.
        progress_bar: Enable or disable progress bar. Default is True.
        calculate_checksum: If True, also calculate CRC32C checksum for all trace data.

    Returns:
        HeaderArray if calculate_checksum is False.
        Tuple of (HeaderArray, combined_crc32c) if calculate_checksum is True.
    """
    # Dynamically import the appropriate header_scan_worker based on checksum requirement
    # if should_calculate_checksum() and is_checksum_available():
    #     from mdio.segy.checksum import header_scan_worker
    # else:
    #     from mdio.segy._workers import header_scan_worker

    # Type hint for the dynamically imported function
    # header_scan_worker: Any
    # Initialize combiner only if checksum calculation is needed and available
    # combiner: Any = None
    # Use UPath for cloud/filesystem agnostic reading
    path = UPath(segy_file_kwargs["url"])
    raw_bytes = path.fs.read_block(
        fn=str(path),
        offset=0,
        length=3600,
    )

    # Calculate trace size to determine total data length
    # We need to open the file briefly to get the spec
    sf = SegyFile(**segy_file_kwargs)
    trace_header_size = sf.spec.trace.header.itemsize
    sample_size = 4  # This will always be a 4-byte float
    num_samples = len(sf.sample_labels)
    trace_size = trace_header_size + (num_samples * sample_size)

    # Total length is header (3600) + trace data
    total_len = 3600 + num_traces * trace_size
    combiner = create_distributed_crc32c(raw_bytes, total_len)

    trace_count = num_traces
    n_blocks = int(ceil(trace_count / block_size))

    trace_ranges = []
    for idx in range(n_blocks):
        start, stop = idx * block_size, (idx + 1) * block_size
        stop = min(stop, trace_count)

        trace_ranges.append((start, stop))

    num_cpus = int(os.getenv("MDIO__IMPORT__CPU_COUNT", default_cpus))
    num_workers = min(n_blocks, num_cpus)

    tqdm_kw = {"unit": "block", "dynamic_ncols": True}
    # For Unix async writes with s3fs/fsspec & multiprocessing, use 'spawn' instead of default
    # 'fork' to avoid deadlocks on cloud stores. Slower but necessary. Default on Windows.
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(num_workers, mp_context=context) as executor:
        lazy_work = executor.map(
            header_scan_worker,
            repeat(segy_file_kwargs),
            trace_ranges,
            repeat(subset),
        )

        if progress_bar is True:
            desc = "Scanning SEG-Y & calculating checksum"
            lazy_work = tqdm(
                iterable=lazy_work,
                total=n_blocks,
                desc=desc,
                **tqdm_kw,
            )

        # This executes the lazy work.
        results = list(lazy_work)

    return finalize_distributed_checksum(results, combiner)


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


def create_distributed_crc32c(initial_bytes: bytes, total_length: int) -> DistributedCRC32C:
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
    results: list[tuple[HeaderArray, tuple[int, int, int]]], combiner: DistributedCRC32C
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
