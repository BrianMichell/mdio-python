"""Parsers for sections of SEG-Y files."""

from __future__ import annotations

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
from tqdm.auto import tqdm
from upath import UPath

# All imports from the `checksum` module are slated to be removed in the future
from mdio.segy.checksum import create_distributed_crc32c
from mdio.segy.checksum import finalize_distributed_checksum
from mdio.segy.checksum import is_checksum_available
from mdio.segy.checksum import should_calculate_checksum

if TYPE_CHECKING:
    from segy.arrays import HeaderArray

    from mdio.segy.file import SegyFileArguments

default_cpus = cpu_count(logical=True)


def parse_headers(  # noqa: PLR0913
    segy_file_kwargs: SegyFileArguments,
    num_traces: int,
    subset: list[str] | None = None,
    block_size: int = 10000,
    progress_bar: bool = True,
    calculate_checksum: bool = True,
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
    if should_calculate_checksum() and is_checksum_available():
        from mdio.segy.checksum import header_scan_worker
    else:
        from mdio.segy._workers import header_scan_worker

    # Type hint for the dynamically imported function
    header_scan_worker: Any
    # Initialize combiner only if checksum calculation is needed and available
    combiner: Any = None
    if calculate_checksum:
        if not is_checksum_available():
            # Checksum was requested but libraries not available - disable it
            calculate_checksum = False
        else:
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
            desc = (
                "Scanning SEG-Y & calculating checksum"
                if calculate_checksum
                else "Scanning SEG-Y for geometry attributes"
            )
            lazy_work = tqdm(
                iterable=lazy_work,
                total=n_blocks,
                desc=desc,
                **tqdm_kw,
            )

        # This executes the lazy work.
        results = list(lazy_work)

    if not calculate_checksum:
        # Merge headers and return
        return np.concatenate(results)

    return finalize_distributed_checksum(results, combiner)

# def _parse_headers(
#     segy_file_kwargs: SegyFileArguments,
#     num_traces: int,
#     subset: list[str] | None = None,
#     block_size: int = 10000,
#     progress_bar: bool = True,
# ) -> HeaderArray:
#     """Read and parse given `byte_locations` from SEG-Y file.

#     Args:
#         segy_file_kwargs: SEG-Y file arguments.
#         num_traces: Total number of traces in the SEG-Y file.
#         subset: List of header names to filter and keep.
#         block_size: Number of traces to read for each block.
#         progress_bar: Enable or disable progress bar. Default is True.

#     Returns:
#         HeaderArray. Keys are the index names, values are numpy arrays of parsed headers for the
#         current block. Array is of type byte_type except IBM32 which is mapped to FLOAT32.
#     """
#     trace_count = num_traces
#     n_blocks = int(ceil(trace_count / block_size))

#     trace_ranges = []
#     for idx in range(n_blocks):
#         start, stop = idx * block_size, (idx + 1) * block_size
#         stop = min(stop, trace_count)

#         trace_ranges.append((start, stop))

#     num_cpus = int(os.getenv("MDIO__IMPORT__CPU_COUNT", default_cpus))
#     num_workers = min(n_blocks, num_cpus)

#     tqdm_kw = {"unit": "block", "dynamic_ncols": True}
#     # For Unix async writes with s3fs/fsspec & multiprocessing, use 'spawn' instead of default
#     # 'fork' to avoid deadlocks on cloud stores. Slower but necessary. Default on Windows.
#     context = mp.get_context("spawn")
#     with ProcessPoolExecutor(num_workers, mp_context=context) as executor:
#         lazy_work = executor.map(header_scan_worker, repeat(segy_file_kwargs), trace_ranges, repeat(subset))

#         if progress_bar is True:
#             lazy_work = tqdm(
#                 iterable=lazy_work,
#                 total=n_blocks,
#                 desc="Scanning SEG-Y for geometry attributes",
#                 **tqdm_kw,
#             )

#         # This executes the lazy work.
#         headers: list[HeaderArray] = list(lazy_work)

#     # Merge blocks before return
#     return np.concatenate(headers)