"""Parsers for sections of SEG-Y files."""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from math import ceil
from typing import TYPE_CHECKING

import numpy as np
from psutil import cpu_count
from tqdm.auto import tqdm

from mdio.segy._workers import header_scan_worker

if TYPE_CHECKING:
    from segy.arrays import HeaderArray

    from mdio.segy._workers import SegyFileArguments

default_cpus = cpu_count(logical=True)


def parse_headers(
    segy_file_kwargs: SegyFileArguments,
    num_traces: int,
    subset: list[str] | None = None,
    block_size: int = 10000,
    progress_bar: bool = True,
    calculate_checksum: bool = False,
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
            repeat(calculate_checksum),
        )

        if progress_bar is True:
            desc = "Scanning SEG-Y & calculating checksum" if calculate_checksum else "Scanning SEG-Y for geometry attributes"
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
    
    # Separate headers and checksums
    headers: list[HeaderArray] = [r[0] for r in results]
    checksum_parts: list[tuple[int, int, int]] = [r[1] for r in results]
    
    # Combine checksums
    from mdio.segy.blocked_io import _combine_crc32c_checksums
    
    combined_crc = _combine_crc32c_checksums(checksum_parts)
    
    # Merge headers and return with checksum
    return np.concatenate(headers), combined_crc
