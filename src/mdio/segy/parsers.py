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
    from segy import SegyFile
    from segy.arrays import HeaderArray

default_cpus = cpu_count(logical=True)


def parse_index_headers(
    segy_file: SegyFile,
    block_size: int = 10000,
    progress_bar: bool = True,
) -> HeaderArray:
    """Read and parse given `byte_locations` from SEG-Y file.

    Args:
        segy_file: SegyFile instance.
        block_size: Number of traces to read for each block.
        progress_bar: Enable or disable progress bar. Default is True.

    Returns:
        HeaderArray. Keys are the index names, values are numpy arrays of parsed headers for the
        current block. Array is of type byte_type except IBM32 which is mapped to FLOAT32.
    """
    trace_count = segy_file.num_traces
    print(f"trace_count: {trace_count}")
    n_blocks = int(ceil(trace_count / block_size))
    print(f"n_blocks: {n_blocks}")

    trace_ranges = []
    for idx in range(n_blocks):
        start, stop = idx * block_size, (idx + 1) * block_size
        stop = min(stop, trace_count)

        trace_ranges.append((start, stop))

    # For Unix async reads with s3fs/fsspec & multiprocessing, use 'spawn' instead of default
    # 'fork' to avoid deadlocks on cloud stores. Slower but necessary. Default on Windows.
    num_cpus = int(os.getenv("MDIO__IMPORT__CPU_COUNT", default_cpus))
    num_workers = min(n_blocks, num_cpus)
    context = mp.get_context("spawn")

    tqdm_kw = {"unit": "block", "dynamic_ncols": True}
    with ProcessPoolExecutor(num_workers, mp_context=context) as executor:
        lazy_work = executor.map(header_scan_worker, repeat(segy_file), trace_ranges)

        if progress_bar is True:
            lazy_work = tqdm(
                iterable=lazy_work,
                total=n_blocks,
                desc="Scanning SEG-Y for geometry attributes",
                **tqdm_kw,
            )

        # This executes the lazy work.
        headers: list[HeaderArray] = list(lazy_work)

    print("Concatenating headers...", flush=True)
    # raise Exception("Stop here")
    # ret = memory_efficient_concatenate(headers)
    ret = np.concatenate(headers)
    print("Finished!", flush=True)
    # Merge blocks before return
    return ret


def memory_efficient_concatenate(headers: list[HeaderArray]) -> HeaderArray:
    """Memory-efficient concatenation for many small header arrays.
    
    Pre-allocates the target array and copies data in place to avoid
    the memory fragmentation and intermediate allocations that occur
    with np.concatenate on many small arrays.
    
    Args:
        headers: List of HeaderArray objects to concatenate
        
    Returns:
        Single concatenated HeaderArray
    """

    # Heartbeat 1: Function entry
    with open("heartbeat_1_entry.txt", "w") as f:
        f.write("Entered memory_efficient_concatenate\n")
        f.flush()

    if not headers:
        raise ValueError("Cannot concatenate empty list of arrays")
    
    # Heartbeat 2: Before size calculation
    with open("heartbeat_2_calculating_size.txt", "w") as f:
        f.write(f"Starting size calculation for {len(headers)} arrays\n")
        f.flush()
    
    # Calculate total size and get array metadata
    total_length = sum(len(arr) for arr in headers)
    first_array = headers[0]
    target_dtype = first_array.dtype
    
    # Heartbeat 3: Before allocation
    estimated_size_mb = total_length * target_dtype.itemsize / 1024**2
    with open("heartbeat_3_before_allocation.txt", "w") as f:
        f.write(f"About to allocate {total_length:,} elements, estimated {estimated_size_mb:.1f} MB\n")
        f.flush()
    
    print(f"Pre-allocating result array: {total_length:,} elements, "
          f"dtype={target_dtype}, estimated size={estimated_size_mb:.1f} MB")
    
    # Pre-allocate the final array - this is the key optimization
    result = np.empty(total_length, dtype=target_dtype)
    
    # Heartbeat 4: After allocation
    actual_size_mb = result.nbytes / 1024**2
    with open("heartbeat_4_allocated.txt", "w") as f:
        f.write(f"Successfully allocated array, actual size {actual_size_mb:.1f} MB\n")
        f.flush()
    
    # Copy arrays sequentially into pre-allocated space
    current_pos = 0
    batch_size = 100  # Process in batches to provide progress feedback
    
    # Heartbeat 5: Before copying loop
    with open("heartbeat_5_start_copying.txt", "w") as f:
        f.write("Starting copy loop\n")
        f.flush()
    
    for i in range(0, len(headers), batch_size):
        batch_end = min(i + batch_size, len(headers))
        
        # Process this batch
        for j in range(i, batch_end):
            arr = headers[j]
            if arr is None:  # Skip if already processed
                continue
                
            end_pos = current_pos + len(arr)
            
            # Direct copy into pre-allocated space - no intermediate allocations
            result[current_pos:end_pos] = arr
            current_pos = end_pos
            
            # Help garbage collector by clearing reference
            headers[j] = None
        
        # Progress update and heartbeat for major milestones
        if batch_end % (5 * batch_size) == 0 or batch_end == len(headers):
            progress_pct = (batch_end / len(headers)) * 100
            with open(f"heartbeat_6_progress_{int(progress_pct)}.txt", "w") as f:
                f.write(f"Progress: {progress_pct:.1f}% ({batch_end:,}/{len(headers):,})\n")
                f.flush()
            print(f"Concatenation progress: {progress_pct:.1f}% ({batch_end:,}/{len(headers):,} arrays)")
    
    # Heartbeat 7: Completion
    with open("heartbeat_7_complete.txt", "w") as f:
        f.write(f"Concatenation complete. Final size: {result.nbytes / 1024**2:.1f} MB\n")
        f.flush()
    
    print(f"Concatenation complete. Final array size: {result.nbytes / 1024**2:.1f} MB")
    return result
