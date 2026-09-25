"""Low level workers for parsing and writing SEG-Y to Zarr."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from segy import SegyFile
from segy.arrays import HeaderArray
from zarr import open_group as zarr_open_group

from mdio.core.config import MDIOSettings
from mdio.segy._raw_trace_wrapper import SegyFileRawTraceWrapper
from mdio.segy.file import SegyFileArguments
from mdio.segy.file import SegyFileWrapper

if TYPE_CHECKING:
    from zarr import Array as zarr_Array

from zarr.core.config import config as zarr_config

from mdio.builder.schemas.v1.stats import CenteredBinHistogram
from mdio.builder.schemas.v1.stats import SummaryStatistics
from mdio.constants import fill_value_map

logger = logging.getLogger(__name__)

# `np.ma.masked_values(samples, 0)` is `isclose` with atol 1e-8, so |x| <= 1e-8 is excluded.
_MASKED_ZERO_ATOL = 1e-8
# Stats walk the sample axis in shard-depth slabs. A full-column masked array plus `np.ma.power`
# keeps another copy of the column resident in every worker.
_STATS_SLAB_SAMPLES = 256


def header_scan_worker(
    segy_file_kwargs: SegyFileArguments,
    trace_range: tuple[int, int],
    subset: tuple[str, ...] | None = None,
) -> HeaderArray:
    """Header scan worker.

    If SegyFile is not open, it can either accept a path string or a handle that was opened in
    a different context manager.

    Args:
        segy_file_kwargs: Arguments to open SegyFile instance.
        trace_range: Tuple consisting of the trace ranges to read.
        subset: Tuple of header names to filter and keep.

    Returns:
        HeaderArray parsed from SEG-Y library.
    """
    settings = MDIOSettings()

    segy_file = SegyFileWrapper(**segy_file_kwargs)

    slice_ = slice(*trace_range)

    trace_header = segy_file.trace[slice_].header if settings.cloud_native else segy_file.header[slice_]

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

    return HeaderArray(trace_header)  # wrap back so we can use aliases


# Per-worker process state populated once by `trace_worker_init`. Keeping the SEG-Y handle,
# Zarr array handles, and the (compressed, in-memory) grid map here lets us pickle them a single
# time per worker via the pool initializer instead of once per submitted block. The grid map is
# retained as a compressed in-memory Zarr array and sliced lazily per region, so each worker only
# materializes its own block rather than the full dense map.
_worker_state: dict[str, object] = {}


def trace_worker_init(  # noqa: PLR0913, PLR0917
    segy_file_kwargs: SegyFileArguments,
    output_path: str,
    storage_options: dict[str, object] | None,
    use_consolidated: bool,
    data_variable_name: str,
    grid_map: zarr_Array,
    sample_slab: int,
) -> None:
    """Initialize per-process state for trace ingestion workers.

    Used as the `ProcessPoolExecutor` initializer so the SEG-Y file, Zarr output handles, and grid
    map are opened/transferred once per worker process rather than re-pickled for every block.

    Args:
        segy_file_kwargs: Arguments to open the SegyFile instance.
        output_path: POSIX path to the output MDIO Zarr store.
        storage_options: fsspec storage options for the output store.
        use_consolidated: Whether to open the group with consolidated metadata (Zarr V2).
        data_variable_name: Name of the data variable in the dataset.
        grid_map: Compressed in-memory Zarr array mapping live traces to their positions.
        sample_slab: Sample-axis length of one storage write. Sharded ingestion uses the shard
            depth so each write is one complete shard object.
    """
    # Keep Zarr thread use explicit so worker processes cannot oversubscribe the host.
    settings = MDIOSettings()
    zarr_config.set({"threading.max_workers": settings.import_zarr_threads})

    zarr_group = zarr_open_group(
        output_path,
        mode="r+",
        storage_options=storage_options,
        use_consolidated=use_consolidated,
    )

    _worker_state["segy_file"] = SegyFile(**segy_file_kwargs)
    _worker_state["data_array"] = zarr_group[data_variable_name]
    _worker_state["header_array"] = zarr_group.get("headers")
    _worker_state["raw_header_array"] = zarr_group.get("raw_headers")
    _worker_state["grid_map"] = grid_map
    _worker_state["sample_slab"] = sample_slab


def summarize_samples(samples: np.ndarray, slab: int = _STATS_SLAB_SAMPLES) -> SummaryStatistics | None:
    """Summarize samples, excluding values with absolute value <= 1e-8.

    Partial sums are accumulated per slab in float64. The same function serves sharded and
    traditional ingestion, and neither path builds a masked array of the whole column.

    Args:
        samples: Decoded samples shaped ``(n_traces, n_samples)``.
        slab: Sample-axis length of one statistics pass. Values below 1 use the whole axis.

    Returns:
        Summary statistics, or None when every value is within 1e-8 of zero.
    """
    count = 0
    total = 0.0
    sum_squares = 0.0
    min_value: float | None = None
    max_value: float | None = None
    n_samples = int(samples.shape[-1])
    step = n_samples if slab < 1 else slab
    for offset in range(0, n_samples, step):
        values = samples[..., offset : offset + step]
        kept = values[np.abs(values) > _MASKED_ZERO_ATOL]
        if kept.size == 0:
            continue
        count += int(kept.size)
        total += float(kept.sum(dtype=np.float64))
        # Not `np.dot`: BLAS ddot wakes a per-process OpenBLAS pool that spin-waits after each call.
        sum_squares += float(np.square(kept, dtype=np.float64).sum())
        slab_min = float(kept.min())
        slab_max = float(kept.max())
        min_value = slab_min if min_value is None else min(min_value, slab_min)
        max_value = slab_max if max_value is None else max(max_value, slab_max)
    if count == 0 or min_value is None or max_value is None:
        return None
    histogram = CenteredBinHistogram(bin_centers=[], counts=[])
    return SummaryStatistics(
        count=count,
        min=min_value,
        max=max_value,
        sum=total,
        sum_squares=sum_squares,
        histogram=histogram,
    )


def _write_sample_slabs(  # noqa: PLR0913
    data_array: zarr_Array,
    region_slices: tuple[slice, ...],
    spatial_shape: tuple[int, ...],
    not_null: np.ndarray,
    samples: np.ndarray,
    sample_slab: int,
) -> None:
    """Write samples in storage-sized slabs along the last axis.

    A slab equal to the shard depth makes each assignment one complete shard object. The full
    column is never materialized as a second array.

    Args:
        data_array: Destination Zarr array.
        region_slices: Region slices, including the sample axis.
        spatial_shape: Spatial shape of the region, without samples.
        not_null: Mask of live traces inside the spatial region.
        samples: Decoded samples for the live traces, shaped ``(n_live, n_samples)``.
        sample_slab: Number of samples per write. Values below 1 write the whole axis.
    """
    sample_slice = region_slices[-1]
    spatial_slices = region_slices[:-1]
    sample_count = sample_slice.stop - sample_slice.start
    slab = sample_count if sample_slab < 1 else sample_slab
    spans: list[tuple[int, int, int]] = []
    offset = 0
    for start in range(sample_slice.start, sample_slice.stop, slab):
        stop = min(start + slab, sample_slice.stop)
        spans.append((start, stop, offset))
        offset += stop - start

    def write_span(span: tuple[int, int, int]) -> None:
        """Write one sample slab. Separate time slabs are separate shard objects."""
        start, stop, sample_offset = span
        width = stop - start
        tmp_samples = np.full((*spatial_shape, width), data_array.fill_value)
        tmp_samples[not_null] = samples[..., sample_offset : sample_offset + width]
        data_array[(*spatial_slices, slice(start, stop))] = tmp_samples

    # The sync shard codec compresses inner chunks on one thread. Overlapping two
    # time-slab writes uses a second core without a second copy of the trace column.
    if len(spans) <= 1:
        for span in spans:
            write_span(span)
        return
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(write_span, spans))


def trace_worker(region: dict[str, slice]) -> SummaryStatistics | None:
    """Writes a subset of traces from a region of the dataset of Zarr file.

    Reads its shared inputs (SEG-Y handle, Zarr arrays, grid map) from the per-process state set up
    by `trace_worker_init`, so only the lightweight `region` is pickled per block.

    Args:
        region: Region of the dataset to write to.

    Returns:
        SummaryStatistics object containing statistics about the written traces.
    """
    segy_file: SegyFile = _worker_state["segy_file"]
    data_array: zarr_Array = _worker_state["data_array"]
    header_array: zarr_Array | None = _worker_state["header_array"]
    raw_header_array: zarr_Array | None = _worker_state["raw_header_array"]
    grid_map: zarr_Array = _worker_state["grid_map"]

    region_slices = tuple(region.values())
    local_grid_map = grid_map[region_slices[:-1]]  # minus last (vertical) axis

    # The dtype.max is the sentinel value for the grid map.
    # Normally, this is uint32, but some grids need to be promoted to uint64.
    not_null = local_grid_map != fill_value_map.get(local_grid_map.dtype.name)
    if not not_null.any():
        return None

    live_trace_indexes = local_grid_map[not_null].tolist()

    # Raw headers are not intended to remain as a feature of the SEGY ingestion.
    # For that reason, we have wrapped the accessors to provide an interface that can be removed
    # and not require additional changes to the below code.
    # NOTE: The `raw_header_key` code block should be removed in full as it will become dead code.
    timing = os.environ.get("MDIO__IMPORT__SHARD_TIMING") == "1"
    started = time.perf_counter()
    traces = SegyFileRawTraceWrapper(
        segy_file,
        live_trace_indexes,
        keep_raw=raw_header_array is not None,
    )
    fetched = time.perf_counter()
    if timing and not _worker_state.get("logged_writeable"):
        print(f"FETCH_WRITEABLE writeable={traces.fetched_writeable}", flush=True)
        _worker_state["logged_writeable"] = True

    # Compute slices once (headers exclude sample dimension)
    header_region_slices = region_slices[:-1]  # Exclude sample dimension
    header_shape = tuple(s.stop - s.start for s in header_region_slices)

    # Write raw headers if array was provided
    # Headers only have spatial dimensions (no sample dimension)
    if raw_header_array is not None:
        tmp_raw_headers = np.full(header_shape, raw_header_array.fill_value)
        tmp_raw_headers[not_null] = traces.raw_header
        raw_header_array[header_region_slices] = tmp_raw_headers

    # Write headers if array was provided
    # Headers only have spatial dimensions (no sample dimension)
    if header_array is not None:
        tmp_headers = np.full(header_shape, header_array.fill_value)
        tmp_headers[not_null] = traces.header
        header_array[header_region_slices] = tmp_headers

    # Write the data variable. Read samples once; the wrapper may otherwise fetch them twice.
    # Assign one storage slab at a time so a shard writer does not also retain a full-column copy.
    samples = traces.sample
    decoded = time.perf_counter()
    _write_sample_slabs(
        data_array=data_array,
        region_slices=region_slices,
        spatial_shape=header_shape,
        not_null=not_null,
        samples=samples,
        sample_slab=int(_worker_state["sample_slab"]),
    )
    written = time.perf_counter()

    stats = summarize_samples(samples)
    if timing:
        finished = time.perf_counter()
        print(
            f"SHARD_TIME fetch={fetched - started:.2f} header_decode={decoded - fetched:.2f} "
            f"write={written - decoded:.2f} stats={finished - written:.2f}",
            flush=True,
        )
    return stats
