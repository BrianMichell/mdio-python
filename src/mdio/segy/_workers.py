"""Low level workers for parsing and writing SEG-Y to Zarr."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
from segy import SegyFile
from segy.arrays import HeaderArray
from zarr import open_group as zarr_open_group

from mdio.core.config import MDIOSettings
from mdio.segy._shard_writer import ShardStreamWriter
from mdio.segy.file import SegyFileArguments
from mdio.segy.file import SegyFileWrapper

if TYPE_CHECKING:
    from collections.abc import Iterator

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
# One range request is at most 8 MiB of whole traces; a batch of them is fetched concurrently
# and then decoded in place, bounding fetch staging to one batch per worker.
_FETCH_REQUEST_BYTES = 8 * 1024**2
_FETCH_BATCH_REQUESTS = 8
# Time shards encoded and uploaded concurrently per worker; each holds one inner chunk and its
# encoded bytes.
SHARD_WRITE_THREADS = 2


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

    data_array = zarr_group[data_variable_name]
    _worker_state["segy_file"] = SegyFile(**segy_file_kwargs)
    _worker_state["data_array"] = data_array
    _worker_state["header_array"] = zarr_group.get("headers")
    _worker_state["raw_header_array"] = zarr_group.get("raw_headers")
    _worker_state["grid_map"] = grid_map
    _worker_state["shard_writer"] = None
    if data_array.shards is not None:
        _worker_state["shard_writer"] = ShardStreamWriter(data_array, output_path, storage_options)
        _worker_state["write_pool"] = ThreadPoolExecutor(SHARD_WRITE_THREADS)


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


def plan_trace_runs(sorted_traces: np.ndarray, max_traces: int) -> list[tuple[int, int]]:
    """Split ascending trace indexes into runs of consecutive traces.

    Args:
        sorted_traces: Trace indexes in ascending order.
        max_traces: Upper bound on traces per run. Values below 1 are treated as 1.

    Returns:
        ``(first_row, row_count)`` pairs into ``sorted_traces``. Each run is one contiguous
        byte range in the SEG-Y file.
    """
    breaks = np.flatnonzero(np.diff(sorted_traces) != 1) + 1
    bounds = [0, *breaks.tolist(), int(sorted_traces.size)]
    step = max(1, max_traces)
    runs: list[tuple[int, int]] = []
    for run_start, run_stop in zip(bounds[:-1], bounds[1:], strict=True):
        runs.extend((row, min(step, run_stop - row)) for row in range(run_start, run_stop, step))
    return runs


def fetch_traces(
    segy_file: SegyFile,
    trace_indexes: np.ndarray,
    keep_raw_headers: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Fetch and decode traces into one preallocated array, in ascending file order.

    Each byte range is copied straight into the output and decoded in place one batch at a
    time, so peak memory stays near one copy of the traces. `SegyFile.trace` instead joins all
    ranges, copies them into a bytearray, and reorders the result, which keeps several copies
    of a shard-sized column alive at once.

    Args:
        segy_file: Open SEG-Y file.
        trace_indexes: Trace indexes to fetch, in any order.
        keep_raw_headers: Also return the undecoded header bytes.

    Returns:
        ``(order, traces, raw_headers)``. Row ``i`` of ``traces`` and ``raw_headers`` is trace
        ``trace_indexes[order[i]]``. ``raw_headers`` is None unless requested.

    Raises:
        OSError: If the store returns fewer bytes than requested.
    """
    order = np.argsort(trace_indexes, kind="stable")
    sorted_traces = trace_indexes[order].astype(np.int64, copy=False)
    trace_dtype = segy_file.spec.trace.dtype
    trace_offset = int(segy_file.spec.trace.offset)
    itemsize = trace_dtype.itemsize
    pipeline = segy_file.accessors.trace_decode_pipeline

    traces = np.empty(sorted_traces.size, dtype=trace_dtype)
    raw_headers = None
    if keep_raw_headers:
        raw_headers = np.empty(sorted_traces.size, dtype=np.dtype((np.void, segy_file.spec.trace.header.itemsize)))

    # Copy through a byte view: structured assignment skips unnamed header bytes, so raw headers
    # would carry uninitialized padding.
    trace_bytes = traces.view(np.uint8)

    runs = plan_trace_runs(sorted_traces, _FETCH_REQUEST_BYTES // itemsize)
    decoded_dtype = trace_dtype
    for batch_start in range(0, len(runs), _FETCH_BATCH_REQUESTS):
        batch = runs[batch_start : batch_start + _FETCH_BATCH_REQUESTS]
        starts = [trace_offset + int(sorted_traces[row]) * itemsize for row, _ in batch]
        ends = [start + count * itemsize for start, (_, count) in zip(starts, batch, strict=True)]
        blobs = segy_file.fs.cat_ranges([segy_file.url] * len(batch), starts, ends, on_error="raise")
        for (row, count), blob in zip(batch, blobs, strict=True):
            if len(blob) != count * itemsize:
                msg = f"Short read from {segy_file.url}: expected {count * itemsize} bytes, got {len(blob)}."
                raise OSError(msg)
            rows = slice(row, row + count)
            trace_bytes[row * itemsize : (row + count) * itemsize] = np.frombuffer(blob, dtype=np.uint8)
            if raw_headers is not None:
                raw_headers[rows] = traces[rows]["header"].view(raw_headers.dtype)
            decoded = pipeline.apply(traces[rows])
            decoded_dtype = decoded.dtype
            if not np.shares_memory(decoded, traces):
                traces.view(decoded_dtype)[rows] = decoded
        del blobs
    return order, traces.view(decoded_dtype), raw_headers


def merge_stats(total: SummaryStatistics | None, partial: SummaryStatistics | None) -> SummaryStatistics | None:
    """Combine two partial summaries. None means no counted samples.

    Args:
        total: Running summary; updated in place when not None.
        partial: Summary to fold in.

    Returns:
        The combined summary.
    """
    if partial is None:
        return total
    if total is None:
        return partial
    total.count += partial.count
    total.sum += partial.sum
    total.sum_squares += partial.sum_squares
    total.min = min(total.min, partial.min)
    total.max = max(total.max, partial.max)
    return total


def _sub_blocks(shape: tuple[int, ...], block: tuple[int, ...]) -> Iterator[tuple[slice, ...]]:
    """Yield slices tiling ``shape`` with ``block``; edge blocks are clipped."""
    starts = [range(0, size, step) for size, step in zip(shape, block, strict=True)]
    for origin in product(*starts):
        yield tuple(slice(o, min(o + b, size)) for o, b, size in zip(origin, block, shape, strict=True))


class _ArraySink:
    """Write a trace column through Zarr as one region assignment."""

    def __init__(self, data_array: zarr_Array, region_slices: tuple[slice, ...]) -> None:
        self._data_array = data_array
        self._region_slices = region_slices

    def write(self, block: tuple[slice, ...], live: tuple[np.ndarray, ...], samples: np.ndarray) -> None:
        sample_slice = self._region_slices[-1]
        shape = tuple(b.stop - b.start for b in block)
        tmp_samples = np.full((*shape, sample_slice.stop - sample_slice.start), self._data_array.fill_value)
        tmp_samples[live] = samples
        target = tuple(
            slice(r.start + b.start, r.start + b.stop) for r, b in zip(self._region_slices, block, strict=False)
        )
        self._data_array[(*target, sample_slice)] = tmp_samples

    def close(self) -> None:
        """Nothing is buffered."""

    def discard(self) -> None:
        """Nothing is buffered."""


class _ShardSink:
    """Stream the inner chunks of one spatial shard column into its time shards.

    Blocks must be inner-chunk aligned within the shard. Each time shard receives its chunks in
    block order, and different time shards are encoded and uploaded concurrently.
    """

    def __init__(
        self,
        writer: ShardStreamWriter,
        spatial_slices: tuple[slice, ...],
        n_samples: int,
        pool: ThreadPoolExecutor,
    ) -> None:
        self._writer = writer
        self._pool = pool
        self._n_samples = n_samples
        shard_spatial = tuple(s.start // d for s, d in zip(spatial_slices, writer.shard_shape[:-1], strict=True))
        n_time_shards = -(-n_samples // writer.shard_shape[-1])
        self._streams = [writer.open((*shard_spatial, t)) for t in range(n_time_shards)]

    def write(self, block: tuple[slice, ...], live: tuple[np.ndarray, ...], samples: np.ndarray) -> None:
        writer = self._writer
        chunk_spatial = writer.chunk_shape[:-1]
        chunk_depth = writer.chunk_shape[-1]
        local = tuple(b.start // c for b, c in zip(block, chunk_spatial, strict=True))

        def write_time_shard(shard_index: int) -> None:
            for depth_index in range(writer.chunks_per_shard[-1]):
                start = shard_index * writer.shard_shape[-1] + depth_index * chunk_depth
                if start >= self._n_samples:
                    return
                width = min(chunk_depth, self._n_samples - start)
                chunk = np.full(writer.chunk_shape, writer.fill_value, dtype=writer.dtype)
                chunk[(*live, slice(0, width))] = samples[:, start : start + width]
                self._streams[shard_index].append((*local, depth_index), writer.encode(chunk))

        list(self._pool.map(write_time_shard, range(len(self._streams))))

    def close(self) -> None:
        list(self._pool.map(lambda stream: stream.close(), self._streams))

    def discard(self) -> None:
        for stream in self._streams:
            stream.discard()


def trace_worker(region: dict[str, slice]) -> SummaryStatistics | None:  # noqa: PLR0915
    """Writes a subset of traces from a region of the dataset of Zarr file.

    Reads its shared inputs (SEG-Y handle, Zarr arrays, grid map) from the per-process state set up
    by `trace_worker_init`, so only the lightweight `region` is pickled per block. A sharded region
    is read one inner-chunk column at a time and streamed into its shard objects, so a worker never
    holds the whole shard column.

    Args:
        region: Region of the dataset to write to.

    Returns:
        SummaryStatistics object containing statistics about the written traces.

    Raises:
        BaseException: Any read or write failure, re-raised after partial shards are discarded.
    """
    segy_file: SegyFile = _worker_state["segy_file"]
    data_array: zarr_Array = _worker_state["data_array"]
    header_array: zarr_Array | None = _worker_state["header_array"]
    raw_header_array: zarr_Array | None = _worker_state["raw_header_array"]
    grid_map: zarr_Array = _worker_state["grid_map"]
    shard_writer: ShardStreamWriter | None = _worker_state["shard_writer"]

    region_slices = tuple(region.values())
    spatial_slices = region_slices[:-1]  # minus last (vertical) axis
    local_grid_map = grid_map[spatial_slices]

    # The dtype.max is the sentinel value for the grid map.
    # Normally, this is uint32, but some grids need to be promoted to uint64.
    not_null = local_grid_map != fill_value_map.get(local_grid_map.dtype.name)
    if not not_null.any():
        return None

    headers = None if header_array is None else np.full(local_grid_map.shape, header_array.fill_value)
    # NOTE: Raw headers are not intended to remain a feature of SEG-Y ingestion; remove them in
    # full once they are dropped.
    raw_headers = None if raw_header_array is None else np.full(local_grid_map.shape, raw_header_array.fill_value)

    if shard_writer is None:
        sink: _ArraySink | _ShardSink = _ArraySink(data_array, region_slices)
        block_shape = local_grid_map.shape
    else:
        sink = _ShardSink(shard_writer, spatial_slices, data_array.shape[-1], _worker_state["write_pool"])
        block_shape = shard_writer.chunk_shape[:-1]

    timing = os.environ.get("MDIO__IMPORT__SHARD_TIMING") == "1"
    fetch_time = stats_time = write_time = 0.0
    stats: SummaryStatistics | None = None
    try:
        for block in _sub_blocks(local_grid_map.shape, block_shape):
            live_mask = not_null[block]
            if not live_mask.any():
                continue
            started = time.perf_counter()
            order, traces, block_raw_headers = fetch_traces(
                segy_file,
                local_grid_map[block][live_mask],
                keep_raw_headers=raw_headers is not None,
            )
            live = tuple(axis[order] for axis in np.nonzero(live_mask))
            fetched = time.perf_counter()
            if headers is not None:
                headers[block][live] = traces["header"]
            if raw_headers is not None:
                raw_headers[block][live] = block_raw_headers
            samples = traces["data"]
            stats = merge_stats(stats, summarize_samples(samples))
            summarized = time.perf_counter()
            sink.write(block, live, samples)
            del traces, samples, block_raw_headers
            fetch_time += fetched - started
            stats_time += summarized - fetched
            write_time += time.perf_counter() - summarized
        started = time.perf_counter()
        sink.close()
        write_time += time.perf_counter() - started
    except BaseException:
        sink.discard()
        raise

    started = time.perf_counter()
    if headers is not None:
        header_array[spatial_slices] = headers
    if raw_headers is not None:
        raw_header_array[spatial_slices] = raw_headers
    header_time = time.perf_counter() - started

    if timing:
        print(
            f"SHARD_TIME fetch_decode={fetch_time:.2f} headers={header_time:.2f} "
            f"write={write_time:.2f} stats={stats_time:.2f}",
            flush=True,
        )
    return stats
