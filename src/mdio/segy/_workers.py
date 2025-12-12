"""Low level workers for parsing and writing SEG-Y to Zarr."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mdio_cpp
import numpy as np
from segy.arrays import HeaderArray

from mdio.core.config import MDIOSettings
from mdio.segy._raw_trace_wrapper import SegyFileRawTraceWrapper
from mdio.segy.file import SegyFileArguments
from mdio.segy.file import SegyFileWrapper

if TYPE_CHECKING:
    from segy import SegyFile
    from zarr import Array as zarr_Array

from zarr.core.config import config as zarr_config

from mdio.builder.schemas.v1.stats import CenteredBinHistogram
from mdio.builder.schemas.v1.stats import SummaryStatistics
from mdio.constants import fill_value_map

logger = logging.getLogger(__name__)


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


def trace_worker(  # noqa: PLR0913
    segy_file: SegyFile,
    ds: mdio_cpp.Dataset,
    # data_array: mdio_cpp.Variable,
    # header_array: mdio_cpp.Variable | None,
    # raw_header_array: mdio_cpp.Variable | None,
    region: dict[str, slice],
    grid_map: zarr_Array,
) -> SummaryStatistics | None:
    """Writes a subset of traces from a region of the dataset of Zarr file.

    Args:
        segy_file: The opened SEG-Y file.
        data_array: Zarr array for writing trace data.
        header_array: Zarr array for writing trace headers (or None if not needed).
        raw_header_array: Zarr array for writing raw headers (or None if not needed).
        region: Region of the dataset to write to.
        grid_map: Zarr array mapping live traces to their positions in the dataset.

    Returns:
        SummaryStatistics object containing statistics about the written traces.
    """
    # Setting the zarr config to 1 thread to ensure we honor the `MDIO__IMPORT__CPU_COUNT` environment variable.
    # The Zarr 3 engine utilizes multiple threads. This can lead to resource contention and unpredictable memory usage.
    zarr_config.set({"threading.max_workers": 1})

    region_slices = tuple(region.values())
    local_grid_map = grid_map[region_slices[:-1]]  # minus last (vertical) axis

    slices = list()
    for key, value in region.items():
        slices.append((key, value.start, value.stop))

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
    traces = SegyFileRawTraceWrapper(segy_file, live_trace_indexes)

    sliced_ds = ds.isel(slices)
    data_array = sliced_ds.get_variable("amplitude")
    header_array = sliced_ds.get_variable("headers")
    raw_header_array = sliced_ds.get_variable("raw_headers")

    # data_array = data_array.slice(slices)
    # header_array = header_array.slice(slices)
    # raw_header_array = raw_header_array.slice(slices)

    # Write raw headers if array was provided
    # Headers only have spatial dimensions (no sample dimension)
    if raw_header_array is not None:
        raw_var = raw_header_array.allocate_data()
        raw_var.data[...][not_null] = (
            np.ascontiguousarray(traces.raw_header).view(np.uint8).reshape(-1, raw_var.data.shape[-1])
        )
        raw_header_array.write_data(raw_var)

    if header_array is not None:
        header_var = header_array.allocate_data()

        # Flatten spatial mask and target buffer
        mask = not_null.ravel()
        header_store = header_var.data.reshape(-1, header_var.data.shape[-1])  # (N_total, 216)

        # Convert structured headers to contiguous bytes and trim 240 -> 216
        header_bytes = (
            np.ascontiguousarray(traces.header).view(np.uint8).reshape(mask.sum(), -1)[:, : header_store.shape[-1]]
        )

        # Masked assignment then write back
        header_store[mask] = header_bytes
        header_array.write_data(header_var)

    # Write the data variable
    data_var = data_array.allocate_data()
    data_var.data[...][not_null] = traces.sample
    data_array.write_data(data_var)

    nonzero_samples = np.ma.masked_values(traces.sample, 0, copy=False)

    nonzero_count = nonzero_samples.count()
    if nonzero_count == 0:
        # Return None to avoid calculating a NaN in sum_squares
        return None

    histogram = CenteredBinHistogram(bin_centers=[], counts=[])
    return SummaryStatistics(
        count=nonzero_count,
        min=nonzero_samples.min(),
        max=nonzero_samples.max(),
        sum=nonzero_samples.sum(dtype="float64"),
        sum_squares=(np.ma.power(nonzero_samples, 2).sum(dtype="float64")),
        histogram=histogram,
    )
