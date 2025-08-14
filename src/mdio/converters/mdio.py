"""Conversion from to MDIO various other formats."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from psutil import cpu_count
from tqdm.dask import TqdmCallback

from mdio.segy.blocked_io import to_segy
from mdio.segy.creation import concat_files
from mdio.segy.creation import get_required_segy_fields
from mdio.segy.creation import mdio_spec_to_segy
from mdio.segy.utilities import segy_export_rechunker

if TYPE_CHECKING:
    from mdio.core.storage_location import StorageLocation

try:
    import distributed
except ImportError:
    distributed = None


default_cpus = cpu_count(logical=True)
NUM_CPUS = int(os.getenv("MDIO__EXPORT__CPU_COUNT", default_cpus))


def mdio_to_segy(  # noqa: PLR0912, PLR0913, PLR0915
    input_location: StorageLocation,
    output_location: StorageLocation,
    *,
    endian: str = "big",
    new_chunks: tuple[int, ...] = None,
    selection_mask: np.ndarray = None,
    client: distributed.Client = None,
    overwrite: bool = False,
) -> None:
    """Convert MDIO file to SEG-Y format.

    We export N-D seismic data to the flattened SEG-Y format used in data transmission.

    The input headers are preserved as is, and will be transferred to the output file.

    Input MDIO can be local or cloud based. However, the output SEG-Y will be generated locally.

    A `selection_mask` can be provided (same shape as spatial grid) to export a subset.

    Args:
        input_location: Location of the input MDIO file.
        output_location: Location of the output SEG-Y file.
        endian: Endianness of the output SEG-Y. Rev.2 allows little endian. Default is "big".
        new_chunks: Set manual chunksize. For development purposes only.
        selection_mask: Array that lists the subset of traces.
        client: Dask client. If `None` we will use local threaded scheduler. If `auto` is used we
            will create multiple processes (with 8 threads each).
        overwrite: Whether to overwrite the SEG-Y file if it already exists.

    Raises:
        FileExistsError: If the output location already exists and `overwrite` is False.
        ImportError: If distributed package isn't installed but requested.
        ValueError: If cut mask is empty, i.e. no traces will be written.

    Examples:
        To export an existing local MDIO file to SEG-Y we use the code snippet below. This will
        export the full MDIO (without padding) to SEG-Y format using IBM floats and big-endian
        byte order.

        >>> from mdio import mdio_to_segy
        >>>
        >>> mdio_to_segy(
        ...     input_location=StorageLocation("prefix2/file.mdio"),
        ...     output_location=StorageLocation("prefix/file.segy"),
        ... )

        If we want to export this as an IEEE big-endian, using a selection mask, we would run:

        >>> mdio_to_segy(
        ...     input_location=StorageLocation("prefix2/file.mdio"),
        ...     output_location=StorageLocation("prefix/file.segy"),
        ...     selection_mask=boolean_mask,
        ... )

    """
    backend = "dask"

    if not overwrite and output_location.exists():
        err = f"Output location '{output_location.uri}' exists. Set `overwrite=True` if intended."
        raise FileExistsError(err)

    output_segy_path = Path(output_location.uri)

    if new_chunks is None:
        ds_tmp = xr.open_dataset(input_location.uri, engine="zarr", mask_and_scale=False)
        amp = ds_tmp["amplitude"]
        chunks = amp.encoding.get("chunks")
        shape = amp.shape
        dtype = amp.dtype
        new_chunks = segy_export_rechunker(chunks, shape, dtype)
        ds_tmp.close()

    creation_args = [
        input_location,
        output_location,
        endian,
        new_chunks,
        backend,
    ]

    if client is not None:
        if distributed is not None:
            # This is in case we work with big data
            feature = client.submit(mdio_spec_to_segy, *creation_args)
            ds, segy_factory = feature.result()
        else:
            msg = "Distributed client was provided, but `distributed` is not installed"
            raise ImportError(msg)
    else:
        ds, segy_factory = mdio_spec_to_segy(*creation_args)

    amp_da, headers_da, trace_mask_da, _, _ = get_required_segy_fields(ds)

    live_mask = trace_mask_da.data.compute()

    if selection_mask is not None:
        live_mask = live_mask & selection_mask

    # This handles the case if we are skipping a whole block.
    if live_mask.sum() == 0:
        msg = "No traces will be written out. Live mask is empty."
        raise ValueError(msg)

    # Find rough dim limits, so we don't unnecessarily hit disk / cloud store.
    # Typically, gets triggered when there is a selection mask
    dim_slices = ()
    live_nonzeros = live_mask.nonzero()
    for dim_nonzeros in live_nonzeros:
        start = np.min(dim_nonzeros)
        stop = np.max(dim_nonzeros) + 1
        dim_slices += (slice(start, stop),)

    # Lazily pull the data with limits now, and limit mask so its the same shape.
    trace_mask_da = trace_mask_da.data
    headers = headers_da.data
    samples = amp_da.data

    live_mask_da = trace_mask_da[dim_slices]
    headers = headers[dim_slices]
    samples = samples[dim_slices]
    live_mask_da = live_mask_da.rechunk(headers.chunks)

    if selection_mask is not None:
        selection_mask = selection_mask[dim_slices]
        live_mask_da = live_mask_da & selection_mask

    # tmp file root
    out_dir = output_segy_path.parent
    tmp_dir = TemporaryDirectory(dir=out_dir)

    with tmp_dir:
        with TqdmCallback(desc="Unwrapping MDIO Blocks"):
            block_records = to_segy(
                samples=samples,
                headers=headers,
                live_mask=live_mask_da,
                segy_factory=segy_factory,
                file_root=tmp_dir.name,
            )

            if client is not None:
                block_records = block_records.compute()
            else:
                block_records = block_records.compute(num_workers=NUM_CPUS)

        ordered_files = [rec.path for rec in block_records.ravel() if rec != 0]
        ordered_files = [output_segy_path] + ordered_files

        if client is not None:
            _ = client.submit(concat_files, paths=ordered_files).result()
        else:
            concat_files(paths=ordered_files, progress=True)
