"""Utils for reading MDIO dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import zarr
from upath import UPath
from xarray import Dataset as xr_Dataset
from xarray import open_zarr as xr_open_zarr
from xarray.backends.writers import to_zarr as xr_to_zarr

from mdio.constants import ZarrFormat
from mdio.core.zarr_io import zarr_warnings_suppress_unstable_structs_v3

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import MutableMapping
    from pathlib import Path

    from xarray import Dataset
    from xarray.core.types import T_Chunks
    from xarray.core.types import ZarrWriteModes


def _normalize_path(path: UPath | Path | str) -> UPath:
    """Normalize input into a UPath, preserving any existing storage options.

    Args:
        path: Path to normalize (can be string, Path, or UPath).

    Returns:
        A UPath instance. If input was already a UPath, it is returned as-is
        to preserve its storage options.
    """
    if isinstance(path, UPath):
        return path
    return UPath(path)


def _normalize_storage_options(path: UPath) -> dict[str, Any] | None:
    """Extract storage options from a UPath."""
    return None if len(path.storage_options) == 0 else path.storage_options


def _get_store(path: UPath) -> MutableMapping[str, bytes]:
    """Return a clean fsspec mapper for this path.

    This intentionally avoids:
      - URI-based access
      - storage_options parameter (uses UPath's filesystem directly)
      - leaking `fs` into async s3fs kwargs

    It reconstructs the filesystem if UPath polluted fs.kwargs.

    Args:
        path: A UPath with the appropriate storage options already set.

    Returns:
        An fsspec mapper suitable for zarr/xarray.
    """
    fs = path.fs

    # Defensive cleanup for UPath + async s3fs interaction
    if hasattr(fs, "kwargs") and "fs" in fs.kwargs:
        clean_kwargs = {k: v for k, v in fs.kwargs.items() if k != "fs"}
        fs = fs.__class__(**clean_kwargs)

    return fs.get_mapper(path.path)


def open_mdio(
    input_path: UPath | Path | str,
    chunks: T_Chunks = None,
) -> xr_Dataset:
    """Open an MDIO (Zarr) dataset from a universal path.

    Args:
        input_path: Universal input path for the MDIO dataset. For cloud storage
            with specific credentials, pass a UPath with storage options:
            ``UPath("s3://bucket/path", profile="my-profile")``
        chunks: Dask chunking specification.
            - ``chunks="auto"`` will use dask ``auto`` chunking
            - ``chunks=None`` skips using dask, which is generally faster for small arrays.
            - ``chunks=-1`` loads the data with dask using a single chunk for all arrays.
            - ``chunks={}`` loads the data with dask using the engine's preferred chunk size (on disk).
            - ``chunks={dim: chunk, ...}`` loads the data with dask using the specified chunk size.

    Returns:
        An opened xarray Dataset.
    """
    input_path = _normalize_path(input_path)
    zarr_format = zarr.config.get("default_zarr_format")

    store = _get_store(input_path)

    return xr_open_zarr(
        store,
        chunks=chunks,
        mask_and_scale=zarr_format == ZarrFormat.V3,  # on for v3
        consolidated=zarr_format == ZarrFormat.V2,  # on for v2
    )


def to_mdio(  # noqa: PLR0913
    dataset: Dataset,
    output_path: UPath | Path | str,
    mode: ZarrWriteModes | None = None,
    *,
    compute: bool = True,
    region: Mapping[str, slice | Literal["auto"]] | Literal["auto"] | None = None,
) -> None:
    """Write a Dataset to an MDIO (Zarr) store.

    Args:
        dataset: Dataset to write.
        output_path: Universal output path. For cloud storage with specific
            credentials, pass a UPath with storage options:
            ``UPath("s3://bucket/path", profile="my-profile")``
        mode: Persistence mode: "w" means create (overwrite if exists)
            "w-" means create (fail if exists)
            "a" means override all existing variables including dimension coordinates
            "a-" means only append those variables that have ``append_dim``.
            "r+" means modify existing array *values* only.
        compute: Whether to compute immediately.
        region: Optional region mapping for partial writes.
    """
    output_path = _normalize_path(output_path)
    zarr_format = zarr.config.get("default_zarr_format")

    store = _get_store(output_path)

    with zarr_warnings_suppress_unstable_structs_v3():
        xr_to_zarr(
            dataset,
            store=store,
            mode=mode,
            compute=compute,
            consolidated=zarr_format == ZarrFormat.V2,
            region=region,
            write_empty_chunks=False,
        )
