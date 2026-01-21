"""Conversion from ZGY to MDIO v1 format."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from mdio.api.io import _normalize_path
from mdio.api.io import to_mdio

if TYPE_CHECKING:
    from pathlib import Path

    from upath import UPath
    from xarray import Dataset as xr_Dataset

    from mdio.builder.templates.base import AbstractDatasetTemplate


logger = logging.getLogger(__name__)


def _import_pyzgy() -> type:
    """Import pyzgy and return the module.

    Returns:
        The pyzgy module.

    Raises:
        ImportError: If pyzgy is not installed.
    """
    try:
        import pyzgy  # noqa: PLC0415

        return pyzgy
    except ImportError as e:
        msg = (
            "The 'pyzgy' package is required for ZGY file support. "
            "Install via 'pip install multidimio[zgy]' or 'pip install pyzgy'."
        )
        raise ImportError(msg) from e


def zgy_to_mdio(
    input_path: UPath | Path | str,
    output_path: UPath | Path | str,
    mdio_template: AbstractDatasetTemplate | None = None,  # noqa: ARG001
    overwrite: bool = False,
) -> None:
    """Convert a ZGY file to an MDIO v1 file.

    ZGY is a seismic data format developed by Schlumberger that provides fast random access
    to 3D seismic volumes. This function converts ZGY files to MDIO format using pyzgy's
    xarray backend for seamless data handling.

    Args:
        input_path: The universal path to the input ZGY file.
        output_path: The universal path for the output MDIO v1 file.
        mdio_template: Optional MDIO template for customization. Currently unused as
            the dataset structure is derived directly from the ZGY file.
        overwrite: Whether to overwrite the output file if it already exists. Defaults to False.

    Raises:
        FileExistsError: If the output location already exists and overwrite is False.

    Example:
        >>> from mdio.converters.zgy import zgy_to_mdio
        >>> zgy_to_mdio("input.zgy", "output.mdio")
    """
    import xarray as xr  # noqa: PLC0415

    # Validate pyzgy is available
    _import_pyzgy()

    input_path = _normalize_path(input_path)
    output_path = _normalize_path(output_path)

    if not overwrite and output_path.exists():
        err = f"Output location '{output_path.as_posix()}' exists. Set `overwrite=True` if intended."
        raise FileExistsError(err)

    # Open ZGY file using pyzgy's xarray backend
    logger.info("Opening ZGY file: %s", input_path.as_posix())
    zgy_ds: xr_Dataset = xr.open_dataset(input_path.as_posix(), engine="pyzgy")

    # Rename dimensions to match MDIO conventions
    dim_mapping = {"iline": "inline", "xline": "crossline"}
    zgy_ds = zgy_ds.rename({k: v for k, v in dim_mapping.items() if k in zgy_ds.dims})

    # Rename data variable if needed
    if "data" in zgy_ds.data_vars:
        zgy_ds = zgy_ds.rename_vars({"data": "amplitude"})

    # Add MDIO metadata
    zgy_ds.attrs["name"] = _get_template_name(zgy_ds)
    zgy_ds.attrs["attributes"] = {
        "surveyType": "3D",
        "gatherType": "stacked",
        "defaultVariableName": "amplitude",
    }

    # Create trace mask (all True for ZGY - no dead traces)
    spatial_dims = [d for d in zgy_ds.dims if d not in ("time", "depth", "sample")]
    mask_shape = tuple(zgy_ds.sizes[d] for d in spatial_dims)
    zgy_ds["trace_mask"] = (spatial_dims, np.ones(mask_shape, dtype=bool))

    # Write to MDIO
    logger.info("Writing MDIO file: %s", output_path.as_posix())
    to_mdio(zgy_ds, output_path=output_path, mode="w", compute=True)

    logger.info("ZGY to MDIO conversion complete")


def _get_template_name(ds: xr_Dataset) -> str:
    """Determine template name based on dataset dimensions."""
    if "depth" in ds.dims:
        return "PostStack3DDepth"
    return "PostStack3DTime"
