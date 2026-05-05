"""Coordinate handling utilities for MDIO ingestion.

This module contains functions for extracting, populating, and managing
coordinates during SEG-Y to MDIO conversion.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from segy.standards.codes import MeasurementSystem as SegyMeasurementSystem
from segy.standards.fields import binary as binary_header_fields

from mdio.builder.schemas.v1.units import AngleUnitEnum
from mdio.builder.schemas.v1.units import AngleUnitModel
from mdio.builder.schemas.v1.units import LengthUnitEnum
from mdio.builder.schemas.v1.units import LengthUnitModel
from mdio.segy.scalar import SCALE_COORDINATE_KEYS
from mdio.segy.scalar import _apply_coordinate_scalar

if TYPE_CHECKING:
    from segy.arrays import HeaderArray as SegyHeaderArray
    from xarray import Dataset as xr_Dataset

    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.core.dimension import Dimension
    from mdio.core.grid import Grid
    from mdio.segy.file import SegyFileInfo

logger = logging.getLogger(__name__)


# Constants for unit handling
MEASUREMENT_SYSTEM_KEY = binary_header_fields.Rev0.MEASUREMENT_SYSTEM_CODE.model.name
ANGLE_UNIT_KEYS = ["angle", "azimuth"]
SPATIAL_UNIT_KEYS = [
    "cdp_x",
    "cdp_y",
    "source_coord_x",
    "source_coord_y",
    "group_coord_x",
    "group_coord_y",
    "offset",
]


def get_coordinates(
    grid: Grid,
    segy_headers: SegyHeaderArray,
    mdio_template: AbstractDatasetTemplate,
) -> tuple[list[Dimension], dict[str, SegyHeaderArray]]:
    """Get the data dim and non-dim coordinates from the SEG-Y headers and MDIO template.

    Select a subset of the segy_dimensions that corresponds to the MDIO dimensions.
    Uses the grid's actual dimensions (which may have been transformed by grid overrides).
    The last dimension is always the vertical domain dimension.

    Args:
        grid: Inferred MDIO grid for SEG-Y file (may have transformed dimensions).
        segy_headers: Headers read in from SEG-Y file.
        mdio_template: The MDIO template to use for the conversion.

    Raises:
        ValueError: If a coordinate name from the MDIO template is not found in
                    the SEG-Y headers.

    Returns:
        A tuple containing:
            - A list of dimension coordinates (1-D arrays).
            - A dict of non-dimension coordinates (str: N-D arrays).
    """
    # Use grid's actual dimensions (may differ from template after grid overrides)
    dimensions_coords = list(grid.dims)

    # Extract non-dimension coordinates from headers
    non_dim_coords: dict[str, SegyHeaderArray] = {}
    for coord_name in mdio_template.coordinate_names:
        if coord_name not in segy_headers.dtype.names:
            err = f"Coordinate '{coord_name}' not found in SEG-Y headers."
            raise ValueError(err)
        # Copy the data to allow segy_headers to be garbage collected
        non_dim_coords[coord_name] = np.array(segy_headers[coord_name])

    return dimensions_coords, non_dim_coords


def populate_dim_coordinates(
    dataset: xr_Dataset, grid: Grid, drop_vars_delayed: list[str]
) -> tuple[xr_Dataset, list[str]]:
    """Populate the xarray dataset with dimension coordinate variables."""
    for dim in grid.dims:
        dataset[dim.name].values[:] = dim.coords
        drop_vars_delayed.append(dim.name)
    return dataset, drop_vars_delayed


def populate_non_dim_coordinates(
    dataset: xr_Dataset,
    grid: Grid,
    coordinates: dict[str, SegyHeaderArray],
    drop_vars_delayed: list[str],
    spatial_coordinate_scalar: int,
) -> tuple[xr_Dataset, list[str]]:
    """Populate the xarray dataset with coordinate variables.

    Memory optimization: Processes coordinates one at a time and explicitly
    releases intermediate arrays to reduce peak memory usage.
    """
    non_data_domain_dims = grid.dim_names[:-1]  # minus the data domain dimension

    # Process coordinates one at a time to minimize peak memory
    coord_names = list(coordinates.keys())
    for coord_name in coord_names:
        coord_values = coordinates.pop(coord_name)  # Remove from dict to free memory
        da_coord = dataset[coord_name]

        # Get coordinate shape from dataset (uses dask shape, no memory allocation)
        coord_shape = da_coord.shape

        # Create output array with fill value
        fill_value = da_coord.encoding.get("_FillValue") or da_coord.encoding.get("fill_value")
        if fill_value is None:
            fill_value = np.nan
        tmp_coord_values = np.full(coord_shape, fill_value, dtype=da_coord.dtype)

        # Compute slices for this coordinate's dimensions
        coord_axes = tuple(non_data_domain_dims.index(coord_dim) for coord_dim in da_coord.dims)
        coord_slices = tuple(slice(None) if idx in coord_axes else 0 for idx in range(len(non_data_domain_dims)))

        # Read only the required slice from grid map
        coord_trace_indices = np.asarray(grid.map[coord_slices])

        # Find valid (non-null) indices
        not_null = coord_trace_indices != grid.map.fill_value

        # Populate values efficiently
        if not_null.any():
            valid_indices = coord_trace_indices[not_null]
            tmp_coord_values[not_null] = coord_values[valid_indices]

        # Apply scalar if needed
        if coord_name in SCALE_COORDINATE_KEYS:
            tmp_coord_values = _apply_coordinate_scalar(tmp_coord_values, spatial_coordinate_scalar)

        # Assign to dataset
        dataset[coord_name][:] = tmp_coord_values
        drop_vars_delayed.append(coord_name)

        # Explicitly release intermediate arrays
        del tmp_coord_values, coord_trace_indices, not_null, coord_values

    return dataset, drop_vars_delayed


def get_spatial_coordinate_unit(segy_file_info: SegyFileInfo) -> LengthUnitModel | None:
    """Get the coordinate unit from the SEG-Y headers."""
    measurement_system_code = int(segy_file_info.binary_header_dict[MEASUREMENT_SYSTEM_KEY])

    if measurement_system_code not in (1, 2):
        logger.warning(
            "Unexpected value in coordinate unit (%s) header: %s. Can't extract coordinate unit and will "
            "ingest without coordinate units.",
            MEASUREMENT_SYSTEM_KEY,
            measurement_system_code,
        )
        return None

    if measurement_system_code == SegyMeasurementSystem.METERS:
        unit = LengthUnitEnum.METER
    if measurement_system_code == SegyMeasurementSystem.FEET:
        unit = LengthUnitEnum.FOOT

    return LengthUnitModel(length=unit)


def apply_runtime_units(template: AbstractDatasetTemplate, segy_file_info: SegyFileInfo) -> AbstractDatasetTemplate:
    """Update the template with dynamic and some pre-defined units."""
    unit = get_spatial_coordinate_unit(segy_file_info)

    # Add units for pre-defined: angle and azimuth etc.
    new_units = {key: AngleUnitModel(angle=AngleUnitEnum.DEGREES) for key in ANGLE_UNIT_KEYS}

    # If a spatial unit is not provided, we return as is
    if unit is None:
        template.add_units(new_units)
        return template

    # Dynamically add units based on the spatial coordinate unit
    for key in SPATIAL_UNIT_KEYS:
        current_value = template.get_unit_by_key(key)
        if current_value is not None:
            logger.warning("Unit for %s already in template. Will keep the original unit: %s", key, current_value)
            continue

        new_units[key] = unit

    template.add_units(new_units)
    return template
