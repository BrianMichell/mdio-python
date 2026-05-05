"""Conversion from SEG-Y to MDIO v1 format."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

import numpy as np
import zarr
from segy.config import SegyFileSettings
from segy.config import SegyHeaderOverrides
from segy.standards.codes import MeasurementSystem as SegyMeasurementSystem
from segy.standards.fields import binary as binary_header_fields

from mdio.api.io import _normalize_path
from mdio.api.io import to_mdio
from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.chunk_grid import RegularChunkShape
from mdio.builder.schemas.compressors import Blosc
from mdio.builder.schemas.compressors import BloscCname
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.units import AngleUnitEnum
from mdio.builder.schemas.v1.units import AngleUnitModel
from mdio.builder.schemas.v1.units import LengthUnitEnum
from mdio.builder.schemas.v1.units import LengthUnitModel
from mdio.builder.schemas.v1.variable import VariableMetadata
from mdio.builder.xarray_builder import to_xarray_dataset
from mdio.constants import ZarrFormat
from mdio.converters.exceptions import GridTraceCountError
from mdio.converters.exceptions import GridTraceSparsityError
from mdio.converters.type_converter import to_structured_type
from mdio.core.config import MDIOSettings
from mdio.core.grid import Grid
from mdio.core.utils_write import MAX_COORDINATES_BYTES
from mdio.core.utils_write import MAX_SIZE_LIVE_MASK
from mdio.core.utils_write import get_constrained_chunksize
from mdio.segy import blocked_io
from mdio.segy.file import get_segy_file_info
from mdio.segy.geometry import GridOverrides
from mdio.segy.scalar import SCALE_COORDINATE_KEYS
from mdio.segy.scalar import _apply_coordinate_scalar
from mdio.segy.utilities import get_grid_plan

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from segy.arrays import HeaderArray as SegyHeaderArray
    from segy.schema import SegySpec
    from upath import UPath
    from xarray import Dataset as xr_Dataset

    from mdio.builder.schemas import Dataset
    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.core.dimension import Dimension
    from mdio.segy.file import SegyFileArguments
    from mdio.segy.file import SegyFileInfo

logger = logging.getLogger(__name__)


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




def segy_to_mdio(  # noqa PLR0913
    segy_spec: SegySpec,
    mdio_template: AbstractDatasetTemplate,
    input_path: UPath | Path | str,
    output_path: UPath | Path | str,
    overwrite: bool = False,
    grid_overrides: GridOverrides | dict[str, Any] | None = None,
    segy_header_overrides: SegyHeaderOverrides | None = None,
) -> None:
    """A function that converts a SEG-Y file to an MDIO v1 file.

    Ingest a SEG-Y file according to the segy_spec. This could be a spec from registry or custom.

    Args:
        segy_spec: The SEG-Y specification to use for the conversion.
        mdio_template: The MDIO template to use for the conversion.
        input_path: The universal path of the input SEG-Y file.
        output_path: The universal path for the output MDIO v1 file.
        overwrite: Whether to overwrite the output file if it already exists. Defaults to False.
        grid_overrides: Grid override configuration. Can be a GridOverrides instance for type
            safety, or a dict for backward compatibility. See GridOverrides class for available
            options.
        segy_header_overrides: Option to override specific SEG-Y headers during ingestion.

    Raises:
        FileExistsError: If the output location already exists and overwrite is False.
    """
    # Convert dict to GridOverrides if needed for type safety
    if isinstance(grid_overrides, dict):
        grid_overrides = GridOverrides.model_validate(grid_overrides) if grid_overrides else None

    # Use ingestion pipeline
    from mdio.ingestion.pipeline import run_segy_ingestion

    return run_segy_ingestion(
        segy_spec=segy_spec,
        mdio_template=mdio_template,
        input_path=input_path,
        output_path=output_path,
        overwrite=overwrite,
        grid_overrides=grid_overrides,
        segy_header_overrides=segy_header_overrides,
    )
