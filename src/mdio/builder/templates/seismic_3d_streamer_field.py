"""Seismic3DStreamerFieldRecordsTemplate MDIO v1 dataset templates."""

from typing import Any

from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.chunk_grid import RegularChunkShape
from mdio.builder.schemas.compressors import Blosc
from mdio.builder.schemas.compressors import BloscCname
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.variable import VariableMetadata
from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import SeismicDataDomain
from mdio.core.utils_write import MAX_COORDINATES_BYTES
from mdio.core.utils_write import get_constrained_chunksize


class Seismic3DStreamerFieldRecordsTemplate(AbstractDatasetTemplate):
    """Seismic 3D streamer shot field records template.

    A generalized template for streamer field records that are optimized for:
        - Common-shot access
        - Common-channel access

    It can also store all the shot-lines of a survey in one MDIO if needed.

    Args:
        data_domain: The domain of the dataset.
    """

    def __init__(self, data_domain: SeismicDataDomain = "time"):
        super().__init__(data_domain=data_domain)

        self._spatial_dim_names = ("sail_line", "gun", "shot_index", "cable", "channel")
        self._calculated_dims = ("shot_index",)
        self._dim_names = (*self._spatial_dim_names, self._data_domain)
        self._physical_coord_names = ("source_coord_x", "source_coord_y", "group_coord_x", "group_coord_y")
        self._logical_coord_names = ("shot_point", "orig_field_record_num")  # ffid
        self._var_chunk_shape = (1, 1, 16, 1, 32, 1024)

    @property
    def _name(self) -> str:
        return "StreamerFieldRecords3D"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyDimensionality": "3D", "gatherType": "common_source"}

    def declare_coordinate_specs(self) -> tuple[Any, ...]:
        from mdio.ingestion.schema_resolver import CoordinateSpec

        shot_dims = ("sail_line", "gun", "shot_index")
        receiver_dims = ("sail_line", "gun", "shot_index", "cable", "channel")
        return (
            CoordinateSpec(name="orig_field_record_num", dimensions=shot_dims, dtype=ScalarType.UINT32),
            CoordinateSpec(name="shot_point", dimensions=shot_dims, dtype=ScalarType.UINT32),
            CoordinateSpec(name="source_coord_x", dimensions=shot_dims, dtype=ScalarType.FLOAT64),
            CoordinateSpec(name="source_coord_y", dimensions=shot_dims, dtype=ScalarType.FLOAT64),
            CoordinateSpec(name="group_coord_x", dimensions=receiver_dims, dtype=ScalarType.FLOAT64),
            CoordinateSpec(name="group_coord_y", dimensions=receiver_dims, dtype=ScalarType.FLOAT64),
        )

    def _add_coordinates(self) -> None:
        # Add dimension coordinates
        # EXCLUDE: `shot_index` since its 0-N
        self._builder.add_coordinate(
            "sail_line",
            dimensions=("sail_line",),
            data_type=ScalarType.UINT32,
        )
        self._builder.add_coordinate(
            "gun",
            dimensions=("gun",),
            data_type=ScalarType.UINT8,
        )
        self._builder.add_coordinate(
            "cable",
            dimensions=("cable",),
            data_type=ScalarType.UINT8,
        )
        self._builder.add_coordinate(
            "channel",
            dimensions=("channel",),
            data_type=ScalarType.UINT16,
        )
        self._builder.add_coordinate(
            self._data_domain,
            dimensions=(self._data_domain,),
            data_type=ScalarType.INT32,
            metadata=self._dim_coord_metadata(self._data_domain),
        )

        # Chunk grids for shot-indexed (3D) and receiver-indexed (5D) non-dim coordinates.
        shot_chunk_shape = get_constrained_chunksize(self._dim_sizes[:3], ScalarType.FLOAT64, MAX_COORDINATES_BYTES)
        chunk_grid_3d = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=shot_chunk_shape))
        receiver_chunk_shape = get_constrained_chunksize(
            self._dim_sizes[:5], ScalarType.FLOAT64, MAX_COORDINATES_BYTES
        )
        chunk_grid_5d = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=receiver_chunk_shape))

        compressor = Blosc(cname=BloscCname.zstd)
        self._builder.add_coordinate(
            "orig_field_record_num",
            dimensions=("sail_line", "gun", "shot_index"),
            data_type=ScalarType.UINT32,
            compressor=compressor,
            metadata=VariableMetadata(chunk_grid=chunk_grid_3d),
        )
        self._builder.add_coordinate(
            "shot_point",
            dimensions=("sail_line", "gun", "shot_index"),
            data_type=ScalarType.UINT32,
            compressor=compressor,
            metadata=VariableMetadata(chunk_grid=chunk_grid_3d),
        )
        self._builder.add_coordinate(
            "source_coord_x",
            dimensions=("sail_line", "gun", "shot_index"),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=VariableMetadata(units_v1=self.get_unit_by_key("source_coord_x"), chunk_grid=chunk_grid_3d),
        )
        self._builder.add_coordinate(
            "source_coord_y",
            dimensions=("sail_line", "gun", "shot_index"),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=VariableMetadata(units_v1=self.get_unit_by_key("source_coord_y"), chunk_grid=chunk_grid_3d),
        )
        self._builder.add_coordinate(
            "group_coord_x",
            dimensions=("sail_line", "gun", "shot_index", "cable", "channel"),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=VariableMetadata(units_v1=self.get_unit_by_key("group_coord_x"), chunk_grid=chunk_grid_5d),
        )
        self._builder.add_coordinate(
            "group_coord_y",
            dimensions=("sail_line", "gun", "shot_index", "cable", "channel"),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=VariableMetadata(units_v1=self.get_unit_by_key("group_coord_y"), chunk_grid=chunk_grid_5d),
        )
