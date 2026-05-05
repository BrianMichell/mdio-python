"""Seismic3DObnReceiverGathersTemplate MDIO v1 dataset templates."""

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


class Seismic3DObnReceiverGathersTemplate(AbstractDatasetTemplate):
    """Seismic 3D OBN (Ocean Bottom Node) receiver gathers template.

    This template uses shot_index as a calculated dimension (similar to StreamerFieldRecords3D).
    The shot_index is computed from shot_point values during SEG-Y import using the
    AutoShotWrap grid override. AutoShotWrap is template-aware and automatically detects
    that this template uses shot_line (not sail_line) based on the dimension names.
    The original shot_point values are preserved as a coordinate indexed by
    (shot_line, gun, shot_index).

    This design handles OBN data where shot points may be interleaved across multiple guns.

    Special handling for component dimension:
        If the SEG-Y spec does not contain a 'component' field, the ingestion process
        will automatically synthesize a component dimension with constant value 1 for
        all traces. A warning is logged when this occurs. This is handled explicitly
        in GridOverrider._synthesize_obn_component().
    """

    def __init__(self, data_domain: SeismicDataDomain = "time"):
        super().__init__(data_domain=data_domain)

        self._spatial_dim_names = ("component", "receiver", "shot_line", "gun", "shot_index")
        self._calculated_dims = ("shot_index",)
        self._dim_names = (*self._spatial_dim_names, self._data_domain)
        self._physical_coord_names = (
            "group_coord_x",
            "group_coord_y",
            "source_coord_x",
            "source_coord_y",
        )
        self._logical_coord_names = ("shot_point", "orig_field_record_num")
        self._var_chunk_shape = (1, 1, 1, 1, 512, 4096)
        self.synthesize_missing_dims = ("component",)

    @property
    def _name(self) -> str:
        return "ObnReceiverGathers3D"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {"surveyType": "3D", "gatherType": "common_receiver"}

    def declare_coordinate_specs(self) -> tuple[Any, ...]:
        from mdio.ingestion.schema_resolver import CoordinateSpec

        return (
            CoordinateSpec(
                name="group_coord_x",
                dimensions=("receiver",),
                dtype=ScalarType.FLOAT64,
                source="header",
                header_key="group_coord_x",
            ),
            CoordinateSpec(
                name="group_coord_y",
                dimensions=("receiver",),
                dtype=ScalarType.FLOAT64,
                source="header",
                header_key="group_coord_y",
            ),
            CoordinateSpec(
                name="shot_point",
                dimensions=("shot_line", "gun", "shot_index"),
                dtype=ScalarType.UINT32,
                source="header",
                header_key="shot_point",
            ),
            CoordinateSpec(
                name="orig_field_record_num",
                dimensions=("shot_line", "gun", "shot_index"),
                dtype=ScalarType.UINT32,
                source="header",
                header_key="orig_field_record_num",
            ),
            CoordinateSpec(
                name="source_coord_x",
                dimensions=("shot_line", "gun", "shot_index"),
                dtype=ScalarType.FLOAT64,
                source="header",
                header_key="source_coord_x",
            ),
            CoordinateSpec(
                name="source_coord_y",
                dimensions=("shot_line", "gun", "shot_index"),
                dtype=ScalarType.FLOAT64,
                source="header",
                header_key="source_coord_y",
            ),
        )

    def _add_coordinates(self) -> None:
        # Add dimension coordinates
        # EXCLUDE: `shot_index` since it's 0-N (calculated dimension)
        self._builder.add_coordinate(
            "component",
            dimensions=("component",),
            data_type=ScalarType.UINT8,
        )
        self._builder.add_coordinate(
            "receiver",
            dimensions=("receiver",),
            data_type=ScalarType.UINT32,
        )
        self._builder.add_coordinate(
            "shot_line",
            dimensions=("shot_line",),
            data_type=ScalarType.UINT32,
        )
        self._builder.add_coordinate(
            "gun",
            dimensions=("gun",),
            data_type=ScalarType.UINT8,
        )
        self._builder.add_coordinate(
            self._data_domain,
            dimensions=(self._data_domain,),
            data_type=ScalarType.INT32,
            metadata=VariableMetadata(units_v1=self.get_unit_by_key(self._data_domain)),
        )

        # Add non-dimension coordinates with computed chunk sizes
        # For 1D coordinates (over receiver)
        coord_spatial_shape_1d = (self._dim_sizes[1],)  # receiver
        coord_chunk_shape_1d = get_constrained_chunksize(
            coord_spatial_shape_1d,
            ScalarType.FLOAT64,
            MAX_COORDINATES_BYTES,
        )
        chunk_grid_1d = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=coord_chunk_shape_1d))

        # For 3D coordinates (over shot_line, gun, shot_index)
        coord_spatial_shape_3d = (
            self._dim_sizes[2],
            self._dim_sizes[3],
            self._dim_sizes[4],
        )  # shot_line, gun, shot_index
        coord_chunk_shape_3d = get_constrained_chunksize(
            coord_spatial_shape_3d,
            ScalarType.FLOAT64,
            MAX_COORDINATES_BYTES,
        )
        chunk_grid_3d = RegularChunkGrid(configuration=RegularChunkShape(chunk_shape=coord_chunk_shape_3d))

        compressor = Blosc(cname=BloscCname.zstd)
        self._builder.add_coordinate(
            "group_coord_x",
            dimensions=("receiver",),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=VariableMetadata(units_v1=self.get_unit_by_key("group_coord_x"), chunk_grid=chunk_grid_1d),
        )
        self._builder.add_coordinate(
            "group_coord_y",
            dimensions=("receiver",),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=VariableMetadata(units_v1=self.get_unit_by_key("group_coord_y"), chunk_grid=chunk_grid_1d),
        )
        self._builder.add_coordinate(
            "shot_point",
            dimensions=("shot_line", "gun", "shot_index"),
            data_type=ScalarType.UINT32,
            compressor=compressor,
            metadata=VariableMetadata(chunk_grid=chunk_grid_3d),
        )
        self._builder.add_coordinate(
            "orig_field_record_num",
            dimensions=("shot_line", "gun", "shot_index"),
            data_type=ScalarType.UINT32,
            compressor=compressor,
            metadata=VariableMetadata(chunk_grid=chunk_grid_3d),
        )
        self._builder.add_coordinate(
            "source_coord_x",
            dimensions=("shot_line", "gun", "shot_index"),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=VariableMetadata(units_v1=self.get_unit_by_key("source_coord_x"), chunk_grid=chunk_grid_3d),
        )
        self._builder.add_coordinate(
            "source_coord_y",
            dimensions=("shot_line", "gun", "shot_index"),
            data_type=ScalarType.FLOAT64,
            compressor=compressor,
            metadata=VariableMetadata(units_v1=self.get_unit_by_key("source_coord_y"), chunk_grid=chunk_grid_3d),
        )
