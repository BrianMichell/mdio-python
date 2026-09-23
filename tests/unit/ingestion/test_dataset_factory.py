"""Tests for the schema-driven ``build_mdio_dataset`` factory."""

from __future__ import annotations

import numpy as np
import pytest

from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.v1.units import LengthUnitEnum
from mdio.builder.schemas.v1.units import LengthUnitModel
from mdio.builder.templates.types import CoordinateSpec
from mdio.converters.type_converter import to_structured_type
from mdio.ingestion.dataset_factory import build_mdio_dataset
from mdio.ingestion.schema import DimensionSpec
from mdio.ingestion.schema import ResolvedSchema


def _vars_by_name(dataset) -> dict:  # noqa: ANN001
    """Index a built Dataset's variables (and coordinates) by name."""
    return {var.name: var for var in dataset.variables}


def _dim_names(variable) -> tuple[str, ...]:  # noqa: ANN001
    """Return the ordered dimension names of a built Variable."""
    return tuple(dim.name for dim in variable.dimensions)


def _chunk_shape(variable) -> tuple[int, ...]:  # noqa: ANN001
    """Return the configured chunk shape of a built Variable."""
    return tuple(variable.metadata.chunk_grid.configuration.chunk_shape)


def _shard_shape(variable) -> tuple[int, ...]:  # noqa: ANN001
    """Return the configured shard shape of a built Variable, or ``()`` when disabled."""
    if variable.metadata.shard_grid is None:
        return ()
    return tuple(variable.metadata.shard_grid.configuration.chunk_shape)


@pytest.fixture
def basic_schema() -> ResolvedSchema:
    """A minimal 2-spatial-dim + vertical schema with one non-dim coordinate."""
    return ResolvedSchema(
        name="Toy3D",
        dimensions=[
            DimensionSpec(name="inline", is_spatial=True, dtype=ScalarType.INT32),
            DimensionSpec(name="crossline", is_spatial=True, dtype=ScalarType.INT32),
            DimensionSpec(name="time", is_spatial=False, dtype=ScalarType.INT16),
        ],
        coordinates=[
            CoordinateSpec(name="cdp_x", dimensions=("inline", "crossline"), dtype=ScalarType.FLOAT64),
        ],
        chunk_shape=(2, -1, 4),
        default_variable_name="amplitude",
    )


class TestBuildMdioDataset:
    """Tests for the structure produced by ``build_mdio_dataset``."""

    def test_builds_dimensions_coordinates_and_data_variable(self, basic_schema: ResolvedSchema) -> None:
        """All dimensions, the non-dim coordinate, trace_mask and data variable are created."""
        dataset = build_mdio_dataset(schema=basic_schema, sizes=(2, 3, 4))
        variables = _vars_by_name(dataset)

        assert {"inline", "crossline", "time", "cdp_x", "trace_mask", "amplitude"}.issubset(variables)
        assert _dim_names(variables["amplitude"]) == ("inline", "crossline", "time")
        assert _dim_names(variables["trace_mask"]) == ("inline", "crossline")
        assert _dim_names(variables["cdp_x"]) == ("inline", "crossline")

    def test_resolves_negative_one_chunk_against_sizes(self, basic_schema: ResolvedSchema) -> None:
        """A ``-1`` chunk entry is replaced by the actual dimension size."""
        dataset = build_mdio_dataset(schema=basic_schema, sizes=(2, 3, 4))
        amplitude = _vars_by_name(dataset)["amplitude"]
        # chunk_shape (2, -1, 4) with sizes (2, 3, 4) -> (2, 3, 4)
        assert _chunk_shape(amplitude) == (2, 3, 4)

    def test_default_variable_name_recorded_in_attributes(self, basic_schema: ResolvedSchema) -> None:
        """The default variable name is written to dataset attributes."""
        dataset = build_mdio_dataset(schema=basic_schema, sizes=(2, 3, 4))
        assert dataset.metadata.attributes["defaultVariableName"] == "amplitude"

    def test_resolves_shards_and_keeps_headers_unsharded(self, basic_schema: ResolvedSchema) -> None:
        """Shard ``-1`` values resolve against sizes and apply only to the data variable."""
        header_dtype = to_structured_type(np.dtype([("cdp_x", "int32"), ("cdp_y", "int32")]))
        schema = basic_schema.model_copy(update={"shard_shape": (4, -1, 8)})

        dataset = build_mdio_dataset(schema=schema, sizes=(4, 6, 8), header_dtype=header_dtype)
        variables = _vars_by_name(dataset)

        assert _chunk_shape(variables["amplitude"]) == (2, 6, 4)
        assert _shard_shape(variables["amplitude"]) == (4, 6, 8)
        assert _shard_shape(variables["headers"]) == ()

    @pytest.mark.parametrize(
        ("shard_shape", "match"),
        [
            ((4, 6), "rank does not match"),
            ((3, 6, 4), "positive whole multiple"),
            ((4, 0, 8), "positive whole multiple"),
        ],
    )
    def test_rejects_invalid_shards(
        self,
        basic_schema: ResolvedSchema,
        shard_shape: tuple[int, ...],
        match: str,
    ) -> None:
        """Invalid shard rank, zero, and non-multiples fail before store creation."""
        schema = basic_schema.model_copy(update={"shard_shape": shard_shape})
        with pytest.raises(ValueError, match=match):
            build_mdio_dataset(schema=schema, sizes=(4, 6, 8))

    def test_header_dtype_adds_headers_variable(self, basic_schema: ResolvedSchema) -> None:
        """Passing a header dtype adds a ``headers`` variable over the spatial dims only."""
        header_dtype = to_structured_type(np.dtype([("cdp_x", "int32"), ("cdp_y", "int32")]))
        dataset = build_mdio_dataset(schema=basic_schema, sizes=(2, 3, 4), header_dtype=header_dtype)

        headers = _vars_by_name(dataset)["headers"]
        assert _dim_names(headers) == ("inline", "crossline")
        # Headers chunk drops the vertical axis.
        assert _chunk_shape(headers) == (2, 3)

    def test_no_headers_variable_when_dtype_missing(self, basic_schema: ResolvedSchema) -> None:
        """Without a header dtype, no ``headers`` variable is created."""
        dataset = build_mdio_dataset(schema=basic_schema, sizes=(2, 3, 4))
        assert "headers" not in _vars_by_name(dataset)

    def test_calculated_dimension_has_no_dimension_coordinate(self) -> None:
        """A calculated spatial dim (e.g. ``shot_index``) gets no dimension coordinate."""
        schema = ResolvedSchema(
            name="Calc",
            dimensions=[
                DimensionSpec(name="receiver", is_spatial=True, dtype=ScalarType.UINT32),
                DimensionSpec(name="shot_index", is_spatial=True, is_calculated=True, dtype=ScalarType.INT32),
                DimensionSpec(name="time", is_spatial=False, dtype=ScalarType.INT16),
            ],
            coordinates=[],
            chunk_shape=(2, 2, 4),
        )
        dataset = build_mdio_dataset(schema=schema, sizes=(2, 2, 4))
        variables = _vars_by_name(dataset)

        # shot_index is still a data-variable dimension, but not its own 1-D coordinate.
        assert _dim_names(variables["amplitude"]) == ("receiver", "shot_index", "time")
        assert "shot_index" not in variables
        assert "receiver" in variables

    def test_extra_variables_are_appended(self, basic_schema: ResolvedSchema) -> None:
        """Extra variable dicts (e.g. raw_headers) are added to the dataset."""
        extra = [
            {
                "name": "raw_headers",
                "dimensions": ("inline", "crossline"),
                "data_type": ScalarType.BYTES240,
            }
        ]
        dataset = build_mdio_dataset(schema=basic_schema, sizes=(2, 3, 4), extra_variables=extra)
        assert "raw_headers" in _vars_by_name(dataset)

    def test_units_attached_to_coordinate(self, basic_schema: ResolvedSchema) -> None:
        """Units passed by name are attached to the matching coordinate metadata."""
        meter = LengthUnitModel(length=LengthUnitEnum.METER)
        dataset = build_mdio_dataset(schema=basic_schema, sizes=(2, 3, 4), units={"cdp_x": meter})
        cdp_x = _vars_by_name(dataset)["cdp_x"]
        assert cdp_x.metadata.units_v1 == meter

    def test_crs_attached_to_dataset_metadata(self, basic_schema: ResolvedSchema) -> None:
        """Optional CRS is stored on dataset metadata, not the attributes bag."""
        schema = basic_schema.model_copy(update={"crs": "EPSG:32610"})
        dataset = build_mdio_dataset(schema=schema, sizes=(2, 3, 4))
        assert dataset.metadata.crs == "EPSG:32610"
        assert "crs" not in dataset.metadata.attributes
