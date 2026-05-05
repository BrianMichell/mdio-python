"""Tests for ingestion schema resolver."""

from mdio.builder.templates.seismic_3d_streamer_shot import Seismic3DStreamerShotGathersTemplate
from mdio.ingestion.schema_resolver import SchemaResolver
from mdio.segy.geometry import GridOverrides


class TestSchemaResolver:
    def test_resolve_without_overrides(self) -> None:
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        resolver = SchemaResolver()
        schema = resolver.resolve(template, grid_overrides=None)

        assert schema.name == "StreamerShotGathers3D"
        assert len(schema.dimensions) == 4
        assert schema.dimensions[0].name == "shot_point"
        assert schema.dimensions[1].name == "cable"

    def test_resolve_with_non_binned_default(self) -> None:
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        resolver = SchemaResolver()
        overrides = GridOverrides(non_binned=True, chunksize=64)
        schema = resolver.resolve(template, grid_overrides=overrides)

        assert len(schema.dimensions) == 3
        assert schema.dimensions[0].name == "shot_point"
        assert schema.dimensions[1].name == "trace"
        assert schema.chunk_shape == (16, 64, 1024)

    def test_resolve_with_has_duplicates(self) -> None:
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        resolver = SchemaResolver()
        overrides = GridOverrides(has_duplicates=True)
        schema = resolver.resolve(template, grid_overrides=overrides)

        assert len(schema.dimensions) == 5
        assert schema.dimensions[3].name == "trace"
        assert schema.chunk_shape == (16, 1, 32, 1, 1024)
