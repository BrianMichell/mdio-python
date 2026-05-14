"""Unit tests for the v1.2 SchemaResolver."""

from __future__ import annotations

import pytest

from mdio.builder.templates.seismic_3d_cdp import Seismic3DCdpGathersTemplate
from mdio.builder.templates.seismic_3d_obn import Seismic3DObnReceiverGathersTemplate
from mdio.builder.templates.seismic_3d_streamer_shot import Seismic3DStreamerShotGathersTemplate
from mdio.ingestion.schema_resolver import SchemaResolver
from mdio.segy.geometry import GridOverrides


class TestSchemaResolverNoOverrides:
    def test_streamer_shot_template_basic(self) -> None:
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        schema = SchemaResolver().resolve(template, grid_overrides=None)

        assert schema.name == "StreamerShotGathers3D"
        assert [d.name for d in schema.dimensions] == ["shot_point", "cable", "channel", "time"]
        assert schema.dimensions[-1].is_spatial is False
        assert schema.dimensions[-1].is_calculated is False
        # Default chunk shape comes straight from the template.
        assert schema.chunk_shape == template.full_chunk_shape

    def test_obn_template_marks_shot_index_as_calculated(self) -> None:
        template = Seismic3DObnReceiverGathersTemplate(data_domain="time")
        schema = SchemaResolver().resolve(template, grid_overrides=None)

        shot_index = next(d for d in schema.dimensions if d.name == "shot_index")
        assert shot_index.is_calculated is True
        assert shot_index.is_spatial is True

    def test_cdp_required_header_fields(self) -> None:
        template = Seismic3DCdpGathersTemplate(data_domain="time", gather_domain="offset")
        schema = SchemaResolver().resolve(template, grid_overrides=None)

        # Spatial dim header keys + coordinate header keys + always-present coordinate_scalar.
        required = schema.required_header_fields()
        assert {"inline", "crossline", "offset", "cdp_x", "cdp_y", "coordinate_scalar"}.issubset(required)


class TestSchemaResolverNonBinned:
    def test_default_collapse_keeps_first_spatial_dim(self) -> None:
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        # Streamer shot template default chunk shape is (8, 1, 128, 2048).
        schema = SchemaResolver().resolve(template, GridOverrides(non_binned=True, chunksize=64))

        names = [d.name for d in schema.dimensions]
        assert names == ["shot_point", "trace", "time"]
        # shot_point keeps its original chunk (8); trace gets the override (64); vertical (2048) preserved.
        assert schema.chunk_shape == (8, 64, 2048)

    def test_explicit_replace_dims(self) -> None:
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        overrides = GridOverrides(non_binned=True, chunksize=128, replace_dims=["channel"])
        schema = SchemaResolver().resolve(template, overrides)

        names = [d.name for d in schema.dimensions]
        assert names == ["shot_point", "cable", "trace", "time"]
        # shot_point=8, cable=1 preserved; trace=128 (override); vertical=2048.
        assert schema.chunk_shape == (8, 1, 128, 2048)

    def test_coordinate_dimensions_collapsed_when_referenced(self) -> None:
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        schema = SchemaResolver().resolve(template, GridOverrides(non_binned=True, chunksize=64))
        # group_coord_x originally depends on (shot_point, cable, channel). After NonBinned
        # collapses cable+channel, it should depend on (shot_point, trace).
        group_coord_x = next(c for c in schema.coordinates if c.name == "group_coord_x")
        assert group_coord_x.dimensions == ("shot_point", "trace")

    def test_non_binned_flag_recorded_in_metadata(self) -> None:
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        overrides = GridOverrides(non_binned=True, chunksize=64)
        schema = SchemaResolver().resolve(template, overrides)
        assert "gridOverrides" in schema.metadata
        assert schema.metadata["gridOverrides"].get("NonBinned") is True


class TestSchemaResolverHasDuplicates:
    def test_inserts_trace_dim_with_chunksize_one(self) -> None:
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        schema = SchemaResolver().resolve(template, GridOverrides(has_duplicates=True))

        names = [d.name for d in schema.dimensions]
        assert names == ["shot_point", "cable", "channel", "trace", "time"]
        # Streamer shot default chunks (8, 1, 128, 2048); trace dim is a 1-wide chunk inserted
        # before the vertical dim.
        assert schema.chunk_shape == (8, 1, 128, 1, 2048)

    def test_has_duplicates_metadata(self) -> None:
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        schema = SchemaResolver().resolve(template, GridOverrides(has_duplicates=True))
        assert schema.metadata["gridOverrides"].get("HasDuplicates") is True


class TestGridOverridesLegacyTranslation:
    def test_calculate_shot_index_maps_to_auto_shot_wrap(self) -> None:
        overrides = GridOverrides.from_legacy_dict({"CalculateShotIndex": True})
        assert overrides.auto_shot_wrap is True

    def test_non_binned_dims_maps_to_replace_dims(self) -> None:
        overrides = GridOverrides.from_legacy_dict(
            {"NonBinned": True, "chunksize": 64, "non_binned_dims": ["cable", "channel"]}
        )
        assert overrides.replace_dims == ["cable", "channel"]

    def test_unknown_key_routed_to_extra_params(self) -> None:
        overrides = GridOverrides.from_legacy_dict({"NonBinned": True, "chunksize": 4, "future_flag": "x"})
        assert overrides.extra_params == {"future_flag": "x"}

    def test_modern_keys_passthrough(self) -> None:
        overrides = GridOverrides.from_legacy_dict({"auto_channel_wrap": True})
        assert overrides.auto_channel_wrap is True

    def test_empty_dict_returns_default(self) -> None:
        overrides = GridOverrides.from_legacy_dict({})
        assert not bool(overrides)


class TestGridOverridesValidation:
    def test_chunksize_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive|greater than"):
            GridOverrides(non_binned=True, chunksize=0)

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValueError, match="Extra inputs are not permitted|extra"):
            GridOverrides(unknown_thing=True)
