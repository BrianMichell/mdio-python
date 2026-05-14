"""Unit tests for the v1.2 HeaderAnalyzer / HeaderRequirements."""

from __future__ import annotations

from mdio.builder.templates.seismic_3d_cdp import Seismic3DCdpGathersTemplate
from mdio.builder.templates.seismic_3d_streamer_shot import Seismic3DStreamerShotGathersTemplate
from mdio.ingestion.header_analyzer import HeaderAnalyzer
from mdio.ingestion.header_analyzer import HeaderRequirements
from mdio.ingestion.schema_resolver import SchemaResolver


class TestHeaderRequirements:
    def test_required_only(self) -> None:
        req = HeaderRequirements(required_fields={"inline", "crossline"})
        assert req.all_fields() == {"inline", "crossline"}

    def test_optional_merged(self) -> None:
        req = HeaderRequirements(
            required_fields={"inline"},
            optional_fields={"shot_point"},
        )
        assert req.all_fields() == {"inline", "shot_point"}


class TestRequirementsFromSchema:
    def test_streamer_schema_includes_coordinate_scalar(self) -> None:
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        schema = SchemaResolver().resolve(template)
        req = HeaderAnalyzer.requirements_from_schema(schema)
        assert "coordinate_scalar" in req.required_fields
        assert {"shot_point", "cable", "channel"}.issubset(req.required_fields)

    def test_cdp_schema_includes_coords(self) -> None:
        template = Seismic3DCdpGathersTemplate(data_domain="time", gather_domain="offset")
        schema = SchemaResolver().resolve(template)
        req = HeaderAnalyzer.requirements_from_schema(schema)
        # CDP coords + dims + always-present coordinate_scalar.
        assert {"inline", "crossline", "offset", "cdp_x", "cdp_y", "coordinate_scalar"}.issubset(req.required_fields)
