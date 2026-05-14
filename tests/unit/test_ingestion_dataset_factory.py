"""Unit tests for the v1.2 DatasetFactory and the ``apply_resolved_dimensions`` API.

Note:
    These tests intentionally exercise CDP/Streamer field templates whose
    ``_add_coordinates`` is compatible with the resolved schema layout. Templates
    whose overridden ``_add_coordinates`` hard-codes coordinate dimension names
    (e.g. ``Seismic3DStreamerShotGathersTemplate``) currently require a separate
    template-aware coord-dim rewrite for grid-override paths and are exercised end-
    to-end via the integration test suite, not here.
"""

from __future__ import annotations

import pytest

from mdio.builder.templates.seismic_3d_cdp import Seismic3DCdpGathersTemplate
from mdio.builder.templates.seismic_3d_streamer_shot import Seismic3DStreamerShotGathersTemplate
from mdio.core.dimension import Dimension
from mdio.ingestion.dataset_factory import DatasetFactory
from mdio.ingestion.schema_resolver import SchemaResolver


class TestDatasetFactoryBuild:
    def test_build_cdp_with_coords(self) -> None:
        template = Seismic3DCdpGathersTemplate(data_domain="time", gather_domain="offset")
        schema = SchemaResolver().resolve(template)
        dimensions = [
            Dimension(coords=[10, 11], name="inline"),
            Dimension(coords=[100, 101, 102], name="crossline"),
            Dimension(coords=[0, 100, 200], name="offset"),
            Dimension(coords=[0, 4], name="time"),
        ]
        ds = DatasetFactory().build(template, schema, dimensions)

        assert ds.metadata.name == template.name
        var_names = {v.name for v in ds.variables}
        assert {"cdp_x", "cdp_y", "trace_mask", "amplitude"}.issubset(var_names)
        # Dimension sizes are carried on each Variable's NamedDimension list.
        amplitude = next(v for v in ds.variables if v.name == "amplitude")
        assert [(d.name, d.size) for d in amplitude.dimensions] == [
            ("inline", 2),
            ("crossline", 3),
            ("offset", 3),
            ("time", 2),
        ]


class TestApplyResolvedDimensions:
    def test_round_trip(self) -> None:
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        new_dims = ("shot_point", "trace", "time")
        new_chunks = (8, 64, 1024)
        template.apply_resolved_dimensions(dim_names=new_dims, chunk_shape=new_chunks)
        assert template.dimension_names == new_dims
        # full_chunk_shape returns the stored shape until dim_sizes are set.
        assert template.full_chunk_shape == new_chunks

    def test_length_mismatch_raises(self) -> None:
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        with pytest.raises(ValueError, match="does not match"):
            template.apply_resolved_dimensions(dim_names=("a", "b"), chunk_shape=(1, 2, 3))
