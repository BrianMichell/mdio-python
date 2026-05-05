"""Tests for dataset factory."""

from mdio.builder.templates.seismic_3d_streamer_shot import Seismic3DStreamerShotGathersTemplate
from mdio.core.dimension import Dimension
from mdio.ingestion.dataset_factory import DatasetFactory
from mdio.ingestion.schema_resolver import SchemaResolver


def test_dataset_factory_build():
    template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
    resolver = SchemaResolver()
    schema = resolver.resolve(template)

    dimensions = [
        Dimension(coords=[1, 2], name="shot_point"),
        Dimension(coords=[1], name="cable"),
        Dimension(coords=[1, 2, 3], name="channel"),
        Dimension(coords=[0, 4, 8], name="time"),
    ]

    factory = DatasetFactory()
    ds = factory.build(template, schema, dimensions)

    assert ds.name == "StreamerShotGathers3D"
    assert len(ds.dimensions) == 4
    assert ds.dimensions[0].size == 2
    assert ds.dimensions[1].size == 1
    assert ds.dimensions[2].size == 3
    assert ds.dimensions[3].size == 3
