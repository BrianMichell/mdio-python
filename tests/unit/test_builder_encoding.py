import numpy as np
from pathlib import Path

from mdio.core.v1.builder import MDIODatasetBuilder


def test_builder_populates_encodings(tmp_path: Path) -> None:
    builder = MDIODatasetBuilder("encodings")
    builder.add_dimension("x", 10)
    builder.add_variable(
        "var",
        metadata={"chunkGrid": {"name": "regular", "configuration": {"chunkShape": [5]}}},
    )
    ds = builder.to_mdio(str(tmp_path / "enc.mdio"))

    assert ds["x"].encoding["chunks"] == (10,)
    assert ds["var"].encoding["chunks"] == (5,)


def test_builder_populates_encodings_multiple(tmp_path: Path) -> None:
    """Ensure encodings are populated for a more complex dataset."""

    builder = MDIODatasetBuilder("encodings_multi")

    # Add several dimensions
    builder.add_dimension("x", 10)
    builder.add_dimension("y", 20)
    builder.add_dimension("z", 5)

    # Add coordinates using different combinations of dimensions
    builder.add_coordinate("x_coord", dimensions=["x"])
    builder.add_coordinate("yz_coord", dimensions=["y", "z"])

    # Add variables with explicit chunking across multiple dimensions
    builder.add_variable(
        "var_xy",
        dimensions=["x", "y"],
        metadata={"chunkGrid": {"name": "regular", "configuration": {"chunkShape": [5, 10]}}},
    )
    builder.add_variable(
        "var_xyz",
        dimensions=["x", "y", "z"],
        metadata={"chunkGrid": {"name": "regular", "configuration": {"chunkShape": [5, 10, 5]}}},
    )
    builder.add_variable(
        "var_z",
        dimensions=["z"],
        metadata={"chunkGrid": {"name": "regular", "configuration": {"chunkShape": [5]}}},
    )

    ds = builder.to_mdio(str(tmp_path / "enc_multi.mdio"))

    # Dimension variables should have full-size chunks
    assert ds["x"].encoding["chunks"] == (10,)
    assert ds["y"].encoding["chunks"] == (20,)
    assert ds["z"].encoding["chunks"] == (5,)

    # Coordinate variables should inherit dimension chunking
    assert ds["x_coord"].encoding["chunks"] == (10,)
    assert ds["yz_coord"].encoding["chunks"] == (20, 5)

    # Data variables should use the provided chunk shapes
    assert ds["var_xy"].encoding["chunks"] == (5, 10)
    assert ds["var_xyz"].encoding["chunks"] == (5, 10, 5)
    assert ds["var_z"].encoding["chunks"] == (5,)
