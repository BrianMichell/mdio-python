"""Dimension (grid) abstraction and serializers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from mdio.core.serialization import Serializer
from mdio.exceptions import ShapeError

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(eq=False, order=False, slots=True)
class Dimension:
    """Dimension class.

    Dimension has a name and coordinates associated with it. The Dimension coordinates can only
    be a vector.

    Args:
        coords: Vector of coordinates.
        name: Name of the dimension.

    Attributes:
        coords: Vector of coordinates.
        name: Name of the dimension.
    """

    coords: list | tuple | NDArray | range
    name: str

    def __post_init__(self) -> None:
        """Post process and validation."""
        self.coords = np.asarray(self.coords)
        if self.coords.ndim != 1:
            msg = "Dimensions can only have vector coordinates"
            raise ShapeError(msg, ("# Dim", "Expected"), (self.coords.ndim, 1))

    @property
    def size(self) -> int:
        """Size of the dimension."""
        return len(self.coords)

    def to_dict(self) -> dict[str, Any]:
        """Convert dimension to dictionary."""
        return {"name": self.name, "coords": self.coords.tolist()}

    @classmethod
    def from_dict(cls, other: dict[str, Any]) -> Dimension:
        """Make dimension from dictionary."""
        return Dimension(**other)

    def __len__(self) -> int:
        """Length magic."""
        return self.size

    def __getitem__(self, item: int | slice | list[int]) -> NDArray:
        """Gets a specific coordinate value by index."""
        return self.coords[item]

    def __setitem__(self, key: int, value: NDArray) -> None:
        """Sets a specific coordinate value by index."""
        self.coords[key] = value

    def __hash__(self) -> int:
        """Hashing magic."""
        return hash(tuple(self.coords) + (self.name,))

    def __eq__(self, other: Dimension) -> bool:
        """Compares if the dimension has same properties."""
        if not isinstance(other, Dimension):
            other_type = type(other).__name__
            msg = f"Can't compare Dimension with {other_type}"
            raise TypeError(msg)

        return hash(self) == hash(other)

    def min(self) -> NDArray[float]:
        """Get minimum value of dimension."""
        return np.min(self.coords)

    def max(self) -> NDArray[float]:
        """Get maximum value of dimension."""
        return np.max(self.coords)

    def serialize(self, stream_format: str) -> str:
        """Serialize the dimension into buffer."""
        serializer = DimensionSerializer(stream_format)
        return serializer.serialize(self)

    @classmethod
    def deserialize(cls, stream: str, stream_format: str) -> Dimension:
        """Deserialize buffer into Dimension."""
        serializer = DimensionSerializer(stream_format)
        return serializer.deserialize(stream)


class DimensionSerializer(Serializer):
    """Serializer implementation for Dimension."""

    def serialize(self, dimension: Dimension) -> str:
        """Serialize Dimension into buffer."""
        payload = {
            "name": dimension.name,
            "length": len(dimension),
            "coords": dimension.coords.tolist(),
        }
        return self.serialize_func(payload)

    def deserialize(self, stream: str) -> Dimension:
        """Deserialize buffer into Dimension."""
        signature = inspect.signature(Dimension)

        payload = self.deserialize_func(stream)
        payload = self.validate_payload(payload, signature)

        return Dimension(**payload)
