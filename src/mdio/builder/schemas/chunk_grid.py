"""This module contains data models for Zarr's chunk grid."""

from __future__ import annotations

from pydantic import Field
from pydantic import model_validator

from mdio.builder.schemas.core import CamelCaseStrictModel


class RegularChunkShape(CamelCaseStrictModel):
    """Represents regular chunk sizes along each dimension."""

    chunk_shape: tuple[int, ...] = Field(..., description="Lengths of the chunk along each dimension of the array.")


class RectilinearChunkShape(CamelCaseStrictModel):
    """Represents irregular chunk sizes along each dimension."""

    chunk_shape: tuple[tuple[int, ...], ...] = Field(
        ...,
        description="Lengths of the chunk along each dimension of the array.",
    )


class ShardedChunkShape(CamelCaseStrictModel):
    """Represents sharded chunk configuration for Zarr v3.

    In Zarr v3 sharding, data is organized in a two-level hierarchy:
    - Shards: Outer containers that are stored as individual files/objects
    - Chunks: Inner units within each shard for compression and access

    The shard_shape must be evenly divisible by chunk_shape along each dimension.
    """

    shard_shape: tuple[int, ...] = Field(
        ...,
        description="Lengths of the shard (outer container) along each dimension of the array.",
    )

    chunk_shape: tuple[int, ...] = Field(
        ...,
        description="Lengths of the chunk (inner unit) along each dimension within each shard.",
    )

    @model_validator(mode="after")
    def validate_shapes(self) -> ShardedChunkShape:
        """Validate that shard_shape is divisible by chunk_shape."""
        if len(self.shard_shape) != len(self.chunk_shape):
            msg = (
                f"shard_shape and chunk_shape must have the same number of dimensions. "
                f"Got shard_shape={self.shard_shape} and chunk_shape={self.chunk_shape}"
            )
            raise ValueError(msg)

        for i, (shard_dim, chunk_dim) in enumerate(zip(self.shard_shape, self.chunk_shape, strict=True)):
            if shard_dim % chunk_dim != 0:
                msg = (
                    f"shard_shape must be evenly divisible by chunk_shape along all dimensions. "
                    f"Dimension {i}: shard_shape[{i}]={shard_dim} is not divisible by chunk_shape[{i}]={chunk_dim}"
                )
                raise ValueError(msg)
            if shard_dim < chunk_dim:
                msg = (
                    f"shard_shape must be >= chunk_shape along all dimensions. "
                    f"Dimension {i}: shard_shape[{i}]={shard_dim} < chunk_shape[{i}]={chunk_dim}"
                )
                raise ValueError(msg)

        return self


class RegularChunkGrid(CamelCaseStrictModel):
    """Represents a rectangular and regularly spaced chunk grid."""

    name: str = Field(default="regular", description="The name of the chunk grid.")

    configuration: RegularChunkShape = Field(..., description="Configuration of the regular chunk grid.")


class RectilinearChunkGrid(CamelCaseStrictModel):
    """Represents a rectangular and irregularly spaced chunk grid."""

    name: str = Field(default="rectilinear", description="The name of the chunk grid.")

    configuration: RectilinearChunkShape = Field(..., description="Configuration of the irregular chunk grid.")


class ShardedChunkGrid(CamelCaseStrictModel):
    """Represents a sharded chunk grid for Zarr v3.

    Sharding enables efficient storage and access patterns by grouping multiple
    chunks into larger shards. Each shard is stored as a single object/file,
    reducing the number of I/O operations for sequential access patterns.
    """

    name: str = Field(default="sharded", description="The name of the chunk grid.")

    configuration: ShardedChunkShape = Field(..., description="Configuration of the sharded chunk grid.")
