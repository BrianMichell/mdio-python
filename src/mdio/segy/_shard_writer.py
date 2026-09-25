"""Stream Zarr v3 shards straight from SEG-Y trace reads.

A shard needs every trace of its spatial extent, and SEG-Y yields whole traces, so writing a shard
through Zarr means holding the full shard column in memory. The shard format does not need that:
inner chunks may sit in any order inside the object, and the index sits at the end. This module
encodes each inner chunk as soon as its traces are read and appends it to a streaming upload of
its shard, so a worker only holds one inner-chunk column of traces at a time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fsspec.core import url_to_fs
from zarr.codecs.sharding import ShardingCodec
from zarr.codecs.sharding import _ShardIndex
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.chunk_utils import encode_or_elide_chunk
from zarr.registry import get_ndbuffer_class

if TYPE_CHECKING:
    from typing import Any

    import numpy as np
    from fsspec.spec import AbstractBufferedFile
    from zarr import Array as zarr_Array
    from zarr.core.buffer import Buffer

# Smallest multipart part S3 accepts. The upload buffer flushes a part once it holds this much,
# so each open shard keeps about one encoded inner chunk resident.
UPLOAD_BLOCK_BYTES = 5 * 1024**2


class ShardStreamWriter:
    """Encode inner chunks with an array's own codecs and stream them into shard objects.

    Args:
        data_array: Sharded Zarr v3 array whose metadata already exists in the store.
        output_path: URL of the store root.
        storage_options: fsspec options for the store.

    Raises:
        ValueError: If the array is not a single sharding codec with the index at the end.
    """

    def __init__(self, data_array: zarr_Array, output_path: str, storage_options: dict[str, Any] | None) -> None:
        metadata = data_array.metadata
        codec = metadata.codecs[0] if len(metadata.codecs) == 1 else None
        if not isinstance(codec, ShardingCodec) or codec.index_location != "end":
            msg = f"Streaming needs one sharding codec with a trailing index; got {metadata.codecs}."
            raise ValueError(msg)

        shard_spec = ArraySpec(
            shape=data_array.shards,
            dtype=metadata.dtype,
            fill_value=metadata.fill_value,
            config=data_array.config,
            prototype=default_buffer_prototype(),
        )
        self.chunk_shape = tuple(codec.chunk_shape)
        self.shard_shape = tuple(data_array.shards)
        self.chunks_per_shard = tuple(s // c for s, c in zip(self.shard_shape, self.chunk_shape, strict=True))
        self.dtype = data_array.dtype
        self.fill_value = data_array.fill_value
        self._codec = codec
        self._chunk_spec = codec._get_chunk_spec(shard_spec)
        self._encode_chunk = codec._get_inner_chunk_transform(shard_spec).encode_chunk
        self._metadata = metadata
        self._fs, root = url_to_fs(output_path, **(storage_options or {}))
        self._local = "file" in self._fs.protocol
        self._array_root = f"{root.rstrip('/')}/{data_array.path}"

    def encode(self, chunk: np.ndarray) -> Buffer | None:
        """Encode one full-shape inner chunk. None means it equals the fill value and is omitted."""
        return encode_or_elide_chunk(get_ndbuffer_class().from_numpy_array(chunk), self._chunk_spec, self._encode_chunk)

    def open(self, shard_coords: tuple[int, ...]) -> ShardStream:
        """Start a shard. Nothing is written until its first non-empty chunk arrives."""
        return ShardStream(self, f"{self._array_root}/{self._metadata.encode_chunk_key(shard_coords)}")


class ShardStream:
    """One shard object being written; inner chunks append in arrival order.

    Args:
        writer: Writer that owns the codecs and filesystem.
        path: Filesystem path of the shard object.
    """

    def __init__(self, writer: ShardStreamWriter, path: str) -> None:
        self._writer = writer
        self._path = path
        self._file: AbstractBufferedFile | None = None
        self._offset = 0
        self._index = _ShardIndex.create_empty(writer.chunks_per_shard)

    def append(self, chunk_coords: tuple[int, ...], encoded: Buffer | None) -> None:
        """Append one encoded inner chunk at its coordinates within the shard."""
        if encoded is None:
            return
        if self._file is None:
            fs = self._writer._fs
            if self._writer._local:
                fs.makedirs(fs._parent(self._path), exist_ok=True)
            self._file = fs.open(self._path, "wb", block_size=UPLOAD_BLOCK_BYTES)
        data = encoded.as_numpy_array()
        self._file.write(data)
        self._index.set_chunk_slice(chunk_coords, slice(self._offset, self._offset + data.size))
        self._offset += data.size

    def close(self) -> None:
        """Append the index and commit. A shard with no chunks writes no object, as in Zarr.

        Raises:
            RuntimeError: If the encoded index size differs from what readers expect.
        """
        if self._file is None:
            return
        codec = self._writer._codec
        index_bytes = codec._encode_shard_index_sync(self._index)
        if len(index_bytes) != codec._shard_index_size(self._writer.chunks_per_shard):
            msg = "Encoded shard index size differs from the size readers expect."
            raise RuntimeError(msg)
        self._file.write(index_bytes.as_numpy_array())
        self._file.close()
        self._file = None

    def discard(self) -> None:
        """Abandon a partial shard: abort the multipart upload, or remove a local partial file."""
        if self._file is None:
            return
        if self._writer._local:
            self._file.close()
            self._writer._fs.rm(self._path)
        else:
            self._file.discard()
        self._file = None
