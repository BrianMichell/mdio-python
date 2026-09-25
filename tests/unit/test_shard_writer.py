"""Unit tests for streaming Zarr v3 shards from trace columns."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

from mdio.segy._shard_writer import ShardStreamWriter
from mdio.segy._workers import _ShardSink
from mdio.segy._workers import _sub_blocks

if TYPE_CHECKING:
    from pathlib import Path

# Edge shards in every dimension, a partial last time chunk, and a time shard past the data.
SHAPE = (6, 7, 21)
SHARDS = (4, 4, 16)
CHUNKS = (2, 2, 8)


def _create(path: Path) -> zarr.Array:
    return zarr.create_array(
        store=str(path),
        shape=SHAPE,
        chunks=CHUNKS,
        shards=SHARDS,
        dtype="float32",
        fill_value=0.0,
        zarr_format=3,
    )


def _reference_data() -> tuple[np.ndarray, np.ndarray]:
    """Data with dead traces, so some inner chunks and one whole time shard are empty."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(SHAPE).astype(np.float32)
    live = np.ones(SHAPE[:-1], dtype=bool)
    live[:2, :2] = False  # one full inner-chunk column is dead
    live[5, 6] = False
    data[~live] = 0
    data[4:, 4:, 16:] = 0  # the far-corner last time shard holds only fill values
    return data, live


def _stream(array: zarr.Array, root: Path, data: np.ndarray, live: np.ndarray) -> None:
    writer = ShardStreamWriter(array, str(root), None)
    with ThreadPoolExecutor(2) as pool:
        for region in _sub_blocks(SHAPE[:-1], SHARDS[:-1]):
            sink = _ShardSink(writer, region, SHAPE[-1], pool)
            region_live = live[region]
            for block in _sub_blocks(region_live.shape, CHUNKS[:-1]):
                block_live = np.nonzero(region_live[block])
                if block_live[0].size == 0:
                    continue
                # Reverse the rows to prove chunk placement follows `live`, not trace order.
                rows = tuple(axis[::-1] for axis in block_live)
                samples = data[region][block][rows]
                sink.write(block, rows, samples)
            sink.close()


def _objects(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in (root / "c").rglob("*") if p.is_file()}


def test_streamed_shards_match_zarr_writes(tmp_path: Path) -> None:
    """Streamed shards decode to the same values and hold the same objects as a Zarr write."""
    data, live = _reference_data()
    expected_root = tmp_path / "expected.zarr"
    _create(expected_root)[:] = data

    streamed_root = tmp_path / "streamed.zarr"
    _stream(_create(streamed_root), streamed_root, data, live)

    np.testing.assert_array_equal(zarr.open_array(str(streamed_root))[:], data)
    expected, streamed = _objects(expected_root), _objects(streamed_root)
    assert sorted(streamed) == sorted(expected)
    assert "c/1/1/1" not in streamed  # the all-fill shard writes no object, as in Zarr
    for key in expected:
        assert len(streamed[key]) == len(expected[key])


def test_discard_removes_partial_shard(tmp_path: Path) -> None:
    """A failed column leaves no partial shard object behind."""
    root = tmp_path / "partial.zarr"
    writer = ShardStreamWriter(_create(root), str(root), None)
    stream = writer.open((0, 0, 0))
    stream.append((0, 0, 0), writer.encode(np.ones(CHUNKS, dtype=np.float32)))

    stream.discard()

    assert _objects(root) == {}


def test_rejects_unsharded_arrays(tmp_path: Path) -> None:
    """Only sharded arrays can be streamed."""
    array = zarr.create_array(store=str(tmp_path / "plain.zarr"), shape=SHAPE, chunks=CHUNKS, dtype="float32")

    with pytest.raises(ValueError, match="sharding codec"):
        ShardStreamWriter(array, str(tmp_path / "plain.zarr"), None)
