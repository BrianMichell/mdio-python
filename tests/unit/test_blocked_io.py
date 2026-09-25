"""Unit tests for blocked SEG-Y ingestion resource controls."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mdio.segy.blocked_io import _estimate_worker_bytes
from mdio.segy.blocked_io import _memory_bounded_worker_count

GIB = 1024**3
FLOAT32 = np.dtype(np.float32)
HOST_32GIB = SimpleNamespace(total=32 * GIB, available=28 * GIB)
N_SAMPLES = 2000
SHARD_COLUMN = (256, 256, N_SAMPLES)
INNER_CHUNKS = (128, 128, 128)
SHARDS = (256, 256, 256)


def _workers_on(memory: SimpleNamespace, worker_bytes: int) -> int:
    with patch("mdio.segy.blocked_io.virtual_memory", return_value=memory):
        return _memory_bounded_worker_count(16, worker_bytes)


def test_traditional_columns_keep_all_cpus() -> None:
    """128x128 trace columns written whole keep 16 CPUs on a 32 GiB host."""
    worker_bytes = _estimate_worker_bytes((128, 128, N_SAMPLES), FLOAT32)

    assert _workers_on(HOST_32GIB, worker_bytes) == 16


def test_streamed_shard_worker_is_no_larger_than_traditional() -> None:
    """Streaming holds one inner-chunk column, so a 256x256 shard worker costs about a 128x128 column."""
    traditional = _estimate_worker_bytes((128, 128, N_SAMPLES), FLOAT32)
    sharded = _estimate_worker_bytes(SHARD_COLUMN, FLOAT32, inner_chunks=INNER_CHUNKS, shards=SHARDS)

    assert sharded < traditional


def test_shard_workers_fit_a_small_host() -> None:
    """An 8 GiB host still runs several streamed shard workers."""
    worker_bytes = _estimate_worker_bytes(SHARD_COLUMN, FLOAT32, inner_chunks=INNER_CHUNKS, shards=SHARDS)

    assert _workers_on(SimpleNamespace(total=8 * GIB, available=6 * GIB), worker_bytes) == 7


def test_worker_count_never_disables_ingestion() -> None:
    """A host smaller than one estimated worker still gets one worker."""
    worker_bytes = _estimate_worker_bytes(SHARD_COLUMN, FLOAT32, inner_chunks=INNER_CHUNKS, shards=SHARDS)

    assert _workers_on(SimpleNamespace(total=GIB, available=GIB // 8), worker_bytes) == 1
