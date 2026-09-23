"""Unit tests for blocked SEG-Y ingestion resource controls."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mdio.segy.blocked_io import _memory_bounded_worker_count


def test_memory_bounded_worker_count_preserves_traditional_cpu_limit() -> None:
    """Traditional 128-cube float32 blocks remain CPU-bound on a normal benchmark host."""
    memory = SimpleNamespace(total=32 * 1024**3, available=24 * 1024**3)
    with patch("mdio.segy.blocked_io.virtual_memory", return_value=memory):
        assert _memory_bounded_worker_count(16, (128, 128, 128), np.dtype(np.float32)) == 16


def test_memory_bounded_worker_count_bounds_large_shards() -> None:
    """Shard-sized write blocks reduce process count when the memory budget is tight."""
    memory = SimpleNamespace(total=16 * 1024**3, available=8 * 1024**3)
    with patch("mdio.segy.blocked_io.virtual_memory", return_value=memory):
        assert _memory_bounded_worker_count(16, (256, 256, 256), np.dtype(np.float32)) == 8


def test_memory_bounded_worker_count_never_disables_ingestion() -> None:
    """A host smaller than one estimated block still gets one worker."""
    memory = SimpleNamespace(total=1024**3, available=128 * 1024**2)
    with patch("mdio.segy.blocked_io.virtual_memory", return_value=memory):
        assert _memory_bounded_worker_count(16, (256, 256, 256), np.dtype(np.float32)) == 1
