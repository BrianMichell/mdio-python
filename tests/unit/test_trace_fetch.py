"""Unit tests for trace worker fetch, decode, and statistics helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from segy.factory import SegyFactory
from segy.schema import Endianness
from segy.standards import get_segy_standard

from mdio.segy import _workers
from mdio.segy._workers import fetch_traces
from mdio.segy._workers import plan_trace_runs
from mdio.segy._workers import summarize_samples
from mdio.segy.file import SegyFileWrapper

if TYPE_CHECKING:
    from pathlib import Path

NUM_TRACES = 40
SAMPLES_PER_TRACE = 101


def test_plan_trace_runs_merges_consecutive_traces() -> None:
    """Consecutive trace indexes share a run; gaps start a new one."""
    assert plan_trace_runs(np.array([3, 4, 5, 9, 10, 20]), max_traces=100) == [(0, 3), (3, 2), (5, 1)]


def test_plan_trace_runs_splits_long_runs() -> None:
    """A run longer than the limit is split without cutting a trace."""
    assert plan_trace_runs(np.arange(10), max_traces=4) == [(0, 4), (4, 4), (8, 2)]


def test_plan_trace_runs_empty() -> None:
    """No traces means no runs."""
    assert plan_trace_runs(np.array([], dtype=np.int64), max_traces=4) == []


@pytest.fixture(
    params=[(Endianness.BIG, 1), (Endianness.BIG, 5), (Endianness.LITTLE, 1), (Endianness.LITTLE, 5)],
    ids=["big_ibm", "big_ieee", "little_ibm", "little_ieee"],
)
def segy_file(request: pytest.FixtureRequest, tmp_path: Path) -> SegyFileWrapper:
    """Small SEG-Y file with distinct samples per trace."""
    endianness, data_format = request.param
    spec = get_segy_standard(1.0)
    spec.endianness = endianness
    factory = SegyFactory(spec=spec, samples_per_trace=SAMPLES_PER_TRACE)
    headers = factory.create_trace_header_template(NUM_TRACES)
    samples = factory.create_trace_sample_template(NUM_TRACES)
    headers["trace_seq_num_line"] = np.arange(NUM_TRACES)
    samples[:] = np.arange(NUM_TRACES)[:, None] + np.linspace(0, 1, SAMPLES_PER_TRACE)

    path = tmp_path / "traces.segy"
    with path.open("wb") as f:
        f.write(factory.create_textual_header())
        f.write(factory.create_binary_header(update={"data_sample_format": data_format}))
        f.write(factory.create_traces(headers, samples))
    return SegyFileWrapper(path, spec=spec)


def test_fetch_traces_matches_segy_reader(segy_file: SegyFileWrapper, monkeypatch: pytest.MonkeyPatch) -> None:
    """Batched in-place decode returns the same traces and raw bytes as the SEG-Y library."""
    # Force several requests per batch and several batches.
    monkeypatch.setattr(_workers, "_FETCH_REQUEST_BYTES", 3 * segy_file.spec.trace.dtype.itemsize)
    monkeypatch.setattr(_workers, "_FETCH_BATCH_REQUESTS", 2)
    indexes = np.array([17, 3, 4, 5, 30, 31, 32, 33, 34, 0, 39, 12])

    order, traces, raw_headers = fetch_traces(segy_file, indexes, keep_raw_headers=True)

    np.testing.assert_array_equal(indexes[order], np.sort(indexes))
    expected = segy_file.trace[np.sort(indexes)]
    np.testing.assert_array_equal(traces["data"], expected.sample)
    np.testing.assert_array_equal(traces["header"]["trace_seq_num_line"], expected.header["trace_seq_num_line"])
    header_size = segy_file.spec.trace.header.itemsize
    trace_offset = segy_file.spec.trace.offset
    itemsize = segy_file.spec.trace.dtype.itemsize
    with open(segy_file.url, "rb") as f:  # noqa: PTH123
        on_disk = []
        for index in np.sort(indexes):
            f.seek(trace_offset + int(index) * itemsize)
            on_disk.append(f.read(header_size))
    assert [bytes(raw) for raw in raw_headers] == on_disk


def test_fetch_traces_without_raw_headers(segy_file: SegyFileWrapper) -> None:
    """Raw header bytes are only kept on request."""
    _, _, raw_headers = fetch_traces(segy_file, np.array([2, 1]), keep_raw_headers=False)

    assert raw_headers is None


def test_summarize_samples_matches_masked_reference() -> None:
    """Slabbed float64 statistics match the masked-array reference, excluding near-zero values."""
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((50, 700)).astype(np.float32)
    samples[:, ::7] = 0

    stats = summarize_samples(samples, slab=256)

    masked = np.ma.masked_values(samples, 0, copy=False)
    assert stats.count == masked.count()
    assert stats.min == pytest.approx(float(masked.min()))
    assert stats.max == pytest.approx(float(masked.max()))
    assert stats.sum == pytest.approx(float(masked.sum(dtype="float64")))
    assert stats.sum_squares == pytest.approx(float(np.ma.power(masked, 2).sum(dtype="float64")))


def test_summarize_samples_all_zero_returns_none() -> None:
    """All-zero input has no counted samples."""
    assert summarize_samples(np.zeros((4, 10), dtype=np.float32)) is None
