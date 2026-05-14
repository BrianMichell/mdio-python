"""Unit tests for the v1.2 ingestion index strategies and the strategy registry.

These tests exercise individual strategies with synthetic structured numpy arrays
(mimicking SEG-Y :class:`segy.arrays.HeaderArray` shape semantics) so they remain
fast and do not require any real SEG-Y data.
"""

from __future__ import annotations

import numpy as np
import pytest

from mdio.builder.templates.seismic_3d_obn import Seismic3DObnReceiverGathersTemplate
from mdio.builder.templates.seismic_3d_streamer_field import Seismic3DStreamerFieldRecordsTemplate
from mdio.builder.templates.seismic_3d_streamer_shot import Seismic3DStreamerShotGathersTemplate
from mdio.ingestion.index_strategies import ChannelWrappingStrategy
from mdio.ingestion.index_strategies import ComponentSynthesisStrategy
from mdio.ingestion.index_strategies import CompositeStrategy
from mdio.ingestion.index_strategies import DuplicateHandlingStrategy
from mdio.ingestion.index_strategies import IndexStrategyRegistry
from mdio.ingestion.index_strategies import NonBinnedStrategy
from mdio.ingestion.index_strategies import RegularGridStrategy
from mdio.ingestion.index_strategies import ShotWrappingStrategy
from mdio.segy.geometry import GridOverrides


def _make_struct(data: dict[str, np.ndarray]) -> np.ndarray:
    """Build a 1-D structured array from a name -> 1-D array mapping."""
    names = list(data.keys())
    arrays = [data[name] for name in names]
    n = len(arrays[0])
    dtype = np.dtype([(name, arr.dtype) for name, arr in zip(names, arrays, strict=True)])
    out = np.empty(n, dtype=dtype)
    for name, arr in zip(names, arrays, strict=True):
        out[name] = arr
    return out


# ---------------------------------------------------------------------------
# IndexStrategyRegistry
# ---------------------------------------------------------------------------


class TestIndexStrategyRegistry:
    def test_default_returns_regular_grid(self) -> None:
        strategy = IndexStrategyRegistry().create_strategy(grid_overrides=None)
        assert isinstance(strategy, RegularGridStrategy)

    def test_falsy_overrides_returns_regular_grid(self) -> None:
        # GridOverrides() with no flags set must be treated as "no overrides".
        strategy = IndexStrategyRegistry().create_strategy(grid_overrides=GridOverrides())
        assert isinstance(strategy, RegularGridStrategy)

    def test_non_binned_only(self) -> None:
        overrides = GridOverrides(non_binned=True, chunksize=64)
        strategy = IndexStrategyRegistry().create_strategy(grid_overrides=overrides)
        assert isinstance(strategy, NonBinnedStrategy)
        assert strategy.chunksize == 64

    def test_has_duplicates_only(self) -> None:
        strategy = IndexStrategyRegistry().create_strategy(grid_overrides=GridOverrides(has_duplicates=True))
        assert isinstance(strategy, DuplicateHandlingStrategy)

    def test_non_binned_and_has_duplicates_are_mutually_exclusive(self) -> None:
        # Both flags set: non_binned wins (matches v1.x semantics) and HasDuplicates is dropped.
        strategy = IndexStrategyRegistry().create_strategy(
            grid_overrides=GridOverrides(non_binned=True, chunksize=8, has_duplicates=True)
        )
        assert isinstance(strategy, NonBinnedStrategy)

    def test_composite_with_channel_wrap(self) -> None:
        overrides = GridOverrides(auto_channel_wrap=True, non_binned=True, chunksize=64)
        strategy = IndexStrategyRegistry().create_strategy(grid_overrides=overrides)
        assert isinstance(strategy, CompositeStrategy)
        assert [s.name for s in strategy.strategies] == ["ChannelWrappingStrategy", "NonBinnedStrategy"]

    def test_synthesize_dims_is_first(self) -> None:
        # ComponentSynthesisStrategy must run BEFORE other strategies that may depend on
        # the synthesized field being present.
        overrides = GridOverrides(auto_shot_wrap=True)
        strategy = IndexStrategyRegistry().create_strategy(
            grid_overrides=overrides,
            synthesize_dims=("component",),
        )
        assert isinstance(strategy, CompositeStrategy)
        assert strategy.strategies[0].name == "ComponentSynthesisStrategy"

    def test_template_drives_shot_wrap_for_obn(self) -> None:
        # OBN: shot_line + shot_index calculated -> always_calculate=True, line_field='shot_line'.
        template = Seismic3DObnReceiverGathersTemplate(data_domain="time")
        line_field, always_calculate = IndexStrategyRegistry._resolve_shot_wrap_params(template)
        assert line_field == "shot_line"
        assert always_calculate is True

    def test_template_drives_shot_wrap_for_streamer_field(self) -> None:
        # Streamer field records: sail_line + shot_index calculated.
        template = Seismic3DStreamerFieldRecordsTemplate(data_domain="time")
        line_field, always_calculate = IndexStrategyRegistry._resolve_shot_wrap_params(template)
        assert line_field == "sail_line"
        assert always_calculate is True

    def test_template_drives_shot_wrap_for_streamer_shot(self) -> None:
        # Streamer shot gathers: sail_line, no calculated shot_index -> Type-B-only.
        template = Seismic3DStreamerShotGathersTemplate(data_domain="time")
        line_field, always_calculate = IndexStrategyRegistry._resolve_shot_wrap_params(template)
        assert line_field == "sail_line"
        assert always_calculate is False

    def test_no_template_falls_back_to_streamer_defaults(self) -> None:
        line_field, always_calculate = IndexStrategyRegistry._resolve_shot_wrap_params(None)
        assert line_field == "sail_line"
        assert always_calculate is False


# ---------------------------------------------------------------------------
# RegularGridStrategy
# ---------------------------------------------------------------------------


class TestRegularGridStrategy:
    def test_returns_unique_dims(self) -> None:
        headers = _make_struct(
            {
                "inline": np.array([1, 1, 2, 2], dtype=np.int32),
                "crossline": np.array([10, 11, 10, 11], dtype=np.int32),
            }
        )
        dims = RegularGridStrategy().compute_dimensions(headers, ("inline", "crossline"))
        assert [d.name for d in dims] == ["inline", "crossline"]
        np.testing.assert_array_equal(dims[0].coords, [1, 2])
        np.testing.assert_array_equal(dims[1].coords, [10, 11])

    def test_unknown_dim_silently_skipped(self) -> None:
        headers = _make_struct({"inline": np.array([1, 2], dtype=np.int32)})
        dims = RegularGridStrategy().compute_dimensions(headers, ("inline", "missing"))
        assert [d.name for d in dims] == ["inline"]


# ---------------------------------------------------------------------------
# NonBinnedStrategy
# ---------------------------------------------------------------------------


class TestNonBinnedStrategy:
    def test_appends_sequential_trace_field(self) -> None:
        headers = _make_struct(
            {
                "shot_point": np.array([1, 1, 2, 2], dtype=np.int32),
                "channel": np.array([1, 2, 1, 2], dtype=np.int32),
            }
        )
        out = NonBinnedStrategy(chunksize=4).transform_headers(headers)
        assert "trace" in out.dtype.names
        np.testing.assert_array_equal(out["trace"], [0, 1, 2, 3])

    def test_default_collapse_keeps_first_spatial_dim(self) -> None:
        headers = _make_struct(
            {
                "shot_point": np.array([1, 1, 2, 2], dtype=np.int32),
                "channel": np.array([1, 2, 1, 2], dtype=np.int32),
            }
        )
        strategy = NonBinnedStrategy(chunksize=4)
        out = strategy.transform_headers(headers)
        dims = strategy.compute_dimensions(out, ("shot_point", "channel"))
        assert [d.name for d in dims] == ["shot_point", "trace"]
        # trace dim has unique coords for every input row
        assert len(dims[1].coords) == 4

    def test_explicit_collapse_dims(self) -> None:
        headers = _make_struct(
            {
                "shot_point": np.array([1, 1, 2, 2], dtype=np.int32),
                "channel": np.array([1, 2, 1, 2], dtype=np.int32),
            }
        )
        strategy = NonBinnedStrategy(chunksize=4, collapse_dims=["channel"])
        out = strategy.transform_headers(headers)
        dims = strategy.compute_dimensions(out, ("shot_point", "channel"))
        # 'channel' is collapsed into 'trace'; 'shot_point' is preserved
        assert [d.name for d in dims] == ["shot_point", "trace"]


# ---------------------------------------------------------------------------
# DuplicateHandlingStrategy
# ---------------------------------------------------------------------------


class TestDuplicateHandlingStrategy:
    def test_appends_per_dim_duplicate_counter(self) -> None:
        headers = _make_struct(
            {
                "inline": np.array([1, 1, 1, 2], dtype=np.int32),
                "crossline": np.array([10, 10, 11, 10], dtype=np.int32),
            }
        )
        strategy = DuplicateHandlingStrategy()
        out = strategy.transform_headers(headers)
        assert "trace" in out.dtype.names
        # Each (inline, crossline) tuple gets a 1-based duplicate counter.
        # (1,10) appears twice -> counters {1, 2}; (1,11) once -> {1}; (2,10) once -> {1}.
        np.testing.assert_array_equal(np.sort(out["trace"]), [1, 1, 1, 2])


# ---------------------------------------------------------------------------
# ChannelWrappingStrategy
# ---------------------------------------------------------------------------


class TestChannelWrappingStrategy:
    def test_type_a_pass_through(self) -> None:
        # Type A: per-cable channel numbering (overlapping ranges) -> headers untouched.
        headers = _make_struct(
            {
                "cable": np.array([1, 1, 2, 2], dtype=np.int32),
                "channel": np.array([1, 2, 1, 2], dtype=np.int32),
            }
        )
        out = ChannelWrappingStrategy().transform_headers(headers)
        np.testing.assert_array_equal(out["channel"], [1, 2, 1, 2])

    def test_type_b_renumbers_per_cable(self) -> None:
        # Type B: channels numbered sequentially across cables -> rebased to 1..N per cable.
        headers = _make_struct(
            {
                "cable": np.array([1, 1, 2, 2], dtype=np.int32),
                "channel": np.array([1, 2, 3, 4], dtype=np.int32),
            }
        )
        out = ChannelWrappingStrategy().transform_headers(headers)
        # Cable 1: 1,2 -> 1,2 ; Cable 2: 3,4 -> 1,2.
        np.testing.assert_array_equal(out["channel"], [1, 2, 1, 2])


# ---------------------------------------------------------------------------
# ShotWrappingStrategy
# ---------------------------------------------------------------------------


class TestShotWrappingStrategy:
    def test_type_b_streamer_emits_shot_index(self) -> None:
        # Sail line 1 with two guns interleaving shot_point: 1,2,3,4 across guns 1,2.
        headers = _make_struct(
            {
                "sail_line": np.array([1, 1, 1, 1], dtype=np.int32),
                "gun": np.array([1, 2, 1, 2], dtype=np.int32),
                "shot_point": np.array([1, 2, 3, 4], dtype=np.int32),
            }
        )
        out = ShotWrappingStrategy(line_field="sail_line").transform_headers(headers)
        assert "shot_index" in out.dtype.names
        # floor(shot_point / 2): 0, 1, 1, 2 -> rebased to 0-based per line: 0,1,1,2
        # min is 0, so values stay as is. Each gun sees indices [0,1] and [1,2].
        np.testing.assert_array_equal(out["shot_index"], [0, 1, 1, 2])

    def test_type_a_streamer_skipped_without_always_calculate(self) -> None:
        # Type A geometry: per gun shots [1, 2, 3] -> floor(/2) yields [0, 1, 1] (2 unique != 3).
        headers = _make_struct(
            {
                "sail_line": np.array([1, 1, 1, 1, 1, 1], dtype=np.int32),
                "gun": np.array([1, 1, 1, 2, 2, 2], dtype=np.int32),
                "shot_point": np.array([1, 2, 3, 1, 2, 3], dtype=np.int32),
            }
        )
        out = ShotWrappingStrategy(line_field="sail_line", always_calculate=False).transform_headers(headers)
        assert "shot_index" not in out.dtype.names

    def test_type_a_obn_always_calculates(self) -> None:
        # OBN-style Type A: per-gun shots [1,2,3,4] collide under floor(/2) so geometry
        # detection picks Type A. always_calculate=True forces a dense per-line index built
        # via searchsorted over the sorted unique shot_point values.
        headers = _make_struct(
            {
                "shot_line": np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
                "gun": np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32),
                "shot_point": np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.int32),
            }
        )
        out = ShotWrappingStrategy(line_field="shot_line", always_calculate=True).transform_headers(headers)
        assert "shot_index" in out.dtype.names
        np.testing.assert_array_equal(out["shot_index"], [0, 1, 2, 3, 0, 1, 2, 3])

    def test_obn_multiline_type_a_processes_all_lines(self) -> None:
        # Regression for v1.x bug: Type A detection on line 1 used to drop later lines.
        headers = _make_struct(
            {
                "shot_line": np.array([1, 1, 2, 2, 3, 3], dtype=np.int32),
                "gun": np.array([1, 2, 1, 2, 1, 2], dtype=np.int32),
                "shot_point": np.array([1, 2, 1, 2, 1, 2], dtype=np.int32),
            }
        )
        out = ShotWrappingStrategy(line_field="shot_line", always_calculate=True).transform_headers(headers)
        # Each line independently gets dense per-line indices over its unique shot_points.
        np.testing.assert_array_equal(out["shot_index"], [0, 1, 0, 1, 0, 1])


# ---------------------------------------------------------------------------
# ComponentSynthesisStrategy
# ---------------------------------------------------------------------------


class TestComponentSynthesisStrategy:
    def test_synthesizes_missing_field(self) -> None:
        headers = _make_struct({"receiver": np.array([1, 2, 3], dtype=np.int32)})
        out = ComponentSynthesisStrategy(("component",)).transform_headers(headers)
        assert "component" in out.dtype.names
        np.testing.assert_array_equal(out["component"], [1, 1, 1])

    def test_existing_field_left_alone(self) -> None:
        headers = _make_struct(
            {
                "receiver": np.array([1, 2, 3], dtype=np.int32),
                "component": np.array([2, 3, 4], dtype=np.uint8),
            }
        )
        out = ComponentSynthesisStrategy(("component",)).transform_headers(headers)
        np.testing.assert_array_equal(out["component"], [2, 3, 4])


# ---------------------------------------------------------------------------
# CompositeStrategy
# ---------------------------------------------------------------------------


class TestCompositeStrategy:
    def test_requires_at_least_one_strategy(self) -> None:
        with pytest.raises(ValueError, match="at least one strategy"):
            CompositeStrategy([])

    def test_strategies_run_in_order(self) -> None:
        # Synthesize 'component' first, then NonBinned collapses 'channel' into trace.
        headers = _make_struct(
            {
                "shot_point": np.array([1, 1, 2, 2], dtype=np.int32),
                "channel": np.array([1, 2, 1, 2], dtype=np.int32),
            }
        )
        composite = CompositeStrategy([ComponentSynthesisStrategy(("component",)), NonBinnedStrategy(chunksize=4)])
        out = composite.transform_headers(headers)
        assert "component" in out.dtype.names
        assert "trace" in out.dtype.names
