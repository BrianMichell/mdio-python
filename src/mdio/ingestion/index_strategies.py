"""Index strategies for transforming SEG-Y headers into indexable dimensions."""

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib import recfunctions as rfn

from mdio.core import Dimension
from mdio.ingestion.header_analysis import ShotGunGeometryType
from mdio.ingestion.header_analysis import StreamerShotGeometryType
from mdio.ingestion.header_analysis import analyze_lines_for_guns
from mdio.ingestion.header_analysis import analyze_non_indexed_headers
from mdio.ingestion.header_analysis import analyze_streamer_headers

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from segy.arrays import HeaderArray

    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.segy.geometry import GridOverrides

logger = logging.getLogger(__name__)


class IndexStrategy(ABC):
    """Base class for indexing strategies.

    Each strategy knows how to:
    1. Transform headers (add/modify header fields)
    2. Compute dimensions from the transformed headers
    """

    @abstractmethod
    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Transform headers for indexing.

        Args:
            headers: Input header array

        Returns:
            Transformed header array (may have additional fields)
        """

    @abstractmethod
    def compute_dimensions(self, headers: HeaderArray, dim_names: tuple[str, ...]) -> list[Dimension]:
        """Compute dimensions from headers.

        Args:
            headers: Transformed header array
            dim_names: Expected dimension names from schema

        Returns:
            List of Dimension objects
        """

    @property
    def name(self) -> str:
        """Return strategy name."""
        return self.__class__.__name__


class RegularGridStrategy(IndexStrategy):
    """Standard grid indexing without transformations.

    This is the default strategy when no grid overrides are specified.
    Headers are used as-is to build dimensions.
    """

    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """No transformation needed for regular grids."""
        return headers

    def compute_dimensions(self, headers: HeaderArray, dim_names: tuple[str, ...]) -> list[Dimension]:
        """Compute dimensions from header values."""
        dimensions = []

        for dim_name in dim_names:
            if dim_name in headers.dtype.names:
                coords = np.unique(headers[dim_name])
                dimensions.append(Dimension(coords=coords, name=dim_name))

        return dimensions


class NonBinnedStrategy(IndexStrategy):
    """Sequential trace indexing for non-binned data.

    Creates a "trace" dimension by sequentially numbering traces,
    replacing specified spatial dimensions.
    """

    def __init__(self, chunksize: int, collapse_dims: list[str] | None = None):
        """Initialize non-binned strategy.

        Args:
            chunksize: Chunk size for the trace dimension
            collapse_dims: Dimension names to collapse into trace.
                If None, inferred from dim_names in compute_dimensions.
        """
        self.chunksize = chunksize
        self.collapse_dims = collapse_dims or []

    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Append a sequential ``trace`` field (0, 1, 2, ...) to the headers."""
        trace_idx = np.arange(len(headers), dtype=np.int32)
        return rfn.append_fields(headers, "trace", trace_idx, usemask=False)

    def compute_dimensions(self, headers: HeaderArray, dim_names: tuple[str, ...]) -> list[Dimension]:
        """Compute dimensions with ``trace`` replacing collapsed dims."""
        if not self.collapse_dims:
            spatial_dims = [d for d in dim_names if d in headers.dtype.names]
            if len(spatial_dims) > 1:
                self.collapse_dims = spatial_dims[1:]

        dimensions: list[Dimension] = []
        trace_added = False

        for dim_name in dim_names:
            if dim_name in self.collapse_dims:
                if not trace_added and "trace" in headers.dtype.names:
                    dimensions.append(Dimension(coords=np.unique(headers["trace"]), name="trace"))
                    trace_added = True
                continue

            if dim_name == "trace" and not trace_added:
                dimensions.append(Dimension(coords=np.unique(headers["trace"]), name="trace"))
                trace_added = True
            elif dim_name in headers.dtype.names:
                dimensions.append(Dimension(coords=np.unique(headers[dim_name]), name=dim_name))

        return dimensions


class DuplicateHandlingStrategy(IndexStrategy):
    """Handle duplicate indices by adding trace dimension.

    Similar to NonBinned but uses a fixed chunksize of 1 and doesn't
    collapse other dimensions.
    """

    def __init__(self, dtype: DTypeLike = np.int16):
        """Initialize duplicate handling strategy.

        Args:
            dtype: Data type for trace index
        """
        self.dtype = dtype

    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Add trace index for duplicates."""
        return analyze_non_indexed_headers(headers, dtype=self.dtype)

    def compute_dimensions(self, headers: HeaderArray, dim_names: tuple[str, ...]) -> list[Dimension]:
        """Compute dimensions including trace for duplicates."""
        dimensions = []

        for dim_name in dim_names:
            if dim_name in headers.dtype.names:
                coords = np.unique(headers[dim_name])
                dimensions.append(Dimension(coords=coords, name=dim_name))

        return dimensions


class ChannelWrappingStrategy(IndexStrategy):
    """Handle streamer acquisition channel wrapping (Type A/B).

    Analyzes channel numbering across cables and adjusts for Type B
    (sequential numbering) to Type A (per-cable numbering).
    """

    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Adjust channel numbers based on geometry type."""
        unique_cables, cable_chan_min, cable_chan_max, geom_type = analyze_streamer_headers(headers)

        logger.info("Ingesting dataset as %s", geom_type.name)
        for cable, chan_min, chan_max in zip(unique_cables, cable_chan_min, cable_chan_max, strict=True):
            logger.info("Cable: %s has min chan: %s and max chan: %s", cable, chan_min, chan_max)

        if geom_type == StreamerShotGeometryType.B:
            for idx, cable in enumerate(unique_cables):
                cable_idxs = np.where(headers["cable"][:] == cable)
                headers["channel"][cable_idxs] = headers["channel"][cable_idxs] - cable_chan_min[idx] + 1

        return headers

    def compute_dimensions(self, headers: HeaderArray, dim_names: tuple[str, ...]) -> list[Dimension]:
        """Compute dimensions after channel wrapping."""
        dimensions = []

        for dim_name in dim_names:
            if dim_name in headers.dtype.names:
                coords = np.unique(headers[dim_name])
                dimensions.append(Dimension(coords=coords, name=dim_name))

        return dimensions


class ShotWrappingStrategy(IndexStrategy):
    """Handle multi-gun acquisition shot wrapping.

    Creates ``shot_index`` from ``shot_point`` for multi-gun acquisitions. The line
    field (``sail_line`` for streamer field records, ``shot_line`` for OBN, etc.) and
    whether ``shot_index`` must always be created (vs. only for Type B / interleaved
    geometries) are template-driven.

    Args:
        line_field: Name of the line header used to group shots (``sail_line`` or
            ``shot_line``). Defaults to ``sail_line`` to match streamer field records.
        always_calculate: When ``True`` (used by templates that declare ``shot_index``
            as a calculated dimension, e.g. OBN), ``shot_index`` is always emitted using
            either the Type B division-by-num_guns rule or, for Type A, a 0-based
            ``np.searchsorted`` over the sorted unique shot points per line. When
            ``False`` (default streamer behavior), ``shot_index`` is only emitted for
            Type B geometries.
    """

    def __init__(self, line_field: str = "sail_line", always_calculate: bool = False) -> None:
        self.line_field = line_field
        self.always_calculate = always_calculate

    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Create ``shot_index`` for multi-gun acquisition."""
        unique_lines, unique_guns_per_line, geom_type = analyze_lines_for_guns(headers, line_field=self.line_field)

        logger.info("Ingesting dataset as shot type: %s (line_field=%s)", geom_type.name, self.line_field)

        max_num_guns = 1
        for line_val in unique_lines:
            guns = unique_guns_per_line[str(line_val)]
            logger.info("%s: %s has guns: %s", self.line_field, line_val, guns)
            max_num_guns = max(len(guns), max_num_guns)

        # Skip the entire transform when there is nothing to do for this geometry
        # (Type A streamer case where caller did not opt-in to always_calculate).
        if geom_type == ShotGunGeometryType.A and not self.always_calculate:
            return headers

        shot_index = np.empty(len(headers), dtype="uint32")
        base_array = headers.base if headers.base is not None else headers
        headers = rfn.append_fields(base_array, "shot_index", shot_index, usemask=False)

        if geom_type == ShotGunGeometryType.B:
            # Interleaved across guns: divide shot_point by max_num_guns then zero-base per line.
            for line_val in unique_lines:
                line_idxs = np.where(headers[self.line_field][:] == line_val)
                headers["shot_index"][line_idxs] = np.floor(headers["shot_point"][line_idxs] / max_num_guns)
                headers["shot_index"][line_idxs] -= headers["shot_index"][line_idxs].min()
        else:
            # Type A always-calculate: shot points already unique per gun, build a dense
            # 0-based index from the sorted unique shot_point values per line.
            for line_val in unique_lines:
                line_idxs = np.where(headers[self.line_field][:] == line_val)
                shot_points = headers["shot_point"][line_idxs]
                unique_shots = np.unique(shot_points)
                headers["shot_index"][line_idxs] = np.searchsorted(unique_shots, shot_points)

        return headers

    def compute_dimensions(self, headers: HeaderArray, dim_names: tuple[str, ...]) -> list[Dimension]:
        """Compute dimensions including ``shot_index`` if created."""
        dimensions = []

        for dim_name in dim_names:
            if dim_name in headers.dtype.names:
                coords = np.unique(headers[dim_name])
                dimensions.append(Dimension(coords=coords, name=dim_name))

        return dimensions


class ComponentSynthesisStrategy(IndexStrategy):
    """Handle missing component dimension for OBN data.

    If component is missing from headers, adds it with a constant value of 1.
    """

    def __init__(self, synthesize_dims: tuple[str, ...]):
        """Initialize component synthesis strategy.

        Args:
            synthesize_dims: Tuple of dimension names to synthesize if missing
        """
        self.synthesize_dims = synthesize_dims

    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Add missing component dimension."""
        for dim in self.synthesize_dims:
            if dim not in headers.dtype.names:
                logger.warning("Synthesizing missing '%s' dimension with constant value 1", dim)
                comp_array = np.ones(len(headers), dtype=np.uint8)
                # `.base` is None for non-view arrays; fall back to the array itself.
                base_array = headers.base if headers.base is not None else headers
                headers = rfn.append_fields(base_array, dim, comp_array, usemask=False)
        return headers

    def compute_dimensions(self, headers: HeaderArray, dim_names: tuple[str, ...]) -> list[Dimension]:
        """Compute dimensions including synthesized component."""
        dimensions = []

        for dim_name in dim_names:
            if dim_name in headers.dtype.names:
                coords = np.unique(headers[dim_name])
                dimensions.append(Dimension(coords=coords, name=dim_name))

        return dimensions


class CompositeStrategy(IndexStrategy):
    """Apply multiple strategies in sequence; each transform feeds into the next."""

    def __init__(self, strategies: list[IndexStrategy]):
        if not strategies:
            raise ValueError("CompositeStrategy requires at least one strategy")
        self.strategies = strategies

    def transform_headers(self, headers: HeaderArray) -> HeaderArray:
        """Apply all strategy transformations in sequence."""
        result = headers
        for strategy in self.strategies:
            logger.debug("Applying strategy: %s", strategy.name)
            result = strategy.transform_headers(result)
        return result

    def compute_dimensions(self, headers: HeaderArray, dim_names: tuple[str, ...]) -> list[Dimension]:
        """Delegate dimension computation to the final strategy.

        Assumes the final strategy is aware of every preceding header transformation.
        """
        return self.strategies[-1].compute_dimensions(headers, dim_names)


class IndexStrategyRegistry:
    """Picks the right :class:`IndexStrategy` from grid overrides + template properties."""

    def create_strategy(
        self,
        grid_overrides: GridOverrides | None = None,
        synthesize_dims: tuple[str, ...] = (),
        template: AbstractDatasetTemplate | None = None,
    ) -> IndexStrategy:
        """Create appropriate index strategy from grid overrides.

        Args:
            grid_overrides: Optional grid override configuration.
            synthesize_dims: Dimensions to synthesize if missing (e.g. ``component``).
            template: Optional dataset template. Used to drive template-aware strategy
                parameters (e.g. picking ``shot_line`` vs. ``sail_line`` for shot
                wrapping and deciding whether ``shot_index`` must always be calculated).

        Returns:
            IndexStrategy (may be composite if multiple overrides).
        """
        strategies: list[IndexStrategy] = []

        if synthesize_dims:
            strategies.append(ComponentSynthesisStrategy(synthesize_dims))

        if grid_overrides:
            if grid_overrides.auto_channel_wrap:
                strategies.append(ChannelWrappingStrategy())

            if grid_overrides.auto_shot_wrap:
                line_field, always_calculate = self._resolve_shot_wrap_params(template)
                strategies.append(ShotWrappingStrategy(line_field=line_field, always_calculate=always_calculate))

            # NonBinned and HasDuplicates are mutually exclusive.
            if grid_overrides.non_binned:
                strategies.append(
                    NonBinnedStrategy(
                        chunksize=grid_overrides.chunksize,
                        collapse_dims=grid_overrides.replace_dims,
                    )
                )
            elif grid_overrides.has_duplicates:
                strategies.append(DuplicateHandlingStrategy())

        if not strategies:
            return RegularGridStrategy()
        if len(strategies) == 1:
            return strategies[0]
        return CompositeStrategy(strategies)

    @staticmethod
    def _resolve_shot_wrap_params(
        template: AbstractDatasetTemplate | None,
    ) -> tuple[str, bool]:
        """Pick ``line_field`` and ``always_calculate`` for shot wrapping from a template.

        Templates that declare ``shot_index`` as a calculated dimension (e.g. OBN) need
        ``shot_index`` to always be emitted. The line field is ``shot_line`` if it appears
        in the template's spatial dimensions, otherwise ``sail_line``.
        """
        if template is None:
            return "sail_line", False

        spatial = set(template.spatial_dimension_names)
        line_field = "shot_line" if "shot_line" in spatial else "sail_line"
        always_calculate = "shot_index" in template.calculated_dimension_names
        return line_field, always_calculate
