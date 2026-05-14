"""Header analysis utilities for MDIO ingestion.

This module contains functions for analyzing SEG-Y headers to determine
geometry and create indices for various acquisition types.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from enum import auto
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib import recfunctions as rfn

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from numpy.typing import NDArray
    from segy.arrays import HeaderArray


logger = logging.getLogger(__name__)


class StreamerShotGeometryType(Enum):
    """Enumerates streamer shot geometry types by channel numbering.

    Type A: Channels restart numbering for each cable (1 to N per cable).
    Type B: Channels are numbered sequentially across all cables.
    Type C: Channels are numbered in reverse sequential order (Type B reversed).
    """

    A = auto()
    B = auto()
    C = auto()


class ShotGunGeometryType(Enum):
    """Enumerates acquisition gun geometries for shot data.

    A (SINGLE/ALTERNATE):  Shot points are already unique per gun (no interleaving).
    B (SIMULTANEOUS):      Shot points are interleaved across guns and must be divided
                           by the number of guns to recover a dense per-gun index.
    """

    A = auto()
    B = auto()


def analyze_streamer_headers(
    index_headers: HeaderArray,
) -> tuple[NDArray, NDArray, NDArray, StreamerShotGeometryType]:
    """Inspect cable/channel headers to determine streamer geometry type.

    Args:
        index_headers: numpy array with index headers

    Returns:
        Tuple of (unique_cables, cable_chan_min, cable_chan_max, geom_type).
    """
    unique_cables = np.sort(np.unique(index_headers["cable"]))

    cable_chan_min = np.empty(unique_cables.shape)
    cable_chan_max = np.empty(unique_cables.shape)

    for idx, cable in enumerate(unique_cables):
        cable_mask = index_headers["cable"] == cable
        current_cable = index_headers["channel"][cable_mask]
        cable_chan_min[idx] = np.min(current_cable)
        cable_chan_max[idx] = np.max(current_cable)

    geom_type = StreamerShotGeometryType.B

    for idx1, cable1 in enumerate(unique_cables):
        min_val1 = cable_chan_min[idx1]
        max_val1 = cable_chan_max[idx1]
        cable1_range = (min_val1, max_val1)
        for idx2, cable2 in enumerate(unique_cables):
            if cable2 == cable1:
                continue
            min_val2 = cable_chan_min[idx2]
            max_val2 = cable_chan_max[idx2]
            cable2_range = (min_val2, max_val2)

            if min_val2 < max_val1 and max_val2 > min_val1:
                geom_type = StreamerShotGeometryType.A
                logger.info("Found overlapping channels, assuming streamer type A")
                overlap_info = (
                    "Cable %s index %s with channel range %s overlaps cable %s index %s with "
                    "channel range %s."
                )
                logger.info(overlap_info, cable1, idx1, cable1_range, cable2, idx2, cable2_range)
                return unique_cables, cable_chan_min, cable_chan_max, geom_type

    return unique_cables, cable_chan_min, cable_chan_max, geom_type


def analyze_lines_for_guns(
    index_headers: HeaderArray,
    line_field: str = "sail_line",
) -> tuple[NDArray, dict[str, list], ShotGunGeometryType]:
    """Inspect line/gun/shot_point headers to determine multi-gun geometry.

    Generalized for any line field name (sail_line for streamer templates,
    shot_line for OBN templates, etc).

    Args:
        index_headers: Numpy array with index headers.
        line_field: Name of the line field (e.g. ``sail_line``, ``shot_line``).

    Returns:
        Tuple of (unique_lines, unique_guns_per_line, geom_type).
    """
    unique_lines = np.sort(np.unique(index_headers[line_field]))
    unique_guns = np.sort(np.unique(index_headers["gun"]))
    logger.info("unique_%s values: %s", line_field, unique_lines)
    logger.info("unique_guns: %s", unique_guns)

    unique_guns_per_line: dict[str, list] = {}

    geom_type = ShotGunGeometryType.B
    for line_val in unique_lines:
        line_mask = index_headers[line_field] == line_val
        shot_current = index_headers["shot_point"][line_mask]
        gun_current = index_headers["gun"][line_mask]

        unique_guns_in_line = np.sort(np.unique(gun_current))
        num_guns = unique_guns_in_line.shape[0]
        unique_guns_per_line[str(line_val)] = list(unique_guns_in_line)

        # Skip geometry detection once we know it is Type A but keep populating the
        # unique_guns_per_line map for downstream consumers.
        if geom_type == ShotGunGeometryType.A:
            continue

        for gun in unique_guns_in_line:
            gun_mask = gun_current == gun
            shots_for_gun = shot_current[gun_mask]
            num_shots = np.unique(shots_for_gun).shape[0]
            mod_shots = np.floor(shots_for_gun / num_guns)
            if len(np.unique(mod_shots)) != num_shots:
                msg = "%s %s has %s shots; div by %s guns gives %s unique mod shots."
                logger.info(msg, line_field, line_val, num_shots, num_guns, len(np.unique(mod_shots)))
                geom_type = ShotGunGeometryType.A
                break

    return unique_lines, unique_guns_per_line, geom_type


def analyze_saillines_for_guns(
    index_headers: HeaderArray,
) -> tuple[NDArray, dict[str, list], ShotGunGeometryType]:
    """Backward-compatible alias of :func:`analyze_lines_for_guns` for ``sail_line``."""
    return analyze_lines_for_guns(index_headers, line_field="sail_line")


def create_counter(
    depth: int,
    total_depth: int,
    unique_headers: dict[str, NDArray],
    header_names: list[str],
) -> dict[str, dict] | int:
    """Helper to build a nested dict tree for counting trace keys for auto index."""
    if depth == total_depth:
        return 0

    counter: dict[str, dict] = {}
    header_key = header_names[depth]
    for header in unique_headers[header_key]:
        counter[header] = create_counter(depth + 1, total_depth, unique_headers, header_names)
    return counter


def create_trace_index(
    depth: int,
    counter: dict,
    index_headers: HeaderArray,
    header_names: list[str],
    dtype: DTypeLike = np.int16,
) -> NDArray | None:
    """Walk the counter tree and assign per-key trace indices to ``index_headers``."""
    if depth == 0:
        return None

    trace_no_field = np.zeros(index_headers.shape, dtype=dtype)
    index_headers = rfn.append_fields(index_headers, "trace", trace_no_field, usemask=False)

    headers = [index_headers[name] for name in header_names[:depth]]
    for idx, idx_values in enumerate(zip(*headers, strict=True)):
        if depth == 1:
            counter[idx_values[0]] += 1
            index_headers["trace"][idx] = counter[idx_values[0]]
        else:
            sub_counter = counter
            for idx_value in idx_values[:-1]:
                sub_counter = sub_counter[idx_value]
            sub_counter[idx_values[-1]] += 1
            index_headers["trace"][idx] = sub_counter[idx_values[-1]]

    return index_headers


def analyze_non_indexed_headers(index_headers: HeaderArray, dtype: DTypeLike = np.int16) -> NDArray:
    """Add a ``trace`` field to headers indexing duplicates within each combination of dims.

    Args:
        index_headers: numpy array with index headers
        dtype: numpy type for value of created trace header.

    Returns:
        Headers augmented with a ``trace`` field counting duplicates per dim combination.
    """
    t_start = time.perf_counter()
    unique_headers: dict[str, NDArray] = {}
    total_depth = 0
    header_names: list[str] = []
    for header_key in index_headers.dtype.names:
        if header_key != "trace":
            unique_headers[header_key] = np.sort(np.unique(index_headers[header_key]))
            header_names.append(header_key)
            total_depth += 1

    counter = create_counter(0, total_depth, unique_headers, header_names)
    index_headers = create_trace_index(total_depth, counter, index_headers, header_names, dtype=dtype)

    t_stop = time.perf_counter()
    logger.debug("Time spent generating trace index: %.4f s", t_stop - t_start)
    return index_headers
