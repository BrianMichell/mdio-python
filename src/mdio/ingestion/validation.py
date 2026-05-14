"""Validation utilities for MDIO ingestion."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from mdio.converters.exceptions import GridTraceSparsityError
from mdio.core.config import MDIOSettings

if TYPE_CHECKING:
    from segy.schema import SegySpec

    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.core.grid import Grid

logger = logging.getLogger(__name__)


def grid_density_qc(grid: Grid, num_traces: int) -> None:
    """Quality control for sensible grid density during SEG-Y to MDIO conversion.

    This function checks the density of the proposed grid by comparing the total possible traces
    (`grid_traces`) to the actual number of traces in the SEG-Y file (`num_traces`). A warning is
    logged if the sparsity ratio (`grid_traces / num_traces`) exceeds a configurable threshold,
    indicating potential inefficiency or misconfiguration.

    The warning threshold is set via the environment variable `MDIO__GRID__SPARSITY_RATIO_WARN`
    (default 2), and the error threshold via `MDIO__GRID__SPARSITY_RATIO_LIMIT` (default 10). To
    suppress the exception (but still log warnings), set `MDIO_IGNORE_CHECKS=1`.

    Args:
        grid: The Grid instance to check.
        num_traces: Expected number of traces from the SEG-Y file.

    Raises:
        GridTraceSparsityError: If the sparsity ratio exceeds `MDIO__GRID__SPARSITY_RATIO_LIMIT`
            and `MDIO_IGNORE_CHECKS` is not set to a truthy value (e.g., "1", "true").
    """
    settings = MDIOSettings()
    grid_traces = np.prod(grid.shape[:-1], dtype=np.uint64)
    sparsity_ratio = float("inf") if num_traces == 0 else grid_traces / num_traces

    warning_ratio = settings.grid_sparsity_ratio_warn
    error_ratio = settings.grid_sparsity_ratio_limit
    ignore_checks = settings.ignore_checks

    should_warn = sparsity_ratio > warning_ratio
    should_error = sparsity_ratio > error_ratio and not ignore_checks

    if not should_warn and not should_error:
        return

    dims = dict(zip(grid.dim_names, grid.shape, strict=True))
    msg = (
        f"Ingestion grid is sparse. Sparsity ratio: {sparsity_ratio:.2f}. "
        f"Ingestion grid: {dims}. "
        f"SEG-Y trace count: {num_traces}, grid trace count: {grid_traces}."
    )
    for dim_name in grid.dim_names:
        msg += f"\n{dim_name} min: {grid.get_min(dim_name)} max: {grid.get_max(dim_name)}"

    if should_warn:
        logger.warning(msg)

    if should_error:
        raise GridTraceSparsityError(grid.shape, num_traces, msg)


def validate_spec_in_template(segy_spec: SegySpec, mdio_template: AbstractDatasetTemplate) -> None:
    """Validate that the SegySpec has all required fields in the MDIO template."""
    header_fields = {field.name for field in segy_spec.trace.header.fields}

    required_fields = set(mdio_template.spatial_dimension_names) | set(mdio_template.coordinate_names)
    required_fields -= set(mdio_template.calculated_dimension_names)

    for dim in getattr(mdio_template, "synthesize_missing_dims", ()):
        required_fields.discard(dim)

    required_fields.add("coordinate_scalar")
    missing_fields = required_fields - header_fields

    if missing_fields:
        err = (
            f"Required fields {sorted(missing_fields)} for template {mdio_template.name} "
            f"not found in the provided segy_spec"
        )
        raise ValueError(err)
