"""Conversion from SEG-Y to MDIO v1 format."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from mdio.ingestion.pipeline import run_segy_ingestion
from mdio.segy.geometry import GridOverrides

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from segy.config import SegyHeaderOverrides
    from segy.schema import SegySpec
    from upath import UPath

    from mdio.builder.templates.base import AbstractDatasetTemplate


def segy_to_mdio(  # noqa: PLR0913
    segy_spec: SegySpec,
    mdio_template: AbstractDatasetTemplate,
    input_path: UPath | Path | str,
    output_path: UPath | Path | str,
    overwrite: bool = False,
    grid_overrides: GridOverrides | dict[str, Any] | None = None,
    segy_header_overrides: SegyHeaderOverrides | None = None,
) -> None:
    """Convert a SEG-Y file to an MDIO v1 file.

    This is the v1.x public entry point. Internally it dispatches to the v1.2
    refactored ingestion pipeline (:func:`mdio.ingestion.pipeline.run_segy_ingestion`).

    Args:
        segy_spec: The SEG-Y specification to use for the conversion.
        mdio_template: The MDIO template to use for the conversion.
        input_path: The universal path of the input SEG-Y file.
        output_path: The universal path for the output MDIO v1 file.
        overwrite: Whether to overwrite the output file if it already exists. Defaults to False.
        grid_overrides: Grid override configuration. Prefer a typed
            :class:`mdio.GridOverrides` instance. A ``dict`` is still accepted for backward
            compatibility but will emit a :class:`DeprecationWarning` and be coerced.
        segy_header_overrides: Option to override specific SEG-Y headers during ingestion.

    Raises:
        FileExistsError: If the output location already exists and ``overwrite`` is False.
    """
    if isinstance(grid_overrides, dict):
        warnings.warn(
            "Passing 'grid_overrides' as a dict is deprecated and will be removed in a future "
            "release. Use 'mdio.GridOverrides(...)' (snake_case fields or legacy aliases both work).",
            DeprecationWarning,
            stacklevel=2,
        )
        grid_overrides = GridOverrides.from_legacy_dict(grid_overrides) if grid_overrides else None

    return run_segy_ingestion(
        segy_spec=segy_spec,
        mdio_template=mdio_template,
        input_path=input_path,
        output_path=output_path,
        overwrite=overwrite,
        grid_overrides=grid_overrides,
        segy_header_overrides=segy_header_overrides,
    )
