"""EXPERIMENTAL: gating for the ``MDIO__IMPORT__RAW_HEADERS`` feature.

This module concentrates all checks tied to the experimental raw-headers feature so
that removing the feature is a single-file delete plus the (small) handful of call
sites that import from here. The behaviour:

* ``maybe_add_raw_headers()`` decides whether the ``raw_headers`` variable should
  be added to the dataset for this run, and if so appends it via the template's
  internal builder. Honours the env-driven :class:`mdio.core.config.MDIOSettings`
  and the active Zarr format.
* ``attach_raw_binary_header()`` adds the base64-encoded binary file header to a
  segy_file_header attribute dict, if (and only if) the feature is enabled.

Removal plan
------------
Delete this module and:

#. Drop the call to :func:`maybe_add_raw_headers` in
   :mod:`mdio.ingestion.pipeline`.
#. Drop the ``_add_raw_headers`` method on
   :class:`~mdio.builder.templates.base.AbstractDatasetTemplate`.
#. Drop the call to :func:`attach_raw_binary_header` in
   :mod:`mdio.ingestion.metadata`.
#. Drop ``MDIOSettings.raw_headers``.
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

import zarr

from mdio.constants import ZarrFormat
from mdio.core.config import MDIOSettings

if TYPE_CHECKING:
    from typing import Any

    from mdio.builder.schemas.v1.dataset import Dataset
    from mdio.builder.templates.base import AbstractDatasetTemplate

logger = logging.getLogger(__name__)


def _should_include_raw_headers() -> bool:
    """True iff the experimental flag is set and the active Zarr format supports it."""
    if not MDIOSettings().raw_headers:
        return False
    if zarr.config.get("default_zarr_format") == ZarrFormat.V2:
        logger.warning("Raw headers are only supported for Zarr v3. Skipping raw headers.")
        return False
    logger.warning("MDIO__IMPORT__RAW_HEADERS is experimental and expected to change or be removed.")
    return True


def maybe_add_raw_headers(template: AbstractDatasetTemplate, dataset: Dataset) -> Dataset:
    """Append the raw_headers variable to ``dataset`` if the experimental flag enables it.

    The template's internal builder (still alive after ``build_dataset``) is reused
    so dimension resolution stays consistent. Returns the original dataset untouched
    when the feature is disabled.
    """
    if not _should_include_raw_headers():
        return dataset
    template._add_raw_headers()  # noqa: SLF001 - experimental seam, see removal plan above
    return template._builder.build()  # noqa: SLF001


def attach_raw_binary_header(attrs: dict[str, Any], raw_binary_headers: bytes) -> None:
    """Attach a base64-encoded raw binary header to ``attrs`` if the feature is enabled."""
    if not MDIOSettings().raw_headers:
        return
    attrs["rawBinaryHeader"] = base64.b64encode(raw_binary_headers).decode("ascii")
