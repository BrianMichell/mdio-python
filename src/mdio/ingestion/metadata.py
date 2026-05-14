"""Metadata utilities for MDIO ingestion."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from mdio.core.config import MDIOSettings
from mdio.ingestion._raw_headers_experimental import attach_raw_binary_header

if TYPE_CHECKING:
    from xarray import Dataset as xr_Dataset

    from mdio.builder.schemas.v1.dataset import Dataset
    from mdio.segy.file import SegyFileInfo


def add_grid_override_to_metadata(dataset: Dataset, grid_overrides: dict[str, Any] | None) -> None:
    """Add grid override to Dataset metadata if needed."""
    if dataset.metadata.attributes is None:
        dataset.metadata.attributes = {}

    if grid_overrides is not None:
        dataset.metadata.attributes["gridOverrides"] = grid_overrides


def add_segy_file_headers(xr_dataset: xr_Dataset, segy_file_info: SegyFileInfo) -> xr_Dataset:
    """Add SEG-Y file headers to the dataset as metadata."""
    if not MDIOSettings().save_segy_file_header:
        return xr_dataset

    expected_rows = 40
    expected_cols = 80

    text_header_rows = segy_file_info.text_header.splitlines()
    text_header_cols_bad = [len(row) != expected_cols for row in text_header_rows]

    if len(text_header_rows) != expected_rows:
        err = f"Invalid text header count: expected {expected_rows}, got {len(segy_file_info.text_header)}"
        raise ValueError(err)

    if any(text_header_cols_bad):
        err = f"Invalid text header columns: expected {expected_cols} per line."
        raise ValueError(err)

    xr_dataset["segy_file_header"] = ((), "")
    file_header_attrs = xr_dataset["segy_file_header"].attrs
    file_header_attrs.update(
        {
            "textHeader": segy_file_info.text_header,
            "binaryHeader": segy_file_info.binary_header_dict,
        }
    )
    attach_raw_binary_header(file_header_attrs, segy_file_info.raw_binary_headers)

    return xr_dataset
