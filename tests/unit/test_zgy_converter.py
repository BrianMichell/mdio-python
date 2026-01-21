"""Unit tests for ZGY to MDIO conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from mdio.api.io import open_mdio
from mdio.converters.zgy import zgy_to_mdio

if TYPE_CHECKING:
    from pathlib import Path


def _create_mock_zgy_dataset(
    n_ilines: int = 5,
    n_xlines: int = 10,
    n_samples: int = 50,
    vertical_dim: str = "time",
) -> xr.Dataset:
    """Create a mock xarray Dataset simulating pyzgy output.

    Args:
        n_ilines: Number of inlines.
        n_xlines: Number of crosslines.
        n_samples: Number of samples.
        vertical_dim: Name of vertical dimension ('time' or 'depth').

    Returns:
        Mock xarray Dataset.
    """
    rng = np.random.default_rng(seed=42)

    data = rng.standard_normal((n_ilines, n_xlines, n_samples)).astype(np.float32)

    return xr.Dataset(
        {"data": (["iline", "xline", vertical_dim], data)},
        coords={
            "iline": np.arange(1, n_ilines + 1),
            "xline": np.arange(1, n_xlines + 1),
            vertical_dim: np.arange(0, n_samples * 4, 4),
        },
    )


class TestZgyToMdio:
    """Tests for zgy_to_mdio function."""

    def test_output_exists_no_overwrite(self, tmp_path: Path) -> None:
        """Test that FileExistsError is raised when output exists and overwrite=False."""
        output_path = tmp_path / "output.mdio"
        output_path.mkdir()

        mock_pyzgy = MagicMock()
        with (
            patch.dict("sys.modules", {"pyzgy": mock_pyzgy}),
            pytest.raises(FileExistsError, match="exists"),
        ):
            zgy_to_mdio("input.zgy", output_path, overwrite=False)

    def test_pyzgy_not_installed(self, tmp_path: Path) -> None:
        """Test that ImportError is raised when pyzgy is not installed."""
        output_path = tmp_path / "output.mdio"

        with (
            patch.dict("sys.modules", {"pyzgy": None}),
            pytest.raises(ImportError, match="pyzgy"),
        ):
            zgy_to_mdio("input.zgy", output_path)

    def test_full_conversion_time_domain(self, tmp_path: Path) -> None:
        """Test full conversion workflow for time-domain data."""
        mock_ds = _create_mock_zgy_dataset(n_ilines=5, n_xlines=10, n_samples=50, vertical_dim="time")
        output_path = tmp_path / "output.mdio"

        mock_pyzgy = MagicMock()
        with (
            patch.dict("sys.modules", {"pyzgy": mock_pyzgy}),
            patch("xarray.open_dataset", return_value=mock_ds),
        ):
            zgy_to_mdio("test.zgy", output_path, overwrite=True)

        # Verify output
        assert output_path.exists()

        ds = open_mdio(output_path)

        # Check dimensions were renamed
        assert "inline" in ds.dims
        assert "crossline" in ds.dims
        assert "time" in ds.dims
        assert ds.sizes["inline"] == 5
        assert ds.sizes["crossline"] == 10
        assert ds.sizes["time"] == 50

        # Check data variable renamed
        assert "amplitude" in ds.data_vars

        # Check trace mask created
        assert "trace_mask" in ds.data_vars
        assert ds["trace_mask"].shape == (5, 10)
        assert ds["trace_mask"].values.all()

        # Check metadata
        assert ds.attrs["name"] == "PostStack3DTime"

    def test_full_conversion_depth_domain(self, tmp_path: Path) -> None:
        """Test full conversion workflow for depth-domain data."""
        mock_ds = _create_mock_zgy_dataset(n_ilines=3, n_xlines=4, n_samples=25, vertical_dim="depth")
        output_path = tmp_path / "output_depth.mdio"

        mock_pyzgy = MagicMock()
        with (
            patch.dict("sys.modules", {"pyzgy": mock_pyzgy}),
            patch("xarray.open_dataset", return_value=mock_ds),
        ):
            zgy_to_mdio("test.zgy", output_path, overwrite=True)

        ds = open_mdio(output_path)

        assert "depth" in ds.dims
        assert "time" not in ds.dims
        assert ds.attrs["name"] == "PostStack3DDepth"

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        """Test that overwrite=True allows overwriting existing files."""
        mock_ds = _create_mock_zgy_dataset()
        output_path = tmp_path / "output.mdio"

        mock_pyzgy = MagicMock()

        # First write
        with (
            patch.dict("sys.modules", {"pyzgy": mock_pyzgy}),
            patch("xarray.open_dataset", return_value=mock_ds),
        ):
            zgy_to_mdio("test.zgy", output_path, overwrite=True)

        assert output_path.exists()

        # Second write with overwrite
        with (
            patch.dict("sys.modules", {"pyzgy": mock_pyzgy}),
            patch("xarray.open_dataset", return_value=mock_ds),
        ):
            zgy_to_mdio("test.zgy", output_path, overwrite=True)

        assert output_path.exists()
