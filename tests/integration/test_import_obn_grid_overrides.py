"""End to end testing for OBN SEG-Y to MDIO conversion with grid overrides."""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

import dask
import numpy as np
import pytest
import xarray.testing as xrt
from tests.integration.conftest import get_segy_mock_obn_spec

from mdio import GridOverrides
from mdio.api.io import open_mdio
from mdio.builder.template_registry import TemplateRegistry
from mdio.converters.segy import segy_to_mdio

if TYPE_CHECKING:
    from pathlib import Path

dask.config.set(scheduler="synchronous")
os.environ["MDIO__IMPORT__SAVE_SEGY_FILE_HEADER"] = "true"


class TestImportObnWithComponent:
    """Test OBN SEG-Y import with component header (standard case)."""

    def test_import_obn_with_calculate_shot_index(
        self,
        segy_mock_obn_with_component: Path,
        zarr_tmp: Path,
    ) -> None:
        """Test importing OBN SEG-Y with the v1.2 ``auto_shot_wrap`` override.

        Replaces the v1.x ``{"CalculateShotIndex": True}`` dict. The strategy registry
        is template-aware and detects that the OBN template uses ``shot_line`` and that
        ``shot_index`` is a calculated dimension (always-emit).
        """
        segy_spec = get_segy_mock_obn_spec(include_component=True)
        grid_override = GridOverrides(auto_shot_wrap=True)

        segy_to_mdio(
            segy_spec=segy_spec,
            mdio_template=TemplateRegistry().get("ObnReceiverGathers3D"),
            input_path=segy_mock_obn_with_component,
            output_path=zarr_tmp,
            overwrite=True,
            grid_overrides=grid_override,
        )

        num_samples = 25
        components = [1, 2, 3, 4]
        receivers = [101, 102, 103]
        shot_lines = [1, 2]
        guns = [1, 2]

        ds = open_mdio(zarr_tmp)

        assert ds["segy_file_header"].attrs["binaryHeader"]["samples_per_trace"] == num_samples
        assert ds.attrs["attributes"]["gridOverrides"] == {"AutoShotWrap": True}

        xrt.assert_duckarray_equal(ds["component"], components)
        xrt.assert_duckarray_equal(ds["receiver"], receivers)
        xrt.assert_duckarray_equal(ds["shot_line"], shot_lines)
        xrt.assert_duckarray_equal(ds["gun"], guns)

        # With interleaved geometry: gun1: 1,3,5 -> indices 0,1,2; gun2: 2,4,6 -> indices 1,2,3
        # Combined unique indices: 0, 1, 2, 3
        expected_shot_index = [0, 1, 2, 3]
        xrt.assert_duckarray_equal(ds["shot_index"], expected_shot_index)

        times_expected = list(range(0, num_samples, 1))
        xrt.assert_duckarray_equal(ds["time"], times_expected)

        # shot_point preserved as coordinate, not a dimension
        assert "shot_point" in ds.coords
        assert ds["shot_point"].dims == ("shot_line", "gun", "shot_index")

    def test_import_obn_with_legacy_dict_calculate_shot_index(
        self,
        segy_mock_obn_with_component: Path,
        zarr_tmp: Path,
    ) -> None:
        """Back-compat: passing the legacy ``{"CalculateShotIndex": True}`` dict still works.

        Verifies (a) the legacy key is translated to ``auto_shot_wrap`` and (b) a
        ``DeprecationWarning`` is emitted to push callers to :class:`GridOverrides`.
        """
        segy_spec = get_segy_mock_obn_spec(include_component=True)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            segy_to_mdio(
                segy_spec=segy_spec,
                mdio_template=TemplateRegistry().get("ObnReceiverGathers3D"),
                input_path=segy_mock_obn_with_component,
                output_path=zarr_tmp,
                overwrite=True,
                grid_overrides={"CalculateShotIndex": True},
            )

        deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning)]
        assert any("grid_overrides" in str(w.message) for w in deprecations)

        ds = open_mdio(zarr_tmp)
        # Legacy key collapses into auto_shot_wrap; metadata reflects the canonical name.
        assert ds.attrs["attributes"]["gridOverrides"] == {"AutoShotWrap": True}
        xrt.assert_duckarray_equal(ds["shot_index"], [0, 1, 2, 3])


class TestImportObnSyntheticComponent:
    """Test OBN SEG-Y import without component header - component is synthesized."""

    def test_import_obn_synthetic_component(
        self,
        segy_mock_obn_no_component: Path,
        zarr_tmp: Path,
    ) -> None:
        """Test importing OBN SEG-Y without component - component is automatically synthesized.

        The OBN template declares ``synthesize_missing_dims = ("component",)``, which the
        :class:`ComponentSynthesisStrategy` consumes during the index-strategy phase.
        """
        segy_spec = get_segy_mock_obn_spec(include_component=False)
        grid_override = GridOverrides(auto_shot_wrap=True)

        segy_to_mdio(
            segy_spec=segy_spec,
            mdio_template=TemplateRegistry().get("ObnReceiverGathers3D"),
            input_path=segy_mock_obn_no_component,
            output_path=zarr_tmp,
            overwrite=True,
            grid_overrides=grid_override,
        )

        num_samples = 25
        receivers = [101, 102, 103]
        shot_lines = [1, 2]
        guns = [1, 2]

        ds = open_mdio(zarr_tmp)

        assert ds["segy_file_header"].attrs["binaryHeader"]["samples_per_trace"] == num_samples
        assert ds.attrs["attributes"]["gridOverrides"] == {"AutoShotWrap": True}

        # Component is a synthesized dimension with constant value 1.
        assert "component" in ds.dims
        xrt.assert_duckarray_equal(ds["component"], [1])

        xrt.assert_duckarray_equal(ds["receiver"], receivers)
        xrt.assert_duckarray_equal(ds["shot_line"], shot_lines)
        xrt.assert_duckarray_equal(ds["gun"], guns)

        expected_shot_index = [0, 1, 2, 3]
        xrt.assert_duckarray_equal(ds["shot_index"], expected_shot_index)

        times_expected = list(range(0, num_samples, 1))
        xrt.assert_duckarray_equal(ds["time"], times_expected)

        assert "shot_point" in ds.coords
        assert ds["shot_point"].dims == ("shot_line", "gun", "shot_index")


class TestImportObnMissingCalculateShotIndex:
    """Test OBN SEG-Y import without ``auto_shot_wrap`` enabled."""

    def test_import_obn_without_calculate_shot_index_raises(
        self,
        segy_mock_obn_with_component: Path,
        zarr_tmp: Path,
    ) -> None:
        """Importing OBN data without ``auto_shot_wrap`` must fail with a clear message.

        OBN's ``shot_index`` is a calculated dimension. With no override that produces it,
        the index strategy phase cannot synthesize ``shot_index`` and ingestion fails.
        """
        segy_spec = get_segy_mock_obn_spec(include_component=True)

        with pytest.raises(ValueError, match=r"Required computed fields.*not found after grid overrides") as exc_info:
            segy_to_mdio(
                segy_spec=segy_spec,
                mdio_template=TemplateRegistry().get("ObnReceiverGathers3D"),
                input_path=segy_mock_obn_with_component,
                output_path=zarr_tmp,
                overwrite=True,
                grid_overrides=None,
            )

        error_message = str(exc_info.value)
        assert "shot_index" in error_message
        assert "ObnReceiverGathers3D" in error_message


class TestImportObnMultilineTypeA:
    """OBN SEG-Y import with multiple shot lines and Type A geometry.

    Regression coverage for a v1.x bug where ``analyze_lines_for_guns`` returned early
    on detecting Type A on an earlier line, leaving ``unique_guns_per_line`` incomplete
    and triggering a ``KeyError`` in shot index calculation. The v1.2
    :func:`analyze_lines_for_guns` keeps populating the map after Type A detection.
    """

    def test_import_obn_multiline_type_a_all_lines_processed(
        self,
        segy_mock_obn_multiline_type_a: Path,
        zarr_tmp: Path,
    ) -> None:
        """All shot lines are processed under Type A geometry."""
        segy_spec = get_segy_mock_obn_spec(include_component=True)
        grid_override = GridOverrides(auto_shot_wrap=True)

        segy_to_mdio(
            segy_spec=segy_spec,
            mdio_template=TemplateRegistry().get("ObnReceiverGathers3D"),
            input_path=segy_mock_obn_multiline_type_a,
            output_path=zarr_tmp,
            overwrite=True,
            grid_overrides=grid_override,
        )

        ds = open_mdio(zarr_tmp)

        # All shot lines present (the v1.x bug would drop later lines).
        expected_shot_lines = [1, 2, 3]
        xrt.assert_duckarray_equal(ds["shot_line"], expected_shot_lines)

        expected_guns = [1, 2]
        xrt.assert_duckarray_equal(ds["gun"], expected_guns)

        # Type A: shot points [1, 2, 3] are already unique per gun -> 0-based indices.
        expected_shot_index = [0, 1, 2]
        xrt.assert_duckarray_equal(ds["shot_index"], expected_shot_index)

        expected_receivers = [101, 102]
        xrt.assert_duckarray_equal(ds["receiver"], expected_receivers)

        expected_components = [1]
        xrt.assert_duckarray_equal(ds["component"], expected_components)

        assert "shot_point" in ds.coords
        assert ds["shot_point"].dims == ("shot_line", "gun", "shot_index")

    def test_import_obn_multiline_type_a_sparse_shot_points(
        self,
        segy_mock_obn_multiline_type_a_sparse: Path,
        zarr_tmp: Path,
    ) -> None:
        """Type A vectorized path holds for sparse, non-contiguous shot points.

        Guards the ``np.searchsorted`` over ``np.unique`` Type A path: ``shot_index``
        must be a dense 0-based sequence over the sorted unique shot points, and the
        original ``shot_point`` values must be preserved as a coordinate.
        """
        segy_spec = get_segy_mock_obn_spec(include_component=True)
        grid_override = GridOverrides(auto_shot_wrap=True)

        segy_to_mdio(
            segy_spec=segy_spec,
            mdio_template=TemplateRegistry().get("ObnReceiverGathers3D"),
            input_path=segy_mock_obn_multiline_type_a_sparse,
            output_path=zarr_tmp,
            overwrite=True,
            grid_overrides=grid_override,
        )

        ds = open_mdio(zarr_tmp)

        # Sparse [10, 50, 100] map to dense 0-based [0, 1, 2].
        expected_shot_index = [0, 1, 2]
        xrt.assert_duckarray_equal(ds["shot_index"], expected_shot_index)

        assert "shot_point" in ds.coords
        assert ds["shot_point"].dims == ("shot_line", "gun", "shot_index")
        unique_shot_points = np.unique(ds["shot_point"].values)
        xrt.assert_duckarray_equal(unique_shot_points, [10, 50, 100])
