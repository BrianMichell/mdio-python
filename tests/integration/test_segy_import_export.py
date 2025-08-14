"""End to end testing for SEG-Y to MDIO conversion and back."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import dask
import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
from segy import SegyFile
from segy.standards import get_segy_standard
from tests.integration.testing_data import binary_header_teapot_dome
from tests.integration.testing_data import text_header_teapot_dome
from tests.integration.testing_helpers import customize_segy_specs
from tests.integration.testing_helpers import get_inline_header_values
from tests.integration.testing_helpers import get_values
from tests.integration.testing_helpers import validate_variable

from mdio import mdio_to_segy
from mdio.converters.segy import segy_to_mdio
from mdio.core.storage_location import StorageLocation
from mdio.schemas.v1.templates.template_registry import TemplateRegistry
from mdio.segy.compat import mdio_segy_spec

if TYPE_CHECKING:
    from pathlib import Path

dask.config.set(scheduler="synchronous")


@pytest.mark.dependency
@pytest.mark.parametrize("index_bytes", [(17, 13, 81, 85)])
@pytest.mark.parametrize("index_names", [("inline", "crossline", "cdp_x", "cdp_y")])
@pytest.mark.parametrize("index_types", [("int32", "int32", "int32", "int32")])
def test_3d_import(
    segy_input: Path,
    zarr_tmp: Path,
    index_bytes: tuple[int, ...],
    index_names: tuple[str, ...],
    index_types: tuple[str, ...],
) -> None:
    """Test importing a SEG-Y file to MDIO."""
    segy_spec = get_segy_standard(1.0)
    segy_spec = customize_segy_specs(
        segy_spec=segy_spec,
        index_bytes=index_bytes,
        index_names=index_names,
        index_types=index_types,
    )

    segy_to_mdio(
        segy_spec=segy_spec,
        mdio_template=TemplateRegistry().get("PostStack3DTime"),
        input_location=StorageLocation(str(segy_input)),
        output_location=StorageLocation(str(zarr_tmp)),
        overwrite=True,
    )


@pytest.mark.dependency("test_3d_import")
class TestReader:
    """Test reader functionality."""

    def test_meta_dataset_read(self, zarr_tmp: Path) -> None:
        """Metadata reading tests."""
        ds = xr.open_dataset(zarr_tmp, engine="zarr", mask_and_scale=False)
        expected_attrs = {
            "apiVersion": "1.0.0a1",
            "createdOn": "2025-08-06 16:21:54.747880+00:00",
            "name": "PostStack3DTime",
        }
        actual_attrs_json = ds.attrs
        for key, value in expected_attrs.items():
            assert key in actual_attrs_json
            if key == "createdOn":
                assert actual_attrs_json[key] is not None
            else:
                assert actual_attrs_json[key] == value

        attributes = ds.attrs["attributes"]
        assert attributes is not None

        assert attributes["surveyDimensionality"] == "3D"
        assert attributes["ensembleType"] == "line"
        assert attributes["processingStage"] == "post-stack"
        assert attributes["textHeader"] == text_header_teapot_dome()
        assert attributes["binaryHeader"] == binary_header_teapot_dome()

    def test_meta_variable_read(self, zarr_tmp: Path) -> None:
        """Metadata reading tests."""
        ds = xr.open_dataset(zarr_tmp, engine="zarr", mask_and_scale=False)
        expected_attrs = {
            "count": 97354860,
            "sum": -8594.551666259766,
            "sum_squares": 40571291.6875,
            "min": -8.375323295593262,
            "max": 0.0,
            "histogram": {"counts": [], "bin_centers": []},
        }
        actual_attrs_json = json.loads(ds["amplitude"].attrs["statsV1"])
        assert actual_attrs_json == expected_attrs

    def test_grid(self, zarr_tmp: Path) -> None:
        """Test validating MDIO variables."""
        ds = xr.open_dataset(zarr_tmp, engine="zarr", mask_and_scale=False)

        validate_variable(ds, "inline", (345,), ["inline"], np.int32, range(1, 346), get_values)
        validate_variable(ds, "crossline", (188,), ["crossline"], np.int32, range(1, 189), get_values)
        validate_variable(ds, "time", (1501,), ["time"], np.int32, range(0, 3002, 2), get_values)

        validate_variable(ds, "cdp_x", (345, 188), ["inline", "crossline"], np.float64, None, None)
        validate_variable(ds, "cdp_y", (345, 188), ["inline", "crossline"], np.float64, None, None)

        data_type = np.dtype([("inline", "<i4"), ("crossline", "<i4"), ("cdp_x", "<i4"), ("cdp_y", "<i4")])
        validate_variable(
            ds,
            "headers",
            (345, 188),
            ["inline", "crossline"],
            data_type,
            range(1, 346),
            get_inline_header_values,
        )

        validate_variable(ds, "trace_mask", (345, 188), ["inline", "crossline"], np.bool_, None, None)
        validate_variable(
            ds,
            "amplitude",
            (345, 188, 1501),
            ["inline", "crossline", "time"],
            np.float32,
            None,
            None,
        )

    def test_inline(self, zarr_tmp: Path) -> None:
        """Read and compare every 75 inlines' mean and std. dev."""
        ds = xr.open_dataset(zarr_tmp, engine="zarr", mask_and_scale=False)
        inlines = ds["amplitude"][::75, :, :]
        mean, std = inlines.mean(), inlines.std()
        npt.assert_allclose([mean, std], [1.0555277e-04, 6.0027051e-01])

    def test_crossline(self, zarr_tmp: Path) -> None:
        """Read and compare every 75 crosslines' mean and std. dev."""
        ds = xr.open_dataset(zarr_tmp, engine="zarr", mask_and_scale=False)
        xlines = ds["amplitude"][::, ::75, :]
        mean, std = xlines.mean(), xlines.std()
        npt.assert_allclose([mean, std], [-5.0329847e-05, 5.9406823e-01])

    def test_zslice(self, zarr_tmp: Path) -> None:
        """Read and compare every 225 z-slices' mean and std. dev."""
        ds = xr.open_dataset(zarr_tmp, engine="zarr", mask_and_scale=False)
        slices = ds["amplitude"][::, ::, ::225]
        mean, std = slices.mean(), slices.std()
        npt.assert_allclose([mean, std], [0.005236923, 0.61279935])


@pytest.mark.dependency("test_3d_import")
class TestExport:
    """Test SEG-Y exporting functionality."""

    def test_3d_export(self, zarr_tmp: Path, segy_export_tmp: Path) -> None:
        """Export the ingested MDIO file back to SEG-Y."""
        mdio_to_segy(
            input_location=StorageLocation(zarr_tmp.__str__()),
            output_location=StorageLocation(segy_export_tmp.__str__()),
        )

    def test_size_equal(self, segy_input: Path, segy_export_tmp: Path) -> None:
        """Confirm file sizes match after export."""
        assert segy_input.stat().st_size == segy_export_tmp.stat().st_size

    def test_rand_equal(self, segy_input: Path, segy_export_tmp: Path) -> None:
        """Verify trace data is preserved after round-trip export."""
        spec = mdio_segy_spec()
        in_segy = SegyFile(segy_input, spec=spec)
        out_segy = SegyFile(segy_export_tmp, spec=spec)

        num_traces = in_segy.num_traces
        random_indices = np.random.choice(num_traces, 100, replace=False)
        in_traces = in_segy.trace[random_indices]
        out_traces = out_segy.trace[random_indices]

        assert in_segy.num_traces == out_segy.num_traces
        assert in_segy.text_header == out_segy.text_header
        assert in_segy.binary_header == out_segy.binary_header
        npt.assert_array_equal(in_traces.header, out_traces.header)
        npt.assert_array_equal(in_traces.sample, out_traces.sample)
