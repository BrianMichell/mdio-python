"""End to end testing for SEG-Y to MDIO conversion with sharding enabled."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import dask
import numpy as np
import numpy.testing as npt
import pytest
import zarr
from segy.schema import HeaderField
from segy.schema import ScalarType
from segy.standards import get_segy_standard
from tests.integration.testing_helpers import get_inline_header_values
from tests.integration.testing_helpers import get_values
from tests.integration.testing_helpers import validate_variable

from mdio import __version__
from mdio import mdio_to_segy
from mdio.api.io import open_mdio
from mdio.builder.template_registry import TemplateRegistry
from mdio.constants import ZarrFormat
from mdio.converters.segy import segy_to_mdio
from mdio.segy.file import SegyFileWrapper

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from segy.schema import SegySpec


dask.config.set(scheduler="synchronous")


@pytest.fixture(scope="module", autouse=True)
def set_zarr_v3() -> Generator[None, None, None]:
    """Set Zarr v3 format for the sharding tests."""
    original_format = zarr.config.get("default_zarr_format")
    zarr.config.set({"default_zarr_format": ZarrFormat.V3})
    yield
    zarr.config.set({"default_zarr_format": original_format})


@pytest.fixture(scope="module")
def zarr_sharded_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Make a temp file for the sharded output MDIO."""
    return tmp_path_factory.mktemp(r"mdio_sharded")


@pytest.fixture(scope="module")
def segy_sharded_export_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Make a temp file for the round-trip sharded SEG-Y."""
    tmp_dir = tmp_path_factory.mktemp("segy_sharded")
    return tmp_dir / "teapot_sharded_roundtrip.segy"


@pytest.fixture
def set_env_vars(monkeypatch: Generator[pytest.MonkeyPatch]) -> None:
    """Set environment variables for the Teapot dome tests."""
    monkeypatch.setenv("MDIO__IMPORT__SAVE_SEGY_FILE_HEADER", "true")
    monkeypatch.setenv("MDIO__IMPORT__RAW_HEADERS", "true")


@pytest.fixture
def teapot_segy_spec() -> SegySpec:
    """Return the customized SEG-Y specification for the teapot dome dataset."""
    teapot_fields = [
        HeaderField(name="inline", byte=17, format=ScalarType.INT32),
        HeaderField(name="crossline", byte=13, format=ScalarType.INT32),
        HeaderField(name="cdp_x", byte=81, format=ScalarType.INT32),
        HeaderField(name="cdp_y", byte=85, format=ScalarType.INT32),
    ]
    return get_segy_standard(1.0).customize(trace_header_fields=teapot_fields)


def text_header_teapot_dome() -> str:
    """Return the teapot dome expected text header."""
    header_rows = [
        "C 1 CLIENT: ROCKY MOUNTAIN OILFIELD TESTING CENTER                              ",
        "C 2 PROJECT: NAVAL PETROLEUM RESERVE #3 (TEAPOT DOME); NATRONA COUNTY, WYOMING  ",
        "C 3 LINE: 3D                                                                    ",
        "C 4                                                                             ",
        "C 5 THIS IS THE FILTERED POST STACK MIGRATION                                   ",
        "C 6                                                                             ",
        "C 7 INLINE 1, XLINE 1:   X COORDINATE: 788937  Y COORDINATE: 938845             ",
        "C 8 INLINE 1, XLINE 188: X COORDINATE: 809501  Y COORDINATE: 939333             ",
        "C 9 INLINE 188, XLINE 1: X COORDINATE: 788039  Y COORDINATE: 976674             ",
        "C10 INLINE NUMBER:    MIN: 1  MAX: 345  TOTAL: 345                              ",
        "C11 CROSSLINE NUMBER: MIN: 1  MAX: 188  TOTAL: 188                              ",
        "C12 TOTAL NUMBER OF CDPS: 64860   BIN DIMENSION: 110' X 110'                    ",
        "C13                                                                             ",
        "C14                                                                             ",
        "C15                                                                             ",
        "C16                                                                             ",
        "C17                                                                             ",
        "C18                                                                             ",
        "C19 GENERAL SEGY INFORMATION                                                    ",
        "C20 RECORD LENGHT (MS): 3000                                                    ",
        "C21 SAMPLE RATE (MS): 2.0                                                       ",
        "C22 DATA FORMAT: 4 BYTE IBM FLOATING POINT                                      ",
        "C23 BYTES  13- 16: CROSSLINE NUMBER (TRACE)                                     ",
        "C24 BYTES  17- 20: INLINE NUMBER (LINE)                                         ",
        "C25 BYTES  81- 84: CDP_X COORD                                                  ",
        "C26 BYTES  85- 88: CDP_Y COORD                                                  ",
        "C27 BYTES 181-184: INLINE NUMBER (LINE)                                         ",
        "C28 BYTES 185-188: CROSSLINE NUMBER (TRACE)                                     ",
        "C29 BYTES 189-192: CDP_X COORD                                                  ",
        "C30 BYTES 193-196: CDP_Y COORD                                                  ",
        "C31                                                                             ",
        "C32                                                                             ",
        "C33                                                                             ",
        "C34                                                                             ",
        "C35                                                                             ",
        "C36 Processed by: Excel Geophysical Services, Inc.                              ",
        "C37               8301 East Prentice Ave. Ste. 402                              ",
        "C38               Englewood, Colorado 80111                                     ",
        "C39               (voice) 303.694.9629 (fax) 303.771.1646                       ",
        "C40 END EBCDIC                                                                  ",
    ]
    return "\n".join(header_rows)


def binary_header_teapot_dome() -> dict[str, int]:
    """Return the teapot dome expected binary header."""
    return {
        "job_id": 9999,
        "line_num": 9999,
        "reel_num": 1,
        "data_traces_per_ensemble": 188,
        "aux_traces_per_ensemble": 0,
        "sample_interval": 2000,
        "orig_sample_interval": 0,
        "samples_per_trace": 1501,
        "orig_samples_per_trace": 1501,
        "data_sample_format": 1,
        "ensemble_fold": 57,
        "trace_sorting_code": 4,
        "vertical_sum_code": 1,
        "sweep_freq_start": 0,
        "sweep_freq_end": 0,
        "sweep_length": 0,
        "sweep_type_code": 0,
        "sweep_trace_num": 0,
        "sweep_taper_start": 0,
        "sweep_taper_end": 0,
        "taper_type_code": 0,
        "correlated_data_code": 2,
        "binary_gain_code": 1,
        "amp_recovery_code": 4,
        "measurement_system_code": 2,
        "impulse_polarity_code": 1,
        "vibratory_polarity_code": 0,
        "fixed_length_trace_flag": 0,
        "num_extended_text_headers": 0,
        "segy_revision_major": 0,
        "segy_revision_minor": 0,
    }


class TestTeapotShardedRoundtrip:
    """Tests for Teapot Dome data ingestion and export with sharding."""

    @pytest.mark.dependency
    @pytest.mark.usefixtures("set_env_vars")
    def test_teapot_sharded_import(
        self,
        segy_input: Path,
        zarr_sharded_tmp: Path,
        teapot_segy_spec: SegySpec,
    ) -> None:
        """Test importing a SEG-Y file to MDIO with sharding enabled.

        Uses 256-cube shards containing 32-cube chunks for efficient I/O.
        """
        # Get the template and configure sharding
        template = TemplateRegistry().get("PostStack3DTime")

        # Configure chunk and shard sizes:
        # - Chunks: 32x32x32 (inner compression unit)
        # - Shards: 256x256x256 (outer storage container)
        template.full_chunk_shape = (32, 32, 32)
        template.full_shard_shape = (256, 256, 256)

        segy_to_mdio(
            segy_spec=teapot_segy_spec,
            mdio_template=template,
            input_path=segy_input,
            output_path=zarr_sharded_tmp,
            overwrite=True,
        )

    @pytest.mark.dependency("test_teapot_sharded_import")
    def test_sharding_configuration(self, zarr_sharded_tmp: Path) -> None:
        """Verify that sharding is correctly configured in the output dataset."""
        # Open the zarr store directly to inspect sharding configuration
        store = zarr.open_group(zarr_sharded_tmp, mode="r")

        # Check the amplitude array has sharding codec
        amplitude_array = store["amplitude"]
        metadata = amplitude_array.metadata

        # Verify the array has sharding configured
        # In Zarr v3 with sharding, the codecs include a sharding codec
        codecs = metadata.codecs
        sharding_codec = None
        for codec in codecs:
            if hasattr(codec, "chunk_shape"):
                sharding_codec = codec
                break

        assert sharding_codec is not None, "Sharding codec not found in amplitude array"

        # Verify chunk shape within shards is 32x32x32
        # Note: The actual chunk shape stored depends on how zarr-python encodes sharding
        inner_chunk_shape = sharding_codec.chunk_shape
        assert inner_chunk_shape == (32, 32, 32), f"Expected chunk shape (32, 32, 32), got {inner_chunk_shape}"

    @pytest.mark.dependency("test_teapot_sharded_import")
    def test_dataset_metadata(self, zarr_sharded_tmp: Path) -> None:
        """Metadata reading tests."""
        ds = open_mdio(zarr_sharded_tmp)
        expected_attrs = {
            "apiVersion": __version__,
            "createdOn": "2025-08-06 16:21:54.747880+00:00",
            "name": "PostStack3DTime",
        }
        actual_attrs_json = ds.attrs
        # compare one by one due to ever changing createdOn. For it, we only check existence
        for key, value in expected_attrs.items():
            assert key in actual_attrs_json
            if key == "createdOn":
                assert actual_attrs_json[key] is not None
            else:
                assert actual_attrs_json[key] == value

        attributes = ds.attrs["attributes"]
        assert attributes is not None
        assert len(attributes) == 3
        # Validate all attributes provided by the abstract template
        assert attributes["defaultVariableName"] == "amplitude"
        assert attributes["surveyType"] == "3D"
        assert attributes["gatherType"] == "stacked"

        segy_file_header = ds["segy_file_header"]
        assert segy_file_header.attrs["textHeader"] == text_header_teapot_dome()
        assert segy_file_header.attrs["binaryHeader"] == binary_header_teapot_dome()

    @pytest.mark.dependency("test_teapot_sharded_import")
    def test_variable_metadata(self, zarr_sharded_tmp: Path) -> None:
        """Metadata reading tests."""
        ds = open_mdio(zarr_sharded_tmp)
        expected_attrs = {
            "count": 46854270,
            "sum": -8594.551589292674,
            "sumSquares": 40571285.42351971,
            "min": -8.375323295593262,
            "max": 7.723702430725098,
            "histogram": {"counts": [], "binCenters": []},
        }
        actual_attrs = json.loads(ds["amplitude"].attrs["statsV1"])
        assert actual_attrs.keys() == expected_attrs.keys()
        actual_attrs.pop("histogram")
        expected_attrs.pop("histogram")
        np.testing.assert_allclose(list(actual_attrs.values()), list(expected_attrs.values()))

    @pytest.mark.dependency("test_teapot_sharded_import")
    def test_grid(self, zarr_sharded_tmp: Path, teapot_segy_spec: SegySpec) -> None:
        """Test validating MDIO variables."""
        ds = open_mdio(zarr_sharded_tmp)

        # Validate the dimension coordinate variables
        validate_variable(ds, "inline", (345,), ("inline",), np.int32, range(1, 346), get_values)
        validate_variable(ds, "crossline", (188,), ("crossline",), np.int32, range(1, 189), get_values)
        validate_variable(ds, "time", (1501,), ("time",), np.int32, range(0, 3002, 2), get_values)

        # Validate the non-dimensional coordinate variables
        validate_variable(ds, "cdp_x", (345, 188), ("inline", "crossline"), np.float64, None, None)
        validate_variable(ds, "cdp_y", (345, 188), ("inline", "crossline"), np.float64, None, None)

        # Validate the headers
        segy_spec = teapot_segy_spec
        data_type = segy_spec.trace.header.dtype

        validate_variable(
            ds,
            "headers",
            (345, 188),
            ("inline", "crossline"),
            data_type.newbyteorder("native"),
            range(1, 346),
            get_inline_header_values,
        )

        # Validate the trace mask
        validate_variable(ds, "trace_mask", (345, 188), ("inline", "crossline"), np.bool, None, None)

        # validate the amplitude data
        validate_variable(
            ds,
            "amplitude",
            (345, 188, 1501),
            ("inline", "crossline", "time"),
            np.float32,
            None,
            None,
        )

    @pytest.mark.dependency("test_teapot_sharded_import")
    def test_inline_reads(self, zarr_sharded_tmp: Path) -> None:
        """Read and compare every 75 inlines' mean and std. dev."""
        ds = open_mdio(zarr_sharded_tmp)
        inlines = ds["amplitude"][::75, :, :]
        mean, std = inlines.mean(dtype="float64"), inlines.std(dtype="float64")
        npt.assert_allclose([mean, std], [0.00010555267, 0.60027058412])

    @pytest.mark.dependency("test_teapot_sharded_import")
    def test_crossline_reads(self, zarr_sharded_tmp: Path) -> None:
        """Read and compare every 75 crosslines' mean and std. dev."""
        ds = open_mdio(zarr_sharded_tmp)
        xlines = ds["amplitude"][:, ::75, :]
        mean, std = xlines.mean(dtype="float64"), xlines.std(dtype="float64")
        npt.assert_allclose([mean, std], [-5.03298501828e-05, 0.59406807762])

    @pytest.mark.dependency("test_teapot_sharded_import")
    def test_zslice_reads(self, zarr_sharded_tmp: Path) -> None:
        """Read and compare every 225 z-slices' mean and std. dev."""
        ds = open_mdio(zarr_sharded_tmp)
        slices = ds["amplitude"][:, :, ::225]
        mean, std = slices.mean(dtype="float64"), slices.std(dtype="float64")
        npt.assert_allclose([mean, std], [0.00523692339, 0.61279943571])

    @pytest.mark.dependency("test_teapot_sharded_import")
    def test_sharded_export(
        self,
        segy_input: Path,
        zarr_sharded_tmp: Path,
        segy_sharded_export_tmp: Path,
        teapot_segy_spec: SegySpec,
    ) -> None:
        """Test 3D export from sharded MDIO."""
        rng = np.random.default_rng(seed=1234)

        mdio_to_segy(segy_spec=teapot_segy_spec, input_path=zarr_sharded_tmp, output_path=segy_sharded_export_tmp)

        # Check if file sizes match on IBM file.
        assert segy_input.stat().st_size == segy_sharded_export_tmp.stat().st_size

        # IBM. Is random original traces and headers match round-trip file?
        in_segy = SegyFileWrapper(segy_input, spec=teapot_segy_spec)
        out_segy = SegyFileWrapper(segy_sharded_export_tmp, spec=teapot_segy_spec)

        num_traces = in_segy.num_traces
        random_indices = rng.choice(num_traces, 100, replace=False)
        in_traces = in_segy.trace[random_indices]
        out_traces = out_segy.trace[random_indices]

        assert in_segy.num_traces == out_segy.num_traces
        assert in_segy.text_header == out_segy.text_header
        assert in_segy.binary_header == out_segy.binary_header
        npt.assert_array_equal(desired=in_traces.header, actual=out_traces.header)
        npt.assert_array_equal(desired=in_traces.sample, actual=out_traces.sample)

