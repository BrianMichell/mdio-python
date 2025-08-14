"""Tests for SEG-Y field validation helper."""

import numpy as np
import pytest
import xarray as xr

from mdio.segy.creation import get_required_segy_fields


def _make_dataset() -> xr.Dataset:
    data = np.zeros((1, 1, 1), dtype=np.float32)
    headers = np.zeros((1, 1, 1), dtype=np.int32)
    mask = np.ones((1, 1, 1), dtype=bool)
    return xr.Dataset(
        {
            "amplitude": (("x", "y", "sample"), data),
            "headers": (("x", "y", "header"), headers),
            "trace_mask": (("x", "y", "mask"), mask),
        },
        attrs={
            "apiVersion": "1.0.0",
            "attributes": {"textHeader": "", "binaryHeader": {}},
        },
    )


def test_get_required_segy_fields_returns_all() -> None:
    """Return tuple when all required fields exist."""
    ds = _make_dataset()
    amp, hdr, mask, attrs, version = get_required_segy_fields(ds)
    assert amp.name == "amplitude"
    assert hdr.name == "headers"
    assert mask.name == "trace_mask"
    assert attrs == {"textHeader": "", "binaryHeader": {}}
    assert version == "1.0.0"


def test_get_required_segy_fields_missing_variable() -> None:
    """Raise when a required data variable is missing."""
    ds = _make_dataset().drop_vars("amplitude")
    with pytest.raises(KeyError):
        get_required_segy_fields(ds)


def test_get_required_segy_fields_missing_attribute() -> None:
    """Raise when a required attribute is missing."""
    ds = _make_dataset()
    del ds.attrs["attributes"]["textHeader"]
    with pytest.raises(KeyError):
        get_required_segy_fields(ds)
