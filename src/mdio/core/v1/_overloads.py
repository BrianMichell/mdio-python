"""Overloads for xarray.

The intent of overloading here is:
1. To provide a consistent mdio.* naming scheme.
"""

from collections.abc import Mapping

import xarray as xr
from xarray import DataArray as _DataArray
from xarray import Dataset as _Dataset


class MDIODataset(_Dataset):
    """xarray.Dataset subclass with MDIO v1 extensions."""

    __slots__ = ()

    def to_mdio(
        self,
        store: str | None = None,
        *args: str | int | float | bool,
        **kwargs: Mapping[str, str | int | float | bool],
    ) -> None:
        """Alias for `.to_zarr()`."""
        # Ensure zarr_version=2 by default unless explicitly overridden
        zarr_version = kwargs.get("zarr_version", 2)
        if zarr_version != 2:
            raise ValueError("MDIO only supports zarr_version=2")
        kwargs["zarr_version"] = zarr_version
        return super().to_zarr(*args, store=store, **kwargs)


class MDIODataArray(_DataArray):
    """xarray.DataArray subclass with MDIO v1 extensions."""

    __slots__ = ()

    def to_mdio(
        self,
        store: str | None = None,
        *args: str | int | float | bool,
        **kwargs: Mapping[str, str | int | float | bool],
    ) -> None:
        """Alias for `.to_zarr()`, and writes to Zarr store."""
        # Ensure zarr_version=2 by default unless explicitly overridden
        zarr_version = kwargs.get("zarr_version", 2)
        if zarr_version != 2:
            raise ValueError("MDIO only supports zarr_version=2")
        kwargs["zarr_version"] = zarr_version
        return super().to_zarr(*args, store=store, **kwargs)


class MDIO:
    """MDIO namespace for overloaded types and functions."""

    Dataset = MDIODataset
    DataArray = MDIODataArray

    @staticmethod
    def open(
        store: str,
        *args: str | int | float | bool,
        engine: str = "zarr",
        consolidated: bool = False,
        **kwargs: Mapping[str, str | int | float | bool],
    ) -> MDIODataset:
        """Open a Zarr store as an MDIODataset.

        Casts the returned xarray.Dataset (and its variables) to the MDIO subclasses.
        """
        ds = xr.open_dataset(
            store,
            *args,
            engine=engine,
            consolidated=consolidated,
            **kwargs,
        )
        # Cast Dataset to MDIODataset
        ds.__class__ = MDIODataset
        # Cast each DataArray in data_vars and coords

        for _name, var in ds.data_vars.items():
            var.__class__ = MDIODataArray
        for _name, coord in ds.coords.items():
            coord.__class__ = MDIODataArray
        return ds


# Create module-level MDIO namespace
mdio = MDIO()
