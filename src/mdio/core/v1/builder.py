"""Builder pattern implementation for MDIO v1 schema models."""

from collections.abc import Mapping
from datetime import UTC
from datetime import datetime
from enum import Enum
from enum import auto
from typing import Any

from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding  # noqa: F401

from mdio.core.v1._overloads import mdio
from mdio.schemas.compressors import ZFP
from mdio.schemas.compressors import Blosc
from mdio.schemas.dimension import NamedDimension
from mdio.schemas.dtype import ScalarType
from mdio.schemas.dtype import StructuredType
from mdio.schemas.metadata import UserAttributes
from mdio.schemas.v1.dataset import Dataset
from mdio.schemas.v1.units import AllUnits
from mdio.schemas.v1.variable import Coordinate
from mdio.schemas.v1.variable import Variable
from mdio.schemas.v1.variable import VariableMetadata

# Import factory functions from serializer module
from ._serializer import _construct_mdio_dataset
from ._serializer import _convert_compressor
from ._serializer import make_coordinate
from ._serializer import make_dataset
from ._serializer import make_dataset_metadata
from ._serializer import make_named_dimension
from ._serializer import make_variable


class _BuilderState(Enum):
    """States for the template builder."""

    INITIAL = auto()
    HAS_DIMENSIONS = auto()
    HAS_COORDINATES = auto()
    HAS_VARIABLES = auto()


class MDIODatasetBuilder:
    """Builder for creating MDIO datasets with enforced build order.

    This builder implements the builder pattern to create MDIO datasets with a v1 schema.
    It enforces a specific build order to ensure valid dataset construction:
    1. Must add dimensions first via add_dimension()
    2. Can optionally add coordinates via add_coordinate()
    3. Must add variables via add_variable()
    4. Must call build() to create the dataset.
    """

    def __init__(self, name: str, attributes: dict[str, Any] | None = None):
        self.name = name
        self.api_version = "1.0.0"  # TODO(BrianMichell, #0): Pull from package metadata
        self.created_on = datetime.now(UTC)
        self.attributes = attributes
        self._dimensions: list[NamedDimension] = []
        self._coordinates: list[Coordinate] = []
        self._variables: list[Variable] = []
        self._state = _BuilderState.INITIAL
        self._unnamed_variable_counter = 0

    def add_dimension(  # noqa: PLR0913
        self,
        name: str,
        size: int,
        long_name: str = None,
        data_type: ScalarType | StructuredType = ScalarType.INT32,
        metadata: list[AllUnits | UserAttributes] | None | dict[str, Any] = None,
    ) -> "MDIODatasetBuilder":
        """Add a dimension.

        This must be called at least once before adding coordinates or variables.

        Args:
            name: Name of the dimension
            size: Size of the dimension
            long_name: Optional long name for the dimension variable
            data_type: Data type for the dimension variable (defaults to INT32)
            metadata: Optional metadata for the dimension variable

        Returns:
            self: Returns self for method chaining
        """
        # Create the dimension
        dimension = make_named_dimension(name, size)
        self._dimensions.append(dimension)

        # Create a variable for the dimension
        dim_var = make_variable(
            name=name,
            long_name=long_name,
            dimensions=[dimension],
            data_type=data_type,
            metadata=metadata,
        )
        self._variables.append(dim_var)

        self._state = _BuilderState.HAS_DIMENSIONS
        return self

    def add_coordinate(  # noqa: PLR0913
        self,
        name: str = "",
        *,
        long_name: str = None,
        dimensions: list[NamedDimension | str] | None = None,
        data_type: ScalarType | StructuredType = ScalarType.FLOAT32,
        metadata: list[AllUnits | UserAttributes] | None | dict[str, Any] = None,
    ) -> "MDIODatasetBuilder":
        """Add a coordinate after adding at least one dimension."""
        if self._state == _BuilderState.INITIAL:
            msg = "Must add at least one dimension before adding coordinates"
            raise ValueError(msg)

        if name == "":
            name = f"coord_{len(self._coordinates)}"
        if dimensions is None:
            dimensions = self._dimensions
        if isinstance(metadata, dict):
            metadata = [metadata]

        # Convert string dimension names to NamedDimension objects
        dim_objects = []
        for dim in dimensions:
            if isinstance(dim, str):
                dim_obj = next((d for d in self._dimensions if d.name == dim), None)
                if dim_obj is None:
                    msg = f"Dimension {dim!r} not found"
                    raise ValueError(msg)
                dim_objects.append(dim_obj)
            else:
                dim_objects.append(dim)

        self._coordinates.append(
            make_coordinate(
                name=name,
                long_name=long_name,
                dimensions=dim_objects,
                data_type=data_type,
                metadata=metadata,
            )
        )
        self._state = _BuilderState.HAS_COORDINATES
        return self

    def add_variable(  # noqa: PLR0913
        self,
        name: str = "",
        *,
        long_name: str = None,
        dimensions: list[NamedDimension | str] | None = None,
        data_type: ScalarType | StructuredType = ScalarType.FLOAT32,
        compressor: Blosc | ZFP | None = None,
        coordinates: list[Coordinate | str] | None = None,
        metadata: VariableMetadata | None = None,
    ) -> "MDIODatasetBuilder":
        """Add a variable after adding at least one dimension."""
        if self._state == _BuilderState.INITIAL:
            msg = "Must add at least one dimension before adding variables"
            raise ValueError(msg)

        if name == "":
            name = f"var_{self._unnamed_variable_counter}"
            self._unnamed_variable_counter += 1
        if dimensions is None:
            dimensions = self._dimensions

        # Convert string dimension names to NamedDimension objects
        dim_objects = []
        for dim in dimensions:
            if isinstance(dim, str):
                dim_obj = next((d for d in self._dimensions if d.name == dim), None)
                if dim_obj is None:
                    msg = f"Dimension {dim!r} not found"
                    raise ValueError(msg)
                dim_objects.append(dim_obj)
            else:
                dim_objects.append(dim)

        self._variables.append(
            make_variable(
                name=name,
                long_name=long_name,
                dimensions=dim_objects,
                data_type=data_type,
                compressor=compressor,
                coordinates=coordinates,
                metadata=metadata,
            )
        )
        self._state = _BuilderState.HAS_VARIABLES
        return self

    def build(self) -> Dataset:
        """Build the final dataset."""
        if self._state == _BuilderState.INITIAL:
            msg = "Must add at least one dimension before building"
            raise ValueError(msg)

        metadata = make_dataset_metadata(
            self.name, self.api_version, self.created_on, self.attributes
        )

        # Add coordinates as variables to the dataset
        # We make a copy so that coordinates are not duplicated if the builder is reused
        all_variables = self._variables.copy()
        for coord in self._coordinates:
            # Convert coordinate to variable
            coord_var = make_variable(
                name=coord.name,
                long_name=coord.long_name,
                dimensions=coord.dimensions,
                data_type=coord.data_type,
                metadata=coord.metadata,
            )
            all_variables.append(coord_var)

        return make_dataset(all_variables, metadata)

    def to_mdio(
        self,
        store: str,
        mode: str = "w",
        compute: bool = False,
        **kwargs: Mapping[str, str | int | float | bool],
    ) -> Dataset:
        """Write the dataset to a Zarr store and return the constructed mdio.Dataset.

        This function constructs an mdio.Dataset from the MDIO dataset and writes its metadata
        to a Zarr store. The actual data is not written, only the metadata structure is created.
        """
        return write_mdio_metadata(self.build(), store, mode, compute, **kwargs)


def write_mdio_metadata(
    mdio_ds: Dataset,
    store: str,
    mode: str = "w",
    compute: bool = False,
    **kwargs: Mapping[str, str | int | float | bool],
) -> mdio.Dataset:
    """Write MDIO metadata to a Zarr store and return the constructed mdio.Dataset.

    This function constructs an mdio.Dataset from the MDIO dataset and writes its metadata
    to a Zarr store. The actual data is not written, only the metadata structure is created.

    Args:
        mdio_ds: The MDIO dataset to serialize
        store: Path to the Zarr or .mdio store
        mode: Write mode to pass to to_mdio(), e.g. 'w' or 'a'
        compute: Whether to compute (write) array chunks (True) or only metadata (False)
        **kwargs: Additional arguments to pass to to_mdio()

    Returns:
        The constructed xarray Dataset with MDIO extensions
    """
    ds = _construct_mdio_dataset(mdio_ds)

    def _generate_encodings() -> dict:
        """Generate encodings for each variable in the MDIO dataset.

        Returns:
            Dictionary mapping variable names to their encoding configurations.
        """
        # TODO(Anybody, #10274): Re-enable chunk_key_encoding when supported by xarray
        # dimension_separator_encoding = V2ChunkKeyEncoding(separator="/").to_dict()
        
        # Collect dimension sizes (same approach as _construct_mdio_dataset)
        dims: dict[str, int] = {}
        for var in mdio_ds.variables:
            for d in var.dimensions:
                if isinstance(d, NamedDimension):
                    dims[d.name] = d.size
        
        global_encodings = {}
        for var in mdio_ds.variables:
            fill_value = 0
            if isinstance(var.data_type, StructuredType):
                continue
            chunks = None
            if var.metadata is not None and var.metadata.chunk_grid is not None:
                chunks = var.metadata.chunk_grid.configuration.chunk_shape
            else:
                # When no chunk_grid is provided, set chunks to shape to avoid chunking
                dim_names = [d.name if isinstance(d, NamedDimension) else d for d in var.dimensions]
                chunks = tuple(dims[name] for name in dim_names)
            global_encodings[var.name] = {
                "chunks": chunks,
                # TODO(Anybody, #10274): Re-enable chunk_key_encoding when supported by xarray
                # "chunk_key_encoding": dimension_separator_encoding,
                "_FillValue": fill_value,
                "dtype": var.data_type,
                "compressors": _convert_compressor(var.compressor),
            }
        return global_encodings

    ds.to_mdio(
        store,
        mode=mode,
        zarr_format=2,
        consolidated=True,
        safe_chunks=False,
        compute=compute,
        encoding=_generate_encodings(),
        **kwargs,
    )
    return ds
