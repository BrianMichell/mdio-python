"""Dataset Factory for MDIO Ingestion.

This module provides a factory for building MDIO datasets from resolved schemas
and dimensions. It centralizes all dataset construction logic in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdio.builder.schemas.dtype import StructuredType
    from mdio.builder.schemas.v1.dataset import Dataset
    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.core.dimension import Dimension
    from mdio.ingestion.schema_resolver import ResolvedSchema


class DatasetFactory:
    """Factory for building MDIO datasets from schemas.

    This class takes a resolved schema and dimensions and builds
    a complete MDIO dataset with all variables and coordinates.
    """

    def build(
        self,
        template: AbstractDatasetTemplate,
        schema: ResolvedSchema,
        dimensions: list[Dimension],
        header_dtype: StructuredType | None = None,
        include_raw_headers: bool = False,
    ) -> Dataset:
        """Build MDIO dataset from schema and dimensions.

        Args:
            template: The template to build from
            schema: Resolved schema specifying dataset structure
            dimensions: List of dimension objects with coordinates
            header_dtype: Optional structured type for trace headers
            include_raw_headers: Whether to include raw binary headers (Zarr v3 only)

        Returns:
            Complete Dataset ready for xarray conversion
        """
        # Create dimension sizes dict
        dim_sizes = {dim.name: len(dim.coords) for dim in dimensions}

        # The sizes tuple must match the template's expected dimension_names
        sizes = tuple(dim_sizes[dim_name] for dim_name in template.dimension_names)

        # Build dataset using the template
        return template.build_dataset(
            name=schema.name,
            sizes=sizes,
            header_dtype=header_dtype,
            include_raw_headers=include_raw_headers,
        )
