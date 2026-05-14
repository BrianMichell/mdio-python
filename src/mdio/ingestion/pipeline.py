"""Ingestion pipeline for SEG-Y to MDIO."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import numpy as np
from segy.config import SegyFileSettings

from mdio.api.io import _normalize_path
from mdio.api.io import to_mdio
from mdio.builder.xarray_builder import to_xarray_dataset
from mdio.converters.exceptions import GridTraceCountError
from mdio.converters.type_converter import to_structured_type
from mdio.core.config import MDIOSettings
from mdio.core.dimension import Dimension
from mdio.core.grid import Grid
from mdio.ingestion._raw_headers_experimental import should_include_raw_headers
from mdio.ingestion.coordinate_utils import apply_runtime_units
from mdio.ingestion.coordinate_utils import get_coordinates
from mdio.ingestion.coordinate_utils import populate_dim_coordinates
from mdio.ingestion.coordinate_utils import populate_non_dim_coordinates
from mdio.ingestion.dataset_factory import DatasetFactory
from mdio.ingestion.header_analyzer import HeaderAnalyzer
from mdio.ingestion.index_strategies import IndexStrategyRegistry
from mdio.ingestion.metadata import add_grid_override_to_metadata
from mdio.ingestion.metadata import add_segy_file_headers
from mdio.ingestion.schema_resolver import SchemaResolver
from mdio.ingestion.validation import grid_density_qc
from mdio.ingestion.validation import validate_spec_in_template
from mdio.segy import blocked_io
from mdio.segy.file import get_segy_file_info

if TYPE_CHECKING:
    from pathlib import Path

    from segy.config import SegyHeaderOverrides
    from segy.schema import SegySpec
    from upath import UPath

    from mdio.builder.schemas.v1.dataset import Dataset
    from mdio.builder.templates.base import AbstractDatasetTemplate
    from mdio.segy.file import SegyFileArguments
    from mdio.segy.geometry import GridOverrides

logger = logging.getLogger(__name__)


def run_segy_ingestion(  # noqa PLR0913
    segy_spec: SegySpec,
    mdio_template: AbstractDatasetTemplate,
    input_path: UPath | Path | str,
    output_path: UPath | Path | str,
    overwrite: bool = False,
    grid_overrides: GridOverrides | None = None,
    segy_header_overrides: SegyHeaderOverrides | None = None,
) -> None:
    """Convert SEG-Y to MDIO.

    Pipeline phases: schema resolution, header analysis, index strategy, grid build,
    dataset build, data write.

    Args:
        segy_spec: The SEG-Y specification to use for the conversion.
        mdio_template: The MDIO template to use for the conversion.
        input_path: The universal path of the input SEG-Y file.
        output_path: The universal path for the output MDIO v1 file.
        overwrite: Whether to overwrite the output file if it already exists. Defaults to False.
        grid_overrides: Grid override configuration for non-standard geometries.
        segy_header_overrides: Option to override specific SEG-Y headers during ingestion.

    Raises:
        FileExistsError: If the output location already exists and overwrite is False.
        ValueError: If required fields are missing from segy_spec.
    """
    settings = MDIOSettings()

    validate_spec_in_template(segy_spec, mdio_template)

    input_path = _normalize_path(input_path)
    output_path = _normalize_path(output_path)

    if not overwrite and output_path.exists():
        err = f"Output location '{output_path.as_posix()}' exists. Set `overwrite=True` if intended."
        raise FileExistsError(err)

    segy_settings = SegyFileSettings(storage_options=input_path.storage_options)
    segy_file_kwargs: SegyFileArguments = {
        "url": input_path.as_posix(),
        "spec": segy_spec,
        "settings": segy_settings,
        "header_overrides": segy_header_overrides,
    }

    segy_file_info = get_segy_file_info(segy_file_kwargs)

    # Work on a private copy so the caller's template is never mutated by ingestion.
    mdio_template = copy.deepcopy(mdio_template)
    mdio_template = apply_runtime_units(mdio_template, segy_file_info)

    schema = SchemaResolver().resolve(mdio_template, grid_overrides)

    # NonBinned / HasDuplicates change the dim layout; sync it back onto the template copy.
    if grid_overrides and (grid_overrides.non_binned or grid_overrides.has_duplicates):
        mdio_template.apply_resolved_dimensions(
            dim_names=tuple(d.name for d in schema.dimensions),
            chunk_shape=schema.chunk_shape,
        )

    analyzer = HeaderAnalyzer()
    requirements = analyzer.requirements_from_schema(schema)
    parsed_headers = analyzer.analyze(
        segy_file_kwargs=segy_file_kwargs,
        requirements=requirements,
        num_traces=segy_file_info.num_traces,
    )

    synthesize_dims = getattr(mdio_template, "synthesize_missing_dims", ())
    strategy = IndexStrategyRegistry().create_strategy(
        grid_overrides=grid_overrides,
        synthesize_dims=synthesize_dims,
        template=mdio_template,
    )
    logger.info("Using strategy: %s", strategy.name)

    indexed_headers = strategy.transform_headers(parsed_headers)
    dim_names = tuple(d.name for d in schema.dimensions if d.is_spatial)
    dimensions = strategy.compute_dimensions(indexed_headers, dim_names)

    # Computed dims (e.g. shot_index for OBN) require an opt-in grid override; if they
    # are missing here, fail with a clear message instead of an opaque downstream KeyError.
    produced_dim_names = {d.name for d in dimensions}
    missing_computed = [
        d.name
        for d in schema.dimensions
        if d.is_spatial and d.is_calculated and d.name not in produced_dim_names
    ]
    if missing_computed:
        err = (
            f"Required computed fields {sorted(missing_computed)} for template "
            f"{mdio_template.name} not found after grid overrides. "
            f"Please ensure the correct grid overrides are applied."
        )
        raise ValueError(err)

    sample_labels = segy_file_info.sample_labels / 1000  # normalize
    if all(sample_labels.astype("int64") == sample_labels):
        sample_labels = sample_labels.astype("int64")

    vertical_dim_name = schema.dimensions[-1].name
    dimensions.append(Dimension(coords=sample_labels, name=vertical_dim_name))

    grid = Grid(dims=dimensions)
    grid_density_qc(grid, segy_file_info.num_traces)
    grid.build_map(indexed_headers)

    live_trace_count = int(np.sum(grid.live_mask))
    if live_trace_count != segy_file_info.num_traces:
        for dim_name in grid.dim_names:
            dim_min, dim_max = grid.get_min(dim_name), grid.get_max(dim_name)
            logger.warning("%s min: %s max: %s", dim_name, dim_min, dim_max)
        logger.warning("Ingestion grid shape: %s.", grid.shape)
        raise GridTraceCountError(live_trace_count, segy_file_info.num_traces)

    _, non_dim_coords = get_coordinates(grid, indexed_headers, mdio_template)
    header_dtype = to_structured_type(segy_spec.trace.header.dtype)

    mdio_ds: Dataset = DatasetFactory().build(
        template=mdio_template,
        schema=schema,
        dimensions=dimensions,
        header_dtype=header_dtype,
        include_raw_headers=should_include_raw_headers(),
    )

    grid_overrides_dict = None
    if grid_overrides is not None:
        grid_overrides_dict = grid_overrides.model_dump(by_alias=True, exclude_defaults=True)
        if grid_overrides.replace_dims is not None:
            grid_overrides_dict["non_binned_dims"] = grid_overrides.replace_dims

    add_grid_override_to_metadata(dataset=mdio_ds, grid_overrides=grid_overrides_dict)

    xr_dataset = to_xarray_dataset(mdio_ds=mdio_ds)

    drop_vars_delayed: list[str] = []
    xr_dataset, drop_vars_delayed = populate_dim_coordinates(xr_dataset, grid, drop_vars_delayed)
    xr_dataset, drop_vars_delayed = populate_non_dim_coordinates(
        xr_dataset,
        grid,
        non_dim_coords,
        drop_vars_delayed,
        segy_file_info.coordinate_scalar,
    )

    if settings.save_segy_file_header:
        xr_dataset = add_segy_file_headers(xr_dataset, segy_file_info)

    xr_dataset.trace_mask.data[:] = grid.live_mask

    to_mdio(xr_dataset, output_path=output_path, mode="w", compute=False)

    unindexed_dims = [d for d in xr_dataset.dims if d not in xr_dataset.coords]
    for d in unindexed_dims:
        if d in drop_vars_delayed:
            drop_vars_delayed.remove(d)

    meta_ds = xr_dataset[drop_vars_delayed + ["trace_mask"]]
    to_mdio(meta_ds, output_path=output_path, mode="r+", compute=True)

    xr_dataset = xr_dataset.drop_vars(drop_vars_delayed)

    blocked_io.to_zarr(
        segy_file_kwargs=segy_file_kwargs,
        output_path=output_path,
        grid_map=grid.map,
        dataset=xr_dataset,
        data_variable_name=schema.default_variable_name,
    )
