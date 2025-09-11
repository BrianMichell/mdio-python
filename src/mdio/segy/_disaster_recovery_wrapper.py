"""Consumer-side utility to get both raw and transformed header data with single filesystem read."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from segy.file import SegyFile
    from segy.transforms import Transform, ByteSwapTransform, IbmFloatTransform
    from numpy.typing import NDArray

def _reverse_single_transform(data: NDArray, transform: Transform) -> NDArray:
    """Reverse a single transform operation."""
    from segy.schema import Endianness

    if isinstance(transform, ByteSwapTransform):
        # Reverse the endianness conversion
        # TODO: I don't think this is correct
        if transform.target_order == Endianness.LITTLE:
            reverse_target = Endianness.BIG
        else:
            reverse_target = Endianness.LITTLE

        reverse_transform = ByteSwapTransform(reverse_target)
        return reverse_transform.apply(data)

    elif isinstance(transform, IbmFloatTransform):
        # Reverse IBM float conversion
        reverse_direction = "to_ibm" if transform.direction == "to_ieee" else "to_ieee"
        reverse_transform = IbmFloatTransform(reverse_direction, transform.keys)
        return reverse_transform.apply(data)

    else:
        # For unknown transforms, return data unchanged
        return data

def get_header_raw_and_transformed(
    segy_file: SegyFile,
    indices: int | list[int] | NDArray | slice
) -> tuple[NDArray, NDArray, NDArray]:
    """Get both raw and transformed header data.

    Args:
        segy_file: The SegyFile instance
        indices: Which headers to retrieve

    Returns:
        Tuple of (raw_headers, transformed_headers, traces)
    """

    traces = segy_file.trace[indices]
    transformed_headers = traces.header

    # Reverse transforms to get raw data
    raw_headers = _reverse_transforms(transformed_headers, segy_file.header.transform_pipeline)

    return raw_headers, transformed_headers, traces

def _reverse_transforms(transformed_data: NDArray, transform_pipeline) -> NDArray:
    """Reverse the transform pipeline to get raw data."""
    raw_data = transformed_data.copy() if hasattr(transformed_data, 'copy') else transformed_data

    # Apply transforms in reverse order
    for transform in reversed(transform_pipeline.transforms):
        raw_data = _reverse_single_transform(raw_data, transform)

    return raw_data
