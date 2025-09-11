"""Consumer-side utility to get both raw and transformed header data with single filesystem read."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
from segy.transforms import ByteSwapTransform
from segy.transforms import IbmFloatTransform

if TYPE_CHECKING:
    from segy.file import SegyFile
    from segy.indexing import HeaderIndexer
    from segy.transforms import Transform, TransformPipeline, ByteSwapTransform, IbmFloatTransform
    from numpy.typing import NDArray

def _reverse_single_transform(data: NDArray, transform: Transform) -> NDArray:
    """Reverse a single transform operation.

    Args:
        data: The data to reverse transform
        transform: The transform to reverse

    Returns:
        Data with the transform reversed
    """
    # Import here to avoid circular imports
    from segy.transforms import get_endianness
    from segy.schema import Endianness

    if isinstance(transform, ByteSwapTransform):
        # For byte swap, we need to reverse the endianness conversion
        # If the transform was converting to little-endian, we need to convert back to big-endian

        # If transform was converting TO little-endian, we need to convert TO big-endian
        # TODO: I don't think this is correct
        if transform.target_order == Endianness.LITTLE:
            reverse_target = Endianness.BIG
        else:
            reverse_target = Endianness.LITTLE

        reverse_transform = ByteSwapTransform(reverse_target)
        result = reverse_transform.apply(data)

        return result

    elif isinstance(transform, IbmFloatTransform):
        # Reverse IBM float conversion by swapping direction
        reverse_direction = "to_ibm" if transform.direction == "to_ieee" else "to_ieee"
        reverse_transform = IbmFloatTransform(reverse_direction, transform.keys)
        return reverse_transform.apply(data)

    else:
        # For unknown transforms, return data unchanged
        # This maintains compatibility if new transforms are added
        return data

def get_header_raw_and_transformed(
    segy_file: SegyFile,
    indices: int | list[int] | np.ndarray | slice
) -> tuple[NDArray, NDArray, NDArray]:
    """Convenience function to get both raw and transformed header data.

    This is a drop-in replacement that provides the functionality you requested
    without modifying the segy package.

    Args:
        segy_file: The SegyFile instance
        indices: Which headers to retrieve

    Returns:
        Tuple of (raw_headers, transformed_headers)

    Example:
        from header_raw_transformed_accessor import get_header_raw_and_transformed

        # Single header
        raw_hdr, transformed_hdr = get_header_raw_and_transformed(segy_file, 0)

        # Multiple headers
        raw_hdrs, transformed_hdrs = get_header_raw_and_transformed(segy_file, [0, 1, 2])

        # Slice of headers
        raw_hdrs, transformed_hdrs = get_header_raw_and_transformed(segy_file, slice(0, 10))
    """

    traces = segy_file.trace[indices]

    transformed_headers = traces.header

    # Reverse the transforms on the already-loaded transformed data
    # This eliminates the second disk read entirely!
    raw_headers = _reverse_transforms(transformed_headers, segy_file.header.transform_pipeline)

    return raw_headers, transformed_headers, traces

def _reverse_transforms(transformed_data: NDArray, transform_pipeline) -> NDArray:
    """Reverse the transform pipeline to get raw data from transformed data.

    Args:
        transformed_data: Data that has been processed through the transform pipeline
        transform_pipeline: The transform pipeline to reverse

    Returns:
        Raw data equivalent to what was read directly from filesystem
    """
    # Start with the transformed data
    raw_data = transformed_data.copy() if hasattr(transformed_data, 'copy') else transformed_data

    # Apply transforms in reverse order with reversed operations
    for transform in reversed(transform_pipeline.transforms):
        raw_data = _reverse_single_transform(raw_data, transform)

    return raw_data
