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


def debug_compare_raw_vs_processed(segy_file, trace_index=0):
    """Debug function to compare raw filesystem data vs processed data."""
    from segy.indexing import HeaderIndexer

    # Create a fresh indexer to get raw data
    indexer = HeaderIndexer(
        segy_file.fs,
        segy_file.url,
        segy_file.spec.trace,
        segy_file.num_traces,
        transform_pipeline=None  # No transforms = raw data
    )

    # Get raw data directly from filesystem
    raw_data = indexer[trace_index]

    # Get processed data with transforms
    processed_data = segy_file.header[trace_index]

    print("=== Raw vs Processed Comparison ===")
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Processed data shape: {processed_data.shape}")

    if hasattr(raw_data, 'dtype') and raw_data.dtype.names:
        if 'inline_number' in raw_data.dtype.names:
            print(f"Raw inline_number: {raw_data['inline_number']}")
            print(f"Raw inline_number (hex): {raw_data['inline_number']:08x}")
            print(f"Processed inline_number: {processed_data['inline_number']}")
            print(f"Processed inline_number (hex): {processed_data['inline_number']:08x}")
            print(f"Are they equal? {raw_data['inline_number'] == processed_data['inline_number']}")

    return raw_data, processed_data


class HeaderRawTransformedAccessor:
    """Utility class to access both raw and transformed header data with single filesystem read.

    This class works as a consumer of SegyFile objects without modifying the package.
    It achieves the goal by:
    1. Reading raw data from filesystem once
    2. Applying transforms to get transformed data
    3. Keeping both versions available

    The transforms used in SEG-Y processing are reversible:
    - ByteSwapTransform: Self-inverse (swapping twice returns to original)
    - IbmFloatTransform: Can be reversed by swapping direction
    """

    def __init__(self, segy_file: SegyFile):
        """Initialize with a SegyFile instance.

        Args:
            segy_file: The SegyFile instance to work with
        """
        self.segy_file = segy_file
        self.header_indexer = segy_file.header
        self.transform_pipeline = self.header_indexer.transform_pipeline

        # Debug: Print transform pipeline information
        import sys
        print(f"Debug: System endianness: {sys.byteorder}")
        print(f"Debug: File endianness: {self.segy_file.spec.endianness}")
        print(f"Debug: Transform pipeline has {len(self.transform_pipeline.transforms)} transforms:")
        for i, transform in enumerate(self.transform_pipeline.transforms):
            print(f"  Transform {i}: {type(transform).__name__}")
            if hasattr(transform, 'target_order'):
                print(f"    Target order: {transform.target_order}")
            if hasattr(transform, 'direction'):
                print(f"    Direction: {transform.direction}")
            if hasattr(transform, 'keys'):
                print(f"    Keys: {transform.keys}")

    def get_raw_and_transformed(
        self, indices: int | list[int] | np.ndarray | slice
    ) -> tuple[NDArray, NDArray]:
        """Get both raw and transformed header data with single filesystem read.

        Args:
            indices: Which headers to retrieve (int, list, ndarray, or slice)

        Returns:
            Tuple of (raw_headers, transformed_headers)
        """
        # Get the transformed data using the normal API
        # This reads from filesystem and applies transforms
        transformed_data = self.header_indexer[indices]

        print(f"Debug: Transformed data shape: {transformed_data.shape}")
        if hasattr(transformed_data, 'dtype') and transformed_data.dtype.names:
            print(f"Debug: Transformed data dtype names: {transformed_data.dtype.names[:5]}...")  # First 5 fields
            if 'inline_number' in transformed_data.dtype.names:
                print(f"Debug: First transformed inline_number: {transformed_data['inline_number'][0]}")
                print(f"Debug: First transformed inline_number (hex): {transformed_data['inline_number'][0]:08x}")

        # Now reverse the transforms to get back to raw data
        raw_data = self._reverse_transforms(transformed_data)

        print(f"Debug: Raw data shape: {raw_data.shape}")
        if hasattr(raw_data, 'dtype') and raw_data.dtype.names:
            if 'inline_number' in raw_data.dtype.names:
                print(f"Debug: First raw inline_number: {raw_data['inline_number'][0]}")
                print(f"Debug: First raw inline_number (hex): {raw_data['inline_number'][0]:08x}")

        return raw_data, transformed_data

    def _reverse_transforms(self, transformed_data: NDArray) -> NDArray:
        """Reverse the transform pipeline to get raw data from transformed data.

        Args:
            transformed_data: Data that has been processed through the transform pipeline

        Returns:
            Raw data equivalent to what was read directly from filesystem
        """
        # Start with the transformed data
        raw_data = transformed_data.copy() if hasattr(transformed_data, 'copy') else transformed_data

        print(f"Debug: Starting reversal with {len(self.transform_pipeline.transforms)} transforms")

        # Apply transforms in reverse order with reversed operations
        for i, transform in enumerate(reversed(self.transform_pipeline.transforms)):
            print(f"Debug: Reversing transform {len(self.transform_pipeline.transforms)-1-i}: {type(transform).__name__}")
            if 'inline_number' in raw_data.dtype.names:
                print(f"Debug: Before reversal - inline_number: {raw_data['inline_number'][0]:08x}")
            raw_data = self._reverse_single_transform(raw_data, transform)
            if 'inline_number' in raw_data.dtype.names:
                print(f"Debug: After reversal - inline_number: {raw_data['inline_number'][0]:08x}")

        return raw_data

    def _reverse_single_transform(self, data: NDArray, transform: Transform) -> NDArray:
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
            print(f"Debug: Reversing byte swap (target was: {transform.target_order})")

            # Get current data endianness
            current_endianness = get_endianness(data)
            print(f"Debug: Current data endianness: {current_endianness}")

            # If transform was converting TO little-endian, we need to convert TO big-endian
            if transform.target_order == Endianness.LITTLE:
                reverse_target = Endianness.BIG
            else:
                reverse_target = Endianness.LITTLE

            print(f"Debug: Reversing to target: {reverse_target}")
            reverse_transform = ByteSwapTransform(reverse_target)
            result = reverse_transform.apply(data)

            if 'inline_number' in data.dtype.names:
                print(f"Debug: Byte swap reversal - before: {data['inline_number'][0]:08x}, after: {result['inline_number'][0]:08x}")
            return result

        elif isinstance(transform, IbmFloatTransform):
            # Reverse IBM float conversion by swapping direction
            reverse_direction = "to_ibm" if transform.direction == "to_ieee" else "to_ieee"
            print(f"Debug: Applying IBM float reversal (direction: {transform.direction} -> {reverse_direction})")
            reverse_transform = IbmFloatTransform(reverse_direction, transform.keys)
            return reverse_transform.apply(data)

        else:
            # For unknown transforms, return data unchanged
            # This maintains compatibility if new transforms are added
            print(f"Warning: Unknown transform type {type(transform).__name__}, cannot reverse")
            return data


def get_header_raw_and_transformed(
    segy_file: SegyFile,
    indices: int | list[int] | np.ndarray | slice
) -> tuple[NDArray, NDArray]:
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
    accessor = HeaderRawTransformedAccessor(segy_file)
    return accessor.get_raw_and_transformed(indices)
