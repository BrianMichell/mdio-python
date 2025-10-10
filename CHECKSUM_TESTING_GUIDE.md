# CRC32C Checksum Testing Guide

## Quick Start

### 1. Install Dependencies

```bash
# Install the package with the new google-crc32c dependency
uv sync
# or
pip install -e .
```

### 2. Run an Ingestion

```python
from mdio.converters.segy import segy_to_mdio
from segy import SegySpec
from mdio.builder.templates import SeismicTemplate3D

# Load or create your SEG-Y spec
spec = SegySpec.from_registry("some_spec")

# Create template
template = SeismicTemplate3D()

# Run ingestion (checksum will be calculated automatically)
segy_to_mdio(
    segy_spec=spec,
    mdio_template=template,
    input_path="input.segy",
    output_path="output.mdio",
    overwrite=True
)
```

### 3. Verify Checksum Was Stored

```python
import zarr

# Open the output store
store = zarr.open_group("output.mdio", mode="r")

# Check the checksum attributes
print(f"CRC32C: {store.attrs['segy_input_crc32c']}")
print(f"Algorithm: {store.attrs['crc32c_algorithm']}")
print(f"Scope: {store.attrs['checksum_scope']}")
print(f"Library: {store.attrs['checksum_library']}")
```

Expected output:
```
CRC32C: 0x12345678
Algorithm: CRC32C
Scope: full_file
Library: google-crc32c
```

## Manual Verification

### Option 1: Using Python Script

Create a script to manually calculate the CRC32C of your SEG-Y file:

```python
#!/usr/bin/env python3
"""Manually calculate CRC32C checksum of a SEG-Y file."""

import sys
import google_crc32c

def calculate_file_crc32c(filepath: str, chunk_size: int = 1024 * 1024) -> int:
    """Calculate CRC32C checksum of a file.
    
    Args:
        filepath: Path to file
        chunk_size: Size of chunks to read (default 1MB)
        
    Returns:
        CRC32C checksum as integer
    """
    checksum = google_crc32c.Checksum()
    
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            checksum.update(chunk)
    
    return int.from_bytes(checksum.digest(), byteorder='big')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <segy_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    crc = calculate_file_crc32c(filepath)
    print(f"CRC32C: 0x{crc:08x}")
    print(f"CRC32C (decimal): {crc}")
```

Usage:
```bash
python calculate_crc32c.py input.segy
```

### Option 2: Using Command Line Tool

If you have the `crc32c` command-line tool installed:

```bash
# Install crc32c CLI (if available)
pip install crc32c-cli  # if such a package exists

# Or use a similar tool
crc32c input.segy
```

### Option 3: Compare with Reference Implementation

```python
import google_crc32c

def verify_checksum(segy_path: str, mdio_path: str) -> bool:
    """Verify that the stored checksum matches the actual file checksum.
    
    Args:
        segy_path: Path to original SEG-Y file
        mdio_path: Path to output MDIO store
        
    Returns:
        True if checksums match, False otherwise
    """
    import zarr
    
    # Calculate actual checksum
    checksum = google_crc32c.Checksum()
    with open(segy_path, 'rb') as f:
        while True:
            chunk = f.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            checksum.update(chunk)
    
    actual_crc = int.from_bytes(checksum.digest(), byteorder='big')
    
    # Get stored checksum
    store = zarr.open_group(mdio_path, mode='r')
    stored_crc_hex = store.attrs['segy_input_crc32c']
    stored_crc = int(stored_crc_hex, 16)
    
    print(f"Actual CRC32C:  0x{actual_crc:08x}")
    print(f"Stored CRC32C:  {stored_crc_hex}")
    print(f"Match: {actual_crc == stored_crc}")
    
    return actual_crc == stored_crc

# Usage
verify_checksum("input.segy", "output.mdio")
```

## Test Cases

### Test Case 1: Small File (Single Chunk)

```python
# Create a small test SEG-Y file or use existing small file
# Run ingestion
# Verify checksum matches manual calculation
```

### Test Case 2: Large File (Multiple Chunks)

```python
# Use a large SEG-Y file that will be processed by multiple workers
# Verify checksum matches manual calculation
# This tests the distributed combination logic
```

### Test Case 3: Sparse Grid (with Dead Traces)

```python
# Use a SEG-Y file that results in dead traces in the grid
# Verify checksum still covers entire file (including dead trace data)
```

### Test Case 4: Different Storage Backends

```python
# Test with local filesystem
segy_to_mdio(..., input_path="file:///path/to/input.segy")

# Test with S3
segy_to_mdio(..., input_path="s3://bucket/input.segy")

# Test with GCS
segy_to_mdio(..., input_path="gs://bucket/input.segy")

# Verify checksums are identical for the same file
```

## Debugging

### Enable Detailed Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("mdio.segy")
logger.setLevel(logging.DEBUG)

# Run ingestion to see detailed checksum calculation logs
```

### Inspect Intermediate Values

Add temporary logging to track checksum calculation:

```python
# In blocked_io.py, add logging in _combine_crc32c_checksums()
print(f"Combining {len(checksum_parts)} partial checksums")
for offset, crc, length in sorted_parts:
    print(f"  Offset: {offset}, CRC: 0x{crc:08x}, Length: {length}")
```

### Check for Byte Alignment Issues

Verify that checksum parts have no gaps:

```python
# The _combine_crc32c_checksums() function already checks for gaps
# If you see "Gap or overlap detected" error, check:
# 1. Trace size calculations
# 2. Byte offset calculations  
# 3. Worker chunk boundaries
```

## Common Issues

### Issue: Import Error for google_crc32c

**Problem**: `ModuleNotFoundError: No module named 'google_crc32c'`

**Solution**: 
```bash
uv sync
# or
pip install google-crc32c>=1.5.0
```

### Issue: Checksum Mismatch

**Problem**: Stored checksum doesn't match manual calculation

**Possible Causes**:
1. Byte order issue in conversion
2. Gap in byte ranges being checksummed
3. Incorrect trace size calculation

**Debug Steps**:
1. Add logging to see all byte ranges being checksummed
2. Verify header checksum separately (3600 bytes)
3. Check trace size calculation matches file structure

### Issue: Performance Degradation

**Problem**: Ingestion is slower with checksum

**Solutions**:
1. Verify using hardware-accelerated `google-crc32c` (not pure Python)
2. Check that raw bytes are being read efficiently
3. Profile to see if checksum calculation is the bottleneck

## Validation Script

Complete validation script:

```python
#!/usr/bin/env python3
"""Validate CRC32C implementation for SEG-Y ingestion."""

import sys
import google_crc32c
import zarr
from pathlib import Path

def calculate_file_checksum(filepath: Path) -> int:
    """Calculate CRC32C of entire file."""
    checksum = google_crc32c.Checksum()
    with open(filepath, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            checksum.update(chunk)
    return int.from_bytes(checksum.digest(), byteorder='big')

def validate_ingestion_checksum(segy_path: str, mdio_path: str) -> None:
    """Validate that ingestion checksum matches actual file."""
    print(f"Validating checksum for {segy_path} -> {mdio_path}")
    print("-" * 60)
    
    # Calculate actual file checksum
    print("Calculating actual file checksum...")
    actual_crc = calculate_file_checksum(Path(segy_path))
    print(f"Actual CRC32C:  0x{actual_crc:08x}")
    
    # Get stored checksum
    print("\nReading stored checksum...")
    store = zarr.open_group(mdio_path, mode='r')
    
    if 'segy_input_crc32c' not in store.attrs:
        print("ERROR: No checksum found in store attributes!")
        sys.exit(1)
    
    stored_crc_hex = store.attrs['segy_input_crc32c']
    stored_crc = int(stored_crc_hex, 16)
    print(f"Stored CRC32C:  {stored_crc_hex}")
    
    # Compare
    print("\n" + "=" * 60)
    if actual_crc == stored_crc:
        print("✓ VALIDATION PASSED - Checksums match!")
    else:
        print("✗ VALIDATION FAILED - Checksum mismatch!")
        print(f"  Expected: 0x{actual_crc:08x}")
        print(f"  Got:      {stored_crc_hex}")
        sys.exit(1)
    
    # Print additional metadata
    print("\nChecksum Metadata:")
    print(f"  Algorithm: {store.attrs.get('crc32c_algorithm', 'N/A')}")
    print(f"  Scope: {store.attrs.get('checksum_scope', 'N/A')}")
    print(f"  Library: {store.attrs.get('checksum_library', 'N/A')}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <segy_file> <mdio_output>")
        sys.exit(1)
    
    validate_ingestion_checksum(sys.argv[1], sys.argv[2])
```

Save as `validate_checksum.py` and run:

```bash
python validate_checksum.py input.segy output.mdio
```

