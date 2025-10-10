# CRC32C Checksum Implementation for SEG-Y Ingestion

## Overview
This document describes the implementation of distributed CRC32C checksum calculation for SEG-Y files during the `segy_to_mdio` ingestion process.

## Implementation Summary

The CRC32C checksum is calculated for **every single byte** of the input SEG-Y file in a distributed manner across parallel workers. The checksum covers:
- Text header (3200 bytes)
- Binary header (400 bytes)  
- All trace data (headers + samples for every trace)

### Architecture

```
SEG-Y File (3600 + trace_data bytes)
├── Headers (3600 bytes) → CRC32C calculated in info_worker
└── Trace Data (N traces) → CRC32C calculated in parallel by trace_workers
    ├── Worker 1: traces 0-999 → partial CRC32C
    ├── Worker 2: traces 1000-1999 → partial CRC32C
    └── Worker N: traces ... → partial CRC32C
                ↓
    Combined in blocked_io.to_zarr() using google_crc32c.extend()
                ↓
    Final CRC32C = combine(header_crc, trace_data_crc)
                ↓
    Stored in Zarr attributes
```

## Changes Made

### 1. Dependencies (`pyproject.toml`)
- Added `google-crc32c>=1.5.0` to dependencies

### 2. Data Structures (`src/mdio/segy/_workers.py`)

#### Modified `SegyFileInfo` dataclass:
```python
@dataclass
class SegyFileInfo:
    # ... existing fields ...
    headers_crc32c: int  # CRC32C of text + binary headers (3600 bytes)
    trace_data_offset: int  # Byte offset where trace data starts
```

#### New `TraceWorkerResult` dataclass:
```python
@dataclass
class TraceWorkerResult:
    statistics: SummaryStatistics | None
    partial_crc32c: int  # Partial CRC32C checksum for this chunk
    byte_offset: int  # Starting byte offset in SEG-Y file
    byte_length: int  # Number of bytes checksummed
```

### 3. Header Checksum Calculation (`src/mdio/segy/_workers.py`)

#### Modified `info_worker()`:
- Reads raw text header bytes (3200 bytes) via `segy_file.fs.read_block()`
- Reads raw binary header bytes (400 bytes) via `segy_file.fs.read_block()`
- Calculates CRC32C for the combined 3600 bytes
- Returns checksum in `SegyFileInfo`

### 4. Trace Data Checksum Calculation (`src/mdio/segy/_workers.py`)

#### Modified `trace_worker()`:
- Calculates byte offset and length for the trace range being processed
- Reads raw bytes from SEG-Y file using `segy_file.fs.read_block()`
- Includes all traces in the contiguous range (including dead traces)
- Calculates partial CRC32C checksum
- Returns `TraceWorkerResult` with statistics and checksum info

### 5. Checksum Combination (`src/mdio/segy/blocked_io.py`)

#### New `_combine_crc32c_checksums()` function:
- Sorts partial checksums by byte offset
- Verifies no gaps or overlaps in byte ranges
- Uses `google_crc32c.extend()` to combine partial checksums
- Returns combined trace data CRC32C

#### Modified `to_zarr()`:
- Collects partial checksums from all workers
- Combines them using `_combine_crc32c_checksums()`
- Returns tuple of `(SummaryStatistics, trace_data_crc32c)`

### 6. Final Checksum Storage (`src/mdio/converters/segy.py`)

#### New `_combine_header_and_trace_crc32c()` function:
- Combines header CRC32C with trace data CRC32C
- Uses `google_crc32c.extend()` for proper CRC32C combination
- Returns final file checksum

#### Modified `segy_to_mdio()`:
- Captures both statistics and trace_data_crc32c from `blocked_io.to_zarr()`
- Calculates total trace data length
- Combines header and trace data checksums
- Stores final CRC32C in Zarr root attributes:
  ```python
  {
      "segy_input_crc32c": "0x12345678",  # Hex format
      "crc32c_algorithm": "CRC32C",
      "checksum_scope": "full_file",
      "checksum_library": "google-crc32c"
  }
  ```

## Key Design Decisions

1. **Distributed Calculation**: Checksum calculated in parallel across workers for performance
2. **Raw Bytes**: Uses `fs.read_block()` to ensure no conversions affect the checksum
3. **Contiguous Ranges**: Each worker checksums contiguous byte ranges (including dead traces)
4. **CRC32C Combination**: Uses `google_crc32c.extend()` for mathematically correct combination
5. **Storage Location**: Stored in Zarr root-level attributes for easy access

## Testing Recommendations

1. **Verify Checksum Accuracy**:
   - Calculate checksum using external tool (e.g., `crc32c` CLI utility)
   - Compare with distributed calculation result
   
2. **Test Edge Cases**:
   - Files with dead traces
   - Very small files (single chunk)
   - Very large files (many chunks)
   - Files on different storage backends (local, S3, GCS)

3. **Performance Testing**:
   - Measure overhead of checksum calculation
   - Compare with/without checksum enabled

## Usage

After ingestion, the checksum can be accessed from the Zarr store:

```python
import zarr

store = zarr.open_group("output.mdio", mode="r")
checksum = store.attrs["segy_input_crc32c"]
print(f"SEG-Y file CRC32C: {checksum}")
```

## Installation

To use this feature, install the package with the updated dependencies:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

The `google-crc32c` package provides hardware-accelerated CRC32C calculation using CPU extensions when available.

## Future Enhancements

1. Add environment variable to optionally disable checksum calculation
2. Add logging of checksum calculation progress
3. Support verification mode (compare stored checksum with recalculated)
4. Add checksum to CLI output
5. Document checksum in user-facing documentation

## Files Modified

1. `pyproject.toml` - Added dependency
2. `src/mdio/segy/_workers.py` - Data structures and worker implementations
3. `src/mdio/segy/blocked_io.py` - Checksum combination logic
4. `src/mdio/converters/segy.py` - Final integration and storage

## Performance Impact

The checksum calculation adds minimal overhead:
- Header checksum: ~negligible (3600 bytes, calculated once)
- Trace data checksum: Calculated while data is already in memory
- No additional file reads beyond what's already happening
- Combination overhead: ~O(N) where N is number of chunks

