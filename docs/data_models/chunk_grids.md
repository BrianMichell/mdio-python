```{eval-rst}
:tocdepth: 3
```

```{currentModule} mdio.builder.schemas.chunk_grid

```

# Chunk Grid Models

```{article-info}
:author: Altay Sansal
:date: "{sub-ref}`today`"
:read-time: "{sub-ref}`wordcount-minutes` min read"
:class-container: sd-p-0 sd-outline-muted sd-rounded-3 sd-font-weight-light
```

The variables in MDIO data model can represent different types of chunk grids.
These grids are essential for managing multi-dimensional data arrays efficiently.
In this breakdown, we will explore the chunk grid data models within the MDIO schema,
each serving a specific purpose in data handling and organization.

MDIO implements data models following the guidelines of the Zarr v3 spec and ZEPs:

- [Zarr core specification (version 3)](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html)
- [ZEP 1 — Zarr specification version 3](https://zarr.dev/zeps/accepted/ZEP0001.html)
- [ZEP 2 — Sharding codec](https://zarr.dev/zeps/accepted/ZEP0002.html)
- [ZEP 3 — Variable chunking](https://zarr.dev/zeps/draft/ZEP0003.html)

## Regular Grid

The regular grid models are designed to represent a rectangular and regularly
paced chunk grid.

```{eval-rst}
.. autosummary::
   RegularChunkGrid
   RegularChunkShape
```

For 1D array with `size = 31`{l=python}, we can divide it into 5 equally sized
chunks. Note that the last chunk will be truncated to match the size of the array.

`{ "name": "regular", "configuration": { "chunkShape": [7] } }`{l=json}

Using the above schema resulting array chunks will look like this:

```bash
 ←─ 7 ─→ ←─ 7 ─→ ←─ 7 ─→ ←─ 7 ─→  ↔ 3
┌───────┬───────┬───────┬───────┬───┐
└───────┴───────┴───────┴───────┴───┘
```

For 2D array with shape `rows, cols = (7, 17)`{l=python}, we can divide it into 9
equally sized chunks.

`{ "name": "regular", "configuration": { "chunkShape": [3, 7] } }`{l=json}

Using the above schema, the resulting 2D array chunks will look like below.
Note that the rows and columns are conceptual and visually not to scale.

```bash
 ←─ 7 ─→ ←─ 7 ─→  ↔ 3
┌───────┬───────┬───┐
│       ╎       ╎   │  ↑
│       ╎       ╎   │  3
│       ╎       ╎   │  ↓
├╶╶╶╶╶╶╶┼╶╶╶╶╶╶╶┼╶╶╶┤
│       ╎       ╎   │  ↑
│       ╎       ╎   │  3
│       ╎       ╎   │  ↓
├╶╶╶╶╶╶╶┼╶╶╶╶╶╶╶┼╶╶╶┤
│       ╎       ╎   │  ↕ 1
└───────┴───────┴───┘
```

## Rectilinear Grid

The [RectilinearChunkGrid](RectilinearChunkGrid) model extends
the concept of chunk grids to accommodate rectangular and irregularly spaced chunks.
This model is useful in data structures where non-uniform chunk sizes are necessary.
[RectilinearChunkShape](RectilinearChunkShape) specifies the chunk sizes for each
dimension as a list allowing for irregular intervals.

```{eval-rst}
.. autosummary::
   RectilinearChunkGrid
   RectilinearChunkShape
```

:::{note}
It's important to ensure that the sum of the irregular spacings specified
in the `chunkShape` matches the size of the respective array dimension.
:::

For 1D array with `size = 39`{l=python}, we can divide it into 5 irregular sized
chunks.

`{ "name": "rectilinear", "configuration": { "chunkShape": [[10, 7, 5, 7, 10]] } }`{l=json}

Using the above schema resulting array chunks will look like this:

```bash
 ←── 10 ──→ ←─ 7 ─→ ← 5 → ←─ 7 ─→ ←── 10 ──→
┌──────────┬───────┬─────┬───────┬──────────┐
└──────────┴───────┴─────┴───────┴──────────┘
```

For 2D array with shape `rows, cols = (7, 25)`{l=python}, we can divide it into 12
rectilinear (rectangular bur irregular) chunks. Note that the rows and columns are
conceptual and visually not to scale.

`{ "name": "rectilinear", "configuration": { "chunkShape": [[3, 1, 3], [10, 5, 7, 3]] } }`{l=json}

```bash
 ←── 10 ──→ ← 5 → ←─ 7 ─→  ↔ 3
┌──────────┬─────┬───────┬───┐
│          ╎     ╎       ╎   │  ↑
│          ╎     ╎       ╎   │  3
│          ╎     ╎       ╎   │  ↓
├╶╶╶╶╶╶╶╶╶╶┼╶╶╶╶╶┼╶╶╶╶╶╶╶┼╶╶╶┤
│          ╎     ╎       ╎   │  ↕ 1
├╶╶╶╶╶╶╶╶╶╶┼╶╶╶╶╶┼╶╶╶╶╶╶╶┼╶╶╶┤
│          ╎     ╎       ╎   │  ↑
│          ╎     ╎       ╎   │  3
│          ╎     ╎       ╎   │  ↓
└──────────┴─────┴───────┴───┘
```

## Sharded Grid

The [ShardedChunkGrid](ShardedChunkGrid) model implements Zarr v3 sharding, which
organizes data in a two-level hierarchy for improved I/O performance:

- **Shards**: Outer containers stored as individual files/objects
- **Chunks**: Inner units within each shard for compression and fine-grained access

Sharding is particularly beneficial for cloud storage where reducing the number of
objects improves performance while still maintaining efficient partial reads.

```{eval-rst}
.. autosummary::
   ShardedChunkGrid
   ShardedChunkShape
```

:::{note}
Sharding is only supported with Zarr v3 format. The `shardShape` must be evenly
divisible by `chunkShape` along all dimensions.
:::

:::{important}
**Non-shardable dtype limitation**: Sharding is only applied to simple scalar-type
variables (e.g., `amplitude`, `cdp_x`, `cdp_y`). Variables with structured dtypes
(e.g., `headers`) or void/bytes types (e.g., `raw_headers`) will automatically use
regular chunking instead, as Zarr's sharding codec does not support these dtypes.
However, these variables will use the **shard shape as their chunk shape** to maintain
consistent I/O patterns across all variables.
:::

For a 2D array with shape `rows, cols = (256, 512)`{l=python}, we can configure
sharding with shard size `(128, 128)` and chunk size `(32, 32)`:

`{ "name": "sharded", "configuration": { "shardShape": [128, 128], "chunkShape": [32, 32] } }`{l=json}

This creates a two-level structure where each shard contains multiple chunks:

```bash
 ←────────── 128 ──────────→ ←────────── 128 ──────────→ ...
┌─────────────────────────────────────────────────────────┐
│ ┌─────┬─────┬─────┬─────┐ ┌─────┬─────┬─────┬─────┐     │  ↑
│ │ 32  │ 32  │ 32  │ 32  │ │ 32  │ 32  │ 32  │ 32  │     │  │
│ ├─────┼─────┼─────┼─────┤ ├─────┼─────┼─────┼─────┤     │  │
│ │ 32  │ 32  │ 32  │ 32  │ │ 32  │ 32  │ 32  │ 32  │     │  128
│ ├─────┼─────┼─────┼─────┤ ├─────┼─────┼─────┼─────┤     │  │
│ │ 32  │ 32  │ 32  │ 32  │ │ 32  │ 32  │ 32  │ 32  │     │  │
│ ├─────┼─────┼─────┼─────┤ ├─────┼─────┼─────┼─────┤     │  │
│ │ 32  │ 32  │ 32  │ 32  │ │ 32  │ 32  │ 32  │ 32  │     │  ↓
│ └─────┴─────┴─────┴─────┘ └─────┴─────┴─────┴─────┘     │
│        Shard (0,0)               Shard (0,1)            │
├─────────────────────────────────────────────────────────┤
│        Shard (1,0)               Shard (1,1)       ...  │
└─────────────────────────────────────────────────────────┘
```

### Using Sharding with Templates

By default, MDIO templates do not use sharding. To enable sharding on a template,
set both `full_chunk_shape` and `full_shard_shape`:

```python
from mdio.builder.templates.seismic_3d_poststack import Seismic3DPostStackTemplate

# Create template with default chunking (no sharding)
template = Seismic3DPostStackTemplate(data_domain='depth')
print(template.full_shard_shape)  # None - sharding disabled

# Enable sharding by setting both chunk and shard shapes
template.full_chunk_shape = (32, 32, 128)    # Inner chunk sizes
template.full_shard_shape = (128, 128, 512)  # Outer shard sizes

# Build dataset - will use ShardedChunkGrid for scalar variables (amplitude, coordinates)
# Note: headers array will use RegularChunkGrid (structured dtypes don't support sharding)
dataset = template.build_dataset('Seismic3D', sizes=(256, 512, 1024))
```

### Ingestion with Sharding

When sharding is enabled, the SEG-Y ingestion process automatically iterates at the
shard level instead of the chunk level. This reduces the number of I/O operations
and improves performance, especially for cloud storage backends.

- **With sharding**: Ingestion iterates over shards (larger units)
- **Without sharding**: Ingestion iterates over chunks (default behavior)

During ingestion, sharding is applied only to simple scalar-type arrays (e.g., `amplitude`).
Arrays with non-shardable dtypes use regular chunking with the shard shape as their chunk
size. For example, with `full_chunk_shape=(32, 32, 32)` and `full_shard_shape=(256, 256, 256)`:

- `amplitude` array: chunks=(32, 32, 32), shards=(256, 256, 256)
- `headers` array (structured dtype): chunks=(256, 256), no sharding
- `raw_headers` array (void/bytes dtype): chunks=(256, 256), no sharding

## Model Reference

:::{dropdown} RegularChunkGrid
:animate: fade-in-slide-down

```{eval-rst}
.. autopydantic_model:: RegularChunkGrid

----------

.. autopydantic_model:: RegularChunkShape
```

:::
:::{dropdown} RectilinearChunkGrid
:animate: fade-in-slide-down

```{eval-rst}
.. autopydantic_model:: RectilinearChunkGrid

----------

.. autopydantic_model:: RectilinearChunkShape
```

:::
:::{dropdown} ShardedChunkGrid
:animate: fade-in-slide-down

```{eval-rst}
.. autopydantic_model:: ShardedChunkGrid

----------

.. autopydantic_model:: ShardedChunkShape
```

:::
