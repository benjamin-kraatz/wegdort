# Snapshot Format

Wegdort persists stores with a compact custom binary snapshot. The format is
little-endian, versioned, and stable for v1.

Snapshots are written and loaded through `Store::save` and `Store::load`, or
through the equivalent reader/writer and byte-buffer APIs. The snapshot captures
the store metric, fixed dimensions, vector ids, and raw `f32` vector rows. It
does not store external metadata for vector ids or internal cached search data.

## Version 1 Layout

The v1 file starts with a fixed 28-byte header:

| Field | Size | Description |
| --- | ---: | --- |
| Magic bytes | 8 bytes | `WEGDORT\0` |
| Format version | 2 bytes | `1` as little-endian `u16` |
| Metric id | 1 byte | `1` cosine, `2` dot product, `3` squared L2 |
| Reserved | 1 byte | Must be `0` |
| Dimensions | 8 bytes | Vector dimension as little-endian `u64` |
| Vector count | 8 bytes | Number of vector rows as little-endian `u64` |

The header is followed by `vector_count` rows. Each row contains:

| Field | Size | Description |
| --- | ---: | --- |
| Vector id | 8 bytes | Caller-supplied id as little-endian `u64` |
| Vector values | `dimensions * 4` bytes | Raw little-endian `f32` values |

Rows are stored contiguously in snapshot order. Loading preserves the stored ids
and vector values, but callers should not rely on iteration order as a stable
semantic contract because removals use swap-remove internally.

## Validation

`Store::load`, `Store::load_reader`, and `Store::from_bytes` reject snapshots
that do not match the v1 format. Validation currently includes:

- missing or invalid magic bytes;
- unsupported format versions;
- invalid metric ids;
- non-zero reserved header byte;
- zero dimensions;
- payload lengths that do not match the header;
- duplicate vector ids;
- non-finite vector values;
- zero vectors when the metric is cosine similarity.

## Compatibility

Version 1 is the current stable snapshot format. Future incompatible changes
should increment the format version. Compatible extensions should use explicit
reserved fields or a new versioned layout rather than changing the meaning of
existing bytes.

The format is optimized for compact files and sequential read/write. It is not a
memory-mapped index format and does not include approximate-nearest-neighbor
state.
