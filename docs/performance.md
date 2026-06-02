# Performance

Wegdort is optimized for exact in-memory search before adding approximate
nearest-neighbor indexes. The current search path uses bounded top-k selection,
so it keeps only the best `k` candidates instead of sorting every stored vector.

## Benchmarks

Run the serial benchmark suite with:

```sh
cargo bench
```

Run benchmarks with optional parallel search enabled:

```sh
cargo bench --features parallel
```

The Criterion suite currently covers:

- cosine, dot product, and squared L2 search;
- several store sizes for exact top-k search;
- insert, replacement upsert, and remove loops;
- binary snapshot save/load round trips;
- parallel exact search when the `parallel` feature is enabled.

Criterion writes reports under `target/criterion/`.

## Serial Search

`Store::search` and `SearchSnapshot::search` are always serial and deterministic.
They are the right default for small and medium stores, low-latency single-query
workloads, embedded callers, and users who want no runtime dependency beyond the
standard library.

Serial search keeps the crate dependency-light by default.

## Parallel Search

Enable Rayon-backed search with:

```toml
[dependencies]
wegdort = { path = ".", features = ["parallel"] }
```

Then call:

```rust
# use wegdort::{Metric, Store, VectorId};
# fn main() -> Result<(), wegdort::Error> {
# let mut store = Store::new(2, Metric::Dot)?;
# store.insert(VectorId::new(1), [1.0, 0.0])?;
let hits = store.search_parallel([1.0, 0.0], 10)?;
# Ok(())
# }
```

Parallel search is useful when stores are large enough that scoring dominates
Rayon scheduling overhead. It validates inputs exactly like serial search and
returns the same ordering.

The demo CLI also exposes the parallel path:

```sh
cargo run --features parallel --example demo_cli -- parallel dot 3 1.0 0.1 0.0
```

## Interpreting Results

Benchmark results depend on CPU, memory bandwidth, dimension count, metric, and
`k`. Compare serial and parallel results with the same store sizes and query
dimensions before choosing a default for an application.
