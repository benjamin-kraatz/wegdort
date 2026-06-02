# Wegdort

Wegdort is a planned lightweight vector store for fast in-memory search, with
optional persistent storage when durability matters. It is intended to be small,
cross-platform, dependency-conscious, and easy to embed in applications that need
local vector similarity search without running a separate database service.

The name comes from the German phrase "weg dort", which sounds close to
"vector".

> Project status: early rewrite. The current repository is being prepared for a
> from-scratch implementation. Public APIs, module names, and persistence
> details may change before the first stable release.

## Goals

- **Fast by default**: keep the hot path cache-friendly, allocation-light, and
  built around predictable exact search before adding more complex indexing.
- **Lightweight**: prefer a small core, focused modules, and carefully justified
  dependencies.
- **Easy to use**: expose a clear Rust API first, with a small number of obvious
  types and typed errors instead of surprising panics.
- **Cross-platform**: support common desktop, server, and mobile targets through
  portable Rust.
- **Embeddable**: store vectors in memory first, with optional persistent
  snapshots separated from the in-memory search path.
- **Well documented**: document public APIs in code and keep Markdown
  documentation accurate as the architecture evolves.

## Planned Features

- In-memory vector storage with a fixed dimension per store.
- Caller-supplied vector ids.
- Exact top-k flat search for the first production-ready search engine.
- Three core scoring modes:
  - cosine similarity, where higher scores are better;
  - dot product, where higher scores are better;
  - squared L2 distance, where lower distances are better.
- Typed errors for dimension mismatches, invalid input, storage failures, and
  persistence failures.
- Optional file-backed snapshots for persistent storage.
- Rust API first.
- Future Swift and TypeScript APIs through a dedicated bindings layer.

## Future Rust API Shape

The final API may differ, but wegdort should feel close to this:

```rust
use wegdort::{Metric, Store, VectorId};

fn main() -> Result<(), wegdort::Error> {
    let mut store = Store::new(3).with_metric(Metric::Cosine);

    store.insert(VectorId::new(1), [1.0, 0.0, 0.0])?;
    store.insert(VectorId::new(2), [0.0, 1.0, 0.0])?;
    store.insert(VectorId::new(3), [0.0, 0.2, 0.8])?;

    let matches = store.search([0.0, 0.0, 1.0], 2)?;

    for hit in matches {
        println!("id={} score={}", hit.id, hit.score);
    }

    Ok(())
}
```

The intended default is simple: create a store, insert vectors, search top-k
neighbors, and optionally save or load a snapshot.

## Architecture Direction

Wegdort should become a library-first Rust crate. A small CLI or demo binary may
exist later, but the crate API is the primary product.

The planned module structure is:

- `metrics`: cosine, dot product, squared L2, and score ordering semantics.
- `store`: the ergonomic user-facing vector store API.
- `storage`: compact in-memory id and vector layout.
- `search`: exact flat top-k search implementation.
- `persistence`: optional file-backed snapshots.
- `error`: typed errors used across public APIs.
- `bindings` or `ffi`: future Swift and TypeScript integration boundary.

Implementation should use modern Rust 2024 practices, strong types, contiguous
storage where practical, minimal allocations during search, and clear separation
between the in-memory hot path and optional persistence.

## Performance Philosophy

The first version should make exact flat search excellent before adding
approximate nearest-neighbor indexes. Flat search is simple, deterministic, easy
to test, and often fast enough for embedded or local workloads when implemented
with cache-friendly storage and tight metric loops.

Performance work should be measured. Future optimization work should include
benchmarks for metric calculation, insertion, deletion, top-k search, snapshot
save/load, and memory usage.

## Documentation Expectations

Public APIs must include Rustdoc comments with examples when useful. Larger
design decisions should be captured in Markdown so future contributors can
understand why the crate is structured the way it is.

Documentation should stay honest about what exists today versus what is planned.
Do not document planned behavior as implemented behavior.

## Bindings Roadmap

Rust is the first-class API. Swift and TypeScript support are planned for later
and should be built on top of a stable core API rather than driving the initial
storage design.

The bindings layer should keep ownership, memory safety, and error reporting
explicit. It should not leak internal storage details into language-specific
interfaces.

## Contributing

The rewrite is still early. Before implementation work, agree on the intended
API and behavior, then keep changes focused and well tested.

Contributors should run:

```sh
cargo fmt
cargo clippy --all-targets --all-features
cargo test --all-features
cargo doc --no-deps --all-features
```

See [AGENTS.md](AGENTS.md) for detailed instructions for AI coding agents and
contributors using agentic tooling.

## License

No license has been selected yet. Add a license before publishing releases or
accepting broad external contributions.
