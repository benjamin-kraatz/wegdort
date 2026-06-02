# Performance Improvements Proposal

This proposal captures the current measured performance state and the next
practical optimization slices for Wegdort exact flat search. The goal is to keep
the Rust API stable while improving search throughput in ways that are easy to
benchmark and reason about.

## Reference

- Date: June 3, 2026
- Baseline commit:
  `e7ed07d3470c467238814692d864e97e717d393d`

## Current State

Wegdort currently uses exact flat search with bounded top-k selection. Vectors
are stored in contiguous row-major `f32` storage, ids are stored separately, and
cosine stores cache one Euclidean norm per vector row. Cosine search computes
the query norm once, then reuses cached row norms for each candidate. Dot product
and squared L2 do not use the norm cache.

Metric hot loops now use explicit indexed loops for dot product, vector norm,
and squared L2. This keeps the hot path predictable for LLVM while preserving
the same public behavior and raw-vector storage semantics.

The production top-k implementation remains a `BinaryHeap`. A bounded buffer
alternative was benchmarked, but it did not beat the heap for the current
10,000-candidate, `k = 10` workload.

## Recent Measurements

Final local Criterion runs on this machine showed:

| Benchmark | Approximate result | Interpretation |
| --- | ---: | --- |
| `search/Cosine/10k/128d/k10` | 297 us | No material change after indexed loops |
| `search/Dot/10k/128d/k10` | 300 us | No material change after indexed loops |
| `search/SquaredL2/10k/128d/k10` | 324 us | No material change after indexed loops |
| `search/cosine_cached_norms/50k/128d/k10` | 1.48 ms | Stable cached-norm path |
| `search_parallel/dot/50k/128d/k10` | 402 us | No material change |
| `top_k/binary_heap/10k/k10` | 8.67 us | Faster than bounded buffer |
| `top_k/bounded_buffer/10k/k10` | 10.09 us | Useful comparison, not production-worthy |

The indexed metric-loop change is harmless and keeps the implementation
explicit, but it is not a major throughput win. The top-k buffer experiment is
valuable as benchmark coverage, but the binary heap should remain the production
path until another candidate clearly beats it.

## Proposed Next Slices

### 1. SIMD Metric Kernels

Add architecture-aware SIMD implementations for dot product, squared L2, vector
norm, and cosine dot products.

Recommended approach:

- Keep scalar indexed loops as the portable fallback.
- Add target-feature-gated fast paths for common desktop/server targets.
- Start with `x86_64` AVX2/FMA where available, then consider ARM NEON for
  Apple Silicon and mobile targets.
- Dispatch inside internal metric helpers; do not change the public API.
- Benchmark dimensions 128, 384, 768, and 1536 because embedding workloads often
  use those widths.

Acceptance criteria:

- Search results match scalar results within a documented `f32` tolerance.
- Non-finite and zero-vector validation behavior remains unchanged.
- Benchmarks show a clear win on at least one common target without regressing
  scalar fallback behavior.

### 2. Cosine Normalized Sidecar Storage

Evaluate storing a normalized vector sidecar for cosine stores while preserving
raw vectors for `get`, `iter`, `remove`, snapshots, and persistence.

Recommended approach:

- Keep raw vectors as the source of truth.
- Add an internal normalized-vector sidecar only for `Metric::Cosine`.
- Score cosine queries as a dot product against normalized rows after
  normalizing the query once.
- Keep the v1 snapshot format unchanged; rebuild normalized sidecar data when
  loading.

Tradeoff:

- Search gets cheaper for cosine.
- Cosine stores use roughly one extra vector payload of memory.

Acceptance criteria:

- Raw vector APIs and snapshot bytes remain unchanged.
- Cosine search ordering and scores remain equivalent within tolerance.
- Benchmarks show enough improvement to justify the memory cost.

### 3. Top-k Candidate Alternatives

Keep benchmarking top-k alternatives, but do not replace `BinaryHeap` until an
alternative wins in full search and isolated selection benchmarks.

Promising candidates:

- Fixed-size array specialized for very small `k` values such as 1, 5, and 10.
- Selection plus partial sort when `k` is large relative to store size.
- Separate `k = 1` fast path with no allocation and no heap.

Acceptance criteria:

- Full `Store::search` benchmarks improve, not only isolated top-k selection.
- Small stores do not regress.
- Tie ordering by id remains deterministic.

### 4. Parallel Search Heuristics

Parallel search is useful once scoring dominates Rayon scheduling overhead. Add
benchmarks and documentation that make the serial/parallel crossover point
clear.

Recommended approach:

- Benchmark serial and parallel search across store sizes from 1,000 to 1M rows.
- Include cosine, dot product, and squared L2.
- Document observed crossover ranges rather than adding automatic dispatch in
  v1.

Acceptance criteria:

- `docs/performance.md` gives concrete guidance for when callers should use
  `search_parallel`.
- Parallel benchmarks remain feature-gated behind `parallel`.

## Verification Plan

For each performance slice, run:

```sh
cargo fmt --check
cargo clippy --all-targets --all-features
cargo test --all-features
cargo doc --no-deps --all-features
cargo bench
cargo bench --features parallel
```

When a slice is target-specific, also record CPU model, target triple, and
enabled target features alongside benchmark output.

## Recommendation

The next serious performance investment should be SIMD metric kernels. The
current scalar implementation is already efficient enough that small Rust-level
loop and top-k tweaks are mostly neutral. SIMD has the best chance of moving
exact flat search throughput materially while keeping the public API and storage
semantics stable.
