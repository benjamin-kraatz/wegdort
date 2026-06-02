# AGENTS.md

This file defines the working rules for AI coding agents and contributors using
agentic tooling in this repository.

## Project Intent

Wegdort is being rewritten from scratch as a small, fast, cross-platform vector
store. The core product is a Rust library for in-memory vector storage and exact
top-k similarity search, with optional persistence and future Swift and
TypeScript APIs.

The first production search engine is exact flat search. The required metrics
are cosine similarity, dot product, and squared L2 distance.

## Before Changing Code

- Read the current README and this file.
- Inspect the relevant modules before proposing or making changes.
- Preserve user changes. Do not revert or overwrite unrelated local work.
- Do not run destructive git commands such as `git reset --hard` or
  `git checkout --` unless the user explicitly asks for that exact action.
- For non-trivial changes, state the intended approach before editing.

## Documentation Requirements

- Public Rust APIs must have useful Rustdoc comments.
- Public examples should be compile-checked as doctests where practical.
- New public behavior must be documented in Markdown when it affects project
  architecture, storage semantics, persistence, bindings, or user-facing APIs.
- Keep documentation accurate. Clearly distinguish implemented behavior from
  planned behavior.
- Prefer concise comments that explain why code exists or why a tradeoff was
  chosen. Do not add comments that merely restate obvious code.

## Project Structure Rules

Keep files focused and modules logically grouped. The intended structure is:

- `metrics`: vector scoring functions and score ordering semantics.
- `store`: ergonomic user-facing store API.
- `storage`: in-memory layout and id-to-vector storage.
- `search`: exact top-k search algorithms.
- `persistence`: optional file-backed snapshots.
- `error`: typed error definitions.
- `bindings` or `ffi`: future Swift and TypeScript API boundaries.

The crate should be library-first. A CLI or demo binary may exist, but it must
not own core behavior.

## Rust Implementation Rules

- Use Rust 2024 idioms and strong types.
- Keep the core dependency-light. Any new dependency must be justified by clear
  value in performance, safety, portability, or maintainability.
- Prefer cache-friendly contiguous storage for vector data where practical.
- Avoid unnecessary allocations in metric and search hot paths.
- Keep optional persistence separate from the in-memory search path.
- Return typed errors for invalid input such as dimension mismatches. Do not
  panic for normal user-facing error cases.
- Make score ordering explicit:
  - cosine similarity: higher is better;
  - dot product: higher is better;
  - squared L2 distance: lower is better.

## Required Tests

Add or update tests for any behavior you change. Future implementation work
should cover at least:

- metric correctness for cosine, dot product, and squared L2;
- score ordering direction for all metrics;
- insert, update or duplicate-id behavior, get, remove, and count;
- dimension validation for inserts and queries;
- top-k search correctness, including empty stores, `k = 0`, ties, and `k`
  greater than the number of stored vectors;
- persistence round trips once a persistence format exists;
- public API examples and doctests;
- benchmark coverage for metric hot paths and flat search once benchmarks are
  introduced.

## Verification

Before finalizing implementation work, run the relevant subset of:

```sh
cargo fmt
cargo clippy --all-targets --all-features
cargo test --all-features
cargo doc --no-deps --all-features
```

For documentation-only changes, proofread the changed Markdown and verify that
links and examples are internally consistent.

## API And Bindings Guidance

Rust API stability comes first. Swift and TypeScript APIs are planned for later
and should be built from a stable core rather than forcing early design
complexity into the storage layer.

Bindings must keep ownership, memory safety, and error reporting explicit. Do
not expose internal storage details unless there is a deliberate, documented
reason.
