use criterion::{Criterion, criterion_group, criterion_main};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hint::black_box;
use std::io::Cursor;
use std::time::{SystemTime, UNIX_EPOCH};
use wegdort::{Metric, Store, VectorId};

const DIMENSIONS: usize = 128;

fn build_store(count: usize, dimensions: usize, metric: Metric) -> Store {
    let mut store = Store::with_capacity(dimensions, metric, count).unwrap();
    for id in 0..count {
        store
            .insert(VectorId::new(id as u64), vector_for(id, dimensions))
            .unwrap();
    }
    store
}

fn vector_for(seed: usize, dimensions: usize) -> Vec<f32> {
    (0..dimensions)
        .map(|index| ((seed + index + 1) % 97) as f32 / 97.0 + 0.001)
        .collect()
}

fn bench_search(c: &mut Criterion) {
    let query = vector_for(42, DIMENSIONS);

    for metric in [Metric::Cosine, Metric::Dot, Metric::SquaredL2] {
        let store = build_store(10_000, DIMENSIONS, metric);
        c.bench_function(&format!("search/{metric:?}/10k/{DIMENSIONS}d/k10"), |b| {
            b.iter(|| black_box(store.search(black_box(&query), 10).unwrap()));
        });
    }

    let cosine_store = build_store(50_000, DIMENSIONS, Metric::Cosine);
    c.bench_function("search/cosine_cached_norms/50k/128d/k10", |b| {
        b.iter(|| black_box(cosine_store.search(black_box(&query), 10).unwrap()));
    });

    for count in [100, 1_000, 10_000] {
        let store = build_store(count, DIMENSIONS, Metric::Dot);
        c.bench_function(&format!("search/dot/{count}/{DIMENSIONS}d/k10"), |b| {
            b.iter(|| black_box(store.search(black_box(&query), 10).unwrap()));
        });
    }
}

#[cfg(feature = "parallel")]
fn bench_parallel_search(c: &mut Criterion) {
    let query = vector_for(42, DIMENSIONS);
    let store = build_store(50_000, DIMENSIONS, Metric::Dot);

    c.bench_function("search_parallel/dot/50k/128d/k10", |b| {
        b.iter(|| black_box(store.search_parallel(black_box(&query), 10).unwrap()));
    });
}

#[cfg(not(feature = "parallel"))]
fn bench_parallel_search(_c: &mut Criterion) {}

fn bench_top_k_selection(c: &mut Criterion) {
    let candidates = (0..10_000)
        .map(|id| BenchRankedHit {
            id,
            score: ((id * 31) % 997) as f32 / 997.0,
        })
        .collect::<Vec<_>>();

    c.bench_function("top_k/binary_heap/10k/k10", |b| {
        b.iter(|| black_box(heap_top_k(black_box(&candidates), 10)));
    });

    c.bench_function("top_k/bounded_buffer/10k/k10", |b| {
        b.iter(|| black_box(buffer_top_k(black_box(&candidates), 10)));
    });
}

fn bench_writes(c: &mut Criterion) {
    c.bench_function("insert/10k/128d", |b| {
        b.iter(|| {
            let mut store = Store::with_capacity(DIMENSIONS, Metric::Dot, 10_000).unwrap();
            for id in 0..10_000 {
                store
                    .insert(
                        VectorId::new(id),
                        black_box(vector_for(id as usize, DIMENSIONS)),
                    )
                    .unwrap();
            }
            black_box(store.len());
        });
    });

    c.bench_function("upsert_replace/10k/128d", |b| {
        let mut store = build_store(10_000, DIMENSIONS, Metric::Dot);
        b.iter(|| {
            for id in 0..10_000 {
                store
                    .upsert(
                        VectorId::new(id),
                        black_box(vector_for(id as usize + 1, DIMENSIONS)),
                    )
                    .unwrap();
            }
            black_box(store.len());
        });
    });

    c.bench_function("remove/10k/128d", |b| {
        b.iter(|| {
            let mut store = build_store(10_000, DIMENSIONS, Metric::Dot);
            for id in 0..10_000 {
                let _ = store.remove(VectorId::new(id));
            }
            black_box(store.len());
        });
    });
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct BenchRankedHit {
    id: u64,
    score: f32,
}

impl BenchRankedHit {
    fn is_better_than(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Less
    }
}

impl Eq for BenchRankedHit {}

impl PartialOrd for BenchRankedHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BenchRankedHit {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.id.cmp(&other.id))
    }
}

fn heap_top_k(candidates: &[BenchRankedHit], k: usize) -> Vec<BenchRankedHit> {
    let mut heap = BinaryHeap::with_capacity(k);
    for hit in candidates.iter().copied() {
        if heap.len() < k {
            heap.push(hit);
        } else if let Some(worst) = heap.peek()
            && hit.is_better_than(worst)
        {
            let _ = heap.pop();
            heap.push(hit);
        }
    }

    let mut hits = heap.into_iter().collect::<Vec<_>>();
    hits.sort();
    hits
}

fn buffer_top_k(candidates: &[BenchRankedHit], k: usize) -> Vec<BenchRankedHit> {
    let mut hits = Vec::with_capacity(k);
    let mut worst_index = None;
    for hit in candidates.iter().copied() {
        if hits.len() < k {
            hits.push(hit);
            if hits.len() == k {
                worst_index = refresh_bench_worst_index(&hits);
            }
            continue;
        }

        let index = worst_index.expect("full top-k buffer has a worst hit");
        if hit.is_better_than(&hits[index]) {
            hits[index] = hit;
            worst_index = refresh_bench_worst_index(&hits);
        }
    }

    hits.sort();
    hits
}

fn refresh_bench_worst_index(hits: &[BenchRankedHit]) -> Option<usize> {
    hits.iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.cmp(right))
        .map(|(index, _)| index)
}

fn bench_persistence(c: &mut Criterion) {
    let store = build_store(5_000, DIMENSIONS, Metric::Dot);

    c.bench_function("snapshot_save_load/5k/128d", |b| {
        b.iter(|| {
            let path = temp_path();
            store.save(&path).unwrap();
            let loaded = Store::load(&path).unwrap();
            let _ = std::fs::remove_file(&path);
            black_box(loaded.len());
        });
    });

    c.bench_function("snapshot_to_bytes_from_bytes/5k/128d", |b| {
        b.iter(|| {
            let bytes = store.to_bytes().unwrap();
            let loaded = Store::from_bytes(black_box(&bytes)).unwrap();
            black_box(loaded.len());
        });
    });

    c.bench_function("snapshot_writer_reader/5k/128d", |b| {
        b.iter(|| {
            let mut bytes = Vec::new();
            store.save_writer(&mut bytes).unwrap();
            let loaded = Store::load_reader(&mut Cursor::new(black_box(bytes))).unwrap();
            black_box(loaded.len());
        });
    });
}

fn temp_path() -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("wegdort-bench-{nanos}.wgd"))
}

criterion_group!(
    benches,
    bench_search,
    bench_parallel_search,
    bench_top_k_selection,
    bench_writes,
    bench_persistence
);
criterion_main!(benches);
