use crate::error::{Error, Result};
use crate::metrics::{Metric, validate_vector};
use crate::storage::{VectorId, VectorIter, VectorStorage};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// One result from a top-k vector search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchHit {
    /// Id of the matching vector.
    pub id: VectorId,
    /// Metric score for this match.
    ///
    /// Higher is better for cosine and dot product. Lower is better for squared
    /// L2 distance.
    pub score: f32,
}

/// Immutable, owned view of a store that can be searched independently.
#[derive(Debug, Clone)]
pub struct SearchSnapshot {
    dimensions: usize,
    metric: Metric,
    ids: Vec<VectorId>,
    vectors: Vec<f32>,
}

impl SearchSnapshot {
    pub(crate) fn new(storage: &VectorStorage, metric: Metric) -> Self {
        Self {
            dimensions: storage.dimensions(),
            metric,
            ids: storage.ids().to_vec(),
            vectors: storage.vectors().to_vec(),
        }
    }

    /// Searches the immutable snapshot and returns up to `k` best matches.
    pub fn search(&self, query: impl AsRef<[f32]>, k: usize) -> Result<Vec<SearchHit>> {
        search_flat(
            self.dimensions,
            self.metric,
            &self.ids,
            &self.vectors,
            query.as_ref(),
            k,
        )
    }

    /// Searches this snapshot with Rayon parallelism.
    ///
    /// This method is available only with the `parallel` feature.
    #[cfg(feature = "parallel")]
    pub fn search_parallel(&self, query: impl AsRef<[f32]>, k: usize) -> Result<Vec<SearchHit>> {
        search_flat_parallel(
            self.dimensions,
            self.metric,
            &self.ids,
            &self.vectors,
            query.as_ref(),
            k,
        )
    }

    /// Iterates over ids and vector slices in this snapshot.
    pub fn iter(&self) -> VectorIter<'_> {
        VectorIter::new(&self.ids, &self.vectors, self.dimensions)
    }

    /// Returns the fixed vector dimension of this snapshot.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Returns the metric used by this snapshot.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Returns the number of vectors in this snapshot.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Returns true when this snapshot contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

pub(crate) fn search_storage(
    storage: &VectorStorage,
    metric: Metric,
    query: &[f32],
    k: usize,
) -> Result<Vec<SearchHit>> {
    search_flat(
        storage.dimensions(),
        metric,
        storage.ids(),
        storage.vectors(),
        query,
        k,
    )
}

#[cfg(feature = "parallel")]
pub(crate) fn search_storage_parallel(
    storage: &VectorStorage,
    metric: Metric,
    query: &[f32],
    k: usize,
) -> Result<Vec<SearchHit>> {
    search_flat_parallel(
        storage.dimensions(),
        metric,
        storage.ids(),
        storage.vectors(),
        query,
        k,
    )
}

fn search_flat(
    dimensions: usize,
    metric: Metric,
    ids: &[VectorId],
    vectors: &[f32],
    query: &[f32],
    k: usize,
) -> Result<Vec<SearchHit>> {
    if query.len() != dimensions {
        return Err(Error::DimensionMismatch {
            expected: dimensions,
            actual: query.len(),
        });
    }

    validate_vector(metric, query)?;

    if k == 0 || ids.is_empty() {
        return Ok(Vec::new());
    }

    let mut heap = BinaryHeap::with_capacity(k.min(ids.len()));
    for (row, id) in ids.iter().copied().enumerate() {
        let start = row * dimensions;
        let vector = &vectors[start..start + dimensions];
        push_top_k(
            &mut heap,
            k,
            RankedHit {
                id,
                score: metric.score(query, vector),
                metric,
            },
        );
    }

    Ok(sorted_hits(heap, metric))
}

#[cfg(feature = "parallel")]
fn search_flat_parallel(
    dimensions: usize,
    metric: Metric,
    ids: &[VectorId],
    vectors: &[f32],
    query: &[f32],
    k: usize,
) -> Result<Vec<SearchHit>> {
    if query.len() != dimensions {
        return Err(Error::DimensionMismatch {
            expected: dimensions,
            actual: query.len(),
        });
    }

    validate_vector(metric, query)?;

    if k == 0 || ids.is_empty() {
        return Ok(Vec::new());
    }

    let hits = ids
        .par_iter()
        .copied()
        .enumerate()
        .fold(
            || BinaryHeap::with_capacity(k),
            |mut heap, (row, id)| {
                let start = row * dimensions;
                let vector = &vectors[start..start + dimensions];
                push_top_k(
                    &mut heap,
                    k,
                    RankedHit {
                        id,
                        score: metric.score(query, vector),
                        metric,
                    },
                );
                heap
            },
        )
        .reduce(
            || BinaryHeap::with_capacity(k),
            |mut left, right| {
                for hit in right {
                    push_top_k(&mut left, k, hit);
                }
                left
            },
        );

    Ok(sorted_hits(hits, metric))
}

fn push_top_k(heap: &mut BinaryHeap<RankedHit>, k: usize, hit: RankedHit) {
    if heap.len() < k {
        heap.push(hit);
    } else if let Some(worst) = heap.peek()
        && hit.is_better_than(worst)
    {
        let _ = heap.pop();
        heap.push(hit);
    }
}

fn sorted_hits(heap: BinaryHeap<RankedHit>, metric: Metric) -> Vec<SearchHit> {
    let mut hits: Vec<_> = heap
        .into_iter()
        .map(|hit| SearchHit {
            id: hit.id,
            score: hit.score,
        })
        .collect();
    hits.sort_by(|left, right| {
        metric
            .compare_scores(left.score, right.score)
            .then_with(|| left.id.cmp(&right.id))
    });
    hits
}

#[derive(Debug, Clone, Copy)]
struct RankedHit {
    id: VectorId,
    score: f32,
    metric: Metric,
}

impl RankedHit {
    fn is_better_than(&self, other: &Self) -> bool {
        self.metric
            .compare_scores(self.score, other.score)
            .then_with(|| self.id.cmp(&other.id))
            == Ordering::Less
    }
}

impl PartialEq for RankedHit {
    fn eq(&self, other: &Self) -> bool {
        self.metric == other.metric
            && self.id == other.id
            && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for RankedHit {}

impl PartialOrd for RankedHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankedHit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.metric
            .compare_scores(self.score, other.score)
            .then_with(|| self.id.cmp(&other.id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_sort(
        storage: &VectorStorage,
        metric: Metric,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchHit>> {
        if query.len() != storage.dimensions() {
            return Err(Error::DimensionMismatch {
                expected: storage.dimensions(),
                actual: query.len(),
            });
        }
        validate_vector(metric, query)?;

        let mut hits = Vec::new();
        for (row, id) in storage.ids().iter().copied().enumerate() {
            let start = row * storage.dimensions();
            let vector = &storage.vectors()[start..start + storage.dimensions()];
            hits.push(SearchHit {
                id,
                score: metric.score(query, vector),
            });
        }
        hits.sort_by(|left, right| {
            metric
                .compare_scores(left.score, right.score)
                .then_with(|| left.id.cmp(&right.id))
        });
        hits.truncate(k.min(hits.len()));
        Ok(hits)
    }

    #[test]
    fn returns_empty_for_zero_k() {
        let mut storage = VectorStorage::with_capacity(2, 0).unwrap();
        storage.insert(VectorId::new(1), &[1.0, 0.0]).unwrap();
        let hits = search_storage(&storage, Metric::Dot, &[1.0, 0.0], 0).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn orders_similarity_highest_first_with_id_tie_break() {
        let mut storage = VectorStorage::with_capacity(2, 0).unwrap();
        storage.insert(VectorId::new(2), &[1.0, 0.0]).unwrap();
        storage.insert(VectorId::new(1), &[1.0, 0.0]).unwrap();
        storage.insert(VectorId::new(3), &[0.0, 1.0]).unwrap();

        let hits = search_storage(&storage, Metric::Dot, &[1.0, 0.0], 3).unwrap();
        assert_eq!(
            hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
            vec![VectorId::new(1), VectorId::new(2), VectorId::new(3),]
        );
    }

    #[test]
    fn orders_distance_lowest_first() {
        let mut storage = VectorStorage::with_capacity(2, 0).unwrap();
        storage.insert(VectorId::new(1), &[5.0, 5.0]).unwrap();
        storage.insert(VectorId::new(2), &[1.0, 1.0]).unwrap();

        let hits = search_storage(&storage, Metric::SquaredL2, &[0.0, 0.0], 2).unwrap();
        assert_eq!(hits[0].id, VectorId::new(2));
    }

    #[test]
    fn rejects_wrong_query_dimensions() {
        let storage = VectorStorage::with_capacity(2, 0).unwrap();
        assert!(matches!(
            search_storage(&storage, Metric::Dot, &[1.0], 1),
            Err(Error::DimensionMismatch {
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn bounded_top_k_matches_full_sort_for_all_metrics() {
        for metric in [Metric::Cosine, Metric::Dot, Metric::SquaredL2] {
            let mut storage = VectorStorage::with_capacity(3, 0).unwrap();
            for id in 1..=12 {
                storage
                    .insert(
                        VectorId::new(id),
                        &[id as f32, (id % 3 + 1) as f32, (13 - id) as f32],
                    )
                    .unwrap();
            }

            let query = [1.0, 2.0, 3.0];
            assert_eq!(
                search_storage(&storage, metric, &query, 5).unwrap(),
                full_sort(&storage, metric, &query, 5).unwrap()
            );
            assert_eq!(
                search_storage(&storage, metric, &query, 99).unwrap(),
                full_sort(&storage, metric, &query, 99).unwrap()
            );
        }
    }

    #[test]
    fn snapshot_iter_returns_ids_and_vectors() {
        let mut storage = VectorStorage::with_capacity(2, 0).unwrap();
        storage.insert(VectorId::new(1), &[1.0, 0.0]).unwrap();
        let snapshot = SearchSnapshot::new(&storage, Metric::Dot);

        let rows = snapshot.iter().collect::<Vec<_>>();
        assert_eq!(rows, vec![(VectorId::new(1), [1.0, 0.0].as_slice())]);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn snapshot_parallel_search_matches_serial_search() {
        let mut storage = VectorStorage::with_capacity(2, 0).unwrap();
        storage.insert(VectorId::new(2), &[1.0, 0.0]).unwrap();
        storage.insert(VectorId::new(1), &[1.0, 0.0]).unwrap();
        storage.insert(VectorId::new(3), &[0.0, 1.0]).unwrap();
        let snapshot = SearchSnapshot::new(&storage, Metric::Dot);

        assert_eq!(
            snapshot.search_parallel([1.0, 0.0], 3).unwrap(),
            snapshot.search([1.0, 0.0], 3).unwrap()
        );
        assert!(snapshot.search_parallel([1.0, 0.0], 0).unwrap().is_empty());
    }
}
