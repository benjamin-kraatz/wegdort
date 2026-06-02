use crate::error::{Error, Result};
use crate::metrics::{Metric, validate_vector};
use crate::storage::{VectorId, VectorStorage};

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

    let mut hits = Vec::with_capacity(ids.len());
    for (row, id) in ids.iter().copied().enumerate() {
        let start = row * dimensions;
        let vector = &vectors[start..start + dimensions];
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_empty_for_zero_k() {
        let mut storage = VectorStorage::new(2).unwrap();
        storage.insert(VectorId::new(1), &[1.0, 0.0]).unwrap();
        let hits = search_storage(&storage, Metric::Dot, &[1.0, 0.0], 0).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn orders_similarity_highest_first_with_id_tie_break() {
        let mut storage = VectorStorage::new(2).unwrap();
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
        let mut storage = VectorStorage::new(2).unwrap();
        storage.insert(VectorId::new(1), &[5.0, 5.0]).unwrap();
        storage.insert(VectorId::new(2), &[1.0, 1.0]).unwrap();

        let hits = search_storage(&storage, Metric::SquaredL2, &[0.0, 0.0], 2).unwrap();
        assert_eq!(hits[0].id, VectorId::new(2));
    }

    #[test]
    fn rejects_wrong_query_dimensions() {
        let storage = VectorStorage::new(2).unwrap();
        assert!(matches!(
            search_storage(&storage, Metric::Dot, &[1.0], 1),
            Err(Error::DimensionMismatch {
                expected: 2,
                actual: 1
            })
        ));
    }
}
