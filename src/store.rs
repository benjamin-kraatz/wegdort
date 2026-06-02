use crate::error::{Error, Result};
use crate::metrics::{Metric, validate_vector};
use crate::persistence;
#[cfg(feature = "parallel")]
use crate::search::search_storage_parallel;
use crate::search::{SearchHit, SearchSnapshot, search_storage};
use crate::storage::{VectorId, VectorIter, VectorStorage};
use std::path::Path;

/// Result of an upsert operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpsertResult {
    /// The id did not exist and a new vector was inserted.
    Inserted,
    /// The id existed and its vector was replaced.
    Replaced,
}

/// Builder for configuring a [`Store`] before construction.
#[derive(Debug, Clone)]
pub struct StoreBuilder {
    dimensions: usize,
    metric: Metric,
    capacity: usize,
}

impl StoreBuilder {
    /// Creates a builder for a store with fixed `dimensions`.
    ///
    /// The default metric is [`Metric::Cosine`] and the default capacity is `0`.
    ///
    /// ```
    /// use wegdort::{Metric, StoreBuilder};
    ///
    /// let store = StoreBuilder::new(3)
    ///     .metric(Metric::Dot)
    ///     .capacity(128)
    ///     .build()?;
    ///
    /// assert_eq!(store.dimensions(), 3);
    /// assert_eq!(store.metric(), Metric::Dot);
    /// assert!(store.capacity() >= 128);
    /// # Ok::<(), wegdort::Error>(())
    /// ```
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            metric: Metric::Cosine,
            capacity: 0,
        }
    }

    /// Sets the metric for the store.
    pub fn metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    /// Sets the initial vector capacity for the store.
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// Builds the configured store.
    pub fn build(self) -> Result<Store> {
        Store::with_capacity(self.dimensions, self.metric, self.capacity)
    }
}

/// In-memory vector store with exact top-k search.
///
/// A store has one fixed dimension and one search metric. All vectors inserted
/// into the store must match that dimension.
#[derive(Debug, Clone)]
pub struct Store {
    metric: Metric,
    storage: VectorStorage,
}

impl Store {
    /// Creates an empty store with fixed `dimensions` and `metric`.
    ///
    /// ```
    /// use wegdort::{Metric, Store};
    ///
    /// let store = Store::new(3, Metric::Dot)?;
    /// assert_eq!(store.dimensions(), 3);
    /// # Ok::<(), wegdort::Error>(())
    /// ```
    pub fn new(dimensions: usize, metric: Metric) -> Result<Self> {
        Self::with_capacity(dimensions, metric, 0)
    }

    /// Creates an empty store with enough capacity for at least `capacity` vectors.
    ///
    /// ```
    /// use wegdort::{Metric, Store};
    ///
    /// let store = Store::with_capacity(3, Metric::Dot, 64)?;
    /// assert!(store.capacity() >= 64);
    /// # Ok::<(), wegdort::Error>(())
    /// ```
    pub fn with_capacity(dimensions: usize, metric: Metric, capacity: usize) -> Result<Self> {
        Ok(Self {
            metric,
            storage: VectorStorage::with_capacity(dimensions, capacity)?,
        })
    }

    pub(crate) fn from_parts(
        dimensions: usize,
        metric: Metric,
        ids: Vec<VectorId>,
        vectors: Vec<f32>,
    ) -> Result<Self> {
        Ok(Self {
            metric,
            storage: VectorStorage::from_parts(dimensions, ids, vectors)?,
        })
    }

    /// Inserts a new vector.
    ///
    /// Duplicate ids are rejected. Use [`Store::upsert`] to insert or replace.
    pub fn insert(&mut self, id: VectorId, vector: impl AsRef<[f32]>) -> Result<()> {
        let vector = vector.as_ref();
        self.validate_input(vector)?;
        self.storage.insert(id, vector)
    }

    /// Inserts a new vector or replaces an existing vector with the same id.
    pub fn upsert(&mut self, id: VectorId, vector: impl AsRef<[f32]>) -> Result<UpsertResult> {
        let vector = vector.as_ref();
        self.validate_input(vector)?;
        if self.storage.upsert(id, vector) {
            Ok(UpsertResult::Replaced)
        } else {
            Ok(UpsertResult::Inserted)
        }
    }

    /// Removes a vector and returns it if the id existed.
    pub fn remove(&mut self, id: VectorId) -> Option<Vec<f32>> {
        self.storage.remove(id)
    }

    /// Returns the vector for `id` without copying it.
    pub fn get(&self, id: VectorId) -> Option<&[f32]> {
        self.storage.get(id)
    }

    /// Returns true if `id` exists in the store.
    pub fn contains(&self, id: VectorId) -> bool {
        self.storage.contains(id)
    }

    /// Searches the store and returns up to `k` best matches.
    pub fn search(&self, query: impl AsRef<[f32]>, k: usize) -> Result<Vec<SearchHit>> {
        search_storage(&self.storage, self.metric, query.as_ref(), k)
    }

    /// Searches the store with Rayon parallelism.
    ///
    /// This method is available only with the `parallel` feature.
    #[cfg(feature = "parallel")]
    pub fn search_parallel(&self, query: impl AsRef<[f32]>, k: usize) -> Result<Vec<SearchHit>> {
        search_storage_parallel(&self.storage, self.metric, query.as_ref(), k)
    }

    /// Creates an owned immutable snapshot that can be searched independently.
    ///
    /// ```
    /// use wegdort::{Metric, Store, VectorId};
    ///
    /// let mut store = Store::new(2, Metric::Dot)?;
    /// store.insert(VectorId::new(1), [1.0, 0.0])?;
    ///
    /// let snapshot = store.snapshot();
    /// store.upsert(VectorId::new(1), [0.0, 1.0])?;
    ///
    /// let hits = snapshot.search([1.0, 0.0], 1)?;
    /// assert_eq!(hits[0].id, VectorId::new(1));
    /// assert_eq!(hits[0].score, 1.0);
    /// # Ok::<(), wegdort::Error>(())
    /// ```
    pub fn snapshot(&self) -> SearchSnapshot {
        SearchSnapshot::new(&self.storage, self.metric)
    }

    /// Saves the store to a stable custom binary snapshot.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        persistence::save(self, path.as_ref())
    }

    /// Loads a store from a stable custom binary snapshot.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        persistence::load(path.as_ref())
    }

    /// Reserves capacity for at least `additional` more vectors.
    ///
    /// ```
    /// use wegdort::{Metric, Store};
    ///
    /// let mut store = Store::new(3, Metric::Dot)?;
    /// store.reserve(32);
    /// assert!(store.capacity() >= 32);
    /// # Ok::<(), wegdort::Error>(())
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        self.storage.reserve(additional);
    }

    /// Returns the current vector capacity.
    pub fn capacity(&self) -> usize {
        self.storage.capacity()
    }

    /// Removes all vectors while retaining allocated capacity.
    pub fn clear(&mut self) {
        self.storage.clear();
    }

    /// Iterates over ids and vector slices.
    ///
    /// ```
    /// use wegdort::{Metric, Store, VectorId};
    ///
    /// let mut store = Store::new(2, Metric::Dot)?;
    /// store.insert(VectorId::new(7), [1.0, 2.0])?;
    ///
    /// let rows: Vec<_> = store.iter().collect();
    /// assert_eq!(rows[0], (VectorId::new(7), [1.0, 2.0].as_slice()));
    /// # Ok::<(), wegdort::Error>(())
    /// ```
    pub fn iter(&self) -> VectorIter<'_> {
        self.storage.iter()
    }

    /// Returns the fixed vector dimension.
    pub fn dimensions(&self) -> usize {
        self.storage.dimensions()
    }

    /// Returns the metric used by this store.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Returns the number of stored vectors.
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Returns true when the store contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    pub(crate) fn ids(&self) -> &[VectorId] {
        self.storage.ids()
    }

    pub(crate) fn vectors(&self) -> &[f32] {
        self.storage.vectors()
    }

    fn validate_input(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions() {
            return Err(Error::DimensionMismatch {
                expected: self.dimensions(),
                actual: vector.len(),
            });
        }

        validate_vector(self.metric, vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_contains_and_len_work() {
        let mut store = Store::new(2, Metric::Dot).unwrap();
        store.insert(VectorId::new(1), [1.0, 2.0]).unwrap();

        assert_eq!(store.get(VectorId::new(1)), Some([1.0, 2.0].as_slice()));
        assert!(store.contains(VectorId::new(1)));
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    #[test]
    fn insert_rejects_duplicate_id() {
        let mut store = Store::new(2, Metric::Dot).unwrap();
        store.insert(VectorId::new(1), [1.0, 2.0]).unwrap();
        assert!(matches!(
            store.insert(VectorId::new(1), [2.0, 3.0]),
            Err(Error::DuplicateId(id)) if id == VectorId::new(1)
        ));
    }

    #[test]
    fn upsert_inserts_and_replaces() {
        let mut store = Store::new(2, Metric::Dot).unwrap();
        assert_eq!(
            store.upsert(VectorId::new(1), [1.0, 2.0]).unwrap(),
            UpsertResult::Inserted
        );
        assert_eq!(
            store.upsert(VectorId::new(1), [3.0, 4.0]).unwrap(),
            UpsertResult::Replaced
        );
        assert_eq!(store.get(VectorId::new(1)), Some([3.0, 4.0].as_slice()));
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let mut store = Store::new(2, Metric::Dot).unwrap();
        assert!(matches!(
            store.insert(VectorId::new(1), [1.0]),
            Err(Error::DimensionMismatch {
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn searches_top_k() {
        let mut store = Store::new(2, Metric::SquaredL2).unwrap();
        store.insert(VectorId::new(1), [5.0, 5.0]).unwrap();
        store.insert(VectorId::new(2), [1.0, 1.0]).unwrap();

        let hits = store.search([0.0, 0.0], 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, VectorId::new(2));
    }

    #[test]
    fn snapshot_is_not_changed_by_store_mutation() {
        let mut store = Store::new(2, Metric::Dot).unwrap();
        store.insert(VectorId::new(1), [1.0, 0.0]).unwrap();
        let snapshot = store.snapshot();

        store.upsert(VectorId::new(1), [0.0, 1.0]).unwrap();

        let snapshot_hits = snapshot.search([1.0, 0.0], 1).unwrap();
        let store_hits = store.search([1.0, 0.0], 1).unwrap();
        assert_eq!(snapshot_hits[0].score, 1.0);
        assert_eq!(store_hits[0].score, 0.0);
    }

    #[test]
    fn with_capacity_reserve_capacity_and_clear_work() {
        let mut store = Store::with_capacity(3, Metric::Dot, 4).unwrap();
        assert!(store.capacity() >= 4);

        let before = store.capacity();
        store.reserve(16);
        assert!(store.capacity() >= before);

        store.insert(VectorId::new(1), [1.0, 2.0, 3.0]).unwrap();
        store.clear();
        assert!(store.is_empty());
        assert!(store.capacity() >= before);
    }

    #[test]
    fn builder_uses_defaults_and_configuration() {
        let default_store = StoreBuilder::new(2).build().unwrap();
        assert_eq!(default_store.metric(), Metric::Cosine);

        let configured = StoreBuilder::new(4)
            .metric(Metric::SquaredL2)
            .capacity(8)
            .build()
            .unwrap();
        assert_eq!(configured.dimensions(), 4);
        assert_eq!(configured.metric(), Metric::SquaredL2);
        assert!(configured.capacity() >= 8);
    }

    #[test]
    fn iter_returns_ids_and_vectors() {
        let mut store = Store::new(2, Metric::Dot).unwrap();
        store.insert(VectorId::new(10), [1.0, 2.0]).unwrap();

        let rows = store.iter().collect::<Vec<_>>();
        assert_eq!(rows, vec![(VectorId::new(10), [1.0, 2.0].as_slice())]);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_search_matches_serial_search() {
        let mut store = Store::new(2, Metric::Dot).unwrap();
        store.insert(VectorId::new(2), [1.0, 0.0]).unwrap();
        store.insert(VectorId::new(1), [1.0, 0.0]).unwrap();
        store.insert(VectorId::new(3), [0.0, 1.0]).unwrap();

        assert_eq!(
            store.search_parallel([1.0, 0.0], 2).unwrap(),
            store.search([1.0, 0.0], 2).unwrap()
        );
    }
}
