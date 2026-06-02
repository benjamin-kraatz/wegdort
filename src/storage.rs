use crate::error::{Error, Result};
use crate::metrics::vector_norm;
use std::collections::HashMap;
use std::fmt;

/// Caller-supplied identifier for a stored vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VectorId(u64);

impl VectorId {
    /// Creates a new vector id from a raw `u64`.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw id value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl From<u64> for VectorId {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

impl fmt::Display for VectorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct VectorStorage {
    dimensions: usize,
    ids: Vec<VectorId>,
    vectors: Vec<f32>,
    norms: Vec<f32>,
    rows_by_id: HashMap<VectorId, usize>,
}

impl VectorStorage {
    pub(crate) fn with_capacity(dimensions: usize, capacity: usize) -> Result<Self> {
        if dimensions == 0 {
            return Err(Error::ZeroDimensions);
        }

        Ok(Self {
            dimensions,
            ids: Vec::with_capacity(capacity),
            vectors: Vec::with_capacity(capacity.saturating_mul(dimensions)),
            norms: Vec::with_capacity(capacity),
            rows_by_id: HashMap::with_capacity(capacity),
        })
    }

    pub(crate) fn from_parts(
        dimensions: usize,
        ids: Vec<VectorId>,
        vectors: Vec<f32>,
    ) -> Result<Self> {
        if dimensions == 0 {
            return Err(Error::ZeroDimensions);
        }

        if ids.len().checked_mul(dimensions) != Some(vectors.len()) {
            return Err(Error::CorruptedFile(
                "vector payload length does not match header",
            ));
        }

        let mut rows_by_id = HashMap::with_capacity(ids.len());
        for (row, id) in ids.iter().copied().enumerate() {
            if rows_by_id.insert(id, row).is_some() {
                return Err(Error::CorruptedFile("duplicate vector id"));
            }
        }
        let norms = vectors
            .chunks_exact(dimensions)
            .map(vector_norm)
            .collect::<Vec<_>>();

        Ok(Self {
            dimensions,
            ids,
            vectors,
            norms,
            rows_by_id,
        })
    }

    pub(crate) fn insert(&mut self, id: VectorId, vector: &[f32]) -> Result<()> {
        if self.rows_by_id.contains_key(&id) {
            return Err(Error::DuplicateId(id));
        }

        let row = self.ids.len();
        self.ids.push(id);
        self.vectors.extend_from_slice(vector);
        self.norms.push(vector_norm(vector));
        self.rows_by_id.insert(id, row);
        Ok(())
    }

    pub(crate) fn upsert(&mut self, id: VectorId, vector: &[f32]) -> bool {
        if let Some(row) = self.rows_by_id.get(&id).copied() {
            let range = self.row_range(row);
            self.vectors[range].copy_from_slice(vector);
            self.norms[row] = vector_norm(vector);
            true
        } else {
            let row = self.ids.len();
            self.ids.push(id);
            self.vectors.extend_from_slice(vector);
            self.norms.push(vector_norm(vector));
            self.rows_by_id.insert(id, row);
            false
        }
    }

    pub(crate) fn remove(&mut self, id: VectorId) -> Option<Vec<f32>> {
        let row = self.rows_by_id.remove(&id)?;
        let removed = self.vector(row).to_vec();
        let last_row = self.ids.len() - 1;

        if row != last_row {
            let last_id = self.ids[last_row];
            let last_vector = self.vector(last_row).to_vec();
            let last_norm = self.norms[last_row];
            let target_range = self.row_range(row);
            self.vectors[target_range].copy_from_slice(&last_vector);
            self.norms[row] = last_norm;
            self.ids[row] = last_id;
            self.rows_by_id.insert(last_id, row);
        }

        self.ids.pop();
        self.norms.pop();
        self.vectors.truncate(self.ids.len() * self.dimensions);
        Some(removed)
    }

    pub(crate) fn get(&self, id: VectorId) -> Option<&[f32]> {
        self.rows_by_id
            .get(&id)
            .copied()
            .map(|row| self.vector(row))
    }

    pub(crate) fn contains(&self, id: VectorId) -> bool {
        self.rows_by_id.contains_key(&id)
    }

    pub(crate) fn reserve(&mut self, additional: usize) {
        self.ids.reserve(additional);
        self.vectors
            .reserve(additional.saturating_mul(self.dimensions));
        self.norms.reserve(additional);
        self.rows_by_id.reserve(additional);
    }

    pub(crate) fn capacity(&self) -> usize {
        self.ids.capacity()
    }

    pub(crate) fn clear(&mut self) {
        self.ids.clear();
        self.vectors.clear();
        self.norms.clear();
        self.rows_by_id.clear();
    }

    pub(crate) fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub(crate) fn ids(&self) -> &[VectorId] {
        &self.ids
    }

    pub(crate) fn vectors(&self) -> &[f32] {
        &self.vectors
    }

    pub(crate) fn norms(&self) -> &[f32] {
        &self.norms
    }

    pub(crate) fn iter(&self) -> VectorIter<'_> {
        VectorIter::new(&self.ids, &self.vectors, self.dimensions)
    }

    pub(crate) fn vector(&self, row: usize) -> &[f32] {
        &self.vectors[self.row_range(row)]
    }

    fn row_range(&self, row: usize) -> std::ops::Range<usize> {
        let start = row * self.dimensions;
        start..start + self.dimensions
    }
}

/// Iterator over vector ids and vector slices.
#[derive(Debug, Clone)]
pub struct VectorIter<'a> {
    ids: &'a [VectorId],
    vectors: &'a [f32],
    dimensions: usize,
    row: usize,
}

impl<'a> VectorIter<'a> {
    pub(crate) fn new(ids: &'a [VectorId], vectors: &'a [f32], dimensions: usize) -> Self {
        Self {
            ids,
            vectors,
            dimensions,
            row: 0,
        }
    }
}

impl<'a> Iterator for VectorIter<'a> {
    type Item = (VectorId, &'a [f32]);

    fn next(&mut self) -> Option<Self::Item> {
        let id = *self.ids.get(self.row)?;
        let start = self.row * self.dimensions;
        let vector = &self.vectors[start..start + self.dimensions];
        self.row += 1;
        Some((id, vector))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.ids.len().saturating_sub(self.row);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for VectorIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_dimensions() {
        assert!(matches!(
            VectorStorage::with_capacity(0, 0),
            Err(Error::ZeroDimensions)
        ));
    }

    #[test]
    fn insert_rejects_duplicates() {
        let mut storage = VectorStorage::with_capacity(2, 0).unwrap();
        storage.insert(VectorId::new(7), &[1.0, 2.0]).unwrap();
        assert!(matches!(
            storage.insert(VectorId::new(7), &[3.0, 4.0]),
            Err(Error::DuplicateId(id)) if id == VectorId::new(7)
        ));
    }

    #[test]
    fn upsert_reports_replacement() {
        let mut storage = VectorStorage::with_capacity(2, 0).unwrap();
        assert!(!storage.upsert(VectorId::new(7), &[1.0, 2.0]));
        assert!(storage.upsert(VectorId::new(7), &[3.0, 4.0]));
        assert_eq!(storage.get(VectorId::new(7)), Some([3.0, 4.0].as_slice()));
        assert_eq!(storage.norms(), &[5.0]);
    }

    #[test]
    fn remove_updates_swapped_row_index() {
        let mut storage = VectorStorage::with_capacity(2, 0).unwrap();
        storage.insert(VectorId::new(1), &[1.0, 1.0]).unwrap();
        storage.insert(VectorId::new(2), &[2.0, 2.0]).unwrap();
        storage.insert(VectorId::new(3), &[3.0, 3.0]).unwrap();

        assert_eq!(storage.remove(VectorId::new(1)), Some(vec![1.0, 1.0]));
        assert_eq!(storage.get(VectorId::new(3)), Some([3.0, 3.0].as_slice()));
        assert_eq!(storage.len(), 2);
        assert!(!storage.contains(VectorId::new(1)));
    }

    #[test]
    fn remove_keeps_norms_aligned_with_swapped_rows() {
        let mut storage = VectorStorage::with_capacity(2, 0).unwrap();
        storage.insert(VectorId::new(1), &[3.0, 4.0]).unwrap();
        storage.insert(VectorId::new(2), &[5.0, 12.0]).unwrap();
        storage.insert(VectorId::new(3), &[8.0, 15.0]).unwrap();

        assert_eq!(storage.remove(VectorId::new(1)), Some(vec![3.0, 4.0]));

        let row = storage
            .ids()
            .iter()
            .position(|id| *id == VectorId::new(3))
            .unwrap();
        assert_eq!(storage.norms()[row], 17.0);
        assert_eq!(storage.vector(row), [8.0, 15.0].as_slice());
    }

    #[test]
    fn from_parts_builds_norms() {
        let storage = VectorStorage::from_parts(
            2,
            vec![VectorId::new(1), VectorId::new(2)],
            vec![3.0, 4.0, 5.0, 12.0],
        )
        .unwrap();

        assert_eq!(storage.norms(), &[5.0, 13.0]);
    }
}
