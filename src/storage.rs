use crate::error::{Error, Result};
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

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
    rows_by_id: HashMap<VectorId, usize>,
}

impl VectorStorage {
    pub(crate) fn new(dimensions: usize) -> Result<Self> {
        if dimensions == 0 {
            return Err(Error::ZeroDimensions);
        }

        Ok(Self {
            dimensions,
            ids: Vec::new(),
            vectors: Vec::new(),
            rows_by_id: HashMap::new(),
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

        Ok(Self {
            dimensions,
            ids,
            vectors,
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
        self.rows_by_id.insert(id, row);
        Ok(())
    }

    pub(crate) fn upsert(&mut self, id: VectorId, vector: &[f32]) -> bool {
        if let Some(row) = self.rows_by_id.get(&id).copied() {
            let range = self.row_range(row);
            self.vectors[range].copy_from_slice(vector);
            true
        } else {
            let row = self.ids.len();
            self.ids.push(id);
            self.vectors.extend_from_slice(vector);
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
            let target_range = self.row_range(row);
            self.vectors[target_range].copy_from_slice(&last_vector);
            self.ids[row] = last_id;
            self.rows_by_id.insert(last_id, row);
        }

        self.ids.pop();
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

    pub(crate) fn vector(&self, row: usize) -> &[f32] {
        &self.vectors[self.row_range(row)]
    }

    fn row_range(&self, row: usize) -> std::ops::Range<usize> {
        let start = row * self.dimensions;
        start..start + self.dimensions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_dimensions() {
        assert!(matches!(VectorStorage::new(0), Err(Error::ZeroDimensions)));
    }

    #[test]
    fn insert_rejects_duplicates() {
        let mut storage = VectorStorage::new(2).unwrap();
        storage.insert(VectorId::new(7), &[1.0, 2.0]).unwrap();
        assert!(matches!(
            storage.insert(VectorId::new(7), &[3.0, 4.0]),
            Err(Error::DuplicateId(id)) if id == VectorId::new(7)
        ));
    }

    #[test]
    fn upsert_reports_replacement() {
        let mut storage = VectorStorage::new(2).unwrap();
        assert!(!storage.upsert(VectorId::new(7), &[1.0, 2.0]));
        assert!(storage.upsert(VectorId::new(7), &[3.0, 4.0]));
        assert_eq!(storage.get(VectorId::new(7)), Some([3.0, 4.0].as_slice()));
    }

    #[test]
    fn remove_updates_swapped_row_index() {
        let mut storage = VectorStorage::new(2).unwrap();
        storage.insert(VectorId::new(1), &[1.0, 1.0]).unwrap();
        storage.insert(VectorId::new(2), &[2.0, 2.0]).unwrap();
        storage.insert(VectorId::new(3), &[3.0, 3.0]).unwrap();

        assert_eq!(storage.remove(VectorId::new(1)), Some(vec![1.0, 1.0]));
        assert_eq!(storage.get(VectorId::new(3)), Some([3.0, 3.0].as_slice()));
        assert_eq!(storage.len(), 2);
        assert!(!storage.contains(VectorId::new(1)));
    }
}
