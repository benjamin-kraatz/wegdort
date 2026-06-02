use crate::core::Vector;
use std::collections::HashMap;

pub type VectorId = u64;

pub trait Storage {
    fn insert(&mut self, id: VectorId, vector: Vector);
    fn get(&self, id: VectorId) -> Option<&Vector>;
    fn remove(&mut self, id: VectorId) -> Option<Vector>;
    fn list_ids(&self) -> Vec<VectorId>;
}

pub struct InMemoryStorage {
    vectors: HashMap<VectorId, Vector>,
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),
        }
    }
}

impl Storage for InMemoryStorage {
    fn insert(&mut self, id: VectorId, vector: Vector) {
        self.vectors.insert(id, vector);
    }

    fn get(&self, id: VectorId) -> Option<&Vector> {
        self.vectors.get(&id)
    }

    fn remove(&mut self, id: VectorId) -> Option<Vector> {
        self.vectors.remove(&id)
    }

    fn list_ids(&self) -> Vec<VectorId> {
        self.vectors.keys().cloned().collect()
    }
}
