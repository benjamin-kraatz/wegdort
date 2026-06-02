mod core;
mod storage;
mod index;
mod persistence;

use crate::core::{Vector, L2Distance};
use crate::storage::{InMemoryStorage, Storage};
use crate::index::{FlatIndex, Index};
use crate::persistence::{FilePersistence, Persistence};

fn main() {
    println!("Wegdort: In-memory Vector Database - DEMO");

    let mut storage = InMemoryStorage::new();
    
    // Insert some dummy vectors
    storage.insert(1, Vector::new(vec![1.0, 0.0, 0.0]));
    storage.insert(2, Vector::new(vec![0.0, 1.0, 0.0]));
    storage.insert(3, Vector::new(vec![0.0, 0.1, 0.9]));

    let query = Vector::new(vec![0.0, 0.0, 1.0]);
    let index = FlatIndex::<InMemoryStorage, L2Distance>::new(&storage);
    
    let results = index.search(&query, 2);
    
    println!("Search results for {:?}:", query);
    for res in results {
        println!("ID: {}, Distance: {}", res.id, res.distance);
    }

    // Demonstrate persistence placeholder
    let persistence = FilePersistence::new("wegdort.db");
    persistence.save(&storage).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_search() {
        let mut storage = InMemoryStorage::new();
        storage.insert(1, Vector::new(vec![1.0, 0.0]));
        storage.insert(2, Vector::new(vec![0.0, 1.0]));

        let query = Vector::new(vec![1.0, 0.1]);
        let index = FlatIndex::<InMemoryStorage, L2Distance>::new(&storage);
        let results = index.search(&query, 1);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1);
    }
}
