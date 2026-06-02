use crate::core::{Vector, Distance};
use crate::storage::{Storage, VectorId};
use std::marker::PhantomData;

pub struct SearchResult {
    pub id: VectorId,
    pub distance: f32,
}

pub trait Index {
    fn search(&self, query: &Vector, k: usize) -> Vec<SearchResult>;
}

pub struct FlatIndex<'a, S, D>
where
    S: Storage,
    D: Distance,
{
    storage: &'a S,
    _distance: PhantomData<D>,
}

impl<'a, S, D> FlatIndex<'a, S, D>
where
    S: Storage,
    D: Distance,
{
    pub fn new(storage: &'a S) -> Self {
        Self {
            storage,
            _distance: PhantomData,
        }
    }
}

impl<'a, S, D> Index for FlatIndex<'a, S, D>
where
    S: Storage,
    D: Distance,
{
    fn search(&self, query: &Vector, k: usize) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = self.storage.list_ids()
            .into_iter()
            .filter_map(|id| {
                self.storage.get(id).map(|v| SearchResult {
                    id,
                    distance: D::calculate(query, v),
                })
            })
            .collect();

        // For distance, smaller is better (e.g. L2). 
        // For similarity, larger is better (e.g. Cosine).
        // This is a simple implementation.
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.into_iter().take(k).collect()
    }
}
