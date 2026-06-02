//! Fast, lightweight in-memory vector storage and exact vector search.
//!
//! Wegdort stores fixed-dimension `f32` vectors by caller-supplied ids and
//! searches them with exact top-k flat search.
//!
//! ```
//! use wegdort::{Metric, Store, VectorId};
//!
//! # fn main() -> Result<(), wegdort::Error> {
//! let mut store = Store::new(3, Metric::Cosine)?;
//!
//! store.insert(VectorId::new(1), [1.0, 0.0, 0.0])?;
//! store.insert(VectorId::new(2), [0.0, 1.0, 0.0])?;
//! store.insert(VectorId::new(3), [0.0, 0.2, 0.8])?;
//!
//! let hits = store.search([0.0, 0.0, 1.0], 2)?;
//! assert_eq!(hits[0].id, VectorId::new(3));
//! # Ok(())
//! # }
//! ```

mod error;
mod metrics;
mod persistence;
mod search;
mod storage;
mod store;

pub use crate::error::{Error, Result};
pub use crate::metrics::Metric;
pub use crate::search::{SearchHit, SearchSnapshot};
pub use crate::storage::{VectorId, VectorIter};
pub use crate::store::{Store, StoreBuilder, UpsertResult};
