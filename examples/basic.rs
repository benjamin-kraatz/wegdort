use wegdort::{Metric, Store, VectorId};

fn main() -> Result<(), wegdort::Error> {
    let mut store = Store::new(3, Metric::Cosine)?;

    store.insert(VectorId::new(1), [1.0, 0.0, 0.0])?;
    store.insert(VectorId::new(2), [0.0, 1.0, 0.0])?;
    store.insert(VectorId::new(3), [0.0, 0.2, 0.8])?;

    for hit in store.search([0.0, 0.0, 1.0], 2)? {
        println!("id={} score={}", hit.id, hit.score);
    }

    Ok(())
}
