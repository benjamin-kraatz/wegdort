use std::ops::Deref;

#[derive(Debug, Clone, PartialEq)]
pub struct Vector(Vec<f32>);

impl Vector {
    pub fn new(data: Vec<f32>) -> Self {
        Self(data)
    }

    pub fn dims(&self) -> usize {
        self.0.len()
    }
}

impl Deref for Vector {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub trait Distance {
    fn calculate(v1: &Vector, v2: &Vector) -> f32;
}

pub struct CosineSimilarity;

impl Distance for CosineSimilarity {
    fn calculate(v1: &Vector, v2: &Vector) -> f32 {
        let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm_v1: f32 = v1.iter().map(|a| a * a).sum::<f32>().sqrt();
        let norm_v2: f32 = v2.iter().map(|a| a * a).sum::<f32>().sqrt();
        dot_product / (norm_v1 * norm_v2)
    }
}

pub struct L2Distance;

impl Distance for L2Distance {
    fn calculate(v1: &Vector, v2: &Vector) -> f32 {
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}
