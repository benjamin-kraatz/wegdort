use crate::error::{Error, Result};
use std::cmp::Ordering;

/// Metric used to score vectors during search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
    /// Cosine similarity. Higher scores are better.
    Cosine,
    /// Dot product similarity. Higher scores are better.
    Dot,
    /// Squared Euclidean distance. Lower scores are better.
    SquaredL2,
}

impl Metric {
    pub(crate) fn score(self, query: &[f32], vector: &[f32]) -> f32 {
        match self {
            Self::Cosine => cosine(query, vector),
            Self::Dot => dot(query, vector),
            Self::SquaredL2 => squared_l2(query, vector),
        }
    }

    pub(crate) fn compare_scores(self, a: f32, b: f32) -> Ordering {
        match self {
            Self::Cosine | Self::Dot => b.total_cmp(&a),
            Self::SquaredL2 => a.total_cmp(&b),
        }
    }

    pub(crate) fn to_u8(self) -> u8 {
        match self {
            Self::Cosine => 1,
            Self::Dot => 2,
            Self::SquaredL2 => 3,
        }
    }

    pub(crate) fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Cosine),
            2 => Some(Self::Dot),
            3 => Some(Self::SquaredL2),
            _ => None,
        }
    }
}

pub(crate) fn validate_vector(metric: Metric, vector: &[f32]) -> Result<()> {
    if !vector.iter().all(|value| value.is_finite()) {
        return Err(Error::NonFiniteValue);
    }

    if metric == Metric::Cosine && is_zero_vector(vector) {
        return Err(Error::ZeroVectorForCosine);
    }

    Ok(())
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    let mut index = 0;
    while index < a.len() {
        sum += a[index] * b[index];
        index += 1;
    }
    sum
}

pub(crate) fn vector_norm(vector: &[f32]) -> f32 {
    let mut sum = 0.0;
    let mut index = 0;
    while index < vector.len() {
        sum += vector[index] * vector[index];
        index += 1;
    }
    sum.sqrt()
}

pub(crate) fn cosine_with_norms(a: &[f32], norm_a: f32, b: &[f32], norm_b: f32) -> f32 {
    dot(a, b) / (norm_a * norm_b)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    cosine_with_norms(a, vector_norm(a), b, vector_norm(b))
}

fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    let mut index = 0;
    while index < a.len() {
        let diff = a[index] - b[index];
        sum += diff * diff;
        index += 1;
    }
    sum
}

fn is_zero_vector(vector: &[f32]) -> bool {
    vector.iter().all(|value| *value == 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_dot_product() {
        assert_eq!(Metric::Dot.score(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
    }

    #[test]
    fn scores_squared_l2() {
        assert_eq!(Metric::SquaredL2.score(&[1.0, 2.0], &[4.0, 6.0]), 25.0);
    }

    #[test]
    fn scores_cosine_similarity() {
        let score = Metric::Cosine.score(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn compares_similarity_highest_first() {
        assert_eq!(Metric::Cosine.compare_scores(0.9, 0.2), Ordering::Less);
        assert_eq!(Metric::Dot.compare_scores(9.0, 2.0), Ordering::Less);
    }

    #[test]
    fn compares_distance_lowest_first() {
        assert_eq!(Metric::SquaredL2.compare_scores(2.0, 9.0), Ordering::Less);
    }

    #[test]
    fn rejects_non_finite_values() {
        assert!(matches!(
            validate_vector(Metric::Dot, &[1.0, f32::NAN]),
            Err(Error::NonFiniteValue)
        ));
    }

    #[test]
    fn rejects_zero_vectors_for_cosine() {
        assert!(matches!(
            validate_vector(Metric::Cosine, &[0.0, 0.0]),
            Err(Error::ZeroVectorForCosine)
        ));
    }
}
