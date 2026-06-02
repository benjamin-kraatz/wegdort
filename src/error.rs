use crate::storage::VectorId;
use std::fmt;
use std::io;

/// Result type used by wegdort public APIs.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors returned by wegdort.
#[derive(Debug)]
pub enum Error {
    /// Store dimensions must be greater than zero.
    ZeroDimensions,
    /// A vector or query had the wrong number of dimensions.
    DimensionMismatch { expected: usize, actual: usize },
    /// The supplied vector id already exists.
    DuplicateId(VectorId),
    /// A vector or query contained a non-finite value.
    NonFiniteValue,
    /// Cosine similarity cannot be computed for a zero vector.
    ZeroVectorForCosine,
    /// The binary snapshot did not match the wegdort format.
    InvalidSnapshot(&'static str),
    /// The binary snapshot uses a newer or unsupported format version.
    UnsupportedSnapshotVersion(u16),
    /// The binary snapshot is truncated or internally inconsistent.
    CorruptedFile(&'static str),
    /// File system or stream I/O failed.
    Io(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimensions => write!(f, "dimensions must be greater than zero"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::DuplicateId(id) => write!(f, "vector id {} already exists", id.get()),
            Self::NonFiniteValue => write!(f, "vectors must contain only finite f32 values"),
            Self::ZeroVectorForCosine => {
                write!(f, "cosine similarity is undefined for zero vectors")
            }
            Self::InvalidSnapshot(reason) => write!(f, "invalid snapshot: {reason}"),
            Self::UnsupportedSnapshotVersion(version) => {
                write!(f, "unsupported snapshot version {version}")
            }
            Self::CorruptedFile(reason) => write!(f, "corrupted snapshot: {reason}"),
            Self::Io(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}
