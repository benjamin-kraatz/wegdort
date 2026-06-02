use crate::error::{Error, Result};
use crate::metrics::{Metric, validate_vector};
use crate::storage::VectorId;
use crate::store::Store;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"WEGDORT\0";
const VERSION: u16 = 1;
// Header fields: magic(8) + version(2) + metric(1) + reserved(1)
// + dimensions(8) + vector_count(8).
const HEADER_LEN: usize = 8 + 2 + 1 + 1 + 8 + 8;

pub(crate) fn save(store: &Store, path: &Path) -> Result<()> {
    let temp_path = temp_path_for(path);
    let write_result = write_snapshot(store, &temp_path);

    match write_result {
        Ok(()) => {
            fs::rename(&temp_path, path)?;
            Ok(())
        }
        Err(error) => {
            let _ = fs::remove_file(&temp_path);
            Err(error)
        }
    }
}

pub(crate) fn load(path: &Path) -> Result<Store> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    read_snapshot(&bytes)
}

fn write_snapshot(store: &Store, path: &Path) -> Result<()> {
    let mut file = File::create(path)?;
    file.write_all(MAGIC)?;
    file.write_all(&VERSION.to_le_bytes())?;
    file.write_all(&[store.metric().to_u8()])?;
    file.write_all(&[0])?;
    file.write_all(&(store.dimensions() as u64).to_le_bytes())?;
    file.write_all(&(store.len() as u64).to_le_bytes())?;

    for (row, id) in store.ids().iter().copied().enumerate() {
        file.write_all(&id.get().to_le_bytes())?;
        let start = row * store.dimensions();
        let vector = &store.vectors()[start..start + store.dimensions()];
        for value in vector {
            file.write_all(&value.to_le_bytes())?;
        }
    }

    file.sync_all()?;
    Ok(())
}

fn read_snapshot(bytes: &[u8]) -> Result<Store> {
    if bytes.len() < HEADER_LEN {
        return Err(Error::CorruptedFile("file is shorter than snapshot header"));
    }

    if &bytes[0..8] != MAGIC {
        return Err(Error::InvalidSnapshot("missing wegdort magic bytes"));
    }

    let version = read_u16(bytes, 8)?;
    if version != VERSION {
        return Err(Error::UnsupportedSnapshotVersion(version));
    }

    let metric = Metric::from_u8(bytes[10]).ok_or(Error::InvalidSnapshot("invalid metric id"))?;
    let reserved = bytes[11];
    if reserved != 0 {
        return Err(Error::InvalidSnapshot("reserved header byte must be zero"));
    }

    let dimensions = usize_from_u64(read_u64(bytes, 12)?, "dimensions do not fit usize")?;
    if dimensions == 0 {
        return Err(Error::ZeroDimensions);
    }

    let count = usize_from_u64(read_u64(bytes, 20)?, "vector count does not fit usize")?;
    let bytes_per_vector = dimensions
        .checked_mul(4)
        .and_then(|value| value.checked_add(8))
        .ok_or(Error::CorruptedFile("snapshot dimensions overflow"))?;
    let expected_len = HEADER_LEN
        .checked_add(
            count
                .checked_mul(bytes_per_vector)
                .ok_or(Error::CorruptedFile("snapshot length overflow"))?,
        )
        .ok_or(Error::CorruptedFile("snapshot length overflow"))?;

    if bytes.len() != expected_len {
        return Err(Error::CorruptedFile(
            "file length does not match snapshot header",
        ));
    }

    let mut offset = HEADER_LEN;
    let mut ids = Vec::with_capacity(count);
    let mut vectors = Vec::with_capacity(count * dimensions);

    for _ in 0..count {
        let id = VectorId::new(read_u64(bytes, offset)?);
        offset += 8;
        ids.push(id);

        let row_start = vectors.len();
        for _ in 0..dimensions {
            let value = read_f32(bytes, offset)?;
            offset += 4;
            vectors.push(value);
        }
        validate_vector(metric, &vectors[row_start..row_start + dimensions])?;
    }

    Store::from_parts(dimensions, metric, ids, vectors)
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let chunk = bytes
        .get(offset..offset + 2)
        .ok_or(Error::CorruptedFile("truncated u16"))?;
    Ok(u16::from_le_bytes([chunk[0], chunk[1]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let chunk = bytes
        .get(offset..offset + 8)
        .ok_or(Error::CorruptedFile("truncated u64"))?;
    Ok(u64::from_le_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
    ]))
}

fn read_f32(bytes: &[u8], offset: usize) -> Result<f32> {
    let chunk = bytes
        .get(offset..offset + 4)
        .ok_or(Error::CorruptedFile("truncated f32"))?;
    Ok(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
}

fn usize_from_u64(value: u64, reason: &'static str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::CorruptedFile(reason))
}

fn temp_path_for(path: &Path) -> PathBuf {
    let mut temp = path.to_path_buf();
    let extension = match path.extension().and_then(|extension| extension.to_str()) {
        Some(extension) => format!("{extension}.tmp"),
        None => "tmp".to_string(),
    };
    temp.set_extension(extension);
    temp
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn round_trips_each_metric() {
        for metric in [Metric::Cosine, Metric::Dot, Metric::SquaredL2] {
            let path = test_path("round-trip");
            let mut store = Store::new(2, metric).unwrap();
            store.insert(VectorId::new(1), [1.0, 0.0]).unwrap();
            store.insert(VectorId::new(2), [0.0, 1.0]).unwrap();

            save(&store, &path).unwrap();
            let loaded = load(&path).unwrap();
            let _ = fs::remove_file(&path);

            assert_eq!(loaded.dimensions(), 2);
            assert_eq!(loaded.metric(), metric);
            assert_eq!(loaded.len(), 2);
            assert_eq!(
                loaded.search([1.0, 0.0], 1).unwrap()[0].id,
                VectorId::new(1)
            );
        }
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = valid_bytes();
        bytes[0] = b'X';
        assert!(matches!(
            read_snapshot(&bytes),
            Err(Error::InvalidSnapshot("missing wegdort magic bytes"))
        ));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = valid_bytes();
        bytes[8..10].copy_from_slice(&2_u16.to_le_bytes());
        assert!(matches!(
            read_snapshot(&bytes),
            Err(Error::UnsupportedSnapshotVersion(2))
        ));
    }

    #[test]
    fn rejects_invalid_metric() {
        let mut bytes = valid_bytes();
        bytes[10] = 99;
        assert!(matches!(
            read_snapshot(&bytes),
            Err(Error::InvalidSnapshot("invalid metric id"))
        ));
    }

    #[test]
    fn rejects_truncated_file() {
        let bytes = valid_bytes();
        assert!(matches!(
            read_snapshot(&bytes[..bytes.len() - 1]),
            Err(Error::CorruptedFile(
                "file length does not match snapshot header"
            ))
        ));
    }

    #[test]
    fn rejects_duplicate_ids() {
        let mut bytes = valid_bytes_with_count(2);
        let first_id = HEADER_LEN;
        let second_id = HEADER_LEN + 8 + 8;
        let id = bytes[first_id..first_id + 8].to_vec();
        bytes[second_id..second_id + 8].copy_from_slice(&id);
        assert!(matches!(
            read_snapshot(&bytes),
            Err(Error::CorruptedFile("duplicate vector id"))
        ));
    }

    #[test]
    fn rejects_non_finite_values() {
        let mut bytes = valid_bytes();
        let first_value = HEADER_LEN + 8;
        bytes[first_value..first_value + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(matches!(read_snapshot(&bytes), Err(Error::NonFiniteValue)));
    }

    #[test]
    fn rejects_cosine_zero_vector() {
        let mut bytes = valid_bytes();
        bytes[10] = Metric::Cosine.to_u8();
        let first_value = HEADER_LEN + 8;
        bytes[first_value..first_value + 4].copy_from_slice(&0.0_f32.to_le_bytes());
        bytes[first_value + 4..first_value + 8].copy_from_slice(&0.0_f32.to_le_bytes());
        assert!(matches!(
            read_snapshot(&bytes),
            Err(Error::ZeroVectorForCosine)
        ));
    }

    fn valid_bytes() -> Vec<u8> {
        valid_bytes_with_count(1)
    }

    fn valid_bytes_with_count(count: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.push(Metric::Dot.to_u8());
        bytes.push(0);
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&(count as u64).to_le_bytes());
        for index in 0..count {
            bytes.extend_from_slice(&((index + 1) as u64).to_le_bytes());
            bytes.extend_from_slice(&1.0_f32.to_le_bytes());
            bytes.extend_from_slice(&0.0_f32.to_le_bytes());
        }
        bytes
    }

    fn test_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("wegdort-{name}-{nanos}.wgd"))
    }
}
