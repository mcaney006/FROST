//! Hardened safetensors reader.
//!
//! Every header field is validated with checked arithmetic before any byte of
//! tensor data is touched: header length vs file length, dtype names, shape
//! dimensions, element-count overflow, byte-size agreement and offset ranges.
//! Malformed files return `SafetensorsError`; nothing here panics on input.

use memmap2::Mmap;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

/// Upstream readers cap the header at 100 MB; a larger value is corruption.
const MAX_HEADER_BYTES: u64 = 100_000_000;
const MAX_RANK: usize = 16;

#[derive(Debug, thiserror::Error)]
pub enum SafetensorsError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("file too small: {0} bytes")]
    TooSmall(u64),
    #[error("header length {0} is larger than the file or the {MAX_HEADER_BYTES}-byte limit")]
    HeaderLength(u64),
    #[error("header is not valid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("header is not a JSON object")]
    HeaderNotObject,
    #[error("tensor {name}: {reason}")]
    Tensor { name: String, reason: String },
    #[error("missing tensor {0}")]
    Missing(String),
    #[error("tensor {name}: expected dtype {expected:?}, found {found:?}")]
    Dtype { name: String, expected: Dtype, found: Dtype },
    #[error("tensor {name}: expected shape {expected:?}, found {found:?}")]
    Shape { name: String, expected: Vec<usize>, found: Vec<usize> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype { Bool, U8, I8, U16, I16, F16, BF16, U32, I32, F32, U64, I64, F64 }

impl Dtype {
    fn parse(s: &str) -> Option<Dtype> {
        Some(match s {
            "BOOL" => Dtype::Bool, "U8" => Dtype::U8, "I8" => Dtype::I8,
            "U16" => Dtype::U16, "I16" => Dtype::I16, "F16" => Dtype::F16, "BF16" => Dtype::BF16,
            "U32" => Dtype::U32, "I32" => Dtype::I32, "F32" => Dtype::F32,
            "U64" => Dtype::U64, "I64" => Dtype::I64, "F64" => Dtype::F64,
            _ => return None,
        })
    }
    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            Dtype::Bool | Dtype::U8 | Dtype::I8 => 1,
            Dtype::U16 | Dtype::I16 | Dtype::F16 | Dtype::BF16 => 2,
            Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
            Dtype::U64 | Dtype::I64 | Dtype::F64 => 8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Absolute byte range within the file (validated at open).
    start: usize,
    end: usize,
}

impl TensorInfo {
    pub fn nbytes(&self) -> usize { self.end - self.start }
    pub fn numel(&self) -> usize { self.shape.iter().product() }
}

pub struct SafeTensors {
    mmap: Mmap,
    tensors: BTreeMap<String, TensorInfo>,
    pub metadata: BTreeMap<String, String>,
}

fn terr(name: &str, reason: impl Into<String>) -> SafetensorsError {
    SafetensorsError::Tensor { name: name.to_string(), reason: reason.into() }
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<SafeTensors, SafetensorsError> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len < 8 { return Err(SafetensorsError::TooSmall(file_len)); }
        // SAFETY: read-only mapping of a regular file we just opened; the file may be
        // truncated by another process later (SIGBUS), which no userland reader can prevent.
        let mmap = unsafe { Mmap::map(&file)? };
        let hlen = u64::from_le_bytes(mmap[0..8].try_into().expect("8 bytes"));
        if hlen > MAX_HEADER_BYTES || hlen > file_len - 8 { return Err(SafetensorsError::HeaderLength(hlen)); }
        let hlen = hlen as usize;
        let data_start = 8usize.checked_add(hlen).ok_or(SafetensorsError::HeaderLength(hlen as u64))?;
        let data_len = (file_len as usize) - data_start;

        let header: serde_json::Value = serde_json::from_slice(&mmap[8..data_start])?;
        let obj = header.as_object().ok_or(SafetensorsError::HeaderNotObject)?;

        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (name, meta) in obj {
            if name == "__metadata__" {
                if let Some(m) = meta.as_object() {
                    for (k, v) in m { if let Some(s) = v.as_str() { metadata.insert(k.clone(), s.to_string()); } }
                }
                continue;
            }
            let m = meta.as_object().ok_or_else(|| terr(name, "entry is not an object"))?;
            let dtype_s = m.get("dtype").and_then(|d| d.as_str()).ok_or_else(|| terr(name, "missing dtype"))?;
            let dtype = Dtype::parse(dtype_s).ok_or_else(|| terr(name, format!("unknown dtype {dtype_s:?}")))?;
            let shape_v = m.get("shape").and_then(|s| s.as_array()).ok_or_else(|| terr(name, "missing shape"))?;
            if shape_v.len() > MAX_RANK { return Err(terr(name, format!("rank {} exceeds {MAX_RANK}", shape_v.len()))); }
            let mut shape = Vec::with_capacity(shape_v.len());
            let mut numel: usize = 1;
            for d in shape_v {
                let d = d.as_u64().ok_or_else(|| terr(name, "shape dimension is not a non-negative integer"))?;
                let d = usize::try_from(d).map_err(|_| terr(name, "shape dimension does not fit usize"))?;
                numel = numel.checked_mul(d).ok_or_else(|| terr(name, "element count overflows"))?;
                shape.push(d);
            }
            let offs = m.get("data_offsets").and_then(|o| o.as_array()).ok_or_else(|| terr(name, "missing data_offsets"))?;
            if offs.len() != 2 { return Err(terr(name, "data_offsets must have two entries")); }
            let start = offs[0].as_u64().ok_or_else(|| terr(name, "data_offsets[0] invalid"))?;
            let end = offs[1].as_u64().ok_or_else(|| terr(name, "data_offsets[1] invalid"))?;
            if start > end { return Err(terr(name, format!("data_offsets start {start} > end {end}"))); }
            if end > data_len as u64 { return Err(terr(name, format!("data_offsets end {end} beyond data section ({data_len} bytes)"))); }
            let expected = numel.checked_mul(dtype.size()).ok_or_else(|| terr(name, "byte size overflows"))?;
            let actual = (end - start) as usize;
            if expected != actual { return Err(terr(name, format!("shape {shape:?} x {dtype:?} needs {expected} bytes, header gives {actual}"))); }
            let abs_start = data_start.checked_add(start as usize).ok_or_else(|| terr(name, "offset overflow"))?;
            let abs_end = data_start.checked_add(end as usize).ok_or_else(|| terr(name, "offset overflow"))?;
            tensors.insert(name.clone(), TensorInfo { dtype, shape, start: abs_start, end: abs_end });
        }
        Ok(SafeTensors { mmap, tensors, metadata })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> { self.tensors.keys().map(|s| s.as_str()) }
    pub fn len(&self) -> usize { self.tensors.len() }
    pub fn is_empty(&self) -> bool { self.tensors.is_empty() }
    pub fn contains(&self, name: &str) -> bool { self.tensors.contains_key(name) }
    pub fn info(&self, name: &str) -> Option<&TensorInfo> { self.tensors.get(name) }

    /// Raw bytes of a tensor (range validated at open).
    pub fn bytes(&self, name: &str) -> Result<(&[u8], &TensorInfo), SafetensorsError> {
        let info = self.tensors.get(name).ok_or_else(|| SafetensorsError::Missing(name.to_string()))?;
        Ok((&self.mmap[info.start..info.end], info))
    }

    /// Raw bytes after checking dtype and exact shape.
    pub fn expect(&self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<&[u8], SafetensorsError> {
        let (b, info) = self.bytes(name)?;
        if info.dtype != dtype {
            return Err(SafetensorsError::Dtype { name: name.into(), expected: dtype, found: info.dtype });
        }
        if info.shape != shape {
            return Err(SafetensorsError::Shape { name: name.into(), expected: shape.to_vec(), found: info.shape.clone() });
        }
        Ok(b)
    }

    /// Copy an F32 tensor out (alignment-safe), returning (data, shape).
    pub fn f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), SafetensorsError> {
        let (b, info) = self.bytes(name)?;
        if info.dtype != Dtype::F32 {
            return Err(SafetensorsError::Dtype { name: name.into(), expected: Dtype::F32, found: info.dtype });
        }
        let v = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok((v, info.shape.clone()))
    }
}

/// bf16 bit pattern -> f32 (exact: bf16 is the top half of an f32).
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 { f32::from_bits((bits as u32) << 16) }

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_file(header: &str, data: &[u8]) -> std::path::PathBuf {
        let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let p = std::env::temp_dir().join(format!("frost-st-{n}-{}.safetensors", std::process::id()));
        let mut f = File::create(&p).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(data).unwrap();
        p
    }

    fn valid() -> (String, Vec<u8>) {
        let mut data = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] { data.extend_from_slice(&v.to_le_bytes()); }
        data.extend_from_slice(&[0x80, 0x3F, 0x00, 0x40]); // two bf16: 1.0, 2.0
        let h = r#"{"__metadata__":{"format":"mlx"},"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"b":{"dtype":"BF16","shape":[2],"data_offsets":[24,28]}}"#;
        (h.to_string(), data)
    }

    #[test]
    fn parses_valid_file() {
        let (h, d) = valid();
        let p = write_file(&h, &d);
        let st = SafeTensors::open(&p).unwrap();
        assert_eq!(st.metadata.get("format").map(String::as_str), Some("mlx"));
        let (v, shape) = st.f32("a").unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(v, vec![1., 2., 3., 4., 5., 6.]);
        let b = st.expect("b", Dtype::BF16, &[2]).unwrap();
        assert_eq!(bf16_to_f32(u16::from_le_bytes([b[0], b[1]])), 1.0);
        assert_eq!(bf16_to_f32(u16::from_le_bytes([b[2], b[3]])), 2.0);
        assert!(matches!(st.f32("b"), Err(SafetensorsError::Dtype { .. })));
        assert!(matches!(st.expect("a", Dtype::F32, &[3, 2]), Err(SafetensorsError::Shape { .. })));
        assert!(matches!(st.bytes("zzz"), Err(SafetensorsError::Missing(_))));
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn rejects_malformed_headers_without_panicking() {
        let (_, d) = valid();
        let cases: Vec<(&str, String)> = vec![
            ("not json", "{oops".into()),
            ("not object", "[1,2]".into()),
            ("unknown dtype", r#"{"a":{"dtype":"Q4","shape":[1],"data_offsets":[0,4]}}"#.into()),
            ("negative dim", r#"{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}"#.into()),
            ("numel overflow", r#"{"a":{"dtype":"F32","shape":[4294967295,4294967295,4294967295],"data_offsets":[0,4]}}"#.into()),
            ("start > end", r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}}"#.into()),
            ("end beyond data", r#"{"a":{"dtype":"F32","shape":[1000],"data_offsets":[0,4000]}}"#.into()),
            ("size mismatch", r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#.into()),
            ("missing offsets", r#"{"a":{"dtype":"F32","shape":[1]}}"#.into()),
            ("rank too large", format!(r#"{{"a":{{"dtype":"F32","shape":[{}],"data_offsets":[0,4]}}}}"#, vec!["1"; 17].join(","))),
        ];
        for (what, h) in cases {
            let p = write_file(&h, &d);
            let r = SafeTensors::open(&p);
            assert!(r.is_err(), "{what} should be rejected");
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn rejects_bad_header_length_and_tiny_files() {
        let p = std::env::temp_dir().join(format!("frost-st-tiny-{}.safetensors", std::process::id()));
        std::fs::write(&p, [1u8, 2, 3]).unwrap();
        assert!(matches!(SafeTensors::open(&p), Err(SafetensorsError::TooSmall(3))));
        std::fs::write(&p, u64::MAX.to_le_bytes()).unwrap();
        assert!(matches!(SafeTensors::open(&p), Err(SafetensorsError::HeaderLength(_))));
        let mut big = (500u64).to_le_bytes().to_vec();
        big.extend_from_slice(b"{}");
        std::fs::write(&p, &big).unwrap();
        assert!(matches!(SafeTensors::open(&p), Err(SafetensorsError::HeaderLength(500))));
        let _ = std::fs::remove_file(p);
    }
}
