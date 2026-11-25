//! Parser for the `.trit` file (layout in docs/format.md, writer in
//! train/trit_train/fileformat.py).

use std::fmt;

pub const MAGIC: &[u8; 4] = b"TRIT";
pub const VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    Truncated(&'static str),
    BadMagic,
    UnsupportedVersion(u32),
    MissingTensor(String),
    BadTensor(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated(what) => write!(f, "file is truncated in the {what}"),
            Error::BadMagic => write!(f, "not a .trit file"),
            Error::UnsupportedVersion(v) => write!(f, "unsupported .trit version {v}"),
            Error::MissingTensor(n) => write!(f, "tensor {n} is missing"),
            Error::BadTensor(n) => write!(f, "tensor {n} has the wrong type or shape"),
        }
    }
}

impl std::error::Error for Error {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub vocab_size: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ffn_dim: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub eos_id: u32,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    I8,
    Ternary,
}

#[derive(Debug, Clone)]
pub struct TensorInfo<'a> {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub scale: f32,
    pub data: &'a [u8],
}

pub struct File<'a> {
    pub config: Config,
    pub tokens: Vec<&'a [u8]>,
    pub merges: Vec<(u16, u16, u16)>,
    pub tensors: Vec<TensorInfo<'a>>,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], Error> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or(Error::Truncated(what))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self, what: &'static str) -> Result<u8, Error> {
        Ok(self.take(1, what)?[0])
    }
    fn u16(&mut self, what: &'static str) -> Result<u16, Error> {
        Ok(u16::from_le_bytes(self.take(2, what)?.try_into().unwrap()))
    }
    fn u32(&mut self, what: &'static str) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
    }
    fn u64(&mut self, what: &'static str) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.take(8, what)?.try_into().unwrap()))
    }
    fn f32(&mut self, what: &'static str) -> Result<f32, Error> {
        Ok(f32::from_bits(self.u32(what)?))
    }
}

impl<'a> File<'a> {
    pub fn parse(buf: &'a [u8]) -> Result<Self, Error> {
        let mut r = Reader { buf, pos: 0 };
        if r.take(4, "header")? != MAGIC {
            return Err(Error::BadMagic);
        }
        let version = r.u32("header")?;
        if version != VERSION {
            return Err(Error::UnsupportedVersion(version));
        }
        let mut u = || r.u32("header").map(|v| v as usize);
        let (vocab_size, dim, n_layers, n_heads, ffn_dim, max_seq_len) =
            (u()?, u()?, u()?, u()?, u()?, u()?);
        let _reserved = (u()?, u()?);
        let rope_theta = r.f32("header")?;
        let norm_eps = r.f32("header")?;
        let eos_id = r.u32("header")?;
        let n_merges = r.u32("header")? as usize;
        let n_tensors = r.u32("header")? as usize;
        r.pos = 64;

        let mut tokens = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let n = r.u8("vocabulary")? as usize;
            tokens.push(r.take(n, "vocabulary")?);
        }
        let mut merges = Vec::with_capacity(n_merges);
        for _ in 0..n_merges {
            merges.push((r.u16("merges")?, r.u16("merges")?, r.u16("merges")?));
        }
        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let n = r.u8("tensor table")? as usize;
            let name = String::from_utf8_lossy(r.take(n, "tensor table")?).into_owned();
            let dtype = match r.u8("tensor table")? {
                0 => DType::F32,
                1 => DType::I8,
                2 => DType::Ternary,
                _ => return Err(Error::BadTensor(name)),
            };
            let ndim = r.u8("tensor table")? as usize;
            let mut shape = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                shape.push(r.u32("tensor table")? as usize);
            }
            let scale = r.f32("tensor table")?;
            let offset = r.u64("tensor table")? as usize;
            let nbytes = r.u64("tensor table")? as usize;
            let data = offset
                .checked_add(nbytes)
                .and_then(|end| buf.get(offset..end))
                .ok_or(Error::Truncated("tensor data"))?;
            tensors.push(TensorInfo {
                name,
                dtype,
                shape,
                scale,
                data,
            });
        }
        Ok(File {
            config: Config {
                vocab_size,
                dim,
                n_layers,
                n_heads,
                ffn_dim,
                max_seq_len,
                rope_theta,
                norm_eps,
                eos_id,
            },
            tokens,
            merges,
            tensors,
        })
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo<'a>, Error> {
        self.tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| Error::MissingTensor(name.into()))
    }

    /// An f32 tensor with exactly `len` elements.
    pub fn f32s(&self, name: &str, len: usize) -> Result<Vec<f32>, Error> {
        let t = self.tensor(name)?;
        if t.dtype != DType::F32 || t.data.len() != len * 4 {
            return Err(Error::BadTensor(name.into()));
        }
        Ok(t.data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }
}
