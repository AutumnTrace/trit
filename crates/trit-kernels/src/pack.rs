//! Ternary matrix layouts.
//!
//! Every layout stores a row as a run of 32-byte blocks. Within a block, byte
//! `j` holds one weight from each of several 32-column stripes, so that a
//! single SIMD operation on the 32 bytes yields 32 weights for 32 *contiguous*
//! columns and can be multiplied against 32 contiguous activations.
//!
//! * [`Layout::I2`]: 4 trits per byte, 2 bits each, storing `t + 1`.
//!   Digit `k` of byte `j` (bits `2k..2k+2`) is column `128 b + 32 k + j`.
//!   2 bits per weight.
//! * [`Layout::Tq`]: 5 trits per byte in base 3, 1.6 bits per weight. The five
//!   digits `d_k = t + 1` form `v = Σ d_k 3^(4-k)`, stored as the fixed-point
//!   byte `q = ceil(256 v / 243)`. Multiplying by 3 then shifts the next digit
//!   into the high byte: `d_k = ((q · 3^k mod 256) · 3) >> 8`, so extraction
//!   needs only byte adds and compares. Column is `160 b + 32 k + j`.
//!
//! Rows are padded with zero weights to a whole number of blocks; callers pad
//! activations with zeros to [`TernaryMatrix::padded_cols`].

/// Bytes per block in every layout.
pub const BLOCK_BYTES: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    /// 2 bits per weight, 128 columns per block.
    I2,
    /// Base 3, 1.6 bits per weight, 160 columns per block.
    Tq,
}

impl Layout {
    pub const ALL: [Layout; 2] = [Layout::I2, Layout::Tq];

    /// Trits stored per byte.
    pub const fn trits_per_byte(self) -> usize {
        match self {
            Layout::I2 => 4,
            Layout::Tq => 5,
        }
    }

    /// Columns covered by one 32-byte block.
    pub const fn block_cols(self) -> usize {
        BLOCK_BYTES * self.trits_per_byte()
    }

    pub fn bits_per_weight(self) -> f64 {
        8.0 / self.trits_per_byte() as f64
    }

    pub const fn name(self) -> &'static str {
        match self {
            Layout::I2 => "i2",
            Layout::Tq => "tq",
        }
    }
}

const POW3: [u32; 5] = [1, 3, 9, 27, 81];

/// Encode one TQ byte from five trits (column order `k = 0..5`).
pub fn tq_encode(trits: [i8; 5]) -> u8 {
    let mut v = 0u32;
    for (k, &t) in trits.iter().enumerate() {
        v += (t + 1) as u32 * POW3[4 - k];
    }
    // ceil(256 v / 243); v <= 242 so this is at most 255.
    ((v * 256).div_ceil(243)) as u8
}

/// Digit `k` (0..5) of a TQ byte, as a trit.
#[inline]
pub fn tq_digit(q: u8, k: usize) -> i8 {
    let qk = q.wrapping_mul(POW3[k] as u8);
    (((qk as u16) * 3) >> 8) as i8 - 1
}

/// A ternary matrix in one of the kernel layouts.
#[derive(Clone, Debug)]
pub struct TernaryMatrix {
    rows: usize,
    cols: usize,
    layout: Layout,
    blocks_per_row: usize,
    data: Vec<u8>,
}

impl TernaryMatrix {
    /// Pack a row-major `rows x cols` matrix of values in {-1, 0, 1}.
    ///
    /// # Panics
    /// If `trits.len() != rows * cols` or a value is outside {-1, 0, 1}.
    pub fn from_trits(layout: Layout, rows: usize, cols: usize, trits: &[i8]) -> Self {
        assert_eq!(trits.len(), rows * cols, "trits.len() must be rows * cols");
        assert!(
            trits.iter().all(|t| (-1..=1).contains(t)),
            "weights must be -1, 0 or 1"
        );
        let bc = layout.block_cols();
        let blocks_per_row = cols.div_ceil(bc).max(1);
        let mut data = vec![0u8; rows * blocks_per_row * BLOCK_BYTES];
        let at = |r: usize, c: usize| if c < cols { trits[r * cols + c] } else { 0 };
        for r in 0..rows {
            for b in 0..blocks_per_row {
                let base = (r * blocks_per_row + b) * BLOCK_BYTES;
                for j in 0..BLOCK_BYTES {
                    let col = |k: usize| b * bc + k * BLOCK_BYTES + j;
                    data[base + j] = match layout {
                        Layout::I2 => (0..4)
                            .fold(0u8, |acc, k| acc | (((at(r, col(k)) + 1) as u8) << (2 * k))),
                        Layout::Tq => tq_encode(std::array::from_fn(|k| at(r, col(k)))),
                    };
                }
            }
        }
        Self {
            rows,
            cols,
            layout,
            blocks_per_row,
            data,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn blocks_per_row(&self) -> usize {
        self.blocks_per_row
    }

    /// Columns including padding; activation vectors must be this long.
    pub fn padded_cols(&self) -> usize {
        self.blocks_per_row * self.layout.block_cols()
    }

    pub fn row_bytes(&self) -> usize {
        self.blocks_per_row * BLOCK_BYTES
    }

    /// The packed bytes, `rows * row_bytes()` long.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Weight at `(r, c)`.
    pub fn get(&self, r: usize, c: usize) -> i8 {
        assert!(r < self.rows && c < self.cols);
        let bc = self.layout.block_cols();
        let (b, w) = (c / bc, c % bc);
        let (k, j) = (w / BLOCK_BYTES, w % BLOCK_BYTES);
        let byte = self.data[(r * self.blocks_per_row + b) * BLOCK_BYTES + j];
        match self.layout {
            Layout::I2 => ((byte >> (2 * k)) & 3) as i8 - 1,
            Layout::Tq => tq_digit(byte, k),
        }
    }

    /// Unpack back to row-major trits.
    pub fn to_trits(&self) -> Vec<i8> {
        let mut out = Vec::with_capacity(self.rows * self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.push(self.get(r, c));
            }
        }
        out
    }

    /// The same weights in another layout.
    pub fn to_layout(&self, layout: Layout) -> Self {
        if layout == self.layout {
            return self.clone();
        }
        Self::from_trits(layout, self.rows, self.cols, &self.to_trits())
    }

    pub(crate) fn view(&self, rows: std::ops::Range<usize>) -> RowsView<'_> {
        let rb = self.row_bytes();
        RowsView {
            data: &self.data[rows.start * rb..rows.end * rb],
            rows: rows.len(),
            blocks: self.blocks_per_row,
            layout: self.layout,
        }
    }
}

/// A contiguous range of rows, which is what the kernels operate on.
#[derive(Clone, Copy)]
pub(crate) struct RowsView<'a> {
    pub data: &'a [u8],
    pub rows: usize,
    pub blocks: usize,
    pub layout: Layout,
}

/// Dense base-3 packing used by the `.trit` file format: five consecutive
/// columns per byte, little-endian digits (`v = Σ (t_i + 1) 3^i`), each row
/// padded with zero weights to a multiple of five.
pub mod base3 {
    /// Bytes for one packed row of `cols` trits.
    pub const fn row_bytes(cols: usize) -> usize {
        cols.div_ceil(5)
    }

    pub fn pack_row(trits: &[i8], out: &mut Vec<u8>) {
        for chunk in trits.chunks(5) {
            let mut v = 0u8;
            let mut p = 1u8;
            for &t in chunk {
                v += (t + 1) as u8 * p;
                p = p.wrapping_mul(3);
            }
            // Missing digits are trit 0 (digit 1).
            for _ in chunk.len()..5 {
                v += p;
                p = p.wrapping_mul(3);
            }
            out.push(v);
        }
    }

    /// The five trits of every byte value 0..243.
    pub const LUT: [[i8; 5]; 243] = {
        let mut lut = [[0i8; 5]; 243];
        let mut v = 0;
        while v < 243 {
            let mut x = v;
            let mut i = 0;
            while i < 5 {
                lut[v][i] = (x % 3) as i8 - 1;
                x /= 3;
                i += 1;
            }
            v += 1;
        }
        lut
    };

    /// Unpack one row of `cols` trits.
    ///
    /// # Errors
    /// If the input is too short or holds a byte >= 243.
    pub fn unpack_row(bytes: &[u8], cols: usize, out: &mut Vec<i8>) -> Result<(), &'static str> {
        let need = row_bytes(cols);
        if bytes.len() < need {
            return Err("base-3 row is truncated");
        }
        let start = out.len();
        for &b in &bytes[..need] {
            let digits = LUT.get(b as usize).ok_or("base-3 byte out of range")?;
            out.extend_from_slice(digits);
        }
        out.truncate(start + cols);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tq_digits_roundtrip_for_every_byte_value() {
        for v in 0..243u32 {
            let trits: [i8; 5] = std::array::from_fn(|k| ((v / POW3[4 - k]) % 3) as i8 - 1);
            let q = tq_encode(trits);
            for (k, &t) in trits.iter().enumerate() {
                assert_eq!(tq_digit(q, k), t, "v={v} k={k} q={q}");
            }
        }
    }

    #[test]
    fn tq_thresholds_match_the_simd_compare_form() {
        // The SIMD kernels compute digit = (q >= 86) + (q >= 171).
        for q in 0..=255u8 {
            let d = (((q as u16) * 3) >> 8) as u8;
            assert_eq!(d, (q >= 86) as u8 + (q >= 171) as u8, "q={q}");
        }
    }

    #[test]
    fn base3_roundtrip_with_ragged_rows() {
        for cols in [1, 4, 5, 6, 384, 1537] {
            let trits: Vec<i8> = (0..cols).map(|i| ((i * 7 + 3) % 3) as i8 - 1).collect();
            let mut packed = Vec::new();
            base3::pack_row(&trits, &mut packed);
            assert_eq!(packed.len(), base3::row_bytes(cols));
            assert!(packed.iter().all(|&b| b < 243));
            let mut back = Vec::new();
            base3::unpack_row(&packed, cols, &mut back).unwrap();
            assert_eq!(back, trits);
        }
    }

    #[test]
    fn padding_reads_as_zero_weights() {
        let m = TernaryMatrix::from_trits(Layout::I2, 1, 3, &[1, -1, 1]);
        assert_eq!(m.padded_cols(), 128);
        assert_eq!(m.data()[3], 0b0101_0101, "padding byte is four zero trits");
        let m = TernaryMatrix::from_trits(Layout::Tq, 1, 3, &[1, -1, 1]);
        assert_eq!(m.padded_cols(), 160);
        assert_eq!(tq_digit(m.data()[5], 0), 0);
    }
}
