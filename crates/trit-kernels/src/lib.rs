//! Packed ternary (1.58-bit) matrices and the GEMV kernels that multiply them
//! against int8 activations.
//!
//! A BitNet b1.58 linear layer computes `y = (W_t · x_q) · γ / s`, where
//! `W_t` has entries in {-1, 0, +1}, `γ` is one scale for the whole matrix and
//! `x_q = round(x · s)` is the input quantized to int8 per token with
//! `s = 127 / max|x|`. This crate does the integer part, `W_t · x_q`, exactly.
//!
//! ```
//! use trit_kernels::{Activations, Isa, Layout, TernaryMatrix, gemv};
//!
//! let w = TernaryMatrix::from_trits(Layout::I2, 2, 3, &[1, 0, -1, -1, 1, 1]);
//! let x = Activations::quantize(&[0.5, -1.0, 0.25]);
//! let mut y = [0i32; 2];
//! gemv(Isa::detect(), &w, &x, &mut y);
//! assert_eq!(y, [64 - 32, -64 - 127 + 32]);
//! ```

pub mod dense;
pub mod pack;
mod scalar;
#[cfg(target_arch = "wasm32")]
mod wasm;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

pub use pack::{Layout, TernaryMatrix, base3};

/// Instruction sets the ternary kernels are written for, in increasing order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Isa {
    Scalar,
    Simd128,
    Avx2,
    Avx512,
}

impl Isa {
    /// The best kernel this machine (or, on wasm32, this build) supports.
    pub fn detect() -> Isa {
        *Self::available().last().unwrap_or(&Isa::Scalar)
    }

    /// Every kernel that can run here, scalar first.
    pub fn available() -> Vec<Isa> {
        [Isa::Scalar, Isa::Simd128, Isa::Avx2, Isa::Avx512]
            .into_iter()
            .filter(|i| i.is_available())
            .collect()
    }

    /// Whether this kernel can run here. Cheap: feature detection is cached by std.
    pub fn is_available(self) -> bool {
        match self {
            Isa::Scalar => true,
            Isa::Simd128 => cfg!(all(target_arch = "wasm32", target_feature = "simd128")),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Isa::Avx2 => std::is_x86_feature_detected!("avx2"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Isa::Avx512 => {
                std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw")
                    && std::is_x86_feature_detected!("avx512vl")
                    && std::is_x86_feature_detected!("avx512vnni")
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            Isa::Avx2 | Isa::Avx512 => false,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::Simd128 => "simd128",
            Isa::Avx2 => "avx2",
            Isa::Avx512 => "avx512-vnni",
        }
    }
}

/// Every padded width the layouts use divides this, so one activation buffer
/// fits any matrix with the same logical column count.
pub const ACTIVATION_ALIGN: usize = 640;

/// An activation vector quantized to int8 the BitNet way.
#[derive(Clone, Debug, Default)]
pub struct Activations {
    /// Quantized values, zero-padded to a multiple of [`ACTIVATION_ALIGN`].
    pub q: Vec<i8>,
    /// `127 / max|x|`; the real value of `q[i]` is `q[i] / scale`.
    pub scale: f32,
    /// `Σ q[i]`, needed by the SIMD kernels.
    pub sum: i32,
    /// Logical length.
    pub len: usize,
}

impl Activations {
    pub fn quantize(x: &[f32]) -> Self {
        let mut a = Self::default();
        a.quantize_from(x);
        a
    }

    /// Quantize into this buffer, reusing its allocation.
    ///
    /// Matches the PyTorch training code bit for bit:
    /// `s = 127 / clamp(max|x|, 1e-5)`, `q = clamp(round_half_even(x * s), -128, 127)`.
    pub fn quantize_from(&mut self, x: &[f32]) {
        let absmax = x.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-5);
        let s = 127.0f32 / absmax;
        self.q.clear();
        self.q.resize(
            x.len()
                .next_multiple_of(ACTIVATION_ALIGN)
                .max(ACTIVATION_ALIGN),
            0,
        );
        let mut sum = 0i32;
        for (q, &v) in self.q.iter_mut().zip(x) {
            let r = (v * s).round_ties_even().clamp(-128.0, 127.0) as i8;
            sum += r as i32;
            *q = r;
        }
        self.scale = s;
        self.sum = sum;
        self.len = x.len();
    }
}

/// `out[r] = Σ_c W[r, c] · x.q[c]`, exactly, with the chosen kernel.
///
/// # Panics
/// If `isa` is not available on this machine, if `x` was quantized from a
/// vector of a different length than `w.cols()`, or if `out` is shorter than `w.rows()`.
pub fn gemv(isa: Isa, w: &TernaryMatrix, x: &Activations, out: &mut [i32]) {
    gemv_rows(isa, w, 0..w.rows(), x, out);
}

/// [`gemv`] for the row range `rows` only; `out` holds just those rows.
pub fn gemv_rows(
    isa: Isa,
    w: &TernaryMatrix,
    rows: std::ops::Range<usize>,
    x: &Activations,
    out: &mut [i32],
) {
    assert_eq!(
        x.len,
        w.cols(),
        "activation length must equal the matrix column count"
    );
    assert!(x.q.len() >= w.padded_cols());
    assert!(rows.end <= w.rows() && out.len() >= rows.len());
    let n = rows.len();
    kernel(isa, w, rows, &x.q, x.sum, &mut out[..n]);
}

/// Like [`gemv`], splitting the rows over `threads` scoped threads.
pub fn gemv_parallel(
    isa: Isa,
    w: &TernaryMatrix,
    x: &Activations,
    out: &mut [i32],
    threads: usize,
) {
    let rows = w.rows();
    assert!(out.len() >= rows);
    let threads = threads.clamp(1, rows.max(1));
    let per = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (i, chunk) in out[..rows].chunks_mut(per).enumerate() {
            let start = i * per;
            let end = start + chunk.len();
            scope.spawn(move || gemv_rows(isa, w, start..end, x, chunk));
        }
    });
}

fn kernel(
    isa: Isa,
    w: &TernaryMatrix,
    rows: std::ops::Range<usize>,
    q: &[i8],
    sum: i32,
    out: &mut [i32],
) {
    assert!(isa.is_available(), "{isa:?} is not available here");
    let view = w.view(rows);
    match isa {
        Isa::Scalar => scalar::gemv(view, q, out),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        // SAFETY: availability checked above; lengths checked by the callers.
        Isa::Avx2 => unsafe { x86::gemv_avx2(view, q, sum, out) },
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        // SAFETY: as above.
        Isa::Avx512 => unsafe { x86::gemv_avx512(view, q, sum, out) },
        #[cfg(target_arch = "wasm32")]
        Isa::Simd128 => wasm::gemv_simd128(view, q, sum, out),
        #[allow(unreachable_patterns)]
        _ => unreachable!("{isa:?} has no kernel on this target"),
    }
    // Only the SIMD kernels need `sum`; on targets with just the scalar one it is unused.
    let _ = sum;
}

/// The obvious dense product, for tests: `trits` is row-major `rows x cols`.
pub fn reference_gemv(trits: &[i8], rows: usize, cols: usize, x: &[i8]) -> Vec<i32> {
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| trits[r * cols + c] as i32 * x[c] as i32)
                .sum()
        })
        .collect()
}
