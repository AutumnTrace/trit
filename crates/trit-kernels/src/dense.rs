//! Dense GEMV kernels, as the baselines the ternary kernels are measured
//! against, plus the f32-by-int8-rows product used for the output head.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::Isa;

/// `out[r] = Σ_c w[r, c] x[c]`, row-major f32 weights.
pub fn gemv_f32(w: &[f32], x: &[f32], out: &mut [f32]) {
    let cols = x.len();
    assert_eq!(w.len(), out.len() * cols);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let isa = Isa::detect();
        if isa == Isa::Avx512 && cols.is_multiple_of(16) {
            // SAFETY: feature detected, lengths checked above.
            return unsafe { x86::gemv_f32_avx512(w, x, out) };
        }
        if isa >= Isa::Avx2 && std::is_x86_feature_detected!("fma") && cols.is_multiple_of(8) {
            // SAFETY: as above.
            return unsafe { x86::gemv_f32_avx2(w, x, out) };
        }
    }
    for (o, row) in out.iter_mut().zip(w.chunks_exact(cols)) {
        *o = dot_f32(row, x);
    }
}

/// Dot product written with 16 independent lanes so it vectorizes without
/// reassociation.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let (ca, ra) = a.as_chunks::<16>();
    let (cb, rb) = b.as_chunks::<16>();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..16 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// `out[r] = scale[r] · Σ_c w[r, c] x[c]` for int8 weights with one scale per
/// row and f32 activations. This is the tied embedding / output head.
pub fn gemv_f32_i8rows(w: &[i8], scales: &[f32], x: &[f32], out: &mut [f32]) {
    let cols = x.len();
    assert_eq!(w.len(), out.len() * cols);
    assert_eq!(scales.len(), out.len());
    for ((o, row), &s) in out.iter_mut().zip(w.chunks_exact(cols)).zip(scales) {
        let mut acc = [0f32; 16];
        let (cw, rw) = row.as_chunks::<16>();
        let (cx, rx) = x.as_chunks::<16>();
        for (a, b) in cw.iter().zip(cx) {
            for i in 0..16 {
                acc[i] += a[i] as f32 * b[i];
            }
        }
        let mut sum: f32 = acc.iter().sum();
        for (a, b) in rw.iter().zip(rx) {
            sum += *a as f32 * b;
        }
        *o = sum * s;
    }
}

/// `out[r] = Σ_c w[r, c] x[c]` for int8 weights and int8 activations
/// (activations must not contain -128).
pub fn gemv_i8(w: &[i8], x: &[i8], out: &mut [i32]) {
    let cols = x.len();
    assert_eq!(w.len(), out.len() * cols);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let isa = Isa::detect();
        if isa == Isa::Avx512 && cols.is_multiple_of(64) {
            // SAFETY: feature detected, lengths checked above.
            return unsafe { x86::gemv_i8_avx512(w, x, out) };
        }
        if isa >= Isa::Avx2 && cols.is_multiple_of(32) {
            // SAFETY: as above.
            return unsafe { x86::gemv_i8_avx2(w, x, out) };
        }
    }
    for (o, row) in out.iter_mut().zip(w.chunks_exact(cols)) {
        *o = row.iter().zip(x).map(|(&a, &b)| a as i32 * b as i32).sum();
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_op_in_unsafe_fn)]
mod x86 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gemv_f32_avx2(w: &[f32], x: &[f32], out: &mut [f32]) {
        let cols = x.len();
        for (r, o) in out.iter_mut().enumerate() {
            let row = w.as_ptr().add(r * cols);
            let (mut a0, mut a1, mut a2, mut a3) = (
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
            );
            let mut c = 0;
            while c + 32 <= cols {
                a0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(row.add(c)),
                    _mm256_loadu_ps(x.as_ptr().add(c)),
                    a0,
                );
                a1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(row.add(c + 8)),
                    _mm256_loadu_ps(x.as_ptr().add(c + 8)),
                    a1,
                );
                a2 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(row.add(c + 16)),
                    _mm256_loadu_ps(x.as_ptr().add(c + 16)),
                    a2,
                );
                a3 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(row.add(c + 24)),
                    _mm256_loadu_ps(x.as_ptr().add(c + 24)),
                    a3,
                );
                c += 32;
            }
            while c < cols {
                a0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(row.add(c)),
                    _mm256_loadu_ps(x.as_ptr().add(c)),
                    a0,
                );
                c += 8;
            }
            let s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
            let h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
            let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
            let h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
            *o = _mm_cvtss_f32(h);
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn gemv_f32_avx512(w: &[f32], x: &[f32], out: &mut [f32]) {
        let cols = x.len();
        for (r, o) in out.iter_mut().enumerate() {
            let row = w.as_ptr().add(r * cols);
            let (mut a0, mut a1, mut a2, mut a3) = (
                _mm512_setzero_ps(),
                _mm512_setzero_ps(),
                _mm512_setzero_ps(),
                _mm512_setzero_ps(),
            );
            let ld = |c: usize| {
                (
                    _mm512_loadu_ps(row.add(c)),
                    _mm512_loadu_ps(x.as_ptr().add(c)),
                )
            };
            let mut c = 0;
            while c + 64 <= cols {
                let (w0, x0) = ld(c);
                let (w1, x1) = ld(c + 16);
                let (w2, x2) = ld(c + 32);
                let (w3, x3) = ld(c + 48);
                a0 = _mm512_fmadd_ps(w0, x0, a0);
                a1 = _mm512_fmadd_ps(w1, x1, a1);
                a2 = _mm512_fmadd_ps(w2, x2, a2);
                a3 = _mm512_fmadd_ps(w3, x3, a3);
                c += 64;
            }
            while c < cols {
                let (w0, x0) = ld(c);
                a0 = _mm512_fmadd_ps(w0, x0, a0);
                c += 16;
            }
            *o = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn gemv_i8_avx2(w: &[i8], x: &[i8], out: &mut [i32]) {
        let cols = x.len();
        let ones = _mm256_set1_epi16(1);
        for (r, o) in out.iter_mut().enumerate() {
            let row = w.as_ptr().add(r * cols);
            let mut acc = _mm256_setzero_si256();
            let mut c = 0;
            while c < cols {
                let wv = _mm256_loadu_si256(row.add(c) as *const __m256i);
                let xv = _mm256_loadu_si256(x.as_ptr().add(c) as *const __m256i);
                // |w| * sign(x, w) == w * x, with the unsigned operand first.
                let p = _mm256_maddubs_epi16(_mm256_abs_epi8(wv), _mm256_sign_epi8(xv, wv));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones));
                c += 32;
            }
            let s = _mm_add_epi32(
                _mm256_castsi256_si128(acc),
                _mm256_extracti128_si256(acc, 1),
            );
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
            *o = _mm_cvtsi128_si32(s);
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub unsafe fn gemv_i8_avx512(w: &[i8], x: &[i8], out: &mut [i32]) {
        let cols = x.len();
        // (w + 128) is unsigned; Σ (w + 128) x = Σ w x + 128 Σ x.
        let sum_x: i32 = x.iter().map(|&v| v as i32).sum();
        let flip = _mm512_set1_epi8(-128);
        for (r, o) in out.iter_mut().enumerate() {
            let row = w.as_ptr().add(r * cols);
            // Four independent chains to hide vpdpbusd latency.
            let (mut a0, mut a1, mut a2, mut a3) = (
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
            );
            let ld = |c: usize| {
                (
                    _mm512_xor_si512(_mm512_loadu_si512(row.add(c) as *const __m512i), flip),
                    _mm512_loadu_si512(x.as_ptr().add(c) as *const __m512i),
                )
            };
            let mut c = 0;
            while c + 256 <= cols {
                let (w0, x0) = ld(c);
                let (w1, x1) = ld(c + 64);
                let (w2, x2) = ld(c + 128);
                let (w3, x3) = ld(c + 192);
                a0 = _mm512_dpbusd_epi32(a0, w0, x0);
                a1 = _mm512_dpbusd_epi32(a1, w1, x1);
                a2 = _mm512_dpbusd_epi32(a2, w2, x2);
                a3 = _mm512_dpbusd_epi32(a3, w3, x3);
                c += 256;
            }
            while c < cols {
                let (w0, x0) = ld(c);
                a0 = _mm512_dpbusd_epi32(a0, w0, x0);
                c += 64;
            }
            let acc = _mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3));
            *o = _mm512_reduce_add_epi32(acc) - 128 * sum_x;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    #[test]
    fn f32_matches_naive() {
        let mut s = 1;
        for (rows, cols) in [(3, 8), (5, 48), (7, 1000), (2, 384)] {
            let w: Vec<f32> = (0..rows * cols)
                .map(|_| lcg(&mut s) as f32 / u32::MAX as f32 - 0.5)
                .collect();
            let x: Vec<f32> = (0..cols)
                .map(|_| lcg(&mut s) as f32 / u32::MAX as f32 - 0.5)
                .collect();
            let mut out = vec![0f32; rows];
            gemv_f32(&w, &x, &mut out);
            for r in 0..rows {
                let want: f64 = (0..cols)
                    .map(|c| w[r * cols + c] as f64 * x[c] as f64)
                    .sum();
                assert!((out[r] as f64 - want).abs() < 1e-4, "{rows}x{cols} row {r}");
            }
        }
    }

    #[test]
    fn i8_matches_naive() {
        let mut s = 2;
        for (rows, cols) in [(3, 32), (5, 64), (4, 384), (2, 100)] {
            let w: Vec<i8> = (0..rows * cols)
                .map(|_| (lcg(&mut s) % 255) as i32 as i8)
                .collect();
            let x: Vec<i8> = (0..cols)
                .map(|_| ((lcg(&mut s) % 255) as i32 - 127) as i8)
                .collect();
            let mut out = vec![0i32; rows];
            gemv_i8(&w, &x, &mut out);
            for r in 0..rows {
                let want: i32 = (0..cols)
                    .map(|c| w[r * cols + c] as i32 * x[c] as i32)
                    .sum();
                assert_eq!(out[r], want, "{rows}x{cols} row {r}");
            }
        }
    }

    #[test]
    fn f32_i8rows_matches_naive() {
        let w: Vec<i8> = (0..3 * 40).map(|i| (i % 21) as i8 - 10).collect();
        let x: Vec<f32> = (0..40).map(|i| i as f32 * 0.25 - 3.0).collect();
        let scales = [0.5, 1.0, 2.0];
        let mut out = [0f32; 3];
        gemv_f32_i8rows(&w, &scales, &x, &mut out);
        for r in 0..3 {
            let want: f32 = (0..40).map(|c| w[r * 40 + c] as f32 * x[c]).sum::<f32>() * scales[r];
            assert!((out[r] - want).abs() < 1e-3);
        }
    }
}
