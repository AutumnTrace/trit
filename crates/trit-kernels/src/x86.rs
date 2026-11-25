//! AVX2 and AVX-512 VNNI kernels.
//!
//! Both use the same trick: decode each weight to its *digit* `d = t + 1` in
//! {0, 1, 2}, which is a valid unsigned byte, multiply-accumulate it against
//! the signed int8 activations with `vpmaddubsw` (AVX2) or `vpdpbusd`
//! (VNNI), and subtract `Σ x` at the end: `Σ t·x = Σ d·x − Σ x`.

#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::pack::{BLOCK_BYTES, Layout, RowsView};

#[inline(always)]
unsafe fn load256(p: *const u8) -> __m256i {
    _mm256_loadu_si256(p as *const __m256i)
}

#[inline(always)]
unsafe fn load256i(p: *const i8) -> __m256i {
    _mm256_loadu_si256(p as *const __m256i)
}

#[inline(always)]
unsafe fn hsum256(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
    _mm_cvtsi128_si32(s)
}

/// Unsigned `a >= b` per byte, as 0xFF / 0x00.
#[inline(always)]
unsafe fn ge_epu8(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a)
}

/// # Safety
/// Requires AVX2. `x` must cover `m.blocks * block_cols` activations and `out` `m.rows` entries.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gemv_avx2(m: RowsView<'_>, x: &[i8], x_sum: i32, out: &mut [i32]) {
    debug_assert!(x.len() >= m.blocks * m.layout.block_cols());
    debug_assert!(out.len() >= m.rows);
    let rb = m.blocks * BLOCK_BYTES;
    let ones = _mm256_set1_epi16(1);
    match m.layout {
        Layout::I2 => {
            let mask = _mm256_set1_epi8(3);
            for (r, o) in out.iter_mut().enumerate().take(m.rows) {
                let row = m.data.as_ptr().add(r * rb);
                let mut acc = _mm256_setzero_si256();
                for b in 0..m.blocks {
                    let w = load256(row.add(b * BLOCK_BYTES));
                    let xb = x.as_ptr().add(b * 128);
                    let d0 = _mm256_and_si256(w, mask);
                    let d1 = _mm256_and_si256(_mm256_srli_epi16(w, 2), mask);
                    let d2 = _mm256_and_si256(_mm256_srli_epi16(w, 4), mask);
                    let d3 = _mm256_and_si256(_mm256_srli_epi16(w, 6), mask);
                    let p0 = _mm256_maddubs_epi16(d0, load256i(xb));
                    let p1 = _mm256_maddubs_epi16(d1, load256i(xb.add(32)));
                    let p2 = _mm256_maddubs_epi16(d2, load256i(xb.add(64)));
                    let p3 = _mm256_maddubs_epi16(d3, load256i(xb.add(96)));
                    // Each lane is at most 4 * 2 * 2 * 127 = 2032: no overflow.
                    let s = _mm256_add_epi16(_mm256_add_epi16(p0, p1), _mm256_add_epi16(p2, p3));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(s, ones));
                }
                *o = hsum256(acc) - x_sum;
            }
        }
        Layout::Tq => {
            let c86 = _mm256_set1_epi8(86u8 as i8);
            let c171 = _mm256_set1_epi8(171u8 as i8);
            let zero = _mm256_setzero_si256();
            for (r, o) in out.iter_mut().enumerate().take(m.rows) {
                let row = m.data.as_ptr().add(r * rb);
                let mut acc = _mm256_setzero_si256();
                for b in 0..m.blocks {
                    let mut q = load256(row.add(b * BLOCK_BYTES));
                    let xb = x.as_ptr().add(b * 160);
                    let mut s = zero;
                    for k in 0..5 {
                        // digit = floor(3q / 256) = (q >= 86) + (q >= 171)
                        let d = _mm256_sub_epi8(
                            _mm256_sub_epi8(zero, ge_epu8(q, c86)),
                            ge_epu8(q, c171),
                        );
                        let p = _mm256_maddubs_epi16(d, load256i(xb.add(k * 32)));
                        s = _mm256_add_epi16(s, p);
                        // q = 3q mod 256 brings the next digit to the top.
                        q = _mm256_add_epi8(q, _mm256_add_epi8(q, q));
                    }
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(s, ones));
                }
                *o = hsum256(acc) - x_sum;
            }
        }
    }
}

/// Two consecutive 32-byte activation stripes from different blocks, as one register.
#[inline(always)]
unsafe fn load_pair(lo: *const i8, hi: *const i8) -> __m512i {
    _mm512_inserti64x4(_mm512_castsi256_si512(load256i(lo)), load256i(hi), 1)
}

/// # Safety
/// Requires AVX-512 F/BW/VL and VNNI. Same buffer requirements as [`gemv_avx2`].
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
pub(crate) unsafe fn gemv_avx512(m: RowsView<'_>, x: &[i8], x_sum: i32, out: &mut [i32]) {
    debug_assert!(x.len() >= m.blocks * m.layout.block_cols());
    debug_assert!(out.len() >= m.rows);
    let rb = m.blocks * BLOCK_BYTES;
    let pairs = m.blocks / 2;
    let tail = m.blocks % 2 == 1;
    match m.layout {
        Layout::I2 => {
            let mask = _mm512_set1_epi8(3);
            let mask256 = _mm256_set1_epi8(3);
            for (r, o) in out.iter_mut().enumerate().take(m.rows) {
                let row = m.data.as_ptr().add(r * rb);
                // One accumulator per digit: vpdpbusd has a latency of several
                // cycles, and a single chain would serialise the four of them.
                let mut acc = [_mm512_setzero_si512(); 4];
                for p in 0..pairs {
                    let w = _mm512_loadu_si512(row.add(p * 64) as *const __m512i);
                    let x0 = x.as_ptr().add(p * 256);
                    let x1 = x0.add(128);
                    acc[0] =
                        _mm512_dpbusd_epi32(acc[0], _mm512_and_si512(w, mask), load_pair(x0, x1));
                    acc[1] = _mm512_dpbusd_epi32(
                        acc[1],
                        _mm512_and_si512(_mm512_srli_epi16(w, 2), mask),
                        load_pair(x0.add(32), x1.add(32)),
                    );
                    acc[2] = _mm512_dpbusd_epi32(
                        acc[2],
                        _mm512_and_si512(_mm512_srli_epi16(w, 4), mask),
                        load_pair(x0.add(64), x1.add(64)),
                    );
                    acc[3] = _mm512_dpbusd_epi32(
                        acc[3],
                        _mm512_and_si512(_mm512_srli_epi16(w, 6), mask),
                        load_pair(x0.add(96), x1.add(96)),
                    );
                }
                let acc = _mm512_add_epi32(
                    _mm512_add_epi32(acc[0], acc[1]),
                    _mm512_add_epi32(acc[2], acc[3]),
                );
                let mut total = _mm512_reduce_add_epi32(acc);
                if tail {
                    let b = m.blocks - 1;
                    let w = load256(row.add(b * BLOCK_BYTES));
                    let xb = x.as_ptr().add(b * 128);
                    let mut a = _mm256_setzero_si256();
                    for k in 0..4 {
                        let d = _mm256_and_si256(
                            _mm256_srlv_epi16(w, _mm256_set1_epi16(2 * k as i16)),
                            mask256,
                        );
                        a = _mm256_dpbusd_epi32(a, d, load256i(xb.add(k * 32)));
                    }
                    total += hsum256(a);
                }
                *o = total - x_sum;
            }
        }
        Layout::Tq => {
            let c86 = _mm512_set1_epi8(86u8 as i8);
            let c171 = _mm512_set1_epi8(171u8 as i8);
            let one = _mm512_set1_epi8(1);
            for (r, o) in out.iter_mut().enumerate().take(m.rows) {
                let row = m.data.as_ptr().add(r * rb);
                let mut acc = [_mm512_setzero_si512(); 5];
                for p in 0..pairs {
                    let mut q = _mm512_loadu_si512(row.add(p * 64) as *const __m512i);
                    let x0 = x.as_ptr().add(p * 320);
                    let x1 = x0.add(160);
                    for (k, a) in acc.iter_mut().enumerate() {
                        let lo = _mm512_maskz_mov_epi8(_mm512_cmpge_epu8_mask(q, c86), one);
                        let d = _mm512_mask_add_epi8(lo, _mm512_cmpge_epu8_mask(q, c171), lo, one);
                        *a = _mm512_dpbusd_epi32(*a, d, load_pair(x0.add(k * 32), x1.add(k * 32)));
                        q = _mm512_add_epi8(q, _mm512_add_epi8(q, q));
                    }
                }
                let acc = acc
                    .into_iter()
                    .fold(_mm512_setzero_si512(), |s, a| _mm512_add_epi32(s, a));
                let mut total = _mm512_reduce_add_epi32(acc);
                if tail {
                    let b = m.blocks - 1;
                    let mut q = load256(row.add(b * BLOCK_BYTES));
                    let xb = x.as_ptr().add(b * 160);
                    let c86 = _mm256_set1_epi8(86u8 as i8);
                    let c171 = _mm256_set1_epi8(171u8 as i8);
                    let one = _mm256_set1_epi8(1);
                    let mut a = _mm256_setzero_si256();
                    for k in 0..5 {
                        let lo = _mm256_maskz_mov_epi8(_mm256_cmpge_epu8_mask(q, c86), one);
                        let d = _mm256_mask_add_epi8(lo, _mm256_cmpge_epu8_mask(q, c171), lo, one);
                        a = _mm256_dpbusd_epi32(a, d, load256i(xb.add(k * 32)));
                        q = _mm256_add_epi8(q, _mm256_add_epi8(q, q));
                    }
                    total += hsum256(a);
                }
                *o = total - x_sum;
            }
        }
    }
}
