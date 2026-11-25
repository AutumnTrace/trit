//! WebAssembly SIMD128 kernels.
//!
//! SIMD128 has no unsigned-by-signed byte dot product, so digits {0, 1, 2}
//! are multiplied against activations with `i16x8.extmul_*_i8x16`, summed in
//! 16-bit lanes for the whole block (at most 20 · 254 per lane), and widened
//! to 32 bits once per block.

use core::arch::wasm32::*;

use crate::pack::{BLOCK_BYTES, Layout, RowsView};

#[inline(always)]
fn load(p: &[u8], at: usize) -> v128 {
    let s = &p[at..at + 16];
    // SAFETY: the slice above guarantees 16 readable bytes.
    unsafe { v128_load(s.as_ptr() as *const v128) }
}

#[inline(always)]
fn load_i8(p: &[i8], at: usize) -> v128 {
    let s = &p[at..at + 16];
    // SAFETY: as above.
    unsafe { v128_load(s.as_ptr() as *const v128) }
}

#[inline(always)]
fn mac(acc: v128, d: v128, x: v128) -> v128 {
    i16x8_add(
        acc,
        i16x8_add(i16x8_extmul_low_i8x16(d, x), i16x8_extmul_high_i8x16(d, x)),
    )
}

#[inline(always)]
fn hsum(v: v128) -> i32 {
    i32x4_extract_lane::<0>(v)
        + i32x4_extract_lane::<1>(v)
        + i32x4_extract_lane::<2>(v)
        + i32x4_extract_lane::<3>(v)
}

pub(crate) fn gemv_simd128(m: RowsView<'_>, x: &[i8], x_sum: i32, out: &mut [i32]) {
    let rb = m.blocks * BLOCK_BYTES;
    let bc = m.layout.block_cols();
    assert!(x.len() >= m.blocks * bc && out.len() >= m.rows);
    for (r, o) in out.iter_mut().enumerate().take(m.rows) {
        let row = &m.data[r * rb..(r + 1) * rb];
        let mut acc = i32x4_splat(0);
        for b in 0..m.blocks {
            let at = b * BLOCK_BYTES;
            let xb = b * bc;
            let mut s = i16x8_splat(0);
            match m.layout {
                Layout::I2 => {
                    let mask = u8x16_splat(3);
                    for h in 0..2 {
                        let w = load(row, at + 16 * h);
                        for k in 0..4u32 {
                            let d = v128_and(u8x16_shr(w, 2 * k), mask);
                            s = mac(s, d, load_i8(x, xb + 32 * k as usize + 16 * h));
                        }
                    }
                }
                Layout::Tq => {
                    let c86 = u8x16_splat(86);
                    let c171 = u8x16_splat(171);
                    let zero = i8x16_splat(0);
                    for h in 0..2 {
                        let mut q = load(row, at + 16 * h);
                        for k in 0..5 {
                            let d = i8x16_sub(i8x16_sub(zero, u8x16_ge(q, c86)), u8x16_ge(q, c171));
                            s = mac(s, d, load_i8(x, xb + 32 * k + 16 * h));
                            q = i8x16_add(q, i8x16_add(q, q));
                        }
                    }
                }
            }
            acc = i32x4_add(acc, i32x4_extadd_pairwise_i16x8(s));
        }
        *o = hsum(acc) - x_sum;
    }
}
