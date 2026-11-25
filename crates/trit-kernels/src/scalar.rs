//! Portable kernels. These are the reference the SIMD paths are tested against
//! (besides the dense [`crate::reference_gemv`]) and the fallback everywhere.

use crate::pack::{BLOCK_BYTES, Layout, RowsView, tq_digit};

pub(crate) fn gemv(m: RowsView<'_>, x: &[i8], out: &mut [i32]) {
    let rb = m.blocks * BLOCK_BYTES;
    let bc = m.layout.block_cols();
    let tpb = m.layout.trits_per_byte();
    for (r, o) in out.iter_mut().enumerate().take(m.rows) {
        let row = &m.data[r * rb..(r + 1) * rb];
        let mut acc = 0i32;
        for b in 0..m.blocks {
            let blk = &row[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
            let xb = &x[b * bc..(b + 1) * bc];
            for k in 0..tpb {
                let xs = &xb[k * BLOCK_BYTES..(k + 1) * BLOCK_BYTES];
                for (&byte, &xv) in blk.iter().zip(xs) {
                    let t = match m.layout {
                        Layout::I2 => ((byte >> (2 * k)) & 3) as i8 - 1,
                        Layout::Tq => tq_digit(byte, k),
                    };
                    acc += t as i32 * xv as i32;
                }
            }
        }
        *o = acc;
    }
}
