//! Every kernel available on the test machine must agree exactly with the
//! dense reference, for arbitrary shapes (including ragged tails and odd
//! block counts), weights and activations.
//!
//! The workspace has no dependencies outside std, so instead of a property
//! testing crate this file draws thousands of seeded random cases and prints
//! the seed of any failure so it can be replayed.

use trit_kernels::{Activations, Isa, Layout, TernaryMatrix, gemv, gemv_parallel, reference_gemv};

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn trits(&mut self, n: usize, zero_bias: usize) -> Vec<i8> {
        (0..n)
            .map(|_| match self.below(3 + zero_bias) {
                0 => -1,
                1 => 1,
                _ => 0,
            })
            .collect()
    }
    fn floats(&mut self, n: usize, mag: f32) -> Vec<f32> {
        (0..n)
            .map(|_| (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 * mag - mag)
            .collect()
    }
}

fn check(seed: u64, layout: Layout, rows: usize, cols: usize, rng: &mut Rng) {
    let bias = rng.below(4);
    let trits = rng.trits(rows * cols, bias);
    let mag = 0.1 + rng.below(8) as f32;
    let x = rng.floats(cols, mag);
    let m = TernaryMatrix::from_trits(layout, rows, cols, &trits);
    assert_eq!(m.to_trits(), trits, "seed {seed}: pack/unpack");
    let a = Activations::quantize(&x);
    let want = reference_gemv(&trits, rows, cols, &a.q[..cols]);
    for isa in Isa::available() {
        let mut out = vec![0i32; rows];
        gemv(isa, &m, &a, &mut out);
        assert_eq!(out, want, "seed {seed}: {isa:?} {layout:?} {rows}x{cols}");
    }
    let mut out = vec![0i32; rows];
    gemv_parallel(Isa::detect(), &m, &a, &mut out, 1 + rng.below(4));
    assert_eq!(out, want, "seed {seed}: parallel {layout:?} {rows}x{cols}");
}

#[test]
fn random_shapes_match_reference() {
    for seed in 0..1500u64 {
        let mut rng = Rng(seed);
        let layout = Layout::ALL[rng.below(2)];
        let rows = 1 + rng.below(12);
        let cols = 1 + rng.below(800);
        check(seed, layout, rows, cols, &mut rng);
    }
}

#[test]
fn block_boundaries_match_reference() {
    // One below, at and above every block edge up to five blocks of either layout.
    let mut edges = vec![1];
    for bc in [128, 160] {
        for b in 1..=5 {
            edges.extend([b * bc - 1, b * bc, b * bc + 1]);
        }
    }
    let mut rng = Rng(42);
    for layout in Layout::ALL {
        for &cols in &edges {
            check(cols as u64, layout, 3, cols, &mut rng);
        }
    }
}

#[test]
fn relayout_preserves_weights() {
    let mut rng = Rng(7);
    for _ in 0..200 {
        let (rows, cols) = (1 + rng.below(6), 1 + rng.below(500));
        let trits = rng.trits(rows * cols, 1);
        for from in Layout::ALL {
            let m = TernaryMatrix::from_trits(from, rows, cols, &trits);
            for to in Layout::ALL {
                assert_eq!(m.to_layout(to).to_trits(), trits);
            }
        }
    }
}

#[test]
fn extreme_activations_do_not_overflow() {
    // All +1 weights against all +127 activations, and all -1 against -127,
    // across many blocks: the widest sums the 16-bit stages ever see.
    let cols = 160 * 32;
    for layout in Layout::ALL {
        for (t, v) in [(1i8, 4.0f32), (-1, -4.0), (1, -4.0)] {
            let m = TernaryMatrix::from_trits(layout, 2, cols, &vec![t; 2 * cols]);
            let a = Activations::quantize(&vec![v; cols]);
            let want = t as i32 * a.q[0] as i32 * cols as i32;
            for isa in Isa::available() {
                let mut out = [0i32; 2];
                gemv(isa, &m, &a, &mut out);
                assert_eq!(out, [want, want], "{isa:?} {layout:?}");
            }
        }
    }
}

#[test]
fn quantization_matches_torch_rounding() {
    // torch.round is round-half-to-even: 0.5 * 127 = 63.5 -> 64, 0.25 * 127 = 31.75 -> 32.
    let a = Activations::quantize(&[0.5, -1.0, 0.25, 0.0]);
    assert_eq!(&a.q[..4], &[64, -127, 32, 0]);
    assert_eq!(a.sum, 64 - 127 + 32);
    // With max|x| = 127 the scale is exactly 1, so 2.5 is a true tie and rounds to 2.
    let a = Activations::quantize(&[127.0, 2.5, 3.5]);
    assert_eq!(&a.q[..3], &[127, 2, 4]);
    let a = Activations::quantize(&[0.0; 5]);
    assert!(a.q.iter().all(|&q| q == 0));
}
