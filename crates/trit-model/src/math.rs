//! Platform-independent float math.
//!
//! `f32::exp`, `sin` and `cos` call the platform's maths library, and native
//! builds and WebAssembly use different ones, which disagree in the last bit
//! now and then. That is enough for a sampled story to drift apart after a
//! few dozen tokens. Everything in the forward pass and the sampler is either
//! exact integer work or basic IEEE arithmetic, so replacing these few calls
//! makes a seed produce the same story in the CLI and in the browser.

/// e^x from basic arithmetic only: `x = k ln2 + r`, a degree-7 polynomial
/// for e^r on |r| <= ln2 / 2, then scaling by 2^k through the exponent bits.
/// Relative error is a few ulp; results below the smallest normal f32 flush to 0.
#[inline]
pub fn exp(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_145_75; // ln 2 split so that k * LN2_HI is exact
    const LN2_LO: f32 = 1.428_606_8e-6;
    if x.is_nan() {
        return x;
    }
    if x > 88.72 {
        return f32::INFINITY;
    }
    if x < -87.33 {
        return 0.0;
    }
    let k = (x * LOG2E).round_ties_even();
    let r = (x - k * LN2_HI) - k * LN2_LO;
    // Taylor series to r^7 in Horner form.
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0
                    + r * (1.0 / 24.0
                        + r * (1.0 / 120.0 + r * (1.0 / 720.0 + r * (1.0 / 5040.0)))))));
    let k = k as i32;
    if k > 127 {
        // e^x just below f32::MAX: 2^128 itself is not representable.
        return p * f32::from_bits(((k - 1 + 127) as u32) << 23) * 2.0;
    }
    p * f32::from_bits(((k + 127) as u32) << 23)
}

/// cos and sin of an f32 angle, computed in f64 and rounded once. Maths
/// libraries disagree by at most an ulp of f64, 29 bits below what survives
/// the rounding to f32, so the f32 results only differ if an f64 result sits
/// right at a rounding boundary. The browser test checks the outcome: the
/// same seed must write the same story as the CLI.
pub fn cos_sin(a: f32) -> (f32, f32) {
    let a = a as f64;
    (a.cos() as f32, a.sin() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_is_accurate() {
        let mut worst = 0f64;
        let mut x = -87.0f32;
        while x < 88.0 {
            let want = (x as f64).exp();
            let got = exp(x) as f64;
            worst = worst.max(((got - want) / want).abs());
            x += 0.0137;
        }
        assert!(worst < 5e-7, "worst relative error {worst:e}");
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(-100.0), 0.0);
        assert_eq!(exp(100.0), f32::INFINITY);
        assert!(exp(f32::NAN).is_nan());
        assert_eq!(exp(f32::NEG_INFINITY), 0.0);
    }
}
