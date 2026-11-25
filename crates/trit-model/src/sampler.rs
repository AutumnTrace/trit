//! Temperature and nucleus (top-p) sampling with a seeded generator.

/// SplitMix64: small, fast, and identical on every platform, so a seed
/// reproduces a story exactly in the CLI and in the browser.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[derive(Clone, Debug)]
pub struct Sampler {
    /// 0 means greedy.
    pub temperature: f32,
    /// Keep the smallest set of tokens whose probability reaches this; 1 keeps all.
    pub top_p: f32,
    pub rng: Rng,
    probs: Vec<(f32, u32)>,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, seed: u64) -> Self {
        Self {
            temperature,
            top_p: top_p.clamp(0.0, 1.0),
            rng: Rng::new(seed),
            probs: Vec::new(),
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.temperature <= 0.0 {
            return argmax(logits);
        }
        let inv_t = 1.0 / self.temperature;
        let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        self.probs.clear();
        let mut sum = 0.0;
        for (i, &l) in logits.iter().enumerate() {
            let p = crate::math::exp((l - m) * inv_t);
            sum += p;
            self.probs.push((p, i as u32));
        }
        let probs = &mut self.probs;
        let mut total = sum;
        if self.top_p < 1.0 {
            // Most of the mass is in a few dozen tokens: sort only the candidates
            // above a cutoff that cannot exclude any of the nucleus.
            let cutoff = (1.0 - self.top_p) / (probs.len() as f32 - 1.0).max(1.0) * sum;
            probs.retain(|&(p, _)| p >= cutoff);
            probs.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let target = self.top_p * sum;
            let mut acc = 0.0;
            let mut keep = probs.len();
            for (i, &(p, _)) in probs.iter().enumerate() {
                acc += p;
                if acc >= target {
                    keep = i + 1;
                    break;
                }
            }
            probs.truncate(keep);
            total = probs.iter().map(|p| p.0).sum();
        }
        let r = self.rng.next_f32() * total;
        let mut acc = 0.0;
        for &(p, i) in probs.iter() {
            acc += p;
            if r < acc {
                return i;
            }
        }
        probs.last().map_or(0, |p| p.1)
    }
}

pub fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_the_max() {
        let mut s = Sampler::new(0.0, 1.0, 1);
        assert_eq!(s.sample(&[0.1, 3.0, -2.0, 2.9]), 1);
    }

    #[test]
    fn top_p_restricts_to_the_nucleus() {
        // Probabilities ~ [0.64, 0.24, 0.09, 0.03]: top_p 0.8 keeps the first two.
        let logits = [3.0f32, 2.0, 1.0, 0.0];
        let mut s = Sampler::new(1.0, 0.8, 7);
        let mut seen = [0u32; 4];
        for _ in 0..4000 {
            seen[s.sample(&logits) as usize] += 1;
        }
        assert!(seen[0] > 0 && seen[1] > 0);
        assert_eq!(seen[2] + seen[3], 0, "{seen:?}");
        // 0.64 / 0.88 of the kept mass
        let frac = seen[0] as f32 / 4000.0;
        assert!((frac - 0.731).abs() < 0.03, "{frac}");
    }

    #[test]
    fn same_seed_same_samples() {
        let logits: Vec<f32> = (0..100).map(|i| ((i * 37) % 11) as f32 * 0.3).collect();
        let a: Vec<u32> = {
            let mut s = Sampler::new(0.9, 0.95, 42);
            (0..50).map(|_| s.sample(&logits)).collect()
        };
        let b: Vec<u32> = {
            let mut s = Sampler::new(0.9, 0.95, 42);
            (0..50).map(|_| s.sample(&logits)).collect()
        };
        assert_eq!(a, b);
    }
}
