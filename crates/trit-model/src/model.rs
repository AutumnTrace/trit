//! The forward pass, one token at a time with a KV cache. Mirrors
//! train/trit_train/reference.py.

use trit_kernels::{Activations, Isa, Layout, TernaryMatrix, base3, dense, gemv};

use crate::format::{Config, DType, Error, File};
use crate::tokenizer::Tokenizer;

/// A ternary linear layer: `y = (W_t · round(x s)) · γ / s`.
pub struct BitLinear {
    pub w: TernaryMatrix,
    pub gamma: f32,
}

impl BitLinear {
    fn load(
        f: &File<'_>,
        name: &str,
        rows: usize,
        cols: usize,
        layout: Layout,
    ) -> Result<Self, Error> {
        let t = f.tensor(name)?;
        let rb = base3::row_bytes(cols);
        if t.dtype != DType::Ternary || t.shape != [rows, cols] || t.data.len() != rows * rb {
            return Err(Error::BadTensor(name.into()));
        }
        let mut trits = Vec::with_capacity(rows * cols);
        for row in t.data.chunks_exact(rb) {
            base3::unpack_row(row, cols, &mut trits).map_err(|_| Error::BadTensor(name.into()))?;
        }
        Ok(Self {
            w: TernaryMatrix::from_trits(layout, rows, cols, &trits),
            gamma: t.scale,
        })
    }

    /// Multiply already-quantized activations; `acc` is scratch of at least `rows` entries.
    #[inline]
    fn apply(&self, isa: Isa, x: &Activations, acc: &mut [i32], out: &mut [f32]) {
        gemv(isa, &self.w, x, acc);
        let k = self.gamma / x.scale;
        for (o, &a) in out.iter_mut().zip(acc.iter()) {
            *o = a as f32 * k;
        }
    }
}

pub struct Layer {
    pub attn_norm: Vec<f32>,
    pub wq: BitLinear,
    pub wk: BitLinear,
    pub wv: BitLinear,
    pub attn_sub_norm: Vec<f32>,
    pub wo: BitLinear,
    pub ffn_norm: Vec<f32>,
    pub w_up: BitLinear,
    pub ffn_sub_norm: Vec<f32>,
    pub w_down: BitLinear,
}

pub struct Model {
    pub config: Config,
    pub tokenizer: Tokenizer,
    pub layers: Vec<Layer>,
    pub embed: Vec<i8>,
    pub embed_scale: Vec<f32>,
    pub norm: Vec<f32>,
    pub isa: Isa,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

impl Model {
    /// Parse a `.trit` file and repack its ternary weights into `layout`.
    pub fn load(bytes: &[u8], layout: Layout) -> Result<Self, Error> {
        let f = File::parse(bytes)?;
        let c = f.config;
        if c.n_heads == 0 || c.dim % c.n_heads != 0 || c.head_dim() % 2 != 0 {
            return Err(Error::BadTensor("config".into()));
        }
        let d = c.dim;
        let mut layers = Vec::with_capacity(c.n_layers);
        for i in 0..c.n_layers {
            let p = |n: &str| format!("layers.{i}.{n}");
            let lin = |n: &str, rows, cols| BitLinear::load(&f, &p(n), rows, cols, layout);
            // This layer's feed-forward width. Version 1 files always use the
            // full width; version 2 files may have pruned dead units.
            let h = f.tensor(&p("w_up"))?.shape.first().copied().unwrap_or(0);
            let full = h == c.ffn_dim;
            if !(full || (f.version >= 2 && (1..c.ffn_dim).contains(&h))) {
                return Err(Error::BadTensor(p("w_up")));
            }
            layers.push(Layer {
                attn_norm: f.f32s(&p("attn_norm"), d)?,
                wq: lin("wq", d, d)?,
                wk: lin("wk", d, d)?,
                wv: lin("wv", d, d)?,
                attn_sub_norm: f.f32s(&p("attn_sub_norm"), d)?,
                wo: lin("wo", d, d)?,
                ffn_norm: f.f32s(&p("ffn_norm"), d)?,
                w_up: lin("w_up", h, d)?,
                ffn_sub_norm: f.f32s(&p("ffn_sub_norm"), h)?,
                w_down: lin("w_down", d, h)?,
            });
        }
        let e = f.tensor("embed.weight")?;
        if e.dtype != DType::I8 || e.shape != [c.vocab_size, d] {
            return Err(Error::BadTensor("embed.weight".into()));
        }
        let embed = e.data.iter().map(|&b| b as i8).collect();
        let embed_scale = f.f32s("embed.scale", c.vocab_size)?;
        let norm = f.f32s("norm", d)?;

        // RoPE tables. The angle is formed in f32 as in PyTorch; the
        // transcendental parts go through f64 so every platform agrees (see math.rs).
        let half = c.head_dim() / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| {
                (1.0 / (c.rope_theta as f64).powf((2 * i) as f64 / c.head_dim() as f64)) as f32
            })
            .collect();
        let mut rope_cos = Vec::with_capacity(c.max_seq_len * half);
        let mut rope_sin = Vec::with_capacity(c.max_seq_len * half);
        for t in 0..c.max_seq_len {
            for &f in &inv {
                let (cos, sin) = crate::math::cos_sin(t as f32 * f);
                rope_cos.push(cos);
                rope_sin.push(sin);
            }
        }
        Ok(Self {
            config: c,
            tokenizer: Tokenizer::from_file(&f),
            layers,
            embed,
            embed_scale,
            norm,
            isa: Isa::detect(),
            rope_cos,
            rope_sin,
        })
    }

    /// Bytes of weights as stored in memory for inference.
    pub fn weight_bytes(&self) -> usize {
        let lin: usize = self
            .layers
            .iter()
            .flat_map(|l| [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_up, &l.w_down])
            .map(|b| b.w.data().len())
            .sum();
        lin + self.embed.len() + 4 * self.embed_scale.len()
    }

    pub fn ternary_params(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|l| [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_up, &l.w_down])
            .map(|b| b.w.rows() * b.w.cols())
            .sum()
    }

    pub fn new_state(&self) -> State {
        State::new(&self.config)
    }

    /// Run one token at position `state.pos` and return the logits for the next one.
    pub fn forward<'s>(&self, state: &'s mut State, token: u32) -> &'s [f32] {
        let c = &self.config;
        let (d, hd) = (c.dim, c.head_dim());
        let pos = state.pos;
        assert!(
            pos < c.max_seq_len,
            "context is full ({} tokens)",
            c.max_seq_len
        );
        let tok = token as usize;
        assert!(
            tok < c.vocab_size,
            "token {token} is outside the vocabulary"
        );
        let s = self.embed_scale[tok];
        for (x, &e) in state.x.iter_mut().zip(&self.embed[tok * d..(tok + 1) * d]) {
            *x = e as f32 * s;
        }
        let isa = self.isa;
        let half = hd / 2;
        let cos = &self.rope_cos[pos * half..(pos + 1) * half];
        let sin = &self.rope_sin[pos * half..(pos + 1) * half];
        let kv = c.max_seq_len * d;
        let scale = 1.0 / (hd as f32).sqrt();

        for (li, l) in self.layers.iter().enumerate() {
            let St {
                x,
                h,
                q,
                k,
                v,
                att,
                scores,
                u,
                act,
                acc,
                k_cache,
                v_cache,
                ..
            } = state.split();
            rmsnorm(h, x, &l.attn_norm, c.norm_eps);
            act.quantize_from(h);
            l.wq.apply(isa, act, acc, q);
            l.wk.apply(isa, act, acc, k);
            l.wv.apply(isa, act, acc, v);
            for head in 0..c.n_heads {
                rope(&mut q[head * hd..(head + 1) * hd], cos, sin);
                rope(&mut k[head * hd..(head + 1) * hd], cos, sin);
            }
            let kc = &mut k_cache[li * kv..(li + 1) * kv];
            let vc = &mut v_cache[li * kv..(li + 1) * kv];
            kc[pos * d..(pos + 1) * d].copy_from_slice(k);
            vc[pos * d..(pos + 1) * d].copy_from_slice(v);

            for head in 0..c.n_heads {
                let o = head * hd;
                let qh = &q[o..o + hd];
                let sc = &mut scores[..=pos];
                for (t, s) in sc.iter_mut().enumerate() {
                    *s = dense::dot_f32(qh, &kc[t * d + o..t * d + o + hd]) * scale;
                }
                softmax(sc);
                let out = &mut att[o..o + hd];
                out.fill(0.0);
                for (t, &p) in sc.iter().enumerate() {
                    for (a, &vv) in out.iter_mut().zip(&vc[t * d + o..t * d + o + hd]) {
                        *a += p * vv;
                    }
                }
            }
            rmsnorm(h, att, &l.attn_sub_norm, c.norm_eps);
            act.quantize_from(h);
            l.wo.apply(isa, act, acc, q);
            for (x, &y) in x.iter_mut().zip(q.iter()) {
                *x += y;
            }

            rmsnorm(h, x, &l.ffn_norm, c.norm_eps);
            act.quantize_from(h);
            let u = &mut u[..l.w_up.w.rows()];
            l.w_up.apply(isa, act, acc, u);
            for v in u.iter_mut() {
                let r = v.max(0.0);
                *v = r * r;
            }
            // Pruned units are zeros that aren't stored, so the mean square
            // still divides by the trained width. Adding 0.0 to a sum changes
            // nothing, so this is bit-identical to the unpruned layer.
            rmsnorm_inplace(u, &l.ffn_sub_norm, c.norm_eps, c.ffn_dim);
            act.quantize_from(u);
            l.w_down.apply(isa, act, acc, q);
            for (x, &y) in x.iter_mut().zip(q.iter()) {
                *x += y;
            }
        }
        let St { x, h, logits, .. } = state.split();
        rmsnorm(h, x, &self.norm, c.norm_eps);
        dense::gemv_f32_i8rows(&self.embed, &self.embed_scale, h, logits);
        state.pos += 1;
        &state.logits
    }
}

/// Per-sequence buffers: the KV cache and scratch space.
pub struct State {
    pub pos: usize,
    x: Vec<f32>,
    h: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    scores: Vec<f32>,
    u: Vec<f32>,
    logits: Vec<f32>,
    act: Activations,
    acc: Vec<i32>,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
}

/// Disjoint mutable borrows of every buffer at once.
struct St<'a> {
    x: &'a mut [f32],
    h: &'a mut [f32],
    q: &'a mut [f32],
    k: &'a mut [f32],
    v: &'a mut [f32],
    att: &'a mut [f32],
    scores: &'a mut [f32],
    u: &'a mut [f32],
    logits: &'a mut [f32],
    act: &'a mut Activations,
    acc: &'a mut [i32],
    k_cache: &'a mut [f32],
    v_cache: &'a mut [f32],
}

impl State {
    fn new(c: &Config) -> Self {
        let (d, f) = (c.dim, c.ffn_dim);
        Self {
            pos: 0,
            x: vec![0.0; d],
            h: vec![0.0; d],
            q: vec![0.0; d],
            k: vec![0.0; d],
            v: vec![0.0; d],
            att: vec![0.0; d],
            scores: vec![0.0; c.max_seq_len],
            u: vec![0.0; f],
            logits: vec![0.0; c.vocab_size],
            act: Activations::default(),
            acc: vec![0; d.max(f)],
            k_cache: vec![0.0; c.n_layers * c.max_seq_len * d],
            v_cache: vec![0.0; c.n_layers * c.max_seq_len * d],
        }
    }

    fn split(&mut self) -> St<'_> {
        St {
            x: &mut self.x,
            h: &mut self.h,
            q: &mut self.q,
            k: &mut self.k,
            v: &mut self.v,
            att: &mut self.att,
            scores: &mut self.scores,
            u: &mut self.u,
            logits: &mut self.logits,
            act: &mut self.act,
            acc: &mut self.acc,
            k_cache: &mut self.k_cache,
            v_cache: &mut self.v_cache,
        }
    }

    /// Forget the sequence so the next token is position 0.
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
}

/// `out = x / sqrt(mean(x²) + eps) · w`.
fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) {
        *o = v * r * g;
    }
}

/// `rmsnorm` in place, with the mean taken over `n` values: `x` plus
/// `n - x.len()` zeros.
fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32, n: usize) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let r = 1.0 / (ms + eps).sqrt();
    for (v, &g) in x.iter_mut().zip(w) {
        *v = *v * r * g;
    }
}

/// Rotate-half RoPE on one head: pairs `(i, i + hd/2)`.
fn rope(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = x.len() / 2;
    let (a, b) = x.split_at_mut(half);
    for i in 0..half {
        let (x1, x2) = (a[i], b[i]);
        a[i] = x1 * cos[i] - x2 * sin[i];
        b[i] = x1 * sin[i] + x2 * cos[i];
    }
}

pub(crate) fn softmax(x: &mut [f32]) {
    let m = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = crate::math::exp(*v - m);
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}
