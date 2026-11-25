//! The Rust engine against the PyTorch reference (train/trit_train/reference.py):
//! same tokens for the same text, same logits for the same tokens.
//!
//! Uses the shipped model and fixtures by default; set TRIT_MODEL and
//! TRIT_FIXTURES to check another export.

use std::path::PathBuf;
use trit_kernels::{Isa, Layout};
use trit_model::{Model, Sampler};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn model_path() -> PathBuf {
    std::env::var_os("TRIT_MODEL")
        .map_or_else(|| root().join("models/tinystories-15m.trit"), PathBuf::from)
}

fn fixture(name: &str) -> Vec<u8> {
    let dir = std::env::var_os("TRIT_FIXTURES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"),
        PathBuf::from,
    );
    std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn u32s(b: &[u8]) -> impl Iterator<Item = u32> + '_ {
    b.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c))
}

fn load(layout: Layout) -> Model {
    let bytes = std::fs::read(model_path()).expect("model file");
    Model::load(&bytes, layout).expect("valid model")
}

#[test]
fn tokenizer_matches_huggingface() {
    let model = load(Layout::I2);
    let buf = fixture("tokenizer.bin");
    let mut pos = 4;
    let n = u32s(&buf[..4]).next().unwrap();
    for _ in 0..n {
        let len = u32s(&buf[pos..pos + 4]).next().unwrap() as usize;
        let text = std::str::from_utf8(&buf[pos + 4..pos + 4 + len]).unwrap();
        pos += 4 + len;
        let k = u32s(&buf[pos..pos + 4]).next().unwrap() as usize;
        let want: Vec<u32> = u32s(&buf[pos + 4..pos + 4 + 4 * k]).collect();
        pos += 4 + 4 * k;
        assert_eq!(model.tokenizer.encode(text), want, "{text:?}");
        assert_eq!(model.tokenizer.decode(&want), text);
    }
}

#[test]
fn logits_match_pytorch() {
    let buf = fixture("logits.bin");
    let mut it = u32s(&buf);
    let (n, vocab) = (it.next().unwrap() as usize, it.next().unwrap() as usize);
    let tokens: Vec<u32> = it.take(n).collect();
    let want: Vec<f32> = buf[8 + 4 * n..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(want.len(), n * vocab);

    let model = load(Layout::I2);
    let mut state = model.new_state();
    let mut worst = 0f32;
    for (i, &t) in tokens.iter().enumerate() {
        let got = model.forward(&mut state, t);
        let w = &want[i * vocab..(i + 1) * vocab];
        let scale = w.iter().fold(0f32, |m, v| m.max(v.abs()));
        let diff = got
            .iter()
            .zip(w)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        worst = worst.max(diff / scale);
        assert_eq!(
            trit_model::sampler::argmax(got),
            trit_model::sampler::argmax(w),
            "argmax differs at position {i}"
        );
    }
    eprintln!("worst relative logit difference: {worst:e}");
    assert!(
        worst < 1e-5,
        "logits differ from PyTorch by up to {worst:e} of their range"
    );
}

#[test]
fn every_kernel_and_layout_gives_identical_logits() {
    // The integer GEMV is exact, so the floating-point work around it sees
    // identical inputs whatever kernel or layout computed it.
    let tokens = [0u32, 400, 271, 262, 1200, 13];
    let mut reference: Option<Vec<f32>> = None;
    for layout in Layout::ALL {
        let mut model = load(layout);
        for isa in Isa::available() {
            model.isa = isa;
            let mut state = model.new_state();
            let mut all = Vec::new();
            for &t in &tokens {
                all.extend_from_slice(model.forward(&mut state, t));
            }
            match &reference {
                None => reference = Some(all),
                Some(r) => assert!(r == &all, "{layout:?} {isa:?} differs"),
            }
        }
    }
}

#[test]
fn generation_is_reproducible_for_a_seed() {
    let model = load(Layout::I2);
    let story = |seed| {
        let mut g = trit_model::Generator::new(&model, Sampler::new(0.8, 0.9, seed));
        g.prompt("Once upon a time");
        (0..40).map_while(|_| g.next_text()).collect::<String>()
    };
    let a = story(5);
    assert_eq!(a, story(5));
    assert!(!a.trim().is_empty());
}
