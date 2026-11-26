//! The shipped model is a version 2 file: export dropped the feed-forward
//! units whose `w_up` row is all zero. Putting dead units back must not change
//! a single bit of the output, and the loader must refuse widths a file can't
//! have.

use std::path::PathBuf;
use trit_kernels::{Isa, Layout, TernaryMatrix};
use trit_model::model::BitLinear;
use trit_model::{Error, Model};

fn model_bytes() -> Vec<u8> {
    let path = std::env::var_os("TRIT_MODEL").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/tinystories-15m.trit"),
        PathBuf::from,
    );
    std::fs::read(path).expect("model file")
}

/// Pads every layer back to the trained width with dead units: zero `w_up`
/// rows, placed first so every live unit moves. Their sub-norm weights and
/// `w_down` columns are deliberately not zero; an activation of 0 has to hide
/// them.
fn widen(model: &mut Model) {
    let (d, f) = (model.config.dim, model.config.ffn_dim);
    for l in &mut model.layers {
        let h = l.w_up.w.rows();
        let extra = f - h;
        let layout = l.w_up.w.layout();
        let mut up = vec![0i8; extra * d];
        up.extend(l.w_up.w.to_trits());
        let live_down = l.w_down.w.to_trits();
        let mut down = Vec::with_capacity(d * f);
        for r in 0..d {
            down.extend((0..extra).map(|c| [1i8, -1, 0][(r + c) % 3]));
            down.extend_from_slice(&live_down[r * h..(r + 1) * h]);
        }
        let mut sub = vec![1.5f32; extra];
        sub.extend_from_slice(&l.ffn_sub_norm);
        l.w_up = BitLinear {
            w: TernaryMatrix::from_trits(layout, f, d, &up),
            gamma: l.w_up.gamma,
        };
        l.w_down = BitLinear {
            w: TernaryMatrix::from_trits(layout, d, f, &down),
            gamma: l.w_down.gamma,
        };
        l.ffn_sub_norm = sub;
    }
}

fn logits(model: &Model, tokens: &[u32]) -> Vec<f32> {
    let mut state = model.new_state();
    tokens
        .iter()
        .flat_map(|&t| model.forward(&mut state, t).to_vec())
        .collect()
}

#[test]
fn dead_units_change_no_bit_of_the_output() {
    let bytes = model_bytes();
    let tokens = [0u32, 400, 271, 262, 1200, 13, 287, 1033];
    for layout in Layout::ALL {
        let mut pruned = Model::load(&bytes, layout).expect("valid model");
        let pruned_units: usize = pruned
            .layers
            .iter()
            .map(|l| pruned.config.ffn_dim - l.w_up.w.rows())
            .sum();
        let mut full = Model::load(&bytes, layout).expect("valid model");
        widen(&mut full);
        assert!(
            full.layers
                .iter()
                .all(|l| l.w_up.w.rows() == full.config.ffn_dim)
        );
        for isa in Isa::available() {
            pruned.isa = isa;
            full.isa = isa;
            assert!(
                logits(&pruned, &tokens) == logits(&full, &tokens),
                "{layout:?} {isa:?}: {pruned_units} pruned units changed the logits"
            );
        }
    }
}

fn load_err(bytes: &[u8]) -> Error {
    match Model::load(bytes, Layout::I2) {
        Ok(_) => panic!("expected the file to be refused"),
        Err(e) => e,
    }
}

#[test]
fn loader_refuses_widths_the_version_does_not_allow() {
    let bytes = model_bytes();
    let model = Model::load(&bytes, Layout::I2).expect("valid model");
    let c = model.config;
    let first_pruned = model
        .layers
        .iter()
        .position(|l| l.w_up.w.rows() < c.ffn_dim)
        .expect("the shipped model has pruned layers");

    // Version 1 files have every layer at the full width.
    let mut v1 = bytes.clone();
    v1[4..8].copy_from_slice(&1u32.to_le_bytes());
    assert_eq!(
        load_err(&v1),
        Error::BadTensor(format!("layers.{first_pruned}.w_up"))
    );

    // No layer may be wider than the header's (trained) width.
    let mut narrow = bytes.clone();
    narrow[24..28].copy_from_slice(&1400u32.to_le_bytes());
    assert_eq!(load_err(&narrow), Error::BadTensor("layers.0.w_up".into()));

    let mut v3 = bytes;
    v3[4..8].copy_from_slice(&3u32.to_le_bytes());
    assert_eq!(load_err(&v3), Error::UnsupportedVersion(3));
}
