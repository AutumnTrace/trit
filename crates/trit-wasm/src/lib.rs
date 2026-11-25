//! WebAssembly surface for the browser demo: a handful of C-ABI exports, no
//! bindings generator. The page copies the model file into memory obtained
//! from `trit_alloc`, calls `trit_load`, then `trit_start` and `trit_next` in a
//! loop, reading each token's bytes from `trit_out_ptr`.
//!
//! Everything lives in one thread-local; wasm32-unknown-unknown is single
//! threaded and the page runs this inside a worker.

#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;

use trit_kernels::Layout;
use trit_model::{Generator, Model, Sampler};

struct App {
    model: &'static Model,
    story: Option<Generator<'static>>,
    out: Vec<u8>,
    trits: Vec<i8>,
}

thread_local! {
    static APP: RefCell<Option<App>> = const { RefCell::new(None) };
}

fn with<R>(f: impl FnOnce(&mut App) -> R) -> Option<R> {
    APP.with(|a| a.borrow_mut().as_mut().map(f))
}

#[unsafe(no_mangle)]
pub extern "C" fn trit_alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// # Safety
/// `ptr` must come from `trit_alloc(len)` and not have been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn trit_free(ptr: *mut u8, len: usize) {
    // SAFETY: per the contract above.
    drop(unsafe { Vec::from_raw_parts(ptr, 0, len) });
}

/// Load a `.trit` file from `ptr..ptr+len` (memory from `trit_alloc`; it is
/// freed here). Returns 0 on success, 1 if the file is invalid.
///
/// # Safety
/// `ptr` must point at `len` initialised bytes allocated by `trit_alloc(len)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn trit_load(ptr: *mut u8, len: usize) -> u32 {
    // SAFETY: per the contract above.
    let bytes = unsafe { Vec::from_raw_parts(ptr, len, len) };
    let Ok(model) = Model::load(&bytes, Layout::I2) else {
        return 1;
    };
    let model: &'static Model = Box::leak(Box::new(model));
    APP.with(|a| {
        *a.borrow_mut() = Some(App {
            model,
            story: None,
            out: Vec::with_capacity(64),
            trits: Vec::new(),
        })
    });
    0
}

/// Model shape: 0 vocab, 1 dim, 2 layers, 3 heads, 4 ffn, 5 context,
/// 6 ternary weight count, 7 in-memory weight bytes, 8 kernel (1 = simd128).
#[unsafe(no_mangle)]
pub extern "C" fn trit_info(what: u32) -> u32 {
    with(|a| {
        let c = a.model.config;
        match what {
            0 => c.vocab_size as u32,
            1 => c.dim as u32,
            2 => c.n_layers as u32,
            3 => c.n_heads as u32,
            4 => c.ffn_dim as u32,
            5 => c.max_seq_len as u32,
            6 => a.model.ternary_params() as u32,
            7 => a.model.weight_bytes() as u32,
            8 => (a.model.isa == trit_kernels::Isa::Simd128) as u32,
            _ => 0,
        }
    })
    .unwrap_or(0)
}

/// Begin a story with the UTF-8 prompt at `ptr..ptr+len`. Returns the number
/// of prompt tokens.
///
/// # Safety
/// `ptr` must point at `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn trit_start(
    ptr: *const u8,
    len: usize,
    temperature: f32,
    top_p: f32,
    seed: u32,
) -> u32 {
    // SAFETY: per the contract above.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    let prompt = String::from_utf8_lossy(bytes).into_owned();
    with(|a| {
        let mut g = Generator::new(a.model, Sampler::new(temperature, top_p, seed as u64));
        let n = g.prompt(&prompt);
        a.story = Some(g);
        n as u32
    })
    .unwrap_or(0)
}

/// Produce one token. Returns its byte length (bytes at `trit_out_ptr`), or
/// -1 at the end of the story.
#[unsafe(no_mangle)]
pub extern "C" fn trit_next() -> i32 {
    with(|a| {
        let Some(g) = a.story.as_mut() else { return -1 };
        match g.next_token() {
            Some(t) => {
                a.out.clear();
                a.out.extend_from_slice(a.model.tokenizer.token_bytes(t));
                a.out.len() as i32
            }
            None => -1,
        }
    })
    .unwrap_or(-1)
}

#[unsafe(no_mangle)]
pub extern "C" fn trit_out_ptr() -> *const u8 {
    with(|a| a.out.as_ptr()).unwrap_or(std::ptr::null())
}

/// Context position of the running story.
#[unsafe(no_mangle)]
pub extern "C" fn trit_pos() -> u32 {
    with(|a| a.story.as_ref().map_or(0, |g| g.state.pos as u32)).unwrap_or(0)
}

fn matrix(m: &Model, layer: u32, which: u32) -> Option<&trit_model::model::BitLinear> {
    let l = m.layers.get(layer as usize)?;
    Some(match which {
        0 => &l.wq,
        1 => &l.wk,
        2 => &l.wv,
        3 => &l.wo,
        4 => &l.w_up,
        5 => &l.w_down,
        _ => return None,
    })
}

/// Unpack one weight matrix as row-major int8 trits; returns a pointer to
/// `rows * cols` bytes valid until the next call. `which` indexes
/// wq, wk, wv, wo, w_up, w_down.
#[unsafe(no_mangle)]
pub extern "C" fn trit_matrix(layer: u32, which: u32) -> *const i8 {
    with(|a| {
        let Some(b) = matrix(a.model, layer, which) else {
            return std::ptr::null();
        };
        a.trits = b.w.to_trits();
        a.trits.as_ptr()
    })
    .unwrap_or(std::ptr::null())
}

/// Rows (0), columns (1) of a matrix, or its scale γ as f32 bits (2).
#[unsafe(no_mangle)]
pub extern "C" fn trit_matrix_info(layer: u32, which: u32, what: u32) -> u32 {
    with(|a| {
        matrix(a.model, layer, which).map_or(0, |b| match what {
            0 => b.w.rows() as u32,
            1 => b.w.cols() as u32,
            _ => b.gamma.to_bits(),
        })
    })
    .unwrap_or(0)
}
