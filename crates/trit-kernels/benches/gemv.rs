//! GEMV throughput: ternary layouts against int8 and f32 weights.
//!
//!     cargo bench -p trit-kernels --bench gemv                 # fixed shapes
//!     cargo bench -p trit-kernels --bench gemv -- --sweep      # working-set sweep
//!     cargo bench -p trit-kernels --bench gemv -- --sweep --json bench/kernels.json
//!
//! Fixed shapes are single-threaded. The sweep multiplies one activation
//! vector through a stack of distinct 2048 x 2048 matrices (like the layers of
//! a model) and grows the stack until the weights no longer fit in cache,
//! single-threaded and with one thread per core.
//!
//! No benchmarking crate: each case is calibrated to ~25 ms per sample and the
//! median of 11 samples is reported.

use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};
use trit_kernels::{Activations, Isa, Layout, TernaryMatrix, dense, gemv, gemv_rows};

fn lcg(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 33) as u32
}

/// Median seconds per call of `f`.
fn time(mut f: impl FnMut()) -> f64 {
    let mut iters = 1u64;
    loop {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        if t.elapsed() > Duration::from_millis(25) || iters > 1 << 24 {
            break;
        }
        iters *= 2;
    }
    let mut samples: Vec<f64> = (0..11)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..iters {
                f();
            }
            t.elapsed().as_secs_f64() / iters as f64
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

struct Row {
    group: String,
    kernel: String,
    weights: u64,
    bytes: u64,
    threads: usize,
    secs: f64,
}

impl Row {
    fn print(&self) {
        println!(
            "{:<22} {:<18} {:>3}t {:>10.1} us {:>8.1} Gweight/s {:>8.1} GB/s",
            self.group,
            self.kernel,
            self.threads,
            self.secs * 1e6,
            self.weights as f64 / self.secs / 1e9,
            self.bytes as f64 / self.secs / 1e9,
        );
    }
    fn json(&self) -> String {
        format!(
            r#"{{"group":"{}","kernel":"{}","threads":{},"weights":{},"bytes":{},"seconds":{:e}}}"#,
            self.group, self.kernel, self.threads, self.weights, self.bytes, self.secs
        )
    }
}

const SHAPES: &[(&str, usize, usize)] = &[
    ("trit 384x384", 384, 384),
    ("trit 1536x384", 1536, 384),
    ("trit 384x1536", 384, 1536),
    ("7b 4096x4096", 4096, 4096),
    ("7b 11008x4096", 11008, 4096),
];

fn fixed_shapes(out: &mut Vec<Row>) {
    let mut seed = 7;
    for &(name, rows, cols) in SHAPES {
        let trits: Vec<i8> = (0..rows * cols)
            .map(|_| (lcg(&mut seed) % 3) as i8 - 1)
            .collect();
        let xf: Vec<f32> = (0..cols)
            .map(|_| lcg(&mut seed) as f32 / u32::MAX as f32 - 0.5)
            .collect();
        let act = Activations::quantize(&xf);
        let mut y = vec![0i32; rows];
        let n = (rows * cols) as u64;
        let mut push = |kernel: String, bytes: usize, secs: f64| {
            let r = Row {
                group: name.into(),
                kernel,
                weights: n,
                bytes: bytes as u64,
                threads: 1,
                secs,
            };
            r.print();
            out.push(r);
        };
        for layout in Layout::ALL {
            let m = TernaryMatrix::from_trits(layout, rows, cols, &trits);
            for isa in Isa::available() {
                if isa == Isa::Scalar && rows * cols > 1 << 20 {
                    continue;
                }
                let s = time(|| gemv(isa, black_box(&m), black_box(&act), &mut y));
                push(
                    format!("{}-{}", layout.name(), isa.name()),
                    m.data().len(),
                    s,
                );
            }
        }
        let w8: Vec<i8> = trits.iter().map(|&t| t * 90).collect();
        let s = time(|| dense::gemv_i8(black_box(&w8), black_box(&act.q[..cols]), &mut y));
        push("i8".into(), w8.len(), s);
        let wf: Vec<f32> = trits.iter().map(|&t| t as f32 * 0.02).collect();
        let mut yf = vec![0f32; rows];
        let s = time(|| dense::gemv_f32(black_box(&wf), black_box(&xf), &mut yf));
        push("f32".into(), wf.len() * 4, s);
    }
}

/// One weight format, as a stack of distinct matrices.
enum Stack {
    Ternary(Vec<TernaryMatrix>),
    I8(Vec<Vec<i8>>),
    F32(Vec<Vec<f32>>),
}

const SWEEP_DIM: usize = 2048;

type MakeStack<'a> = Box<dyn Fn() -> Stack + 'a>;

impl Stack {
    fn bytes(&self) -> usize {
        match self {
            Stack::Ternary(v) => v.iter().map(|m| m.data().len()).sum(),
            Stack::I8(v) => v.iter().map(Vec::len).sum(),
            Stack::F32(v) => v.iter().map(|m| m.len() * 4).sum(),
        }
    }

    /// Rows `r0..r1` of every matrix in the stack.
    #[allow(clippy::too_many_arguments)]
    fn pass(
        &self,
        isa: Isa,
        act: &Activations,
        xf: &[f32],
        r0: usize,
        r1: usize,
        y: &mut [i32],
        yf: &mut [f32],
    ) {
        let d = SWEEP_DIM;
        match self {
            Stack::Ternary(v) => v.iter().for_each(|m| gemv_rows(isa, m, r0..r1, act, y)),
            Stack::I8(v) => v
                .iter()
                .for_each(|m| dense::gemv_i8(&m[r0 * d..r1 * d], &act.q[..d], &mut y[..r1 - r0])),
            Stack::F32(v) => v
                .iter()
                .for_each(|m| dense::gemv_f32(&m[r0 * d..r1 * d], xf, &mut yf[..r1 - r0])),
        }
    }
}

fn run_stack(
    stack: Arc<Stack>,
    isa: Isa,
    act: Arc<Activations>,
    xf: Arc<Vec<f32>>,
    threads: usize,
) -> f64 {
    let d = SWEEP_DIM;
    if threads == 1 {
        let mut y = vec![0i32; d];
        let mut yf = vec![0f32; d];
        return time(|| stack.pass(isa, &act, &xf, 0, d, &mut y, &mut yf));
    }
    // Persistent workers: spawning threads per pass would dominate small
    // stacks. Waking them still costs tens of microseconds, so each wake-up
    // runs enough passes to cover at least 64 MB of weights.
    let reps = ((64usize << 20) / stack.bytes()).max(1);
    let start = Arc::new(Barrier::new(threads + 1));
    let done = Arc::new(Barrier::new(threads + 1));
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let per = d.div_ceil(threads);
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let (stack, act, xf) = (stack.clone(), act.clone(), xf.clone());
            let (start, done, stop) = (start.clone(), done.clone(), stop.clone());
            std::thread::spawn(move || {
                let (r0, r1) = (t * per, ((t + 1) * per).min(d));
                let mut y = vec![0i32; per];
                let mut yf = vec![0f32; per];
                loop {
                    start.wait();
                    if stop.load(std::sync::atomic::Ordering::Relaxed) {
                        return;
                    }
                    for _ in 0..reps {
                        stack.pass(isa, &act, &xf, r0, r1, &mut y, &mut yf);
                    }
                    done.wait();
                }
            })
        })
        .collect();
    let secs = time(|| {
        start.wait();
        done.wait();
    }) / reps as f64;
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    start.wait();
    handles.into_iter().for_each(|h| h.join().unwrap());
    secs
}

fn sweep(out: &mut Vec<Row>) {
    let d = SWEEP_DIM;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let isa = Isa::detect();
    let mut seed = 11;
    let trits: Vec<i8> = (0..d * d).map(|_| (lcg(&mut seed) % 3) as i8 - 1).collect();
    let xf: Vec<f32> = (0..d)
        .map(|_| lcg(&mut seed) as f32 / u32::MAX as f32 - 0.5)
        .collect();
    let act = Arc::new(Activations::quantize(&xf));
    let xf = Arc::new(xf);
    let i2 = TernaryMatrix::from_trits(Layout::I2, d, d, &trits);
    let tq = TernaryMatrix::from_trits(Layout::Tq, d, d, &trits);
    let w8: Vec<i8> = trits.iter().map(|&t| t * 90).collect();
    let wf: Vec<f32> = trits.iter().map(|&t| t as f32 * 0.02).collect();
    for n in [1usize, 2, 4, 8, 16, 32, 64, 128, 256] {
        let formats: Vec<(&str, MakeStack)> = vec![
            ("f32", Box::new(|| Stack::F32(vec![wf.clone(); n]))),
            ("i8", Box::new(|| Stack::I8(vec![w8.clone(); n]))),
            ("i2", Box::new(|| Stack::Ternary(vec![i2.clone(); n]))),
            ("tq", Box::new(|| Stack::Ternary(vec![tq.clone(); n]))),
        ];
        for (name, make) in formats {
            // Keep each stack under ~1.1 GB.
            let per_matrix = match name {
                "f32" => d * d * 4,
                "i8" => d * d,
                _ => d * d / 4,
            };
            if per_matrix * n > 1100 << 20 {
                continue;
            }
            let stack = Arc::new(make());
            for t in [1, threads] {
                let secs = run_stack(stack.clone(), isa, act.clone(), xf.clone(), t);
                let r = Row {
                    group: format!("stack {n} x 2048^2"),
                    kernel: format!(
                        "{name}-{}",
                        if name.starts_with('f') || name == "i8" {
                            "best"
                        } else {
                            isa.name()
                        }
                    ),
                    weights: (n * d * d) as u64,
                    bytes: stack.bytes() as u64,
                    threads: t,
                    secs,
                };
                r.print();
                out.push(r);
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json = args
        .iter()
        .position(|a| a == "--json")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let mut rows = Vec::new();
    println!("kernels available: {:?}", Isa::available());
    if args.iter().any(|a| a == "--sweep") {
        sweep(&mut rows);
    } else {
        fixed_shapes(&mut rows);
    }
    if let Some(path) = json {
        // cargo runs benchmarks from the crate directory; resolve relative
        // paths against the workspace root instead.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(path);
        let body: Vec<String> = rows.iter().map(Row::json).collect();
        std::fs::write(&path, format!("[\n{}\n]\n", body.join(",\n"))).expect("write json");
        println!("wrote {}", path.display());
    }
}
