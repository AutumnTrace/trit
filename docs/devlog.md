# Devlog

Notes from building v0.1.0, mostly the things that didn't go to plan.

## The GPU was busy

The training box is my desktop (RTX 5070, 12 GB), and when I started the
first throughput test the card was already at 80–99% utilization and 5 GB of
memory from other programs. The first eager PyTorch run managed **26k
tokens/s** at batch 16, and batch 32 was *slower* (15k tokens/s), which made
no sense until I watched memory: at batch 32 the model needed 6 GB on a card
with 7 GB left, and on Windows that spills into shared system memory instead
of failing. A plain 4096² bf16 matmul measured 13.8 TFLOPS on the shared
card, a fraction of what it does idle.

The fix that mattered was `torch.compile`. BitLinear is mostly elementwise
work (absmax, scale, round, clamp, the straight-through trick, all of it per
call), and eager mode launches a kernel for each step and writes every
intermediate to memory. On Windows that needs the `triton-windows` package;
with it the same model ran at **87k tokens/s** at batch 16 and **104k** at
batch 32, using 2.4 GB instead of 6. Over the whole run, evaluations
included, the ternary model averaged **106k tokens/s** and the full-precision
baseline **206k**: re-quantizing every weight and activation on every
forward pass is not free.

So the training numbers in the README are from a shared GPU. They are what
it took, not what the card can do.

## No dependencies, by accident

The first version of the kernel tests used `proptest` and the benchmarks
used `criterion`. Neither would build: this machine's Rust toolchain is the
`windows-gnu` one, and recent `getrandom` and `windows-sys` link Windows
system libraries with `raw-dylib`, which on that toolchain needs `dlltool`
and an assembler that aren't installed.

Rather than install a MinGW toolchain to run two dev-dependencies, I wrote
the tests with a 15-line SplitMix64 and seeded random cases (1,500 random
shapes plus every block edge, and any failure prints its seed), and the
benchmark with a calibrate-then-median-of-11 loop. The workspace now has no
dependencies outside `std` at all, which also keeps the WebAssembly build at
120 KB.

## The AVX-512 kernel was no faster than AVX2

The first AVX-512 VNNI kernel processed twice as many bytes per instruction
as the AVX2 one and was, at the 4096 x 4096 shape, slightly *slower*
(87 vs 91 Gweights/s). It was latency-bound: four `vpdpbusd` per iteration
all accumulating into the same register, so each had to wait for the one
before. One accumulator per stripe made it about twice as fast.

Then I noticed I'd been unfair to the baselines: the int8 kernel had the same
single-chain problem. My first fix for it indexed an array of four
accumulators with `(c / 64) % 4`, which compiled to loads and stores through
the stack and made it *slower*. Unrolling by hand into four named variables
fixed that. The f32 baseline got the same four-accumulator treatment.

## A layout for each job

I wanted the file to be as small as the information allows, so it stores
five trits per byte (1.6 bits). But the fast kernels want 32-byte blocks
whose bytes each feed 32 consecutive columns, and with 160 columns per base-3
block, this model's width of 384 doesn't divide: every 384-wide row would
need three blocks, 25% padding. So the file format and the in-memory layouts
are separate things. The file is plain base 3, five consecutive columns per
byte, and on load each matrix is unpacked and repacked into whichever layout
the engine is using (I2 by default). Loading the whole model, including
that repacking, takes about 16 ms.

## Re-quantizing a quantized matrix

To check the Rust engine against PyTorch I first planned to load the
exported trits back into the training model, setting each weight to
`t * γ`. That doesn't work, and silently: BitLinear quantizes its weights on
every forward pass, and the absmean of `t * γ` is `γ * mean(|t|)`, not `γ`,
because a third of the trits are zero. The trits survive the round trip but
the scale shrinks by a third and every output with it. So
`trit_train.reference` is its own small forward pass written directly against
the file, with no `nn.Module` in between, and a separate Python test checks
that it reproduces the training model.

## Rounding

`torch.round` rounds halves to even; Rust's `f32::round` rounds them away
from zero. Activation quantization produces exact halves more often than
you'd think (any `x * s` that lands on `n + 0.5`), so the engine uses
`round_ties_even`. With that, the largest logit difference between Rust and
PyTorch on the test prompt is about 5e-7 of the logit range.

## Same seed, different story

The first time I ran the same seed in the CLI and in the browser, the two
stories agreed word for word for about seventy tokens ("Mia asked her mom,
'Mom, can I open this box?' Her mom said, 'Of course, Mia. You can't open
it") and then went different ways ("...yet." against "...It is not
yours."). The kernels are exact and everything else is ordinary float
arithmetic, which x86 and WebAssembly compute identically, so the culprit had
to be the maths library: `f32::exp` in the softmax and the sampler, and
`sin`/`cos` for the RoPE tables, are implemented differently on each
platform. One last-bit difference in one probability is enough to tip a
sample eventually.

`exp` is now a small polynomial with Cody-Waite range reduction, built from
adds and multiplies only, and the RoPE tables are computed in f64 and
rounded once. Since then the CLI and the browser write identical stories,
and an end-to-end test keeps it that way. PyTorch parity didn't move.

## A regex without look-ahead

GPT-2's pre-tokenizer ends with `\s+(?!\S)|\s+`, which keeps the last space
of a run attached to the next word. Supporting look-ahead would mean a regex
engine in the WebAssembly build, so I dropped it and wrote the pattern so that
a hand-written scanner can follow it exactly. The difference shows up only on
runs of several spaces: `"a  b"` splits as `a` / `  ` / `b` here, where GPT-2
gives `a` / ` ` / ` b`. TinyStories has almost no double spaces, and the
Rust scanner is tested against Hugging Face's output on a set of awkward
strings (contractions, digits, tabs, accented letters, curly quotes, an
emoji, CJK).

## Sixteen threads, slower than one

The first run of the working-set sweep reported that sixteen threads did
one in-cache 2048 x 2048 ternary GEMV at 28 billion weights per second, a
tenth of what one thread managed. The work takes a few microseconds; waking
sixteen threads through a `std::sync::Barrier` and waiting for all of them
took about 150. The benchmark now has every wake-up run enough passes to
cover at least 64 MB of weights, and the in-cache number became 975 billion.
The engine itself stays single-threaded: at this model's size a token's
whole forward pass takes about 0.4 ms, not much more than that overhead.

## Small things the tests and screenshots caught

- The validation split starts in the middle of a story ("...u don't have to
  be scared of the loud dog"). Harmless, but it's the first thing you see.
- The model is useful well before training ends: the checkpoint at step
  5,000 of 20,000 already wrote whole, coherent stories, so the browser demo
  was built against that one while the real run finished.
- An end-to-end test for share links failed because navigating to the same
  page with a different `#hash` doesn't reload it. The page now listens for
  `hashchange`.
- The first screenshot showed every story starting on its second line: the
  story element uses `white-space: pre-wrap`, and the HTML had a newline and
  indentation inside it.
- The capture script read the prompt from a `PROMPT` environment variable,
  and the first GIF began "$P$G". On Windows, `PROMPT` is the shell's prompt
  string. It is `STORY_PROMPT` now.

## After v0.1.0: pruning the dead units

The 396 switched-off feed-forward units looked like free savings, and they
nearly were. The catch was the sub-norm in front of `w_down`: it divides the
sum of squares by the layer's width, so dividing by the pruned width changes
every remaining activation slightly. The parity test fails with that: the
worst logit is off by 1.9% of the range against PyTorch. With the divisor kept
at the trained width (stored in the header as before), the sum itself
doesn't change, because adding 0.0 to a float sum leaves it exactly as it
was. The pruned model's output is bit-identical, and a test puts dead units
back to check exactly that.

Only units with an all-zero `w_up` row qualify. In a few cases `w_down`'s
column is zero but `w_up`'s row isn't. Those units still produce
activations, which count in the sub-norm and the absmax, so they stay.

It doesn't make generation measurably faster: `w_up` loses 3.2% of its rows,
but `w_down` rows are padded to whole 128- or 160-column blocks, so most
layers keep the same number of blocks there. The file is 63 KB smaller.
