/// <reference lib="webworker" />
/**
 * Owns the WebAssembly instance. Generation runs in short slices so a stop
 * request (or a weight-matrix request) is handled between tokens.
 */
import type { FromWorker, GenerateOptions, ModelInfo, StopReason, ToWorker } from './protocol.ts';

interface Exports {
  memory: WebAssembly.Memory;
  trit_alloc(len: number): number;
  trit_free(ptr: number, len: number): void;
  trit_load(ptr: number, len: number): number;
  trit_info(what: number): number;
  trit_start(ptr: number, len: number, temperature: number, topP: number, seed: number): number;
  trit_next(): number;
  trit_out_ptr(): number;
  trit_pos(): number;
  trit_matrix(layer: number, which: number): number;
  trit_matrix_info(layer: number, which: number, what: number): number;
}

const base = import.meta.env.BASE_URL;
let wasm: Exports | null = null;
let info: ModelInfo | null = null;
let runId = 0;

function post(msg: FromWorker, transfer: Transferable[] = []): void {
  (self as DedicatedWorkerGlobalScope).postMessage(msg, transfer);
}

async function fetchWithProgress(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length') ?? 0);
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    post({ type: 'progress', loaded, total });
  }
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

async function load(): Promise<void> {
  const t0 = performance.now();
  const [module, file] = await Promise.all([
    WebAssembly.compileStreaming(fetch(`${base}trit.wasm`)),
    fetchWithProgress(`${base}tinystories-15m.trit`),
  ]);
  const instance = await WebAssembly.instantiate(module, {});
  wasm = instance.exports as unknown as Exports;
  const ptr = wasm.trit_alloc(file.length);
  new Uint8Array(wasm.memory.buffer, ptr, file.length).set(file);
  if (wasm.trit_load(ptr, file.length) !== 0) throw new Error('the model file is not valid');
  const i = (n: number) => wasm!.trit_info(n);
  info = {
    vocab: i(0),
    dim: i(1),
    layers: i(2),
    heads: i(3),
    ffn: i(4),
    context: i(5),
    ternaryWeights: i(6),
    memoryBytes: i(7),
    fileBytes: file.length,
    simd: i(8) === 1,
  };
  post({ type: 'ready', info, loadMs: performance.now() - t0 });
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function generate(run: number, o: GenerateOptions): Promise<void> {
  const w = wasm!;
  const id = ++runId;
  const enc = new TextEncoder().encode(o.prompt);
  const p = w.trit_alloc(Math.max(1, enc.length));
  new Uint8Array(w.memory.buffer, p, enc.length).set(enc);
  const t0 = performance.now();
  const promptTokens = w.trit_start(p, enc.length, o.temperature, o.topP, o.seed >>> 0);
  w.trit_free(p, Math.max(1, enc.length));
  post({ type: 'prompt', run, tokens: promptTokens, ms: performance.now() - t0 });

  let generated = 0;
  let computeMs = 0;
  let reason: StopReason = 'end';
  const sliceMs = o.pace > 0 ? 0 : 12;
  const started = performance.now();
  for (;;) {
    const batch: Uint8Array[] = [];
    const sliceStart = performance.now();
    let ended = false;
    do {
      if (generated >= o.maxTokens) {
        reason = 'limit';
        ended = true;
        break;
      }
      const t = performance.now();
      const n = w.trit_next();
      computeMs += performance.now() - t;
      if (n < 0) {
        reason = w.trit_pos() >= info!.context ? 'context' : 'end';
        ended = true;
        break;
      }
      batch.push(new Uint8Array(w.memory.buffer, w.trit_out_ptr(), n).slice());
      generated++;
    } while (performance.now() - sliceStart < sliceMs);
    if (batch.length) post({ type: 'tokens', run, tokens: batch, position: w.trit_pos(), computeMs, generated });
    if (ended) break;
    // Yield so 'stop' and 'matrix' messages get through.
    await sleep(o.pace > 0 ? Math.max(0, started + (generated * 1000) / o.pace - performance.now()) : 0);
    if (id !== runId) {
      reason = 'stopped';
      break;
    }
  }
  post({ type: 'done', run, generated, computeMs, reason });
}

self.onmessage = async (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  try {
    switch (m.type) {
      case 'load':
        await load();
        break;
      case 'generate':
        await generate(m.run, m.options);
        break;
      case 'stop':
        runId++;
        break;
      case 'matrix': {
        const w = wasm!;
        const rows = w.trit_matrix_info(m.layer, m.which, 0);
        const cols = w.trit_matrix_info(m.layer, m.which, 1);
        const gamma = new Float32Array(new Uint32Array([w.trit_matrix_info(m.layer, m.which, 2)]).buffer)[0]!;
        const ptr = w.trit_matrix(m.layer, m.which);
        const trits = new Int8Array(w.memory.buffer, ptr, rows * cols).slice();
        post({ type: 'matrix', layer: m.layer, which: m.which, rows, cols, gamma, trits }, [trits.buffer]);
        break;
      }
    }
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
