import './style.css';
import { MATRICES, type FromWorker, type GenerateOptions, type ModelInfo, type StopReason } from './protocol.ts';
import { drawMatrix, pick } from './weights.ts';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const form = $<HTMLFormElement>('controls');
const promptIn = $<HTMLInputElement>('prompt');
const temperature = $<HTMLInputElement>('temperature');
const topP = $<HTMLInputElement>('top-p');
const seedIn = $<HTMLInputElement>('seed');
const pace = $<HTMLSelectElement>('pace');
const writeBtn = $<HTMLButtonElement>('write');
const stopBtn = $<HTMLButtonElement>('stop');
const showTokens = $<HTMLInputElement>('show-tokens');
const autoSeed = $<HTMLInputElement>('auto-seed');
const page = $('page');
const promptText = $('prompt-text');
const storyText = $('story-text');

const STARTERS = [
  'Once upon a time',
  'Lily found a tiny door in the garden.',
  'The dragon was sad because',
  'Tom and his dog Max went to the beach.',
  'One day, a little robot',
];

// ---- state shared with the e2e tests and the capture script --------------

interface Story {
  prompt: string;
  text: string;
  tokens: number;
  computeMs: number;
  reason: StopReason;
  options: GenerateOptions;
}

declare global {
  interface Window {
    trit?: {
      ready: Promise<ModelInfo>;
      generate(o: Partial<GenerateOptions>): Promise<Story>;
      showMatrix(layer: number, which: number): Promise<void>;
      last: Story | null;
    };
  }
}

const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
let info: ModelInfo | null = null;
let run = 0;
let current: { options: GenerateOptions; decoder: TextDecoder; text: string; resolve?: (s: Story) => void } | null =
  null;
let resolveReady: (i: ModelInfo) => void;
const ready = new Promise<ModelInfo>((r) => (resolveReady = r));
let matrixWaiter: (() => void) | null = null;

const fmtMB = (b: number) => `${(b / 1e6).toFixed(1)} MB`;
const fmtInt = (n: number) => n.toLocaleString('en-US');

function status(text: string): void {
  $('stat-status').textContent = text;
}

// ---- hash state ----------------------------------------------------------

function readHash(): void {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.has('prompt')) promptIn.value = h.get('prompt')!;
  if (h.has('seed')) seedIn.value = String(Number(h.get('seed')) >>> 0);
  if (h.has('t')) temperature.value = h.get('t')!;
  if (h.has('p')) topP.value = h.get('p')!;
  if (h.has('seed')) autoSeed.checked = false;
  syncOutputs();
}

function link(o: GenerateOptions): string {
  const h = new URLSearchParams({ prompt: o.prompt, seed: String(o.seed), t: String(o.temperature), p: String(o.topP) });
  return `${location.origin}${location.pathname}#${h}`;
}

function syncOutputs(): void {
  $('temperature-out').textContent = Number(temperature.value).toFixed(2);
  $('top-p-out').textContent = Number(topP.value).toFixed(2);
}

// ---- generation ----------------------------------------------------------

function options(over: Partial<GenerateOptions> = {}): GenerateOptions {
  return {
    prompt: promptIn.value,
    temperature: Number(temperature.value),
    topP: Number(topP.value),
    seed: Number(seedIn.value) >>> 0,
    maxTokens: 480,
    pace: Number(pace.value),
    ...over,
  };
}

function start(o: GenerateOptions): Promise<Story> {
  run++;
  storyText.replaceChildren();
  promptText.textContent = o.prompt;
  page.classList.add('writing');
  writeBtn.disabled = true;
  stopBtn.disabled = false;
  $('stat-tokens').textContent = '0';
  status('writing…');
  return new Promise((resolve) => {
    current = { options: o, decoder: new TextDecoder(), text: '', resolve };
    worker.postMessage({ type: 'generate', run, options: o });
  });
}

function finish(m: Extract<FromWorker, { type: 'done' }>): void {
  if (!current) return;
  const tail = current.decoder.decode();
  if (tail) appendToken(tail);
  page.classList.remove('writing');
  writeBtn.disabled = false;
  stopBtn.disabled = true;
  const why = { end: 'the model ended the story', limit: 'token limit', stopped: 'stopped', context: 'context is full' };
  status(`${why[m.reason]} · seed ${current.options.seed}`);
  const story: Story = {
    prompt: current.options.prompt,
    text: current.text,
    tokens: m.generated,
    computeMs: m.computeMs,
    reason: m.reason,
    options: current.options,
  };
  window.trit!.last = story;
  history.replaceState(null, '', link(current.options));
  current.resolve?.(story);
  current = null;
  if (autoSeed.checked) seedIn.value = String((Math.random() * 2 ** 32) >>> 0);
}

function appendToken(text: string): void {
  const span = document.createElement('span');
  span.className = 'tok';
  span.textContent = text;
  storyText.append(span);
}

function onTokens(m: Extract<FromWorker, { type: 'tokens' }>): void {
  if (!current) return;
  for (const bytes of m.tokens) {
    const text = current.decoder.decode(bytes, { stream: true });
    current.text += text;
    if (text) appendToken(text);
  }
  $('stat-tokens').textContent = String(m.generated);
  $('stat-context').textContent = String(m.position);
  if (m.computeMs > 0) $('stat-speed').textContent = fmtInt(Math.round((m.generated / m.computeMs) * 1000));
}

// ---- weights -------------------------------------------------------------

let shown: { rows: number; cols: number; trits: Int8Array; transposed: boolean } | null = null;
const canvas = $<HTMLCanvasElement>('weights-canvas');
const layerSel = $<HTMLSelectElement>('layer');
const matrixGroup = $('matrix');
let matrixIndex = 0;

function requestMatrix(): Promise<void> {
  return new Promise((resolve) => {
    matrixWaiter = resolve;
    worker.postMessage({ type: 'matrix', layer: Number(layerSel.value), which: matrixIndex });
  });
}

function onMatrix(m: Extract<FromWorker, { type: 'matrix' }>): void {
  const { transposed, counts } = drawMatrix(canvas, m.rows, m.cols, m.trits);
  shown = { rows: m.rows, cols: m.cols, trits: m.trits, transposed };
  const n = m.rows * m.cols;
  const pct = (k: number) => `${((100 * k) / n).toFixed(1)}%`;
  const name = MATRICES[m.which];
  // Export drops feed-forward units that training switched off (an all-zero
  // w_up row), so these two matrices can be narrower than the trained width.
  const width = name === 'w_up' ? m.rows : name === 'w_down' ? m.cols : null;
  const dropped = info && width !== null && width < info.ffn ? info.ffn - width : 0;
  $('weight-stats').innerHTML = `
    <dt>matrix</dt><dd>layers.${m.layer}.${name}: ${m.rows} × ${m.cols}${transposed ? ' (drawn rotated)' : ''}${
      dropped ? `, ${dropped} dead units dropped at export` : ''
    }</dd>
    <dt>scale &gamma;</dt><dd>${m.gamma.toPrecision(4)}</dd>
    <dt>share</dt><dd>
      <span style="color:var(--neg)">−1 ${pct(counts[0])}</span> ·
      <span style="color:#8d93a8">0 ${pct(counts[1])}</span> ·
      <span style="color:var(--pos)">+1 ${pct(counts[2])}</span>
      <div class="share"><i style="width:${pct(counts[0])};background:var(--neg)"></i><i style="width:${pct(counts[1])};background:var(--zero)"></i><i style="width:${pct(counts[2])};background:var(--pos)"></i></div>
    </dd>`;
  matrixWaiter?.();
  matrixWaiter = null;
}

canvas.addEventListener('pointermove', (e) => {
  const hover = $('weights-hover');
  if (!shown) return;
  const at = pick(canvas, e, shown.rows, shown.cols, shown.transposed);
  hover.hidden = !at;
  if (at) {
    const t = shown.trits[at.row * shown.cols + at.col]!;
    hover.textContent = `row ${at.row}, col ${at.col}: ${t > 0 ? '+1' : t < 0 ? '−1' : '0'}`;
  }
});
canvas.addEventListener('pointerleave', () => ($('weights-hover').hidden = true));

function buildPicker(layers: number): void {
  for (let i = 0; i < layers; i++) layerSel.add(new Option(String(i), String(i)));
  MATRICES.forEach((name, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = name;
    b.setAttribute('role', 'radio');
    b.setAttribute('aria-checked', String(i === 0));
    b.onclick = () => {
      matrixIndex = i;
      for (const el of matrixGroup.children) el.setAttribute('aria-checked', String(el === b));
      void requestMatrix();
    };
    matrixGroup.append(b);
  });
  layerSel.onchange = () => void requestMatrix();
}

function showFacts(i: ModelInfo): void {
  const embed = i.vocab * i.dim;
  const f32 = (i.ternaryWeights + embed) * 4;
  $('file-size').textContent = fmtMB(i.fileBytes);
  $('stat-context-max').textContent = String(i.context);
  $('facts').innerHTML = `
    <dt>shape</dt><dd>${i.layers} layers · dim ${i.dim} · ${i.heads} heads · ffn ${i.ffn}</dd>
    <dt>vocabulary</dt><dd>${fmtInt(i.vocab)} byte-level BPE tokens</dd>
    <dt>context</dt><dd>${i.context} tokens</dd>
    <dt>ternary weights</dt><dd>${fmtInt(i.ternaryWeights)}</dd>
    <dt>embedding</dt><dd>${fmtInt(embed)} × int8, tied to the output</dd>
    <dt>file</dt><dd>${fmtMB(i.fileBytes)} (${fmtMB(f32)} as float32)</dd>
    <dt>kernel</dt><dd>${i.simd ? 'WebAssembly SIMD128' : 'scalar WebAssembly'}</dd>`;
}

// ---- wiring --------------------------------------------------------------

worker.onmessage = (e: MessageEvent<FromWorker>) => {
  const m = e.data;
  switch (m.type) {
    case 'progress':
      if (m.total) status(`loading the model… ${Math.round((100 * m.loaded) / m.total)}%`);
      break;
    case 'ready':
      info = m.info;
      showFacts(m.info);
      buildPicker(m.info.layers);
      writeBtn.disabled = false;
      writeBtn.textContent = 'Write';
      status(`ready (loaded in ${(m.loadMs / 1000).toFixed(1)} s)`);
      void requestMatrix();
      resolveReady(m.info);
      break;
    case 'error':
      status(`error: ${m.message}`);
      break;
    case 'prompt':
      break;
    case 'tokens':
      if (m.run === run) onTokens(m);
      break;
    case 'done':
      if (m.run === run) finish(m);
      break;
    case 'matrix':
      onMatrix(m);
      break;
  }
};

form.addEventListener('submit', (e) => {
  e.preventDefault();
  if (info && !current) void start(options());
});
stopBtn.onclick = () => worker.postMessage({ type: 'stop' });
$('reseed').onclick = () => (seedIn.value = String((Math.random() * 2 ** 32) >>> 0));
temperature.oninput = topP.oninput = syncOutputs;
showTokens.onchange = () => page.classList.toggle('tokens', showTokens.checked);
$('copy-link').onclick = async () => {
  const o = window.trit?.last?.options ?? options();
  await navigator.clipboard?.writeText(link(o)).catch(() => {});
  status('link copied');
};
for (const s of STARTERS) {
  const b = document.createElement('button');
  b.type = 'button';
  b.textContent = s;
  b.onclick = () => {
    promptIn.value = s;
    promptIn.focus();
  };
  $('starters').append(b);
}

window.trit = {
  ready,
  last: null,
  generate: async (over) => {
    await ready;
    if (current) {
      worker.postMessage({ type: 'stop' });
      await new Promise<void>((r) => {
        const t = setInterval(() => !current && (clearInterval(t), r()), 10);
      });
    }
    return start(options(over));
  },
  showMatrix: async (layer, which) => {
    await ready;
    layerSel.value = String(layer);
    (matrixGroup.children[which] as HTMLButtonElement).click();
    await new Promise<void>((r) => {
      const t = setInterval(() => !matrixWaiter && (clearInterval(t), r()), 10);
    });
  },
};

readHash();
addEventListener('hashchange', readHash);
worker.postMessage({ type: 'load' });
