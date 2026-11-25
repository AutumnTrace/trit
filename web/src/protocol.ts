/** Messages between the page and the worker that owns the WebAssembly instance. */

export interface ModelInfo {
  vocab: number;
  dim: number;
  layers: number;
  heads: number;
  ffn: number;
  context: number;
  ternaryWeights: number;
  memoryBytes: number;
  fileBytes: number;
  simd: boolean;
}

export interface GenerateOptions {
  prompt: string;
  temperature: number;
  topP: number;
  seed: number;
  maxTokens: number;
  /** Tokens per second to reveal at; 0 for as fast as the model runs. */
  pace: number;
}

export type StopReason = 'end' | 'limit' | 'stopped' | 'context';

export type ToWorker =
  | { type: 'load' }
  | { type: 'generate'; run: number; options: GenerateOptions }
  | { type: 'stop' }
  | { type: 'matrix'; layer: number; which: number };

export type FromWorker =
  | { type: 'progress'; loaded: number; total: number }
  | { type: 'ready'; info: ModelInfo; loadMs: number }
  | { type: 'error'; message: string }
  | { type: 'prompt'; run: number; tokens: number; ms: number }
  /** One or more tokens, each as its raw bytes (a token can end mid-character). */
  | { type: 'tokens'; run: number; tokens: Uint8Array[]; position: number; computeMs: number; generated: number }
  | { type: 'done'; run: number; generated: number; computeMs: number; reason: StopReason }
  | { type: 'matrix'; layer: number; which: number; rows: number; cols: number; gamma: number; trits: Int8Array };

export const MATRICES = ['wq', 'wk', 'wv', 'wo', 'w_up', 'w_down'] as const;
