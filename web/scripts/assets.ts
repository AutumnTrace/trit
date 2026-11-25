/** Puts the release wasm build and the model file into public/, where Vite serves them from. */
import { copyFileSync, mkdirSync, statSync } from 'node:fs';

mkdirSync('public', { recursive: true });
for (const [src, dst] of [
  ['../target/wasm32-unknown-unknown/release/trit_wasm.wasm', 'public/trit.wasm'],
  ['../models/tinystories-15m.trit', 'public/tinystories-15m.trit'],
]) {
  copyFileSync(src!, dst!);
  console.log(`${dst}: ${(statSync(dst!).size / 1024).toFixed(0)} KiB`);
}
