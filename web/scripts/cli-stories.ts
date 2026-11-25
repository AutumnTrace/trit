/**
 * Records a few stories with the native CLI for the browser test that checks
 * the WebAssembly build writes exactly the same ones.
 *
 *   cargo build -p trit-cli --release && node scripts/cli-stories.ts
 */
import { execFileSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';

const bin = `../target/release/trit${process.platform === 'win32' ? '.exe' : ''}`;
const cases = [
  { prompt: 'Once upon a time', seed: 1, temperature: 0.8, topP: 0.9, maxTokens: 200 },
  { prompt: 'The dragon was sad because', seed: 2, temperature: 1.0, topP: 0.95, maxTokens: 200 },
  { prompt: 'Tom and his dog Max went to the beach.', seed: 3, temperature: 0.6, topP: 0.8, maxTokens: 200 },
];
const stories = cases.map((c) => {
  const out = execFileSync(
    bin,
    [
      'generate',
      '--model', '../models/tinystories-15m.trit',
      '--prompt', c.prompt,
      '--seed', String(c.seed),
      '--temperature', String(c.temperature),
      '--top-p', String(c.topP),
      '--max-tokens', String(c.maxTokens),
    ],
    { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] },
  );
  return { ...c, text: out.replace(/\n$/, '') };
});
writeFileSync(
  'e2e/cli-stories.json',
  JSON.stringify({ command: 'node scripts/cli-stories.ts', stories }, null, 2) + '\n',
);
console.log(`e2e/cli-stories.json: ${stories.length} stories`);
