/**
 * Generation speed of the WebAssembly build in headless Chrome: five stories
 * of up to 256 tokens at full speed, counting only time spent inside the
 * model (not yielding to the page).
 *
 *   npm run build && npm run bench
 */
import { writeFileSync } from 'node:fs';
import { chromium } from '@playwright/test';
import { preview } from 'vite';

const PORT = 4404;
const server = await preview({ preview: { port: PORT, strictPort: true }, logLevel: 'warn' });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
try {
  const page = await browser.newPage();
  await page.goto(`http://localhost:${PORT}/trit/`);
  await page.waitForFunction(() => window.trit !== undefined);
  await page.evaluate(() => window.trit!.ready);
  // Warm-up story, not counted.
  await page.evaluate(() => window.trit!.generate({ seed: 999, pace: 0, maxTokens: 64 }));
  let tokens = 0;
  let ms = 0;
  for (let seed = 0; seed < 5; seed++) {
    const s = await page.evaluate((seed) => window.trit!.generate({ seed, pace: 0, maxTokens: 256 }), seed);
    tokens += s.tokens;
    ms += s.computeMs;
  }
  const version = browser.version();
  const result = { browser: `Chrome ${version}`, tokens, seconds: ms / 1000, tok_per_s: (tokens * 1000) / ms };
  console.log(result);
  writeFileSync('../bench/wasm.json', JSON.stringify(result, null, 2) + '\n');
} finally {
  await browser.close();
  await new Promise<void>((resolve) => server.httpServer.close(() => resolve()));
}
