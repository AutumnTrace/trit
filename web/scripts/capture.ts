/**
 * Records the README media from the built site:
 *
 *   docs/media/hero.gif      a story being written, at the page's reading pace
 *   docs/media/tokens.png    the same story with token boundaries shown
 *   docs/media/weights.png   the weight viewer
 *
 *   npm run build && npm run capture
 *
 * The GIF is a sequence of real screenshots taken while the worker generates;
 * each frame's delay is the time that actually passed between screenshots.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { chromium } from '@playwright/test';
import { preview } from 'vite';
import gifenc from 'gifenc';
import { PNG } from 'pngjs';

const { GIFEncoder, quantize, applyPalette } = gifenc;
const PORT = 4403;
const OUT = '../docs/media';
const SEED = Number(process.env.SEED ?? 1);
// Not PROMPT: on Windows that variable is the shell prompt ("$P$G").
const PROMPT = process.env.STORY_PROMPT ?? 'Once upon a time';

const server = await preview({ preview: { port: PORT, strictPort: true }, logLevel: 'warn' });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
mkdirSync(OUT, { recursive: true });

try {
  const page = await browser.newPage({ viewport: { width: 1000, height: 900 }, deviceScaleFactor: 1 });
  await page.goto(`http://localhost:${PORT}/trit/`);
  await page.waitForFunction(() => window.trit !== undefined);
  await page.evaluate(() => window.trit!.ready);
  await page.locator('#prompt').fill(PROMPT);
  await page.locator('#seed').fill(String(SEED));
  await page.locator('#auto-seed').uncheck();

  // ---- hero GIF: controls and page while the story is written ----------
  // Write the story once at full speed to find how tall the finished page is,
  // so every frame can share one crop that ends below the stats line.
  await page.evaluate(() => window.trit!.generate({ pace: 0, maxTokens: 260 }));
  const top = await page.locator('#controls').boundingBox();
  const stats = await page.locator('#stats').boundingBox();
  const clip = { x: top!.x - 8, y: top!.y - 8, width: top!.width + 16, height: stats!.y + stats!.height + 12 - (top!.y - 8) };
  const frames: { rgba: Uint8Array; t: number }[] = [];
  // Then for real, at the page's own reading pace (the Speed control's default).
  const done = page.evaluate(() => window.trit!.generate({ maxTokens: 260 }));
  let finished = false;
  void done.then(() => (finished = true));
  const t0 = Date.now();
  while (!finished || frames.length === 0) {
    const png = await page.screenshot({ clip, type: 'png' });
    frames.push({ rgba: new Uint8Array(PNG.sync.read(png).data), t: Date.now() - t0 });
  }
  for (let i = 0; i < 12; i++) frames.push({ rgba: frames.at(-1)!.rgba, t: frames.at(-1)!.t + 250 });
  const story = await done;
  const W = Math.round(clip.width);
  const H = Math.round(clip.height);

  const sample = new Uint8Array(W * H * 4 * 3);
  for (let k = 0; k < 3; k++) sample.set(frames[Math.floor(((k + 1) * (frames.length - 1)) / 3)]!.rgba, k * W * H * 4);
  const palette = quantize(sample, 63);
  while (palette.length < 63) palette.push([0, 0, 0]);
  const KEEP = palette.length;
  palette.push([0, 0, 0]);
  const gif = GIFEncoder();
  let previous: Uint8Array | null = null;
  for (let i = 0; i < frames.length; i++) {
    const delay = Math.max(20, (frames[i + 1]?.t ?? frames[i]!.t + 1500) - frames[i]!.t);
    const index = applyPalette(frames[i]!.rgba, palette.slice(0, KEEP));
    if (!previous) {
      gif.writeFrame(index, W, H, { palette, delay });
    } else {
      const delta = new Uint8Array(index.length);
      for (let p = 0; p < index.length; p++) delta[p] = index[p] === previous[p] ? KEEP : index[p]!;
      gif.writeFrame(delta, W, H, { palette, delay, transparent: true, transparentIndex: KEEP });
    }
    previous = index;
  }
  gif.finish();
  writeFileSync(`${OUT}/hero.gif`, gif.bytes());
  console.log(`hero.gif: ${W}x${H}, ${frames.length} frames, ${(gif.bytes().length / 1e6).toFixed(2)} MB`);
  console.log(`story (seed ${SEED}, ${story.tokens} tokens): ${story.text}`);

  // ---- tokens view ------------------------------------------------------
  await page.locator('#show-tokens').check();
  const pageBox = await page.locator('#page').boundingBox();
  await page.screenshot({
    path: `${OUT}/tokens.png`,
    clip: { x: pageBox!.x, y: pageBox!.y, width: pageBox!.width, height: Math.min(pageBox!.height, 420) },
  });
  console.log('tokens.png');

  // ---- weights ------------------------------------------------------------
  await page.setViewportSize({ width: 1280, height: 1400 });
  await page.evaluate(() => window.trit!.showMatrix(3, 4));
  const weights = page.locator('section.weights');
  await weights.scrollIntoViewIfNeeded();
  await weights.screenshot({ path: `${OUT}/weights.png` });
  console.log('weights.png');
} finally {
  await browser.close();
  await new Promise<void>((resolve) => server.httpServer.close(() => resolve()));
}
