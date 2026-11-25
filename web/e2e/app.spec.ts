import { readFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.goto('./');
  await page.waitForFunction(() => window.trit !== undefined);
  await page.evaluate(() => window.trit!.ready);
});

test('loads the model and describes it', async ({ page }) => {
  await expect(page.locator('#write')).toBeEnabled();
  await expect(page.locator('#facts')).toContainText('ternary weights');
  await expect(page.locator('#facts')).toContainText('WebAssembly SIMD128');
});

test('writes a story that continues the prompt', async ({ page }) => {
  const story = await page.evaluate(() =>
    window.trit!.generate({ prompt: 'Once upon a time', seed: 3, pace: 0, maxTokens: 120 }),
  );
  expect(story.tokens).toBeGreaterThan(10);
  expect(story.text.trim().length).toBeGreaterThan(30);
  await expect(page.locator('#story-text')).toHaveText(story.text);
  await expect(page.locator('#prompt-text')).toHaveText('Once upon a time');
  expect(Number(await page.locator('#stat-speed').textContent())).toBeGreaterThan(0);
});

test('the same seed writes the same story', async ({ page }) => {
  const opts = { prompt: 'Tom and his dog', seed: 99, pace: 0, maxTokens: 80, temperature: 0.9, topP: 0.95 };
  const a = await page.evaluate((o) => window.trit!.generate(o), opts);
  const b = await page.evaluate((o) => window.trit!.generate(o), opts);
  expect(b.text).toBe(a.text);
  const c = await page.evaluate((o) => window.trit!.generate({ ...o, seed: 100 }), opts);
  expect(c.text).not.toBe(a.text);
});

test('stop ends a paced story early', async ({ page }) => {
  const pending = page.evaluate(() => window.trit!.generate({ seed: 5, pace: 20, maxTokens: 400 }));
  await expect(page.locator('#stop')).toBeEnabled();
  await page.waitForTimeout(400);
  await page.locator('#stop').click();
  const story = await pending;
  expect(story.reason).toBe('stopped');
  expect(story.tokens).toBeLessThan(60);
});

test('a link reproduces the story', async ({ page }) => {
  const story = await page.evaluate(() =>
    window.trit!.generate({ prompt: 'The dragon was sad because', seed: 7, pace: 0, maxTokens: 60 }),
  );
  const url = page.url();
  expect(url).toContain('seed=7');
  await page.goto('about:blank');
  await page.goto(url);
  await page.waitForFunction(() => window.trit !== undefined);
  await page.evaluate(() => window.trit!.ready);
  await expect(page.locator('#prompt')).toHaveValue('The dragon was sad because');
  const again = await page.evaluate(() => window.trit!.generate({ pace: 0, maxTokens: 60 }));
  expect(again.text).toBe(story.text);
});

test('draws every weight matrix as trits', async ({ page }) => {
  for (const which of [0, 4, 5]) {
    await page.evaluate((w) => window.trit!.showMatrix(2, w), which);
    await expect(page.locator('#weight-stats')).toContainText(`layers.2.`);
  }
  const size = await page.locator('#weights-canvas').evaluate((c: HTMLCanvasElement) => [c.width, c.height]);
  expect(size).toEqual([1536, 384]); // w_down is 384 x 1536, drawn as is
  const share = await page.locator('#weight-stats').textContent();
  expect(share).toMatch(/−1 [\d.]+%.*0 [\d.]+%.*\+1 [\d.]+%/s);
});

test('writes the same stories as the native CLI', async ({ page }) => {
  // Recorded with `trit generate` (see e2e/cli-stories.json for the command).
  const { stories } = JSON.parse(readFileSync(new URL('./cli-stories.json', import.meta.url), 'utf8')) as {
    stories: { prompt: string; seed: number; temperature: number; topP: number; maxTokens: number; text: string }[];
  };
  for (const s of stories) {
    const got = await page.evaluate(
      (s) => window.trit!.generate({ prompt: s.prompt, seed: s.seed, temperature: s.temperature, topP: s.topP, maxTokens: s.maxTokens, pace: 0 }),
      s,
    );
    expect(got.prompt + got.text).toBe(s.text);
  }
});
