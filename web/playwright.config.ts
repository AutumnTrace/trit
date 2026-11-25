import { defineConfig } from '@playwright/test';

const PORT = 4401;

export default defineConfig({
  testDir: 'e2e',
  timeout: 60_000,
  fullyParallel: false,
  workers: 1,
  reporter: process.env.CI ? [['github'], ['list']] : 'list',
  use: {
    baseURL: `http://localhost:${PORT}/trit/`,
    channel: 'chrome',
    headless: true,
    viewport: { width: 1280, height: 900 },
  },
  webServer: {
    command: `npm run build && npx vite preview --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}/trit/`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
