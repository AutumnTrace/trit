import { defineConfig } from 'vite';

export default defineConfig({
  base: '/trit/',
  build: {
    target: 'es2022',
    assetsInlineLimit: 0,
  },
  worker: {
    format: 'es',
  },
});
