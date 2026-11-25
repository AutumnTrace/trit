/** Draws a ternary matrix as an image, one pixel per weight. */

const NEG = [240, 88, 125];
const ZERO = [24, 26, 34];
const POS = [58, 209, 192];

export interface Drawn {
  /** True when rows are drawn horizontally (the matrix is taller than wide). */
  transposed: boolean;
  counts: [number, number, number];
}

export function drawMatrix(canvas: HTMLCanvasElement, rows: number, cols: number, trits: Int8Array): Drawn {
  const transposed = rows > cols;
  const w = transposed ? rows : cols;
  const h = transposed ? cols : rows;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(w, h);
  const px = img.data;
  const counts: [number, number, number] = [0, 0, 0];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const t = trits[r * cols + c]!;
      counts[t + 1]!++;
      const color = t < 0 ? NEG : t > 0 ? POS : ZERO;
      const [x, y] = transposed ? [r, c] : [c, r];
      const o = (y * w + x) * 4;
      px[o] = color[0]!;
      px[o + 1] = color[1]!;
      px[o + 2] = color[2]!;
      px[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return { transposed, counts };
}

/** Matrix coordinates under a pointer event, or null outside the image. */
export function pick(
  canvas: HTMLCanvasElement,
  e: PointerEvent,
  rows: number,
  cols: number,
  transposed: boolean,
): { row: number; col: number } | null {
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor(((e.clientX - rect.left) / rect.width) * canvas.width);
  const y = Math.floor(((e.clientY - rect.top) / rect.height) * canvas.height);
  if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return null;
  const [row, col] = transposed ? [x, y] : [y, x];
  return row < rows && col < cols ? { row, col } : null;
}
