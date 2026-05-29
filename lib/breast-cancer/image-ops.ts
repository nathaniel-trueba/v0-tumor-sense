// Browser-side image operations used by the forward-pass viz, Grad-CAM
// overlay and eigen-breasts panel. Everything here is deterministic and runs
// on the canvas — no backend required.

export type Kernel = number[][];

export const KERNELS: Record<string, Kernel> = {
  identity: [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
  ],
  sobelX: [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
  ],
  sobelY: [
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1],
  ],
  laplacian: [
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0],
  ],
  blur: [
    [1 / 16, 2 / 16, 1 / 16],
    [2 / 16, 4 / 16, 2 / 16],
    [1 / 16, 2 / 16, 1 / 16],
  ],
  sharpen: [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
  ],
  emboss: [
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2],
  ],
  diag: [
    [-1, -1, 2],
    [-1, 2, -1],
    [2, -1, -1],
  ],
};

export function loadImageData(url: string): Promise<ImageData> {
  return new Promise((resolve, reject) => {
    const img = new window.Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const ctx = c.getContext("2d");
      if (!ctx) return reject(new Error("no 2d ctx"));
      ctx.drawImage(img, 0, 0);
      resolve(ctx.getImageData(0, 0, c.width, c.height));
    };
    img.onerror = reject;
    img.src = url;
  });
}

export function emptyImageData(w: number, h: number): ImageData {
  return new ImageData(w, h);
}

export function cloneImageData(src: ImageData): ImageData {
  const dst = new ImageData(src.width, src.height);
  dst.data.set(src.data);
  return dst;
}

export function applyKernel(src: ImageData, kernel: Kernel, opts: { bias?: number; mono?: boolean } = {}): ImageData {
  const { bias = 128, mono = false } = opts;
  const w = src.width;
  const h = src.height;
  const out = new ImageData(w, h);
  const k = kernel;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let r = 0;
      let g = 0;
      let b = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const px = Math.min(w - 1, Math.max(0, x + kx));
          const py = Math.min(h - 1, Math.max(0, y + ky));
          const i = (py * w + px) * 4;
          const weight = k[ky + 1][kx + 1];
          r += src.data[i] * weight;
          g += src.data[i + 1] * weight;
          b += src.data[i + 2] * weight;
        }
      }
      const oi = (y * w + x) * 4;
      const rv = clamp8(r + bias);
      const gv = clamp8(g + bias);
      const bv = clamp8(b + bias);
      if (mono) {
        const v = clamp8((rv + gv + bv) / 3);
        out.data[oi] = v;
        out.data[oi + 1] = v;
        out.data[oi + 2] = v;
      } else {
        out.data[oi] = rv;
        out.data[oi + 1] = gv;
        out.data[oi + 2] = bv;
      }
      out.data[oi + 3] = 255;
    }
  }
  return out;
}

function clamp8(v: number) {
  return Math.max(0, Math.min(255, v));
}

export function maxPool(src: ImageData, factor = 2): ImageData {
  const w = Math.max(1, Math.floor(src.width / factor));
  const h = Math.max(1, Math.floor(src.height / factor));
  const out = new ImageData(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let r = 0;
      let g = 0;
      let b = 0;
      for (let dy = 0; dy < factor; dy++) {
        for (let dx = 0; dx < factor; dx++) {
          const sx = x * factor + dx;
          const sy = y * factor + dy;
          const i = (sy * src.width + sx) * 4;
          r = Math.max(r, src.data[i]);
          g = Math.max(g, src.data[i + 1]);
          b = Math.max(b, src.data[i + 2]);
        }
      }
      const oi = (y * w + x) * 4;
      out.data[oi] = r;
      out.data[oi + 1] = g;
      out.data[oi + 2] = b;
      out.data[oi + 3] = 255;
    }
  }
  return out;
}

// Approximate ReLU after a centered convolution result.
export function relu(src: ImageData): ImageData {
  const out = new ImageData(src.width, src.height);
  for (let i = 0; i < src.data.length; i += 4) {
    out.data[i] = Math.max(0, src.data[i] - 128) * 2;
    out.data[i + 1] = Math.max(0, src.data[i + 1] - 128) * 2;
    out.data[i + 2] = Math.max(0, src.data[i + 2] - 128) * 2;
    out.data[i + 3] = 255;
  }
  return out;
}

// Tinted version of a feature map — used for the "channel" visualisation in
// deeper layers.
export function tint(src: ImageData, rgb: [number, number, number], strength = 0.65): ImageData {
  const out = new ImageData(src.width, src.height);
  for (let i = 0; i < src.data.length; i += 4) {
    const luma = (src.data[i] + src.data[i + 1] + src.data[i + 2]) / 3 / 255;
    out.data[i] = clamp8(luma * 255 * (1 - strength) + rgb[0] * strength * luma);
    out.data[i + 1] = clamp8(luma * 255 * (1 - strength) + rgb[1] * strength * luma);
    out.data[i + 2] = clamp8(luma * 255 * (1 - strength) + rgb[2] * strength * luma);
    out.data[i + 3] = 255;
  }
  return out;
}

// Paste an ImageData into a canvas, scaling with nearest-neighbour so the
// per-pixel structure stays visible.
export function blitTo(canvas: HTMLCanvasElement, src: ImageData, dpr = 1) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const w = canvas.width;
  const h = canvas.height;
  const off = document.createElement("canvas");
  off.width = src.width;
  off.height = src.height;
  off.getContext("2d")!.putImageData(src, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(off, 0, 0, w, h);
}
