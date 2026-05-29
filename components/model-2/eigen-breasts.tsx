"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Image from "next/image";
import { loadImageData } from "@/lib/breast-cancer/image-ops";
import { IMAGE_CATALOG, type PatchImage } from "@/lib/breast-cancer/image-dataset";

const PATCH_W = 50;
const PATCH_H = 50;
const D = PATCH_W * PATCH_H * 3;
const K = 8;

interface PcaResult {
  mean: ImageData;
  eigenImages: ImageData[];
  variance: number[]; // explained ratio per component
  cumulative: number[]; // cumulative explained
  // For the selected image: a (length-K) coordinate vector in the eigenspace.
  project: (img: ImageData) => number[];
}

// Real PCA via the dual covariance trick. 36 samples × 7500 features means the
// 36×36 dual matrix is trivial to eigendecompose with deflated power iteration.
async function computePca(catalog: PatchImage[]): Promise<PcaResult> {
  const datasets = await Promise.all(catalog.map((p) => loadImageData(p.url)));
  const N = datasets.length;

  // Pack into N × D matrix of floats in [0,1].
  const X = new Float32Array(N * D);
  for (let n = 0; n < N; n++) {
    const data = datasets[n].data;
    for (let p = 0; p < PATCH_W * PATCH_H; p++) {
      X[n * D + p * 3] = data[p * 4] / 255;
      X[n * D + p * 3 + 1] = data[p * 4 + 1] / 255;
      X[n * D + p * 3 + 2] = data[p * 4 + 2] / 255;
    }
  }

  // Mean.
  const mean = new Float32Array(D);
  for (let n = 0; n < N; n++) {
    for (let d = 0; d < D; d++) mean[d] += X[n * D + d];
  }
  for (let d = 0; d < D; d++) mean[d] /= N;

  // Centered data.
  const Xc = new Float32Array(N * D);
  for (let n = 0; n < N; n++) {
    for (let d = 0; d < D; d++) Xc[n * D + d] = X[n * D + d] - mean[d];
  }

  // Dual covariance K = Xc · Xc^T (N×N).
  const Kmat = new Float32Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = i; j < N; j++) {
      let s = 0;
      for (let d = 0; d < D; d++) s += Xc[i * D + d] * Xc[j * D + d];
      Kmat[i * N + j] = s;
      Kmat[j * N + i] = s;
    }
  }

  // Deflated power iteration to extract top-K eigenvectors of Kmat.
  const eigVecs: Float32Array[] = [];
  const eigVals: number[] = [];
  const Kwork = new Float32Array(Kmat);
  for (let k = 0; k < K; k++) {
    let v = new Float32Array(N);
    // Deterministic init using a per-component sine pattern so PCA is
    // reproducible across renders.
    for (let i = 0; i < N; i++) v[i] = Math.sin(i * (k + 1) * 0.7) + 0.001;
    normalise(v);
    let lambda = 0;
    for (let iter = 0; iter < 60; iter++) {
      const next = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        let s = 0;
        for (let j = 0; j < N; j++) s += Kwork[i * N + j] * v[j];
        next[i] = s;
      }
      lambda = norm(next);
      if (lambda < 1e-9) break;
      for (let i = 0; i < N; i++) v[i] = next[i] / lambda;
    }
    eigVecs.push(v);
    eigVals.push(lambda);
    // Deflate: Kwork -= lambda · v · vᵀ
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) Kwork[i * N + j] -= lambda * v[i] * v[j];
    }
  }

  // Reconstruct image-space eigenvectors u_k = Xcᵀ · v_k (then normalise).
  const eigImagesRaw: Float32Array[] = [];
  for (let k = 0; k < K; k++) {
    const u = new Float32Array(D);
    const vk = eigVecs[k];
    for (let d = 0; d < D; d++) {
      let s = 0;
      for (let n = 0; n < N; n++) s += Xc[n * D + d] * vk[n];
      u[d] = s;
    }
    const nrm = norm(u);
    if (nrm > 0) for (let d = 0; d < D; d++) u[d] /= nrm;
    eigImagesRaw.push(u);
  }

  // Render eigen-breasts: bipolar map → red / blue tinted.
  const eigenImages: ImageData[] = eigImagesRaw.map((u) => bipolarToImage(u));

  // Total variance for ratios.
  const totalVar = eigVals.reduce((s, v) => s + v, 0) || 1;
  const variance = eigVals.map((v) => v / totalVar);
  let acc = 0;
  const cumulative = variance.map((v) => (acc += v));

  // Projection: x_proj_k = u_k · (img - mean).
  const project = (img: ImageData) => {
    const out: number[] = [];
    for (let k = 0; k < K; k++) {
      let s = 0;
      for (let p = 0; p < PATCH_W * PATCH_H; p++) {
        const r = img.data[p * 4] / 255 - mean[p * 3];
        const g = img.data[p * 4 + 1] / 255 - mean[p * 3 + 1];
        const b = img.data[p * 4 + 2] / 255 - mean[p * 3 + 2];
        s +=
          eigImagesRaw[k][p * 3] * r +
          eigImagesRaw[k][p * 3 + 1] * g +
          eigImagesRaw[k][p * 3 + 2] * b;
      }
      out.push(s);
    }
    return out;
  };

  // Mean image.
  const meanImage = new ImageData(PATCH_W, PATCH_H);
  for (let p = 0; p < PATCH_W * PATCH_H; p++) {
    meanImage.data[p * 4] = Math.round(mean[p * 3] * 255);
    meanImage.data[p * 4 + 1] = Math.round(mean[p * 3 + 1] * 255);
    meanImage.data[p * 4 + 2] = Math.round(mean[p * 3 + 2] * 255);
    meanImage.data[p * 4 + 3] = 255;
  }

  return { mean: meanImage, eigenImages, variance, cumulative, project };
}

function normalise(v: Float32Array) {
  const n = norm(v);
  if (n > 0) for (let i = 0; i < v.length; i++) v[i] /= n;
}

function norm(v: Float32Array): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

// Render a bipolar eigenvector (centred near 0) as a tinted patch — positive
// channels in pink, negative channels in blue, intensity by magnitude.
function bipolarToImage(u: Float32Array): ImageData {
  const img = new ImageData(PATCH_W, PATCH_H);
  let maxAbs = 0;
  for (const x of u) if (Math.abs(x) > maxAbs) maxAbs = Math.abs(x);
  if (maxAbs === 0) maxAbs = 1;
  for (let p = 0; p < PATCH_W * PATCH_H; p++) {
    const r = u[p * 3] / maxAbs;
    const g = u[p * 3 + 1] / maxAbs;
    const b = u[p * 3 + 2] / maxAbs;
    const score = r + g + b; // bipolar around 0
    // Convert to a colour: positive => pink, negative => blue, magnitude => intensity.
    const t = Math.max(-1, Math.min(1, score / 1.5));
    const pos: [number, number, number] = [236, 72, 153]; // CANCER pink
    const neg: [number, number, number] = [15, 107, 255]; // NO_CANCER blue
    const intensity = Math.pow(Math.abs(t), 0.5);
    const base = 240;
    const col: [number, number, number] = t >= 0
      ? [
          base + (pos[0] - base) * intensity,
          base + (pos[1] - base) * intensity,
          base + (pos[2] - base) * intensity,
        ]
      : [
          base + (neg[0] - base) * intensity,
          base + (neg[1] - base) * intensity,
          base + (neg[2] - base) * intensity,
        ];
    img.data[p * 4] = Math.round(col[0]);
    img.data[p * 4 + 1] = Math.round(col[1]);
    img.data[p * 4 + 2] = Math.round(col[2]);
    img.data[p * 4 + 3] = 255;
  }
  return img;
}

interface EigenBreastsProps {
  image: PatchImage;
}

export function EigenBreasts({ image }: EigenBreastsProps) {
  const [pca, setPca] = useState<PcaResult | null>(null);
  const [selectedImgData, setSelectedImgData] = useState<ImageData | null>(null);
  const meanCanvasRef = useRef<HTMLCanvasElement>(null);

  // Compute PCA once on mount.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const result = await computePca(IMAGE_CATALOG);
      if (!cancelled) setPca(result);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Reload the selected image whenever it changes.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const d = await loadImageData(image.url);
      if (!cancelled) setSelectedImgData(d);
    })();
    return () => {
      cancelled = true;
    };
  }, [image.url]);

  // Draw the mean image.
  useEffect(() => {
    if (!pca || !meanCanvasRef.current) return;
    drawToCanvas(meanCanvasRef.current, pca.mean);
  }, [pca]);

  const coords = useMemo(() => {
    if (!pca || !selectedImgData) return null;
    return pca.project(selectedImgData);
  }, [pca, selectedImgData]);

  return (
    <div className="space-y-6">
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <div>
          <h3 className="font-display text-2xl lg:text-3xl tracking-tight">
            Representation learning — eigen-tissues
          </h3>
          <p className="text-sm text-muted-foreground mt-1 max-w-xl">
            Real PCA over all {IMAGE_CATALOG.length} histology patches. Top-{K} principal
            components reveal the axes along which tissue varies most.
          </p>
        </div>
        <span className="text-xs font-mono text-muted-foreground">
          {pca
            ? `${(pca.cumulative[pca.cumulative.length - 1] * 100).toFixed(1)}% variance captured`
            : "computing PCA…"}
        </span>
      </div>

      <div className="grid lg:grid-cols-[1fr_1.6fr] gap-px bg-foreground/10 border border-foreground/10">
        {/* Mean image + selected projection */}
        <div className="bg-background p-6">
          <div className="text-xs font-mono uppercase tracking-wider text-muted-foreground mb-3">
            Dataset mean
          </div>
          <div className="aspect-square border border-foreground/10 mb-5 overflow-hidden">
            <canvas
              ref={meanCanvasRef}
              className="w-full h-full block [image-rendering:pixelated]"
            />
          </div>

          <div className="text-xs font-mono uppercase tracking-wider text-muted-foreground mb-3">
            Selected patch
          </div>
          <div className="flex items-center gap-3">
            <div className="relative w-20 h-20 border border-foreground/10 overflow-hidden">
              <Image
                src={image.url}
                alt={image.id}
                fill
                sizes="80px"
                unoptimized
                className="object-cover [image-rendering:pixelated]"
              />
            </div>
            <div className="text-xs font-mono">
              <div>{image.id}</div>
              <div className="text-muted-foreground">
                {image.label.replace("_", " ")}
              </div>
            </div>
          </div>

          {coords && (
            <div className="mt-5">
              <div className="text-xs font-mono uppercase tracking-wider text-muted-foreground mb-2">
                Eigen-coordinates
              </div>
              <div className="space-y-1.5">
                {coords.map((c, i) => {
                  const max = Math.max(...coords.map((x) => Math.abs(x))) || 1;
                  const pct = Math.min(100, (Math.abs(c) / max) * 100);
                  const positive = c >= 0;
                  return (
                    <div key={i} className="grid grid-cols-[30px_1fr_60px] items-center gap-2 text-[11px] font-mono">
                      <span className="text-muted-foreground">PC{i + 1}</span>
                      <div className="relative h-2 bg-foreground/5 rounded-full">
                        <span className="absolute inset-y-0 left-1/2 w-px bg-foreground/20" />
                        <span
                          className="absolute top-0 bottom-0 rounded-full"
                          style={{
                            left: positive ? "50%" : `calc(50% - ${pct / 2}%)`,
                            width: `${pct / 2}%`,
                            background: positive ? "#EC4899" : "#0F6BFF",
                            opacity: 0.85,
                          }}
                        />
                      </div>
                      <span className="text-right tabular-nums text-muted-foreground">
                        {c >= 0 ? "+" : ""}
                        {c.toFixed(2)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Eigen-breasts grid */}
        <div className="bg-background p-6">
          <div className="text-xs font-mono uppercase tracking-wider text-muted-foreground mb-3">
            Top-{K} principal components
          </div>
          {pca ? (
            <div className="grid grid-cols-4 gap-4">
              {pca.eigenImages.map((img, i) => (
                <EigenTile
                  key={i}
                  img={img}
                  index={i}
                  variance={pca.variance[i]}
                  cumulative={pca.cumulative[i]}
                />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-4 gap-4">
              {Array.from({ length: K }).map((_, i) => (
                <div key={i} className="aspect-square animate-pulse bg-foreground/5" />
              ))}
            </div>
          )}
          <p className="text-xs text-muted-foreground mt-5 leading-relaxed">
            Pink ⇢ pixel pushes toward malignant cytology · Blue ⇢ pushes toward benign.
            Magnitude encodes how strongly that pixel contributes to the component.
          </p>
        </div>
      </div>
    </div>
  );
}

function EigenTile({
  img,
  index,
  variance,
  cumulative,
}: {
  img: ImageData;
  index: number;
  variance: number;
  cumulative: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    drawToCanvas(ref.current, img);
  }, [img]);
  return (
    <div className="space-y-1.5">
      <div className="aspect-square border border-foreground/10 overflow-hidden">
        <canvas
          ref={ref}
          className="w-full h-full block [image-rendering:pixelated]"
        />
      </div>
      <div className="flex items-baseline justify-between text-[10px] font-mono">
        <span className="font-display text-xs text-foreground">PC{index + 1}</span>
        <span className="text-muted-foreground tabular-nums">
          {(variance * 100).toFixed(1)}%
        </span>
      </div>
      <div className="h-0.5 bg-foreground/5 rounded-full overflow-hidden">
        <div
          className="h-full bg-foreground/40"
          style={{ width: `${cumulative * 100}%` }}
        />
      </div>
    </div>
  );
}

function drawToCanvas(canvas: HTMLCanvasElement, src: ImageData) {
  const cssW = canvas.clientWidth || 96;
  const cssH = canvas.clientHeight || 96;
  const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
  canvas.width = Math.max(1, Math.floor(cssW * dpr));
  canvas.height = Math.max(1, Math.floor(cssH * dpr));
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const off = document.createElement("canvas");
  off.width = src.width;
  off.height = src.height;
  off.getContext("2d")!.putImageData(src, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
}
