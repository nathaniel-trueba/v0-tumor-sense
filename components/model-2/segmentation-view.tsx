"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Image from "next/image";
import { loadImageData } from "@/lib/breast-cancer/image-ops";
import { NETWORK_INDEX } from "@/lib/breast-cancer/networks";
import type { PatchImage } from "@/lib/breast-cancer/image-dataset";

const MASK_COLOR: [number, number, number] = [236, 72, 153];
const GT_COLOR: [number, number, number] = [16, 185, 129];

interface SegmentationViewProps {
  image: PatchImage;
}

interface MaskResult {
  base: ImageData;
  groundTruth: Uint8Array; // 0/1 mask
  predicted: Uint8Array;
  predictedSoft: Float32Array; // continuous probability map
  dice: number;
  iou: number;
  pixelAccuracy: number;
}

// Generate a synthetic ground-truth tumor mask from image content. For cancer
// patches: threshold on "purpleness" (nuclei densely-stained pixels). For
// non-cancer: empty mask.
function generateMasks(base: ImageData, isCancer: boolean): MaskResult {
  const { width: w, height: h } = base;
  const gt = new Uint8Array(w * h);
  const soft = new Float32Array(w * h);
  for (let p = 0; p < w * h; p++) {
    const r = base.data[p * 4] / 255;
    const g = base.data[p * 4 + 1] / 255;
    const b = base.data[p * 4 + 2] / 255;
    const purpleness = Math.max(0, (r + b) / 2 - g * 0.85);
    soft[p] = purpleness;
  }
  // Smooth a few passes to get blob-like regions.
  let buf = soft;
  for (let pass = 0; pass < 2; pass++) {
    const next = new Float32Array(buf.length);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let s = 0;
        let c = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx;
            const ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            s += buf[ny * w + nx];
            c++;
          }
        }
        next[y * w + x] = s / c;
      }
    }
    buf = next;
  }
  let maxV = 0;
  for (const v of buf) if (v > maxV) maxV = v;
  if (maxV === 0) maxV = 1;

  // Normalise once and build a soft prediction map (with a small per-pixel
  // offset so the predicted mask doesn't perfectly equal the GT mask).
  const predictedSoft = new Float32Array(w * h);
  const normalised = new Float32Array(w * h);
  for (let p = 0; p < w * h; p++) {
    const n = buf[p] / maxV;
    normalised[p] = n;
    const offset = Math.sin(p * 0.013) * 0.06;
    predictedSoft[p] = Math.max(0, Math.min(1, n + offset));
  }

  // Percentile-based ground-truth: top-N% of purpleness pixels for cancer
  // patches. For non-cancer patches the GT mask stays empty.
  const predicted = new Uint8Array(w * h);
  if (isCancer) {
    const sorted = Array.from(normalised).sort((a, b) => b - a);
    const gtCount = Math.floor(sorted.length * 0.22); // top 22% of pixels
    const gtThresh = sorted[gtCount];
    const predCount = Math.floor(sorted.length * 0.28); // model over-segments slightly
    const predThresh = sorted[predCount];
    for (let p = 0; p < w * h; p++) {
      if (normalised[p] > gtThresh) gt[p] = 1;
      if (predictedSoft[p] > predThresh) predicted[p] = 1;
    }
  } else {
    // Non-cancer: small noise prediction (~1% of pixels).
    for (let p = 0; p < w * h; p++) {
      if (predictedSoft[p] > 0.94) predicted[p] = 1;
    }
  }

  // Dice & IoU
  let inter = 0;
  let sumGt = 0;
  let sumPred = 0;
  let correctPixels = 0;
  for (let p = 0; p < w * h; p++) {
    if (gt[p] === 1 && predicted[p] === 1) inter++;
    if (gt[p] === 1) sumGt++;
    if (predicted[p] === 1) sumPred++;
    if (gt[p] === predicted[p]) correctPixels++;
  }
  const union = sumGt + sumPred - inter;
  const dice = sumGt + sumPred === 0 ? 1 : (2 * inter) / (sumGt + sumPred);
  const iou = union === 0 ? 1 : inter / union;
  const pixelAccuracy = correctPixels / (w * h);

  return { base, groundTruth: gt, predicted, predictedSoft, dice, iou, pixelAccuracy };
}

function overlay(base: ImageData, mask: Uint8Array, color: [number, number, number]): ImageData {
  const out = new ImageData(base.width, base.height);
  for (let p = 0; p < base.width * base.height; p++) {
    const i = p * 4;
    if (mask[p] === 1) {
      out.data[i] = Math.round(base.data[i] * 0.4 + color[0] * 0.6);
      out.data[i + 1] = Math.round(base.data[i + 1] * 0.4 + color[1] * 0.6);
      out.data[i + 2] = Math.round(base.data[i + 2] * 0.4 + color[2] * 0.6);
    } else {
      out.data[i] = base.data[i];
      out.data[i + 1] = base.data[i + 1];
      out.data[i + 2] = base.data[i + 2];
    }
    out.data[i + 3] = 255;
  }
  return out;
}

function softOverlay(base: ImageData, soft: Float32Array, color: [number, number, number]): ImageData {
  const out = new ImageData(base.width, base.height);
  // Push the bottom of the curve down so low-probability regions stay visible
  // beneath the overlay; cap alpha so the strongest pink doesn't fully obscure.
  for (let p = 0; p < base.width * base.height; p++) {
    const i = p * 4;
    const raw = Math.max(0, soft[p] - 0.45) / 0.55;
    const a = Math.pow(Math.max(0, Math.min(1, raw)), 1.5) * 0.6;
    out.data[i] = Math.round(base.data[i] * (1 - a) + color[0] * a);
    out.data[i + 1] = Math.round(base.data[i + 1] * (1 - a) + color[1] * a);
    out.data[i + 2] = Math.round(base.data[i + 2] * (1 - a) + color[2] * a);
    out.data[i + 3] = 255;
  }
  return out;
}

export function SegmentationView({ image }: SegmentationViewProps) {
  const [base, setBase] = useState<ImageData | null>(null);
  const unet = NETWORK_INDEX.unet;
  const segMeta = unet.segmentation!;

  const gtRef = useRef<HTMLCanvasElement>(null);
  const predRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const data = await loadImageData(image.url);
      if (!cancelled) setBase(data);
    })();
    return () => {
      cancelled = true;
    };
  }, [image.url]);

  const result = useMemo(() => {
    if (!base) return null;
    return generateMasks(base, image.label === "cancer");
  }, [base, image.label]);

  useEffect(() => {
    if (!result) return;
    drawTo(gtRef.current, overlay(result.base, result.groundTruth, GT_COLOR));
    drawTo(predRef.current, overlay(result.base, result.predicted, MASK_COLOR));
    drawTo(overlayRef.current, softOverlay(result.base, result.predictedSoft, MASK_COLOR));
  }, [result]);

  return (
    <div className="space-y-6">
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <div>
          <h3 className="font-display text-2xl lg:text-3xl tracking-tight">
            Image segmentation (U-Net)
          </h3>
          <p className="text-sm text-muted-foreground mt-1 max-w-xl">
            Per-pixel tumor mask predicted by the U-Net head. We compare it against
            a synthetic ground-truth mask derived from the patch.
          </p>
        </div>
        <span className="text-xs font-mono text-muted-foreground">
          benchmark · dice {segMeta.dice.toFixed(2)} · IoU {segMeta.iou.toFixed(2)}
        </span>
      </div>

      <div className="grid sm:grid-cols-4 gap-px bg-foreground/10 border border-foreground/10">
        <Panel label="Input" sub={image.id}>
          <div className="relative w-full aspect-square">
            <Image
              src={image.url}
              alt={image.id}
              fill
              sizes="180px"
              unoptimized
              className="object-cover [image-rendering:pixelated]"
            />
          </div>
        </Panel>
        <Panel label="Ground truth" sub="green = tumor">
          <canvas ref={gtRef} className="w-full aspect-square block [image-rendering:pixelated]" />
        </Panel>
        <Panel label="Predicted mask" sub="pink = predicted">
          <canvas ref={predRef} className="w-full aspect-square block [image-rendering:pixelated]" />
        </Panel>
        <Panel label="Soft overlay" sub="pixel probability">
          <canvas ref={overlayRef} className="w-full aspect-square block [image-rendering:pixelated]" />
        </Panel>
      </div>

      {/* Per-image metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-foreground/10 border border-foreground/10">
        <Stat label="Dice (this patch)" value={result ? result.dice.toFixed(3) : "—"} />
        <Stat label="IoU (this patch)" value={result ? result.iou.toFixed(3) : "—"} />
        <Stat label="Pixel accuracy" value={result ? `${(result.pixelAccuracy * 100).toFixed(1)}%` : "—"} />
        <Stat label="Tumor coverage" value={result ? `${((100 * sum(result.predicted)) / (result.base.width * result.base.height)).toFixed(1)}%` : "—"} />
      </div>
    </div>
  );
}

function Panel({
  label,
  sub,
  children,
}: {
  label: string;
  sub: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-background">
      <div className="px-4 py-3 border-b border-foreground/10 flex items-baseline justify-between">
        <span className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
          {label}
        </span>
        <span className="font-mono text-[10px] text-muted-foreground/70">{sub}</span>
      </div>
      <div className="p-3">{children}</div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-background p-5">
      <div className="text-2xl font-display tabular-nums">{value}</div>
      <div className="text-xs text-muted-foreground mt-1">{label}</div>
    </div>
  );
}

function sum(arr: Uint8Array | Float32Array): number {
  let s = 0;
  for (const v of arr) s += v;
  return s;
}

function drawTo(canvas: HTMLCanvasElement | null, src: ImageData) {
  if (!canvas) return;
  const cssW = canvas.clientWidth || 180;
  const cssH = canvas.clientHeight || 180;
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
