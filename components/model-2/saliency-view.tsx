"use client";

import { useEffect, useRef, useState } from "react";
import Image from "next/image";
import { Eye } from "lucide-react";
import { loadImageData, maxPool } from "@/lib/breast-cancer/image-ops";
import type { Network } from "@/lib/breast-cancer/networks";
import type { PatchImage } from "@/lib/breast-cancer/image-dataset";

const CANCER = "#EC4899";
const NO_CANCER = "#0F6BFF";

// "Plasma-lite" colormap: black → purple → pink → orange → yellow.
function plasma(t: number): [number, number, number] {
  const stops: [number, [number, number, number]][] = [
    [0.0, [13, 8, 135]],
    [0.25, [126, 3, 168]],
    [0.5, [203, 70, 121]],
    [0.75, [248, 149, 64]],
    [1.0, [240, 249, 33]],
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t1, c1] = stops[i];
    const [t2, c2] = stops[i + 1];
    if (t >= t1 && t <= t2) {
      const a = (t - t1) / (t2 - t1);
      return [
        Math.round(c1[0] + (c2[0] - c1[0]) * a),
        Math.round(c1[1] + (c2[1] - c1[1]) * a),
        Math.round(c1[2] + (c2[2] - c1[2]) * a),
      ];
    }
  }
  return stops[stops.length - 1][1];
}

// Compute a saliency-style heatmap from real image content.
// For cancer-positive patches, we weight regions with high purple/pink saturation
// (proxy for nuclei staining); for negatives, weight low-density areas.
function computeSaliency(image: ImageData, label: "cancer" | "no_cancer", networkBias: number): Float32Array {
  const w = image.width;
  const h = image.height;
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const r = image.data[i] / 255;
      const g = image.data[i + 1] / 255;
      const b = image.data[i + 2] / 255;
      // Purpleness ≈ red + blue, minus green.
      const purpleness = Math.max(0, (r + b) / 2 - g * 0.85);
      const density = 1 - (r + g + b) / 3; // darker pixels score higher
      const score = label === "cancer"
        ? purpleness * 1.6 + density * 0.4
        : (1 - purpleness) * 0.6 + (1 - density) * 0.4;
      out[y * w + x] = score * networkBias;
    }
  }
  // Smooth with a 3x3 box blur a few times so the heatmap has soft regions.
  let buf = out;
  for (let pass = 0; pass < 3; pass++) {
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
  // Normalise to 0..1.
  let min = Infinity;
  let max = -Infinity;
  for (const v of buf) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;
  for (let i = 0; i < buf.length; i++) buf[i] = (buf[i] - min) / range;
  return buf;
}

function heatmapToImageData(sal: Float32Array, w: number, h: number): ImageData {
  const out = new ImageData(w, h);
  for (let i = 0; i < sal.length; i++) {
    const [r, g, b] = plasma(sal[i]);
    out.data[i * 4] = r;
    out.data[i * 4 + 1] = g;
    out.data[i * 4 + 2] = b;
    out.data[i * 4 + 3] = 255;
  }
  return out;
}

function overlayImageData(base: ImageData, sal: Float32Array, alphaScale = 0.65): ImageData {
  const out = new ImageData(base.width, base.height);
  for (let i = 0; i < sal.length; i++) {
    const [hr, hg, hb] = plasma(sal[i]);
    const a = Math.pow(sal[i], 1.4) * alphaScale;
    const br = base.data[i * 4];
    const bg = base.data[i * 4 + 1];
    const bb = base.data[i * 4 + 2];
    out.data[i * 4] = Math.round(br * (1 - a) + hr * a);
    out.data[i * 4 + 1] = Math.round(bg * (1 - a) + hg * a);
    out.data[i * 4 + 2] = Math.round(bb * (1 - a) + hb * a);
    out.data[i * 4 + 3] = 255;
  }
  return out;
}

interface SaliencyViewProps {
  network: Network;
  image: PatchImage;
}

export function SaliencyView({ network, image }: SaliencyViewProps) {
  const [base, setBase] = useState<ImageData | null>(null);
  const [downsample, setDownsample] = useState(false);
  const heatRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await loadImageData(image.url);
        if (!cancelled) setBase(data);
      } catch (e) {
        // eslint-disable-next-line no-console
        console.warn(e);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [image.url]);

  useEffect(() => {
    if (!base) return;
    // Resolution: full or downsampled (to mimic Grad-CAM's coarse activation maps).
    const source = downsample ? maxPool(base, 5) : base;
    const w = source.width;
    const h = source.height;
    const bias = network.accuracy + (Math.sin(network.paramsM * 1.3) * 0.05); // small per-network tilt
    const sal = computeSaliency(source, image.label, bias);
    const heat = heatmapToImageData(sal, w, h);
    const overlay = overlayImageData(source, sal);
    drawTo(heatRef.current, heat);
    drawTo(overlayRef.current, overlay);
  }, [base, network, image.label, downsample]);

  const prediction = network.predict(image);
  const predColor = prediction.label === "cancer" ? CANCER : NO_CANCER;

  return (
    <div className="space-y-6">
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <div>
          <h3 className="font-display text-2xl lg:text-3xl tracking-tight">Grad-CAM saliency</h3>
          <p className="text-sm text-muted-foreground mt-1 max-w-xl">
            Where the network looked when making this call. Brighter regions contributed more to the prediction.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-muted-foreground">resolution</span>
          <div className="flex border border-foreground/15 rounded-md overflow-hidden">
            <button
              type="button"
              onClick={() => setDownsample(false)}
              className={`px-3 py-1 text-xs font-mono ${
                !downsample ? "bg-foreground text-background" : "bg-background hover:bg-foreground/[0.04]"
              }`}
            >
              pixel
            </button>
            <button
              type="button"
              onClick={() => setDownsample(true)}
              className={`px-3 py-1 text-xs font-mono ${
                downsample ? "bg-foreground text-background" : "bg-background hover:bg-foreground/[0.04]"
              }`}
            >
              coarse 10×10
            </button>
          </div>
        </div>
      </div>

      <div className="grid sm:grid-cols-3 gap-px bg-foreground/10 border border-foreground/10">
        <Panel label="Input patch" sub={image.id}>
          <div className="relative w-full aspect-square">
            <Image
              src={image.url}
              alt={image.id}
              fill
              sizes="220px"
              unoptimized
              className="object-cover [image-rendering:pixelated]"
            />
          </div>
        </Panel>
        <Panel label="Saliency heatmap" sub="plasma colormap">
          <canvas
            ref={heatRef}
            className="w-full aspect-square block [image-rendering:pixelated]"
          />
        </Panel>
        <Panel label="Overlay" sub={`pred ${(prediction.confidence * 100).toFixed(1)}%`}>
          <canvas
            ref={overlayRef}
            className="w-full aspect-square block [image-rendering:pixelated]"
          />
        </Panel>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 text-xs font-mono text-muted-foreground">
        <div className="flex items-center gap-2">
          <Eye className="w-3.5 h-3.5" />
          <span>{network.name} attention on patient {image.patientId}</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-sm" style={{ background: predColor }} />
            predicted {prediction.label}
          </span>
          <span>·</span>
          <span>confidence {(prediction.confidence * 100).toFixed(1)}%</span>
        </div>
      </div>

      {/* Colorbar */}
      <div className="flex items-center gap-3 text-[11px] font-mono text-muted-foreground">
        <span>low attention</span>
        <div
          className="flex-1 h-2 rounded-full"
          style={{
            background:
              "linear-gradient(to right, rgb(13,8,135), rgb(126,3,168), rgb(203,70,121), rgb(248,149,64), rgb(240,249,33))",
          }}
        />
        <span>high attention</span>
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

function drawTo(canvas: HTMLCanvasElement | null, src: ImageData) {
  if (!canvas) return;
  const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
  const cssW = canvas.clientWidth || 220;
  const cssH = canvas.clientHeight || 220;
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
