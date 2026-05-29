"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play, RotateCcw } from "lucide-react";
import {
  KERNELS,
  applyKernel,
  blitTo,
  loadImageData,
  maxPool,
  relu,
  tint,
} from "@/lib/breast-cancer/image-ops";
import type { Layer, Network } from "@/lib/breast-cancer/networks";
import type { PatchImage } from "@/lib/breast-cancer/image-dataset";

const CANCER = "#EC4899";
const NO_CANCER = "#0F6BFF";

// Each layer surfaces a set of thumbnails so the user can see what the layer
// "computes". The first item is always the canonical activation; the rest are
// alternate filter views that we surface inside the active-layer detail panel.
interface LayerMaps {
  layerId: string;
  maps: ImageData[];
  shape: [number, number, number];
}

function buildMapsForLayer(
  layer: Layer,
  prev: ImageData,
  prevPrev: ImageData | null,
  index: number
): LayerMaps {
  let primary: ImageData;
  const alternates: ImageData[] = [];

  switch (layer.kind) {
    case "input": {
      primary = prev;
      // R, G, B channel split
      for (const channel of [0, 1, 2]) {
        const c = new ImageData(prev.width, prev.height);
        for (let i = 0; i < prev.data.length; i += 4) {
          c.data[i] = channel === 0 ? prev.data[i] : 0;
          c.data[i + 1] = channel === 1 ? prev.data[i + 1] : 0;
          c.data[i + 2] = channel === 2 ? prev.data[i + 2] : 0;
          c.data[i + 3] = 255;
        }
        alternates.push(c);
      }
      break;
    }
    case "conv": {
      const k1 = applyKernel(prev, KERNELS.sobelX, { mono: true });
      const k2 = applyKernel(prev, KERNELS.sobelY, { mono: true });
      const k3 = applyKernel(prev, KERNELS.blur);
      const k4 = applyKernel(prev, KERNELS.sharpen);
      const k5 = applyKernel(prev, KERNELS.emboss, { mono: true });
      const k6 = applyKernel(prev, KERNELS.diag, { mono: true });
      const candidates = [k1, k2, k3, k4, k5, k6];
      // Pick the "primary" deterministically by layer index so layers don't all
      // show the same edge view.
      primary = candidates[index % candidates.length];
      // Push more alternates so the detail panel has variety.
      alternates.push(
        ...candidates,
        tint(k1, [236, 72, 153]),
        tint(k2, [15, 107, 255]),
        tint(k3, [16, 185, 129])
      );
      break;
    }
    case "pool": {
      primary = maxPool(prev, 2);
      alternates.push(
        maxPool(applyKernel(prev, KERNELS.sobelX, { mono: true }), 2),
        maxPool(applyKernel(prev, KERNELS.sobelY, { mono: true }), 2),
        maxPool(applyKernel(prev, KERNELS.laplacian, { mono: true }), 2),
        maxPool(applyKernel(prev, KERNELS.sharpen), 2)
      );
      break;
    }
    case "residual": {
      const conv = applyKernel(prev, KERNELS.sobelX, { mono: true });
      const reluOut = relu(applyKernel(prev, KERNELS.laplacian));
      // "Identity" path tinted to differentiate from "conv" path.
      primary = relu(conv);
      alternates.push(
        primary,
        reluOut,
        tint(prev, [236, 72, 153], 0.4),
        tint(conv, [15, 107, 255], 0.6)
      );
      break;
    }
    case "depthwise": {
      const r = applyKernel(prev, KERNELS.sobelX, { mono: true });
      const g = applyKernel(prev, KERNELS.sobelY, { mono: true });
      const b = applyKernel(prev, KERNELS.laplacian, { mono: true });
      primary = combineRgb(r, g, b);
      alternates.push(primary, r, g, b);
      break;
    }
    case "bn":
    case "relu":
      primary = relu(prev);
      alternates.push(primary);
      break;
    case "flatten":
      primary = activationBar(layer.shape[2], "single", prevPrev ?? prev);
      alternates.push(primary);
      break;
    case "fc":
      primary = activationBar(layer.shape[2], "fc", prevPrev ?? prev);
      alternates.push(primary);
      break;
    case "softmax":
      primary = softmaxBar(prevPrev ?? prev);
      alternates.push(primary);
      break;
    default:
      primary = prev;
  }

  return { layerId: layer.id, maps: [primary, ...alternates], shape: layer.shape };
}

// Combine 3 single-channel ImageDatas into one RGB ImageData.
function combineRgb(r: ImageData, g: ImageData, b: ImageData): ImageData {
  const w = r.width;
  const h = r.height;
  const out = new ImageData(w, h);
  for (let i = 0; i < out.data.length; i += 4) {
    out.data[i] = r.data[i];
    out.data[i + 1] = g.data[i + 1];
    out.data[i + 2] = b.data[i + 2];
    out.data[i + 3] = 255;
  }
  return out;
}

// Render a 50x50 ImageData representing FC activations. Each "neuron" gets a
// vertical bar based on a pseudo-deterministic activation derived from the
// previous map's mean intensity.
function activationBar(units: number, mode: "single" | "fc", seedSrc: ImageData): ImageData {
  const w = 50;
  const h = 50;
  const out = new ImageData(w, h);
  // Background subtle grey.
  for (let i = 0; i < out.data.length; i += 4) {
    out.data[i] = 245;
    out.data[i + 1] = 245;
    out.data[i + 2] = 245;
    out.data[i + 3] = 255;
  }
  // Derive seed from seedSrc mean.
  let sum = 0;
  const limit = Math.min(seedSrc.data.length, 4096);
  for (let i = 0; i < limit; i += 4) sum += seedSrc.data[i] + seedSrc.data[i + 1] + seedSrc.data[i + 2];
  const seed = sum % 1000;
  const visibleBars = Math.min(units, 32);
  const barW = Math.max(1, Math.floor(w / visibleBars));
  for (let b = 0; b < visibleBars; b++) {
    const activation = (Math.sin(b * 1.7 + seed * 0.01) + 1) / 2; // 0..1
    const barH = Math.max(1, Math.round(activation * (h - 4)));
    const baseY = h - 2 - barH;
    const x0 = b * barW;
    for (let x = 0; x < barW - 1; x++) {
      for (let y = baseY; y < h - 2; y++) {
        const i = (y * w + (x0 + x)) * 4;
        if (mode === "fc") {
          out.data[i] = 20;
          out.data[i + 1] = 20;
          out.data[i + 2] = 20;
        } else {
          out.data[i] = 130;
          out.data[i + 1] = 130;
          out.data[i + 2] = 130;
        }
        out.data[i + 3] = 255;
      }
    }
  }
  return out;
}

function softmaxBar(seedSrc: ImageData): ImageData {
  const w = 50;
  const h = 50;
  const out = new ImageData(w, h);
  for (let i = 0; i < out.data.length; i += 4) {
    out.data[i] = 250;
    out.data[i + 1] = 250;
    out.data[i + 2] = 250;
    out.data[i + 3] = 255;
  }
  let sum = 0;
  const limit = Math.min(seedSrc.data.length, 4096);
  for (let i = 0; i < limit; i += 4) sum += seedSrc.data[i];
  const probCancer = Math.min(0.95, Math.max(0.05, (sum % 100) / 100));
  const probNoCancer = 1 - probCancer;
  const drawBar = (xStart: number, prob: number, color: [number, number, number]) => {
    const barH = Math.round(prob * (h - 8));
    const baseY = h - 4 - barH;
    for (let x = 0; x < 20; x++) {
      for (let y = baseY; y < h - 4; y++) {
        const i = (y * w + (xStart + x)) * 4;
        out.data[i] = color[0];
        out.data[i + 1] = color[1];
        out.data[i + 2] = color[2];
        out.data[i + 3] = 255;
      }
    }
  };
  drawBar(4, probNoCancer, [15, 107, 255]);
  drawBar(26, probCancer, [236, 72, 153]);
  return out;
}

interface ForwardPassVizProps {
  network: Network;
  image: PatchImage;
}

export function ForwardPassViz({ network, image }: ForwardPassVizProps) {
  const [allMaps, setAllMaps] = useState<LayerMaps[]>([]);
  const [activeIndex, setActiveIndex] = useState<number>(network.layers.length - 1);
  const [isPlaying, setIsPlaying] = useState(false);
  const stripRef = useRef<HTMLDivElement>(null);
  const cardRefs = useRef<(HTMLDivElement | null)[]>([]);

  // Compute feature maps whenever the input image or network changes.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const input = await loadImageData(image.url);
        const built: LayerMaps[] = [];
        let prev: ImageData = input;
        let prevPrev: ImageData | null = null;
        network.layers.forEach((layer, idx) => {
          const m = buildMapsForLayer(layer, prev, prevPrev, idx);
          built.push(m);
          prevPrev = prev;
          prev = m.maps[0];
        });
        if (!cancelled) {
          setAllMaps(built);
          setActiveIndex(built.length - 1);
        }
      } catch (e) {
        if (!cancelled) {
          // eslint-disable-next-line no-console
          console.warn("forward pass viz failed to build maps", e);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [image.url, network.id, network.layers]);

  // Animation tick.
  useEffect(() => {
    if (!isPlaying) return;
    setActiveIndex(0);
    const step = (i: number) => {
      setActiveIndex(i);
      cardRefs.current[i]?.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
        inline: "center",
      });
      if (i + 1 < network.layers.length) {
        timer = window.setTimeout(() => step(i + 1), 650);
      } else {
        timer = window.setTimeout(() => setIsPlaying(false), 800);
      }
    };
    let timer = window.setTimeout(() => step(0), 100);
    return () => {
      window.clearTimeout(timer);
    };
  }, [isPlaying, network.layers.length]);

  const activeMaps = allMaps[activeIndex];
  const activeLayer = network.layers[activeIndex];
  const prediction = useMemo(() => network.predict(image), [network, image]);
  const perLayerProb = useMemo(() => {
    const labelIsCancer = prediction.label === "cancer";
    return prediction.perLayerConfidence.map((c) => (labelIsCancer ? c : 1 - c));
  }, [prediction]);

  const finalColor = prediction.label === "cancer" ? CANCER : NO_CANCER;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setIsPlaying((p) => !p)}
            className="flex items-center gap-2 px-4 py-2 rounded-full bg-foreground text-background text-sm font-mono hover:opacity-90"
          >
            {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            {isPlaying ? "pause" : "play forward pass"}
          </button>
          <button
            type="button"
            onClick={() => {
              setIsPlaying(false);
              setActiveIndex(network.layers.length - 1);
            }}
            className="flex items-center gap-2 px-4 py-2 rounded-full border border-foreground/15 text-sm font-mono hover:border-foreground/40"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            jump to output
          </button>
        </div>
        <div className="text-xs font-mono text-muted-foreground">
          step {activeIndex + 1} / {network.layers.length}{" "}
          ·{" "}
          {activeLayer?.name} — out{" "}
          {activeLayer?.shape[0]}×{activeLayer?.shape[1]}×{activeLayer?.shape[2]}
        </div>
      </div>

      <div
        ref={stripRef}
        className="relative overflow-x-auto pb-3 border border-foreground/10"
      >
        <div className="flex items-stretch gap-3 p-4 min-w-max">
          {network.layers.map((layer, idx) => {
            const maps = allMaps[idx];
            const isActive = idx === activeIndex;
            const reached = idx <= activeIndex;
            return (
              <div
                key={layer.id}
                ref={(el) => {
                  cardRefs.current[idx] = el;
                }}
                className={`relative flex flex-col items-stretch gap-2 transition-all duration-300 ${
                  reached ? "opacity-100" : "opacity-50"
                }`}
              >
                <button
                  type="button"
                  onClick={() => {
                    setIsPlaying(false);
                    setActiveIndex(idx);
                  }}
                  className={`relative w-[88px] aspect-square bg-foreground/[0.02] border rounded-md overflow-hidden text-left transition-colors ${
                    isActive
                      ? "border-foreground"
                      : "border-foreground/10 hover:border-foreground/30"
                  }`}
                >
                  {maps ? (
                    <FeatureMapCanvas src={maps.maps[0]} width={88} height={88} />
                  ) : (
                    <div className="w-full h-full animate-pulse bg-foreground/5" />
                  )}
                  {isActive && (
                    <span
                      className="absolute inset-x-0 bottom-0 h-0.5"
                      style={{ background: "#EC4899" }}
                    />
                  )}
                </button>
                <div className="w-[88px] space-y-0.5">
                  <div className="font-display text-xs truncate">{layer.name}</div>
                  <div className="font-mono text-[10px] text-muted-foreground">
                    {layer.shape[0]}×{layer.shape[1]}×{layer.shape[2]}
                  </div>
                  <div className="h-1 rounded-full overflow-hidden bg-foreground/5">
                    <div
                      className="h-full transition-all duration-500"
                      style={{
                        width: `${(perLayerProb[idx] ?? 0.5) * 100}%`,
                        background:
                          (perLayerProb[idx] ?? 0.5) > 0.5 ? CANCER : NO_CANCER,
                        opacity: 0.75,
                      }}
                    />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Active-layer detail panel + final prediction */}
      <div className="grid lg:grid-cols-[1.4fr_1fr] gap-px bg-foreground/10 border border-foreground/10">
        <div className="bg-background p-6">
          <div className="flex items-baseline justify-between mb-4">
            <div>
              <div className="text-xs font-mono text-muted-foreground">
                Layer {activeIndex + 1} of {network.layers.length}
              </div>
              <h4 className="font-display text-2xl mt-1">{activeLayer?.name}</h4>
            </div>
            <div className="text-right">
              <div className="text-xs font-mono text-muted-foreground">tensor</div>
              <div className="font-mono text-base">
                {activeLayer?.shape[0]}×{activeLayer?.shape[1]}×{activeLayer?.shape[2]}
              </div>
            </div>
          </div>
          {activeLayer?.detail && (
            <p className="text-sm text-muted-foreground leading-relaxed mb-4">{activeLayer.detail}</p>
          )}
          <div className="flex flex-wrap gap-3 text-xs font-mono text-muted-foreground mb-5">
            {activeLayer?.receptiveField !== undefined && (
              <span>
                receptive ≈ {activeLayer.receptiveField}px
              </span>
            )}
            {activeLayer?.params !== undefined && <span>{activeLayer.params.toFixed(1)} K params</span>}
            {activeLayer?.flops !== undefined && <span>{activeLayer.flops.toFixed(1)} M FLOPs</span>}
          </div>
          {activeMaps && (
            <div>
              <div className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground mb-2">
                Sample activation maps
              </div>
              <div className="grid grid-cols-4 gap-2">
                {activeMaps.maps.slice(0, 8).map((m, i) => (
                  <div
                    key={i}
                    className="aspect-square border border-foreground/10 bg-foreground/[0.02] overflow-hidden rounded-sm"
                  >
                    <FeatureMapCanvas src={m} />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        <div className="bg-background p-6 flex flex-col">
          <div className="text-xs font-mono text-muted-foreground">Final prediction</div>
          <div className="mt-2 font-display text-4xl lg:text-5xl tracking-tight" style={{ color: finalColor }}>
            {prediction.label === "cancer" ? "Cancer" : "No cancer"}
          </div>
          <div className="text-sm text-muted-foreground mt-1">
            {(prediction.confidence * 100).toFixed(1)}% confidence
          </div>

          <div className="mt-6 space-y-3">
            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span style={{ color: NO_CANCER }}>no cancer</span>
                <span className="text-muted-foreground tabular-nums">
                  {((1 - prediction.probability) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-1.5 bg-foreground/5 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-[width] duration-500"
                  style={{
                    width: `${(1 - prediction.probability) * 100}%`,
                    background: NO_CANCER,
                  }}
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span style={{ color: CANCER }}>cancer</span>
                <span className="text-muted-foreground tabular-nums">
                  {(prediction.probability * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-1.5 bg-foreground/5 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-[width] duration-500"
                  style={{
                    width: `${prediction.probability * 100}%`,
                    background: CANCER,
                  }}
                />
              </div>
            </div>
          </div>

          <div className="mt-auto pt-6 text-xs font-mono text-muted-foreground border-t border-foreground/10">
            Per-layer confidence bars above each layer card show how the model&apos;s certainty rises as the input traverses the network.
          </div>
        </div>
      </div>
    </div>
  );
}

function FeatureMapCanvas({
  src,
  width,
  height,
}: {
  src: ImageData;
  width?: number;
  height?: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const w = width ?? c.clientWidth;
    const h = height ?? c.clientHeight;
    const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
    c.width = Math.max(1, Math.floor(w * dpr));
    c.height = Math.max(1, Math.floor(h * dpr));
    c.style.width = `${w}px`;
    c.style.height = `${h}px`;
    blitTo(c, src);
  }, [src, width, height]);
  return <canvas ref={ref} className="w-full h-full block" />;
}
