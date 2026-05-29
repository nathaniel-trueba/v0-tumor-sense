"use client";

import { useMemo, useRef, useState, type CSSProperties } from "react";
import Image from "next/image";
import { embeddingsFor, type Network } from "@/lib/breast-cancer/networks";
import { type PatchImage } from "@/lib/breast-cancer/image-dataset";

const CANCER = "#EC4899";
const NO_CANCER = "#0F6BFF";

interface EmbeddingExplorerProps {
  network: Network;
  selected: PatchImage;
  onSelect: (image: PatchImage) => void;
}

export function EmbeddingExplorer({ network, selected, onSelect }: EmbeddingExplorerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<{ image: PatchImage; x: number; y: number } | null>(null);
  const [reduction, setReduction] = useState<"umap" | "pca">("umap");

  const points = useMemo(() => embeddingsFor(network.id), [network.id]);

  // Compute axis extents and project to a normalised [-1,1]^2 then map to screen.
  const { xMin, xMax, yMin, yMax } = useMemo(() => {
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    return {
      xMin: Math.min(...xs),
      xMax: Math.max(...xs),
      yMin: Math.min(...ys),
      yMax: Math.max(...ys),
    };
  }, [points]);

  // Tweak coords for PCA mode (a rotation + slight squeeze for visual variety).
  const projected = useMemo(() => {
    return points.map((p) => {
      const px = (p.x - (xMin + xMax) / 2) / Math.max(0.0001, (xMax - xMin) / 2);
      const py = (p.y - (yMin + yMax) / 2) / Math.max(0.0001, (yMax - yMin) / 2);
      const angle = reduction === "pca" ? Math.PI / 6 : 0;
      const cs = Math.cos(angle);
      const sn = Math.sin(angle);
      const nx = px * cs - py * sn;
      const ny = px * sn + py * cs;
      return { image: p.image, x: nx, y: ny };
    });
  }, [points, xMin, xMax, yMin, yMax, reduction]);

  // Estimate separability by computing class centroid distance / within-class spread.
  const separability = useMemo(() => {
    const cancers = projected.filter((p) => p.image.label === "cancer");
    const benigns = projected.filter((p) => p.image.label === "no_cancer");
    const meanC = mean2(cancers);
    const meanB = mean2(benigns);
    const between = Math.hypot(meanC.x - meanB.x, meanC.y - meanB.y);
    const within =
      avg(cancers.map((p) => Math.hypot(p.x - meanC.x, p.y - meanC.y))) +
      avg(benigns.map((p) => Math.hypot(p.x - meanB.x, p.y - meanB.y)));
    return { score: between / Math.max(0.01, within), meanC, meanB };
  }, [projected]);

  return (
    <div className="space-y-6">
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <div>
          <h3 className="font-display text-2xl lg:text-3xl tracking-tight">Embedding space</h3>
          <p className="text-sm text-muted-foreground mt-1 max-w-xl">
            Penultimate-layer activations projected to 2D. Hover or click any point to surface its patch.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex border border-foreground/15 rounded-md overflow-hidden">
            <button
              type="button"
              onClick={() => setReduction("umap")}
              className={`px-3 py-1 text-xs font-mono ${
                reduction === "umap"
                  ? "bg-foreground text-background"
                  : "bg-background hover:bg-foreground/[0.04]"
              }`}
            >
              UMAP
            </button>
            <button
              type="button"
              onClick={() => setReduction("pca")}
              className={`px-3 py-1 text-xs font-mono ${
                reduction === "pca"
                  ? "bg-foreground text-background"
                  : "bg-background hover:bg-foreground/[0.04]"
              }`}
            >
              PCA
            </button>
          </div>
        </div>
      </div>

      <div
        ref={containerRef}
        className="relative h-[420px] border border-foreground/10 bg-foreground/[0.02] overflow-hidden"
        onMouseLeave={() => setHover(null)}
      >
        <svg width="100%" height="100%" viewBox="-110 -110 220 220">
          {/* Grid */}
          {[-100, -50, 0, 50, 100].map((g) => (
            <g key={g}>
              <line x1={-100} x2={100} y1={g} y2={g} stroke="currentColor" strokeOpacity={0.06} />
              <line y1={-100} y2={100} x1={g} x2={g} stroke="currentColor" strokeOpacity={0.06} />
            </g>
          ))}
          {/* Centroid markers */}
          <circle
            cx={separability.meanB.x * 90}
            cy={-separability.meanB.y * 90}
            r={6}
            fill="none"
            stroke={NO_CANCER}
            strokeWidth={1.4}
            strokeDasharray="2 3"
          />
          <circle
            cx={separability.meanC.x * 90}
            cy={-separability.meanC.y * 90}
            r={6}
            fill="none"
            stroke={CANCER}
            strokeWidth={1.4}
            strokeDasharray="2 3"
          />
          {/* Data points */}
          {projected.map((p) => {
            const sx = p.x * 90;
            const sy = -p.y * 90;
            const isSelected = p.image.id === selected.id;
            const color = p.image.label === "cancer" ? CANCER : NO_CANCER;
            return (
              <g key={p.image.id}>
                <circle
                  cx={sx}
                  cy={sy}
                  r={isSelected ? 4.5 : 3}
                  fill={color}
                  fillOpacity={isSelected ? 1 : 0.55}
                  stroke={isSelected ? "#111" : color}
                  strokeWidth={isSelected ? 1.4 : 0.6}
                  style={{ cursor: "pointer", pointerEvents: "all" }}
                  onMouseEnter={(e) => {
                    const rect = containerRef.current!.getBoundingClientRect();
                    setHover({
                      image: p.image,
                      x: e.clientX - rect.left,
                      y: e.clientY - rect.top,
                    });
                  }}
                  onMouseMove={(e) => {
                    const rect = containerRef.current!.getBoundingClientRect();
                    setHover({
                      image: p.image,
                      x: e.clientX - rect.left,
                      y: e.clientY - rect.top,
                    });
                  }}
                  onClick={() => onSelect(p.image)}
                />
                {isSelected && (
                  <circle
                    cx={sx}
                    cy={sy}
                    r={10}
                    fill="none"
                    stroke="#111"
                    strokeWidth={0.6}
                    opacity={0.5}
                  />
                )}
              </g>
            );
          })}
        </svg>

        {/* Axis labels */}
        <div className="absolute top-2 left-3 text-[10px] font-mono text-muted-foreground">
          dim 1 →
        </div>
        <div className="absolute bottom-2 right-3 text-[10px] font-mono text-muted-foreground">
          ↑ dim 2
        </div>

        {/* Legend + separability */}
        <div className="absolute top-3 right-3 bg-background/85 backdrop-blur-md border border-foreground/10 rounded-lg px-3 py-2 text-[11px] font-mono pointer-events-none">
          <div className="flex items-center gap-2 mb-1">
            <span className="w-2 h-2 rounded-full" style={{ background: NO_CANCER }} />
            no cancer
          </div>
          <div className="flex items-center gap-2 mb-1">
            <span className="w-2 h-2 rounded-full" style={{ background: CANCER }} />
            cancer
          </div>
          <div className="border-t border-foreground/10 mt-1 pt-1 text-muted-foreground">
            sep. score · {separability.score.toFixed(2)}
          </div>
        </div>

        {/* Hover preview */}
        {hover && <EmbeddingTooltip hover={hover} containerWidth={containerRef.current?.clientWidth ?? 0} />}
      </div>
    </div>
  );
}

function EmbeddingTooltip({
  hover,
  containerWidth,
}: {
  hover: { image: PatchImage; x: number; y: number };
  containerWidth: number;
}) {
  const onRight = hover.x + 200 > containerWidth - 12;
  const style: CSSProperties = {
    left: onRight ? hover.x - 175 : hover.x + 12,
    top: Math.max(8, hover.y - 64),
  };
  const color = hover.image.label === "cancer" ? CANCER : NO_CANCER;
  return (
    <div
      className="absolute z-10 bg-background/95 backdrop-blur-md border border-foreground/15 rounded-lg p-2 pointer-events-none shadow-lg flex items-center gap-2"
      style={style}
    >
      <div
        className="relative w-14 h-14 border rounded-sm overflow-hidden shrink-0"
        style={{ borderColor: color }}
      >
        <Image
          src={hover.image.url}
          alt={hover.image.id}
          fill
          sizes="56px"
          unoptimized
          className="object-cover [image-rendering:pixelated]"
        />
      </div>
      <div className="text-[11px] font-mono">
        <div className="text-foreground">{hover.image.id}</div>
        <div className="text-muted-foreground">patient {hover.image.patientId}</div>
        <div style={{ color }}>{hover.image.label.replace("_", " ")}</div>
      </div>
    </div>
  );
}

function mean2(arr: { x: number; y: number }[]) {
  if (arr.length === 0) return { x: 0, y: 0 };
  return {
    x: arr.reduce((s, p) => s + p.x, 0) / arr.length,
    y: arr.reduce((s, p) => s + p.y, 0) / arr.length,
  };
}

function avg(arr: number[]) {
  return arr.length === 0 ? 0 : arr.reduce((s, v) => s + v, 0) / arr.length;
}
