"use client";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import {
  FEATURE_INDEX,
  FEATURES,
  FeatureId,
  type FeatureMeta,
} from "@/lib/breast-cancer/features";
import { DATASET, type DiagnosisLabel } from "@/lib/breast-cancer/dataset";
import type { SVMModel } from "@/lib/breast-cancer/models";

const BENIGN_COLOR = "#0F6BFF";
const MALIGNANT_COLOR = "#EC4899";

interface FeatureSpaceProps {
  model: SVMModel;
  values: number[];
  xFeature: FeatureId;
  yFeature: FeatureId;
  onXFeatureChange: (f: FeatureId) => void;
  onYFeatureChange: (f: FeatureId) => void;
}

interface ScreenPoint {
  x: number;
  y: number;
}

interface HoverState {
  type: "dataset" | "user";
  label: DiagnosisLabel;
  pointId?: number;
  x: number; // screen
  y: number; // screen
  values: number[]; // feature values (original units)
  decision: number;
  confidence: number;
}

const RESOLUTION = 80; // boundary mesh resolution
const PADDING = 16;

function scaleToScreen(
  value: number,
  domainMin: number,
  domainMax: number,
  rangeMin: number,
  rangeMax: number
): number {
  const t = (value - domainMin) / (domainMax - domainMin);
  return rangeMin + t * (rangeMax - rangeMin);
}

function formatValue(value: number, feature: FeatureMeta): string {
  if (feature.step >= 1) return value.toFixed(0);
  if (feature.step >= 0.1) return value.toFixed(1);
  if (feature.step >= 0.01) return value.toFixed(2);
  if (feature.step >= 0.001) return value.toFixed(3);
  return value.toFixed(4);
}

export function FeatureSpace({
  model,
  values,
  xFeature,
  yFeature,
  onXFeatureChange,
  onYFeatureChange,
}: FeatureSpaceProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 720, h: 540 });
  const [hover, setHover] = useState<HoverState | null>(null);
  const [showGrid, setShowGrid] = useState(true);

  const xi = FEATURE_INDEX[xFeature];
  const yi = FEATURE_INDEX[yFeature];
  const xMeta = FEATURES[xi];
  const yMeta = FEATURES[yi];

  // Resize observer.
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      const e = entries[0];
      const w = Math.max(360, Math.floor(e.contentRect.width));
      const h = Math.max(360, Math.floor(w * 0.7));
      setSize({ w, h });
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  const innerWidth = size.w - PADDING * 2;
  const innerHeight = size.h - PADDING * 2;

  // Project an (x_feature_value, y_feature_value) to screen coordinates.
  function toScreen(fx: number, fy: number): ScreenPoint {
    return {
      x: scaleToScreen(fx, xMeta.min, xMeta.max, PADDING, PADDING + innerWidth),
      y: scaleToScreen(fy, yMeta.min, yMeta.max, PADDING + innerHeight, PADDING),
    };
  }

  // Project from screen back to feature units (used for hover lookups).
  function fromScreen(px: number, py: number): { fx: number; fy: number } {
    const fx = xMeta.min + ((px - PADDING) / innerWidth) * (xMeta.max - xMeta.min);
    const fy = yMeta.min + ((PADDING + innerHeight - py) / innerHeight) * (yMeta.max - yMeta.min);
    return { fx, fy };
  }

  // Pre-projected dataset points for the current axes.
  const projected = useMemo(() => {
    return DATASET.map((d) => ({
      id: d.id,
      label: d.label,
      values: d.values,
      x: d.values[xi],
      y: d.values[yi],
    }));
  }, [xi, yi]);

  // Decision boundary mesh — recomputed when model, sliders, or axes change.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
    canvas.width = size.w * dpr;
    canvas.height = size.h * dpr;
    canvas.style.width = `${size.w}px`;
    canvas.style.height = `${size.h}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);

    const cellW = innerWidth / RESOLUTION;
    const cellH = innerHeight / RESOLUTION;

    // Track decision values so we can scale color intensity.
    const grid: number[] = new Array(RESOLUTION * RESOLUTION);
    let maxAbs = 0;
    const probe = [...values];
    for (let gy = 0; gy < RESOLUTION; gy++) {
      const fy = yMeta.max - ((gy + 0.5) / RESOLUTION) * (yMeta.max - yMeta.min);
      probe[yi] = fy;
      for (let gx = 0; gx < RESOLUTION; gx++) {
        const fx = xMeta.min + ((gx + 0.5) / RESOLUTION) * (xMeta.max - xMeta.min);
        probe[xi] = fx;
        const d = model.decision(probe);
        grid[gy * RESOLUTION + gx] = d;
        if (Math.abs(d) > maxAbs) maxAbs = Math.abs(d);
      }
    }
    if (maxAbs === 0) maxAbs = 1;

    // Fill the mesh.
    for (let gy = 0; gy < RESOLUTION; gy++) {
      for (let gx = 0; gx < RESOLUTION; gx++) {
        const d = grid[gy * RESOLUTION + gx];
        const t = Math.max(-1, Math.min(1, d / maxAbs));
        // Closer to 0 -> closer to background (low opacity).
        const intensity = Math.pow(Math.abs(t), 0.55);
        const color = t >= 0 ? MALIGNANT_COLOR : BENIGN_COLOR;
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.05 + intensity * 0.4;
        ctx.fillRect(
          PADDING + gx * cellW,
          PADDING + gy * cellH,
          cellW + 0.5,
          cellH + 0.5
        );
      }
    }
    ctx.globalAlpha = 1;

    // Draw the actual boundary (d ≈ 0) by tracing isolines between cells.
    ctx.strokeStyle = "rgba(20, 20, 20, 0.55)";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 3]);
    for (let gy = 0; gy < RESOLUTION - 1; gy++) {
      for (let gx = 0; gx < RESOLUTION - 1; gx++) {
        const tl = grid[gy * RESOLUTION + gx];
        const tr = grid[gy * RESOLUTION + gx + 1];
        const bl = grid[(gy + 1) * RESOLUTION + gx];
        const br = grid[(gy + 1) * RESOLUTION + gx + 1];
        const segments: [ScreenPoint, ScreenPoint][] = [];
        const horiz = (
          v1: number,
          v2: number,
          baseX: number,
          y: number
        ): ScreenPoint | null => {
          if ((v1 > 0) === (v2 > 0)) return null;
          const t = v1 / (v1 - v2);
          return { x: baseX + cellW * t, y };
        };
        const vert = (
          v1: number,
          v2: number,
          x: number,
          baseY: number
        ): ScreenPoint | null => {
          if ((v1 > 0) === (v2 > 0)) return null;
          const t = v1 / (v1 - v2);
          return { x, y: baseY + cellH * t };
        };
        const ax = PADDING + gx * cellW;
        const ay = PADDING + gy * cellH;
        const top = horiz(tl, tr, ax, ay);
        const bottom = horiz(bl, br, ax, ay + cellH);
        const left = vert(tl, bl, ax, ay);
        const right = vert(tr, br, ax + cellW, ay);
        const pts = [top, right, bottom, left].filter(
          (p): p is ScreenPoint => p !== null
        );
        if (pts.length === 2) segments.push([pts[0], pts[1]]);
        else if (pts.length === 4) {
          segments.push([pts[0], pts[1]]);
          segments.push([pts[2], pts[3]]);
        }
        ctx.beginPath();
        for (const seg of segments) {
          ctx.moveTo(seg[0].x, seg[0].y);
          ctx.lineTo(seg[1].x, seg[1].y);
        }
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);
  }, [model, values, xi, yi, size.w, size.h, innerWidth, innerHeight, xMeta, yMeta]);

  // User point projection + prediction.
  const userPoint = toScreen(values[xi], values[yi]);
  const userPrediction = useMemo(() => model.predict(values), [model, values]);

  function onMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Hover the user point if close.
    const userDx = mx - userPoint.x;
    const userDy = my - userPoint.y;
    if (Math.sqrt(userDx * userDx + userDy * userDy) < 14) {
      setHover({
        type: "user",
        label: userPrediction.label,
        x: userPoint.x,
        y: userPoint.y,
        values,
        decision: userPrediction.decision,
        confidence: userPrediction.confidence,
      });
      return;
    }

    // Otherwise hover the closest dataset point within 8px.
    let best: typeof projected[number] | null = null;
    let bestDist = 8 * 8;
    for (const p of projected) {
      const sp = toScreen(p.x, p.y);
      const dx = sp.x - mx;
      const dy = sp.y - my;
      const dSq = dx * dx + dy * dy;
      if (dSq < bestDist) {
        bestDist = dSq;
        best = p;
      }
    }
    if (best) {
      const sp = toScreen(best.x, best.y);
      const pred = model.predict(best.values);
      setHover({
        type: "dataset",
        label: best.label,
        pointId: best.id,
        x: sp.x,
        y: sp.y,
        values: best.values,
        decision: pred.decision,
        confidence: pred.confidence,
      });
    } else {
      setHover(null);
    }
  }

  function onMouseLeave() {
    setHover(null);
  }

  const axisOptions = FEATURES.map((f) => (
    <option key={f.id} value={f.id}>
      {f.label}
    </option>
  ));

  return (
    <div className="space-y-4">
      {/* Axis selectors */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 text-sm">
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs text-muted-foreground">X axis</span>
          <select
            value={xFeature}
            onChange={(e) => onXFeatureChange(e.target.value as FeatureId)}
            className="bg-transparent border border-foreground/15 rounded-md px-2 py-1 text-sm focus:outline-none focus:border-foreground/40"
          >
            {axisOptions}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs text-muted-foreground">Y axis</span>
          <select
            value={yFeature}
            onChange={(e) => onYFeatureChange(e.target.value as FeatureId)}
            className="bg-transparent border border-foreground/15 rounded-md px-2 py-1 text-sm focus:outline-none focus:border-foreground/40"
          >
            {axisOptions}
          </select>
        </div>
        <label className="flex items-center gap-2 ml-auto cursor-pointer select-none">
          <input
            type="checkbox"
            checked={showGrid}
            onChange={(e) => setShowGrid(e.target.checked)}
            className="accent-foreground"
          />
          <span className="text-xs font-mono text-muted-foreground">grid</span>
        </label>
      </div>

      <div
        ref={containerRef}
        className="relative border border-foreground/10 bg-background overflow-hidden"
        style={{ height: size.h }}
        onMouseMove={onMouseMove}
        onMouseLeave={onMouseLeave}
      >
        <canvas
          ref={canvasRef}
          className="absolute inset-0 pointer-events-none"
        />

        {/* SVG overlay for points + grid lines + labels */}
        <svg
          width={size.w}
          height={size.h}
          className="absolute inset-0 pointer-events-none"
        >
          {/* Grid lines */}
          {showGrid && (
            <g opacity={0.18}>
              {Array.from({ length: 9 }, (_, i) => i + 1).map((i) => {
                const x = PADDING + (innerWidth * i) / 10;
                return (
                  <line
                    key={`gx-${i}`}
                    x1={x}
                    y1={PADDING}
                    x2={x}
                    y2={PADDING + innerHeight}
                    stroke="currentColor"
                    strokeWidth={0.5}
                  />
                );
              })}
              {Array.from({ length: 9 }, (_, i) => i + 1).map((i) => {
                const y = PADDING + (innerHeight * i) / 10;
                return (
                  <line
                    key={`gy-${i}`}
                    x1={PADDING}
                    y1={y}
                    x2={PADDING + innerWidth}
                    y2={y}
                    stroke="currentColor"
                    strokeWidth={0.5}
                  />
                );
              })}
            </g>
          )}

          {/* Frame */}
          <rect
            x={PADDING}
            y={PADDING}
            width={innerWidth}
            height={innerHeight}
            fill="none"
            stroke="currentColor"
            strokeOpacity={0.18}
            strokeWidth={1}
          />

          {/* Dataset points */}
          {projected.map((p) => {
            const sp = toScreen(p.x, p.y);
            const isMalignant = p.label === "malignant";
            return (
              <circle
                key={p.id}
                cx={sp.x}
                cy={sp.y}
                r={3.4}
                fill={isMalignant ? MALIGNANT_COLOR : BENIGN_COLOR}
                fillOpacity={0.55}
                stroke={isMalignant ? MALIGNANT_COLOR : BENIGN_COLOR}
                strokeOpacity={0.9}
                strokeWidth={0.5}
              />
            );
          })}

          {/* User point */}
          <g>
            <circle
              cx={userPoint.x}
              cy={userPoint.y}
              r={16}
              fill={userPrediction.label === "malignant" ? MALIGNANT_COLOR : BENIGN_COLOR}
              fillOpacity={0.12}
            >
              <animate
                attributeName="r"
                values="14;22;14"
                dur="2.2s"
                repeatCount="indefinite"
              />
              <animate
                attributeName="fill-opacity"
                values="0.18;0.02;0.18"
                dur="2.2s"
                repeatCount="indefinite"
              />
            </circle>
            <circle
              cx={userPoint.x}
              cy={userPoint.y}
              r={8}
              fill="#fff"
              stroke="#111"
              strokeWidth={2}
            />
            <circle
              cx={userPoint.x}
              cy={userPoint.y}
              r={3.2}
              fill={userPrediction.label === "malignant" ? MALIGNANT_COLOR : BENIGN_COLOR}
            />
          </g>

          {/* Axis labels */}
          <text
            x={PADDING + innerWidth / 2}
            y={size.h - 4}
            textAnchor="middle"
            fontSize={11}
            fill="currentColor"
            opacity={0.65}
            className="font-mono"
          >
            {xMeta.label}
            {xMeta.unit ? ` (${xMeta.unit})` : ""}
          </text>
          <text
            x={4}
            y={PADDING + innerHeight / 2}
            textAnchor="middle"
            fontSize={11}
            fill="currentColor"
            opacity={0.65}
            transform={`rotate(-90, 4, ${PADDING + innerHeight / 2})`}
            className="font-mono"
          >
            {yMeta.label}
            {yMeta.unit ? ` (${yMeta.unit})` : ""}
          </text>

          {/* Tick labels */}
          <text
            x={PADDING}
            y={PADDING + innerHeight + 14}
            fontSize={10}
            fill="currentColor"
            opacity={0.55}
            className="font-mono"
          >
            {formatValue(xMeta.min, xMeta)}
          </text>
          <text
            x={PADDING + innerWidth}
            y={PADDING + innerHeight + 14}
            fontSize={10}
            textAnchor="end"
            fill="currentColor"
            opacity={0.55}
            className="font-mono"
          >
            {formatValue(xMeta.max, xMeta)}
          </text>
          <text
            x={PADDING + 4}
            y={PADDING + 12}
            fontSize={10}
            fill="currentColor"
            opacity={0.55}
            className="font-mono"
          >
            {formatValue(yMeta.max, yMeta)}
          </text>
          <text
            x={PADDING + 4}
            y={PADDING + innerHeight}
            fontSize={10}
            fill="currentColor"
            opacity={0.55}
            className="font-mono"
          >
            {formatValue(yMeta.min, yMeta)}
          </text>
        </svg>

        {/* Legend overlay */}
        <div className="absolute top-3 right-3 bg-background/85 backdrop-blur-md border border-foreground/10 rounded-lg px-3 py-2 text-xs font-mono pointer-events-none">
          <div className="flex items-center gap-2 mb-1">
            <span
              className="w-2.5 h-2.5 rounded-full"
              style={{ background: BENIGN_COLOR }}
            />
            <span>Benign region · {DATASET.filter((d) => d.label === "benign").length} pts</span>
          </div>
          <div className="flex items-center gap-2 mb-1">
            <span
              className="w-2.5 h-2.5 rounded-full"
              style={{ background: MALIGNANT_COLOR }}
            />
            <span>Malignant region · {DATASET.filter((d) => d.label === "malignant").length} pts</span>
          </div>
          <div className="flex items-center gap-2 pt-1 border-t border-foreground/10 mt-1">
            <span className="w-2.5 h-2.5 rounded-full border-2 border-foreground bg-background" />
            <span>Your input</span>
          </div>
          <div className="flex items-center gap-2 mt-1">
            <span
              className="w-3 h-px"
              style={{
                background: "linear-gradient(to right, currentColor 50%, transparent 50%)",
                backgroundSize: "4px 1px",
              }}
            />
            <span>Decision boundary</span>
          </div>
        </div>

        {/* Hover tooltip */}
        {hover && (
          <HoverTooltip
            hover={hover}
            xFeature={xFeature}
            yFeature={yFeature}
            size={size}
          />
        )}
      </div>

      {/* Caption */}
      <p className="text-xs text-muted-foreground font-mono">
        Mesh shading = signed decision value · darker = farther from the dashed boundary
        · all other features are held at slider values.
      </p>
    </div>
  );
}

function HoverTooltip({
  hover,
  xFeature,
  yFeature,
  size,
}: {
  hover: HoverState;
  xFeature: FeatureId;
  yFeature: FeatureId;
  size: { w: number; h: number };
}) {
  const tooltipWidth = 240;
  const offset = 14;
  const onRight = hover.x + offset + tooltipWidth > size.w - 8;
  const style: CSSProperties = {
    left: onRight ? hover.x - offset - tooltipWidth : hover.x + offset,
    top: Math.min(size.h - 120, Math.max(8, hover.y - 8)),
    width: tooltipWidth,
  };
  const color = hover.label === "malignant" ? MALIGNANT_COLOR : BENIGN_COLOR;

  const xMeta = FEATURES[FEATURE_INDEX[xFeature]];
  const yMeta = FEATURES[FEATURE_INDEX[yFeature]];
  const xVal = hover.values[FEATURE_INDEX[xFeature]];
  const yVal = hover.values[FEATURE_INDEX[yFeature]];

  return (
    <div
      className="absolute z-20 bg-background/95 backdrop-blur-md border border-foreground/15 rounded-lg p-3 text-xs shadow-lg pointer-events-none"
      style={style}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="font-mono text-muted-foreground">
          {hover.type === "user" ? "Your input" : `Patient #${hover.pointId}`}
        </span>
        <span
          className="px-1.5 py-0.5 rounded text-[10px] font-mono uppercase tracking-wide"
          style={{ background: `${color}1f`, color }}
        >
          {hover.label}
        </span>
      </div>
      <div className="space-y-1 font-mono">
        <div className="flex justify-between">
          <span className="text-muted-foreground">{xMeta.short}</span>
          <span>{formatValue(xVal, xMeta)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">{yMeta.short}</span>
          <span>{formatValue(yVal, yMeta)}</span>
        </div>
        <div className="flex justify-between pt-1.5 mt-1.5 border-t border-foreground/10">
          <span className="text-muted-foreground">decision</span>
          <span>{hover.decision.toFixed(3)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">confidence</span>
          <span>{(hover.confidence * 100).toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}
