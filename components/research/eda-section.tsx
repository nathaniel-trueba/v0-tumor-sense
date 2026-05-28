"use client";

import { useMemo, useState } from "react";
import { FEATURES, type FeatureId } from "@/lib/breast-cancer/features";
import { DATASET } from "@/lib/breast-cancer/dataset";

const BENIGN_COLOR = "#0F6BFF";
const MALIGNANT_COLOR = "#EC4899";
const BIN_COUNT = 24;

function buildHistogram(featureId: FeatureId) {
  const f = FEATURES.find((x) => x.id === featureId)!;
  const i = FEATURES.indexOf(f);
  const bins = Array.from({ length: BIN_COUNT }, () => ({ benign: 0, malignant: 0 }));
  for (const point of DATASET) {
    const t = (point.values[i] - f.min) / (f.max - f.min);
    const idx = Math.min(BIN_COUNT - 1, Math.max(0, Math.floor(t * BIN_COUNT)));
    bins[idx][point.label]++;
  }
  return { feature: f, bins };
}

interface FeatureDistributionPlotProps {
  featureId: FeatureId;
  large?: boolean;
}

function FeatureDistributionPlot({ featureId, large }: FeatureDistributionPlotProps) {
  const { feature, bins } = useMemo(() => buildHistogram(featureId), [featureId]);
  const maxCount = Math.max(...bins.map((b) => Math.max(b.benign, b.malignant)));
  const width = large ? 640 : 220;
  const height = large ? 260 : 90;
  const padX = large ? 24 : 4;
  const padY = large ? 20 : 6;
  const innerW = width - padX * 2;
  const innerH = height - padY * 2;
  const barW = innerW / BIN_COUNT;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full" preserveAspectRatio="none">
      <g>
        {bins.map((b, i) => {
          const x = padX + i * barW;
          const benignH = (b.benign / maxCount) * innerH;
          const malignantH = (b.malignant / maxCount) * innerH;
          return (
            <g key={i}>
              <rect
                x={x + 0.5}
                y={padY + innerH - benignH}
                width={Math.max(0.5, barW - 1)}
                height={benignH}
                fill={BENIGN_COLOR}
                opacity={0.42}
              />
              <rect
                x={x + 0.5}
                y={padY + innerH - malignantH}
                width={Math.max(0.5, barW - 1)}
                height={malignantH}
                fill={MALIGNANT_COLOR}
                opacity={0.55}
              />
            </g>
          );
        })}
        {/* Baseline */}
        <line
          x1={padX}
          y1={padY + innerH}
          x2={padX + innerW}
          y2={padY + innerH}
          stroke="currentColor"
          strokeOpacity={0.2}
          strokeWidth={0.5}
        />
        {/* Means */}
        {(() => {
          const benignX =
            padX + ((feature.benignMean - feature.min) / (feature.max - feature.min)) * innerW;
          const malignantX =
            padX + ((feature.malignantMean - feature.min) / (feature.max - feature.min)) * innerW;
          return (
            <>
              <line
                x1={benignX}
                x2={benignX}
                y1={padY}
                y2={padY + innerH}
                stroke={BENIGN_COLOR}
                strokeWidth={large ? 1.2 : 0.9}
                strokeDasharray="3 3"
              />
              <line
                x1={malignantX}
                x2={malignantX}
                y1={padY}
                y2={padY + innerH}
                stroke={MALIGNANT_COLOR}
                strokeWidth={large ? 1.2 : 0.9}
                strokeDasharray="3 3"
              />
            </>
          );
        })()}
      </g>
    </svg>
  );
}

export function EDASection() {
  const [focus, setFocus] = useState<FeatureId>("concave_points_mean");

  return (
    <section id="eda" className="space-y-12">
      <div>
        <span className="inline-flex items-center gap-3 text-sm font-mono text-muted-foreground mb-6">
          <span className="w-8 h-px bg-foreground/30" />
          Exploratory data analysis
        </span>
        <h2 className="text-3xl lg:text-5xl font-display tracking-tight max-w-3xl">
          The signal hiding in the histograms.
        </h2>
        <p className="mt-4 text-lg text-muted-foreground max-w-3xl leading-relaxed">
          Each cell below overlays the benign and malignant distributions for one of
          the ten mean-summary features. Hover a card to inspect it; the focused
          view on the right exposes the full distribution and the per-class means.
        </p>
      </div>

      <div className="grid lg:grid-cols-[1.2fr_1fr] gap-px bg-foreground/10 border border-foreground/10">
        {/* Small multiples grid */}
        <div className="bg-background p-4 lg:p-6">
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-3 gap-3">
            {FEATURES.map((f) => (
              <button
                key={f.id}
                type="button"
                onMouseEnter={() => setFocus(f.id)}
                onFocus={() => setFocus(f.id)}
                onClick={() => setFocus(f.id)}
                className={`text-left p-3 border rounded-md transition-all ${
                  focus === f.id
                    ? "border-foreground/40 bg-foreground/[0.03]"
                    : "border-foreground/10 hover:border-foreground/25"
                }`}
              >
                <div className="flex items-baseline justify-between mb-1">
                  <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                    {f.short}
                  </span>
                  <span className="text-[10px] font-mono text-muted-foreground">
                    RFE {f.rfeRank}
                  </span>
                </div>
                <div className="h-16 -mx-1 text-foreground/80">
                  <FeatureDistributionPlot featureId={f.id} />
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Focused histogram */}
        <div className="bg-background p-6 lg:p-8 flex flex-col">
          {(() => {
            const f = FEATURES.find((x) => x.id === focus)!;
            return (
              <>
                <div className="space-y-2">
                  <span className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
                    {f.id}
                  </span>
                  <h3 className="font-display text-3xl">{f.label}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">{f.description}</p>
                </div>

                <div className="flex-1 mt-6 min-h-[200px] text-foreground/85">
                  <FeatureDistributionPlot featureId={f.id} large />
                </div>

                <div className="grid grid-cols-2 gap-px bg-foreground/10 border border-foreground/10 mt-6">
                  <div className="bg-background p-4">
                    <div className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
                      Benign μ ± σ
                    </div>
                    <div
                      className="text-xl font-display mt-1"
                      style={{ color: BENIGN_COLOR }}
                    >
                      {f.benignMean.toFixed(3)}{" "}
                      <span className="text-sm text-muted-foreground">± {f.benignSd.toFixed(3)}</span>
                    </div>
                  </div>
                  <div className="bg-background p-4">
                    <div className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
                      Malignant μ ± σ
                    </div>
                    <div
                      className="text-xl font-display mt-1"
                      style={{ color: MALIGNANT_COLOR }}
                    >
                      {f.malignantMean.toFixed(3)}{" "}
                      <span className="text-sm text-muted-foreground">± {f.malignantSd.toFixed(3)}</span>
                    </div>
                  </div>
                </div>

                <div className="mt-4 flex items-center gap-5 text-xs font-mono text-muted-foreground">
                  <span className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-sm" style={{ background: BENIGN_COLOR, opacity: 0.5 }} />
                    benign
                  </span>
                  <span className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-sm" style={{ background: MALIGNANT_COLOR, opacity: 0.6 }} />
                    malignant
                  </span>
                  <span className="ml-auto">dashed = per-class mean</span>
                </div>
              </>
            );
          })()}
        </div>
      </div>
    </section>
  );
}
