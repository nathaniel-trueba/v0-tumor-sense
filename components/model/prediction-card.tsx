"use client";

import { useEffect, useState } from "react";
import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { FEATURES } from "@/lib/breast-cancer/features";
import type { SVMModel, SVMPrediction } from "@/lib/breast-cancer/models";

const BENIGN_COLOR = "#0F6BFF";
const MALIGNANT_COLOR = "#EC4899";

interface PredictionCardProps {
  model: SVMModel;
  prediction: SVMPrediction;
  values: number[];
}

export function PredictionCard({ model, prediction, values }: PredictionCardProps) {
  const isMalignant = prediction.label === "malignant";
  const color = isMalignant ? MALIGNANT_COLOR : BENIGN_COLOR;
  const [latency, setLatency] = useState<number | null>(null);

  useEffect(() => {
    const start = performance.now();
    // Re-evaluate so we report a fresh latency reading whenever inputs change.
    model.predict(values);
    setLatency(performance.now() - start);
  }, [model, values]);

  // Surface the three features that most pushed the decision in either direction.
  const contributions = values.map((v, i) => {
    const f = FEATURES[i];
    const range = f.max - f.min;
    const fromBenign = (v - f.benignMean) / range;
    const fromMalignant = (v - f.malignantMean) / range;
    const direction = Math.abs(fromBenign) > Math.abs(fromMalignant) ? "malignant" : "benign";
    const magnitude =
      direction === "malignant"
        ? Math.abs(fromBenign) * f.importance
        : Math.abs(fromMalignant) * f.importance;
    return {
      feature: f,
      direction,
      magnitude,
      value: v,
    };
  });
  const top = [...contributions].sort((a, b) => b.magnitude - a.magnitude).slice(0, 3);

  return (
    <div className="border border-foreground/10 bg-background overflow-hidden">
      <div className="grid lg:grid-cols-[1.2fr_1fr]">
        <div className="p-8 border-b lg:border-b-0 lg:border-r border-foreground/10">
          <span
            className="inline-flex items-center gap-2 text-[11px] font-mono uppercase tracking-wider px-2 py-1 rounded-full"
            style={{ background: `${color}1f`, color }}
          >
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
            Prediction
          </span>
          <div
            className="mt-4 text-5xl lg:text-6xl font-display tracking-tight"
            style={{ color }}
          >
            {isMalignant ? "Malignant" : "Benign"}
          </div>
          <p className="mt-3 text-muted-foreground max-w-md">
            Based on {model.name.replace("SVM — ", "")}, the model classifies this profile as{" "}
            <span style={{ color }}>{prediction.label}</span> with{" "}
            <span className="font-mono">{(prediction.confidence * 100).toFixed(1)}%</span> confidence.
          </p>

          {/* Confidence bar */}
          <div className="mt-6">
            <div className="flex items-center justify-between text-xs font-mono mb-2">
              <span style={{ color: BENIGN_COLOR }}>benign</span>
              <span className="text-muted-foreground">
                P(malignant) = {(prediction.probability * 100).toFixed(1)}%
              </span>
              <span style={{ color: MALIGNANT_COLOR }}>malignant</span>
            </div>
            <div className="relative h-2 rounded-full overflow-hidden bg-foreground/10">
              <div
                className="absolute inset-y-0 left-0"
                style={{
                  width: `${prediction.probability * 100}%`,
                  background: `linear-gradient(to right, ${BENIGN_COLOR}, ${MALIGNANT_COLOR})`,
                  transition: "width 0.5s cubic-bezier(0.22, 1, 0.36, 1)",
                }}
              />
              <div
                className="absolute -top-1 w-px h-4 bg-foreground"
                style={{ left: `${prediction.probability * 100}%` }}
              />
            </div>
          </div>
        </div>

        <div className="p-8 space-y-5">
          <div>
            <div className="text-xs font-mono text-muted-foreground">Signed decision</div>
            <div className="text-2xl font-mono tabular-nums mt-1 flex items-baseline gap-2">
              {prediction.decision >= 0 ? (
                <ArrowUpRight className="w-5 h-5" style={{ color: MALIGNANT_COLOR }} />
              ) : (
                <ArrowDownRight className="w-5 h-5" style={{ color: BENIGN_COLOR }} />
              )}
              {prediction.decision.toFixed(4)}
            </div>
          </div>
          <div>
            <div className="text-xs font-mono text-muted-foreground">Distance to boundary</div>
            <div className="text-2xl font-mono tabular-nums mt-1">
              {Math.abs(prediction.decision).toFixed(4)}
            </div>
          </div>
          <div>
            <div className="text-xs font-mono text-muted-foreground">Latency (live)</div>
            <div className="text-2xl font-mono tabular-nums mt-1">
              {latency !== null ? `${latency.toFixed(2)} ms` : "—"}
            </div>
          </div>
        </div>
      </div>

      {/* Top contributing features */}
      <div className="border-t border-foreground/10 p-6">
        <div className="text-xs font-mono text-muted-foreground mb-3">
          Top features pulling this prediction
        </div>
        <div className="grid sm:grid-cols-3 gap-3">
          {top.map((t) => {
            const directionColor = t.direction === "malignant" ? MALIGNANT_COLOR : BENIGN_COLOR;
            const magnitude = Math.min(1, t.magnitude * 4);
            return (
              <div
                key={t.feature.id}
                className="border border-foreground/10 p-3 rounded-md flex items-center gap-3"
              >
                <div
                  className="w-1 h-10 rounded-full"
                  style={{ background: directionColor, opacity: 0.3 + magnitude * 0.7 }}
                />
                <div className="min-w-0 flex-1">
                  <div className="font-display text-base truncate">{t.feature.short}</div>
                  <div className="font-mono text-xs text-muted-foreground">
                    {t.value.toFixed(t.feature.step < 0.01 ? 3 : 1)} ·{" "}
                    <span style={{ color: directionColor }}>→ {t.direction}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
