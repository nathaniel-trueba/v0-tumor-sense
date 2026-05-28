"use client";

import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { FEATURES } from "@/lib/breast-cancer/features";
import { DATASET } from "@/lib/breast-cancer/dataset";
import { MODELS } from "@/lib/breast-cancer/models";

const BENIGN_COLOR = "#0F6BFF";
const MALIGNANT_COLOR = "#EC4899";

function shapData() {
  // Mock mean(|SHAP|) per feature aligned with the RFE rank — high-importance
  // features land first in the chart.
  return [...FEATURES]
    .map((f) => ({
      feature: f.short,
      benign: -f.importance * (f.benignMean < f.malignantMean ? 1 : -1) * 0.6,
      malignant: f.importance * (f.benignMean < f.malignantMean ? 1 : -1) * 0.85,
    }))
    .sort((a, b) => Math.abs(b.malignant) - Math.abs(a.malignant));
}

function misclassificationData() {
  // Plot (decision, confidence) for every dataset point under the RBF model and
  // colour-code by whether the prediction matches the true label.
  const rbf = MODELS.find((m) => m.id === "rbf")!;
  return DATASET.map((d) => {
    const p = rbf.predict(d.values);
    return {
      id: d.id,
      decision: +p.decision.toFixed(3),
      confidence: +(p.confidence * 100).toFixed(1),
      trueLabel: d.label,
      pred: p.label,
      correct: p.label === d.label,
    };
  });
}

const LIME_EXAMPLES = [
  {
    case: "Patient #042",
    label: "malignant",
    confidence: 0.97,
    weights: [
      { feature: "concave_points_mean", weight: 0.34, direction: "malignant" },
      { feature: "perimeter_mean", weight: 0.27, direction: "malignant" },
      { feature: "radius_mean", weight: 0.21, direction: "malignant" },
      { feature: "area_mean", weight: 0.15, direction: "malignant" },
      { feature: "smoothness_mean", weight: -0.05, direction: "benign" },
    ],
  },
  {
    case: "Patient #137",
    label: "benign",
    confidence: 0.92,
    weights: [
      { feature: "radius_mean", weight: -0.28, direction: "benign" },
      { feature: "concave_points_mean", weight: -0.22, direction: "benign" },
      { feature: "compactness_mean", weight: -0.12, direction: "benign" },
      { feature: "texture_mean", weight: 0.07, direction: "malignant" },
      { feature: "symmetry_mean", weight: -0.05, direction: "benign" },
    ],
  },
];

export function MLView() {
  const shap = useMemo(shapData, []);
  const misclass = useMemo(misclassificationData, []);

  return (
    <div className="space-y-12 lg:space-y-16">
      {/* SHAP */}
      <div className="grid lg:grid-cols-[1.2fr_1fr] gap-px bg-foreground/10 border border-foreground/10">
        <div className="bg-background p-6">
          <div className="flex items-baseline justify-between mb-4">
            <h3 className="font-display text-2xl">Global SHAP attributions</h3>
            <span className="text-xs font-mono text-muted-foreground">mean(|SHAP|) · RBF kernel</span>
          </div>
          <div className="h-80">
            <ResponsiveContainer>
              <BarChart
                data={shap}
                layout="vertical"
                margin={{ top: 4, right: 16, bottom: 4, left: 16 }}
                barCategoryGap={6}
              >
                <CartesianGrid strokeOpacity={0.08} horizontal={false} />
                <XAxis
                  type="number"
                  domain={[-1, 1]}
                  tickFormatter={(v) => v.toFixed(1)}
                  fontSize={10}
                  axisLine={{ stroke: "rgba(20,20,20,0.18)" }}
                  tickLine={false}
                  className="font-mono"
                />
                <YAxis
                  type="category"
                  dataKey="feature"
                  width={100}
                  fontSize={10}
                  axisLine={false}
                  tickLine={false}
                  className="font-mono"
                />
                <ReferenceLine x={0} stroke="rgba(20,20,20,0.25)" />
                <Tooltip
                  contentStyle={{
                    background: "rgba(255,255,255,0.95)",
                    border: "1px solid rgba(20,20,20,0.1)",
                    borderRadius: 6,
                    fontFamily: "monospace",
                    fontSize: 12,
                  }}
                />
                <Bar dataKey="benign" stackId="a" fill={BENIGN_COLOR} radius={[2, 0, 0, 2]} />
                <Bar dataKey="malignant" stackId="a" fill={MALIGNANT_COLOR} radius={[0, 2, 2, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-sm text-muted-foreground mt-4 leading-relaxed">
            Negative bars pull the prediction toward benign, positive bars toward malignant.
            Concave-points, perimeter, and radius dominate the global ranking — consistent
            with the original Wolberg/Mangasarian findings.
          </p>
        </div>

        {/* Misclassification risk */}
        <div className="bg-background p-6">
          <div className="flex items-baseline justify-between mb-4">
            <h3 className="font-display text-2xl">Misclassification risk zones</h3>
            <span className="text-xs font-mono text-muted-foreground">decision vs. confidence</span>
          </div>
          <div className="h-80">
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 0 }}>
                <CartesianGrid strokeOpacity={0.08} />
                <XAxis
                  type="number"
                  dataKey="decision"
                  name="decision"
                  domain={[-2, 2]}
                  fontSize={10}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(20,20,20,0.18)" }}
                  className="font-mono"
                  label={{
                    value: "signed decision",
                    position: "bottom",
                    fontSize: 10,
                    fill: "rgba(20,20,20,0.55)",
                  }}
                />
                <YAxis
                  type="number"
                  dataKey="confidence"
                  domain={[50, 100]}
                  fontSize={10}
                  tickLine={false}
                  axisLine={false}
                  className="font-mono"
                  tickFormatter={(v) => `${v}%`}
                />
                <Tooltip
                  cursor={{ strokeOpacity: 0.2 }}
                  contentStyle={{
                    background: "rgba(255,255,255,0.95)",
                    border: "1px solid rgba(20,20,20,0.1)",
                    borderRadius: 6,
                    fontFamily: "monospace",
                    fontSize: 12,
                  }}
                  formatter={(value: number | string, name: string) => {
                    if (name === "confidence") return [`${value}%`, "confidence"];
                    return [value, name];
                  }}
                />
                <ReferenceLine
                  x={0}
                  stroke="rgba(20,20,20,0.4)"
                  strokeDasharray="3 4"
                  label={{
                    value: "boundary",
                    position: "top",
                    fontSize: 10,
                    fill: "rgba(20,20,20,0.45)",
                  }}
                />
                <Scatter data={misclass}>
                  {misclass.map((p) => (
                    <Cell
                      key={p.id}
                      fill={p.trueLabel === "malignant" ? MALIGNANT_COLOR : BENIGN_COLOR}
                      fillOpacity={p.correct ? 0.55 : 1}
                      stroke={p.correct ? "transparent" : "#111"}
                      strokeWidth={p.correct ? 0 : 1.5}
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="flex items-center gap-5 mt-4 text-xs font-mono text-muted-foreground">
            <span className="flex items-center gap-2">
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ background: BENIGN_COLOR, opacity: 0.55 }}
              />
              benign — correct
            </span>
            <span className="flex items-center gap-2">
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ background: MALIGNANT_COLOR, opacity: 0.55 }}
              />
              malignant — correct
            </span>
            <span className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full border border-foreground bg-background" />
              misclassified
            </span>
          </div>
        </div>
      </div>

      {/* LIME-style local explanations */}
      <div className="space-y-6">
        <h3 className="font-display text-2xl lg:text-3xl tracking-tight">
          LIME — local explanations
        </h3>
        <div className="grid lg:grid-cols-2 gap-px bg-foreground/10 border border-foreground/10">
          {LIME_EXAMPLES.map((ex) => {
            const color = ex.label === "malignant" ? MALIGNANT_COLOR : BENIGN_COLOR;
            return (
              <div key={ex.case} className="bg-background p-6">
                <div className="flex items-baseline justify-between mb-5">
                  <div>
                    <div className="font-mono text-xs text-muted-foreground">{ex.case}</div>
                    <div
                      className="font-display text-2xl mt-1"
                      style={{ color }}
                    >
                      Predicted {ex.label}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs font-mono text-muted-foreground">confidence</div>
                    <div className="font-display text-2xl">{(ex.confidence * 100).toFixed(0)}%</div>
                  </div>
                </div>
                <div className="space-y-2.5">
                  {ex.weights.map((w) => {
                    const width = Math.min(100, Math.abs(w.weight) * 220);
                    const right = w.direction === "malignant";
                    const fillColor = right ? MALIGNANT_COLOR : BENIGN_COLOR;
                    return (
                      <div
                        key={w.feature}
                        className="grid grid-cols-[150px_1fr_60px] items-center gap-3 text-xs font-mono"
                      >
                        <span className="text-muted-foreground truncate">{w.feature}</span>
                        <div className="relative h-3 bg-foreground/5 rounded-full overflow-hidden">
                          <div className="absolute inset-y-0 left-1/2 w-px bg-foreground/30" />
                          <div
                            className="absolute top-0 bottom-0 rounded-full"
                            style={{
                              left: right ? "50%" : `calc(50% - ${width / 2}%)`,
                              width: `${width / 2}%`,
                              background: fillColor,
                              opacity: 0.85,
                            }}
                          />
                        </div>
                        <span className="text-right tabular-nums">{w.weight > 0 ? "+" : ""}{w.weight.toFixed(2)}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
