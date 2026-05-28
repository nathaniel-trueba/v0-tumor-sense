"use client";

import { useMemo } from "react";
import {
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SVMModel, SVMPrediction } from "@/lib/breast-cancer/models";

interface ModelStatsProps {
  model: SVMModel;
  prediction: SVMPrediction;
}

function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div>
      <div className="text-3xl lg:text-4xl font-display tracking-tight tabular-nums">{value}</div>
      <div className="mt-1 text-sm text-muted-foreground">{label}</div>
      {hint && <div className="text-xs text-muted-foreground/70 font-mono mt-0.5">{hint}</div>}
    </div>
  );
}

export function ModelStats({ model, prediction }: ModelStatsProps) {
  const distance = Math.abs(prediction.decision);

  const charts = useMemo(() => {
    return {
      learning: model.learningCurve.map((p) => ({
        epoch: p.epoch,
        train: +(p.train * 100).toFixed(2),
        test: +(p.test * 100).toFixed(2),
      })),
      bv: model.biasVariance.map((p) => ({
        complexity: p.complexity,
        train: +(p.train * 100).toFixed(2),
        test: +(p.test * 100).toFixed(2),
      })),
    };
  }, [model]);

  return (
    <div className="space-y-12">
      {/* Performance metrics grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-foreground/10 border border-foreground/10">
        <div className="bg-background p-6">
          <Stat
            label="Accuracy"
            value={`${(model.accuracy * 100).toFixed(1)}%`}
            hint="held-out test split"
          />
        </div>
        <div className="bg-background p-6">
          <Stat
            label="ROC-AUC"
            value={(model.rocAuc * 100).toFixed(1) + "%"}
            hint="area under ROC curve"
          />
        </div>
        <div className="bg-background p-6">
          <Stat
            label="F1 Score"
            value={(model.f1 * 100).toFixed(1) + "%"}
            hint="harmonic precision / recall"
          />
        </div>
        <div className="bg-background p-6">
          <Stat
            label="Prediction latency"
            value={`${model.latencyMs.toFixed(1)} ms`}
            hint="single-sample inference"
          />
        </div>
        <div className="bg-background p-6">
          <Stat
            label="Precision"
            value={(model.precision * 100).toFixed(1) + "%"}
            hint="TP / (TP+FP)"
          />
        </div>
        <div className="bg-background p-6">
          <Stat
            label="Recall"
            value={(model.recall * 100).toFixed(1) + "%"}
            hint="TP / (TP+FN)"
          />
        </div>
        <div className="bg-background p-6">
          <Stat
            label="Support vectors"
            value={model.supportVectors.toString()}
            hint={`trained in ${model.trainingTimeMs} ms`}
          />
        </div>
        <div className="bg-background p-6">
          <Stat
            label="Distance to boundary"
            value={distance.toFixed(3)}
            hint={`malignant prob. ${(prediction.probability * 100).toFixed(1)}%`}
          />
        </div>
      </div>

      {/* Learning curve + bias-variance side by side */}
      <div className="grid lg:grid-cols-2 gap-px bg-foreground/10 border border-foreground/10">
        <div className="bg-background p-6">
          <div className="flex items-baseline justify-between mb-5">
            <h4 className="font-display text-xl">Learning curve</h4>
            <span className="text-xs font-mono text-muted-foreground">accuracy vs. epoch</span>
          </div>
          <div className="h-60">
            <ResponsiveContainer>
              <LineChart data={charts.learning} margin={{ top: 8, right: 12, left: -20, bottom: 0 }}>
                <XAxis
                  dataKey="epoch"
                  tickLine={false}
                  axisLine={{ stroke: "rgba(20,20,20,0.18)" }}
                  fontSize={10}
                  className="font-mono"
                />
                <YAxis
                  domain={[80, 100]}
                  tickLine={false}
                  axisLine={false}
                  fontSize={10}
                  className="font-mono"
                  tickFormatter={(v) => `${v}%`}
                />
                <Tooltip
                  contentStyle={{
                    background: "rgba(255,255,255,0.95)",
                    border: "1px solid rgba(20,20,20,0.1)",
                    borderRadius: 6,
                    fontFamily: "monospace",
                    fontSize: 12,
                  }}
                  formatter={(v: number) => `${v.toFixed(2)}%`}
                />
                <Line
                  type="monotone"
                  dataKey="train"
                  stroke="#111"
                  strokeWidth={1.6}
                  dot={false}
                  name="train"
                />
                <Line
                  type="monotone"
                  dataKey="test"
                  stroke="#EC4899"
                  strokeWidth={1.6}
                  dot={false}
                  name="test"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="flex items-center gap-5 mt-4 text-xs font-mono text-muted-foreground">
            <span className="flex items-center gap-2">
              <span className="w-4 h-px bg-foreground" />
              train
            </span>
            <span className="flex items-center gap-2">
              <span className="w-4 h-px bg-[#EC4899]" />
              test
            </span>
          </div>
        </div>

        <div className="bg-background p-6">
          <div className="flex items-baseline justify-between mb-5">
            <h4 className="font-display text-xl">Bias / variance diagnostic</h4>
            <span className="text-xs font-mono text-muted-foreground">error vs. complexity (C)</span>
          </div>
          <div className="h-60">
            <ResponsiveContainer>
              <LineChart data={charts.bv} margin={{ top: 8, right: 12, left: -20, bottom: 0 }}>
                <XAxis
                  dataKey="complexity"
                  tickLine={false}
                  axisLine={{ stroke: "rgba(20,20,20,0.18)" }}
                  fontSize={10}
                  className="font-mono"
                />
                <YAxis
                  domain={[0, 12]}
                  tickLine={false}
                  axisLine={false}
                  fontSize={10}
                  className="font-mono"
                  tickFormatter={(v) => `${v}%`}
                />
                <Tooltip
                  contentStyle={{
                    background: "rgba(255,255,255,0.95)",
                    border: "1px solid rgba(20,20,20,0.1)",
                    borderRadius: 6,
                    fontFamily: "monospace",
                    fontSize: 12,
                  }}
                  formatter={(v: number) => `${v.toFixed(2)}%`}
                />
                <ReferenceLine
                  y={3}
                  stroke="rgba(20,20,20,0.2)"
                  strokeDasharray="2 4"
                  label={{
                    value: "target err.",
                    position: "right",
                    fontSize: 10,
                    fill: "rgba(20,20,20,0.45)",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="train"
                  stroke="#0F6BFF"
                  strokeWidth={1.6}
                  dot={false}
                  name="bias (train err.)"
                />
                <Line
                  type="monotone"
                  dataKey="test"
                  stroke="#EC4899"
                  strokeWidth={1.6}
                  dot={false}
                  name="variance proxy (cv err.)"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="flex items-center gap-5 mt-4 text-xs font-mono text-muted-foreground">
            <span className="flex items-center gap-2">
              <span className="w-4 h-px bg-[#0F6BFF]" />
              bias (train err.)
            </span>
            <span className="flex items-center gap-2">
              <span className="w-4 h-px bg-[#EC4899]" />
              variance (cv err.)
            </span>
          </div>
        </div>
      </div>

      {/* Hyperparameter card */}
      <div className="border border-foreground/10 p-6">
        <div className="flex items-baseline justify-between mb-4">
          <h4 className="font-display text-xl">Hyperparameters</h4>
          <span className="text-xs font-mono text-muted-foreground">{model.formula}</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          {model.hyperparameters.map((h) => (
            <div key={h.name} className="font-mono text-sm">
              <div className="text-muted-foreground">{h.name}</div>
              <div className="text-foreground">{h.value}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
