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
import type { Network } from "@/lib/breast-cancer/networks";

const CANCER = "#EC4899";
const NO_CANCER = "#0F6BFF";

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div>
      <div className="text-2xl lg:text-3xl font-display tracking-tight tabular-nums">{value}</div>
      <div className="mt-1 text-sm text-muted-foreground">{label}</div>
      {hint && <div className="text-xs text-muted-foreground/70 font-mono mt-0.5">{hint}</div>}
    </div>
  );
}

export function NetworkStats({ network }: { network: Network }) {
  const c = network.confusion;
  const total = c.tp + c.fp + c.fn + c.tn;
  const cells = [
    {
      key: "tn",
      value: c.tn,
      label: "True negative",
      x: 0,
      y: 0,
    },
    {
      key: "fp",
      value: c.fp,
      label: "False positive",
      x: 1,
      y: 0,
    },
    {
      key: "fn",
      value: c.fn,
      label: "False negative",
      x: 0,
      y: 1,
    },
    {
      key: "tp",
      value: c.tp,
      label: "True positive",
      x: 1,
      y: 1,
    },
  ];
  const max = Math.max(...cells.map((cell) => cell.value));

  const learning = useMemo(
    () =>
      network.learningCurve.map((p) => ({
        epoch: p.epoch,
        train: +(p.train * 100).toFixed(2),
        val: +(p.val * 100).toFixed(2),
      })),
    [network]
  );
  const bv = useMemo(
    () =>
      network.biasVariance.map((p) => ({
        complexity: p.complexity,
        train: +(p.train * 100).toFixed(2),
        val: +(p.val * 100).toFixed(2),
      })),
    [network]
  );

  return (
    <div className="space-y-10">
      {/* Top stats grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-foreground/10 border border-foreground/10">
        <div className="bg-background p-6">
          <Stat
            label="Accuracy"
            value={`${(network.accuracy * 100).toFixed(1)}%`}
            hint="held-out 200 patches"
          />
        </div>
        <div className="bg-background p-6">
          <Stat label="F1 score" value={`${(network.f1 * 100).toFixed(1)}%`} />
        </div>
        <div className="bg-background p-6">
          <Stat label="ROC-AUC" value={`${(network.rocAuc * 100).toFixed(1)}%`} />
        </div>
        <div className="bg-background p-6">
          <Stat
            label="Latency"
            value={`${network.latencyMs.toFixed(1)} ms`}
            hint="per patch on CPU"
          />
        </div>
        <div className="bg-background p-6">
          <Stat
            label="Precision"
            value={`${(network.precision * 100).toFixed(1)}%`}
            hint="TP / (TP + FP)"
          />
        </div>
        <div className="bg-background p-6">
          <Stat
            label="Recall"
            value={`${(network.recall * 100).toFixed(1)}%`}
            hint="TP / (TP + FN)"
          />
        </div>
        <div className="bg-background p-6">
          <Stat
            label="Parameters"
            value={`${network.paramsM.toFixed(2)} M`}
            hint={`${network.flopsG.toFixed(3)} GFLOPs`}
          />
        </div>
        <div className="bg-background p-6">
          <Stat
            label="Epochs trained"
            value={network.trainedEpochs.toString()}
            hint="Adam · cosine LR"
          />
        </div>
      </div>

      <div className="grid lg:grid-cols-[1fr_1.2fr_1.2fr] gap-px bg-foreground/10 border border-foreground/10">
        {/* Confusion matrix */}
        <div className="bg-background p-6">
          <div className="flex items-baseline justify-between mb-4">
            <h4 className="font-display text-xl">Confusion matrix</h4>
            <span className="text-xs font-mono text-muted-foreground">n = {total}</span>
          </div>
          <div className="grid grid-cols-[40px_1fr_1fr] grid-rows-[28px_1fr_1fr] gap-1 items-center text-xs font-mono text-muted-foreground">
            <span />
            <span className="text-center">pred ¬</span>
            <span className="text-center">pred ✓</span>
            <span className="-rotate-90 text-center">true ¬</span>
            <ConfusionCell value={c.tn} max={max} kind="tn" />
            <ConfusionCell value={c.fp} max={max} kind="fp" />
            <span className="-rotate-90 text-center">true ✓</span>
            <ConfusionCell value={c.fn} max={max} kind="fn" />
            <ConfusionCell value={c.tp} max={max} kind="tp" />
          </div>
          <div className="mt-5 grid grid-cols-2 gap-3 text-xs">
            <div>
              <span className="text-muted-foreground">Precision</span>
              <div className="font-mono">{(network.precision * 100).toFixed(1)}%</div>
            </div>
            <div>
              <span className="text-muted-foreground">Recall</span>
              <div className="font-mono">{(network.recall * 100).toFixed(1)}%</div>
            </div>
          </div>
        </div>

        {/* Learning curve */}
        <div className="bg-background p-6">
          <div className="flex items-baseline justify-between mb-4">
            <h4 className="font-display text-xl">Learning curve</h4>
            <span className="text-xs font-mono text-muted-foreground">accuracy vs. epoch</span>
          </div>
          <div className="h-56">
            <ResponsiveContainer>
              <LineChart data={learning} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <XAxis
                  dataKey="epoch"
                  tickLine={false}
                  axisLine={{ stroke: "rgba(20,20,20,0.18)" }}
                  fontSize={10}
                />
                <YAxis
                  domain={[60, 100]}
                  tickLine={false}
                  axisLine={false}
                  fontSize={10}
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
                <Line type="monotone" dataKey="train" stroke="#111" strokeWidth={1.6} dot={false} name="train" />
                <Line type="monotone" dataKey="val" stroke={CANCER} strokeWidth={1.6} dot={false} name="val" />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <Legend />
        </div>

        {/* Bias / variance */}
        <div className="bg-background p-6">
          <div className="flex items-baseline justify-between mb-4">
            <h4 className="font-display text-xl">Bias / variance</h4>
            <span className="text-xs font-mono text-muted-foreground">error vs. complexity</span>
          </div>
          <div className="h-56">
            <ResponsiveContainer>
              <LineChart data={bv} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <XAxis
                  dataKey="complexity"
                  tickLine={false}
                  axisLine={{ stroke: "rgba(20,20,20,0.18)" }}
                  fontSize={10}
                />
                <YAxis
                  domain={[0, 18]}
                  tickLine={false}
                  axisLine={false}
                  fontSize={10}
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
                <ReferenceLine y={3} stroke="rgba(20,20,20,0.2)" strokeDasharray="2 4" />
                <Line type="monotone" dataKey="train" stroke={NO_CANCER} strokeWidth={1.6} dot={false} />
                <Line type="monotone" dataKey="val" stroke={CANCER} strokeWidth={1.6} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="flex items-center gap-5 mt-4 text-xs font-mono text-muted-foreground">
            <span className="flex items-center gap-2">
              <span className="w-4 h-px" style={{ background: NO_CANCER }} />
              bias (train)
            </span>
            <span className="flex items-center gap-2">
              <span className="w-4 h-px" style={{ background: CANCER }} />
              variance (val)
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function ConfusionCell({
  value,
  max,
  kind,
}: {
  value: number;
  max: number;
  kind: "tp" | "tn" | "fp" | "fn";
}) {
  const intensity = value / max;
  const isError = kind === "fp" || kind === "fn";
  const color = isError ? CANCER : "#111";
  return (
    <div
      className="aspect-square rounded-md border border-foreground/10 flex flex-col items-center justify-center font-mono text-sm tabular-nums"
      style={{
        background: `rgba(${isError ? "236,72,153" : "17,17,17"}, ${0.05 + intensity * 0.18})`,
        color,
      }}
    >
      <div className="text-xl font-display">{value}</div>
      <div className="text-[10px] uppercase tracking-wider opacity-60">{kind}</div>
    </div>
  );
}

function Legend() {
  return (
    <div className="flex items-center gap-5 mt-4 text-xs font-mono text-muted-foreground">
      <span className="flex items-center gap-2">
        <span className="w-4 h-px bg-foreground" />
        train
      </span>
      <span className="flex items-center gap-2">
        <span className="w-4 h-px" style={{ background: CANCER }} />
        validation
      </span>
    </div>
  );
}
