"use client";

import { useMemo } from "react";
import { FEATURES } from "@/lib/breast-cancer/features";
import { DATASET } from "@/lib/breast-cancer/dataset";

const BENIGN_COLOR = "#0F6BFF";
const MALIGNANT_COLOR = "#EC4899";

function correlationMatrix(): number[][] {
  const n = FEATURES.length;
  const m = DATASET.length;
  const means = FEATURES.map((_, i) =>
    DATASET.reduce((s, d) => s + d.values[i], 0) / m
  );
  const sds = FEATURES.map((_, i) => {
    const mean = means[i];
    return Math.sqrt(DATASET.reduce((s, d) => s + (d.values[i] - mean) ** 2, 0) / m);
  });
  const mat: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      if (i === j) {
        mat[i][j] = 1;
        continue;
      }
      let s = 0;
      for (let k = 0; k < m; k++) {
        s += (DATASET[k].values[i] - means[i]) * (DATASET[k].values[j] - means[j]);
      }
      const r = s / (m * sds[i] * sds[j]);
      mat[i][j] = r;
      mat[j][i] = r;
    }
  }
  return mat;
}

function tTests() {
  return FEATURES.map((f, i) => {
    const benignVals = DATASET.filter((d) => d.label === "benign").map((d) => d.values[i]);
    const malignantVals = DATASET.filter((d) => d.label === "malignant").map((d) => d.values[i]);
    const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
    const variance = (a: number[], m: number) =>
      a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1);
    const m1 = mean(benignVals);
    const m2 = mean(malignantVals);
    const v1 = variance(benignVals, m1);
    const v2 = variance(malignantVals, m2);
    const n1 = benignVals.length;
    const n2 = malignantVals.length;
    const se = Math.sqrt(v1 / n1 + v2 / n2);
    const t = se === 0 ? 0 : (m2 - m1) / se;
    // Cohen's d
    const pooled = Math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2));
    const d = pooled === 0 ? 0 : (m2 - m1) / pooled;
    // Crude p-value approximation using Welch–Satterthwaite df and a normal cdf.
    const z = Math.abs(t);
    // Abramowitz & Stegun normal CDF approximation:
    const k = 1 / (1 + 0.2316419 * z);
    const px =
      1 -
      (1 / Math.sqrt(2 * Math.PI)) *
        Math.exp(-(z * z) / 2) *
        (0.319381530 * k -
          0.356563782 * k ** 2 +
          1.781477937 * k ** 3 -
          1.821255978 * k ** 4 +
          1.330274429 * k ** 5);
    const p = 2 * (1 - px);
    return {
      feature: f,
      meanBenign: m1,
      meanMalignant: m2,
      cohensD: d,
      t,
      p,
    };
  });
}

export function StatsView() {
  const corr = useMemo(correlationMatrix, []);
  const stats = useMemo(tTests, []);
  const sortedByEffect = useMemo(
    () => [...stats].sort((a, b) => Math.abs(b.cohensD) - Math.abs(a.cohensD)),
    [stats]
  );
  const maxAbsCorr = useMemo(() => {
    let m = 0;
    for (let i = 0; i < corr.length; i++)
      for (let j = 0; j < corr.length; j++)
        if (i !== j) m = Math.max(m, Math.abs(corr[i][j]));
    return m;
  }, [corr]);

  return (
    <div className="space-y-12 lg:space-y-16">
      {/* Correlation heatmap */}
      <div className="grid lg:grid-cols-[1.1fr_1fr] gap-px bg-foreground/10 border border-foreground/10">
        <div className="bg-background p-6">
          <div className="flex items-baseline justify-between mb-4">
            <h3 className="font-display text-2xl">Pearson correlation</h3>
            <span className="text-xs font-mono text-muted-foreground">10 × 10 matrix</span>
          </div>
          <div className="overflow-x-auto">
            <table className="text-[10px] font-mono w-full">
              <thead>
                <tr>
                  <th className="text-left p-1 text-muted-foreground" />
                  {FEATURES.map((f) => (
                    <th
                      key={f.id}
                      className="p-1 text-muted-foreground text-left whitespace-nowrap"
                      style={{ minWidth: 36 }}
                    >
                      <span
                        className="inline-block"
                        style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
                      >
                        {f.short}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {FEATURES.map((rowF, i) => (
                  <tr key={rowF.id}>
                    <td className="text-muted-foreground p-1 whitespace-nowrap">{rowF.short}</td>
                    {FEATURES.map((_, j) => {
                      const r = corr[i][j];
                      const norm = Math.min(1, Math.abs(r));
                      const bg =
                        r >= 0
                          ? `rgba(236, 72, 153, ${norm * 0.85 + 0.05})`
                          : `rgba(15, 107, 255, ${norm * 0.85 + 0.05})`;
                      return (
                        <td
                          key={j}
                          className="p-0 text-center tabular-nums"
                          style={{
                            background: bg,
                            color: norm > 0.55 ? "#fff" : "rgba(20,20,20,0.7)",
                            width: 38,
                            height: 24,
                          }}
                          title={`${rowF.short} × ${FEATURES[j].short}: r = ${r.toFixed(2)}`}
                        >
                          {r.toFixed(2)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex items-center gap-5 mt-5 text-xs font-mono text-muted-foreground">
            <span className="flex items-center gap-2">
              <span className="w-4 h-3 rounded-sm" style={{ background: BENIGN_COLOR, opacity: 0.7 }} />
              negative
            </span>
            <span className="flex items-center gap-2">
              <span className="w-4 h-3 rounded-sm" style={{ background: MALIGNANT_COLOR, opacity: 0.7 }} />
              positive
            </span>
            <span className="ml-auto">|r|max ≈ {maxAbsCorr.toFixed(2)}</span>
          </div>
        </div>

        {/* Significance table */}
        <div className="bg-background p-6">
          <div className="flex items-baseline justify-between mb-4">
            <h3 className="font-display text-2xl">Two-sample tests</h3>
            <span className="text-xs font-mono text-muted-foreground">Welch t · effect size</span>
          </div>
          <div className="overflow-x-auto -mx-2 px-2">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs font-mono text-muted-foreground border-b border-foreground/10">
                  <th className="text-left py-2 pr-3">Feature</th>
                  <th className="text-right py-2 px-3">μ benign</th>
                  <th className="text-right py-2 px-3">μ malignant</th>
                  <th className="text-right py-2 px-3">|t|</th>
                  <th className="text-right py-2 px-3">p</th>
                  <th className="text-right py-2 pl-3">Cohen&apos;s d</th>
                </tr>
              </thead>
              <tbody>
                {sortedByEffect.map((s) => {
                  const sig = s.p < 0.001 ? "***" : s.p < 0.01 ? "**" : s.p < 0.05 ? "*" : "ns";
                  return (
                    <tr
                      key={s.feature.id}
                      className="border-b border-foreground/5 hover:bg-foreground/[0.02]"
                    >
                      <td className="py-2 pr-3 font-mono text-xs truncate max-w-[140px]">
                        {s.feature.id}
                      </td>
                      <td className="text-right tabular-nums font-mono text-xs py-2 px-3">
                        {s.meanBenign.toFixed(2)}
                      </td>
                      <td className="text-right tabular-nums font-mono text-xs py-2 px-3">
                        {s.meanMalignant.toFixed(2)}
                      </td>
                      <td className="text-right tabular-nums font-mono text-xs py-2 px-3">
                        {Math.abs(s.t).toFixed(2)}
                      </td>
                      <td className="text-right tabular-nums font-mono text-xs py-2 px-3">
                        {s.p < 0.0001 ? "<0.0001" : s.p.toFixed(4)}{" "}
                        <span
                          className={`ml-1 ${
                            sig === "ns" ? "text-muted-foreground" : "text-[#EC4899]"
                          }`}
                        >
                          {sig}
                        </span>
                      </td>
                      <td className="text-right tabular-nums font-mono text-xs py-2 pl-3">
                        {s.cohensD.toFixed(2)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-muted-foreground mt-4">
            *** p&lt;0.001 · ** p&lt;0.01 · * p&lt;0.05. P-values computed via Welch&apos;s t and an Abramowitz–Stegun normal approximation.
          </p>
        </div>
      </div>

      {/* Effect-size ranking */}
      <div className="border border-foreground/10 p-6 lg:p-8">
        <div className="flex items-baseline justify-between mb-4">
          <h3 className="font-display text-2xl">Effect size ranking</h3>
          <span className="text-xs font-mono text-muted-foreground">|Cohen&apos;s d|</span>
        </div>
        <div className="space-y-2.5">
          {sortedByEffect.map((s) => {
            const width = Math.min(100, Math.abs(s.cohensD) * 28);
            return (
              <div
                key={s.feature.id}
                className="grid grid-cols-[180px_1fr_70px] items-center gap-3 text-sm"
              >
                <span className="font-mono text-xs truncate">{s.feature.id}</span>
                <div className="h-2.5 bg-foreground/5 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${width}%`,
                      background: s.cohensD >= 0 ? MALIGNANT_COLOR : BENIGN_COLOR,
                      opacity: 0.85,
                      transition: "width 0.4s cubic-bezier(0.22,1,0.36,1)",
                    }}
                  />
                </div>
                <span className="text-right tabular-nums font-mono text-xs">
                  {s.cohensD.toFixed(2)}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
