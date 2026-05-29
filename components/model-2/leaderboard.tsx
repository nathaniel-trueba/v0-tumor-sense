"use client";

import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";
import { NETWORKS, type Network } from "@/lib/breast-cancer/networks";

type SortKey = "accuracy" | "f1" | "rocAuc" | "precision" | "recall" | "latencyMs" | "paramsM";
type SortDir = "asc" | "desc";

const COLS: { key: SortKey; label: string; suffix?: string; betterLower?: boolean }[] = [
  { key: "accuracy", label: "Accuracy", suffix: "%" },
  { key: "f1", label: "F1", suffix: "%" },
  { key: "rocAuc", label: "ROC-AUC", suffix: "%" },
  { key: "precision", label: "Precision", suffix: "%" },
  { key: "recall", label: "Recall", suffix: "%" },
  { key: "latencyMs", label: "Latency", suffix: " ms", betterLower: true },
  { key: "paramsM", label: "Params", suffix: " M", betterLower: true },
];

interface LeaderboardProps {
  selectedId: string;
  onSelect: (network: Network) => void;
}

export function Leaderboard({ selectedId, onSelect }: LeaderboardProps) {
  const [sortKey, setSortKey] = useState<SortKey>("f1");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const sorted = useMemo(() => {
    const cmp = (a: Network, b: Network) => {
      const av = a[sortKey] as number;
      const bv = b[sortKey] as number;
      return sortDir === "desc" ? bv - av : av - bv;
    };
    return [...NETWORKS].sort(cmp);
  }, [sortKey, sortDir]);

  function format(value: number, col: (typeof COLS)[number]) {
    if (col.suffix === "%") return `${(value * 100).toFixed(1)}%`;
    if (col.suffix === " ms") return `${value.toFixed(1)} ms`;
    if (col.suffix === " M") return `${value.toFixed(2)} M`;
    return value.toString();
  }

  function toggleSort(k: SortKey) {
    if (k === sortKey) setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    else {
      setSortKey(k);
      setSortDir("desc");
    }
  }

  // Highlight the best value in each column (after format) so the leaderboard
  // self-narrates which model wins on which dimension.
  const bestPerColumn = useMemo(() => {
    return Object.fromEntries(
      COLS.map((c) => {
        const values = NETWORKS.map((n) => n[c.key] as number);
        const best = c.betterLower ? Math.min(...values) : Math.max(...values);
        return [c.key, best];
      })
    ) as Record<SortKey, number>;
  }, []);

  return (
    <section className="space-y-6">
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <h2 className="font-display text-2xl lg:text-3xl tracking-tight">Benchmark leaderboard</h2>
        <span className="text-xs font-mono text-muted-foreground">
          Sorted by {COLS.find((c) => c.key === sortKey)?.label.toLowerCase()} · click any header
          to re-sort
        </span>
      </div>
      <div className="border border-foreground/10 overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="border-b border-foreground/10 bg-foreground/[0.02]">
            <tr>
              <th className="text-left py-3 px-4 font-mono text-[11px] uppercase tracking-wider text-muted-foreground w-10">
                #
              </th>
              <th className="text-left py-3 px-4 font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
                Architecture
              </th>
              {COLS.map((c) => {
                const isSort = c.key === sortKey;
                return (
                  <th
                    key={c.key}
                    className="text-right py-3 px-4 font-mono text-[11px] uppercase tracking-wider text-muted-foreground"
                  >
                    <button
                      type="button"
                      onClick={() => toggleSort(c.key)}
                      className="inline-flex items-center gap-1 hover:text-foreground transition-colors"
                    >
                      {c.label}
                      {isSort ? (
                        sortDir === "desc" ? (
                          <ArrowDown className="w-3 h-3" />
                        ) : (
                          <ArrowUp className="w-3 h-3" />
                        )
                      ) : (
                        <ArrowUpDown className="w-3 h-3 opacity-40" />
                      )}
                    </button>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {sorted.map((n, idx) => {
              const isSelected = n.id === selectedId;
              return (
                <tr
                  key={n.id}
                  className={`border-b border-foreground/5 last:border-b-0 cursor-pointer transition-colors ${
                    isSelected ? "bg-foreground/[0.04]" : "hover:bg-foreground/[0.02]"
                  }`}
                  onClick={() => onSelect(n)}
                >
                  <td className="py-3 px-4 font-mono text-xs text-muted-foreground tabular-nums">
                    {String(idx + 1).padStart(2, "0")}
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-3">
                      <span
                        className={`w-1.5 h-6 rounded-full ${
                          isSelected ? "bg-[#EC4899]" : "bg-foreground/15"
                        }`}
                      />
                      <div className="min-w-0">
                        <div className="font-display text-base truncate">{n.name}</div>
                        <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                          {n.family}
                        </div>
                      </div>
                    </div>
                  </td>
                  {COLS.map((c) => {
                    const v = n[c.key] as number;
                    const isBest = bestPerColumn[c.key] === v;
                    return (
                      <td
                        key={c.key}
                        className={`text-right py-3 px-4 font-mono text-sm tabular-nums ${
                          isBest ? "text-foreground" : "text-foreground/70"
                        }`}
                      >
                        {format(v, c)}
                        {isBest && (
                          <span
                            className="inline-block ml-1.5 align-middle w-1.5 h-1.5 rounded-full"
                            style={{ background: "#EC4899" }}
                            title="best in column"
                          />
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
