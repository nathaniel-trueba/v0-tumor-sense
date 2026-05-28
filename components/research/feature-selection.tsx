"use client";

import { useMemo } from "react";
import { Check, X } from "lucide-react";
import { FEATURES } from "@/lib/breast-cancer/features";

const SELECTED_RANK_CUTOFF = 6;
const BENIGN_COLOR = "#0F6BFF";
const MALIGNANT_COLOR = "#EC4899";

export function FeatureSelectionSection() {
  const ranked = useMemo(
    () => [...FEATURES].sort((a, b) => a.rfeRank - b.rfeRank),
    []
  );
  const maxImportance = Math.max(...ranked.map((f) => f.importance));

  return (
    <section id="feature-selection" className="space-y-10">
      <div>
        <span className="inline-flex items-center gap-3 text-sm font-mono text-muted-foreground mb-6">
          <span className="w-8 h-px bg-foreground/30" />
          Feature selection
        </span>
        <h2 className="text-3xl lg:text-5xl font-display tracking-tight max-w-3xl">
          Recursive feature elimination.
        </h2>
        <p className="mt-4 text-lg text-muted-foreground max-w-3xl leading-relaxed">
          We ran 10-fold RFE with a linear SVM as the base estimator and tracked the
          drop in cross-validated F1 as features were eliminated. Six features survive
          the 0.05 significance cut.
        </p>
      </div>

      <div className="border border-foreground/10">
        {ranked.map((f, idx) => {
          const keep = f.rfeRank <= SELECTED_RANK_CUTOFF;
          const pct = (f.importance / maxImportance) * 100;
          return (
            <div
              key={f.id}
              className="grid grid-cols-[40px_1fr_minmax(140px,1.4fr)_120px_60px] items-center gap-4 px-6 py-4 border-t first:border-t-0 border-foreground/10"
            >
              <div className="font-mono text-xs text-muted-foreground tabular-nums">
                {String(idx + 1).padStart(2, "0")}
              </div>
              <div className="min-w-0">
                <div className="font-display text-base truncate">{f.label}</div>
                <div className="font-mono text-[11px] text-muted-foreground truncate">{f.id}</div>
              </div>
              <div>
                <div className="relative h-2 rounded-full bg-foreground/5 overflow-hidden">
                  <div
                    className="absolute inset-y-0 left-0 rounded-full"
                    style={{
                      width: `${pct}%`,
                      background: keep ? MALIGNANT_COLOR : BENIGN_COLOR,
                      opacity: keep ? 0.85 : 0.45,
                      transition: "width 0.5s cubic-bezier(0.22,1,0.36,1)",
                    }}
                  />
                </div>
                <div className="font-mono text-[10px] text-muted-foreground mt-1">
                  importance {(f.importance * 100).toFixed(0)}%
                </div>
              </div>
              <div className="font-mono text-xs text-muted-foreground">
                rank {f.rfeRank}
              </div>
              <div className="flex justify-end">
                {keep ? (
                  <span
                    className="inline-flex items-center gap-1 text-[11px] font-mono uppercase tracking-wider px-2 py-1 rounded-full"
                    style={{ background: `${MALIGNANT_COLOR}1f`, color: MALIGNANT_COLOR }}
                  >
                    <Check className="w-3 h-3" /> keep
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-[11px] font-mono uppercase tracking-wider px-2 py-1 rounded-full bg-foreground/5 text-muted-foreground">
                    <X className="w-3 h-3" /> drop
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
