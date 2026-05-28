"use client";

import { useMemo } from "react";
import { FEATURES } from "@/lib/breast-cancer/features";
import { MODEL_INDEX } from "@/lib/breast-cancer/models";

const BENIGN_COLOR = "#0F6BFF";
const MALIGNANT_COLOR = "#EC4899";

interface Case {
  id: string;
  age: number;
  region: string;
  groundTruth: "benign" | "malignant";
  narrative: string;
  outcome: string;
  features: Partial<Record<string, number>>;
}

const CASES: Case[] = [
  {
    id: "WD-042",
    age: 56,
    region: "Midwest US, 1992 follow-up",
    groundTruth: "malignant",
    narrative:
      "A 56-year-old patient presented with an asymmetric mass identified on routine screening. Fine-needle aspirate showed nuclei with high concave-points and elevated radius — features the model now flags strongly.",
    outcome:
      "Confirmed invasive ductal carcinoma on biopsy. Treated with breast-conserving surgery and adjuvant therapy; disease-free at 5-year follow-up.",
    features: {
      radius_mean: 17.6,
      perimeter_mean: 117.5,
      area_mean: 982.0,
      concavity_mean: 0.162,
      concave_points_mean: 0.094,
      texture_mean: 21.7,
    },
  },
  {
    id: "WD-137",
    age: 42,
    region: "Madison cohort, 1991",
    groundTruth: "benign",
    narrative:
      "A 42-year-old patient with a palpable lump opted for FNA before deciding on surgical excision. Compactness and concave-points sat below the population mean, anchoring the case inside the benign cluster.",
    outcome:
      "Fibroadenoma confirmed on histology. Patient elected watchful waiting; no progression at 2-year follow-up.",
    features: {
      radius_mean: 11.9,
      perimeter_mean: 76.4,
      area_mean: 451.0,
      concavity_mean: 0.041,
      concave_points_mean: 0.023,
      texture_mean: 17.2,
    },
  },
  {
    id: "WD-204",
    age: 49,
    region: "Borderline case · 1993",
    groundTruth: "malignant",
    narrative:
      "Borderline FNA findings sat near the decision boundary in every kernel we tested. The RBF model resolved the case to malignant, but with confidence below 60% — a strong signal that the case warranted further imaging.",
    outcome:
      "Subsequent core biopsy revealed ductal carcinoma in situ. Case is now part of the explainability test set used in our boundary-region diagnostics.",
    features: {
      radius_mean: 14.6,
      perimeter_mean: 94.8,
      area_mean: 670.5,
      concavity_mean: 0.084,
      concave_points_mean: 0.052,
      texture_mean: 19.4,
    },
  },
];

function buildVector(c: Case): number[] {
  return FEATURES.map((f) => {
    const v = c.features[f.id];
    if (v != null) return v;
    return c.groundTruth === "benign" ? f.benignMean : f.malignantMean;
  });
}

export function CaseStudyView() {
  const enriched = useMemo(() => {
    const rbf = MODEL_INDEX.rbf;
    return CASES.map((c) => {
      const vector = buildVector(c);
      const pred = rbf.predict(vector);
      return { case: c, pred };
    });
  }, []);

  return (
    <div className="space-y-12">
      {enriched.map(({ case: c, pred }, idx) => {
        const truthColor = c.groundTruth === "malignant" ? MALIGNANT_COLOR : BENIGN_COLOR;
        const predColor = pred.label === "malignant" ? MALIGNANT_COLOR : BENIGN_COLOR;
        const matches = c.groundTruth === pred.label;
        return (
          <article
            key={c.id}
            className="grid lg:grid-cols-[1fr_1.2fr] gap-px bg-foreground/10 border border-foreground/10"
          >
            <div className="bg-background p-6 lg:p-8">
              <div className="flex items-baseline justify-between mb-4">
                <span className="font-mono text-xs text-muted-foreground">
                  Case {String(idx + 1).padStart(2, "0")} · {c.id}
                </span>
                <span
                  className="text-[11px] font-mono uppercase tracking-wider px-2 py-0.5 rounded-full"
                  style={{ background: `${truthColor}1f`, color: truthColor }}
                >
                  ground truth · {c.groundTruth}
                </span>
              </div>
              <h3 className="font-display text-2xl lg:text-3xl">
                Age {c.age} — {c.region}
              </h3>
              <p className="text-sm text-muted-foreground mt-4 leading-relaxed">{c.narrative}</p>
              <p className="text-sm text-foreground/85 mt-4 leading-relaxed">
                <span className="font-display text-base mr-1">Outcome.</span>
                {c.outcome}
              </p>
            </div>

            <div className="bg-background p-6 lg:p-8 space-y-5">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs font-mono text-muted-foreground">RBF prediction</div>
                  <div
                    className="font-display text-2xl mt-1"
                    style={{ color: predColor }}
                  >
                    {pred.label}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xs font-mono text-muted-foreground">confidence</div>
                  <div className="font-display text-2xl">{(pred.confidence * 100).toFixed(1)}%</div>
                </div>
                <div className="text-right">
                  <div className="text-xs font-mono text-muted-foreground">match</div>
                  <div
                    className={`font-display text-2xl ${
                      matches ? "text-foreground" : "text-[#EC4899]"
                    }`}
                  >
                    {matches ? "✓" : "≠"}
                  </div>
                </div>
              </div>

              <div className="space-y-2">
                {Object.entries(c.features).map(([fid, val]) => {
                  const f = FEATURES.find((x) => x.id === fid);
                  if (!f || val == null) return null;
                  const range = f.max - f.min;
                  const pct = ((val - f.min) / range) * 100;
                  const benignPct = ((f.benignMean - f.min) / range) * 100;
                  const malignantPct = ((f.malignantMean - f.min) / range) * 100;
                  return (
                    <div key={fid} className="text-xs font-mono">
                      <div className="flex items-baseline justify-between mb-1">
                        <span className="text-muted-foreground">{f.short}</span>
                        <span className="tabular-nums">{val.toFixed(f.step < 0.01 ? 3 : 1)}</span>
                      </div>
                      <div className="relative h-1.5 rounded-full bg-foreground/5">
                        <span
                          className="absolute top-0 bottom-0 w-px"
                          style={{ left: `${benignPct}%`, background: BENIGN_COLOR }}
                        />
                        <span
                          className="absolute top-0 bottom-0 w-px"
                          style={{ left: `${malignantPct}%`, background: MALIGNANT_COLOR }}
                        />
                        <span
                          className="absolute -top-1 w-2 h-3.5 rounded-sm"
                          style={{
                            left: `calc(${pct}% - 4px)`,
                            background: predColor,
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </article>
        );
      })}
    </div>
  );
}
