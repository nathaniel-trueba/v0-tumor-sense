"use client";

import { useMemo } from "react";
import { FileText, Quote, Sparkles } from "lucide-react";
import { FEATURES } from "@/lib/breast-cancer/features";
import type { SVMModel, SVMPrediction } from "@/lib/breast-cancer/models";

interface ExplanationPanelProps {
  model: SVMModel;
  prediction: SVMPrediction;
  values: number[];
}

// Mock "ground truth" research excerpts. The RAG retrieval is mocked here —
// in production this would be an API call to a vector store backed by clinical
// reference documents.
const GROUND_TRUTH_SNIPPETS = [
  {
    id: "wdbc-1995",
    title: "Wolberg & Mangasarian (1990): Multi-surface method for breast tumor diagnosis",
    quote:
      "Across 569 fine-needle aspirates the mean radius, perimeter, and concave-points features were the most discriminative — together they reach a > 97% cross-validated accuracy with a linear classifier.",
    tags: ["radius_mean", "perimeter_mean", "concave_points_mean"],
  },
  {
    id: "rfe-ucsd-2024",
    title: "DS3 internal — Recursive feature elimination report (Spring 2026)",
    quote:
      "After RFE only six 'mean' features survive the 0.05 significance cut; smoothness and fractal dimension consistently drop out as redundant.",
    tags: ["smoothness_mean", "fractal_dimension_mean"],
  },
  {
    id: "rbf-vs-linear",
    title: "Kernel-comparison study, UCI-WDBC benchmark",
    quote:
      "The RBF kernel narrows the boundary in the high-concavity region where the linear kernel tends to over-classify benign borderline cases.",
    tags: ["concavity_mean", "concave_points_mean"],
  },
  {
    id: "case-A",
    title: "Case A-123 — historical positive prediction (chart match 91%)",
    quote:
      "Inputs in this neighborhood (radius ≈ 17, concavity ≈ 0.15) historically resolved malignant in 41 of 44 retrospective cases.",
    tags: ["radius_mean", "concavity_mean"],
  },
  {
    id: "case-B",
    title: "Case B-002 — historical benign cluster (chart match 87%)",
    quote:
      "Profiles with radius < 13 and concave points < 0.04 cluster strongly inside the benign region of every kernel we evaluated.",
    tags: ["radius_mean", "concave_points_mean"],
  },
];

const RECOMMENDED_FEATURES = new Set([
  "radius_mean",
  "perimeter_mean",
  "concave_points_mean",
  "concavity_mean",
  "area_mean",
  "texture_mean",
]);

interface ExplainerLine {
  text: string;
  weight: "high" | "medium" | "low";
}

function buildSummary(model: SVMModel, prediction: SVMPrediction, values: number[]): ExplainerLine[] {
  const lines: ExplainerLine[] = [];
  const isMalignant = prediction.label === "malignant";
  const confidence = prediction.confidence;

  lines.push({
    text: `Using the ${model.name.toLowerCase()}, your input falls in the ${
      isMalignant ? "malignant" : "benign"
    } region of feature space with ${(confidence * 100).toFixed(1)}% confidence (signed decision ${prediction.decision.toFixed(
      3
    )}).`,
    weight: "high",
  });

  // Closest reference cluster.
  const radius = values[0];
  const concavity = values[6];
  const concavePoints = values[7];
  if (isMalignant && radius > 15 && concavity > 0.1) {
    lines.push({
      text: `This neighborhood (radius ≈ ${radius.toFixed(1)}, concavity ≈ ${concavity.toFixed(
        3
      )}) closely matches Case A-123 — a historical malignant cluster with 91% chart match.`,
      weight: "medium",
    });
  } else if (!isMalignant && radius < 13 && concavePoints < 0.04) {
    lines.push({
      text: `Your radius (${radius.toFixed(
        1
      )}) and concave points (${concavePoints.toFixed(
        3
      )}) sit inside the benign reference cluster B-002 (chart match 87%).`,
      weight: "medium",
    });
  } else {
    lines.push({
      text: `This input lies near the boundary — small movements in concavity or radius can flip the prediction. We recommend reviewing the bias-variance panel below.`,
      weight: "medium",
    });
  }

  // Feature-choice critique. A feature counts as "active" if the user has moved
  // it meaningfully away from the midpoint between the two class means.
  const ACTIVE_TOLERANCE = 0.02; // fraction of the feature range
  const chosen = FEATURES.filter((f, i) => {
    const midpoint = (f.benignMean + f.malignantMean) / 2;
    const range = f.max - f.min;
    return Math.abs(values[i] - midpoint) / range > ACTIVE_TOLERANCE;
  });
  if (chosen.length === 0) {
    lines.push({
      text: `Your input sits at the neutral midpoint of every feature. Try adjusting radius, perimeter, area, or concave points to see how the prediction responds.`,
      weight: "low",
    });
  } else {
    const overlap = chosen.filter((c) => RECOMMENDED_FEATURES.has(c.id)).length;
    const ratio = overlap / chosen.length;
    if (ratio >= 0.6) {
      lines.push({
        text: `${(ratio * 100).toFixed(0)}% of the ${chosen.length} feature${
          chosen.length === 1 ? "" : "s"
        } you are adjusting overlap with our RFE-selected set — the model is operating on its strongest signal.`,
        weight: "low",
      });
    } else {
      lines.push({
        text: `Only ${(ratio * 100).toFixed(0)}% of your ${chosen.length} active feature${
          chosen.length === 1 ? "" : "s"
        } are in our RFE-selected set; consider focusing on radius, perimeter, area, and concave points for sharper predictions.`,
        weight: "low",
      });
    }
  }

  return lines;
}

export function ExplanationPanel({ model, prediction, values }: ExplanationPanelProps) {
  const summary = useMemo(() => buildSummary(model, prediction, values), [model, prediction, values]);

  // Retrieve the snippets most relevant to the prediction.
  const retrieved = useMemo(() => {
    const importantFeatureIds = new Set<string>();
    if (Math.abs(prediction.decision) > 0.4) {
      importantFeatureIds.add("radius_mean");
      importantFeatureIds.add("perimeter_mean");
      importantFeatureIds.add("concave_points_mean");
    } else {
      importantFeatureIds.add("concavity_mean");
      importantFeatureIds.add("smoothness_mean");
    }
    return GROUND_TRUTH_SNIPPETS.filter((s) =>
      s.tags.some((t) => importantFeatureIds.has(t))
    ).slice(0, 3);
  }, [prediction.decision]);

  return (
    <div className="grid lg:grid-cols-[1.1fr_1fr] gap-px bg-foreground/10 border border-foreground/10">
      {/* AI summary */}
      <div className="bg-background p-6">
        <div className="flex items-center gap-2 mb-4">
          <Sparkles className="w-4 h-4" />
          <h4 className="font-display text-xl">Why this prediction?</h4>
          <span className="ml-auto text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded-full bg-foreground/5 text-muted-foreground">
            RAG · mock
          </span>
        </div>
        <ul className="space-y-3">
          {summary.map((s, i) => (
            <li
              key={i}
              className={`text-sm leading-relaxed ${
                s.weight === "high"
                  ? "text-foreground"
                  : s.weight === "medium"
                  ? "text-foreground/80"
                  : "text-muted-foreground"
              }`}
            >
              {s.text}
            </li>
          ))}
        </ul>
        <div className="mt-5 pt-5 border-t border-foreground/10 text-xs text-muted-foreground font-mono">
          Predictions are advisory and require clinical validation before acting on them.
        </div>
      </div>

      {/* Retrieved snippets */}
      <div className="bg-background p-6">
        <div className="flex items-center gap-2 mb-4">
          <FileText className="w-4 h-4" />
          <h4 className="font-display text-xl">Retrieved reference passages</h4>
        </div>
        <div className="space-y-4">
          {retrieved.map((r) => (
            <div key={r.id} className="border border-foreground/10 p-4 rounded-md">
              <div className="text-xs font-mono text-muted-foreground mb-2">{r.id}</div>
              <div className="font-display text-base leading-snug mb-2">{r.title}</div>
              <div className="text-sm text-foreground/80 leading-relaxed flex gap-2">
                <Quote className="w-3.5 h-3.5 shrink-0 mt-1 text-foreground/40" />
                <span>{r.quote}</span>
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                {r.tags.map((t) => (
                  <span
                    key={t}
                    className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-foreground/5 text-muted-foreground"
                  >
                    {t}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
