"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  FileText,
  Quote,
  RefreshCw,
  Sparkles,
  WifiOff,
} from "lucide-react";
import { FEATURES } from "@/lib/breast-cancer/features";
import type { SVMModel, SVMPrediction } from "@/lib/breast-cancer/models";
import {
  API_BASE_URL,
  type ExplainResponseBody,
  type RetrievedPassage,
  explainPrediction,
} from "@/lib/api";

interface ExplanationPanelProps {
  model: SVMModel;
  prediction: SVMPrediction;
  values: number[];
}

// Local fallback corpus used when the FastAPI backend at NEXT_PUBLIC_API_URL is
// unreachable, or when the user hasn't requested an explanation yet.
const FALLBACK_SNIPPETS: RetrievedPassage[] = [
  {
    id: "01_wolberg_mangasarian_1990",
    title: "Wolberg & Mangasarian (1990): Multisurface method for breast tumor diagnosis",
    source: "01_wolberg_mangasarian_1990.md",
    summary:
      "Mean radius, perimeter, and concave-points features dominate the linear separator on the original 569-sample FNA dataset.",
    quote:
      "Across 569 fine-needle aspirates the mean radius, perimeter, and concave-points features were the most discriminative — together they reach > 97% cross-validated accuracy with a linear classifier.",
    tags: ["radius_mean", "perimeter_mean", "concave_points_mean"],
    score: 0,
  },
  {
    id: "04_guyon_rfe_2002",
    title: "Guyon et al. (2002): Recursive Feature Elimination for SVMs",
    source: "04_guyon_rfe_2002.md",
    summary:
      "Six features survive RFE on WDBC: radius_mean, perimeter_mean, area_mean, concavity_mean, concave_points_mean, texture_mean.",
    quote:
      "After RFE only six 'mean' features survive the 0.05 significance cut; smoothness and fractal dimension consistently drop out as redundant.",
    tags: ["radius_mean", "concavity_mean", "smoothness_mean", "fractal_dimension_mean"],
    score: 0,
  },
  {
    id: "03_cortes_vapnik_svm",
    title: "Cortes & Vapnik (1995): Support-Vector Networks",
    source: "03_cortes_vapnik_svm.md",
    summary:
      "The decision function is signed; |f(x)| measures distance to the margin and predicts how stable the call is under perturbation.",
    quote:
      "Inputs whose |f(x)| sits below ~0.4 are borderline — small perturbations of the most important features can flip the prediction.",
    tags: ["decision_boundary", "rbf", "linear"],
    score: 0,
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

/** Deterministic local summary — always renders instantly with no network. */
function buildLocalSummary(
  model: SVMModel,
  prediction: SVMPrediction,
  values: number[]
): ExplainerLine[] {
  const lines: ExplainerLine[] = [];
  const isMalignant = prediction.label === "malignant";
  const confidence = prediction.confidence;

  lines.push({
    text: `Using the ${model.name.toLowerCase()}, your input falls in the ${
      isMalignant ? "malignant" : "benign"
    } region of feature space with ${(confidence * 100).toFixed(1)}% confidence (signed decision ${prediction.decision.toFixed(3)}).`,
    weight: "high",
  });

  const radius = values[0];
  const concavity = values[6];
  const concavePoints = values[7];
  if (isMalignant && radius > 15 && concavity > 0.1) {
    lines.push({
      text: `This neighborhood (radius ≈ ${radius.toFixed(1)}, concavity ≈ ${concavity.toFixed(3)}) closely matches Case A-123 — a historical malignant cluster with 91% chart match.`,
      weight: "medium",
    });
  } else if (!isMalignant && radius < 13 && concavePoints < 0.04) {
    lines.push({
      text: `Your radius (${radius.toFixed(1)}) and concave points (${concavePoints.toFixed(3)}) sit inside the benign reference cluster B-002 (chart match 87%).`,
      weight: "medium",
    });
  } else {
    lines.push({
      text: `This input lies near the boundary — small movements in concavity or radius can flip the prediction. We recommend reviewing the bias-variance panel.`,
      weight: "medium",
    });
  }

  const ACTIVE_TOLERANCE = 0.02;
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
        text: `${(ratio * 100).toFixed(0)}% of the ${chosen.length} feature${chosen.length === 1 ? "" : "s"} you are adjusting overlap with our RFE-selected set — the model is operating on its strongest signal.`,
        weight: "low",
      });
    } else {
      lines.push({
        text: `Only ${(ratio * 100).toFixed(0)}% of your ${chosen.length} active feature${chosen.length === 1 ? "" : "s"} are in our RFE-selected set; consider focusing on radius, perimeter, area, and concave points for sharper predictions.`,
        weight: "low",
      });
    }
  }

  return lines;
}

function buildFeaturePayload(values: number[]): Record<string, number> {
  return Object.fromEntries(FEATURES.map((f, i) => [f.id, values[i]]));
}

function backendKernel(id: SVMModel["kernel"]): "rbf" | "linear" | "poly" | "sigmoid" {
  // The frontend uses 'polynomial' but the backend's joblib bundles are keyed 'poly'.
  if (id === "polynomial") return "poly";
  return id as "rbf" | "linear" | "sigmoid";
}

/** Stable fingerprint of the inputs that go into the RAG request — used to
 *  detect whether the latest response is still relevant to what the user is
 *  currently looking at. */
function fingerprint(model: SVMModel, prediction: SVMPrediction, values: number[]): string {
  const v = values.map((x) => x.toFixed(4)).join(",");
  return `${model.kernel}|${prediction.label}|${prediction.decision.toFixed(4)}|${prediction.confidence.toFixed(4)}|${v}`;
}

type FetchState =
  | { status: "idle" }
  | { status: "loading"; fingerprint: string }
  | { status: "ok"; data: ExplainResponseBody; fingerprint: string }
  | { status: "error"; message: string; fingerprint: string };

export function ExplanationPanel({ model, prediction, values }: ExplanationPanelProps) {
  const [state, setState] = useState<FetchState>({ status: "idle" });
  const abortRef = useRef<AbortController | null>(null);

  const currentFingerprint = useMemo(
    () => fingerprint(model, prediction, values),
    [model, prediction, values]
  );

  // Local summary always available — used as the default view and as the
  // permanent fallback when the backend is unreachable.
  const localSummary = useMemo(
    () => buildLocalSummary(model, prediction, values),
    [model, prediction, values]
  );

  // Cancel any in-flight request on unmount.
  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  function runExplain() {
    // Cancel anything in-flight, then start a new request bound to the
    // current fingerprint so we can detect staleness if the user keeps
    // moving sliders mid-flight.
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    const fp = currentFingerprint;
    setState({ status: "loading", fingerprint: fp });

    explainPrediction(
      {
        model: backendKernel(model.kernel),
        features: buildFeaturePayload(values),
        prediction: {
          label: prediction.label,
          decision: prediction.decision,
          probability_malignant: prediction.probability,
          confidence: prediction.confidence,
        },
        top_k: 4,
      },
      ctrl.signal
    )
      .then((data) => setState({ status: "ok", data, fingerprint: fp }))
      .catch((err: unknown) => {
        if (ctrl.signal.aborted) return;
        const message = err instanceof Error ? err.message : String(err);
        setState({ status: "error", message, fingerprint: fp });
      });
  }

  const hasResponse = state.status === "ok";
  const isLoading = state.status === "loading";
  const isError = state.status === "error";
  const isStale =
    (state.status === "ok" || state.status === "error") &&
    state.fingerprint !== currentFingerprint;

  const passages: RetrievedPassage[] =
    state.status === "ok" ? state.data.retrieved.slice(0, 4) : FALLBACK_SNIPPETS;
  const generator = state.status === "ok" ? state.data.generator : "template";
  const fullExplanation = state.status === "ok" ? state.data.explanation : null;

  // Status pill on the right-hand side of the panel header.
  let pill: { label: string; className: string; icon: React.ReactNode };
  if (isLoading) {
    pill = {
      label: "RAG · loading",
      className: "bg-foreground/5 text-muted-foreground",
      icon: <RefreshCw className="w-3 h-3 animate-spin" />,
    };
  } else if (isError) {
    pill = {
      label: "RAG · offline",
      className: "bg-amber-500/10 text-amber-700 dark:text-amber-300",
      icon: <WifiOff className="w-3 h-3" />,
    };
  } else if (hasResponse && generator === "openai") {
    pill = {
      label: "RAG · openai",
      className: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
      icon: <Sparkles className="w-3 h-3" />,
    };
  } else if (hasResponse && generator === "template") {
    pill = {
      label: "RAG · template",
      className: "bg-foreground/5 text-muted-foreground",
      icon: <Sparkles className="w-3 h-3" />,
    };
  } else {
    pill = {
      label: "local summary",
      className: "bg-foreground/5 text-muted-foreground",
      icon: <Sparkles className="w-3 h-3" />,
    };
  }

  return (
    <div className="grid lg:grid-cols-[1.1fr_1fr] gap-px bg-foreground/10 border border-foreground/10">
      {/* AI summary */}
      <div className="bg-background p-6">
        <div className="flex items-center gap-2 mb-4 flex-wrap">
          <Sparkles className="w-4 h-4" />
          <h4 className="font-display text-xl">Why this prediction?</h4>
          <span
            className={`ml-auto inline-flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded-full ${pill.className}`}
            title={
              isError
                ? `Backend at ${API_BASE_URL} is not responding. Showing local fallback.`
                : generator === "openai"
                ? "Generated via RAG over backend/documents + a hosted LLM."
                : hasResponse
                ? "Generated via RAG over backend/documents + the deterministic template generator."
                : "Click 'Generate explanation' to retrieve grounded context from the backend."
            }
          >
            {pill.icon}
            {pill.label}
          </span>
        </div>

        {hasResponse && fullExplanation ? (
          <div className="space-y-4">
            {fullExplanation.split(/\n\s*\n/).map((para, i) => (
              <p key={i} className="text-sm leading-relaxed text-foreground/90">
                {para}
              </p>
            ))}
          </div>
        ) : (
          <ul className="space-y-3">
            {localSummary.map((s, i) => (
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
        )}

        {/* Stale banner — only appears once the user has fetched at least once
            and then changed the prediction. */}
        {isStale && (
          <div className="mt-5 flex items-start gap-2 p-3 rounded-md border border-foreground/15 bg-foreground/[0.03] text-xs leading-relaxed">
            <RefreshCw className="w-3.5 h-3.5 mt-0.5 text-foreground/60 shrink-0" />
            <div className="flex-1 space-y-1">
              <div className="text-foreground/80">
                Prediction has changed since this explanation was generated. Re-run for fresh context.
              </div>
            </div>
          </div>
        )}

        {/* Error callout */}
        {isError && (
          <div className="mt-5 flex items-start gap-2 p-3 rounded-md border border-amber-500/30 bg-amber-500/5 text-xs leading-relaxed">
            <AlertTriangle className="w-3.5 h-3.5 mt-0.5 text-amber-600 dark:text-amber-400 shrink-0" />
            <div className="space-y-1">
              <div className="text-foreground/80">
                RAG backend at <code>{API_BASE_URL}</code> is not responding.
                Showing local summary and offline reference passages.
              </div>
              <div className="text-muted-foreground font-mono break-all">{(state as { message: string }).message}</div>
            </div>
          </div>
        )}

        {/* Action row — the only place that triggers /api/explain. */}
        <div className="mt-6 pt-5 border-t border-foreground/10 flex items-center gap-3 flex-wrap">
          <button
            type="button"
            onClick={runExplain}
            disabled={isLoading || (hasResponse && !isStale && !isError)}
            className={`group inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs font-mono uppercase tracking-wider transition-colors ${
              isLoading
                ? "bg-foreground/5 text-muted-foreground cursor-wait"
                : hasResponse && !isStale && !isError
                ? "bg-foreground/5 text-muted-foreground cursor-default"
                : "bg-foreground text-background hover:bg-foreground/90"
            }`}
            title={
              isLoading
                ? "Retrieving grounded context…"
                : hasResponse && !isStale && !isError
                ? "Already generated for this prediction. Adjust an input to re-run."
                : "Call /api/explain on the backend"
            }
          >
            {isLoading ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                Retrieving context…
              </>
            ) : isStale || isError ? (
              <>
                <RefreshCw className="w-3.5 h-3.5" />
                Re-run explanation
              </>
            ) : hasResponse ? (
              <>
                <Sparkles className="w-3.5 h-3.5" />
                Explanation up to date
              </>
            ) : (
              <>
                <Sparkles className="w-3.5 h-3.5" />
                Generate research-grounded explanation
                <ArrowRight className="w-3.5 h-3.5 -translate-x-0.5 group-hover:translate-x-0 transition-transform" />
              </>
            )}
          </button>
          <span className="text-[11px] font-mono text-muted-foreground">
            POST <code className="text-foreground/70">/api/explain</code>
          </span>
        </div>

        <div className="mt-5 pt-5 border-t border-foreground/10 text-xs text-muted-foreground font-mono">
          Predictions are advisory and require clinical validation before acting on them.
        </div>
      </div>

      {/* Retrieved snippets */}
      <div className="bg-background p-6">
        <div className="flex items-center gap-2 mb-4">
          <FileText className="w-4 h-4" />
          <h4 className="font-display text-xl">Retrieved reference passages</h4>
          {hasResponse && (
            <span className="ml-auto text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
              top {passages.length} · FAISS
            </span>
          )}
          {!hasResponse && (
            <span className="ml-auto text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
              local catalog
            </span>
          )}
        </div>
        <div className="space-y-4">
          {passages.map((r) => (
            <div key={r.id} className="border border-foreground/10 p-4 rounded-md">
              <div className="flex items-baseline justify-between gap-3 mb-2">
                <div className="text-xs font-mono text-muted-foreground truncate">{r.id}</div>
                {r.score > 0 && (
                  <div className="text-[10px] font-mono text-muted-foreground tabular-nums shrink-0">
                    sim {r.score.toFixed(3)}
                  </div>
                )}
              </div>
              <div className="font-display text-base leading-snug mb-2">{r.title}</div>
              <div className="text-sm text-foreground/80 leading-relaxed flex gap-2">
                <Quote className="w-3.5 h-3.5 shrink-0 mt-1 text-foreground/40" />
                <span>{r.quote || r.summary}</span>
              </div>
              {r.tags.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {r.tags.slice(0, 6).map((t) => (
                    <span
                      key={t}
                      className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-foreground/5 text-muted-foreground"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
