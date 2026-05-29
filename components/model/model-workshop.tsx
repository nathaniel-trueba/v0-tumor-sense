"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ChevronDown, ArrowUpRight, ImageIcon } from "lucide-react";
import { FEATURES, type FeatureId } from "@/lib/breast-cancer/features";
import { getDefaultInput } from "@/lib/breast-cancer/dataset";
import { MODELS, MODEL_INDEX, type KernelId } from "@/lib/breast-cancer/models";
import { FeatureSpace } from "./feature-space";
import { FeatureSliders } from "./feature-sliders";
import { PredictionCard } from "./prediction-card";
import { ModelStats } from "./model-stats";
import { ExplanationPanel } from "./explanation-panel";

export function ModelWorkshop() {
  const [modelId, setModelId] = useState<KernelId>("rbf");
  const [values, setValues] = useState<number[]>(() => getDefaultInput());
  const [xFeature, setXFeature] = useState<FeatureId>("radius_mean");
  const [yFeature, setYFeature] = useState<FeatureId>("concave_points_mean");
  const [pickerOpen, setPickerOpen] = useState(false);

  // Defer mount-dependent things to avoid SSR mismatch.
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  const model = MODEL_INDEX[modelId];
  const prediction = useMemo(() => model.predict(values), [model, values]);

  function updateFeature(idx: number, value: number) {
    setValues((prev) => {
      const next = [...prev];
      next[idx] = value;
      return next;
    });
  }

  function resetNeutral() {
    setValues(getDefaultInput());
  }

  function useBenignMean() {
    setValues(FEATURES.map((f) => f.benignMean));
  }

  function useMalignantMean() {
    setValues(FEATURES.map((f) => f.malignantMean));
  }

  function randomize() {
    setValues(
      FEATURES.map((f) => {
        const mean = (f.benignMean + f.malignantMean) / 2;
        const spread = (f.malignantMean - f.benignMean) / 2;
        const v = mean + (Math.random() - 0.5) * spread * 4;
        return Math.min(f.max, Math.max(f.min, v));
      })
    );
  }

  function selectAxis(setter: (f: FeatureId) => void, other: FeatureId) {
    return (next: FeatureId) => {
      // Don't allow both axes to be the same feature.
      if (next === other) return;
      setter(next);
    };
  }

  return (
    <div className="space-y-12 lg:space-y-16">
      {/* Header */}
      <div className="space-y-6">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <span className="inline-flex items-center gap-3 text-sm font-mono text-muted-foreground">
            <span className="w-8 h-px bg-foreground/30" />
            Model workshop · SVM
          </span>
          <Link
            href="/model-2"
            className="group inline-flex items-center gap-2 rounded-full border border-foreground/15 hover:border-foreground/40 bg-background/40 px-4 py-1.5 text-xs font-mono text-muted-foreground hover:text-foreground transition-colors"
          >
            <ImageIcon className="w-3.5 h-3.5" />
            <span className="uppercase tracking-wider">Try the image workshop</span>
            <ArrowUpRight className="w-3.5 h-3.5 -translate-x-0.5 group-hover:translate-x-0 transition-transform" />
          </Link>
        </div>
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <h1 className="text-4xl lg:text-6xl font-display tracking-tight max-w-3xl">
            Probe the model. <br />
            <span className="text-muted-foreground">See what it sees.</span>
          </h1>

          {/* Model picker */}
          <div className="relative">
            <button
              type="button"
              onClick={() => setPickerOpen((p) => !p)}
              className="flex items-center gap-3 border border-foreground/15 rounded-full px-5 py-3 text-sm hover:border-foreground/40 transition-colors"
            >
              <span className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
                kernel
              </span>
              <span className="font-display text-lg">
                {model.name.replace("SVM — ", "")}
              </span>
              <ChevronDown
                className={`w-4 h-4 transition-transform ${pickerOpen ? "rotate-180" : ""}`}
              />
            </button>
            {pickerOpen && (
              <div className="absolute right-0 top-full mt-2 z-30 w-80 bg-background border border-foreground/15 rounded-2xl shadow-lg overflow-hidden">
                {MODELS.map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => {
                      setModelId(m.id);
                      setPickerOpen(false);
                    }}
                    className={`w-full text-left px-5 py-4 border-b border-foreground/10 last:border-b-0 transition-colors ${
                      m.id === modelId
                        ? "bg-foreground/[0.04]"
                        : "hover:bg-foreground/[0.02]"
                    }`}
                  >
                    <div className="flex items-baseline justify-between gap-3">
                      <span className="font-display text-base">{m.name.replace("SVM — ", "")}</span>
                      <span className="text-xs font-mono text-muted-foreground">
                        {(m.accuracy * 100).toFixed(1)}%
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
                      {m.description}
                    </p>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
        <p className="text-lg text-muted-foreground max-w-3xl leading-relaxed">
          Switch kernels on the right, then drag the sliders below. The decision boundary,
          your prediction, and the surrounding region update live. Pick any two features
          as visualisation axes — the remaining eight are held at the slider values.
        </p>
      </div>

      {/* Main interaction surface */}
      <div className="grid lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)] gap-10 lg:gap-14">
        {/* Sliders */}
        <div className="lg:sticky lg:top-28 self-start space-y-6">
          <div className="flex items-baseline justify-between">
            <h2 className="font-display text-2xl">Features</h2>
            <span className="text-xs font-mono text-muted-foreground">10 inputs</span>
          </div>
          <FeatureSliders
            values={values}
            onChange={updateFeature}
            xFeature={xFeature}
            yFeature={yFeature}
            onReset={resetNeutral}
            onRandomize={randomize}
            onUseBenignMean={useBenignMean}
            onUseMalignantMean={useMalignantMean}
          />
        </div>

        {/* Visualization */}
        <div className="space-y-8">
          <div className="space-y-2">
            <h2 className="font-display text-2xl">Decision boundary</h2>
            <p className="text-sm text-muted-foreground">
              Live feature space for the selected kernel. Hover any point for details.
            </p>
          </div>
          {mounted ? (
            <FeatureSpace
              model={model}
              values={values}
              xFeature={xFeature}
              yFeature={yFeature}
              onXFeatureChange={selectAxis(setXFeature, yFeature)}
              onYFeatureChange={selectAxis(setYFeature, xFeature)}
            />
          ) : (
            <div className="h-[420px] border border-foreground/10 animate-pulse bg-foreground/[0.02]" />
          )}

          <PredictionCard model={model} prediction={prediction} values={values} />
        </div>
      </div>

      {/* Stats */}
      <div className="space-y-6">
        <div className="flex items-baseline justify-between gap-4 flex-wrap">
          <h2 className="font-display text-3xl lg:text-5xl tracking-tight">Model-specific stats</h2>
          <span className="text-sm font-mono text-muted-foreground">
            {model.name} · {model.supportVectors} support vectors
          </span>
        </div>
        <ModelStats model={model} prediction={prediction} />
      </div>

      {/* Explanation / RAG */}
      <div className="space-y-6">
        <div className="flex items-baseline justify-between gap-4 flex-wrap">
          <h2 className="font-display text-3xl lg:text-5xl tracking-tight">Explainability</h2>
          <span className="text-sm font-mono text-muted-foreground">
            RAG retrieval over reference clinical documents
          </span>
        </div>
        <ExplanationPanel model={model} prediction={prediction} values={values} />
      </div>

      {/* Active feature snapshot */}
      <div className="border border-foreground/10 p-6 lg:p-8">
        <div className="flex items-baseline justify-between mb-6">
          <h3 className="font-display text-2xl">Input snapshot</h3>
          <span className="text-xs font-mono text-muted-foreground">
            JSON · ready to send to /api/predict (when wired up)
          </span>
        </div>
        <pre className="text-xs font-mono overflow-x-auto leading-relaxed text-foreground/80">
          {JSON.stringify(
            {
              model: model.id,
              kernel: model.kernel,
              features: Object.fromEntries(
                FEATURES.map((f, i) => [f.id, values[i]])
              ),
              prediction: {
                label: prediction.label,
                probability_malignant: +prediction.probability.toFixed(4),
                confidence: +prediction.confidence.toFixed(4),
                decision: +prediction.decision.toFixed(4),
              },
            },
            null,
            2
          )}
        </pre>
      </div>
    </div>
  );
}
