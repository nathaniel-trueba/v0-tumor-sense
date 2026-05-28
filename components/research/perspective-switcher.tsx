"use client";

import { useState } from "react";
import { BookOpen, ChartLine, Sparkles } from "lucide-react";
import { MLView } from "./ml-view";
import { StatsView } from "./stats-view";
import { CaseStudyView } from "./case-study-view";

type ViewId = "ml" | "stats" | "case";

const VIEWS: { id: ViewId; label: string; subtitle: string; icon: typeof Sparkles }[] = [
  {
    id: "ml",
    label: "Machine learning",
    subtitle: "SHAP · LIME · misclassification zones",
    icon: Sparkles,
  },
  {
    id: "stats",
    label: "Statistical",
    subtitle: "correlations · significance · effect sizes",
    icon: ChartLine,
  },
  {
    id: "case",
    label: "Case study",
    subtitle: "three patient walk-throughs",
    icon: BookOpen,
  },
];

export function PerspectiveSwitcher() {
  const [active, setActive] = useState<ViewId>("ml");

  return (
    <section id="analysis" className="space-y-10">
      <div>
        <span className="inline-flex items-center gap-3 text-sm font-mono text-muted-foreground mb-6">
          <span className="w-8 h-px bg-foreground/30" />
          Analysis
        </span>
        <h2 className="text-3xl lg:text-5xl font-display tracking-tight max-w-3xl">
          Choose a lens. <br />
          <span className="text-muted-foreground">Same data, different posture.</span>
        </h2>
      </div>

      <div className="grid sm:grid-cols-3 gap-px bg-foreground/10 border border-foreground/10">
        {VIEWS.map((v) => {
          const Icon = v.icon;
          const isActive = v.id === active;
          return (
            <button
              key={v.id}
              type="button"
              onClick={() => setActive(v.id)}
              className={`text-left p-6 transition-colors group relative ${
                isActive
                  ? "bg-foreground text-background"
                  : "bg-background hover:bg-foreground/[0.03]"
              }`}
            >
              <div className="flex items-center gap-3 mb-3">
                <Icon
                  className={`w-4 h-4 ${
                    isActive ? "text-background" : "text-foreground/70"
                  }`}
                />
                <span
                  className={`font-mono text-[11px] uppercase tracking-wider ${
                    isActive ? "text-background/70" : "text-muted-foreground"
                  }`}
                >
                  perspective
                </span>
              </div>
              <div className="font-display text-2xl">{v.label}</div>
              <div
                className={`text-xs font-mono mt-1 ${
                  isActive ? "text-background/70" : "text-muted-foreground"
                }`}
              >
                {v.subtitle}
              </div>
              {isActive && (
                <span className="absolute -bottom-px left-0 right-0 h-0.5 bg-background" />
              )}
            </button>
          );
        })}
      </div>

      <div className="pt-2">
        {active === "ml" && <MLView />}
        {active === "stats" && <StatsView />}
        {active === "case" && <CaseStudyView />}
      </div>
    </section>
  );
}
