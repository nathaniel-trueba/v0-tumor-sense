"use client";

import { useEffect, useRef, useState } from "react";

const stats = [
  { value: "569", label: "Patient samples", hint: "Wisconsin Diagnostic" },
  { value: "30", label: "Source features", hint: "10 mean · 10 SE · 10 worst" },
  { value: "62.7%", label: "Benign prevalence", hint: "357 / 212 split" },
  { value: "1990", label: "Original publication", hint: "Wolberg & Mangasarian" },
];

export function BackgroundSection() {
  const [visible, setVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => e.isIntersecting && setVisible(true), {
      threshold: 0.15,
    });
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);

  return (
    <section id="background" ref={ref} className="space-y-12 lg:space-y-16">
      <div>
        <span className="inline-flex items-center gap-3 text-sm font-mono text-muted-foreground mb-6">
          <span className="w-8 h-px bg-foreground/30" />
          Background
        </span>
        <h1
          className={`text-4xl lg:text-6xl font-display tracking-tight max-w-4xl transition-all duration-700 ${
            visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
          }`}
        >
          Reading nuclei <br />
          <span className="text-muted-foreground">as evidence.</span>
        </h1>
      </div>

      <div className="grid lg:grid-cols-2 gap-12 lg:gap-20">
        <div className="space-y-5 text-lg leading-relaxed text-foreground/85">
          <p>
            Breast cancer is the most commonly diagnosed cancer worldwide; early
            cytological screening through fine-needle aspirate (FNA) imaging has
            been an important first line of triage since the early 1990s.
            The Wisconsin Diagnostic Breast Cancer (WDBC) dataset, collected at
            UW-Madison, encodes 30 morphometric features computed from digitised
            FNA images into a structured tabular form.
          </p>
          <p>
            Our project re-examines this dataset through the lens of support
            vector machines. We compare four kernels — linear, polynomial, RBF
            and sigmoid — under a unified feature-selection and validation
            pipeline, then expose the model behaviour interactively so that
            clinicians and students alike can build intuition.
          </p>
          <p>
            The model is <em className="text-foreground">not</em> a diagnostic tool.
            It is a structured perspective on a classic dataset and a vehicle
            for explainable-AI experimentation.
          </p>
        </div>

        <div
          className={`transition-all duration-700 delay-200 ${
            visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"
          }`}
        >
          <h3 className="font-display text-xl mb-6">Motivation in numbers</h3>
          <div className="grid grid-cols-2 gap-px bg-foreground/10 border border-foreground/10">
            {stats.map((s) => (
              <div key={s.label} className="bg-background p-6">
                <div className="text-3xl lg:text-4xl font-display tracking-tight">{s.value}</div>
                <div className="text-sm text-foreground/80 mt-1">{s.label}</div>
                <div className="text-xs font-mono text-muted-foreground mt-1">{s.hint}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
