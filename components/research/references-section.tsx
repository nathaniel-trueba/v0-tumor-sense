"use client";

import { useState } from "react";
import { ChevronDown, ExternalLink } from "lucide-react";

interface Reference {
  id: string;
  title: string;
  authors: string;
  venue: string;
  year: number;
  url?: string;
  doi?: string;
}

const REFERENCES: Reference[] = [
  {
    id: "wolberg-1990",
    title: "Multisurface method of pattern separation for medical diagnosis applied to breast cytology",
    authors: "W. Wolberg, O. Mangasarian",
    venue: "Proceedings of the National Academy of Sciences",
    year: 1990,
    doi: "10.1073/pnas.87.23.9193",
  },
  {
    id: "uci-wdbc",
    title: "Wisconsin Diagnostic Breast Cancer (WDBC) Dataset",
    authors: "W. Street, W. Wolberg, O. Mangasarian",
    venue: "UCI Machine Learning Repository",
    year: 1995,
    url: "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
  },
  {
    id: "cortes-1995",
    title: "Support-vector networks",
    authors: "C. Cortes, V. Vapnik",
    venue: "Machine Learning, 20(3)",
    year: 1995,
    doi: "10.1007/BF00994018",
  },
  {
    id: "guyon-2002",
    title: "Gene selection for cancer classification using support vector machines",
    authors: "I. Guyon, J. Weston, S. Barnhill, V. Vapnik",
    venue: "Machine Learning, 46",
    year: 2002,
    doi: "10.1023/A:1012487302797",
  },
  {
    id: "lundberg-2017",
    title: "A unified approach to interpreting model predictions",
    authors: "S. Lundberg, S.-I. Lee",
    venue: "Advances in Neural Information Processing Systems",
    year: 2017,
    url: "https://arxiv.org/abs/1705.07874",
  },
  {
    id: "ribeiro-2016",
    title: "“Why should I trust you?”: Explaining the predictions of any classifier",
    authors: "M. Ribeiro, S. Singh, C. Guestrin",
    venue: "KDD",
    year: 2016,
    url: "https://arxiv.org/abs/1602.04938",
  },
  {
    id: "ds3-2026",
    title: "TumorSense — DS3 Spring Project (internal report)",
    authors: "N. Trueba, K. Shah, S. Ngo, E. Park",
    venue: "Data Science Student Society @ UC San Diego",
    year: 2026,
  },
];

export function ReferencesSection() {
  const [open, setOpen] = useState(false);

  return (
    <section id="references" className="border-t border-foreground/10 pt-12">
      <button
        type="button"
        onClick={() => setOpen((p) => !p)}
        className="flex items-center justify-between w-full text-left group"
        aria-expanded={open}
      >
        <div>
          <span className="inline-flex items-center gap-3 text-sm font-mono text-muted-foreground mb-3">
            <span className="w-8 h-px bg-foreground/30" />
            References
          </span>
          <h2 className="text-2xl lg:text-4xl font-display tracking-tight">
            Show the {REFERENCES.length} works informing this project
          </h2>
        </div>
        <ChevronDown
          className={`w-6 h-6 text-foreground/60 transition-transform duration-300 shrink-0 ${
            open ? "rotate-180" : ""
          }`}
        />
      </button>

      <div
        className={`grid transition-all duration-500 ${
          open ? "grid-rows-[1fr] mt-10 opacity-100" : "grid-rows-[0fr] mt-0 opacity-0"
        }`}
      >
        <div className="overflow-hidden">
          <ol className="space-y-3">
            {REFERENCES.map((r, idx) => {
              const href = r.url ?? (r.doi ? `https://doi.org/${r.doi}` : undefined);
              const Container = href ? "a" : "div";
              return (
                <li key={r.id}>
                  <Container
                    {...(href ? { href, target: "_blank", rel: "noreferrer" } : {})}
                    className="grid grid-cols-[40px_1fr_auto] gap-4 items-baseline py-3 border-b border-foreground/5 hover:bg-foreground/[0.02] transition-colors px-2 rounded-sm"
                  >
                    <span className="font-mono text-xs text-muted-foreground tabular-nums">
                      [{String(idx + 1).padStart(2, "0")}]
                    </span>
                    <div className="min-w-0">
                      <div className="font-display text-base leading-snug">{r.title}</div>
                      <div className="text-xs text-muted-foreground mt-1 font-mono">
                        {r.authors} · {r.venue} · {r.year}
                      </div>
                    </div>
                    {href && (
                      <ExternalLink className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                    )}
                  </Container>
                </li>
              );
            })}
          </ol>
        </div>
      </div>
    </section>
  );
}
