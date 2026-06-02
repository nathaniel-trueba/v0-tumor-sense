"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Activity, Layers, Microscope } from "lucide-react";

// Patches arrive as [col, row, true, predicted, proba] tuples — keeps the
// shipped JSON small (~56 KB for 3 patients × ~1000 patches).
type Patch = [number, number, number, number, number];

interface PatientLandscape {
  id: string;
  label: string;
  caption: string;
  cols: number;
  rows: number;
  n_total: number;
  n_cancer_true: number;
  n_cancer_pred: number;
  patch_accuracy?: number;
  patches: Patch[];
  /** When present, the tissue and mask panels render the real stitched RGB
   *  slide instead of falling back to a silhouette. */
  tissue_image?: string;
  tissue_image_size?: [number, number];
}

interface LandscapeData {
  patch_size: number;
  model?: string;
  source: string;
  patients: PatientLandscape[];
}

type LayerId = "tissue" | "mask" | "proba";

const LAYERS: { id: LayerId; label: string; tagline: string; icon: typeof Microscope }[] = [
  {
    id: "tissue",
    label: "Tissue slice",
    tagline: "Stitched H&E slide reconstructed from the raw 50 px patches.",
    icon: Microscope,
  },
  {
    id: "mask",
    label: "Cancer tissue · red mask",
    tagline: "Same tissue panel overlaid with patches the network predicted as malignant.",
    icon: Layers,
  },
  {
    id: "proba",
    label: "Cancer probability",
    tagline: "Per-patch P(cancer) heatmap — yellow → orange → deep red.",
    icon: Activity,
  },
];

// matplotlib YlOrRd ramp, sampled at 5 stops.
const YLORRD: ReadonlyArray<[number, number, number, number]> = [
  [0.0, 255, 255, 204],
  [0.25, 254, 217, 118],
  [0.5, 253, 141, 60],
  [0.75, 227, 26, 28],
  [1.0, 128, 0, 38],
];

function sampleYlOrRd(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t));
  for (let i = 1; i < YLORRD.length; i++) {
    const [tb, rb, gb, bb] = YLORRD[i];
    const [ta, ra, ga, ba] = YLORRD[i - 1];
    if (x <= tb) {
      const u = (x - ta) / (tb - ta || 1);
      return [Math.round(ra + (rb - ra) * u), Math.round(ga + (gb - ga) * u), Math.round(ba + (bb - ba) * u)];
    }
  }
  const last = YLORRD[YLORRD.length - 1];
  return [last[1], last[2], last[3]];
}

// Histology-tissue colour used when we only care about the silhouette
// (where there is data at all) — based on hematoxylin/eosin's pink/lilac.
const TISSUE_RGB: [number, number, number] = [232, 207, 222]; // soft lilac-pink

// Site malignancy red, with alpha applied in the canvas draw.
const CANCER_RGB: [number, number, number] = [236, 72, 153]; // #EC4899

interface CanvasPlotProps {
  patient: PatientLandscape;
  layer: LayerId;
  size: number;
  tissueImage: HTMLImageElement | null;
}

function CanvasPlot({ patient, layer, size, tissueImage }: CanvasPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
    const pxW = Math.round(size * dpr);
    const pxH = Math.round(size * dpr);
    canvas.width = pxW;
    canvas.height = pxH;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;

    // Background — soft cream so YlOrRd's pale yellow has contrast.
    ctx.fillStyle = layer === "proba" ? "rgba(255, 252, 240, 1)" : "rgba(255, 255, 255, 1)";
    ctx.fillRect(0, 0, pxW, pxH);

    // Fit grid into canvas, preserving aspect ratio.
    const aspect = patient.cols / patient.rows;
    let drawW: number;
    let drawH: number;
    if (aspect >= 1) {
      drawW = pxW * 0.94;
      drawH = drawW / aspect;
    } else {
      drawH = pxH * 0.94;
      drawW = drawH * aspect;
    }
    const offsetX = (pxW - drawW) / 2;
    const offsetY = (pxH - drawH) / 2;
    const cellW = drawW / patient.cols;
    const cellH = drawH / patient.rows;

    const hasTissue = !!tissueImage && (layer === "tissue" || layer === "mask");

    if (hasTissue && tissueImage) {
      // Match the matplotlib alpha=0.9 / 0.8 the notebook uses on the tissue.
      ctx.globalAlpha = layer === "mask" ? 0.8 : 0.9;
      ctx.drawImage(tissueImage, offsetX, offsetY, drawW, drawH);
      ctx.globalAlpha = 1;
    } else if (layer === "tissue" || layer === "mask") {
      // Fallback for patients without a stitched RGB image — draw silhouette.
      ctx.fillStyle = `rgba(${TISSUE_RGB[0]}, ${TISSUE_RGB[1]}, ${TISSUE_RGB[2]}, 0.9)`;
      for (const [c, r] of patient.patches) {
        ctx.fillRect(
          Math.round(offsetX + c * cellW),
          Math.round(offsetY + r * cellH),
          Math.ceil(cellW + 0.5),
          Math.ceil(cellH + 0.5)
        );
      }
    }

    if (layer === "mask") {
      // Notebook used alpha=0.7 for the red mask.
      const alpha = hasTissue ? 0.55 : 0.65;
      ctx.fillStyle = `rgba(${CANCER_RGB[0]}, ${CANCER_RGB[1]}, ${CANCER_RGB[2]}, ${alpha})`;
      for (const [c, r, , predicted] of patient.patches) {
        if (predicted !== 1) continue;
        ctx.fillRect(
          Math.round(offsetX + c * cellW),
          Math.round(offsetY + r * cellH),
          Math.ceil(cellW + 0.5),
          Math.ceil(cellH + 0.5)
        );
      }
    } else if (layer === "proba") {
      for (const [c, r, , , proba] of patient.patches) {
        const [pr, pg, pb] = sampleYlOrRd(proba);
        ctx.fillStyle = `rgba(${pr}, ${pg}, ${pb}, 1)`;
        ctx.fillRect(
          Math.round(offsetX + c * cellW),
          Math.round(offsetY + r * cellH),
          Math.ceil(cellW + 0.5),
          Math.ceil(cellH + 0.5)
        );
      }
    }

    // Frame the plot with a subtle hairline so it reads as a panel.
    ctx.strokeStyle = "rgba(20, 20, 20, 0.08)";
    ctx.lineWidth = 1 * dpr;
    ctx.strokeRect(0.5, 0.5, pxW - 1, pxH - 1);
  }, [patient, layer, size, tissueImage]);

  return (
    <canvas
      ref={canvasRef}
      className="block w-full h-auto"
      role="img"
      aria-label={`${LAYERS.find((l) => l.id === layer)?.label} for patient ${patient.id}`}
    />
  );
}

/** Loads the patient's tissue PNG once and re-emits when it's decoded. */
function useTissueImage(url: string | undefined): HTMLImageElement | null {
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    if (!url) {
      setImg(null);
      return;
    }
    let cancelled = false;
    const el = new window.Image();
    el.crossOrigin = "anonymous";
    el.onload = () => {
      if (!cancelled) setImg(el);
    };
    el.src = url;
    return () => {
      cancelled = true;
    };
  }, [url]);
  return img;
}

function PatientRow({ patient, size }: { patient: PatientLandscape; size: number }) {
  const cancerPct = (patient.n_cancer_pred / patient.n_total) * 100;
  const truePct = (patient.n_cancer_true / patient.n_total) * 100;
  const tissueImage = useTissueImage(patient.tissue_image);

  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <div>
          <h4 className="font-display text-xl lg:text-2xl tracking-tight">{patient.label}</h4>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl leading-relaxed">
            {patient.caption}
          </p>
        </div>
        <div className="text-xs font-mono text-muted-foreground text-right tabular-nums leading-relaxed">
          <div>
            patient · <span className="text-foreground/80">{patient.id}</span>
          </div>
          <div>
            grid <span className="text-foreground/80">{patient.cols}×{patient.rows}</span> ·{" "}
            <span className="text-foreground/80">{patient.n_total}</span> patches
          </div>
          <div>
            P(cancer · pred) <span className="text-foreground/80">{cancerPct.toFixed(0)}%</span> · GT{" "}
            <span className="text-foreground/80">{truePct.toFixed(0)}%</span>
          </div>
          {patient.patch_accuracy !== undefined && (
            <div>
              patch acc <span className="text-foreground/80">{(patient.patch_accuracy * 100).toFixed(1)}%</span>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-foreground/10 border border-foreground/10">
        {LAYERS.map((layer) => {
          const Icon = layer.icon;
          return (
            <div key={layer.id} className="bg-background p-4 space-y-3">
              <div className="flex items-center gap-2">
                <Icon className="w-3.5 h-3.5 text-muted-foreground" />
                <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  {layer.label}
                </span>
              </div>
              <CanvasPlot
                patient={patient}
                layer={layer.id}
                size={size}
                tissueImage={tissueImage}
              />
              <p className="text-[11px] font-mono text-muted-foreground leading-relaxed">
                {layer.tagline}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ColorRamp() {
  const stops = Array.from({ length: 28 }, (_, i) => i / 27);
  return (
    <div className="flex items-center gap-3 text-[10px] font-mono text-muted-foreground">
      <span>0.00</span>
      <div className="flex h-2 flex-1 overflow-hidden rounded-full border border-foreground/10">
        {stops.map((t, i) => {
          const [r, g, b] = sampleYlOrRd(t);
          return <div key={i} style={{ background: `rgb(${r}, ${g}, ${b})`, flex: 1 }} />;
        })}
      </div>
      <span>1.00</span>
    </div>
  );
}

export function ProbabilityLandscape() {
  const [data, setData] = useState<LandscapeData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/data/probability_landscape.json")
      .then((r) => {
        if (!r.ok) throw new Error(`fetch failed: ${r.status}`);
        return r.json() as Promise<LandscapeData>;
      })
      .then((json) => {
        if (!cancelled) setData(json);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Static-sized canvases keep the layout calm — each plot draws at 320 px wide
  // on desktop, 280 px on tablet, falling through to full-width column on small.
  const size = useMemo(() => 320, []);

  return (
    <section className="space-y-8">
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <div className="max-w-2xl">
          <h3 className="font-display text-2xl lg:text-3xl tracking-tight">
            The probability landscape of invasive ductal carcinoma
          </h3>
          <p className="text-sm text-muted-foreground mt-2 leading-relaxed">
            A whole-slide reconstruction from {data ? data.patients[0]?.n_total.toLocaleString() : "~500"} raw
            50 px patches, with every patch coloured by what the network
            thought of it. The three views together let you read the slide
            three ways: where the tissue lives, what the model called
            malignant, and how confident it was at every step.
          </p>
        </div>
        <div className="text-[11px] font-mono text-muted-foreground text-right max-w-xs">
          {data ? (
            <>
              {data.model && (
                <>
                  model <span className="text-foreground/80">{data.model}</span>
                  <br />
                </>
              )}
              source <span className="text-foreground/80">{data.source}</span>
              <br />
              patch size <span className="text-foreground/80">{data.patch_size}px</span>
            </>
          ) : (
            "loading patches…"
          )}
        </div>
      </div>

      <div className="grid lg:grid-cols-[1fr_minmax(0,260px)] gap-6 items-end">
        <ColorRamp />
        <div className="flex items-center gap-4 text-[11px] font-mono text-muted-foreground">
          <span className="flex items-center gap-2">
            <span
              className="inline-block w-3 h-3 rounded-sm"
              style={{ background: `rgb(${TISSUE_RGB[0]}, ${TISSUE_RGB[1]}, ${TISSUE_RGB[2]})` }}
            />
            tissue
          </span>
          <span className="flex items-center gap-2">
            <span
              className="inline-block w-3 h-3 rounded-sm"
              style={{ background: `rgba(${CANCER_RGB[0]}, ${CANCER_RGB[1]}, ${CANCER_RGB[2]}, 0.85)` }}
            />
            predicted cancer
          </span>
        </div>
      </div>

      {error && (
        <div className="border border-amber-500/30 bg-amber-500/5 text-xs leading-relaxed p-4 rounded-md text-foreground/80">
          Could not load <code>/data/probability_landscape.json</code>: {error}
        </div>
      )}

      {!data && !error && (
        <div className="grid lg:grid-cols-3 gap-px bg-foreground/10 border border-foreground/10">
          {[0, 1, 2].map((i) => (
            <div key={i} className="bg-background p-4 aspect-square animate-pulse">
              <div className="w-full h-full bg-foreground/[0.03] rounded-sm" />
            </div>
          ))}
        </div>
      )}

      {data && (
        <div className="space-y-12">
          {data.patients.map((p) => (
            <PatientRow key={p.id} patient={p} size={size} />
          ))}
        </div>
      )}

      <p className="text-xs text-muted-foreground leading-relaxed max-w-3xl">
        Adapted from the <em>Probability landscape of invasive ductal carcinoma</em>{" "}
        figure in the project's training notebook. The slide is rebuilt from
        the original 50 px patches; the mask and probability overlays are the
        ResNet-18's actual predicted class and P(cancer) for each patch.
      </p>
    </section>
  );
}
