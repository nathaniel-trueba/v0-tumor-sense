"use client";

import { useEffect, useState } from "react";

// Shape of /data/gallery_predictions.json produced by scripts/predict_gallery.
// One row per (frontend network id, gallery image id) pair.
export interface RealPrediction {
  label: "cancer" | "no_cancer";
  predicted_class: 0 | 1;
  true_class: 0 | 1;
  correct: boolean;
  probability_cancer: number;
  probability_no_cancer: number;
  confidence: number;
  latency_ms: number;
}

export interface GalleryModelMeta {
  id: string;
  backend_id: string;
  name: string;
  family: string;
  weights_file: string;
  patch_accuracy: number;
  avg_latency_ms: number;
}

export interface GalleryPredictionsDoc {
  n_images: number;
  labels: [string, string];
  models: GalleryModelMeta[];
  predictions: Record<string, Record<string, RealPrediction>>;
}

let cache: GalleryPredictionsDoc | null = null;
let inflight: Promise<GalleryPredictionsDoc> | null = null;

/** One-shot, process-wide fetch of the pre-computed gallery predictions. */
export async function loadGalleryPredictions(): Promise<GalleryPredictionsDoc> {
  if (cache) return cache;
  if (!inflight) {
    inflight = fetch("/data/gallery_predictions.json")
      .then((r) => {
        if (!r.ok) throw new Error(`gallery_predictions: HTTP ${r.status}`);
        return r.json() as Promise<GalleryPredictionsDoc>;
      })
      .then((doc) => {
        cache = doc;
        return doc;
      })
      .catch((err) => {
        inflight = null;
        throw err;
      });
  }
  return inflight;
}

/** React hook — returns the cached doc once loaded, plus a status flag. */
export function useGalleryPredictions(): {
  doc: GalleryPredictionsDoc | null;
  status: "idle" | "loading" | "ok" | "error";
} {
  const [doc, setDoc] = useState<GalleryPredictionsDoc | null>(cache);
  const [status, setStatus] = useState<"idle" | "loading" | "ok" | "error">(
    cache ? "ok" : "loading"
  );

  useEffect(() => {
    if (cache) {
      setDoc(cache);
      setStatus("ok");
      return;
    }
    let cancelled = false;
    loadGalleryPredictions()
      .then((d) => {
        if (cancelled) return;
        setDoc(d);
        setStatus("ok");
      })
      .catch(() => {
        if (cancelled) return;
        setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { doc, status };
}

/** Look up a real prediction; returns null if no match (e.g. U-Net). */
export function lookupRealPrediction(
  doc: GalleryPredictionsDoc | null,
  networkId: string,
  imageId: string
): RealPrediction | null {
  if (!doc) return null;
  return doc.predictions[networkId]?.[imageId] ?? null;
}
