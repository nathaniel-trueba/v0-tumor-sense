import { FEATURES, FeatureId, FEATURE_INDEX } from "./features";

export type DiagnosisLabel = "benign" | "malignant";

export interface DataPoint {
  id: number;
  label: DiagnosisLabel;
  values: number[]; // ordered like FEATURES
}

// Small seedable PRNG (mulberry32) — keeps the dataset stable between renders.
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rand: () => number, mean: number, sd: number): number {
  // Box-Muller transform.
  const u = Math.max(rand(), 1e-12);
  const v = rand();
  const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  return mean + z * sd;
}

function generate(
  count: number,
  label: DiagnosisLabel,
  rand: () => number,
  startId: number
): DataPoint[] {
  return Array.from({ length: count }, (_, i) => ({
    id: startId + i,
    label,
    values: FEATURES.map((f) => {
      const mean = label === "benign" ? f.benignMean : f.malignantMean;
      const sd = label === "benign" ? f.benignSd : f.malignantSd;
      const v = gauss(rand, mean, sd);
      return Math.min(f.max, Math.max(f.min, v));
    }),
  }));
}

// Approximate the WDBC class balance: ~62.7% benign, 37.3% malignant.
const TOTAL = 220;
const BENIGN_COUNT = Math.round(TOTAL * 0.627);
const MALIGNANT_COUNT = TOTAL - BENIGN_COUNT;

const rand = mulberry32(20260528);
export const DATASET: DataPoint[] = [
  ...generate(BENIGN_COUNT, "benign", rand, 0),
  ...generate(MALIGNANT_COUNT, "malignant", rand, BENIGN_COUNT),
];

export function getDefaultInput(): number[] {
  // A "neutral" input sitting between the two class means — perfect starting
  // location to make the boundary visualisation interesting from frame 1.
  return FEATURES.map((f) => (f.benignMean + f.malignantMean) / 2);
}

export function getFeatureExtents(featureId: FeatureId) {
  const f = FEATURES[FEATURE_INDEX[featureId]];
  return { min: f.min, max: f.max };
}

// Helper to project the dataset down to two chosen feature axes for the 2D
// decision-boundary view.
export function project2D(
  data: DataPoint[],
  xFeature: FeatureId,
  yFeature: FeatureId
): { x: number; y: number; label: DiagnosisLabel; id: number; values: number[] }[] {
  const xi = FEATURE_INDEX[xFeature];
  const yi = FEATURE_INDEX[yFeature];
  return data.map((d) => ({
    id: d.id,
    label: d.label,
    x: d.values[xi],
    y: d.values[yi],
    values: d.values,
  }));
}
