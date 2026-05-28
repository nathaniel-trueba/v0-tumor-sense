// Feature definitions for the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
// The min / max / step values approximate the empirical ranges in the public dataset.
// Each feature stores both class means so we can generate realistic mock data and
// compute interpretable decision boundaries without a live backend.

export type FeatureId =
  | "radius_mean"
  | "texture_mean"
  | "perimeter_mean"
  | "area_mean"
  | "smoothness_mean"
  | "compactness_mean"
  | "concavity_mean"
  | "concave_points_mean"
  | "symmetry_mean"
  | "fractal_dimension_mean";

export interface FeatureMeta {
  id: FeatureId;
  label: string;
  short: string;
  unit?: string;
  min: number;
  max: number;
  step: number;
  benignMean: number;
  benignSd: number;
  malignantMean: number;
  malignantSd: number;
  rfeRank: number; // 1 = strongest selected, higher = less informative
  importance: number; // 0..1 SHAP-style importance score
  description: string;
}

export const FEATURES: FeatureMeta[] = [
  {
    id: "radius_mean",
    label: "Radius (mean)",
    short: "Radius",
    unit: "mm",
    min: 6,
    max: 30,
    step: 0.1,
    benignMean: 12.2,
    benignSd: 1.8,
    malignantMean: 17.5,
    malignantSd: 3.2,
    rfeRank: 1,
    importance: 0.92,
    description: "Mean distance from cell-nucleus center to its perimeter.",
  },
  {
    id: "texture_mean",
    label: "Texture (mean)",
    short: "Texture",
    min: 9,
    max: 40,
    step: 0.1,
    benignMean: 17.9,
    benignSd: 4,
    malignantMean: 21.6,
    malignantSd: 4,
    rfeRank: 6,
    importance: 0.41,
    description: "Standard deviation of greyscale values in the cell nucleus.",
  },
  {
    id: "perimeter_mean",
    label: "Perimeter (mean)",
    short: "Perimeter",
    unit: "px",
    min: 40,
    max: 200,
    step: 0.5,
    benignMean: 78.1,
    benignSd: 11.8,
    malignantMean: 115.4,
    malignantSd: 21.9,
    rfeRank: 2,
    importance: 0.9,
    description: "Average perimeter of the cell nucleus.",
  },
  {
    id: "area_mean",
    label: "Area (mean)",
    short: "Area",
    unit: "px²",
    min: 140,
    max: 2500,
    step: 1,
    benignMean: 462.8,
    benignSd: 134,
    malignantMean: 978.4,
    malignantSd: 367,
    rfeRank: 3,
    importance: 0.88,
    description: "Average area of the cell nucleus.",
  },
  {
    id: "smoothness_mean",
    label: "Smoothness (mean)",
    short: "Smoothness",
    min: 0.05,
    max: 0.17,
    step: 0.001,
    benignMean: 0.0925,
    benignSd: 0.0134,
    malignantMean: 0.1029,
    malignantSd: 0.0125,
    rfeRank: 9,
    importance: 0.23,
    description: "Local variation in radius lengths.",
  },
  {
    id: "compactness_mean",
    label: "Compactness (mean)",
    short: "Compactness",
    min: 0.02,
    max: 0.36,
    step: 0.001,
    benignMean: 0.0801,
    benignSd: 0.0337,
    malignantMean: 0.1452,
    malignantSd: 0.054,
    rfeRank: 7,
    importance: 0.38,
    description: "Perimeter² / area − 1.0 — captures how compact a nucleus is.",
  },
  {
    id: "concavity_mean",
    label: "Concavity (mean)",
    short: "Concavity",
    min: 0,
    max: 0.43,
    step: 0.001,
    benignMean: 0.0461,
    benignSd: 0.0436,
    malignantMean: 0.16,
    malignantSd: 0.075,
    rfeRank: 4,
    importance: 0.81,
    description: "Severity of concave portions of the cell-nucleus contour.",
  },
  {
    id: "concave_points_mean",
    label: "Concave points (mean)",
    short: "Concave pts.",
    min: 0,
    max: 0.21,
    step: 0.001,
    benignMean: 0.0258,
    benignSd: 0.0159,
    malignantMean: 0.0879,
    malignantSd: 0.0345,
    rfeRank: 5,
    importance: 0.78,
    description: "Number of concave portions of the contour.",
  },
  {
    id: "symmetry_mean",
    label: "Symmetry (mean)",
    short: "Symmetry",
    min: 0.1,
    max: 0.31,
    step: 0.001,
    benignMean: 0.1742,
    benignSd: 0.0248,
    malignantMean: 0.1928,
    malignantSd: 0.0276,
    rfeRank: 8,
    importance: 0.27,
    description: "Symmetry of the cell-nucleus shape.",
  },
  {
    id: "fractal_dimension_mean",
    label: "Fractal dim. (mean)",
    short: "Fractal dim.",
    min: 0.045,
    max: 0.1,
    step: 0.0005,
    benignMean: 0.0629,
    benignSd: 0.0067,
    malignantMean: 0.0627,
    malignantSd: 0.0076,
    rfeRank: 10,
    importance: 0.09,
    description: "Coastline approximation − 1.0 (boundary complexity).",
  },
];

export const FEATURE_INDEX = Object.fromEntries(
  FEATURES.map((f, i) => [f.id, i])
) as Record<FeatureId, number>;

export const FEATURE_IDS = FEATURES.map((f) => f.id);

// Normalize a value to 0..1 based on its feature range.
export function normalize(value: number, feature: FeatureMeta): number {
  return (value - feature.min) / (feature.max - feature.min);
}

export function denormalize(value: number, feature: FeatureMeta): number {
  return feature.min + value * (feature.max - feature.min);
}

// Convert a full feature vector (in original units, ordered like FEATURES) into
// a normalized [0,1] vector — what most of the SVM math operates on.
export function normalizeVector(values: number[]): number[] {
  return values.map((v, i) => normalize(v, FEATURES[i]));
}
