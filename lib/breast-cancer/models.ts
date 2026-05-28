import { FEATURES, normalize } from "./features";

export type KernelId = "linear" | "polynomial" | "rbf" | "sigmoid";
export type DiagnosisLabel = "benign" | "malignant";

export interface SVMPrediction {
  label: DiagnosisLabel;
  decision: number; // signed distance to boundary (negative = benign, positive = malignant)
  probability: number; // 0..1 probability of malignant
  confidence: number; // 0..1 confidence in the predicted label
}

export interface SVMModel {
  id: KernelId;
  name: string;
  kernel: KernelId;
  formula: string;
  description: string;
  // pre-computed mock performance metrics
  accuracy: number;
  rocAuc: number;
  f1: number;
  precision: number;
  recall: number;
  latencyMs: number;
  trainingTimeMs: number;
  supportVectors: number;
  hyperparameters: { name: string; value: string }[];
  predict: (values: number[]) => SVMPrediction;
  decision: (values: number[]) => number;
  // bias-variance proxy for the research page (training error vs cv error per complexity step)
  biasVariance: { complexity: number; train: number; test: number }[];
  learningCurve: { epoch: number; train: number; test: number }[];
}

// ─── Weights derived from a mock-fit ───────────────────────────────────────────
// Direction roughly aligned with the SHAP-style "importance" so that the
// decision values move in a sensible direction as the user adjusts sliders.
const W = FEATURES.map((f) => {
  // Heuristic — features whose malignant mean is meaningfully higher than the
  // benign mean get a positive weight, otherwise negative.
  const span = f.malignantMean - f.benignMean;
  const denom = Math.max(1e-6, f.max - f.min);
  return (span / denom) * (0.6 + f.importance * 1.8);
});

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function normVec(values: number[]): number[] {
  return values.map((v, i) => normalize(v, FEATURES[i]));
}

// A scalar threshold so the boundary roughly bisects the two class means.
const BENIGN_NORM = normVec(FEATURES.map((f) => f.benignMean));
const MALIGNANT_NORM = normVec(FEATURES.map((f) => f.malignantMean));
const BIAS = -(dot(W, BENIGN_NORM) + dot(W, MALIGNANT_NORM)) / 2;

function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x));
}

function probability(decision: number): number {
  // Stretch the sigmoid so confidence behaves nicely near the boundary.
  return sigmoid(decision * 3.4);
}

function packPrediction(decision: number): SVMPrediction {
  const probability_malignant = probability(decision);
  const label: DiagnosisLabel = decision >= 0 ? "malignant" : "benign";
  const confidence = label === "malignant" ? probability_malignant : 1 - probability_malignant;
  return { label, decision, probability: probability_malignant, confidence };
}

// ─── Kernel functions on normalized vectors ────────────────────────────────────
function linearDecision(values: number[]): number {
  const x = normVec(values);
  return dot(W, x) + BIAS;
}

function polynomialDecision(values: number[]): number {
  const x = normVec(values);
  const lin = dot(W, x) + BIAS;
  const sign = Math.sign(lin) || 1;
  // Degree-3 polynomial decision — same sign as linear but curved.
  return sign * Math.pow(Math.abs(lin) * 1.4, 1.8);
}

function sigmoidDecision(values: number[]): number {
  const x = normVec(values);
  const z = dot(W, x) + BIAS;
  return Math.tanh(z * 1.6);
}

// ─── RBF support vectors (mock anchors near each cluster) ──────────────────────
function rbfAnchorsFor(label: DiagnosisLabel): number[][] {
  const anchors: number[][] = [];
  // 3 jittered anchors per class so the boundary curves like a true RBF SVM.
  const seedOffsets = [
    [-0.5, -0.3, 0.2, 0.1, -0.2, 0.4, -0.1, 0.3, -0.4, 0.2],
    [0.4, -0.2, -0.3, 0.5, 0.1, -0.4, 0.2, -0.5, 0.3, -0.1],
    [-0.2, 0.4, 0.5, -0.3, 0.2, -0.1, 0.4, 0.2, 0.1, -0.5],
  ];
  for (const o of seedOffsets) {
    anchors.push(
      FEATURES.map((f, i) => {
        const base = label === "benign" ? f.benignMean : f.malignantMean;
        const sd = label === "benign" ? f.benignSd : f.malignantSd;
        return normalize(base + o[i] * sd * 0.6, f);
      })
    );
  }
  return anchors;
}

const RBF_BENIGN = rbfAnchorsFor("benign");
const RBF_MALIGNANT = rbfAnchorsFor("malignant");
const RBF_GAMMA = 6;

function squaredDist(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

function rbfDecision(values: number[]): number {
  const x = normVec(values);
  let pos = 0;
  let neg = 0;
  for (const a of RBF_MALIGNANT) pos += Math.exp(-RBF_GAMMA * squaredDist(x, a));
  for (const a of RBF_BENIGN) neg += Math.exp(-RBF_GAMMA * squaredDist(x, a));
  return (pos - neg) * 1.6;
}

// Bias / variance curves & learning curves are illustrative.
function bvCurve(bias0: number, variance0: number) {
  return Array.from({ length: 9 }, (_, i) => {
    const c = i + 1;
    const train = Math.max(0.005, bias0 * Math.exp(-c / 3));
    const test = Math.max(0.015, bias0 * Math.exp(-c / 2.5) + variance0 * (1 - Math.exp(-c / 4)));
    return { complexity: c, train: +train.toFixed(4), test: +test.toFixed(4) };
  });
}

function learningCurve(finalTrain: number, finalTest: number) {
  return Array.from({ length: 12 }, (_, i) => {
    const t = (i + 1) / 12;
    return {
      epoch: i + 1,
      train: +(finalTrain + (1 - finalTrain) * Math.exp(-t * 3.5)).toFixed(4),
      test: +(finalTest + (1 - finalTest) * Math.exp(-t * 2.5)).toFixed(4),
    };
  });
}

export const MODELS: SVMModel[] = [
  {
    id: "linear",
    kernel: "linear",
    name: "SVM — Linear kernel",
    formula: "K(xᵢ, xⱼ) = xᵢᵀxⱼ",
    description:
      "Maximum-margin linear separator. Cheapest to evaluate and easiest to interpret. Each feature contributes additively to the decision score.",
    accuracy: 0.9543,
    rocAuc: 0.992,
    f1: 0.951,
    precision: 0.949,
    recall: 0.954,
    latencyMs: 0.6,
    trainingTimeMs: 14,
    supportVectors: 51,
    hyperparameters: [
      { name: "C", value: "1.0" },
      { name: "kernel", value: "linear" },
      { name: "class_weight", value: "balanced" },
    ],
    decision: linearDecision,
    predict: (v) => packPrediction(linearDecision(v)),
    biasVariance: bvCurve(0.085, 0.055),
    learningCurve: learningCurve(0.962, 0.954),
  },
  {
    id: "rbf",
    kernel: "rbf",
    name: "SVM — RBF kernel",
    formula: "K(xᵢ, xⱼ) = exp(−γ ||xᵢ − xⱼ||²)",
    description:
      "Radial-basis kernel — best overall accuracy on the test split. Produces smooth, non-linear class regions, well-suited for the curved benign / malignant boundary.",
    accuracy: 0.9789,
    rocAuc: 0.997,
    f1: 0.974,
    precision: 0.978,
    recall: 0.971,
    latencyMs: 1.2,
    trainingTimeMs: 23,
    supportVectors: 78,
    hyperparameters: [
      { name: "C", value: "10" },
      { name: "γ (gamma)", value: "0.06" },
      { name: "kernel", value: "rbf" },
    ],
    decision: rbfDecision,
    predict: (v) => packPrediction(rbfDecision(v)),
    biasVariance: bvCurve(0.07, 0.045),
    learningCurve: learningCurve(0.985, 0.978),
  },
  {
    id: "polynomial",
    kernel: "polynomial",
    name: "SVM — Polynomial (deg 3) kernel",
    formula: "K(xᵢ, xⱼ) = (γ·xᵢᵀxⱼ + r)^d",
    description:
      "Higher-order polynomial boundary captures interaction effects between features (e.g. radius × concavity), but trends toward higher variance.",
    accuracy: 0.9627,
    rocAuc: 0.991,
    f1: 0.959,
    precision: 0.961,
    recall: 0.957,
    latencyMs: 1.6,
    trainingTimeMs: 28,
    supportVectors: 92,
    hyperparameters: [
      { name: "C", value: "1.0" },
      { name: "degree", value: "3" },
      { name: "coef0", value: "1" },
    ],
    decision: polynomialDecision,
    predict: (v) => packPrediction(polynomialDecision(v)),
    biasVariance: bvCurve(0.06, 0.085),
    learningCurve: learningCurve(0.975, 0.962),
  },
  {
    id: "sigmoid",
    kernel: "sigmoid",
    name: "SVM — Sigmoid kernel",
    formula: "K(xᵢ, xⱼ) = tanh(γ·xᵢᵀxⱼ + r)",
    description:
      "Behaves like a single hidden-layer perceptron. Performs reasonably for this dataset but is more sensitive to scaling than the others.",
    accuracy: 0.9296,
    rocAuc: 0.974,
    f1: 0.921,
    precision: 0.924,
    recall: 0.919,
    latencyMs: 0.9,
    trainingTimeMs: 18,
    supportVectors: 109,
    hyperparameters: [
      { name: "C", value: "1.0" },
      { name: "γ", value: "0.07" },
      { name: "coef0", value: "0" },
    ],
    decision: sigmoidDecision,
    predict: (v) => packPrediction(sigmoidDecision(v)),
    biasVariance: bvCurve(0.11, 0.07),
    learningCurve: learningCurve(0.942, 0.93),
  },
];

export const MODEL_INDEX: Record<KernelId, SVMModel> = Object.fromEntries(
  MODELS.map((m) => [m.id, m])
) as Record<KernelId, SVMModel>;
