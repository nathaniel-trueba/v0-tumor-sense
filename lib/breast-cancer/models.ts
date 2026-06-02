import { FEATURES, normalize } from "./features";

export type KernelId = "linear" | "polynomial" | "rbf" | "sigmoid";
export type DiagnosisLabel = "benign" | "malignant";

export interface SVMPrediction {
  label: DiagnosisLabel;
  decision: number; // signed distance to boundary (negative = benign, positive = malignant)
  probability: number; // 0..1 probability of malignant
  confidence: number; // 0..1 confidence in the predicted label
}

// Held-out confusion matrix using the sklearn convention with
// `target_names=['malignant', 'benign']` — row = actual, col = predicted.
// values are real test-split counts from backend/outputs/svm_out/metrics.json.
export interface ConfusionMatrix {
  // rows = actual, cols = predicted, ordered [malignant, benign]
  // matrix[0][0] = true malignant, matrix[0][1] = false benign
  // matrix[1][0] = false malignant, matrix[1][1] = true benign
  matrix: [[number, number], [number, number]];
  support: number; // total test samples
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
  confusionMatrix: ConfusionMatrix;
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

// All accuracy / AUC / precision / recall / F1 / confusion-matrix values below
// come straight from backend/outputs/svm_out/metrics.json. Precision, recall,
// and F1 are reported for the *malignant* class (the medically relevant
// positive). Latency, training time and support-vector counts are illustrative
// — they aren't in the metrics file and depend on hardware.
export const MODELS: SVMModel[] = [
  {
    id: "linear",
    kernel: "linear",
    name: "SVM — Linear kernel",
    formula: "K(xᵢ, xⱼ) = xᵢᵀxⱼ",
    description:
      "Maximum-margin linear separator. Cheapest to evaluate and easiest to interpret. Each feature contributes additively to the decision score.",
    accuracy: 0.9825,
    rocAuc: 0.9937,
    f1: 0.9762,
    precision: 0.9762,
    recall: 0.9762,
    latencyMs: 0.6,
    trainingTimeMs: 14,
    supportVectors: 51,
    hyperparameters: [
      { name: "C", value: "0.1" },
      { name: "kernel", value: "linear" },
    ],
    decision: linearDecision,
    predict: (v) => packPrediction(linearDecision(v)),
    biasVariance: bvCurve(0.045, 0.025),
    learningCurve: learningCurve(0.99, 0.9825),
    confusionMatrix: {
      matrix: [
        [41, 1],
        [1, 71],
      ],
      support: 114,
    },
  },
  {
    id: "rbf",
    kernel: "rbf",
    name: "SVM — RBF kernel",
    formula: "K(xᵢ, xⱼ) = exp(−γ ||xᵢ − xⱼ||²)",
    description:
      "Radial-basis kernel — top-tier accuracy on the test split. Produces smooth, non-linear class regions, well-suited for the curved benign / malignant boundary.",
    accuracy: 0.9825,
    rocAuc: 0.9977,
    f1: 0.9762,
    precision: 0.9762,
    recall: 0.9762,
    latencyMs: 1.2,
    trainingTimeMs: 23,
    supportVectors: 78,
    hyperparameters: [
      { name: "C", value: "10" },
      { name: "γ (gamma)", value: "0.01" },
      { name: "kernel", value: "rbf" },
    ],
    decision: rbfDecision,
    predict: (v) => packPrediction(rbfDecision(v)),
    biasVariance: bvCurve(0.04, 0.022),
    learningCurve: learningCurve(0.99, 0.9825),
    confusionMatrix: {
      matrix: [
        [41, 1],
        [1, 71],
      ],
      support: 114,
    },
  },
  {
    id: "polynomial",
    kernel: "polynomial",
    name: "SVM — Polynomial (deg 2) kernel",
    formula: "K(xᵢ, xⱼ) = (γ·xᵢᵀxⱼ + r)^d",
    description:
      "Degree-2 polynomial boundary captures pairwise interactions between features (e.g. radius × concavity) and reaches the highest test ROC-AUC of the four kernels.",
    accuracy: 0.9825,
    rocAuc: 0.998,
    f1: 0.9762,
    precision: 0.9762,
    recall: 0.9762,
    latencyMs: 1.6,
    trainingTimeMs: 28,
    supportVectors: 92,
    hyperparameters: [
      { name: "C", value: "1" },
      { name: "degree", value: "2" },
      { name: "coef0", value: "1.0" },
      { name: "γ (gamma)", value: "scale" },
    ],
    decision: polynomialDecision,
    predict: (v) => packPrediction(polynomialDecision(v)),
    biasVariance: bvCurve(0.045, 0.028),
    learningCurve: learningCurve(0.995, 0.9825),
    confusionMatrix: {
      matrix: [
        [41, 1],
        [1, 71],
      ],
      support: 114,
    },
  },
  {
    id: "sigmoid",
    kernel: "sigmoid",
    name: "SVM — Sigmoid kernel",
    formula: "K(xᵢ, xⱼ) = tanh(γ·xᵢᵀxⱼ + r)",
    description:
      "Behaves like a single hidden-layer perceptron. Trails the other three kernels on this dataset — three malignant cases slip through as false benigns.",
    accuracy: 0.9649,
    rocAuc: 0.996,
    f1: 0.9512,
    precision: 0.975,
    recall: 0.9286,
    latencyMs: 0.9,
    trainingTimeMs: 18,
    supportVectors: 109,
    hyperparameters: [
      { name: "C", value: "10" },
      { name: "γ (gamma)", value: "0.001" },
      { name: "coef0", value: "0.0" },
    ],
    decision: sigmoidDecision,
    predict: (v) => packPrediction(sigmoidDecision(v)),
    biasVariance: bvCurve(0.08, 0.04),
    learningCurve: learningCurve(0.975, 0.9649),
    confusionMatrix: {
      matrix: [
        [39, 3],
        [1, 71],
      ],
      support: 114,
    },
  },
];

export const MODEL_INDEX: Record<KernelId, SVMModel> = Object.fromEntries(
  MODELS.map((m) => [m.id, m])
) as Record<KernelId, SVMModel>;
