// Mock catalog of CNN architectures benchmarked on the IDC histopathology task.
// Each network defines its layer stack (used by the 3D architecture viz and the
// animated forward pass), pre-computed benchmark metrics, and a per-patch
// prediction function (mock — uses the patch label + a tiny dose of noise so
// the page feels live without a backend).

import { IMAGE_CATALOG, type PatchImage } from "./image-dataset";

export type LayerKind =
  | "input"
  | "conv"
  | "bn"
  | "relu"
  | "pool"
  | "residual"
  | "depthwise"
  | "flatten"
  | "fc"
  | "softmax";

export interface Layer {
  id: string;
  name: string;
  kind: LayerKind;
  // Output tensor shape [height, width, channels]. For FC: [1, 1, units].
  shape: [number, number, number];
  // Optional helpful details surfaced in the tooltip.
  detail?: string;
  receptiveField?: number;
  params?: number; // in thousands (K) — keeps the UI compact
  flops?: number; // in millions (M)
}

export interface NetworkPrediction {
  label: "cancer" | "no_cancer";
  probability: number; // P(cancer)
  confidence: number;
  // Decision value across the forward pass — used to animate per-layer
  // confidence inside the architecture viz.
  perLayerConfidence: number[];
}

export interface NetworkArchitecture {
  id: string;
  name: string;
  family: "cnn" | "resnet" | "vgg" | "mobilenet" | "vit" | "unet";
  paramsM: number; // total parameters, millions
  flopsG: number; // GFLOPs per forward pass
  trainedEpochs: number;
  layers: Layer[];
  // benchmark
  accuracy: number;
  f1: number;
  rocAuc: number;
  precision: number;
  recall: number;
  latencyMs: number; // per-patch
  // training curves (compact mock data)
  learningCurve: { epoch: number; train: number; val: number }[];
  biasVariance: { complexity: number; train: number; val: number }[];
  // confusion-matrix on a held-out 200-sample test split
  confusion: { tp: number; fp: number; fn: number; tn: number };
  description: string;
  // optional segmentation head
  segmentation?: {
    dice: number;
    iou: number;
  };
}

// ─── Curve helpers ─────────────────────────────────────────────────────────────
function learning(finalTrain: number, finalVal: number, epochs = 20) {
  return Array.from({ length: epochs }, (_, i) => {
    const t = (i + 1) / epochs;
    return {
      epoch: i + 1,
      train: +(finalTrain + (0.5 - finalTrain) * Math.exp(-t * 4.2)).toFixed(4),
      val: +(finalVal + (0.5 - finalVal) * Math.exp(-t * 3.4)).toFixed(4),
    };
  });
}

function bv(bias0: number, variance0: number, steps = 9) {
  return Array.from({ length: steps }, (_, i) => {
    const c = i + 1;
    const train = Math.max(0.005, bias0 * Math.exp(-c / 3));
    const val = Math.max(0.015, bias0 * Math.exp(-c / 2.5) + variance0 * (1 - Math.exp(-c / 4)));
    return { complexity: c, train: +train.toFixed(4), val: +val.toFixed(4) };
  });
}

// ─── Layer builders ────────────────────────────────────────────────────────────
function conv(
  id: string,
  channels: number,
  hw: number,
  receptive: number,
  paramsK: number,
  flopsM: number,
  detail?: string
): Layer {
  return {
    id,
    name: `Conv ${channels}`,
    kind: "conv",
    shape: [hw, hw, channels],
    receptiveField: receptive,
    params: paramsK,
    flops: flopsM,
    detail,
  };
}

function pool(id: string, hw: number, channels: number): Layer {
  return {
    id,
    name: `Pool ${hw}`,
    kind: "pool",
    shape: [hw, hw, channels],
    detail: "MaxPool 2x2 · stride 2",
  };
}

function residual(id: string, hw: number, channels: number, paramsK: number): Layer {
  return {
    id,
    name: `Residual ${channels}`,
    kind: "residual",
    shape: [hw, hw, channels],
    detail: "Identity + Conv block",
    params: paramsK,
  };
}

function fc(id: string, units: number, paramsK: number): Layer {
  return {
    id,
    name: `FC ${units}`,
    kind: "fc",
    shape: [1, 1, units],
    params: paramsK,
  };
}

const inputLayer: Layer = {
  id: "input",
  name: "Input",
  kind: "input",
  shape: [50, 50, 3],
  detail: "Histopathology patch · 50×50 RGB",
};

const softmaxLayer: Layer = {
  id: "softmax",
  name: "Softmax",
  kind: "softmax",
  shape: [1, 1, 2],
  detail: "P(no_cancer), P(cancer)",
};

// ─── Mock prediction ───────────────────────────────────────────────────────────
function makePredict(
  baseAcc: number,
  layerCount: number
): (image: PatchImage) => NetworkPrediction {
  return (image) => {
    // Use the seed for deterministic per-image noise.
    const noise = (Math.sin(image.seed * 12.9898) * 43758.5453) % 1;
    const truth = image.label === "cancer" ? 1 : 0;
    // Base prob aligned with the truth but pulled toward 0.5 if the model is
    // weak (lower baseAcc).
    const margin = baseAcc - 0.5;
    const probCancer = truth === 1
      ? 0.5 + margin + Math.abs(noise) * (1 - baseAcc) * 0.6
      : 0.5 - margin - Math.abs(noise) * (1 - baseAcc) * 0.6;
    const clamped = Math.max(0.02, Math.min(0.98, probCancer));
    const label: "cancer" | "no_cancer" = clamped > 0.5 ? "cancer" : "no_cancer";
    const confidence = label === "cancer" ? clamped : 1 - clamped;

    // Per-layer confidence: rises smoothly from 0.5 to the final probability so
    // the forward-pass animation feels like real conviction building up.
    const perLayerConfidence = Array.from({ length: layerCount }, (_, i) => {
      const t = (i + 1) / layerCount;
      const eased = 1 - Math.pow(1 - t, 2.2);
      return 0.5 + (clamped - 0.5) * eased;
    });

    return {
      label,
      probability: clamped,
      confidence,
      perLayerConfidence,
    };
  };
}

// ─── Architectures ─────────────────────────────────────────────────────────────
const SIMPLE_CNN_LAYERS: Layer[] = [
  inputLayer,
  conv("c1", 16, 50, 3, 0.5, 0.7, "3×3 · stride 1 · ReLU"),
  pool("p1", 25, 16),
  conv("c2", 32, 25, 7, 4.7, 2.9, "3×3 · ReLU"),
  pool("p2", 12, 32),
  conv("c3", 64, 12, 15, 18.5, 2.7, "3×3 · ReLU"),
  pool("p3", 6, 64),
  { id: "flat", name: "Flatten", kind: "flatten", shape: [1, 1, 2304] },
  fc("fc1", 128, 295.0),
  fc("fc2", 2, 0.3),
  softmaxLayer,
];

const RESNET18_LAYERS: Layer[] = [
  inputLayer,
  conv("stem", 32, 25, 7, 4.8, 3.0, "7×7 stride 2 · BN · ReLU"),
  pool("stem_pool", 13, 32),
  residual("r1a", 13, 64, 18.5),
  residual("r1b", 13, 64, 18.5),
  residual("r2a", 7, 128, 73.5),
  residual("r2b", 7, 128, 73.5),
  residual("r3a", 4, 256, 294.0),
  residual("r3b", 4, 256, 294.0),
  residual("r4a", 2, 512, 1180.0),
  residual("r4b", 2, 512, 1180.0),
  { id: "gap", name: "GAP", kind: "pool", shape: [1, 1, 512], detail: "Global average pool" },
  fc("fc_out", 2, 1.0),
  softmaxLayer,
];

const VGG16_SLIM_LAYERS: Layer[] = [
  inputLayer,
  conv("c1_1", 64, 50, 3, 1.7, 4.3),
  conv("c1_2", 64, 50, 5, 36.9, 92.2),
  pool("p1", 25, 64),
  conv("c2_1", 128, 25, 7, 73.8, 46.1),
  conv("c2_2", 128, 25, 9, 147.5, 92.2),
  pool("p2", 12, 128),
  conv("c3_1", 256, 12, 15, 294.9, 42.5),
  conv("c3_2", 256, 12, 17, 589.8, 85.0),
  pool("p3", 6, 256),
  { id: "flat", name: "Flatten", kind: "flatten", shape: [1, 1, 9216] },
  fc("fc1", 256, 2359.0),
  fc("fc2", 2, 0.5),
  softmaxLayer,
];

const MOBILENET_LAYERS: Layer[] = [
  inputLayer,
  conv("stem", 32, 25, 3, 0.9, 0.6),
  { id: "dw1", name: "DW 32", kind: "depthwise", shape: [25, 25, 32], detail: "Depthwise 3×3 + Pointwise" },
  { id: "dw2", name: "DW 64", kind: "depthwise", shape: [13, 13, 64], detail: "Depthwise 3×3 + Pointwise" },
  { id: "dw3", name: "DW 128", kind: "depthwise", shape: [13, 13, 128], detail: "Depthwise 3×3 + Pointwise" },
  { id: "dw4", name: "DW 128", kind: "depthwise", shape: [7, 7, 128], detail: "Depthwise 3×3 + Pointwise" },
  { id: "dw5", name: "DW 256", kind: "depthwise", shape: [7, 7, 256], detail: "Depthwise 3×3 + Pointwise" },
  { id: "dw6", name: "DW 512", kind: "depthwise", shape: [4, 4, 512], detail: "Depthwise 3×3 + Pointwise" },
  { id: "gap", name: "GAP", kind: "pool", shape: [1, 1, 512], detail: "Global average pool" },
  fc("fc_out", 2, 1.0),
  softmaxLayer,
];

const VIT_TINY_LAYERS: Layer[] = [
  inputLayer,
  { id: "patch", name: "Patch embed", kind: "conv", shape: [10, 10, 96], detail: "5×5 stride 5 patch embedding" },
  { id: "block1", name: "Attn 96", kind: "residual", shape: [10, 10, 96], detail: "MSA + MLP block ×1" },
  { id: "block2", name: "Attn 96", kind: "residual", shape: [10, 10, 96], detail: "MSA + MLP block ×2" },
  { id: "block3", name: "Attn 96", kind: "residual", shape: [10, 10, 96], detail: "MSA + MLP block ×3" },
  { id: "cls", name: "CLS pool", kind: "pool", shape: [1, 1, 96], detail: "Class token" },
  fc("fc_out", 2, 0.2),
  softmaxLayer,
];

const UNET_LAYERS: Layer[] = [
  inputLayer,
  conv("enc1", 32, 50, 3, 0.9, 0.7),
  pool("p1", 25, 32),
  conv("enc2", 64, 25, 7, 18.5, 11.5),
  pool("p2", 12, 64),
  conv("enc3", 128, 12, 15, 73.7, 10.6),
  { id: "bot", name: "Bottleneck", kind: "conv", shape: [6, 6, 256], detail: "Bottleneck 3×3 · ReLU" },
  { id: "up1", name: "Up 128", kind: "conv", shape: [12, 12, 128], detail: "Up-conv 2×2 + skip from enc3" },
  { id: "up2", name: "Up 64", kind: "conv", shape: [25, 25, 64], detail: "Up-conv 2×2 + skip from enc2" },
  { id: "up3", name: "Up 32", kind: "conv", shape: [50, 50, 32], detail: "Up-conv 2×2 + skip from enc1" },
  { id: "mask", name: "Mask", kind: "conv", shape: [50, 50, 1], detail: "1×1 sigmoid → tumor mask" },
];

export const NETWORKS: NetworkArchitecture[] = [
  {
    id: "simple-cnn",
    name: "SimpleCNN-3",
    family: "cnn",
    paramsM: 0.32,
    flopsG: 0.008,
    trainedEpochs: 25,
    layers: SIMPLE_CNN_LAYERS,
    accuracy: 0.842,
    f1: 0.831,
    rocAuc: 0.901,
    precision: 0.838,
    recall: 0.824,
    latencyMs: 0.9,
    learningCurve: learning(0.91, 0.842),
    biasVariance: bv(0.13, 0.09),
    confusion: { tp: 91, fp: 16, fn: 15, tn: 78 },
    description:
      "Tiny baseline: three 3×3 conv blocks with max-pool and a 128-unit FC head. Fast and surprisingly strong on 50×50 patches.",
    predict: undefined as never,
  } as unknown as NetworkArchitecture,
  {
    id: "resnet18",
    name: "ResNet-18",
    family: "resnet",
    paramsM: 11.7,
    flopsG: 0.34,
    trainedEpochs: 30,
    layers: RESNET18_LAYERS,
    accuracy: 0.917,
    f1: 0.913,
    rocAuc: 0.969,
    precision: 0.92,
    recall: 0.906,
    latencyMs: 2.1,
    learningCurve: learning(0.98, 0.917),
    biasVariance: bv(0.06, 0.045),
    confusion: { tp: 96, fp: 8, fn: 9, tn: 87 },
    description:
      "Residual learning unlocks deeper representations. Our best mid-size model on the IDC patches — strong recall without sacrificing precision.",
    predict: undefined as never,
  } as unknown as NetworkArchitecture,
  {
    id: "vgg16",
    name: "VGG-16 (slim)",
    family: "vgg",
    paramsM: 14.3,
    flopsG: 0.51,
    trainedEpochs: 28,
    layers: VGG16_SLIM_LAYERS,
    accuracy: 0.894,
    f1: 0.889,
    rocAuc: 0.95,
    precision: 0.891,
    recall: 0.887,
    latencyMs: 3.4,
    learningCurve: learning(0.985, 0.894),
    biasVariance: bv(0.04, 0.075),
    confusion: { tp: 93, fp: 11, fn: 10, tn: 86 },
    description:
      "Classic 3×3 stack — over-parameterised for 50×50 patches but reliable. Tends to overfit when run for too many epochs.",
    predict: undefined as never,
  } as unknown as NetworkArchitecture,
  {
    id: "mobilenet",
    name: "MobileNet-V2",
    family: "mobilenet",
    paramsM: 2.2,
    flopsG: 0.06,
    trainedEpochs: 28,
    layers: MOBILENET_LAYERS,
    accuracy: 0.886,
    f1: 0.879,
    rocAuc: 0.943,
    precision: 0.883,
    recall: 0.875,
    latencyMs: 1.3,
    learningCurve: learning(0.94, 0.886),
    biasVariance: bv(0.08, 0.06),
    confusion: { tp: 92, fp: 12, fn: 11, tn: 85 },
    description:
      "Depthwise-separable convolutions — the latency / accuracy sweet spot for edge inference.",
    predict: undefined as never,
  } as unknown as NetworkArchitecture,
  {
    id: "vit-tiny",
    name: "ViT-Tiny",
    family: "vit",
    paramsM: 0.95,
    flopsG: 0.07,
    trainedEpochs: 40,
    layers: VIT_TINY_LAYERS,
    accuracy: 0.878,
    f1: 0.872,
    rocAuc: 0.938,
    precision: 0.881,
    recall: 0.863,
    latencyMs: 1.8,
    learningCurve: learning(0.92, 0.878),
    biasVariance: bv(0.09, 0.08),
    confusion: { tp: 90, fp: 12, fn: 13, tn: 85 },
    description:
      "Patch-token transformer — competitive even at tiny scale; needs more epochs but generalises gracefully.",
    predict: undefined as never,
  } as unknown as NetworkArchitecture,
  {
    id: "unet",
    name: "U-Net (seg.)",
    family: "unet",
    paramsM: 1.9,
    flopsG: 0.21,
    trainedEpochs: 35,
    layers: UNET_LAYERS,
    accuracy: 0.901, // patch-level when threshold applied to predicted mask
    f1: 0.896,
    rocAuc: 0.954,
    precision: 0.892,
    recall: 0.9,
    latencyMs: 4.1,
    learningCurve: learning(0.94, 0.901),
    biasVariance: bv(0.07, 0.05),
    confusion: { tp: 94, fp: 10, fn: 9, tn: 87 },
    segmentation: { dice: 0.83, iou: 0.71 },
    description:
      "Encoder–decoder with skip connections. Produces per-pixel tumor masks alongside a patch-level classification.",
    predict: undefined as never,
  } as unknown as NetworkArchitecture,
];

// Attach predict() with closure over each network's accuracy + layer count.
for (const n of NETWORKS) {
  (n as unknown as { predict: (image: PatchImage) => NetworkPrediction }).predict = makePredict(
    n.accuracy,
    n.layers.length
  );
}

export const NETWORK_INDEX: Record<string, NetworkArchitecture & { predict: (image: PatchImage) => NetworkPrediction }> =
  Object.fromEntries(
    NETWORKS.map((n) => [n.id, n as NetworkArchitecture & { predict: (image: PatchImage) => NetworkPrediction }])
  );

export type Network = NetworkArchitecture & { predict: (image: PatchImage) => NetworkPrediction };

// Pre-compute embeddings: project all patches to 2D using a deterministic
// pseudo-embedding (cancer cluster tilted right, non-cancer cluster left).
// Each model gets its own slight perturbation so the embedding shifts when the
// user changes networks.
export function embeddingsFor(networkId: string) {
  const idx = NETWORKS.findIndex((n) => n.id === networkId);
  const tilt = (idx % 3) - 1;
  return IMAGE_CATALOG.map((img) => {
    const noise = Math.sin(img.seed * 1.7 + idx) * 0.6;
    const noise2 = Math.cos(img.seed * 2.3 + idx) * 0.6;
    const baseX = img.label === "cancer" ? 1.2 + tilt * 0.15 : -1.2 - tilt * 0.15;
    const baseY = img.label === "cancer" ? 0.4 : -0.4;
    return {
      image: img,
      x: baseX + noise,
      y: baseY + noise2,
    };
  });
}
