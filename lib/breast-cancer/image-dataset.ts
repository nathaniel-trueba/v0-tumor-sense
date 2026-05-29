// Catalog of the 36 histopathology patches the user uploaded to /public/histopathology.
// Each patch is a 50x50 RGB image from the IDC (Invasive Ductal Carcinoma) dataset.
// Filenames follow:
//   grid_{NN}_{label}_{patient_id}_idx5_x{X}_y{Y}_class{0|1}.png
// class0 = IDC negative (no cancer in this patch)
// class1 = IDC positive (cancer in this patch)

export type ImageLabel = "cancer" | "no_cancer";

export interface PatchImage {
  id: string; // grid_NN
  filename: string;
  url: string;
  label: ImageLabel;
  patientId: string;
  x: number;
  y: number;
  // Synthetic seed used by feature/embedding generators so the same image
  // always projects to the same UMAP / Grad-CAM coordinates.
  seed: number;
}

const FILES: string[] = [
  "grid_01_no_cancer_10254_idx5_x1001_y1251_class0.png",
  "grid_02_cancer_12900_idx5_x2251_y1051_class1.png",
  "grid_03_no_cancer_10277_idx5_x1801_y1551_class0.png",
  "grid_04_cancer_10302_idx5_x1201_y2001_class1.png",
  "grid_05_no_cancer_12872_idx5_x951_y401_class0.png",
  "grid_06_cancer_12935_idx5_x1401_y2001_class1.png",
  "grid_07_no_cancer_10290_idx5_x1801_y1451_class0.png",
  "grid_08_cancer_12898_idx5_x1651_y301_class1.png",
  "grid_09_no_cancer_12954_idx5_x1001_y901_class0.png",
  "grid_10_cancer_9226_idx5_x1801_y2501_class1.png",
  "grid_11_no_cancer_10260_idx5_x2251_y1301_class0.png",
  "grid_12_cancer_12908_idx5_x2601_y1051_class1.png",
  "grid_13_no_cancer_9173_idx5_x351_y301_class0.png",
  "grid_14_cancer_10277_idx5_x1201_y1501_class1.png",
  "grid_15_no_cancer_12891_idx5_x2801_y401_class0.png",
  "grid_16_cancer_9023_idx5_x2051_y1951_class1.png",
  "grid_17_no_cancer_9173_idx5_x2951_y201_class0.png",
  "grid_18_cancer_12894_idx5_x1901_y401_class1.png",
  "grid_19_no_cancer_13400_idx5_x1601_y1801_class0.png",
  "grid_20_cancer_14211_idx5_x2451_y601_class1.png",
  "grid_21_no_cancer_10285_idx5_x1151_y1551_class0.png",
  "grid_22_cancer_9124_idx5_x901_y401_class1.png",
  "grid_23_no_cancer_9266_idx5_x2051_y51_class0.png",
  "grid_24_cancer_9077_idx5_x2351_y1251_class1.png",
  "grid_25_no_cancer_12826_idx5_x1801_y1901_class0.png",
  "grid_26_cancer_9081_idx5_x2601_y1101_class1.png",
  "grid_27_no_cancer_13106_idx5_x2701_y1901_class0.png",
  "grid_28_cancer_14155_idx5_x3401_y451_class1.png",
  "grid_29_no_cancer_9177_idx5_x2551_y701_class0.png",
  "grid_30_cancer_10277_idx5_x1501_y1551_class1.png",
  "grid_31_no_cancer_10301_idx5_x1951_y1101_class0.png",
  "grid_32_cancer_15473_idx5_x1501_y751_class1.png",
  "grid_33_no_cancer_12905_idx5_x2851_y601_class0.png",
  "grid_34_cancer_13613_idx5_x2051_y2101_class1.png",
  "grid_35_no_cancer_9029_idx5_x901_y1201_class0.png",
  "grid_36_cancer_10273_idx5_x1451_y2201_class1.png",
];

function parse(file: string, idx: number): PatchImage {
  // grid_{NN}_{label_words}_{patient_id}_idx5_x{X}_y{Y}_class{0|1}.png
  const stem = file.replace(/\.png$/, "");
  const parts = stem.split("_");
  // parts: [grid, NN, labelWord(s)..., patientId, idx5, xNNN, yNNN, classN]
  const id = `grid_${parts[1]}`;
  const isCancer = parts.includes("cancer") && !parts.includes("no");
  const label: ImageLabel = isCancer ? "cancer" : "no_cancer";
  const classIdx = parts.findIndex((p) => /^class\d+$/.test(p));
  const yIdx = classIdx - 1;
  const xIdx = classIdx - 2;
  const patientIdx = parts.findIndex((p) => p === "idx5") - 1;
  const patientId = parts[patientIdx];
  const x = parseInt(parts[xIdx].slice(1), 10);
  const y = parseInt(parts[yIdx].slice(1), 10);
  return {
    id,
    filename: file,
    url: `/histopathology/${file}`,
    label,
    patientId,
    x,
    y,
    seed: idx * 73 + (isCancer ? 13 : 31),
  };
}

export const IMAGE_CATALOG: PatchImage[] = FILES.map(parse);

export const IMAGE_INDEX: Record<string, PatchImage> = Object.fromEntries(
  IMAGE_CATALOG.map((p) => [p.id, p])
);

export function getDefaultImage(): PatchImage {
  // Reasonable default: first cancer-positive patch.
  return IMAGE_CATALOG.find((i) => i.label === "cancer") ?? IMAGE_CATALOG[0];
}

export const COUNTS = {
  total: IMAGE_CATALOG.length,
  cancer: IMAGE_CATALOG.filter((i) => i.label === "cancer").length,
  no_cancer: IMAGE_CATALOG.filter((i) => i.label === "no_cancer").length,
};
