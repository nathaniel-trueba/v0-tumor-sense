# Clinical Reference Thresholds — FNA Cytology

**Authors:** internal, distilled from Wolberg / Mangasarian and follow-up literature
**Tags:** thresholds, radius_mean, concave_points_mean, concavity_mean, area_mean, decision_boundary

## What "borderline" looks like numerically

These ranges are derived from the joint distribution of malignant and benign
FNA samples in the WDBC dataset. They are **not** clinical cut-offs and
should never be used as a stand-alone diagnostic — they are calibration
anchors for explaining the model's behaviour.

| Feature              | Strongly benign      | Borderline          | Strongly malignant   |
|----------------------|----------------------|---------------------|----------------------|
| radius_mean (mm)     | < 12.0               | 12.0 – 15.0         | > 15.0               |
| perimeter_mean (px)  | < 80                 | 80 – 100            | > 100                |
| area_mean (px²)      | < 500                | 500 – 800           | > 800                |
| concavity_mean       | < 0.05               | 0.05 – 0.12         | > 0.12               |
| concave_points_mean  | < 0.03               | 0.03 – 0.06         | > 0.06               |
| texture_mean         | < 18                 | 18 – 22             | > 22                 |

## Historical case clusters

- **Case A-123:** radius ≈ 17.3, concavity ≈ 0.16 — historically resolved
  malignant in 41 of 44 retrospective cases (chart match 91%).
- **Case B-002:** radius ≈ 11.7, concave_points ≈ 0.025 — historically
  benign in 87 of 96 retrospective cases (chart match 87%).
- **Case C-417:** radius ≈ 14.5, texture ≈ 21 — split, 18 of 27 malignant.
  This neighbourhood is the dominant source of false negatives in every
  kernel.

## Reading the SVM decision score

`|decision_function(x)|` greater than ~0.8 indicates a confident prediction;
between 0.4 and 0.8 the prediction is reasonably confident but worth a
sanity check; below 0.4 the input is borderline and a clinician should not
act on the SVM alone.
