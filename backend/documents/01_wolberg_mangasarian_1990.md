# Multisurface Method of Pattern Separation for Medical Diagnosis Applied to Breast Cytology

**Authors:** W. H. Wolberg, O. L. Mangasarian
**Venue:** Proceedings of the National Academy of Sciences, 87(23), pp. 9193–9196 (1990)
**Tags:** radius_mean, perimeter_mean, area_mean, concave_points_mean, concavity_mean

## Premise

Fine-needle aspirate (FNA) samples from 569 patients were digitised and ten
morphologic features were measured per cell nucleus. The original study showed
that nuclear geometry alone — radius, perimeter, area, and the shape of the
boundary — could separate malignant from benign tissue with greater than 97%
cross-validated accuracy using a single linear classifier.

## Feature ranking (mean-summary subset)

| Rank | Feature              | Benign mean | Malignant mean |
|------|----------------------|-------------|----------------|
| 1    | radius_mean          | 12.2        | 17.5           |
| 2    | perimeter_mean       | 78.1        | 115.4          |
| 3    | area_mean            | 462.8       | 978.4          |
| 4    | concavity_mean       | 0.046       | 0.160          |
| 5    | concave_points_mean  | 0.026       | 0.088          |

Larger, more concave nuclei correlate strongly with invasive ductal carcinoma.

## What this implies for prediction

A patient profile whose radius_mean exceeds ~15 mm **and** whose concavity_mean
exceeds ~0.10 sits squarely in the malignant region of the linear separator
described above. Profiles with radius_mean < 13 mm and concave_points_mean <
0.04 land deep inside the benign region for every kernel evaluated since.
