# Wisconsin Diagnostic Breast Cancer (WDBC) Dataset

**Authors:** W. N. Street, W. H. Wolberg, O. L. Mangasarian
**Venue:** UCI Machine Learning Repository (1995)
**Tags:** dataset, radius_mean, perimeter_mean, area_mean, smoothness_mean, fractal_dimension_mean

## Composition

- 569 samples (357 benign, 212 malignant)
- 30 numeric features per sample: ten *mean* metrics, ten *standard error* metrics, and ten *worst* metrics
- Class labels: `malignant`, `benign`

## Selected feature definitions

| Feature                | Definition                                          |
|------------------------|-----------------------------------------------------|
| radius_mean            | mean distance from nucleus center to perimeter      |
| texture_mean           | std. of greyscale values inside the nucleus         |
| perimeter_mean         | average perimeter of the nucleus                    |
| area_mean              | average area of the nucleus                         |
| smoothness_mean        | local variation in radius lengths                   |
| compactness_mean       | perimeter² / area − 1.0                             |
| concavity_mean         | severity of concave portions of the contour        |
| concave_points_mean    | number of concave portions of the contour          |
| symmetry_mean          | symmetry of the nucleus shape                       |
| fractal_dimension_mean | "coastline" approximation − 1.0                     |

## Modelling implications

- **smoothness_mean** and **fractal_dimension_mean** have heavily overlapping
  class distributions. Models that rely on them in isolation are easy to
  fool with borderline cases.
- **concavity_mean** and **concave_points_mean** are highly correlated
  (r ≈ 0.92) — using both rarely adds information; using neither typically
  collapses test accuracy by 4–6%.
- All distance and area metrics are scale-dependent — never feed the raw
  features to an SVM, always standard-scale first.
