# Gene Selection for Cancer Classification Using Support Vector Machines

**Authors:** I. Guyon, J. Weston, S. Barnhill, V. Vapnik
**Venue:** Machine Learning, 46, pp. 389–422 (2002)
**Tags:** rfe, feature_selection, radius_mean, perimeter_mean, area_mean, concave_points_mean

## Method recap

Recursive Feature Elimination (RFE) repeatedly trains a linear SVM, ranks each
feature by the squared component of the weight vector, and drops the lowest
rank. A 10-fold cross-validated F1 is recorded at every step; the final
feature set is the smallest subset whose F1 is statistically
indistinguishable from the full model.

## Results on the WDBC "mean" subset

After 10-fold RFE only six features survive the 0.05 significance cut:

1. radius_mean
2. perimeter_mean
3. area_mean
4. concavity_mean
5. concave_points_mean
6. texture_mean

`compactness_mean`, `symmetry_mean`, `smoothness_mean`, and
`fractal_dimension_mean` are eliminated — they carry redundant signal once the
six features above are known.

## Implication for the workshop

When the user moves sliders that are **not** in the RFE-selected set, the
prediction tends to barely move; the decision score is dominated by the six
features above. When auditing a prediction, prioritise checking these six.
