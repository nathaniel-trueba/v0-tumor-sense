# Support-Vector Networks

**Authors:** C. Cortes, V. Vapnik
**Venue:** Machine Learning, 20(3), pp. 273–297 (1995)
**Tags:** svm, linear, rbf, polynomial, kernel, decision_boundary

## Why an SVM

The SVM finds a separating hyperplane that maximises the margin to the closest
training points (the **support vectors**). The decision function
`f(x) = wᵀφ(x) + b` is signed — positive values fall on one side of the
boundary, negative on the other, and the magnitude `|f(x)|` quantifies how far
the input is from the boundary.

## Kernel intuition

| Kernel       | Decision shape                | When to prefer                                                            |
|--------------|-------------------------------|---------------------------------------------------------------------------|
| linear       | hyperplane                    | high-dim, additive contributions, easiest to interpret                    |
| polynomial   | curved, interaction-aware     | when feature interactions matter (e.g. radius × concavity)                |
| RBF          | smooth, locally radial        | best default for low-to-mid dimensional tabular data with non-linear class shapes |
| sigmoid      | tanh-like                     | behaves like a 1-hidden-layer perceptron; sensitive to scaling             |

## What `decision_function(x)` means in this project

For a malignant-positive convention the predicted label is `malignant` when
`f(x) ≥ 0` and `benign` when `f(x) < 0`. Inputs whose `|f(x)|` is below ~0.4
sit close to the boundary — small perturbations of the most important features
(radius_mean, perimeter_mean, concave_points_mean) can flip the label.

## Probability calibration

`SVC(probability=True)` fits a one-dim Platt-scaling model on the decision
score. The reported "confidence" is the probability of the predicted class,
**not** the magnitude of the decision score. Both are useful: confidence is
calibrated, decision score is more discriminative near the boundary.
