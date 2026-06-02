# A Unified Approach to Interpreting Model Predictions (SHAP)

**Authors:** S. M. Lundberg, S.-I. Lee
**Venue:** Advances in Neural Information Processing Systems (NeurIPS) (2017)
**Tags:** shap, explainability, attribution, radius_mean, perimeter_mean

## Idea in one paragraph

SHAP values are Shapley values from cooperative game theory applied to a
prediction: each feature is a "player" and the "payout" is the model's output
relative to the expected output. The result is an additive decomposition of
the prediction into per-feature contributions that respects local accuracy,
missingness, and consistency.

## Global SHAP ranking on this SVM

Across the 569 WDBC samples the global mean |SHAP| ranking, computed with the
KernelExplainer, is:

| Rank | Feature              | Mean |SHAP| |
|------|----------------------|-------------|
| 1    | radius_mean          | 0.91        |
| 2    | perimeter_mean       | 0.89        |
| 3    | area_mean            | 0.87        |
| 4    | concavity_mean       | 0.79        |
| 5    | concave_points_mean  | 0.76        |
| 6    | texture_mean         | 0.40        |

## How to read a single-prediction SHAP value

Positive SHAP ⇒ that feature pushed the prediction toward **malignant**.
Negative SHAP ⇒ it pulled toward **benign**. A prediction near the boundary
is one whose SHAP contributions roughly cancel out.
