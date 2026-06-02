# "Why Should I Trust You?" — Explaining the Predictions of Any Classifier (LIME)

**Authors:** M. T. Ribeiro, S. Singh, C. Guestrin
**Venue:** ACM SIGKDD (2016)
**Tags:** lime, explainability, local, decision_boundary

## TL;DR

LIME explains a single prediction by fitting a tiny interpretable model
(weighted linear regression) on a cloud of perturbations of the input
sample. The output is a per-feature coefficient that says how that feature
locally affects the prediction.

## Reading the LIME output on this workshop

- Coefficients are **local** — they describe how the SVM behaves in a small
  neighbourhood around the user's input. The sign and magnitude can differ
  meaningfully between a near-boundary input and a deep-interior input.
- Where global SHAP and local LIME disagree, trust LIME for the **specific
  patient** in front of you and trust SHAP for what the model *typically*
  uses.

## When LIME helps clinicians most

- Borderline predictions (`|decision_function(x)| < 0.4`).
- Predictions where the user is uncertain about whether a moved slider was
  decisive.
- Sanity-checking a malignant call when the user changed only one or two
  features.
