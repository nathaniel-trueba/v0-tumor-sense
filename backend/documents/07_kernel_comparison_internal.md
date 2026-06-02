# Kernel Comparison Study — Internal Benchmark on UCI-WDBC

**Authors:** N. Trueba, K. Shah, S. Ngo, E. Park (DS3 Spring 2026)
**Venue:** Internal report
**Tags:** rbf, linear, polynomial, sigmoid, kernel, test_auc, confusion_matrix

## Setup

- 80 / 20 stratified split, `random_state=42`
- StandardScaler fit on training data only
- 5-fold stratified `GridSearchCV` scored on ROC-AUC
- Probability calibration via `SVC(probability=True)`

## Held-out test results (n = 114)

| Kernel    | CV-AUC | Test-AUC | Test accuracy | TP/FP/FN/TN     |
|-----------|--------|----------|----------------|------------------|
| rbf       | 0.9962 | 0.9977   | 98.2%          | 41 / 1 / 1 / 71  |
| linear    | 0.9955 | 0.9937   | 98.2%          | 41 / 1 / 1 / 71  |
| poly (d=2)| 0.9963 | 0.9980   | 98.2%          | 41 / 1 / 1 / 71  |
| sigmoid   | 0.9943 | 0.9960   | 96.5%          | 39 / 1 / 3 / 71  |

The RBF kernel narrows the boundary in the high-concavity region where the
linear kernel tends to over-classify benign borderline cases. Polynomial degree
2 wins by 3 thousandths of a point on test-AUC but is twice as slow at
inference time.

## Recommended default

For interactive use we ship **rbf** as the default — best margin shape, fast
enough at < 2 ms / sample, and most stable across feature subsets in our
ablation.
