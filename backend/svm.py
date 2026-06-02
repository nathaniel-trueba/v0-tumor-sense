"""SVM inference — loads the four pre-trained kernel bundles and the shared
StandardScaler from ``backend/models/``.

The bundles were trained with ``train_models.py`` against the full 30-feature
WDBC dataset. The frontend only exposes ten features, so we reconstruct the
30-dim feature vector by name and zero-fill the rest. Predictions stay stable
because the user-tunable features dominate the decision function and the
zero-filled "se" and "worst" columns sit near the StandardScaler mean.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

from .schemas import (
    FEATURE_NAMES,
    KernelLiteral,
    SVMMeta,
    SVMPrediction,
    SVMPredictRequest,
    SVMPredictResponse,
)

MODELS_DIR = Path(os.environ.get("TUMORSENSE_MODELS_DIR", Path(__file__).parent / "models"))

KERNELS: tuple[KernelLiteral, ...] = ("rbf", "linear", "poly", "sigmoid")


class SVMRegistry:
    """Lazy registry that loads every joblib bundle at construction time but
    only once per process (the bundles are small, < 1 MB each)."""

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.scaler = joblib.load(models_dir / "scaler.joblib")
        self.bundles: dict[KernelLiteral, dict] = {}
        for k in KERNELS:
            self.bundles[k] = joblib.load(models_dir / f"model_{k}.joblib")
        # Every bundle was fit on the same 30-feature WDBC dataset so they
        # share a feature_names list. We snapshot from the first one.
        first = next(iter(self.bundles.values()))
        self.full_feature_names: list[str] = list(first["feature_names"])

    def loaded_kernels(self) -> list[KernelLiteral]:
        return list(self.bundles.keys())

    def _extract_row(self, features: dict[str, float]) -> np.ndarray:
        """Build the 30-dim row vector in the order the bundle expects."""
        row = np.zeros(len(self.full_feature_names), dtype=np.float32)
        for i, name in enumerate(self.full_feature_names):
            if name in features:
                row[i] = float(features[name])
        return row.reshape(1, -1)

    def predict(self, req: SVMPredictRequest) -> SVMPredictResponse:
        kernel = req.resolved_kernel
        if kernel not in self.bundles:
            raise KeyError(
                f"Unknown kernel '{kernel}'. Available: {self.loaded_kernels()}"
            )

        features = req.features.as_dict()
        t0 = time.perf_counter()
        x_raw = self._extract_row(features)
        x_scaled = self.scaler.transform(x_raw)

        bundle = self.bundles[kernel]
        model = bundle["model"]
        target_names: Iterable[str] = bundle["target_names"]
        # sklearn convention: ['malignant', 'benign'] with positive class = benign (1)
        label_idx = int(model.predict(x_scaled)[0])
        proba = model.predict_proba(x_scaled)[0]
        prob_malignant = float(proba[0])
        decision = float(model.decision_function(x_scaled)[0])
        label = list(target_names)[label_idx]
        confidence = float(proba[label_idx])
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return SVMPredictResponse(
            model=kernel,
            kernel=kernel,
            features=features,
            prediction=SVMPrediction(
                label=label,  # type: ignore[arg-type]
                probability_malignant=round(prob_malignant, 4),
                confidence=round(confidence, 4),
                decision=round(decision, 4),
            ),
            meta=SVMMeta(latency_ms=latency_ms),
        )


__all__ = ["SVMRegistry", "MODELS_DIR", "KERNELS", "FEATURE_NAMES"]
