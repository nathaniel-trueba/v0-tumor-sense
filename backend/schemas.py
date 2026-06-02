"""Pydantic schemas for the FastAPI server.

All API payloads — both directions — are validated through these models. Field
names mirror the existing frontend contract so the React client can keep its
existing types unchanged.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ─── Shared kernel + label types ──────────────────────────────────────────────

KernelLiteral = Literal["rbf", "linear", "poly", "sigmoid"]
LabelLiteral = Literal["malignant", "benign"]

# These ten "mean" features form the public contract. The trained SVMs were
# fit on all 30 WDBC features; the frontend only exposes these 10, so we
# extract them by name and zero-fill the rest.
FEATURE_NAMES: tuple[str, ...] = (
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
)


# ─── /api/predict (SVM) ───────────────────────────────────────────────────────


class SVMFeatures(BaseModel):
    """The 10 'mean' WDBC features the workshop exposes to the user."""

    model_config = ConfigDict(extra="ignore")

    radius_mean: float = Field(ge=0)
    texture_mean: float = Field(ge=0)
    perimeter_mean: float = Field(ge=0)
    area_mean: float = Field(ge=0)
    smoothness_mean: float = Field(ge=0)
    compactness_mean: float = Field(ge=0)
    concavity_mean: float = Field(ge=0)
    concave_points_mean: float = Field(ge=0)
    symmetry_mean: float = Field(ge=0)
    fractal_dimension_mean: float = Field(ge=0)

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


class SVMPredictRequest(BaseModel):
    """Body for POST /api/predict.

    ``model`` and ``kernel`` are both accepted for backwards compatibility with
    the original Flask contract. If both are present they must agree.
    """

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    model: KernelLiteral | None = None
    kernel: KernelLiteral | None = None
    features: SVMFeatures

    @field_validator("kernel")
    @classmethod
    def _kernel_matches_model(cls, v, info):
        m = info.data.get("model")
        if m is not None and v is not None and m != v:
            raise ValueError(f"'model' ({m}) and 'kernel' ({v}) must match")
        return v

    @property
    def resolved_kernel(self) -> KernelLiteral:
        return self.model or self.kernel or "rbf"


class SVMPrediction(BaseModel):
    label: LabelLiteral
    probability_malignant: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    decision: float


class SVMMeta(BaseModel):
    latency_ms: float
    scaler: str = "StandardScaler"


class SVMPredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: KernelLiteral
    kernel: KernelLiteral
    features: dict[str, float]
    prediction: SVMPrediction
    meta: SVMMeta


# ─── /api/image/* (PyTorch CNNs) ──────────────────────────────────────────────

ImageModelLiteral = Literal[
    "baseline_cnn",
    "vgg16",
    "mobilenet_v2",
    "resnet18",
    "vit_tiny",
]

ImageLabelLiteral = Literal["no_cancer", "cancer"]


class ImagePredictRequest(BaseModel):
    """Body for POST /api/image/predict.

    Image is sent as a base64-encoded string. The image bytes themselves can be
    any format Pillow can open (PNG, JPEG, ...). Multipart upload is also
    supported by a separate route.
    """

    model_config = ConfigDict(protected_namespaces=())

    model: ImageModelLiteral = "resnet18"
    image_base64: str = Field(..., description="base64-encoded image bytes")


class ImagePrediction(BaseModel):
    label: ImageLabelLiteral
    probability_cancer: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    logits: list[float]


class ImagePredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: ImageModelLiteral
    prediction: ImagePrediction
    meta: dict[str, Any]


class ImageModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: ImageModelLiteral
    name: str
    family: str
    params: int
    weights_path: str
    weights_loaded: bool


# ─── /api/explain (RAG) ───────────────────────────────────────────────────────


class ExplainPrediction(BaseModel):
    """Just enough of the prediction to ground the explanation in the user's
    actual model output. We don't recompute the prediction server-side here —
    the client's `prediction` block is treated as authoritative."""

    label: LabelLiteral
    decision: float
    probability_malignant: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


class ExplainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: KernelLiteral = "rbf"
    features: SVMFeatures
    prediction: ExplainPrediction
    top_k: int = Field(default=4, ge=1, le=8)


class RetrievedPassage(BaseModel):
    id: str
    title: str
    source: str
    summary: str
    quote: str
    tags: list[str] = []
    score: float = Field(ge=0)


class ExplainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    explanation: str = Field(..., description="Grounded natural-language explanation")
    bullets: list[str] = Field(default_factory=list)
    retrieved: list[RetrievedPassage]
    generator: Literal["openai", "template"]
    meta: dict[str, Any]


# ─── Misc ─────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: Literal["ok"]
    svm_models_loaded: list[KernelLiteral]
    image_models_available: list[ImageModelLiteral]
    rag_ready: bool
