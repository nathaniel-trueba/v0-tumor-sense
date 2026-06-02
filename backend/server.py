"""Tumor Sense — FastAPI server.

Endpoints:

  GET  /health                  service status + which backends are loaded
  GET  /api/models              list trained SVM kernels
  GET  /api/feature-names       the 10 'mean' features the workshop exposes
  GET  /api/metrics             training / test metrics from outputs/svm_out
  POST /api/predict             SVM inference (Pydantic-validated)

  GET  /api/image/models        list image-classification architectures
  POST /api/image/predict       image inference (base64 body)
  POST /api/image/predict-file  image inference (multipart upload)

  POST /api/explain             RAG-grounded explanation of a prediction

Run with:

    cd backend
    uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .image_models import ImageInferer, WEIGHT_FILES
from .rag import RagPipeline
from .schemas import (
    ExplainRequest,
    ExplainResponse,
    FEATURE_NAMES,
    HealthResponse,
    ImageModelInfo,
    ImageModelLiteral,
    ImagePredictRequest,
    ImagePredictResponse,
    KernelLiteral,
    SVMPredictRequest,
    SVMPredictResponse,
)
from .svm import KERNELS, SVMRegistry

BACKEND_DIR = Path(__file__).parent
SVM_METRICS_PATH = BACKEND_DIR / "outputs" / "svm_out" / "metrics.json"


# ─── App + middleware ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Tumor Sense API",
    version="0.2.0",
    description=(
        "FastAPI server backing the Tumor Sense workshop. Serves SVM "
        "predictions, image-classification predictions, and a LangChain "
        "RAG-grounded explanation endpoint."
    ),
)

# CORS configurable via env so production deployments can lock it down.
_origins = os.environ.get("TUMORSENSE_CORS_ORIGINS", "*")
origins = [o.strip() for o in _origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ─── Lazy singletons ──────────────────────────────────────────────────────────

_svm: SVMRegistry | None = None
_image: ImageInferer | None = None
_rag: RagPipeline | None = None


def get_svm() -> SVMRegistry:
    global _svm
    if _svm is None:
        _svm = SVMRegistry()
        print(
            f"[server] svm registry ready: {len(_svm.bundles)} kernels "
            f"({', '.join(_svm.loaded_kernels())})"
        )
    return _svm


def get_image() -> ImageInferer:
    global _image
    if _image is None:
        _image = ImageInferer()
    return _image


def get_rag() -> RagPipeline:
    global _rag
    if _rag is None:
        _rag = RagPipeline()
    return _rag


@app.on_event("startup")
async def _eager_warm_svm() -> None:
    # SVM startup is cheap (joblib). Image + RAG stay lazy.
    get_svm()


# ─── Meta ─────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    svm = get_svm()
    img = get_image()
    available_images: list[ImageModelLiteral] = []
    for mid in WEIGHT_FILES:
        if (img.models_dir / WEIGHT_FILES[mid]).exists():
            available_images.append(mid)
    return HealthResponse(
        status="ok",
        svm_models_loaded=svm.loaded_kernels(),
        image_models_available=available_images,
        rag_ready=get_rag().ready,
    )


# ─── SVM ──────────────────────────────────────────────────────────────────────


@app.get("/api/models")
async def list_models() -> dict[str, list[KernelLiteral]]:
    return {"available_kernels": list(KERNELS)}


@app.get("/api/feature-names")
async def feature_names() -> dict[str, list[str]]:
    return {"feature_names": list(FEATURE_NAMES)}


@app.get("/api/metrics")
async def metrics() -> dict:
    if not SVM_METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail=f"metrics file not found at {SVM_METRICS_PATH}")
    with open(SVM_METRICS_PATH) as f:
        return json.load(f)


@app.post("/api/predict", response_model=SVMPredictResponse)
async def predict(req: SVMPredictRequest) -> SVMPredictResponse:
    try:
        return get_svm().predict(req)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"inference failed: {e}") from e


# ─── Image classification ────────────────────────────────────────────────────


@app.get("/api/image/models", response_model=list[ImageModelInfo])
async def image_models() -> list[ImageModelInfo]:
    return get_image().info()


@app.post("/api/image/predict", response_model=ImagePredictResponse)
async def image_predict(req: ImagePredictRequest) -> ImagePredictResponse:
    try:
        return get_image().predict(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"image inference failed: {e}") from e


@app.post("/api/image/predict-file", response_model=ImagePredictResponse)
async def image_predict_file(
    file: UploadFile = File(...),
    model: ImageModelLiteral = Form("resnet18"),
) -> ImagePredictResponse:
    import base64

    raw = await file.read()
    req = ImagePredictRequest(model=model, image_base64=base64.b64encode(raw).decode("ascii"))
    return await image_predict(req)


# ─── RAG explanation ─────────────────────────────────────────────────────────


@app.post("/api/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest) -> ExplainResponse:
    try:
        return get_rag().explain(req)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"explanation failed: {e}") from e


# ─── Local dev entrypoint ─────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=bool(int(os.environ.get("RELOAD", "1"))),
    )
