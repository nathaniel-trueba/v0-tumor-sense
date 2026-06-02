"""Image-classification inference for the histopathology breast-tissue patches.

Five architectures are supported, mirroring those defined in
``image_classification.ipynb``. The saved ``.pth`` files in
``backend/models/`` are pure ``state_dict``s (saved with
``torch.save(model.state_dict(), ...)``), so we must instantiate the matching
architecture before calling ``load_state_dict``.

Each model is wrapped behind a thin :class:`ImageInferer` registry that
lazy-loads weights on first use — this avoids forcing the user to pay the
torchvision pretrained-download cost at FastAPI startup just to talk to the
SVM endpoints.

Input contract: 50×50 RGB images, normalised with the ImageNet
mean/std the notebook used. Two output classes: ``no_cancer`` (0) and
``cancer`` (1).
"""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path
from typing import Callable

from .schemas import (
    ImageLabelLiteral,
    ImageModelInfo,
    ImageModelLiteral,
    ImagePrediction,
    ImagePredictRequest,
    ImagePredictResponse,
)

MODELS_DIR = Path(os.environ.get("TUMORSENSE_MODELS_DIR", Path(__file__).parent / "models"))

# Map model id → checkpoint filename (matches what's already on disk).
WEIGHT_FILES: dict[ImageModelLiteral, str] = {
    "baseline_cnn": "BaselineCNN_weights.pth",
    "vgg16": "VGG16_weights.pth",
    "mobilenet_v2": "MobileNetV2_weights.pth",
    "resnet18": "resnet18_weights.pth",
    "vit_tiny": "ViTTiny_weights.pth",
}

FAMILY: dict[ImageModelLiteral, str] = {
    "baseline_cnn": "cnn",
    "vgg16": "vgg",
    "mobilenet_v2": "mobilenet",
    "resnet18": "resnet",
    "vit_tiny": "vit",
}

DISPLAY_NAME: dict[ImageModelLiteral, str] = {
    "baseline_cnn": "Baseline 3-layer CNN",
    "vgg16": "VGG-16 (slim head)",
    "mobilenet_v2": "MobileNet-V2",
    "resnet18": "ResNet-18 (custom fc)",
    "vit_tiny": "ViT-Tiny (224 upsample)",
}

NUM_CLASSES = 2
LABELS: tuple[ImageLabelLiteral, ImageLabelLiteral] = ("no_cancer", "cancer")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# Heavy deps are torch-only; defer the import so the API can still serve SVM
# requests even on machines without torch installed.
def _torch_modules():
    import torch
    import torch.nn as nn
    from torchvision import models as tvm
    from torchvision import transforms
    from PIL import Image

    return torch, nn, tvm, transforms, Image


# ─── Architecture builders ────────────────────────────────────────────────────


def _build_baseline_cnn():
    _, nn, _, _, _ = _torch_modules()

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes: int = NUM_CLASSES) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 6 * 6, 128),  # 50 -> 25 -> 12 -> 6
                nn.ReLU(),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    return SimpleCNN()


def _build_vgg16():
    _, nn, tvm, _, _ = _torch_modules()
    model = tvm.vgg16(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, NUM_CLASSES),
    )
    return model


def _build_mobilenet_v2():
    _, nn, tvm, _, _ = _torch_modules()
    model = tvm.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    return model


def _build_resnet18():
    _, nn, tvm, _, _ = _torch_modules()
    model = tvm.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),
        nn.Linear(256, NUM_CLASSES),
    )
    return model


def _build_vit_tiny():
    _, nn, _, _, _ = _torch_modules()
    try:
        import timm
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "timm is required for the ViT-Tiny model — `pip install timm`"
        ) from e

    class ViTTinyWrapper(nn.Module):
        def __init__(self, num_classes: int = NUM_CLASSES) -> None:
            super().__init__()
            self.resize = nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)
            self.vit = timm.create_model(
                "vit_tiny_patch16_224",
                pretrained=False,
                num_classes=num_classes,
            )

        def forward(self, x):
            return self.vit(self.resize(x))

    return ViTTinyWrapper()


BUILDERS: dict[ImageModelLiteral, Callable[[], "object"]] = {
    "baseline_cnn": _build_baseline_cnn,
    "vgg16": _build_vgg16,
    "mobilenet_v2": _build_mobilenet_v2,
    "resnet18": _build_resnet18,
    "vit_tiny": _build_vit_tiny,
}


# ─── Inference registry ───────────────────────────────────────────────────────


class ImageInferer:
    """Holds lazy-loaded torch modules behind an in-process cache."""

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self._cache: dict[ImageModelLiteral, "object"] = {}
        self._transform = None
        self._device = None

    # ── public surface ────────────────────────────────────────────────────────

    def info(self) -> list[ImageModelInfo]:
        """Static metadata about every model, regardless of load state."""
        rows: list[ImageModelInfo] = []
        for mid, fname in WEIGHT_FILES.items():
            path = self.models_dir / fname
            params = 0
            if mid in self._cache:
                # Cheap to compute once loaded.
                params = sum(p.numel() for p in self._cache[mid].parameters())  # type: ignore[attr-defined]
            rows.append(
                ImageModelInfo(
                    id=mid,
                    name=DISPLAY_NAME[mid],
                    family=FAMILY[mid],
                    params=params,
                    weights_path=str(path),
                    weights_loaded=mid in self._cache,
                )
            )
        return rows

    def predict(self, req: ImagePredictRequest) -> ImagePredictResponse:
        torch, _, _, _, _ = _torch_modules()
        try:
            raw_bytes = base64.b64decode(req.image_base64, validate=True)
        except Exception as e:
            raise ValueError(f"image_base64 is not valid base64: {e}") from e

        model = self._load(req.model)
        device = self._get_device()

        t0 = time.perf_counter()
        tensor = self._image_to_tensor(raw_bytes).to(device)
        with torch.inference_mode():
            logits = model(tensor)  # type: ignore[operator]
            probs = torch.softmax(logits, dim=1)[0]
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        prob_cancer = float(probs[1].item())
        label_idx = int(probs.argmax().item())
        label = LABELS[label_idx]
        confidence = float(probs[label_idx].item())

        return ImagePredictResponse(
            model=req.model,
            prediction=ImagePrediction(
                label=label,
                probability_cancer=round(prob_cancer, 4),
                confidence=round(confidence, 4),
                logits=[round(float(l.item()), 4) for l in logits[0]],
            ),
            meta={
                "latency_ms": latency_ms,
                "device": str(device),
                "input_size": [50, 50],
                "normalize": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            },
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _get_device(self):
        torch, _, _, _, _ = _torch_modules()
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def _get_transform(self):
        if self._transform is None:
            _, _, _, transforms, _ = _torch_modules()
            self._transform = transforms.Compose(
                [
                    transforms.Resize((50, 50)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
        return self._transform

    def _image_to_tensor(self, raw_bytes: bytes):
        _, _, _, _, Image = _torch_modules()
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        tensor = self._get_transform()(img)
        return tensor.unsqueeze(0)

    def _load(self, mid: ImageModelLiteral):
        torch, _, _, _, _ = _torch_modules()
        if mid in self._cache:
            return self._cache[mid]

        if mid not in BUILDERS:
            raise KeyError(f"Unknown image model '{mid}'")

        path = self.models_dir / WEIGHT_FILES[mid]
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {mid}: {path}")

        model = BUILDERS[mid]()
        state = torch.load(path, map_location="cpu")
        # Notebook saves either a bare state_dict or a checkpoint dict; handle both.
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Strip any DataParallel "module." prefixes.
        state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)  # type: ignore[attr-defined]
        if missing or unexpected:
            # Not fatal — log so the operator sees it on stdout.
            print(
                f"[image_models] {mid}: missing={len(missing)} unexpected={len(unexpected)}"
            )
        model.eval()
        model = model.to(self._get_device())
        self._cache[mid] = model
        return model


__all__ = [
    "ImageInferer",
    "WEIGHT_FILES",
    "BUILDERS",
    "FAMILY",
    "DISPLAY_NAME",
    "LABELS",
]
