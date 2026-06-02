"""Download missing model weight files to ``$TUMORSENSE_MODELS_DIR``.

Render (and any other PaaS) does not ship the SVM joblibs or the .pth weights
through git, because torchvision checkpoints exceed GitHub's 100 MB per-file
limit and we don't want to pay for Git LFS.

This script is idempotent — already-present files are skipped — and is safe
to run on every cold-start as part of ``startCommand`` in ``render.yaml``.

Configuration:

    TUMORSENSE_MODELS_DIR          target directory (default backend/models)
    TUMORSENSE_WEIGHTS_BASE_URL    public base URL. Filename is appended.
                                   Examples:
                                     https://huggingface.co/<user>/<repo>/resolve/main/
                                     https://my-bucket.s3.amazonaws.com/tumorsense/
                                     https://github.com/<user>/<repo>/releases/download/v1/

Per-file explicit URLs can also be set via env vars, e.g.::

    TUMORSENSE_WEIGHT_URL_VGG16_WEIGHTS_PTH=https://…/VGG16_weights.pth

(filename uppercased, dots → underscores).

Run locally::

    python backend/scripts/download_weights.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "weights_manifest.json"

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR = Path(os.environ.get("TUMORSENSE_MODELS_DIR", DEFAULT_DIR))
BASE_URL = (os.environ.get("TUMORSENSE_WEIGHTS_BASE_URL") or "").rstrip("/")


def _per_file_env(filename: str) -> str:
    """Compute the env var name that can override a single file's URL."""
    safe = filename.upper().replace(".", "_").replace("-", "_")
    return f"TUMORSENSE_WEIGHT_URL_{safe}"


def _resolve_url(filename: str, explicit: str | None) -> str | None:
    env_override = os.environ.get(_per_file_env(filename))
    if env_override:
        return env_override
    if explicit:
        return explicit
    if BASE_URL:
        return f"{BASE_URL}/{filename}"
    return None


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  → fetching {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                f.write(chunk)
        tmp.rename(dest)
        print(f"    saved {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")
    except urllib.error.HTTPError as e:
        if tmp.exists():
            tmp.unlink()
        print(f"    ✗ HTTP {e.code} for {url}", file=sys.stderr)
        raise
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def main() -> int:
    if not MANIFEST_PATH.exists():
        print(f"manifest missing: {MANIFEST_PATH}", file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST_PATH.read_text())
    files: dict[str, dict] = manifest.get("files", {})

    print(f"[download_weights] models_dir = {MODELS_DIR}")
    print(f"[download_weights] base_url   = {BASE_URL or '(unset)'}")

    missing: list[tuple[str, dict]] = []
    for fname, meta in files.items():
        dest = MODELS_DIR / fname
        if dest.exists() and dest.stat().st_size > 0:
            continue
        missing.append((fname, meta))

    if not missing:
        print(f"[download_weights] all {len(files)} weight files already present — nothing to do")
        return 0

    print(f"[download_weights] need to fetch {len(missing)} / {len(files)} files")
    errors = 0
    for fname, meta in missing:
        url = _resolve_url(fname, meta.get("url"))
        if not url:
            print(
                f"  ! {fname}: no URL configured. Set TUMORSENSE_WEIGHTS_BASE_URL "
                f"or {_per_file_env(fname)}.",
                file=sys.stderr,
            )
            errors += 1
            continue
        try:
            _download(url, MODELS_DIR / fname)
        except Exception as e:  # pragma: no cover
            print(f"  ! {fname}: {e}", file=sys.stderr)
            errors += 1

    if errors:
        print(
            f"[download_weights] WARNING: {errors} file(s) could not be fetched. "
            f"Endpoints that need them will return 503 until they are present.",
            file=sys.stderr,
        )
    return 0  # never block startup — SVM + RAG should still come up


if __name__ == "__main__":
    sys.exit(main())
