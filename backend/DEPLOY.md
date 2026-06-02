# Deploying the Tumor Sense backend

This guide covers **Render** and **Railway**. Both run the same FastAPI app;
the only difference is how you wire the service in each platform's dashboard.

The trained model weights (~190 MB total) are **not** in git (see `.gitignore`).
On every deploy / cold start the service runs `backend/scripts/download_weights.py`,
which fetches any missing `.pth` / `.joblib` files from a public URL you
configure via `TUMORSENSE_WEIGHTS_BASE_URL` (typically a Hugging Face Hub repo).

---

## 0. One-time prep — publish the weights

Upload `backend/models/` to a **public** Hugging Face model repo. From the
**repo root** (`v0-tumor-sense`):

```bash
# optional: use the project venv
source backend/.venv/bin/activate

pip install huggingface_hub   # provides the `hf` CLI

hf auth login

# create the repo once
hf repo create tumorsense-weights --type model

# upload all weights + SVM joblibs (run from repo root)
hf upload <your-username>/tumorsense-weights backend/models/ . \
  --commit-message "Initial upload"
```

For a large folder you can also use `hf upload-large-folder`.

Confirm the files appear at `https://huggingface.co/<your-username>/tumorsense-weights`.

Your base URL for every host below:

```
https://huggingface.co/<your-username>/tumorsense-weights/resolve/main
```

(Render / Railway append each filename, e.g. `.../resolve/main/resnet18_weights.pth`.)

**Alternative hosts** (any public HTTPS base works):

| Host | Base URL pattern |
| --- | --- |
| GitHub Releases | `https://github.com/<user>/<repo>/releases/download/<tag>/` |
| AWS S3 (public) | `https://<bucket>.s3.<region>.amazonaws.com/<prefix>/` |
| Cloudflare R2   | `https://<account>.r2.cloudflarestorage.com/<bucket>/` |

Per-file override: `TUMORSENSE_WEIGHT_URL_<FILENAME>` (uppercase, dots → underscores),
e.g. `TUMORSENSE_WEIGHT_URL_VGG16_WEIGHTS_PTH=https://.../VGG16_weights.pth`.

---

## Environment variables (all hosts)

Set these on Render, Railway, or Docker — names are identical.

| Key | Required | Example | Notes |
| --- | --- | --- | --- |
| `TUMORSENSE_WEIGHTS_BASE_URL` | **Yes** (prod) | `https://huggingface.co/<user>/tumorsense-weights/resolve/main` | No trailing slash on the base |
| `TUMORSENSE_CORS_ORIGINS` | Recommended | `https://your-app.vercel.app` | Comma-separated origins; `*` for quick tests |
| `OPENAI_API_KEY` | No | `sk-...` | Enables LLM-backed `/api/explain`; template fallback if unset |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Only used when `OPENAI_API_KEY` is set |
| `TUMORSENSE_MODELS_DIR` | No | `/app/backend/models` | Docker sets this automatically; Nixpacks usually needs the default |
| `PORT` | Auto | — | Set by Render / Railway; do not hard-code in app code |

---

## Health check (all hosts)

After deploy, open `https://<your-service-url>/health`. You want:

```json
{
  "status": "ok",
  "svm_models_loaded": ["rbf", "linear", "poly", "sigmoid"],
  "image_models_available": ["baseline_cnn", "vgg16", "mobilenet_v2", "resnet18", "vit_tiny"],
  "rag_ready": true
}
```

If `image_models_available` is empty, check deploy logs for `[download_weights]`
— usually a missing or private HF repo, or `TUMORSENSE_WEIGHTS_BASE_URL` not set.

Swagger UI: `https://<your-service-url>/docs`

---

## Connect the frontend

Wherever the Next.js app is hosted (Vercel, etc.), set:

```
NEXT_PUBLIC_API_URL=https://<your-service-url>
```

(no trailing slash). Redeploy the frontend. The `/api-docs` page, `/model` RAG
panel, and `lib/api.ts` will use the new base URL automatically.

---

## Render

The repo includes `render.yaml` for a one-click **Blueprint** deploy.

### Render — Option A: Blueprint (native Python)

1. Push this repo to GitHub.
2. Render → **New +** → **Blueprint** → select the repo → service `tumorsense-api`.
3. Before **Apply**, **Edit Environment** and set at least `TUMORSENSE_WEIGHTS_BASE_URL`
   and `TUMORSENSE_CORS_ORIGINS` (see table above). `render.yaml` ships
   `TUMORSENSE_WEIGHTS_BASE_URL` as empty — you must fill it in.
4. **Apply**. First build is often 5–10 minutes (`torch` + deps). Default plan in
   `render.yaml` is **starter** (~2 GB RAM).

**What Render runs** (from `render.yaml`):

```bash
pip install -r backend/requirements.txt
python backend/scripts/download_weights.py
uvicorn backend.server:app --host 0.0.0.0 --port $PORT --workers 1
```

| Plan | RAM | Notes |
| --- | --- | --- |
| Free | 512 MB | Spins down after idle; image models may OOM |
| Starter | 2 GB | Recommended — all five image models fit |

For Free, change `plan: starter` → `plan: free` in `render.yaml`.

### Render — Option B: Docker

Use when the native build OOMs on `pip install torch`.

```bash
docker build -t tumorsense-api -f backend/Dockerfile .
docker run --rm -p 8000:8000 \
  -e TUMORSENSE_WEIGHTS_BASE_URL="https://huggingface.co/<user>/tumorsense-weights/resolve/main" \
  tumorsense-api
curl localhost:8000/health
```

Render → **New +** → **Web Service** → **Deploy an existing image**, or configure
the Dockerfile path in the service settings. Use the same env vars as Option A.

---

## Railway

There is no `railway.toml` in the repo yet — configure the service in the
[Railway dashboard](https://railway.app) after connecting GitHub.

### Railway — Option A: Dockerfile (recommended)

Best match for the heavy `torch` + `langchain` stack; same image as local Docker.

1. **New Project** → **Deploy from GitHub** → select `v0-tumor-sense`.
2. Open the service **Settings**:
   - **Root Directory:** `/` (repository root — **not** `backend/`)
   - **Dockerfile Path:** `backend/Dockerfile`
3. **Variables** → add at minimum:

   ```
   TUMORSENSE_WEIGHTS_BASE_URL=https://huggingface.co/<user>/tumorsense-weights/resolve/main
   TUMORSENSE_CORS_ORIGINS=https://<your-frontend-host>
   ```

   Optional: `OPENAI_API_KEY`, `OPENAI_MODEL=gpt-4o-mini`.

   You do **not** need to set `PORT` — Railway injects it; the Dockerfile CMD
   already uses `${PORT:-8000}`.

4. **Settings** → **Networking** → **Generate Domain**.
5. Open `https://<your-railway-domain>/health` and confirm the JSON above.

The Dockerfile CMD runs `download_weights.py` then `uvicorn` on every container start.

### Railway — Option B: Nixpacks / custom start (no Docker)

If Railway auto-detects Python instead of the Dockerfile, set:

| Setting | Value |
| --- | --- |
| **Root Directory** | `/` (repo root) |
| **Build Command** | `pip install --upgrade pip && pip install -r backend/requirements.txt` |
| **Start Command** | `python backend/scripts/download_weights.py && uvicorn backend.server:app --host 0.0.0.0 --port $PORT --workers 1` |

Add the same **Variables** as Option A.

**Do not** use `backend/Procfile` alone — it only starts uvicorn and skips the
weight download step.

### Railway — RAM

Pick a plan with **≥ 2 GB RAM** if you want all five image models loaded without
OOM. SVM + RAG alone are lighter.

---

## Render vs Railway (quick comparison)

| | Render | Railway |
| --- | --- | --- |
| In-repo config | `render.yaml` (Blueprint) | Dashboard (or add `railway.toml` yourself) |
| Recommended path | Blueprint native Python | `backend/Dockerfile` |
| Weights on boot | `download_weights.py` in start command | Same (Docker CMD or custom start) |
| Public URL | `*.onrender.com` | `*.up.railway.app` (after Generate Domain) |
| Env vars | Same names | Same names |

---

## Common gotchas

| Symptom | Fix |
| --- | --- |
| `503 model not found at backend/models/...` | Set `TUMORSENSE_WEIGHTS_BASE_URL`; confirm HF repo is **public** and filenames match `backend/scripts/weights_manifest.json`. |
| `[download_weights] … no URL configured` | Base URL env var missing on the host. |
| Build fails at `pip install torch` with `MemoryError` | Use Docker (`backend/Dockerfile`) on Render or Railway. |
| `/api/explain` returns `503` for langchain | Build log must show `langchain` installed; check `backend/requirements.txt`. |
| Browser CORS error | Add exact frontend origin to `TUMORSENSE_CORS_ORIGINS` (no trailing slash). |
| Slow first `/api/explain` | Embedding model downloads on first use (~90 MB). Docker image pre-bakes MiniLM; native/Nixpacks may not. |
| `huggingface-cli` command not found | Use `hf` instead: `hf auth login`, `hf upload`, etc. |

---

## Local Docker (parity check)

```bash
# from repo root
docker build -t tumorsense-api -f backend/Dockerfile .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/backend/models:/app/backend/models" \
  -v "$(pwd)/backend/outputs:/app/backend/outputs" \
  tumorsense-api
```

Mounting `backend/models` skips the HF download when weights are already local.

**Local dev without Docker** (repo root):

```bash
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.server:app --reload --port 8000
```

Copy `backend/.env.example` → `backend/.env` for `OPENAI_API_KEY` and CORS settings.
