"use client";

import { useState } from "react";
import { Check, Copy, ExternalLink, Server, Sparkles, Image as ImageIcon, Activity, FileJson } from "lucide-react";
import { API_BASE_URL } from "@/lib/api";

type Method = "GET" | "POST";

interface Endpoint {
  method: Method;
  path: string;
  title: string;
  summary: string;
  request?: string; // JSON request body example
  response: string; // JSON response body example
  cls?: string; // tailwind background for method badge
}

const meta: Endpoint[] = [
  {
    method: "GET",
    path: "/health",
    title: "Service health",
    summary: "Reports which SVM kernels, image models, and RAG components are loaded. Use this as a deploy smoke check.",
    response: `{
  "status": "ok",
  "svm_models_loaded": ["rbf", "linear", "poly", "sigmoid"],
  "image_models_available": ["baseline_cnn", "vgg16", "mobilenet_v2", "resnet18", "vit_tiny"],
  "rag_ready": true
}`,
  },
  {
    method: "GET",
    path: "/api/feature-names",
    title: "Feature schema",
    summary: "The 10 WDBC 'mean' features the workshop exposes, in canonical order.",
    response: `{
  "feature_names": [
    "radius_mean", "texture_mean", "perimeter_mean",
    "area_mean", "smoothness_mean", "compactness_mean",
    "concavity_mean", "concave_points_mean",
    "symmetry_mean", "fractal_dimension_mean"
  ]
}`,
  },
  {
    method: "GET",
    path: "/api/metrics",
    title: "SVM training metrics",
    summary: "Raw contents of backend/outputs/svm_out/metrics.json: per-kernel best_params, CV AUC, test AUC, confusion matrices and full classification reports.",
    response: `{
  "rbf":    { "best_params": { "C": 10, "gamma": 0.01 }, "cv_auc": 0.9962, "test_auc": 0.9977, "confusion_matrix": [[41,1],[1,71]], "classification_report": { … } },
  "linear": { … },
  "poly":   { … },
  "sigmoid":{ … }
}`,
  },
];

const svm: Endpoint[] = [
  {
    method: "GET",
    path: "/api/models",
    title: "List SVM kernels",
    summary: "Returns the kernels backed by trained joblib bundles in backend/models.",
    response: `{ "available_kernels": ["rbf", "linear", "poly", "sigmoid"] }`,
  },
  {
    method: "POST",
    path: "/api/predict",
    title: "SVM inference",
    summary: "Run a single prediction. Pass any subset of the 10 mean features — missing ones are imputed with the dataset mean. The original 30 WDBC features are reconstructed before scaling (the trained SVMs expect 30-dim input).",
    request: `{
  "model": "rbf",
  "features": {
    "radius_mean": 17.2,
    "perimeter_mean": 113.0,
    "area_mean": 904.0,
    "concavity_mean": 0.18,
    "concave_points_mean": 0.10
  }
}`,
    response: `{
  "label": "malignant",
  "decision": 1.34,
  "probability_malignant": 0.91,
  "confidence": 0.91,
  "meta": {
    "kernel": "rbf",
    "latency_ms": 1.2,
    "n_support_vectors": 78,
    "model_path": "backend/models/model_rbf.joblib"
  }
}`,
  },
];

const image: Endpoint[] = [
  {
    method: "GET",
    path: "/api/image/models",
    title: "List image architectures",
    summary: "Returns the 5 image-classification models, with weights paths and whether each one has been lazily loaded into memory yet.",
    response: `[
  { "id": "baseline_cnn", "name": "Baseline 3-layer CNN", "family": "cnn", "params": 297218, "weights_path": ".../BaselineCNN_weights.pth", "weights_loaded": false },
  { "id": "vgg16",        "name": "VGG-16 (slim head)",   "family": "vgg",     … },
  { "id": "mobilenet_v2", "name": "MobileNet-V2",         "family": "mobilenet", … },
  { "id": "resnet18",     "name": "ResNet-18 (custom fc)","family": "resnet",  … },
  { "id": "vit_tiny",     "name": "ViT-Tiny (224 upsample)","family": "vit",   … }
]`,
  },
  {
    method: "POST",
    path: "/api/image/predict",
    title: "Image inference · base64",
    summary: "Accepts a base64-encoded 50×50 RGB patch (or any size — it's resized internally) and returns class probabilities + latency.",
    request: `{
  "model": "resnet18",
  "image_base64": "iVBORw0KGgoAAAANSUh…"
}`,
    response: `{
  "label": "cancer",
  "predicted_class": 1,
  "probability_no_cancer": 0.07,
  "probability_cancer":    0.93,
  "confidence": 0.93,
  "meta": {
    "model": "resnet18",
    "latency_ms": 12.4
  }
}`,
  },
  {
    method: "POST",
    path: "/api/image/predict-file",
    title: "Image inference · multipart",
    summary: "Same as /api/image/predict but takes a raw image upload (multipart form-data). Useful for curl-from-the-terminal demos.",
    request: `curl -X POST $API/api/image/predict-file \\
  -F "model=resnet18" \\
  -F "file=@patch.png"`,
    response: `// identical schema to /api/image/predict`,
  },
];

const rag: Endpoint[] = [
  {
    method: "POST",
    path: "/api/explain",
    title: "RAG-grounded explanation",
    summary: "Builds a natural-language rationale from the prediction + the user's adjusted features, grounded in the markdown corpus under backend/documents/. Returns the explanation plus the top-K retrieved passages with their similarity scores. Uses an OpenAI chat model when OPENAI_API_KEY is set, otherwise falls back to a deterministic template generator (still grounded on retrieval).",
    request: `{
  "model": "rbf",
  "features": {
    "radius_mean": 17.2,
    "concavity_mean": 0.18
  },
  "prediction": {
    "label": "malignant",
    "decision": 1.34,
    "probability_malignant": 0.91,
    "confidence": 0.91
  },
  "top_k": 4
}`,
    response: `{
  "explanation": "The model flagged this case as malignant with high confidence…",
  "bullets": ["radius_mean is above the malignant centroid", "…"],
  "retrieved": [
    {
      "id": "01_wolberg_mangasarian_1990",
      "title": "Wolberg & Mangasarian (1990): Multisurface method…",
      "source": "01_wolberg_mangasarian_1990.md",
      "summary": "Mean radius / perimeter / concave-points dominate…",
      "quote": "Across 569 fine-needle aspirates …",
      "tags": ["radius_mean", "perimeter_mean", "concave_points_mean"],
      "score": 0.812
    }
  ],
  "generator": "openai",
  "meta": { "model": "gpt-4o-mini", "retrieved_count": 4 }
}`,
  },
];

const SECTIONS: { id: string; label: string; icon: typeof Server; endpoints: Endpoint[]; intro: string }[] = [
  {
    id: "meta",
    label: "Meta",
    icon: Server,
    intro: "Service info, schema, and the raw training metrics file.",
    endpoints: meta,
  },
  {
    id: "svm",
    label: "SVM",
    icon: Activity,
    intro: "Inference against the four scikit-learn SVMs trained on the WDBC dataset.",
    endpoints: svm,
  },
  {
    id: "image",
    label: "Image classification",
    icon: ImageIcon,
    intro: "PyTorch image classification across BaselineCNN, VGG-16, MobileNet-V2, ResNet-18, and ViT-Tiny.",
    endpoints: image,
  },
  {
    id: "rag",
    label: "RAG",
    icon: Sparkles,
    intro: "LangChain retrieval-augmented explanations, grounded in the corpus under backend/documents/.",
    endpoints: rag,
  },
];

function methodClass(m: Method): string {
  return m === "GET"
    ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
    : "bg-foreground text-background";
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        } catch {
          // ignore (e.g. unfocused permissions)
        }
      }}
      className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] font-mono uppercase tracking-wider text-muted-foreground hover:text-foreground hover:bg-foreground/5 transition-colors"
      title="Copy to clipboard"
    >
      {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
      {copied ? "copied" : "copy"}
    </button>
  );
}

function CodeBlock({ children, label }: { children: string; label?: string }) {
  return (
    <div className="bg-foreground/[0.03] border border-foreground/10 rounded-md overflow-hidden">
      {label && (
        <div className="flex items-center justify-between px-3 py-2 border-b border-foreground/10 bg-foreground/[0.02]">
          <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">{label}</span>
          <CopyButton text={children} />
        </div>
      )}
      <pre className="text-xs font-mono leading-relaxed p-4 overflow-x-auto whitespace-pre">
        <code>{children}</code>
      </pre>
    </div>
  );
}

function EndpointCard({ e }: { e: Endpoint }) {
  const url = `${API_BASE_URL}${e.path}`;
  return (
    <div className="border border-foreground/10 bg-background rounded-md overflow-hidden">
      <div className="p-5 border-b border-foreground/10 space-y-3">
        <div className="flex items-baseline gap-3 flex-wrap">
          <span
            className={`inline-flex items-center px-2 py-0.5 rounded-sm text-[10px] font-mono uppercase tracking-wider ${methodClass(e.method)}`}
          >
            {e.method}
          </span>
          <code className="font-mono text-sm text-foreground">{e.path}</code>
          <a
            href={url}
            target="_blank"
            rel="noreferrer"
            className="ml-auto inline-flex items-center gap-1.5 text-[11px] font-mono text-muted-foreground hover:text-foreground transition-colors"
          >
            try it
            <ExternalLink className="w-3 h-3" />
          </a>
        </div>
        <h4 className="font-display text-xl">{e.title}</h4>
        <p className="text-sm text-foreground/80 leading-relaxed">{e.summary}</p>
      </div>
      <div className="p-5 space-y-4">
        {e.request && <CodeBlock label="request body">{e.request}</CodeBlock>}
        <CodeBlock label="response">{e.response}</CodeBlock>
      </div>
    </div>
  );
}

export function ApiDocs() {
  return (
    <div className="space-y-16">
      {/* Hero */}
      <header className="space-y-4">
        <span className="inline-flex items-center gap-3 text-sm font-mono text-muted-foreground">
          <span className="w-8 h-px bg-foreground/30" />
          API reference · v0.2
        </span>
        <h1 className="font-display text-5xl lg:text-7xl tracking-tight leading-[0.95]">
          The Tumor Sense
          <br />
          <span className="text-muted-foreground">inference API.</span>
        </h1>
        <p className="text-base lg:text-lg text-foreground/80 leading-relaxed max-w-2xl">
          A FastAPI server backing the workshops on this site: SVM tabular
          predictions, five image-classification networks, and a RAG-grounded
          explanation endpoint built on LangChain. Everything is JSON in, JSON
          out — interactive Swagger UI lives at{" "}
          <code className="font-mono text-foreground">{API_BASE_URL}/docs</code>.
        </p>
        <div className="flex flex-wrap items-center gap-3 pt-2">
          <a
            href={`${API_BASE_URL}/docs`}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs font-mono uppercase tracking-wider bg-foreground text-background hover:bg-foreground/90 transition-colors"
          >
            <FileJson className="w-3.5 h-3.5" />
            Open Swagger UI
            <ExternalLink className="w-3 h-3" />
          </a>
          <a
            href={`${API_BASE_URL}/health`}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 rounded-full border border-foreground/15 hover:border-foreground/40 px-4 py-2 text-xs font-mono uppercase tracking-wider text-foreground/80 hover:text-foreground transition-colors"
          >
            <Server className="w-3.5 h-3.5" />
            GET /health
          </a>
          <code className="text-[11px] font-mono text-muted-foreground">
            base URL · {API_BASE_URL}
          </code>
        </div>
      </header>

      {/* Quickstart */}
      <section className="grid lg:grid-cols-[1fr_minmax(0,2fr)] gap-px bg-foreground/10 border border-foreground/10">
        <div className="bg-background p-6 space-y-2">
          <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
            01 · Quickstart
          </span>
          <h3 className="font-display text-2xl">Three calls, end-to-end</h3>
          <p className="text-sm text-foreground/80 leading-relaxed">
            Health check → SVM predict → RAG explanation, in one go.
            Substitute <code className="font-mono">$API</code> with the base
            URL above.
          </p>
        </div>
        <div className="bg-background p-6">
          <CodeBlock label="curl">{`export API="${API_BASE_URL}"

# 1. is the server up?
curl -s $API/health | jq

# 2. predict — minimum payload is just the kernel id; other features get
#    imputed with the dataset mean.
curl -s -X POST $API/api/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "rbf",
    "features": {
      "radius_mean": 17.2,
      "concavity_mean": 0.18,
      "concave_points_mean": 0.10
    }
  }' | jq

# 3. explain — feed the prediction back in.
curl -s -X POST $API/api/explain \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "rbf",
    "features": { "radius_mean": 17.2, "concavity_mean": 0.18 },
    "prediction": {
      "label": "malignant",
      "decision": 1.34,
      "probability_malignant": 0.91,
      "confidence": 0.91
    },
    "top_k": 4
  }' | jq`}</CodeBlock>
        </div>
      </section>

      {/* Endpoint sections */}
      {SECTIONS.map((s) => {
        const Icon = s.icon;
        return (
          <section key={s.id} id={s.id} className="space-y-6 scroll-mt-32">
            <div className="flex items-baseline justify-between gap-4 flex-wrap">
              <div className="flex items-center gap-3">
                <Icon className="w-4 h-4 text-muted-foreground" />
                <h2 className="font-display text-3xl lg:text-4xl tracking-tight">{s.label}</h2>
              </div>
              <p className="text-sm text-muted-foreground max-w-xl text-right">{s.intro}</p>
            </div>
            <div className="grid lg:grid-cols-2 gap-4">
              {s.endpoints.map((e) => (
                <EndpointCard key={e.path + e.method} e={e} />
              ))}
            </div>
          </section>
        );
      })}

      {/* Errors */}
      <section className="space-y-4">
        <div className="flex items-center gap-3">
          <h2 className="font-display text-3xl lg:text-4xl tracking-tight">Errors</h2>
          <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
            standard FastAPI shape
          </span>
        </div>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-px bg-foreground/10 border border-foreground/10">
          {[
            { code: "400", title: "Bad request", desc: "Unknown model id, unknown feature name, or malformed image bytes." },
            { code: "404", title: "Not found", desc: "Returned by /api/metrics when the metrics.json file isn't bundled." },
            { code: "503", title: "Service unavailable", desc: "A model weight file is missing, or the LLM/embeddings backend isn't installed." },
            { code: "500", title: "Internal", desc: "Catch-all for unexpected inference failures. Check the server logs." },
          ].map((e) => (
            <div key={e.code} className="bg-background p-5 space-y-2">
              <div className="flex items-baseline gap-2">
                <span className="font-display text-2xl tabular-nums">{e.code}</span>
                <span className="text-xs font-mono text-muted-foreground">{e.title}</span>
              </div>
              <p className="text-xs leading-relaxed text-foreground/80">{e.desc}</p>
            </div>
          ))}
        </div>
        <CodeBlock label="error body">{`{ "detail": "human-readable message" }`}</CodeBlock>
      </section>

      {/* CORS / auth */}
      <section className="space-y-4">
        <h2 className="font-display text-3xl lg:text-4xl tracking-tight">CORS &amp; auth</h2>
        <div className="grid md:grid-cols-2 gap-px bg-foreground/10 border border-foreground/10">
          <div className="bg-background p-6 space-y-2">
            <h4 className="font-display text-xl">CORS</h4>
            <p className="text-sm text-foreground/80 leading-relaxed">
              Origin allow-list is configured via the{" "}
              <code className="font-mono">TUMORSENSE_CORS_ORIGINS</code> env
              var (comma-separated). Defaults to <code className="font-mono">*</code> for
              local development.
            </p>
          </div>
          <div className="bg-background p-6 space-y-2">
            <h4 className="font-display text-xl">Auth</h4>
            <p className="text-sm text-foreground/80 leading-relaxed">
              No authentication today — this server is meant for the in-site
              workshops and local exploration. For public deployment, gate
              behind a reverse proxy or sit it behind Cloudflare Access.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
