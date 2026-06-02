/**
 * Backend API helpers.
 *
 * The base URL is read from `NEXT_PUBLIC_API_URL` so the frontend can talk to
 * the FastAPI server running locally (default http://localhost:8000) or to a
 * deployed instance. All endpoints are scoped under `/api/`.
 */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";

export type ExplainGenerator = "openai" | "template";

export interface ExplainRequestBody {
  model: "rbf" | "linear" | "poly" | "sigmoid";
  features: Record<string, number>;
  prediction: {
    label: "malignant" | "benign";
    decision: number;
    probability_malignant: number;
    confidence: number;
  };
  top_k?: number;
}

export interface RetrievedPassage {
  id: string;
  title: string;
  source: string;
  summary: string;
  quote: string;
  tags: string[];
  score: number;
}

export interface ExplainResponseBody {
  explanation: string;
  bullets: string[];
  retrieved: RetrievedPassage[];
  generator: ExplainGenerator;
  meta: Record<string, unknown>;
}

export async function explainPrediction(
  body: ExplainRequestBody,
  signal?: AbortSignal
): Promise<ExplainResponseBody> {
  const res = await fetch(`${API_BASE_URL}/api/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const json = (await res.json()) as { detail?: string };
      if (json?.detail) detail = json.detail;
    } catch {
      // ignore non-JSON error bodies
    }
    throw new Error(`POST /api/explain → ${res.status}: ${detail}`);
  }
  return (await res.json()) as ExplainResponseBody;
}
