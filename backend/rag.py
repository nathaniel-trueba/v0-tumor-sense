"""LangChain RAG pipeline for the /api/explain endpoint.

Pipeline:

1. Load every markdown file in ``backend/documents/`` (one document per file).
2. Split into overlapping chunks with ``RecursiveCharacterTextSplitter``.
3. Embed with ``sentence-transformers/all-MiniLM-L6-v2`` (local, no API key).
4. Index in an in-memory FAISS vector store.
5. On each request, build a query that combines:
   - the predicted label and confidence,
   - the kernel name,
   - the user's notable feature values (those moved meaningfully off the midpoint),
6. Retrieve the top-K chunks, then either:
   - hand them to ``langchain-openai.ChatOpenAI`` for a grounded summary
     (when ``OPENAI_API_KEY`` is set), or
   - fall back to a deterministic template-based summariser that quotes the
     retrieved passages and ties them to the user's specific inputs.

The vector store is built lazily on first request so SVM-only deployments
don't pay the cold-start cost of downloading the embedding model.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import (
    ExplainRequest,
    ExplainResponse,
    RetrievedPassage,
)

DOCUMENTS_DIR = Path(os.environ.get("TUMORSENSE_DOCUMENTS_DIR", Path(__file__).parent / "documents"))

# These feature ids are what the frontend exposes — used both for "interesting
# feature" detection and as keywords boosted in retrieval queries.
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

# Coarse "is this value pushing toward malignant" thresholds — kept in sync
# with backend/documents/08_clinical_reference_thresholds.md so the model
# narrative agrees with the retrieved doc.
THRESHOLDS: dict[str, tuple[float, float]] = {
    "radius_mean": (12.0, 15.0),
    "perimeter_mean": (80.0, 100.0),
    "area_mean": (500.0, 800.0),
    "concavity_mean": (0.05, 0.12),
    "concave_points_mean": (0.03, 0.06),
    "texture_mean": (18.0, 22.0),
}

# Midpoint between class means for each feature (used for "actively adjusted"
# detection). Numbers match lib/breast-cancer/features.ts on the frontend.
CLASS_MIDPOINTS: dict[str, tuple[float, float]] = {
    # (midpoint, full range)
    "radius_mean": ((12.2 + 17.5) / 2, 24),
    "texture_mean": ((17.9 + 21.6) / 2, 31),
    "perimeter_mean": ((78.1 + 115.4) / 2, 160),
    "area_mean": ((462.8 + 978.4) / 2, 2360),
    "smoothness_mean": ((0.0925 + 0.1029) / 2, 0.12),
    "compactness_mean": ((0.0801 + 0.1452) / 2, 0.34),
    "concavity_mean": ((0.0461 + 0.16) / 2, 0.43),
    "concave_points_mean": ((0.0258 + 0.0879) / 2, 0.21),
    "symmetry_mean": ((0.1742 + 0.1928) / 2, 0.21),
    "fractal_dimension_mean": ((0.0629 + 0.0627) / 2, 0.055),
}

RFE_SELECTED = {
    "radius_mean",
    "perimeter_mean",
    "area_mean",
    "concavity_mean",
    "concave_points_mean",
    "texture_mean",
}


# ─── Helpers shared by both generator paths ───────────────────────────────────


@dataclass
class _FeatureSignal:
    name: str
    value: float
    direction: str  # 'malignant-leaning', 'benign-leaning', 'borderline'
    is_rfe: bool
    is_active: bool


def _summarise_features(features: dict[str, float]) -> list[_FeatureSignal]:
    """Tag each feature as malignant-leaning / benign-leaning / borderline."""
    signals: list[_FeatureSignal] = []
    for name in FEATURE_NAMES:
        v = float(features.get(name, 0.0))
        # Direction: only meaningful for features we have thresholds for.
        if name in THRESHOLDS:
            low, high = THRESHOLDS[name]
            if v >= high:
                direction = "malignant-leaning"
            elif v <= low:
                direction = "benign-leaning"
            else:
                direction = "borderline"
        else:
            direction = "borderline"
        # Active: moved noticeably away from the class midpoint.
        mid, rng = CLASS_MIDPOINTS[name]
        is_active = abs(v - mid) / rng > 0.02
        signals.append(
            _FeatureSignal(
                name=name,
                value=v,
                direction=direction,
                is_rfe=name in RFE_SELECTED,
                is_active=is_active,
            )
        )
    return signals


def _build_query(req: ExplainRequest, signals: list[_FeatureSignal]) -> str:
    notable = [s for s in signals if s.is_active and s.is_rfe]
    if not notable:
        notable = [s for s in signals if s.is_active][:3]
    descriptors = ", ".join(f"{s.name}={s.value:.3f} ({s.direction})" for s in notable[:5])
    return (
        f"Explain an SVM ({req.model} kernel) breast-tumor prediction of "
        f"'{req.prediction.label}' with decision score "
        f"{req.prediction.decision:+.3f} and confidence "
        f"{req.prediction.confidence * 100:.1f}%. "
        f"Notable inputs: {descriptors or 'none meaningfully adjusted'}. "
        f"Reference SVM kernels, RFE feature selection, and SHAP/LIME explainability."
    )


# ─── Document loading + chunking ──────────────────────────────────────────────


def _parse_doc(path: Path) -> tuple[str, str, list[str], str]:
    """Return (id, title, tags, body) for a single markdown document."""
    text = path.read_text()
    doc_id = path.stem
    # First H1 → title.
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else doc_id
    # `**Tags:** a, b, c` line → list of tags.
    tag_match = re.search(r"\*\*Tags:\*\*\s*(.+)", text)
    tags = [t.strip() for t in tag_match.group(1).split(",")] if tag_match else []
    return doc_id, title, tags, text


def _quote_for(passage: str, max_chars: int = 220) -> str:
    """Pick a representative sentence from a passage."""
    # Prefer the first sentence that is not a markdown header / table row.
    for line in passage.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("|") or line.startswith("-"):
            continue
        # Trim to roughly one sentence.
        if "." in line[:max_chars]:
            sentence = line[: line.index(".") + 1]
        else:
            sentence = line[:max_chars]
        return sentence.strip()
    # Fallback: just truncate the whole thing.
    return passage.strip()[:max_chars]


# ─── Template fallback (no API key) ───────────────────────────────────────────


def _build_template_explanation(
    req: ExplainRequest, signals: list[_FeatureSignal], passages: list[RetrievedPassage]
) -> tuple[str, list[str]]:
    pred = req.prediction
    is_malignant = pred.label == "malignant"
    confident = pred.confidence > 0.8
    near_boundary = abs(pred.decision) < 0.4

    pushing = [s for s in signals if s.is_active and s.is_rfe and (
        (is_malignant and s.direction == "malignant-leaning")
        or (not is_malignant and s.direction == "benign-leaning")
    )]
    pulling = [s for s in signals if s.is_active and s.is_rfe and (
        (is_malignant and s.direction == "benign-leaning")
        or (not is_malignant and s.direction == "malignant-leaning")
    )]
    active_count = sum(1 for s in signals if s.is_active)
    rfe_active_count = sum(1 for s in signals if s.is_active and s.is_rfe)
    rfe_ratio = rfe_active_count / max(1, active_count)

    lead = (
        f"The {req.model}-kernel SVM places this profile in the "
        f"{'malignant' if is_malignant else 'benign'} region of feature space "
        f"with {pred.confidence * 100:.1f}% confidence "
        f"(signed decision score {pred.decision:+.3f})."
    )

    if pushing:
        push_clause = (
            "The strongest drivers in your input are " +
            ", ".join(f"{s.name} = {s.value:.3f}" for s in pushing[:3]) +
            f" — each sits in the {'high' if is_malignant else 'low'} range that the "
            "reference documents flag as " + ("malignant-leaning" if is_malignant else "benign-leaning") + "."
        )
    else:
        push_clause = (
            "None of the six RFE-selected features sit deep in the "
            f"{'malignant' if is_malignant else 'benign'} range, so this prediction "
            "is being driven mostly by the joint geometry rather than any single feature."
        )

    if near_boundary:
        boundary_clause = (
            "Because |decision| < 0.4, this is a borderline case — small slider "
            "moves on radius_mean, perimeter_mean, or concave_points_mean can flip "
            "the label. Treat the confidence as a soft prior, not a verdict."
        )
    elif confident:
        boundary_clause = (
            "The decision magnitude indicates a confident call, and the retrieved "
            "passages back it up: the user input matches the historical pattern for this class."
        )
    else:
        boundary_clause = (
            "The decision sits in the moderate-confidence band — the call is reasonable "
            "but it is worth reviewing the bias-variance and SHAP panels before acting."
        )

    if pulling:
        pull_clause = (
            "Counter-evidence in your input: " +
            ", ".join(f"{s.name} = {s.value:.3f}" for s in pulling[:2]) +
            ", which pull in the opposite direction. The model is weighing them against the stronger drivers above."
        )
    else:
        pull_clause = ""

    if rfe_active_count == 0:
        rfe_clause = (
            "You have not yet moved any of the six RFE-selected features off "
            "their midpoint. Adjusting radius, perimeter, area, or concave-points "
            "would let the model lean on its strongest signals."
        )
    else:
        rfe_clause = (
            f"{rfe_active_count} of your {active_count} adjusted features "
            f"({rfe_ratio * 100:.0f}%) are in the RFE-selected set, so the model "
            f"is operating on its high-signal features."
        )

    # Ground every claim in a retrieved passage.
    citation_clause = ""
    if passages:
        cited = passages[0]
        citation_clause = (
            f' Grounded in {cited.title} (\"{cited.quote}\")'
            f"{'; further support in ' + passages[1].title + '.' if len(passages) > 1 else '.'}"
        )

    paragraphs = [lead, push_clause, boundary_clause]
    if pull_clause:
        paragraphs.append(pull_clause)
    paragraphs.append(rfe_clause + citation_clause)
    explanation = "\n\n".join(paragraphs)

    bullets = [lead, push_clause, boundary_clause]
    if pull_clause:
        bullets.append(pull_clause)
    bullets.append(rfe_clause)
    return explanation, bullets


# ─── LangChain RAG store ──────────────────────────────────────────────────────


class RagPipeline:
    """Lazily-initialised LangChain pipeline.

    All heavy imports happen the first time :meth:`explain` runs.
    """

    def __init__(self, docs_dir: Path = DOCUMENTS_DIR) -> None:
        self.docs_dir = docs_dir
        self._lock = threading.Lock()
        self._vectorstore = None
        self._splits: list[Any] = []
        self._documents_meta: dict[str, dict[str, Any]] = {}
        self._llm = None
        self._llm_init_attempted = False

    @property
    def ready(self) -> bool:
        return self._vectorstore is not None

    # ── public surface ────────────────────────────────────────────────────────

    def ensure_ready(self) -> None:
        """Build the vector store if it hasn't been built yet."""
        if self._vectorstore is not None:
            return
        with self._lock:
            if self._vectorstore is not None:
                return
            self._build_store()

    def explain(self, req: ExplainRequest) -> ExplainResponse:
        self.ensure_ready()
        signals = _summarise_features(req.features.as_dict())
        query = _build_query(req, signals)
        retrieved = self._retrieve(query, k=req.top_k)

        llm = self._get_llm()
        generator = "openai" if llm is not None else "template"
        if llm is not None:
            try:
                explanation, bullets = self._llm_explanation(llm, req, signals, retrieved)
            except Exception as e:  # pragma: no cover
                # LLM call failed mid-flight — fall back to the template generator.
                print(f"[rag] LLM call failed, falling back to template: {e}")
                explanation, bullets = _build_template_explanation(req, signals, retrieved)
                generator = "template"
        else:
            explanation, bullets = _build_template_explanation(req, signals, retrieved)

        return ExplainResponse(
            explanation=explanation,
            bullets=bullets,
            retrieved=retrieved,
            generator=generator,
            meta={
                "query": query,
                "documents_indexed": len(self._documents_meta),
                "chunks_indexed": len(self._splits),
                "rfe_active": [s.name for s in signals if s.is_active and s.is_rfe],
            },
        )

    # ── store + retrieval ─────────────────────────────────────────────────────

    def _build_store(self) -> None:
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_core.documents import Document
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as e:
            raise RuntimeError(
                "LangChain RAG deps not installed — see backend/requirements.txt"
            ) from e

        documents: list[Any] = []
        for md in sorted(self.docs_dir.glob("*.md")):
            doc_id, title, tags, body = _parse_doc(md)
            self._documents_meta[doc_id] = {"title": title, "tags": tags, "path": str(md)}
            documents.append(
                Document(
                    page_content=body,
                    metadata={"id": doc_id, "title": title, "tags": tags, "source": str(md.name)},
                )
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=80, separators=["\n## ", "\n### ", "\n\n", "\n", " "]
        )
        self._splits = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vectorstore = FAISS.from_documents(self._splits, embeddings)

    def _retrieve(self, query: str, k: int) -> list[RetrievedPassage]:
        assert self._vectorstore is not None
        docs_and_scores = self._vectorstore.similarity_search_with_score(query, k=k)
        passages: list[RetrievedPassage] = []
        for doc, score in docs_and_scores:
            meta = doc.metadata
            content = doc.page_content
            passages.append(
                RetrievedPassage(
                    id=meta.get("id", "unknown"),
                    title=meta.get("title", "Untitled"),
                    source=meta.get("source", ""),
                    summary=_quote_for(content, max_chars=180),
                    quote=_quote_for(content, max_chars=320),
                    tags=meta.get("tags", []) or [],
                    score=round(1.0 / (1.0 + float(score)), 4),  # FAISS returns L2; map to (0,1] similarity
                )
            )
        return passages

    # ── LLM init (optional) ───────────────────────────────────────────────────

    def _get_llm(self):
        if self._llm_init_attempted:
            return self._llm
        self._llm_init_attempted = True
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            return None
        try:
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.2,
                api_key=key,
                timeout=30,
            )
            return self._llm
        except Exception as e:  # pragma: no cover
            print(f"[rag] OpenAI LLM init failed, falling back to template: {e}")
            return None

    def _llm_explanation(
        self,
        llm: Any,
        req: ExplainRequest,
        signals: list[_FeatureSignal],
        passages: list[RetrievedPassage],
    ) -> tuple[str, list[str]]:
        from langchain_core.prompts import ChatPromptTemplate

        passage_block = "\n\n".join(
            f"[{i+1}] {p.title}\n{p.quote}" for i, p in enumerate(passages)
        )
        feature_block = "\n".join(
            f"- {s.name} = {s.value:.4f}  ({s.direction}{', RFE' if s.is_rfe else ''}"
            f"{', active' if s.is_active else ''})"
            for s in signals
            if s.is_active
        ) or "(no features moved off their class midpoint)"

        system_msg = (
            "You are a clinical-ML explainer for a breast-tumor SVM classifier. "
            "You speak in calm, accurate, non-diagnostic language. Always ground every "
            "claim in the supplied passages — cite them inline as [1], [2], etc. "
            "Never invent numbers. Never give clinical advice. Three short paragraphs "
            "plus a short bullet list of takeaways."
        )
        user_template = (
            "User prediction:\n"
            "  kernel = {kernel}\n"
            "  label = {label}\n"
            "  decision_score = {decision:+.3f}\n"
            "  confidence = {confidence_pct:.1f}%\n\n"
            "User feature values:\n{features}\n\n"
            "Retrieved passages:\n{passages}\n\n"
            "Write the explanation now."
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_msg), ("human", user_template)]
        )
        chain = prompt | llm
        response = chain.invoke(
            {
                "kernel": req.model,
                "label": req.prediction.label,
                "decision": req.prediction.decision,
                "confidence_pct": req.prediction.confidence * 100,
                "features": feature_block,
                "passages": passage_block,
            }
        )
        content = getattr(response, "content", str(response)).strip()
        # Pull out bullets (lines starting with '-' or '*').
        bullets = [
            re.sub(r"^[-*]\s*", "", line).strip()
            for line in content.splitlines()
            if re.match(r"^\s*[-*]\s+", line)
        ]
        return content, bullets


__all__ = ["RagPipeline", "DOCUMENTS_DIR"]
