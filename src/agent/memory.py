import json
import math
import struct

from . import db
from .embedding_client import EmbeddingClient
from .llm_client import LLMClient

STRONG_MATCH = 0.75
WEAK_MATCH = 0.5
TOP_K = 3
MIN_CANDIDATES = 2


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def categorize(llm: LLMClient, question: str) -> str:
    result = llm.complete_json(
        system_prompt="Classify the interview question into one category tag.",
        user_prompt=question,
        schema_hint='{"category": str}',
    )
    return result["category"]


def retrieve(conn, llm: LLMClient, embedder: EmbeddingClient, question: str) -> dict:
    category = categorize(llm, question)
    rows = db.list_memory_by_category(conn, category)

    query_vec = embedder.embed(question)
    scored = []
    for row in rows:
        score = _cosine(query_vec, _unpack(row["embedding"]))
        if score >= WEAK_MATCH:
            scored.append((score, row))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    if len(scored) < MIN_CANDIDATES:
        return {"status": "needs_input", "category": category}

    top = scored[:TOP_K]
    confidence = "strong" if top[0][0] >= STRONG_MATCH else "weak"
    return {
        "status": "match",
        "category": category,
        "confidence": confidence,
        "candidates": [row["canonical_answer"] for _, row in top],
    }


def synthesize_answer(llm: LLMClient, question: str, candidates: list[str]) -> str:
    result = llm.complete_json(
        system_prompt=(
            "Blend the given past answers into one answer for the new question. "
            "Respond with only the answer text itself, no commentary, no meta-discussion, "
            "and no requests for clarification."
        ),
        user_prompt=json.dumps({"question": question, "past_answers": candidates}),
        schema_hint='{"answer": str}',
    )
    return result["answer"]


def store_answer(
    conn,
    llm: LLMClient,
    embedder: EmbeddingClient,
    question: str,
    raw_answer: str,
    category: str,
) -> int:
    canonicalized = llm.complete_json(
        system_prompt=(
            "The user answered a form question with the given raw answer. Extract a "
            "generalized, reusable version of that exact answer for future similar "
            "questions. Respond with only the answer text itself, never a definition, "
            "explanation, or request for clarification, even if the answer is short or "
            "looks like a bare word or number. Stay as close to the raw answer as "
            "possible — do not add extra explanation, padding, or restate the question. "
            "Only generalize wording that was specific to this one application (e.g. a "
            "company name), nothing else."
        ),
        user_prompt=json.dumps({"question": question, "raw_answer": raw_answer}),
        schema_hint='{"canonical_answer": str}',
    )["canonical_answer"]
    vector = embedder.embed(canonicalized)
    return db.add_memory_entry(conn, question, canonicalized, _pack(vector), category)
