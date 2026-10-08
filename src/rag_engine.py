from __future__ import annotations

from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.data_utils import clean_text


class SimpleRAG:
    """A lightweight semantic retriever backed by FAISS embeddings."""

    def __init__(self, documents: List[dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self._build_index()

    def _build_index(self) -> None:
        texts = [clean_text(item["text"]) for item in self.documents]
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 4) -> List[dict]:
        query_vector = self.model.encode([clean_text(query)], convert_to_numpy=True, normalize_embeddings=True)
        query_vector = np.asarray(query_vector, dtype="float32")
        scores, indices = self.index.search(query_vector, top_k)

        results: List[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = dict(self.documents[idx])
            doc["score"] = float(score)
            results.append(doc)
        return results


def build_response(query: str, intent: str, sentiment: str, risk_level: str, retrieved_docs: List[dict]) -> str:
    """Compose a context-aware support response from the retrieved documents."""
    top_doc = retrieved_docs[0] if retrieved_docs else None
    if top_doc and "answer" in top_doc:
        base_template = top_doc["answer"]
    elif top_doc:
        base_template = top_doc["text"]
    else:
        base_template = "We do not have enough context yet. Please contact the helpdesk for a manual review."

    action_note = ""
    if intent == "Fraud":
        action_note = "Please secure the account immediately and escalate to the fraud team if the transaction appears unauthorized."
    elif intent == "Loan":
        action_note = "Please share the application number and any supporting documents if the issue is about disbursement or rejection."
    elif intent == "KYC":
        action_note = "Please ensure your documents are available and follow the KYC update procedure if your account is restricted."
    else:
        action_note = "Please provide any additional details so we can route the case correctly."

    return (
        f"Based on the retrieved banking guidance, here is a suggested response:\n"
        f"{base_template}\n\n"
        f"Customer sentiment: {sentiment}. Risk level: {risk_level}. \n"
        f"{action_note}"
    )
