"""
retriever.py — FAISS-backed retrieval over the SHL catalog.

The catalog JSON is loaded once.  Embeddings are computed from
sentence-transformers (all-MiniLM-L6-v2) and kept in an in-memory
FAISS flat index (cosine similarity via L2-normalised inner product).
"""
import json
import logging
import os
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")
EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, ~22 MB


class Retriever:
    """Semantic retrieval over the SHL individual-test catalog."""

    def __init__(self):
        self.catalog: List[Dict] = self._load_catalog()
        self._model = SentenceTransformer(EMBED_MODEL)
        self._index, self._embeddings = self._build_index()
        # Fast lookup: URL → item
        self._url_map: Dict[str, Dict] = {item["url"]: item for item in self.catalog}

    # ── public ───────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 25) -> List[Dict]:
        """Return up to *top_k* catalog items most similar to *query*."""
        if not query.strip():
            return self.catalog[:top_k]

        q_vec = self._embed([query])
        faiss.normalize_L2(q_vec)

        k = min(top_k, len(self.catalog))
        scores, indices = self._index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                item = self.catalog[idx].copy()
                item["_score"] = float(score)
                results.append(item)
        return results

    def is_valid_url(self, url: str) -> bool:
        return url in self._url_map

    def get_by_name(self, name: str) -> List[Dict]:
        """Fuzzy-ish name lookup (case-insensitive substring)."""
        name_lower = name.lower()
        return [
            item for item in self.catalog
            if name_lower in item["name"].lower()
        ]

    # ── private ──────────────────────────────────────────────────────────────

    def _load_catalog(self) -> List[Dict]:
        if not os.path.exists(CATALOG_PATH):
            raise FileNotFoundError(
                f"catalog.json not found at {CATALOG_PATH}. "
                "Run scraper.py first (see README)."
            )
        with open(CATALOG_PATH, "r", encoding="utf-8") as fh:
            catalog = json.load(fh)
        logger.info("Loaded %d catalog items.", len(catalog))
        return catalog

    def _item_to_text(self, item: Dict) -> str:
        """Concatenate every useful field into a single searchable string."""
        fields = [
            item.get("name", ""),
            item.get("test_type_full", ""),
            " ".join(item.get("job_levels", [])),
            " ".join(item.get("job_families", [])),
            " ".join(item.get("competencies", [])),
            " ".join(item.get("languages", [])),
            item.get("description", ""),
        ]
        return " ".join(f for f in fields if f)

    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return vecs.astype(np.float32)

    def _build_index(self):
        texts = [self._item_to_text(item) for item in self.catalog]
        logger.info("Computing embeddings for %d catalog items …", len(texts))
        embeddings = self._embed(texts)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)   # cosine sim after L2 normalise
        index.add(embeddings)
        logger.info("FAISS index built (%d vectors, dim=%d).", index.ntotal, dim)
        return index, embeddings
