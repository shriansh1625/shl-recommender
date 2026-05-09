"""
retriever.py — Catalog loader for SHL Recommender.

The catalog JSON is loaded once. We removed FAISS and sentence-transformers
since the full catalog is now passed directly into the Gemini context.
"""
import json
import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")


class Retriever:
    """Catalog manager for the SHL individual-test catalog."""

    def __init__(self):
        self.catalog: List[Dict] = self._load_catalog()
        # Fast lookup: URL → item
        self._url_map: Dict[str, Dict] = {item["url"]: item for item in self.catalog}

    # ── public ───────────────────────────────────────────────────────────────

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
