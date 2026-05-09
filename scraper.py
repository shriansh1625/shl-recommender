"""
scraper.py — One-time script to build catalog.json from the SHL product catalog.

The SHL catalog is JavaScript-rendered, so we use Playwright to load it fully.

Usage
-----
    pip install playwright beautifulsoup4
    playwright install chromium
    python scraper.py

Output
------
    catalog.json   ← commit this to your repo; the API loads it at start-up.

Notes
-----
* Only Individual Test Solutions are scraped (Pre-packaged Job Solutions are
  excluded as per the assignment brief).
* Run from the project root.  The script is idempotent.
"""

import json
import logging
import re
import time
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"

# SHL test-type abbreviations (from catalog icon titles)
TEST_TYPE_MAP: Dict[str, str] = {
    "Ability & Aptitude":              "A",
    "Biodata & Situational Judgement": "B",
    "Competencies":                    "C",
    "Development & 360":               "D",
    "Assessment Exercises":            "E",
    "Knowledge & Skills":              "K",
    "Personality & Behaviour":         "P",
    "Simulations":                     "S",
}


# ── Playwright scraper ────────────────────────────────────────────────────────

def scrape_with_playwright() -> List[Dict]:
    """
    Use Playwright to render the JS catalog page and extract all
    Individual Test Solution rows.
    """
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    items: List[Dict] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})

        start = 0
        page_size = 12          # SHL catalog shows 12 items per page

        while True:
            url = f"{CATALOG_URL}?start={start}&type=1"   # type=1 = Individual
            logger.info("Fetching %s …", url)

            try:
                page.goto(url, wait_until="networkidle", timeout=30_000)
            except PWTimeout:
                logger.warning("Timeout on %s; stopping pagination.", url)
                break

            # Wait for the table to appear
            try:
                page.wait_for_selector("table", timeout=10_000)
            except PWTimeout:
                logger.info("No table found — end of catalog.")
                break

            rows = page.query_selector_all("table tbody tr")
            if not rows:
                logger.info("Empty page — stopping.")
                break

            batch = []
            for row in rows:
                item = _parse_row(row)
                if item:
                    batch.append(item)

            if not batch:
                break

            items.extend(batch)
            logger.info("  Got %d items (total so far: %d)", len(batch), len(items))

            if len(batch) < page_size:
                break           # last page

            start += page_size
            time.sleep(1)       # be polite

        browser.close()

    # Enrich with detail pages
    logger.info("Fetching detail pages for %d items …", len(items))
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()

        for item in items:
            _enrich_from_detail(page, item)
            time.sleep(0.5)

        browser.close()

    return items


def _parse_row(row) -> Optional[Dict]:
    """Extract fields from a single catalog table row."""
    # Name + URL
    link = row.query_selector("a")
    if not link:
        return None
    name = (link.inner_text() or "").strip()
    href = link.get_attribute("href") or ""
    if not name or not href:
        return None
    url = href if href.startswith("http") else f"{BASE_URL}{href}"

    # Test type — look for icons / spans with title attributes
    test_type = "K"   # sensible default
    type_full = ""
    spans = row.query_selector_all("[title]")
    for span in spans:
        title = (span.get_attribute("title") or "").strip()
        if title in TEST_TYPE_MAP:
            test_type = TEST_TYPE_MAP[title]
            type_full = title
            break

    # Remote testing / Adaptive checkmarks
    row_text = row.inner_text()
    remote_testing = bool(re.search(r"remote", row_text, re.I))
    adaptive = bool(re.search(r"adaptive|irt", row_text, re.I))

    return {
        "name":          name,
        "url":           url,
        "test_type":     test_type,
        "test_type_full": type_full,
        "remote_testing": remote_testing,
        "adaptive":      adaptive,
        "duration":      None,
        "description":   "",
        "job_levels":    [],
        "job_families":  [],
        "languages":     ["English"],
        "competencies":  [],
    }


def _enrich_from_detail(page, item: Dict) -> None:
    """Visit the product detail page and fill in richer metadata."""
    try:
        page.goto(item["url"], wait_until="networkidle", timeout=20_000)
    except Exception as e:
        logger.debug("Could not load %s: %s", item["url"], e)
        return

    # Description — try common selectors
    for sel in [".product-description", ".hero-description", "meta[name='description']"]:
        el = page.query_selector(sel)
        if el:
            text = el.get_attribute("content") or el.inner_text()
            if text and len(text) > 20:
                item["description"] = text.strip()[:600]
                break

    # Duration
    page_text = page.inner_text("body") if page.query_selector("body") else ""
    dur = re.search(r"(\d{1,3})\s*(?:min(?:utes?)?)", page_text, re.I)
    if dur:
        item["duration"] = int(dur.group(1))

    # Job levels
    levels_found = []
    for level in ["Entry", "Graduate", "Mid", "Manager", "Director", "Executive", "Professional"]:
        if level.lower() in page_text.lower():
            levels_found.append(level)
    if levels_found:
        item["job_levels"] = levels_found

    # Languages — rough extraction
    lang_match = re.findall(
        r"\b(English|French|German|Spanish|Chinese|Arabic|Portuguese|Dutch|Italian|Russian)\b",
        page_text,
    )
    if lang_match:
        item["languages"] = list(dict.fromkeys(lang_match))  # deduplicate, preserve order


# ── Fallback: requests + BeautifulSoup ───────────────────────────────────────

def scrape_with_requests() -> List[Dict]:
    """
    Fallback scraper using requests + BeautifulSoup.
    Works only if SHL serves catalog content server-side on that URL.
    """
    import requests
    from bs4 import BeautifulSoup

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    items: List[Dict] = []
    start, page_size = 0, 12

    while True:
        url = f"{CATALOG_URL}?start={start}&type=1"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        rows = (
            soup.select("table.custom-table tbody tr") or
            soup.select("table tbody tr") or
            soup.select(".catalogue-item") or
            []
        )
        if not rows:
            break

        batch = []
        for row in rows:
            link = row.find("a")
            if not link:
                continue
            name = link.get_text(strip=True)
            href = link.get("href", "")
            if not name or not href:
                continue
            url_item = href if href.startswith("http") else f"{BASE_URL}{href}"

            # Test type from icon titles
            test_type, type_full = "K", ""
            for tag in row.find_all(title=True):
                t = tag["title"].strip()
                if t in TEST_TYPE_MAP:
                    test_type = TEST_TYPE_MAP[t]
                    type_full = t
                    break

            batch.append({
                "name": name, "url": url_item,
                "test_type": test_type, "test_type_full": type_full,
                "remote_testing": False, "adaptive": False,
                "duration": None, "description": "",
                "job_levels": [], "job_families": [],
                "languages": ["English"], "competencies": [],
            })

        items.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
        time.sleep(1)

    return items


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=== SHL Catalog Scraper ===")

    items: List[Dict] = []

    # Try Playwright first (handles JS-rendered pages)
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
        items = scrape_with_playwright()
    except ImportError:
        logger.warning("Playwright not installed — falling back to requests scraper.")
        items = scrape_with_requests()
    except Exception as e:
        logger.error("Playwright scrape failed: %s — trying requests fallback.", e)
        items = scrape_with_requests()

    if not items:
        logger.error(
            "No items scraped!\n"
            "Possible causes:\n"
            "  • The catalog page structure has changed.\n"
            "  • Bot detection blocked the request.\n"
            "Try opening the catalog in a browser and using the browser console to\n"
            "copy the rendered HTML, then parse it manually.\n"
        )
    else:
        out_path = "catalog.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(items, fh, indent=2, ensure_ascii=False)
        logger.info("Saved %d items to %s", len(items), out_path)

        # Quick summary
        from collections import Counter
        by_type = Counter(i["test_type"] for i in items)
        logger.info("Test-type breakdown: %s", dict(by_type))
