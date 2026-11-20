"""Collect public data about DarGlobal (darglobal.co.uk).

The site's HTML pages sit behind an Incapsula bot-protection layer, so this
module never tries to defeat it. It supports three honest modes:

  sitemap  (default) - read the public sitemap.xml (served without protection)
                       and build a catalogue of projects / partners / insights /
                       press headlines from the URL slugs.
  browser            - plain headless Chromium (no stealth flags, honest UA),
                       robots.txt respected. Works only if the site lets it through.
  html               - parse pages you saved yourself from a normal browser
                       ("Save page as... -> Webpage, HTML only") into data/darglobal_html/.

Usage:
    python -m scraper.darglobal                     # sitemap catalogue
    python -m scraper.darglobal --mode html         # + parse saved pages
    python -m scraper.darglobal --mode browser      # + try a plain browser crawl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

from .common import ROOT, RAW_DIR, USER_AGENT, PoliteSession, log, write_jsonl

BASE = "https://darglobal.co.uk"
SITEMAP = f"{BASE}/sitemap.xml"
CATALOG_OUT = RAW_DIR / "darglobal_catalog.jsonl"
PAGES_OUT = RAW_DIR / "darglobal_pages.jsonl"
HTML_DIR = ROOT / "data" / "darglobal_html"

LANG_PREFIX = re.compile(r"^/(ar|ru|zh|fr|de|es|it|tr|cn)(/|$)")
SKIP = {"pay-online", "thank-you", "success", "bookYourUnit", "landingpage", "new-home-page",
        "internship-application", "privacy-policy", "terms-conditions", "terms-of-uses",
        "landing-page", "status"}

# Known context for slugs, used only to label categories (not invented facts).
SECTION_LABEL = {
    "partners": "Brand partner",
    "insights": "Insight article",
    "press": "Press release",
    "one-of-one": "One of One (ultra-luxury) collection",
    "rayana": "Rayana (Riyadh) community project",
    "amaya": "Amaya project",
    "trump-plaza": "Trump Plaza (Jeddah) project",
    "blog": "Blog article",
    "projects": "Project",
}


def slug_to_title(slug: str) -> str:
    words = slug.replace("_", "-").split("-")
    small = {"and", "of", "the", "in", "at", "by", "for", "to", "a", "with", "on"}
    out = []
    for i, w in enumerate(words):
        if not w:
            continue
        if w.isupper() or any(c.isdigit() for c in w):
            out.append(w.upper() if len(w) <= 3 and w.isalpha() else w)
        elif i and w.lower() in small:
            out.append(w.lower())
        else:
            out.append(w.capitalize())
    return " ".join(out)


def classify(path: str) -> tuple[str, str]:
    parts = [p for p in path.strip("/").split("/") if p]
    if not parts:
        return "page", "Home page"
    if len(parts) >= 2 and parts[0] in SECTION_LABEL:
        return parts[0], SECTION_LABEL[parts[0]]
    if parts[0] in {"about", "why-invest", "faq", "investor", "hospitality", "commercial",
                    "development-management", "tokenization", "partners", "one-of-one",
                    "insights", "press", "blog", "careers", "live-all-in", "dg-circle",
                    "exclusive-membership", "insignia", "luxury-golf-communities"}:
        return "page", "Corporate page"
    return "projects", "Project / campaign page"


def build_catalog(sess: PoliteSession) -> list[dict]:
    r = sess.get(SITEMAP)
    if r is None:
        log.error("could not read DarGlobal sitemap")
        return []
    urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", r.text)
    rows, seen = [], set()
    for u in urls:
        path = re.sub(r"^https?://[^/]+", "", u) or "/"
        if LANG_PREFIX.match(path):
            continue
        slug = path.rstrip("/").rsplit("/", 1)[-1] or "home"
        if slug in SKIP or path in seen:
            continue
        seen.add(path)
        section, label = classify(path)
        rows.append({
            "source": "darglobal",
            "kind": "catalog",
            "id": f"darglobal-cat-{path.strip('/').replace('/', '--') or 'home'}",
            "url": BASE + path,
            "section": section,
            "label": label,
            "title": slug_to_title(slug),
        })
    write_jsonl(CATALOG_OUT, rows)
    return rows


# ---------- page text extraction (shared by browser + html modes) ----------

def extract_page(url: str, html: str) -> dict | None:
    soup = BeautifulSoup(html, "lxml")
    if "Incapsula" in html[:3000] and len(html) < 5000:
        return None  # blocked page, do not store
    for t in soup(["script", "style", "noscript", "svg", "iframe", "form"]):
        if t.get("type") == "application/ld+json":
            continue
        t.decompose()
    title = (soup.title.string or "").strip() if soup.title else ""
    meta = soup.find("meta", attrs={"name": "description"})
    desc = meta.get("content", "").strip() if meta else ""
    for t in soup.select("header, footer, nav"):
        t.decompose()
    blocks, seen = [], set()
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "span", "div"]):
        if el.name in ("span", "div") and el.find(["p", "div", "li", "h1", "h2", "h3"]):
            continue
        txt = " ".join(el.get_text(" ", strip=True).split())
        if len(txt) < 3 or txt in seen:
            continue
        seen.add(txt)
        blocks.append(("## " + txt) if el.name in ("h1", "h2", "h3") else txt)
    text = "\n".join(blocks)
    if len(text) < 200:
        return None
    path = re.sub(r"^https?://[^/]+", "", url) or "/"
    section, label = classify(path)
    return {
        "source": "darglobal",
        "kind": "page",
        "id": f"darglobal-page-{path.strip('/').replace('/', '--') or 'home'}",
        "url": url,
        "section": section,
        "label": label,
        "title": title or slug_to_title(path.rsplit("/", 1)[-1]),
        "meta_description": desc,
        "text": text[:20000],
    }


def from_saved_html() -> list[dict]:
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(HTML_DIR.glob("*.htm*")):
        html = f.read_text(encoding="utf-8", errors="ignore")
        canon = re.search(r'<link[^>]+rel="canonical"[^>]+href="([^"]+)"', html)
        og = re.search(r'<meta[^>]+property="og:url"[^>]+content="([^"]+)"', html)
        url = (canon or og).group(1) if (canon or og) else f"{BASE}/{f.stem}"
        rec = extract_page(url, html)
        log.info("%s -> %s", f.name, "ok" if rec else "no usable content")
        if rec:
            rows.append(rec)
    return rows


def from_browser(catalog: list[dict], limit: int, delay: float) -> list[dict]:
    from playwright.sync_api import sync_playwright

    sess = PoliteSession()
    targets = [c["url"] for c in catalog if c["section"] != "press"][:limit]
    targets += [c["url"] for c in catalog if c["section"] == "press"][: max(0, limit - len(targets))]
    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)
        for i, u in enumerate(targets, 1):
            if not sess.allowed(u):
                continue
            try:
                page.goto(u, wait_until="networkidle", timeout=60000)
                rec = extract_page(u, page.content())
            except Exception as e:  # noqa: BLE001
                log.warning("browser error %s: %s", u, e)
                rec = None
            log.info("browser %d/%d %s %s", i, len(targets), "ok" if rec else "blocked/empty", u)
            if rec:
                rows.append(rec)
            elif i >= 5 and not rows:
                log.error("first pages all blocked by bot protection - stopping (not bypassing it)")
                break
            page.wait_for_timeout(int(delay * 1000))
        browser.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sitemap", "html", "browser"], default="sitemap")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--delay", type=float, default=2.0)
    args = ap.parse_args()

    catalog = build_catalog(PoliteSession())
    if args.mode == "sitemap":
        return
    pages = from_saved_html() if args.mode == "html" else from_browser(catalog, args.limit, args.delay)
    existing = {}
    if PAGES_OUT.exists():
        for ln in PAGES_OUT.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                existing[r["id"]] = r
    for r in pages:
        existing[r["id"]] = r
    write_jsonl(PAGES_OUT, list(existing.values()))


if __name__ == "__main__":
    main()
