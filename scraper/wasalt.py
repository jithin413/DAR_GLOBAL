"""Scrape public listing + market-guide data from Wasalt.sa.

Respects https://wasalt.sa/robots.txt and https://wasalt.sa/llms.txt:
  * only public listing pages and public category (market guide) pages
  * no /search pages, no phone numbers / WhatsApp / agent personal data
  * moderate crawl rate (default 1.5 s between requests)
  * every record keeps its source URL so the bot can attribute "Source: Wasalt.sa"

Usage:
    python -m scraper.wasalt --listings 700 --delay 1.5
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict

from .common import RAW_DIR, PoliteSession, html_to_text, log, next_data, read_jsonl

SITEMAP_PRODUCTS = "https://cdn.wasalt.sa/sitemap/product_sitemap_en_sa.xml.gz"
SITEMAP_CATEGORIES = "https://cdn.wasalt.sa/sitemap/category_sitemap_en_sa.xml.gz"

LISTINGS_OUT = RAW_DIR / "wasalt_listings.jsonl"
GUIDES_OUT = RAW_DIR / "wasalt_guides.jsonl"

CITIES = ["riyadh", "jeddah", "dammam", "al-khobar", "makkah", "madinah",
          "khamis-mushayt", "taif", "abha", "buraidah", "dhahran", "tabuk"]
KINDS = ["properties-for-sale", "apartments-for-sale", "villas-for-sale", "lands-for-sale",
         "properties-for-rent", "apartments-for-rent", "villas-for-rent"]

_LOC = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")
_LISTING = re.compile(r"/en/property/(sale|rent)/([a-z]+)[^/]*-(\d+)$")


def sitemap_urls(sess: PoliteSession, url: str) -> list[str]:
    r = sess.get(url)
    if r is None:
        return []
    body = r.content
    if body[:2] == b"\x1f\x8b":
        import gzip
        body = gzip.decompress(body)
    return _LOC.findall(body.decode("utf-8", "ignore"))


def stratified_sample(urls: list[str], n: int, seed: int = 42) -> list[str]:
    """Sample listings proportionally per (purpose, type) with a floor so rare types appear."""
    buckets: dict[tuple, list[str]] = defaultdict(list)
    for u in urls:
        m = _LISTING.search(u)
        if m:
            buckets[(m.group(1), m.group(2))].append(u)
    total = sum(len(v) for v in buckets.values())
    rnd = random.Random(seed)
    out: list[str] = []
    for key, items in sorted(buckets.items()):
        k = max(8, round(n * len(items) / total))
        rnd.shuffle(items)
        out.extend(items[: min(k, len(items))])
    rnd.shuffle(out)
    return out[:n]


def _strings(obj, key="name") -> list[str]:
    """Collect human-readable names from a nested amenities/fittings structure."""
    found: list[str] = []
    if isinstance(obj, dict):
        v = obj.get(key) or obj.get("title") or obj.get("label")
        if isinstance(v, str) and v.strip():
            found.append(v.strip())
        for val in obj.values():
            if isinstance(val, (dict, list)):
                found.extend(_strings(val, key))
    elif isinstance(obj, list):
        for it in obj:
            found.extend(_strings(it, key))
    return found


def parse_listing(url: str, html: str) -> dict | None:
    nd = next_data(html)
    if not nd:
        return None
    pp = nd.get("props", {}).get("pageProps", {})
    p = pp.get("propertyDetailsV3") or {}
    pi = p.get("propertyInfo") or {}
    if not pi:
        return None

    attrs: dict[str, str] = {}
    for a in (p.get("attributes") or []) + (p.get("additionalAttributes") or []):
        if isinstance(a, dict) and a.get("key") and a.get("value") not in (None, "", "-"):
            unit = f" {a['unit']}" if a.get("unit") else ""
            attrs[a["key"]] = f"{a['value']}{unit}"

    rega = {}
    for group in p.get("regaVerifiedInfo") or []:
        if group.get("key") in ("rega_information", "property_specs"):
            for f in group.get("fields") or []:
                if f.get("value") not in (None, "", "-") and "Url" not in f.get("key", ""):
                    rega[f.get("label") or f.get("key")] = f["value"]

    info = pp.get("infoDetails") or {}
    amenities = sorted(set(_strings(info.get("amenities")) + _strings(info.get("fittings"))))

    price = pi.get("salePrice") if pi.get("propertyFor") == "sale" else (pi.get("expectedRent") or pi.get("salePrice"))
    return {
        "source": "wasalt",
        "kind": "listing",
        "id": f"wasalt-{p.get('id')}",
        "url": url,
        "title": pi.get("title") or pi.get("propertyName"),
        "purpose": pi.get("propertyFor"),
        "property_type": pi.get("propertySubType"),
        "main_type": pi.get("propertyMainType"),
        "country": pi.get("country"),
        "city": pi.get("city"),
        "territory": pi.get("territory"),
        "district": pi.get("district") or pi.get("zone"),
        "address": pi.get("address"),
        "price": _num(price),
        "currency": pi.get("currencyType") or "SAR",
        "rent_frequency": pi.get("rentFreq") or None,
        "bedrooms": _num(attrs.get("noOfBedrooms")),
        "bathrooms": _num(attrs.get("noOfBathrooms")),
        "area_sqm": _num((attrs.get("builtUpArea") or attrs.get("landArea") or "").split(" ")[0]),
        "furnishing": pi.get("furnishingType"),
        "facing": pi.get("facingType"),
        "attributes": attrs,
        "amenities": amenities,
        "rega": rega,
        "description": html_to_text(pi.get("description")),
        "published_at": p.get("publishedAt"),
        "updated_at": p.get("updatedAt"),
        "lat": _num((p.get("location") or {}).get("lat")),
        "lon": _num((p.get("location") or {}).get("lon")),
    }


def parse_guide(url: str, html: str) -> dict | None:
    nd = next_data(html)
    if not nd:
        return None
    pp = nd.get("props", {}).get("pageProps", {})
    meta = pp.get("metaData") or {}
    faqs = []
    for q in (pp.get("FAQSchema") or {}).get("mainEntity", []) or []:
        a = (q.get("acceptedAnswer") or {}).get("text")
        if q.get("name") and a:
            faqs.append({"q": q["name"], "a": html_to_text(a)})
    para = pp.get("para")
    para_text = html_to_text(para if isinstance(para, str) else json.dumps(para, ensure_ascii=False)) if para else ""
    # summarise the visible search results (counts / price range) without personal data
    sr = pp.get("searchResult") or {}
    props = sr.get("properties") or sr.get("data") or []
    prices = [_num((x.get("propertyInfo") or x).get("salePrice") or (x.get("propertyInfo") or x).get("expectedRent"))
              for x in props if isinstance(x, dict)]
    prices = [x for x in prices if x]
    title = meta.get("title") if isinstance(meta, dict) else None
    slug = url.rsplit("/", 1)[-1]
    return {
        "source": "wasalt",
        "kind": "market_guide",
        "id": f"wasalt-guide-{slug}",
        "url": url,
        "title": title or slug.replace("-", " ").title(),
        "meta_description": meta.get("description") if isinstance(meta, dict) else None,
        "total_listings": sr.get("total") or sr.get("count") or sr.get("totalCount"),
        "sample_price_min": min(prices) if prices else None,
        "sample_price_max": max(prices) if prices else None,
        "faqs": faqs,
        "text": para_text[:6000],
    }


def _num(v):
    if v in (None, "", "-"):
        return None
    try:
        f = float(str(v).replace(",", "").strip())
        return int(f) if f.is_integer() else f
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listings", type=int, default=700)
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--skip-guides", action="store_true")
    args = ap.parse_args()
    sess = PoliteSession(delay=args.delay)

    # 1) market guide pages (city x kind) that exist in the category sitemap
    if not args.skip_guides:
        cats = set(sitemap_urls(sess, SITEMAP_CATEGORIES))
        wanted = [f"https://wasalt.sa/en/{k}-in-{c}" for c in CITIES for k in KINDS]
        wanted = [u for u in wanted if u in cats] or wanted[:30]
        done = {r["url"] for r in read_jsonl(GUIDES_OUT)}
        with GUIDES_OUT.open("a", encoding="utf-8") as f:
            for i, u in enumerate(wanted, 1):
                if u in done:
                    continue
                r = sess.get(u)
                rec = parse_guide(u, r.text) if r else None
                if rec:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                log.info("guide %d/%d %s %s", i, len(wanted), "ok" if rec else "skip", u)

    # 2) listing pages (stratified sample from the public product sitemap)
    urls = sitemap_urls(sess, SITEMAP_PRODUCTS)
    log.info("product sitemap: %d urls", len(urls))
    sample = stratified_sample(urls, args.listings)
    done = {r["url"] for r in read_jsonl(LISTINGS_OUT)}
    ok = len(done)
    with LISTINGS_OUT.open("a", encoding="utf-8") as f:
        for i, u in enumerate(sample, 1):
            if u in done:
                continue
            r = sess.get(u)
            rec = parse_listing(u, r.text) if r else None
            if rec:
                ok += 1
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            if i % 25 == 0:
                log.info("listings %d/%d (saved %d)", i, len(sample), ok)
    log.info("done: %d listings saved", ok)


if __name__ == "__main__":
    main()
