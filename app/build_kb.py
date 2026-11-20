"""Turn raw scraped records into retrieval documents + embeddings.

    python -m app.build_kb

Inputs : data/raw/*.jsonl
Outputs: data/kb/docs.jsonl, data/kb/embeddings.npy, data/kb/manifest.json
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
KB = ROOT / "data" / "kb"


def _read(name: str) -> list[dict]:
    p = RAW / name
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def fmt_price(v, cur="SAR") -> str:
    if v is None:
        return "price on request"
    return f"{v:,.0f} {cur}"


def chunk(text: str, size: int = 1100, overlap: int = 150) -> list[str]:
    paras, out, cur = text.split("\n"), [], ""
    for p in paras:
        if len(cur) + len(p) + 1 > size and cur:
            out.append(cur.strip())
            cur = cur[-overlap:] + "\n"
        cur += p + "\n"
    if cur.strip():
        out.append(cur.strip())
    return out


def listing_doc(r: dict) -> dict:
    purpose = "for rent" if r.get("purpose") == "rent" else "for sale"
    price = fmt_price(r.get("price"), r.get("currency") or "SAR")
    if r.get("purpose") == "rent" and r.get("rent_frequency"):
        price += f" ({r['rent_frequency']})"
    facts = [
        f"{r.get('property_type') or 'Property'} {purpose} in {r.get('address') or r.get('city')}",
        f"Price: {price}",
    ]
    if r.get("bedrooms"):
        facts.append(f"Bedrooms: {r['bedrooms']}")
    if r.get("bathrooms"):
        facts.append(f"Bathrooms: {r['bathrooms']}")
    if r.get("area_sqm"):
        facts.append(f"Area: {r['area_sqm']} sqm")
        if r.get("price") and r.get("purpose") == "sale":
            facts.append(f"Price per sqm: {r['price'] / r['area_sqm']:,.0f} SAR")
    for k in ("furnishing", "facing"):
        if r.get(k):
            facts.append(f"{k.title()}: {r[k]}")
    a = r.get("attributes") or {}
    for k, label in (("completionYear", "Building age"), ("propertyFacade", "Facade"),
                     ("noOfFloors", "Floors"), ("streetWidth", "Street width")):
        if a.get(k):
            facts.append(f"{label}: {a[k]}")
    if r.get("amenities"):
        facts.append("Amenities: " + ", ".join(r["amenities"][:20]))
    rega = r.get("rega") or {}
    if rega.get("REGA Ad License No"):
        facts.append(f"REGA ad licence: {rega['REGA Ad License No']} ({rega.get('Status', 'n/a')})")
    if r.get("published_at"):
        facts.append(f"Listed: {r['published_at'][:10]}")
    desc = (r.get("description") or "")[:900]
    text = "\n".join(facts) + ("\nDescription: " + desc if desc else "")
    return {
        "id": r["id"],
        "source": "wasalt",
        "kind": "listing",
        "title": f"{r.get('title') or r.get('property_type')} - {r.get('district') or ''}, {r.get('city') or ''}".strip(" ,-"),
        "url": r["url"],
        "text": text,
        "meta": {
            "city": (r.get("city") or "").lower() or None,
            "district": r.get("district"),
            "purpose": r.get("purpose"),
            "type": (r.get("property_type") or "").lower() or None,
            "price": r.get("price"),
            "bedrooms": r.get("bedrooms"),
            "area_sqm": r.get("area_sqm"),
        },
    }


def guide_docs(r: dict) -> list[dict]:
    docs = []
    head = r["title"]
    stats = []
    if r.get("total_listings"):
        stats.append(f"Wasalt shows about {r['total_listings']:,} active listings on this page.")
    if r.get("sample_price_min") and r.get("sample_price_max"):
        stats.append(f"Prices on the first results page ranged {r['sample_price_min']:,.0f}-{r['sample_price_max']:,.0f} SAR.")
    faq = "\n".join(f"Q: {f['q']}\nA: {f['a']}" for f in r.get("faqs") or [])
    first = "\n".join(filter(None, [head, " ".join(stats), faq]))
    docs.append({"id": r["id"] + "-faq", "source": "wasalt", "kind": "market_guide",
                 "title": head, "url": r["url"], "text": first, "meta": {}})
    for i, c in enumerate(chunk(r.get("text") or "")):
        docs.append({"id": f"{r['id']}-{i}", "source": "wasalt", "kind": "market_guide",
                     "title": head, "url": r["url"], "text": f"{head}\n{c}", "meta": {}})
    return docs


def darglobal_docs(catalog: list[dict], pages: list[dict]) -> list[dict]:
    docs = []
    have_page = {p["url"] for p in pages}
    for p in pages:
        for i, c in enumerate(chunk(p["text"])):
            docs.append({"id": f"{p['id']}-{i}", "source": "darglobal", "kind": p["section"],
                         "title": p["title"], "url": p["url"],
                         "text": f"DarGlobal - {p['label']}: {p['title']}\n"
                                 f"{p.get('meta_description') or ''}\n{c}", "meta": {}})
    # Catalogue entries (from the public sitemap) for pages we have no full text for.
    groups: dict[str, list[dict]] = {}
    for c in catalog:
        if c["url"] in have_page:
            continue
        if c["section"] in ("blog", "press", "insights"):
            groups.setdefault(c["section"], []).append(c)
            continue
        docs.append({"id": c["id"], "source": "darglobal", "kind": c["section"], "title": c["title"],
                     "url": c["url"],
                     "text": f"DarGlobal {c['label']}: {c['title']}. Official page: {c['url']}", "meta": {}})
    for section, items in groups.items():
        for i in range(0, len(items), 12):
            part = items[i:i + 12]
            label = part[0]["label"]
            lines = "\n".join(f"- {x['title']} ({x['url']})" for x in part)
            docs.append({"id": f"darglobal-{section}-list-{i // 12}", "source": "darglobal", "kind": section,
                         "title": f"DarGlobal {label}s", "url": f"https://darglobal.co.uk/{section}",
                         "text": f"DarGlobal {label} headlines:\n{lines}", "meta": {}})
    # One overview doc listing every project / partner name (helps "what projects..." questions).
    for section in ("projects", "partners", "one-of-one"):
        names = sorted({f"{c['title']} <{c['url']}>" for c in catalog if c["section"] == section})
        if names:
            docs.append({"id": f"darglobal-overview-{section}", "source": "darglobal", "kind": "overview",
                         "title": f"DarGlobal {section.replace('-', ' ')} overview",
                         "url": f"https://darglobal.co.uk/{'' if section == 'projects' else section}",
                         "text": f"Complete list of DarGlobal {SECTION_NAMES[section]} found on darglobal.co.uk "
                                 f"({len(names)}): " + "; ".join(names), "meta": {}})
    return docs


SECTION_NAMES = {"projects": "projects and campaigns", "partners": "brand partners (luxury brand collaborations)",
                 "one-of-one": "'One of One' ultra-luxury residences"}


def main() -> None:
    from app.embedder import Embedder

    t0 = time.time()
    docs: list[dict] = []
    listings = _read("wasalt_listings.jsonl")
    seen = set()
    for r in listings:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        docs.append(listing_doc(r))
    for g in _read("wasalt_guides.jsonl"):
        docs.extend(guide_docs(g))
    docs.extend(darglobal_docs(_read("darglobal_catalog.jsonl"), _read("darglobal_pages.jsonl")))

    KB.mkdir(parents=True, exist_ok=True)
    emb = Embedder().embed_docs([f"{d['title']}\n{d['text']}" for d in docs])
    np.save(KB / "embeddings.npy", emb)
    with (KB / "docs.jsonl").open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "documents": len(docs),
        "by_source_kind": Counter(f"{d['source']}:{d['kind']}" for d in docs),
        "wasalt_listings": len(seen),
        "cities": Counter(d["meta"].get("city") for d in docs if d["kind"] == "listing").most_common(15),
    }
    (KB / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"built in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
