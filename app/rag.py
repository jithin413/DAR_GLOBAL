"""Hybrid retrieval (BM25 + dense) with structured filters for property queries."""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .embedder import Embedder

KB = Path(__file__).resolve().parent.parent / "data" / "kb"
_TOKEN = re.compile(r"\w+", re.UNICODE)

TYPE_WORDS = {
    "apartment": ["apartment", "apartments", "flat", "flats", "شقة", "شقق"],
    "villa": ["villa", "villas", "townhouse", "فيلا", "فلل", "فله"],
    "land": ["land", "lands", "plot", "plots", "أرض", "ارض", "اراضي", "أراضي"],
    "floor": ["floor", "دور"],
    "building": ["building", "buildings", "عمارة", "مبنى"],
    "rest house": ["rest house", "rest-house", "istiraha", "استراحة"],
    "chalet": ["chalet", "شاليه"],
}
CITY_ALIASES = {
    "riyadh": ["riyadh", "الرياض", "riyad"],
    "jeddah": ["jeddah", "jedda", "جدة", "جده"],
    "dammam": ["dammam", "الدمام"],
    "al khobar": ["khobar", "al khobar", "al-khobar", "الخبر"],
    "makkah": ["makkah", "mecca", "مكة"],
    "madinah": ["madinah", "medina", "المدينة"],
    "khamis mushait": ["khamis", "خميس مشيط"],
    "taif": ["taif", "الطائف"],
    "abha": ["abha", "أبها", "ابها"],
    "buraidah": ["buraidah", "buraydah", "بريدة"],
    "dhahran": ["dhahran", "الظهران"],
    "tabuk": ["tabuk", "تبوك"],
}
DARGLOBAL_HINTS = ["dar global", "darglobal", "dar-global", "trump", "pagani", "aida", "missoni", "lamborghini",
                   "aston martin", "fendi", "elie saab", "mouawad", "w residences", "astera", "one of one",
                   "rayana", "amaya", "marbella", "oman", "dubai", "qatar", "maldives", "london", "branded",
                   "off-plan", "off plan", "tokeni", "dar al arkan", "داركلوبال", "دار جلوبال", "دار غلوبال"]
WASALT_HINTS = ["wasalt", "وصلت", "listing", "for sale", "for rent", "saudi", "ksa", "sar", "riyal"]


# Arabic -> English terms appended to the retrieval query (KB + embedder are English).
AR_GLOSSARY = {
    "دبي": "dubai", "عمان": "oman", "عُمان": "oman", "مسقط": "muscat oman", "قطر": "qatar", "الدوحة": "doha qatar",
    "جدة": "jeddah", "جده": "jeddah", "الرياض": "riyadh", "الدمام": "dammam", "الخبر": "khobar", "مكة": "makkah",
    "المدينة": "madinah", "لندن": "london", "إسبانيا": "spain marbella", "ماربيا": "marbella", "المالديف": "maldives",
    "مشاريع": "projects", "مشروع": "project", "شقة": "apartment", "شقق": "apartments", "فيلا": "villa",
    "فلل": "villas", "أرض": "land", "ارض": "land", "للبيع": "for sale", "للإيجار": "for rent", "إيجار": "rent",
    "سعر": "price", "أسعار": "prices", "غرف": "bedrooms", "غرفة": "bedroom", "فاخرة": "luxury", "فخمة": "luxury",
    "ترامب": "trump", "استثمار": "investment invest", "شركاء": "partners", "علامة": "brand", "العلامات": "brands",
    "دار جلوبال": "darglobal", "دار غلوبال": "darglobal", "داركلوبال": "darglobal", "وصلت": "wasalt",
    "فندق": "hotel", "برج": "tower", "مجمع": "community", "تسليم": "handover", "تأشيرة": "visa golden visa",
}


def expand_query(text: str) -> str:
    extra = [en for ar, en in AR_GLOSSARY.items() if ar in text]
    return text + (" " + " ".join(extra) if extra else "")


def tokenize(s: str) -> list[str]:
    return [t for t in _TOKEN.findall(s.lower()) if len(t) > 1 or t.isdigit()]


@dataclass
class Query:
    text: str
    purpose: str | None = None
    ptype: str | None = None
    city: str | None = None
    bedrooms: int | None = None
    min_price: float | None = None
    max_price: float | None = None
    prefer: str | None = None  # "darglobal" | "wasalt"
    notes: list[str] = field(default_factory=list)

    @property
    def is_listing_search(self) -> bool:
        return any([self.purpose, self.ptype, self.city, self.bedrooms, self.min_price, self.max_price])

    def describe(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v and k not in ("text", "notes")}


def _amount(num: str, unit: str | None) -> float:
    v = float(num.replace(",", ""))
    unit = (unit or "").lower()
    if unit in ("k", "thousand", "ألف", "الف"):
        v *= 1_000
    elif unit in ("m", "mn", "million", "مليون"):
        v *= 1_000_000
    return v


_NUM = r"([\d][\d,]*(?:\.\d+)?)\s*(k|m|mn|million|thousand|مليون|ألف|الف)?\b"


def parse_query(text: str) -> Query:
    t = text.lower()
    q = Query(text=text)
    if re.search(r"\b(rent|rental|renting|lease|leasing|monthly|yearly|annual rent)\b|إيجار|ايجار|للإيجار", t):
        q.purpose = "rent"
    elif re.search(r"\b(buy|buying|sale|purchase|purchasing|invest(?:ment)? property|own)\b|للبيع|شراء", t):
        q.purpose = "sale"
    for ptype, words in TYPE_WORDS.items():
        if any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", t) for w in words):
            q.ptype = ptype
            break
    for city, words in CITY_ALIASES.items():
        if any(w in t for w in words):
            q.city = city
            break
    m = re.search(r"(\d+)\s*(?:-|\s)?(?:bed(?:room)?s?|br|bhk|غرف|غرفة)", t)
    if m:
        q.bedrooms = int(m.group(1))
    m = re.search(rf"(?:under|below|less than|max(?:imum)?|up to|upto|within|budget(?: of| is)?|<|أقل من|حتى)\s*(?:sar|sr)?\s*{_NUM}", t)
    if m:
        q.max_price = _amount(m.group(1), m.group(2))
    m = re.search(rf"(?:over|above|more than|at least|min(?:imum)?|>|أكثر من)\s*(?:sar|sr)?\s*{_NUM}", t)
    if m:
        q.min_price = _amount(m.group(1), m.group(2))
    m = re.search(rf"between\s*(?:sar)?\s*{_NUM}\s*(?:and|-|to)\s*(?:sar)?\s*{_NUM}", t)
    if m:
        q.min_price, q.max_price = _amount(m.group(1), m.group(2) or m.group(4)), _amount(m.group(3), m.group(4))
    dg = sum(h in t for h in DARGLOBAL_HINTS)
    ws = sum(h in t for h in WASALT_HINTS) + (2 if q.is_listing_search else 0)
    if dg > ws:
        q.prefer = "darglobal"
    elif ws > dg:
        q.prefer = "wasalt"
    return q


class Retriever:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.docs = [json.loads(ln) for ln in (KB / "docs.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.emb = np.load(KB / "embeddings.npy")
        self.manifest = json.loads((KB / "manifest.json").read_text(encoding="utf-8"))
        self.bm25 = BM25Okapi([tokenize(f"{d['title']} {d['text']}") for d in self.docs])
        self.embedder = embedder or Embedder()
        self.is_listing = np.array([d["kind"] == "listing" for d in self.docs])
        self.source = np.array([d["source"] for d in self.docs])

    # ---------- filtering ----------
    def _listing_mask(self, q: Query, relax: int = 0) -> np.ndarray:
        mask = np.ones(len(self.docs), dtype=bool)
        for i, d in enumerate(self.docs):
            if d["kind"] != "listing":
                continue
            m = d["meta"]
            ok = True
            if q.city:
                dc = (m.get("city") or "").replace("-", " ")
                if not any(a in dc for a in CITY_ALIASES[q.city] if a.isascii()):
                    ok = False
            if q.purpose and m.get("purpose") != q.purpose:
                ok = False
            if q.ptype and q.ptype.split()[0] not in (m.get("type") or ""):
                ok = False
            if relax < 2:
                if q.bedrooms and m.get("bedrooms") is not None and m["bedrooms"] != q.bedrooms and relax == 0:
                    ok = False
                price = m.get("price")
                if q.max_price and price and price > q.max_price * (1.0 if relax == 0 else 1.15):
                    ok = False
                if q.min_price and price and price < q.min_price * (1.0 if relax == 0 else 0.85):
                    ok = False
            mask[i] = ok
        return mask

    def stats(self, mask: np.ndarray) -> dict | None:
        idx = [i for i in np.where(mask & self.is_listing)[0]]
        prices = [self.docs[i]["meta"]["price"] for i in idx if self.docs[i]["meta"].get("price")]
        if len(prices) < 3:
            return None
        ppsqm = [self.docs[i]["meta"]["price"] / self.docs[i]["meta"]["area_sqm"] for i in idx
                 if self.docs[i]["meta"].get("price") and self.docs[i]["meta"].get("area_sqm")
                 and self.docs[i]["meta"].get("purpose") == "sale"]
        return {
            "matching_listings_in_sample": len(idx),
            "price_min": min(prices), "price_median": statistics.median(prices), "price_max": max(prices),
            "median_price_per_sqm": round(statistics.median(ppsqm)) if len(ppsqm) >= 3 else None,
        }

    # ---------- search ----------
    def search(self, text: str, k: int = 8) -> tuple[list[dict], Query, dict | None]:
        text = expand_query(text)
        q = parse_query(text)
        mask = np.ones(len(self.docs), dtype=bool)
        if q.is_listing_search:
            for relax in (0, 1, 2):
                mask = self._listing_mask(q, relax)
                if (mask & self.is_listing).sum() >= 3:
                    if relax:
                        q.notes.append("filters relaxed to find close matches")
                    break

        bm = self.bm25.get_scores(tokenize(text))
        dense = self.emb @ self.embedder.embed_query(text)
        score = np.zeros(len(self.docs))
        for arr, w in ((bm, 1.0), (dense, 1.2)):
            ranked = np.argsort(-arr)
            ranks = np.empty(len(arr))
            ranks[ranked] = np.arange(len(arr))
            score += w / (60 + ranks)
        if q.prefer:
            score[self.source == q.prefer] *= 1.35
        if q.is_listing_search:
            score[self.is_listing & mask] *= 1.5
        score[~mask] = -1

        out, n_list = [], 0
        for i in np.argsort(-score):
            if score[i] <= 0 or len(out) >= k:
                break
            d = self.docs[i]
            if d["kind"] == "listing":
                if n_list >= 6:
                    continue
                n_list += 1
            out.append({**d, "score": float(score[i])})
        out = self._pin_overviews(text, q, out, k)
        st = self.stats(mask) if q.is_listing_search else None
        return out, q, st

    OVERVIEW_TRIGGERS = {
        "darglobal-overview-projects": ("project", "projects", "portfolio", "which", "list", "all", "branded", "developments"),
        "darglobal-overview-partners": ("brand", "brands", "branded", "partner", "partners", "collaborat"),
        "darglobal-overview-one-of-one": ("one of one", "ultra", "penthouse", "exclusive", "most luxurious"),
    }

    def _pin_overviews(self, text: str, q: Query, out: list[dict], k: int) -> list[dict]:
        """Catalogue-style questions ("which projects...") need the full list, not a few fragments."""
        if q.prefer != "darglobal":
            return out
        t, have = text.lower(), {d["id"] for d in out}
        pinned = [d for d in self.docs if d["id"] in self.OVERVIEW_TRIGGERS and d["id"] not in have
                  and any(w in t for w in self.OVERVIEW_TRIGGERS[d["id"]])]
        return ([{**d, "score": 1.0} for d in pinned] + out)[:k + len(pinned)]
