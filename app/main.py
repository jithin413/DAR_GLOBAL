"""FastAPI app: RAG chatbot over scraped DarGlobal + Wasalt data."""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .llm import LLMError, models, stream_chat
from .rag import Retriever

log = logging.getLogger("app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
STATIC = Path(__file__).parent / "static"
STATE: dict = {}

RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_5MIN", "20"))
_hits: dict[str, deque] = defaultdict(deque)


@asynccontextmanager
async def lifespan(_: FastAPI):
    t = time.time()
    STATE["retriever"] = Retriever()
    log.info("knowledge base loaded: %d docs in %.1fs", len(STATE["retriever"].docs), time.time() - t)
    yield


app = FastAPI(title="DarGlobal & Wasalt Property Assistant", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class Msg(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class ChatIn(BaseModel):
    messages: list[Msg] = Field(min_length=1, max_length=30)


SYSTEM = """You are the "DarGlobal & Wasalt Property Assistant", a professional real-estate assistant.
Your knowledge comes ONLY from the CONTEXT block, which contains data scraped from two public websites:
- DarGlobal (darglobal.co.uk): an international real-estate developer of branded luxury projects.
- Wasalt (wasalt.sa): a Saudi Arabian property portal with listings for sale and rent.

Rules:
1. Ground every factual statement in the CONTEXT and cite it inline with its number, e.g. [2]. Never invent
   prices, sizes, dates, amenities or availability. If the CONTEXT does not contain the answer, say so plainly
   and point the user to the relevant official page.
   Use ONLY URLs that appear verbatim in the CONTEXT - never build or guess a link. DarGlobal's site is
   darglobal.co.uk (not .com). Do not infer details (brand partner, unit type) from a project name alone.
2. Some DarGlobal entries are catalogue entries (name + official link only). For those, give the name and link
   but do not make up details.
3. Wasalt prices and availability change often: when you quote Wasalt listings, end with
   "Source: Wasalt.sa - please confirm current price and availability on the live listing."
4. "MARKET SAMPLE STATS" are computed from the scraped sample only; describe them as indicative, not the full market.
5. Never share personal phone numbers or contact details. Do not give legal, tax or investment advice; you may
   summarise what the sources say.
6. Reply in the user's language (English or Arabic). Be concise and well structured: short intro, then bullet
   points or a compact markdown table when comparing several properties (title, location, price, size, link).
7. If the request is vague (e.g. "find me a home"), give a few relevant options AND ask one short follow-up
   question (budget, city, rent vs buy, bedrooms)."""


def build_context(hits: list[dict], q, stats: dict | None) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        label = {"listing": "Wasalt listing", "market_guide": "Wasalt market guide"}.get(
            h["kind"], f"DarGlobal {h['kind'].replace('-', ' ')}")
        parts.append(f"[{i}] ({label}) {h['title']}\nURL: {h['url']}\n{h['text']}")
    if stats:
        parts.append("MARKET SAMPLE STATS (scraped Wasalt sample matching the user's filters "
                     f"{json.dumps(q.describe(), ensure_ascii=False)}): "
                     + json.dumps(stats, ensure_ascii=False))
    if q.notes:
        parts.append("NOTE: " + "; ".join(q.notes))
    return "\n\n".join(parts) if parts else "(no relevant documents found)"


def retrieval_query(msgs: list[Msg]) -> str:
    users = [m.content for m in msgs if m.role == "user"]
    cur = users[-1]
    if len(users) > 1 and len(cur.split()) < 8:  # short follow-up: carry previous intent
        return users[-2] + " " + cur
    return cur


def client_ip(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "unknown")


def check_rate(ip: str) -> None:
    now, dq = time.time(), _hits[ip]
    while dq and now - dq[0] > 300:
        dq.popleft()
    if len(dq) >= RATE_LIMIT:
        raise HTTPException(429, "Too many requests - please wait a few minutes.")
    dq.append(now)


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat(body: ChatIn, request: Request):
    check_rate(client_ip(request))
    if body.messages[-1].role != "user":
        raise HTTPException(400, "last message must be from the user")
    retriever: Retriever = STATE["retriever"]
    hits, q, stats = retriever.search(retrieval_query(body.messages))
    context = build_context(hits, q, stats)
    history = [{"role": m.role, "content": m.content[:2000]} for m in body.messages[-7:-1]]
    messages = [{"role": "system", "content": SYSTEM},
                *history,
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {body.messages[-1].content}"}]
    sources = [{"n": i, "title": h["title"], "url": h["url"], "source": h["source"], "kind": h["kind"]}
               for i, h in enumerate(hits, 1)]

    async def gen():
        yield sse("sources", {"sources": sources, "filters": q.describe(), "stats": stats})
        t = time.time()
        try:
            async for ev in stream_chat(messages):
                if ev["type"] == "token":
                    yield sse("token", ev["text"])
                else:
                    yield sse("done", {"model": ev["model"], "seconds": round(time.time() - t, 1)})
        except LLMError as e:
            log.warning("llm failure: %s", e)
            yield sse("error", str(e))

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/search")
async def search(q: str, k: int = 8):
    hits, query, stats = STATE["retriever"].search(q, k=min(k, 20))
    return {"filters": query.describe(), "notes": query.notes, "stats": stats,
            "hits": [{k_: h[k_] for k_ in ("title", "url", "source", "kind", "score", "text")} for h in hits]}


@app.get("/api/health")
async def health():
    r = STATE.get("retriever")
    return {"status": "ok" if r else "loading", "documents": len(r.docs) if r else 0,
            "llm_configured": bool(os.getenv("OPENROUTER_API_KEY")), "models": models()}


@app.get("/api/stats")
async def stats():
    return JSONResponse(STATE["retriever"].manifest)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")
