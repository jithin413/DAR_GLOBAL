---
title: DarGlobal & Wasalt Property Assistant
emoji: 🏙️
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: RAG chatbot over DarGlobal projects and Wasalt listings
---

# DarGlobal & Wasalt Property Assistant

An AI chatbot that answers questions about **DarGlobal** (darglobal.co.uk) projects and **Wasalt** (wasalt.sa) Saudi
property listings. It uses data scraped from both public websites, retrieval-augmented generation (RAG), and free
LLMs served through **OpenRouter**. The whole app ships as a single **Docker** container.

## Architecture

```
 scraper/ (offline)                 app/build_kb.py                     app/ (Docker container)
 ┌──────────────────────┐   raw    ┌──────────────────────┐   kb     ┌───────────────────────────────┐
 │ wasalt.py            │ ───────► │ normalise + chunk     │ ───────► │ FastAPI  /api/chat (SSE)      │
 │  sitemap → listing   │  jsonl   │ embed (bge-small ONNX)│ docs +   │  ├─ query parser (city, rent/ │
 │  pages (__NEXT_DATA__)│         │ manifest              │ vectors  │  │  sale, type, beds, budget)  │
 │ darglobal.py         │          └──────────────────────┘          │  ├─ hybrid retrieval           │
 │  sitemap catalogue   │                                            │  │  BM25 + dense, RRF fusion   │
 └──────────────────────┘                                            │  ├─ sample price statistics    │
                                                                     │  └─ OpenRouter free models     │
                                                                     │     (fallback chain, streaming)│
                                                                     │ static/index.html chat UI      │
                                                                     └───────────────────────────────┘
```

| Layer | Choice | Why |
|---|---|---|
| Scraping | `requests` + embedded Next.js JSON (Wasalt), public sitemap (DarGlobal) | Structured and reliable; respects robots.txt |
| Embeddings | `bge-small-en-v1.5` int8 ONNX via `onnxruntime` | ~35 MB, CPU-only, no PyTorch → small image |
| Retrieval | BM25 + cosine similarity, reciprocal-rank fusion, metadata filters | Handles exact names ("Pagani") and paraphrases |
| LLM | OpenRouter free models with client-side fallback | Free tier; survives per-model rate limits |
| API / UI | FastAPI + server-sent events, single-page HTML | Token streaming, clickable citations, Arabic (RTL) |

## Features

- Cited answers. Each answer links back to the source listing or page, and the model may only use URLs that appear in the retrieved data.
- Understands property searches, e.g. *"3-bedroom apartments for sale in Riyadh under 1M"*, and applies them as hard filters, relaxing them gradually when nothing matches.
- Indicative market statistics (min / median / max price, price per m²) computed from the matching scraped sample.
- Answers in English or Arabic (Arabic queries are expanded with English search terms).
- Free-model fallback chain. Reasoning models run with `reasoning.exclude` so hidden reasoning output doesn't hold up the answer stream.
- Per-IP rate limiting, a non-root container, a healthcheck, and the API key supplied through the environment only.

## Run locally

```bash
cp .env.example .env            # add your OPENROUTER_API_KEY
docker compose up --build       # → http://localhost:7860
```

Without Docker:

```bash
python -m venv .venv && .venv/Scripts/activate        # (Linux/macOS: source .venv/bin/activate)
pip install -r requirements-scraper.txt
python -m scraper.darglobal            # DarGlobal sitemap catalogue
python -m scraper.wasalt --listings 700 --delay 2.5
python -m app.build_kb                 # → data/kb/
uvicorn app.main:app --port 7860
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat` | `{"messages":[{"role":"user","content":"..."}]}` → SSE events: `sources`, `token`, `done`, `error` |
| `GET` | `/api/search?q=...` | Retrieval only (debugging / evaluation) |
| `GET` | `/api/health` | Liveness + configuration |
| `GET` | `/api/stats` | Knowledge-base manifest |

## Deploy (Render, free)

1. Push this repository to GitHub.
2. Render dashboard → **New → Blueprint** → select the repo (`render.yaml` configures a free Docker web service).
3. When prompted, paste `OPENROUTER_API_KEY`. The app goes live at `https://darglobal-wasalt-assistant.onrender.com`.

The free plan sleeps after 15 minutes idle, so the first request afterwards takes about a minute.
(A Hugging Face Space also works via the YAML header above, but Docker Spaces now require a PRO plan.)

## Data collection & compliance

- **Wasalt**: their `robots.txt` allows listing pages and `llms.txt` allows *"summarizing property listings,
  recommending properties"*. The crawler fetches only public listing and market-guide pages, never `/search`, with
  at least 1.5 s between requests. It does **not** store agent names, phone numbers or WhatsApp links. Answers carry
  "Source: Wasalt.sa" and send users to the live listing for current prices. `llms.txt` disallows commercial use;
  this project is a non-commercial technical assignment.
- **DarGlobal**: the HTML pages are behind bot protection, which this project deliberately does not bypass. Data
  comes from the public `sitemap.xml` (project, partner, insight and press names with their official links), so
  the bot gives names and links but does not invent specifications.
