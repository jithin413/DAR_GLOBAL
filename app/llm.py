"""OpenRouter chat client (OpenAI-compatible) with streaming + free-model fallback."""
from __future__ import annotations

import json
import os
from typing import AsyncIterator

import logging

import httpx

log = logging.getLogger("llm")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = ("nvidia/nemotron-3.5-lightning:free,google/gemma-4-31b-it:free,"
                  "nvidia/nemotron-3-ultra-550b-a55b:free,baidu/ernie-5.5-flash-preview:free,"
                  "dots-studio/dots-3-note-preview:free")


def models() -> list[str]:
    return [m.strip() for m in os.getenv("OPENROUTER_MODELS", DEFAULT_MODELS).split(",") if m.strip()]


class LLMError(RuntimeError):
    pass


async def stream_chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 2500) -> AsyncIterator[dict]:
    """Yield {"type": "token", "text": ...} and finally {"type": "model", "model": ...}.

    Tries each configured model in turn (OpenRouter `models` routing does the same
    server-side; we also loop client-side so a 429 on a free model falls through).
    """
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise LLMError("OPENROUTER_API_KEY is not configured on the server.")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("APP_URL", "https://huggingface.co/spaces"),
        "X-Title": "DarGlobal & Wasalt Property Assistant",
    }
    chain = models()
    last_err = "no models configured"
    async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=15)) as client:
        for i, model in enumerate(chain):
            body = {
                "model": model,
                # reasoning models: keep thinking short and out of the answer stream
                "reasoning": {"effort": "low", "exclude": True},
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
            got_any = False
            try:
                async with client.stream("POST", OPENROUTER_URL, headers=headers, json=body) as r:
                    if r.status_code != 200:
                        last_err = f"{model}: HTTP {r.status_code} {(await r.aread())[:300]!r}"
                        log.warning("model skipped: %s", last_err)
                        continue
                    used = model
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("error"):
                            last_err = f"{model}: {chunk['error']}"
                            break
                        used = chunk.get("model", used)
                        for ch in chunk.get("choices", []):
                            txt = (ch.get("delta") or {}).get("content")
                            if txt:
                                got_any = True
                                yield {"type": "token", "text": txt}
                    if got_any:
                        yield {"type": "model", "model": used}
                        return
                    log.warning("model %s returned no content (%s)", model, last_err)
            except httpx.HTTPError as e:
                last_err = f"{model}: {e}"
                if got_any:
                    yield {"type": "model", "model": model}
                    return
    raise LLMError(f"All free models are currently unavailable ({last_err}). Please retry in a moment.")
