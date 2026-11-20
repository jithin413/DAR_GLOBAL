"""Shared helpers for polite, robots-aware scraping."""
from __future__ import annotations

import json
import logging
import re
import time
from html import unescape
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = (
    "DarWasaltChatbot/1.0 (+technical-assignment; polite crawler; "
    "respects robots.txt and llms.txt)"
)

log = logging.getLogger("scraper")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class PoliteSession:
    """requests.Session with robots.txt checks, a fixed delay and retries."""

    def __init__(self, delay: float = 1.5, timeout: int = 30):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
        self.delay = delay
        self.timeout = timeout
        self._last = 0.0
        self._robots: dict[str, robotparser.RobotFileParser] = {}

    def allowed(self, url: str) -> bool:
        host = urlparse(url)
        base = f"{host.scheme}://{host.netloc}"
        if base not in self._robots:
            rp = robotparser.RobotFileParser()
            try:
                txt = self.s.get(base + "/robots.txt", timeout=self.timeout).text
                rp.parse(txt.splitlines())
            except requests.RequestException:
                rp.parse([])
            self._robots[base] = rp
        return self._robots[base].can_fetch(USER_AGENT, url)

    def get(self, url: str, retries: int = 3) -> requests.Response | None:
        if not self.allowed(url):
            log.warning("robots.txt disallows %s - skipped", url)
            return None
        for attempt in range(retries):
            wait = self.delay - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.s.get(url, timeout=self.timeout)
                if r.status_code == 200:
                    return r
                if r.status_code in (404, 410):
                    return None
                log.warning("HTTP %s for %s (attempt %d)", r.status_code, url, attempt + 1)
            except requests.RequestException as e:
                log.warning("error %s for %s (attempt %d)", e, url, attempt + 1)
            time.sleep(2 ** (attempt + 1))
        return None


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")


def html_to_text(html: str | None) -> str:
    if not html:
        return ""
    txt = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</h\d>|</div>", "\n", html)
    txt = unescape(_TAG.sub(" ", txt))
    lines = [_WS.sub(" ", ln).strip() for ln in txt.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def next_data(html: str) -> dict | None:
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    return json.loads(m.group(1)) if m else None


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("wrote %d rows -> %s", len(rows), path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]
