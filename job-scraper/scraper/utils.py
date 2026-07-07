"""Shared helpers: logging, human-like sleeps, ID hashing, JSON path lookup."""

from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .config import SLEEP_RANGE


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure a console logger for the whole `scraper` package."""
    logger = logging.getLogger("scraper")
    if logger.handlers:  # already configured (idempotent for tests/imports)
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger


@dataclass(frozen=True)
class Listing:
    """One scraped item, normalized across all sources."""
    source: str   # e.g. "ashby:Notion"
    title: str
    link: str
    # When the source told us the posting/publish time (UTC). None means the
    # source has no reliable timestamp (e.g. Google CSE results) -- such
    # listings are still stored, just excluded from the "last 48h" SMS digest
    # since we can't verify how new they actually are.
    posted_at: datetime | None = None
    # Plain-text job description, when the source's API/feed provides one
    # (Ashby, Greenhouse with content=true, RSS <description>). Empty string
    # when unavailable (Workday's list endpoint, GitHub, Instagram, Google
    # CSE) -- filters.py treats "no description" as "can't verify, don't
    # exclude on experience-level grounds" rather than assuming the worst.
    description: str = ""

    @property
    def job_id(self) -> str:
        """Stable dedup key: SHA-256 of the normalized link + title."""
        raw = f"{self.link.strip().lower()}|{self.title.strip().lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def is_recent(self, hours: int) -> bool:
        """True if `posted_at` is known and falls within the last `hours`."""
        if self.posted_at is None:
            return False
        return datetime.now(timezone.utc) - self.posted_at <= timedelta(hours=hours)


def parse_timestamp(value) -> datetime | None:
    """Best-effort parse of a posting timestamp into a UTC-aware datetime.

    Accepts ISO-8601 strings (with or without a trailing "Z"), RFC-2822
    strings (RSS `pubDate`), or epoch seconds/millis. Returns None instead of
    raising on anything unrecognized -- callers treat that as "unknown".
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        ts = value / 1000 if value > 10_000_000_000 else value  # ms -> s
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

    text = str(value).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


async def human_sleep(scale: float = 1.0) -> None:
    """Sleep a random 1-5s (scaled) to mimic a human between actions."""
    lo, hi = SLEEP_RANGE
    await asyncio.sleep(random.uniform(lo, hi) * scale)


def dig(obj: dict | list, dotted_path: str):
    """Resolve a dotted path like ``"data.jobs"`` inside nested JSON.

    Numeric segments index into lists (``"results.0.title"``).
    Returns None instead of raising when any segment is missing, so parsers
    can treat an unexpected payload shape as "no results" rather than crash.
    """
    current = obj
    if not dotted_path:
        return current
    for part in dotted_path.split("."):
        try:
            current = current[int(part)] if part.isdigit() else current[part]
        except (KeyError, IndexError, TypeError):
            return None
    return current


def truncate(text: str, limit: int) -> str:
    """Trim text to `limit` chars, appending an ellipsis when cut."""
    text = " ".join(text.split())  # collapse whitespace/newlines
    return text if len(text) <= limit else text[: limit - 1] + "…"


def strip_html(raw: str) -> str:
    """Reduce an HTML job-description blob to plain text for keyword/regex
    matching in filters.py. Not meant to preserve formatting -- just enough
    to search for phrases like "senior" or "2+ years of experience".
    """
    if not raw:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    return " ".join(text.split())
