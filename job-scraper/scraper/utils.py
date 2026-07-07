"""Shared helpers: logging, human-like sleeps, ID hashing, JSON path lookup."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import sys
from dataclasses import dataclass

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
    source: str   # e.g. "fortune500:Netflix"
    title: str
    link: str

    @property
    def job_id(self) -> str:
        """Stable dedup key: SHA-256 of the normalized link + title."""
        raw = f"{self.link.strip().lower()}|{self.title.strip().lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
