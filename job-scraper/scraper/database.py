"""SQLite deduplication layer.

Schema (created on first run):

    jobs(
        job_id     TEXT PRIMARY KEY,   -- sha256(link|title), see utils.Listing
        source     TEXT NOT NULL,
        title      TEXT NOT NULL,
        link       TEXT NOT NULL,
        scraped_at TEXT NOT NULL       -- ISO-8601 UTC
    )

The table doubles as a run history: nothing is ever deleted, so a listing
that reappears weeks later is still recognized and suppressed.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterable, Iterator

from .utils import Listing

log = logging.getLogger("scraper.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id     TEXT PRIMARY KEY,
    source     TEXT NOT NULL,
    title      TEXT NOT NULL,
    link       TEXT NOT NULL,
    scraped_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs (source);
"""


class JobStore:
    """Thin wrapper around a local SQLite file used to filter duplicates."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def exists(self, job_id: str) -> bool:
        """Return True if this listing has been seen on any previous run."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM jobs WHERE job_id = ? LIMIT 1", (job_id,)
            ).fetchone()
        return row is not None

    def add(self, listing: Listing) -> bool:
        """Insert a listing; return True if it was new, False if duplicate.

        INSERT OR IGNORE makes this safe even if two sources yield the same
        link in a single run.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO jobs (job_id, source, title, link, scraped_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (listing.job_id, listing.source, listing.title, listing.link, now),
            )
        return cur.rowcount == 1

    def filter_new(self, listings: Iterable[Listing]) -> list[Listing]:
        """Persist and return only brand-new listings, preserving order."""
        fresh: list[Listing] = []
        for listing in listings:
            if self.add(listing):
                fresh.append(listing)
        log.info("dedup: %d new listing(s) after filtering", len(fresh))
        return fresh
