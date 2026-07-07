"""SQLite deduplication layer.

Schema (created on first run):

    jobs(
        job_id     TEXT PRIMARY KEY,   -- sha256(link|title), see utils.Listing
        source     TEXT NOT NULL,
        title      TEXT NOT NULL,
        link       TEXT NOT NULL,
        posted_at  TEXT,               -- ISO-8601 UTC, NULL if source has no timestamp
        scraped_at TEXT NOT NULL       -- ISO-8601 UTC, when *we* found it
    )

The table doubles as a run history and as the "everything we've ever seen"
archive: nothing is ever deleted, so a listing that reappears weeks later is
still recognized and suppressed, and every non-urgent listing (anything
outside the 48h SMS window) still lands here and stays searchable via
`search()` below or `python -m scraper.query`.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterable, Iterator

from .utils import Listing

log = logging.getLogger("scraper.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id     TEXT PRIMARY KEY,
    source     TEXT NOT NULL,
    title      TEXT NOT NULL,
    link       TEXT NOT NULL,
    posted_at  TEXT,
    scraped_at TEXT NOT NULL
);
"""
# Indexes are created *after* migration (below), since a pre-existing jobs.db
# from before `posted_at` was added won't have that column until migrated.
_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs (source);
CREATE INDEX IF NOT EXISTS idx_jobs_posted_at ON jobs (posted_at);
"""


class JobStore:
    """Thin wrapper around a local SQLite file used to filter duplicates."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        with self._connect() as conn:
            conn.executescript(_CREATE_TABLE)
            self._migrate(conn)
            conn.executescript(_CREATE_INDEXES)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add columns introduced after a user's jobs.db already existed."""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        if "posted_at" not in cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN posted_at TEXT")

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
        link in a single run. Every listing is stored regardless of age or
        whether it makes the SMS digest -- this table is the full archive.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        posted = listing.posted_at.isoformat(timespec="seconds") if listing.posted_at else None
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO jobs (job_id, source, title, link, posted_at, scraped_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (listing.job_id, listing.source, listing.title, listing.link, posted, now),
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

    def search(self, keyword: str = "", source: str = "", limit: int = 50) -> list[sqlite3.Row]:
        """Search the full archive by title/source substring (the "open
        search" surface for everything that didn't make the SMS digest).
        Also exposed as a CLI: `python -m scraper.query <keyword>`.
        """
        sql = "SELECT * FROM jobs WHERE 1=1"
        params: list = []
        if keyword:
            sql += " AND title LIKE ?"
            params.append(f"%{keyword}%")
        if source:
            sql += " AND source LIKE ?"
            params.append(f"%{source}%")
        sql += " ORDER BY scraped_at DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(sql, params).fetchall()
