"""Daily run orchestrator.

    python -m scraper.main                 # full run: scrape -> dedup -> SMS
    python -m scraper.main --dry-run       # scrape only; touch nothing
    python -m scraper.main --no-notify     # scrape + store, skip the SMS
    python -m scraper.main --only github   # run a single source family
    python -m scraper.main -v              # debug logging

Sources run concurrently (they hit unrelated hosts, so parallelism is safe);
each source's *internal* requests stay sequential with 1-5 s human pauses.
A source that throws is logged and contributes zero listings -- one broken
site never cancels the others or the notification.

Every new listing is stored in jobs.db regardless of age. Only listings
posted within the last `notify.recent_hours` (48h by default, see config.py)
go out in the SMS digest -- the rest just sits in the database, searchable
with `python -m scraper.query`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys

from .config import CONFIG
from .database import JobStore
from .notifier import build_digest, send_digest
from .sources import ashby, fortune500, github_repos, instagram, job_boards, workday
from .utils import Listing, setup_logging

log = logging.getLogger("scraper.main")

# source-family name -> (scrape coroutine fn, config key)
SOURCE_REGISTRY = {
    "fortune500": fortune500.scrape,
    "ashby": ashby.scrape,
    "workday": workday.scrape,
    "github": github_repos.scrape,
    "instagram": instagram.scrape,
    "job_boards": job_boards.scrape,
}


async def run_all(only: str | None = None) -> list[Listing]:
    """Run the selected source families concurrently and merge results."""
    families = [only] if only else list(SOURCE_REGISTRY)

    async def guarded(name: str) -> list[Listing]:
        # Stagger startup so four browsers/requests don't fire in the same
        # instant -- looks more human and smooths local resource usage.
        await asyncio.sleep(random.uniform(0, 3))
        try:
            return await SOURCE_REGISTRY[name](CONFIG["sources"].get(name, []))
        except Exception as exc:  # belt-and-braces: sources shouldn't raise
            log.error("source %r failed at top level: %s", name, exc)
            return []

    batches = await asyncio.gather(*(guarded(name) for name in families))
    merged = [listing for batch in batches for listing in batch]
    log.info("scraped %d raw listing(s) across %d source(s)",
             len(merged), len(families))
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Zero-cost daily job scraper (Playwright + SQLite + SMS)")
    parser.add_argument("--dry-run", action="store_true",
                        help="scrape and print, but skip DB writes and SMS")
    parser.add_argument("--no-notify", action="store_true",
                        help="store new listings but do not send the digest")
    parser.add_argument("--only", choices=sorted(SOURCE_REGISTRY),
                        help="run a single source family")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="debug logging")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    raw = asyncio.run(run_all(args.only))

    if args.dry_run:
        print(f"\n--- DRY RUN: {len(raw)} listing(s), nothing persisted ---")
        for item in raw:
            print(f"[{item.source}] {item.title}\n    {item.link}")
        return 0

    store = JobStore(CONFIG["database_path"])
    fresh = store.filter_new(raw)  # persists ALL new listings, any age

    if not fresh:
        log.info("no new listings today -- done")
        return 0

    recent_hours = CONFIG["notify"]["recent_hours"]
    recent = [item for item in fresh if item.is_recent(recent_hours)]
    stale = len(fresh) - len(recent)
    log.info("%d new listing(s) within the last %dh (%d older/undated stored "
             "only -- see `python -m scraper.query`)",
             len(recent), recent_hours, stale)

    if not recent:
        log.info("nothing within the %dh SMS window -- done", recent_hours)
        return 0

    print(build_digest(recent))  # always echo the digest to the console/log
    if args.no_notify:
        return 0
    return 0 if send_digest(recent) else 1


if __name__ == "__main__":
    sys.exit(main())
