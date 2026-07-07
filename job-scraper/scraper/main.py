"""Daily run orchestrator.

    python -m scraper.main                 # full run: scrape -> dedup -> append to CSV
    python -m scraper.main --dry-run       # scrape only; touch nothing
    python -m scraper.main --only github   # run a single source family
    python -m scraper.main -v              # debug logging

Sources run concurrently (they hit unrelated hosts, so parallelism is safe);
each source's *internal* requests stay sequential with 1-5 s human pauses.
A source that throws is logged and contributes zero listings -- one broken
site never cancels the others or the CSV write.

No credentials, no email, no SMS gateway, no residential-IP considerations --
every new listing found gets appended to `listings.csv` (see config.py) and
also stored in jobs.db for dedup + `python -m scraper.query` lookups.

Before writing to the CSV, every new listing passes through filters.py:
senior/staff/lead/manager-type titles are dropped entirely, as are postings
whose description explicitly requires more than CONFIG["max_years_experience"]
years -- both are hard filters. New-grad-signal listings aren't required,
just sorted first and flagged (`new_grad_signal` column) as a soft target.
Filtered-out listings are still recorded in jobs.db (so they're not
re-evaluated every run) but never appear in the CSV; use --no-filter to see
everything unfiltered.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys

from .config import CONFIG
from .csv_export import append_listings
from .database import JobStore
from .filters import apply_filters
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
        description="Zero-cost daily job scraper (Playwright + SQLite + CSV)")
    parser.add_argument("--dry-run", action="store_true",
                        help="scrape and print, but skip DB writes and the CSV")
    parser.add_argument("--only", choices=sorted(SOURCE_REGISTRY),
                        help="run a single source family")
    parser.add_argument("--no-filter", action="store_true",
                        help="skip seniority/experience filtering -- write everything new")
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

    if args.no_filter:
        kept = [(item, False) for item in fresh]
        dropped = []
    else:
        kept, dropped = apply_filters(fresh, max_years=CONFIG["max_years_experience"])

    if dropped:
        log.info("filtered out %d listing(s) as senior/over-experienced "
                 "(still recorded in jobs.db, just not in the CSV)", len(dropped))

    if not kept:
        log.info("nothing passed the filters this run -- done")
        return 0

    append_listings(CONFIG["csv_path"], kept)
    print(f"\n{len(kept)} new listing(s) appended to {CONFIG['csv_path']}"
         f" ({sum(1 for _, ng in kept if ng)} new-grad signal)")
    for item, new_grad in kept:
        tag = " [NEW GRAD]" if new_grad else ""
        print(f"  [{item.source}] {item.title}{tag}\n    {item.link}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
