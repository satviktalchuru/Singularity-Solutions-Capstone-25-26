"""Search the full local archive (jobs.db) -- everything scraped, not just
what made the last 48h SMS digest.

    python -m scraper.query                       # last 50 listings, any source
    python -m scraper.query python                # title contains "python"
    python -m scraper.query --source ashby         # only Ashby-sourced listings
    python -m scraper.query python --source workday --limit 20
"""

from __future__ import annotations

import argparse

from .config import CONFIG
from .database import JobStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search the local job archive")
    parser.add_argument("keyword", nargs="?", default="",
                        help="substring to match against the listing title")
    parser.add_argument("--source", default="",
                        help="substring to match against the source tag, "
                             "e.g. 'ashby', 'workday', 'zero2sudo'")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)

    store = JobStore(CONFIG["database_path"])
    rows = store.search(keyword=args.keyword, source=args.source, limit=args.limit)

    if not rows:
        print("no matches")
        return 0

    for row in rows:
        posted = row["posted_at"] or "unknown"
        print(f"[{row['source']}] {row['title']}\n"
              f"    posted: {posted}  scraped: {row['scraped_at']}\n"
              f"    {row['link']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
