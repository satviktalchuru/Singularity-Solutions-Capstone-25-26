"""CSV output -- the replacement for the SMS/email digest. No credentials,
no residential-IP considerations, no third-party gateway: every run appends
its newly-found listings to a local CSV file you can open in Excel/Sheets or
read with pandas.

Columns: source, title, link, posted_at, scraped_at, recent_48h
`recent_48h` is just a convenience flag (True/False/"" for unknown-age) so
you can filter/sort for "what's new today" without recomputing it yourself;
it doesn't gate what gets written -- every new listing is written, always.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .utils import Listing

log = logging.getLogger("scraper.csv")

FIELDNAMES = ["source", "title", "link", "posted_at", "scraped_at", "recent_48h"]


def append_listings(csv_path: str, listings: Iterable[Listing]) -> int:
    """Append listings to `csv_path`, writing a header if the file is new.
    Returns the number of rows written.
    """
    listings = list(listings)
    if not listings:
        return 0

    path = Path(csv_path)
    is_new = not path.exists()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        for item in listings:
            writer.writerow({
                "source": item.source,
                "title": item.title,
                "link": item.link,
                "posted_at": item.posted_at.isoformat(timespec="seconds") if item.posted_at else "",
                "scraped_at": now,
                "recent_48h": item.is_recent(48) if item.posted_at else "",
            })

    log.info("wrote %d row(s) to %s", len(listings), path)
    return len(listings)
