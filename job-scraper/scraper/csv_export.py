"""CSV output -- the replacement for the SMS/email digest. No credentials,
no residential-IP considerations, no third-party gateway: every run appends
its newly-found (and filter-passing -- see filters.py) listings to a local
CSV file you can open in Excel/Sheets or read with pandas.

Columns: source, title, link, posted_at, scraped_at, recent_48h, new_grad_signal
`recent_48h` and `new_grad_signal` are convenience flags (True/False/"" for
unknown-age) for filtering/sorting in Excel/Sheets -- they don't gate what
gets written. Senior-role and over-experience filtering (which DOES decide
what gets written) happens one layer up, in filters.py / main.py.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .utils import Listing

log = logging.getLogger("scraper.csv")

FIELDNAMES = ["source", "title", "link", "posted_at", "scraped_at",
             "recent_48h", "new_grad_signal"]


def append_listings(csv_path: str, listings_with_flags: Iterable[tuple[Listing, bool]]) -> int:
    """Append (listing, new_grad_signal) pairs to `csv_path`, writing a
    header if the file is new. Returns the number of rows written.
    """
    rows = list(listings_with_flags)
    if not rows:
        return 0

    path = Path(csv_path)
    is_new = not path.exists()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        for item, new_grad in rows:
            writer.writerow({
                "source": item.source,
                "title": item.title,
                "link": item.link,
                "posted_at": item.posted_at.isoformat(timespec="seconds") if item.posted_at else "",
                "scraped_at": now,
                "recent_48h": item.is_recent(48) if item.posted_at else "",
                "new_grad_signal": new_grad,
            })

    log.info("wrote %d row(s) to %s", len(rows), path)
    return len(rows)
