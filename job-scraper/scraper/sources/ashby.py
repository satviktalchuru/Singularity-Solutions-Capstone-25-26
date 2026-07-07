"""Ashby job-board watcher (pure httpx -- no browser needed).

Ashby (used by Notion, Ramp, Linear, and many other tech companies) exposes
a public, unauthenticated JSON API per company job board:

    https://api.ashbyhq.com/posting-api/job-board/<board-name>

Find <board-name> in the company's careers URL -- e.g.
`jobs.ashbyhq.com/notion` -> board name "notion". CONFIG entry:

    {"name": "Notion", "board": "notion"}

Each posting carries a `publishedDate` (-> `Listing.posted_at`) and usually a
`descriptionHtml`/`descriptionPlain` body (-> `Listing.description`, stripped
to plain text), which is what lets filters.py check for seniority/years-of-
experience language instead of just guessing off the title.
"""

from __future__ import annotations

import logging
import random

import httpx

from ..config import MAX_RESULTS_PER_SOURCE, USER_AGENTS
from ..utils import Listing, human_sleep, parse_timestamp, strip_html

log = logging.getLogger("scraper.ashby")

API = "https://api.ashbyhq.com/posting-api/job-board"


async def _scrape_company(client: httpx.AsyncClient, entry: dict) -> list[Listing]:
    headers = {"User-Agent": random.choice(USER_AGENTS), "Accept": "application/json"}
    resp = await client.get(f"{API}/{entry['board']}", headers=headers,
                            params={"includeCompensation": "false"})
    resp.raise_for_status()
    postings = resp.json().get("jobs", [])

    listings: list[Listing] = []
    for posting in postings[:MAX_RESULTS_PER_SOURCE]:
        title = posting.get("title")
        link = posting.get("jobUrl") or posting.get("applyUrl")
        if not (title and link):
            continue
        posted_at = parse_timestamp(
            posting.get("publishedDate") or posting.get("publishedAt")
            or posting.get("updatedAt")
        )
        description = strip_html(
            posting.get("descriptionPlain") or posting.get("descriptionHtml") or ""
        )
        listings.append(Listing(
            source=f"ashby:{entry['name']}",
            title=title,
            link=link,
            posted_at=posted_at,
            description=description,
        ))
    return listings


async def scrape(entries: list[dict]) -> list[Listing]:
    """Check every configured company's Ashby board."""
    results: list[Listing] = []
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for entry in entries:
            try:
                found = await _scrape_company(client, entry)
                log.info("%s: %d listing(s)", entry["name"], len(found))
                results.extend(found)
            except httpx.HTTPStatusError as exc:
                log.error("%s: HTTP %d (bad board name?)",
                         entry.get("name", "?"), exc.response.status_code)
            except Exception as exc:
                log.error("%s: scrape failed (%s)", entry.get("name", "?"), exc)
            await human_sleep(scale=0.5)
    return results
