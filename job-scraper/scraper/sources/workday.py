"""Workday job-board watcher (pure httpx -- no browser needed).

Most large enterprises (much of the actual Fortune 500) run their careers
site on Workday, backed by a JSON search API the page itself calls via POST
-- the "CXS" API:

    https://<tenant>.<wdN>.myworkdayjobs.com/wday/cxs/<tenant>/<site>/jobs

To find it: open the company's careers page, devtools -> Network -> filter
XHR -> reload -> look for a POST request ending in `/jobs`. Copy that exact
URL into `api_url`. CONFIG entry:

    {
        "name": "Acme",
        "api_url": "https://acme.wd5.myworkdayjobs.com/wday/cxs/acme/External/jobs",
        # Optional: the human-facing careers base the job detail links live
        # under (visible in the browser address bar). If omitted we guess by
        # stripping "/wday/cxs/..." off api_url, which is usually close but
        # occasionally missing a locale segment -- set this if links 404.
        "career_site_url": "https://acme.wd5.myworkdayjobs.com/en-US/External",
        "search_text": "",  # optional keyword filter, same as the site's search box
    }

Workday doesn't return an exact posting timestamp, only relative text like
"Posted Today" / "Posted 3 Days Ago" / "Posted 30+ Days Ago". That text is
converted to an approximate UTC datetime so the 48h SMS filter still works;
"30+ Days Ago" (and anything unparsable) is treated as unknown-age and simply
excluded from the digest, not from storage.
"""

from __future__ import annotations

import logging
import random
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin

import httpx

from ..config import MAX_RESULTS_PER_SOURCE, USER_AGENTS
from ..utils import Listing, human_sleep

log = logging.getLogger("scraper.workday")

_RELATIVE_RE = re.compile(r"posted\s+(today|yesterday|(\d+)\+?\s*days?\s+ago)", re.I)


def _parse_relative_posted(text: str | None) -> datetime | None:
    """Convert Workday's "Posted N Days Ago" style text into a UTC datetime."""
    if not text:
        return None
    match = _RELATIVE_RE.search(text)
    if not match:
        return None
    phrase = match.group(1).lower()
    now = datetime.now(timezone.utc)
    if phrase == "today":
        return now
    if phrase == "yesterday":
        return now - timedelta(days=1)
    if "+" in match.group(0):  # "30+ Days Ago" -- deliberately left as unknown
        return None
    days = match.group(2)
    return now - timedelta(days=int(days)) if days else None


async def _scrape_company(client: httpx.AsyncClient, entry: dict) -> list[Listing]:
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = {
        "appliedFacets": {},
        "limit": MAX_RESULTS_PER_SOURCE,
        "offset": 0,
        "searchText": entry.get("search_text", ""),
    }
    resp = await client.post(entry["api_url"], json=body, headers=headers)
    resp.raise_for_status()
    postings = resp.json().get("jobPostings", [])

    base = entry.get("career_site_url") or entry["api_url"].split("/wday/cxs/")[0]
    listings: list[Listing] = []
    for posting in postings[:MAX_RESULTS_PER_SOURCE]:
        title = posting.get("title")
        path = posting.get("externalPath")
        if not (title and path):
            continue
        listings.append(Listing(
            source=f"workday:{entry['name']}",
            title=title,
            link=urljoin(base + "/", path.lstrip("/")),
            posted_at=_parse_relative_posted(posting.get("postedOn")),
        ))
    return listings


async def scrape(entries: list[dict]) -> list[Listing]:
    """Check every configured company's Workday CXS API."""
    results: list[Listing] = []
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for entry in entries:
            try:
                found = await _scrape_company(client, entry)
                log.info("%s: %d listing(s)", entry["name"], len(found))
                results.extend(found)
            except httpx.HTTPStatusError as exc:
                log.error("%s: HTTP %d (check api_url -- copy it fresh from "
                         "devtools if the tenant moved)",
                         entry.get("name", "?"), exc.response.status_code)
            except Exception as exc:
                log.error("%s: scrape failed (%s)", entry.get("name", "?"), exc)
            await human_sleep()
    return results
