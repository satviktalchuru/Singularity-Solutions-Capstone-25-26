"""Fortune 500 / company career-page scraper.

Two interchangeable modes per CONFIG entry (see config.py for live examples):

mode "json"  -- the career site is backed by a clean JSON endpoint
                (Greenhouse, Lever, Workday, custom XHR found via devtools).
                Fetched with httpx; no browser involved.
    {
        "name": "Acme", "mode": "json",
        "url": "https://boards-api.greenhouse.io/v1/boards/acme/jobs",
        "list_path": "jobs",           # dotted path to the postings array
        "title_key": "title",          # dotted path inside each posting
        "link_key": "absolute_url",
        "posted_at_key": "updated_at", # optional; omit if the feed has none
    }

Dedicated Ashby (`sources/ashby.py`) and Workday (`sources/workday.py`)
modules exist for those two ATS platforms specifically -- prefer those when
a company uses one, since they parse posting dates for you. Use this
generic "json"/"html" scraper for everything else (Greenhouse, Lever,
custom career pages, etc.).

mode "html"  -- no JSON endpoint exists; render the page with stealth
                Playwright and pull cards out with CSS selectors.
    {
        "name": "Acme", "mode": "html",
        "url": "https://acme.com/careers",
        "card_selector": "a.job-card",  # one element per posting
        "title_selector": ".job-title", # inside the card; None = card text
        "link_attr": "href",
        "base_url": "https://acme.com", # prefix for relative links
    }
"""

from __future__ import annotations

import logging
import random
from urllib.parse import urljoin

import httpx

from ..config import MAX_RESULTS_PER_SOURCE, USER_AGENTS
from ..utils import Listing, dig, human_sleep, parse_timestamp
from .browser import stealth_page

log = logging.getLogger("scraper.fortune500")


async def _scrape_json(entry: dict) -> list[Listing]:
    """Hit a JSON careers endpoint and map postings to Listings."""
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(entry["url"], headers=headers)
        resp.raise_for_status()
        payload = resp.json()

    postings = dig(payload, entry.get("list_path", ""))
    if not isinstance(postings, list):
        log.warning("%s: list_path %r not found in response",
                    entry["name"], entry.get("list_path"))
        return []

    listings: list[Listing] = []
    for posting in postings[:MAX_RESULTS_PER_SOURCE]:
        title = dig(posting, entry["title_key"])
        link = dig(posting, entry["link_key"])
        if not (title and link):
            continue  # tolerate partial/odd records instead of crashing
        posted_at = None
        if entry.get("posted_at_key"):
            posted_at = parse_timestamp(dig(posting, entry["posted_at_key"]))
        listings.append(Listing(
            source=f"fortune500:{entry['name']}",
            title=str(title),
            link=urljoin(entry.get("base_url", entry["url"]), str(link)),
            posted_at=posted_at,
        ))
    return listings


async def _scrape_html(entry: dict) -> list[Listing]:
    """Render the careers page with stealth Playwright and parse job cards."""
    listings: list[Listing] = []
    async with stealth_page() as page:
        await page.goto(entry["url"], wait_until="domcontentloaded",
                        timeout=60_000)
        # Let lazy-loaded job boards hydrate, with a human-length pause.
        await human_sleep()

        cards = await page.query_selector_all(entry["card_selector"])
        if not cards:
            log.warning("%s: selector %r matched nothing (page layout "
                        "changed?)", entry["name"], entry["card_selector"])
            return []

        for card in cards[:MAX_RESULTS_PER_SOURCE]:
            try:
                if entry.get("title_selector"):
                    title_el = await card.query_selector(entry["title_selector"])
                    title = (await title_el.inner_text()) if title_el else None
                else:
                    title = await card.inner_text()

                link = await card.get_attribute(entry.get("link_attr", "href"))
                if not link and entry.get("link_attr", "href") != "href":
                    # Card itself isn't the anchor -- look for one inside.
                    anchor = await card.query_selector("a[href]")
                    link = await anchor.get_attribute("href") if anchor else None

                if not (title and title.strip() and link):
                    continue
                listings.append(Listing(
                    source=f"fortune500:{entry['name']}",
                    title=title.strip(),
                    link=urljoin(entry.get("base_url", entry["url"]), link),
                ))
            except Exception as exc:  # one bad card shouldn't kill the page
                log.debug("%s: skipping card (%s)", entry["name"], exc)
    return listings


async def scrape(entries: list[dict]) -> list[Listing]:
    """Scrape every configured company sequentially (gentler than parallel)."""
    results: list[Listing] = []
    for entry in entries:
        try:
            if entry.get("mode") == "html":
                found = await _scrape_html(entry)
            else:
                found = await _scrape_json(entry)
            log.info("%s: %d listing(s)", entry["name"], len(found))
            results.extend(found)
        except Exception as exc:
            log.error("%s: scrape failed (%s)", entry.get("name", "?"), exc)
        await human_sleep()  # pause between companies
    return results
