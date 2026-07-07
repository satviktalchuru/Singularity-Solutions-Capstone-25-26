"""LinkedIn / job-board watcher via safe, login-free surfaces.

Scraping LinkedIn directly is both against its ToS and heavily bot-protected,
so this module deliberately stays off linkedin.com and uses two indirect,
free channels instead:

mode "google_cse" -- Google Programmable Search JSON API (free tier:
    100 queries/day) scoped with `site:linkedin.com/jobs`. Google has already
    indexed the postings; we just query the index. Requires GOOGLE_CSE_KEY
    and GOOGLE_CSE_CX in `.env`; the source is skipped cleanly if unset.

        {"name": "...", "mode": "google_cse",
         "query": 'site:linkedin.com/jobs "python" "remote"',
         "num_results": 10}

mode "rss" -- any public jobs RSS/Atom feed (WeWorkRemotely, RemoteOK,
    HN "Who is hiring" mirrors, company feeds...). Parsed with the stdlib.

        {"name": "...", "mode": "rss",
         "url": "https://weworkremotely.com/...jobs.rss",
         "keywords": ["python", "backend"]}   # [] = keep all
"""

from __future__ import annotations

import logging
import random
import xml.etree.ElementTree as ET

import httpx

from ..config import (GOOGLE_CSE_CX, GOOGLE_CSE_KEY, MAX_RESULTS_PER_SOURCE,
                      USER_AGENTS)
from ..utils import Listing, human_sleep, parse_timestamp, strip_html, truncate

log = logging.getLogger("scraper.job_boards")

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


async def _scrape_google_cse(client: httpx.AsyncClient, entry: dict) -> list[Listing]:
    """Query the free Google Programmable Search JSON API."""
    if not (GOOGLE_CSE_KEY and GOOGLE_CSE_CX):
        log.info("%s: GOOGLE_CSE_KEY/CX not set -- skipping", entry["name"])
        return []

    resp = await client.get(
        "https://www.googleapis.com/customsearch/v1",
        params={
            "key": GOOGLE_CSE_KEY,
            "cx": GOOGLE_CSE_CX,
            "q": entry["query"],
            "num": min(int(entry.get("num_results", 10)), 10),  # API max
            "dateRestrict": "d2",  # indexed within the last 2 days
        },
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])

    return [
        Listing(
            source=f"job_boards:{entry['name']}",
            title=truncate(item.get("title", "untitled"), 100),
            link=item["link"],
        )
        for item in items
        if item.get("link")
    ]


def _parse_feed(xml_text: str, name: str, keywords: list[str]) -> list[Listing]:
    """Parse RSS 2.0 or Atom, with optional keyword filtering on titles."""
    root = ET.fromstring(xml_text)

    # RSS 2.0: <rss><channel><item>...  |  Atom: <feed><entry>...
    # description/summary is captured too -- many job RSS feeds (WeWorkRemotely,
    # RemoteOK) include the full posting body here, which is what lets
    # filters.py check for seniority/years-of-experience language.
    items: list[tuple[str, str, str, str]] = []  # (title, link, published, description)
    for item in root.iter("item"):  # RSS
        title = item.findtext("title", default="").strip()
        link = item.findtext("link", default="").strip()
        published = item.findtext("pubDate", default="").strip()
        description = item.findtext("description", default="") or ""
        items.append((title, link, published, description))
    if not items:  # fall back to Atom
        for entry in root.findall("atom:entry", ATOM_NS):
            title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS)
                     or "").strip()
            link_el = entry.find("atom:link", ATOM_NS)
            link = link_el.get("href", "") if link_el is not None else ""
            published = (entry.findtext("atom:published", default="", namespaces=ATOM_NS)
                        or entry.findtext("atom:updated", default="", namespaces=ATOM_NS)
                        or "").strip()
            description = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS)
                          or entry.findtext("atom:content", default="", namespaces=ATOM_NS)
                          or "")
            items.append((title, link, published, description))

    wanted = [kw.lower() for kw in keywords]
    listings: list[Listing] = []
    for title, link, published, description in items[:MAX_RESULTS_PER_SOURCE]:
        if not (title and link):
            continue
        if wanted and not any(kw in title.lower() for kw in wanted):
            continue
        listings.append(Listing(
            source=f"job_boards:{name}",
            title=truncate(title, 100),
            link=link,
            posted_at=parse_timestamp(published),
            description=strip_html(description),
        ))
    return listings


async def _scrape_rss(client: httpx.AsyncClient, entry: dict) -> list[Listing]:
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    resp = await client.get(entry["url"], headers=headers)
    resp.raise_for_status()
    return _parse_feed(resp.text, entry["name"], entry.get("keywords", []))


async def scrape(entries: list[dict]) -> list[Listing]:
    """Run every configured job-board source with per-entry isolation."""
    results: list[Listing] = []
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for entry in entries:
            try:
                if entry.get("mode") == "google_cse":
                    found = await _scrape_google_cse(client, entry)
                else:
                    found = await _scrape_rss(client, entry)
                log.info("%s: %d listing(s)", entry["name"], len(found))
                results.extend(found)
            except ET.ParseError:
                log.error("%s: feed returned invalid XML", entry.get("name", "?"))
            except Exception as exc:
                log.error("%s: scrape failed (%s)", entry.get("name", "?"), exc)
            await human_sleep()
    return results
