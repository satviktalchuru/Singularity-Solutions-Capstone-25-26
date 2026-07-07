"""Public Instagram channel watcher -- no login, no paid API.

Instagram's old anonymous JSON endpoints (`?__a=1`) are login-walled now, so
the reliable zero-cost path is RSS-Bridge (https://rss-bridge.org): a free,
self-hostable proxy that turns a *public* profile into an Atom feed. Run one
locally in seconds:

    docker run -d -p 3000:80 rssbridge/rss-bridge

CONFIG entry:

    {"username": "nasa", "bridge_url": "http://localhost:3000"}

Each new post caption becomes a Listing (the permalink dedupes it in SQLite).
Only public accounts work -- private profiles have no anonymous surface.
"""

from __future__ import annotations

import logging
import random
import xml.etree.ElementTree as ET

import httpx

from ..config import MAX_RESULTS_PER_SOURCE, USER_AGENTS
from ..utils import Listing, human_sleep, parse_timestamp, truncate

log = logging.getLogger("scraper.instagram")

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _bridge_feed_url(bridge_url: str, username: str) -> str:
    """Build the RSS-Bridge Atom URL for a public Instagram profile."""
    return (
        f"{bridge_url.rstrip('/')}/?action=display&bridge=Instagram"
        f"&context=Username&u={username}&media_type=all&format=Atom"
    )


def _parse_atom(xml_text: str, username: str) -> list[Listing]:
    """Extract (caption, permalink) pairs from an Atom feed document."""
    listings: list[Listing] = []
    root = ET.fromstring(xml_text)
    for entry in root.findall("atom:entry", ATOM_NS)[:MAX_RESULTS_PER_SOURCE]:
        title_el = entry.find("atom:title", ATOM_NS)
        link_el = entry.find("atom:link", ATOM_NS)
        published_el = (entry.find("atom:published", ATOM_NS)
                       or entry.find("atom:updated", ATOM_NS))
        caption = (title_el.text or "").strip() if title_el is not None else ""
        link = link_el.get("href", "") if link_el is not None else ""
        if not link:
            continue
        listings.append(Listing(
            source=f"instagram:{username}",
            title=truncate(caption, 100) or "(no caption)",
            link=link,
            posted_at=parse_timestamp(published_el.text if published_el is not None else None),
        ))
    return listings


async def scrape(entries: list[dict]) -> list[Listing]:
    """Pull the latest public posts for every configured account."""
    results: list[Listing] = []
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for entry in entries:
            username = entry.get("username", "")
            try:
                url = _bridge_feed_url(entry["bridge_url"], username)
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                found = _parse_atom(resp.text, username)
                log.info("@%s: %d post(s) in feed", username, len(found))
                results.extend(found)
            except httpx.ConnectError:
                log.error("@%s: cannot reach RSS-Bridge at %s -- is the "
                          "container running?", username, entry.get("bridge_url"))
            except ET.ParseError:
                log.error("@%s: bridge returned non-XML (profile private or "
                          "bridge rate-limited?)", username)
            except Exception as exc:
                log.error("@%s: scrape failed (%s)", username, exc)
            await human_sleep()
    return results
