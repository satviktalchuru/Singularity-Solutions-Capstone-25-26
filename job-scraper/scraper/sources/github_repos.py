"""GitHub repository watcher (pure httpx -- no browser needed).

Uses the public REST API, which allows 60 unauthenticated requests/hour;
set GITHUB_TOKEN in `.env` for 5000/hour. Each CONFIG entry:

    {"repo": "owner/name", "watch": ["commits", "releases"]}

"commits"  -> the newest commits on the default branch (great for repos like
              SimplifyJobs/Summer2026-Internships where each posting is a
              commit).
"releases" -> the latest published release.
"""

from __future__ import annotations

import logging
import random

import httpx

from ..config import GITHUB_TOKEN, USER_AGENTS
from ..utils import Listing, human_sleep, parse_timestamp, truncate

log = logging.getLogger("scraper.github")

API = "https://api.github.com"
COMMITS_PER_REPO = 5  # newest N commits considered per run


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": random.choice(USER_AGENTS),
    }
    if GITHUB_TOKEN:  # optional -- unauthenticated works, just rate-limited
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


async def _fetch_commits(client: httpx.AsyncClient, repo: str) -> list[Listing]:
    resp = await client.get(
        f"{API}/repos/{repo}/commits",
        params={"per_page": COMMITS_PER_REPO},
        headers=_headers(),
    )
    resp.raise_for_status()
    listings = []
    for commit in resp.json():
        message = commit.get("commit", {}).get("message", "")
        first_line = message.splitlines()[0] if message else "(no message)"
        commit_date = commit.get("commit", {}).get("author", {}).get("date")
        listings.append(Listing(
            source=f"github:{repo}",
            title=f"commit: {truncate(first_line, 80)}",
            link=commit.get("html_url", f"https://github.com/{repo}/commits"),
            posted_at=parse_timestamp(commit_date),
        ))
    return listings


async def _fetch_latest_release(client: httpx.AsyncClient, repo: str) -> list[Listing]:
    resp = await client.get(f"{API}/repos/{repo}/releases/latest",
                            headers=_headers())
    if resp.status_code == 404:  # repo simply has no releases -- not an error
        return []
    resp.raise_for_status()
    release = resp.json()
    name = release.get("name") or release.get("tag_name", "unnamed release")
    return [Listing(
        source=f"github:{repo}",
        title=f"release: {truncate(name, 80)}",
        link=release.get("html_url", f"https://github.com/{repo}/releases"),
        posted_at=parse_timestamp(release.get("published_at")),
    )]


async def scrape(entries: list[dict]) -> list[Listing]:
    """Check commits/releases for every configured repository."""
    results: list[Listing] = []
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for entry in entries:
            repo = entry.get("repo", "")
            try:
                if "commits" in entry.get("watch", []):
                    results.extend(await _fetch_commits(client, repo))
                if "releases" in entry.get("watch", []):
                    results.extend(await _fetch_latest_release(client, repo))
                log.info("%s: checked ok", repo)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 403:
                    log.error("%s: GitHub rate limit hit -- set GITHUB_TOKEN "
                              "in .env to raise it", repo)
                else:
                    log.error("%s: HTTP %d", repo, exc.response.status_code)
            except Exception as exc:
                log.error("%s: scrape failed (%s)", repo, exc)
            await human_sleep(scale=0.5)  # APIs need less courtesy than pages
    return results
