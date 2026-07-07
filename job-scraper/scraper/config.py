"""Central configuration.

Everything you are likely to tweak lives in this one file:
  * which sites/repos/accounts get scraped (CONFIG["sources"])
  * stealth parameters (viewports, user agents, sleep range)
  * database path and notification settings

Secrets (Gmail app password, API tokens) are NOT stored here -- they are read
from the environment / a `.env` file so this file is safe to commit.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load `.env` sitting next to the project root (silently ignored if missing).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Stealth: realistic desktop fingerprints. One (viewport, UA) pair is picked
# at random per browser context so consecutive runs don't look identical.
# ---------------------------------------------------------------------------
VIEWPORTS: list[dict[str, int]] = [
    {"width": 1920, "height": 1080},
    {"width": 1536, "height": 864},
    {"width": 1440, "height": 900},
    {"width": 1366, "height": 768},
    {"width": 2560, "height": 1440},
]

USER_AGENTS: list[str] = [
    # Chrome on Windows 11
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    # Chrome on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 "
    "Firefox/127.0",
]

# Human-like pause between page actions / requests, in seconds.
SLEEP_RANGE: tuple[float, float] = (1.0, 5.0)

# Hard cap per source so a broken selector can't flood the digest.
MAX_RESULTS_PER_SOURCE = 25

CONFIG: dict = {
    "database_path": str(PROJECT_ROOT / "jobs.db"),

    # -----------------------------------------------------------------------
    # NOTIFICATIONS -- SMS gateways truncate around 160 chars, so the digest
    # is chunked; `max_sms_parts` caps how many messages one run may send.
    # -----------------------------------------------------------------------
    "notify": {
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 465,  # implicit SSL
        "sender": os.getenv("GMAIL_ADDRESS", ""),
        "app_password": os.getenv("GMAIL_APP_PASSWORD", ""),
        "recipients": [
            r.strip()
            for r in os.getenv("NOTIFY_RECIPIENTS", "").split(",")
            if r.strip()
        ],
        "sms_chunk_size": 150,
        "max_sms_parts": 4,
    },

    # -----------------------------------------------------------------------
    # SOURCES -- add/remove entries freely; each scraper module documents its
    # own entry format at the top of the file.
    # -----------------------------------------------------------------------
    "sources": {

        # ---- Fortune 500 / company career pages -------------------------
        # mode "json": a clean JSON endpoint (many career sites are backed by
        #   Workday/Greenhouse/Lever-style APIs -- find the XHR in devtools).
        # mode "html": rendered page scraped with stealth Playwright.
        "fortune500": [
            {
                "name": "Netflix",
                "mode": "json",
                "url": (
                    "https://explore.jobs.netflix.net/api/apply/v2/jobs"
                    "?domain=netflix.com&num=10&query=software%20engineer"
                ),
                # Dotted path from the JSON root to the list of postings.
                "list_path": "positions",
                # Keys (dotted paths allowed) inside each posting.
                "title_key": "name",
                "link_key": "canonicalPositionUrl",
            },
            {
                "name": "Greenhouse demo (Stripe)",
                "mode": "json",
                # Greenhouse exposes a public, keyless JSON board per company:
                # https://boards-api.greenhouse.io/v1/boards/<company>/jobs
                "url": "https://boards-api.greenhouse.io/v1/boards/stripe/jobs",
                "list_path": "jobs",
                "title_key": "title",
                "link_key": "absolute_url",
            },
            {
                "name": "Anthropic (HTML example)",
                "mode": "html",
                "url": "https://www.anthropic.com/jobs",
                # CSS selectors: one per job card, then title/link inside it.
                "card_selector": "a[href*='/jobs/']",
                "title_selector": None,   # None => use the card's own text
                "link_attr": "href",      # attribute holding the URL
                "base_url": "https://www.anthropic.com",  # for relative hrefs
            },
        ],

        # ---- GitHub repositories (httpx, no browser needed) -------------
        # watch: "releases", "commits", or both.
        "github": [
            {"repo": "SimplifyJobs/Summer2026-Internships", "watch": ["commits"]},
            {"repo": "microsoft/playwright-python", "watch": ["releases"]},
        ],

        # ---- Public Instagram channels via RSS-Bridge --------------------
        # Instagram's anonymous JSON endpoints are login-walled these days;
        # RSS-Bridge (free, self-hostable: https://rss-bridge.org) proxies a
        # public profile into a stable Atom feed. Run your own instance with
        #   docker run -d -p 3000:80 rssbridge/rss-bridge
        # or pick a public instance from the RSS-Bridge wiki.
        "instagram": [
            {
                "username": "nasa",
                "bridge_url": "http://localhost:3000",
            },
        ],

        # ---- LinkedIn / job boards ---------------------------------------
        # mode "google_cse": free Google Programmable Search JSON API scoped
        #   to a jobs site -- avoids scraping LinkedIn itself (login-walled
        #   and aggressively bot-protected).
        # mode "rss": any public RSS/Atom jobs feed.
        "job_boards": [
            {
                "name": "LinkedIn via Google CSE",
                "mode": "google_cse",
                "query": 'site:linkedin.com/jobs "software engineer" "new grad"',
                "num_results": 10,
            },
            {
                "name": "WeWorkRemotely - Programming",
                "mode": "rss",
                "url": "https://weworkremotely.com/categories/remote-programming-jobs.rss",
                # Optional case-insensitive keyword filter on titles
                # (empty list = keep everything).
                "keywords": ["python", "backend", "engineer"],
            },
            {
                "name": "RemoteOK RSS",
                "mode": "rss",
                "url": "https://remoteok.com/remote-python-jobs.rss",
                "keywords": [],
            },
        ],
    },
}

# Optional tokens (read once here so scrapers never touch os.environ).
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
GOOGLE_CSE_KEY = os.getenv("GOOGLE_CSE_KEY", "").strip()
GOOGLE_CSE_CX = os.getenv("GOOGLE_CSE_CX", "").strip()
