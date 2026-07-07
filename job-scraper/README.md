# job-scraper — zero-cost daily job & content alerts

A locally hosted, 100% free daily scraper. Runs on your own residential
connection, stores everything in a local SQLite file to suppress duplicate
alerts, and texts you a condensed digest through a free email-to-SMS gateway.
No paid APIs, no cloud services, no subscriptions.

```
┌────────────────────────── daily run (cron / Task Scheduler) ─────────────────────────┐
│                                                                                      │
│  SCRAPING LAYER                     DATABASE LAYER            NOTIFICATION LAYER     │
│  ─────────────────────────          ────────────────          ────────────────────   │
│  fortune500  (Playwright/httpx) ─┐                                                   │
│  github      (httpx REST API)   ─┤─▶  jobs.db (SQLite) ─▶  new only ─▶ Gmail SMTP    │
│  instagram   (RSS-Bridge feed)  ─┤    sha256(link|title)              ─▶ SMS gateway │
│  job_boards  (Google CSE / RSS) ─┘    INSERT OR IGNORE                 (@vtext.com…) │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

## Quick start

```bash
cd job-scraper
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium

cp .env.example .env        # then fill in your Gmail app password etc.

python -m scraper.main --dry-run    # test scrape, nothing saved or sent
python -m scraper.main              # real run: dedupe + SMS digest
```

## Configuration

Every URL, repo, account, selector, and keyword filter lives in one dictionary:
[`scraper/config.py`](scraper/config.py) → `CONFIG["sources"]`. Each scraper
module documents its entry format at the top of the file:

| Family | Module | Transport | Entry format |
|---|---|---|---|
| `fortune500` | `sources/fortune500.py` | httpx (JSON) or stealth Playwright (HTML) | `mode: "json"` with `list_path`/`title_key`/`link_key`, or `mode: "html"` with CSS selectors |
| `github` | `sources/github_repos.py` | httpx → GitHub REST API | `{"repo": "owner/name", "watch": ["commits", "releases"]}` |
| `instagram` | `sources/instagram.py` | httpx → RSS-Bridge Atom feed | `{"username": "...", "bridge_url": "http://localhost:3000"}` |
| `job_boards` | `sources/job_boards.py` | httpx → Google CSE API or RSS | `mode: "google_cse"` (needs free key) or `mode: "rss"` |

Secrets go in `.env` (git-ignored) — see `.env.example` for every variable,
including the carrier gateway address table (`@vtext.com`, `@txt.att.net`, …).

### Finding a company's JSON careers endpoint

Open the careers page with browser devtools → Network → filter XHR → look for
a request returning job JSON (Greenhouse boards are simply
`https://boards-api.greenhouse.io/v1/boards/<company>/jobs`, no key needed).
Copy the URL into a `mode: "json"` entry and set `list_path` / `title_key` /
`link_key` to match the payload. Only fall back to `mode: "html"` + CSS
selectors when no clean endpoint exists.

### Instagram prerequisite

Instagram has no anonymous JSON surface anymore, so this source reads a
[RSS-Bridge](https://rss-bridge.org) feed. Run a free local instance once:

```bash
docker run -d --name rss-bridge --restart unless-stopped -p 3000:80 rssbridge/rss-bridge
```

Only public profiles work. If you don't care about Instagram, just empty the
`"instagram"` list in the config.

## Scheduling the daily run

**Linux/macOS (cron)** — 8:00 AM daily, with a jittered start (0–20 min) so
the run doesn't fire at a robotic fixed second:

```cron
0 8 * * * sleep $((RANDOM \% 1200)) && cd /path/to/job-scraper && .venv/bin/python -m scraper.main >> logs/run.log 2>&1
```

**Windows (Task Scheduler)** — create a daily task running:

```
C:\path\to\job-scraper\.venv\Scripts\python.exe -m scraper.main
```

with *Start in* set to `C:\path\to\job-scraper` (enable "random delay" in the
trigger settings for the same jitter effect).

## Design notes

- **Stealth**: `playwright-stealth` patches `navigator.webdriver` and other
  headless fingerprints; each run picks a random realistic viewport + UA pair
  (`sources/browser.py`), and every request is separated by a random 1–5 s
  human pause (`utils.human_sleep`). Sources start with staggered jitter so
  nothing fires simultaneously.
- **Deduplication**: `job_id = sha256(lowercased link | title)`; `INSERT OR
  IGNORE` into `jobs.db` means a listing alerts exactly once, ever. The table
  is append-only, doubling as a searchable history of everything found.
- **Fault isolation**: every source and every item is wrapped in its own
  try/except — a changed page layout or a dead feed logs an error and yields
  zero listings instead of killing the run or the SMS.
- **SMS budget**: the digest is compressed (abbreviated per-source counts,
  40-char titles), chunked at 150 chars per message, and hard-capped at 4
  parts per day so a scraper bug can't flood your phone.
- **LinkedIn**: never scraped directly (login-walled, ToS-protected).
  Postings are pulled from Google's already-public index via the free
  Programmable Search API, or from job-board RSS feeds.

## Respectful-use notes

This tool is built for low-volume personal use (one run per day, a handful of
pages). Keep it that way: it deliberately avoids login walls, honors public
endpoints, and paces itself. Check the terms of any site you add to the
config, and prefer official feeds/APIs (Greenhouse boards, RSS, GitHub REST)
over HTML scraping whenever they exist.
