# job-scraper — zero-cost daily job & content alerts

A locally hosted, 100% free daily scraper. Runs on your own residential
connection, stores everything in a local SQLite file to suppress duplicate
alerts, and texts you a condensed digest — **only for postings from the last
48 hours** — through a free email-to-SMS gateway. Everything older (or
undated) still gets scraped and stored, just not texted; search it anytime
with `python -m scraper.query`. No paid APIs, no cloud services, no
subscriptions.

```
┌──────────────────────────── daily run (cron / Task Scheduler) ─────────────────────────────┐
│                                                                                             │
│  SCRAPING LAYER                        DATABASE LAYER          NOTIFICATION LAYER          │
│  ─────────────────────────             ───────────────         ────────────────────        │
│  fortune500  (Playwright/httpx)   ─┐                                                        │
│  ashby       (httpx JSON API)     ─┤                          ┌─▶ new & <48h ─▶ SMS digest  │
│  workday     (httpx CXS API)      ─┤─▶ jobs.db (SQLite) ──────┤   (Gmail SMTP → carrier      │
│  github      (httpx REST API)     ─┤   sha256(link|title)     │    gateway, e.g. @vtext.com) │
│  instagram   (RSS-Bridge, zero2sudo)┤  INSERT OR IGNORE       └─▶ everything else: stored    │
│  job_boards  (Google CSE / RSS)   ─┘  full history, always        only, `scraper.query`      │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Quick start

```bash
cd job-scraper
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium

cp .env.example .env        # then fill in your Gmail app password etc.

python -m scraper.main --dry-run    # test scrape, nothing saved or sent
python -m scraper.main              # real run: dedupe + SMS digest (<48h only)
python -m scraper.query python      # search everything ever scraped, any age
```

## Configuration

Every URL, repo, account, selector, and keyword filter lives in one dictionary:
[`scraper/config.py`](scraper/config.py) → `CONFIG["sources"]`. Each scraper
module documents its entry format at the top of the file:

| Family | Module | Transport | Entry format | Has posting date? |
|---|---|---|---|---|
| `fortune500` | `sources/fortune500.py` | httpx (JSON) or stealth Playwright (HTML) | `mode: "json"` with `list_path`/`title_key`/`link_key`(+optional `posted_at_key`), or `mode: "html"` with CSS selectors | JSON: if `posted_at_key` set. HTML: no. |
| `ashby` | `sources/ashby.py` | httpx → Ashby's public JSON board API | `{"name": "...", "board": "..."}` | Yes — always. |
| `workday` | `sources/workday.py` | httpx → Workday CXS API (POST) | `{"name": "...", "api_url": "...", "career_site_url": "..."}` | Approximate — Workday only gives relative text ("Posted 3 Days Ago"); "30+ Days Ago" counts as unknown. |
| `github` | `sources/github_repos.py` | httpx → GitHub REST API | `{"repo": "owner/name", "watch": ["commits", "releases"]}` | Yes — commit/release timestamps. |
| `instagram` | `sources/instagram.py` | httpx → RSS-Bridge Atom feed | `{"username": "zero2sudo", "bridge_url": "http://localhost:3000"}` (only `zero2sudo` is configured) | Yes — Atom `published`/`updated`. |
| `job_boards` | `sources/job_boards.py` | httpx → Google CSE API or RSS | `mode: "google_cse"` (needs free key) or `mode: "rss"` | RSS: yes (`pubDate`). Google CSE: no — Google's index date isn't a reliable posting date, so these never make the SMS digest, only the archive. |

**Why this matters**: only listings with a known `posted_at` within the last
48 hours (`CONFIG["notify"]["recent_hours"]`) go into the SMS. Ashby and
Workday are the two ATS platforms most Fortune-500-and-up companies actually
run their career sites on, so they're the primary sources for time-sensitive
alerts — add companies there first.

Secrets go in `.env` (git-ignored) — see `.env.example` for every variable,
including the carrier gateway address table (`@vtext.com`, `@txt.att.net`, …).

### Finding a company's JSON careers endpoint

Open the careers page with browser devtools → Network → filter XHR → look for
a request returning job JSON (Greenhouse boards are simply
`https://boards-api.greenhouse.io/v1/boards/<company>/jobs`, no key needed).
Copy the URL into a `mode: "json"` entry and set `list_path` / `title_key` /
`link_key` to match the payload. Only fall back to `mode: "html"` + CSS
selectors when no clean endpoint exists.

### Adding an Ashby or Workday company

**Ashby**: find the board name in the company's `jobs.ashbyhq.com/<board>` URL
and add `{"name": "...", "board": "<board>"}` to `CONFIG["sources"]["ashby"]`.
No key needed, and postings carry a real timestamp.

**Workday**: open the company's careers page, devtools → Network → filter
XHR → reload → find the POST request ending in `/jobs` (e.g.
`https://acme.wd5.myworkdayjobs.com/wday/cxs/acme/External/jobs`) and drop it
into `CONFIG["sources"]["workday"]` as `api_url`. Workday only reports
relative age ("Posted Today", "Posted 3 Days Ago"), which is converted to an
approximate timestamp — anything "30+ Days Ago" is treated as unknown-old and
excluded from the SMS (still stored).

### Instagram prerequisite

Instagram has no anonymous JSON surface anymore, so this source reads a
[RSS-Bridge](https://rss-bridge.org) feed. Run a free local instance once:

```bash
docker run -d --name rss-bridge --restart unless-stopped -p 3000:80 rssbridge/rss-bridge
```

Only public profiles work. Only `zero2sudo` is configured — add more entries
to `CONFIG["sources"]["instagram"]` if you want additional accounts, or empty
the list to drop Instagram entirely.

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
- **48-hour window**: every scraper attaches a `posted_at` timestamp when the
  source provides one (`Listing.posted_at` in `utils.py`). `main.py` texts
  only listings where `is_recent(48)` is true; everything else — older
  postings, or sources with no reliable date (HTML-scraped pages, Google CSE)
  — is still persisted in `jobs.db` and searchable with `scraper.query`, just
  silently, without a text.
- **LinkedIn**: never scraped directly (login-walled, ToS-protected).
  Postings are pulled from Google's already-public index via the free
  Programmable Search API, or from job-board RSS feeds.

## Respectful-use notes

This tool is built for low-volume personal use (one run per day, a handful of
pages). Keep it that way: it deliberately avoids login walls, honors public
endpoints, and paces itself. Check the terms of any site you add to the
config, and prefer official feeds/APIs (Greenhouse boards, RSS, GitHub REST)
over HTML scraping whenever they exist.
