# job-scraper — zero-cost daily job & content scraper

A locally hosted, 100% free daily scraper. Runs on your own machine, stores
everything in a local SQLite file to suppress duplicate rows, and appends
every newly-found listing to a plain CSV file — no email, no SMS, no
credentials, no gateway to configure. Open `listings.csv` in Excel/Sheets/
pandas whenever you want to check what's new.

```
┌──────────────────────── daily run (cron / Task Scheduler) ─────────────────────────────┐
│                                                                                         │
│  SCRAPING LAYER                    DEDUP LAYER          FILTER LAYER        OUTPUT      │
│  ─────────────────────────         ───────────────      ──────────────     ────────    │
│  fortune500  (Playwright/httpx)─┐                                                      │
│  ashby       (httpx JSON API)  ─┤                       drop senior/                   │
│  workday     (httpx CXS API)   ─┤─▶ jobs.db (SQLite) ─▶ staff/lead titles ─▶ listings.csv│
│  github      (httpx REST API)  ─┤   sha256(link|title)  & >2yr-experience              │
│  instagram   (RSS-Bridge)      ─┤   INSERT OR IGNORE     descriptions;                  │
│  job_boards  (Google CSE / RSS) ─┘   full history         new-grad first                │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## Quick start

```bash
cd job-scraper
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium

python -m scraper.main --dry-run    # test scrape, nothing saved
python -m scraper.main              # real run: dedupe + filter + append to listings.csv
python -m scraper.main --no-filter  # same, but skip seniority/experience filtering
python -m scraper.query python      # search everything ever scraped (unfiltered), any age
```

No `.env` file is required to run this at all — `cp .env.example .env` only
matters if you want the two optional tokens described below (higher GitHub
rate limits, the Google CSE job-board source).

## Filtering: no senior roles, 0-2 years target

Every new listing is checked by [`scraper/filters.py`](scraper/filters.py)
before it reaches the CSV:

- **Senior/staff/lead/manager/director/architect/"Engineer III"-style titles
  are dropped entirely** — checked against both the title and, when a source
  provides one, the full job description.
- **Postings whose description explicitly states more than 2 years of
  required experience are also dropped** (`CONFIG["max_years_experience"]`,
  config.py). If a source doesn't give a description (Workday's list
  endpoint, GitHub, Instagram, Google CSE), there's nothing to check, so
  it's kept rather than guessed away.
- **New grad / entry-level / intern keywords are a soft target, not a
  requirement**: matching listings get `new_grad_signal=True` and are sorted
  to the top of each run's output, but a listing without that keyword isn't
  penalized — it just wasn't stated either way.

Dropped listings are still recorded in `jobs.db` (so they're not re-evaluated
every run) but never written to the CSV. Run with `--no-filter` to see
everything unfiltered, or use `python -m scraper.query` to search the full
archive directly.

Only a few sources currently have description text to filter on: Ashby
(always), Greenhouse-via-`fortune500.py` (with `?content=true` +
`description_key` set, see below), and RSS feeds that include a
`<description>` (WeWorkRemotely, RemoteOK). Workday's search API and
HTML-scraped pages don't expose a description without an extra per-posting
fetch, so those are filtered on title only.

## Output: `listings.csv`

Every run appends new, filter-passing rows (existing ones are never
rewritten) with columns:

```
source, title, link, posted_at, scraped_at, recent_48h, new_grad_signal
```

`posted_at` is filled in when the source provides a real timestamp (Ashby,
GitHub, RSS `pubDate`, Instagram); `recent_48h` and `new_grad_signal` are
convenience True/False columns (blank if unknown) for filtering/sorting in
Excel/Sheets without recomputing them yourself.

## Configuration

Every URL, repo, account, selector, and keyword filter lives in one dictionary:
[`scraper/config.py`](scraper/config.py) → `CONFIG["sources"]`. Each scraper
module documents its entry format at the top of the file:

| Family | Module | Transport | Entry format | Has posting date? |
|---|---|---|---|---|
| `fortune500` | `sources/fortune500.py` | httpx (JSON) or stealth Playwright (HTML) | `mode: "json"` with `list_path`/`title_key`/`link_key`(+optional `posted_at_key`), or `mode: "html"` with CSS selectors | JSON: if `posted_at_key` set. HTML: no. |
| `ashby` | `sources/ashby.py` | httpx → Ashby's public JSON board API | `{"name": "...", "board": "..."}` | Yes — always. |
| `workday` | `sources/workday.py` | httpx → Workday CXS API (POST) | `{"name": "...", "api_url": "...", "career_site_url": "..."}` | Approximate — Workday only gives relative text ("Posted 3 Days Ago"). |
| `github` | `sources/github_repos.py` | httpx → GitHub REST API | `{"repo": "owner/name", "watch": ["commits", "releases"]}` | Yes — commit/release timestamps. |
| `instagram` | `sources/instagram.py` | httpx → RSS-Bridge Atom feed | `{"username": "zero2sudo", "bridge_url": "http://localhost:3000"}` (only `zero2sudo` is configured) | Yes — Atom `published`/`updated`. |
| `job_boards` | `sources/job_boards.py` | httpx → Google CSE API or RSS | `mode: "google_cse"` (needs free key) or `mode: "rss"` | RSS: yes (`pubDate`). Google CSE: no. |

Ashby and Workday are the two ATS platforms most Fortune-500-and-up companies
actually run their career sites on, so they're the best sources to add
companies to first if you want reliable posting dates.

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
into `CONFIG["sources"]["workday"]` as `api_url`.

### Instagram prerequisite

Instagram has no anonymous JSON surface anymore, so this source reads a
[RSS-Bridge](https://rss-bridge.org) feed. Run a free local instance once:

```bash
docker run -d --name rss-bridge --restart unless-stopped -p 3000:80 rssbridge/rss-bridge
```

Only public profiles work. Only `zero2sudo` is configured — add more entries
to `CONFIG["sources"]["instagram"]` if you want additional accounts, or empty
the list to drop Instagram entirely.

### Optional tokens (`.env`, not required to run)

- `GITHUB_TOKEN` — raises the unauthenticated GitHub API limit (60/hr) to
  5000/hr. Leave blank and it still works, just more likely to get
  rate-limited on a busy day.
- `GOOGLE_CSE_KEY` / `GOOGLE_CSE_CX` — free Google Programmable Search
  (100 queries/day), only used by the LinkedIn-via-Google-CSE job_boards
  entry. Leave blank to skip just that one source.

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
  IGNORE` into `jobs.db` means a listing is only ever appended to the CSV
  once, no matter how many times it's re-scraped. The table is append-only,
  doubling as a searchable history of everything found (`scraper.query`).
- **Fault isolation**: every source and every item is wrapped in its own
  try/except — a changed page layout or a dead feed logs an error and yields
  zero listings instead of killing the run.
- **No credentials required.** The old version of this project sent an SMS
  digest via Gmail SMTP + a carrier email-to-SMS gateway; that's gone. Output
  is just a local CSV file, so there's nothing to authenticate and nothing
  that leaves your machine.
- **LinkedIn**: never scraped directly (login-walled, ToS-protected).
  Postings are pulled from Google's already-public index via the free
  Programmable Search API, or from job-board RSS feeds.

## Respectful-use notes

This tool is built for low-volume personal use (one run per day, a handful of
pages). Keep it that way: it deliberately avoids login walls, honors public
endpoints, and paces itself. Check the terms of any site you add to the
config, and prefer official feeds/APIs (Greenhouse boards, RSS, GitHub REST)
over HTML scraping whenever they exist.
