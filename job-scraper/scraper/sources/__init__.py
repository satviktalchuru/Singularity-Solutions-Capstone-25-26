"""Source scrapers. Each module exposes an async `scrape(...)` coroutine
returning a list of `scraper.utils.Listing` objects and never raises --
per-source failures are logged and yield an empty list so one broken site
cannot take down the whole daily run.
"""

from . import ashby, fortune500, github_repos, instagram, job_boards, workday  # noqa: F401
