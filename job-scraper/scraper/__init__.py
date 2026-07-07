"""Locally hosted, zero-cost daily job/content scraper.

Layers:
    scraper.sources    -- Playwright/httpx scrapers (one module per source type)
    scraper.database   -- SQLite deduplication store
    scraper.csv_export -- appends new listings to a local CSV, no credentials
    scraper.main       -- async orchestrator / CLI entry point
"""

__version__ = "1.0.0"
