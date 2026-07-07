"""Locally hosted, zero-cost daily job/content scraper.

Layers:
    scraper.sources   -- Playwright/httpx scrapers (one module per source type)
    scraper.database  -- SQLite deduplication store
    scraper.notifier  -- Gmail SMTP -> email-to-SMS digest
    scraper.main      -- async orchestrator / CLI entry point
"""

__version__ = "1.0.0"
