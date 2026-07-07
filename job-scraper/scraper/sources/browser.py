"""Stealth Playwright helpers.

`stealth_page()` yields a ready-to-use Playwright page inside a Chromium
context with:
  * a randomly chosen realistic viewport + User-Agent pair per run
  * playwright-stealth patches applied (navigator.webdriver, plugins,
    languages, WebGL vendor strings, etc.)
  * sane locale/timezone so the fingerprint is internally consistent

Works with both playwright-stealth 1.x (`stealth_async`) and 2.x (`Stealth`).
"""

from __future__ import annotations

import logging
import random
from contextlib import asynccontextmanager
from typing import AsyncIterator

from playwright.async_api import Page, async_playwright

from ..config import USER_AGENTS, VIEWPORTS

log = logging.getLogger("scraper.browser")

# ---------------------------------------------------------------------------
# playwright-stealth changed its public API between 1.x and 2.x; support both
# and degrade gracefully (plain Playwright) if the package is missing.
# ---------------------------------------------------------------------------
try:  # 2.x
    from playwright_stealth import Stealth  # type: ignore

    _stealth = Stealth()

    async def _apply_stealth(page: Page) -> None:
        await _stealth.apply_stealth_async(page)

except ImportError:
    try:  # 1.x
        from playwright_stealth import stealth_async  # type: ignore

        async def _apply_stealth(page: Page) -> None:
            await stealth_async(page)

    except ImportError:  # not installed at all

        async def _apply_stealth(page: Page) -> None:
            log.warning("playwright-stealth not installed; running unpatched")


@asynccontextmanager
async def stealth_page(headless: bool = True) -> AsyncIterator[Page]:
    """Async context manager yielding a stealth-patched Playwright page.

    Usage:
        async with stealth_page() as page:
            await page.goto(url, wait_until="domcontentloaded")
    """
    viewport = random.choice(VIEWPORTS)
    user_agent = random.choice(USER_AGENTS)
    log.debug("launching chromium viewport=%s ua=%.40s…", viewport, user_agent)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=headless,
            args=[
                # Removes the "Chrome is being controlled by automated
                # software" flag from the navigator fingerprint.
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = await browser.new_context(
            viewport=viewport,
            user_agent=user_agent,
            locale="en-US",
            timezone_id="America/New_York",
        )
        page = await context.new_page()
        await _apply_stealth(page)
        try:
            yield page
        finally:
            await context.close()
            await browser.close()
