"""Zero-cost notification layer: Gmail SMTP -> email-to-SMS gateway.

Carrier gateways (e.g. 9259185702@vtext.com) accept plain email and deliver
the body as a text message, but truncate around 160 characters. The digest is
therefore compressed hard and split into numbered parts, capped at
`max_sms_parts` so a busy scrape day can't spam your phone. Any recipient
that looks like a normal inbox address also works -- it simply receives the
same condensed text.

This module only formats/sends whatever list of listings it's given -- the
"only listings from the last 48h" filtering happens one layer up, in
main.py, via `Listing.is_recent()`.
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from .config import CONFIG
from .utils import Listing, truncate

log = logging.getLogger("scraper.notify")


def build_digest(listings: list[Listing]) -> str:
    """Condense new listings into a compact plain-text digest.

    Format (one listing per line, grouped counts up front):

        3 new | F500:2 GH:1
        1.Sr Python Eng @Netflix bit.ly-style-link
        ...
    """
    if not listings:
        return ""

    # Per-source counts, abbreviated to keep the header short.
    abbrev = {"fortune500": "F500", "ashby": "ASHB", "workday": "WD",
              "github": "GH", "instagram": "IG", "job_boards": "JOBS"}
    counts: dict[str, int] = {}
    for item in listings:
        family = item.source.split(":", 1)[0]
        key = abbrev.get(family, family[:4].upper())
        counts[key] = counts.get(key, 0) + 1

    header = f"{len(listings)} new | " + " ".join(
        f"{k}:{v}" for k, v in sorted(counts.items())
    )

    lines = [header]
    for i, item in enumerate(listings, start=1):
        lines.append(f"{i}.{truncate(item.title, 40)} {item.link}")
    return "\n".join(lines)


def _chunk_for_sms(digest: str, chunk_size: int, max_parts: int) -> list[str]:
    """Split the digest on line boundaries into <=chunk_size char parts."""
    parts: list[str] = []
    current = ""
    for line in digest.splitlines():
        line = truncate(line, chunk_size)  # a single huge line still fits
        if current and len(current) + len(line) + 1 > chunk_size:
            parts.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
        if len(parts) == max_parts:
            break
    if current and len(parts) < max_parts:
        parts.append(current)

    if len(parts) > 1:  # tag multi-part messages: "(1/3) ..."
        parts = [f"({i}/{len(parts)}) {p}" for i, p in enumerate(parts, 1)]
    return parts


def send_digest(listings: list[Listing]) -> bool:
    """Email the condensed digest to every configured recipient.

    Returns True on success, False if config was incomplete or SMTP failed.
    Never raises -- a notification failure must not kill the scrape run
    (the DB already recorded the listings; check the logs instead).
    """
    cfg = CONFIG["notify"]
    if not listings:
        log.info("no new listings -- skipping notification")
        return True
    if not (cfg["sender"] and cfg["app_password"] and cfg["recipients"]):
        log.warning("notify config incomplete (check .env) -- digest below:\n%s",
                    build_digest(listings))
        return False

    digest = build_digest(listings)
    parts = _chunk_for_sms(digest, cfg["sms_chunk_size"], cfg["max_sms_parts"])

    try:
        # Implicit-SSL connection on port 465; one session for all messages.
        with smtplib.SMTP_SSL(cfg["smtp_host"], cfg["smtp_port"], timeout=30) as smtp:
            smtp.login(cfg["sender"], cfg["app_password"])
            for part in parts:
                msg = MIMEMultipart()
                msg["From"] = cfg["sender"]
                msg["To"] = ", ".join(cfg["recipients"])
                # Many SMS gateways prepend the subject -- keep it tiny.
                msg["Subject"] = "Jobs"
                msg.attach(MIMEText(part, "plain", "utf-8"))
                smtp.sendmail(cfg["sender"], cfg["recipients"], msg.as_string())
        log.info("sent %d message part(s) to %d recipient(s)",
                 len(parts), len(cfg["recipients"]))
        return True
    except smtplib.SMTPAuthenticationError:
        log.error("Gmail login failed -- verify GMAIL_ADDRESS and the App "
                  "Password (regular passwords are rejected).")
    except (smtplib.SMTPException, OSError) as exc:
        log.error("SMTP delivery failed: %s", exc)
    return False
