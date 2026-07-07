"""Seniority/experience-level filtering.

Two different behaviors, per what was asked for:
  1. "No senior roles at all" -- a HARD filter. Anything whose title (or
     description, when available) reads as senior/staff/lead/manager/
     director/etc. is dropped entirely, never written to the CSV.
  2. "Max 0-2 years of experience" -- also a HARD filter, but only applied
     when the job description actually states a required years-of-
     experience range. If a listing has no description text (most sources
     only give title+link -- see Listing.description in utils.py), there's
     nothing to check, so it's kept rather than guessed away.
  3. "New grad positions as a soft target" -- never excludes anything.
     Listings whose title/description mention new-grad-style keywords get
     `new_grad_signal=True` in the CSV and are sorted first in each run's
     output, but a listing missing that keyword is not penalized for it.

Keyword/regex matching is inherently imperfect (e.g. "Team Lead" reads as
senior, but "Leadership Development Program" -- a common new-grad rotational
program name -- would too if not special-cased). Treat this as a solid first
pass, not a guarantee; skip it and use `python -m scraper.query` directly if
you want to see everything unfiltered.
"""

from __future__ import annotations

import re

from .utils import Listing

# Ordered so more-specific phrases are checked before generic ones matter
# less here since these are independent regexes, but grouped for readability.
SENIOR_PATTERNS = [
    r"\bsenior\b", r"\bsr\.?\b", r"\bstaff\b", r"\bprincipal\b",
    r"\blead\b", r"\btech(?:nical)? lead\b", r"\bmanager\b", r"\bdirector\b",
    r"\bhead of\b", r"\bvp\b", r"\bvice president\b", r"\bexecutive\b",
    r"\barchitect\b", r"\bdistinguished\b", r"\bfellow\b",
    r"\biii\b", r"\biv\b", r"\bv\b(?!\w)",  # level suffixes: Engineer III/IV
]
_SENIOR_RE = re.compile("|".join(SENIOR_PATTERNS), re.IGNORECASE)

# Explicitly NOT senior even though they might contain a senior-ish word
# fragment -- e.g. "Leadership Development Program" is a common new-grad
# rotational-program name, not a leadership hire.
SENIOR_FALSE_POSITIVE_RE = re.compile(
    r"\bleadership (development|rotational) program\b", re.IGNORECASE
)

NEW_GRAD_PATTERNS = [
    r"\bnew grad(uate)?\b", r"\bnew college grad(uate)?\b",
    r"\brecent grad(uate)?\b", r"\bentry[\s-]?level\b",
    r"\bearly[\s-]?career\b", r"\bcampus\b", r"\buniversity grad(uate)?\b",
    r"\bclass of 20\d\d\b", r"\bjunior\b", r"\bjr\.?\b",
    r"\bassociate\b", r"\bgraduate program\b", r"\brotational program\b",
    r"\bintern(ship)?\b",
]
_NEW_GRAD_RE = re.compile("|".join(NEW_GRAD_PATTERNS), re.IGNORECASE)

# Matches "0-2 years", "1+ years", "at least 3 years", "minimum of 2 years",
# "2 to 4 years" etc., in the neighborhood of the word "experience".
_YEARS_RE = re.compile(
    r"(\d+)\s*(?:\+|-|to)?\s*(?:\d+)?\s*\+?\s*years?\s*(?:of\s*)?(?:relevant\s*|professional\s*|work(?:ing)?\s*)?experience",
    re.IGNORECASE,
)


def is_senior(title: str, description: str = "") -> bool:
    text = f"{title} {description}"
    if SENIOR_FALSE_POSITIVE_RE.search(text):
        # Strip the false-positive phrase out before testing, in case the
        # rest of the text independently contains a real senior signal.
        text = SENIOR_FALSE_POSITIVE_RE.sub("", text)
    return bool(_SENIOR_RE.search(text))


def has_new_grad_signal(title: str, description: str = "") -> bool:
    return bool(_NEW_GRAD_RE.search(f"{title} {description}"))


def min_years_required(description: str) -> int | None:
    """Smallest stated minimum years-of-experience requirement found in the
    description, or None if the text doesn't mention one at all.
    """
    if not description:
        return None
    matches = [int(m.group(1)) for m in _YEARS_RE.finditer(description)]
    return min(matches) if matches else None


def evaluate(listing: Listing, max_years: int = 2) -> tuple[bool, bool]:
    """Returns (keep, new_grad_signal).

    keep=False means the listing should be dropped entirely (senior role,
    or description explicitly requires more than `max_years`).
    """
    if is_senior(listing.title, listing.description):
        return False, False

    years = min_years_required(listing.description)
    if years is not None and years > max_years:
        return False, False

    return True, has_new_grad_signal(listing.title, listing.description)


def apply_filters(
    listings: list[Listing], max_years: int = 2
) -> tuple[list[tuple[Listing, bool]], list[Listing]]:
    """Split listings into (kept, dropped).

    `kept` is a list of (listing, new_grad_signal) pairs, sorted so
    new-grad-signal listings come first (the "soft target" behavior).
    `dropped` is listings excluded as senior or over-experienced.
    """
    kept_with_flag: list[tuple[Listing, bool]] = []
    dropped: list[Listing] = []
    for listing in listings:
        keep, new_grad = evaluate(listing, max_years)
        if keep:
            kept_with_flag.append((listing, new_grad))
        else:
            dropped.append(listing)

    kept_with_flag.sort(key=lambda pair: not pair[1])  # new_grad=True first
    return kept_with_flag, dropped
