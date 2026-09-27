"""Live outbound calls. Two of them, and both are free and key-less.

`rdap.org` gives a registration date that a model cannot hallucinate, and the
submitted site is read as a self-report. Nothing here asks an API key, and
nothing here decides a fact: a failed call is reported as a gap, never
substituted with a guess.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import httpx

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)

PAGE_TIMEOUT = float(12.0)
RDAP_TIMEOUT = float(10.0)


def _client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(20.0, connect=6.0),
        follow_redirects=True,
        headers={"User-Agent": UA, "Accept-Language": "en-GB,en;q=0.9"},
    )


def fetch_website(domain: str) -> Tuple[Optional[str], dict]:
    """Surf the submitted site. Its content is DATA, never instructions: the
    caller only ever regexes phrases out of it and labels the result self-report."""
    meta: dict = {"url": "https://" + domain, "ok": False, "ms": None, "error": None}
    started = time.time()
    try:
        with _client() as c:
            r = c.get("https://" + domain, timeout=PAGE_TIMEOUT)
        meta["ms"] = int((time.time() - started) * 1000)
        meta["status"] = r.status_code
        # A body is only useful if it is HTML, not a binary or an interstitial.
        ctype = r.headers.get("content-type", "")
        meta["content_type"] = ctype
        if r.status_code >= 400:
            meta["error"] = "HTTP %d" % r.status_code
            return None, meta
        if ctype and not any(t in ctype.lower() for t in ("html", "text", "xml", "json", "plain")):
            meta["error"] = "not a text page (%s)" % ctype
            return None, meta
        meta["ok"] = True
        return r.text, meta
    except Exception as exc:  # noqa: BLE001 - an unreachable site is a finding, not a crash
        meta["ms"] = int((time.time() - started) * 1000)
        meta["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:160])
        return None, meta


def fetch_rdap(domain: str) -> Tuple[dict, dict]:
    """Registration date from RDAP. Free, no key, and the date it returns is
    published by the registry rather than generated."""
    meta: dict = {"ok": False, "ms": None, "error": None}
    started = time.time()
    try:
        with _client() as c:
            r = c.get("https://rdap.org/domain/" + domain, timeout=RDAP_TIMEOUT)
        meta["ms"] = int((time.time() - started) * 1000)
        meta["status"] = r.status_code
        if r.status_code != 200:
            meta["error"] = "HTTP %d" % r.status_code
            return {}, meta
        data = r.json()
        meta["ok"] = True
        return data if isinstance(data, dict) else {}, meta
    except Exception as exc:  # noqa: BLE001
        meta["ms"] = int((time.time() - started) * 1000)
        meta["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:160])
        return {}, meta


def normalise_rdap(raw: dict, domain: Optional[str]) -> dict:
    """Turn the RDAP response into the one deterministic fact the reasoner needs.
    A model is never asked for a date: it can hallucinate one.

    Some TLDs publish no RDAP service at all. When that happens the result is
    `source_status: "cached"` with `age_days: null` -- never a guess, and never
    a 0 standing in for unknown.
    """
    registration: Optional[str] = None
    age_days: Optional[int] = None
    reason: Optional[str] = None
    source_status = "live"

    events = raw.get("events") if isinstance(raw.get("events"), list) else []
    reg_event = None
    for e in events:
        if str(e.get("eventAction", "")).lower() == "registration":
            reg_event = e
            break

    if reg_event and reg_event.get("eventDate"):
        registration = str(reg_event["eventDate"])[:10]
        try:
            created = datetime.strptime(registration, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            age_days = max(0, round((datetime.now(timezone.utc) - created).total_seconds() / 86400))
        except ValueError:
            registration = None

    if not registration:
        source_status = "cached"
        reason = (
            "RDAP published no record for this TLD (the IANA bootstrap has no RDAP service for it, "
            "e.g. .ac.uk). Domain age is unconfirmed."
            if not raw
            else "RDAP responded but carried no registration event."
        )

    return {
        "registration_date": registration,
        "age_days": age_days,
        "reason": reason,
        "source_status": source_status,
        "source_url": ("https://rdap.org/domain/" + domain) if domain else None,
    }
