"""
tba_client.py

TBA API client with:
  - ETag / If-None-Match caching (respects TBA's cache hints)
  - Stale fallback on network failure (serves last cached payload + warning banner)
  - Exponential backoff retry (3 attempts, 1s/2s/4s)
  - Global semaphore (max 10 concurrent requests — stays well under TBA limits)
  - Filtered event list (skips offseason/preseason before fetching matches)
"""

import asyncio
import json
import logging
from typing import Any, List, Optional, Dict

import httpx

from config import CONFIG
from cache import get_cached, set_cached

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_STALE_WARNING: bool = False

# Cap concurrent TBA requests to avoid rate-limiting
_SEM = asyncio.Semaphore(10)

# Event types to skip when fetching matches (no match data worth processing)
_SKIP_EVENT_TYPES = {99, 100}   # Offseason, Preseason

# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

async def _tba_fetch(path: str) -> Optional[Any]:
    """
    Fetch one TBA endpoint with ETag caching, stale fallback, and retry.

    Returns parsed JSON or None on failure.  Never raises — failures result
    in either a stale cache hit or None.
    """
    global _STALE_WARNING

    if not CONFIG.tba_auth_key:
        logger.warning("TBA_AUTH_KEY not set — request to %s will fail", path)

    endpoint = f"{CONFIG.tba_base_url}{path}"
    cached = get_cached(endpoint)

    headers = {"X-TBA-Auth-Key": CONFIG.tba_auth_key}
    if cached and cached.get("etag"):
        headers["If-None-Match"] = cached["etag"]

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            async with _SEM:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(endpoint, headers=headers)

            if resp.status_code == 304:
                logger.debug("TBA 304 Not Modified: %s", path)
                _STALE_WARNING = False
                return json.loads(cached["response_body"])

            resp.raise_for_status()
            body = resp.text
            etag = resp.headers.get("ETag", "")
            set_cached(endpoint, etag, body, "fresh")
            _STALE_WARNING = False
            return resp.json()

        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            last_exc = exc
            wait = 2 ** attempt  # 1s, 2s, 4s
            logger.warning(
                "TBA fetch attempt %d/3 failed for %s: %s. Retrying in %ds.",
                attempt + 1, path, exc, wait,
            )
            if attempt < 2:
                await asyncio.sleep(wait)

    # All retries exhausted — try stale cache
    if cached and cached.get("response_body"):
        logger.warning("All retries failed for %s; serving stale cache.", path)
        _STALE_WARNING = True
        return json.loads(cached["response_body"])

    logger.error("TBA fetch failed with no cache for %s: %s", path, last_exc)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_data_stale() -> bool:
    return _STALE_WARNING


async def get_season_events(year: int) -> List[Dict]:
    """
    Fetch all events for a season, filtering out offseason and preseason entries.
    Only returns events within the target year (by start_date).
    """
    data = await _tba_fetch(f"/events/{year}")
    if not isinstance(data, list):
        return []

    filtered: List[Dict] = []
    for ev in data:
        et = ev.get("event_type")
        # Skip offseason / preseason types
        if et in _SKIP_EVENT_TYPES:
            continue
        # Skip events whose start_date is not in the target year
        start = ev.get("start_date", "")
        if start and not start.startswith(str(year)):
            continue
            
        # Filter strictly down to California to reduce API calls
        state = ev.get("state_prov", "")
        if state not in ("CA", "California"):
            continue
            
        filtered.append(ev)

    logger.info(
        "Season %d: %d events after filtering (%d total returned by TBA)",
        year, len(filtered), len(data),
    )
    return filtered


async def get_event_matches(event_key: str) -> List[Dict]:
    data = await _tba_fetch(f"/event/{event_key}/matches")
    return data if isinstance(data, list) else []


async def get_event_teams(event_key: str) -> List[Dict]:
    data = await _tba_fetch(f"/event/{event_key}/teams/simple")
    return data if isinstance(data, list) else []


async def get_team_info(team_number: int) -> Optional[Dict]:
    return await _tba_fetch(f"/team/frc{team_number}")
