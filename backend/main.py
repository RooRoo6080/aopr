"""
main.py

FastAPI application entry point.

Run with:
    cd backend
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import CONFIG
from cache import init_db, get_latest_snapshot
from tba_client import is_data_stale, get_team_info
from pipeline import run_season_pipeline, get_cached_results
from models import (
    TeamStats,
    TeamHistory,
    SeasonSummary,
    HealthResponse,
    MatchAuditRow,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AOPR Engine",
    description="Adjusted OPR — cross-event, season-wide FRC rating system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ---------------------------------------------------------------------------
# Background refresh
# ---------------------------------------------------------------------------

_refresh_lock = asyncio.Lock()


async def _refresh_loop() -> None:
    # Jitter: stagger multiple workers up to 60 s apart
    jitter = random.uniform(0, 60)
    logger.info("Refresh loop starting (jitter=%.1fs)", jitter)
    await asyncio.sleep(jitter)
    while True:
        try:
            async with _refresh_lock:
                logger.info("Background refresh starting…")
                await run_season_pipeline()
                logger.info("Background refresh complete.")
        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)
        await asyncio.sleep(CONFIG.refresh_interval_minutes * 60)


@app.on_event("startup")
async def startup_event() -> None:
    init_db()

    if not CONFIG.tba_auth_key:
        logger.error(
            "TBA_AUTH_KEY is not set. All TBA API requests will fail with 401. "
            "Set it in your .env file or environment before starting."
        )
    else:
        logger.info("TBA_AUTH_KEY configured (length=%d)", len(CONFIG.tba_auth_key))

    snap = get_latest_snapshot(CONFIG.season_year)
    if snap:
        logger.info("Loaded prior snapshot from DB (ts=%.0f)", snap.get("timestamp", 0))

    asyncio.create_task(_refresh_loop())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_results() -> Dict[str, Any]:
    results = get_cached_results()
    if not results:
        raise HTTPException(
            status_code=503,
            detail="No solve results yet. The pipeline is running — check /api/v1/health and try again shortly.",
        )
    return results


def _teams_for_event(results: Dict[str, Any], event_key: str) -> set:
    """Return the set of team numbers (int) that attended a given event."""
    membership: Dict[str, List[int]] = results.get("event_membership", {})
    return set(membership.get(event_key, []))


def _rank_rows(rows: list, sort_by: str, ascending: bool) -> list:
    valid = {"opr", "dpr", "aopr", "delta", "variability", "match_count"}
    key = sort_by if sort_by in valid else "aopr"
    rows.sort(key=lambda r: r.get(key, 0), reverse=not ascending)
    return [dict(r, rank=i + 1) for i, r in enumerate(rows)]


def _to_team_stats(r: dict) -> TeamStats:
    return TeamStats(
        rank=r.get("rank", 0),
        team_number=r["team_number"],
        nickname=r.get("nickname", ""),
        opr=r["opr"],
        dpr=r["dpr"],
        synergy=r.get("synergy", 0.0),
        primary_role=r.get("primary_role", ""),
        aopr=r["aopr"],
        event_opr=r.get("event_opr"),
        delta=r["delta"],
        variability=r["variability"],
        match_count=r["match_count"],
        breaker_count=r.get("breaker_count", 0),
        is_defender=r["is_defender"],
        low_match_warning=r["low_match_warning"],
    )


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    results = get_cached_results()
    ts = results.get("meta", {}).get("solve_timestamp") if results else None
    return HealthResponse(
        status="ok" if CONFIG.tba_auth_key else "no_api_key",
        season_year=CONFIG.season_year,
        data_stale=is_data_stale(),
        last_solve=ts,
    )


@app.get("/api/v1/meta", tags=["System"])
async def solver_meta() -> Dict[str, Any]:
    return _require_results().get("meta", {})


# ---------------------------------------------------------------------------
# Season
# ---------------------------------------------------------------------------

@app.get("/api/v1/season/current", response_model=SeasonSummary, tags=["Season"])
async def season_summary() -> SeasonSummary:
    results = _require_results()
    meta = results.get("meta", {})
    return SeasonSummary(
        season_year=CONFIG.season_year,
        total_events=meta.get("total_events", 0),
        total_matches=meta.get("total_matches", 0),
        total_teams=meta.get("total_teams", 0),
        last_solve_timestamp=meta.get("solve_timestamp"),
        data_stale_warning=is_data_stale(),
    )


@app.post("/api/v1/season/refresh", tags=["Season"])
async def trigger_refresh(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Manually trigger a pipeline refresh (non-blocking)."""
    async def _run() -> None:
        async with _refresh_lock:
            await run_season_pipeline()
    background_tasks.add_task(_run)
    return {"message": "Refresh started in the background."}


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@app.get("/api/v1/events", tags=["Events"])
async def list_events() -> List[Dict[str, Any]]:
    """All events in the season snapshot, sorted by week."""
    results = _require_results()
    event_meta: Dict[str, dict] = results.get("meta", {}).get("event_meta", {})
    membership: Dict[str, List[int]] = results.get("event_membership", {})
    team_results: Dict[int, Dict] = results.get("team_results", {})

    out = []
    for ek, em in event_meta.items():
        teams_at_event = membership.get(ek, [])
        
        # Calculate ECI (Event Competitiveness Index)
        # Average global Season AOPR of the event's top 24 playoff cohort
        aoprs = [team_results.get(t, {}).get("aopr", 0.0) for t in teams_at_event if t in team_results]
        aoprs.sort(reverse=True)
        top_cohort = aoprs[:24]
        eci = round(sum(top_cohort) / len(top_cohort), 1) if top_cohort else 0.0

        out.append({
            "event_key":   ek,
            "name":        em.get("name", ek),
            "week":        em.get("week"),
            "event_type":  em.get("event_type"),
            "district_key": em.get("district_key"),
            "team_count":  len(teams_at_event) or em.get("team_count", 0),
            "eci":         eci,
        })
        
    out.sort(key=lambda e: (e.get("week") or 99, e["event_key"]))
    return out


@app.get("/api/v1/event/{event_key}/stats", tags=["Events"])
async def event_stats(
    event_key: str,
    mode: str = Query("adjusted", pattern="^(raw|adjusted)$"),
    sort_by: str = Query("aopr"),
    ascending: bool = Query(False),
) -> Dict[int, Any]:
    """
    Event-level stats: ranked team list + aggregates (avg OPR/AOPR, top teams,
    defender list).  mode=raw sorts by OPR; mode=adjusted (default) by AOPR.
    """
    results = _require_results()
    team_results: Dict[int, Dict] = dict(results.get("team_results", {}))
    event_meta: Dict[str, dict] = results.get("meta", {}).get("event_meta", {})

    allowed = _teams_for_event(results, event_key)
    if not allowed:
        raise HTTPException(
            status_code=404,
            detail=f"Event '{event_key}' not found or has no team membership data.",
        )

    # Attach event_opr exclusively for this event
    event_oprs = results.get("event_oprs", {}).get(event_key, {})
    rows = []
    for k, v in team_results.items():
        if k in allowed:
            row_copy = dict(v)
            row_copy["event_opr"] = event_oprs.get(k)
            rows.append(row_copy)
    primary = "opr" if mode == "raw" else "aopr"
    sort_field = sort_by if sort_by in {"opr","dpr","aopr","delta","variability","match_count"} else primary
    rows = _rank_rows(rows, sort_field, ascending)

    ranked = [r for r in rows if r.get("match_count", 0) >= CONFIG.min_matches_to_rank]
    oprs  = [r["opr"]  for r in ranked]
    aoprs = [r["aopr"] for r in ranked]
    dprs  = [r["dpr"]  for r in ranked]

    def _avg(lst: list) -> float:
        return round(sum(lst) / len(lst), 1) if lst else 0.0

    def _top(lst: list) -> float:
        return round(max(lst), 1) if lst else 0.0

    em = event_meta.get(event_key, {})
    return {
        "event_key":  event_key,
        "event_name": em.get("name", event_key),
        "week":       em.get("week"),
        "mode":       mode,
        "team_stats": [_to_team_stats(r) for r in rows],
        "aggregates": {
            "team_count":     len(rows),
            "ranked_count":   len(ranked),
            "avg_opr":        _avg(oprs),
            "avg_aopr":       _avg(aoprs),
            "avg_dpr":        _avg(dprs),
            "top_opr":        _top(oprs),
            "top_aopr":       _top(aoprs),
            "defender_count": sum(1 for r in rows if r.get("is_defender")),
            "top3_aopr":      [r["team_number"] for r in rows[:3]],
            "defenders":      [r["team_number"] for r in rows if r.get("is_defender")],
        },
    }


@app.get("/api/v1/event/{event_key}/matches", tags=["Events"])
async def event_matches(event_key: str) -> List[Dict[str, Any]]:
    """
    All match rows for an event, de-duplicated by (match_key, side), sorted
    by timestamp.  Returns [] before the first pipeline run completes.
    """
    results = _require_results()
    team_match_rows: Dict = results.get("team_match_rows", {})
    allowed = _teams_for_event(results, event_key)

    seen: set = set()
    out: list = []
    for team_num, rows in team_match_rows.items():
        if team_num not in allowed:
            continue
        for row in rows:
            if row.get("event_key") != event_key:
                continue
            dedup_key = f"{row.get('match_key','')}_{row.get('side','')}"
            if dedup_key not in seen:
                seen.add(dedup_key)
                out.append(row)

    out.sort(key=lambda r: (r.get("timestamp", 0), r.get("match_key", "")))
    return out


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

@app.get("/api/v1/teams", response_model=List[TeamStats], tags=["Teams"])
async def all_teams(
    event_key: Optional[str] = Query(None, description="Filter to a specific event key"),
    min_matches: int = Query(0, ge=0),
    defenders_only: bool = Query(False),
    sort_by: str = Query("aopr"),
    ascending: bool = Query(False),
) -> List[TeamStats]:
    results = _require_results()
    team_results: Dict[int, Dict] = dict(results.get("team_results", {}))

    if event_key:
        allowed = _teams_for_event(results, event_key)
        if allowed:
            team_results = {t: v for t, v in team_results.items() if t in allowed}

    rows = list(team_results.values())
    if min_matches > 0:
        rows = [r for r in rows if r.get("match_count", 0) >= min_matches]
    if defenders_only:
        rows = [r for r in rows if r.get("is_defender", False)]

    rows = _rank_rows(rows, sort_by, ascending)
    return [_to_team_stats(r) for r in rows]


@app.get("/api/v1/team/{team_number}", response_model=TeamHistory, tags=["Teams"])
async def team_detail(team_number: int) -> TeamHistory:
    results = _require_results()
    team_results = results.get("team_results", {})

    r = team_results.get(team_number) or team_results.get(str(team_number))
    if not r:
        raise HTTPException(status_code=404, detail=f"Team {team_number} not found in season data")

    membership: Dict[str, List[int]] = results.get("event_membership", {})
    event_meta: Dict[str, dict] = results.get("meta", {}).get("event_meta", {})

    attended = [
        {"event_key": ek, "name": event_meta.get(ek, {}).get("name", ek)}
        for ek, teams in membership.items()
        if team_number in teams
    ]

    # Use the pipeline-collected nickname first (free — already in results).
    # Fall back to a live TBA call only when it's missing (e.g. cold start from DB).
    nickname = r.get("nickname", "")
    if not nickname:
        try:
            info = await get_team_info(team_number)
            if info:
                nickname = (info.get("nickname") or "").strip()
        except Exception:
            pass

    return TeamHistory(
        team_number=team_number,
        nickname=nickname,
        events=attended,
        season_opr=r["opr"],
        season_dpr=r["dpr"],
        season_aopr=r["aopr"],
        season_match_count=r["match_count"],
    )


@app.get("/api/v1/team/{team_number}/matches", tags=["Teams"])
async def team_matches(team_number: int) -> List[Dict[str, Any]]:
    """
    Full match-by-match history for a team.
    Rebuilt on every pipeline run — returns [] on cold start.
    """
    results = _require_results()
    rows = results.get("team_match_rows", {})
    return rows.get(team_number, rows.get(str(team_number), []))


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

@app.get("/api/v1/audit", response_model=List[MatchAuditRow], tags=["Audit"])
async def audit_rows(
    event_key: Optional[str] = Query(None),
    has_refund: bool = Query(False),
    limit: int = Query(500, le=2000),
) -> List[MatchAuditRow]:
    results = _require_results()
    audit = list(results.get("audit", []))

    if event_key:
        audit = [r for r in audit if r.get("event_key") == event_key]
    if has_refund:
        audit = [r for r in audit if (r.get("refund") or 0) > 0]

    return [
        MatchAuditRow(
            match_key=r["match_key"],
            event_key=r["event_key"],
            alliance_side=r.get("alliance_side", ""),
            expected_score=r["expected_score"],
            actual_score=r["actual_score"],
            residual=r["residual"],
            refund=r["refund"],
            defender_keys=r.get("defender_keys", []),
            row_weight=r["row_weight"],
            is_breaker=r.get("is_breaker", False),
            is_excluded=r.get("is_excluded", False),
        )
        for r in audit[:limit]
    ]


# ---------------------------------------------------------------------------
# Serve dashboard
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    if not INDEX_HTML.exists():
        return JSONResponse(
            {"error": "Frontend not found. Serve frontend/index.html separately."},
            status_code=404,
        )
    return FileResponse(str(INDEX_HTML))


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str) -> FileResponse:
    if not full_path.startswith("api/") and INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML))
    raise HTTPException(status_code=404)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host=CONFIG.host, port=CONFIG.port, reload=False)
