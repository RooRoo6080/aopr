from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TeamStats(BaseModel):
    team_number: int
    nickname: str = ""
    opr: float = Field(description="Raw offensive power rating")
    dpr: float = Field(description="Defensive power rating")
    synergy: float = Field(default=0.0, description="Average positive residual (Boost ceiling)")
    aopr: float = Field(description="Adjusted OPR after defensive refunds")
    event_opr: Optional[float] = Field(default=None, description="Event-specific OPR")
    delta: float = Field(description="AOPR − OPR")
    variability: float = Field(description="Match-to-match instability score")
    match_count: int
    breaker_count: int = Field(default=0, description="Matches flagged as OPR breakers or excluded")
    primary_role: str = Field(default="", description="Algorithmic role classification")
    low_match_warning: bool = False
    rank: int = 0


class MatchAuditRow(BaseModel):
    match_key: str
    event_key: str
    alliance_side: str
    expected_score: float
    actual_score: float
    residual: float
    refund: float
    defender_keys: List[str] = []
    row_weight: float
    is_breaker: bool = False
    is_excluded: bool = False


class MatchDetail(BaseModel):
    match_key: str
    event_key: str
    comp_level: str
    red_teams: List[int]
    blue_teams: List[int]
    red_score: int
    blue_score: int
    timestamp: float
    is_playoff: bool
    status_flags: List[str] = []
    quality_weight: float = 1.0


class SolverDiagnostics(BaseModel):
    damp: float
    istop: int
    iterations: int
    r1norm: float
    n_teams: int
    n_rows: int


class EventStats(BaseModel):
    event_key: str
    season_year: int
    team_stats: List[TeamStats]
    audit_rows: List[MatchAuditRow] = []
    match_details: List[MatchDetail] = []
    noise_sigma: float
    avg_score: float
    defender_count: int
    solve_timestamp: float
    cache_fresh: bool = True
    data_stale_warning: bool = False
    solver_diagnostics: Optional[Dict[str, Any]] = None


class TeamHistory(BaseModel):
    team_number: int
    nickname: str = ""
    events: List[Dict[str, Any]] = []
    season_opr: float = 0.0
    season_dpr: float = 0.0
    season_aopr: float = 0.0
    season_match_count: int = 0


class SeasonSummary(BaseModel):
    season_year: int
    total_events: int
    total_matches: int
    total_teams: int
    last_solve_timestamp: Optional[float] = None
    data_stale_warning: bool = False


class HealthResponse(BaseModel):
    status: str
    season_year: int
    data_stale: bool
    last_solve: Optional[float] = None
    db_ok: bool = True
