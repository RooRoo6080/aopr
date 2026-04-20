import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Load .env from the project root (two levels up from this file)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv optional; rely on real env vars


@dataclass
class SeasonConfig:
    season_year: int = field(default_factory=lambda: datetime.now().year)
    refresh_interval_minutes: int = 10
    time_decay_half_life_days: float = 21.0
    min_matches_to_rank: int = 6
    defender_threshold_multiplier: float = 1.5
    noise_exclusion_sigma: float = 2.5
    oppr_breaker_sigma: float = 2.0
    missed_elims_penalty: float = 12.0
    refund_credit_multiplier: float = 1.75
    tba_base_url: str = "https://www.thebluealliance.com/api/v3"
    tba_auth_key: str = field(default_factory=lambda: os.environ.get("TBA_AUTH_KEY", ""))
    db_path: str = field(default_factory=lambda: os.environ.get("DB_PATH", "aopr.db"))
    host: str = field(default_factory=lambda: os.environ.get("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "8000")))


CONFIG = SeasonConfig()
