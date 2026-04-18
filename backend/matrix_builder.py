from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from config import CONFIG
from match_normalizer import MatchRecord, STAGE_WEIGHTS


# TBA event_type → weight multiplier
# Types: 0=Regional, 1=District, 2=DistrictChamp, 3=ChampDiv, 4=ChampFinals,
#        5=DistrictChampDiv, 6=FestivalOfChampions, 7=Remote, 99=Offseason, 100=Preseason
EVENT_TYPE_WEIGHTS: Dict[int, float] = {
    0:   1.00,   # Regional
    1:   1.00,   # District
    2:   1.15,   # District Championship
    5:   1.15,   # District Championship Division
    6:   1.10,   # Festival of Champions
    3:   1.30,   # Championship Division (worlds)
    4:   1.40,   # Championship Finals (worlds finals)
    7:   0.85,   # Remote events — lower data quality
    99:  0.00,   # Offseason — exclude entirely
    100: 0.00,   # Preseason — exclude entirely
    -1:  1.00,   # Unlabeled
}


class MatrixBundle(NamedTuple):
    """Everything the solver needs for one weighted least-squares solve."""
    A: sp.csr_matrix          # scoring alliance design matrix   [n_rows × n_teams]
    A_opp: sp.csr_matrix      # opposing alliance design matrix  [n_rows × n_teams]
    y: np.ndarray             # scoring alliance actual scores   [n_rows]
    y_opp: np.ndarray         # opposing alliance actual scores  [n_rows]
    weights: np.ndarray       # per-row weights (before quality adj) [n_rows]
    team_list: List[int]      # sorted list of team numbers
    row_to_match: List[Tuple[int, int]]  # (match_idx, side) for each row


def build_matrices(
    matches: List[MatchRecord],
    event_team_counts: Dict[str, int] | None = None,
    event_types: Dict[str, int] | None = None,
    now: float | None = None,
) -> MatrixBundle:
    """
    Build the sparse design matrices for a list of matches.

    Two rows per match:
      row 2i   → red alliance scored red_score; blue is the opponent
      row 2i+1 → blue alliance scored blue_score; red is the opponent

    Parameters
    ----------
    matches:
        Normalized match records (may span multiple events).
    event_team_counts:
        Optional dict {event_key: team_count} for event-size weighting.
    now:
        Reference timestamp for time-decay (defaults to current time).
    """
    if now is None:
        now = datetime.now().timestamp()

    all_teams: set = set()
    for m in matches:
        all_teams.update(m.red_teams)
        all_teams.update(m.blue_teams)

    team_list = sorted(all_teams)
    team_idx: Dict[int, int] = {t: i for i, t in enumerate(team_list)}
    n_teams = len(team_list)
    n_matches = len(matches)
    n_rows = 2 * n_matches

    rows_A: List[int] = []
    cols_A: List[int] = []
    y = np.zeros(n_rows)
    y_opp = np.zeros(n_rows)
    weights = np.zeros(n_rows)
    row_to_match: List[Tuple[int, int]] = []

    for mi, m in enumerate(matches):
        w_base = _row_weight(m, now, event_team_counts, event_types)

        for side in range(2):
            ri = 2 * mi + side
            if side == 0:
                scoring_teams, opp_teams = m.red_teams, m.blue_teams
                score, opp_score = m.red_score, m.blue_score
            else:
                scoring_teams, opp_teams = m.blue_teams, m.red_teams
                score, opp_score = m.blue_score, m.red_score

            for t in scoring_teams:
                if t in team_idx:
                    rows_A.append(ri)
                    cols_A.append(team_idx[t])

            # Apply match quality weight (set later by breaker detection)
            q = m.quality_weight if not m.is_excluded else 0.0
            y[ri] = score
            y_opp[ri] = opp_score
            weights[ri] = w_base * q
            row_to_match.append((mi, side))

    A = sp.csr_matrix(
        ([1.0] * len(rows_A), (rows_A, cols_A)), shape=(n_rows, n_teams)
    )
    A_opp = A

    return MatrixBundle(A, A_opp, y, y_opp, weights, team_list, row_to_match)


# ---------------------------------------------------------------------------
# Weight components
# ---------------------------------------------------------------------------

def _row_weight(
    m: MatchRecord,
    now: float,
    event_team_counts: Dict[str, int] | None,
    event_types: Dict[str, int] | None = None,
) -> float:
    w = _time_weight(m.timestamp, now)
    w *= STAGE_WEIGHTS.get(m.comp_level, 1.0)
    if event_team_counts and m.event_key in event_team_counts:
        w *= _event_size_weight(event_team_counts[m.event_key])
    if event_types and m.event_key in event_types:
        et = event_types[m.event_key]
        w *= EVENT_TYPE_WEIGHTS.get(et, 1.0)
    return w


def _time_weight(ts: float, now: float) -> float:
    if ts <= 0:
        return 0.5  # unknown timestamp → penalise slightly
    delta_days = (now - ts) / 86400.0
    # Clamp to 0: a match timestamped in the future gets weight 1.0, not >1.0
    delta_days = max(0.0, delta_days)
    return 2.0 ** (-delta_days / CONFIG.time_decay_half_life_days)


def _event_size_weight(team_count: int) -> float:
    w = math.sqrt(team_count / 40.0)
    return max(0.85, min(1.35, w))
