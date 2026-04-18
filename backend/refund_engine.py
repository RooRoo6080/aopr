from __future__ import annotations

import math
import logging
from typing import Dict, List, Set, Tuple

import numpy as np

from config import CONFIG
from match_normalizer import MatchRecord

logger = logging.getLogger(__name__)

# Diminishing-credit splits for 1/2/3 defenders on the same alliance
_DEFENDER_SPLITS: Dict[int, List[float]] = {
    1: [1.00],
    2: [0.80, 0.20],
    3: [0.70, 0.20, 0.10],
}


# ---------------------------------------------------------------------------
# Defender detection
# ---------------------------------------------------------------------------

def detect_defenders(
    team_list: List[int],
    opr: np.ndarray,
    dpr: np.ndarray,
    match_counts: Dict[int, int],
) -> Set[int]:
    """
    A team is a defender if:
      1. DPR > threshold_multiplier × OPR
      2. match_count >= min_matches_to_rank
      3. DPR > 10% of the event's max DPR  (avoids tiny-sample artefacts)
    """
    if len(dpr) == 0:
        return set()

    max_dpr = float(np.max(dpr)) if len(dpr) > 0 else 1.0
    significance_floor = 0.10 * max_dpr

    defenders: Set[int] = set()
    for i, team in enumerate(team_list):
        o = max(opr[i], 1e-6)
        d = dpr[i]
        if d < CONFIG.defender_threshold_multiplier * o:
            continue
        if match_counts.get(team, 0) < CONFIG.min_matches_to_rank:
            continue
        if d < significance_floor:
            continue
        defenders.add(team)
        logger.debug("Defender detected: team %d  OPR=%.2f  DPR=%.2f", team, o, d)

    return defenders


# ---------------------------------------------------------------------------
# Refund engine
# ---------------------------------------------------------------------------

def compute_refunds(
    matches: List[MatchRecord],
    residuals: np.ndarray,        # shape: (2 × n_matches,)
    opr: np.ndarray,
    dpr: np.ndarray,
    team_list: List[int],
    defenders: Set[int],
    weights: np.ndarray,
    event_avg_score: float,
) -> Tuple[np.ndarray, List[dict]]:
    """
    For each alliance row that has positive residual *and* defenders on the
    opposing side, compute a defensive refund that credits back suppressed offense.

    Returns
    -------
    refunds    : per-row refund amounts (shape: 2 × n_matches)
    audit_rows : list of audit dicts, one per refunded row
    """
    team_idx: Dict[int, int] = {t: i for i, t in enumerate(team_list)}
    n_rows = 2 * len(matches)
    refunds = np.zeros(n_rows)
    audit_rows: List[dict] = []

    avg_score = max(event_avg_score, 1e-6)

    for mi, m in enumerate(matches):
        for side in range(2):
            ri = 2 * mi + side

            if side == 0:
                scoring_teams = m.red_teams
                opposing_teams = m.blue_teams
                actual_score = m.red_score
                alliance_side = "red"
            else:
                scoring_teams = m.blue_teams
                opposing_teams = m.red_teams
                actual_score = m.blue_score
                alliance_side = "blue"

            # Which defenders are on the opposing side?
            active_defenders = [t for t in opposing_teams if t in defenders]
            if not active_defenders:
                continue

            residual = float(residuals[ri])
            if residual <= 0:
                continue  # alliance matched or exceeded expectation → no refund

            # Expected score for the scoring alliance
            opp_expected = sum(
                opr[team_idx[t]] for t in scoring_teams if t in team_idx
            )

            # Nonlinear strength multiplier: defense against a stronger team is worth more
            raw_mult = math.sqrt(max(opp_expected, 0.0) / avg_score)
            multiplier = max(0.85, min(1.25, raw_mult))

            # Sort defenders by OPR ascending: lowest OPR → most likely pure defender
            active_defenders.sort(
                key=lambda t: opr[team_idx[t]] if t in team_idx else 0.0
            )

            n_def = min(len(active_defenders), 3)
            splits = _DEFENDER_SPLITS[n_def][:n_def]
            split_sum = sum(splits)
            splits = [s / split_sum for s in splits]

            # Aggregate defensive credit across all active defenders.
            # Credit per defender = how much their DPR exceeds the threshold × OPR,
            # i.e. the suppression they provide *above* what their offensive role explains.
            total_credit = 0.0
            for def_team, split in zip(active_defenders[:n_def], splits):
                if def_team not in team_idx:
                    continue
                di = team_idx[def_team]
                # suppression surplus: DPR − (threshold × OPR)
                # positive only when the defender is genuinely above threshold
                surplus = dpr[di] - CONFIG.defender_threshold_multiplier * max(opr[di], 0.0)
                credit = (
                    max(0.0, surplus)
                    * split
                    * multiplier
                    * CONFIG.refund_credit_multiplier
                )
                total_credit += credit

            # Hard cap: refund ≤ positive residual
            refund = min(total_credit, residual)
            refunds[ri] = refund

            audit_rows.append(
                {
                    "match_key": m.match_key,
                    "event_key": m.event_key,
                    "alliance_side": alliance_side,
                    "expected_score": round(opp_expected, 3),
                    "actual_score": actual_score,
                    "residual": round(residual, 3),
                    "refund": round(refund, 3),
                    "defender_keys": [f"frc{t}" for t in active_defenders[:n_def]],
                    "row_weight": round(float(weights[ri]), 4),
                    "is_breaker": "breaker" in m.status_flags,
                    "is_excluded": m.is_excluded,
                }
            )

    logger.info(
        "Refund engine: %d refunded rows  total_refund=%.1f",
        len(audit_rows),
        float(refunds.sum()),
    )
    return refunds, audit_rows
