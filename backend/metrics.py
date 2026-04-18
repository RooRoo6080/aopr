from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp

from config import CONFIG
from match_normalizer import MatchRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------

def compute_residuals(A: sp.spmatrix, opr: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    r[i] = (A @ opr)[i] - y[i]

    Positive residual  →  alliance underperformed vs expectation (candidate for refund)
    Negative residual  →  alliance overperformed
    """
    return np.asarray(A @ opr).ravel() - y


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

def compute_noise_sigma(residuals: np.ndarray) -> float:
    """
    Robust scale estimate using the Median Absolute Deviation.
    σ_noise = 1.4826 · MAD(r)
    """
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    sigma = 1.4826 * mad
    return max(sigma, 1e-6)  # guard against zero


# ---------------------------------------------------------------------------
# OPR-breaker detection
# ---------------------------------------------------------------------------

def detect_breakers(
    residuals: np.ndarray,
    sigma: float,
    threshold: float | None = None,
) -> np.ndarray:
    """
    Return a boolean mask of rows whose residual exceeds the breaker threshold.
    These rows are downweighted to 0.1 (not excluded outright).
    """
    if threshold is None:
        threshold = CONFIG.oppr_breaker_sigma
    return residuals > threshold * sigma


def detect_exclusions(
    residuals: np.ndarray,
    sigma: float,
    threshold: float | None = None,
) -> np.ndarray:
    """
    Return a boolean mask of rows whose residual exceeds the full-exclusion
    threshold (noise_exclusion_sigma, default 2.5).  These rows are zeroed
    from the solve entirely.  The exclusion gate is stricter than the breaker
    gate — only the most extreme anomalies are excluded.
    """
    if threshold is None:
        threshold = CONFIG.noise_exclusion_sigma
    return residuals > threshold * sigma


def apply_breaker_weights(
    matches: List[MatchRecord],
    row_to_match: List[Tuple[int, int]],
    breaker_mask: np.ndarray,
    exclusion_mask: np.ndarray | None = None,
    breaker_weight: float = 0.1,
) -> None:
    """
    Propagate anomaly status back to MatchRecord.quality_weight (in-place).

    Two tiers:
      exclusion_mask (2.5σ) → quality_weight = 0.0, is_excluded = True
      breaker_mask   (2.0σ) → quality_weight = 0.1, status_flag "breaker"
    """
    if exclusion_mask is None:
        exclusion_mask = np.zeros(len(breaker_mask), dtype=bool)

    for ri, (mi, _side) in enumerate(row_to_match):
        m = matches[mi]
        if exclusion_mask[ri]:
            m.is_excluded = True
            m.quality_weight = 0.0
        elif breaker_mask[ri]:
            if "breaker" not in m.status_flags:
                m.status_flags.append("breaker")
            m.quality_weight = min(m.quality_weight, breaker_weight)


# ---------------------------------------------------------------------------
# Per-team variability
# ---------------------------------------------------------------------------

def compute_variability(
    A: sp.spmatrix,
    opr: np.ndarray,
    residuals: np.ndarray,
    team_list: List[int],
) -> Dict[int, float]:
    """
    Variability for each team as a standard deviation around that team's OPR.

    For each alliance row a team appears in:
      expected_alliance = sum(team OPRs)
      actual_alliance   = expected_alliance - residual

    We estimate the team's row-level offensive contribution by allocating the
    alliance residual evenly across alliance members:
      observed_team_contrib ~= team_opr - residual / alliance_size

    The returned variability is the standard deviation of those observed
    per-row team contributions across the season.
    """
    var: Dict[int, float] = {}
    A_csc = A.tocsc()
    for col_idx, team in enumerate(team_list):
        col = A_csc.getcol(col_idx)
        row_indices = col.nonzero()[0]
        if len(row_indices) < 2:
            var[team] = 0.0
        else:
            team_opr = float(opr[col_idx])
            row_contribs = []
            for ri in row_indices:
                alliance_size = int(A.getrow(ri).nnz)
                if alliance_size <= 0:
                    continue
                row_contribs.append(team_opr - (float(residuals[ri]) / alliance_size))

            var[team] = float(np.std(row_contribs)) if len(row_contribs) >= 2 else 0.0
    return var


# ---------------------------------------------------------------------------
# Per-team match count
# ---------------------------------------------------------------------------

def compute_match_counts(
    A: sp.spmatrix,
    team_list: List[int],
) -> Dict[int, int]:
    """Number of alliance rows each team appears in (≈ matches played × 1 row per match)."""
    counts: Dict[int, int] = {}
    A_csc = A.tocsc()
    for col_idx, team in enumerate(team_list):
        col = A_csc.getcol(col_idx)
        # Each match contributes exactly one row per side for this team
        counts[team] = int(col.nnz)
    return counts
