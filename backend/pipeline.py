"""
pipeline.py

Orchestrates a full season-wide AOPR solve:
  1. Fetch all events for the current season
  2. Fetch and normalize all matches
  3. Build weighted sparse matrices
  4. Solve OPR + DPR
  5. Compute residuals, noise sigma, breakers
  6. Detect defenders
  7. Compute refunds → adjusted scores
  8. Re-solve for AOPR
  9. Store results + audit trail
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

import scipy.sparse as sp

from config import CONFIG
from tba_client import get_season_events, get_event_matches, get_event_teams, is_data_stale
from match_normalizer import normalize_matches, MatchRecord
from matrix_builder import build_matrices, MatrixBundle
from solver import _choose_damp, solve_opr_dpr, solve_weighted
from metrics import (
    compute_residuals,
    compute_noise_sigma,
    detect_breakers,
    detect_exclusions,
    apply_breaker_weights,
    compute_variability,
    compute_match_counts,
)
from refund_engine import detect_defenders, compute_refunds
from cache import (
    save_audit_records,
    save_solver_snapshot,
    get_latest_snapshot,
)

logger = logging.getLogger(__name__)

# Module-level result cache so the API can serve instantly while refresh runs
_last_results: Optional[Dict[str, Any]] = None
_last_solve_time: float = 0.0


def _safe(v: Any, decimals: int = 3) -> float:
    """Return a finite float rounded to `decimals`, or 0.0 for NaN/Inf/None."""
    try:
        f = float(v)
        return round(f, decimals) if math.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def run_season_pipeline() -> Dict[str, Any]:
    """
    Execute the full pipeline for the current season.
    Returns a results dict keyed by team number.
    """
    global _last_results, _last_solve_time
    t0 = time.perf_counter()
    year = CONFIG.season_year

    logger.info("=== Starting AOPR pipeline for %d ===", year)

    # --- 1. Fetch events ---------------------------------------------------
    events = await get_season_events(year)
    logger.info("Fetched %d events for %d", len(events), year)

    # --- 2. Fetch + normalize matches per event (concurrent) ---------------
    all_matches: List[MatchRecord] = []
    event_team_counts: Dict[str, int] = {}
    event_types: Dict[str, int] = {}          # event_key → TBA event_type int
    event_meta: Dict[str, dict] = {}
    event_membership: Dict[str, List[int]] = {}  # event_key → [team_numbers]
    team_nicknames: Dict[int, str] = {}           # team_number → nickname

    async def _fetch_event(ev: dict) -> None:
        """Fetch matches + teams for one event and merge results in-place."""
        ek = ev.get("key", "")
        if not ek:
            return
        try:
            raw_matches, teams = await asyncio.gather(
                get_event_matches(ek),
                get_event_teams(ek),
            )

            normalized = normalize_matches(raw_matches, ek)
            # Accumulate into shared structures — list.extend is thread-safe in CPython
            # but we do it from a single event loop so no locking needed.
            all_matches.extend(normalized)

            team_nums: List[int] = []
            for t in teams:
                try:
                    tn = int(t.get("key", "frc0").replace("frc", ""))
                    team_nums.append(tn)
                    if tn not in team_nicknames:
                        nick = (t.get("nickname") or "").strip()
                        if nick:
                            team_nicknames[tn] = nick
                except (ValueError, AttributeError):
                    pass

            et = ev.get("event_type")
            event_team_counts[ek] = len(teams)
            event_membership[ek] = team_nums
            if et is not None:
                event_types[ek] = int(et)
            event_meta[ek] = {
                "name": ev.get("name", ek),
                "week": ev.get("week"),
                "event_type": et,
                "district_key": (ev.get("district") or {}).get("key"),
                "team_count": len(teams),
            }
        except Exception as exc:
            logger.warning("Skipping event %s: %s", ek, exc)

    # Fetch all events concurrently (semaphore in tba_client caps to 10 in-flight)
    await asyncio.gather(*[_fetch_event(ev) for ev in events])

    logger.info("Total matches normalized: %d", len(all_matches))
    if not all_matches:
        logger.warning("No matches available; aborting pipeline")
        return {}

    # --- 3. Build matrices (initial pass, quality_weight=1.0 everywhere) --
    bundle = build_matrices(all_matches, event_team_counts, event_types)
    team_list = bundle.team_list
    n_teams = len(team_list)

    # --- 4. Baseline OPR + DPR solve --------------------------------------
    opr, dpr, solver_info = solve_opr_dpr(
        bundle.A, bundle.A_opp, bundle.y, bundle.y_opp, bundle.weights
    )

    # --- 5. Residuals, noise sigma, two-tier anomaly detection --------------
    residuals = compute_residuals(bundle.A, opr, bundle.y)
    sigma = compute_noise_sigma(residuals)
    breaker_mask   = detect_breakers(residuals, sigma)    # 2.0σ → downweight 0.1
    exclusion_mask = detect_exclusions(residuals, sigma)  # 2.5σ → exclude entirely
    apply_breaker_weights(all_matches, bundle.row_to_match, breaker_mask, exclusion_mask)

    # --- 6. Rebuild matrices with breaker weights applied -----------------
    bundle = build_matrices(all_matches, event_team_counts, event_types)

    # --- 7. Re-solve baseline with updated weights ------------------------
    opr, dpr, solver_info = solve_opr_dpr(
        bundle.A, bundle.A_opp, bundle.y, bundle.y_opp, bundle.weights
    )
    residuals = compute_residuals(bundle.A, opr, bundle.y)
    sigma = compute_noise_sigma(residuals)

    # --- 8. Defender detection --------------------------------------------
    match_counts = compute_match_counts(bundle.A, team_list)
    defenders: Set[int] = detect_defenders(team_list, opr, dpr, match_counts)

    # --- 9. Refund computation --------------------------------------------
    scored_y = bundle.y[bundle.weights > 0]
    avg_score = float(np.mean(scored_y)) if len(scored_y) > 0 else 0.0
    refunds, audit_rows = compute_refunds(
        all_matches, residuals, opr, dpr, team_list, defenders, bundle.weights, avg_score
    )

    # --- 10. Direct Proportional AOPR Allocation --------------------------
    # Instead of doing a global LSQR matrix re-solve (which smears the refund across
    # innocent/broken robots on the same alliance), we proportionally assign the 
    # refunded match points to the alliance's primary offensive threats (Strikers).
    team_refund_totals = np.zeros_like(opr)
    for ri, (mi, side) in enumerate(bundle.row_to_match):
        r_val = refunds[ri]
        if r_val > 0:
            m = all_matches[mi]
            scoring_teams = m.red_teams if side == 0 else m.blue_teams
            valid_indices = [team_idx[t] for t in scoring_teams if t in team_idx]
            
            # Distribute refund proportionally by each team's positive OPR
            oprs = [max(opr[ti], 0.0) for ti in valid_indices]
            total_opr = sum(oprs)
            if total_opr > 0:
                for ti, o in zip(valid_indices, oprs):
                    team_refund_totals[ti] += r_val * (o / total_opr)
            else:
                # If all zero, distribute equally
                for ti in valid_indices:
                    team_refund_totals[ti] += r_val / len(valid_indices)

    # Convert total seasonal refunded points into a per-match AOPR boost
    match_counts_arr = np.array([match_counts.get(t, 0) for t in team_list])
    aopr = opr + (team_refund_totals / np.maximum(match_counts_arr, 1.0))

    # --- 11. Variability --------------------------------------------------
    variability = compute_variability(bundle.A, residuals, team_list)

    # Sanitise raw solver arrays
    opr  = np.where(np.isfinite(opr),  opr,  0.0)
    dpr  = np.where(np.isfinite(dpr),  dpr,  0.0)
    aopr = np.where(np.isfinite(aopr), aopr, 0.0)

    # --- 12. Assemble per-team results + Synergy + Roles ------------------
    results: Dict[int, Dict[str, Any]] = {}
    
    # Collect synergy (average positive residual)
    team_positive_residuals = {t: [] for t in team_list}
    team_breaker_counts: Dict[int, int] = {t: 0 for t in team_list}
    
    for ri, (mi, side) in enumerate(bundle.row_to_match):
        res_val = residuals[ri]
        m = all_matches[mi]
        scoring = m.red_teams if side == 0 else m.blue_teams
        for t in scoring:
            if res_val > 0:
                if t in team_positive_residuals:
                    team_positive_residuals[t].append(res_val)
                    
        if "breaker" in m.status_flags or m.is_excluded:
            for t in scoring:
                if t in team_breaker_counts:
                    team_breaker_counts[t] += 1
                    
    # Pre-calculate role thresholds
    opr_valid = opr[match_counts_arr >= CONFIG.min_matches_to_rank]
    if len(opr_valid) > 0:
        opr_mean = float(np.mean(opr_valid))
        opr_p75 = float(np.percentile(opr_valid, 75))
        opr_p25 = float(np.percentile(opr_valid, 25))
    else:
        opr_mean = opr_p75 = opr_p25 = 0.0
        
    var_vals = [variability.get(t, 0) for t in team_list if match_counts.get(t, 0) >= CONFIG.min_matches_to_rank]
    var_p25 = float(np.percentile(var_vals, 25)) if var_vals else 0.0

    for i, team in enumerate(team_list):
        mc = match_counts.get(team, 0)
        pos_res_list = team_positive_residuals.get(team, [])
        
        # 1. Basic Synergy (Average Positive Residual)
        synergy = sum(pos_res_list) / len(pos_res_list) if pos_res_list else 0.0
        
        # 2. Assign Algorithmic Roles
        role = ""
        if team in defenders:
            role = "Defender"
        elif synergy > 3.0 and opr[i] < opr_mean:
            role = "Enabler"
        elif opr[i] > opr_p75 and variability.get(team, 0.0) < var_p25:
            role = "Anchor"
        elif opr[i] > opr_p75:
            role = "Striker"
        elif opr[i] < opr_p25:
            role = "Support"

        results[team] = {
            "team_number": team,
            "nickname":    team_nicknames.get(team, ""),
            "opr":         _safe(opr[i]),
            "dpr":         _safe(dpr[i]),
            "synergy":     _safe(synergy),
            "aopr":        _safe(aopr[i]),
            "delta":       _safe(float(aopr[i]) - float(opr[i])),
            "variability": _safe(variability.get(team, 0.0)),
            "match_count": mc,
            "breaker_count": team_breaker_counts.get(team, 0),
            "primary_role": role,
            "low_match_warning": mc < CONFIG.min_matches_to_rank,
        }

    # --- 13. Build full match-row index (in-memory only, not persisted) ------
    #   Two rows per match. Each row records everything needed for team history.
    team_idx: Dict[int, int] = {t: i for i, t in enumerate(team_list)}
    aopr_arr = np.asarray(aopr)
    opr_arr  = np.asarray(opr)

    # Build a dict: team_number → [match row dicts], sorted by timestamp
    team_match_rows: Dict[int, List[Dict[str, Any]]] = {t: [] for t in team_list}

    for mi, m in enumerate(all_matches):
        for side in range(2):
            ri = 2 * mi + side
            if side == 0:
                scoring, opposing = m.red_teams, m.blue_teams
                actual_score = m.red_score
                side_label = "red"
            else:
                scoring, opposing = m.blue_teams, m.red_teams
                actual_score = m.blue_score
                side_label = "blue"

            expected_score = float(sum(
                opr_arr[team_idx[t]] for t in scoring if t in team_idx
            ))
            adj_score = float(expected_score + refunds[ri])
            row_residual = float(residuals[ri])

            # defender keys on opposing side
            def_keys = [f"frc{t}" for t in opposing if t in defenders]

            row = {
                "match_key":    m.match_key,
                "event_key":    m.event_key,
                "event_name":   event_meta.get(m.event_key, {}).get("name", m.event_key),
                "comp_level":   m.comp_level,
                "timestamp":    m.timestamp,
                "is_playoff":   m.is_playoff,
                "side":         side_label,
                "alliance_teams": scoring,
                "opponent_teams": opposing,
                "actual_score":   actual_score,
                "expected_score": _safe(expected_score, 1),
                "residual":       _safe(row_residual, 1),
                "refund":         _safe(float(refunds[ri]), 1),
                "adj_score":      _safe(adj_score, 1),
                "row_weight":     _safe(float(bundle.weights[ri]), 4),
                "is_breaker":     "breaker" in m.status_flags,
                "is_excluded":    m.is_excluded,
                "defender_keys":  def_keys,
            }

            for t in scoring:
                if t in team_match_rows:
                    team_match_rows[t].append(row)

    # Sort each team's rows by timestamp ascending
    for t in team_match_rows:
        team_match_rows[t].sort(key=lambda r: r["timestamp"])

    # --- 13.5 Compute Event OPRs ----------------------------------------------
    event_oprs: Dict[str, Dict[int, float]] = {}
    for ek in event_meta.keys():
        ev_matches = [m for m in all_matches if m.event_key == ek]
        if not ev_matches:
            continue
        try:
            eb = build_matrices(ev_matches)
            e_opr, _, _ = solve_weighted(eb.A, eb.y, eb.weights)
            e_dict = {}
            for ti, team in enumerate(eb.team_list):
                e_dict[team] = _safe(e_opr[ti])
            event_oprs[ek] = e_dict
        except Exception as exc:
            logger.warning("Failed event OPR for %s: %s", ek, exc)

    # --- 14. Persist audit + snapshot; keep match rows in memory only ---------
    save_audit_records(audit_rows)
    snapshot_payload = {
        "team_results": results,
        "event_oprs": event_oprs,
        "meta": {
            "year": year,
            "total_matches": len(all_matches),
            "total_events": len(event_meta),
            "total_teams": n_teams,
            "noise_sigma": _safe(sigma, 3),
            "avg_score":   _safe(avg_score, 3),
            "defender_count": len(defenders),
            "solver_info": solver_info,
            "aopr_solver_info": solver_info,
            "solve_timestamp": datetime.now().timestamp(),
            "event_meta": event_meta,
        },
        "audit": audit_rows,
        "event_membership": event_membership,
        # team_match_rows is NOT serialised to SQLite (too large, rebuilt each run)
    }
    save_solver_snapshot(year, snapshot_payload)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Pipeline complete: %d teams  %d matches  %.2fs  sigma=%.2f",
        n_teams, len(all_matches), elapsed, sigma,
    )

    _last_results = snapshot_payload
    # Attach match rows to in-memory copy ONLY (not in the SQLite snapshot)
    _last_results["team_match_rows"] = team_match_rows
    _last_solve_time = datetime.now().timestamp()
    return _last_results


def get_cached_results() -> Optional[Dict[str, Any]]:
    """Return in-memory results if available, else load from DB."""
    global _last_results
    if _last_results:
        return _last_results
    snap = get_latest_snapshot(CONFIG.season_year)
    if snap:
        _last_results = _coerce_keys(snap["results"])
        return _last_results
    return None


def _coerce_keys(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    JSON serialisation turns integer dict keys into strings.
    Re-coerce team_results, event_membership, and event_oprs back to int keys.
    """
    if "team_results" in results:
        results["team_results"] = {
            int(k): v for k, v in results["team_results"].items()
        }
    if "event_membership" in results:
        results["event_membership"] = {
            k: [int(t) for t in v]
            for k, v in results["event_membership"].items()
        }
    if "event_oprs" in results:
        results["event_oprs"] = {
            ek: {int(t): opr for t, opr in val_dict.items()}
            for ek, val_dict in results["event_oprs"].items()
        }
    return results
