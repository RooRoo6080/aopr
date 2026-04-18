from __future__ import annotations

import logging
from typing import Tuple, Dict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adaptive damping
# ---------------------------------------------------------------------------

def _choose_damp(Aw: sp.spmatrix) -> float:
    """
    Estimate the condition number of the weighted design matrix and
    return an appropriate ridge-damping value for lsqr.
    """
    try:
        k = min(6, min(Aw.shape) - 1)
        if k < 1:
            return 0.05
        sv = spla.svds(Aw, k=k, return_singular_vectors=False)
        cond = float(sv.max()) / (float(sv.min()) + 1e-12)
        if cond < 1e6:
            return 0.0
        elif cond < 1e8:
            return 0.05
        else:
            return 0.2
    except Exception as exc:
        logger.debug("svds failed (%s); defaulting damp=0.05", exc)
        return 0.05


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def solve_weighted(
    A: sp.spmatrix,
    y: np.ndarray,
    weights: np.ndarray,
    damp: float | None = None,
) -> Tuple[np.ndarray, float, dict]:
    """
    Solve the weighted least-squares problem:
        min  ||W^½ (Ax − y)||²  +  damp² ||x||²

    Parameters
    ----------
    A       : design matrix [n_rows × n_teams]
    y       : target vector [n_rows]
    weights : per-row non-negative weights [n_rows]
    damp    : ridge damping (None → adaptive)

    Returns
    -------
    x       : solution vector [n_teams]
    damp_used : damping value actually applied
    info    : lsqr diagnostic dict
    """
    sqrt_w = np.sqrt(np.clip(weights, 0.0, None))

    # Zero-weight rows contribute nothing; sparse diagonal is fine
    W = sp.diags(sqrt_w, format="csr")
    Aw = W @ A
    yw = sqrt_w * y

    if damp is None:
        damp = _choose_damp(Aw)

    result = spla.lsqr(Aw, yw, damp=damp, iter_lim=50_000, atol=1e-10, btol=1e-10)
    x, istop, itn, r1norm = result[0], result[1], result[2], result[3]

    info = {
        "damp": damp,
        "istop": int(istop),
        "iterations": int(itn),
        "r1norm": float(r1norm),
        "n_teams": int(A.shape[1]),
        "n_rows": int(A.shape[0]),
    }
    logger.debug("lsqr solve: damp=%.4f istop=%d itn=%d r1norm=%.3f", damp, istop, itn, r1norm)
    return x, damp, info


# ---------------------------------------------------------------------------
# Convenience: solve OPR + DPR in one call
# ---------------------------------------------------------------------------

def solve_opr_dpr(
    A: sp.spmatrix,
    A_opp: sp.spmatrix,
    y: np.ndarray,
    y_opp: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns (opr_vector, dpr_vector, solver_info).
    Both solves share the same adaptive damp so diagnostics are comparable.
    """
    # Compute damp from OPR system (same conditioning for both)
    sqrt_w = np.sqrt(np.clip(weights, 0.0, None))
    W = sp.diags(sqrt_w, format="csr")
    Aw = W @ A
    damp = _choose_damp(Aw)

    opr, _, info_opr = solve_weighted(A, y, weights, damp=damp)
    
    y_pred = A.dot(opr)
    res = y - y_pred
    
    y_dpr = np.zeros_like(y)

    for i in range(0, len(y), 2):
        # Pure Points Suppressed: Target = Expected - Actual
        # Empirically the most resilient FRC defensive tracking metric
        y_dpr[i] = -res[i+1]
        y_dpr[i+1] = -res[i]

    dpr, _, info_dpr = solve_weighted(A, y_dpr, weights, damp=damp)

    return opr, dpr, {"opr": info_opr, "dpr": info_dpr}
