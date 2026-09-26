"""3D wireframe reconstruction utilities.

This module provides functions to build normalized landmark positions from
ratio tables and to assemble Bézier controls suitable for 3D reconstruction
and plotting. Public API highlights:

- `build_landmarks(row, BC30_scale)`: construct normalized landmark coords
    from a pandas Series containing either r_* ratios (relative to BC30) or
    C*B_over_AB ratios (relative to AB).
- `build_controls(lm, u_AB)`: derive control points for Bézier segments from
    normalized landmarks and a unit AB direction vector.
- `compute_2d_curves(ctrl, n)`: sample the Bézier segments for plotting.

Functions raise informative ValueError when input ratios would produce
geometrically invalid control points (e.g. invalid ordering or degeneracy).
"""

import math
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from core.math.geometry import rot_np, normalize_np, polygon_area_np, y_on_line_through_np
from core.math.bezier import bezier_quad_np, bezier_cubic_np
from core.io.data_loader import get_any

# 3D thickness model
def morteo_SP_T(C: float) -> Tuple[float, float]:
    """Return SP and T thickness model parameters for chord length C.

    The empirical linear relations were derived from morphometric fits and
    are used to parameterize section thickness models.
    """
    SP = 0.4357 * C - 1.8803
    T = 0.1191 * C + 0.5164
    return SP, T

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def thickness_piecewise(c: np.ndarray, C: float, SP: float, T: float) -> np.ndarray:
    """Piecewise thickness profile for normalized chord positions.

    c : array-like of chordwise coordinates
    C, SP, T : model parameters (see `morteo_SP_T`).
    Returns per-position thickness, clipped to zero.
    """
    c = np.asarray(c, float)
    out = np.zeros_like(c)

    maskL = c <= SP
    if np.any(maskL) and SP > 1e-9:
        x = c[maskL]
        out[maskL] = (-T * (x ** 2) / (SP ** 2)) + (2 * T * x / SP)

    denom = (C - SP)
    maskR = ~maskL
    if np.any(maskR) and denom > 1e-9:
        x = c[maskR]
        u = (C - x)
        out[maskR] = (-T * (u ** 2) / (denom ** 2)) + (2 * T * u / denom)

    out[out < 0] = 0.0
    return out

def invert_thickness_for_c(target: float, C: float, SP: float, T: float) -> Optional[Tuple[float, float]]:
    """Invert the thickness model to find chordwise locations corresponding
    to a target thickness value.

    Returns a tuple (cL, cR) with left and right chord abscissae, or None when
    inversion is impossible for the provided parameters.
    """
    if target < 1e-12:
        return 0.0, C
    if target > T - 1e-12:
        return SP, SP

    K = target * (SP ** 2) / max(T, 1e-12)
    disc = SP ** 2 - K
    if disc < 0:
        return None
    cL = SP - math.sqrt(max(0.0, disc))

    denom = (C - SP)
    if denom <= 1e-12:
        return None
    K2 = target * (denom ** 2) / max(T, 1e-12)
    disc2 = denom ** 2 - K2
    if disc2 < 0:
        return None
    u = denom - math.sqrt(max(0.0, disc2))
    cR = C - u
    return cL, cR

# Landmarks (normalized)
def build_landmarks(row: pd.Series, BC30_scale: float = 10.0) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Construct normalized 2D landmark coordinates from a ratios row.

    The function accepts either the r_* columns (ratios relative to BC30)
    or columns named like `C30B_over_AB` (ratios of C*B distances over AB).
    Returns a dict of numpy arrays for landmarks and the unit vector `u_AB`.
    """
    B = np.array([0.0, 0.0], float)
    C30 = np.array([-BC30_scale, 0.0], float)

    u_BC30 = (C30 - B) / np.linalg.norm(C30 - B)
    u_AB = rot_np(u_BC30, math.radians(-30.0))

    r_BA = get_any(row, ["r_BA", "BA_over_BC30"])
    r_BC5 = get_any(row, ["r_BC5", "BC5_over_BC30"])
    r_BC10 = get_any(row, ["r_BC10", "BC10_over_BC30"])
    r_BC20 = get_any(row, ["r_BC20", "BC20_over_BC30"])
    r_BC30 = get_any(row, ["r_BC30"])

    if (r_BA is not None) and (r_BC5 is not None) and (r_BC10 is not None) and (r_BC20 is not None):
        if r_BC30 is None:
            r_BC30 = 1.0
        if abs(r_BC30) < 1e-12:
            raise ValueError("r_BC30 is zero/invalid.")
        scale = BC30_scale / r_BC30
        BA = r_BA * scale
        BC5 = r_BC5 * scale
        BC10 = r_BC10 * scale
        BC20 = r_BC20 * scale
    else:
        c30b_over_ab = get_any(row, ["C30B_over_AB"])
        c5b_over_ab = get_any(row, ["C5B_over_AB"])
        c10b_over_ab = get_any(row, ["C10B_over_AB"])
        c20b_over_ab = get_any(row, ["C20B_over_AB"])
        if None in (c30b_over_ab, c5b_over_ab, c10b_over_ab, c20b_over_ab):
            raise ValueError("Missing required ratios. Provide either r_* columns or C*B_over_AB columns.")
        AB = BC30_scale / c30b_over_ab
        BA = AB
        BC5 = c5b_over_ab * AB
        BC10 = c10b_over_ab * AB
        BC20 = c20b_over_ab * AB

    A = B + BA * u_AB
    C5 = B + BC5 * rot_np(u_AB, math.radians(5.0))
    C10 = B + BC10 * rot_np(u_AB, math.radians(10.0))
    C20 = B + BC20 * rot_np(u_AB, math.radians(20.0))

    return {"B": B, "A": A, "C5": C5, "C10": C10, "C20": C20, "C30": C30}, u_AB

def _te2_control_through_C10_at_half(C5: np.ndarray, C10: np.ndarray, C20: np.ndarray) -> np.ndarray:
    return 2.0 * C10 - 0.5 * (C5 + C20)

def build_controls(lm: Dict[str, np.ndarray], u_AB: np.ndarray) -> Dict[str, np.ndarray]:
    """Derive Bézier control points for leading and trailing edges.

    The routine enforces geometric ordering constraints (e.g. TE1 ordering,
    TE2 tangent below AB, TE3 control inside the C20–C30 span) and raises
    ValueError when constraints cannot be satisfied. The returned dict is
    directly consumable by `compute_2d_curves`.
    """

    B, A, C5, C10, C20, C30 = lm["B"], lm["A"], lm["C5"], lm["C10"], lm["C20"], lm["C30"]
    BA = float(np.linalg.norm(A - B))
    n_left = rot_np(u_AB, math.radians(90.0))

    LE_P0 = B
    LE_P1 = B + u_AB * (0.27 * BA) + n_left * (0.08 * BA)
    LE_P2 = A + np.array([0.25 * BA, 0.06 * BA], float)
    if LE_P2[0] <= A[0]:
        raise ValueError("LE_P2 must lie to the right of A.")
    LE_P3 = A

    TE2_Q0 = C5
    TE2_Q2 = C20
    TE2_Q1 = _te2_control_through_C10_at_half(C5, C10, C20)
    if TE2_Q1[1] >= y_on_line_through_np(B, A, float(TE2_Q1[0])):
        raise ValueError("TE2_Q1 must be below AB.")

    dA = (A - LE_P2)
    dA /= (np.linalg.norm(dA) + 1e-12)
    if dA[1] >= -1e-9:
        raise ValueError("Leading-edge tangent at A not downward; cannot build TE1.")

    dir_C5 = (TE2_Q1 - C5)
    if np.linalg.norm(dir_C5) < 1e-12:
        raise ValueError("Degenerate TE2 tangent at C5.")

    L = 0.5 * float(np.linalg.norm(A - C5))
    alpha = L / float(np.linalg.norm(dir_C5))
    TE1_P2 = C5 - alpha * dir_C5

    kA = min(0.49, max(0.10, 0.18 * BA))
    margin_y = 0.06 * BA
    kA_max_from_y = (TE1_P2[1] + margin_y - A[1]) / dA[1]
    kA = min(kA, kA_max_from_y, 0.49)
    if kA <= 0.0:
        raise ValueError("No feasible TE1_P1 satisfying ordering and |A-TE1_P1|<0.5.")
    TE1_P1 = A + kA * dA

    if not (C5[1] + 1e-6 < TE1_P2[1] < TE1_P1[1] - 1e-6 < A[1] - 1e-6):
        raise ValueError("TE1 ordering violated (C5 < TE1_P2 < TE1_P1 < A).")

    d20 = (C20 - TE2_Q1)
    if np.linalg.norm(d20) < 1e-12:
        raise ValueError("Degenerate TE2 end tangent at C20.")
    d20 /= (np.linalg.norm(d20) + 1e-12)

    seg2030 = float(np.linalg.norm(C20 - C30))
    TE3_P1 = C20 + 0.5 * seg2030 * d20

    if TE3_P1[1] <= C30[1] + 1e-6:
        raise ValueError("TE3_P1 must be above C30.")
    if TE3_P1[1] >= A[1] - 1e-6:
        raise ValueError("TE3_P1 must be below A.")
    x_lo = min(C20[0], C30[0]) - 1e-6
    x_hi = max(C20[0], C30[0]) + 1e-6
    if not (x_lo <= TE3_P1[0] <= x_hi):
        # TE3_P1 fell outside the horizontal span of C20–C30. This can
        # occur with noisy or extreme ratio inputs. Instead of failing,
        # project TE3_P1 onto the C20–C30 segment (closest point) so the
        # control remains well-formed and downstream reconstruction can
        # proceed. If the segment is degenerate, preserve the original
        # failure behavior.
        seg_vec = (C30 - C20)
        denom = float(np.dot(seg_vec, seg_vec))
        if denom <= 1e-12:
            raise ValueError("TE3_P1 must lie within x-span of segment C20–C30.")
        tproj = float(np.dot(TE3_P1 - C20, seg_vec) / denom)
        tproj_clamped = max(0.0, min(1.0, tproj))
        TE3_P1 = C20 + tproj_clamped * seg_vec

    return {
        "LE_P0": LE_P0, "LE_P1": LE_P1, "LE_P2": LE_P2, "LE_P3": LE_P3,
        "TE1_P0": A, "TE1_P1": TE1_P1, "TE1_P2": TE1_P2, "TE1_P3": C5,
        "TE2_Q0": TE2_Q0, "TE2_Q1": TE2_Q1, "TE2_Q2": TE2_Q2,
        "TE3_Q0": C20, "TE3_P1": TE3_P1, "TE3_Q2": C30,
    }

def compute_2d_curves(ctrl: Dict[str, np.ndarray], n: int = 900) -> Dict[str, np.ndarray]:
    """Sample the provided Bézier control dictionary and return arrays.

    Returns a dict containing sampled arrays for each segment plus an overall
    `outline` array useful for plotting and area computations.
    """
    t = np.linspace(0.0, 1.0, n)
    leading = bezier_cubic_np(ctrl["LE_P0"], ctrl["LE_P1"], ctrl["LE_P2"], ctrl["LE_P3"], t)
    te1 = bezier_cubic_np(ctrl["TE1_P0"], ctrl["TE1_P1"], ctrl["TE1_P2"], ctrl["TE1_P3"], t)
    te2 = bezier_quad_np(ctrl["TE2_Q0"], ctrl["TE2_Q1"], ctrl["TE2_Q2"], t)
    te3 = bezier_quad_np(ctrl["TE3_Q0"], ctrl["TE3_P1"], ctrl["TE3_Q2"], t)
    trailing = np.vstack([te1, te2[1:], te3[1:]])
    outline = np.vstack([leading, trailing[1:]])
    return {"leading": leading, "te1": te1, "te2": te2, "te3": te3, "trailing": trailing, "outline": outline}

def fin_area_S(outline: np.ndarray, B: np.ndarray) -> float:
    return polygon_area_np(np.vstack([outline, B[None, :]]))
