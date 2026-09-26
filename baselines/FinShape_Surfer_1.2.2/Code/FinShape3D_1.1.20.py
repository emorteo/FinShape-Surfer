#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinShape3D_1.1.20

Fixes vs 1.1.12
---------------
1) Denser 3D wireframe defaults:
   The 3D wireframe now uses at least 30% more default section/detail points: y-slices 30→39,
   chord points 90→117, and z-slices 15→20. Existing CLI options can still override them.

2) Plotting initialization fix:
   plot_2d now initializes Matplotlib before calling pyplot functions. Version 1.1.11
   initialized Matplotlib for 3D plotting and montages but missed the 2D plotting path,
   which caused AttributeError: 'NoneType' object has no attribute 'subplots'.

3) Exact output-directory reporting:
   The program now prints the resolved input path and exact output directory before processing,
   and prints the exact output directory again at completion.

4) Matplotlib cache-directory robustness:
   If the user's default Matplotlib configuration directory is not writable, plotting uses a
   writable temporary FinShape cache directory to avoid noisy startup warnings.

5) TE3 control-point robustness:
   Some ratio rows produce a C20 outgoing tangent whose natural TE3 control point falls outside
   the C20-C30 x-span. Version 1.1.9 raised ValueError and stopped the batch. This version
   keeps the output auditable by clamping the TE3 control point into the valid x-span, reporting
   a warning, and continuing the run. The endpoint landmarks and input ratios are not changed.

6) Landmark label placement robustness:
   Some fins (e.g., FIN_001_TB, FIN_004_BB) had C5 and/or C20 labels displaced because the
   previous algorithm used a simple centroid ray to decide the "outside" direction. For
   non-convex polygons, the vertex-mean centroid may lie outside the polygon or yield a
   direction that is not locally outward at boundary points.

   This version places LANDMARK labels using a *local outward normal* computed from the
   contour tangent near the landmark, verified by point-in-polygon tests. This makes the
   "outside" placement consistent and near the landmark.

   Control-point labels are still forced INSIDE the contour, but now use a robust interior
   point (guaranteed inside polygon) and point-in-polygon stepping.

7) Still PyCharm / Python 3.12 safe:
   - matplotlib backend forced to Agg when plotting starts.
   - argparse uses parse_known_args() and --input/--outdir are optional.
   - --help and non-plotting code paths no longer require Matplotlib to import successfully.
   - If --input missing, program FIRST asks user to choose CSV/XLS/XLSX (GUI picker if available).

8) Tip-complete wireframe rendering:
   The 3D plot now appends a zero-width, zero-thickness terminal section at the measured
   outline apex so chordwise rails converge to the fin tip instead of stopping at the last
   non-degenerate horizontal slice.

9) Readable contour-referenced wireframe:
   The renderer now samples 100 points along the measured contour edge, preserves curve
   endpoints/landmark-derived edge anchors, and uses those references to choose section levels.
   Chordwise rails are capped at a readable count so the plot remains a wireframe instead of
   becoming a visually solid blue surface.

10) Documented structural visualization floor:
   The renderer can impose a minimum displayed edge thickness and tip chord to prevent
   visual collapse near the apex. This is a visualization assumption, not a measured
   biological thickness and not CFD validation.

11) Reusable 3D axes renderer:
   The wireframe renderer can now draw onto an existing Matplotlib 3D axes. The saved PNG
   and the integrated GUI's rotatable live view use the same drawing path.

Dependencies (Python 3.12 / macOS / PyCharm):
  pip install numpy pandas matplotlib openpyxl
"""
from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

plt = None
mpimg = None
MplPath = None


def ensure_matplotlib() -> None:
    global plt, mpimg, MplPath
    if plt is not None and mpimg is not None and MplPath is not None:
        return
    if "MPLCONFIGDIR" not in os.environ:
        default_cfg = Path.home() / ".matplotlib"
        if (not default_cfg.exists()) or (not os.access(default_cfg, os.W_OK)):
            fallback_cfg = Path(tempfile.gettempdir()) / "finshape_matplotlib_config"
            fallback_cfg.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(fallback_cfg)
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as pyplot
        import matplotlib.image as mpl_image
        from matplotlib.path import Path as mpl_path
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "FinShape3D plotting requires matplotlib. Install it with: "
            "pip install matplotlib"
        ) from exc
    plt = pyplot
    mpimg = mpl_image
    MplPath = mpl_path


VERSION = "1.1.20"

# Global formatting multipliers
FONT_SCALE = 3.0
LEGEND_SCALE = 2.0

# Landmark/control label scaling (relative to 1.1.7 rule set)
LM_LABEL_MULT = 0.5
CTRL_LABEL_MULT = 0.5

FS_POINT = 10
FS_CTRL = 9
FS_AXLAB = 10
FS_LEGEND = 9

# Headers not scaled
FS_TITLE_2D = 12
FS_TITLE_3D = 11
FS_MONT_TILE = 9
FS_MONT_TITLE = 14

# Structural visualization assumptions, in AB=10 coordinate units.
# These prevent the plotted wireframe from collapsing to an invisible zero-thickness tip.
DEFAULT_EDGE_MIN_THICKNESS = 0.05
DEFAULT_TIP_MIN_CHORD = 0.05


# -----------------------------
# FIRST STEP: ask for input file
# -----------------------------
def select_input_file_interactive() -> Path:
    """Ask the user to select a ratios table (CSV/XLS/XLSX)."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass

        file_path = filedialog.askopenfilename(
            title="Select ratios table (CSV/XLS/XLSX)",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if file_path:
            pth = Path(file_path).expanduser().resolve()
            if pth.exists():
                return pth
    except Exception:
        pass

    while True:
        raw = input("Enter path to ratios CSV/XLS/XLSX (or press Enter to cancel): ").strip().strip('"').strip("'")
        if not raw:
            raise SystemExit("No input file selected. Exiting.")
        pth = Path(raw).expanduser().resolve()
        if pth.exists():
            return pth
        print(f"File not found: {pth}")


def default_outdir_for_input(input_path: Path) -> Path:
    return input_path.parent / f"{input_path.stem}_FinShape3D_outputs_v{VERSION}"


# -----------------------------
# Geometry utilities
# -----------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def rot(v: np.ndarray, ang_rad: float) -> np.ndarray:
    c, s = math.cos(ang_rad), math.sin(ang_rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=float)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0], float)
    return v / n


def bezier_quad(q0: np.ndarray, q1: np.ndarray, q2: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return ((1 - t) ** 2)[:, None] * q0 + (2 * (1 - t) * t)[:, None] * q1 + (t ** 2)[:, None] * q2


def bezier_cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return ((1 - t) ** 3)[:, None] * p0 + (3 * (1 - t) ** 2 * t)[:, None] * p1 + (3 * (1 - t) * t ** 2)[:, None] * p2 + (t ** 3)[:, None] * p3


def y_on_line_through(B: np.ndarray, A: np.ndarray, x: float) -> float:
    if abs(A[0] - B[0]) < 1e-12:
        return float("inf")
    m = (A[1] - B[1]) / (A[0] - B[0])
    return float(B[1] + m * (x - B[0]))


def polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def x_intersections_with_horizontal(polyline: np.ndarray, yval: float) -> List[float]:
    xs: List[float] = []
    for i in range(len(polyline) - 1):
        x0, y0 = float(polyline[i, 0]), float(polyline[i, 1])
        x1, y1 = float(polyline[i + 1, 0]), float(polyline[i + 1, 1])
        if abs(y1 - y0) < 1e-12:
            continue
        if (y0 - yval) * (y1 - yval) <= 0:
            t = (yval - y0) / (y1 - y0)
            if 0.0 <= t <= 1.0:
                xs.append(x0 + t * (x1 - x0))
    return xs


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _get_any(row: pd.Series, keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in row.index and pd.notna(row[k]):
            return float(row[k])
    return None


# -----------------------------
# Landmarks (normalized)
# -----------------------------
def build_landmarks(row: pd.Series, BC30_scale: float = 10.0) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    B = np.array([0.0, 0.0], float)
    C30 = np.array([-BC30_scale, 0.0], float)

    u_BC30 = (C30 - B) / np.linalg.norm(C30 - B)
    u_AB = rot(u_BC30, math.radians(-30.0))

    r_BA = _get_any(row, ["r_BA", "BA_over_BC30"])
    r_BC5 = _get_any(row, ["r_BC5", "BC5_over_BC30"])
    r_BC10 = _get_any(row, ["r_BC10", "BC10_over_BC30"])
    r_BC20 = _get_any(row, ["r_BC20", "BC20_over_BC30"])
    r_BC30 = _get_any(row, ["r_BC30"])

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
        c30b_over_ab = _get_any(row, ["C30B_over_AB"])
        c5b_over_ab = _get_any(row, ["C5B_over_AB"])
        c10b_over_ab = _get_any(row, ["C10B_over_AB"])
        c20b_over_ab = _get_any(row, ["C20B_over_AB"])
        if None in (c30b_over_ab, c5b_over_ab, c10b_over_ab, c20b_over_ab):
            raise ValueError("Missing required ratios. Provide either r_* columns or C*B_over_AB columns.")
        AB = BC30_scale / c30b_over_ab
        BA = AB
        BC5 = c5b_over_ab * AB
        BC10 = c10b_over_ab * AB
        BC20 = c20b_over_ab * AB

    A = B + BA * u_AB
    C5 = B + BC5 * rot(u_AB, math.radians(5.0))
    C10 = B + BC10 * rot(u_AB, math.radians(10.0))
    C20 = B + BC20 * rot(u_AB, math.radians(20.0))

    return {"B": B, "A": A, "C5": C5, "C10": C10, "C20": C20, "C30": C30}, u_AB


# -----------------------------
# Controls
# -----------------------------
def _te2_control_through_C10_at_half(C5: np.ndarray, C10: np.ndarray, C20: np.ndarray) -> np.ndarray:
    return 2.0 * C10 - 0.5 * (C5 + C20)


def build_controls(lm: Dict[str, np.ndarray], u_AB: np.ndarray, warnings: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    B, A, C5, C10, C20, C30 = lm["B"], lm["A"], lm["C5"], lm["C10"], lm["C20"], lm["C30"]
    BA = float(np.linalg.norm(A - B))
    n_left = rot(u_AB, math.radians(90.0))

    LE_P0 = B
    LE_P1 = B + u_AB * (0.27 * BA) + n_left * (0.08 * BA)
    LE_P2 = A + np.array([0.25 * BA, 0.06 * BA], float)
    if LE_P2[0] <= A[0]:
        raise ValueError("LE_P2 must lie to the right of A.")
    LE_P3 = A

    TE2_Q0 = C5
    TE2_Q2 = C20
    TE2_Q1 = _te2_control_through_C10_at_half(C5, C10, C20)
    if TE2_Q1[1] >= y_on_line_through(B, A, float(TE2_Q1[0])):
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

    x_lo_raw = min(float(C20[0]), float(C30[0]))
    x_hi_raw = max(float(C20[0]), float(C30[0]))
    if not (x_lo_raw <= float(TE3_P1[0]) <= x_hi_raw):
        original_x = float(TE3_P1[0])
        TE3_P1 = TE3_P1.copy()
        TE3_P1[0] = clamp(original_x, x_lo_raw, x_hi_raw)
        if warnings is not None:
            warnings.append(
                "TE3_P1 x-coordinate adjusted from "
                f"{original_x:.6g} to {float(TE3_P1[0]):.6g} to keep the C20-C30 segment bounded; "
                "TE2-to-TE3 tangent continuity is relaxed for this row."
            )

    return {
        "LE_P0": LE_P0, "LE_P1": LE_P1, "LE_P2": LE_P2, "LE_P3": LE_P3,
        "TE1_P0": A, "TE1_P1": TE1_P1, "TE1_P2": TE1_P2, "TE1_P3": C5,
        "TE2_Q0": TE2_Q0, "TE2_Q1": TE2_Q1, "TE2_Q2": TE2_Q2,
        "TE3_Q0": C20, "TE3_P1": TE3_P1, "TE3_Q2": C30,
    }


def compute_2d_curves(ctrl: Dict[str, np.ndarray], n: int = 900) -> Dict[str, np.ndarray]:
    t = np.linspace(0.0, 1.0, n)
    leading = bezier_cubic(ctrl["LE_P0"], ctrl["LE_P1"], ctrl["LE_P2"], ctrl["LE_P3"], t)
    te1 = bezier_cubic(ctrl["TE1_P0"], ctrl["TE1_P1"], ctrl["TE1_P2"], ctrl["TE1_P3"], t)
    te2 = bezier_quad(ctrl["TE2_Q0"], ctrl["TE2_Q1"], ctrl["TE2_Q2"], t)
    te3 = bezier_quad(ctrl["TE3_Q0"], ctrl["TE3_P1"], ctrl["TE3_Q2"], t)
    trailing = np.vstack([te1, te2[1:], te3[1:]])
    outline = np.vstack([leading, trailing[1:]])
    return {"leading": leading, "te1": te1, "te2": te2, "te3": te3, "trailing": trailing, "outline": outline}


def fin_area_S(outline: np.ndarray, B: np.ndarray) -> float:
    return polygon_area(np.vstack([outline, B[None, :]]))


# -----------------------------
# 3D thickness model
# -----------------------------
def morteo_SP_T(C: float) -> Tuple[float, float]:
    SP = 0.4357 * C - 1.8803
    T = 0.1191 * C + 0.5164
    return SP, T


def thickness_piecewise(c: np.ndarray, C: float, SP: float, T: float) -> np.ndarray:
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


def set_equal_3d_aspect(ax, xlim, ylim, zlim) -> None:
    xr = abs(xlim[1] - xlim[0]) or 1.0
    yr = abs(ylim[1] - ylim[0]) or 1.0
    zr = abs(zlim[1] - zlim[0]) or 1.0
    ax.set_box_aspect((xr, yr, zr))


# -----------------------------
# Robust label placement
# -----------------------------
def find_interior_point(poly_path, poly: np.ndarray) -> np.ndarray:
    """Return a point guaranteed to be inside polygon (best-effort)."""
    # Try the mean of vertices first
    cand = np.mean(poly, axis=0)
    if poly_path.contains_point((float(cand[0]), float(cand[1]))):
        return cand

    # Try mean of outline (excluding B) if available
    cand2 = np.mean(poly[:-1], axis=0)
    if poly_path.contains_point((float(cand2[0]), float(cand2[1]))):
        return cand2

    # Deterministic pseudo-random sampling inside bbox
    xmin, xmax = float(np.min(poly[:, 0])), float(np.max(poly[:, 0]))
    ymin, ymax = float(np.min(poly[:, 1])), float(np.max(poly[:, 1]))
    rng = np.random.default_rng(12345)
    for _ in range(500):
        x = float(rng.uniform(xmin, xmax))
        y = float(rng.uniform(ymin, ymax))
        if poly_path.contains_point((x, y)):
            return np.array([x, y], float)

    # Fallback: return cand even if outside (should be rare)
    return cand


def nearest_index(polyline: np.ndarray, pt: np.ndarray) -> int:
    d2 = np.sum((polyline - pt[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))


def local_outward_normal(poly_path: MplPath, polyline: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """Compute outward unit normal near pt using local polyline tangent and point-in-polygon test."""
    i = nearest_index(polyline, pt)
    n = len(polyline)
    im1 = (i - 1) % n
    ip1 = (i + 1) % n
    tangent = _normalize(polyline[ip1] - polyline[im1])
    normal = _normalize(np.array([-tangent[1], tangent[0]], float))

    eps = 0.05
    test = pt + eps * normal
    if poly_path.contains_point((float(test[0]), float(test[1]))):
        normal = -normal  # flip: we want outward
    return normal


def repel_from_taken(pos: np.ndarray, taken: List[np.ndarray], step: float = 0.20,
                     min_dist: float = 0.55, max_iter: int = 40) -> np.ndarray:
    """Simple label collision avoidance: nudge until far from prior labels."""
    p = pos.copy()
    for _ in range(max_iter):
        ok = True
        for q in taken:
            if float(np.linalg.norm(p - q)) < min_dist:
                ok = False
                # nudge perpendicular-ish
                d = _normalize(p - q)
                perp = np.array([-d[1], d[0]], float)
                p = p + step * perp
                break
        if ok:
            return p
    return p


def place_landmark_label(poly_path: MplPath, polyline: np.ndarray, pt: np.ndarray,
                         taken: List[np.ndarray]) -> np.ndarray:
    n_out = local_outward_normal(poly_path, polyline, pt)
    # start close to point and expand outward until outside
    offset = 0.35
    for _ in range(35):
        cand = pt + offset * n_out
        if not poly_path.contains_point((float(cand[0]), float(cand[1]))):
            cand = repel_from_taken(cand, taken)
            taken.append(cand)
            return cand
        offset += 0.15
    cand = pt + 1.0 * n_out
    cand = repel_from_taken(cand, taken)
    taken.append(cand)
    return cand


def place_control_label(poly_path: MplPath, interior: np.ndarray, pt: np.ndarray,
                        taken: List[np.ndarray]) -> np.ndarray:
    d_in = _normalize(interior - pt)
    offset = 0.15
    for _ in range(60):
        cand = pt + offset * d_in
        if poly_path.contains_point((float(cand[0]), float(cand[1]))):
            cand = repel_from_taken(cand, taken, min_dist=0.50)
            taken.append(cand)
            return cand
        offset += 0.12
    # fallback: try interior itself
    cand = interior.copy()
    cand = repel_from_taken(cand, taken, min_dist=0.50)
    taken.append(cand)
    return cand


# -----------------------------
# Plotting
# -----------------------------
def plot_2d(curves2d: Dict[str, np.ndarray], lm: Dict[str, np.ndarray], ctrl: Dict[str, np.ndarray],
            fin_id: str, out_png: Path, legend_anchor_xy: Tuple[float, float] = (-4.0, 4.0)) -> None:
    ensure_matplotlib()
    B, A, C5, C10, C20, C30 = lm["B"], lm["A"], lm["C5"], lm["C10"], lm["C20"], lm["C30"]
    outline = curves2d["outline"]
    leading = curves2d["leading"]
    te1, te2, te3 = curves2d["te1"], curves2d["te2"], curves2d["te3"]
    S = fin_area_S(outline, B)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    poly = np.vstack([outline, B[None, :]])
    poly_path = MplPath(poly)

    # interior point for control label placement
    interior = find_interior_point(poly_path, poly)

    # shaded fin area
    ax.fill(poly[:, 0], poly[:, 1], alpha=0.18, label=f"S={S:.3f} (rel²)")

    # curves
    ax.plot(leading[:, 0], leading[:, 1], linewidth=2.0, label="Leading edge (cubic)")
    ax.plot(te1[:, 0], te1[:, 1], linewidth=2.0, label="TE1 A→C5 (cubic)")
    ax.plot(te2[:, 0], te2[:, 1], linewidth=2.0, label="TE2 C5→C20 (quadratic)")
    ax.plot(te3[:, 0], te3[:, 1], linewidth=2.0, label="TE3 C20→C30 (quadratic)")

    # rays from B (dotted)
    for pt in (C5, C10, C20, C30):
        ax.plot([B[0], pt[0]], [B[1], pt[1]], linestyle=":", linewidth=1.6, alpha=0.85)

    # control polygons (dotted) to show placement
    def dotted_poly(points: List[np.ndarray]) -> None:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        ax.plot(xs, ys, linestyle=":", linewidth=1.2, alpha=0.75)

    dotted_poly([ctrl["LE_P0"], ctrl["LE_P1"], ctrl["LE_P2"], ctrl["LE_P3"]])
    dotted_poly([ctrl["TE1_P0"], ctrl["TE1_P1"], ctrl["TE1_P2"], ctrl["TE1_P3"]])
    dotted_poly([ctrl["TE2_Q0"], ctrl["TE2_Q1"], ctrl["TE2_Q2"]])
    dotted_poly([ctrl["TE3_Q0"], ctrl["TE3_P1"], ctrl["TE3_Q2"]])

    # label placement with simple collision avoidance
    taken: List[np.ndarray] = []

    # landmarks: OUTSIDE + bold
    lm_fs = int(FS_POINT * FONT_SCALE * LM_LABEL_MULT)
    landmarks = {"B": B, "A": A, "C5": C5, "C10": C10, "C20": C20, "C30": C30}
    for name, pt in landmarks.items():
        ax.scatter([pt[0]], [pt[1]], s=45)
        lab = place_landmark_label(poly_path, outline, pt, taken)
        ax.text(lab[0], lab[1], name, fontsize=lm_fs, fontweight="bold")

    # control points: label INSIDE
    cp_fs = int(FS_CTRL * FONT_SCALE * CTRL_LABEL_MULT)
    ctrl_points = {
        "LE-P1": ctrl["LE_P1"], "LE-P2": ctrl["LE_P2"],
        "TE1-P1": ctrl["TE1_P1"], "TE1-P2": ctrl["TE1_P2"],
        "TE2-Q1": ctrl["TE2_Q1"], "TE3-P1": ctrl["TE3_P1"],
    }
    for name, pt in ctrl_points.items():
        ax.scatter([pt[0]], [pt[1]], s=35, marker="x")
        lab = place_control_label(poly_path, interior, pt, taken)
        ax.text(lab[0], lab[1], name, fontsize=cp_fs)

    # trim axes to nearest integers and include legend anchor
    ax_xmin = min(float(poly[:, 0].min()), legend_anchor_xy[0]) - 1.0
    ax_xmax = max(float(poly[:, 0].max()), legend_anchor_xy[0]) + 1.0
    ax_ymin = min(float(poly[:, 1].min()), 0.0, legend_anchor_xy[1]) - 1.0
    ax_ymax = max(float(poly[:, 1].max()), legend_anchor_xy[1]) + 1.0
    ax.set_xlim(math.floor(ax_xmin), math.ceil(ax_xmax))
    ax.set_ylim(math.floor(ax_ymin), math.ceil(ax_ymax))
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("X", fontsize=int(FS_AXLAB * FONT_SCALE))
    ax.set_ylabel("Y (height)", fontsize=int(FS_AXLAB * FONT_SCALE))
    ax.tick_params(labelsize=int(8 * FONT_SCALE))

    ax.set_title(f"{fin_id}: 2D Bézier contour (FinShape3D {VERSION})", fontsize=FS_TITLE_2D)

    ax.legend(
        loc="lower left",
        bbox_to_anchor=legend_anchor_xy,
        bbox_transform=ax.transData,
        framealpha=0.90,
        fontsize=int(FS_LEGEND * LEGEND_SCALE),
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)


def chordwise_surface_rails(sections: List[Tuple[float, float, float, float, float]],
                            rail_fracs: np.ndarray,
                            edge_min_thickness: float = 0.0,
                            tip_min_chord: float = 0.0) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return longitudinal upper/lower surface rails at fixed chord fractions.

    Each rail spans every valid horizontal section. Fractions 0 and 1 are
    the leading and trailing boundary rails, where model thickness is zero.
    """
    rails = []
    for frac in np.asarray(rail_fracs, dtype=float):
        frac = float(clamp(float(frac), 0.0, 1.0))
        xs, y_vals, z_up, z_dn = [], [], [], []
        for (y, x_le, C, SP, Tm) in sections:
            x_le_vis, C_vis, SP_vis, Tm_vis = structural_visual_section(
                x_le, C, SP, Tm, edge_min_thickness=edge_min_thickness, tip_min_chord=tip_min_chord
            )
            c = frac * C_vis
            x = x_le_vis - c
            th = float(structural_visual_thickness(
                np.array([c], dtype=float), C_vis, SP_vis, Tm_vis, edge_min_thickness=edge_min_thickness
            )[0])
            xs.append(float(x))
            y_vals.append(float(y))
            z_up.append(0.5 * th)
            z_dn.append(-0.5 * th)
        rails.append((np.array(xs, float), np.array(y_vals, float), np.array(z_up, float), np.array(z_dn, float)))
    return rails


def append_terminal_apex_section(sections: List[Tuple[float, float, float, float, float]],
                                 outline2d: np.ndarray,
                                 y_max: float,
                                 tol: float = 1e-9) -> List[Tuple[float, float, float, float, float]]:
    """Return sections with a zero-width terminal apex section added when needed."""
    if not sections:
        return sections

    top_mask = np.isclose(outline2d[:, 1], y_max, rtol=0.0, atol=max(tol, abs(y_max) * 1e-9))
    if np.any(top_mask):
        apex_x = float(np.mean(outline2d[top_mask, 0]))
    else:
        apex_x = float(outline2d[int(np.argmax(outline2d[:, 1])), 0])

    if abs(float(sections[-1][0]) - y_max) <= max(tol, abs(y_max) * 1e-9):
        _last_y, _last_x, last_c, _last_sp, last_t = sections[-1]
        if last_c <= tol and last_t <= tol:
            return sections

    return sections + [(float(y_max), apex_x, 0.0, 0.0, 0.0)]


def sample_polyline_by_arclength(polyline: np.ndarray, n: int) -> np.ndarray:
    """Sample n approximately equally spaced points along a polyline."""
    pts = np.asarray(polyline, dtype=float)
    if len(pts) == 0:
        return pts.reshape(0, 2)
    if len(pts) == 1 or n <= 1:
        return pts[:1].copy()

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(dist[-1])
    if total <= 1e-12:
        return np.repeat(pts[:1], n, axis=0)

    targets = np.linspace(0.0, total, n)
    xs = np.interp(targets, dist, pts[:, 0])
    ys = np.interp(targets, dist, pts[:, 1])
    return np.column_stack([xs, ys])


def _unique_points(points: np.ndarray, decimals: int = 9) -> np.ndarray:
    seen = set()
    unique = []
    for x, y in np.asarray(points, dtype=float):
        key = (round(float(x), decimals), round(float(y), decimals))
        if key not in seen:
            seen.add(key)
            unique.append((float(x), float(y)))
    return np.array(unique, dtype=float)


def contour_reference_points(curves2d: Dict[str, np.ndarray], n_edge_points: int = 100) -> np.ndarray:
    """Return edge reference points from arclength samples plus available curve endpoints."""
    refs = [sample_polyline_by_arclength(curves2d["outline"], n_edge_points)]
    for key in ("leading", "trailing", "le0", "le1", "te1", "te2", "te3"):
        curve = curves2d.get(key)
        if curve is not None and len(curve) > 0:
            refs.append(np.asarray([curve[0], curve[-1]], dtype=float))
    return _unique_points(np.vstack(refs))


def contour_reference_y_values(curves2d: Dict[str, np.ndarray], y_max: float,
                               n_y_slices: int, top_focus_power: float,
                               n_edge_points: int = 100, max_sections: int = 55) -> np.ndarray:
    """Choose readable section levels from contour references, always including base and apex."""
    t = np.linspace(0.0, 1.0, n_y_slices)
    base = y_max * (1.0 - (1.0 - t) ** top_focus_power)
    base = np.clip(base, 0.0, y_max * 0.9995)

    refs = contour_reference_points(curves2d, n_edge_points=n_edge_points)
    candidates = np.concatenate([base, refs[:, 1], np.array([0.0, y_max], dtype=float)])
    candidates = candidates[np.isfinite(candidates)]
    candidates = np.clip(candidates, 0.0, y_max)
    candidates = np.unique(np.round(candidates, 9))
    candidates.sort()

    target_count = max(n_y_slices, min(max_sections, n_edge_points // 2 + 5))
    if len(candidates) > target_count:
        idx = np.unique(np.round(np.linspace(0, len(candidates) - 1, target_count)).astype(int))
        candidates = candidates[idx]
    if candidates[0] != 0.0:
        candidates = np.insert(candidates, 0, 0.0)
    if candidates[-1] != round(float(y_max), 9):
        candidates = np.append(candidates, round(float(y_max), 9))
    return candidates.astype(float)


def readable_chordwise_rail_fractions(n_chord_pts: int, max_rails: int = 35) -> np.ndarray:
    """Return a readable number of chordwise rail fractions, including both edges."""
    rail_count = max(13, min(max_rails, int(n_chord_pts)))
    return np.linspace(0.0, 1.0, rail_count)


def structural_visual_section(x_le: float, C: float, SP: float, Tm: float,
                              edge_min_thickness: float = DEFAULT_EDGE_MIN_THICKNESS,
                              tip_min_chord: float = DEFAULT_TIP_MIN_CHORD) -> Tuple[float, float, float, float]:
    """Return visualization-only section geometry with documented nonzero edge/tip floors."""
    edge_floor = max(0.0, float(edge_min_thickness))
    chord_floor = max(0.0, float(tip_min_chord))
    c_orig = max(0.0, float(C))
    c_vis = max(c_orig, chord_floor)
    center_x = float(x_le) - 0.5 * c_orig
    x_le_vis = center_x + 0.5 * c_vis
    if c_vis <= 1e-12:
        return x_le_vis, 0.0, 0.0, max(float(Tm), edge_floor)
    sp_vis = float(SP) if c_orig > 1e-12 else 0.5 * c_vis
    sp_vis = clamp(sp_vis, 0.05 * c_vis, 0.95 * c_vis)
    tm_vis = max(float(Tm), edge_floor)
    return x_le_vis, c_vis, sp_vis, tm_vis


def structural_visual_thickness(c: np.ndarray, C: float, SP: float, Tm: float,
                                edge_min_thickness: float = DEFAULT_EDGE_MIN_THICKNESS) -> np.ndarray:
    """Return plotted thickness with a visualization-only minimum structural floor."""
    th = thickness_piecewise(c, C, SP, Tm)
    floor = max(0.0, float(edge_min_thickness))
    if floor > 0.0:
        th = np.maximum(th, floor)
    return th


def draw_3d_wireframe_axes(ax, curves2d: Dict[str, np.ndarray], fin_id: str,
                           n_y_slices: int = 39, n_chord_pts: int = 117, n_z_slices: int = 20,
                           top_focus_power: float = 3.0,
                           edge_min_thickness: float = DEFAULT_EDGE_MIN_THICKNESS,
                           tip_min_chord: float = DEFAULT_TIP_MIN_CHORD,
                           pitch: float = 18.0, yaw: float = -65.0, roll: float = 0.0) -> None:
    """Draw the 3D wireframe on an existing axes using visualization-only structural floors."""
    leading2d = curves2d["leading"]
    trailing2d = curves2d["trailing"]
    outline2d = curves2d["outline"]

    y_max = float(outline2d[:, 1].max())
    if y_max <= 1e-9:
        raise ValueError(f"{fin_id}: degenerate outline (y_max≈0).")

    y_vals = contour_reference_y_values(
        curves2d, y_max, n_y_slices=n_y_slices, top_focus_power=top_focus_power,
        n_edge_points=100, max_sections=55,
    )

    sections = []
    halfTmax = 0.0
    for y in y_vals:
        xs_le = x_intersections_with_horizontal(leading2d, float(y))
        xs_tr = x_intersections_with_horizontal(trailing2d, float(y))
        if (not xs_le) or (not xs_tr):
            continue
        x_le = max(xs_le)
        x_tr = min(xs_tr)
        C = float(x_le - x_tr)
        if C <= 0.10:
            continue

        SP, Tm = morteo_SP_T(C)
        SP = clamp(SP, 0.05 * C, 0.95 * C)
        Tm = max(Tm, 1e-4)
        halfTmax = max(halfTmax, 0.5 * Tm)
        sections.append((float(y), float(x_le), float(C), float(SP), float(Tm)))

    if len(sections) < 3:
        raise ValueError(f"{fin_id}: insufficient slices for 3D wireframe.")

    sections = append_terminal_apex_section(sections, outline2d, y_max)

    halfTmax = max(halfTmax, 0.5 * max(0.0, float(edge_min_thickness)))
    z_levels = np.linspace(0.0, halfTmax, n_z_slices)
    wire_color = "blue"

    c_frac = np.linspace(0.0, 1.0, n_chord_pts)
    for (y, x_le, C, SP, Tm) in sections:
        x_le_vis, C_vis, SP_vis, Tm_vis = structural_visual_section(
            x_le, C, SP, Tm, edge_min_thickness=edge_min_thickness, tip_min_chord=tip_min_chord
        )
        c = c_frac * C_vis
        x = x_le_vis - c
        th = structural_visual_thickness(c, C_vis, SP_vis, Tm_vis, edge_min_thickness=edge_min_thickness)
        z_up = 0.5 * th
        z_dn = -0.5 * th
        ax.plot(x, z_up, np.full_like(x, y), linewidth=0.85, color=wire_color)
        ax.plot(x, z_dn, np.full_like(x, y), linewidth=0.85, color=wire_color)

    rail_fracs = readable_chordwise_rail_fractions(n_chord_pts, max_rails=35)
    for xs, ys, z_up, z_dn in chordwise_surface_rails(
        sections, rail_fracs, edge_min_thickness=edge_min_thickness, tip_min_chord=tip_min_chord
    ):
        ax.plot(xs, z_up, ys, linewidth=0.8, color=wire_color)
        ax.plot(xs, z_dn, ys, linewidth=0.8, color=wire_color)

    for zl in z_levels[1:]:
        for sign in (+1.0, -1.0):
            zc = sign * zl
            xs_front, xs_back, ys = [], [], []
            for (y, x_le, C, SP, Tm) in sections:
                ys.append(y)
                x_le_vis, C_vis, SP_vis, Tm_vis = structural_visual_section(
                    x_le, C, SP, Tm, edge_min_thickness=edge_min_thickness, tip_min_chord=tip_min_chord
                )
                target = 2.0 * abs(zc)
                if target > Tm_vis:
                    xs_front.append(np.nan); xs_back.append(np.nan)
                    continue
                inv = invert_thickness_for_c(target, C_vis, SP_vis, Tm_vis)
                if inv is None:
                    xs_front.append(np.nan); xs_back.append(np.nan)
                    continue
                cL, cR = inv
                xs_front.append(x_le_vis - cL)
                xs_back.append(x_le_vis - cR)

            xs_front = np.array(xs_front, float)
            xs_back = np.array(xs_back, float)
            ys = np.array(ys, float)

            for xs in (xs_front, xs_back):
                mask = np.isfinite(xs)
                if mask.sum() < 2:
                    continue
                idx = np.where(mask)[0]
                blocks = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
                for b in blocks:
                    if len(b) >= 2:
                        ax.plot(xs[b], np.full(len(b), zc), ys[b], linewidth=0.8, color=wire_color)

    ax.plot(outline2d[:, 0], np.zeros(len(outline2d)), outline2d[:, 1], linewidth=2.2, color="black")
    ax.scatter([0.0], [0.0], [0.0], s=45, color="black")

    ax.set_xlabel("X", fontsize=int(FS_AXLAB * FONT_SCALE), labelpad=18)
    ax.set_ylabel("Z", fontsize=int(FS_AXLAB * FONT_SCALE), labelpad=24)
    ax.set_zlabel("Y", fontsize=int(FS_AXLAB * FONT_SCALE), labelpad=18)
    ax.tick_params(labelsize=int(8 * FONT_SCALE))

    xmin = float(outline2d[:, 0].min()) - 1.0
    xmax = float(outline2d[:, 0].max()) + 1.0
    xlim = (math.floor(xmin), math.ceil(xmax))

    thickness_lim = max(halfTmax * 1.2, 1.2)
    ylim = (-thickness_lim, thickness_lim)
    zlim = (0.0, y_max * 1.02)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    ax.set_yticks([-1.0, 0.0, 1.0])

    set_equal_3d_aspect(ax, xlim, ylim, zlim)
    try:
        ax.view_init(elev=float(pitch), azim=float(yaw), roll=float(roll))
    except TypeError:
        ax.view_init(elev=float(pitch), azim=float(yaw))
    ax.set_title(f"{fin_id}: 3D wireframe (FinShape3D {VERSION})", fontsize=FS_TITLE_3D)


def plot_3d_wireframe(curves2d: Dict[str, np.ndarray], fin_id: str, out_png: Path,
                      n_y_slices: int = 39, n_chord_pts: int = 117, n_z_slices: int = 20,
                      top_focus_power: float = 3.0,
                      edge_min_thickness: float = DEFAULT_EDGE_MIN_THICKNESS,
                      tip_min_chord: float = DEFAULT_TIP_MIN_CHORD) -> None:
    ensure_matplotlib()
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    draw_3d_wireframe_axes(
        ax, curves2d, fin_id, n_y_slices=n_y_slices, n_chord_pts=n_chord_pts,
        n_z_slices=n_z_slices, top_focus_power=top_focus_power,
        edge_min_thickness=edge_min_thickness, tip_min_chord=tip_min_chord,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)

def make_montage(png_paths: List[Path], out_path: Path, title: Optional[str] = None) -> None:
    ensure_matplotlib()
    n = len(png_paths)
    cols = 2
    rows = int(math.ceil(n / cols)) if n > 0 else 1

    imgs = [mpimg.imread(str(p)) for p in png_paths]
    fig, axes = plt.subplots(rows, cols, figsize=(7.0 * cols, 6.2 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for k, (img, p) in enumerate(zip(imgs, png_paths)):
        r = k // cols
        c = k % cols
        axes[r, c].imshow(img)
        axes[r, c].set_title(p.name, fontsize=FS_MONT_TILE)

    if title:
        fig.suptitle(title, fontsize=FS_MONT_TITLE)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Pipeline
# -----------------------------
def run(input_path: Path, outdir: Path, bc30scale: float, y_slices: int, z_slices: int,
        top_focus_power: float, montage: bool,
        edge_min_thickness: float = DEFAULT_EDGE_MIN_THICKNESS,
        tip_min_chord: float = DEFAULT_TIP_MIN_CHORD) -> None:
    input_path = input_path.expanduser().resolve()
    outdir = outdir.expanduser().resolve()
    print(f"[STARTUP] FinShape3D {VERSION}")
    print(f"[STARTUP] Input file: {input_path}")
    print(f"[STARTUP] Output directory: {outdir}")
    print(f"[STARTUP] Structural visualization assumption: edge_min_thickness={edge_min_thickness:g}, tip_min_chord={tip_min_chord:g} AB units")

    df = read_table(input_path)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[STARTUP] Output directory ready: {outdir}")

    outs2d: List[Path] = []
    outs3d: List[Path] = []
    failures: List[Tuple[str, str]] = []

    for i, row in df.iterrows():
        fin_id = str(row.get("id", f"row{i}"))
        try:
            row_warnings: List[str] = []
            lm, u_AB = build_landmarks(row, BC30_scale=bc30scale)
            ctrl = build_controls(lm, u_AB, warnings=row_warnings)
            for warning in row_warnings:
                print(f"[WARN] row {i} ({fin_id}): {warning}")
            curves2d = compute_2d_curves(ctrl, n=1170)

            p2d = outdir / f"FinShape3D_{VERSION}_2D_{fin_id}.png"
            p3d = outdir / f"FinShape3D_{VERSION}_3D_{fin_id}.png"
            plot_2d(curves2d, lm, ctrl, fin_id, p2d, legend_anchor_xy=(-4.0, 4.0))
            plot_3d_wireframe(
                curves2d, fin_id, p3d, n_y_slices=y_slices, n_z_slices=z_slices,
                top_focus_power=top_focus_power, edge_min_thickness=edge_min_thickness,
                tip_min_chord=tip_min_chord,
            )

            outs2d.append(p2d)
            outs3d.append(p3d)
            print(f"[OK] row {i} ({fin_id}) -> {p2d.name}, {p3d.name}")
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            failures.append((fin_id, msg))
            print(f"[ERROR] row {i} ({fin_id}) skipped: {msg}")

    if failures:
        print(f"[SUMMARY] {len(failures)} row(s) skipped; {len(outs2d)} row(s) processed.")
        for fin_id, msg in failures:
            print(f"[SUMMARY] skipped {fin_id}: {msg}")
        if not outs2d:
            raise SystemExit("No rows could be processed; see errors above.")

    if montage and outs2d:
        make_montage(outs2d, outdir / f"FinShape3D_{VERSION}_montage_2D.png",
                     title=f"FinShape3D {VERSION} — 2D Contours")
    if montage and outs3d:
        make_montage(outs3d, outdir / f"FinShape3D_{VERSION}_montage_3D.png",
                     title=f"FinShape3D {VERSION} — 3D Wireframes")

    print(f"[DONE] Output directory: {outdir}")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Reconstruct 2D/3D dorsal fins from ratio tables.")
    ap.add_argument("--input", type=Path, default=None, help="CSV or XLSX containing ratios.")
    ap.add_argument("--outdir", type=Path, default=None, help="Output directory (default next to input).")
    ap.add_argument("--bc30scale", type=float, default=10.0, help="Set |BC30| to this value (relative units).")
    ap.add_argument("--y_slices", type=int, default=39, help="Horizontal slices for 3D wireframe (default increased 30 percent from 30).")
    ap.add_argument("--z_slices", type=int, default=20, help="Vertical slices for 3D wireframe (default increased at least 30 percent from 15).")
    ap.add_argument("--top_focus_power", type=float, default=3.0, help="Concentrate y-slices near tip.")
    ap.add_argument("--edge_min_thickness", type=float, default=DEFAULT_EDGE_MIN_THICKNESS, help="Visualization-only minimum displayed edge/tip thickness in AB=10 units; not measured biology.")
    ap.add_argument("--tip_min_chord", type=float, default=DEFAULT_TIP_MIN_CHORD, help="Visualization-only minimum displayed apex chord in AB=10 units; prevents tip collapse.")
    ap.add_argument("--no_montage", action="store_true", help="Disable montage creation.")

    # PyCharm-safe: ignore foreign args
    args, _unknown = ap.parse_known_args(argv)

    input_path = args.input if args.input is not None else select_input_file_interactive()
    outdir = args.outdir if args.outdir is not None else default_outdir_for_input(input_path)

    run(
        input_path=input_path,
        outdir=outdir,
        bc30scale=args.bc30scale,
        y_slices=args.y_slices,
        z_slices=args.z_slices,
        top_focus_power=args.top_focus_power,
        montage=(not args.no_montage),
        edge_min_thickness=args.edge_min_thickness,
        tip_min_chord=args.tip_min_chord,
    )


if __name__ == "__main__":
    main()
