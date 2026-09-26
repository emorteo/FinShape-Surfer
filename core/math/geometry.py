"""Geometry helpers used by GUI and reconstruction code.

This module exposes both tuple-based vector helpers (used by the Tk GUI)
and NumPy-based equivalents for numerical algorithms used in reconstruction
and plotting.
"""

import math
from typing import Tuple, List
import numpy as np


def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp a scalar to the closed interval [lo, hi]."""
    return max(lo, min(hi, v))


# Tuple-based geometry (for Tkinter GUI)
def v_add(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """Component-wise addition of 2D tuples."""
    return (a[0] + b[0], a[1] + b[1])


def v_sub(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """Component-wise subtraction a - b."""
    return (a[0] - b[0], a[1] - b[1])


def v_mul(a: Tuple[float, float], s: float) -> Tuple[float, float]:
    """Scale a 2D vector by scalar s."""
    return (a[0] * s, a[1] * s)


def v_len(a: Tuple[float, float]) -> float:
    """Euclidean length of a 2D tuple."""
    return math.hypot(a[0], a[1])


def v_norm(a: Tuple[float, float]) -> Tuple[float, float]:
    """Return a unit vector in the direction of `a`.

    Falls back to (1,0) when the input is near zero to avoid division errors.
    """
    n = v_len(a)
    if n < 1e-12:
        return (1.0, 0.0)
    return (a[0] / n, a[1] / n)


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Distance between two 2D points."""
    return v_len(v_sub(a, b))


def shoelace_area(poly: List[Tuple[float, float]]) -> float:
    """Signed polygon area via the shoelace formula (absolute value returned)."""
    if len(poly) < 3:
        return 0.0
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def rot2(v: Tuple[float, float], ang_rad: float) -> Tuple[float, float]:
    """Rotate a 2D tuple by `ang_rad` radians about the origin."""
    c, s = math.cos(ang_rad), math.sin(ang_rad)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1])


def rotate_about_origin(p: Tuple[float, float], ang: float) -> Tuple[float, float]:
    """Alias for `rot2` kept for historical compatibility."""
    return rot2(p, ang)


# Numpy-based geometry (for 3D generation and matplotlib)
def rot_np(v: np.ndarray, ang_rad: float) -> np.ndarray:
    """Rotate a numpy 2-vector by `ang_rad` radians."""
    c, s = math.cos(ang_rad), math.sin(ang_rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=float)


def normalize_np(v: np.ndarray) -> np.ndarray:
    """Return the unit-normalized numpy vector (fallback value when near-zero)."""
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0], float)
    return v / n


def y_on_line_through_np(B: np.ndarray, A: np.ndarray, x: float) -> float:
    """Compute y at given x on the line through B and A. Returns inf for vertical lines."""
    if abs(A[0] - B[0]) < 1e-12:
        return float("inf")
    m = (A[1] - B[1]) / (A[0] - B[0])
    return float(B[1] + m * (x - B[0]))


def polygon_area_np(poly: np.ndarray) -> float:
    """Compute polygon area for a (N,2) numpy array of points."""
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def x_intersections_with_horizontal_np(polyline: np.ndarray, yval: float) -> List[float]:
    """Return x coordinates where the polyline crosses horizontal line y=yval."""
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
