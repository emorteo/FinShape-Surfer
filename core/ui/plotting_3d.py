import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d import Axes3D

from core.math.geometry import rot_np, normalize_np
from core.reconstruction.wireframe import fin_area_S, morteo_SP_T, thickness_piecewise, invert_thickness_for_c
from core.math.geometry import x_intersections_with_horizontal_np

VERSION = "1.1.9" 
FONT_SCALE = 3.0
LEGEND_SCALE = 2.0
LM_LABEL_MULT = 0.5
CTRL_LABEL_MULT = 0.5
FS_POINT = 10
FS_CTRL = 9
FS_AXLAB = 10
FS_LEGEND = 9
FS_TITLE_2D = 12
FS_TITLE_3D = 11
FS_MONT_TILE = 9
FS_MONT_TITLE = 14

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def set_equal_3d_aspect(ax, xlim, ylim, zlim) -> None:
    xr = abs(xlim[1] - xlim[0]) or 1.0
    yr = abs(ylim[1] - ylim[0]) or 1.0
    zr = abs(zlim[1] - zlim[0]) or 1.0
    ax.set_box_aspect((xr, yr, zr))

def find_interior_point(poly_path: MplPath, poly: np.ndarray) -> np.ndarray:
    cand = np.mean(poly, axis=0)
    if poly_path.contains_point((float(cand[0]), float(cand[1]))): return cand
    cand2 = np.mean(poly[:-1], axis=0)
    if poly_path.contains_point((float(cand2[0]), float(cand2[1]))): return cand2
    xmin, xmax = float(np.min(poly[:, 0])), float(np.max(poly[:, 0]))
    ymin, ymax = float(np.min(poly[:, 1])), float(np.max(poly[:, 1]))
    rng = np.random.default_rng(12345)
    for _ in range(500):
        x = float(rng.uniform(xmin, xmax))
        y = float(rng.uniform(ymin, ymax))
        if poly_path.contains_point((x, y)): return np.array([x, y], float)
    return cand

def nearest_index(polyline: np.ndarray, pt: np.ndarray) -> int:
    d2 = np.sum((polyline - pt[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))

def local_outward_normal(poly_path: MplPath, polyline: np.ndarray, pt: np.ndarray) -> np.ndarray:
    i = nearest_index(polyline, pt)
    n = len(polyline)
    im1 = (i - 1) % n
    ip1 = (i + 1) % n
    tangent = normalize_np(polyline[ip1] - polyline[im1])
    normal = normalize_np(np.array([-tangent[1], tangent[0]], float))
    eps = 0.05
    test = pt + eps * normal
    if poly_path.contains_point((float(test[0]), float(test[1]))): normal = -normal
    return normal

def repel_from_taken(pos: np.ndarray, taken: List[np.ndarray], step: float = 0.20, min_dist: float = 0.55, max_iter: int = 40) -> np.ndarray:
    p = pos.copy()
    for _ in range(max_iter):
        ok = True
        for q in taken:
            if float(np.linalg.norm(p - q)) < min_dist:
                ok = False
                d = normalize_np(p - q)
                perp = np.array([-d[1], d[0]], float)
                p = p + step * perp
                break
        if ok: return p
    return p

def place_landmark_label(poly_path: MplPath, polyline: np.ndarray, pt: np.ndarray, taken: List[np.ndarray]) -> np.ndarray:
    n_out = local_outward_normal(poly_path, polyline, pt)
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

def place_control_label(poly_path: MplPath, interior: np.ndarray, pt: np.ndarray, taken: List[np.ndarray]) -> np.ndarray:
    d_in = normalize_np(interior - pt)
    offset = 0.15
    for _ in range(60):
        cand = pt + offset * d_in
        if poly_path.contains_point((float(cand[0]), float(cand[1]))):
            cand = repel_from_taken(cand, taken, min_dist=0.50)
            taken.append(cand)
            return cand
        offset += 0.12
    cand = interior.copy()
    cand = repel_from_taken(cand, taken, min_dist=0.50)
    taken.append(cand)
    return cand

def plot_2d(curves2d: Dict[str, np.ndarray], lm: Dict[str, np.ndarray], ctrl: Dict[str, np.ndarray],
            fin_id: str, out_png: Path, legend_anchor_xy: Tuple[float, float] = (-4.0, 4.0)) -> None:
    B, A, C5, C10, C20, C30 = lm["B"], lm["A"], lm["C5"], lm["C10"], lm["C20"], lm["C30"]
    outline = curves2d["outline"]
    leading = curves2d["leading"]
    te1, te2, te3 = curves2d["te1"], curves2d["te2"], curves2d["te3"]
    S = fin_area_S(outline, B)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    poly = np.vstack([outline, B[None, :]])
    poly_path = MplPath(poly)
    interior = find_interior_point(poly_path, poly)

    ax.fill(poly[:, 0], poly[:, 1], alpha=0.18, label=f"S={S:.3f} (rel²)")
    ax.plot(leading[:, 0], leading[:, 1], linewidth=2.0, label="Leading edge (cubic)")
    ax.plot(te1[:, 0], te1[:, 1], linewidth=2.0, label="TE1 A→C5 (cubic)")
    ax.plot(te2[:, 0], te2[:, 1], linewidth=2.0, label="TE2 C5→C20 (quadratic)")
    ax.plot(te3[:, 0], te3[:, 1], linewidth=2.0, label="TE3 C20→C30 (quadratic)")

    for pt in (C5, C10, C20, C30):
        ax.plot([B[0], pt[0]], [B[1], pt[1]], linestyle=":", linewidth=1.6, alpha=0.85)

    def dotted_poly(points: List[np.ndarray]) -> None:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        ax.plot(xs, ys, linestyle=":", linewidth=1.2, alpha=0.75)

    dotted_poly([ctrl["LE_P0"], ctrl["LE_P1"], ctrl["LE_P2"], ctrl["LE_P3"]])
    dotted_poly([ctrl["TE1_P0"], ctrl["TE1_P1"], ctrl["TE1_P2"], ctrl["TE1_P3"]])
    dotted_poly([ctrl["TE2_Q0"], ctrl["TE2_Q1"], ctrl["TE2_Q2"]])
    dotted_poly([ctrl["TE3_Q0"], ctrl["TE3_P1"], ctrl["TE3_Q2"]])

    taken: List[np.ndarray] = []
    lm_fs = int(FS_POINT * FONT_SCALE * LM_LABEL_MULT)
    landmarks = {"B": B, "A": A, "C5": C5, "C10": C10, "C20": C20, "C30": C30}
    for name, pt in landmarks.items():
        ax.scatter([pt[0]], [pt[1]], s=45)
        lab = place_landmark_label(poly_path, outline, pt, taken)
        ax.text(lab[0], lab[1], name, fontsize=lm_fs, fontweight="bold")

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
    ax.legend(loc="lower left", bbox_to_anchor=legend_anchor_xy, bbox_transform=ax.transData, framealpha=0.90, fontsize=int(FS_LEGEND * LEGEND_SCALE))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)

def plot_3d_wireframe(curves2d: Dict[str, np.ndarray], fin_id: str, out_png: Path,
                      n_y_slices: int = 30, n_chord_pts: int = 90, n_z_slices: int = 15,
                      top_focus_power: float = 3.0) -> None:
    leading2d = curves2d["leading"]
    trailing2d = curves2d["trailing"]
    outline2d = curves2d["outline"]

    y_max = float(outline2d[:, 1].max())
    if y_max <= 1e-9: raise ValueError(f"{fin_id}: degenerate outline (y_max≈0).")

    t = np.linspace(0.0, 1.0, n_y_slices)
    y_vals = y_max * (1.0 - (1.0 - t) ** top_focus_power)
    y_vals = np.clip(y_vals, 0.0, y_max * 0.9995)

    sections = []
    halfTmax = 0.0
    for y in y_vals:
        xs_le = x_intersections_with_horizontal_np(leading2d, float(y))
        xs_tr = x_intersections_with_horizontal_np(trailing2d, float(y))
        if (not xs_le) or (not xs_tr): continue
        x_le = max(xs_le)
        x_tr = min(xs_tr)
        C = float(x_le - x_tr)
        if C <= 0.10: continue

        SP, Tm = morteo_SP_T(C)
        SP = clamp(SP, 0.05 * C, 0.95 * C)
        Tm = max(Tm, 1e-4)
        halfTmax = max(halfTmax, 0.5 * Tm)
        sections.append((float(y), float(x_le), float(C), float(SP), float(Tm)))

    if len(sections) < 3: raise ValueError(f"{fin_id}: insufficient slices for 3D wireframe.")
    z_levels = np.linspace(0.0, halfTmax, n_z_slices)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    wire_color = "blue"

    c_frac = np.linspace(0.0, 1.0, n_chord_pts)
    for (y, x_le, C, SP, Tm) in sections:
        c = c_frac * C
        x = x_le - c
        th = thickness_piecewise(c, C, SP, Tm)
        z_up = 0.5 * th
        z_dn = -0.5 * th
        ax.plot(x, z_up, np.full_like(x, y), linewidth=0.85, color=wire_color)
        ax.plot(x, z_dn, np.full_like(x, y), linewidth=0.85, color=wire_color)

    for zl in z_levels[1:]:
        for sign in (+1.0, -1.0):
            zc = sign * zl
            xs_front, xs_back, ys = [], [], []
            for (y, x_le, C, SP, Tm) in sections:
                ys.append(y)
                target = 2.0 * abs(zc)
                if target > Tm:
                    xs_front.append(np.nan); xs_back.append(np.nan)
                    continue
                inv = invert_thickness_for_c(target, C, SP, Tm)
                if inv is None:
                    xs_front.append(np.nan); xs_back.append(np.nan)
                    continue
                cL, cR = inv
                xs_front.append(x_le - cL)
                xs_back.append(x_le - cR)

            xs_front = np.array(xs_front, float)
            xs_back = np.array(xs_back, float)
            ys = np.array(ys, float)

            for xs in (xs_front, xs_back):
                mask = np.isfinite(xs)
                if mask.sum() < 2: continue
                idx = np.where(mask)[0]
                blocks = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
                for b in blocks:
                    if len(b) >= 2: ax.plot(xs[b], np.full(len(b), zc), ys[b], linewidth=0.8, color=wire_color)

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
    ax.view_init(elev=18, azim=-65)
    ax.set_title(f"{fin_id}: 3D wireframe (FinShape3D {VERSION})", fontsize=FS_TITLE_3D)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)

def make_montage(png_paths: List[Path], out_path: Path, title: Optional[str] = None) -> None:
    n = len(png_paths)
    cols = 2
    rows = int(math.ceil(n / cols)) if n > 0 else 1
    imgs = [mpimg.imread(str(p)) for p in png_paths]
    fig, axes = plt.subplots(rows, cols, figsize=(7.0 * cols, 6.2 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for ax in axes.flat: ax.axis("off")
    for k, (img, p) in enumerate(zip(imgs, png_paths)):
        r = k // cols
        c = k % cols
        axes[r, c].imshow(img)
        axes[r, c].set_title(p.name, fontsize=FS_MONT_TILE)
    if title: fig.suptitle(title, fontsize=FS_MONT_TITLE)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
