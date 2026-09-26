#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinShape2D R-GUI 2.0.20

Fixes vs 2.0.7 (requested)
--------------------------
1) Button workflow order + Apply Crop activation:
   - Step order: 1 Load, 2 Flip, 3 Rotate, 4 Crop, 5 Apply Crop (enabled only after a crop box is defined), 6 Reset Crop.
2) Row labels:
   - Row 1 labeled **Edit image**
   - Row 2 labeled **Morphometry**
3) Current image name displayed at the upper-right of the main window.
4) Saving filenames:
   - When choosing the CSV file for the session, default filename is based on the CURRENT image name.
   - Contour exports default to "<image_stem>_contour.<ext>".
5) 2D display y-axis:
   - Uses a true y-up plotting convention (values increase upwards on screen) by flipping y during canvas mapping.
6) Save 2D contour (rotated) in common formats:
   - Adds "Save contour…" button in the 2D window.
   - Saves rotated so that segment B→C30 is horizontal in standardized coordinates.
   - Mirrors final x-values so the exported/displayed contour follows the right-swimming convention.
   - Supports SVG export natively (no extra deps).
   - Supports PNG/JPG/BMP exports if Pillow is installed (via PostScript rasterization; may require Ghostscript on some systems).
7) Canvas navigation + contour orientation maintenance:
   - Main image canvas is scrollable so Bezier handles outside a tight crop remain reachable.
   - Final 2D contour x-values are mirrored to match the right-swimming standard output convention.
8) Leading-edge B→D smoothing:
   - The B→D leading-edge segment is now cubic, with an additional editable control point referenced from B.
9) Zoom placement:
   - The zoom window is anchored at the upper-right of the image canvas so it does not interfere with point B after tight crops.
10) Point-color clarity:
   - Fixed/reference landmarks are drawn in red.
   - Derived reference point O is shown in red when A, B, and C30 exist.
   - Editable Bezier control points are drawn in green.
11) Fixed landmark protection:
   - Bezier endpoint fields coincident with fixed landmarks are no longer draggable handles.

Dependencies
------------
- Standard library: tkinter, math, csv, pathlib, dataclasses.
- Optional: Pillow (PIL) for JPEG + flip/rotate/crop + zoom + raster exports.
  Install: pip install pillow
"""

from __future__ import annotations

from types import SimpleNamespace

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PIL_OK = False
try:
    from PIL import Image, ImageTk, ImageDraw  # type: ignore
    PIL_OK = True
except Exception:
    PIL_OK = False

import tkinter as tk
from tkinter import filedialog, messagebox


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def v_add(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] + b[0], a[1] + b[1])


def v_sub(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def v_mul(a: Tuple[float, float], s: float) -> Tuple[float, float]:
    return (a[0] * s, a[1] * s)


def v_len(a: Tuple[float, float]) -> float:
    return math.hypot(a[0], a[1])


def v_norm(a: Tuple[float, float]) -> Tuple[float, float]:
    n = v_len(a)
    if n < 1e-12:
        return (1.0, 0.0)
    return (a[0] / n, a[1] / n)


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return v_len(v_sub(a, b))


def shoelace_area(poly: List[Tuple[float, float]]) -> float:
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
    c, s = math.cos(ang_rad), math.sin(ang_rad)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1])


def bezier_quad(q0, q1, q2, t: float) -> Tuple[float, float]:
    u = 1.0 - t
    return (
        u * u * q0[0] + 2 * u * t * q1[0] + t * t * q2[0],
        u * u * q0[1] + 2 * u * t * q1[1] + t * t * q2[1],
    )


def bezier_cubic(p0, p1, p2, p3, t: float) -> Tuple[float, float]:
    u = 1.0 - t
    return (
        (u ** 3) * p0[0] + 3 * (u ** 2) * t * p1[0] + 3 * u * (t ** 2) * p2[0] + (t ** 3) * p3[0],
        (u ** 3) * p0[1] + 3 * (u ** 2) * t * p1[1] + 3 * u * (t ** 2) * p2[1] + (t ** 3) * p3[1],
    )


def sample_curve(fn, n: int = 280) -> List[Tuple[float, float]]:
    return [fn(i / (n - 1)) for i in range(n)] if n > 1 else [fn(0.0)]


@dataclass
class Landmarks:
    A: Tuple[float, float]
    B: Tuple[float, float]
    C5: Tuple[float, float]
    C10: Tuple[float, float]
    C20: Tuple[float, float]
    C30: Tuple[float, float]
    D: Tuple[float, float]


@dataclass
class BezierControls:
    LE0_P0: Tuple[float, float]
    LE0_P1: Tuple[float, float]
    LE0_P2: Tuple[float, float]
    LE0_P3: Tuple[float, float]
    LE1_P0: Tuple[float, float]
    LE1_P1: Tuple[float, float]
    LE1_P2: Tuple[float, float]
    LE1_P3: Tuple[float, float]
    TE1_P0: Tuple[float, float]
    TE1_P1: Tuple[float, float]
    TE1_P2: Tuple[float, float]
    TE1_P3: Tuple[float, float]
    TE2_Q0: Tuple[float, float]
    TE2_Q1: Tuple[float, float]
    TE2_Q2: Tuple[float, float]
    TE3_Q0: Tuple[float, float]
    TE3_Q1: Tuple[float, float]
    TE3_Q2: Tuple[float, float]


FIXED_BEZIER_ENDPOINT_FIELDS = frozenset({
    "LE0_P0", "LE0_P3", "LE1_P0", "LE1_P3",
    "TE1_P0", "TE1_P3", "TE2_Q0", "TE2_Q2", "TE3_Q0", "TE3_Q2",
})
DRAGGABLE_BEZIER_CONTROL_FIELDS = tuple(
    name for name in BezierControls.__dataclass_fields__
    if name not in FIXED_BEZIER_ENDPOINT_FIELDS
)


def build_initial_controls(lm: Landmarks) -> BezierControls:
    A, B, D = lm.A, lm.B, lm.D
    C5, C10, C20, C30 = lm.C5, lm.C10, lm.C20, lm.C30

    uAB = v_norm(v_sub(A, B))
    nAB = rot2(uAB, -math.pi / 2.0)

    BD = dist(B, D)
    uBD = v_norm(v_sub(D, B))
    LE0_P0, LE0_P3 = B, D
    LE0_P1 = v_add(B, v_add(v_mul(uBD, 0.25 * BD), v_mul(nAB, -0.08 * BD)))
    LE0_P2 = v_add(D, v_add(v_mul(uBD, -0.25 * BD), v_mul(nAB, -0.06 * BD)))

    DA = dist(D, A)
    uDA = v_norm(v_sub(A, D))
    nDA = rot2(uDA, -math.pi / 2.0)
    LE1_P0, LE1_P3 = D, A
    LE1_P1 = v_add(D, v_add(v_mul(uDA, 0.25 * DA), v_mul(nDA, -0.10 * DA)))
    LE1_P2 = v_add(A, v_add(v_mul(uDA, -0.20 * DA), v_mul(nDA, -0.07 * DA)))

    AC5 = dist(A, C5)
    uAC5 = v_norm(v_sub(C5, A))
    nAC5 = rot2(uAC5, math.pi / 2.0)
    TE1_P1 = v_add(A, v_add(v_mul(uAC5, 0.35 * AC5), v_mul(nAC5, 0.15 * AC5)))
    TE1_P2 = v_add(C5, v_add(v_mul(uAC5, -0.25 * AC5), v_mul(nAC5, 0.08 * AC5)))

    TE2_Q1 = (2 * C10[0] - 0.5 * (C5[0] + C20[0]), 2 * C10[1] - 0.5 * (C5[1] + C20[1]))

    mid = ((C20[0] + C30[0]) / 2.0, (C20[1] + C30[1]) / 2.0)
    u = v_norm(v_sub(C30, C20))
    n = rot2(u, math.pi / 2.0)
    seg = dist(C20, C30)
    TE3_Q1 = v_add(mid, v_mul(n, 0.08 * seg))

    return BezierControls(
        LE0_P0=LE0_P0, LE0_P1=LE0_P1, LE0_P2=LE0_P2, LE0_P3=LE0_P3,
        LE1_P0=D, LE1_P1=LE1_P1, LE1_P2=LE1_P2, LE1_P3=A,
        TE1_P0=A, TE1_P1=TE1_P1, TE1_P2=TE1_P2, TE1_P3=C5,
        TE2_Q0=C5, TE2_Q1=TE2_Q1, TE2_Q2=C20,
        TE3_Q0=C20, TE3_Q1=TE3_Q1, TE3_Q2=C30
    )


def curves_from_controls(ctrl: BezierControls, n_per_seg: int = 320) -> Dict[str, List[Tuple[float, float]]]:
    le0 = sample_curve(lambda t: bezier_cubic(ctrl.LE0_P0, ctrl.LE0_P1, ctrl.LE0_P2, ctrl.LE0_P3, t), n_per_seg)
    le1 = sample_curve(lambda t: bezier_cubic(ctrl.LE1_P0, ctrl.LE1_P1, ctrl.LE1_P2, ctrl.LE1_P3, t), n_per_seg)
    te1 = sample_curve(lambda t: bezier_cubic(ctrl.TE1_P0, ctrl.TE1_P1, ctrl.TE1_P2, ctrl.TE1_P3, t), n_per_seg)
    te2 = sample_curve(lambda t: bezier_quad(ctrl.TE2_Q0, ctrl.TE2_Q1, ctrl.TE2_Q2, t), n_per_seg)
    te3 = sample_curve(lambda t: bezier_quad(ctrl.TE3_Q0, ctrl.TE3_Q1, ctrl.TE3_Q2, t), n_per_seg)
    return {"LE0": le0, "LE1": le1, "TE1": te1, "TE2": te2, "TE3": te3}


def outline_polygon(ctrl: BezierControls, n_per_seg: int = 500) -> List[Tuple[float, float]]:
    c = curves_from_controls(ctrl, n_per_seg)
    le0 = c["LE0"]
    le1 = c["LE1"][1:]
    te = (c["TE1"][:-1] + c["TE2"][:-1] + c["TE3"])[1:]
    return le0 + le1 + te


def derived_O_point(B: Tuple[float, float], A: Tuple[float, float], C30: Tuple[float, float]) -> Tuple[float, float]:
    """Return the foot of C30 on the AB axis, i.e. the AB/C30-D construction intersection."""
    u_ab = v_norm(v_sub(A, B))
    c30_from_b = v_sub(C30, B)
    along_ab = c30_from_b[0] * u_ab[0] + c30_from_b[1] * u_ab[1]
    return v_add(B, v_mul(u_ab, along_ab))


def compute_weller_ratios(lm: Landmarks) -> Dict[str, float]:
    AB = dist(lm.A, lm.B)
    if AB < 1e-9:
        raise ValueError("AB length is zero.")
    return {
        "AB_px": AB,
        "C5B_over_AB": dist(lm.C5, lm.B) / AB,
        "C10B_over_AB": dist(lm.C10, lm.B) / AB,
        "C20B_over_AB": dist(lm.C20, lm.B) / AB,
        "C30B_over_AB": dist(lm.C30, lm.B) / AB,
        "DB_over_AB": dist(lm.D, lm.B) / AB,
    }


def clip_ray_to_rect(origin: Tuple[float, float], direction: Tuple[float, float], rect: Tuple[float, float, float, float]):
    ox, oy = origin
    dx, dy = direction
    xmin, ymin, xmax, ymax = rect
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return None
    t0, t1 = 0.0, float("inf")

    def clip(p, q):
        nonlocal t0, t1
        if abs(p) < 1e-12:
            return q >= 0
        r = q / p
        if p < 0:
            if r > t1:
                return False
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return False
            if r < t1:
                t1 = r
        return True

    if not clip(-dx, ox - xmin): return None
    if not clip(dx, xmax - ox):  return None
    if not clip(-dy, oy - ymin): return None
    if not clip(dy, ymax - oy):  return None
    if t1 < t0: return None
    p0 = (ox + t0*dx, oy + t0*dy)
    if math.isinf(t1): t1 = t0 + 1e6
    p1 = (ox + t1*dx, oy + t1*dy)
    return p0, p1


def rotate_about_origin(p: Tuple[float, float], ang: float) -> Tuple[float, float]:
    c, s = math.cos(ang), math.sin(ang)
    x, y = p
    return (c*x - s*y, s*x + c*y)


def standardize_contour_units(
    poly: List[Tuple[float, float]],
    B: Tuple[float, float],
    A: Tuple[float, float],
    C30: Tuple[float, float],
    ab_units: float = 10.0,
) -> List[Tuple[float, float]]:
    def rel_px(p):
        return (p[0] - B[0], p[1] - B[1])

    def to_yup(p):
        return (p[0], -p[1])

    poly_rel_yup = [to_yup(rel_px(p)) for p in poly]
    A_rel_yup = to_yup(rel_px(A))
    C30_rel_yup = to_yup(rel_px(C30))

    ang = -math.atan2(C30_rel_yup[1], C30_rel_yup[0])
    poly_rot = [rotate_about_origin(p, ang) for p in poly_rel_yup]
    A_rot = rotate_about_origin(A_rel_yup, ang)

    if A_rot[0] < 0:
        poly_rot = [(-p[0], p[1]) for p in poly_rot]
        A_rot = (-A_rot[0], A_rot[1])

    AB_px = v_len(A_rot)
    if AB_px < 1e-9:
        raise ValueError("AB length is zero.")

    scale = ab_units / AB_px
    poly_units = [(p[0] * scale, p[1] * scale) for p in poly_rot]
    return [(-x, -y) for (x, y) in poly_units]


def rotate_and_orient_swim_right(
    pts_rel_px: List[Tuple[float, float]],
    A_rel_px: Tuple[float, float],
    C30_rel_px: Tuple[float, float],
) -> Tuple[List[Tuple[float, float]], Tuple[float, float], Tuple[float, float]]:
    ang = -math.atan2(C30_rel_px[1], C30_rel_px[0])
    pts_r = [rotate_about_origin(p, ang) for p in pts_rel_px]
    A_r = rotate_about_origin(A_rel_px, ang)
    C30_r = rotate_about_origin(C30_rel_px, ang)
    if A_r[0] < 0:
        pts_r = [(-p[0], p[1]) for p in pts_r]
        A_r = (-A_r[0], A_r[1])
        C30_r = (-C30_r[0], C30_r[1])
    return pts_r, A_r, C30_r


class FinShape2DApp(tk.Tk):
    CSV_COLUMNS = [
        "image_name",
        "AB_px",
        "area_px2",
        "C5B_over_AB",
        "C10B_over_AB",
        "C20B_over_AB",
        "C30B_over_AB",
        "DB_over_AB",
        "A_dx", "A_dy",
        "C5_dx", "C5_dy",
        "C10_dx", "C10_dy",
        "C20_dx", "C20_dy",
        "C30_dx", "C30_dy",
        "D_dx", "D_dy",
    ]

    def __init__(self):
        super().__init__()
        self.title("FinShape2D R-GUI 2.0.20")
        self.geometry("1280x820")

        self.image_path: Optional[Path] = None
        self.pil_image = None
        self.tk_image: Optional[tk.PhotoImage] = None
        self.display_scale = 1.0

        self.crop_box: Optional[Tuple[int, int, int, int]] = None
        self.flip_x = False
        self.rotate_deg = 0.0

        self.landmarks: Dict[str, Tuple[float, float]] = {}
        self.controls: Optional[BezierControls] = None
        self.mode = "idle"
        self._drag_start = None
        self._crop_rect_id = None
        self._active_handle = None

        self.csv_path: Optional[Path] = None
        self.measurements: List[Dict[str, str]] = []

        self.zoom_half = 22
        self.zoom_w, self.zoom_h = 240, 200
        self.zoom_imgtk = None
        self.view_margin_px = 120


        self.locked_after_save: bool = False
        self.btn_apply_crop: Optional[tk.Button] = None
        self.lbl_image_name: Optional[tk.Label] = None

        self.btn_flip = None
        self.scale_rotate = None
        self.btn_crop = None
        self.btn_reset_crop = None
        self.btn_set_landmarks = None
        self.btn_save_meas = None
        self.btn_new_image = None
        self.btn_exit = None
        self._build_ui()
        self._setup_zoom()

        self._update_button_states()
    def _build_ui(self):
        bar = tk.Frame(self, bd=1, relief="raised")
        bar.pack(side="top", fill="x")

        row1 = tk.Frame(bar); row1.pack(side="top", fill="x")
        row2 = tk.Frame(bar); row2.pack(side="top", fill="x")

        tk.Label(row1, text="Edit image", font=("Helvetica", 12, "bold")).pack(side="left", padx=(10, 6))
        tk.Button(row1, text="1. Load Image", command=self.load_image).pack(side="left", padx=4, pady=3)
        self.btn_flip = tk.Button(row1, text="2. Flip ↔", command=self.toggle_flip)
        self.btn_flip.pack(side="left", padx=4, pady=3)

        tk.Label(row1, text="3. Rotate").pack(side="left", padx=(10, 0))
        self.rot_var = tk.DoubleVar(value=0.0)
        self.scale_rotate = tk.Scale(row1, from_=-30, to=30, orient="horizontal", resolution=0.5,
                 variable=self.rot_var, length=170, command=self.on_rotate_change)
        self.scale_rotate.pack(side="left", padx=4)

        self.btn_crop = tk.Button(row1, text="4. Crop", command=self.enter_crop)
        self.btn_crop.pack(side="left", padx=(10, 4), pady=3)

        self.btn_apply_crop = tk.Button(row1, text="5. Apply Crop", command=self.apply_crop, state="disabled")
        self.btn_apply_crop.pack(side="left", padx=4, pady=3)

        self.btn_reset_crop = tk.Button(row1, text="6. Reset Crop", command=self.reset_crop)
        self.btn_reset_crop.pack(side="left", padx=4, pady=3)

        self.lbl_image_name = tk.Label(row1, text="", anchor="e")
        self.lbl_image_name.pack(side="right", padx=10)

        tk.Label(row2, text="Morphometry", font=("Helvetica", 12, "bold")).pack(side="left", padx=(10, 6))
        self.btn_set_landmarks = tk.Button(row2, text="7. Set Landmarks", command=self.start_landmarks)
        self.btn_set_landmarks.pack(side="left", padx=4, pady=3)
        self.btn_save_meas = tk.Button(row2, text="8. Save Measurement (CSV)", command=self.save_measurement)
        self.btn_save_meas.pack(side="left", padx=4, pady=3)
        self.btn_show2d = tk.Button(row2, text="9. Show 2D", command=self.show_2d_window, state="disabled")
        self.btn_show2d.pack(side="left", padx=4, pady=3)
        self.btn_new_image = tk.Button(row2, text="10. Measure new image", command=self.measure_new_image)
        self.btn_new_image.pack(side="left", padx=14, pady=3)
        self.btn_exit = tk.Button(row2, text="Exit (Save CSV)", command=self.exit_save)
        self.btn_exit.pack(side="right", padx=10, pady=3)

        self.status = tk.StringVar(value="Load an image to begin.")
        tk.Label(self, textvariable=self.status, anchor="w").pack(side="bottom", fill="x")

        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(side="left", fill="both", expand=True)
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#222", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar = tk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.h_scrollbar = tk.Scrollbar(self.canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)

        side = tk.Frame(self, width=360); side.pack(side="right", fill="y")
        self.info = tk.Text(side, height=34, width=58)
        self.info.pack(side="top", fill="both", expand=True, padx=6, pady=6)
        self.info.insert("end",
                         "Instructions\n\n"
                         "• Zoom (lower-right): move mouse over image to see 5× zoom with stable rays.\n"
                         "• IMPORTANT: click using the TIP of the mouse cursor.\n\n"
                         "Edit image: Flip / Rotate / Crop.\n"
                         "  Apply Crop activates only after you draw a crop box.\n\n"
                         "Morphometry:\n"
                         "  Landmarks: B → A → C30 → C20 → C10 → C5 → D\n"
                         "  After D, Bézier control points appear automatically.\n"
                         "  Drag green control points to fit contour.\n"
                         "  Save Measurement updates the session CSV.\n"
                         "  Show 2D displays rotated contour (B→C30 horizontal) with axes (AB=10).\n")
        self.info.configure(state="disabled")

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self.on_shift_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)

    def _setup_zoom(self):
        self.zoom_canvas = tk.Canvas(self.canvas_frame, width=self.zoom_w, height=self.zoom_h, bg="black", highlightthickness=1)
        self.zoom_canvas.place(relx=1.0, rely=0.0, x=-28, y=28, anchor="ne")
        self.zoom_canvas.create_text(6, 6, anchor="nw", text="5× zoom", fill="white", font=("Helvetica", 10, "bold"))

    def _disable_2d(self):
        self.btn_show2d.configure(state="disabled")

    def _enable_2d(self):
        self.btn_show2d.configure(state="normal")


    def _update_button_states(self):
        """Enable/disable controls depending on whether measurement has been saved for current image."""
        locked = getattr(self, "locked_after_save", False)

        edit_state = "disabled" if locked else "normal"
        for w in [getattr(self, "btn_flip", None), getattr(self, "btn_crop", None),
                  getattr(self, "btn_reset_crop", None), getattr(self, "scale_rotate", None)]:
            if w is not None:
                try:
                    w.configure(state=edit_state)
                except Exception:
                    pass

        if self.btn_apply_crop is not None:
            self.btn_apply_crop.configure(state=("normal" if (self.crop_box is not None and not locked) else "disabled"))

        for w in [getattr(self, "btn_set_landmarks", None), getattr(self, "btn_save_meas", None)]:
            if w is not None:
                w.configure(state=("disabled" if locked else "normal"))

        if getattr(self, "btn_show2d", None) is not None:
            self.btn_show2d.configure(state=("normal" if locked else "disabled"))

        if getattr(self, "btn_new_image", None) is not None:
            self.btn_new_image.configure(state="normal")

        if getattr(self, "btn_exit", None) is not None:
            self.btn_exit.configure(state="normal")

    def measure_new_image(self):
        """Reset the workflow and prompt for a new image."""
        self._cancel_pending_zoom()
        self._working_image_cache = None
        self._display_image_cache = None
        self.image_path = None
        self.pil_image = None
        self.tk_image = None
        self.canvas.delete("all")

        self.crop_box = None
        self.flip_x = False
        self.rotate_deg = 0.0
        self.rot_var.set(0.0)

        self.landmarks.clear()
        self.controls = None
        self.mode = "idle"
        self._active_handle = None
        self._crop_rect_id = None
        self._drag_start = None

        self.locked_after_save = False
        self._disable_2d()

        if self.lbl_image_name is not None:
            self.lbl_image_name.configure(text="")

        self.status.set("Step 1: Load Image.")
        self._update_button_states()
        self.load_image()

    def _default_csv_name(self) -> str:
        if self.image_path is None:
            return "morphometry.csv"
        return f"{self.image_path.stem}_morphometry.csv"

    def _ensure_csv_path(self) -> Optional[Path]:
        if self.csv_path is not None:
            return self.csv_path
        p = filedialog.asksaveasfilename(
            title="Choose CSV file for all measurements",
            initialfile=self._default_csv_name(),
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not p:
            return None
        self.csv_path = Path(p)
        return self.csv_path

    def _write_csv(self):
        if self.csv_path is None:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            w.writeheader()
            for row in self.measurements:
                w.writerow(row)

    def load_image(self):
        p = filedialog.askopenfilename(
            title="Select dorsal fin image",
            filetypes=[("Images", "*.png *.gif *.ppm *.pgm *.jpg *.jpeg *.tif *.tiff"), ("All files", "*.*")]
        )
        if not p:
            return
        self.image_path = Path(p)
        self.locked_after_save = False
        if self.lbl_image_name is not None:
            self.lbl_image_name.configure(text=self.image_path.name)

        self.crop_box = None
        self.flip_x = False
        self.rotate_deg = 0.0
        self.rot_var.set(0.0)
        self.landmarks.clear()
        self.controls = None
        self.mode = "idle"
        self._active_handle = None
        self._crop_rect_id = None
        self._drag_start = None
        self._disable_2d()
        if self.btn_apply_crop is not None:
            self.btn_apply_crop.configure(state="disabled")

        try:
            self._load_image_backend(self.image_path)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return

        self._auto_fit_image()
        self.status.set("Image loaded. Edit image then Set Landmarks.")
        self._update_button_states()

    def _load_image_backend(self, path: Path):
        self._cancel_pending_zoom()
        self._working_image_cache = None
        self._display_image_cache = None
        ext = path.suffix.lower()
        if PIL_OK:
            self.pil_image = Image.open(path).convert("RGBA")
        else:
            if ext not in [".png", ".gif", ".ppm", ".pgm"]:
                raise RuntimeError("This image format needs Pillow. Install: pip install pillow")
            self.pil_image = None
            self.tk_image = tk.PhotoImage(file=str(path))
        self._render_image()

    def _apply_transforms(self, exclude_crop: bool = False):
        if not PIL_OK:
            raise RuntimeError("Transforms require Pillow (pip install pillow).")
        crop = None if exclude_crop else self.crop_box
        params = (self.flip_x, self.rotate_deg, crop)
        cached = self.__dict__.get("_working_image_cache")
        if cached is not None and cached[0] is self.pil_image and cached[1] == params:
            return cached[2]
        img = self.pil_image
        if self.flip_x:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if abs(self.rotate_deg) > 1e-6:
            img = img.rotate(self.rotate_deg, expand=True, resample=Image.BICUBIC)
        if crop is not None:
            img = img.crop(crop)
        self._working_image_cache = (self.pil_image, params, img)
        return img

    def on_canvas_resize(self, _evt=None):
        if self.image_path is None:
            return
        self._auto_fit_image()

    def _auto_fit_image(self):
        self.update_idletasks()
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        if PIL_OK:
            img = self._apply_transforms(exclude_crop=False)
            w, h = img.size
            self.display_scale = clamp(min(cw / max(w, 1), ch / max(h, 1)), 0.05, 8.0)
        else:
            self.display_scale = 1.0
        self._render_image()

    def _render_image(self):
        if self.image_path is None:
            self.canvas.delete("all")
            self._working_image_cache = None
            self._display_image_cache = None
            return
        if PIL_OK:
            img = self._apply_transforms(exclude_crop=False)
            w, h = img.size
            dw, dh = max(1, int(w * self.display_scale)), max(1, int(h * self.display_scale))
            cached = self.__dict__.get("_display_image_cache")
            if cached is None or cached[0] is not img or cached[1:] != (dw, dh):
                disp = img.resize((dw, dh), resample=Image.BILINEAR)
                self.tk_image = ImageTk.PhotoImage(disp)
                self._display_image_cache = (img, dw, dh)
        else:
            self.display_scale = 1.0
        if self.canvas.find_withtag("img"):
            self.canvas.itemconfigure("img", image=self.tk_image)
        else:
            self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw", tags=("img",))
        # Landmark edits invalidate overlays, not the unchanged photograph.
        self.canvas.delete("guide", "active_guide_label", "fitted_curve", "handle", "fixed_landmark", "fixed_label")
        if self._crop_rect_id is not None:
            self.canvas.delete(self._crop_rect_id)
            self._crop_rect_id = None
        self._draw_overlays()
        self._refresh_scrollregion()

    def toggle_flip(self):
        if getattr(self, 'locked_after_save', False):
            return
        if self.image_path is None:
            return
        self.flip_x = not self.flip_x
        self._disable_2d()
        self._auto_fit_image()

    def on_rotate_change(self, _=None):
        if getattr(self, 'locked_after_save', False):
            return
        if self.image_path is None:
            return
        self.rotate_deg = float(self.rot_var.get())
        self._disable_2d()
        self._auto_fit_image()

    def enter_crop(self):
        if getattr(self, 'locked_after_save', False):
            return
        if self.image_path is None:
            return
        self.mode = "crop"
        self.status.set("Crop: drag a rectangle (green). Apply Crop activates after selection.")

    def reset_crop(self):
        if getattr(self, 'locked_after_save', False):
            return
        self.crop_box = None
        self._crop_rect_id = None
        self._drag_start = None
        self._disable_2d()
        if self.btn_apply_crop is not None:
            self.btn_apply_crop.configure(state="disabled")
        self._auto_fit_image()
        self.status.set("Crop reset.")
        self._update_button_states()

    def apply_crop(self):
        if getattr(self, 'locked_after_save', False):
            return
        if self.image_path is None:
            return
        if not PIL_OK:
            messagebox.showwarning("Pillow required", "Cropping requires Pillow. Install: pip install pillow")
            return
        if self.crop_box is None:
            messagebox.showinfo("No crop", "No crop rectangle defined.")
            return
        self.mode = "idle"
        self._disable_2d()
        if self.btn_apply_crop is not None:
            self.btn_apply_crop.configure(state="disabled")
        self._auto_fit_image()
        self.status.set("Crop applied.")
        self._update_button_states()

    def _canvas_to_working(self, x: float, y: float) -> Tuple[float, float]:
        return (x / max(self.display_scale, 1e-9), y / max(self.display_scale, 1e-9))

    def _event_canvas_xy(self, event) -> Tuple[float, float]:
        return (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))

    def _refresh_scrollregion(self):
        bbox = self.canvas.bbox("img")
        if bbox is None:
            return
        xs = [float(bbox[0]), float(bbox[2])]
        ys = [float(bbox[1]), float(bbox[3])]

        for p in self.landmarks.values():
            xs.append(p[0] * self.display_scale)
            ys.append(p[1] * self.display_scale)

        if self.controls is not None:
            for field in DRAGGABLE_BEZIER_CONTROL_FIELDS:
                p = getattr(self.controls, field)
                xs.append(p[0] * self.display_scale)
                ys.append(p[1] * self.display_scale)
            for pts in curves_from_controls(self.controls, n_per_seg=80).values():
                for x, y in pts:
                    xs.append(x * self.display_scale)
                    ys.append(y * self.display_scale)

        m = self.view_margin_px
        self.canvas.configure(scrollregion=(min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m))

    def on_mouse_wheel(self, event):
        if getattr(event, "num", None) == 4:
            units = -3
        elif getattr(event, "num", None) == 5:
            units = 3
        else:
            units = -3 if getattr(event, "delta", 0) > 0 else 3
        self.canvas.yview_scroll(units, "units")

    def on_shift_mouse_wheel(self, event):
        units = -3 if getattr(event, "delta", 0) > 0 else 3
        self.canvas.xview_scroll(units, "units")

    def _ray_dirs_from_AB(self):
        if "A" not in self.landmarks or "B" not in self.landmarks:
            return None
        A = self.landmarks["A"]; B = self.landmarks["B"]
        uAB = v_norm(v_sub(A, B))
        def dir_deg(deg):
            return v_norm(rot2(uAB, math.radians(-deg)))
        return {"ray5": dir_deg(5.0), "ray10": dir_deg(10.0), "ray20": dir_deg(20.0), "ray30": dir_deg(30.0)}

    def _ray_dir_D_from_C30(self):
        if "A" not in self.landmarks or "B" not in self.landmarks or "C30" not in self.landmarks:
            return None
        A = self.landmarks["A"]; B = self.landmarks["B"]; C30 = self.landmarks["C30"]
        uAB = v_norm(v_sub(A, B))
        perp = rot2(uAB, math.pi / 2.0)
        if perp[0] < 0:
            perp = v_mul(perp, -1.0)
        return C30, v_norm(perp)

    def _draw_line_on_image(self, img, p0, p1, color, dash=False):
        x0, y0 = p0; x1, y1 = p1
        dx = abs(x1 - x0); dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        step = 0
        while True:
            if not dash or (step // 6) % 2 == 0:
                if 0 <= x0 < img.size[0] and 0 <= y0 < img.size[1]:
                    img.putpixel((x0, y0), color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy; x0 += sx
            if e2 <= dx:
                err += dx; y0 += sy
            step += 1

    def _cancel_pending_zoom(self):
        token = self.__dict__.get("_zoom_after_id")
        if token is not None:
            self.after_cancel(token)
        self._zoom_after_id = None
        self._pending_zoom_event = None

    def destroy(self):
        self._cancel_pending_zoom()
        super().destroy()

    def on_mouse_move(self, event):
        if not PIL_OK or self.image_path is None:
            return
        self._pending_zoom_event = SimpleNamespace(x=event.x, y=event.y)
        if self.__dict__.get("_zoom_after_id") is None:
            self._zoom_after_id = self.after(16, self._flush_pending_zoom)

    def _flush_pending_zoom(self):
        event = self.__dict__.get("_pending_zoom_event")
        self._zoom_after_id = None
        self._pending_zoom_event = None
        if event is not None and self.image_path is not None:
            self._render_zoom_at_pointer(event)

    def _render_zoom_at_pointer(self, event):
        if (not PIL_OK) or self.image_path is None:
            return
        img = self._apply_transforms(exclude_crop=False)
        cx_event, cy_event = self._event_canvas_xy(event)
        wx, wy = self._canvas_to_working(cx_event, cy_event)
        x, y = float(wx), float(wy)
        w, h = img.size

        half = self.zoom_half
        x0 = clamp(x - half, 0.0, w - 1.0)
        y0 = clamp(y - half, 0.0, h - 1.0)
        x1 = clamp(x + half, 1.0, float(w))
        y1 = clamp(y + half, 1.0, float(h))

        crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
        zoom = crop.resize((self.zoom_w, self.zoom_h), resample=Image.NEAREST).convert("RGBA")

        def map_pt(px, py):
            zx = (px - x0) / max((x1 - x0), 1e-9) * (self.zoom_w - 1)
            zy = (py - y0) / max((y1 - y0), 1e-9) * (self.zoom_h - 1)
            return int(clamp(zx, 0, self.zoom_w - 1)), int(clamp(zy, 0, self.zoom_h - 1))

        rect = (x0, y0, x1, y1)

        rays = self._ray_dirs_from_AB()
        if rays is not None and "B" in self.landmarks:
            B = self.landmarks["B"]
            zoom_rays = [v_norm(v_sub(self.landmarks["A"], B)), *rays.values()]
            for d in zoom_rays:
                seg = clip_ray_to_rect(B, d, rect)
                if seg:
                    p0, p1 = seg
                    self._draw_line_on_image(zoom, map_pt(*p0), map_pt(*p1), (255,234,0,255), dash=True)

        d_ray = self._ray_dir_D_from_C30()
        if d_ray is not None:
            origin, ddir = d_ray
            seg = clip_ray_to_rect(origin, ddir, rect)
            if seg:
                p0, p1 = seg
                self._draw_line_on_image(zoom, map_pt(*p0), map_pt(*p1), (0,230,118,255), dash=True)

        cx, cy = map_pt(x, y)
        for dx in range(-8, 9):
            zoom.putpixel((int(clamp(cx + dx, 0, self.zoom_w - 1)), cy), (255,64,64,255))
        for dy in range(-8, 9):
            zoom.putpixel((cx, int(clamp(cy + dy, 0, self.zoom_h - 1))), (255,64,64,255))

        self.zoom_imgtk = ImageTk.PhotoImage(zoom)
        self.zoom_canvas.delete("zoomimg")
        self.zoom_canvas.create_image(0, 0, image=self.zoom_imgtk, anchor="nw", tags=("zoomimg",))
        try:
            self.zoom_canvas.tag_raise('zoomimg')
        except Exception:
            pass

    def start_landmarks(self):
        if getattr(self, 'locked_after_save', False):
            return
        if self.image_path is None:
            return
        self.landmarks.clear()
        self.controls = None
        self.mode = "lm_B"
        self.landmark_placement_notes = []
        self._drag_start = None
        self._active_handle = None
        self._disable_2d()
        self.status.set("Landmarks: click B.")
        self._render_image()

    def save_measurement(self):
        if self.controls is None:
            messagebox.showwarning("Not ready", "Finish landmarks first (controls appear after D), then adjust Bézier.")
            return
        required = ["B","A","C30","C20","C10","C5","D"]
        if any(k not in self.landmarks for k in required):
            messagebox.showwarning("Missing landmarks", "Some landmarks are missing; restart 'Set Landmarks'.")
            return
        if self._ensure_csv_path() is None:
            return

        lm = Landmarks(
            A=self.landmarks["A"], B=self.landmarks["B"],
            C5=self.landmarks["C5"], C10=self.landmarks["C10"],
            C20=self.landmarks["C20"], C30=self.landmarks["C30"],
            D=self.landmarks["D"]
        )
        ratios = compute_weller_ratios(lm)
        poly = outline_polygon(self.controls, n_per_seg=900)
        area_px2 = shoelace_area(poly)

        Bx, By = lm.B
        def rel(p): return (p[0]-Bx, p[1]-By)

        row: Dict[str, str] = {k: "" for k in self.CSV_COLUMNS}
        row["image_name"] = self.image_path.name if self.image_path else ""
        row["AB_px"] = f"{ratios['AB_px']:.6f}"
        row["area_px2"] = f"{area_px2:.6f}"
        row["C5B_over_AB"]  = f"{ratios['C5B_over_AB']:.10f}"
        row["C10B_over_AB"] = f"{ratios['C10B_over_AB']:.10f}"
        row["C20B_over_AB"] = f"{ratios['C20B_over_AB']:.10f}"
        row["C30B_over_AB"] = f"{ratios['C30B_over_AB']:.10f}"
        row["DB_over_AB"]   = f"{ratios['DB_over_AB']:.10f}"

        Ax, Ay = rel(lm.A); row["A_dx"], row["A_dy"] = f"{Ax:.6f}", f"{Ay:.6f}"
        x, y = rel(lm.C5);  row["C5_dx"], row["C5_dy"] = f"{x:.6f}", f"{y:.6f}"
        x, y = rel(lm.C10); row["C10_dx"], row["C10_dy"] = f"{x:.6f}", f"{y:.6f}"
        x, y = rel(lm.C20); row["C20_dx"], row["C20_dy"] = f"{x:.6f}", f"{y:.6f}"
        x, y = rel(lm.C30); row["C30_dx"], row["C30_dy"] = f"{x:.6f}", f"{y:.6f}"
        x, y = rel(lm.D);   row["D_dx"], row["D_dy"] = f"{x:.6f}", f"{y:.6f}"

        self.measurements.append(row)
        self._write_csv()
        self.locked_after_save = True
        self._update_button_states()
        self.status.set(f"Saved to CSV ({len(self.measurements)} row(s)). Editing locked for this image.")

    def exit_save(self):
        if self.csv_path is None and self.measurements:
            if self._ensure_csv_path() is None:
                return
        if self.csv_path is not None:
            self._write_csv()
        self.destroy()

    def show_2d_window(self):
        if not getattr(self, 'locked_after_save', False):
            messagebox.showinfo('Not ready', 'Save Measurement (CSV) first.')
            return
        if self.controls is None or "B" not in self.landmarks or "A" not in self.landmarks or "C30" not in self.landmarks:
            messagebox.showwarning("Not ready", "Need fitted controls and landmarks B, A, C30.")
            return

        poly = outline_polygon(self.controls, n_per_seg=1100)
        B = self.landmarks["B"]
        A = self.landmarks["A"]
        C30 = self.landmarks["C30"]

        try:
            poly_units = standardize_contour_units(poly, B, A, C30, ab_units=10.0)
        except ValueError as exc:
            messagebox.showwarning("Invalid", str(exc))
            return
        win = tk.Toplevel(self)
        win.title("2D Contour (FinShape2D R-GUI 2.0.20)")
        win.geometry("980x880")

        top = tk.Frame(win, bd=1, relief="raised"); top.pack(side="top", fill="x")
        tk.Label(top,
                 text=f"{self.image_path.name if self.image_path else ''} | Rotated: B→C30 horizontal | Scaled: |AB|=10",
                 anchor="w").pack(side="left", padx=8, pady=4)

        tk.Button(top, text="Save contour…", command=lambda: self.save_contour_dialog(poly_units)).pack(side="right", padx=8, pady=4)

        cv = tk.Canvas(win, bg="white"); cv.pack(side="top", fill="both", expand=True)

        def fit_points_yup(pts, W, H, pad=90):
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
            sx = (W - 2*pad) / max(maxx - minx, 1e-9)
            sy = (H - 2*pad) / max(maxy - miny, 1e-9)
            sc = min(sx, sy)

            def tx(p):
                x, y = p
                cx = pad + (x - minx)*sc
                cy = pad + (maxy - y)*sc
                return (cx, cy)

            return [tx(p) for p in pts], (minx, miny, maxx, maxy, sc, pad)

        def draw_axes(W, H, bbox):
            minx, miny, maxx, maxy, sc, pad = bbox
            ox = pad + (0.0 - minx) * sc
            oy = pad + (maxy - 0.0) * sc
            cv.create_line(0, oy, W, oy, fill="#666", width=1)
            cv.create_line(ox, 0, ox, H, fill="#666", width=1)

            tick = 1.0
            k0 = int(math.floor(minx / tick)) - 1
            k1 = int(math.ceil(maxx / tick)) + 1
            for k in range(k0, k1 + 1):
                x = pad + (k*tick - minx) * sc
                cv.create_line(x, oy-4, x, oy+4, fill="#666", width=1)
                if k % 2 == 0:
                    cv.create_text(x, oy+14, text=str(k), fill="#444", font=("Helvetica", 9))

            k0 = int(math.floor(miny / tick)) - 1
            k1 = int(math.ceil(maxy / tick)) + 1
            for k in range(k0, k1 + 1):
                y = pad + (maxy - k*tick) * sc
                cv.create_line(ox-4, y, ox+4, y, fill="#666", width=1)
                if k % 2 == 0:
                    cv.create_text(ox-14, y, text=str(k), fill="#444", font=("Helvetica", 9))

            cv.create_text(W-10, oy-10, anchor="ne", text="x (AB=10 units)", fill="#444", font=("Helvetica", 10, "bold"))
            cv.create_text(ox+10, 10, anchor="nw", text="y (AB=10 units)", fill="#444", font=("Helvetica", 10, "bold"))

        def redraw(_=None):
            cv.delete("all")
            W = max(1, cv.winfo_width()); H = max(1, cv.winfo_height())
            pts, bbox = fit_points_yup(poly_units, W, H, pad=90)
            draw_axes(W, H, bbox)
            for i in range(len(pts)-1):
                cv.create_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], fill="black", width=2)
            cv.create_line(pts[-1][0], pts[-1][1], pts[0][0], pts[0][1], fill="black", width=2)

        cv.bind("<Configure>", redraw)
        redraw()

    def save_contour_dialog(self, poly_units: List[Tuple[float, float]]):
        """Save the 2D contour. SVG is always supported; PNG/JPG/BMP require Pillow but do NOT require Ghostscript."""
        stem = self.image_path.stem if self.image_path else "contour"
        filetypes = [("SVG", "*.svg")]
        if PIL_OK:
            filetypes += [("PNG", "*.png"), ("JPG", "*.jpg"), ("BMP", "*.bmp")]
        filetypes += [("PostScript", "*.ps")]

        out = filedialog.asksaveasfilename(
            title="Save contour image",
            initialfile=f"{stem}_contour.svg",
            defaultextension=".svg",
            filetypes=filetypes,
        )
        if not out:
            return
        out_path = Path(out)
        ext = out_path.suffix.lower()

        if ext == ".svg":
            self._save_svg(poly_units, out_path)
            messagebox.showinfo("Saved", f"Saved: {out_path.name}")
            return

        if ext in [".png", ".jpg", ".bmp"]:
            if not PIL_OK:
                messagebox.showwarning("Pillow missing", "Raster export requires Pillow (pip install pillow).")
                return
            self._save_raster_contour(poly_units, out_path, ext)
            messagebox.showinfo("Saved", f"Saved: {out_path.name}")
            return

        # Vector PostScript
        self._save_ps_contour(poly_units, out_path)
        messagebox.showinfo("Saved", f"Saved: {out_path.name}")

    def _save_raster_contour(self, poly_units: List[Tuple[float, float]], out_path: Path, ext: str):
        """Direct raster rendering using Pillow (no PostScript/Ghostscript)."""
        W, H = 1400, 1400
        pad = 140
        img = Image.new("RGB", (W, H), "white")
        dr = ImageDraw.Draw(img)

        xs = [p[0] for p in poly_units]; ys = [p[1] for p in poly_units]
        minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
        sx = (W - 2*pad) / max(maxx - minx, 1e-9)
        sy = (H - 2*pad) / max(maxy - miny, 1e-9)
        sc = min(sx, sy)

        def tx(p):
            x, y = p
            cx = pad + (x - minx) * sc
            cy = pad + (maxy - y) * sc  # y-up to canvas
            return (cx, cy)

        pts = [tx(p) for p in poly_units]

        ox = pad + (0.0 - minx) * sc
        oy = pad + (maxy - 0.0) * sc
        dr.line([(0, oy), (W, oy)], fill=(90, 90, 90), width=3)
        dr.line([(ox, 0), (ox, H)], fill=(90, 90, 90), width=3)

        for i in range(len(pts)-1):
            dr.line([pts[i], pts[i+1]], fill=(0, 0, 0), width=4)
        dr.line([pts[-1], pts[0]], fill=(0, 0, 0), width=4)

        fmt = "PNG" if ext == ".png" else ("JPEG" if ext == ".jpg" else "BMP")
        img.save(str(out_path), fmt)

    def _save_ps_contour(self, poly_units: List[Tuple[float, float]], out_path: Path):
        """Vector PostScript export using Tk canvas."""
        W, H = 1200, 1200
        pad = 120
        tmp = tk.Toplevel(self)
        tmp.withdraw()
        cv = tk.Canvas(tmp, width=W, height=H, bg="white")
        cv.pack()

        xs = [p[0] for p in poly_units]; ys = [p[1] for p in poly_units]
        minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
        sx = (W - 2*pad) / max(maxx - minx, 1e-9)
        sy = (H - 2*pad) / max(maxy - miny, 1e-9)
        sc = min(sx, sy)

        def tx(p):
            x, y = p
            cx = pad + (x - minx) * sc
            cy = pad + (maxy - y) * sc
            return (cx, cy)

        pts = [tx(p) for p in poly_units]

        ox = pad + (0.0 - minx) * sc
        oy = pad + (maxy - 0.0) * sc
        cv.create_line(0, oy, W, oy, fill="#666", width=3)
        cv.create_line(ox, 0, ox, H, fill="#666", width=3)
        for i in range(len(pts)-1):
            cv.create_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], fill="black", width=4)
        cv.create_line(pts[-1][0], pts[-1][1], pts[0][0], pts[0][1], fill="black", width=4)

        cv.postscript(file=str(out_path), colormode="color")
        tmp.destroy()

    def _save_svg(self, poly_units: List[Tuple[float, float]], out_path: Path):
        W, H = 1000, 1000
        pad = 100
        xs = [p[0] for p in poly_units]; ys = [p[1] for p in poly_units]
        minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
        sx = (W - 2*pad) / max(maxx - minx, 1e-9)
        sy = (H - 2*pad) / max(maxy - miny, 1e-9)
        sc = min(sx, sy)

        def tx(p):
            x, y = p
            cx = pad + (x - minx)*sc
            cy = pad + (maxy - y)*sc
            return (cx, cy)

        pts = [tx(p) for p in poly_units]
        pts_str = " ".join([f"{x:.2f},{y:.2f}" for x, y in pts])

        ox = pad + (0.0 - minx) * sc
        oy = pad + (maxy - 0.0) * sc

        svg = []
        svg.append('<?xml version="1.0" encoding="UTF-8"?>')
        svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
        svg.append('<rect width="100%" height="100%" fill="white"/>')
        svg.append(f'<line x1="0" y1="{oy:.2f}" x2="{W}" y2="{oy:.2f}" stroke="#666" stroke-width="2"/>')
        svg.append(f'<line x1="{ox:.2f}" y1="0" x2="{ox:.2f}" y2="{H}" stroke="#666" stroke-width="2"/>')
        svg.append(f'<polyline points="{pts_str}" fill="none" stroke="black" stroke-width="3"/>')
        svg.append(f'<line x1="{pts[-1][0]:.2f}" y1="{pts[-1][1]:.2f}" x2="{pts[0][0]:.2f}" y2="{pts[0][1]:.2f}" stroke="black" stroke-width="3"/>')
        svg.append('</svg>')
        out_path.write_text("\n".join(svg), encoding="utf-8")

    def _confirm_landmark_click(self, point):
        """Validate clicks without snapping or changing the measurement convention."""
        name = self.mode.removeprefix("lm_")
        if PIL_OK:
            width, height = self._apply_transforms().size
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                self.status.set(f"Landmark {name}: click inside the displayed image.")
                return False
        if any(dist(point, p) < 1e-9 for p in self.landmarks.values()):
            self.status.set(f"Landmark {name}: choose a distinct point; coincident landmarks are not accepted.")
            return False
        rays = self._ray_dirs_from_AB()
        if rays is None or name not in ("C5", "C10", "C20", "C30"):
            return True
        expected = "ray" + name[1:]
        delta = v_sub(point, self.landmarks["B"])
        # Six displayed pixels is a UI hit tolerance, not a biological tolerance.
        distances = {
            key: abs(delta[0] * direction[1] - delta[1] * direction[0]) * self.display_scale
            for key, direction in rays.items()
            if delta[0] * direction[0] + delta[1] * direction[1] > 0
        }
        if not distances:
            return True
        nearest = min(distances, key=distances.get)
        if nearest != expected and distances[nearest] <= 6 and distances.get(expected, float("inf")) > 6:
            other = "C" + nearest.removeprefix("ray")
            accepted = messagebox.askyesno(
                "Check landmark guide",
                f"Placing {name}, but this click is near the B-{other} guide.\n"
                f"Use the highlighted {name} guide for the standard construction.\n\n"
                "Keep this exact off-guide point anyway? No coordinates will be snapped.",
                parent=self, default="no",
            )
            if not accepted:
                self.status.set(f"Landmark {name} not placed. Click on its highlighted guide at the fin edge.")
                return False
            notes = self.__dict__.setdefault("landmark_placement_notes", [])
            notes.append({"landmark": name, "point": list(point), "near_guide": other,
                          "operator_accepted_off_guide": True, "ui_tolerance_display_pixels": 6})
        return True

    def on_mouse_down(self, event):
        if self.image_path is None or getattr(self, 'locked_after_save', False):
            return
        x, y = self._event_canvas_xy(event)

        if self.mode == "crop":
            self._drag_start = (x, y)
            if self._crop_rect_id is not None:
                self.canvas.delete(self._crop_rect_id)
            self._crop_rect_id = self.canvas.create_rectangle(x, y, x, y, outline="#00e676", width=2)
            return

        if self.controls is not None and self.mode == "bezier":
            item = self.canvas.find_withtag("current")
            if item:
                tags = self.canvas.gettags(item)
                for t in tags:
                    if t in DRAGGABLE_BEZIER_CONTROL_FIELDS:
                        self._active_handle = t
                        self._drag_start = (x, y)
                        return

        wx, wy = self._canvas_to_working(x, y)
        if self.mode.startswith("lm_") and not self._confirm_landmark_click((wx, wy)):
            return

        if self.mode == "lm_B":
            self.landmarks["B"] = (wx, wy); self.mode = "lm_A"; self.status.set("Landmarks: click A."); self._render_image()
        elif self.mode == "lm_A":
            self.landmarks["A"] = (wx, wy); self.mode = "lm_C30"; self.status.set("Landmarks: click C30."); self._render_image()
        elif self.mode == "lm_C30":
            self.landmarks["C30"] = (wx, wy); self.mode = "lm_C20"; self.status.set("Landmarks: click C20."); self._render_image()
        elif self.mode == "lm_C20":
            self.landmarks["C20"] = (wx, wy); self.mode = "lm_C10"; self.status.set("Landmarks: click C10."); self._render_image()
        elif self.mode == "lm_C10":
            self.landmarks["C10"] = (wx, wy); self.mode = "lm_C5"; self.status.set("Landmarks: click C5."); self._render_image()
        elif self.mode == "lm_C5":
            self.landmarks["C5"] = (wx, wy); self.mode = "lm_D"; self.status.set("Landmarks: click D."); self._render_image()
        elif self.mode == "lm_D":
            self.landmarks["D"] = (wx, wy)
            lm = Landmarks(
                A=self.landmarks["A"], B=self.landmarks["B"],
                C5=self.landmarks["C5"], C10=self.landmarks["C10"],
                C20=self.landmarks["C20"], C30=self.landmarks["C30"],
                D=self.landmarks["D"]
            )
            self.controls = build_initial_controls(lm)
            self.mode = "bezier"
            self.status.set("Landmarks complete. Drag green control points, then Save Measurement (CSV).")
            self._render_image()

    def on_mouse_drag(self, event):
        if self.image_path is None or getattr(self, 'locked_after_save', False):
            return
        x, y = self._event_canvas_xy(event)

        if self.mode == "crop" and self._drag_start is not None and self._crop_rect_id is not None:
            x0, y0 = self._drag_start
            self.canvas.coords(self._crop_rect_id, x0, y0, x, y)
            return

        if self.controls is not None and self._active_handle in DRAGGABLE_BEZIER_CONTROL_FIELDS and self._drag_start is not None:
            dx = (x - self._drag_start[0]) / max(self.display_scale, 1e-9)
            dy = (y - self._drag_start[1]) / max(self.display_scale, 1e-9)
            self._drag_start = (x, y)
            p = getattr(self.controls, self._active_handle)
            setattr(self.controls, self._active_handle, (p[0] + dx, p[1] + dy))
            self._disable_2d()
            self._render_image()

    def on_mouse_up(self, event):
        if self.image_path is None:
            return

        if self.mode == "crop" and self._crop_rect_id is not None:
            if not PIL_OK:
                return
            img_t = self._apply_transforms(exclude_crop=True)
            tw, th = img_t.size
            x0, y0, x1, y1 = self.canvas.coords(self._crop_rect_id)
            sx = 1.0 / max(self.display_scale, 1e-9)
            x0i, y0i = int(min(x0, x1) * sx), int(min(y0, y1) * sx)
            x1i, y1i = int(max(x0, x1) * sx), int(max(y0, y1) * sx)

            x0i = int(clamp(x0i, 0, tw - 1)); y0i = int(clamp(y0i, 0, th - 1))
            x1i = int(clamp(x1i, 1, tw));     y1i = int(clamp(y1i, 1, th))

            if abs(x1i - x0i) > 5 and abs(y1i - y0i) > 5:
                self.crop_box = (x0i, y0i, x1i, y1i)
                if self.btn_apply_crop is not None:
                    self.btn_apply_crop.configure(state="normal")
                self._update_button_states()
                self.status.set("Crop box set. Click 'Apply Crop' to finalize.")
            self._drag_start = None
            return

        self._active_handle = None
        self._drag_start = None

    def _draw_overlays(self):
        rays = self._ray_dirs_from_AB()
        if rays is not None:
            B, A = self.landmarks["B"], self.landmarks["A"]
            self.canvas.create_line(
                *(v * self.display_scale for p in (B, A) for v in p),
                fill="#ffea00", width=2, tags=("guide", "guide_AB"),
            )
            for key, direction in rays.items():
                name = "C" + key.removeprefix("ray")
                active = self.mode == "lm_" + name
                end = v_add(B, v_mul(direction, 2400))
                self.canvas.create_line(
                    *(v * self.display_scale for p in (B, end) for v in p),
                    fill="#ffea00", width=3 if active else 1, dash=(4, 4),
                    tags=("guide", "guide_" + name),
                )
                if active:
                    label = v_add(B, v_mul(direction, dist(B, A) * 0.55))
                    self.canvas.create_text(
                        label[0] * self.display_scale + 10, label[1] * self.display_scale - 10,
                        text=name, fill="#ffea00", anchor="sw", tags=("active_guide_label",),
                    )
        d_ray = self._ray_dir_D_from_C30()
        if d_ray is not None:
            origin, direction = d_ray
            end = v_add(origin, v_mul(direction, 2400))
            self.canvas.create_line(
                *(v * self.display_scale for p in (origin, end) for v in p),
                fill="#00e676", width=2 if self.mode == "lm_D" else 1, dash=(6, 4),
                tags=("guide", "guide_D"),
            )
        if self.controls is not None:
            curves = curves_from_controls(self.controls, n_per_seg=240)
            for name, points in curves.items():
                self.canvas.create_line(
                    *(v * self.display_scale for p in points for v in p),
                    fill="#00ff6a" if name.startswith("LE") else "#ff5252",
                    width=2, tags=("fitted_curve", name),
                )
            for field in DRAGGABLE_BEZIER_CONTROL_FIELDS:
                cx, cy = (v * self.display_scale for v in getattr(self.controls, field))
                self.canvas.create_rectangle(
                    cx - 4, cy - 4, cx + 4, cy + 4, fill="#2e7d32", outline="",
                    tags=("handle", field),
                )
        # Fixed landmarks are drawn last so curves or coincident handles cannot hide them.
        fixed_points = dict(self.landmarks)
        if all(k in self.landmarks for k in ("A", "B", "C30")):
            fixed_points["O"] = derived_O_point(self.landmarks["B"], self.landmarks["A"], self.landmarks["C30"])
        for name, point in fixed_points.items():
            cx, cy = (v * self.display_scale for v in point)
            self.canvas.create_oval(
                cx - 5, cy - 5, cx + 5, cy + 5, fill="#e53935", outline="",
                tags=("fixed_landmark", "landmark_" + name),
            )
            self.canvas.create_text(
                cx + 10, cy - 10, text=name, fill="#e53935",
                tags=("fixed_label", "label_" + name),
            )


def main():
    app = FinShape2DApp()
    if not PIL_OK:
        messagebox.showwarning(
            "Optional dependency missing",
            "Pillow is not installed.\n\n"
            "You can still load PNG/GIF/PPM/PGM.\n"
            "Flip/Rotate/Crop accuracy + Zoom + JPEG require Pillow.\n\n"
            "Install Pillow:\n    pip install pillow"
        )
    app.mainloop()


if __name__ == "__main__":
    main()


