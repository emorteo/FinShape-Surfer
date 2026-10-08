#!/usr/bin/env python3
"""FinShape Surfer 1.3.13 with optional, operator-reviewed FinFindR assistance.

The frozen 1.2.2 application remains authoritative for manual measurement,
2D/3D reconstruction, Photo-ID rendering, and experimental CFD export. This
subclass adds asynchronous crop and full Photo-ID contour suggestions; it never
places landmarks or changes Bezier controls automatically.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
import threading
import weakref
from dataclasses import replace
from pathlib import Path

import numpy as np

from core.rendering.airfoil_surface import (
    MAX_FULL_THICKNESS_RATIO,
    airfoil_full_thickness,
    airfoil_max_full_thickness,
    invert_airfoil_thickness,
    rebuild_texture_scene_airfoil,
)
from core.rendering.photo_display import (
    DISPLAY_THICKNESS_SCALE,
    MOTION_OUTLINE_TOLERANCE,
    draw_photo_axis_guide,
    render_smooth_photo_frame,
    simplified_motion_outline,
)
from core.rendering.photo_mask import PhotoMaskTextureProxy

SURFER_VERSION = "1.3.13"
ASSISTED_EDGE_COLOR = "#ff8c00"
CONTROL_HIT_RADIUS_PX = 14.0
IMAGE_ALGORITHM_CREDIT = (
    "Image algorithms: FinShape2D R-GUI 2.0.20, Eduardo Morteo & Rodrigo Morteo"
)
MORPHOMETRY_CREDIT = (
    "Ratios by David Weller & Eduardo Morteo, in: Morteo et al. (2017)"
)
INTERACTIVE_PHOTO_MAX_PIXELS = 420_000
INTERACTIVE_MOTION_MAX_PIXELS = 350_000
MOTION_PREVIEW_SUPERSAMPLE = 2
MOTION_PREVIEW_SPAN_STATIONS = 41
MOTION_PREVIEW_CHORD_STATIONS = 9
MOTION_PREVIEW_BOUNDARY_MIN_TRIANGLES = 300
MOTION_RENDER_DELAY_MS = 35
TIP_CAP_START_FRACTION = 0.84
TIP_CAP_RING_PARAMETERS = (0.20, 0.40, 0.60, 0.80)
ROOT = Path(__file__).resolve().parent
BASELINE_GUI_PATH = (
    ROOT
    / "baselines"
    / "FinShape_Surfer_1.2.2"
    / "Code"
    / "FinShape_Surfer_GUI_1.2.2.py"
)


def _load_module(module_name: str, path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


baseline = _load_module("finshape_surfer_gui_1_2_2_frozen", BASELINE_GUI_PATH)
baseline.INTEGRATED_VERSION = SURFER_VERSION
baseline.base.INTEGRATED_VERSION = SURFER_VERSION
baseline.base.base.INTEGRATED_VERSION = SURFER_VERSION
baseline.base.base.base.INTEGRATED_VERSION = SURFER_VERSION
fin2d = baseline.fin2d
_FROZEN_CURVES_FROM_CONTROLS = fin2d.curves_from_controls


def _points_sha256(points: np.ndarray) -> str:
    canonical = np.round(np.asarray(points, dtype=np.float64), 6).astype("<f8", copy=False)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _nearest_control_field(controls, fields, working_point, display_scale, max_distance_px):
    """Return the nearest draggable control independently of canvas stacking order."""

    if controls is None:
        return None
    px_per_unit = max(float(display_scale), 1e-9)
    distances = {
        field: math.hypot(
            (getattr(controls, field)[0] - working_point[0]) * px_per_unit,
            (getattr(controls, field)[1] - working_point[1]) * px_per_unit,
        )
        for field in fields
    }
    if not distances:
        return None
    nearest = min(distances, key=distances.get)
    return nearest if distances[nearest] <= float(max_distance_px) else None


def _anchored_te2_curve(p0, control, anchor, p2, n_samples: int):
    """Sample a free, C1-continuous C5-C10-C20 curve anchored at C10.

    The operator moves one unconstrained handle.  Its reflection about C10 is
    the second quadratic handle, so the two segments meet at the fixed
    landmark with the same tangent instead of confining the handle to an
    artificial one-dimensional locus.
    """

    points = tuple(np.asarray(value, dtype=float) for value in (p0, control, anchor, p2))
    if any(value.shape != (2,) for value in points):
        raise ValueError("Bezier points must be finite two-dimensional coordinates.")
    if not np.isfinite(np.concatenate(points)).all():
        raise ValueError("Bezier points must be finite two-dimensional coordinates.")
    p0, control, anchor, p2 = points
    reflected = 2.0 * anchor - control
    count = max(5, int(n_samples))

    def quadratic(a, b, c, values):
        values = values[:, None]
        return (1.0 - values) ** 2 * a + 2.0 * (1.0 - values) * values * b + values ** 2 * c

    parameter = np.linspace(0.0, 1.0, count)
    left_mask = parameter <= 0.5
    curve = np.empty((count, 2), dtype=float)
    curve[left_mask] = quadratic(p0, control, anchor, 2.0 * parameter[left_mask])
    curve[~left_mask] = quadratic(
        anchor, reflected, p2, 2.0 * parameter[~left_mask] - 1.0
    )
    return curve


def _surfer_curves_from_controls(ctrl, n_per_seg: int = 320):
    """Use the C10-anchored extension only for controls owned by this GUI."""

    curves = _FROZEN_CURVES_FROM_CONTROLS(ctrl, n_per_seg=n_per_seg)
    anchor = getattr(ctrl, "_surfer_c10_anchor", None)
    if anchor is None:
        return curves
    te2 = _anchored_te2_curve(
        ctrl.TE2_Q0, ctrl.TE2_Q1, anchor, ctrl.TE2_Q2, n_per_seg
    )
    curves["TE2"] = [tuple(map(float, point)) for point in te2]
    return curves


def _prepare_c10_anchored_control(ctrl, anchor) -> None:
    """Convert the frozen TE2 control once without changing its initial curve.

    De Casteljau subdivision at t=0.5 turns the original C5-C20 quadratic
    control into the left segment's control. Its reflection around C10 is the
    right segment's control. The resulting two-part curve is exactly the same
    quadratic until the operator moves the now-unrestricted handle.
    """

    anchor = tuple(map(float, anchor))
    if not hasattr(ctrl, "_surfer_c10_anchor"):
        p0 = np.asarray(ctrl.TE2_Q0, dtype=float)
        original = np.asarray(ctrl.TE2_Q1, dtype=float)
        ctrl.TE2_Q1 = tuple(map(float, 0.5 * (p0 + original)))
    ctrl._surfer_c10_anchor = anchor


# Frozen controls have no marker and therefore retain byte-for-byte baseline
# behavior. Controls created by the active GUI receive the marker in
# ``_render_image`` before any drawing, saving, or export path uses them.
fin2d.curves_from_controls = _surfer_curves_from_controls


def _draw_assisted_edge_3d(axis, edge_points) -> bool:
    """Draw the review-only A-to-C30 suggestion in the wireframe view."""

    points = np.asarray(edge_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        return False
    if not np.isfinite(points).all():
        return False
    axis.plot(
        points[:, 0],
        np.zeros(len(points)),
        points[:, 1],
        color=ASSISTED_EDGE_COLOR,
        linewidth=1.2,
        linestyle="--",
    )
    axis.text2D(
        0.01,
        0.02,
        "Orange: FinFindR A-to-C30 review suggestion; measured wireframe unchanged",
        transform=axis.transAxes,
        fontsize=8,
        color="#a65300",
    )
    return True


def _fit_wireframe_axes(axis) -> None:
    """Use the available canvas while keeping axes typography subordinate."""

    title_size = float(axis.title.get_fontsize())
    label_size = min(10.0, title_size)
    tick_size = min(8.0, title_size)
    axis.xaxis.label.set_fontsize(label_size)
    axis.yaxis.label.set_fontsize(label_size)
    axis.zaxis.label.set_fontsize(label_size)
    axis.xaxis.labelpad = 6
    axis.yaxis.labelpad = 8
    axis.zaxis.labelpad = 6
    axis.tick_params(labelsize=tick_size, pad=1)
    xlim, ylim, zlim = axis.get_xlim3d(), axis.get_ylim3d(), axis.get_zlim3d()
    spans = tuple(max(abs(bounds[1] - bounds[0]), 1e-9) for bounds in (xlim, ylim, zlim))
    axis.figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.96)
    axis.set_position((0.0, 0.0, 1.0, 0.94), which="both")
    axis.set_box_aspect(spans, zoom=1.18)


def _camera_drag_value(start, displacement_px, lower, upper):
    """Map direct canvas movement to the intentionally reversed GUI scales."""

    return max(lower, min(upper, start - displacement_px * 0.35))


def _readable_rail_fractions(n_chord_points: int, max_rails: int = 15) -> np.ndarray:
    """Return uniformly spaced display rails without changing mesh resolution."""

    available = max(3, int(n_chord_points))
    count = max(3, min(15, int(max_rails), available))
    if count % 2 == 0:
        count -= 1
    return np.linspace(0.0, 1.0, count)


def _clip_rail_at_y(rail, target_y: float):
    """End one monotonic rail at an interpolated existing-surface position."""

    xs, ys, z_up, z_dn = (np.asarray(values, dtype=float) for values in rail)
    if not (len(xs) == len(ys) == len(z_up) == len(z_dn)) or len(ys) < 2:
        raise ValueError("A wireframe rail must contain matching coordinate arrays.")
    if np.any(np.diff(ys) < -1e-10):
        raise ValueError("Wireframe rail heights must be monotonic.")
    target = float(np.clip(target_y, ys[0], ys[-1]))
    if target >= ys[-1] - 1e-10:
        return xs.copy(), ys.copy(), z_up.copy(), z_dn.copy()

    upper = int(np.searchsorted(ys, target, side="left"))
    if upper == 0:
        upper = 1
    lower = upper - 1
    span = ys[upper] - ys[lower]
    weight = 0.0 if abs(span) <= 1e-12 else (target - ys[lower]) / span

    def clipped(values):
        endpoint = values[lower] + weight * (values[upper] - values[lower])
        return np.concatenate((values[:upper], np.array([endpoint], dtype=float)))

    return clipped(xs), clipped(ys), clipped(z_up), clipped(z_dn)


def _polyline_run_above_y(polyline, level: float) -> np.ndarray:
    """Return the connected measured tip arc above one horizontal chord."""

    points = np.asarray(polyline, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3:
        raise ValueError("The measured outline must contain finite (x, y) points.")
    if not np.isfinite(points).all():
        raise ValueError("The measured outline must contain finite (x, y) points.")

    runs = []
    active = []
    for start, end in zip(points[:-1], points[1:]):
        start_in = start[1] >= level
        end_in = end[1] >= level
        if start_in and not active:
            active.append(start.copy())
        if start_in and end_in:
            active.append(end.copy())
            continue
        if start_in != end_in:
            weight = (level - start[1]) / (end[1] - start[1])
            crossing = start + weight * (end - start)
            if end_in:
                active = [crossing, end.copy()]
            else:
                active.append(crossing)
                if len(active) >= 2:
                    runs.append(np.asarray(active, dtype=float))
                active = []
    if len(active) >= 2:
        runs.append(np.asarray(active, dtype=float))
    if not runs:
        raise ValueError("The measured outline has no connected tip above the cap chord.")
    return max(runs, key=lambda run: (float(run[:, 1].max()), len(run)))


def _sample_polyline_arclength(polyline, count: int) -> np.ndarray:
    """Sample a polyline at exactly equal cumulative arclength fractions."""

    points = np.asarray(polyline, dtype=float)
    if type(count) is not int or count < 2:
        raise ValueError("At least two tip-contour samples are required.")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    keep = np.r_[True, np.diff(cumulative) > 1e-12]
    points, cumulative = points[keep], cumulative[keep]
    if len(points) < 2 or cumulative[-1] <= 1e-12:
        raise ValueError("The measured tip contour has zero arclength.")
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack((
        np.interp(targets, cumulative, points[:, 0]),
        np.interp(targets, cumulative, points[:, 1]),
    ))


def _airfoil_tip_cap_grid(
    rails,
    rail_fractions,
    outline,
    *,
    samples: int = 25,
    ring_parameters=TIP_CAP_RING_PARAMETERS,
):
    """Build a ruled airfoil tip patch ending at distinct contour points.

    The body rail endpoints form the lower cap chord. Each rail terminates at a
    unique, equally spaced arclength point on the measured tip contour, where
    thickness is exactly zero. Intermediate rings and rails are evaluations of
    the same smooth parameterization, so every displayed intersection agrees.
    """

    fractions = np.asarray(rail_fractions, dtype=float)
    if len(rails) != len(fractions) or len(rails) < 3:
        raise ValueError("Wireframe rails and fractions must have matching lengths.")
    if type(samples) is not int or samples < 3:
        raise ValueError("The tip cap needs at least three samples per rail.")
    if np.any(np.diff(fractions) <= 0.0):
        raise ValueError("Wireframe rail fractions must be strictly increasing.")

    body = np.asarray([
        [float(rail[0][-1]), float(rail[1][-1]), float(rail[2][-1])]
        for rail in rails
    ])
    if not np.allclose(body[:, 1], body[0, 1], rtol=0.0, atol=1e-8):
        raise ValueError("Tip-cap body rails must end on one horizontal chord.")
    tip_arc = _polyline_run_above_y(outline, float(body[0, 1]))
    targets = _sample_polyline_arclength(tip_arc, len(rails))
    direct = np.linalg.norm(targets[0] - body[0, :2]) + np.linalg.norm(
        targets[-1] - body[-1, :2]
    )
    reverse = np.linalg.norm(targets[-1] - body[0, :2]) + np.linalg.norm(
        targets[0] - body[-1, :2]
    )
    if reverse < direct:
        targets = targets[::-1]

    parameter = np.linspace(0.0, 1.0, samples)
    eased = 0.5 - 0.5 * np.cos(np.pi * parameter)
    upper = np.empty((len(rails), samples, 3), dtype=float)
    lower = np.empty_like(upper)
    for index, (start, target) in enumerate(zip(body, targets)):
        upper[index, :, 0] = (1.0 - eased) * start[0] + eased * target[0]
        upper[index, :, 1] = (1.0 - eased) * start[1] + eased * target[1]
        # A quarter ellipse gives the airfoil tip a convex thickness envelope
        # in end view while retaining the exact zero-thickness planform target.
        thickness_scale = np.sqrt(np.maximum(0.0, 1.0 - eased**2))
        upper[index, :, 2] = thickness_scale * start[2]
        lower[index] = upper[index]
        lower[index, :, 2] *= -1.0

    rings = []
    for value in ring_parameters:
        value = float(value)
        if not 0.0 < value < 1.0:
            raise ValueError("Tip-cap ring parameters must lie strictly inside (0, 1).")
        position = value * (samples - 1)
        left = math.floor(position)
        right = min(left + 1, samples - 1)
        weight = position - left
        upper_ring = (1.0 - weight) * upper[:, left] + weight * upper[:, right]
        lower_ring = (1.0 - weight) * lower[:, left] + weight * lower[:, right]
        rings.append((upper_ring, lower_ring))
    return upper, lower, tuple(rings), targets


def _remove_incomplete_iso_thickness_lines(axis) -> None:
    """Remove the legacy constant-Z curves that terminate inside the surface."""

    for line in list(axis.lines):
        if line.get_color() != "blue":
            continue
        xs, zs, ys = (np.asarray(values, dtype=float) for values in line.get_data_3d())
        if len(xs) < 2 or np.ptp(ys) <= 1e-9:
            continue
        if np.ptp(zs) <= 1e-9 and abs(float(zs[0])) > 1e-9:
            line.remove()


def _draw_airfoil_tip_cap(axis, rails, rail_fractions, outline) -> None:
    """Draw the longitudinal and transverse families of one tip patch."""

    upper, lower, rings, _targets = _airfoil_tip_cap_grid(
        rails, rail_fractions, outline
    )
    for surface in (upper, lower):
        for rail in surface:
            axis.plot(rail[:, 0], rail[:, 2], rail[:, 1], linewidth=0.8, color="blue")
    for upper_ring, lower_ring in rings:
        for ring in (upper_ring, lower_ring):
            axis.plot(ring[:, 0], ring[:, 2], ring[:, 1], linewidth=0.85, color="blue")


def _uniform_section_y_values(curves2d, y_max, n_y_slices, *_args, **_kwargs):
    """Use an even body grid ending on the lower chord of the tip patch."""

    y_max = float(y_max)
    count = max(17, min(int(n_y_slices), 33))
    cap_y = TIP_CAP_START_FRACTION * y_max
    uniform = np.linspace(0.0, cap_y, count)
    anchors = [0.0, cap_y]
    for name in ("le0", "le1", "te1", "te2", "te3"):
        curve = curves2d.get(name)
        if curve is None or len(curve) == 0:
            continue
        values = np.asarray(curve, dtype=float)
        anchors.extend(value for value in (float(values[0, 1]), float(values[-1, 1])) if value <= cap_y)
    anchors = np.unique(np.clip(np.asarray(anchors, dtype=float), 0.0, cap_y))
    spacing = cap_y / max(count - 1, 1)
    if spacing > 0.0:
        keep = np.ones(len(uniform), dtype=bool)
        for anchor in anchors:
            keep &= np.abs(uniform - anchor) > 0.35 * spacing
        uniform = uniform[keep]
    return np.unique(np.round(np.concatenate((uniform, anchors)), 9))


def _hybrid_photo_id_outline(controls, trailing_points, n_per_leading_segment: int = 420):
    """Combine smooth measured leading geometry with learned trailing detail.

    Individual nicks and notches remain supplied by FinFindR from A to C30.
    The streamlined B-to-D-to-A edge comes from the operator-fitted cubic
    Beziers, avoiding source-pixel stair steps at tangent viewing angles.
    """

    trailing = np.asarray(trailing_points, dtype=float)
    if trailing.ndim != 2 or trailing.shape[1:] != (2,) or len(trailing) < 2:
        raise ValueError("The learned A-to-C30 trailing contour is invalid.")
    raw = fin2d.curves_from_controls(controls, n_per_seg=n_per_leading_segment)
    leading = np.asarray(raw["LE0"] + raw["LE1"][1:], dtype=float)
    if np.linalg.norm(trailing[0] - leading[-1]) > np.linalg.norm(trailing[-1] - leading[-1]):
        trailing = trailing[::-1]
    trailing = trailing.copy()
    trailing[0] = leading[-1]
    return np.vstack((leading, trailing[1:]))


def _clean_measurement_example(image):
    """Remove the unrelated top-right scan artifact from the cited figure."""

    cleaned = image.convert("RGB").copy()
    x0 = round(cleaned.width * 0.70)
    y1 = round(cleaned.height * 0.09)
    fin2d.ImageDraw.Draw(cleaned).rectangle((x0, 0, cleaned.width, y1), fill="white")
    return cleaned


def _install_low_latency_motion_scheduler(window) -> None:
    """Shorten only the frozen Photo-ID motion debounce on this window."""

    original_after = window.after

    def after(delay_ms, callback=None, *args):
        try:
            is_photo_debounce = int(delay_ms) == 140
        except (TypeError, ValueError):
            is_photo_debounce = False
        if callback is not None and is_photo_debounce:
            delay_ms = MOTION_RENDER_DELAY_MS
        return original_after(delay_ms, callback, *args)

    window.after = after


class _HQDisplayProxy:
    """Delegate to the frozen renderer while applying display-only corrections."""

    def __init__(self, renderer):
        self._renderer = renderer
        self._interactive_motion = False
        self._motion_scenes = {}
        self.VERSION = f"{renderer.VERSION}+surfer-{SURFER_VERSION}"

    def __getattr__(self, name):
        return getattr(self._renderer, name)

    def set_interactive_motion(self, active: bool) -> None:
        """Select the low-latency preview used only while a control is held."""

        self._interactive_motion = bool(active)

    @staticmethod
    def _motion_scene(scene):
        """Keep only the photographed skin in the transient motion scene."""

        labels = np.asarray(scene.face_labels)
        faces = np.asarray(scene.faces)
        selected = labels == str(scene.textured_skin)
        return replace(scene, faces=faces[selected], face_labels=labels[selected])

    def build_hq_scene(self, base_scene, snapshot, base_renderer):
        scene = self._renderer.build_hq_scene(base_scene, snapshot, base_renderer)
        scene = self._retarget_scene(scene, scene.photo_side)
        preview_base = rebuild_texture_scene_airfoil(
            replace(
                base_scene,
                outline=simplified_motion_outline(base_scene.outline),
            ),
            span_stations=MOTION_PREVIEW_SPAN_STATIONS,
            chord_stations=MOTION_PREVIEW_CHORD_STATIONS,
            boundary_min_triangles=MOTION_PREVIEW_BOUNDARY_MIN_TRIANGLES,
        )
        preview = self._renderer.build_hq_scene(preview_base, snapshot, base_renderer)
        preview = self._retarget_scene(preview, scene.photo_side)
        self._motion_scenes[id(scene)] = self._motion_scene(preview)
        return scene

    @staticmethod
    def _retarget_scene(scene, photo_side):
        side = str(photo_side).strip().lower()
        skin_by_side = {"left": "positive_z_skin", "right": "negative_z_skin"}
        if side not in skin_by_side:
            raise ValueError("Select the photographed anatomical side explicitly: Left or Right.")
        skin = skin_by_side[side]
        base_scene = replace(scene.base_scene, photo_side=side, textured_skin=skin)
        return replace(scene, photo_side=side, textured_skin=skin, base_scene=base_scene)

    def retarget_photo_side(self, scene, photo_side):
        preview = self._motion_scenes.get(id(scene))
        retargeted = self._retarget_scene(scene, photo_side)
        if preview is not None:
            self._motion_scenes[id(retargeted)] = self._motion_scene(
                self._retarget_scene(preview, photo_side)
            )
        return retargeted

    def render_hq_frame(self, scene, width, height, **kwargs):
        requested_width, requested_height = int(width), int(height)
        # Active camera motion uses a cached boundary-preserving preview mesh
        # and bounded framebuffer. Released views and the 1920x1440 save path
        # retain the full mesh and requested antialiasing.
        is_export = requested_width >= 1800 and requested_height >= 1300
        is_motion_preview = self._interactive_motion and not is_export
        render_scene = self._motion_scenes.get(id(scene), scene) if is_motion_preview else scene
        pixel_budget = (
            INTERACTIVE_MOTION_MAX_PIXELS
            if is_motion_preview
            else INTERACTIVE_PHOTO_MAX_PIXELS
        )
        render_width, render_height = requested_width, requested_height
        if not is_export and requested_width * requested_height > pixel_budget:
            scale = math.sqrt(
                pixel_budget / (requested_width * requested_height)
            )
            render_width = max(64, math.floor(requested_width * scale))
            render_height = max(64, math.floor(requested_height * scale))
        render_options = dict(kwargs)
        if is_motion_preview:
            render_options["supersample"] = MOTION_PREVIEW_SUPERSAMPLE
        result = render_smooth_photo_frame(
            self._renderer, render_scene, render_width, render_height, **render_options
        )
        preview = result.image
        if (render_width, render_height) != (requested_width, requested_height):
            preview = preview.resize(
                (requested_width, requested_height), fin2d.Image.Resampling.LANCZOS
            )
        if not is_export:
            preview = draw_photo_axis_guide(
                preview,
                self._renderer.project_vertices,
                pitch=float(render_options["pitch"]),
                yaw=float(render_options["yaw"]),
                roll=float(render_options["roll"]),
            )
        return replace(
            result,
            image=preview,
            output_pixels=requested_width * requested_height,
        )

    def renderer_metadata(self, scene, result):
        metadata = self._renderer.renderer_metadata(scene, result)
        metadata.update({
            "display_thickness_scale": DISPLAY_THICKNESS_SCALE,
            "display_max_full_thickness_ratio": MAX_FULL_THICKNESS_RATIO,
            "display_section_law": "closed symmetric NACA-style thickness distribution",
            "display_thickness_scope": "interactive wireframe and Photo-ID view only",
            "selected_photo_skin_backface_culling": False,
            "opposite_photo_skin_rendered": False,
            "edge_wall_rendered_in_photo_mode": False,
            "photo_mode_faces": "operator-selected photographed skin only",
            "anatomical_side_mapping": {"left": "+Z", "right": "-Z"},
            "silhouette_geometry": "the single reconstructed airfoil mesh",
            "auxiliary_silhouette_geometry": False,
            "duplicate_surface_geometry": False,
            "measurement_geometry_changed": False,
            "cfd_surface_changed": False,
            "interactive_motion_preview": {
                "temporary_only": True,
                "maximum_pixels": INTERACTIVE_MOTION_MAX_PIXELS,
                "boundary_tolerance_ab_units": MOTION_OUTLINE_TOLERANCE,
                "supersample": MOTION_PREVIEW_SUPERSAMPLE,
                "saved_output_changed": False,
            },
            "interactive_orientation_guide": "projected X/Y/Z axes; excluded from saved photo",
        })
        return metadata


class IntegratedFinShapeApp(baseline.IntegratedFinShapeApp):
    """Add review-only FinFindR assistance to the frozen 1.2.2 UI."""

    def __init__(self):
        self._vision_assistant = None
        self._vision_busy = False
        self._vision_generation = 0
        self._suggested_fin_box = None
        self._suggested_trailing_edge = None
        self._suggested_leading_edge = None
        self._suggested_photo_contour = None
        self._assisted_crop_record = None
        self._assisted_edge_record = None
        self._crop_preview_pending = False
        self.btn_detect_fin = None
        self.btn_apply_detected_crop = None
        self.btn_suggest_edge = None
        self._hq_display_proxy = None
        self._texture_display_proxy = None
        super().__init__()
        self.title(f"FinShape Surfer GUI {SURFER_VERSION}")
        self._configure_workflow_attribution()
        self._correct_instruction_landmark_order()
        self._add_assisted_vision_row()
        self._append_assisted_vision_instructions()
        self._update_button_states()

    def _configure_workflow_attribution(self) -> None:
        """Put provenance on the toolbars and the filename over the workspace."""

        filename_label = getattr(self, "lbl_image_name", None)
        edit_row = getattr(filename_label, "master", None)
        morphometry_row = getattr(getattr(self, "btn_show2d", None), "master", None)
        if edit_row is None or morphometry_row is None:
            raise RuntimeError("Cannot locate the FinShape workflow rows.")

        filename_label.pack_forget()
        for widget in edit_row.winfo_children():
            if widget is filename_label:
                continue
            try:
                widget.pack_configure(padx=1)
            except fin2d.tk.TclError:
                pass
        for widget in morphometry_row.winfo_children():
            try:
                widget.pack_configure(padx=1)
            except fin2d.tk.TclError:
                pass
        self.btn_save_meas.configure(text="7. Save CSV")
        self.btn_new_image.configure(text="10. New image")
        self.btn_exit.configure(text="Exit + Save")
        self.scale_rotate.configure(length=105)
        self.image_algorithm_credit_label = fin2d.tk.Label(
            edit_row,
            text=IMAGE_ALGORITHM_CREDIT,
            anchor="e",
            fg="#555555",
        )
        self.image_algorithm_credit_label.pack(side="right", padx=(4, 6))
        self.morphometry_credit_label = fin2d.tk.Label(
            morphometry_row,
            text=MORPHOMETRY_CREDIT,
            anchor="e",
            fg="#555555",
        )
        try:
            self.morphometry_credit_label.pack(
                side="right", padx=(4, 6), before=self.btn_exit
            )
        except fin2d.tk.TclError:
            self.morphometry_credit_label.pack(side="right", padx=(4, 6))

        self.measurement_filename_var = fin2d.tk.StringVar(value="")
        self.measurement_filename_label = fin2d.tk.Label(
            self.canvas_frame,
            textvariable=self.measurement_filename_var,
            anchor="w",
            padx=10,
            pady=3,
        )
        self.canvas_frame.rowconfigure(0, weight=0)
        self.canvas_frame.rowconfigure(1, weight=1)
        self.measurement_filename_label.grid(
            row=0, column=0, columnspan=2, sticky="ew"
        )
        self.canvas.grid_configure(row=1, column=0)
        self.v_scrollbar.grid_configure(row=1, column=1)
        self.h_scrollbar.grid_configure(row=2, column=0)
        self.zoom_canvas.place_configure(y=54)
        self._sync_measurement_filename()

    def _sync_measurement_filename(self) -> None:
        path = getattr(self, "image_path", None)
        value = f"Measuring: {path.name}" if path is not None else ""
        if getattr(self, "measurement_filename_var", None) is not None:
            self.measurement_filename_var.set(value)

    def _insert_measurement_example_image(self) -> None:
        """Display the cited example without its unrelated scan-corner artifact."""

        if not fin2d.PIL_OK:
            super()._insert_measurement_example_image()
            return
        path = BASELINE_GUI_PATH.parent / "assets" / "dorsal_fin_tricolor_otsu.png"
        try:
            with fin2d.Image.open(path) as source:
                image = _clean_measurement_example(source)
        except (OSError, ValueError):
            super()._insert_measurement_example_image()
            return
        scale = min(330 / max(image.width, 1), 1.0)
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(size, resample=fin2d.Image.Resampling.LANCZOS)
        self.measurement_example_imgtk = fin2d.ImageTk.PhotoImage(image)
        self.info.image_create("end", image=self.measurement_example_imgtk)
        self.info.insert("end", "\n")

    def _correct_instruction_landmark_order(self) -> None:
        """Correct the frozen text in place without deleting embedded images."""

        self.info.configure(state="normal")
        old = "Landmarks: B → A → C30 → C20 → C10 → C5 → D"
        new = "Landmarks: A → B → C30 → C20 → C10 → C5 → D"
        start = self.info.search(old, "1.0", stopindex="end")
        if start:
            end = f"{start}+{len(old)}c"
            self.info.delete(start, end)
            self.info.insert(start, new)
        self.info.configure(state="disabled")

    def _add_assisted_vision_row(self) -> None:
        morphometry_row = getattr(getattr(self, "btn_show2d", None), "master", None)
        toolbar = getattr(morphometry_row, "master", None)
        if toolbar is None:
            raise RuntimeError("Cannot locate the FinShape Surfer toolbar.")
        separator = fin2d.tk.Frame(toolbar, height=1, bg="#b0b0b0")
        separator.pack(side="top", fill="x", padx=6, pady=(1, 1))
        row = fin2d.tk.Frame(toolbar)
        row.pack(side="top", fill="x")
        fin2d.tk.Label(row, text="Assisted vision", font=("Helvetica", 12, "bold")).pack(
            side="left", padx=(10, 6)
        )
        self.btn_detect_fin = fin2d.tk.Button(
            row, text="Automatic fin detection", command=self.detect_fin_suggestion
        )
        self.btn_detect_fin.pack(side="left", padx=4, pady=3)
        self.btn_apply_detected_crop = fin2d.tk.Button(
            row,
            text="Apply detected crop",
            command=self.apply_detected_crop,
            state="disabled",
        )
        self.btn_apply_detected_crop.pack(side="left", padx=4, pady=3)
        self.btn_suggest_edge = fin2d.tk.Button(
            row, text="Automatic edge detection", command=self.suggest_trailing_edge
        )
        self.btn_suggest_edge.pack(side="left", padx=4, pady=3)
        self.vision_credit_label = fin2d.tk.Label(
            row,
            text="Vision algorithms: FinFindR 0.1.21, Jaime Thompson.",
            anchor="e",
            fg="#555555",
        )
        self.vision_credit_label.pack(side="right", padx=(4, 6))
        self.assisted_vision_row = row
        self.assisted_vision_separator = separator
        self._button_registry.update(
            {
                "detect_fin": self.btn_detect_fin,
                "apply_detected_crop": self.btn_apply_detected_crop,
                "suggest_edge": self.btn_suggest_edge,
            }
        )
        self._apply_active_button_style()

    def _append_assisted_vision_instructions(self) -> None:
        self.info.configure(state="normal")
        self.info.insert(
            "end",
            "\nAssisted vision (experimental):\n"
            "  Automatic fin detection displays a green crop; Apply detected crop is a separate action.\n"
            "  Place A, B and C30 before requesting Automatic edge detection.\n"
            "  The learned contour masks Photo-ID rendering only; measured Beziers remain the CFD boundary.\n"
            "  Results never place landmarks, alter Bezier controls, or declare identity.\n",
        )
        self.info.configure(state="disabled")

    def _invalidate_vision_suggestions(self, *, clear_records: bool = False) -> None:
        self._vision_generation += 1
        self._suggested_fin_box = None
        self._suggested_trailing_edge = None
        self._suggested_leading_edge = None
        self._suggested_photo_contour = None
        canvas = getattr(self, "canvas", None)
        if canvas is not None:
            canvas.delete("assisted_crop", "assisted_edge")
        if getattr(self, "btn_apply_detected_crop", None) is not None:
            self.btn_apply_detected_crop.configure(state="disabled")
        if clear_records:
            self._assisted_crop_record = None
            self._assisted_edge_record = None

    def _get_vision_assistant(self):
        if self._vision_assistant is None:
            from core.vision.finfindr import FinFindRAssistant

            self._vision_assistant = FinFindRAssistant()
        return self._vision_assistant

    def _run_vision_job(
        self, description, image, operation, completion, *, active_button: str
    ) -> None:
        if self._vision_busy:
            return
        generation = self._vision_generation
        self._vision_busy = True
        self._set_active_button(active_button)
        self._update_button_states()
        self.status.set(f"{description} running in background...")

        def worker():
            try:
                result = operation(self._get_vision_assistant(), image)
                error = None
            except Exception as exc:  # noqa: BLE001 - worker errors are reported safely on the UI thread
                result, error = None, exc

            def finish():
                self._vision_busy = False
                self._set_active_button(None)
                self._update_button_states()
                if generation != self._vision_generation:
                    self.status.set(f"{description} discarded because the image changed.")
                    return
                if error is not None:
                    fin2d.messagebox.showerror(
                        f"{description} unavailable",
                        f"{type(error).__name__}: {error}\n\n"
                        "The frozen manual workflow remains available and unchanged.",
                        parent=self,
                    )
                    self.status.set(f"{description} failed; manual workflow is unchanged.")
                    return
                completion(result)

            self.after(0, finish)

        threading.Thread(target=worker, name="finfindr-assistance", daemon=True).start()

    def detect_fin_suggestion(self) -> None:
        if self.image_path is None or self.locked_after_save or not fin2d.PIL_OK:
            return
        image = self._apply_transforms(exclude_crop=True).convert("RGB").copy()

        def completed(result):
            if not result.boxes:
                self.status.set("FinFindR found no fin crop above the review threshold.")
                fin2d.messagebox.showinfo(
                    "No crop suggested", "No fin crop met the detection criteria.", parent=self
                )
                return
            box = result.boxes[0]
            self._suggested_fin_box = (box.x0, box.y0, box.x1, box.y1)
            self._render_image()
            self._assisted_crop_record = {
                "accepted": False,
                "box_xyxy": list(self._suggested_fin_box),
                "confidence": box.confidence,
                "threshold": result.threshold,
                "provenance": result.provenance,
            }
            self.status.set(
                "Green crop suggestion displayed. Review it, then click Apply detected crop or draw a manual crop."
            )
            self._update_button_states()

        self._run_vision_job(
            "Automatic fin detection",
            image,
            lambda assistant, value: assistant.detect_fin_crop(value),
            completed,
            active_button="detect_fin",
        )

    def apply_detected_crop(self) -> None:
        if self._suggested_fin_box is None or self.locked_after_save:
            return
        selected = tuple(self._suggested_fin_box)
        self.crop_box = selected
        self._crop_preview_pending = False
        if self._assisted_crop_record is not None:
            self._assisted_crop_record["accepted"] = True
        self._suggested_fin_box = None
        self.canvas.delete("assisted_crop")
        self.apply_crop()
        self.canvas.delete("assisted_crop")
        self.status.set("Detected crop applied. Review the complete fin before placing landmarks.")
        self._update_button_states()

    def suggest_trailing_edge(self) -> None:
        # This review layer does not modify the saved measurement, so it remains
        # available after Save Measurement has locked the morphometry controls.
        if self.image_path is None or not fin2d.PIL_OK:
            return
        if not all(name in self.landmarks for name in ("B", "A", "C30")):
            fin2d.messagebox.showinfo(
                "A, B and C30 required",
                "Place landmarks A, B and C30 first. They anchor the complete Photo-ID contour; "
                "the learned model only proposes the paths between them.",
                parent=self,
            )
            return
        image = self._apply_transforms(exclude_crop=False).convert("RGB").copy()
        anchors = {name: self.landmarks[name] for name in ("B", "A", "C30")}
        tissue_polygon = fin2d.outline_polygon(self.controls, n_per_seg=600)

        def completed(result):
            self._suggested_leading_edge = result.leading_points.copy()
            self._suggested_trailing_edge = result.trailing_points.copy()
            self._suggested_photo_contour = result.outline_points.copy()
            self._assisted_edge_record = {
                "displayed_for_review": True,
                "accepted_as_measurement": False,
                "used_as_photo_id_mask": True,
                "point_count": len(result.outline_points),
                "points_sha256": _points_sha256(result.outline_points),
                "leading_confidence": result.leading_confidence,
                "trailing_confidence": result.trailing_confidence,
                "provenance": result.provenance,
            }
            cached = getattr(self, "last_integrated_texture_snapshot", None)
            if cached is not None:
                self.last_integrated_texture_snapshot = self._apply_photo_mask_to_snapshot(
                    cached
                )
            self._render_image()
            self.status.set(
                "Orange B-to-A-to-C30 contour will mask Photo-ID only; measured and CFD geometry are unchanged."
            )

        self._run_vision_job(
            "Automatic edge detection",
            image,
            lambda assistant, value: assistant.suggest_fin_contour(
                value,
                b_xy=anchors["B"],
                a_xy=anchors["A"],
                c30_xy=anchors["C30"],
                tissue_polygon_xy=tissue_polygon,
            ),
            completed,
            active_button="suggest_edge",
        )

    def _update_button_states(self):
        super()._update_button_states()
        loaded = getattr(self, "image_path", None) is not None
        locked = getattr(self, "locked_after_save", False)
        busy = getattr(self, "_vision_busy", False)
        general_state = "normal" if loaded and not locked and not busy else "disabled"
        if getattr(self, "btn_detect_fin", None) is not None:
            self.btn_detect_fin.configure(state=general_state)
        if getattr(self, "btn_apply_detected_crop", None) is not None:
            apply_ready = self._suggested_fin_box is not None and general_state == "normal"
            self.btn_apply_detected_crop.configure(state="normal" if apply_ready else "disabled")
        if getattr(self, "btn_suggest_edge", None) is not None:
            endpoints_ready = all(name in getattr(self, "landmarks", {}) for name in ("B", "A", "C30"))
            self.btn_suggest_edge.configure(
                state="normal" if loaded and not busy and endpoints_ready else "disabled"
            )

    def _close_2d_review_window(self, window) -> None:
        """Close the derived view and restore access to read-only review actions."""

        try:
            if window.winfo_exists():
                window.destroy()
        finally:
            self._set_active_button(None)
            self._update_button_states()
            try:
                self.lift()
                self.focus_set()
            except fin2d.tk.TclError:
                pass

    def _draw_overlays(self):
        # These extension-owned layers are not known to the frozen redraw
        # routine. Replace them on every frame so dragging never leaves trails.
        self.canvas.delete("assisted_crop", "assisted_edge", "leading_control_link")
        super()._draw_overlays()
        if self._suggested_fin_box is not None and self.crop_box is None:
            x0, y0, x1, y1 = self._suggested_fin_box
            self.canvas.create_rectangle(
                x0 * self.display_scale,
                y0 * self.display_scale,
                x1 * self.display_scale,
                y1 * self.display_scale,
                outline="#00dc5a",
                width=3,
                dash=(8, 4),
                tags=("assisted_crop",),
            )
        if self._suggested_photo_contour is not None and len(self._suggested_photo_contour) > 1:
            coordinates = []
            for x, y in self._suggested_photo_contour:
                coordinates.extend((float(x) * self.display_scale, float(y) * self.display_scale))
            self.canvas.create_line(
                *coordinates,
                fill=ASSISTED_EDGE_COLOR,
                width=1,
                smooth=False,
                tags=("assisted_edge",),
            )
        if self.controls is not None and all(name in self.landmarks for name in ("B", "D")):
            for endpoint_name, field in (("B", "LE0_P1"), ("D", "LE0_P2")):
                endpoint = self.landmarks[endpoint_name]
                control = getattr(self.controls, field)
                self.canvas.create_line(
                    endpoint[0] * self.display_scale,
                    endpoint[1] * self.display_scale,
                    control[0] * self.display_scale,
                    control[1] * self.display_scale,
                    fill="#43a047",
                    width=1,
                    dash=(4, 3),
                    tags=("leading_control_link",),
                )
                cx, cy = control[0] * self.display_scale, control[1] * self.display_scale
                self.canvas.create_rectangle(
                    cx - 6,
                    cy - 6,
                    cx + 6,
                    cy + 6,
                    outline="#66ff7a",
                    width=1,
                    tags=("leading_control_link", field),
                )
            try:
                self.canvas.tag_lower("leading_control_link", "fixed_landmark")
            except fin2d.tk.TclError:
                pass

    def load_image(self):
        previous = getattr(self, "image_path", None)
        super().load_image()
        if getattr(self, "image_path", None) != previous:
            self._crop_preview_pending = False
            self._invalidate_vision_suggestions(clear_records=True)
            self._render_image()
        self._sync_measurement_filename()

    def measure_new_image(self):
        self._crop_preview_pending = False
        self._invalidate_vision_suggestions(clear_records=True)
        super().measure_new_image()
        self._sync_measurement_filename()

    def toggle_flip(self):
        previous = getattr(self, "flip_x", False)
        self._invalidate_vision_suggestions(clear_records=True)
        super().toggle_flip()
        if getattr(self, "flip_x", False) == previous:
            self._update_button_states()

    def on_rotate_change(self, event=None):
        previous = getattr(self, "rotate_deg", 0.0)
        requested_clockwise = float(self.rot_var.get()) if hasattr(self, "rot_var") else -previous
        requested_internal = -requested_clockwise
        if requested_internal != previous:
            self._invalidate_vision_suggestions(clear_records=True)
        if getattr(self, "locked_after_save", False) or self.image_path is None:
            return
        self.rotate_deg = requested_internal
        self._disable_2d()
        self._auto_fit_image()

    def enter_crop(self):
        self._invalidate_vision_suggestions(clear_records=True)
        self.canvas.delete("assisted_crop", "assisted_edge")
        super().enter_crop()

    def reset_crop(self):
        self._crop_preview_pending = False
        self._invalidate_vision_suggestions(clear_records=True)
        super().reset_crop()

    def apply_crop(self):
        self._crop_preview_pending = False
        super().apply_crop()

    def _apply_transforms(self, exclude_crop: bool = False):
        """Keep a selected manual crop pending until the operator applies it."""

        pending = self.__dict__.get("_crop_preview_pending", False)
        return super()._apply_transforms(exclude_crop=exclude_crop or pending)

    def start_landmarks(self):
        super().start_landmarks()
        if self.mode == "lm_B":
            self.mode = "lm_A"
            self.status.set("Landmarks: click A.")
            self._render_image()

    def on_mouse_down(self, event):
        if self.image_path is None or getattr(self, "locked_after_save", False):
            return
        x, y = self._event_canvas_xy(event)
        if self.mode == "crop":
            self._invalidate_vision_suggestions(clear_records=True)
            self.canvas.delete("assisted_crop", "assisted_edge")
        elif self.controls is not None and self.mode == "bezier":
            working = self._canvas_to_working(x, y)
            field = _nearest_control_field(
                self.controls,
                fin2d.DRAGGABLE_BEZIER_CONTROL_FIELDS,
                working,
                self.display_scale,
                CONTROL_HIT_RADIUS_PX,
            )
            if field is not None:
                self._active_handle = field
                self._drag_start = (x, y)
                return
        working = self._canvas_to_working(x, y)
        if self.mode == "lm_A" and "B" not in self.landmarks:
            if self._confirm_landmark_click(working):
                self.landmarks["A"] = working
                self.mode = "lm_B"
                self.status.set("Landmarks: click B.")
                self._render_image()
            return
        if self.mode == "lm_B" and "A" in self.landmarks:
            if self._confirm_landmark_click(working):
                self.landmarks["B"] = working
                self.mode = "lm_C30"
                self.status.set("Landmarks: click C30.")
                self._render_image()
            return
        super().on_mouse_down(event)

    def on_mouse_up(self, event):
        previous = getattr(self, "crop_box", None)
        super().on_mouse_up(event)
        if getattr(self, "crop_box", None) != previous:
            self._crop_preview_pending = True
            self._invalidate_vision_suggestions(clear_records=True)

    def on_mouse_drag(self, event):
        # TE2_Q1 is deliberately unrestricted in 1.3.5. The active curve is
        # split at fixed C10, so ordinary handle motion cannot detach C10.
        super().on_mouse_drag(event)

    def _render_image(self):
        controls = getattr(self, "controls", None)
        c10 = getattr(self, "landmarks", {}).get("C10")
        if controls is not None and c10 is not None:
            _prepare_c10_anchored_control(controls, c10)
        super()._render_image()

    def _standardized_suggested_edge(self):
        if self._suggested_trailing_edge is None:
            return None
        if not all(name in self.landmarks for name in ("B", "A", "C30")):
            return None
        try:
            points = fin2d.standardize_contour_units(
                [tuple(map(float, point)) for point in self._suggested_trailing_edge],
                self.landmarks["B"],
                self.landmarks["A"],
                self.landmarks["C30"],
                ab_units=10.0,
            )
        except ValueError:
            return None
        return np.asarray(points, dtype=float)

    def _standardized_photo_contour(self):
        if self._suggested_photo_contour is None:
            return None
        try:
            points = fin2d.standardize_contour_units(
                [tuple(map(float, point)) for point in self._suggested_photo_contour],
                self.landmarks["B"], self.landmarks["A"], self.landmarks["C30"],
                ab_units=10.0,
            )
        except (KeyError, ValueError):
            return None
        return np.asarray(points, dtype=float)

    def show_2d_window(self):
        before = set(self.winfo_children())
        super().show_2d_window()
        new_windows = [
            child for child in self.winfo_children()
            if child not in before and isinstance(child, fin2d.tk.Toplevel)
        ]
        if not new_windows:
            return
        window = new_windows[-1]
        window.protocol(
            "WM_DELETE_WINDOW",
            lambda current=window: self._close_2d_review_window(current),
        )
        edge = self._standardized_suggested_edge()
        if edge is None or len(edge) < 2:
            return

        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)

        canvases = [child for child in descendants(window) if isinstance(child, fin2d.tk.Canvas)]
        if not canvases:
            return
        canvas = canvases[0]
        poly = fin2d.outline_polygon(self.controls, n_per_seg=1100)
        contour = np.asarray(
            fin2d.standardize_contour_units(
                poly,
                self.landmarks["B"],
                self.landmarks["A"],
                self.landmarks["C30"],
                ab_units=10.0,
            ),
            dtype=float,
        )

        def draw_edge(_event=None):
            canvas.delete("assisted_edge_2d")
            width = max(1, canvas.winfo_width())
            height = max(1, canvas.winfo_height())
            pad = 90.0
            minimum = contour.min(axis=0)
            maximum = contour.max(axis=0)
            scale = min(
                (width - 2.0 * pad) / max(maximum[0] - minimum[0], 1e-9),
                (height - 2.0 * pad) / max(maximum[1] - minimum[1], 1e-9),
            )
            display = np.column_stack((
                pad + (edge[:, 0] - minimum[0]) * scale,
                pad + (maximum[1] - edge[:, 1]) * scale,
            ))
            canvas.create_line(
                *display.ravel().tolist(),
                fill=ASSISTED_EDGE_COLOR,
                width=1,
                dash=(7, 3),
                tags=("assisted_edge_2d",),
            )
            canvas.create_text(
                12,
                height - 12,
                anchor="sw",
                text="Orange dashed line: FinFindR A-to-C30 review suggestion (not measurement geometry)",
                fill="#a65300",
                tags=("assisted_edge_2d",),
            )

        canvas.bind("<Configure>", draw_edge, add="+")
        canvas.after_idle(draw_edge)

    def _load_fin3d(self):
        module = super()._load_fin3d()
        module._surfer_app_ref = weakref.ref(self)
        if not hasattr(module, "_surfer_unscaled_morteo_SP_T"):
            module._surfer_unscaled_morteo_SP_T = module.morteo_SP_T
            module._surfer_original_structural_visual_thickness = (
                module.structural_visual_thickness
            )
            module._surfer_original_invert_thickness_for_c = (
                module.invert_thickness_for_c
            )

            def display_morteo_SP_T(chord):
                _shoulder, empirical = module._surfer_unscaled_morteo_SP_T(chord)
                maximum = airfoil_max_full_thickness(
                    chord,
                    empirical,
                    display_scale=DISPLAY_THICKNESS_SCALE,
                    max_ratio=MAX_FULL_THICKNESS_RATIO,
                )
                return 0.30 * max(0.0, float(chord)), maximum

            def airfoil_visual_thickness(c, chord, _shoulder, maximum, **_kwargs):
                return airfoil_full_thickness(c, chord, maximum)

            def invert_airfoil(target, chord, _shoulder, maximum):
                return invert_airfoil_thickness(target, chord, maximum)

            module.morteo_SP_T = display_morteo_SP_T
            module.structural_visual_thickness = airfoil_visual_thickness
            module.invert_thickness_for_c = invert_airfoil
        if not hasattr(module, "_surfer_unwrapped_readable_chordwise_rail_fractions"):
            module._surfer_unwrapped_readable_chordwise_rail_fractions = (
                module.readable_chordwise_rail_fractions
            )

            def readable_rail_fractions(n_chord_points, max_rails=35):
                return _readable_rail_fractions(
                    n_chord_points,
                    max_rails=min(int(max_rails), 15),
                )

            module.readable_chordwise_rail_fractions = readable_rail_fractions
        if not hasattr(module, "_surfer_unwrapped_contour_reference_y_values"):
            module._surfer_unwrapped_contour_reference_y_values = (
                module.contour_reference_y_values
            )
            module.contour_reference_y_values = _uniform_section_y_values
        if not hasattr(module, "_surfer_unwrapped_append_terminal_apex_section"):
            module._surfer_unwrapped_append_terminal_apex_section = (
                module.append_terminal_apex_section
            )

            def omit_display_apex_section(sections, _outline, _y_max, **_kwargs):
                return sections

            module.append_terminal_apex_section = omit_display_apex_section
        if not hasattr(module, "_surfer_unwrapped_chordwise_surface_rails"):
            module._surfer_unwrapped_chordwise_surface_rails = (
                module.chordwise_surface_rails
            )

            def recorded_body_rails(sections, rail_fracs, **kwargs):
                rails = module._surfer_unwrapped_chordwise_surface_rails(
                    sections, rail_fracs, **kwargs
                )
                module._surfer_last_body_rails = rails
                module._surfer_last_rail_fractions = np.asarray(rail_fracs, dtype=float)
                return rails

            module.chordwise_surface_rails = recorded_body_rails
        if not hasattr(module, "_surfer_unwrapped_draw_3d_wireframe_axes"):
            module._surfer_unwrapped_draw_3d_wireframe_axes = module.draw_3d_wireframe_axes

            def draw_with_assisted_edge(axis, curves2d, fin_id, *args, **kwargs):
                kwargs["top_focus_power"] = 1.0
                kwargs["edge_min_thickness"] = 0.0
                kwargs["tip_min_chord"] = 0.0
                module._surfer_unwrapped_draw_3d_wireframe_axes(
                    axis, curves2d, fin_id, *args, **kwargs
                )
                _remove_incomplete_iso_thickness_lines(axis)
                body_rails = getattr(module, "_surfer_last_body_rails", None)
                rail_fractions = getattr(module, "_surfer_last_rail_fractions", None)
                if body_rails is None or rail_fractions is None:
                    raise RuntimeError("The wireframe body grid was not available for tip closure.")
                _draw_airfoil_tip_cap(
                    axis,
                    body_rails,
                    rail_fractions,
                    np.asarray(curves2d["outline"], dtype=float),
                )
                app_ref = getattr(module, "_surfer_app_ref", None)
                app = app_ref() if app_ref is not None else None
                edge_points = app._standardized_suggested_edge() if app is not None else None
                _draw_assisted_edge_3d(axis, edge_points)
                _fit_wireframe_axes(axis)

            module.draw_3d_wireframe_axes = draw_with_assisted_edge
        return module

    def _load_hq_renderer(self):
        renderer = super()._load_hq_renderer()
        if renderer is None:
            return None
        if self._hq_display_proxy is None or self._hq_display_proxy._renderer is not renderer:
            self._hq_display_proxy = _HQDisplayProxy(renderer)
        return self._hq_display_proxy

    def _load_texture_renderer(self):
        renderer = super()._load_texture_renderer()
        if renderer is None:
            return None
        if self._texture_display_proxy is None or self._texture_display_proxy._renderer is not renderer:
            self._texture_display_proxy = PhotoMaskTextureProxy(renderer)
        return self._texture_display_proxy

    def _texture_registration_snapshot(self, curves, fin_id, surface_snapshot, n_per_seg=234):
        snapshot = super()._texture_registration_snapshot(
            curves, fin_id, surface_snapshot, n_per_seg=n_per_seg
        )
        return self._apply_photo_mask_to_snapshot(snapshot)

    def _apply_photo_mask_to_snapshot(self, snapshot):
        """Return a current Photo-ID snapshot without changing measurement data."""

        if snapshot is None or self._suggested_trailing_edge is None:
            return snapshot
        snapshot = dict(snapshot)
        image_outline = _hybrid_photo_id_outline(
            self.controls, self._suggested_trailing_edge
        )
        standardized = np.asarray(
            fin2d.standardize_contour_units(
                [tuple(map(float, point)) for point in image_outline],
                self.landmarks["B"],
                self.landmarks["A"],
                self.landmarks["C30"],
                ab_units=10.0,
            ),
            dtype=float,
        )
        apex_index = int(np.argmin(np.linalg.norm(
            image_outline - np.asarray(self.landmarks["A"], dtype=float), axis=1
        )))
        image_outline[0] = self.landmarks["B"]
        image_outline[apex_index] = self.landmarks["A"]
        image_outline[-1] = self.landmarks["C30"]
        standardized_landmarks = snapshot["standardized_landmarks"]
        standardized[0] = standardized_landmarks["B"]
        standardized[apex_index] = standardized_landmarks["A"]
        standardized[-1] = standardized_landmarks["C30"]
        snapshot["photo_mask_curves"] = {"outline": standardized.tolist()}
        snapshot["photo_mask_image_outline"] = image_outline.tolist()
        snapshot["photo_mask_provenance"] = self._assisted_edge_record
        snapshot["photo_mask_scope"] = "high-resolution Photo-ID rendering only"
        snapshot["photo_mask_boundary_policy"] = (
            "operator-fitted cubic Bezier B-to-D-to-A leading edge plus "
            "operator-anchored FinFindR A-to-C30 trailing edge"
        )
        return snapshot

    def show_3d_window(self) -> None:
        """Open the inherited inspector and enforce the project side convention."""

        super().show_3d_window()
        controls = getattr(self, "last_3d_view_controls", None)
        if not controls:
            return
        window = controls.get("window")
        if window is None:
            return
        _install_low_latency_motion_scheduler(window)

        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)

        for widget in descendants(window):
            if not isinstance(widget, fin2d.tk.Radiobutton):
                continue
            text = str(widget.cget("text"))
            if text.startswith("Left ("):
                widget.configure(text="Left (+Z)")
            elif text.startswith("Right ("):
                widget.configure(text="Right (-Z)")

        scales = [widget for widget in descendants(window) if isinstance(widget, fin2d.tk.Scale)]
        variables = {
            str(controls["pitch"]): (90.0, -90.0),
            str(controls["yaw"]): (180.0, -180.0),
        }
        scale_by_variable = {}
        for scale in scales:
            variable_name = str(scale.cget("variable"))
            scale_by_variable[variable_name] = scale
            if variable_name in variables:
                scale.configure(from_=variables[variable_name][0], to=variables[variable_name][1])

        axis = getattr(self, "last_3d_view_figure", None)
        axis = axis.axes[0] if axis is not None and axis.axes else None
        if axis is not None and hasattr(axis, "disable_mouse_rotation"):
            axis.disable_mouse_rotation()

        drag = {}
        pitch_scale = scale_by_variable.get(str(controls["pitch"]))
        yaw_scale = scale_by_variable.get(str(controls["yaw"]))
        roll_scale = scale_by_variable.get(str(controls["roll"]))

        def set_motion_preview(active):
            proxy = getattr(self, "_hq_display_proxy", None)
            if proxy is not None:
                proxy.set_interactive_motion(active)

        def request_final_render(scale):
            set_motion_preview(False)
            command = str(scale.cget("command")) if scale is not None else ""
            if command:
                scale.tk.call(command, scale.get())

        for scale in (pitch_scale, yaw_scale, roll_scale):
            if scale is None:
                continue
            scale.bind("<ButtonPress-1>", lambda _event: set_motion_preview(True), add="+")
            scale.bind(
                "<ButtonRelease-1>",
                lambda _event, source=scale: request_final_render(source),
                add="+",
            )

        def begin_drag(event):
            set_motion_preview(True)
            drag.update(x=event.x, y=event.y, pitch=controls["pitch"].get(),
                        yaw=controls["yaw"].get(), roll=controls["roll"].get(),
                        roll_mode=bool(event.state & 0x0001))

        def move_drag(event):
            if "x" not in drag:
                return
            dx, dy = event.x - drag["x"], event.y - drag["y"]
            if drag["roll_mode"]:
                if roll_scale is not None:
                    roll_scale.set(_camera_drag_value(drag["roll"], dx, -180.0, 180.0))
                return
            if yaw_scale is not None:
                yaw_scale.set(_camera_drag_value(drag["yaw"], dx, -180.0, 180.0))
            if pitch_scale is not None:
                pitch_scale.set(_camera_drag_value(drag["pitch"], dy, -90.0, 90.0))

        def end_drag(_event):
            drag.clear()
            request_final_render(yaw_scale or pitch_scale or roll_scale)

        figure = getattr(self, "last_3d_view_figure", None)
        mpl_widget = figure.canvas.get_tk_widget() if figure is not None else None
        candidates = [widget for widget in descendants(window) if widget.master is getattr(mpl_widget, "master", None)]
        for widget in [mpl_widget, *candidates]:
            if widget is None or isinstance(widget, fin2d.tk.Scale):
                continue
            widget.bind("<ButtonPress-1>", begin_drag, add="+")
            widget.bind("<B1-Motion>", move_drag, add="+")
            widget.bind("<ButtonRelease-1>", end_drag, add="+")
        controls.update({"pitch_scale": pitch_scale, "yaw_scale": yaw_scale,
                         "roll_scale": roll_scale, "direct_drag_enabled": True})

    def _surface_export_snapshot(self, curves, fin_id):
        snapshot = super()._surface_export_snapshot(curves, fin_id)
        snapshot["provenance"]["assisted_vision"] = {
            "surfer_extension_version": SURFER_VERSION,
            "automatic_landmark_placement": False,
            "crop": self._assisted_crop_record,
            "trailing_edge_review_overlay": self._assisted_edge_record,
            "trailing_edge_geometry_replacement": False,
            "photo_id_mask_boundary": self._assisted_edge_record,
            "photo_id_mask_changes_cfd": False,
        }
        snapshot["provenance"]["display_geometry"] = {
            "thickness_scale": DISPLAY_THICKNESS_SCALE,
            "scope": "wireframe and Photo-ID display only",
            "section_model": "closed symmetric NACA-style thickness distribution",
            "maximum_full_thickness_ratio": MAX_FULL_THICKNESS_RATIO,
            "contour_closure": "zero thickness at both planform edges and the apex",
            "auxiliary_silhouette_geometry": False,
            "reason": "continuous airfoil-like reconstruction without edge floors or overlays",
            "scientific_status": "documented reconstruction assumption; not measured biology",
            "measurement_geometry_changed": False,
            "cfd_surface_changed": False,
            "cfd_boundary_source": "measured Bezier contour; assisted edge remains review-only",
            "cfd_export_decision": (
                "retain the separately validated constrained solid; the display airfoil mesh "
                "has coincident zero-thickness edge vertices and is not solver-export geometry"
            ),
            "wireframe_rail_limit": 15,
            "wireframe_section_limit": 43,
            "wireframe_apex_rule": (
                "a ruled tip patch maps each longitudinal rail to a distinct equal-arclength "
                "point on the measured tip contour; transverse rings share the same parameterization; "
                "thickness closes on a convex quarter-elliptic envelope"
            ),
            "wireframe_iso_thickness_curves": "omitted because they terminate inside tapering sections",
            "wireframe_density_changes_cfd": False,
        }
        return snapshot

    def _build_manifest(self, fin_id, outdir, outputs, warnings):
        inherited = super()._build_manifest(fin_id, outdir, outputs, warnings)
        state = "not used"
        if self._assisted_crop_record or self._assisted_edge_record:
            state = "used as operator-reviewed visual assistance"
        addition = (
            f"Assisted vision {SURFER_VERSION}: {state}; no automatic landmarks, Bezier controls, "
            "identity decision, or direct geometry replacement.\n"
            f"Display geometry {SURFER_VERSION}: one closed symmetric NACA-style section is reconstructed at "
            f"each planform station, capped at {MAX_FULL_THICKNESS_RATIO:g} of local chord and calibrated from "
            f"the inherited thickness at {DISPLAY_THICKNESS_SCALE:g}x. It closes to zero on both contour edges "
            "and at the apex. CFD export and the measured planform are unchanged.\n"
            "Photo-ID surface: single-chord planforms use the dense structured 321-by-41 airfoil mesh restored "
            "from 1.3.8. Multi-interval outlines use a dense constrained-Delaunay mesh rather "
            "than the legacy coarse triangulation. Exact notch turning heights participate in topology selection. No "
            "ribbon, cap, duplicate edge, or other auxiliary surface is rendered. The dense "
            "FinFindR A-to-C30 boundary retains individual nicks and notches; the measured Bezier boundary "
            "remains the CFD authority.\n"
            "Photo-ID color boundary: robust CIE Lab tissue and water evidence is seeded from the operator-fitted "
            "Bezier interior and the outer crop frame, then combined at source resolution with learned edge and "
            "local perceptual-color gradients. The Photo-ID leading boundary uses the smooth measured cubic edge; "
            "trailing-"
            "edge detail is retained.\n"
            "Wireframe display density: at most 15 uniformly spaced chordwise rails and an even body grid are "
            "shown. A continuous ruled tip patch closes those rails at distinct equal-arclength points on the "
            "measured tip contour, with a convex quarter-elliptic thickness envelope and intersecting transverse "
            "rings from the same parameterization. Legacy "
            "constant-thickness curves that ended inside tapering sections are omitted. CFD triangulation is "
            "unchanged.\n"
            "CFD geometry decision: the experimental exporter retains its independently validated constrained "
            "solid and measured Bezier planform. The display airfoil mesh is not substituted because its "
            "zero-thickness contour closure is unsuitable as a watertight solver boundary without a separate "
            "mesh-generation and validation study.\n"
            "Landmark workflow: operator places A, then B, C30, C20, C10, C5 and D. The B-to-D leading segment is "
            "a cubic Bezier with two independently movable controls visibly linked to B and D. The C5-to-C20 "
            "trailing segment uses an unrestricted handle and two C1-continuous quadratics meeting at fixed C10.\n"
            "Anatomical side convention: dolphin swimming right; Left = +Z and Right = -Z.\n"
        )
        return inherited.replace("Scope:", addition + "Scope:")


def main() -> None:
    app = IntegratedFinShapeApp()
    if not fin2d.PIL_OK:
        fin2d.messagebox.showwarning(
            "Pillow required",
            "Image measurement and assisted vision require Pillow. Install project dependencies and restart.",
            parent=app,
        )
    app.mainloop()


if __name__ == "__main__":
    main()
