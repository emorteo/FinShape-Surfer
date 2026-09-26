#!/usr/bin/env python3
"""Corrected and antialiased high-resolution Photo-ID renderer.

This display-only extension preserves the measured planform and simulation
surface. It corrects anatomical-side mapping, omits edge-wall faces from the
photo view, and culls back-facing triangles to fund 2x silhouette antialiasing.
"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import time
from typing import Tuple

import numpy as np
from PIL import Image


VERSION = "0.2.0"
BASE_PATH = Path(__file__).resolve().with_name("FinShapeTextureHQ_0.1.0.py")
ANATOMICAL_SIDE_TO_SKIN = {
    "left": "negative_z_skin",
    "right": "positive_z_skin",
}
METHOD = (
    "CPU UV rasterization with corrected anatomical-side mapping, barycentric "
    "image-coordinate interpolation, bilinear sampling, a per-pixel depth "
    "buffer, back-face culling and supersampled silhouette antialiasing. "
    "Edge-wall faces are intentionally omitted from the Photo-ID view only."
)
LIMITATION = (
    "High-resolution display does not add camera calibration, camera-pose "
    "estimation, lens-distortion correction, perspective correction, biological "
    "validation or automatic identity matching. Render-only refinement does not "
    "change or validate CFD geometry."
)


def _load_base():
    name = "finshape_texture_hq_0_1_0_base"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BASE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the preserved high-resolution renderer: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base()
HQTextureScene = base.HQTextureScene
RenderResult = base.RenderResult
def project_vertices(
    vertices,
    width: int,
    height: int,
    *,
    pitch: float,
    yaw: float,
    roll: float,
    margin_fraction: float = 0.07,
):
    """Project with Matplotlib's elevation, azimuth and roll convention."""
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Projection requires finite 3D vertices.")
    if type(width) is not int or type(height) is not int or width < 64 or height < 64:
        raise ValueError("Framebuffer dimensions must be integers of at least 64 pixels.")
    if not np.isfinite([pitch, yaw, roll]).all():
        raise ValueError("Pitch, yaw and roll must be finite.")
    if not 0.0 <= margin_fraction < 0.4:
        raise ValueError("Projection margin must be in [0, 0.4).")

    # Match Matplotlib's display order: X, Z-thickness, Y-height.
    display = points[:, [0, 2, 1]]
    center = (display.min(axis=0) + display.max(axis=0)) / 2.0
    centered = display - center
    elevation, azimuth, camera_roll = np.deg2rad([pitch, yaw, roll])
    view_out = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])
    vertical_axis = np.array([0.0, 0.0, 1.0])
    screen_right = np.cross(vertical_axis, view_out)
    if np.linalg.norm(screen_right) < 1e-12:
        screen_right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    screen_right /= np.linalg.norm(screen_right)
    screen_up = np.cross(view_out, screen_right)
    if camera_roll != 0.0:
        cos_roll, sin_roll = np.cos(-camera_roll), np.sin(-camera_roll)
        old_right = screen_right.copy()
        screen_right = old_right * cos_roll + screen_up * sin_roll
        screen_up = screen_up * cos_roll - old_right * sin_roll

    horizontal = centered @ screen_right
    vertical_position = centered @ screen_up
    depth = centered @ view_out
    span_x = max(float(np.ptp(horizontal)), 1e-12)
    span_y = max(float(np.ptp(vertical_position)), 1e-12)
    scale = min(
        width * (1.0 - 2.0 * margin_fraction) / span_x,
        height * (1.0 - 2.0 * margin_fraction) / span_y,
    )
    screen = np.column_stack([
        horizontal * scale + width / 2.0,
        -vertical_position * scale + height / 2.0,
    ])
    return screen, depth


def build_hq_scene(base_scene, snapshot, base_renderer) -> HQTextureScene:
    """Build an immutable scene with the corrected anatomical skin mapping."""
    scene = base.build_hq_scene(base_scene, snapshot, base_renderer)
    return retarget_photo_side(scene, scene.photo_side)


def retarget_photo_side(scene: HQTextureScene, photo_side: str) -> HQTextureScene:
    """Reuse immutable geometry while selecting the other anatomical skin."""
    side = str(photo_side).strip().lower()
    if side not in ANATOMICAL_SIDE_TO_SKIN:
        raise ValueError("Select the photographed anatomical side explicitly: Left or Right.")
    skin = ANATOMICAL_SIDE_TO_SKIN[side]
    corrected_base = replace(scene.base_scene, photo_side=side, textured_skin=skin)
    return replace(scene, photo_side=side, textured_skin=skin, base_scene=corrected_base)


def _front_facing(screen_faces: np.ndarray) -> np.ndarray:
    """Return front-facing triangles in the renderer's screen convention."""
    denominator = (
        (screen_faces[:, 1, 1] - screen_faces[:, 2, 1])
        * (screen_faces[:, 0, 0] - screen_faces[:, 2, 0])
        + (screen_faces[:, 2, 0] - screen_faces[:, 1, 0])
        * (screen_faces[:, 0, 1] - screen_faces[:, 2, 1])
    )
    return denominator > 1e-10


def render_hq_frame(
    scene: HQTextureScene,
    width: int,
    height: int,
    *,
    pitch: float,
    yaw: float,
    roll: float,
    supersample: int = 1,
    background_rgb: Tuple[int, int, int] = (248, 249, 250),
) -> RenderResult:
    """Render an antialiased photo view without drawing the edge-wall strip."""
    if type(supersample) is not int or not 1 <= supersample <= 3:
        raise ValueError("Supersampling must be an integer from 1 to 3.")
    if type(width) is not int or type(height) is not int or width < 64 or height < 64:
        raise ValueError("Output dimensions must be integers of at least 64 pixels.")
    render_width = width * supersample
    render_height = height * supersample
    if render_width * render_height > 16_000_000:
        raise ValueError("Requested render exceeds the 16-megapixel safety limit.")

    started = time.perf_counter()
    screen, depth = project_vertices(
        scene.vertices,
        render_width,
        render_height,
        pitch=pitch,
        yaw=yaw,
        roll=roll,
    )
    framebuffer = np.empty((render_height, render_width, 3), dtype=np.uint8)
    framebuffer[:] = np.asarray(background_rgb, dtype=np.uint8)
    zbuffer = np.full((render_height, render_width), -np.inf, dtype=float)
    textured_mask = np.zeros((render_height, render_width), dtype=bool)
    front_facing = _front_facing(screen[scene.faces])

    for face, label, is_front in zip(scene.faces, scene.face_labels, front_facing):
        label = str(label)
        if not is_front or label == "edge_wall":
            continue
        neutral = base.NEUTRAL_RGB.get(label, np.array([190, 192, 194], dtype=np.uint8))
        base._rasterize_triangle(
            framebuffer,
            zbuffer,
            textured_mask,
            screen[face],
            depth[face],
            scene.vertex_uv_px[face],
            texture=scene.texture_rgb,
            textured=(label == scene.textured_skin),
            neutral_rgb=neutral,
        )

    image = Image.fromarray(framebuffer, "RGB")
    if supersample > 1:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return RenderResult(
        image=image,
        render_seconds=time.perf_counter() - started,
        textured_pixels=int(np.count_nonzero(textured_mask)),
        output_pixels=width * height,
        supersample=supersample,
    )


def renderer_metadata(scene: HQTextureScene, result: RenderResult) -> dict:
    metadata = base.renderer_metadata(scene, result)
    metadata.update({
        "hq_texture_module_version": VERSION,
        "hq_method": METHOD,
        "hq_limitations": LIMITATION,
        "photo_side_operator_selected": scene.photo_side,
        "geometric_skin": scene.textured_skin,
        "edge_wall_rendered_in_photo_view": False,
        "back_face_culling": True,
        "silhouette_antialias": f"{result.supersample}x supersampling with Lanczos downsampling",
        "cfd_surface_changed": False,
    })
    return metadata
