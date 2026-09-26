#!/usr/bin/env python3
"""High-resolution software texture rasterizer for FinShape Surfer.

The scientific surface remains the geometry authority. This module adds UV
coordinates and samples the measured photograph for every visible output pixel,
so photographic resolution is independent of triangle count. It intentionally
does not alter landmarks, Bezier curves, thickness, or CFD export geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping, Sequence, Tuple

import numpy as np
from PIL import Image


VERSION = "0.1.0"
METHOD = (
    "CPU UV rasterization with barycentric image-coordinate interpolation, "
    "bilinear texture sampling and a per-pixel depth buffer. Photographic detail "
    "is determined by the transformed source image and output framebuffer, not "
    "by one solid color per mesh triangle."
)
LIMITATION = (
    "High-resolution display does not add camera calibration, camera-pose "
    "estimation, lens-distortion correction, perspective correction, biological "
    "validation or automatic identity matching."
)
NEUTRAL_RGB = {
    "positive_z_skin": np.array([210, 212, 214], dtype=np.uint8),
    "negative_z_skin": np.array([210, 212, 214], dtype=np.uint8),
    "edge_wall": np.array([72, 76, 80], dtype=np.uint8),
    "base_attachment": np.array([120, 124, 128], dtype=np.uint8),
}


@dataclass(frozen=True)
class HQTextureScene:
    vertices: np.ndarray
    faces: np.ndarray
    face_labels: np.ndarray
    vertex_uv_px: np.ndarray
    texture_rgb: np.ndarray
    textured_skin: str
    photo_side: str
    outline: np.ndarray
    base_scene: object


@dataclass(frozen=True)
class RenderResult:
    image: Image.Image
    render_seconds: float
    textured_pixels: int
    output_pixels: int
    supersample: int


def _readonly_copy(values, dtype=None) -> np.ndarray:
    copied = np.array(values, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


def build_hq_scene(base_scene, snapshot: Mapping, base_renderer) -> HQTextureScene:
    """Attach immutable per-vertex image coordinates to a validated base scene."""
    required_scene = ("vertices", "faces", "face_labels", "affine_planform_to_image", "textured_skin", "photo_side")
    missing = [name for name in required_scene if not hasattr(base_scene, name)]
    if missing:
        raise ValueError("Texture scene is incomplete: " + ", ".join(missing))
    if "working_image" not in snapshot:
        raise ValueError("Texture snapshot has no transformed working image.")
    mapper = getattr(base_renderer, "map_planform_to_image", None)
    if not callable(mapper):
        raise RuntimeError("The base texture renderer cannot map planform coordinates.")

    vertices = np.asarray(base_scene.vertices, dtype=float)
    faces = np.asarray(base_scene.faces, dtype=np.int64)
    labels = np.asarray(base_scene.face_labels, dtype=str)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("HQ scene vertices must be finite 3D coordinates.")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) != len(labels):
        raise ValueError("HQ scene faces and labels must describe matching triangles.")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("HQ scene contains an invalid vertex index.")

    uv = mapper(vertices[:, :2], base_scene.affine_planform_to_image)
    texture = np.asarray(snapshot["working_image"].convert("RGB"), dtype=np.uint8)
    if texture.ndim != 3 or texture.shape[2] != 3 or min(texture.shape[:2]) < 2:
        raise ValueError("The transformed photograph must be an RGB image of at least 2 by 2 pixels.")

    return HQTextureScene(
        vertices=_readonly_copy(vertices, float),
        faces=_readonly_copy(faces, np.int64),
        face_labels=_readonly_copy(labels, str),
        vertex_uv_px=_readonly_copy(uv, float),
        texture_rgb=_readonly_copy(texture, np.uint8),
        textured_skin=str(base_scene.textured_skin),
        photo_side=str(base_scene.photo_side),
        outline=_readonly_copy(base_scene.outline, float),
        base_scene=base_scene,
    )


def _rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    if not np.isfinite([pitch, yaw, roll]).all():
        raise ValueError("Pitch, yaw and roll must be finite.")
    p, y, r = np.deg2rad([pitch, yaw, roll])
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)
    yaw_matrix = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    pitch_matrix = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    roll_matrix = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]])
    return roll_matrix @ pitch_matrix @ yaw_matrix


def project_vertices(
    vertices: Sequence[Sequence[float]],
    width: int,
    height: int,
    *,
    pitch: float,
    yaw: float,
    roll: float,
    margin_fraction: float = 0.07,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project FinShape coordinates to a fitted orthographic framebuffer."""
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Projection requires finite 3D vertices.")
    if type(width) is not int or type(height) is not int or width < 64 or height < 64:
        raise ValueError("Framebuffer dimensions must be integers of at least 64 pixels.")
    if not 0.0 <= margin_fraction < 0.4:
        raise ValueError("Projection margin must be in [0, 0.4).")

    # FinShape stores X, Y-height, Z-thickness. Use X/right, Z/depth, Y/up.
    display = points[:, [0, 2, 1]]
    center = (display.min(axis=0) + display.max(axis=0)) / 2.0
    rotated = (display - center) @ _rotation_matrix(pitch, yaw, roll).T
    horizontal = rotated[:, 0]
    vertical = rotated[:, 2]
    depth = rotated[:, 1]
    span_x = max(float(np.ptp(horizontal)), 1e-12)
    span_y = max(float(np.ptp(vertical)), 1e-12)
    available_w = width * (1.0 - 2.0 * margin_fraction)
    available_h = height * (1.0 - 2.0 * margin_fraction)
    scale = min(available_w / span_x, available_h / span_y)
    screen = np.column_stack([
        (horizontal - (horizontal.min() + horizontal.max()) / 2.0) * scale + width / 2.0,
        -(vertical - (vertical.min() + vertical.max()) / 2.0) * scale + height / 2.0,
    ])
    return screen, depth


def _sample_bilinear_uint8(texture: np.ndarray, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height, width = texture.shape[:2]
    valid = (
        (uv[:, 0] >= 0.0) & (uv[:, 0] <= width - 1) &
        (uv[:, 1] >= 0.0) & (uv[:, 1] <= height - 1)
    )
    colors = np.full((len(uv), 3), 140, dtype=np.uint8)
    if not np.any(valid):
        return colors, valid
    x = uv[valid, 0]
    y = uv[valid, 1]
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    top = texture[y0, x0].astype(float) * (1.0 - wx) + texture[y0, x1].astype(float) * wx
    bottom = texture[y1, x0].astype(float) * (1.0 - wx) + texture[y1, x1].astype(float) * wx
    colors[valid] = np.clip(top * (1.0 - wy) + bottom * wy, 0.0, 255.0).astype(np.uint8)
    return colors, valid


def _rasterize_triangle(
    framebuffer: np.ndarray,
    zbuffer: np.ndarray,
    textured_mask: np.ndarray,
    points: np.ndarray,
    depths: np.ndarray,
    uv: np.ndarray,
    *,
    texture: np.ndarray,
    textured: bool,
    neutral_rgb: np.ndarray,
) -> None:
    height, width = zbuffer.shape
    min_x = max(0, int(math.floor(points[:, 0].min())))
    max_x = min(width - 1, int(math.ceil(points[:, 0].max())))
    min_y = max(0, int(math.floor(points[:, 1].min())))
    max_y = min(height - 1, int(math.ceil(points[:, 1].max())))
    if min_x > max_x or min_y > max_y:
        return

    x0, y0 = points[0]
    x1, y1 = points[1]
    x2, y2 = points[2]
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denominator) < 1e-12:
        return
    xs = np.arange(min_x, max_x + 1, dtype=float) + 0.5
    ys = np.arange(min_y, max_y + 1, dtype=float) + 0.5
    xx, yy = np.meshgrid(xs, ys)
    w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denominator
    w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denominator
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
    if not np.any(inside):
        return
    interpolated_depth = w0 * depths[0] + w1 * depths[1] + w2 * depths[2]
    zview = zbuffer[min_y:max_y + 1, min_x:max_x + 1]
    visible = inside & (interpolated_depth > zview)
    if not np.any(visible):
        return

    rows, columns = np.nonzero(visible)
    target_y = rows + min_y
    target_x = columns + min_x
    zbuffer[target_y, target_x] = interpolated_depth[visible]
    if textured:
        sample_uv = (
            w0[visible, None] * uv[0] +
            w1[visible, None] * uv[1] +
            w2[visible, None] * uv[2]
        )
        colors, valid = _sample_bilinear_uint8(texture, sample_uv)
        framebuffer[target_y, target_x] = colors
        textured_mask[target_y[valid], target_x[valid]] = True
        textured_mask[target_y[~valid], target_x[~valid]] = False
    else:
        framebuffer[target_y, target_x] = neutral_rgb
        textured_mask[target_y, target_x] = False


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
    """Render a UV-textured frame while keeping geometry density unchanged."""
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
        scene.vertices, render_width, render_height,
        pitch=pitch, yaw=yaw, roll=roll,
    )
    framebuffer = np.empty((render_height, render_width, 3), dtype=np.uint8)
    framebuffer[:] = np.asarray(background_rgb, dtype=np.uint8)
    zbuffer = np.full((render_height, render_width), -np.inf, dtype=float)
    textured_mask = np.zeros((render_height, render_width), dtype=bool)

    for face, label in zip(scene.faces, scene.face_labels):
        neutral = NEUTRAL_RGB.get(str(label), np.array([190, 192, 194], dtype=np.uint8))
        _rasterize_triangle(
            framebuffer, zbuffer, textured_mask,
            screen[face], depth[face], scene.vertex_uv_px[face],
            texture=scene.texture_rgb,
            textured=(str(label) == scene.textured_skin),
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
    return {
        "hq_texture_module_version": VERSION,
        "hq_method": METHOD,
        "hq_limitations": LIMITATION,
        "texture_interpolation": "per-output-pixel bilinear RGB",
        "geometry_interpolation": "barycentric UV and depth",
        "depth_test": "per-output-pixel z-buffer",
        "mesh_triangles_unchanged": int(len(scene.faces)),
        "texture_source_size_px": [int(scene.texture_rgb.shape[1]), int(scene.texture_rgb.shape[0])],
        "render_output_size_px": list(result.image.size),
        "render_supersample": result.supersample,
        "visible_textured_pixels": result.textured_pixels,
        "render_seconds": result.render_seconds,
    }
