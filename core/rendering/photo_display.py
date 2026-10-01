"""Auditable display-only geometry and high-resolution Photo-ID rendering.

These helpers do not modify measured landmarks, Bezier curves, or the CFD
surface exporter. They operate on immutable display scenes.
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image

DISPLAY_THICKNESS_SCALE = 0.5


def render_smooth_photo_frame(
    renderer,
    scene,
    width: int,
    height: int,
    *,
    pitch: float,
    yaw: float,
    roll: float,
    supersample: int = 1,
    background_rgb: tuple[int, int, int] = (248, 249, 250),
):
    """Render exactly the selected skin of the reconstructed airfoil mesh."""

    if type(supersample) is not int or not 1 <= supersample <= 3:
        raise ValueError("Supersampling must be an integer from 1 to 3.")
    if type(width) is not int or type(height) is not int or width < 64 or height < 64:
        raise ValueError("Output dimensions must be integers of at least 64 pixels.")
    render_width = width * supersample
    render_height = height * supersample
    if render_width * render_height > 16_000_000:
        raise ValueError("Requested render exceeds the 16-megapixel safety limit.")

    started = time.perf_counter()
    scene_vertices = np.asarray(scene.vertices, dtype=float)
    projected, projected_depth = renderer.project_vertices(
        scene_vertices,
        render_width,
        render_height,
        pitch=pitch,
        yaw=yaw,
        roll=roll,
    )
    screen = projected
    depth = projected_depth
    framebuffer = np.empty((render_height, render_width, 3), dtype=np.uint8)
    framebuffer[:] = np.asarray(background_rgb, dtype=np.uint8)
    zbuffer = np.full((render_height, render_width), -np.inf, dtype=float)
    textured_mask = np.zeros((render_height, render_width), dtype=bool)
    for face, label in zip(scene.faces, scene.face_labels):
        label = str(label)
        # Photo-ID mode displays exactly the operator-selected photographed
        # skin.  Drawing the opposite neutral skin at a tangent view lets its
        # triangulation intermittently occlude the measured Bezier silhouette,
        # which appears as a false serrated leading edge.
        if label != scene.textured_skin:
            continue
        neutral = renderer.base.NEUTRAL_RGB.get(
            label, np.array([190, 192, 194], dtype=np.uint8)
        )
        renderer.base._rasterize_triangle(
            framebuffer,
            zbuffer,
            textured_mask,
            screen[face],
            depth[face],
            scene.vertex_uv_px[face],
            texture=scene.texture_rgb,
            textured=True,
            neutral_rgb=neutral,
        )

    image = Image.fromarray(framebuffer, "RGB")
    if supersample > 1:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return renderer.RenderResult(
        image=image,
        render_seconds=time.perf_counter() - started,
        textured_pixels=int(np.count_nonzero(textured_mask)),
        output_pixels=width * height,
        supersample=supersample,
    )
