"""Auditable display-only geometry and high-resolution Photo-ID rendering.

These helpers do not modify measured landmarks, Bezier curves, or the CFD
surface exporter. They operate on immutable display scenes.
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, Polygon

DISPLAY_THICKNESS_SCALE = 0.5
MOTION_OUTLINE_TOLERANCE = 0.0025


def simplified_motion_outline(outline, tolerance: float = MOTION_OUTLINE_TOLERANCE):
    """Discard subpixel boundary samples for the transient camera preview only."""

    source = np.asarray(outline, dtype=float)
    if source.ndim != 2 or source.shape[1] != 2 or len(source) < 3:
        raise ValueError("The Photo-ID preview needs a two-dimensional contour.")
    if not np.isfinite(source).all() or not 0.0 < tolerance < 0.1:
        raise ValueError("The Photo-ID preview contour or tolerance is invalid.")
    original = Polygon(source)
    if not original.is_valid or original.area <= 0.0:
        raise ValueError("The Photo-ID preview contour must be a valid fin polygon.")
    candidate = np.asarray(
        LineString(source).simplify(tolerance, preserve_topology=True).coords,
        dtype=float,
    )
    simplified = Polygon(candidate)
    if (
        len(candidate) < 3 or not simplified.is_valid or simplified.area <= 0.0
        or original.boundary.hausdorff_distance(simplified.boundary) > tolerance * 1.1
    ):
        return source
    candidate[0], candidate[-1] = source[0], source[-1]
    return candidate


def photo_axis_directions(project_vertices, pitch: float, yaw: float, roll: float):
    """Project unit X/Y/Z with the same camera math as the Photo-ID surface."""

    reference = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0],
    ])
    screen, depth = project_vertices(
        reference, 128, 128, pitch=pitch, yaw=yaw, roll=roll
    )
    directions = np.asarray(screen[1:4] - screen[0], dtype=float)
    maximum = float(np.max(np.linalg.norm(directions, axis=1)))
    if not np.isfinite(directions).all() or maximum <= 1e-12:
        raise ValueError("The camera orientation cannot be projected.")
    return directions / maximum, np.asarray(depth[1:4] - depth[0], dtype=float)


def draw_photo_axis_guide(image, project_vertices, *, pitch: float, yaw: float, roll: float):
    """Annotate the interactive frame, without changing the scientific image."""

    directions, depth = photo_axis_directions(project_vertices, pitch, yaw, roll)
    output = image.copy()
    draw = ImageDraw.Draw(output)
    origin = np.array([88.0, float(output.height - 84)])
    length = 48.0
    colors = ((190, 55, 48), (45, 125, 70), (55, 98, 175))
    labels = ("X", "Y", "Z")
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    endpoints = origin + length * directions
    occupied = []
    for index in np.argsort(depth):
        index = int(index)
        endpoint = endpoints[index]
        start = tuple(np.rint(origin).astype(int))
        finish = tuple(np.rint(endpoint).astype(int))
        draw.line((start, finish), fill=(255, 255, 255), width=5)
        draw.line((start, finish), fill=colors[index], width=3)
        norm = float(np.linalg.norm(directions[index]))
        offset = (
            8.0 * directions[index] / norm
            if norm > 0.2 else np.array(((0, -18), (18, 0), (-18, 0))[index])
        )
        label_xy = endpoint + offset
        for _ in range(8):
            box = draw.textbbox(tuple(label_xy), labels[index], font=font)
            if not any(
                box[0] < other[2] + 2 and box[2] + 2 > other[0]
                and box[1] < other[3] + 2 and box[3] + 2 > other[1]
                for other in occupied
            ):
                break
            label_xy[1] -= 11
        box = draw.textbbox(tuple(label_xy), labels[index], font=font)
        occupied.append(box)
        draw.text(
            tuple(label_xy), labels[index], font=font, fill=colors[index],
            stroke_width=2, stroke_fill=(255, 255, 255),
        )
    x, y = np.rint(origin).astype(int)
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(30, 35, 38))
    return output


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
