#!/usr/bin/env python3
"""Experimental one-side photograph draping for FinShape Surfer.

This module maps the already transformed measurement image back from the
standardized AB=10 planform and colors one skin of the existing modeled
surface. It does not estimate camera pose, correct perspective, infer an
anatomical side, identify an animal, or alter the CFD surface exporter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


VERSION = "0.1.0"
PHOTO_SIDE_TO_SKIN = {
    "left": "positive_z_skin",
    "right": "negative_z_skin",
}
METHOD = (
    "Experimental boundary-conforming 2D photograph drape. The inverse of the "
    "FinShape AB=10 standardization is recovered from B, A and C30. The complete "
    "measured Bezier boundary is checked against that mapping. The B-C30 base is "
    "an exact planform anchor and is not textured. Surface curvature produces the "
    "drape stretch; no extra biological deformation is invented."
)
LIMITATION = (
    "No camera calibration, lens-distortion correction, camera-pose estimation, "
    "visibility solution or perspective correction is performed. This first-"
    "iteration overlay is for visual inspection only and is not validated Photo-ID."
)


@dataclass(frozen=True)
class TextureScene:
    vertices: np.ndarray
    faces: np.ndarray
    face_labels: np.ndarray
    face_colors: np.ndarray
    outline: np.ndarray
    photo_side: str
    textured_skin: str
    coverage_fraction: float
    landmark_residual_px: Mapping[str, float]
    boundary_residual_max_px: float
    base_residual_max_px: float
    affine_planform_to_image: np.ndarray
    surface_stretch_min: float
    surface_stretch_mean: float
    surface_stretch_max: float


def _points(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite two-dimensional points.")
    return array


def validate_photo_side(photo_side: str) -> str:
    side = str(photo_side).strip().lower()
    if side not in PHOTO_SIDE_TO_SKIN:
        raise ValueError("Select the photographed anatomical side explicitly: Left or Right.")
    return side


def fit_planform_to_image_affine(
    standardized_landmarks: Mapping[str, Sequence[float]],
    image_landmarks: Mapping[str, Sequence[float]],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Recover the exact inverse of standardization from non-collinear anchors."""
    anchors = ("B", "A", "C30")
    for key in anchors:
        if key not in standardized_landmarks or key not in image_landmarks:
            raise ValueError(f"Missing {key} landmark for texture registration.")
    source = _points([standardized_landmarks[key] for key in anchors], "standardized anchors")
    target = _points([image_landmarks[key] for key in anchors], "image anchors")
    design = np.column_stack([source, np.ones(len(source))])
    if np.linalg.matrix_rank(design) != 3:
        raise ValueError("B, A and C30 are collinear; a two-dimensional image mapping cannot be recovered.")
    affine = np.linalg.solve(design, target)

    residuals: Dict[str, float] = {}
    for key in sorted(set(standardized_landmarks) & set(image_landmarks)):
        standardized = np.asarray(standardized_landmarks[key], dtype=float)
        measured = np.asarray(image_landmarks[key], dtype=float)
        if standardized.shape != (2,) or measured.shape != (2,):
            raise ValueError(f"Invalid {key} landmark for texture registration.")
        predicted = np.append(standardized, 1.0) @ affine
        residuals[key] = float(np.linalg.norm(predicted - measured))
    return affine, residuals


def map_planform_to_image(points: Sequence[Sequence[float]], affine: np.ndarray) -> np.ndarray:
    planform = _points(points, "planform points")
    matrix = np.asarray(affine, dtype=float)
    if matrix.shape != (3, 2) or not np.isfinite(matrix).all():
        raise ValueError("The planform-to-image affine matrix must be finite and 3 by 2.")
    return np.column_stack([planform, np.ones(len(planform))]) @ matrix


def sample_image_bilinear(image, pixel_points: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Sample RGB values without clamping out-of-image coordinates."""
    if hasattr(image, "convert"):
        pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
    else:
        pixels = np.asarray(image, dtype=np.float64)
    if pixels.ndim != 3 or pixels.shape[2] < 3 or pixels.shape[0] < 2 or pixels.shape[1] < 2:
        raise ValueError("The working texture image must be an RGB image of at least 2 by 2 pixels.")
    pixels = pixels[:, :, :3]
    points = _points(pixel_points, "texture sample coordinates")
    height, width = pixels.shape[:2]
    valid = (
        (points[:, 0] >= 0.0) & (points[:, 0] <= width - 1) &
        (points[:, 1] >= 0.0) & (points[:, 1] <= height - 1)
    )
    colors = np.full((len(points), 3), 0.72, dtype=float)
    if np.any(valid):
        x = points[valid, 0]
        y = points[valid, 1]
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, width - 1)
        y1 = np.minimum(y0 + 1, height - 1)
        wx = (x - x0)[:, None]
        wy = (y - y0)[:, None]
        top = pixels[y0, x0] * (1.0 - wx) + pixels[y0, x1] * wx
        bottom = pixels[y1, x0] * (1.0 - wx) + pixels[y1, x1] * wx
        colors[valid] = (top * (1.0 - wy) + bottom * wy) / 255.0
    return np.clip(colors, 0.0, 1.0), valid


def _build_normalized_surface(exporter, curves, landmarks, edge_min_thickness: float, subdivisions: int):
    required = ("_polygon_from_input", "_triangulate", "_thickness")
    if any(not callable(getattr(exporter, name, None)) for name in required):
        raise RuntimeError("The installed FinShape surface module is incompatible with texture rendering.")
    if type(subdivisions) is not int or not 0 <= subdivisions <= 2:
        raise ValueError("Texture subdivisions must be an integer from 0 to 2.")
    if not np.isfinite(edge_min_thickness) or edge_min_thickness <= 0:
        raise ValueError("Texture surface thickness floor must be finite and positive.")

    data = {
        "schema": "finshape.surface-input/1",
        "curves": {key: np.asarray(value, dtype=float).tolist() for key, value in curves.items()},
        "landmarks": {key: np.asarray(value, dtype=float).tolist() for key, value in landmarks.items()},
    }
    polygon, _input_points, _landmarks = exporter._polygon_from_input(data)
    xy, planar_faces, boundary = exporter._triangulate(polygon, subdivisions)
    thickness = exporter._thickness(xy, boundary, polygon, float(edge_min_thickness))
    n = len(xy)
    vertices = np.vstack([
        np.column_stack([xy, thickness / 2.0]),
        np.column_stack([xy, -thickness / 2.0]),
    ])
    faces = [*planar_faces.tolist(), *(planar_faces[:, ::-1] + n).tolist()]
    labels = ["positive_z_skin"] * len(planar_faces) + ["negative_z_skin"] * len(planar_faces)
    for a, b in boundary:
        faces.extend([[int(a), int(a + n), int(b)], [int(b), int(a + n), int(b + n)]])
        label = "base_attachment" if abs(xy[a, 1]) < 1e-9 and abs(xy[b, 1]) < 1e-9 else "edge_wall"
        labels.extend([label, label])
    return vertices, np.asarray(faces, dtype=np.int64), np.asarray(labels, dtype=str)


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    return 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1,
    )


def build_texture_scene(
    snapshot: Mapping,
    exporter,
    photo_side: str,
    *,
    edge_min_thickness: float = 0.05,
    subdivisions: int = 1,
) -> TextureScene:
    """Build a cached one-side scene from an immutable measurement snapshot."""
    side = validate_photo_side(photo_side)
    required = (
        "working_image", "curves", "standardized_landmarks", "image_landmarks",
        "image_outline", "flip_applied",
    )
    missing = [key for key in required if key not in snapshot]
    if missing:
        raise ValueError(f"Texture snapshot is incomplete: {', '.join(missing)}")

    affine, landmark_residuals = fit_planform_to_image_affine(
        snapshot["standardized_landmarks"], snapshot["image_landmarks"],
    )
    outline = _points(snapshot["curves"]["outline"], "standardized Bezier outline")
    image_outline = _points(snapshot["image_outline"], "image Bezier outline")
    if len(outline) != len(image_outline):
        raise ValueError("Standardized and image Bezier outlines have different sample counts.")
    mapped_outline = map_planform_to_image(outline, affine)
    boundary_residual = np.linalg.norm(mapped_outline - image_outline, axis=1)
    boundary_max = float(boundary_residual.max(initial=0.0))

    base_standardized = np.linspace(
        np.asarray(snapshot["standardized_landmarks"]["B"], dtype=float),
        np.asarray(snapshot["standardized_landmarks"]["C30"], dtype=float),
        21,
    )
    base_image = np.linspace(
        np.asarray(snapshot["image_landmarks"]["B"], dtype=float),
        np.asarray(snapshot["image_landmarks"]["C30"], dtype=float),
        21,
    )
    base_max = float(np.linalg.norm(map_planform_to_image(base_standardized, affine) - base_image, axis=1).max())
    tolerance = 1e-6
    if boundary_max > tolerance or base_max > tolerance or max(landmark_residuals.values(), default=0.0) > tolerance:
        raise ValueError(
            "The image and standardized geometry do not share one transform; texture registration was refused."
        )

    vertices, faces, labels = _build_normalized_surface(
        exporter, snapshot["curves"], snapshot["standardized_landmarks"],
        edge_min_thickness=edge_min_thickness, subdivisions=subdivisions,
    )
    textured_skin = PHOTO_SIDE_TO_SKIN[side]
    selected = labels == textured_skin
    face_colors = np.empty((len(faces), 4), dtype=float)
    face_colors[:] = (0.78, 0.80, 0.82, 1.0)
    face_colors[labels == "edge_wall"] = (0.30, 0.32, 0.34, 1.0)
    face_colors[labels == "base_attachment"] = (0.46, 0.48, 0.50, 1.0)
    face_colors[(labels == "positive_z_skin") | (labels == "negative_z_skin")] = (0.82, 0.83, 0.84, 1.0)

    selected_faces = faces[selected]
    centroids_xy = vertices[selected_faces, :2].mean(axis=1)
    pixel_points = map_planform_to_image(centroids_xy, affine)
    sampled_rgb, valid = sample_image_bilinear(snapshot["working_image"], pixel_points)
    selected_colors = np.column_stack([sampled_rgb, np.ones(len(sampled_rgb))])
    selected_colors[~valid] = (0.55, 0.55, 0.55, 1.0)
    face_colors[selected] = selected_colors

    planar_areas = _triangle_areas(np.column_stack([vertices[:, :2], np.zeros(len(vertices))]), selected_faces)
    surface_areas = _triangle_areas(vertices, selected_faces)
    coverage = float(planar_areas[valid].sum() / planar_areas.sum()) if planar_areas.sum() > 0 else 0.0
    nonzero = planar_areas > 1e-15
    stretch = surface_areas[nonzero] / planar_areas[nonzero]
    if not len(stretch) or not np.isfinite(stretch).all():
        raise ValueError("Texture surface stretch could not be evaluated.")

    return TextureScene(
        vertices=vertices,
        faces=faces,
        face_labels=labels,
        face_colors=face_colors,
        outline=outline,
        photo_side=side,
        textured_skin=textured_skin,
        coverage_fraction=coverage,
        landmark_residual_px=landmark_residuals,
        boundary_residual_max_px=boundary_max,
        base_residual_max_px=base_max,
        affine_planform_to_image=affine,
        surface_stretch_min=float(stretch.min()),
        surface_stretch_mean=float(np.average(stretch, weights=planar_areas[nonzero])),
        surface_stretch_max=float(stretch.max()),
    )


def draw_texture_scene(ax, scene: TextureScene, fin3d, fin_id: str, *, pitch: float, yaw: float, roll: float):
    """Draw one cached textured mesh into an existing Matplotlib 3D axes."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ax.clear()
    display_vertices = scene.vertices[:, [0, 2, 1]]
    collection = Poly3DCollection(
        display_vertices[scene.faces],
        facecolors=scene.face_colors,
        edgecolors="none",
        linewidths=0.0,
        antialiaseds=False,
        shade=False,
        zsort="average",
    )
    ax.add_collection3d(collection)
    ax.plot(scene.outline[:, 0], np.zeros(len(scene.outline)), scene.outline[:, 1], color="black", linewidth=1.6)

    x_min, x_max = float(scene.vertices[:, 0].min()), float(scene.vertices[:, 0].max())
    y_min, y_max = float(scene.vertices[:, 1].min()), float(scene.vertices[:, 1].max())
    z_abs = max(float(np.abs(scene.vertices[:, 2]).max()), 0.6)
    x_pad = max((x_max - x_min) * 0.06, 0.25)
    y_pad = max((y_max - y_min) * 0.02, 0.10)
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (-z_abs * 1.15, z_abs * 1.15)
    zlim = (min(0.0, y_min) - y_pad, y_max + y_pad)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel("X (AB=10)", fontsize=9, labelpad=7)
    ax.set_ylabel("Z thickness", fontsize=9, labelpad=7)
    ax.set_zlabel("Y height", fontsize=9, labelpad=7)
    ax.tick_params(labelsize=8, pad=1)
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    try:
        ax.view_init(elev=float(pitch), azim=float(yaw), roll=float(roll))
    except TypeError:
        ax.view_init(elev=float(pitch), azim=float(yaw))
    ax.set_title(f"{fin_id}: experimental {scene.photo_side}-side photo drape")
    ax.text2D(
        0.01, 0.98,
        "Visual registration only: no camera-pose or perspective correction.",
        transform=ax.transAxes, fontsize=8, color="0.25", va="top",
    )
    return collection


def scene_metadata(scene: TextureScene, snapshot: Mapping) -> Dict:
    return {
        "schema": "finshape.texture-registration/1",
        "texture_module_version": VERSION,
        "fin_id": snapshot.get("fin_id"),
        "photo_side_operator_selected": scene.photo_side,
        "geometric_skin": scene.textured_skin,
        "flip_applied_before_measurement": bool(snapshot["flip_applied"]),
        "working_image_size_px": list(snapshot["working_image"].size),
        "source_image": snapshot.get("source_image"),
        "source_image_sha256": snapshot.get("source_image_sha256"),
        "crop_box": snapshot.get("crop_box"),
        "rotate_deg": snapshot.get("rotate_deg"),
        "affine_planform_to_image": scene.affine_planform_to_image.tolist(),
        "landmark_mapping_residual_px": dict(scene.landmark_residual_px),
        "bezier_boundary_mapping_max_residual_px": scene.boundary_residual_max_px,
        "base_B_C30_mapping_max_residual_px": scene.base_residual_max_px,
        "texture_coverage_fraction_by_planform_area": scene.coverage_fraction,
        "camera_calibrated": False,
        "camera_pose_estimated": False,
        "lens_distortion_corrected": False,
        "perspective_corrected": False,
        "identity_matching_performed": False,
        "surface_to_planform_area_stretch": {
            "minimum": scene.surface_stretch_min,
            "area_weighted_mean": scene.surface_stretch_mean,
            "maximum": scene.surface_stretch_max,
        },
        "method": METHOD,
        "limitations": LIMITATION,
        "anatomical_side_note": (
            "Horizontal image flipping changes pixel orientation but not the animal's anatomical side. "
            "The anatomical side is therefore an explicit operator selection."
        ),
        "base_note": "The B-C30 attachment is an exact planform anchor and is not photo-textured.",
    }
