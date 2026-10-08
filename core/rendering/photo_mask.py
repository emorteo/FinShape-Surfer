"""Photo-ID-only learned-contour scene adaptation.

The measured curve snapshot consumed by CFD export is never modified. A
separately recorded, operator-reviewed image contour is substituted only while
the photograph-drape mesh is built.
"""

from __future__ import annotations

import copy
from typing import Mapping

import numpy as np
from shapely.geometry import Polygon

from core.rendering.airfoil_surface import (
    MAX_FULL_THICKNESS_RATIO,
    airfoil_mesh_strategy,
    rebuild_texture_scene_airfoil,
)


def photo_mask_snapshot(snapshot: Mapping) -> dict:
    """Return a shallow snapshot copy configured for Photo-ID mesh creation."""

    adapted = dict(snapshot)
    curves = snapshot.get("photo_mask_curves")
    image_outline = snapshot.get("photo_mask_image_outline")
    if curves is None or image_outline is None:
        adapted["photo_mask_applied"] = False
        adapted["photo_mask_base_clipped"] = False
        adapted["photo_mask_terminal_measured_splice"] = False
        return adapted

    standardized = np.asarray(curves.get("outline"), dtype=float)
    image = np.asarray(image_outline, dtype=float)
    if (
        standardized.ndim != 2
        or standardized.shape[1:] != (2,)
        or len(standardized) < 3
        or image.shape != standardized.shape
        or not np.isfinite(standardized).all()
        or not np.isfinite(image).all()
    ):
        raise ValueError("The learned Photo-ID contour snapshot is invalid.")
    adapted["photo_mask_base_clipped"] = bool(np.any(standardized[:, 1] < -1e-8))
    adapted["photo_mask_terminal_measured_splice"] = False
    try:
        standardized, image = _clip_photo_outline_to_base(standardized, image)
    except ValueError:
        standardized, image = _repair_terminal_photo_outline(
            standardized, image, snapshot
        )
        adapted["photo_mask_terminal_measured_splice"] = True
    adapted["curves"] = copy.deepcopy(curves)
    adapted["curves"]["outline"] = standardized.tolist()
    adapted["image_outline"] = image.tolist()
    adapted["photo_mask_applied"] = True
    return adapted


def _clip_photo_outline_to_base(standardized, image):
    """Clip below-base Photo-ID samples while retaining aligned image points.

    The reviewed A-to-C30 path can briefly run below the B-C30 attachment
    line. Clip its line segments at that line; do not replace the detected
    edge or alter the measured contour used by CFD.
    """

    if float(standardized[:, 1].min()) >= -1e-8:
        return standardized, image

    result_standardized = [standardized[0]]
    result_image = [image[0]]
    for start, end, image_start, image_end in zip(
        standardized[:-1], standardized[1:], image[:-1], image[1:]
    ):
        start_inside = start[1] >= 0.0
        end_inside = end[1] >= 0.0
        if start_inside != end_inside:
            fraction = float(-start[1] / (end[1] - start[1]))
            result_standardized.append(start + fraction * (end - start))
            result_image.append(image_start + fraction * (image_end - image_start))
        if end_inside:
            result_standardized.append(end)
            result_image.append(image_end)

    clipped = np.asarray(result_standardized, dtype=float)
    image_clipped = np.asarray(result_image, dtype=float)
    keep = np.r_[True, np.any(np.diff(clipped, axis=0) != 0.0, axis=1)]
    clipped, image_clipped = clipped[keep], image_clipped[keep]
    clipped[0], clipped[-1] = standardized[0], standardized[-1]
    image_clipped[0], image_clipped[-1] = image[0], image[-1]
    if len(clipped) < 3:
        raise ValueError("The detected Photo-ID edge has fewer than three points.")
    polygon = Polygon(clipped)
    if not polygon.is_valid or polygon.area <= 0.0:
        raise ValueError(
            "The detected Photo-ID edge cannot form one fin above the attachment base; "
            "review the edge or landmarks."
        )
    return clipped, image_clipped


def _repair_terminal_photo_outline(standardized, image, snapshot):
    """Join a below-base terminal tail to the operator's measured base curve."""

    message = (
        "The detected Photo-ID edge cannot form one fin above the attachment base; "
        "review the edge or landmarks."
    )
    bad = np.flatnonzero(standardized[:, 1] < -1e-8)
    if not len(bad):
        raise ValueError(message)
    excursion = -float(standardized[bad, 1].min())
    if excursion > 0.25 or np.max(standardized[bad[0]:, 1]) > 0.25:
        raise ValueError(message)

    measured = np.asarray(snapshot.get("curves", {}).get("outline"), dtype=float)
    measured_image = np.asarray(snapshot.get("image_outline"), dtype=float)
    if (
        measured.ndim != 2 or measured.shape[1] != 2
        or measured_image.shape != measured.shape or len(measured) < 3
        or not np.isfinite(measured).all() or not np.isfinite(measured_image).all()
        or np.linalg.norm(measured[-1] - standardized[-1]) > 1e-5
        or np.linalg.norm(measured_image[-1] - image[-1]) > 1e-3
    ):
        raise ValueError(message)

    transition_y = max(0.05, 3.0 * excursion)
    before_bad = np.flatnonzero(standardized[:bad[0], 1] >= transition_y)
    if not len(before_bad):
        raise ValueError(message)
    keep_until = int(before_bad[-1])
    if np.max(standardized[keep_until + 1:, 1]) > transition_y + 0.05:
        raise ValueError(message)

    apex = int(np.argmax(measured[:, 1]))
    if apex >= len(measured) - 1:
        raise ValueError(message)
    tail = np.arange(apex + 1, len(measured))
    join = int(tail[np.argmin(np.linalg.norm(
        measured[tail] - standardized[keep_until], axis=1
    ))])
    if np.linalg.norm(measured[join] - standardized[keep_until]) > 0.3:
        raise ValueError(message)

    repaired = np.vstack((standardized[:keep_until + 1], measured[join:]))
    repaired_image = np.vstack((image[:keep_until + 1], measured_image[join:]))
    polygon = Polygon(repaired)
    if (
        not polygon.is_valid or polygon.area <= 0.0
        or np.min(repaired[:, 1]) < -1e-8
    ):
        raise ValueError(message)
    return repaired, repaired_image


class PhotoMaskTextureProxy:
    """Delegate texture rendering after applying the Photo-ID-only contour."""

    def __init__(self, renderer):
        self._renderer = renderer
        self.VERSION = f"{renderer.VERSION}+learned-photo-mask+adaptive-airfoil"

    def __getattr__(self, name):
        return getattr(self._renderer, name)

    def build_texture_scene(self, snapshot, exporter, photo_side, **kwargs):
        adapted = photo_mask_snapshot(snapshot)
        scene = self._renderer.build_texture_scene(
            adapted, exporter, photo_side, **kwargs
        )
        return rebuild_texture_scene_airfoil(scene)

    def scene_metadata(self, scene, snapshot):
        adapted = photo_mask_snapshot(snapshot)
        metadata = self._renderer.scene_metadata(scene, adapted)
        applied = bool(adapted.get("photo_mask_applied"))
        metadata.update({
            "photo_id_boundary_source": (
                "operator-fitted Bezier B-to-D-to-A leading edge and "
                "operator-anchored FinFindR A-to-C30 trailing edge"
                if applied else "measured Bezier fallback"
            ),
            "learned_photo_mask_applied": applied,
            "photo_mask_scope": "high-resolution Photo-ID rendering only",
            "photo_surface_geometry": airfoil_mesh_strategy(scene),
            "photo_surface_vertices": len(scene.vertices),
            "photo_surface_triangles": len(scene.faces),
            "photo_surface_max_full_thickness_ratio": MAX_FULL_THICKNESS_RATIO,
            "auxiliary_silhouette_geometry": False,
            "duplicate_surface_geometry": False,
            "measurement_geometry_changed": False,
            "cfd_surface_changed": False,
            "photo_mask_fallback": False,
            "photo_mask_base_clipped": adapted["photo_mask_base_clipped"],
            "photo_mask_terminal_measured_splice": adapted[
                "photo_mask_terminal_measured_splice"
            ],
        })
        return metadata
