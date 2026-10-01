"""Photo-ID-only learned-contour scene adaptation.

The measured curve snapshot consumed by CFD export is never modified. A
separately recorded, operator-reviewed image contour is substituted only while
the photograph-drape mesh is built.
"""

from __future__ import annotations

import copy
from typing import Mapping

import numpy as np

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
    adapted["curves"] = copy.deepcopy(curves)
    adapted["image_outline"] = image.tolist()
    adapted["photo_mask_applied"] = True
    return adapted


class PhotoMaskTextureProxy:
    """Delegate texture rendering after applying the Photo-ID-only contour."""

    def __init__(self, renderer):
        self._renderer = renderer
        self.VERSION = f"{renderer.VERSION}+learned-photo-mask+adaptive-airfoil"

    def __getattr__(self, name):
        return getattr(self._renderer, name)

    def build_texture_scene(self, snapshot, exporter, photo_side, **kwargs):
        scene = self._renderer.build_texture_scene(
            photo_mask_snapshot(snapshot), exporter, photo_side, **kwargs
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
        })
        return metadata
