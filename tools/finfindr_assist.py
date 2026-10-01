#!/usr/bin/env python3
"""Generate reviewable automatic fin and trailing-edge suggestions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.vision.finfindr import FinFindRAssistant  # noqa: E402
from core.vision.finfindr.assets import sha256_file  # noqa: E402
from core.vision.finfindr.detection import (  # noqa: E402
    TrailingEdgeResult,
    draw_assistance_preview,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run FinFindR-assisted detection without changing the source image or "
            "FinShape Surfer measurements. All output requires operator review."
        )
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--skip-fin-detection", action="store_true")
    parser.add_argument("--skip-trailing-edge", action="store_true")
    parser.add_argument("--crop-threshold", type=float, default=0.45)
    parser.add_argument("--edge-threshold", type=float, default=0.35)
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    image = Image.open(image_path).convert("RGB")
    assistant = FinFindRAssistant(args.model_dir)
    detection = None
    edge = None
    warnings = []
    if not args.skip_fin_detection:
        try:
            detection = assistant.detect_fin_crop(
                image, coarse_threshold=args.crop_threshold
            )
        except Exception as exc:
            warnings.append(f"Automatic fin detection failed: {exc}")
    if not args.skip_trailing_edge:
        try:
            edge_image = image
            offset_x = offset_y = 0
            if detection is not None and detection.boxes:
                best = detection.boxes[0]
                offset_x, offset_y = best.x0, best.y0
                edge_image = image.crop((best.x0, best.y0, best.x1, best.y1))
            local_edge = assistant.suggest_trailing_edge(edge_image, threshold=args.edge_threshold)
            points = local_edge.points.copy()
            points[:, 0] += offset_x
            points[:, 1] += offset_y
            edge = TrailingEdgeResult(
                points=points,
                confidence=local_edge.confidence,
                trailing_probability=local_edge.trailing_probability,
                tip_probability=local_edge.tip_probability,
                provenance=dict(local_edge.provenance, crop_offset_xy=[offset_x, offset_y]),
            )
        except Exception as exc:
            warnings.append(f"Trailing-edge suggestion failed: {exc}")

    preview = draw_assistance_preview(image, detection, edge)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    preview.save(args.preview)
    metadata = {
        "source_image": image_path.name,
        "source_image_sha256": sha256_file(image_path),
        "source_absolute_path_recorded": False,
        "warning": "Automatic suggestions require operator review and were not applied to measurements.",
        "processing_warnings": warnings,
        "fin_detection": None if detection is None else {
            "boxes": [box.__dict__ for box in detection.boxes],
            "threshold": detection.threshold,
            "provenance": detection.provenance,
        },
        "trailing_edge": None if edge is None else {
            "points_xy": edge.points.tolist(),
            "confidence": edge.confidence,
            "threshold": args.edge_threshold,
            "provenance": edge.provenance,
        },
        "provenance": assistant.provenance,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Review preview: {args.preview.resolve()}")
    print(f"Provenance: {args.metadata.resolve()}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    print("No crop, landmark, Bezier control, 2D contour, or 3D geometry was modified.")
    return 0 if detection is not None or edge is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
