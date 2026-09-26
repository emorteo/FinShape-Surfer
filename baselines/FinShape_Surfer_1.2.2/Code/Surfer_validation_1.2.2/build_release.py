#!/usr/bin/env python3
"""Build the deterministic FinShape Surfer 1.2.2 release archive."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Code" / "FinShape_Surfer_1.2.2_release.zip"
PREFIX = Path("FinShape_Surfer_1.2.2")
FILES = (
    "Code/FinShape_Surfer_GUI_1.2.2.py",
    "Code/FinShapeTextureHQ_0.2.0.py",
    "Code/FinShape_Surfer_GUI_1.2.1.py",
    "Code/FinShape_Surfer_GUI_1.2.0.py",
    "Code/FinShape_Surfer_GUI_1.1.3.py",
    "Code/FinShapeTextureHQ_0.1.0.py",
    "Code/FinShapeTexture_0.1.0.py",
    "Code/FinShape2D_R-GUI_2.0.20.py",
    "Code/FinShape3D_1.1.20.py",
    "Code/FinShapeCFD_0.1.0.py",
    "Code/requirements-cfd-0.1.0.txt",
    "Code/assets/dorsal_fin_tricolor_otsu.png",
    "Code/Surfer_validation_1.2.2/test_hq_view_fixes.py",
    "Code/Surfer_validation_1.2.2/test_live_tk_hq_fixes.py",
    "Code/Surfer_validation_1.2.2/commands_executed.txt",
    "Code/Surfer_validation_1.2.2/TEST_COMPLETION.md",
    "Code/Surfer_validation_1.2.2/results.json",
    "Code/Surfer_validation_1.2.2/README_release.md",
    "Code/Surfer_validation_1.2.2/build_release.py",
    "Code/Surfer_validation_1.2.2/artifacts/antialiased_oblique_silhouette.png",
    "Code/Surfer_validation_1.2.2/artifacts/live_hq_122_right_front.png",
    "Documentation/FinShape_Surfer_1.2.2_3D_Display_Corrections.md",
)


def fixed_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=(2026, 9, 11, 12, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def main() -> None:
    missing = [relative for relative in FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError("Release inputs missing: " + ", ".join(missing))
    checksums = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in FILES
    }
    with ZipFile(OUTPUT, "w") as archive:
        for relative in FILES:
            archive.writestr(fixed_info(str(PREFIX / relative)), (ROOT / relative).read_bytes())
        archive.writestr(
            fixed_info(str(PREFIX / "README.md")),
            (ROOT / "Code/Surfer_validation_1.2.2/README_release.md").read_bytes(),
        )
        archive.writestr(
            fixed_info(str(PREFIX / "SHA256.json")),
            (json.dumps(checksums, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    print(OUTPUT)


if __name__ == "__main__":
    main()
