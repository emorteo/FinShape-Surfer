"""Locate and verify the exact FinFindR 0.1.21 model assets."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .errors import FinFindRModelError

FINFINDR_REPOSITORY = "https://github.com/haimeh/finFindR.git"
FINFINDR_COMMIT = "5d7babd208af9642e440ef9143ec679c1154ff91"
DEFAULT_CACHE_DIRECTORY = Path(
    ".cache/finshape-surfer/finfindr-0.1.21/source/inst/extdata"
)

CROPPER_SYMBOL = "cropperInit-symbol.json"
CROPPER_PARAMS = "cropperInit-0941.params"
EDGE_SYMBOL = "SWA_cont2_traceLong7_bn_6,10,5_RGB_fin-symbol.json"
EDGE_PARAMS = "SWA_cont2_traceLong7_bn_6,10,5_RGB_fin-0000.params"

EXPECTED_SHA256 = {
    CROPPER_SYMBOL: "31adf8e3bebfbcdea7466110ca7b6011b9e6e9f3cbd4f2ce1e916664b0b53c74",
    CROPPER_PARAMS: "a0126ad21eea359bb567fdceb3cab3c1585a39b0db2a3017666cb2ce292966e9",
    EDGE_SYMBOL: "fcdb42b69399986ebbcd41be45343f75a04f347ec55d5021a5b530305ee5e5fc",
    EDGE_PARAMS: "06f885818de586857e5537b23c8e99e3333a3b3482a7b856ae7d81308db12d61",
}


@dataclass(frozen=True)
class ModelBundle:
    directory: Path
    cropper_symbol: Path
    cropper_params: Path
    edge_symbol: Path
    edge_params: Path
    upstream_commit: str = FINFINDR_COMMIT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_hashes(directory: Path) -> Dict[str, str]:
    return {name: sha256_file(Path(directory) / name) for name in (
        CROPPER_SYMBOL, CROPPER_PARAMS, EDGE_SYMBOL, EDGE_PARAMS
    )}


def default_model_directory() -> Path:
    """Return the model location created by the preparation utility."""

    return Path.home() / DEFAULT_CACHE_DIRECTORY


def locate_model_bundle(directory: Optional[Path] = None) -> ModelBundle:
    """Resolve a model bundle without downloading or changing user files."""

    configured = (
        directory
        or os.environ.get("FINSHAPE_FINFINR_MODEL_DIR")
        or default_model_directory()
    )
    if not configured:
        raise FinFindRModelError(
            "FinFindR models are not configured. Run tools/prepare_finfindr_models.py."
        )
    base = Path(configured).expanduser().resolve()
    paths = {name: base / name for name in (
        CROPPER_SYMBOL, CROPPER_PARAMS, EDGE_SYMBOL, EDGE_PARAMS
    )}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FinFindRModelError(
            "Incomplete FinFindR model bundle. Run tools/prepare_finfindr_models.py; "
            "missing: " + ", ".join(missing)
        )
    actual_hashes = model_hashes(base)
    altered = [
        f"{name} ({actual_hashes[name]})"
        for name, expected in EXPECTED_SHA256.items()
        if actual_hashes[name] != expected
    ]
    if altered:
        raise FinFindRModelError(
            "FinFindR model files do not match pinned commit "
            f"{FINFINDR_COMMIT}: " + ", ".join(altered)
        )
    return ModelBundle(
        directory=base,
        cropper_symbol=paths[CROPPER_SYMBOL],
        cropper_params=paths[CROPPER_PARAMS],
        edge_symbol=paths[EDGE_SYMBOL],
        edge_params=paths[EDGE_PARAMS],
    )
