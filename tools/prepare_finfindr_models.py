#!/usr/bin/env python3
"""Fetch the exact external FinFindR 0.1.21 model files into a local cache."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.vision.finfindr.assets import (  # noqa: E402
    FINFINDR_COMMIT,
    FINFINDR_REPOSITORY,
    locate_model_bundle,
)


def _run(arguments, cwd=None):
    subprocess.run(arguments, cwd=cwd, check=True)


def prepare(cache_directory: Path) -> Path:
    checkout = cache_directory.expanduser().resolve() / "source"
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", str(checkout)])
        _run(["git", "remote", "add", "origin", FINFINDR_REPOSITORY], cwd=checkout)
    elif not (checkout / ".git").is_dir():
        raise RuntimeError(f"Refusing to reuse a non-Git directory: {checkout}")

    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        revision = ""
    model_directory = checkout / "inst" / "extdata"
    if revision == FINFINDR_COMMIT:
        locate_model_bundle(model_directory)
        return model_directory

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=checkout, text=True
    ).strip()
    if dirty:
        raise RuntimeError(
            f"Refusing to replace a modified FinFindR cache checkout: {checkout}"
        )

    _run(["git", "fetch", "--depth=1", "origin", FINFINDR_COMMIT], cwd=checkout)
    _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=checkout)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if revision != FINFINDR_COMMIT:
        raise RuntimeError(f"Expected FinFindR {FINFINDR_COMMIT}, checked out {revision}.")

    locate_model_bundle(model_directory)
    return model_directory


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch and verify the exact external FinFindR 0.1.21 checkpoint."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "finshape-surfer" / "finfindr-0.1.21",
        help="Local cache root; no files are copied into the FinShape Surfer repository.",
    )
    args = parser.parse_args()
    model_directory = prepare(args.cache_dir)
    print("Verified FinFindR model directory:")
    print(model_directory)
    print("Configure this shell with:")
    print(f"export FINSHAPE_FINFINR_MODEL_DIR={str(model_directory)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
