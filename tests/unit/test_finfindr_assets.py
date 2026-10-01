from pathlib import Path

import pytest

from core.vision.finfindr import assets
from core.vision.finfindr.assets import (
    CROPPER_PARAMS,
    CROPPER_SYMBOL,
    EDGE_PARAMS,
    EDGE_SYMBOL,
    locate_model_bundle,
)
from core.vision.finfindr.errors import FinFindRModelError


def test_model_bundle_rejects_missing_files(tmp_path):
    with pytest.raises(FinFindRModelError, match="missing"):
        locate_model_bundle(tmp_path)


def test_model_bundle_rejects_unpinned_content(tmp_path):
    for name in (CROPPER_SYMBOL, CROPPER_PARAMS, EDGE_SYMBOL, EDGE_PARAMS):
        (tmp_path / name).write_bytes(b"altered")
    with pytest.raises(FinFindRModelError, match="do not match pinned commit"):
        locate_model_bundle(tmp_path)


def test_model_bundle_discovers_prepared_default_cache(monkeypatch, tmp_path):
    default_directory = (
        tmp_path / ".cache/finshape-surfer/finfindr-0.1.21/source/inst/extdata"
    )
    default_directory.mkdir(parents=True)
    for name in (CROPPER_SYMBOL, CROPPER_PARAMS, EDGE_SYMBOL, EDGE_PARAMS):
        (default_directory / name).write_bytes(b"verified by the hash stub")

    monkeypatch.delenv("FINSHAPE_FINFINR_MODEL_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        assets,
        "model_hashes",
        lambda directory: dict(assets.EXPECTED_SHA256),
    )

    bundle = locate_model_bundle()

    assert bundle.directory == default_directory.resolve()


def test_environment_model_directory_overrides_default_cache(monkeypatch, tmp_path):
    configured = tmp_path / "configured"
    configured.mkdir()
    for name in (CROPPER_SYMBOL, CROPPER_PARAMS, EDGE_SYMBOL, EDGE_PARAMS):
        (configured / name).write_bytes(b"verified by the hash stub")

    monkeypatch.setenv("FINSHAPE_FINFINR_MODEL_DIR", str(configured))
    monkeypatch.setattr(
        assets,
        "model_hashes",
        lambda directory: dict(assets.EXPECTED_SHA256),
    )

    bundle = locate_model_bundle()

    assert bundle.directory == configured.resolve()
