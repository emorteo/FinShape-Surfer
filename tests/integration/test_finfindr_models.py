import json
import os
from pathlib import Path

import numpy as np
import pytest

from core.vision.finfindr import locate_model_bundle
from core.vision.finfindr.executor import MXNetSymbolExecutor

MODEL_DIRECTORY = os.environ.get("FINSHAPE_FINFINR_MODEL_DIR")
GOLDEN = json.loads(
    (Path(__file__).resolve().parents[2]
     / "references" / "finfindr-0.1.21" / "golden" / "ml-inference-golden.json")
    .read_text(encoding="utf-8")
)
pytestmark = pytest.mark.skipif(
    not MODEL_DIRECTORY,
    reason="Set FINSHAPE_FINFINR_MODEL_DIR to run the external FinFindR checkpoint tests.",
)


def _gradient(height, width):
    values = np.linspace(0.0, 1.0, 3 * height * width, dtype=np.float32)
    return values.reshape(1, 3, height, width)


def test_exact_crop_checkpoint_executes_deterministically():
    bundle = locate_model_bundle(Path(MODEL_DIRECTORY))
    model = MXNetSymbolExecutor(bundle.cropper_symbol, bundle.cropper_params)
    first = model.predict(_gradient(150, 225))
    second = model.predict(_gradient(150, 225))
    assert first.shape == (1, 2, 150, 225)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(
        first.mean(axis=(0, 2, 3)), GOLDEN["crop"]["channel_means"], rtol=2e-5, atol=2e-7
    )


def test_exact_six_class_checkpoint_executes_and_normalizes_classes():
    bundle = locate_model_bundle(Path(MODEL_DIRECTORY))
    model = MXNetSymbolExecutor(bundle.edge_symbol, bundle.edge_params)
    output = model.predict(_gradient(140, 200))
    assert output.shape == (1, 6, 140, 200)
    assert np.isfinite(output).all()
    np.testing.assert_allclose(output.sum(axis=1), 1.0, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        output.mean(axis=(0, 2, 3)), GOLDEN["edge"]["channel_means"], rtol=2e-5, atol=2e-7
    )
