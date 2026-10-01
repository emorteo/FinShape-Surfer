"""Auditable FinFindR 0.1.21 inference and post-processing adapters."""

from .assets import ModelBundle, locate_model_bundle
from .detection import (
    BoundingBox,
    FinContourResult,
    FinDetectionResult,
    FinFindRAssistant,
    TrailingEdgeResult,
)
from .errors import FinFindRDependencyError, FinFindRError, FinFindRModelError

__all__ = [
    "BoundingBox",
    "FinContourResult",
    "FinDetectionResult",
    "FinFindRAssistant",
    "FinFindRDependencyError",
    "FinFindRError",
    "FinFindRModelError",
    "ModelBundle",
    "TrailingEdgeResult",
    "locate_model_bundle",
]
