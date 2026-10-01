"""Errors with actionable messages for the optional FinFindR adapter."""


class FinFindRError(RuntimeError):
    """Base class for FinFindR adapter failures."""


class FinFindRDependencyError(FinFindRError):
    """Raised when an optional inference dependency is unavailable."""


class FinFindRModelError(FinFindRError):
    """Raised when a model is missing, altered, or unsupported."""

