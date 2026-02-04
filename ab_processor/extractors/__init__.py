"""Feature extractor registry."""

from typing import Dict, Type
from ..base import FeatureExtractor

# Registry populated by imports below
EXTRACTOR_REGISTRY: Dict[str, Type[FeatureExtractor]] = {}


def register_extractor(cls: Type[FeatureExtractor]) -> Type[FeatureExtractor]:
    """Decorator to register an extractor class."""
    EXTRACTOR_REGISTRY[cls.name] = cls
    return cls


def get_extractor(name: str) -> Type[FeatureExtractor]:
    """Get an extractor class by name."""
    if name not in EXTRACTOR_REGISTRY:
        raise ValueError(f"Unknown extractor: {name}. Available: {list(EXTRACTOR_REGISTRY.keys())}")
    return EXTRACTOR_REGISTRY[name]


def get_all_extractors() -> Dict[str, Type[FeatureExtractor]]:
    """Get all registered extractors."""
    return EXTRACTOR_REGISTRY.copy()


# Import extractors to trigger registration
from . import tonal
from . import rhythm
from . import lowlevel

__all__ = ['EXTRACTOR_REGISTRY', 'register_extractor', 'get_extractor', 'get_all_extractors']
