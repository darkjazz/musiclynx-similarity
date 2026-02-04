"""Base classes for feature extraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ColumnType(Enum):
    """PostgreSQL column types."""
    INTEGER = "INTEGER"
    REAL = "REAL"
    TEXT = "TEXT"
    BOOLEAN = "BOOLEAN"
    REAL_ARRAY = "REAL[]"
    UUID = "UUID"


@dataclass
class ColumnDefinition:
    """Definition of a database column."""
    name: str
    col_type: ColumnType
    nullable: bool = True
    description: str = ""

    def to_sql(self) -> str:
        """Generate SQL column definition."""
        null_clause = "" if self.nullable else " NOT NULL"
        return f"{self.name} {self.col_type.value}{null_clause}"


@dataclass
class ExtractionResult:
    """Result of feature extraction."""
    success: bool
    values: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @classmethod
    def ok(cls, values: dict[str, Any]) -> "ExtractionResult":
        """Create successful result."""
        return cls(success=True, values=values)

    @classmethod
    def fail(cls, error: str) -> "ExtractionResult":
        """Create failed result."""
        return cls(success=False, error=error)


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this extractor."""
        pass

    @property
    @abstractmethod
    def version(self) -> int:
        """Version number. Increment when extraction logic changes."""
        pass

    @property
    @abstractmethod
    def columns(self) -> list[ColumnDefinition]:
        """Database columns this extractor produces."""
        pass

    @property
    @abstractmethod
    def json_paths(self) -> list[str]:
        """Dot-notation paths in the JSON this extractor needs."""
        pass

    @abstractmethod
    def extract(self, json_data: dict) -> ExtractionResult:
        """Extract features from JSON data.

        Args:
            json_data: Full JSON document from AcousticBrainz

        Returns:
            ExtractionResult with column values or error
        """
        pass

    def get_nested(self, data: dict, path: str, default: Any = None) -> Any:
        """Get a value from nested dict using dot notation.

        Args:
            data: Dictionary to traverse
            path: Dot-separated path (e.g., "tonal.key_key")
            default: Value to return if path not found

        Returns:
            Value at path or default
        """
        keys = path.split(".")
        current = data
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current
