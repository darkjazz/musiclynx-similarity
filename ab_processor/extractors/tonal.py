"""Tonal feature extractors."""

from ..base import FeatureExtractor, ColumnDefinition, ColumnType, ExtractionResult
from . import register_extractor


@register_extractor
class THPCPExtractor(FeatureExtractor):
    """Extract Tonal Histogram Profile (THPCP) - 36-dimensional chroma vector."""

    name = "thpcp"
    version = 1

    @property
    def columns(self) -> list[ColumnDefinition]:
        return [
            ColumnDefinition("thpcp", ColumnType.REAL_ARRAY, description="36-dim tonal histogram profile"),
        ]

    @property
    def json_paths(self) -> list[str]:
        return ["tonal.thpcp"]

    def extract(self, json_data: dict) -> ExtractionResult:
        thpcp = self.get_nested(json_data, "tonal.thpcp")
        if thpcp is None:
            return ExtractionResult.fail("Missing tonal.thpcp")
        if not isinstance(thpcp, list) or len(thpcp) != 36:
            return ExtractionResult.fail(f"Invalid THPCP: expected 36 values, got {len(thpcp) if isinstance(thpcp, list) else 'non-list'}")
        return ExtractionResult.ok({"thpcp": thpcp})


@register_extractor
class KeyExtractor(FeatureExtractor):
    """Extract musical key information."""

    name = "key"
    version = 1

    @property
    def columns(self) -> list[ColumnDefinition]:
        return [
            ColumnDefinition("key_key", ColumnType.TEXT, description="Musical key (A-G)"),
            ColumnDefinition("key_scale", ColumnType.TEXT, description="Scale (major/minor)"),
            ColumnDefinition("key_strength", ColumnType.REAL, description="Key detection confidence"),
        ]

    @property
    def json_paths(self) -> list[str]:
        return ["tonal.key_key", "tonal.key_scale", "tonal.key_strength"]

    def extract(self, json_data: dict) -> ExtractionResult:
        key = self.get_nested(json_data, "tonal.key_key")
        scale = self.get_nested(json_data, "tonal.key_scale")
        strength = self.get_nested(json_data, "tonal.key_strength")

        if key is None or scale is None:
            return ExtractionResult.fail("Missing key/scale data")

        return ExtractionResult.ok({
            "key_key": key,
            "key_scale": scale,
            "key_strength": float(strength) if strength is not None else None,
        })


@register_extractor
class ChordsExtractor(FeatureExtractor):
    """Extract chord-related features."""

    name = "chords"
    version = 2

    @property
    def columns(self) -> list[ColumnDefinition]:
        return [
            ColumnDefinition("chords_changes_rate", ColumnType.REAL, description="Rate of chord changes per second"),
            ColumnDefinition("chords_number_rate", ColumnType.REAL, description="Number of chord types used"),
            ColumnDefinition("chords_key", ColumnType.TEXT, description="Key detected from chords"),
            ColumnDefinition("chords_scale", ColumnType.TEXT, description="Scale detected from chords"),
            ColumnDefinition("chords_histogram", ColumnType.REAL_ARRAY, description="Chord type histogram (24-dim)"),
        ]

    @property
    def json_paths(self) -> list[str]:
        return [
            "tonal.chords_changes_rate",
            "tonal.chords_number_rate",
            "tonal.chords_key",
            "tonal.chords_scale",
            "tonal.chords_histogram",
        ]

    def extract(self, json_data: dict) -> ExtractionResult:
        changes_rate = self.get_nested(json_data, "tonal.chords_changes_rate")
        number_rate = self.get_nested(json_data, "tonal.chords_number_rate")
        key = self.get_nested(json_data, "tonal.chords_key")
        scale = self.get_nested(json_data, "tonal.chords_scale")
        histogram = self.get_nested(json_data, "tonal.chords_histogram")

        return ExtractionResult.ok({
            "chords_changes_rate": float(changes_rate) if changes_rate is not None else None,
            "chords_number_rate": float(number_rate) if number_rate is not None else None,
            "chords_key": key,
            "chords_scale": scale,
            "chords_histogram": histogram if isinstance(histogram, list) else None,
        })


@register_extractor
class TuningExtractor(FeatureExtractor):
    """Extract tuning information."""

    name = "tuning"
    version = 1

    @property
    def columns(self) -> list[ColumnDefinition]:
        return [
            ColumnDefinition("tuning_frequency", ColumnType.REAL, description="Tuning frequency in Hz"),
            ColumnDefinition("tuning_equal_tempered_deviation", ColumnType.REAL, description="Deviation from equal temperament"),
        ]

    @property
    def json_paths(self) -> list[str]:
        return ["tonal.tuning_frequency", "tonal.tuning_equal_tempered_deviation"]

    def extract(self, json_data: dict) -> ExtractionResult:
        freq = self.get_nested(json_data, "tonal.tuning_frequency")
        deviation = self.get_nested(json_data, "tonal.tuning_equal_tempered_deviation")

        return ExtractionResult.ok({
            "tuning_frequency": float(freq) if freq is not None else None,
            "tuning_equal_tempered_deviation": float(deviation) if deviation is not None else None,
        })
