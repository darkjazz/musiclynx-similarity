"""Rhythm feature extractors."""

from ..base import FeatureExtractor, ColumnDefinition, ColumnType, ExtractionResult
from . import register_extractor


@register_extractor
class BPMExtractor(FeatureExtractor):
    """Extract tempo (BPM) information."""

    name = "bpm"
    version = 1

    @property
    def columns(self) -> list[ColumnDefinition]:
        return [
            ColumnDefinition("bpm", ColumnType.REAL, description="Beats per minute"),
            ColumnDefinition("bpm_confidence", ColumnType.REAL, description="BPM detection confidence"),
        ]

    @property
    def json_paths(self) -> list[str]:
        return ["rhythm.bpm", "rhythm.bpm_histogram_first_peak_weight"]

    def extract(self, json_data: dict) -> ExtractionResult:
        bpm = self.get_nested(json_data, "rhythm.bpm")
        # Use first peak weight as confidence
        confidence = self.get_nested(json_data, "rhythm.bpm_histogram_first_peak_weight.mean")

        if bpm is None:
            return ExtractionResult.fail("Missing rhythm.bpm")

        return ExtractionResult.ok({
            "bpm": float(bpm),
            "bpm_confidence": float(confidence) if confidence is not None else None,
        })


@register_extractor
class DanceabilityExtractor(FeatureExtractor):
    """Extract danceability score."""

    name = "danceability"
    version = 1

    @property
    def columns(self) -> list[ColumnDefinition]:
        return [
            ColumnDefinition("danceability", ColumnType.REAL, description="Danceability score 0-3"),
        ]

    @property
    def json_paths(self) -> list[str]:
        return ["rhythm.danceability"]

    def extract(self, json_data: dict) -> ExtractionResult:
        danceability = self.get_nested(json_data, "rhythm.danceability")

        if danceability is None:
            return ExtractionResult.fail("Missing rhythm.danceability")

        return ExtractionResult.ok({
            "danceability": float(danceability),
        })


@register_extractor
class BeatsExtractor(FeatureExtractor):
    """Extract beat-related features."""

    name = "beats"
    version = 1

    @property
    def columns(self) -> list[ColumnDefinition]:
        return [
            ColumnDefinition("beats_count", ColumnType.INTEGER, description="Number of detected beats"),
            ColumnDefinition("beats_loudness_mean", ColumnType.REAL, description="Mean loudness at beats"),
            ColumnDefinition("beats_loudness_std", ColumnType.REAL, description="Std dev of loudness at beats"),
            ColumnDefinition("onset_rate", ColumnType.REAL, description="Rate of note onsets per second"),
        ]

    @property
    def json_paths(self) -> list[str]:
        return [
            "rhythm.beats_count",
            "rhythm.beats_loudness",
            "rhythm.onset_rate",
        ]

    def extract(self, json_data: dict) -> ExtractionResult:
        beats_count = self.get_nested(json_data, "rhythm.beats_count")
        beats_loudness = self.get_nested(json_data, "rhythm.beats_loudness")
        onset_rate = self.get_nested(json_data, "rhythm.onset_rate")

        # Extract mean/std from beats_loudness if available
        # Note: AcousticBrainz uses "var" (variance) not "stdev"
        loudness_mean = None
        loudness_var = None
        if isinstance(beats_loudness, dict):
            loudness_mean = beats_loudness.get("mean")
            loudness_var = beats_loudness.get("var")

        # Convert variance to std dev (sqrt)
        import math
        loudness_std = math.sqrt(loudness_var) if loudness_var is not None else None

        return ExtractionResult.ok({
            "beats_count": int(beats_count) if beats_count is not None else None,
            "beats_loudness_mean": float(loudness_mean) if loudness_mean is not None else None,
            "beats_loudness_std": float(loudness_std) if loudness_std is not None else None,
            "onset_rate": float(onset_rate) if onset_rate is not None else None,
        })
