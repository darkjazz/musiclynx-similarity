"""Low-level audio feature extractors."""

import hashlib
import json
import math
from ..base import FeatureExtractor, ColumnDefinition, ColumnType, ExtractionResult
from . import register_extractor


def get_mean(d):
    """Get mean value from AcousticBrainz stats dict."""
    return float(d.get("mean")) if isinstance(d, dict) and d.get("mean") is not None else None


def get_std(d):
    """Get std dev (sqrt of variance) from AcousticBrainz stats dict."""
    if not isinstance(d, dict):
        return None
    var = d.get("var")
    if var is None:
        return None
    return math.sqrt(float(var))


@register_extractor
class MFCCExtractor(FeatureExtractor):
    """Extract MFCC (Mel-Frequency Cepstral Coefficients) statistics."""

    name = "mfcc"
    version = 1

    @property
    def columns(self) -> list[ColumnDefinition]:
        return [
            ColumnDefinition("mfcc_mean", ColumnType.REAL_ARRAY, description="Mean MFCC coefficients (13-dim)"),
            ColumnDefinition("mfcc_cov", ColumnType.REAL_ARRAY, description="MFCC covariance matrix (flattened)"),
        ]

    @property
    def json_paths(self) -> list[str]:
        return ["lowlevel.mfcc"]

    def extract(self, json_data: dict) -> ExtractionResult:
        mfcc = self.get_nested(json_data, "lowlevel.mfcc")
        if mfcc is None:
            return ExtractionResult.fail("Missing lowlevel.mfcc")

        mfcc_mean = mfcc.get("mean") if isinstance(mfcc, dict) else None
        mfcc_cov = mfcc.get("cov") if isinstance(mfcc, dict) else None

        if mfcc_mean is None:
            return ExtractionResult.fail("Missing MFCC mean")

        # Flatten covariance matrix if present
        cov_flat = None
        if mfcc_cov is not None and isinstance(mfcc_cov, list):
            cov_flat = [val for row in mfcc_cov for val in row]

        return ExtractionResult.ok({
            "mfcc_mean": mfcc_mean,
            "mfcc_cov": cov_flat,
        })


@register_extractor
class MelbandsExtractor(FeatureExtractor):
    """Extract mel-band energy statistics."""

    name = "melbands"
    version = 1

    @property
    def columns(self) -> list[ColumnDefinition]:
        return [
            ColumnDefinition("melbands_mean", ColumnType.REAL_ARRAY, description="Mean mel-band energies"),
            ColumnDefinition("melbands_std", ColumnType.REAL_ARRAY, description="Std dev of mel-band energies"),
        ]

    @property
    def json_paths(self) -> list[str]:
        return ["lowlevel.melbands"]

    def extract(self, json_data: dict) -> ExtractionResult:
        melbands = self.get_nested(json_data, "lowlevel.melbands")
        if melbands is None:
            return ExtractionResult.fail("Missing lowlevel.melbands")

        mean = melbands.get("mean") if isinstance(melbands, dict) else None
        # Convert variance to std dev for array (element-wise sqrt)
        var = melbands.get("var") if isinstance(melbands, dict) else None
        std = [math.sqrt(v) for v in var] if var else None

        return ExtractionResult.ok({
            "melbands_mean": mean,
            "melbands_std": std,
        })


@register_extractor
class SpectralExtractor(FeatureExtractor):
    """Extract spectral features."""

    name = "spectral"
    version = 1

    @property
    def columns(self) -> list[ColumnDefinition]:
        return [
            ColumnDefinition("spectral_centroid_mean", ColumnType.REAL, description="Mean spectral centroid"),
            ColumnDefinition("spectral_centroid_std", ColumnType.REAL, description="Std dev spectral centroid"),
            ColumnDefinition("spectral_flux_mean", ColumnType.REAL, description="Mean spectral flux"),
            ColumnDefinition("spectral_rolloff_mean", ColumnType.REAL, description="Mean spectral rolloff"),
            ColumnDefinition("spectral_complexity_mean", ColumnType.REAL, description="Mean spectral complexity"),
            ColumnDefinition("spectral_contrast_mean", ColumnType.REAL_ARRAY, description="Mean spectral contrast coefficients"),
        ]

    @property
    def json_paths(self) -> list[str]:
        return [
            "lowlevel.spectral_centroid",
            "lowlevel.spectral_flux",
            "lowlevel.spectral_rolloff",
            "lowlevel.spectral_complexity",
            "lowlevel.spectral_contrast_coeffs",
        ]

    def extract(self, json_data: dict) -> ExtractionResult:
        centroid = self.get_nested(json_data, "lowlevel.spectral_centroid")
        flux = self.get_nested(json_data, "lowlevel.spectral_flux")
        rolloff = self.get_nested(json_data, "lowlevel.spectral_rolloff")
        complexity = self.get_nested(json_data, "lowlevel.spectral_complexity")
        contrast = self.get_nested(json_data, "lowlevel.spectral_contrast_coeffs")

        return ExtractionResult.ok({
            "spectral_centroid_mean": get_mean(centroid),
            "spectral_centroid_std": get_std(centroid),
            "spectral_flux_mean": get_mean(flux),
            "spectral_rolloff_mean": get_mean(rolloff),
            "spectral_complexity_mean": get_mean(complexity),
            "spectral_contrast_mean": contrast.get("mean") if isinstance(contrast, dict) else None,
        })


@register_extractor
class LoudnessExtractor(FeatureExtractor):
    """Extract loudness features."""

    name = "loudness"
    version = 1

    @property
    def columns(self) -> list[ColumnDefinition]:
        return [
            ColumnDefinition("loudness_mean", ColumnType.REAL, description="Mean loudness in dB"),
            ColumnDefinition("loudness_std", ColumnType.REAL, description="Std dev of loudness"),
            ColumnDefinition("dynamic_complexity", ColumnType.REAL, description="Dynamic range complexity"),
        ]

    @property
    def json_paths(self) -> list[str]:
        return ["lowlevel.average_loudness", "lowlevel.dynamic_complexity"]

    def extract(self, json_data: dict) -> ExtractionResult:
        # AcousticBrainz has average_loudness (scalar) not loudness (stats)
        avg_loudness = self.get_nested(json_data, "lowlevel.average_loudness")
        dynamic = self.get_nested(json_data, "lowlevel.dynamic_complexity")

        return ExtractionResult.ok({
            "loudness_mean": float(avg_loudness) if avg_loudness is not None else None,
            "loudness_std": None,  # Not available in AcousticBrainz
            "dynamic_complexity": float(dynamic) if dynamic is not None else None,
        })


@register_extractor
class DissonanceExtractor(FeatureExtractor):
    """Extract dissonance features."""

    name = "dissonance"
    version = 1

    @property
    def columns(self) -> list[ColumnDefinition]:
        return [
            ColumnDefinition("dissonance_mean", ColumnType.REAL, description="Mean dissonance"),
            ColumnDefinition("dissonance_std", ColumnType.REAL, description="Std dev of dissonance"),
        ]

    @property
    def json_paths(self) -> list[str]:
        return ["lowlevel.dissonance"]

    def extract(self, json_data: dict) -> ExtractionResult:
        dissonance = self.get_nested(json_data, "lowlevel.dissonance")

        return ExtractionResult.ok({
            "dissonance_mean": get_mean(dissonance),
            "dissonance_std": get_std(dissonance),
        })


@register_extractor
class TrackHashExtractor(FeatureExtractor):
    """Generate a hash for track deduplication based on audio features."""

    name = "track_hash"
    version = 1

    @property
    def columns(self) -> list[ColumnDefinition]:
        return [
            ColumnDefinition("track_hash", ColumnType.TEXT, nullable=False, description="Hash for deduplication"),
        ]

    @property
    def json_paths(self) -> list[str]:
        return [
            "lowlevel.mfcc.mean",
            "lowlevel.spectral_centroid.mean",
            "metadata.audio_properties.length",
        ]

    def extract(self, json_data: dict) -> ExtractionResult:
        mfcc_mean = self.get_nested(json_data, "lowlevel.mfcc.mean")
        centroid = self.get_nested(json_data, "lowlevel.spectral_centroid.mean")
        length = self.get_nested(json_data, "metadata.audio_properties.length")

        if mfcc_mean is None or centroid is None or length is None:
            return ExtractionResult.fail("Missing data for hash generation")

        # Create hash from rounded feature values for deduplication
        # Round to reduce sensitivity to minor encoding differences
        hash_input = {
            "mfcc": [round(v, 2) for v in mfcc_mean[:5]],  # First 5 coefficients
            "centroid": round(centroid, 0),
            "length": round(length, 1),
        }

        hash_str = hashlib.sha256(json.dumps(hash_input, sort_keys=True).encode()).hexdigest()[:32]

        return ExtractionResult.ok({"track_hash": hash_str})
