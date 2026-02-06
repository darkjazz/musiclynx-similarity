"""Batch processing with progress tracking."""

import sys
import time
from dataclasses import dataclass, field
from typing import Optional

from ..config import Config
from ..base import FeatureExtractor, ExtractionResult
from ..extractors import get_all_extractors
from ..db import DatabaseOperations, SchemaManager
from .archive_reader import StreamingArchiveReader, ArchiveEntry


@dataclass
class ProcessingStats:
    """Statistics for a processing run."""
    archive_name: str
    tracks_processed: int = 0
    tracks_skipped: int = 0
    tracks_failed: int = 0
    bytes_processed: int = 0
    start_time: float = field(default_factory=time.time)
    extraction_errors: dict = field(default_factory=dict)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def tracks_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0
        return self.tracks_processed / self.elapsed_seconds

    @property
    def bytes_per_track(self) -> float:
        if self.tracks_processed == 0:
            return 0
        return self.bytes_processed / self.tracks_processed


class BatchProcessor:
    """Process archives in batches with progress tracking."""

    def __init__(self, config: Config, db_ops: DatabaseOperations):
        self.config = config
        self.db_ops = db_ops
        self.extractors = [cls() for cls in get_all_extractors().values()]

    def process_archive(
        self,
        archive_index: int,
        extractors: Optional[list[FeatureExtractor]] = None,
        rerun: bool = False,
    ) -> ProcessingStats:
        """Process a single archive.

        Args:
            archive_index: Index of archive (0-29)
            extractors: Specific extractors to run (None = all)
            rerun: If True, update existing records

        Returns:
            ProcessingStats with results
        """
        archive_path = self.config.get_archive_path(archive_index)
        archive_name = archive_path.name

        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        # Get resumption point
        skip_count = 0
        if not rerun:
            state = self.db_ops.get_processing_state(archive_name)
            if state and state.get("status") == "completed":
                print(f"Archive {archive_name} already completed, skipping")
                return ProcessingStats(archive_name=archive_name)
            skip_count = state.get("files_processed", 0) if state else 0
            if skip_count > 0:
                print(f"Resuming from file {skip_count}")

        stats = ProcessingStats(archive_name=archive_name)
        stats.tracks_processed = skip_count  # Account for already processed

        reader = StreamingArchiveReader(archive_path)
        extractors_to_use = extractors or self.extractors

        # Always include track_hash extractor (needed for deduplication)
        from ..extractors import get_extractor
        track_hash_extractor = get_extractor('track_hash')()
        if not any(e.name == 'track_hash' for e in extractors_to_use):
            extractors_to_use = list(extractors_to_use) + [track_hash_extractor]

        batch = []
        size_estimated = False

        print(f"Processing {archive_name}...")
        sys.stdout.flush()

        for entry in reader.iter_json_files(skip_count=skip_count):
            # Extract features
            track_data = self._extract_track_data(entry, extractors_to_use, stats)
            if track_data is None:
                stats.tracks_failed += 1
                continue

            batch.append(track_data)
            stats.tracks_processed += 1
            stats.bytes_processed += entry.size_bytes

            # Batch insert
            if len(batch) >= self.config.batch_size:
                self._insert_batch(batch, rerun)
                self.db_ops.update_processing_state(
                    archive_name, stats.tracks_processed, "in_progress"
                )
                batch = []

                # Size estimation after first batch
                if not size_estimated and stats.tracks_processed >= self.config.estimate_after_tracks:
                    self._print_size_estimate(stats)
                    size_estimated = True

                # Progress update
                self._print_progress(stats)

        # Insert remaining
        if batch:
            self._insert_batch(batch, rerun)

        # Mark complete
        self.db_ops.update_processing_state(
            archive_name, stats.tracks_processed, "completed"
        )

        self._print_final_stats(stats)
        return stats

    def rerun_extractors(
        self,
        archive_index: int,
        extractors: list[FeatureExtractor],
    ) -> ProcessingStats:
        """Rerun multiple extractors in a single archive pass.

        Reads the archive once, runs all extractors on each entry, and
        UPDATEs existing rows in track_features matched by mbid.
        """
        archive_path = self.config.get_archive_path(archive_index)
        archive_name = archive_path.name

        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        stats = ProcessingStats(archive_name=archive_name)
        reader = StreamingArchiveReader(archive_path)

        # Collect all columns across all extractors
        all_columns = []
        for ext in extractors:
            all_columns.extend(col.name for col in ext.columns)

        batch = []
        names = ", ".join(e.name for e in extractors)
        print(f"Rerunning [{names}] on {archive_name}...")
        sys.stdout.flush()

        for entry in reader.iter_json_files():
            update = {"mbid": entry.mbid}
            any_success = False

            for extractor in extractors:
                result = extractor.extract(entry.data)
                if result.success:
                    update.update(result.values)
                    any_success = True
                else:
                    if extractor.name not in stats.extraction_errors:
                        stats.extraction_errors[extractor.name] = 0
                    stats.extraction_errors[extractor.name] += 1

            if not any_success:
                stats.tracks_failed += 1
                continue

            batch.append(update)
            stats.tracks_processed += 1

            if len(batch) >= self.config.batch_size:
                self.db_ops.batch_update_features(batch, all_columns)
                batch = []
                self._print_progress(stats)

        if batch:
            self.db_ops.batch_update_features(batch, all_columns)

        self._print_final_stats(stats)
        return stats

    def _extract_track_data(
        self,
        entry: ArchiveEntry,
        extractors: list[FeatureExtractor],
        stats: ProcessingStats,
    ) -> Optional[dict]:
        """Extract all features from a track."""
        # Get metadata
        metadata = entry.data.get("metadata", {})
        tags = metadata.get("tags", {})
        audio_props = metadata.get("audio_properties", {})

        # Extract track info
        track_data = {
            "mbid": entry.mbid,
            "title": self._get_tag(tags, "title"),
            "album_name": self._get_tag(tags, "album"),
            "artist_name": self._get_tag(tags, "artist"),
            "artist_mbid": self._get_tag(tags, "musicbrainz_artistid"),
            "length_seconds": audio_props.get("length"),
            "source_archive": stats.archive_name,
        }

        # Run extractors
        for extractor in extractors:
            result = extractor.extract(entry.data)
            if result.success:
                track_data.update(result.values)
            else:
                # Track extraction errors
                if extractor.name not in stats.extraction_errors:
                    stats.extraction_errors[extractor.name] = 0
                stats.extraction_errors[extractor.name] += 1

        # Must have track_hash for deduplication
        if "track_hash" not in track_data:
            return None

        # Must have valid artist_mbid (skip tracks without artist)
        artist_mbid = track_data.get("artist_mbid")
        if not artist_mbid or not isinstance(artist_mbid, str) or len(artist_mbid) != 36:
            return None

        return track_data

    def _get_tag(self, tags: dict, key: str) -> Optional[str]:
        """Get a tag value, handling list format and multi-value separators."""
        value = tags.get(key)
        if isinstance(value, list) and value:
            value = value[0]

        if not value or not isinstance(value, str):
            return value

        # Handle multiple values separated by various delimiters (take first one)
        # Common separators: /, ;, \, ,
        for sep in ['/', ';', '\\', ',']:
            if sep in value:
                parts = value.split(sep)
                # Only split if first part looks like valid length (e.g., UUID = 36 chars)
                if parts[0].strip():
                    value = parts[0].strip()
                    break

        return value

    def _insert_batch(self, batch: list[dict], update: bool = False):
        """Insert a batch of tracks."""
        for track_data in batch:
            self.db_ops.upsert_track(track_data, update_features=update)

    def _print_progress(self, stats: ProcessingStats):
        """Print progress update."""
        sys.stdout.write(
            f"\r  Processed: {stats.tracks_processed:,} "
            f"({stats.tracks_per_second:.1f}/s) "
            f"Failed: {stats.tracks_failed:,}    "
        )
        sys.stdout.flush()

    def _print_size_estimate(self, stats: ProcessingStats):
        """Print database size estimate."""
        bytes_per_track = stats.bytes_per_track
        total_tracks = self.config.estimated_total_tracks
        estimated_raw_size_gb = (bytes_per_track * total_tracks) / (1024**3)

        # Estimate DB size (rough: ~30% of raw JSON due to binary storage)
        estimated_db_size_gb = estimated_raw_size_gb * 0.3

        print(f"\n\n  Size Estimate (after {stats.tracks_processed:,} tracks):")
        print(f"    Avg JSON size: {bytes_per_track:.0f} bytes/track")
        print(f"    Raw data: ~{estimated_raw_size_gb:.1f} GB for {total_tracks/1e6:.0f}M tracks")
        print(f"    Est. DB size: ~{estimated_db_size_gb:.1f} GB")
        print()

    def _print_final_stats(self, stats: ProcessingStats):
        """Print final processing statistics."""
        print(f"\n\n  Completed {stats.archive_name}:")
        print(f"    Tracks processed: {stats.tracks_processed:,}")
        print(f"    Tracks failed: {stats.tracks_failed:,}")
        print(f"    Time: {stats.elapsed_seconds:.1f}s ({stats.tracks_per_second:.1f}/s)")

        if stats.extraction_errors:
            print("    Extraction errors by extractor:")
            for name, count in sorted(stats.extraction_errors.items()):
                print(f"      {name}: {count:,}")
