"""Database operations for track insertion and updates."""

from typing import Optional
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import execute_values

from ..config import Config
from ..extractors import get_all_extractors


class DatabaseOperations:
    """Handle database insert/update operations."""

    def __init__(self, config: Config):
        self.config = config
        self._conn = None
        self._feature_columns = self._get_feature_columns()

    def _get_feature_columns(self) -> list[str]:
        """Get list of all feature column names."""
        columns = []
        for extractor_cls in get_all_extractors().values():
            extractor = extractor_cls()
            for col in extractor.columns:
                columns.append(col.name)
        return columns

    @contextmanager
    def connection(self):
        """Get a database connection using Unix socket (peer auth)."""
        conn = psycopg2.connect(
            dbname=self.config.db_name,
            user=self.config.db_user,
        )
        try:
            yield conn
        finally:
            conn.close()

    def _is_valid_uuid(self, value) -> bool:
        """Check if a string is a valid UUID (36 chars, correct format)."""
        if not value or not isinstance(value, str) or len(value) == 0:
            return False
        if len(value) != 36:
            return False
        # Basic format check: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        parts = value.split('-')
        if len(parts) != 5:
            return False
        return len(parts[0]) == 8 and len(parts[1]) == 4 and len(parts[2]) == 4 and len(parts[3]) == 4 and len(parts[4]) == 12

    def upsert_track(self, track_data: dict, update_features: bool = False):
        """Insert or update a track and its features.

        Args:
            track_data: Dictionary with track metadata and features
            update_features: If True, update features for existing tracks
        """
        with self.connection() as conn:
            with conn.cursor() as cur:
                # Upsert artist if we have valid artist_mbid
                artist_mbid = track_data.get("artist_mbid")
                # Validate: must be valid UUID, otherwise set to None
                if not self._is_valid_uuid(artist_mbid):
                    artist_mbid = None
                if artist_mbid:
                    cur.execute("""
                        INSERT INTO artists (mbid, name)
                        VALUES (%s, %s)
                        ON CONFLICT (mbid) DO UPDATE SET name = COALESCE(EXCLUDED.name, artists.name)
                    """, (artist_mbid, track_data.get("artist_name")))

                # Check if track already exists by mbid or track_hash
                cur.execute("""
                    SELECT mbid FROM tracks WHERE mbid = %s OR track_hash = %s
                """, (track_data["mbid"], track_data["track_hash"]))
                existing = cur.fetchone()

                if existing:
                    # Track already exists
                    if not update_features:
                        conn.commit()
                        return
                    mbid = existing[0]
                else:
                    # Insert new track
                    cur.execute("""
                        INSERT INTO tracks (mbid, title, album_name, artist_mbid, track_hash, length_seconds, source_archive)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        track_data["mbid"],
                        track_data.get("title"),
                        track_data.get("album_name"),
                        artist_mbid,
                        track_data["track_hash"],
                        track_data.get("length_seconds"),
                        track_data.get("source_archive"),
                    ))
                    mbid = track_data["mbid"]

                # Build feature values
                feature_values = {col: track_data.get(col) for col in self._feature_columns}
                feature_values["mbid"] = mbid

                # Build upsert query for features
                columns = list(feature_values.keys())
                values = [feature_values[c] for c in columns]

                placeholders = ", ".join(["%s"] * len(columns))
                col_names = ", ".join(columns)

                if update_features:
                    # Update on conflict
                    update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c != "mbid")
                    cur.execute(f"""
                        INSERT INTO track_features ({col_names})
                        VALUES ({placeholders})
                        ON CONFLICT (mbid) DO UPDATE SET {update_clause}
                    """, values)
                else:
                    # Ignore on conflict
                    cur.execute(f"""
                        INSERT INTO track_features ({col_names})
                        VALUES ({placeholders})
                        ON CONFLICT (mbid) DO NOTHING
                    """, values)

            conn.commit()

    def get_processing_state(self, archive_name: str) -> Optional[dict]:
        """Get processing state for an archive."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT files_processed, status FROM processing_state
                    WHERE archive_name = %s
                """, (archive_name,))
                result = cur.fetchone()
                if result:
                    return {"files_processed": result[0], "status": result[1]}
                return None

    def update_processing_state(self, archive_name: str, files_processed: int, status: str):
        """Update processing state for an archive."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO processing_state (archive_name, files_processed, status, updated_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (archive_name) DO UPDATE SET
                        files_processed = EXCLUDED.files_processed,
                        status = EXCLUDED.status,
                        updated_at = CURRENT_TIMESTAMP
                """, (archive_name, files_processed, status))
            conn.commit()

    def batch_update_features(self, updates: list[dict], columns: list[str]):
        """Update specific feature columns for existing tracks.

        Args:
            updates: List of dicts with 'mbid' and feature values
            columns: Column names to update (excluding mbid)
        """
        if not updates or not columns:
            return

        with self.connection() as conn:
            with conn.cursor() as cur:
                set_clause = ", ".join(f"{col} = %s" for col in columns)
                sql = f"UPDATE track_features SET {set_clause} WHERE mbid = %s"

                for update in updates:
                    values = [update.get(col) for col in columns]
                    values.append(update["mbid"])
                    cur.execute(sql, values)

            conn.commit()

    def record_extractor_run(self, extractor_name: str, version: int, archive_name: str):
        """Record that an extractor has been run on an archive."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO extractor_runs (extractor_name, version, archive_name)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (extractor_name, archive_name) DO UPDATE SET
                        version = EXCLUDED.version,
                        completed_at = CURRENT_TIMESTAMP
                """, (extractor_name, version, archive_name))
            conn.commit()

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                stats = {}

                # Track counts
                cur.execute("SELECT COUNT(*) FROM tracks")
                stats["total_tracks"] = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM artists")
                stats["total_artists"] = cur.fetchone()[0]

                # Feature coverage
                feature_stats = {}
                for col in self._feature_columns:
                    cur.execute(f"SELECT COUNT(*) FROM track_features WHERE {col} IS NOT NULL")
                    feature_stats[col] = cur.fetchone()[0]
                stats["feature_coverage"] = feature_stats

                # Processing state
                cur.execute("""
                    SELECT archive_name, files_processed, status
                    FROM processing_state ORDER BY archive_name
                """)
                stats["processing_state"] = [
                    {"archive": r[0], "processed": r[1], "status": r[2]}
                    for r in cur.fetchall()
                ]

                # Table sizes
                cur.execute("""
                    SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
                    FROM pg_catalog.pg_statio_user_tables
                    WHERE schemaname = 'public'
                """)
                stats["table_sizes"] = {r[0]: r[1] for r in cur.fetchall()}

                return stats

    def reset_archive_state(self, archive_name: str):
        """Reset processing state for an archive (for reprocessing)."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE processing_state
                    SET files_processed = 0, status = 'pending', updated_at = CURRENT_TIMESTAMP
                    WHERE archive_name = %s
                """, (archive_name,))
            conn.commit()
