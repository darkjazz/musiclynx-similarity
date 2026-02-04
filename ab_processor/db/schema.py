"""Database schema management."""

import psycopg2
from psycopg2 import sql

from ..config import Config
from ..extractors import get_all_extractors


class SchemaManager:
    """Manage database schema creation and updates."""

    def __init__(self, config: Config):
        self.config = config

    def get_connection(self):
        """Get a database connection using Unix socket (peer auth)."""
        return psycopg2.connect(
            dbname=self.config.db_name,
            user=self.config.db_user,
        )

    def init_schema(self):
        """Create all database tables and indices."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Enable UUID extension
                cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

                # Artists table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS artists (
                        mbid UUID PRIMARY KEY,
                        name TEXT
                    )
                """)

                # Tracks table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS tracks (
                        mbid UUID PRIMARY KEY,
                        title TEXT,
                        album_name TEXT,
                        artist_mbid UUID REFERENCES artists(mbid),
                        track_hash TEXT UNIQUE NOT NULL,
                        length_seconds REAL,
                        source_archive TEXT
                    )
                """)

                # Track features table - start with base columns
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS track_features (
                        mbid UUID PRIMARY KEY REFERENCES tracks(mbid) ON DELETE CASCADE
                    )
                """)

                # Add columns from all extractors
                self._add_extractor_columns(cur)

                # Processing state table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS processing_state (
                        archive_name TEXT PRIMARY KEY,
                        files_processed INTEGER NOT NULL DEFAULT 0,
                        status TEXT NOT NULL DEFAULT 'pending',
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Extractor runs table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS extractor_runs (
                        id SERIAL PRIMARY KEY,
                        extractor_name TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        archive_name TEXT NOT NULL,
                        completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(extractor_name, archive_name)
                    )
                """)

                # Create indices
                self._create_indices(cur)

            conn.commit()
            print("Schema initialized successfully")

    def _add_extractor_columns(self, cursor):
        """Add columns for all registered extractors."""
        extractors = get_all_extractors()

        for extractor_cls in extractors.values():
            extractor = extractor_cls()
            for col in extractor.columns:
                # Check if column exists
                cursor.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'track_features' AND column_name = %s
                """, (col.name,))

                if cursor.fetchone() is None:
                    # Add column
                    cursor.execute(sql.SQL(
                        "ALTER TABLE track_features ADD COLUMN {} {}"
                    ).format(
                        sql.Identifier(col.name),
                        sql.SQL(col.col_type.value)
                    ))
                    print(f"  Added column: track_features.{col.name}")

    def _create_indices(self, cursor):
        """Create performance indices."""
        indices = [
            ("idx_tracks_track_hash", "tracks", "track_hash"),
            ("idx_tracks_artist_mbid", "tracks", "artist_mbid"),
            ("idx_track_features_bpm", "track_features", "bpm"),
            ("idx_track_features_danceability", "track_features", "danceability"),
        ]

        # Composite index for key lookup
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_track_features_key
            ON track_features (key_key, key_scale)
        """)

        for idx_name, table, column in indices:
            cursor.execute(sql.SQL(
                "CREATE INDEX IF NOT EXISTS {} ON {} ({})"
            ).format(
                sql.Identifier(idx_name),
                sql.Identifier(table),
                sql.Identifier(column)
            ))

    def add_column(self, column_name: str, column_type: str):
        """Add a new column to track_features table."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL(
                    "ALTER TABLE track_features ADD COLUMN IF NOT EXISTS {} {}"
                ).format(
                    sql.Identifier(column_name),
                    sql.SQL(column_type)
                ))
            conn.commit()
