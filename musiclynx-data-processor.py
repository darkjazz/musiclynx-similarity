from datetime import date
import json
import tarfile
import zstandard as zstd
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import psycopg2
from psycopg2.extras import execute_batch, Json
from psycopg2.pool import ThreadedConnectionPool
from typing import Dict, List, Optional, Any
import hashlib
import pickle
import gc
from collections import defaultdict
import multiprocessing as mp
from functools import partial
from contextlib import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_nested_value(obj, path):
    """Get value from nested object using dot notation path."""
    try:
        keys = path.split('.')
        current = obj
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    except (KeyError, TypeError):
        return None


def set_nested_value(obj, path, value):
    """Set value in nested object using dot notation path."""
    keys = path.split('.')
    current = obj

    # Navigate/create structure up to the last key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # Set the final value
    current[keys[-1]] = value

def extract_features(json_data, feature_list):
    """
    Extract features from JSON data using dot notation while maintaining original structure.

    Args:
        json_data (dict): The source JSON data
        feature_list (list): List of dot notation paths to extract

    Returns:
        dict: New object with extracted features maintaining nested structure
    """
    result = {}

    # Extract each feature
    for feature in feature_list:
        value = get_nested_value(json_data, feature)
        if value is not None:
            set_nested_value(result, feature, value)

    return result


def extract_features_with_defaults(json_data, feature_list, default_value=None):
    """
    Extract features with default values for missing keys.

    Args:
        json_data (dict): The source JSON data
        feature_list (list): List of dot notation paths to extract
        default_value: Value to use for missing features

    Returns:
        dict: New object with extracted features maintaining nested structure
    """
    result = {}

    # Extract each feature, using default for missing values
    for feature in feature_list:
        value = get_nested_value(json_data, feature)
        set_nested_value(result, feature, value if value is not None else default_value)

    return result

class AcousticBrainzProcessor:
    """Process AcousticBrainz dump files and extract features for similarity analysis"""

    def __init__(self,
                 db_config: Dict[str, str],
                 output_dir: str = "processed_data",
                 chunk_size: int = 1000,
                 max_connections: int = 10):
        """
        Initialize processor with PostgreSQL configuration

        db_config example:
        {
            'host': 'localhost',
            'database': 'acousticbrainz',
            'user': 'username',
            'password': 'password',
            'port': 5432
        }
        """
        self.db_config = db_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.chunk_size = chunk_size

        # Connection pool for concurrent processing
        self.connection_pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=max_connections,
            **db_config
        )

        # Initialize database schema
        self._init_database()

    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections"""
        conn = self.connection_pool.getconn()
        try:
            yield conn
        finally:
            self.connection_pool.putconn(conn)

    def _init_database(self):
        """Initialize PostgreSQL database schema"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()

            # Enable UUID extension
            cursor.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')

            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tracks (
                    mbid UUID PRIMARY KEY,
                    title TEXT,
                    album_name TEXT,
                    disc_number TEXT,
                    track_number TEXT,
                    date TEXT,
                    country TEXT,
                    label TEXT,
                    artist_mbid UUID,
                    track_hash VARCHAR(32) UNIQUE NOT NULL,
                    length REAL,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS features (
                    mbid UUID PRIMARY KEY,
                    features JSONB,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS artists (
                    mbid UUID PRIMARY KEY,
                    name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            ''')

            # Create indices for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_track_hash ON tracks (track_hash);')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_artist_mbid ON tracks (artist_mbid);')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed ON tracks (processed);')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metadata_gin ON features USING GIN (metadata);')

            # Create view for easy querying
            cursor.execute('''
                CREATE OR REPLACE VIEW tracks_with_features AS
                SELECT
                    t.mbid,
                    t.artist_mbid,
                    a.name,
                    t.length,
                    f.features,
                    f.metadata
                FROM tracks t
                JOIN features f ON t.mbid = f.mbid
                JOIN artists a on a.mbid = t.artist_mbid
                WHERE t.artist_mbid IS NOT NULL;
            ''')

            conn.commit()
            logger.info("Database schema initialized")

    def decompress_zst_file(self, zst_file_path: str, output_path: str = None) -> str:
        """Decompress a .zst file"""
        if output_path is None:
            output_path = str(zst_file_path).replace('.zst', '')

        if os.path.exists(output_path):
            logger.info(f"Decompressed file already exists: {output_path}")
            return output_path

        logger.info(f"Decompressing {zst_file_path}...")

        with open(zst_file_path, 'rb') as compressed:
            dctx = zstd.ZstdDecompressor()
            with open(output_path, 'wb') as destination:
                dctx.copy_stream(compressed, destination)

        logger.info(f"Decompressed to: {output_path}")
        return output_path

    def extract_tar_file(self, tar_file_path: str, extract_to: str = None) -> str:
        """Extract tar file"""
        if extract_to is None:
            extract_to = str(tar_file_path).replace('.tar', '_extracted')

        extract_path = Path(extract_to)
        if extract_path.exists():
            logger.info(f"Extracted directory already exists: {extract_to}")
            return extract_to

        logger.info(f"Extracting {tar_file_path}...")
        extract_path.mkdir(exist_ok=True)

        with tarfile.open(tar_file_path, 'r') as tar:
            tar.extractall(path=extract_to)

        logger.info(f"Extracted to: {extract_to}")
        return extract_to

    def compute_track_hash(self, features_dict: Dict) -> str:
        """Compute hash of track features for deduplication"""
        hash_components = []

        if 'lowlevel' in features_dict:
            lowlevel = features_dict['lowlevel']

            # MFCC means (most stable identifier)
            if 'mfcc' in lowlevel and 'mean' in lowlevel['mfcc']:
                hash_components.extend(lowlevel['mfcc']['mean'])

            # Spectral centroid mean
            if 'spectral_centroid' in lowlevel and 'mean' in lowlevel['spectral_centroid']:
                hash_components.append(lowlevel['spectral_centroid']['mean'])

            # Track length
            if 'length' in features_dict:
                hash_components.append(features_dict['length'])

        # Create hash
        hash_string = ''.join(f"{x:.6f}" for x in hash_components)
        return hashlib.md5(hash_string.encode()).hexdigest()

    def safe_uuid(self, uuid_string: str) -> Optional[str]:
        """Safely convert string to UUID, return None if invalid"""
        if not uuid_string or len(uuid_string) != 36:
            return None
        try:
            # Validate UUID format
            import uuid
            uuid.UUID(uuid_string)
            return uuid_string
        except (ValueError, AttributeError):
            return None

    def extract_features_from_json(self, json_data: Dict) -> Optional[Dict]:
        """Extract relevant features from AcousticBrainz JSON"""
        try:
            feature_paths = [
                "audio_properties.length",
                "audio_properties.md5_encoded",
                "tonal.thpcp",
                "tonal.key_key",
                "tonal.key_scale",
                "tonal.key_strength",
                "tonal.chords_strength.mean",
                "tonal.chords_histogram",
                "tonal.chords_changes_rate",
                "rhythm.bpm",
                "rhythm.onset_rate",
                "rhythm.beats_count",
                "rhythm.danceability",
                "rhythm.beats_loudness.mean",
                "rhythm.beats_loudness.var",
                "rhythm.beats_position",
                "rhythm.beats_loudness_band_ratio.mean",
                "lowlevel.mfcc.cov",
                "lowlevel.mfcc.mean",
                "lowlevel.melbands.mean",
                "lowlevel.melbands.var",
                "lowlevel.dissonance.mean",
                "lowlevel.dissonance.var",
                "lowlevel.spectral_energy.mean",
                "lowlevel.spectral_energy.var",
                "lowlevel.spectral_spread.mean",
                "lowlevel.spectral_spread.var",
                "lowlevel.average_loudness",
                "lowlevel.spectral_rolloff.mean",
                "lowlevel.zerocrossingrate.mean",
                "lowlevel.spectral_centroid.mean",
                "lowlevel.dynamic_complexity",
                "lowlevel.spectral_contrast_coeffs.mean",
                "lowlevel.spectral_contrast_coeffs.var",
                "lowlevel.spectral_contrast_valleys.mean",
                "lowlevel.spectral_contrast_valleys.var",
            ]
            features = {}

            metadata_tags = json_data.get('metadata', {}).get('tags', {})

            mbid = metadata_tags.get("musicbrainz_recordingid")[0] if metadata_tags.get("musicbrainz_recordingid") else None
            if not mbid:
                return

            track_hash = self.compute_track_hash(json_data)

            features = extract_features(json_data, feature_paths)

            doc = {
                'mbid': mbid,
                'metadata': metadata_tags,
                'track_hash': track_hash,
                'features': features
            }

            return doc

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None

    def batch_insert_tracks(self, tracks_batch: List[Dict]):
        """Batch insert tracks into PostgreSQL"""
        if not tracks_batch:
            return

        with self.get_db_connection() as conn:
            cursor = conn.cursor()

            # Prepare data for batch insert
            track_data = []
            feature_data = []
            artist_data = []

            for track in tracks_batch:
                metadata = track['metadata']
                features = track['features']

                # Track data
                track_data.append((
                    track['mbid'],
                    metadata['title'][0],
                    metadata['album'][0],
                    metadata['discnumber'][0] if 'discnumber' in metadata else None,
                    metadata['tracknumber'][0] if 'tracknumber' in metadata else None,
                    metadata['date'][0] if 'date' in metadata else None,
                    metadata['country'][0] if 'country' in metadata else None,
                    metadata['label'][0] if 'label' in metadata else None,
                    metadata['musicbrainz_artistid'][0],
                    track['track_hash'],
                    features['audio_properties']['length'],
                    True  # processed
                ))

                # Feature data
                feature_data.append((
                    metadata['mbid'],
                    Json(features),
                    Json(metadata)
                ))

                artist_data.append((
                    metadata["musicbrainz_artistid"][0],
                    metadata["artist"][0]
                ))

            try:
                # Insert tracks
                execute_batch(
                    cursor,
                    '''INSERT INTO tracks
                       (mbid, title, album, discnumber, tracknumber, date, country, label, artist_mbid, track_hash, length, processed)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (track_hash) DO NOTHING''',
                    track_data,
                    page_size=100
                )

                # Insert features
                execute_batch(
                    cursor,
                    '''INSERT INTO features
                       (mbid, features, metadata)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (mbid) DO NOTHING''',
                    feature_data,
                    page_size=100
                )

                execute_batch(
                    cursor,
                    '''INSERT INTO artists
                       (mbid, name)
                       VALUES (%s, %s)
                       ON CONFLICT (mbid) DO NOTHING''',
                    feature_data,
                    page_size=100
                )

                conn.commit()

            except psycopg2.Error as e:
                conn.rollback()
                logger.error(f"Database error during batch insert: {e}")
                raise

    def process_json_file(self, json_file_path: str) -> int:
        """Process a single JSON file and extract features"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            features = self.extract_features_from_json(json_data)
            if features is None:
                return 0

            # Check for duplicates
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT mbid FROM tracks WHERE track_hash = %s', (features['track_hash'],))
                existing = cursor.fetchone()

                if existing:
                    logger.debug(f"Duplicate track found: {features['metadata']['mbid']}")
                    return 0

            # Insert single track
            self.batch_insert_tracks([features])
            return 1

        except Exception as e:
            logger.error(f"Error processing {json_file_path}: {e}")
            return 0

    def process_directory_batch(self, json_files: List[str]) -> int:
        """Process a batch of JSON files efficiently"""
        processed_count = 0
        batch = []

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                features = self.extract_features_from_json(json_data)
                if features is None:
                    continue

                # Check for duplicates (simplified check)
                with self.get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT 1 FROM tracks WHERE track_hash = %s', (features['track_hash'],))
                    if cursor.fetchone():
                        continue

                batch.append(features)

                # Process batch when it reaches chunk size
                if len(batch) >= self.chunk_size:
                    self.batch_insert_tracks(batch)
                    processed_count += len(batch)
                    batch = []

            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
                continue

        # Process remaining batch
        if batch:
            self.batch_insert_tracks(batch)
            processed_count += len(batch)

        return processed_count

    def process_directory(self, directory_path: str) -> int:
        """Process all JSON files in a directory"""
        directory = Path(directory_path)
        json_files = list(directory.rglob("*.json"))

        logger.info(f"Found {len(json_files)} JSON files to process")

        # Process in batches for efficiency
        total_processed = 0
        batch_size = self.chunk_size * 10  # Larger batches for file processing

        with tqdm(total=len(json_files), desc="Processing JSON files") as pbar:
            for i in range(0, len(json_files), batch_size):
                batch_files = json_files[i:i + batch_size]
                processed = self.process_directory_batch(batch_files)
                total_processed += processed
                pbar.update(len(batch_files))
                pbar.set_postfix({'processed': total_processed})

        return total_processed

    def export_features_to_csv(self, output_file: str = None, min_tracks_per_artist: int = 2) -> str:
        """Export processed features to CSV format"""
        if output_file is None:
            output_file = self.output_dir / "acousticbrainz_features.csv"

        logger.info("Exporting features to CSV...")

        with self.get_db_connection() as conn:
            # Get artist track counts first
            cursor = conn.cursor()
            cursor.execute('''
                SELECT artist_mbid, COUNT(*) as track_count
                FROM tracks_with_features
                WHERE artist_mbid IS NOT NULL
                GROUP BY artist_mbid
                HAVING COUNT(*) >= %s
            ''', (min_tracks_per_artist,))

            valid_artists = {row[0] for row in cursor.fetchall()}
            logger.info(f"Found {len(valid_artists)} artists with >= {min_tracks_per_artist} tracks")

            # Export data in chunks to avoid memory issues
            cursor.execute('''
                SELECT
                    mbid,
                    artist_mbid,
                    length,
                    timbre_features,
                    tonality_features,
                    tempo_features,
                    metadata->>'artist_name' as artist_name,
                    metadata->>'title' as title
                FROM tracks_with_features
                WHERE artist_mbid = ANY(%s)
                ORDER BY artist_mbid, mbid
            ''', (list(valid_artists),))

            # Process results in chunks
            rows = []
            while True:
                chunk = cursor.fetchmany(self.chunk_size)
                if not chunk:
                    break

                for row in chunk:
                    mbid, artist_mbid, length, timbre_features, tonality_features, tempo_features, artist_name, title = row

                    # Prepare row data
                    row_data = {
                        'track_id': str(mbid),
                        'artist_id': str(artist_mbid),
                        'artist_name': artist_name or '',
                        'title': title or '',
                        'length': length or 0.0
                    }

                    # Add timbre features
                    for i, val in enumerate(timbre_features or []):
                        row_data[f'timbre_{i}'] = float(val)

                    # Add tonality features
                    for i, val in enumerate(tonality_features or []):
                        row_data[f'tonality_{i}'] = float(val)

                    # Add tempo features
                    for i, val in enumerate(tempo_features or []):
                        row_data[f'tempo_{i}'] = float(val)

                    rows.append(row_data)

        # Convert to DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)

        logger.info(f"Features exported to: {output_file}")
        logger.info(f"Dataset shape: {df.shape}")

        # Print summary statistics
        logger.info(f"Artists: {df['artist_id'].nunique()}")
        logger.info(f"Tracks: {len(df)}")
        logger.info(f"Timbre features: {len([col for col in df.columns if col.startswith('timbre_')])}")
        logger.info(f"Tonality features: {len([col for col in df.columns if col.startswith('tonality_')])}")
        logger.info(f"Tempo features: {len([col for col in df.columns if col.startswith('tempo_')])}")

        return str(output_file)

    def should_process_dir(self, dirpath, filenames):
        return any(f.endswith('.json') for f in filenames)

    def process_acousticbrainz_dump(self, zst_file_path: str) -> str:
        """Complete pipeline to process AcousticBrainz dump file"""
        logger.info(f"Processing AcousticBrainz dump: {zst_file_path}")

        # Step 1: Decompress .zst file
        tar_file_path = self.decompress_zst_file(zst_file_path)

        # Step 2: Extract tar file
        extracted_dir = self.extract_tar_file(tar_file_path)

        # Step 3: Process JSON files
        processed_count = 0
        for dirpath, dirnames, filenames in os.walk(extracted_dir):
            if self.should_process_dir(dirpath, filenames):
                processed_count += self.process_directory(extracted_dir)
        logger.info(f"Processed {processed_count} unique tracks")

        # Cleanup extracted files to save space
        try:
            import shutil
            shutil.rmtree(extracted_dir)
            os.remove(tar_file_path)
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Could not clean up temporary files: {e}")

    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Total tracks
            cursor.execute('SELECT COUNT(*) FROM tracks')
            stats['total_tracks'] = cursor.fetchone()[0]

            # Unique artists
            cursor.execute('SELECT COUNT(DISTINCT artist_mbid) FROM tracks WHERE artist_mbid IS NOT NULL')
            stats['unique_artists'] = cursor.fetchone()[0]

            # Tracks with features
            cursor.execute('SELECT COUNT(*) FROM features')
            stats['tracks_with_features'] = cursor.fetchone()[0]

            # Feature dimensions
            cursor.execute('''
                SELECT
                    AVG(array_length(timbre_features, 1)) as avg_timbre_dims,
                    AVG(array_length(tonality_features, 1)) as avg_tonality_dims,
                    AVG(array_length(tempo_features, 1)) as avg_tempo_dims
                FROM features
                WHERE timbre_features IS NOT NULL
            ''')
            dims = cursor.fetchone()
            if dims:
                stats['avg_feature_dimensions'] = {
                    'timbre': dims[0],
                    'tonality': dims[1],
                    'tempo': dims[2]
                }

            # Top artists by track count
            cursor.execute('''
                SELECT
                    t.artist_mbid,
                    f.metadata->>'artist_name' as artist_name,
                    COUNT(*) as track_count
                FROM tracks t
                JOIN features f ON t.mbid = f.mbid
                WHERE t.artist_mbid IS NOT NULL
                GROUP BY t.artist_mbid, f.metadata->>'artist_name'
                ORDER BY track_count DESC
                LIMIT 10
            ''')
            stats['top_artists'] = cursor.fetchall()

            return stats

    def close(self):
        """Close connection pool"""
        if hasattr(self, 'connection_pool'):
            self.connection_pool.closeall()

# Command line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process AcousticBrainz dump files with PostgreSQL")
    parser.add_argument("zst_file", help="Path to the .zst dump file")
    parser.add_argument("--host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--database", default="acousticbrainz", help="PostgreSQL database name")
    parser.add_argument("--user", required=True, help="PostgreSQL username")
    parser.add_argument("--password", help="PostgreSQL password (or set PGPASSWORD env var)")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--output-dir", default="processed_data", help="Output directory")
    parser.add_argument("--min-tracks", type=int, default=2, help="Minimum tracks per artist to include")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Batch size for database operations")

    args = parser.parse_args()

    # Get password from environment if not provided
    password = args.password or os.environ.get('PGPASSWORD')
    if not password:
        import getpass
        password = getpass.getpass("PostgreSQL password: ")

    # Database configuration
    db_config = {
        'host': args.host,
        'database': args.database,
        'user': args.user,
        'password': password,
        'port': args.port
    }

    # Process the dump file
    processor = AcousticBrainzProcessor(
        db_config=db_config,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size
    )

    try:
        processor.process_acousticbrainz_dump(
            args.zst_file
        )

        # Print statistics
        stats = processor.get_statistics()
        print("\n=== Processing Statistics ===")
        for key, value in stats.items():
            if key not in ['top_artists', 'avg_feature_dimensions']:
                print(f"{key}: {value}")
            elif key == 'avg_feature_dimensions':
                print(f"Average feature dimensions: {value}")

        if 'top_artists' in stats:
            print("\nTop 10 artists by track count:")
            for artist_mbid, artist_name, track_count in stats['top_artists'][:10]:
                print(f"  {artist_name or 'Unknown'}: {track_count} tracks")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise
    finally:
        processor.close()
