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
                    artist_mbid UUID,
                    release_mbid UUID,
                    recording_mbid UUID,
                    track_hash VARCHAR(32) UNIQUE NOT NULL,
                    length REAL,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS features (
                    mbid UUID PRIMARY KEY REFERENCES tracks(mbid) ON DELETE CASCADE,
                    timbre_features REAL[],
                    tonality_features REAL[],
                    tempo_features REAL[],
                    metadata JSONB,
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
                    t.release_mbid,
                    t.recording_mbid,
                    t.length,
                    f.timbre_features,
                    f.tonality_features,
                    f.tempo_features,
                    f.metadata
                FROM tracks t
                JOIN features f ON t.mbid = f.mbid
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
            features = {}

            # Basic metadata - safely extract UUIDs
            mbid = self.safe_uuid(json_data.get('mbid', ''))
            if not mbid:
                return None  # Skip records without valid MBID

            metadata_tags = json_data.get('metadata', {}).get('tags', {})

            metadata = {
                'mbid': mbid,
                'length': float(json_data.get('length', 0)),
                'artist_mbid': self.safe_uuid(metadata_tags.get('musicbrainz_artistid', [''])[0] if metadata_tags.get('musicbrainz_artistid') else ''),
                'release_mbid': self.safe_uuid(metadata_tags.get('musicbrainz_albumid', [''])[0] if metadata_tags.get('musicbrainz_albumid') else ''),
                'recording_mbid': self.safe_uuid(metadata_tags.get('musicbrainz_trackid', [''])[0] if metadata_tags.get('musicbrainz_trackid') else ''),
                'artist_name': metadata_tags.get('artist', [''])[0] if metadata_tags.get('artist') else '',
                'title': metadata_tags.get('title', [''])[0] if metadata_tags.get('title') else '',
                'album': metadata_tags.get('album', [''])[0] if metadata_tags.get('album') else ''
            }

            lowlevel = json_data.get('lowlevel', {})
            tonal = json_data.get('tonal', {})
            rhythm = json_data.get('rhythm', {})

            # TIMBRE FEATURES (MFCC + Spectral)
            timbre_features = []

            # MFCC coefficients (mean and var)
            if 'mfcc' in lowlevel:
                if 'mean' in lowlevel['mfcc']:
                    timbre_features.extend(lowlevel['mfcc']['mean'])
                if 'var' in lowlevel['mfcc']:
                    timbre_features.extend(lowlevel['mfcc']['var'])

            # Spectral features
            spectral_features = [
                'spectral_centroid', 'spectral_bandwidth', 'spectral_contrast',
                'spectral_rolloff', 'spectral_decrease', 'spectral_energy',
                'spectral_energyband_low', 'spectral_energyband_middle_low',
                'spectral_energyband_middle_high', 'spectral_energyband_high',
                'spectral_flatness_db', 'spectral_flux', 'spectral_kurtosis',
                'spectral_skewness', 'spectral_spread', 'spectral_strongpeak'
            ]

            for feature in spectral_features:
                if feature in lowlevel:
                    if 'mean' in lowlevel[feature]:
                        if isinstance(lowlevel[feature]['mean'], list):
                            timbre_features.extend(lowlevel[feature]['mean'])
                        else:
                            timbre_features.append(lowlevel[feature]['mean'])
                    if 'var' in lowlevel[feature]:
                        if isinstance(lowlevel[feature]['var'], list):
                            timbre_features.extend(lowlevel[feature]['var'])
                        else:
                            timbre_features.append(lowlevel[feature]['var'])

            # Zero crossing rate
            if 'zerocrossingrate' in lowlevel:
                if 'mean' in lowlevel['zerocrossingrate']:
                    timbre_features.append(lowlevel['zerocrossingrate']['mean'])
                if 'var' in lowlevel['zerocrossingrate']:
                    timbre_features.append(lowlevel['zerocrossingrate']['var'])

            # TONALITY FEATURES
            tonality_features = []

            # Chroma features (chord histograms)
            if 'chords_histogram' in tonal:
                tonality_features.extend(tonal['chords_histogram'])

            if 'hpcp' in tonal:
                if 'mean' in tonal['hpcp']:
                    tonality_features.extend(tonal['hpcp']['mean'])
                if 'var' in tonal['hpcp']:
                    tonality_features.extend(tonal['hpcp']['var'])

            # Key and scale features
            if 'key_key' in tonal:
                tonality_features.append(float(tonal['key_key']))
            if 'key_scale' in tonal:
                tonality_features.append(float(tonal['key_scale']))
            if 'key_strength' in tonal:
                tonality_features.append(float(tonal['key_strength']))

            # Tuning frequency
            if 'tuning_frequency' in tonal:
                tonality_features.append(float(tonal['tuning_frequency']))

            # TEMPO/RHYTHM FEATURES
            tempo_features = []

            # BPM and tempo
            if 'bpm' in rhythm:
                tempo_features.append(float(rhythm['bpm']))
            if 'bpm_histogram_first_peak_bpm' in rhythm:
                tempo_features.append(float(rhythm['bpm_histogram_first_peak_bpm']))
            if 'bpm_histogram_second_peak_bpm' in rhythm:
                tempo_features.append(float(rhythm['bpm_histogram_second_peak_bpm']))

            # Onset rate
            if 'onset_rate' in rhythm:
                tempo_features.append(float(rhythm['onset_rate']))

            # Beats loudness statistics
            if 'beats_loudness' in rhythm:
                if 'mean' in rhythm['beats_loudness']:
                    tempo_features.append(float(rhythm['beats_loudness']['mean']))
                if 'var' in rhythm['beats_loudness']:
                    tempo_features.append(float(rhythm['beats_loudness']['var']))

            # Danceability
            if 'danceability' in rhythm:
                tempo_features.append(float(rhythm['danceability']))

            # Ensure we have features
            if not timbre_features and not tonality_features and not tempo_features:
                return None

            # Convert to numpy arrays with proper handling of NaN/inf
            timbre_array = np.array(timbre_features, dtype=np.float32)
            tonality_array = np.array(tonality_features, dtype=np.float32)
            tempo_array = np.array(tempo_features, dtype=np.float32)

            # Replace NaN/inf with 0
            timbre_array = np.nan_to_num(timbre_array, nan=0.0, posinf=0.0, neginf=0.0)
            tonality_array = np.nan_to_num(tonality_array, nan=0.0, posinf=0.0, neginf=0.0)
            tempo_array = np.nan_to_num(tempo_array, nan=0.0, posinf=0.0, neginf=0.0)

            features = {
                'metadata': metadata,
                'timbre': timbre_array.tolist(),
                'tonality': tonality_array.tolist(),
                'tempo': tempo_array.tolist(),
                'track_hash': self.compute_track_hash(json_data)
            }

            return features

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

            for track in tracks_batch:
                metadata = track['metadata']

                # Track data
                track_data.append((
                    metadata['mbid'],
                    metadata['artist_mbid'],
                    metadata['release_mbid'],
                    metadata['recording_mbid'],
                    track['track_hash'],
                    metadata['length'],
                    True  # processed
                ))

                # Feature data
                feature_data.append((
                    metadata['mbid'],
                    track['timbre'],
                    track['tonality'],
                    track['tempo'],
                    Json(metadata)  # Store full metadata as JSONB
                ))

            try:
                # Insert tracks
                execute_batch(
                    cursor,
                    '''INSERT INTO tracks
                       (mbid, artist_mbid, release_mbid, recording_mbid, track_hash, length, processed)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (track_hash) DO NOTHING''',
                    track_data,
                    page_size=100
                )

                # Insert features
                execute_batch(
                    cursor,
                    '''INSERT INTO features
                       (mbid, timbre_features, tonality_features, tempo_features, metadata)
                       VALUES (%s, %s, %s, %s, %s)
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

    def process_directory(self, directory_path: str, max_files: int = None) -> int:
        """Process all JSON files in a directory"""
        directory = Path(directory_path)
        json_files = list(directory.rglob("*.json"))

        if max_files:
            json_files = json_files[:max_files]

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

    def process_acousticbrainz_dump(self, zst_file_path: str, max_files: int = None) -> str:
        """Complete pipeline to process AcousticBrainz dump file"""
        logger.info(f"Processing AcousticBrainz dump: {zst_file_path}")

        # Step 1: Decompress .zst file
        tar_file_path = self.decompress_zst_file(zst_file_path)

        # Step 2: Extract tar file
        extracted_dir = self.extract_tar_file(tar_file_path)

        # Step 3: Process JSON files
        processed_count = self.process_directory(extracted_dir, max_files)
        logger.info(f"Processed {processed_count} unique tracks")

        # Step 4: Export to CSV
        csv_file = self.export_features_to_csv()

        # Cleanup extracted files to save space
        try:
            import shutil
            shutil.rmtree(extracted_dir)
            os.remove(tar_file_path)
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Could not clean up temporary files: {e}")

        return csv_file

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
    parser.add_argument("--max-files", type=int, help="Maximum number of JSON files to process (for testing)")
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
        csv_file = processor.process_acousticbrainz_dump(
            args.zst_file,
            max_files=args.max_files
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

        print(f"\nFeatures dataset saved to: {csv_file}")
        print("Ready for similarity analysis!")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise
    finally:
        processor.close()
