#!/usr/bin/env python3
"""
HDBSCAN clustering experiments for artist similarity.

Creates track clusters from audio features, then projects to artist similarity
via bipartite graph (artist -> clusters <- artist).
"""

import numpy as np
import pandas as pd
import psycopg2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class FeatureSet:
    """Definition of a feature set for clustering."""
    name: str
    columns: list[str]
    description: str


# Feature set definitions
FEATURE_SETS = {
    "spectral": FeatureSet(
        name="spectral",
        columns=[
            "mfcc_mean",           # 13 dims
            "mfcc_std",            # 13 dims
            "spectral_contrast_coeffs_mean",  # 6 dims
            "spectral_contrast_coeffs_std",   # 6 dims
            "spectral_complexity_mean",       # 1 dim
            "dynamic_complexity",             # 1 dim
        ],
        description="Timbre/instrumentation features (40 dims)",
    ),
    "tonal": FeatureSet(
        name="tonal",
        columns=[
            "chords_histogram",    # 24 dims
            "thpcp",               # 36 dims
        ],
        description="Harmonic/tonal features (60 dims)",
    ),
    "rhythm": FeatureSet(
        name="rhythm",
        columns=[
            "bpm",                 # 1 dim (will convert to bps)
            "onset_rate",          # 1 dim
            "beats_loudness_mean", # 1 dim
            "beats_loudness_std",  # 1 dim
            "beats_count",         # 1 dim
        ],
        description="Rhythmic features (5 dims)",
    ),
    "combined": FeatureSet(
        name="combined",
        columns=[
            # Spectral (40)
            "mfcc_mean", "mfcc_std",
            "spectral_contrast_coeffs_mean", "spectral_contrast_coeffs_std",
            "spectral_complexity_mean", "dynamic_complexity",
            # Tonal (60)
            "chords_histogram", "thpcp",
            # Rhythm (5)
            "bpm", "onset_rate", "beats_loudness_mean", "beats_loudness_std", "beats_count",
        ],
        description="All features combined (105 dims)",
    ),
}


class ClusteringExperiment:
    """Run HDBSCAN clustering and compute artist similarity."""

    def __init__(
        self,
        min_cluster_size: int = 23,
        min_samples: int = 11,
        output_dir: Path = Path("results"),
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_features(self, feature_set: FeatureSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load features from database.

        Returns:
            features: (n_tracks, n_dims) feature matrix
            track_ids: (n_tracks,) track MBIDs
            artist_ids: (n_tracks,) artist MBIDs
        """
        print(f"Loading {feature_set.name} features...")

        columns = ", ".join(feature_set.columns)
        query = f"""
            SELECT t.mbid, t.artist_mbid, {columns}
            FROM track_features tf
            JOIN tracks t ON t.mbid = tf.mbid
            WHERE t.artist_mbid IS NOT NULL
        """

        # Use Unix socket (peer auth)
        conn = psycopg2.connect(
            dbname="acousticbrainz",
            user="alo",
            host="/var/run/postgresql",
        )
        df = pd.read_sql(query, conn)
        conn.close()

        print(f"  Loaded {len(df):,} tracks")

        # Extract track and artist IDs
        track_ids = df["mbid"].values
        artist_ids = df["artist_mbid"].values

        # Build feature matrix - expand array columns
        feature_arrays = []
        for col in feature_set.columns:
            values = df[col].values

            if col == "bpm":
                # Convert BPM to BPS (beats per second)
                arr = np.array([v / 60.0 if v is not None else np.nan for v in values])
                feature_arrays.append(arr.reshape(-1, 1))
            elif isinstance(values[0], list):
                # Array column - stack into matrix
                arr = np.array([v if v is not None else [np.nan] * len(values[1]) for v in values])
                feature_arrays.append(arr)
            else:
                # Scalar column
                arr = np.array([v if v is not None else np.nan for v in values])
                feature_arrays.append(arr.reshape(-1, 1))

        features = np.hstack(feature_arrays)
        print(f"  Feature matrix shape: {features.shape}")

        # Handle missing values - drop rows with any NaN
        valid_mask = ~np.isnan(features).any(axis=1)
        features = features[valid_mask]
        track_ids = track_ids[valid_mask]
        artist_ids = artist_ids[valid_mask]

        print(f"  After dropping NaN: {len(features):,} tracks")

        return features, track_ids, artist_ids

    def cluster_tracks(self, features: np.ndarray) -> tuple[np.ndarray, hdbscan.HDBSCAN]:
        """Run HDBSCAN clustering on features.

        Returns:
            labels: cluster labels (-1 for noise)
            clusterer: fitted HDBSCAN object
        """
        print(f"Clustering {len(features):,} tracks...")
        print(f"  min_cluster_size={self.min_cluster_size}, min_samples={self.min_samples}")

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Run HDBSCAN
        start = time.time()
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(features_scaled)
        elapsed = time.time() - start

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        noise_pct = n_noise / len(labels) * 100

        print(f"  Found {n_clusters:,} clusters in {elapsed:.1f}s")
        print(f"  Noise points: {n_noise:,} ({noise_pct:.1f}%)")

        return labels, clusterer

    def build_artist_cluster_matrix(
        self,
        artist_ids: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build artist-cluster membership matrix.

        Returns:
            matrix: (n_artists, n_clusters) membership counts
            unique_artists: artist IDs
            unique_clusters: cluster IDs (excluding noise)
        """
        print("Building artist-cluster matrix...")

        # Get unique artists and clusters (excluding noise)
        unique_artists = np.unique(artist_ids)
        unique_clusters = np.array([c for c in np.unique(cluster_labels) if c >= 0])

        artist_to_idx = {a: i for i, a in enumerate(unique_artists)}
        cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}

        # Build matrix
        matrix = np.zeros((len(unique_artists), len(unique_clusters)))

        for artist_id, cluster_id in zip(artist_ids, cluster_labels):
            if cluster_id >= 0:  # Skip noise
                artist_idx = artist_to_idx[artist_id]
                cluster_idx = cluster_to_idx[cluster_id]
                matrix[artist_idx, cluster_idx] += 1

        # Normalize by artist track count (L1 norm per row)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix_normalized = matrix / row_sums

        print(f"  Matrix shape: {matrix.shape}")
        print(f"  Artists with cluster membership: {(matrix.sum(axis=1) > 0).sum():,}")

        return matrix_normalized, unique_artists, unique_clusters

    def compute_artist_similarity(
        self,
        artist_cluster_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute pairwise artist similarity from cluster membership.

        Returns:
            similarity: (n_artists, n_artists) cosine similarity matrix
        """
        print("Computing artist similarity...")

        similarity = cosine_similarity(artist_cluster_matrix)

        print(f"  Similarity matrix shape: {similarity.shape}")

        return similarity

    def run_experiment(self, feature_set_name: str) -> dict:
        """Run full experiment for a feature set."""
        print(f"\n{'='*60}")
        print(f"Running experiment: {feature_set_name}")
        print(f"{'='*60}")

        feature_set = FEATURE_SETS[feature_set_name]
        print(f"Description: {feature_set.description}")

        # Load features
        features, track_ids, artist_ids = self.load_features(feature_set)

        # Cluster
        cluster_labels, clusterer = self.cluster_tracks(features)

        # Build bipartite projection
        artist_cluster_matrix, unique_artists, unique_clusters = self.build_artist_cluster_matrix(
            artist_ids, cluster_labels
        )

        # Compute similarity
        artist_similarity = self.compute_artist_similarity(artist_cluster_matrix)

        # Package results
        results = {
            "feature_set": feature_set_name,
            "description": feature_set.description,
            "n_tracks": len(track_ids),
            "n_features": features.shape[1],
            "n_clusters": len(unique_clusters),
            "n_noise": (cluster_labels == -1).sum(),
            "noise_pct": (cluster_labels == -1).sum() / len(cluster_labels) * 100,
            "n_artists": len(unique_artists),
            "track_ids": track_ids,
            "artist_ids": artist_ids,
            "cluster_labels": cluster_labels,
            "unique_artists": unique_artists,
            "unique_clusters": unique_clusters,
            "artist_cluster_matrix": artist_cluster_matrix,
            "artist_similarity": artist_similarity,
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
        }

        # Save results
        output_path = self.output_dir / f"{feature_set_name}_results.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(results, f)
        print(f"\nSaved results to {output_path}")

        return results

    def run_all_experiments(self) -> dict[str, dict]:
        """Run experiments for all feature sets."""
        all_results = {}

        for name in FEATURE_SETS:
            results = self.run_experiment(name)
            all_results[name] = results

            # Print summary
            print(f"\nSummary for {name}:")
            print(f"  Tracks: {results['n_tracks']:,}")
            print(f"  Features: {results['n_features']}")
            print(f"  Clusters: {results['n_clusters']:,}")
            print(f"  Noise: {results['noise_pct']:.1f}%")
            print(f"  Artists: {results['n_artists']:,}")

        return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run clustering experiments")
    parser.add_argument(
        "--feature-set", "-f",
        choices=list(FEATURE_SETS.keys()) + ["all"],
        default="all",
        help="Feature set to use (default: all)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=23,
        help="HDBSCAN min_cluster_size (default: 23)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=11,
        help="HDBSCAN min_samples (default: 11)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("results"),
        help="Output directory for results (default: results)",
    )

    args = parser.parse_args()

    experiment = ClusteringExperiment(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        output_dir=args.output_dir,
    )

    if args.feature_set == "all":
        experiment.run_all_experiments()
    else:
        experiment.run_experiment(args.feature_set)


if __name__ == "__main__":
    main()
