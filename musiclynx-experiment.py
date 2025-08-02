import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
import pickle
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AcousticBrainzEncoder(nn.Module):
    """Neural encoder for AcousticBrainz features"""

    def __init__(self, timbre_dim=40, tonality_dim=36, tempo_dim=8, embedding_dim=128):
        super().__init__()

        # Separate encoders for each feature category
        self.timbre_encoder = nn.Sequential(
            nn.Linear(timbre_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.tonality_encoder = nn.Sequential(
            nn.Linear(tonality_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.tempo_encoder = nn.Sequential(
            nn.Linear(tempo_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(192, 256),  # 64 + 64 + 64 = 192
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, timbre, tonality, tempo):
        timbre_emb = self.timbre_encoder(timbre)
        tonality_emb = self.tonality_encoder(tonality)
        tempo_emb = self.tempo_encoder(tempo)

        # Concatenate and fuse
        combined = torch.cat([timbre_emb, tonality_emb, tempo_emb], dim=1)
        embedding = self.fusion(combined)

        # L2 normalize for cosine similarity
        return F.normalize(embedding, p=2, dim=1)

class ContrastiveLoss(nn.Module):
    """Contrastive loss for similarity learning"""

    def __init__(self, margin=1.0, temperature=0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, embeddings, labels):
        # Compute pairwise cosine similarities
        similarities = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Create positive and negative masks
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.t()).float()
        negative_mask = (labels != labels.t()).float()

        # Remove diagonal (self-similarity)
        positive_mask.fill_diagonal_(0)

        # InfoNCE-style loss
        exp_similarities = torch.exp(similarities)

        # Positive pairs
        positive_similarities = similarities * positive_mask

        # Negative pairs (all other samples)
        denominator = torch.sum(exp_similarities * negative_mask, dim=1, keepdim=True)

        # Compute loss for each positive pair
        loss = 0
        num_positives = 0

        for i in range(embeddings.size(0)):
            positive_indices = positive_mask[i].nonzero().squeeze()
            if positive_indices.numel() > 0:
                if positive_indices.dim() == 0:
                    positive_indices = positive_indices.unsqueeze(0)

                for j in positive_indices:
                    numerator = exp_similarities[i, j]
                    loss += -torch.log(numerator / (denominator[i] + numerator))
                    num_positives += 1

        return loss / max(num_positives, 1)

class AcousticBrainzExperiment:
    """Main experiment class comparing raw features vs embeddings"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.scalers = {
            'timbre': RobustScaler(),
            'tonality': StandardScaler(),
            'tempo': StandardScaler()
        }

    def load_data(self, features_file: str, metadata_file: str = None) -> Dict:
        """
        Load AcousticBrainz features
        Expected format: CSV with columns for track_id, artist_id, and feature columns
        """
        print("Loading AcousticBrainz data...")

        # Load features (you'll need to adapt this to your data format)
        df = pd.read_csv(features_file)

        # Separate feature categories (adapt column names to your data)
        timbre_cols = [col for col in df.columns if 'mfcc' in col.lower() or 'spectral' in col.lower()]
        tonality_cols = [col for col in df.columns if 'chord' in col.lower() or 'key' in col.lower()]
        tempo_cols = [col for col in df.columns if 'bpm' in col.lower() or 'onset' in col.lower() or 'tempo' in col.lower()]

        print(f"Found {len(timbre_cols)} timbre features, {len(tonality_cols)} tonality features, {len(tempo_cols)} tempo features")

        # Extract features
        timbre_features = df[timbre_cols].values
        tonality_features = df[tonality_cols].values
        tempo_features = df[tempo_cols].values

        # Handle missing values
        timbre_features = np.nan_to_num(timbre_features, nan=0.0)
        tonality_features = np.nan_to_num(tonality_features, nan=0.0)
        tempo_features = np.nan_to_num(tempo_features, nan=0.0)

        return {
            'timbre': timbre_features,
            'tonality': tonality_features,
            'tempo': tempo_features,
            'track_ids': df['track_id'].values if 'track_id' in df.columns else np.arange(len(df)),
            'artist_ids': df['artist_id'].values if 'artist_id' in df.columns else np.zeros(len(df)),
            'raw_data': df
        }

    def preprocess_features(self, data: Dict, fit_scalers=True) -> Dict:
        """Normalize features"""
        print("Preprocessing features...")

        if fit_scalers:
            data['timbre'] = self.scalers['timbre'].fit_transform(data['timbre'])
            data['tonality'] = self.scalers['tonality'].fit_transform(data['tonality'])
            data['tempo'] = self.scalers['tempo'].fit_transform(data['tempo'])
        else:
            data['timbre'] = self.scalers['timbre'].transform(data['timbre'])
            data['tonality'] = self.scalers['tonality'].transform(data['tonality'])
            data['tempo'] = self.scalers['tempo'].transform(data['tempo'])

        return data

    def create_training_pairs(self, artist_ids: np.ndarray, n_positive=5000, n_negative=10000) -> Tuple:
        """Create positive and negative pairs for contrastive learning"""
        print("Creating training pairs...")

        # Group tracks by artist
        artist_to_tracks = {}
        for i, artist_id in enumerate(artist_ids):
            if artist_id not in artist_to_tracks:
                artist_to_tracks[artist_id] = []
            artist_to_tracks[artist_id].append(i)

        positive_pairs = []
        negative_pairs = []

        # Create positive pairs (same artist)
        for artist_id, track_indices in artist_to_tracks.items():
            if len(track_indices) > 1:
                for i in range(min(len(track_indices), 10)):  # Max 10 tracks per artist
                    for j in range(i+1, min(len(track_indices), 10)):
                        positive_pairs.append((track_indices[i], track_indices[j]))
                        if len(positive_pairs) >= n_positive:
                            break
                if len(positive_pairs) >= n_positive:
                    break

        # Create negative pairs (different artists)
        artists = list(artist_to_tracks.keys())
        for _ in range(n_negative):
            artist1, artist2 = np.random.choice(artists, 2, replace=False)
            track1 = np.random.choice(artist_to_tracks[artist1])
            track2 = np.random.choice(artist_to_tracks[artist2])
            negative_pairs.append((track1, track2))

        print(f"Created {len(positive_pairs)} positive pairs, {len(negative_pairs)} negative pairs")
        return positive_pairs, negative_pairs

    def train_embedding_model(self, data: Dict, epochs=100, batch_size=512, lr=1e-3):
        """Train the neural embedding model"""
        print("Training embedding model...")

        # Initialize model
        self.model = AcousticBrainzEncoder(
            timbre_dim=data['timbre'].shape[1],
            tonality_dim=data['tonality'].shape[1],
            tempo_dim=data['tempo'].shape[1],
            embedding_dim=128
        ).to(self.device)

        # Create training pairs
        positive_pairs, negative_pairs = self.create_training_pairs(data['artist_ids'])

        # Prepare training data
        all_pairs = positive_pairs + negative_pairs
        pair_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

        # Convert to tensors
        timbre_tensor = torch.FloatTensor(data['timbre']).to(self.device)
        tonality_tensor = torch.FloatTensor(data['tonality']).to(self.device)
        tempo_tensor = torch.FloatTensor(data['tempo']).to(self.device)

        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = ContrastiveLoss()

        self.model.train()
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0

            # Shuffle pairs
            indices = np.random.permutation(len(all_pairs))

            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_pairs = [all_pairs[j] for j in batch_indices]
                batch_labels = [pair_labels[j] for j in batch_indices]

                # Get embeddings for all tracks in batch
                track_indices = list(set([idx for pair in batch_pairs for idx in pair]))

                if len(track_indices) == 0:
                    continue

                track_embeddings = self.model(
                    timbre_tensor[track_indices],
                    tonality_tensor[track_indices],
                    tempo_tensor[track_indices]
                )

                # Create labels for contrastive loss
                track_to_embedding_idx = {track_idx: i for i, track_idx in enumerate(track_indices)}

                # Simple approach: use artist IDs as labels
                artist_labels = torch.LongTensor([data['artist_ids'][idx] for idx in track_indices]).to(self.device)

                optimizer.zero_grad()
                loss = criterion(track_embeddings, artist_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        return losses

    def get_embeddings(self, data: Dict) -> np.ndarray:
        """Generate embeddings for all tracks"""
        print("Generating embeddings...")

        if self.model is None:
            raise ValueError("Model not trained yet!")

        self.model.eval()
        embeddings = []

        timbre_tensor = torch.FloatTensor(data['timbre']).to(self.device)
        tonality_tensor = torch.FloatTensor(data['tonality']).to(self.device)
        tempo_tensor = torch.FloatTensor(data['tempo']).to(self.device)

        batch_size = 1000
        with torch.no_grad():
            for i in range(0, len(data['timbre']), batch_size):
                end_idx = min(i + batch_size, len(data['timbre']))
                batch_embeddings = self.model(
                    timbre_tensor[i:end_idx],
                    tonality_tensor[i:end_idx],
                    tempo_tensor[i:end_idx]
                )
                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def cluster_features(self, features: np.ndarray, method='hdbscan') -> np.ndarray:
        """Cluster features using HDBSCAN"""
        print(f"Clustering with {method}...")

        if method == 'hdbscan':
            clusterer = HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean')
            cluster_labels = clusterer.fit_predict(features)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        print(f"Found {n_clusters} clusters, {n_noise} noise points")
        return cluster_labels

    def bipartite_projection_similarity(self, artist_ids: np.ndarray, cluster_labels: np.ndarray) -> Dict:
        """Your original bipartite graph projection for artist similarity"""
        print("Computing bipartite projection similarity...")

        # Create artist-cluster matrix
        unique_artists = np.unique(artist_ids)
        unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])  # Exclude noise

        artist_to_idx = {artist: i for i, artist in enumerate(unique_artists)}
        cluster_to_idx = {cluster: i for i, cluster in enumerate(unique_clusters)}

        # Build bipartite adjacency matrix
        artist_cluster_matrix = np.zeros((len(unique_artists), len(unique_clusters)))

        for track_idx, (artist_id, cluster_id) in enumerate(zip(artist_ids, cluster_labels)):
            if cluster_id >= 0:  # Skip noise points
                artist_idx = artist_to_idx[artist_id]
                cluster_idx = cluster_to_idx[cluster_id]
                artist_cluster_matrix[artist_idx, cluster_idx] += 1

        # Normalize by artist track counts
        artist_track_counts = np.sum(artist_cluster_matrix, axis=1, keepdims=True)
        artist_track_counts[artist_track_counts == 0] = 1  # Avoid division by zero
        artist_cluster_matrix = artist_cluster_matrix / artist_track_counts

        # Compute artist-artist similarity (cosine similarity of cluster profiles)
        from sklearn.metrics.pairwise import cosine_similarity
        artist_similarity_matrix = cosine_similarity(artist_cluster_matrix)

        return {
            'similarity_matrix': artist_similarity_matrix,
            'artists': unique_artists,
            'artist_cluster_matrix': artist_cluster_matrix
        }

    def evaluate_clustering(self, true_labels: np.ndarray, predicted_labels: np.ndarray, features: np.ndarray) -> Dict:
        """Evaluate clustering quality"""
        # Remove noise points for evaluation
        mask = predicted_labels >= 0
        if np.sum(mask) == 0:
            return {'ari': 0, 'silhouette': 0}

        true_labels_clean = true_labels[mask]
        predicted_labels_clean = predicted_labels[mask]
        features_clean = features[mask]

        ari = adjusted_rand_score(true_labels_clean, predicted_labels_clean)

        if len(np.unique(predicted_labels_clean)) > 1:
            silhouette = silhouette_score(features_clean, predicted_labels_clean)
        else:
            silhouette = 0

        return {'ari': ari, 'silhouette': silhouette}

    def run_experiment(self, features_file: str, save_results=True):
        """Run the complete experiment comparing raw vs embedding features"""
        print("=== Starting AcousticBrainz Embedding Experiment ===")

        # Load and preprocess data
        data = self.load_data(features_file)
        data = self.preprocess_features(data)

        # Combine raw features for baseline
        raw_features = np.hstack([data['timbre'], data['tonality'], data['tempo']])

        print(f"\nDataset: {len(data['track_ids'])} tracks, {len(np.unique(data['artist_ids']))} artists")

        # Experiment 1: Raw features + HDBSCAN
        print("\n=== Experiment 1: Raw Features ===")
        raw_clusters = self.cluster_features(raw_features)
        raw_similarity = self.bipartite_projection_similarity(data['artist_ids'], raw_clusters)
        raw_metrics = self.evaluate_clustering(data['artist_ids'], raw_clusters, raw_features)

        # Experiment 2: Train embeddings + HDBSCAN
        print("\n=== Experiment 2: Neural Embeddings ===")
        train_losses = self.train_embedding_model(data)
        embeddings = self.get_embeddings(data)
        embedding_clusters = self.cluster_features(embeddings)
        embedding_similarity = self.bipartite_projection_similarity(data['artist_ids'], embedding_clusters)
        embedding_metrics = self.evaluate_clustering(data['artist_ids'], embedding_clusters, embeddings)

        # Compare results
        results = {
            'raw_features': {
                'clusters': raw_clusters,
                'similarity': raw_similarity,
                'metrics': raw_metrics
            },
            'embeddings': {
                'clusters': embedding_clusters,
                'similarity': embedding_similarity,
                'metrics': embedding_metrics,
                'training_losses': train_losses
            },
            'data': data
        }

        print("\n=== Results Comparison ===")
        print(f"Raw Features - ARI: {raw_metrics['ari']:.3f}, Silhouette: {raw_metrics['silhouette']:.3f}")
        print(f"Embeddings   - ARI: {embedding_metrics['ari']:.3f}, Silhouette: {embedding_metrics['silhouette']:.3f}")

        if save_results:
            with open('experiment_results.pkl', 'wb') as f:
                pickle.dump(results, f)
            print("\nResults saved to experiment_results.pkl")

        return results

# Example usage
if __name__ == "__main__":
    # Initialize experiment
    experiment = AcousticBrainzExperiment()

    # Run experiment (replace with your data file)
    # results = experiment.run_experiment('acousticbrainz_features.csv')

    print("Experiment code ready! To run:")
    print("1. Prepare your AcousticBrainz data as a CSV with columns: track_id, artist_id, [feature_columns]")
    print("2. Call: experiment.run_experiment('your_data_file.csv')")
