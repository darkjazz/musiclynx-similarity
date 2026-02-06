# musiclynx-similarity

AcousticBrainz feature extraction system for music similarity analysis.

## Overview

Processes AcousticBrainz data dumps into PostgreSQL for music similarity experiments. Extracts audio features from ~30M JSON files, deduplicates to ~8M unique tracks.

## Requirements

- Python 3.10+
- PostgreSQL
- `psycopg2-binary`
- `zstandard`
- `scikit-learn`, `hdbscan`, `pandas`, `numpy` (for experiments)

## Data Source

AcousticBrainz low-level JSON dumps:
- 30 tar.zst archives (~589GB compressed)
- Location: `/media/alo/TOSHIBA EXT/data/ab/`

## Usage

```bash
# Initialize database schema
python main.py init

# Process all archives
python main.py process

# Process specific archives
python main.py process --archives 0 1 2

# Rerun a single extractor
python main.py rerun mfcc --archives 0

# Show statistics
python main.py stats

# List available extractors
python main.py list-extractors
```

## Extractors

| Name | Features |
|------|----------|
| thpcp | 36-dim tonal histogram profile |
| key | key, scale, strength |
| chords | changes_rate, number_rate, key, scale |
| tuning | frequency, equal_tempered_deviation |
| bpm | tempo, confidence |
| danceability | danceability score |
| beats | count, loudness_mean/std, onset_rate |
| mfcc | mean (13-dim), covariance matrix |
| melbands | mean, std arrays |
| spectral | centroid, flux, rolloff, complexity, contrast |
| loudness | mean, dynamic_complexity |
| dissonance | mean, std |
| track_hash | SHA256 hash for deduplication |

## Database Schema

- `artists` (mbid, name)
- `tracks` (mbid, title, album_name, artist_mbid, track_hash, length_seconds, source_archive)
- `track_features` (mbid + all extractor columns)
- `processing_state` (archive progress for resumability)

## Experiments

### Clustering-based Artist Similarity

`experiments/clustering.py` computes artist similarity via HDBSCAN clustering. Tracks are clustered by audio features, then artist similarity is derived from shared cluster membership (bipartite graph projection with cosine similarity).

**Feature sets:**

| Name | Dimensions | Description |
|------|-----------|-------------|
| spectral | 40 | MFCC, spectral contrast, complexity |
| tonal | 60 | Chords histogram, tonal profile |
| rhythm | 5 | BPM, onset rate, beats loudness |
| combined | 105 | All of the above |

**Usage:**

```bash
# Run all feature sets
python experiments/clustering.py

# Run a specific feature set
python experiments/clustering.py -f spectral

# Custom HDBSCAN parameters
python experiments/clustering.py -f combined --min-cluster-size 30 --min-samples 15

# Custom output directory
python experiments/clustering.py -o my_results/
```

Results are saved as pickle files in the output directory (default: `results/`).

## Adding New Extractors

1. Create class in `ab_processor/extractors/`
2. Inherit from `FeatureExtractor`
3. Decorate with `@register_extractor`
4. Run `python main.py init` to add columns
5. Run `python main.py rerun <name>` to populate
