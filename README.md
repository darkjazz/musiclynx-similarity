# musiclynx-similarity

AcousticBrainz feature extraction system for music similarity analysis.

## Overview

Processes AcousticBrainz data dumps into PostgreSQL for music similarity experiments. Extracts audio features from ~30M JSON files, deduplicates to ~8M unique tracks.

## Requirements

- Python 3.10+
- PostgreSQL
- `psycopg2-binary`
- `zstandard`

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

## Adding New Extractors

1. Create class in `ab_processor/extractors/`
2. Inherit from `FeatureExtractor`
3. Decorate with `@register_extractor`
4. Run `python main.py init` to add columns
5. Run `python main.py rerun <name>` to populate
