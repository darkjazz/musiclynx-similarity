"""Configuration for AcousticBrainz processor."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration settings."""

    # Database settings
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "acousticbrainz"
    db_user: str = "alo"
    db_password: str = "05+gr35"

    # Data paths
    data_path: Path = Path("/media/alo/TOSHIBA EXT/data/ab/")
    archive_pattern: str = "acousticbrainz-lowlevel-json-20220623-{}.tar.zst"
    num_archives: int = 30  # Archives numbered 0-29

    # Processing settings
    batch_size: int = 1000
    estimate_after_tracks: int = 1000
    estimated_total_tracks: int = 30_000_000

    @property
    def db_url(self) -> str:
        """Get database connection URL."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    def get_archive_path(self, index: int) -> Path:
        """Get path to a specific archive file."""
        return self.data_path / self.archive_pattern.format(index)

    def get_all_archive_paths(self) -> list[Path]:
        """Get paths to all archive files."""
        return [self.get_archive_path(i) for i in range(self.num_archives)]
