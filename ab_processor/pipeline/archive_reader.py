"""Streaming reader for tar.zst archives."""

import json
import tarfile
from pathlib import Path
from typing import Iterator, Optional
from dataclasses import dataclass

import zstandard as zstd


@dataclass
class ArchiveEntry:
    """A single entry from the archive."""
    mbid: str
    filename: str
    data: dict
    size_bytes: int


class StreamingArchiveReader:
    """Stream JSON files from tar.zst archives without full extraction."""

    def __init__(self, archive_path: Path):
        self.archive_path = archive_path
        self.dctx = zstd.ZstdDecompressor()

    def iter_json_files(self, skip_count: int = 0) -> Iterator[ArchiveEntry]:
        """Iterate over JSON files in the archive.

        Args:
            skip_count: Number of files to skip (for resuming)

        Yields:
            ArchiveEntry for each JSON file
        """
        with open(self.archive_path, 'rb') as fh:
            with self.dctx.stream_reader(fh) as stream:
                with tarfile.open(fileobj=stream, mode='r|') as tar:
                    skipped = 0
                    for member in tar:
                        if not member.isfile() or not member.name.endswith('.json'):
                            continue

                        # Skip files for resumption
                        if skipped < skip_count:
                            skipped += 1
                            continue

                        # Extract MBID from path (format: xx/uuid.json)
                        mbid = self._extract_mbid(member.name)
                        if mbid is None:
                            continue

                        # Read and parse JSON
                        f = tar.extractfile(member)
                        if f is None:
                            continue

                        try:
                            content = f.read()
                            data = json.loads(content)
                            yield ArchiveEntry(
                                mbid=mbid,
                                filename=member.name,
                                data=data,
                                size_bytes=len(content),
                            )
                        except json.JSONDecodeError:
                            continue
                        finally:
                            f.close()

    def count_files(self) -> int:
        """Count total JSON files in archive (slow - reads entire archive)."""
        count = 0
        with open(self.archive_path, 'rb') as fh:
            with self.dctx.stream_reader(fh) as stream:
                with tarfile.open(fileobj=stream, mode='r|') as tar:
                    for member in tar:
                        if member.isfile() and member.name.endswith('.json'):
                            count += 1
        return count

    def _extract_mbid(self, path: str) -> Optional[str]:
        """Extract MBID from archive path.

        Expected format: acousticbrainz-lowlevel-json-20220623/lowlevel/xx/x/uuid-N.json
        where N is the submission number (we strip it to get the bare UUID).
        """
        parts = Path(path).parts
        if len(parts) < 2:
            return None

        filename = parts[-1]
        if not filename.endswith('.json'):
            return None

        # Remove .json extension
        name = filename[:-5]

        # Strip submission number suffix (-0, -1, etc.) to get bare UUID
        # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (36 chars)
        if len(name) > 36 and name[36] == '-':
            return name[:36]

        return name
