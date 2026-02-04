#!/usr/bin/env python3
"""CLI entry point for AcousticBrainz feature extraction."""

import argparse
import sys

from ab_processor.config import Config
from ab_processor.db import SchemaManager, DatabaseOperations
from ab_processor.pipeline import BatchProcessor
from ab_processor.extractors import get_extractor, get_all_extractors


def cmd_init(args, config: Config):
    """Initialize database schema."""
    print("Initializing database schema...")
    schema = SchemaManager(config)
    schema.init_schema()
    print("Done.")


def cmd_process(args, config: Config):
    """Process archives."""
    db_ops = DatabaseOperations(config)
    processor = BatchProcessor(config, db_ops)

    # Determine which archives to process
    if args.archives:
        archive_indices = args.archives
    else:
        archive_indices = list(range(config.num_archives))

    print(f"Processing {len(archive_indices)} archives...")

    for idx in archive_indices:
        try:
            processor.process_archive(idx)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
        except KeyboardInterrupt:
            print("\nInterrupted. Progress saved.")
            sys.exit(1)

    print("\nAll archives processed.")


def cmd_rerun(args, config: Config):
    """Rerun a specific extractor."""
    extractor_name = args.extractor

    try:
        extractor_cls = get_extractor(extractor_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    extractor = extractor_cls()
    print(f"Rerunning extractor: {extractor_name} (v{extractor.version})")

    db_ops = DatabaseOperations(config)
    processor = BatchProcessor(config, db_ops)

    # Determine which archives to process
    if args.archives:
        archive_indices = args.archives
    else:
        archive_indices = list(range(config.num_archives))

    for idx in archive_indices:
        try:
            processor.process_archive(
                idx,
                extractors=[extractor],
                rerun=True,
            )
            db_ops.record_extractor_run(extractor_name, extractor.version, config.get_archive_path(idx).name)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
        except KeyboardInterrupt:
            print("\nInterrupted.")
            sys.exit(1)


def cmd_stats(args, config: Config):
    """Show database statistics."""
    db_ops = DatabaseOperations(config)
    stats = db_ops.get_stats()

    print("\n=== AcousticBrainz Database Statistics ===\n")

    print(f"Total tracks: {stats['total_tracks']:,}")
    print(f"Total artists: {stats['total_artists']:,}")

    print("\n--- Table Sizes ---")
    for table, size in stats['table_sizes'].items():
        print(f"  {table}: {size}")

    print("\n--- Feature Coverage ---")
    total = stats['total_tracks'] or 1
    for feature, count in sorted(stats['feature_coverage'].items()):
        pct = (count / total) * 100
        print(f"  {feature}: {count:,} ({pct:.1f}%)")

    print("\n--- Processing State ---")
    for state in stats['processing_state']:
        print(f"  {state['archive']}: {state['processed']:,} files ({state['status']})")


def cmd_list_extractors(args, config: Config):
    """List all registered extractors."""
    extractors = get_all_extractors()

    print("\n=== Registered Extractors ===\n")
    for name, cls in sorted(extractors.items()):
        extractor = cls()
        print(f"{name} (v{extractor.version})")
        print(f"  Columns: {', '.join(c.name for c in extractor.columns)}")
        print(f"  JSON paths: {', '.join(extractor.json_paths)}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="AcousticBrainz feature extraction system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize database schema")

    # process command
    process_parser = subparsers.add_parser("process", help="Process archives")
    process_parser.add_argument(
        "--archives", "-a",
        type=int,
        nargs="+",
        help="Specific archive indices to process (0-29)",
    )

    # rerun command
    rerun_parser = subparsers.add_parser("rerun", help="Rerun a specific extractor")
    rerun_parser.add_argument(
        "extractor",
        help="Name of extractor to rerun",
    )
    rerun_parser.add_argument(
        "--archives", "-a",
        type=int,
        nargs="+",
        help="Specific archive indices to process (0-29)",
    )

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")

    # list-extractors command
    list_parser = subparsers.add_parser("list-extractors", help="List all registered extractors")

    args = parser.parse_args()
    config = Config()

    if args.command == "init":
        cmd_init(args, config)
    elif args.command == "process":
        cmd_process(args, config)
    elif args.command == "rerun":
        cmd_rerun(args, config)
    elif args.command == "stats":
        cmd_stats(args, config)
    elif args.command == "list-extractors":
        cmd_list_extractors(args, config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
