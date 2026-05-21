"""
episodic-memory CLI entry point.

Exposes administrative commands that are not part of the main Hermes integration.
"""

import os
import shutil
import sys
from pathlib import Path

def _get_store_dir(profile_name: str) -> Path:
    """Get default store path for a Hermes profile."""
    hermes_home = os.getenv("HERMES_HOME", "~/.hermes").replace("~", str(Path.home()))
    return Path(hermes_home) / "episodic_memory" / profile_name

def repair_store(profile_name: str, dry_run: bool = False) -> int:
    """Detect and remove episodes corrupted by Hermes scaffold turns."""
    from episodic_memory import EpisodicMemoryStore

    store_path = _get_store_dir(profile_name)
    if not store_path.exists():
        print(f"Store not found at {store_path}", file=sys.stderr)
        return 1

    store = EpisodicMemoryStore(store_path)
    if not (store_path / "episodes.db").exists():
        print(f"episodes.db not found in {store_path}", file=sys.stderr)
        return 1

    # Known scaffold prefixes
    SCAFFOLD_PREFIXES = (
        "Review the conversation above and consider saving",
        "Please summarize the conversation",
        "[CONTEXT COMPACTION",
        "Conversation summary:",
    )

    corrupted_ids = []
    for episode in store.iterate_episodes():
        summary = episode.get("summary", "")
        if summary.startswith(SCAFFOLD_PREFIXES):
            corrupted_ids.append(episode["key"])

    if not corrupted_ids:
        print(f"No corrupted episodes found in {store_path}")
        return 0

    if dry_run:
        print(f"DRY RUN: Would delete {len(corrupted_ids)} corrupted episodes:")
        for key in corrupted_ids:
            print(f"  - {key}")
        return 0

    # Remove from episodes.db
    with store._conn:
        store._conn.executemany(
            "DELETE FROM episodes WHERE key = ?",
            [(key,) for key in corrupted_ids]
        )
    print(f"Deleted {len(corrupted_ids)} corrupted episodes from episodes.db")

    # Prune hot_metadata.json
    hot_path = store_path / "hot_metadata.json"
    if hot_path.exists():
        import json
        with open(hot_path, 'r') as f:
            hot_data = json.load(f)

        # Filter out entries for corrupted episodes
        filtered = {
            k: v for k, v in hot_data.items()
            if k not in corrupted_ids
        }
        if len(filtered) != len(hot_data):
            with open(hot_path, 'w') as f:
                json.dump(filtered, f, indent=2)
            print(f"Pruned {len(hot_data)-len(filtered)} entries from hot_metadata.json")
        
    else:
        print("hot_metadata.json not found -- only episodes.db was cleaned")

    return 0


def cli():
    """Main CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Episodic Memory maintenance tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    repair_parser = subparsers.add_parser("repair", help="Repair scaffold-corrupted store")
    repair_parser.add_argument("profile", help="Hermes profile name")
    repair_parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")

    args = parser.parse_args()

    if args.command == "repair":
        exit(repair_store(args.profile, dry_run=args.dry_run))

if __name__ == "__main__":
    cli()
