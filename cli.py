"""
episodic-memory CLI entry point.

Exposes administrative commands that are not part of the main Hermes integration.
"""

import os
import sys
from pathlib import Path
import sqlite3
from typing import Optional
import json

# Add the src directory to the path so we can import from episodic_memory
src_dir = os.path.join(os.path.dirname(__file__), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from episodic_memory import EpisodicMemoryStore  # type: ignore # noqa: E402


def _get_store_path(agent_name: str) -> Path:
    """Get the store path for the given agent. Follows the same logic as the plugin."""
    hermes_home = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
    # If agent_name is a path, use it directly
    if "/" in agent_name or "." in agent_name:
        return Path(agent_name)
    return hermes_home / "episodic_memory" / agent_name


def _connect_db(db_path: Path) -> sqlite3.Connection:
    """Connect to the episodes.db sqlite database."""
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}", file=sys.stderr)
        sys.exit(1)
    return sqlite3.connect(str(db_path))


def repair_corrupted_episodes(agent_name: str = "") -> None:
    """Detect and remove episodes with corrupted summaries due to scaffold turns.

    Scaffold turns (context compaction, summarization prompts) were accidentally stored
    as episode summaries and must be purged from both the DB and hot metadata.

    This should be run once after installing from a version that contained the bug.
    """
    store_path = _get_store_path(agent_name)
    db_path = store_path / "episodes.db"
    hot_path = store_path / "hot_metadata.json"

    # List of known scaffold prefixes that indicate corruption
    CORRUPTED_PREFIXES = [
        "Review the conversation above and consider saving",
        "Please summarize the conversation",
        "[CONTEXT COMPACTION",
        "Conversation summary:",
    ]

    # First, get list of episodes to delete from the DB
    conn = _connect_db(db_path)
    cursor = conn.cursor()
    to_delete: list[int] = []
    try:
        cursor.execute("SELECT id, summary FROM episodes")
        rows = cursor.fetchall()
        for row_id, summary in rows:
            if any(summary.startswith(prefix) for prefix in CORRUPTED_PREFIXES):
                to_delete.append(row_id)
    finally:
        conn.close()

    if not to_delete:
        print("No scaffold-corrupted episodes found.")
        return

    # Delete from DB
    conn = _connect_db(db_path)
    cursor = conn.cursor()
    try:
        placeholders = ",".join("?" * len(to_delete))
        cursor.execute(f"DELETE FROM episodes WHERE id IN ({placeholders})", to_delete)
        conn.commit()
        print(f"Deleted {len(to_delete)} corrupted episodes from episodes.db.")
    except Exception as e:
        print(f"Error deleting from DB: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()

    # Prune hot_metadata.json
    if not hot_path.exists():
        print(f"Warning: {hot_path} not found -- cannot prune hot metadata.")
        return

    try:
        with open(hot_path, "r") as f:
            hot_data = json.load(f)
    except Exception as e:
        print(f"Error reading {hot_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Remove metadata for deleted episodes
    original_count = len(hot_data)
    hot_data = [m for m in hot_data if m.get("episode_id") not in to_delete]
    if len(hot_data) == original_count:
        print("No matching metadata found in hot_metadata.json."
        " It might already be clean.")
    else:
        try:
            with open(hot_path, "w") as f:
                json.dump(hot_data, f, indent=2)
            print(f"Pruned hot metadata: removed {original_count - len(hot_data)} entries.")
        except Exception as e:
            print(f"Error writing {hot_path}: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"Successfully cleaned {len(to_delete)} corrupted episodes.")


if __name__ == "__main__":
    # This script is invoked as the entry point from the plugin directory
    # The first argument is the command, the rest are passed to it
    if len(sys.argv) < 2:
        print("Usage: hermes episodic-memory <command> [args]", file=sys.stderr)
        print("Available commands: repair", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    if command == "repair":
        agent_name = sys.argv[2] if len(sys.argv) > 2 else ""
        repair_corrupted_episodes(agent_name)
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)
