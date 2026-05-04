"""
migrate_tags.py -- retroactively add tags + expires_at to all existing episodes.

Usage:
    PYENV_VERSION=aura-env python3 scripts/migrate_tags.py --store ~/.ctm/memory/tars
    PYENV_VERSION=aura-env python3 scripts/migrate_tags.py --store ~/.ctm/memory/aura

Safe to run multiple times (idempotent). Skips episodes that already have tags.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from episodic_memory.tagger import EpisodicTagger


def migrate(store_path: str, dry_run: bool = False) -> None:
    p = Path(store_path).expanduser()
    db_path   = p / "episodes.db"
    meta_path = p / "hot_metadata.json"

    if not db_path.exists():
        print(f"ERROR: episodes.db not found at {p}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))

    # ── Add columns if they don't exist ───────────────────────────────────────
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(episodes)")}
    if "tags" not in existing_cols:
        print("Adding 'tags' column to episodes table...")
        if not dry_run:
            conn.execute("ALTER TABLE episodes ADD COLUMN tags TEXT DEFAULT '[]'")
            conn.commit()

    if "expires_at" not in existing_cols:
        print("Adding 'expires_at' column to episodes table...")
        if not dry_run:
            conn.execute("ALTER TABLE episodes ADD COLUMN expires_at REAL DEFAULT NULL")
            conn.commit()

    # ── Load all episodes that need tagging ────────────────────────────────────
    rows = conn.execute(
        "SELECT session_id, summary, stored_at, tags FROM episodes"
    ).fetchall()

    tagger    = EpisodicTagger(use_roleplay_filter=True)
    total     = len(rows)
    updated   = 0
    skipped   = 0
    tag_stats: dict[str, int] = {}

    print(f"\nTagging {total} episodes in {p}...")

    for session_id, summary, stored_at, existing_tags_json in rows:
        # Skip if already tagged (non-empty tags column)
        try:
            existing = json.loads(existing_tags_json or "[]")
        except json.JSONDecodeError:
            existing = []

        if existing:
            skipped += 1
            continue

        result = tagger.tag(summary=summary or "", stored_at=stored_at or time.time())

        for t in result.tags:
            tag_stats[t] = tag_stats.get(t, 0) + 1

        if not dry_run:
            conn.execute(
                "UPDATE episodes SET tags=?, expires_at=? WHERE session_id=?",
                (json.dumps(result.tags), result.expires_at, session_id),
            )

        updated += 1

    if not dry_run:
        conn.commit()

    conn.close()

    # ── Update hot_metadata.json with tags ─────────────────────────────────────
    if meta_path.exists() and not dry_run:
        with open(meta_path) as f:
            hot_meta = json.load(f)

        # Re-tag everything (fast, no DB needed)
        meta_by_sid = {m["session_id"]: m for m in hot_meta}

        # Reload tags from DB to keep in sync
        conn2 = sqlite3.connect(str(db_path))
        db_tags = {
            row[0]: json.loads(row[1] or "[]")
            for row in conn2.execute("SELECT session_id, tags FROM episodes")
        }
        conn2.close()

        for m in hot_meta:
            sid = m["session_id"]
            if sid in db_tags:
                m["tags"] = db_tags[sid]

        with open(meta_path, "w") as f:
            json.dump(hot_meta, f, indent=2)

        print(f"hot_metadata.json updated with tags.")

    # ── Report ─────────────────────────────────────────────────────────────────
    print(f"\n{'DRY RUN -- ' if dry_run else ''}Results:")
    print(f"  Total episodes : {total}")
    print(f"  Tagged now     : {updated}")
    print(f"  Already tagged : {skipped}")
    print(f"\nTag distribution:")
    for tag, count in sorted(tag_stats.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(updated, 1)
        print(f"  {tag:<16} {count:>4}  ({pct:.0f}%)")

    # Count expiring entries
    conn3 = sqlite3.connect(str(db_path))
    expiring = conn3.execute(
        "SELECT COUNT(*) FROM episodes WHERE expires_at IS NOT NULL"
    ).fetchone()[0]
    expired = conn3.execute(
        "SELECT COUNT(*) FROM episodes WHERE expires_at IS NOT NULL AND expires_at < ?",
        (time.time(),)
    ).fetchone()[0]
    conn3.close()
    print(f"\n  With TTL       : {expiring}")
    print(f"  Already expired: {expired}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate episodic memory store to add tags + TTL")
    parser.add_argument("--store", required=True, help="Path to the store directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be tagged without writing")
    args = parser.parse_args()
    migrate(args.store, dry_run=args.dry_run)
