#!/usr/bin/env python3
"""
ingest_sessions.py -- Ingest Claude Code sessions into the episodic memory store.

Scans ~/.claude/projects/**/*.jsonl, extracts conversation turns, embeds them
with BGE, encodes with EpisodicEncoder, and writes to the episodic store.

Tracks already-ingested sessions in a ledger so re-runs are safe and fast.

Usage:
    python3 ingest_sessions.py [OPTIONS]

    --store PATH      Episodic store directory  (default: ~/.ctm/memory/tars)
    --projects PATH   Claude projects directory (default: ~/.claude/projects)
    --ledger PATH     Ingestion ledger DB       (default: <store>/claude_ingested.db)
    --model NAME      BGE model name            (default: BAAI/bge-large-en-v1.5)
    --min-turns N     Skip sessions with fewer than N turns (default: 3)
    --dry-run         Parse and report without writing to store
    --verbose         Show per-session detail

The script is intentionally standalone -- no Hermes dependency. It imports
directly from the episodic_memory library.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Claude Code JSONL parsing
# ---------------------------------------------------------------------------

def _text_from_content(content) -> str:
    """Extract plain text from a Claude Code message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "tool_result":
                # Include tool output so the store captures what was retrieved/done
                nested = block.get("content", "")
                parts.append(_text_from_content(nested))
        return " ".join(p for p in parts if p).strip()
    return ""


def parse_session(path: Path) -> tuple[str, list[dict], float]:
    """
    Parse a Claude Code JSONL file into a list of {role, content} turn dicts.

    Returns (session_id, turns, stored_at).
    session_id is derived from the filename UUID.
    stored_at is the timestamp of the last message in the file.
    """
    session_id = path.stem  # UUID filename without extension
    turns = []
    last_ts = path.stat().st_mtime

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        log.warning("Could not read %s: %s", path, e)
        return session_id, [], last_ts

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue

        msg_type = obj.get("type")
        if msg_type not in ("user", "assistant"):
            continue

        # Timestamp tracking
        ts = obj.get("timestamp")
        if ts:
            try:
                last_ts = float(ts) / 1000.0 if float(ts) > 1e10 else float(ts)
            except (ValueError, TypeError):
                pass

        msg = obj.get("message", {})
        role = msg.get("role", msg_type)
        content = _text_from_content(msg.get("content", ""))

        # Skip empty turns and pure tool-call turns with no readable text
        if not content.strip():
            continue

        # Normalise role to user/assistant
        if role not in ("user", "assistant"):
            role = "user" if msg_type == "user" else "assistant"

        turns.append({"role": role, "content": content})

    return session_id, turns, last_ts


# ---------------------------------------------------------------------------
# Ledger -- tracks which session files have been ingested
# ---------------------------------------------------------------------------

class IngestLedger:
    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ingested (
                session_id TEXT PRIMARY KEY,
                path       TEXT NOT NULL,
                ingested_at REAL NOT NULL,
                turn_count  INTEGER NOT NULL DEFAULT 0
            )
        """)
        self._conn.commit()

    def has(self, session_id: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM ingested WHERE session_id = ?", (session_id,)
        )
        return cur.fetchone() is not None

    def mark(self, session_id: str, path: Path, turn_count: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO ingested (session_id, path, ingested_at, turn_count) "
            "VALUES (?, ?, ?, ?)",
            (session_id, str(path), time.time(), turn_count),
        )
        self._conn.commit()

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM ingested").fetchone()[0]

    def close(self):
        self._conn.close()


# ---------------------------------------------------------------------------
# Encoding helpers (lifted from Hermes plugin, dependency-free)
# ---------------------------------------------------------------------------

def _make_summary(turns: list[dict], max_chars: int = 800) -> str:
    """Compact plain-text digest -- no LLM required."""
    prefix = {"user": "U", "assistant": "A"}
    lines = []
    for t in turns:
        role = prefix.get(t["role"], t["role"][0].upper())
        snippet = t["content"].strip().replace("\n", " ")[:160]
        lines.append(f"{role}: {snippet}")
    digest = "\n".join(lines)
    if len(digest) > max_chars:
        digest = digest[:max_chars].rsplit("\n", 1)[0] + "\n[...]"
    return digest


def _infer_project(path: Path) -> str:
    """Derive a human-readable project name from the Claude projects folder name.

    Examples:
        -home-richard-projects-gimli  → gimli
        -home-richard-projects-ctm    → ctm
        -home-richard                 → global
        -home-richard--claude         → claude-config
    """
    folder = path.parent.name  # e.g. "-home-richard-projects-gimli"
    SKIP = {"home", "richard", "projects", "claude", "hermes", "agent",
            "venv", "lib", "site", "packages", "profiles", ""}
    parts = [p for p in re.split(r"-+", folder) if p and p not in SKIP
             and not p.isdigit() and not p.startswith("python")]
    if parts:
        # Prefer the last segment -- most specific part of the path
        return parts[-1]
    return "global"


def encode_and_store(
    session_id: str,
    turns: list[dict],
    stored_at: float,
    project: str,
    store,
    embed_client,
    encoder,
) -> bool:
    """
    Embed turns, encode latent, write to store. Returns True on success.
    Mirrors the Hermes plugin _encode_and_store() logic exactly.
    """
    user_turns  = [t["content"] for t in turns if t["role"] == "user"]
    agent_turns = [t["content"] for t in turns if t["role"] == "assistant"]

    if not user_turns:
        log.debug("Session %s: no user turns, skipping.", session_id)
        return False

    # Pad to equal length (encoder expects paired turns)
    max_t = max(len(user_turns), len(agent_turns))
    user_turns  += [""] * (max_t - len(user_turns))
    agent_turns += [""] * (max_t - len(agent_turns))

    user_embs  = embed_client.embed(user_turns)   # (T, D)
    agent_embs = embed_client.embed(agent_turns)  # (T, D)

    # Normalise embedding dim to 1536 (encoder input_dim)
    target_dim = 1536
    for embs_ref in (user_embs, agent_embs):
        pass  # we rebuild below
    def _fix_dim(arr):
        if arr.shape[1] < target_dim:
            return np.pad(arr, ((0, 0), (0, target_dim - arr.shape[1])))
        return arr[:, :target_dim]
    user_embs  = _fix_dim(user_embs).astype(np.float32)
    agent_embs = _fix_dim(agent_embs).astype(np.float32)

    latent, coherence = encoder.encode_numpy(user_embs, agent_embs)

    summary = _make_summary(turns)

    store.add(
        session_id=session_id,
        latent=latent,
        transcript=turns,
        metadata={
            "turn_count":          len(turns),
            "stored_at":           stored_at,
            "coherence":           float(coherence) if coherence is not None else None,
            "project":             project,
            "source":              "claude_code",
            "emotion_cats":        [0.0] * 8,
            "dominant_emotion":    "neutral",
            "dominant_archetype":  "companion",
        },
        summary=summary,
    )
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Ingest Claude Code sessions into episodic memory")
    p.add_argument("--store",    default="~/.ctm/memory/tars",     help="Episodic store directory")
    p.add_argument("--projects", default="~/.claude/projects",     help="Claude projects root")
    p.add_argument("--ledger",   default=None,                     help="Ledger DB path (default: <store>/claude_ingested.db)")
    p.add_argument("--model",    default="BAAI/bge-large-en-v1.5", help="BGE embedding model")
    p.add_argument("--min-turns", type=int, default=3,             help="Skip sessions with fewer turns")
    p.add_argument("--dry-run",  action="store_true",              help="Parse without writing")
    p.add_argument("--verbose",  action="store_true",              help="Per-session logging")
    return p.parse_args()


def main():
    args = parse_args()

    store_path   = Path(args.store).expanduser()
    projects_dir = Path(args.projects).expanduser()
    ledger_path  = Path(args.ledger).expanduser() if args.ledger else store_path / "claude_ingested.db"

    if not store_path.exists():
        log.error("Store path does not exist: %s", store_path)
        sys.exit(1)
    if not projects_dir.exists():
        log.error("Claude projects directory does not exist: %s", projects_dir)
        sys.exit(1)

    # Discover all session JSONLs
    jsonl_files = sorted(projects_dir.rglob("*.jsonl"))
    log.info("Found %d session files under %s", len(jsonl_files), projects_dir)

    ledger = IngestLedger(ledger_path)
    log.info("Ledger: %d sessions previously ingested", ledger.count())

    # Filter to uningested
    pending = [f for f in jsonl_files if not ledger.has(f.stem)]
    log.info("%d sessions pending ingestion", len(pending))

    if not pending:
        log.info("Nothing to do.")
        ledger.close()
        return

    if args.dry_run:
        for f in pending:
            sid, turns, ts = parse_session(f)
            project = _infer_project(f)
            log.info("DRY-RUN  %s  project=%-20s  turns=%d", sid[:8], project, len(turns))
        ledger.close()
        return

    # Lazy-load heavy deps only when we have work to do
    log.info("Loading BGE model (%s)...", args.model)
    try:
        from sentence_transformers import SentenceTransformer

        _st = SentenceTransformer(args.model, device="cpu")

        class _EmbedClient:
            def embed(self, texts):
                return _st.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        embed_client = _EmbedClient()
    except Exception as e:
        log.error("Failed to load BGE model: %s", e)
        sys.exit(1)

    log.info("Loading EpisodicEncoder...")
    try:
        from episodic_memory import EpisodicEncoder, EpisodicEncoderConfig
        encoder = EpisodicEncoder(EpisodicEncoderConfig()).eval()
    except Exception as e:
        log.error("Failed to load EpisodicEncoder: %s", e)
        sys.exit(1)

    log.info("Opening episodic store at %s...", store_path)
    try:
        from episodic_memory import EpisodicMemoryStore
        store = EpisodicMemoryStore(str(store_path))
        log.info("Store has %d existing episodes.", store.n_episodes)
    except Exception as e:
        log.error("Failed to open store: %s", e)
        sys.exit(1)

    # Ingest
    ingested = 0
    skipped  = 0
    errors   = 0

    for f in pending:
        sid, turns, stored_at = parse_session(f)
        project = _infer_project(f)

        if len(turns) < args.min_turns:
            if args.verbose:
                log.info("SKIP  %s  project=%-20s  turns=%d  (< min %d)",
                         sid[:8], project, len(turns), args.min_turns)
            skipped += 1
            ledger.mark(sid, f, len(turns))  # mark so we don't revisit
            continue

        try:
            ok = encode_and_store(sid, turns, stored_at, project, store, embed_client, encoder)
            if ok:
                ledger.mark(sid, f, len(turns))
                ingested += 1
                if args.verbose:
                    log.info("OK    %s  project=%-20s  turns=%d", sid[:8], project, len(turns))
            else:
                skipped += 1
                ledger.mark(sid, f, len(turns))
        except Exception as e:
            log.error("ERROR %s  project=%-20s  %s", sid[:8], project, e)
            errors += 1

    log.info(
        "Done. ingested=%d  skipped=%d  errors=%d  store_total=%d",
        ingested, skipped, errors, store.n_episodes,
    )
    ledger.close()


if __name__ == "__main__":
    main()
