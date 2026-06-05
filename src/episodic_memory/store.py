"""
EpisodicMemoryStore -- two-tier episodic memory storage.

HOT TIER (fast, always in RAM):
    numpy matrix of L2-normalised latent vectors (L, 256-dim).
    Cosine similarity query = matrix-vector multiply → sub-millisecond.
    Persisted as a pair of files:
        <store_path>/hot_latents.npy      -- float32 (N, latent_dim)
        <store_path>/hot_metadata.json    -- list of metadata dicts

COLD TIER (slow, on-demand):
    SQLite database at <store_path>/episodes.db
    Schema:
        episodes(session_id TEXT PK,
                 transcript  TEXT,   -- JSON list of {role, content} dicts
                 summary     TEXT,   -- LLM-generated gist (may be empty)
                 stored_at   REAL)

Design notes:
  - No FAISS dependency for v1. numpy dot-product is fast enough up to ~100K
    memories (< 5ms at 100K on CPU). FAISS can be swapped in later by
    replacing _query_hot_tier() with a faiss.IndexFlatIP lookup.
  - store.add() is atomic at the hot-tier level: we append to the in-memory
    array first, then persist to disk. A crash between the numpy write and
    the SQLite write is recoverable by re-encoding the raw transcript.
  - session_id is the canonical key across both tiers.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np

from episodic_memory.schemas import (
    EpisodicMemory,
    LATENT_DIM,
    EMO_SIG_DIM,
    ARCH_SIG_DIM,
)

# Optional graph client -- import lazily to avoid hard dependency
try:
    from episodic_memory.graph import MemgraphClient as _MemgraphClient
except ImportError:
    _MemgraphClient = None  # type: ignore


def _emit_store_event(event: dict) -> None:
    """Append to <store_path>/mcp_events.jsonl if reachable. Never raises."""
    event.setdefault("ts", time.time())
    log_path = os.environ.get("EPISODIC_MEMORY_STORE_PATH", "")
    if not log_path:
        return
    try:
        path = Path(log_path).expanduser() / "mcp_events.jsonl"
        with open(path, "a") as fh:
            fh.write(json.dumps(event) + "\n")
    except Exception:
        pass


class EpisodicMemoryStore:
    """
    Persistent two-tier episodic memory store.

    Hot tier: numpy latent matrix + JSON metadata (always in RAM, fast queries).
    Cold tier: SQLite with full conversation transcripts (fetched on demand).

    Usage::

        store = EpisodicMemoryStore(Path("~/.ctm/memory"))

        # Store a new episode
        store.add(
            session_id="conv_123",
            latent=encoder.encode_numpy(user_embs, agent_embs),
            transcript=[{"role": "user", "content": "..."}, ...],
            metadata={"dominant_emotion": "joy", "turn_count": 12},
        )

        # Query by similarity (fast -- hot tier only)
        results = store.query(query_vector, top_k=5)
        # results: list of (session_id, similarity, metadata)

        # Fetch full transcript (slow -- cold tier)
        transcript = store.fetch_transcript("conv_123")

    Thread safety: not thread-safe. Use external locking for concurrent access.
    """

    _HOT_LATENTS_FILE  = "hot_latents.npy"
    _HOT_META_FILE     = "hot_metadata.json"
    _DB_FILE           = "episodes.db"

    def __init__(
        self,
        store_path: Path,
        latent_dim:   int = LATENT_DIM,
        graph_client: Optional[object] = None,   # MemgraphClient, if available
    ) -> None:
        self.store_path = Path(store_path).expanduser()
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.latent_dim   = latent_dim
        self._graph       = graph_client          # None → graph writes silently skipped

        # Hot tier (in-memory)
        self._hot_latents:  np.ndarray  = np.zeros((0, latent_dim), dtype=np.float32)
        self._hot_metadata: list[dict]  = []
        # Index: session_id → row index for O(1) lookup
        self._session_index: dict[str, int] = {}

        # Initialise both tiers
        self._init_db()
        self._load_hot_tier()

    # ── Public API ─────────────────────────────────────────────────────────────

    def add(
        self,
        session_id:  str,
        latent:      np.ndarray,        # (latent_dim,) float32 -- should be L2-normalised
        transcript:  list[dict],        # [{role, content}, ...] conversation turns
        metadata:    Optional[dict] = None,
        summary:     str = "",
    ) -> EpisodicMemory:
        """
        Store a new episodic memory.

        If session_id already exists, the existing record is overwritten
        (both hot and cold tier).

        Args:
            session_id: unique identifier for this conversation.
            latent:     L2-normalised (latent_dim,) vector from EpisodicEncoder.
            transcript: list of {role: str, content: str} turn dicts.
            metadata:   optional dict -- stored as JSON alongside the latent.
                        Recommended keys: dominant_emotion, dominant_archetype,
                        turn_count, stored_at.
            summary:    LLM-generated gist (can be filled in later via update_summary).

        Returns:
            EpisodicMemory dataclass representing the stored record.
        """
        meta = dict(metadata or {})
        meta.setdefault("dominant_emotion",   "neutral")
        meta.setdefault("dominant_archetype", "sage")
        meta.setdefault("turn_count",         len(transcript))
        meta.setdefault("stored_at",          time.time())
        meta["session_id"] = session_id

        # Auto-tag if not already provided -- uses summary if available
        if "tags" not in meta:
            try:
                from episodic_memory.tagger import EpisodicTagger
                _tagger = EpisodicTagger()
                tag_result = _tagger.tag(
                    summary=summary or "",
                    stored_at=meta["stored_at"],
                    metadata=meta,
                )
                meta["tags"]       = tag_result.tags
                meta["expires_at"] = tag_result.expires_at
            except Exception:
                meta["tags"]       = []
                meta["expires_at"] = None

        latent_f32 = np.asarray(latent, dtype=np.float32)
        if latent_f32.shape != (self.latent_dim,):
            raise ValueError(
                f"latent must be shape ({self.latent_dim},), got {latent_f32.shape}"
            )

        # Warn if not L2-normalised (don't hard-fail -- caller may have good reason)
        norm = float(np.linalg.norm(latent_f32))
        if not (0.99 < norm < 1.01):
            import warnings
            warnings.warn(
                f"latent norm={norm:.4f} -- expected L2-normalised vector (norm≈1). "
                "Cosine similarity queries will still work but may be less accurate."
            )

        # Overwrite existing record if session_id exists
        if session_id in self._session_index:
            idx = self._session_index[session_id]
            self._hot_latents[idx] = latent_f32
            self._hot_metadata[idx] = meta
        else:
            self._hot_latents = np.vstack([self._hot_latents, latent_f32[np.newaxis]])
            self._hot_metadata.append(meta)
            self._session_index[session_id] = len(self._hot_metadata) - 1

        self._save_hot_tier()
        self._upsert_cold(
            session_id, transcript, summary, meta["stored_at"],
            dominant_emotion   = meta["dominant_emotion"],
            dominant_archetype = meta["dominant_archetype"],
        )

        _emit_store_event({
            "event":             "memory_stored",
            "session_id":        session_id,
            "turn_count":        meta["turn_count"],
            "dominant_emotion":  meta["dominant_emotion"],
            "project":           meta.get("project", ""),
            "total_episodes":    len(self._hot_metadata),
        })

        # ── Graph side-write (optional) ────────────────────────────────────────
        if self._graph is not None:
            try:
                self._graph.upsert_session(
                    session_id         = session_id,
                    dominant_emotion   = meta["dominant_emotion"],
                    dominant_archetype = meta["dominant_archetype"],
                    emotion_dist       = meta.get("emotion_dist",   {}),
                    archetype_dist     = meta.get("archetype_dist", {}),
                    turn_count         = meta["turn_count"],
                    stored_at          = meta["stored_at"],
                    character          = meta.get("character"),
                    topics             = meta.get("topics"),
                )
                # Chain to previous session if available
                prev_sid = meta.get("prev_session_id")
                if prev_sid:
                    self._graph.link_sessions(prev_sid, session_id)
                # Write top-k cosine similarity edges
                if len(self._hot_metadata) > 1:
                    neighbours = self.query(latent_f32, top_k=6, min_similarity=0.5)
                    # Exclude self, cap at 5
                    edges = [
                        (sid, score)
                        for sid, score, _ in neighbours
                        if sid != session_id
                    ][:5]
                    if edges:
                        self._graph.upsert_similarity_edges(session_id, edges)
            except Exception as exc:  # never let graph errors block memory storage
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "MemgraphClient.upsert_session failed (non-fatal): %s", exc
                )

        return EpisodicMemory(
            session_id         = session_id,
            latent             = latent_f32.copy(),
            stored_at          = meta["stored_at"],
            turn_count         = meta["turn_count"],
            dominant_emotion   = meta["dominant_emotion"],
            dominant_archetype = meta["dominant_archetype"],
            metadata           = {k: v for k, v in meta.items()
                                  if k not in ("session_id",)},
        )

    def query(
        self,
        query_vector:   np.ndarray,         # (latent_dim,) -- should be L2-normalised
        top_k:          int   = 5,
        min_similarity: float = 0.0,
    ) -> list[tuple[str, float, dict]]:
        """
        Fast cosine similarity search over all stored memories.

        Args:
            query_vector:   (latent_dim,) query. Should be L2-normalised so that
                            dot product == cosine similarity.
            top_k:          number of results to return.
            min_similarity: exclude results below this threshold.

        Returns:
            list of (session_id, similarity, metadata) tuples, sorted descending
            by similarity. Empty list if store is empty or no results exceed
            min_similarity.
        """
        if len(self._hot_latents) == 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)  # ensure normalised

        # Cosine similarity via dot product (L2-normalised vectors)
        similarities = self._hot_latents @ q              # (N,)

        # Get top-k indices (unsorted first for speed, then sort)
        k = min(top_k, len(similarities))
        top_idx = np.argpartition(similarities, -k)[-k:]  # (k,) unordered
        top_idx = top_idx[np.argsort(similarities[top_idx])[::-1]]  # sort desc

        results = []
        for idx in top_idx:
            sim = float(similarities[idx])
            if sim >= min_similarity:
                meta = dict(self._hot_metadata[idx])
                sid  = meta.get("session_id", "")
                results.append((sid, sim, meta))

        if results:
            _emit_store_event({
                "event":          "memory_queried",
                "n_candidates":   len(self._hot_metadata),
                "n_results":      len(results),
                "top_similarity": round(results[0][1], 4) if results else None,
                "top_k":          top_k,
            })

        return results

    def get_latent(self, session_id: str) -> Optional[np.ndarray]:
        """Return the stored latent vector for a session_id, or None."""
        if session_id not in self._session_index:
            return None
        idx = self._session_index[session_id]
        return self._hot_latents[idx].copy()

    def get_emotional_signatures(self, session_ids: list[str]) -> np.ndarray:
        """
        Return emotional sub-vectors (L[:EMO_SIG_DIM]) for a list of session_ids.

        Used by MemoryResonanceModule for fast affective blending.

        Returns:
            (len(session_ids), EMO_SIG_DIM) float32 array.
            Rows for unknown session_ids are zero vectors.
        """
        out = np.zeros((len(session_ids), EMO_SIG_DIM), dtype=np.float32)
        for i, sid in enumerate(session_ids):
            if sid in self._session_index:
                idx = self._session_index[sid]
                out[i] = self._hot_latents[idx, :EMO_SIG_DIM]
        return out

    def get_archetypal_signatures(self, session_ids: list[str]) -> np.ndarray:
        """
        Return archetypal sub-vectors (L[EMO_SIG_DIM:EMO_SIG_DIM+ARCH_SIG_DIM]).
        Returns (len(session_ids), ARCH_SIG_DIM) float32 array.
        """
        out = np.zeros((len(session_ids), ARCH_SIG_DIM), dtype=np.float32)
        for i, sid in enumerate(session_ids):
            if sid in self._session_index:
                idx = self._session_index[sid]
                out[i] = self._hot_latents[idx, EMO_SIG_DIM:EMO_SIG_DIM + ARCH_SIG_DIM]
        return out

    def fetch_transcript(self, session_id: str) -> Optional[list[dict]]:
        """
        Fetch the full conversation transcript from the cold tier (SQLite).

        Returns:
            list of {role, content} dicts, or None if session not found.
        """
        with self._db_connect() as conn:
            row = conn.execute(
                "SELECT transcript FROM episodes WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def fetch_summary(self, session_id: str) -> Optional[str]:
        """Return stored summary text, or None if not found."""
        with self._db_connect() as conn:
            row = conn.execute(
                "SELECT summary FROM episodes WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row[0] if row else None

    def update_summary(self, session_id: str, summary: str) -> bool:
        """
        Store an LLM-generated summary for a session after the fact.

        Returns True if the session exists and was updated, False otherwise.
        """
        with self._db_connect() as conn:
            cur = conn.execute(
                "UPDATE episodes SET summary = ? WHERE session_id = ?",
                (summary, session_id),
            )
            conn.commit()
        return cur.rowcount > 0

    def fetch_technical_index(self, session_id: str) -> Optional[str]:
        """Return stored technical index text, or None if not found or empty."""
        with self._db_connect() as conn:
            row = conn.execute(
                "SELECT technical_index FROM episodes WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        val = row[0] if row else None
        return val if val else None

    def update_technical_index(self, session_id: str, technical_index: str) -> bool:
        """
        Store an LLM-generated technical index for a session after the fact.

        Returns True if the session exists and was updated, False otherwise.
        """
        with self._db_connect() as conn:
            cur = conn.execute(
                "UPDATE episodes SET technical_index = ? WHERE session_id = ?",
                (technical_index, session_id),
            )
            conn.commit()
        return cur.rowcount > 0

    def remove(self, session_id: str) -> bool:
        """
        Remove a memory from both tiers.

        Returns True if the memory was found and removed.
        Note: does not compact the numpy array (marks the row as zero latent).
        Call rebuild_index() to compact after bulk removals.
        """
        if session_id not in self._session_index:
            return False

        idx = self._session_index.pop(session_id)
        # Zero out the latent so it won't match queries
        self._hot_latents[idx] = 0.0
        self._hot_metadata[idx] = {"session_id": session_id, "_removed": True}

        self._save_hot_tier()

        with self._db_connect() as conn:
            conn.execute("DELETE FROM episodes WHERE session_id = ?", (session_id,))
            conn.commit()

        return True

    def rebuild_index(self) -> None:
        """
        Compact the hot tier by removing rows that have been soft-deleted.
        Rebuilds the session_index mapping. Use after bulk remove() calls.
        """
        live_mask = np.array(
            [not m.get("_removed", False) for m in self._hot_metadata],
            dtype=bool,
        )
        self._hot_latents  = self._hot_latents[live_mask]
        self._hot_metadata = [m for m in self._hot_metadata if not m.get("_removed")]
        self._session_index = {
            m["session_id"]: i for i, m in enumerate(self._hot_metadata)
        }
        self._save_hot_tier()

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def    n_episodes(self) -> int:
        """Number of stored (non-removed) episodes."""
        return sum(1 for m in self._hot_metadata if not m.get("_removed", False))

    def iterate_episodes(self):
        """Yield full episode records (with metadata and summary). """
        for sid in self.session_ids:
            meta = self._hot_metadata[self._session_index[sid]]
            transcript = self.fetch_transcript(sid)
            summary = self.fetch_summary(sid)
            yield {
                "key": sid,
                "metadata": meta,
                "transcript": transcript,
                "summary": summary or "",
            }

    @property
    def session_ids(self) -> list[str]:
        """All stored session_ids (non-removed)."""
        return [
            m["session_id"]
            for m in self._hot_metadata
            if not m.get("_removed", False)
        ]

    # ── Hot-tier persistence ───────────────────────────────────────────────────

    def _load_hot_tier(self) -> None:
        latents_path = self.store_path / self._HOT_LATENTS_FILE
        meta_path    = self.store_path / self._HOT_META_FILE

        if latents_path.exists() and meta_path.exists():
            self._hot_latents  = np.load(str(latents_path))
            with open(meta_path) as f:
                self._hot_metadata = json.load(f)
            self._session_index = {
                m["session_id"]: i
                for i, m in enumerate(self._hot_metadata)
                if "session_id" in m
            }
        else:
            self._hot_latents  = np.zeros((0, self.latent_dim), dtype=np.float32)
            self._hot_metadata = []
            self._session_index = {}

    def _save_hot_tier(self) -> None:
        np.save(str(self.store_path / self._HOT_LATENTS_FILE), self._hot_latents)
        with open(self.store_path / self._HOT_META_FILE, "w") as f:
            json.dump(self._hot_metadata, f, indent=2)

    # ── Cold-tier (SQLite) ─────────────────────────────────────────────────────

    def _db_connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.store_path / self._DB_FILE))

    def _init_db(self) -> None:
        with self._db_connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    session_id         TEXT PRIMARY KEY,
                    transcript         TEXT NOT NULL,
                    summary            TEXT NOT NULL DEFAULT '',
                    stored_at          REAL NOT NULL,
                    dominant_emotion   TEXT NOT NULL DEFAULT 'neutral',
                    dominant_archetype TEXT NOT NULL DEFAULT 'companion'
                )
            """)
            # Migration: add columns if they don't exist (for existing DBs)
            existing = {r[1] for r in conn.execute("PRAGMA table_info(episodes)")}
            if "dominant_emotion" not in existing:
                conn.execute("ALTER TABLE episodes ADD COLUMN dominant_emotion TEXT NOT NULL DEFAULT 'neutral'")
            if "dominant_archetype" not in existing:
                conn.execute("ALTER TABLE episodes ADD COLUMN dominant_archetype TEXT NOT NULL DEFAULT 'companion'")
            if "tags" not in existing:
                conn.execute("ALTER TABLE episodes ADD COLUMN tags TEXT")
            if "expires_at" not in existing:
                conn.execute("ALTER TABLE episodes ADD COLUMN expires_at REAL")
            if "technical_index" not in existing:
                conn.execute("ALTER TABLE episodes ADD COLUMN technical_index TEXT")
            conn.commit()

    def _upsert_cold(
        self,
        session_id:         str,
        transcript:         list[dict],
        summary:            str,
        stored_at:          float,
        dominant_emotion:   str = "neutral",
        dominant_archetype: str = "companion",
    ) -> None:
        with self._db_connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes (session_id, transcript, summary, stored_at,
                                      dominant_emotion, dominant_archetype)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    transcript         = excluded.transcript,
                    summary            = excluded.summary,
                    stored_at          = excluded.stored_at,
                    dominant_emotion   = excluded.dominant_emotion,
                    dominant_archetype = excluded.dominant_archetype
                """,
                (session_id, json.dumps(transcript), summary, stored_at,
                 dominant_emotion, dominant_archetype),
            )
            conn.commit()

    def __repr__(self) -> str:
        return (
            f"EpisodicMemoryStore("
            f"episodes={self.n_episodes}, "
            f"path={self.store_path})"
        )
