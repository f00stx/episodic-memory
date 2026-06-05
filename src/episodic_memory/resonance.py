"""
MemoryResonanceModule -- the fast amygdala pathway.

Analogy: hearing a familiar song fragment and immediately feeling a mood shift
before any specific memory surfaces. The emotional colouring happens reflexively,
driven by pattern-match on the affective signature, not by conscious recall.

Mechanism:
  1. QueryProjector maps current context_emb → query_vector in latent space
     (< 1ms, pure MLP -- no conversation re-encoding needed)
  2. Hot-tier query: cosine similarity against all stored latents
     (< 5ms for 100K memories on CPU, pure numpy)
  3. Top-k emotional signatures weighted by similarity → blended resonance vector
  4. Resonance vector is injected into CognitiveState as a continuous bias
     -- it "colours" future processing without surfacing explicit memory content

Thresholds:
  resonance_threshold: minimum similarity to include a memory in the blend.
      Below this, the memory is effectively "not felt" -- too dissimilar.
  recall_threshold: minimum similarity to trigger the slow hippocampal path.
      Only memories that strongly match cause explicit recall (RecallResult).

Tuning guidance:
  resonance_threshold=0.3 -- loose, picks up weak thematic similarity.
      Good for ambient mood colouring across many conversations.
  resonance_threshold=0.5 -- moderate, only well-matched memories contribute.
  recall_threshold=0.7    -- tight, only very similar episodes trigger recall.
      Avoid setting this too low or the slow path fires too often.
"""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from episodic_memory.schemas import (
    ResonanceResult,
    EMO_SIG_DIM,
    LATENT_DIM,
)
from episodic_memory.roleplay_filter import RoleplayFilter
from episodic_memory.contradiction import ContradictionDetector

if TYPE_CHECKING:
    from episodic_memory.store import EpisodicMemoryStore
    from episodic_memory.encoder import QueryProjector


class MemoryResonanceModule:
    """
    Fast amygdala-style affective resonance from episodic memory.

    Wraps an EpisodicMemoryStore and a QueryProjector to provide a single
    query() call that returns a ResonanceResult -- the blended emotional
    signature of the most relevant past memories.

    This module is stateless apart from references to the store and projector.
    No gradients flow through it at inference time.

    Usage::

        resonance_module = MemoryResonanceModule(
            store=store,
            query_projector=query_proj,
            resonance_threshold=0.35,
            recall_threshold=0.70,
            top_k=7,
        )

        # Called every turn with current context_emb
        result = resonance_module.query(current_context_emb)

        if result.has_resonance:
            # blend result.resonance_vector into cognitive state
            state.resonance_blend = result.resonance_vector

        if result.triggered_recall:
            # hand off to EpisodicRecall for the slow path
            recall = episodic_recall.recall(result.top_k_ids[0])

    Args:
        store:               EpisodicMemoryStore to query.
        query_projector:     Trained QueryProjector (maps context_emb → L-space).
        resonance_threshold: minimum cosine similarity to include in blend (0-1).
        recall_threshold:    minimum cosine similarity to trigger slow recall (0-1).
        top_k:               max memories to blend into resonance_vector.
        device:              torch device for QueryProjector inference.
    """

    def __init__(
        self,
        store:               "EpisodicMemoryStore",
        query_projector:     "QueryProjector",
        resonance_threshold: float = 0.35,
        recall_threshold:    float = 0.70,
        top_k:               int   = 7,
        device:              str   = "cpu",
    ) -> None:
        self.store               = store
        self.query_projector     = query_projector
        self.resonance_threshold = resonance_threshold
        self.recall_threshold    = recall_threshold
        self.top_k               = top_k
        self.device              = device

    def query(
        self,
        context_emb: np.ndarray,           # (context_emb_dim,) from CognitiveState
        exclude_session_id: Optional[str] = None,
    ) -> ResonanceResult:
        """
        Run the fast resonance query.

        1. Project context_emb → query_vector (QueryProjector MLP)
        2. Cosine similarity query against hot tier
        3. Filter by resonance_threshold
        4. Blend top-k emotional signatures (similarity-weighted)

        Args:
            context_emb:        (context_emb_dim,) current context embedding.
            exclude_session_id: session_id to exclude (e.g. current conversation).

        Returns:
            ResonanceResult with blended vector, top-k IDs, and recall flag.
        """
        if self.store.n_episodes == 0:
            return ResonanceResult.null()

        # ── Step 1: Project context_emb → query vector ─────────────────────────
        query_vec = self.query_projector.project_numpy(
            context_emb, device=self.device
        )  # (latent_dim,) L2-normalised

        # ── Step 2: Hot-tier similarity search ─────────────────────────────────
        raw_results = self.store.query(
            query_vector   = query_vec,
            top_k          = self.top_k + (1 if exclude_session_id else 0),
            min_similarity = self.resonance_threshold,
        )  # list of (session_id, similarity, metadata)

        # Filter out current session if requested
        if exclude_session_id:
            raw_results = [
                r for r in raw_results if r[0] != exclude_session_id
            ]

        # Trim back to top_k after exclusion
        raw_results = raw_results[:self.top_k]

        if not raw_results:
            return ResonanceResult.null()

        # ── Step 3: Extract emotional signatures ───────────────────────────────
        top_ids  = [r[0] for r in raw_results]
        top_sims = [r[1] for r in raw_results]

        emo_sigs = self.store.get_emotional_signatures(top_ids)  # (k, EMO_SIG_DIM)

        # ── Step 4: Similarity-weighted blend ──────────────────────────────────
        sims_arr = np.array(top_sims, dtype=np.float32)  # (k,)
        # Softmax-weighted to prevent a single strong match from dominating
        weights  = self._softmax(sims_arr)                 # (k,)
        blended  = (emo_sigs * weights[:, np.newaxis]).sum(axis=0)  # (EMO_SIG_DIM,)

        # Re-normalise blended vector so magnitude reflects resonance_strength
        blended_norm = np.linalg.norm(blended)
        if blended_norm > 1e-9:
            # Scale magnitude by mean similarity (strength) but keep direction
            resonance_strength = float(weights @ sims_arr)
            blended = (blended / blended_norm) * resonance_strength

        max_sim = float(top_sims[0]) if top_sims else 0.0

        return ResonanceResult(
            resonance_vector   = blended.astype(np.float32),
            max_similarity     = max_sim,
            top_k_ids          = top_ids,
            top_k_similarities = top_sims,
            triggered_recall   = max_sim >= self.recall_threshold,
        )

    def query_raw(
        self,
        query_vec: np.ndarray,              # (latent_dim,) -- pre-projected
        exclude_session_id: Optional[str] = None,
    ) -> ResonanceResult:
        """
        Run the resonance query with a pre-projected query vector.

        Use this when you already have a latent vector (e.g. from EpisodicEncoder
        at conversation end) and don't want to go through the QueryProjector.

        Args:
            query_vec:          (latent_dim,) L2-normalised vector.
            exclude_session_id: session_id to exclude.

        Returns:
            ResonanceResult.
        """
        if self.store.n_episodes == 0:
            return ResonanceResult.null()

        raw_results = self.store.query(
            query_vector   = query_vec,
            top_k          = self.top_k + (1 if exclude_session_id else 0),
            min_similarity = self.resonance_threshold,
        )

        if exclude_session_id:
            raw_results = [r for r in raw_results if r[0] != exclude_session_id]
        raw_results = raw_results[:self.top_k]

        if not raw_results:
            return ResonanceResult.null()

        top_ids  = [r[0] for r in raw_results]
        top_sims = [r[1] for r in raw_results]
        emo_sigs = self.store.get_emotional_signatures(top_ids)

        sims_arr = np.array(top_sims, dtype=np.float32)
        weights  = self._softmax(sims_arr)
        blended  = (emo_sigs * weights[:, np.newaxis]).sum(axis=0)

        blended_norm = np.linalg.norm(blended)
        if blended_norm > 1e-9:
            resonance_strength = float(weights @ sims_arr)
            blended = (blended / blended_norm) * resonance_strength

        max_sim = float(top_sims[0]) if top_sims else 0.0

        return ResonanceResult(
            resonance_vector   = blended.astype(np.float32),
            max_similarity     = max_sim,
            top_k_ids          = top_ids,
            top_k_similarities = top_sims,
            triggered_recall   = max_sim >= self.recall_threshold,
        )

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e = np.exp(x - x.max())
        return e / e.sum()

    def __repr__(self) -> str:
        return (
            f"MemoryResonanceModule("
            f"n_episodes={self.store.n_episodes}, "
            f"resonance_threshold={self.resonance_threshold}, "
            f"recall_threshold={self.recall_threshold}, "
            f"top_k={self.top_k})"
        )


class DirectTextResonance:
    """
    BGE semantic search fallback for episodic recall.

    Bypasses the broken QueryProjector path entirely.  Instead of projecting
    a 96-dim context_emb into latent space, this embeds the *current utterance*
    directly with BGE and cosine-searches the pre-computed summary embeddings
    stored in episodes.db.

    Why this works when QueryProjector doesn't:
      - QueryProjector was trained on EpisodicEncoder's own context_emb outputs,
        which come from a full multi-turn transformer pass.  At inference we only
        have a single-utterance 96-dim BGE→reshape vector -- completely different
        subspace.  Max cosine sim plateaus at ~0.27 regardless of input.
      - BGE embeddings of the *utterance text* vs BGE embeddings of the *summary
        text* share the same semantic space.  Cosine similarities of 0.60-0.70
        are achievable for genuinely relevant queries.

    The emotional blend path (resonance_vector) is still powered by the hot-tier
    metadata -- we look up session_ids from direct search, then pull emotion_cats
    from hot_metadata.json.
    """

    def __init__(
        self,
        db_path:           str,
        hot_metadata_path: str,
        embedding_client,                  # EmbeddingClient instance
        recall_threshold:  float = 0.55,   # lower than QP path -- BGE sims are higher
        resonance_threshold: float = 0.45,
        top_k:             int   = 5,
        filter_roleplay:   bool  = True,   # exclude intimate/roleplay episodes from factual recall
    ) -> None:
        import json, sqlite3, os

        self.embedding_client    = embedding_client
        self.recall_threshold    = recall_threshold
        self.resonance_threshold = resonance_threshold
        self.top_k               = top_k
        self._roleplay_filter    = RoleplayFilter() if filter_roleplay else None

        # Load summaries and technical indexes from cold store
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT session_id, summary, dominant_emotion, dominant_archetype, stored_at, "
            "COALESCE(tags, '[]') as tags, expires_at, technical_index "
            "FROM episodes WHERE summary IS NOT NULL AND summary != ''"
        ).fetchall()
        conn.close()

        self._session_ids       = [r[0] for r in rows]
        self._summaries         = [r[1] for r in rows]
        self._dom_emotions      = [r[2] for r in rows]
        self._dom_archetypes    = [r[3] for r in rows]
        self._stored_ats        = [float(r[4]) if r[4] else 0.0 for r in rows]
        self._tags              = [json.loads(r[5]) if r[5] else [] for r in rows]
        self._expires_ats       = [float(r[6]) if r[6] is not None else None for r in rows]
        self._technical_indexes = [r[7] or "" for r in rows]

        # Blended search texts: summary + technical index when available.
        # Used for embedding only -- roleplay filter and contradiction detection
        # still operate on affective summaries alone.
        self._search_texts = [
            (s + "\n\n" + t).strip() if t else s
            for s, t in zip(self._summaries, self._technical_indexes)
        ]

        # Project field from hot_metadata -- used for project-scope filtering.
        # Sessions not found in hot_meta get "" (invisible to scoped agents).
        self._projects = [
            self._hot_meta.get(sid, {}).get("project", "")
            for sid in self._session_ids
        ]

        # Load hot metadata for emotion_cats lookup
        with open(hot_metadata_path) as f:
            hot_meta_list = json.load(f)
        self._hot_meta = {m["session_id"]: m for m in hot_meta_list}

        # Pre-compute summary embeddings (1536-dim, unit-normed) -- batched for speed
        # Cache to disk keyed by MD5 of session IDs -- avoids recompute on restart
        import hashlib, logging
        log = logging.getLogger(__name__)

        cache_dir  = os.path.join(os.path.dirname(db_path), "embed_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_key  = hashlib.md5("".join(self._session_ids).encode()).hexdigest()
        # blended_embs: summary + technical_index combined text (separate from legacy summary_embs)
        cache_path = os.path.join(cache_dir, f"blended_embs_{cache_key}.npy")
        db_mtime   = os.path.getmtime(db_path)

        if (
            os.path.exists(cache_path)
            and os.path.getmtime(cache_path) >= db_mtime
        ):
            log.info(
                "DirectTextResonance: loading %d cached blended embeddings from disk...",
                len(self._search_texts),
            )
            self._summary_embs = np.load(cache_path)
        else:
            n_blended = sum(1 for t in self._technical_indexes if t)
            log.info(
                "DirectTextResonance: pre-computing %d blended embeddings (%d with technical index)...",
                len(self._search_texts), n_blended,
            )
            raw_embs = self.embedding_client.embed(self._search_texts)  # (N, dim)
            norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
            self._summary_embs = (raw_embs / (norms + 1e-8)).astype(np.float32)
            np.save(cache_path, self._summary_embs)
            log.info(
                "DirectTextResonance: blended embeddings cached -> %s", cache_path,
            )
        log.info("DirectTextResonance: ready (%d episodes indexed)", len(self._session_ids))

        # Build ContradictionDetector over the same embeddings (no extra embedding cost)
        self._contradiction_detector = ContradictionDetector(
            session_ids            = self._session_ids,
            summaries              = self._summaries,
            stored_ats             = self._stored_ats,
            summary_embs           = self._summary_embs,
            supersession_threshold = 0.75,
            min_age_gap_days       = 1.0,
        )

    @property
    def n_episodes(self) -> int:
        return len(self._session_ids)

    def query(
        self,
        utterance_text:     str,
        exclude_session_id: str | None = None,
        exclude_tags:       list[str] | None = None,
        only_tags:          list[str] | None = None,
        include_expired:    bool = False,
        project_scope:      str | None = None,
    ) -> "ResonanceResult":
        """
        Search episode summaries for semantic matches to *utterance_text*.

        Returns a ResonanceResult populated from the hot-tier metadata of
        matching episodes.  triggered_recall=True when max_sim >= recall_threshold.

        Args:
            utterance_text:     Text to search for.
            exclude_session_id: Session ID to exclude from results.
            exclude_tags:       Exclude episodes that have ANY of these tags.
                                e.g. exclude_tags=["roleplay"] filters intimate episodes.
            only_tags:          Only include episodes that have ALL of these tags.
                                e.g. only_tags=["hardware"] to search hardware notes.
            include_expired:    If False (default), skip episodes whose expires_at
                                is in the past. Set True to search all episodes.
        """
        if self.n_episodes == 0:
            return ResonanceResult.null()

        # Embed current utterance
        qe = self.embedding_client.embed_one(utterance_text)
        qnorm = np.linalg.norm(qe)
        if qnorm < 1e-8:
            return ResonanceResult.null()
        qe = qe / qnorm  # (1536,)

        # Cosine similarity against all summaries
        sims = self._summary_embs @ qe  # (N,)

        # Exclude current session
        if exclude_session_id:
            for i, sid in enumerate(self._session_ids):
                if sid == exclude_session_id:
                    sims[i] = -1.0

        # Exclude expired episodes (unless include_expired=True)
        if not include_expired:
            now = time.time()
            for i, exp in enumerate(self._expires_ats):
                if exp is not None and now > exp:
                    sims[i] = -1.0

        # Tag filtering
        if exclude_tags:
            exclude_set = set(exclude_tags)
            for i, ep_tags in enumerate(self._tags):
                if exclude_set.intersection(ep_tags):
                    sims[i] = -1.0

        if only_tags:
            only_set = set(only_tags)
            for i, ep_tags in enumerate(self._tags):
                if not only_set.issubset(set(ep_tags)):
                    sims[i] = -1.0

        if project_scope:
            for i, proj in enumerate(self._projects):
                if proj != project_scope:
                    sims[i] = -1.0

        top_idx = np.argsort(sims)[::-1][: self.top_k * 4]   # over-fetch to survive filtering
        # Filter by resonance_threshold
        top_idx = [i for i in top_idx if sims[i] >= self.resonance_threshold]

        # Roleplay filter: remove intimate/roleplay episodes from factual recall.
        # We fetch 4xtop_k candidates above so there's headroom after filtering.
        if self._roleplay_filter is not None:
            top_idx = [
                i for i in top_idx
                if not self._roleplay_filter.is_roleplay(self._summaries[i])
            ]

        top_idx = top_idx[: self.top_k]

        if not top_idx:
            return ResonanceResult.null()

        top_ids  = [self._session_ids[i] for i in top_idx]
        top_sims = [float(sims[i]) for i in top_idx]

        # Build emotional blend from hot_metadata
        emo_sigs = []
        for sid in top_ids:
            meta = self._hot_meta.get(sid)
            if meta and "emotion_cats" in meta:
                emo_sigs.append(np.array(meta["emotion_cats"], dtype=np.float32))
            else:
                # Use length of first entry, or 8 (standard Plutchik)
                _emo_dim = len(self._hot_meta[self._session_ids[0]]["emotion_cats"]) \
                    if self._hot_meta else 8
                emo_sigs.append(np.zeros(_emo_dim, dtype=np.float32))

        emo_sigs = np.stack(emo_sigs, axis=0)  # (k, EMO_SIG_DIM)
        sims_arr = np.array(top_sims, dtype=np.float32)
        weights  = self._softmax(sims_arr)
        blended  = (emo_sigs * weights[:, np.newaxis]).sum(axis=0)

        blended_norm = np.linalg.norm(blended)
        if blended_norm > 1e-9:
            resonance_strength = float(weights @ sims_arr)
            blended = (blended / blended_norm) * resonance_strength

        max_sim = top_sims[0]

        # Zero-pad blended to EMO_SIG_DIM so ResonancePrimer's warmth_vector
        # (initialized to EMO_SIG_DIM=64) can broadcast correctly.
        if len(blended) < EMO_SIG_DIM:
            padded = np.zeros(EMO_SIG_DIM, dtype=np.float32)
            padded[:len(blended)] = blended
            blended = padded

        return ResonanceResult(
            resonance_vector   = blended.astype(np.float32),
            max_similarity     = max_sim,
            top_k_ids          = top_ids,
            top_k_similarities = top_sims,
            triggered_recall   = max_sim >= self.recall_threshold,
        )

    def get_summary(self, session_id: str) -> str | None:
        """Return the LLM-generated summary for a session, or None."""
        try:
            idx = self._session_ids.index(session_id)
            return self._summaries[idx]
        except ValueError:
            return None

    def check_supersession(self, session_id: str) -> "SupersessionResult":
        """
        Check whether *session_id* has been superseded by a newer episode
        on the same topic.

        Returns a SupersessionResult -- is_superseded=False if the episode
        is current, is_superseded=True if a newer episode covers the same
        topic at similarity >= supersession_threshold (default 0.75).

        Delegates to the ContradictionDetector built at init time -- no
        extra embedding work.
        """
        from episodic_memory.contradiction import SupersessionResult
        return self._contradiction_detector.check(session_id)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()
