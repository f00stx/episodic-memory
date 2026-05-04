"""
RecallEngine -- clean standalone interface to the episodic memory layer.

This is the publishable API surface. All CTM-internal coupling lives in the
modules *below* this; RecallEngine itself has zero dependency on CTMSession,
EMA tensors, or any other CTM-specific concept.

Usage (standalone)::

    from episodic_memory.recall_engine import RecallEngine

    engine = RecallEngine("/path/to/memory/store")
    result = engine.query("V100 SXM2 NVLink installation")
    if result:
        print(result.summary)          # LLM-generated gist of the recalled episode
        print(result.similarity)       # cosine similarity (0-1)
        print(result.dominant_emotion) # e.g. "joy"
        print(result.context_injection())  # formatted block for system-prompt injection

Architecture:

    ┌─────────────────────────────────────────┐
    │              RecallEngine               │  ← this module (public API)
    ├──────────────┬──────────────────────────┤
    │  Resonance   │  DirectTextResonance     │  ← fast BGE cosine search
    │  (fast path) │  + RoleplayFilter        │     filters roleplay, <5ms
    ├──────────────┼──────────────────────────┤
    │  Recall      │  EpisodicRecall          │  ← slow SQLite lookup + summary
    │  (slow path) │                          │     triggered only on high sim
    ├──────────────┴──────────────────────────┤
    │              EpisodicMemoryStore        │  ← hot JSON + cold SQLite
    └─────────────────────────────────────────┘

Two-tier architecture mirrors the neuroscience:
    Fast path  = amygdala-style affective resonance (sub-5ms)
    Slow path  = hippocampal episodic recall (100-500ms)

query() runs the fast path always. The slow path fires automatically when
max_similarity >= recall_threshold (default 0.55). Set recall_threshold=1.0
to disable slow path entirely (resonance-only mode).

Thread safety: RecallEngine is read-only at query time. Safe for concurrent
use across threads once constructed (BGE embedding is the only shared state,
and sentence-transformers is thread-safe for inference).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class RecallEngine:
    """
    Unified episodic memory query interface.

    Parameters
    ----------
    store_path:
        Path to the memory store directory. Must contain:
          - ``episodes.db``        (SQLite cold tier)
          - ``hot_metadata.json``  (JSON hot tier)
    recall_threshold:
        Minimum cosine similarity to trigger slow-path recall and return a
        ``RecallResult``.  Below this, ``query()`` returns ``None``.
        Default 0.55 is calibrated for BGE-large-en-v1.5 on Aura session data.
    resonance_threshold:
        Minimum similarity for a candidate to contribute to the emotional
        resonance blend (fast path).  Default 0.45.
    top_k:
        Number of candidate episodes to consider in the blend (fast path).
        The best-matching *non-roleplay* factual episode is used for slow-path
        recall (always the top result after roleplay filtering).
    filter_roleplay:
        If True (default), intimate/roleplay episodes are excluded from factual
        recall.  The ``RoleplayFilter`` is applied after cosine ranking, so
        roleplay episodes still contribute to the emotional resonance blend
        (they *feel* relevant) but their content is never injected as context.
        Set to False to disable filtering (e.g. when querying from a roleplay
        session context).
    embedding_device:
        Device for BGE embeddings.  Default "cpu" -- avoids competing with the
        main LLM for GPU VRAM.
    """

    def __init__(
        self,
        store_path:          str | Path,
        recall_threshold:    float = 0.55,
        resonance_threshold: float = 0.45,
        top_k:               int   = 5,
        filter_roleplay:     bool  = True,
        embedding_device:    str   = "cpu",
        embedding_model:     str   = "BAAI/bge-large-en-v1.5",
    ) -> None:
        self._store_path = Path(store_path).expanduser()
        self._recall_threshold    = recall_threshold
        self._resonance_threshold = resonance_threshold
        self._top_k               = top_k
        self._filter_roleplay     = filter_roleplay
        self._embedding_device    = embedding_device
        self._embedding_model     = embedding_model

        self._resonance: Optional["DirectTextResonance"] = None  # lazy-loaded
        self._recall_mod: Optional["EpisodicRecall"]     = None  # lazy-loaded
        self._store: Optional["EpisodicMemoryStore"]     = None  # lazy-loaded
        self._embed_client                               = None  # lazy-loaded
        self._embedding_device                           = embedding_device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        text:               str,
        exclude_session_id: Optional[str] = None,
    ) -> Optional["RecallResult"]:
        """
        Query episodic memory for episodes relevant to *text*.

        Fast path (always runs):
            BGE cosine search across all episode summaries.  Roleplay episodes
            are filtered out.  Returns ``None`` immediately if nothing exceeds
            ``recall_threshold``.

        Slow path (runs only when fast path triggers):
            Fetches the full ``RecallResult`` including the LLM-generated
            natural-language summary, suitable for ``context_injection()``.

        Parameters
        ----------
        text:
            The current utterance or context to search against.
        exclude_session_id:
            Session ID to exclude (e.g. the current live session).

        Returns
        -------
        ``RecallResult`` if a relevant episode is found, ``None`` otherwise.
        """
        resonance = self._get_resonance()
        recall_mod = self._get_recall()

        res = resonance.query(text, exclude_session_id=exclude_session_id)

        if not res.triggered_recall or not res.top_k_ids:
            return None

        top_session_id = res.top_k_ids[0]
        top_sim        = res.top_k_similarities[0]

        recall_result = recall_mod.recall(
            session_id=top_session_id,
            similarity=top_sim,
        )

        # Supersession check -- annotate RecallResult with staleness info
        if recall_result is not None:
            sup = self._get_resonance().check_supersession(top_session_id)
            if sup.is_superseded:
                recall_result.is_superseded            = True
                recall_result.superseded_by            = sup.superseded_by
                recall_result.superseded_by_summary    = sup.superseded_by_summary
                recall_result.supersession_age_gap_str = sup.age_gap_str
                logger.debug(
                    "RecallEngine: episode %r flagged as superseded by %r (+%s)",
                    top_session_id, sup.superseded_by, sup.age_gap_str,
                )

        return recall_result

    def query_resonance(
        self,
        text:               str,
        exclude_session_id: Optional[str] = None,
    ) -> "ResonanceResult":
        """
        Fast path only -- returns the raw ``ResonanceResult`` without triggering
        slow-path recall.  Useful for injecting emotional colouring into state
        without the latency of a SQLite + LLM call.
        """
        return self._get_resonance().query(text, exclude_session_id=exclude_session_id)

    @property
    def n_episodes(self) -> int:
        """Number of episodes currently indexed."""
        return self._get_resonance().n_episodes

    @property
    def store_path(self) -> Path:
        return self._store_path

    # ------------------------------------------------------------------
    # Lazy initialisation -- keeps __init__ fast, defers BGE model load
    # ------------------------------------------------------------------

    def _get_embed_client(self):
        """Return a lightweight embed wrapper backed by SentenceTransformer.

        In the standalone library we use SentenceTransformer directly rather
        than the CTM-specific EmbeddingClient.  The object returned exposes a
        single method ``embed(texts) -> np.ndarray`` compatible with the rest
        of the recall pipeline.
        """
        if self._embed_client is None:
            from sentence_transformers import SentenceTransformer

            _model_name = getattr(self, "_embedding_model",
                                  "BAAI/bge-large-en-v1.5")
            _device = getattr(self, "_embedding_device", "cpu")
            _st = SentenceTransformer(_model_name, device=_device)

            class _STClient:
                def __init__(self, st):
                    self._st = st

                def embed(self, texts):
                    import numpy as np
                    vecs = self._st.encode(
                        texts,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )
                    return np.array(vecs, dtype=np.float32)

            self._embed_client = _STClient(_st)
        return self._embed_client

    def _get_resonance(self) -> "DirectTextResonance":
        if self._resonance is None:
            from episodic_memory.resonance import DirectTextResonance
            db   = self._store_path / "episodes.db"
            hot  = self._store_path / "hot_metadata.json"
            if not db.exists():
                raise FileNotFoundError(
                    f"episodes.db not found at {db}. "
                    "Run encode_aura_memories.py to build the memory store."
                )
            if not hot.exists():
                raise FileNotFoundError(
                    f"hot_metadata.json not found at {hot}."
                )
            self._resonance = DirectTextResonance(
                db_path             = str(db),
                hot_metadata_path   = str(hot),
                embedding_client    = self._get_embed_client(),
                recall_threshold    = self._recall_threshold,
                resonance_threshold = self._resonance_threshold,
                top_k               = self._top_k,
                filter_roleplay     = self._filter_roleplay,
            )
        return self._resonance

    def _get_store(self) -> "EpisodicMemoryStore":
        if self._store is None:
            from episodic_memory.store import EpisodicMemoryStore
            self._store = EpisodicMemoryStore(self._store_path)
        return self._store

    def _get_recall(self) -> "EpisodicRecall":
        if self._recall_mod is None:
            from episodic_memory.recall import EpisodicRecall
            self._recall_mod = EpisodicRecall(store=self._get_store())
        return self._recall_mod

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        loaded = self._resonance is not None
        n = self._resonance.n_episodes if loaded else "?"
        return (
            f"RecallEngine(store={self._store_path}, "
            f"recall_threshold={self._recall_threshold}, "
            f"episodes={n}, "
            f"loaded={loaded})"
        )
