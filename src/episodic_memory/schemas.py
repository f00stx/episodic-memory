"""
Schemas for the CTM episodic memory system.

Two-tier architecture:

  HOT TIER  -- compact latent vectors (L, 256 dims) for fast similarity search.
              Always in memory (numpy matrix). Queried on every turn.
              Analogous to the amygdala's fast affective tagging.

  COLD TIER -- full conversation transcripts in SQLite.
              Fetched on demand when resonance exceeds recall_threshold.
              Analogous to hippocampal episodic playback.

Data flow:
  conversation ends
      │
      ├─► EpisodicEncoder(turns, terminal_state) → L  (store hot tier)
      └─► raw transcript → SQLite                      (store cold tier)

  new turn arrives
      │
      ├─► MemoryResonanceModule(current_context_emb)
      │       query hot tier → blended emotional resonance (fast, < 1ms)
      └─► if resonance > threshold:
              EpisodicRecall(top_session_id)
                  fetch cold tier → LLM summary → inject context (slow, 100-500ms)

Design decisions:
  D010 -- Latent L is L2-normalised so cosine similarity == dot product.
          This makes hot-tier queries a simple matrix-vector multiply.
  D011 -- L is 256 dims. Not explicitly partitioned during training, but
          probe losses encourage implicit structure:
            L[:64]    → emotional signature  (fast-path resonance query)
            L[64:128] → archetypal signature
            L[128:]   → contextual signature
          Empirically verifiable via t-SNE / k-means after training.
  D012 -- QueryProjector maps current context_emb (96 dims) → L-space (256 dims).
          Trained jointly with encoder. Enables mid-conversation querying without
          waiting for conversation end. Key to the "amygdala reflex" behaviour.
  D013 -- CoherenceGate sits between transformer CLS output and output projection.
          TrajectoryCoherenceScorer measures smoothness of the turn embedding
          trajectory; low coherence routes CLS output toward a learned
          confused_prior vector. Prevents word bombs and incoherent input from
          activating emotional memory clusters they shouldn't activate.
          Analogous to prefrontal modulation of amygdala response: degraded
          semantic structure → attenuated emotional cascade.
          coherence_score is stored in EpisodicMemory.metadata and feeds into
          consolidation_strength (incoherent memories consolidate weakly).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Dimension constants ────────────────────────────────────────────────────────

LATENT_DIM        = 256   # full episodic latent vector
EMO_SIG_DIM       = 64    # emotional signature sub-vector (L[:64])
ARCH_SIG_DIM      = 64    # archetypal signature sub-vector (L[64:128])
CTX_SIG_DIM       = 128   # contextual signature sub-vector (L[128:])
EMBEDDING_DIM     = 1536  # text-embedding-3-small output dim
CONTEXT_EMB_DIM   = 96    # context_emb dim in CognitiveState


# ── Encoder configuration ──────────────────────────────────────────────────────

@dataclass
class EpisodicEncoderConfig:
    """
    Hyperparameters for EpisodicEncoder.

    Architecture controls:
      input_dim   -- dim of each per-turn embedding (user or agent)
      d_model     -- transformer hidden size (projected from 2*input_dim)
      n_heads     -- attention heads (d_model must be divisible by n_heads)
      n_layers    -- number of TransformerEncoderLayer stacks
      dropout     -- applied inside transformer layers
      max_turns   -- max positional encoding slots (conversations longer than
                    this are truncated from the start, keeping the most recent)
      latent_dim  -- output dimension L

    Cognitive seed (D012):
      use_cognitive_seed  -- if True, project terminal context_emb and add
                            to pooled representation before output projection
      context_emb_dim     -- expected dim of context_emb (default 96)

    Query projector:
      query_context_dim   -- input dim for QueryProjector (context_emb dim)
      query_hidden_dim    -- hidden dim in QueryProjector MLP
    """
    # Encoder
    input_dim:          int   = EMBEDDING_DIM    # 1536
    d_model:            int   = 256
    n_heads:            int   = 4
    n_layers:           int   = 2
    dropout:            float = 0.1
    max_turns:          int   = 64
    latent_dim:         int   = LATENT_DIM       # 256

    # Cognitive seed
    use_cognitive_seed: bool  = True
    context_emb_dim:    int   = CONTEXT_EMB_DIM  # 96

    # Query projector
    query_context_dim:  int   = CONTEXT_EMB_DIM  # 96
    query_hidden_dim:   int   = 256

    # Coherence gate (D013)
    # When True, a TrajectoryCoherenceScorer + CoherenceGate are added to the
    # encoder. Low-coherence conversations produce latents pulled toward the
    # learned confused_prior rather than the semantic content of their turns.
    # This prevents word bombs and incoherent input from polluting emotional
    # memory clusters.
    use_coherence_gate:    bool  = True
    coherence_temperature: float = 1.0  # lower = sharper coherence boundary

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.latent_dim != LATENT_DIM:
            import warnings
            warnings.warn(
                f"latent_dim={self.latent_dim} differs from default {LATENT_DIM}. "
                "The EMO/ARCH/CTX signature slicing constants will not apply."
            )


# ── Episodic memory record ─────────────────────────────────────────────────────

@dataclass
class EpisodicMemory:
    """
    A single encoded episodic memory -- what the hot tier stores per conversation.

    latent: L2-normalised vector of shape (latent_dim,).
        The full encoding. Sliced views give interpretable sub-signatures
        (after training -- the probe losses encourage this organisation).

    emotional_signature: latent[:EMO_SIG_DIM]  (64,)
        Used by MemoryResonanceModule for fast affective blending.

    archetypal_signature: latent[EMO_SIG_DIM:EMO_SIG_DIM+ARCH_SIG_DIM]  (64,)
        Relational / behavioural mode signature.

    stored_at: unix timestamp (float).
    turn_count: how many turns the conversation had.
    dominant_emotion: highest-intensity emotion at conversation end.
    dominant_archetype: highest-intensity archetype at conversation end.
    metadata: arbitrary dict for downstream use (tags, user_id, etc.).

    consolidation_strength: float in [0, 1].
        How strongly consolidated this memory is -- affects how much weight it
        receives in the resonance blend relative to raw similarity.

        Phase 1 estimator (simple, good enough to start):
            consolidation_strength = clip(abs(pad_arousal), 0.0, 1.0)
        High emotional intensity at encoding time → stronger consolidation.

        Analogous to why certain childhood songs never fade: high arousal at
        original encoding + repeated exposure = near-permanent anchor.

        Phase 2 extensions (deferred):
          - Strengthening via spaced repetition (each explicit recall += delta)
          - Anchor designation: strength > 0.9 → decay_factor → 1.0 in primer
          - Time-weighted decay of consolidation_strength between sessions
    """
    session_id:              str
    latent:                  np.ndarray          # (latent_dim,) float32, L2-normalised
    stored_at:               float = field(default_factory=time.time)
    turn_count:              int   = 0
    dominant_emotion:        str   = "neutral"
    dominant_archetype:      str   = "sage"
    consolidation_strength:  float = 0.5         # [0, 1] -- estimated from PAD arousal
    metadata:                dict  = field(default_factory=dict)

    @property
    def emotional_signature(self) -> np.ndarray:
        """Fast-path affective sub-vector (64,)."""
        return self.latent[:EMO_SIG_DIM]

    @property
    def archetypal_signature(self) -> np.ndarray:
        """Relational sub-vector (64,)."""
        return self.latent[EMO_SIG_DIM:EMO_SIG_DIM + ARCH_SIG_DIM]

    @property
    def contextual_signature(self) -> np.ndarray:
        """Contextual sub-vector (128,)."""
        return self.latent[EMO_SIG_DIM + ARCH_SIG_DIM:]

    def to_dict(self) -> dict:
        """Serialise to JSON-compatible dict (latent as list)."""
        return {
            "session_id":             self.session_id,
            "latent":                 self.latent.tolist(),
            "stored_at":              self.stored_at,
            "turn_count":             self.turn_count,
            "dominant_emotion":       self.dominant_emotion,
            "dominant_archetype":     self.dominant_archetype,
            "consolidation_strength": self.consolidation_strength,
            "metadata":               self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodicMemory":
        return cls(
            session_id             = d["session_id"],
            latent                 = np.array(d["latent"], dtype=np.float32),
            stored_at              = d.get("stored_at", 0.0),
            turn_count             = d.get("turn_count", 0),
            dominant_emotion       = d.get("dominant_emotion", "neutral"),
            dominant_archetype     = d.get("dominant_archetype", "sage"),
            consolidation_strength = float(d.get("consolidation_strength", 0.5)),
            metadata               = d.get("metadata", {}),
        )


# ── Resonance result ───────────────────────────────────────────────────────────

@dataclass
class ResonanceResult:
    """
    Output of MemoryResonanceModule -- the fast amygdala pathway.

    resonance_vector: (EMO_SIG_DIM,) = (64,) blended emotional signature.
        Weighted average of top-k emotional signatures, weighted by similarity.
        All-zeros if no memories exceed resonance_threshold.

    max_similarity: strongest match in [0, 1].
    top_k_ids: session_ids of top-k matches (empty if none exceed threshold).
    top_k_similarities: corresponding cosine similarities.
    triggered_recall: True if max_similarity >= recall_threshold.
        The slow path (EpisodicRecall) should be triggered when this is True.

    resonance_strength: overall signal strength -- mean of top_k similarities.
        Use this to modulate how much the resonance_vector colours the state.
    """
    resonance_vector:    np.ndarray         # (EMO_SIG_DIM,) float32
    max_similarity:      float
    top_k_ids:           list[str]
    top_k_similarities:  list[float]
    triggered_recall:    bool

    @property
    def resonance_strength(self) -> float:
        """Mean similarity of matches (0 if no matches)."""
        if not self.top_k_similarities:
            return 0.0
        return float(np.mean(self.top_k_similarities))

    @property
    def has_resonance(self) -> bool:
        return len(self.top_k_ids) > 0

    @classmethod
    def null(cls) -> "ResonanceResult":
        """No-match result -- used when the store is empty or nothing crosses threshold."""
        return cls(
            resonance_vector   = np.zeros(EMO_SIG_DIM, dtype=np.float32),
            max_similarity     = 0.0,
            top_k_ids          = [],
            top_k_similarities = [],
            triggered_recall   = False,
        )


# ── Recall result ──────────────────────────────────────────────────────────────

@dataclass
class RecallResult:
    """
    Output of EpisodicRecall -- the slow hippocampal pathway.

    Triggered only when ResonanceResult.triggered_recall is True.
    Contains both the latent signatures (for state blending) and a
    natural-language summary (for context injection into the LLM prompt).

    summary: LLM-generated gist of the recalled conversation.
        Suitable for direct injection into system prompt context.

    emotional_signature: (EMO_SIG_DIM,) -- from hot tier (fast).
    archetypal_signature: (ARCH_SIG_DIM,) -- from hot tier (fast).
    similarity: cosine similarity of this memory to the current query.
    turn_count: length of the recalled conversation.
    stored_at: when this conversation was stored.
    dominant_emotion / dominant_archetype: labels for the recalled memory.
    """
    session_id:          str
    summary:             str
    emotional_signature: np.ndarray         # (EMO_SIG_DIM,)
    archetypal_signature: np.ndarray        # (ARCH_SIG_DIM,)
    similarity:          float
    turn_count:          int                = 0
    stored_at:           float              = 0.0
    dominant_emotion:    str                = "neutral"
    dominant_archetype:  str                = "sage"
    metadata:            dict               = field(default_factory=dict)
    is_superseded:       bool               = False
    superseded_by:       Optional[str]      = None          # session_id of newer episode
    superseded_by_summary: Optional[str]    = None          # summary of newer episode
    supersession_age_gap_str: Optional[str] = None          # human-readable age gap

    def context_injection(self) -> str:
        """
        Format for system prompt injection.

        Returns a grounded factual-memory block. If the episode has been
        flagged as superseded (is_superseded=True), the injection leads with
        a staleness warning and appends the newer episode's summary so the
        model can prefer the more recent picture.

        Includes explicit instruction not to confabulate details not present
        in the summary.
        """
        import datetime
        ts = datetime.datetime.fromtimestamp(self.stored_at).strftime("%Y-%m-%d")

        lines = []

        if self.is_superseded:
            age_gap = self.supersession_age_gap_str or "unknown time"
            lines.append(
                f"[POSSIBLY OUTDATED -- a newer memory ({age_gap} later) covers this "
                f"same topic. Prefer the newer context if they conflict.]"
            )

        lines.append(
            f"[Episodic memory -- {ts}  similarity={self.similarity:.2f}  "
            f"emotion={self.dominant_emotion}  archetype={self.dominant_archetype}]"
        )
        if self.summary:
            lines.append(self.summary.strip())

        if self.is_superseded and self.superseded_by_summary:
            import datetime as _dt
            # newer_stored_at is not stored on RecallResult directly, use superseded_by_summary
            lines.append("[Newer memory on same topic:]")
            lines.append(self.superseded_by_summary.strip())

        lines.append(
            "FACTUAL RECORD -- real past conversation, not fiction. "
            "Reference this accurately if relevant. "
            "Do NOT add physical descriptions, romantic elements, roleplay framing, "
            "or any detail not explicitly stated above. "
            "If the summary conflicts with the current conversation context, "
            "state the discrepancy plainly rather than blending the two."
        )
        return "\n".join(lines)

