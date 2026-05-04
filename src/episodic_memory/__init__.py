"""episodic-memory -- Standalone episodic memory system with roleplay filtering
and temporal contradiction detection.

Core components:
  RecallEngine       -- high-level API: query → RecallResult (roleplay-filtered,
                       supersession-aware). Start here.
  RoleplayFilter     -- keyword triage to exclude RP/fiction from factual recall.
  ContradictionDetector -- temporal supersession: newer episode wins on same topic.
  EpisodicMemoryStore   -- two-tier hot/cold SQLite + numpy store.
  EpisodicEncoder       -- CTM-trained encoder (optional; falls back to BGE).
"""
from episodic_memory.schemas import (
    EpisodicEncoderConfig,
    EpisodicMemory,
    ResonanceResult,
    RecallResult,
    LATENT_DIM,
    EMO_SIG_DIM,
    ARCH_SIG_DIM,
    CTX_SIG_DIM,
    EMBEDDING_DIM,
    CONTEXT_EMB_DIM,
)
from episodic_memory.encoder import EpisodicEncoder, QueryProjector
from episodic_memory.store import EpisodicMemoryStore
from episodic_memory.resonance import MemoryResonanceModule
from episodic_memory.recall import EpisodicRecall
from episodic_memory.roleplay_filter import RoleplayFilter, is_roleplay
from episodic_memory.recall_engine import RecallEngine
from episodic_memory.contradiction import ContradictionDetector, SupersessionResult

__all__ = [
    # High-level API
    "RecallEngine",
    "RecallResult",
    "RoleplayFilter",
    "is_roleplay",
    "ContradictionDetector",
    "SupersessionResult",
    # Mid-level
    "EpisodicMemoryStore",
    "EpisodicRecall",
    "MemoryResonanceModule",
    "ResonanceResult",
    # Low-level / config
    "EpisodicEncoder",
    "QueryProjector",
    "EpisodicEncoderConfig",
    "EpisodicMemory",
    "LATENT_DIM",
    "EMO_SIG_DIM",
    "ARCH_SIG_DIM",
    "CTX_SIG_DIM",
    "EMBEDDING_DIM",
    "CONTEXT_EMB_DIM",
]
