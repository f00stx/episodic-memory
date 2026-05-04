"""
ResonancePrimer -- persistent associative warmth state.

The gap in a stateless resonance module: it queries once per turn and returns,
but the smell-analogy requires a PERSISTING background state that accumulates
across turns and decays when not reinforced.

Biological model:
  - Amygdala/olfactory path: sensory signal → fast limbic response, no cortex needed
  - Signal persists as long as the stimulus is present (or recently present)
  - Multiple converging signals to same emotional cluster compound (superposition)
  - Signal decays with a half-life when un-reinforced
  - If sustained warmth exceeds a secondary threshold → involuntary recall
    (the Proustian moment: smell of madeleine → full episodic flood)

Architecture:

  Per-turn:
    1. New resonance signal arrives (from MemoryResonanceModule.query)
    2. Blend new signal into running warmth_vector with momentum (accumulation)
    3. Decay warmth_vector by decay_factor (exponential decay between turns)
    4. Compute warmth_magnitude -- scalar summary of current arousal level
    5. If warmth_magnitude >= recall_trigger_threshold AND max_sim >= recall_threshold
       → set triggered_recall flag (Proustian path)
    6. Expose warmth_vector as a continuous bias to CognitiveState

The warmth_vector lives in emotional-signature space (EMO_SIG_DIM), same as
the resonance_vector from MemoryResonanceModule. It can be directly injected
into the hybrid cognitive state as a persistent affective bias.

Threshold semantics:
  ambient_threshold:        warmth_magnitude >= this → "something feels familiar"
                            Mild mood colouring, no explicit content surfaced.
  familiarity_threshold:    warmth_magnitude >= this → "I recognise this pattern"
                            Stronger colouring; model should note the resonance.
  recall_trigger_threshold: warmth_magnitude >= this AND resonance strong
                            → involuntary recall (Proustian trigger)

Decay guidance:
  decay_factor=0.85 per turn → half-life ≈ 4-5 turns
      Good for conversational context (warmth persists over a few exchanges)
  decay_factor=0.70 per turn → half-life ≈ 2 turns
      Faster fading; better for topic-switching conversations
  decay_factor=0.95 per turn → half-life ≈ 14 turns
      Slow decay; warmth lingers for most of a conversation

Accumulation guidance:
  accumulation_rate=0.4 → new signal contributes 40%, existing warmth 60%
      Conservative blending; warmth builds slowly but surely
  accumulation_rate=0.6 → faster build-up, can spike quickly on strong signals
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from episodic_memory.schemas import ResonanceResult, EMO_SIG_DIM


@dataclass
class PrimerState:
    """
    The current warmth state of the ResonancePrimer.

    Injected into CognitiveState each turn to provide a persistent affective
    bias. The warmth_vector is in emotional-signature space -- same dims as
    ResonanceResult.resonance_vector.
    """

    # Accumulated warmth vector -- persists and decays across turns
    warmth_vector: np.ndarray = field(
        default_factory=lambda: np.zeros(EMO_SIG_DIM, dtype=np.float32)
    )

    # Scalar summary: L2 norm of warmth_vector, re-scaled to [0, 1]
    warmth_magnitude: float = 0.0

    # How many turns since warmth was last reinforced (0 = reinforced this turn)
    turns_since_reinforcement: int = 0

    # Current state label (for introspection / debugging)
    # 'cold' | 'ambient' | 'familiar' | 'primed'
    state_label: str = "cold"

    # Whether the Proustian threshold was crossed this turn
    triggered_involuntary_recall: bool = False

    # IDs of memories contributing most to current warmth (for slow-path routing)
    top_memory_ids: list[str] = field(default_factory=list)

    @property
    def is_warm(self) -> bool:
        return self.state_label != "cold"

    @property
    def is_primed(self) -> bool:
        return self.state_label == "primed"


class ResonancePrimer:
    """
    Maintains a persistent warmth state across conversational turns.

    Wraps MemoryResonanceModule results to add temporal dynamics:
    accumulation, decay, and threshold-based state transitions.

    Usage::

        primer = ResonancePrimer(
            decay_factor=0.85,
            accumulation_rate=0.4,
            ambient_threshold=0.15,
            familiarity_threshold=0.35,
            recall_trigger_threshold=0.55,
        )

        # Each turn: feed resonance result, get updated PrimerState
        resonance = resonance_module.query(context_emb)
        state = primer.update(resonance)

        # Inspect current warmth
        if state.is_primed:
            recall_ids = state.top_memory_ids[:3]
            # Hand off to EpisodicRecall

        # Inject into CognitiveState
        cognitive_state.resonance_bias = state.warmth_vector

        # Reset at conversation start (new episode = cold start)
        primer.reset()

    Args:
        decay_factor:             Per-turn exponential decay multiplier [0, 1].
                                  0.85 → half-life ≈ 4-5 turns.
        accumulation_rate:        Weight of new resonance signal in blend [0, 1].
                                  0.4 → 40% new, 60% existing.
        ambient_threshold:        warmth_magnitude threshold for 'ambient' state.
        familiarity_threshold:    warmth_magnitude threshold for 'familiar' state.
        recall_trigger_threshold: warmth_magnitude threshold for 'primed' state
                                  (involuntary recall trigger -- requires also that
                                  ResonanceResult.triggered_recall is True).
    """

    def __init__(
        self,
        decay_factor:             float = 0.85,
        accumulation_rate:        float = 0.4,
        ambient_threshold:        float = 0.15,
        familiarity_threshold:    float = 0.35,
        recall_trigger_threshold: float = 0.55,
    ) -> None:
        if not (0.0 < decay_factor <= 1.0):
            raise ValueError(f"decay_factor must be in (0, 1], got {decay_factor}")
        if not (0.0 < accumulation_rate <= 1.0):
            raise ValueError(f"accumulation_rate must be in (0, 1], got {accumulation_rate}")

        self.decay_factor             = decay_factor
        self.accumulation_rate        = accumulation_rate
        self.ambient_threshold        = ambient_threshold
        self.familiarity_threshold    = familiarity_threshold
        self.recall_trigger_threshold = recall_trigger_threshold

        self._state = PrimerState()

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, resonance: ResonanceResult) -> PrimerState:
        """
        Incorporate a new resonance query result, update and return PrimerState.

        Call once per conversational turn with the result from
        MemoryResonanceModule.query().

        Steps:
          1. Decay existing warmth (exponential)
          2. If resonance has signal: blend in (accumulation)
          3. Compute warmth_magnitude
          4. Classify state label
          5. Set Proustian trigger flag if conditions met

        Args:
            resonance: Result from MemoryResonanceModule.query().

        Returns:
            Updated PrimerState (also stored as self.state).
        """
        # ── Step 1: Decay ───────────────────────────────────────────────────────
        self._state.warmth_vector = self._state.warmth_vector * self.decay_factor

        # ── Step 2: Accumulate new signal ──────────────────────────────────────
        if resonance.has_resonance:
            new_signal = resonance.resonance_vector  # (EMO_SIG_DIM,)
            self._state.warmth_vector = (
                (1.0 - self.accumulation_rate) * self._state.warmth_vector
                + self.accumulation_rate * new_signal
            )
            self._state.turns_since_reinforcement = 0
            self._state.top_memory_ids = list(resonance.top_k_ids)
        else:
            self._state.turns_since_reinforcement += 1

        # ── Step 3: Magnitude ──────────────────────────────────────────────────
        mag = float(np.linalg.norm(self._state.warmth_vector))
        # Normalise: warmth_vector L2 norm is at most 1.0 if fully accumulated
        # and not decayed. We soft-clip to [0, 1] just in case.
        self._state.warmth_magnitude = min(mag, 1.0)

        # ── Step 4: State label ────────────────────────────────────────────────
        m = self._state.warmth_magnitude
        if m >= self.recall_trigger_threshold:
            self._state.state_label = "primed"
        elif m >= self.familiarity_threshold:
            self._state.state_label = "familiar"
        elif m >= self.ambient_threshold:
            self._state.state_label = "ambient"
        else:
            self._state.state_label = "cold"

        # ── Step 5: Proustian trigger ──────────────────────────────────────────
        # Involuntary recall fires when BOTH:
        #   - warmth has built up to 'primed' level (sustained accumulation)
        #   - AND the incoming resonance signal independently crossed recall threshold
        # (prevents cold-start single-strong-hit from triggering -- requires warmth buildup)
        self._state.triggered_involuntary_recall = (
            self._state.state_label == "primed"
            and resonance.triggered_recall
        )

        return self._state

    def reset(self) -> None:
        """
        Reset warmth state -- call at the start of a new conversation.

        Each new episode starts cold. This prevents warmth from one
        conversation bleeding into the next (though intentionally NOT
        resetting between turns within a conversation).
        """
        self._state = PrimerState()

    @property
    def state(self) -> PrimerState:
        """Current PrimerState (read-only view)."""
        return self._state

    def describe(self) -> str:
        """
        Human-readable description of current warmth state.

        Useful for debug logging or introspection tools.
        """
        s = self._state
        lines = [
            f"ResonancePrimer -- {s.state_label.upper()}",
            f"  warmth_magnitude:    {s.warmth_magnitude:.3f}",
            f"  turns_no_signal:     {s.turns_since_reinforcement}",
            f"  involuntary_recall:  {s.triggered_involuntary_recall}",
            f"  top_memory_ids:      {s.top_memory_ids[:3]}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ResonancePrimer("
            f"state={self._state.state_label!r}, "
            f"warmth={self._state.warmth_magnitude:.3f}, "
            f"decay={self.decay_factor}, "
            f"accum={self.accumulation_rate})"
        )
