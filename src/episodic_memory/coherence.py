"""
Coherence gate for episodic memory encoding and retrieval.

The core insight: emotional activation should be gated by semantic coherence,
not just semantic content. A coherent mention of food ("mmm, I could really
go for a chicken burger right now") should activate food memories. A word bomb
("taco pizza hamburger ice-cream oreo rack of lamb") should not -- it is
semantically incoherent and should instead activate confusion/concern.

Architecture:

    Turn embeddings (B, T, D)          ← per-turn sentence transformer embeddings
         │
         ├──► TrajectoryCoherenceScorer
         │         Measures smoothness of the turn embedding trajectory.
         │         Adjacent turns in coherent discourse have high cosine
         │         similarity. Word bombs and random topic jumps produce
         │         jagged, low-similarity transitions.
         │         → coherence_score ∈ [0, 1]  (B, 1)
         │
         └──► CoherenceGate
                  Takes CLS pooled representation + coherence_score.
                  Interpolates between the full emotional signal and a
                  learned "confused_prior" vector.
                  
                  gated = gate * features + (1 - gate) * confused_prior
                  
                  The confused_prior is a learned parameter -- it trains to
                  represent the emotional signature of incoherence itself:
                  not silence, but active confusion/concern. This is the
                  right emotional response to word salad.

Training signal for CoherenceScorer:
    Real conversations                  → coherence label 1.0
    Corrupted conversations             → coherence label 0.0
        - word-shuffled turns
        - keyword-bombed turns  
        - sentences from different conversations mixed together
        - sentence order scrambled within a conversation

CoherenceAugmenter provides the corruption functions for dataset generation.

Integration with encoder:
    Sits between the transformer's CLS output and the final output projection.
    Low coherence attenuates the emotional signal, pulling it toward the
    confused_prior before latent projection.
    
    coherence_score is also stored in EpisodicMemory, contributing to
    consolidation_strength: incoherent memories consolidate weakly.

Neurological analogy:
    The prefrontal cortex modulates amygdala response based on semantic/
    contextual processing. Degraded semantic structure → degraded emotional
    cascade. The coherence gate is that modulation pathway.
    
    Fast amygdala reflex (hot tier): attenuated by coherence gate.
    Slow hippocampal recall (cold tier): incoherent cues produce weak queries.
"""
from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Coherence scorer ───────────────────────────────────────────────────────────

class TrajectoryCoherenceScorer(nn.Module):
    """
    Measures semantic coherence by computing smoothness of the turn embedding
    trajectory. Adjacent turns in coherent discourse have high cosine similarity.

    A coherent conversation about food:
        "I've been craving comfort food lately"
        "Yeah, there's something about a good burger"
        "Exactly, especially with cheese and pickles"
        → high adjacent cosine similarity → high coherence score

    Word bomb in a turn:
        "taco pizza hamburger ice-cream oreo rack-of-lamb"
        → that turn's embedding is semantically distant from adjacent turns
        → low trajectory smoothness → low coherence score

    Implementation notes:
        Uses L2-normalised embeddings for stable cosine similarity.
        A learned calibration head maps raw smoothness → gating weight.
        Temperature controls the sharpness of the coherence boundary.

    Args:
        temperature: sharpness of coherence sigmoid (lower = sharper boundary).
                     Default 1.0 is a good starting point; reduce to 0.5 to
                     make the gate more decisive.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        # Learned calibration: maps raw smoothness score → coherence gate weight
        # Single linear + bias so the scorer can adjust the operating point
        self.calibration = nn.Linear(1, 1)
        nn.init.ones_(self.calibration.weight)
        nn.init.zeros_(self.calibration.bias)

    def forward(
        self,
        turn_embeddings: torch.Tensor,   # (B, T, D) -- per-turn embeddings
        turn_mask:        Optional[torch.Tensor] = None,  # (B, T) True = padding
    ) -> torch.Tensor:                   # (B, 1) coherence score in [0, 1]
        """
        Compute coherence score from turn embedding trajectory.

        Args:
            turn_embeddings: (B, T, D) -- per-turn sentence embeddings.
                             These are the raw embeddings before projection,
                             e.g. text-embedding-3-small outputs at 1536 dims,
                             or bge-large-en-v1.5 outputs at 1024 dims.
            turn_mask:       (B, T) bool -- True marks padding positions.
                             Used to exclude padding turns from similarity
                             calculation. If None, assumes no padding.

        Returns:
            coherence_score: (B, 1) float32 in [0, 1].
                1.0 = highly coherent, gate fully open.
                0.0 = incoherent, gate routes toward confused_prior.
        """
        B, T, D = turn_embeddings.shape

        if T < 2:
            # Can't measure trajectory with fewer than 2 turns.
            # Return neutral coherence (0.5) -- don't penalise or reward.
            return torch.full((B, 1), 0.5, device=turn_embeddings.device)

        # L2 normalise for stable cosine similarity
        normed = F.normalize(turn_embeddings, dim=-1)   # (B, T, D)

        # Adjacent pairwise cosine similarity
        sim = (normed[:, :-1] * normed[:, 1:]).sum(dim=-1)  # (B, T-1)

        # If padding mask is provided, zero out similarities involving padding turns.
        # A transition into or out of a padding turn is not a real discourse transition.
        if turn_mask is not None:
            # turn_mask: True = padding. Similarity at position i covers turns i and i+1.
            # Mask out if either end of the transition is padding.
            pair_mask = turn_mask[:, :-1] | turn_mask[:, 1:]   # (B, T-1)
            sim = sim.masked_fill(pair_mask, float("nan"))

        # Mean of valid similarities (ignoring NaN from masked positions)
        # nan_to_num converts NaN→0 before mean to avoid propagation
        valid_sim = torch.where(
            torch.isnan(sim),
            torch.zeros_like(sim),
            sim,
        )
        n_valid = (~torch.isnan(sim)).float().sum(dim=-1, keepdim=True).clamp(min=1)
        smoothness = valid_sim.sum(dim=-1, keepdim=True) / n_valid   # (B, 1)

        # Calibrated sigmoid: maps smoothness ∈ [-1, 1] → coherence ∈ [0, 1]
        coherence = torch.sigmoid(
            self.calibration(smoothness) / self.temperature
        )

        return coherence   # (B, 1)


# ── Coherence gate ─────────────────────────────────────────────────────────────

class CoherenceGate(nn.Module):
    """
    Gates emotional features based on semantic coherence.

    Low coherence → signal drifts toward confused_prior.
    High coherence → full emotional signal passes through.

    The confused_prior is a *learned* vector, not a zero vector. It trains
    to represent the emotional signature of confusion/concern -- the correct
    affective response to incoherent input. This means:

        - Word salad doesn't produce emotional silence (wrong)
        - Word salad doesn't produce the same emotional response as the
          content words it contains (wrong -- that's the pollution problem)
        - Word salad produces confusion/concern (correct)

    The gate_head is a small MLP that refines the raw coherence score before
    applying it. This allows the gate to learn a non-linear decision boundary
    -- for example, "coherence > 0.6 = open, < 0.4 = closed, 0.4-0.6 = blend"
    rather than a simple linear interpolation.

    Args:
        feature_dim: dimensionality of the features being gated (d_model).
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        # Learned "confused/concerned" baseline state.
        # Initialised near zero; trains to represent the emotional signature
        # of incoherence. Regularise this with L2 during training to keep it
        # compact -- we don't want confusion to be as expressive as real emotions.
        self.confused_prior = nn.Parameter(
            torch.randn(feature_dim) * 0.01
        )

        # Small MLP to map raw coherence score → gate weight.
        # Input: (B, 1) coherence score
        # Output: (B, 1) gate weight ∈ [0, 1]
        self.gate_head = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self._init_gate()

    def _init_gate(self) -> None:
        """
        Initialise gate_head to be near-linear at start of training.
        The sigmoid output should start close to the input coherence score
        so the gate doesn't immediately collapse the signal during early training.
        """
        for m in self.gate_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        features:         torch.Tensor,   # (B, feature_dim) -- CLS pooled rep
        coherence_score:  torch.Tensor,   # (B, 1) -- from TrajectoryCoherenceScorer
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply coherence gating to emotional features.

        Args:
            features:        (B, feature_dim) -- typically the CLS output from
                             the transformer encoder.
            coherence_score: (B, 1) in [0, 1] -- from TrajectoryCoherenceScorer.

        Returns:
            gated:      (B, feature_dim) -- coherence-gated features.
            gate_value: (B, 1) -- the actual gate weight applied [0, 1].
                        Useful for logging, debugging, and loss computation.
        """
        gate_value = self.gate_head(coherence_score)   # (B, 1)

        confused = self.confused_prior.unsqueeze(0).expand_as(features)  # (B, D)

        # Linear interpolation between full signal and confused prior
        gated = gate_value * features + (1.0 - gate_value) * confused

        return gated, gate_value


# ── Augmentation for coherence training ───────────────────────────────────────

class CoherenceAugmenter:
    """
    Generates incoherent conversation variants for coherence classifier training.

    The CoherenceScorer needs a training signal: real conversations should score
    high, corrupted versions should score low. This class provides four
    corruption strategies covering different incoherence failure modes.

    Usage::

        augmenter = CoherenceAugmenter(seed=42)
        
        turns = ["I'm really hungry", "Yeah, a burger sounds amazing"]
        corrupted = augmenter.apply_random(turns)
        # e.g. → ["amazing burger Yeah sounds a", "really hungry I'm"]
        
        # For dataset generation, produce (original, label=1) and
        # (corrupted, label=0) pairs:
        pairs = augmenter.make_training_pair(turns)
        # → {"coherent": turns, "incoherent": corrupted, "coherent_label": 1.0}

    Corruption strategies:

        word_shuffle: shuffles words within each turn independently.
            Breaks syntactic structure, preserves vocabulary.
            Targets: intra-turn incoherence, word bomb detection.

        sentence_scramble: randomly reorders the turns in the conversation.
            Each turn is internally coherent, but discourse flow is broken.
            Targets: inter-turn incoherence.

        keyword_bomb: replaces turns with random selections from a topic
            wordlist. The result is semantically related but structurally
            empty -- exactly the "mmm taco pizza hamburger" scenario.
            Targets: keyword stuffing, emotional manipulation attempts.

        cross_mix: interleaves turns from two different conversations.
            Each turn is coherent on its own; the discourse is not.
            Targets: the hardest case -- requires global discourse tracking.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    # ── Corruption methods ─────────────────────────────────────────────────────

    def word_shuffle(self, turns: list[str]) -> list[str]:
        """
        Shuffle words within each turn independently.
        Preserves vocabulary, destroys syntax and flow.
        """
        result = []
        for turn in turns:
            words = turn.split()
            self._rng.shuffle(words)
            result.append(" ".join(words))
        return result

    def sentence_scramble(self, turns: list[str]) -> list[str]:
        """
        Randomly reorder the turns in the conversation.
        Each turn is internally coherent; discourse flow is broken.
        """
        shuffled = list(turns)
        self._rng.shuffle(shuffled)
        return shuffled

    def keyword_bomb(
        self,
        turns: list[str],
        topic_words: Optional[list[str]] = None,
        words_per_turn: int = 12,
    ) -> list[str]:
        """
        Replace turns with random keyword lists.
        
        If topic_words is provided, uses those (e.g. food vocabulary to
        simulate emotionally loaded but contextually meaningless input).
        If not, extracts content words from the original turns.
        
        Args:
            turns:          original turn list (used to extract vocab if
                            topic_words is None)
            topic_words:    explicit vocabulary to sample from
            words_per_turn: how many keywords per synthesised "turn"
        """
        if topic_words is None:
            # Extract all words from original turns as the vocabulary
            all_words = []
            for t in turns:
                all_words.extend(t.split())
            topic_words = all_words if all_words else ["something"]

        return [
            " ".join(self._rng.choices(topic_words, k=words_per_turn))
            for _ in turns
        ]

    def cross_mix(
        self,
        turns_a: list[str],
        turns_b: list[str],
    ) -> list[str]:
        """
        Interleave turns from two different conversations.
        Each turn is coherent; the combined discourse is not.
        Result length matches the longer of the two inputs.
        """
        combined = turns_a + turns_b
        self._rng.shuffle(combined)
        target_len = max(len(turns_a), len(turns_b))
        return combined[:target_len]

    # ── Composite interface ────────────────────────────────────────────────────

    def apply_random(
        self,
        turns:        list[str],
        turns_b:      Optional[list[str]] = None,
        topic_words:  Optional[list[str]] = None,
    ) -> list[str]:
        """
        Apply a randomly chosen corruption strategy.
        cross_mix requires turns_b; if turns_b is None it is skipped.

        Returns the corrupted turn list.
        """
        strategies = [self.word_shuffle, self.sentence_scramble]
        if topic_words is not None:
            strategies.append(
                lambda t: self.keyword_bomb(t, topic_words=topic_words)
            )
        if turns_b is not None:
            strategies.append(lambda t: self.cross_mix(t, turns_b))

        strategy = self._rng.choice(strategies)
        return strategy(turns)

    def make_training_pair(
        self,
        turns:        list[str],
        turns_b:      Optional[list[str]] = None,
        topic_words:  Optional[list[str]] = None,
    ) -> dict:
        """
        Produce a (coherent, incoherent) training pair with labels.

        Returns::

            {
                "coherent":       list[str]  -- original turns, label 1.0
                "incoherent":     list[str]  -- corrupted turns, label 0.0
                "coherent_label": 1.0
                "incoherent_label": 0.0
                "strategy":       str        -- which corruption was applied
            }
        """
        strategy_name = "unknown"
        strategies_named = [
            ("word_shuffle",     lambda t: self.word_shuffle(t)),
            ("sentence_scramble", lambda t: self.sentence_scramble(t)),
        ]
        if topic_words is not None:
            strategies_named.append((
                "keyword_bomb",
                lambda t: self.keyword_bomb(t, topic_words=topic_words),
            ))
        if turns_b is not None:
            strategies_named.append((
                "cross_mix",
                lambda t: self.cross_mix(t, turns_b),
            ))

        strategy_name, strategy_fn = self._rng.choice(strategies_named)
        corrupted = strategy_fn(turns)

        return {
            "coherent":           turns,
            "incoherent":         corrupted,
            "coherent_label":     1.0,
            "incoherent_label":   0.0,
            "strategy":           strategy_name,
        }


# ── Coherence loss ─────────────────────────────────────────────────────────────

class CoherenceLoss(nn.Module):
    """
    Training objective for the TrajectoryCoherenceScorer.

    Binary cross-entropy: real conversations → 1.0, corrupted → 0.0.

    Also includes a regularisation term on the confused_prior from CoherenceGate:
    we want the confused_prior to be compact (small L2 norm) so it acts as a
    soft attractor rather than a full alternative representation. The gate should
    pull toward a minimal "confusion" state, not toward a rich confused emotion.

    Args:
        confused_prior_weight: weight of the L2 regularisation on confused_prior.
                               Default 0.01 is light -- just prevents it from
                               growing unbounded.
    """

    def __init__(self, confused_prior_weight: float = 0.01) -> None:
        super().__init__()
        self.confused_prior_weight = confused_prior_weight

    def forward(
        self,
        coherence_score:  torch.Tensor,                  # (B, 1) predicted
        coherence_label:  torch.Tensor,                  # (B,) 0.0 or 1.0
        confused_prior:   Optional[torch.Tensor] = None, # (feature_dim,) parameter
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            coherence_score: (B, 1) output of TrajectoryCoherenceScorer.
            coherence_label: (B,) ground truth. 1.0 = coherent, 0.0 = corrupted.
            confused_prior:  Optional -- CoherenceGate.confused_prior parameter.
                             If provided, L2 regularisation is applied.

        Returns:
            dict with keys:
                "coherence_bce"   -- binary cross-entropy loss (main signal)
                "prior_reg"       -- L2 regularisation on confused_prior (if provided)
                "total"           -- weighted sum
        """
        bce = F.binary_cross_entropy(
            coherence_score.squeeze(-1),
            coherence_label.to(coherence_score.device),
        )

        losses = {"coherence_bce": bce}

        if confused_prior is not None:
            prior_reg = (confused_prior ** 2).mean()
            losses["prior_reg"] = prior_reg
            losses["total"] = bce + self.confused_prior_weight * prior_reg
        else:
            losses["total"] = bce

        return losses
