"""
EpisodicEncoder -- encodes a full conversation into a compact latent L.

Architecture:

  Input: N turns, each with (user_emb: 1536, agent_emb: 1536)

  1. Concatenate per-turn: (N, 3072) → project → (N, d_model)
  2. Prepend learnable [CLS] token → (N+1, d_model)
  3. Add learned positional embeddings
  4. TransformerEncoder (n_layers=2, n_heads=4 by default)
  5. Pool [CLS] output → (d_model,)
  6. Optional: add projected terminal context_emb (D012)
  7. Linear → Tanh → (latent_dim,)
  8. L2-normalise → L

  QueryProjector (trained jointly):
    context_emb (96,) → MLP → (latent_dim,) → L2-normalise
    Trained to approximate encoder(full_conversation) from mid-conversation
    context_emb. Enables fast querying without re-encoding.

Why CLS pooling over mean pooling:
  CLS learns to attend to the most emotionally/contextually salient turns.
  A farewell turn may be more diagnostically important than 10 factual turns.
  Mean pooling would dilute that signal. We want the model to learn which
  turns are episodically "load-bearing" -- CLS attention lets it do that.

Why learned positional embeddings over sinusoidal:
  Conversations are short (typically 5-30 turns). The model only needs to
  distinguish "early vs late" and "first vs last" -- sinusoidal over-engineers
  this. Learned embeddings are more expressive and faster to train on short
  sequences.

Why residual addition of cognitive seed (D012):
  The terminal CognitiveState is a compressed summary from the per-turn state
  manager -- it already "knows" what the conversation felt like overall. Adding
  it residually to the CLS output lets the episodic encoder incorporate that
  signal without needing to re-learn it from scratch. Think of it as giving
  the encoder a "ground truth hint" during training that it can also use at
  inference time.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from episodic_memory.schemas import EpisodicEncoderConfig, LATENT_DIM, CONTEXT_EMB_DIM
from episodic_memory.coherence import TrajectoryCoherenceScorer, CoherenceGate


class EpisodicEncoder(nn.Module):
    """
    Encodes a full conversation into a compact, L2-normalised latent vector L.

    Usage::

        config = EpisodicEncoderConfig()
        encoder = EpisodicEncoder(config)

        # user_embs, agent_embs: (batch, turns, 1536)
        # turn_mask: (batch, turns) -- True = padding position
        # terminal_context_emb: (batch, 96) or None
        L = encoder(user_embs, agent_embs, turn_mask, terminal_context_emb)
        # L: (batch, 256) L2-normalised

    Args:
        config: EpisodicEncoderConfig controlling all hyperparameters.
    """

    def __init__(self, config: EpisodicEncoderConfig) -> None:
        super().__init__()
        self.config = config

        # Project concatenated (user_emb ‖ agent_emb) → d_model
        # Input: (batch, turns, 2 * input_dim)
        self.turn_proj = nn.Sequential(
            nn.Linear(config.input_dim * 2, config.d_model),
            nn.LayerNorm(config.d_model),
        )

        # Coherence gate -- measures trajectory smoothness across turns,
        # gates CLS output toward confused_prior when coherence is low.
        # See ctm/memory/coherence.py for full design rationale.
        if config.use_coherence_gate:
            self.coherence_scorer = TrajectoryCoherenceScorer(
                temperature=config.coherence_temperature
            )
            self.coherence_gate = CoherenceGate(feature_dim=config.d_model)

        # Learnable [CLS] token -- one vector broadcast over batch
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        # Learned positional embeddings: max_turns + 1 positions (CLS = position 0)
        self.pos_embedding = nn.Embedding(config.max_turns + 1, config.d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = config.d_model,
            nhead           = config.n_heads,
            dim_feedforward = config.d_model * 4,
            dropout         = config.dropout,
            batch_first     = True,
            norm_first      = True,   # pre-LN -- more stable with small data
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = config.n_layers,
        )

        # Optional: project terminal context_emb → d_model (cognitive seed)
        if config.use_cognitive_seed:
            self.cognitive_seed_proj = nn.Sequential(
                nn.Linear(config.context_emb_dim, config.d_model),
                nn.Tanh(),
            )

        # Output projection: d_model → latent_dim
        self.output_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.latent_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform for linear layers, small init for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        user_embs:             torch.Tensor,            # (B, T, input_dim)
        agent_embs:            torch.Tensor,            # (B, T, input_dim)
        turn_mask:             torch.Tensor,            # (B, T) bool -- True = padding
        terminal_context_emb:  Optional[torch.Tensor] = None,  # (B, context_emb_dim)
    ) -> torch.Tensor:
        """
        Encode a batch of conversations into L2-normalised latent vectors.

        Args:
            user_embs:            (B, T, 1536) -- per-turn user embeddings.
            agent_embs:           (B, T, 1536) -- per-turn agent embeddings.
            turn_mask:            (B, T) bool -- True marks padding positions.
                                  Real turns should be False.
            terminal_context_emb: (B, 96) -- context_emb from terminal
                                  CognitiveState. Optional. If provided and
                                  config.use_cognitive_seed=True, it is added
                                  residually to the CLS output.

        Returns:
            L:               (B, latent_dim) -- L2-normalised episodic latent vectors.
            coherence_score: (B, 1) in [0, 1] if config.use_coherence_gate,
                             else None. Store this in EpisodicMemory.metadata
                             to feed into consolidation_strength.
        """
        B, T, _ = user_embs.shape

        # Truncate if conversation exceeds max_turns (keep most recent turns)
        if T > self.config.max_turns:
            user_embs  = user_embs[:, -self.config.max_turns:, :]
            agent_embs = agent_embs[:, -self.config.max_turns:, :]
            turn_mask  = turn_mask[:, -self.config.max_turns:]
            T = self.config.max_turns

        # ── 1. Per-turn projection ─────────────────────────────────────────────
        # Concatenate user + agent embeddings for each turn
        turn_input = torch.cat([user_embs, agent_embs], dim=-1)   # (B, T, 2*input_dim)
        turn_feats = self.turn_proj(turn_input)                    # (B, T, d_model)

        # ── 2. Prepend [CLS] token ─────────────────────────────────────────────
        cls = self.cls_token.expand(B, -1, -1)                    # (B, 1, d_model)
        sequence = torch.cat([cls, turn_feats], dim=1)            # (B, T+1, d_model)

        # ── 3. Positional encoding ─────────────────────────────────────────────
        positions = torch.arange(T + 1, device=user_embs.device)  # (T+1,)
        sequence = sequence + self.pos_embedding(positions)        # (B, T+1, d_model)

        # ── 4. Build padding mask for transformer ──────────────────────────────
        # TransformerEncoder expects True = ignore this position
        # CLS token (position 0) is never masked
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=turn_mask.device)
        full_mask = torch.cat([cls_mask, turn_mask], dim=1)        # (B, T+1)

        # ── 5. Transformer encoding ────────────────────────────────────────────
        encoded = self.transformer(
            sequence,
            src_key_padding_mask=full_mask,
        )                                                           # (B, T+1, d_model)

        # ── 6. Pool [CLS] output ───────────────────────────────────────────────
        cls_out = encoded[:, 0, :]                                  # (B, d_model)

        # ── 7. Optional: add cognitive seed ───────────────────────────────────
        if (terminal_context_emb is not None
                and self.config.use_cognitive_seed
                and hasattr(self, "cognitive_seed_proj")):
            seed = self.cognitive_seed_proj(terminal_context_emb)  # (B, d_model)
            cls_out = cls_out + seed                               # residual

        # ── 7b. Optional: coherence gate ──────────────────────────────────────
        # Measure trajectory smoothness across turns. Attenuate cls_out toward
        # the learned confused_prior when the conversation is incoherent.
        # Raw user_embs used (not projected) to measure discourse coherence --
        # we want the coherence signal to reflect the input semantics, not the
        # encoder's own representation (which may have learned to smooth noise).
        coherence_score = None
        if self.config.use_coherence_gate and hasattr(self, "coherence_scorer"):
            coherence_score = self.coherence_scorer(user_embs, turn_mask)  # (B, 1)
            cls_out, gate_value = self.coherence_gate(cls_out, coherence_score)
            # gate_value available for logging: high = coherent, low = confused

        # ── 8. Project to latent + L2-normalise ───────────────────────────────
        L = self.output_proj(cls_out)                              # (B, latent_dim)
        L = F.normalize(L, p=2, dim=-1)                           # unit hypersphere

        return L, coherence_score

    @torch.no_grad()
    def encode_numpy(
        self,
        user_embs:             "np.ndarray",            # (T, 1536)
        agent_embs:            "np.ndarray",            # (T, 1536)
        terminal_context_emb:  Optional["np.ndarray"] = None,  # (96,)
        device:                str = "cpu",
    ) -> "np.ndarray":
        """
        Convenience method: encode a single conversation from numpy arrays.

        Returns L as a (latent_dim,) float32 numpy array.
        Useful for storing memories without managing batch dimensions.
        """
        import numpy as np

        was_training = self.training
        self.eval()

        dev = torch.device(device)
        u = torch.from_numpy(user_embs).float().unsqueeze(0).to(dev)   # (1, T, 1536)
        a = torch.from_numpy(agent_embs).float().unsqueeze(0).to(dev)  # (1, T, 1536)
        mask = torch.zeros(1, u.shape[1], dtype=torch.bool, device=dev) # no padding

        ctx = None
        if terminal_context_emb is not None:
            ctx = torch.from_numpy(terminal_context_emb).float().unsqueeze(0).to(dev)

        L, coh = self.forward(u, a, mask, ctx)                         # (1, latent_dim)
        result = L.squeeze(0).cpu().numpy()
        coherence = float(coh.squeeze().cpu().item()) if coh is not None else None

        if was_training:
            self.train()

        return result, coherence


class QueryProjector(nn.Module):
    """
    Fast inference-time query encoder: context_emb → L-space.

    Maps the current conversation's context_emb (from HybridStateManager)
    to the episodic latent space without needing to re-encode the full
    conversation. Trained jointly with EpisodicEncoder.

    Architecture: 2-layer MLP with GELU activation + L2-normalisation.

    Training signal: the projected vector should approximate
    EpisodicEncoder(full_conversation) for the same conversation.
    This is the query_alignment_loss in EpisodicObjectives.

    Usage::

        proj = QueryProjector(config)
        # context_emb: (B, 96) from current CognitiveState
        query_vec = proj(context_emb)  # (B, 256) L2-normalised
        # Use query_vec to query EpisodicMemoryStore.query()
    """

    def __init__(self, config: EpisodicEncoderConfig) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.query_context_dim, config.query_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.query_hidden_dim),
            nn.Linear(config.query_hidden_dim, config.latent_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, context_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_emb: (B, query_context_dim) -- context_emb from CognitiveState.

        Returns:
            query_vec: (B, latent_dim) -- L2-normalised query vector.
        """
        out = self.net(context_emb)
        return F.normalize(out, p=2, dim=-1)

    @torch.no_grad()
    def project_numpy(
        self,
        context_emb: "np.ndarray",  # (context_emb_dim,)
        device: str = "cpu",
    ) -> "np.ndarray":
        """
        Convenience method: project a single context_emb from numpy.
        Returns (latent_dim,) float32 numpy array.
        """
        import numpy as np

        was_training = self.training
        self.eval()

        dev = torch.device(device)
        x = torch.from_numpy(context_emb).float().unsqueeze(0).to(dev)
        out = self.forward(x).squeeze(0).cpu().numpy()

        if was_training:
            self.train()

        return out
