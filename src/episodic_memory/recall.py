"""
EpisodicRecall -- the slow hippocampal pathway.

Triggered only when MemoryResonanceModule fires (max_similarity >= recall_threshold).
Fetches the full transcript from the cold tier (SQLite), generates or retrieves a
natural-language summary, and returns a RecallResult for context injection.

Analogy: the resonance module says "something about this moment feels familiar --
check the archives". EpisodicRecall is the archivist: it pulls the actual record,
reads it, and writes a brief note to hand back to consciousness.

Latency contract:
  Fast path (summary already cached): ~1ms (SQLite read)
  Slow path (summary generation):     100-500ms (local LLM call)
  The slow path is only triggered once per session -- after generation the
  summary is persisted via store.update_summary() for instant future access.

LLM prompt strategy:
  We do NOT ask the LLM to reproduce the conversation. We ask it to produce a
  gist that preserves:
    - Dominant emotional tone
    - Relational / archetypal mode (e.g. "you were in a teaching role")
    - Key contextual theme ("the conversation was about X")
    - Affective resolution ("it ended warmly / with tension / inconclusively")
  This is exactly the information that should be injected back into context
  to prime the model's processing -- not verbatim quotes.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional, TYPE_CHECKING

import numpy as np

from episodic_memory.schemas import (
    RecallResult,
    EMO_SIG_DIM,
    ARCH_SIG_DIM,
)

if TYPE_CHECKING:
    from episodic_memory.store import EpisodicMemoryStore

logger = logging.getLogger(__name__)


# ── Prompt template ────────────────────────────────────────────────────────────

_SUMMARY_SYSTEM = """You are a cognitive memory summariser for an AI agent.
Your task is to distill a past conversation into a compact episodic gist.

Focus ONLY on:
1. The dominant emotional tone throughout (and any significant shifts)
2. The relational mode -- was the AI acting as teacher, peer, collaborator, supporter?
3. The core contextual theme -- what was the conversation fundamentally about?
4. How it resolved -- warmly, with tension, openly, with a clear outcome?

Do NOT reproduce quotes. Do NOT list facts or topics discussed.
Write 2-4 sentences maximum. Be specific about affect and relational quality.
Write in past tense, third person ("The user and agent...")."""

_SUMMARY_USER = """Conversation to summarise (turn count: {turn_count}):

{transcript_excerpt}

Dominant emotion at session end: {dominant_emotion}
Dominant relational mode: {dominant_archetype}

Write the episodic gist now."""

# ── Technical index prompt ─────────────────────────────────────────────────────

_TECHNICAL_SYSTEM = """You are a technical artefact extractor for an AI agent's memory system.
Your task is to identify and list named technical facts from a conversation transcript.

Extract ONLY items that are explicitly named or stated:
- File paths, module names, line numbers
- Metric values and scores (AUROC, loss, accuracy, F1, latency, etc.) with their context
- Model names, checkpoint paths, version numbers, revision IDs
- CLI flags, config keys, and their values
- Ports, hostnames, service names
- Explicit decisions, conclusions, or action items agreed on

Output as a compact bulleted list (- item). One fact per line.
If nothing technical is present, output exactly: NONE
Do NOT infer, paraphrase, or add context. Only extract what is explicitly stated."""

_TECHNICAL_USER = """Conversation (turn count: {turn_count}):

{transcript_excerpt}

Extract named technical artefacts now."""


class EpisodicRecall:
    """
    Slow hippocampal recall path -- triggered by strong memory resonance.

    Fetches full transcript from cold tier, generates (or retrieves cached)
    natural-language summary, returns RecallResult for prompt injection.

    Usage::

        recall = EpisodicRecall(
            store=store,
            llm_base_url="http://localhost:11434/v1",  # Ollama default; any OpenAI-compatible endpoint
            llm_model="llama3",
        )

        # Called when ResonanceResult.triggered_recall is True
        result = recall.recall(session_id="conv_123", similarity=0.82)
        if result:
            system_prompt += result.context_injection()

    Args:
        store:               EpisodicMemoryStore providing cold-tier access.
        llm_base_url:        OpenAI-compatible endpoint for summary generation.
        llm_model:           Model name to use for summarisation.
        llm_api_key:         API key (default "none" for local endpoints).
        max_transcript_turns: Max turns to send to LLM for summarisation.
                             Older turns are truncated from the front.
        summary_max_tokens:  Max tokens in generated summary.
        timeout:             HTTP timeout for LLM calls (seconds).
    """

    def __init__(
        self,
        store:                          "EpisodicMemoryStore",
        llm_base_url:                   str   = "http://localhost:11434/v1",
        llm_model:                      str   = "llama3",
        llm_api_key:                    str   = "none",
        technical_llm_base_url:         Optional[str] = None,
        technical_llm_model:            Optional[str] = None,
        technical_llm_api_key:          Optional[str] = None,
        max_transcript_turns:           int   = 30,
        summary_max_tokens:             int   = 150,
        technical_index_max_tokens:     int   = 400,
        timeout:                        float = 20.0,
    ) -> None:
        self.store                      = store
        self.llm_base_url               = llm_base_url
        self.llm_model                  = llm_model
        self.llm_api_key                = llm_api_key
        # Technical index LLM -- falls back to affective summary endpoint if not set
        self.technical_llm_base_url     = technical_llm_base_url or llm_base_url
        self.technical_llm_model        = technical_llm_model or llm_model
        self.technical_llm_api_key      = technical_llm_api_key or llm_api_key
        self.max_transcript_turns       = max_transcript_turns
        self.summary_max_tokens         = summary_max_tokens
        self.technical_index_max_tokens = technical_index_max_tokens
        self.timeout                    = timeout
        self._client: Optional[object]          = None
        self._technical_client: Optional[object] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def recall(
        self,
        session_id:          str,
        similarity:          float = 0.0,
        generate_if_missing: bool  = True,
    ) -> Optional[RecallResult]:
        """
        Recall an episode by session_id.

        Fast path: if a summary is already cached in the cold tier, return
        immediately without calling the LLM.

        Slow path: if no summary exists and generate_if_missing=True, call
        the LLM to generate one, cache it, then return.

        Args:
            session_id:          Session to recall.
            similarity:          Cosine similarity of this memory (from hot tier).
                                 Stored in RecallResult for caller use.
            generate_if_missing: Whether to call LLM if no cached summary exists.

        Returns:
            RecallResult, or None if session_id not found in store.
        """
        # Fetch metadata from hot tier
        meta = self._get_metadata(session_id)
        if meta is None:
            logger.warning(f"EpisodicRecall: session_id {session_id!r} not found")
            return None

        # Try cached summary first (fast path)
        summary = self.store.fetch_summary(session_id) or ""
        transcript: Optional[list] = None

        # Generate summary if missing (slow path)
        if not summary and generate_if_missing:
            transcript = self.store.fetch_transcript(session_id)
            if transcript:
                summary = self._generate_summary(transcript, meta)
                if summary:
                    self.store.update_summary(session_id, summary)
            else:
                logger.warning(
                    f"EpisodicRecall: no transcript found for {session_id!r}"
                )
                summary = self._fallback_summary(meta)

        if not summary:
            summary = self._fallback_summary(meta)

        # Technical index -- parallel fact-preserving path
        technical_index = self.store.fetch_technical_index(session_id)
        if not technical_index and generate_if_missing:
            if transcript is None:
                transcript = self.store.fetch_transcript(session_id)
            if transcript:
                raw = self._generate_technical_index(transcript, meta)
                if raw and raw.strip().upper() != "NONE":
                    technical_index = raw
                    self.store.update_technical_index(session_id, technical_index)

        # Get latent sub-vectors from hot tier
        latent = self.store.get_latent(session_id)
        if latent is None:
            emo_sig  = np.zeros(EMO_SIG_DIM, dtype=np.float32)
            arch_sig = np.zeros(ARCH_SIG_DIM, dtype=np.float32)
        else:
            emo_sig  = latent[:EMO_SIG_DIM]
            arch_sig = latent[EMO_SIG_DIM:EMO_SIG_DIM + ARCH_SIG_DIM]

        return RecallResult(
            session_id           = session_id,
            summary              = summary,
            emotional_signature  = emo_sig,
            archetypal_signature = arch_sig,
            similarity           = similarity,
            turn_count           = meta.get("turn_count", 0),
            stored_at            = meta.get("stored_at", 0.0),
            dominant_emotion     = meta.get("dominant_emotion", "neutral"),
            dominant_archetype   = meta.get("dominant_archetype", "sage"),
            technical_index      = technical_index,
            metadata             = {
                k: v for k, v in meta.items()
                if k not in ("session_id", "_removed")
            },
        )

    def recall_batch(
        self,
        session_ids: list[str],
        similarities: list[float],
        generate_if_missing: bool = True,
    ) -> list[Optional[RecallResult]]:
        """
        Recall multiple episodes. Returns results in the same order as session_ids.
        None entries indicate sessions that were not found.
        """
        results = []
        for sid, sim in zip(session_ids, similarities):
            results.append(self.recall(sid, sim, generate_if_missing))
        return results

    def precompute_summaries(
        self,
        session_ids:       Optional[list[str]] = None,
        batch_size:        int  = 10,
        include_technical: bool = True,
    ) -> dict[str, dict]:
        """
        Pre-generate summaries (and optionally technical indexes) for sessions
        that don't yet have them cached.

        Args:
            session_ids:       Sessions to process. If None, processes ALL sessions
                               missing a summary.
            batch_size:        Log progress every N sessions.
            include_technical: Also generate technical indexes for sessions that
                               don't have one. Default True.

        Returns:
            dict mapping session_id → {"summary": ..., "technical_index": ...}
            for sessions where at least one field was newly generated.
        """
        if session_ids is None:
            session_ids = [
                sid for sid in self.store.session_ids
                if not self.store.fetch_summary(sid)
                or (include_technical and not self.store.fetch_technical_index(sid))
            ]

        generated: dict[str, dict] = {}
        total = len(session_ids)

        for i, sid in enumerate(session_ids):
            if i % batch_size == 0:
                logger.info("Precomputing: %d/%d", i, total)

            try:
                transcript = self.store.fetch_transcript(sid)
                if not transcript:
                    continue

                meta = self._get_metadata(sid) or {}
                result: dict = {}

                if not self.store.fetch_summary(sid):
                    summary = self._generate_summary(transcript, meta)
                    if summary:
                        self.store.update_summary(sid, summary)
                        result["summary"] = summary

                if include_technical and not self.store.fetch_technical_index(sid):
                    raw = self._generate_technical_index(transcript, meta)
                    if raw and raw.strip().upper() != "NONE":
                        self.store.update_technical_index(sid, raw)
                        result["technical_index"] = raw

                if result:
                    generated[sid] = result

            except Exception as e:
                logger.error("Precompute failed for %s: %s", sid, e)

        logger.info(
            "Precompute complete: %d/%d sessions updated (technical=%s)",
            len(generated), total, include_technical,
        )
        return generated

    # ── LLM interaction ────────────────────────────────────────────────────────

    def _generate_summary(
        self,
        transcript: list[dict],
        meta:       dict,
    ) -> str:
        """
        Call local LLM to generate an episodic gist from transcript.

        Returns empty string on failure (caller handles fallback).
        """
        client = self._get_client()

        # Truncate transcript for LLM context window
        turns = transcript[-self.max_transcript_turns:]
        excerpt_lines = []
        for turn in turns:
            role    = turn.get("role", "unknown").title()
            content = turn.get("content", "").strip()
            if content:
                # Truncate very long individual turns
                if len(content) > 500:
                    content = content[:500] + "..."
                excerpt_lines.append(f"{role}: {content}")

        transcript_excerpt = "\n".join(excerpt_lines)

        user_msg = _SUMMARY_USER.format(
            turn_count         = meta.get("turn_count", len(transcript)),
            transcript_excerpt = transcript_excerpt,
            dominant_emotion   = meta.get("dominant_emotion", "neutral"),
            dominant_archetype = meta.get("dominant_archetype", "sage"),
        )

        try:
            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model       = self.llm_model,
                messages    = [
                    {"role": "system", "content": _SUMMARY_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens  = self.summary_max_tokens,
                temperature = 0.3,
                timeout     = self.timeout,
            )
            elapsed = time.perf_counter() - t0
            summary = response.choices[0].message.content.strip()
            logger.debug(
                f"EpisodicRecall: generated summary in {elapsed:.2f}s "
                f"({len(summary)} chars)"
            )
            return summary
        except Exception as e:
            logger.error(f"EpisodicRecall: LLM summary generation failed: {e}")
            return ""

    def _generate_technical_index(
        self,
        transcript: list[dict],
        meta:       dict,
    ) -> str:
        """
        Call local LLM to extract named technical artefacts from transcript.

        Returns empty string on failure or "NONE" if no technical content.
        The caller should discard "NONE" responses rather than storing them.
        """
        client = self._get_client()

        turns = transcript[-self.max_transcript_turns:]
        excerpt_lines = []
        for turn in turns:
            role    = turn.get("role", "unknown").title()
            content = turn.get("content", "").strip()
            if content:
                if len(content) > 500:
                    content = content[:500] + "..."
                excerpt_lines.append(f"{role}: {content}")

        transcript_excerpt = "\n".join(excerpt_lines)

        user_msg = _TECHNICAL_USER.format(
            turn_count         = meta.get("turn_count", len(transcript)),
            transcript_excerpt = transcript_excerpt,
        )

        try:
            t0 = time.perf_counter()
            response = self._get_technical_client().chat.completions.create(
                model       = self.technical_llm_model,
                messages    = [
                    {"role": "system", "content": _TECHNICAL_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens  = self.technical_index_max_tokens,
                temperature = 0.1,
                timeout     = self.timeout,
            )
            elapsed = time.perf_counter() - t0
            result = response.choices[0].message.content.strip()
            logger.debug(
                f"EpisodicRecall: generated technical index in {elapsed:.2f}s "
                f"({len(result)} chars)"
            )
            return result
        except Exception as e:
            logger.error(f"EpisodicRecall: technical index generation failed: {e}")
            return ""

    def _get_client(self) -> object:
        """Lazy-init OpenAI-compatible client for affective summary."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url = self.llm_base_url,
                api_key  = self.llm_api_key,
            )
        return self._client

    def _get_technical_client(self) -> object:
        """Lazy-init OpenAI-compatible client for technical index (may differ from summary client)."""
        if self._technical_client is None:
            from openai import OpenAI
            self._technical_client = OpenAI(
                base_url = self.technical_llm_base_url,
                api_key  = self.technical_llm_api_key,
            )
        return self._technical_client

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _get_metadata(self, session_id: str) -> Optional[dict]:
        """Fetch metadata from hot-tier index. Returns None if not found."""
        if session_id not in self.store._session_index:
            return None
        idx = self.store._session_index[session_id]
        return dict(self.store._hot_metadata[idx])

    @staticmethod
    def _fallback_summary(meta: dict) -> str:
        """
        Minimal fallback summary when LLM is unavailable.
        Uses only metadata fields -- no LLM call.
        """
        import datetime
        ts = meta.get("stored_at", 0.0)
        date_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else "unknown date"
        turns    = meta.get("turn_count", 0)
        emotion  = meta.get("dominant_emotion", "neutral")
        archetype = meta.get("dominant_archetype", "sage")
        return (
            f"A {turns}-turn conversation from {date_str}. "
            f"Dominant tone: {emotion}. Relational mode: {archetype}."
        )

    def __repr__(self) -> str:
        return (
            f"EpisodicRecall("
            f"model={self.llm_model!r}, "
            f"store_episodes={self.store.n_episodes})"
        )
