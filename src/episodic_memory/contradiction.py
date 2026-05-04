"""
ContradictionDetector -- temporal supersession detection for episodic recall.

Problem:
    Episodic memory is append-only. A session from 3 months ago that says
    "Richard uses a headset microphone" is stored alongside a session from
    last week that says "Richard uses NT-1 → PreSonus → XMOS LINE IN".
    Both score similarly for any audio-related query.  Without recency
    awareness, the older episode can be injected as confident context --
    resulting in confabulation downstream.

Design philosophy:
    We do NOT attempt logical negation detection (NLI models, symbolic
    reasoning, etc.).  Instead we use *temporal supersession*: if a newer
    episode covers the same topic with sufficiently high similarity, the
    older episode is treated as superseded -- regardless of whether the
    newer episode explicitly contradicts the older one.

    This is conservative and correct for our use case:
      - Topics evolve, not just contradict. "We set up the XMOS board"
        supersedes "we haven't set up the XMOS board yet" without needing
        an explicit negation signal.
      - False positives (flagging an episode as superseded when it isn't)
        are cheap -- we still inject it, just with a staleness warning.
      - False negatives (missing a contradiction) leave the old behaviour.
        Temporal recency gives us most of the signal we need.

Algorithm (per recalled episode):
    1. Query all episodes semantically similar to the recalled summary
       (using the same BGE embeddings already in memory).
    2. Filter to episodes newer than the recalled one.
    3. If any newer episode has similarity >= supersession_threshold,
       the recalled episode is flagged is_superseded=True.
    4. The newest superseding episode is returned as superseded_by.
    5. context_injection() uses is_superseded to qualify the injected text.

Supersession threshold tuning:
    0.75 -- tight, only very similar topic clusters trigger supersession.
        Recommended default: avoids flagging unrelated episodes.
    0.65 -- moderate, picks up loose thematic successors.
    0.55 -- loose, may supersede too aggressively.

The threshold should be higher than recall_threshold because supersession
is a stronger claim than "related to this query".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SupersessionResult:
    """
    Result of checking whether a recalled episode has been superseded.

    Attributes
    ----------
    is_superseded:
        True if a newer episode covers the same topic with similarity
        >= supersession_threshold.
    superseded_by:
        session_id of the newest superseding episode, or None.
    superseded_by_stored_at:
        Unix timestamp of the superseding episode, or None.
    superseded_by_summary:
        Summary of the superseding episode, or None.
    supersession_similarity:
        Cosine similarity between recalled summary and superseding summary.
    age_gap_days:
        Days between the recalled episode and the superseding episode.
        Positive = superseding episode is newer.
    """
    is_superseded:             bool
    superseded_by:             Optional[str]   = None
    superseded_by_stored_at:   Optional[float] = None
    superseded_by_summary:     Optional[str]   = None
    supersession_similarity:   float           = 0.0
    age_gap_days:              float           = 0.0

    @property
    def age_gap_str(self) -> str:
        d = abs(self.age_gap_days)
        if d < 1:
            return "same day"
        if d < 7:
            return f"{int(d)} day{'s' if d > 1 else ''}"
        if d < 30:
            return f"{int(d / 7)} week{'s' if d >= 14 else ''}"
        if d < 365:
            return f"{int(d / 30)} month{'s' if d >= 60 else ''}"
        return f"{d / 365:.1f} years"


class ContradictionDetector:
    """
    Detects temporal supersession: is a recalled episode outdated relative
    to newer episodes on the same topic?

    Uses the pre-computed BGE summary embeddings from DirectTextResonance
    (passed in at construction time -- no extra embedding work).

    Parameters
    ----------
    session_ids:
        Ordered list of session IDs (same order as embeddings).
    summaries:
        Corresponding summary strings.
    stored_ats:
        Corresponding Unix timestamps.
    summary_embs:
        Pre-computed BGE embeddings, shape (N, embed_dim), unit-normed.
        These are the same embeddings held by DirectTextResonance -- pass
        a reference, not a copy.
    supersession_threshold:
        Minimum cosine similarity between recalled summary and a newer
        episode's summary to trigger supersession.  Default 0.75.
    min_age_gap_days:
        Newer episode must be at least this many days newer than the
        recalled episode to count as a supersession.  Prevents same-day
        updates from being flagged.  Default 1.0.
    """

    def __init__(
        self,
        session_ids:           list[str],
        summaries:             list[str | None],
        stored_ats:            list[float],
        summary_embs:          np.ndarray,        # (N, embed_dim) unit-normed
        supersession_threshold: float = 0.75,
        min_age_gap_days:       float = 1.0,
    ) -> None:
        self._session_ids   = session_ids
        self._summaries     = summaries
        self._stored_ats    = np.array(stored_ats, dtype=np.float64)
        self._summary_embs  = summary_embs
        self.supersession_threshold = supersession_threshold
        self.min_age_gap_days       = min_age_gap_days
        self._min_age_gap_secs      = min_age_gap_days * 86400.0

    def check(
        self,
        session_id: str,
        summary:    str | None = None,
    ) -> SupersessionResult:
        """
        Check whether *session_id* has been superseded by a newer episode.

        Parameters
        ----------
        session_id:
            The recalled episode to check.
        summary:
            The episode's summary (used only for logging; the pre-computed
            embedding is used for similarity).

        Returns
        -------
        SupersessionResult -- always returned, even if not superseded.
        """
        try:
            idx = self._session_ids.index(session_id)
        except ValueError:
            logger.debug("ContradictionDetector: session_id %r not in index", session_id)
            return SupersessionResult(is_superseded=False)

        recalled_stored_at = float(self._stored_ats[idx])
        recalled_emb       = self._summary_embs[idx]  # (embed_dim,) unit-normed

        # Cosine similarities to all other episodes
        sims = self._summary_embs @ recalled_emb  # (N,)

        # Only consider episodes that are:
        #   (a) not the recalled episode itself
        #   (b) newer by at least min_age_gap_days
        age_gaps = self._stored_ats - recalled_stored_at  # positive = newer
        mask = (
            (np.arange(len(self._session_ids)) != idx)
            & (age_gaps >= self._min_age_gap_secs)
        )

        if not mask.any():
            return SupersessionResult(is_superseded=False)

        masked_sims = np.where(mask, sims, -1.0)

        if masked_sims.max() < self.supersession_threshold:
            return SupersessionResult(is_superseded=False)

        # Find the *newest* superseding episode (not just the most similar)
        # Rationale: we want the most current picture, not just any successor.
        superseding_indices = np.where(
            mask & (sims >= self.supersession_threshold)
        )[0]

        newest_idx = superseding_indices[
            np.argmax(self._stored_ats[superseding_indices])
        ]

        age_gap_days = float(age_gaps[newest_idx]) / 86400.0

        logger.debug(
            "ContradictionDetector: %r superseded by %r (sim=%.3f, +%.0f days)",
            session_id,
            self._session_ids[newest_idx],
            float(sims[newest_idx]),
            age_gap_days,
        )

        return SupersessionResult(
            is_superseded           = True,
            superseded_by           = self._session_ids[newest_idx],
            superseded_by_stored_at = float(self._stored_ats[newest_idx]),
            superseded_by_summary   = self._summaries[newest_idx],
            supersession_similarity = float(sims[newest_idx]),
            age_gap_days            = age_gap_days,
        )

    def check_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, SupersessionResult]:
        """Check multiple session IDs in one call. Returns a dict keyed by session_id."""
        return {sid: self.check(sid) for sid in session_ids}


# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------

def supersession_preamble(result: SupersessionResult) -> str:
    """
    Return a one-line preamble to prepend to a superseded episode's injection.

    Example output:
        "[POSSIBLY OUTDATED -- a newer memory (3 weeks later) covers this topic]"
    """
    if not result.is_superseded:
        return ""
    return (
        f"[POSSIBLY OUTDATED -- a newer memory ({result.age_gap_str} later, "
        f"similarity={result.supersession_similarity:.2f}) covers this same topic. "
        f"Prefer the newer context if they conflict.]"
    )


def newer_episode_note(result: SupersessionResult) -> str:
    """
    Return a brief note summarising the superseding episode.

    Used when you want to inject both the old and new episode.
    """
    if not result.is_superseded or not result.superseded_by_summary:
        return ""
    import datetime
    ts = datetime.datetime.fromtimestamp(result.superseded_by_stored_at).strftime("%Y-%m-%d")
    return (
        f"[Newer memory -- {ts}  similarity={result.supersession_similarity:.2f}]\n"
        f"{result.superseded_by_summary.strip()}"
    )
