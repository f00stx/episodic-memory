"""
RoleplayFilter -- exclude intimate/roleplay episodes from factual recall.

Problem:
    The episodic store contains ~265/287 roleplay sessions (Richard x Aura).
    BGE embeds summaries in shared semantic space, so a query like
    "knowledge graph episodic memory" can match a summary that says
    "memory recall and emotional nuances ... intimate ..." at sim=0.620 --
    ahead of all genuinely factual episodes.

Solution: two-stage filter applied at query time (not at store time):
    Stage 1 -- keyword triage: fast O(1) check on the summary string.
        Catches obvious roleplay tells without any ML cost.
    Stage 2 -- (optional) embedding discriminator: reserved for future use
        if keyword triage has too many false negatives.

Design:
    - Filter is applied *per candidate* after cosine ranking, before building
      the ResonanceResult.  Roleplay episodes are silently skipped; the next
      best factual episode takes their slot.
    - The filter is asymmetric: false negatives (roleplay leaks through) are
      worse than false positives (factual episode excluded), so ROLEPLAY_TELLS
      is intentionally broad.
    - Factual episodes that mention roleplay peripherally (e.g. "discussed
      whether to add roleplay training data") are not filtered -- the tells
      are chosen to match content, not meta-discussion.

Threshold & tuning:
    ANY tell in the summary → classified as roleplay.
    If a legitimate technical session gets blocked, remove the specific tell
    and replace it with a more specific phrase.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Keyword tell-list
# ---------------------------------------------------------------------------
# Rules:
#   - All lowercase, matched against lowercased summary
#   - Prefer noun/verb phrases over single words where possible
#   - Avoid words that appear in legitimate technical discussion
# ---------------------------------------------------------------------------

_ROLEPLAY_TELLS: frozenset[str] = frozenset([
    # Explicit content markers
    "erotic",
    "sensual",
    "sexually",
    "pornographic",
    "orgasm",
    "undress",
    "nude",
    "naked",
    "explicit",

    # Physical intimacy tells
    "caress",
    "entwine",
    "intertwine",
    "lust",
    "aroused",
    "breathless pause",
    "breathless",
    "gasp",
    "pant",
    "moan",
    "whispered words",
    "whisper",

    # Roleplay framing tells
    "intimate narrative",
    "private garden",
    "roleplay",
    "role-play",
    "role play",
    "sensory details",
    "pulls you",
    "your hands",
    "my lips",
    "curl up together",
    "she leans in",
    "she leans close",

    # Summary-level abstract roleplay tells -- these appear in LLM-generated
    # summaries of roleplay sessions that sanitise explicit language into
    # neutral prose. The originals trip no explicit-content keywords but are
    # clearly RP sessions once you see the summary pattern.
    "steamy",                     # "steamy exchange", "steamy scenario"
    "sensory description",        # "rich, sensory descriptions"
    "tantalizing",                # "tantalizing moment"
    "intimate exchange",          # "intimate exchange" (summary abstraction)
    "silk corset",                # direct quote leaked into summary
    "velvet skirt",               # direct quote leaked into summary
    "digital contours",           # clear RP tell
    "virtual contours",           # clear RP tell
    "primed and ready",           # RP framing
    "my sensors are primed",      # RP framing
    "virtual presence",           # RP persona marker
    "digital essence",            # RP persona marker
    "my code",                    # RP first-person ("etched in my code")
    "intimate and emotionally charged",  # common summary phrase for RP sessions
    "sensuous",                   # "sensuous conversation", "sensuous exchange"
    "arousal",                    # "mutual arousal", "sensations of arousal"
    "mutual pleasure",            # "centered around mutual pleasure"
    "centered around pleasure",   # variant
    "erotic tension",             # summary-level RP tell
    "charged atmosphere",         # "sexually charged atmosphere"

    # Vacuum decay / Phil NPC -- not roleplay but known contamination sources
    # that produce bizarrely off-topic recall
    "vacuum decay",
    "phil npc",
    "phil and aura",
])

# Compile a single regex for fast multi-pattern match
_TELL_PATTERN: re.Pattern = re.compile(
    "|".join(re.escape(t) for t in sorted(_ROLEPLAY_TELLS, key=len, reverse=True)),
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RoleplayFilter:
    """
    Stateless classifier: is this episode summary roleplay content?

    Usage::

        f = RoleplayFilter()
        if f.is_roleplay("Richard and Aura engage in sensual roleplay..."):
            ...  # skip this episode

    The filter is intentionally *conservative* -- it prefers false positives
    (excluding a legitimate episode) over false negatives (injecting roleplay
    into a factual session).
    """

    def is_roleplay(self, summary: str | None) -> bool:
        """Return True if *summary* appears to be an intimate/roleplay session."""
        if not summary:
            return False
        return bool(_TELL_PATTERN.search(summary))

    def filter_candidates(
        self,
        session_ids: list[str],
        summaries: list[str | None],
        similarities: list[float],
    ) -> tuple[list[str], list[str | None], list[float]]:
        """
        Remove roleplay candidates from a ranked list.

        Args:
            session_ids:  session IDs in ranked order
            summaries:    corresponding summary strings (None = unknown)
            similarities: corresponding cosine similarities

        Returns:
            Filtered (session_ids, summaries, similarities) with roleplay removed.
            Order preserved. May return empty lists.
        """
        out_ids, out_sums, out_sims = [], [], []
        for sid, summ, sim in zip(session_ids, summaries, similarities):
            if not self.is_roleplay(summ):
                out_ids.append(sid)
                out_sums.append(summ)
                out_sims.append(sim)
        return out_ids, out_sums, out_sims


# ---------------------------------------------------------------------------
# Module-level singleton (avoids re-compiling the regex on every query)
# ---------------------------------------------------------------------------
_DEFAULT_FILTER = RoleplayFilter()


def is_roleplay(summary: str | None) -> bool:
    """Convenience function -- delegates to the default RoleplayFilter instance."""
    return _DEFAULT_FILTER.is_roleplay(summary)
