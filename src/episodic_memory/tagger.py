"""
EpisodicTagger -- lightweight heuristic auto-tagger for episodic memories.

Zero LLM cost: all tagging is regex/keyword matching against the episode
summary text (and optionally metadata fields).

Tags are used for:
  1. Filtered recall -- exclude or require specific tag classes
  2. TTL/expiry -- speculation + date_sensitive entries auto-expire
  3. Staleness detection -- hardware/config entries flagged for review

Tag vocabulary
──────────────
  hardware        GPU, V100, baseboard, NVMe, PCIe, server rack, heatsink, ...
  speculation     "arriving", "ETA", "expected", "should be", "ordered", "when it"
  person          named individuals (Richard, Charlie, Sally, Aura, Ava, ...)
  project         CTM, TheCog, Anima, substrate, Aura training, Kinect, XMOS, ...
  config          port numbers, file paths, env vars, docker, nginx, systemd
  date_sensitive  explicit dates, version numbers (e.g. v1.2, 2026-05-xx)
  roleplay        intimate/fictional framing (uses RoleplayFilter)
  completed       task explicitly marked done / complete / committed
  error           exception, traceback, bug, crash, OOM, failed

TTL rules (applied at encode time):
  speculation  → expires_at = stored_at + 30 days
  date_sensitive with speculation → same
  hardware     → expires_at = stored_at + 90 days   (hardware changes, but slowly)
  All others   → expires_at = None (no expiry)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional


# ── Tag constants ──────────────────────────────────────────────────────────────

TAG_HARDWARE      = "hardware"
TAG_SPECULATION   = "speculation"
TAG_PERSON        = "person"
TAG_PROJECT       = "project"
TAG_CONFIG        = "config"
TAG_DATE_SENSITIVE = "date_sensitive"
TAG_ROLEPLAY      = "roleplay"
TAG_COMPLETED     = "completed"
TAG_ERROR         = "error"

ALL_TAGS = [
    TAG_HARDWARE, TAG_SPECULATION, TAG_PERSON, TAG_PROJECT,
    TAG_CONFIG, TAG_DATE_SENSITIVE, TAG_ROLEPLAY, TAG_COMPLETED, TAG_ERROR,
]

# TTL in seconds for auto-expiring tags
_TTL_SPECULATION   = 30  * 24 * 3600   # 30 days
_TTL_HARDWARE      = 90  * 24 * 3600   # 90 days


# ── Pattern definitions ────────────────────────────────────────────────────────

_HARDWARE_PAT = re.compile(
    r"\b(gpu|v100|v[0-9]+\s*sxm|baseboard|nvlink|nvme|pcie|"
    r"server\s*rack|heatsink|vram|cuda\s*device|driver|nvidia|"
    r"memory\s*bandwidth|thermal|cooling|fan|shroud|card\s*install|"
    r"rack\s*move|4090|3080|sxm2|v100\s*sxm)\b",
    re.IGNORECASE,
)

_SPECULATION_PAT = re.compile(
    r"\b(arriving|eta\b|expected\s+to|should\s+(be|arrive|work|fix)|"
    r"ordered\b|when\s+it\s+(arrives?|comes?)|not\s+yet|pending|"
    r"todo\b|to\s+do\b|planned|upcoming|will\s+be\s+done|"
    r"next\s+step|deferred|in\s+progress)\b",
    re.IGNORECASE,
)

_PERSON_PAT = re.compile(
    # Generic first-name list -- extend this with names relevant to your agent
    # and the people it regularly talks to or about.
    r"\b(alice|bob|charlie|dave|eve|frank|grace|heidi|"
    r"ivan|judy|mallory|oscar|peggy|trent|victor|wendy)\b",
    re.IGNORECASE,
)

_PROJECT_PAT = re.compile(
    r"\b(episodic.memory|recall.engine|workspace|reservoir|"
    r"attractor|axolotl|qlora|lora\b|vllm|"
    r"llamacpp|memgraph|knowledge.graph|coral\b|gaze|visual.cortex)\b",
    re.IGNORECASE,
)

_CONFIG_PAT = re.compile(
    r"("
    r"port\s+\d{4,5}|"                          # port 8001, port 5050
    r"/home/[^/\s]+/[^\s,]+|"                   # file paths (/home/<user>/...)
    r"localhost:\d{4,5}|"                       # localhost:5050
    r"\b(nginx|docker|systemd|compose|env\s+var|"
    r"config\.yaml|\.service|\.conf|\.env|"
    r"CUDA_VISIBLE_DEVICES|PYENV_VERSION|"
    r"proxy_pass|server\.py|docker-compose)\b"
    r")",
    re.IGNORECASE,
)

_DATE_SENSITIVE_PAT = re.compile(
    r"("
    r"202[0-9]-[01][0-9]-[0-3][0-9]|"          # ISO date
    r"v\d+\.\d+(\.\d+)?|"                       # version numbers
    r"checkpoint-\d+|"                           # checkpoint refs
    r"commit\s+[0-9a-f]{7,}|"                   # git commits
    r"epoch\s+\d+|"                             # training epoch refs
    r"step\s+\d+\b"                             # training step refs
    r")",
    re.IGNORECASE,
)

_COMPLETED_PAT = re.compile(
    r"\b(complete[d]?|committed|merged|pushed|deployed|"
    r"passing|✅|done\b|finished|resolved|fixed)\b",
    re.IGNORECASE,
)

_ERROR_PAT = re.compile(
    r"\b(exception|traceback|error\b|crash|oom|"
    r"out\s+of\s+memory|failed|failure|bug\b|"
    r"segfault|killed|exit\s+code\s+[^0])\b",
    re.IGNORECASE,
)


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class TagResult:
    """Result of tagging a single episode."""
    tags:       list[str]       = field(default_factory=list)
    expires_at: Optional[float] = None   # Unix timestamp, or None (no expiry)

    def to_dict(self) -> dict:
        return {"tags": self.tags, "expires_at": self.expires_at}

    @classmethod
    def empty(cls) -> "TagResult":
        return cls(tags=[], expires_at=None)


# ── Tagger ─────────────────────────────────────────────────────────────────────

class EpisodicTagger:
    """
    Heuristic auto-tagger for episodic memory summaries.

    Usage::

        tagger = EpisodicTagger()
        result = tagger.tag(summary="V100 cards arrived today, installed on GPU2/3")
        # result.tags == ["hardware", "date_sensitive"]
        # result.expires_at == stored_at + 90 days

    Args:
        use_roleplay_filter: if True, import RoleplayFilter and tag intimate
                             episodes with TAG_ROLEPLAY.
        stored_at:           default base timestamp for TTL calculation
                             (defaults to time.time() if not provided on .tag() call).
    """

    def __init__(self, use_roleplay_filter: bool = True) -> None:
        self._rp_filter = None
        if use_roleplay_filter:
            try:
                from episodic_memory.roleplay_filter import RoleplayFilter
                self._rp_filter = RoleplayFilter()
            except ImportError:
                pass

    def tag(
        self,
        summary:   str,
        stored_at: Optional[float] = None,
        metadata:  Optional[dict]  = None,
    ) -> TagResult:
        """
        Tag a single episode from its summary text.

        Args:
            summary:   LLM-generated episode summary (or raw transcript excerpt).
            stored_at: Unix timestamp of when the episode was stored.
                       Used for TTL calculation. Defaults to now.
            metadata:  Optional hot-metadata dict for additional signal
                       (e.g. dominant_emotion, turn_count).

        Returns:
            TagResult with tags list and optional expires_at timestamp.
        """
        if stored_at is None:
            stored_at = time.time()

        tags: list[str] = []
        text = summary or ""

        # ── Heuristic matches ──────────────────────────────────────────────────
        if _HARDWARE_PAT.search(text):
            tags.append(TAG_HARDWARE)

        if _SPECULATION_PAT.search(text):
            tags.append(TAG_SPECULATION)

        if _PERSON_PAT.search(text):
            tags.append(TAG_PERSON)

        if _PROJECT_PAT.search(text):
            tags.append(TAG_PROJECT)

        if _CONFIG_PAT.search(text):
            tags.append(TAG_CONFIG)

        if _DATE_SENSITIVE_PAT.search(text):
            tags.append(TAG_DATE_SENSITIVE)

        if _COMPLETED_PAT.search(text):
            tags.append(TAG_COMPLETED)

        if _ERROR_PAT.search(text):
            tags.append(TAG_ERROR)

        # ── Roleplay filter ────────────────────────────────────────────────────
        if self._rp_filter is not None and self._rp_filter.is_roleplay(text):
            tags.append(TAG_ROLEPLAY)

        # ── TTL calculation ────────────────────────────────────────────────────
        expires_at = self._compute_ttl(tags, stored_at)

        return TagResult(tags=sorted(set(tags)), expires_at=expires_at)

    def tag_batch(
        self,
        summaries:  list[str],
        stored_ats: Optional[list[float]] = None,
        metadatas:  Optional[list[dict]]  = None,
    ) -> list[TagResult]:
        """Tag a batch of summaries. Returns results in same order."""
        if stored_ats is None:
            stored_ats = [time.time()] * len(summaries)
        if metadatas is None:
            metadatas = [{}] * len(summaries)

        return [
            self.tag(s, st, m)
            for s, st, m in zip(summaries, stored_ats, metadatas)
        ]

    @staticmethod
    def _compute_ttl(tags: list[str], stored_at: float) -> Optional[float]:
        """
        Compute expires_at from tag set.

        Rules (shortest TTL wins when multiple apply):
          speculation  → stored_at + 30 days
          hardware     → stored_at + 90 days
          everything else → None (no expiry)
        """
        if TAG_SPECULATION in tags:
            return stored_at + _TTL_SPECULATION
        if TAG_HARDWARE in tags:
            return stored_at + _TTL_HARDWARE
        return None

    def is_expired(self, expires_at: Optional[float], now: Optional[float] = None) -> bool:
        """Return True if expires_at is set and in the past."""
        if expires_at is None:
            return False
        return (now or time.time()) > expires_at
