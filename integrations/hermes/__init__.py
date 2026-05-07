"""Episodic Memory provider plugin for Hermes Agent.

Wraps the standalone ``episodic-memory`` library (github: f00stx/episodic-memory)
as a Hermes MemoryProvider.  Provides:

  * Semantic recall injected into the system prompt before each turn
    (``prefetch`` / ``queue_prefetch`` pattern -- no turn latency)
  * Supersession annotations -- stale memories flagged with "[POSSIBLY OUTDATED]"
  * Emotional resonance -- fast-path affective colouring without a full recall hit
  * ``episodic_recall`` tool -- deliberate agent-triggered recall

Store layout (agent-scoped, never shared):
    $HERMES_HOME/episodic_memory/<agent_name>/
        episodes.db          ← cold SQLite tier
        hot_metadata.json    ← hot JSON tier

The ``$HERMES_HOME`` path is provided by Hermes at ``initialize()`` time, so each
profile gets its own isolated store automatically.  No hardcoded paths.

Configuration (config.yaml):
    memory:
      provider: episodic_memory
      episodic_memory:
        store_path: ~/.ctm/memory/aura          # optional override; default = $HERMES_HOME/episodic_memory/<profile>
        embedding_model: BAAI/bge-large-en-v1.5 # model name or absolute local path
        recall_threshold: 0.55                  # minimum cosine similarity to inject
        filter_roleplay: true                   # exclude RP/fiction from factual recall
        agent_name: aura                        # store subdirectory name (defaults to profile name)

Dependencies (must be available in Hermes venv):
    pip install git+https://github.com/f00stx/episodic-memory
    # or: pip install /home/richard/projects/episodic-memory
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    "name": "episodic_recall",
    "description": (
        "Search episodic memory for context relevant to the current conversation. "
        "Episodes are semantic snapshots of past sessions, filtered to remove roleplay "
        "and annotated when a newer memory supersedes an older one. "
        "Use this when the prefetched context doesn't cover a topic the user is referencing, "
        "or when you want to check whether something has changed since a previous session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "What to search for -- a topic, question, or phrase. "
                    "Natural language works best; the engine uses semantic similarity."
                ),
            },
        },
        "required": ["query"],
    },
}

ALL_TOOL_SCHEMAS = [RECALL_SCHEMA]


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class EpisodicMemoryProvider(MemoryProvider):
    """Hermes memory provider backed by the episodic-memory library.

    Lifecycle:
        initialize()        -- resolve store path, record config (engine lazy-loads)
        prefetch()          -- return pre-warmed recall result (from queue_prefetch)
        queue_prefetch()    -- background semantic query for next turn
        system_prompt_block() -- static description (engine availability check)
        sync_turn()         -- no-op (episodic store is read-only from plugin side;
                               sessions are encoded offline via encode_*_memories.py)
        on_session_end()    -- no-op (same reason)
        shutdown()          -- join prefetch thread
    """

    def __init__(self) -> None:
        self._store_path: Optional[Path] = None
        self._embedding_model: str = "BAAI/bge-large-en-v1.5"
        self._recall_threshold: float = 0.55
        self._filter_roleplay: bool = True
        self._active: bool = False

        # Lazy-loaded RecallEngine (BGE model load is ~2s -- defer to first query)
        self._engine: Optional[Any] = None  # episodic_memory.RecallEngine
        self._engine_lock = threading.Lock()

        # Prefetch cache: queue_prefetch → prefetch
        self._prefetch_result: str = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None

    # -- Identity ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "episodic_memory"

    # -- Availability --------------------------------------------------------

    def is_available(self) -> bool:
        """Check episodic-memory package is importable. No network calls."""
        try:
            import episodic_memory  # noqa: F401
            return True
        except ImportError:
            logger.debug("episodic_memory package not installed -- provider inactive")
            return False

    # -- Lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        """Resolve store path from config or Hermes home, record settings."""
        try:
            hermes_home = kwargs.get("hermes_home") or str(Path.home() / ".hermes")
            config: Dict[str, Any] = kwargs.get("config", {}) or {}

            # Determine store path: explicit config → $HERMES_HOME/episodic_memory/<agent>
            if config.get("store_path"):
                self._store_path = Path(config["store_path"]).expanduser().resolve()
            else:
                # Derive agent name from profile name (hermes_home ends in profiles/<name>)
                agent_name = config.get("agent_name") or Path(hermes_home).name
                self._store_path = Path(hermes_home) / "episodic_memory" / agent_name

            self._embedding_model = config.get("embedding_model", self._embedding_model)
            self._recall_threshold = float(config.get("recall_threshold", self._recall_threshold))
            self._filter_roleplay = bool(config.get("filter_roleplay", self._filter_roleplay))

            # Verify store exists (db + hot tier)
            db_path = self._store_path / "episodes.db"
            hot_path = self._store_path / "hot_metadata.json"

            if not db_path.exists() or not hot_path.exists():
                logger.warning(
                    "Episodic memory store incomplete at %s "
                    "(expected episodes.db + hot_metadata.json) -- provider inactive. "
                    "Run encode_*_memories.py to build the store.",
                    self._store_path,
                )
                return

            self._active = True
            logger.info(
                "Episodic memory provider active: store=%s, model=%s, threshold=%.2f",
                self._store_path, self._embedding_model, self._recall_threshold,
            )

        except Exception as e:
            logger.warning("Episodic memory provider init failed: %s", e)

    def system_prompt_block(self) -> str:
        if not self._active:
            return ""
        engine = self._get_engine()
        if engine is None:
            return ""
        return (
            f"# Episodic Memory\n"
            f"Active. {engine.n_episodes} episodes indexed. "
            "Relevant context is automatically surfaced before each response. "
            "Use episodic_recall to deliberately search for a topic -- "
            "especially when something feels like it's come up before or when "
            "the user references a past event."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return pre-warmed recall result from the last queue_prefetch call."""
        if not self._active:
            return ""
        # Wait briefly for background thread if still running
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Episodic Memory Context\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Fire background semantic recall for the next turn."""
        if not self._active or not query:
            return

        engine = self._get_engine()
        if engine is None:
            return

        def _run():
            try:
                result = engine.query(query.strip()[:400])
                if result is not None:
                    with self._prefetch_lock:
                        self._prefetch_result = result.context_injection()
            except Exception as e:
                logger.debug("Episodic prefetch failed: %s", e)

        # Wait for previous prefetch to finish before starting a new one
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=1.0)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="episodic-prefetch"
        )
        self._prefetch_thread.start()

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        """No-op -- episodic store is built offline via encode_*_memories.py."""
        pass

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """No-op -- sessions are encoded in batch, not streamed live."""
        pass

    # -- Tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return ALL_TOOL_SCHEMAS if self._active else []

    def handle_tool_call(
        self, tool_name: str, args: Dict[str, Any], **kwargs
    ) -> str:
        if tool_name != "episodic_recall":
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        query = (args.get("query") or "").strip()
        if not query:
            return json.dumps({"error": "query is required"})

        engine = self._get_engine()
        if engine is None:
            return json.dumps({"error": "Episodic memory engine not available"})

        try:
            result = engine.query(query[:400])
            if result is None:
                return json.dumps({
                    "result": "No relevant episodic memory found for that query.",
                    "similarity": 0.0,
                })
            return json.dumps({
                "result": result.context_injection(),
                "similarity": float(result.similarity),
                "dominant_emotion": result.dominant_emotion,
                "is_superseded": result.is_superseded,
            })
        except Exception as e:
            logger.warning("Episodic recall tool error: %s", e)
            return json.dumps({"error": str(e)})

    # -- Config schema -------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "store_path",
                "description": (
                    "Path to the episodic memory store directory "
                    "(must contain episodes.db + hot_metadata.json). "
                    "Leave blank to use $HERMES_HOME/episodic_memory/<profile_name>/"
                ),
                "required": False,
                "default": "",
            },
            {
                "key": "embedding_model",
                "description": (
                    "BGE model name or absolute local path. "
                    "Default: BAAI/bge-large-en-v1.5. "
                    "Use a local path (e.g. /models/huggingface/hub/models--BAAI--bge-large-en-v1.5) "
                    "to avoid network fetches."
                ),
                "required": False,
                "default": "BAAI/bge-large-en-v1.5",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        config_path = Path(hermes_home) / "episodic_memory.json"
        config_path.write_text(json.dumps(values, indent=2))
        logger.debug("Episodic memory config saved to %s", config_path)

    # -- Shutdown ------------------------------------------------------------

    def shutdown(self) -> None:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)

    # -- Internal ------------------------------------------------------------

    def _get_engine(self) -> Optional[Any]:
        """Lazy-load RecallEngine (BGE model load ~2s, done once)."""
        if self._engine is not None:
            return self._engine
        if not self._active or self._store_path is None:
            return None

        with self._engine_lock:
            if self._engine is not None:
                return self._engine
            try:
                from episodic_memory import RecallEngine
                self._engine = RecallEngine(
                    store_path=str(self._store_path),
                    recall_threshold=self._recall_threshold,
                    filter_roleplay=self._filter_roleplay,
                    embedding_device="cpu",  # don't compete with main LLM for VRAM
                    embedding_model=self._embedding_model,
                )
                logger.info(
                    "Episodic RecallEngine loaded: %d episodes",
                    self._engine.n_episodes,
                )
            except Exception as e:
                logger.warning("RecallEngine init failed: %s", e)
                self._engine = None

        return self._engine


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Called by Hermes plugin loader."""
    ctx.register_memory_provider(EpisodicMemoryProvider())