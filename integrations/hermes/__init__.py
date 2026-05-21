"""Episodic Memory provider plugin for Hermes Agent.

Wraps the standalone ``episodic-memory`` library (github: f00stx/episodic-memory)
as a Hermes MemoryProvider.  Provides:

  * Semantic recall injected into the system prompt before each turn
    (``prefetch`` / ``queue_prefetch`` pattern -- no turn latency)
  * Supersession annotations -- stale memories flagged with "[POSSIBLY OUTDATED]"
  * Emotional resonance -- fast-path affective colouring without a full recall hit
  * ``episodic_recall`` tool -- deliberate agent-triggered recall
  * Live encoding -- turns buffered in sync_turn() and flushed to the store
    every ``flush_min_turns`` turns and at session end, matching Hermes memory
    management conventions (nudge_interval / flush_min_turns from config.yaml)

Store layout (agent-scoped, never shared):
    $HERMES_HOME/episodic_memory/<agent_name>/
        episodes.db          <- cold SQLite tier
        hot_metadata.json    <- hot JSON tier (created on first flush if absent)

The ``$HERMES_HOME`` path is provided by Hermes at ``initialize()`` time, so each
profile gets its own isolated store automatically.  No hardcoded paths.

Configuration (config.yaml):
    memory:
      provider: episodic_memory
      nudge_interval: 10          # turns between memory-use reminders (Hermes native)
      flush_min_turns: 6          # turns to buffer before encoding to store (Hermes native)
      episodic_memory:
        store_path: ~/.ctm/memory/aura          # optional override
        embedding_model: BAAI/bge-large-en-v1.5
        recall_threshold: 0.55
        filter_roleplay: true
        agent_name: aura                        # store subdirectory (defaults to profile name)

Dependencies (must be available in Hermes venv):
    pip install episodic-memory
    # or: pip install /home/richard/projects/episodic-memory
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

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
        sync_turn()         -- buffer turn; flush to store every flush_min_turns turns
        on_session_end()    -- force-flush remaining buffered turns then join threads
        shutdown()          -- join all background threads
    """

    # Minimum number of turns to buffer before flush is considered worthwhile.
    # Overridden by config flush_min_turns at initialize() time.
    _DEFAULT_FLUSH_MIN_TURNS: int = 6

    def __init__(self) -> None:
        self._store_path: Optional[Path] = None
        self._embedding_model: str = "BAAI/bge-large-en-v1.5"
        self._recall_threshold: float = 0.55
        self._filter_roleplay: bool = True

        # True once store is confirmed to exist OR has been successfully created.
        self._active: bool = False

        # Flush config -- read from Hermes memory block at initialize() time.
        self._flush_min_turns: int = self._DEFAULT_FLUSH_MIN_TURNS

        # Current session ID assigned at initialize() -- used as the episode key.
        self._session_id: str = ""

        # Turn buffer: list of {"role": "user"|"assistant", "content": str}
        self._turn_buffer: List[Dict[str, str]] = []
        self._turn_buffer_lock = threading.Lock()
        self._turns_since_flush: int = 0

        # Lazy-loaded RecallEngine (BGE model load is ~2s -- defer to first query)
        self._engine: Optional[Any] = None
        self._engine_lock = threading.Lock()

        # Lazy-loaded write-side objects (EpisodicMemoryStore + EpisodicEncoder).
        # These are only needed for flushing, not for recall.
        self._store: Optional[Any] = None      # episodic_memory.EpisodicMemoryStore
        self._encoder: Optional[Any] = None    # episodic_memory.EpisodicEncoder
        self._write_lock = threading.Lock()

        # Background thread pool: prefetch + flush
        self._prefetch_result: str = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._flush_thread: Optional[threading.Thread] = None

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

            # Hermes passes flush_min_turns at the top-level memory config block.
            # Fall back to our own default if not provided.
            mem_config: Dict[str, Any] = kwargs.get("memory_config", {}) or {}
            self._flush_min_turns = int(
                mem_config.get("flush_min_turns", config.get("flush_min_turns", self._DEFAULT_FLUSH_MIN_TURNS))
            )

            # Determine store path: explicit config -> $HERMES_HOME/episodic_memory/<agent>
            if config.get("store_path"):
                self._store_path = Path(config["store_path"]).expanduser().resolve()
            else:
                agent_name = config.get("agent_name") or Path(hermes_home).name
                self._store_path = Path(hermes_home) / "episodic_memory" / agent_name

            self._embedding_model = config.get("embedding_model", self._embedding_model)
            self._recall_threshold = float(config.get("recall_threshold", self._recall_threshold))
            self._filter_roleplay = bool(config.get("filter_roleplay", self._filter_roleplay))

            # Store the session ID for episode keying.
            self._session_id = session_id or str(uuid.uuid4())

            # Ensure store directory exists (we create episodes on flush).
            self._store_path.mkdir(parents=True, exist_ok=True)

            # Mark active if recall store exists OR we can create it (i.e. dir is writable).
            db_path = self._store_path / "episodes.db"
            hot_path = self._store_path / "hot_metadata.json"
            store_ready = db_path.exists() and hot_path.exists()

            self._active = True  # Always active once store dir is available.

            if store_ready:
                logger.info(
                    "Episodic memory provider active (recall+encode): store=%s, model=%s, "
                    "threshold=%.2f, flush_min_turns=%d",
                    self._store_path, self._embedding_model, self._recall_threshold,
                    self._flush_min_turns,
                )
            else:
                logger.info(
                    "Episodic memory provider active (encode-only, store empty): store=%s -- "
                    "recall will activate after first flush. flush_min_turns=%d",
                    self._store_path, self._flush_min_turns,
                )

        except Exception as e:
            logger.warning("Episodic memory provider init failed: %s", e)

    def system_prompt_block(self) -> str:
        if not self._active:
            return ""
        engine = self._get_engine()
        if engine is None:
            # Store is empty / not yet built -- still worth mentioning encode is live.
            return (
                "# Episodic Memory\n"
                "Active (encode-only). No episodes indexed yet -- "
                "memories will accumulate as the session progresses."
            )
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
            return  # Store not yet built -- skip silently.

        def _run():
            try:
                result = engine.query(query.strip()[:400])
                if result is not None:
                    with self._prefetch_lock:
                        self._prefetch_result = result.context_injection()
            except Exception as e:
                logger.debug("Episodic prefetch failed: %s", e)

        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=1.0)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="episodic-prefetch"
        )
        self._prefetch_thread.start()

    # Hermes-internal scaffolding prefixes that should never be stored as
    # episodic content.  These are compression prompts, context-compaction
    # markers, and summarisation instructions injected by the framework itself.
    _SYSTEM_TURN_PREFIXES: tuple = (
        "Review the conversation above and consider saving",
        "Please summarize the conversation",
        "[CONTEXT COMPACTION",
        "Conversation summary:",
    )

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        """Buffer this turn; flush to store when flush_min_turns is reached."""
        if not self._active:
            return

        # Drop Hermes compression/summarisation scaffold turns -- they are
        # internal framework messages, not real conversation content, and will
        # corrupt episode summaries if stored.
        if user_content and any(
            user_content.startswith(p) for p in self._SYSTEM_TURN_PREFIXES
        ):
            logger.debug("sync_turn: skipping system scaffold turn")
            return

        with self._turn_buffer_lock:
            if user_content:
                self._turn_buffer.append({"role": "user", "content": user_content})
            if assistant_content:
                self._turn_buffer.append({"role": "assistant", "content": assistant_content})
            self._turns_since_flush += 1
            should_flush = (
                self._flush_min_turns > 0
                and self._turns_since_flush >= self._flush_min_turns
            )

        if should_flush:
            self._trigger_flush(force=False)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Force-flush any remaining buffered turns and wait for completion."""
        if not self._active:
            return
        # Wait for any in-progress flush to complete first.
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=30.0)
        # Flush remaining buffer (may be < flush_min_turns -- flush anyway).
        self._trigger_flush(force=True)
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=60.0)

    # -- Tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # Even when _active=False, return non-empty list to satisfy Hermes schema check
        # The tool itself will no-op. This prevents schema drift in multi-agent workflows.
        return ALL_TOOL_SCHEMAS # if self._active else []

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
            return json.dumps({
                "result": "Episodic memory store is empty -- no episodes indexed yet.",
                "similarity": 0.0,
            })

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
                    "Path to the episodic memory store directory. "
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
                    "Use a local path to avoid network fetches."
                ),
                "required": False,
                "default": "BAAI/bge-large-en-v1.5",
            },
            {
                "key": "flush_min_turns",
                "description": (
                    "Number of turns to buffer before encoding to the episodic store. "
                    "Set to 0 to disable live encoding (read-only mode). Default: 6."
                ),
                "required": False,
                "default": 6,
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
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=30.0)

    # -- Internal: flush pipeline --------------------------------------------

    def _trigger_flush(self, force: bool = False) -> None:
        """Snapshot current buffer and schedule a background encode+store."""
        with self._turn_buffer_lock:
            if not self._turn_buffer:
                return
            # If not forcing, check threshold again under lock.
            if not force and self._turns_since_flush < self._flush_min_turns:
                return
            snapshot = list(self._turn_buffer)
            self._turn_buffer.clear()
            self._turns_since_flush = 0

        if not snapshot:
            return

        session_id = self._session_id
        store_path = self._store_path

        def _run():
            try:
                self._encode_and_store(snapshot, session_id, store_path)
            except Exception as e:
                logger.warning("Episodic flush failed: %s", e, exc_info=True)

        # Don't pile up flush threads -- wait for previous one first.
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=30.0)

        self._flush_thread = threading.Thread(
            target=_run, daemon=True, name="episodic-flush"
        )
        self._flush_thread.start()

    def _encode_and_store(
        self,
        turns: List[Dict[str, str]],
        session_id: str,
        store_path: Path,
    ) -> None:
        """Embed turns with BGE, encode with EpisodicEncoder, write to store.

        This runs in a background thread -- never call from the main agent loop.
        """
        import numpy as np

        # Separate user and assistant turns.
        user_turns = [t["content"] for t in turns if t["role"] == "user"]
        agent_turns = [t["content"] for t in turns if t["role"] == "assistant"]

        if not user_turns:
            logger.debug("Episodic flush: no user turns to encode, skipping.")
            return

        # Pad so both lists are the same length (encoder expects paired turns).
        max_t = max(len(user_turns), len(agent_turns))
        user_turns += [""] * (max_t - len(user_turns))
        agent_turns += [""] * (max_t - len(agent_turns))

        embed_client = self._get_embed_client()
        if embed_client is None:
            logger.warning("Episodic flush: embed client unavailable, skipping.")
            return

        # Embed all turns in two batches (one shot each, no for-loop).
        user_embs = embed_client.embed(user_turns)    # (T, D)
        agent_embs = embed_client.embed(agent_turns)  # (T, D)

        # If embeddings are 1024-dim (bge-small), pad to 1536 to match encoder input_dim.
        target_dim = 1536
        if user_embs.shape[1] < target_dim:
            pad = target_dim - user_embs.shape[1]
            user_embs = np.pad(user_embs, ((0, 0), (0, pad)))
            agent_embs = np.pad(agent_embs, ((0, 0), (0, pad)))
        elif user_embs.shape[1] > target_dim:
            user_embs = user_embs[:, :target_dim]
            agent_embs = agent_embs[:, :target_dim]

        encoder = self._get_encoder()
        if encoder is None:
            logger.warning("Episodic flush: encoder unavailable, skipping.")
            return

        latent, coherence = encoder.encode_numpy(
            user_embs.astype(np.float32),
            agent_embs.astype(np.float32),
        )

        # Build a compact plain-text summary (no LLM needed).
        summary = self._make_summary(turns)

        store = self._get_store(store_path)
        if store is None:
            logger.warning("Episodic flush: store unavailable, skipping.")
            return

        with self._write_lock:
            store.add(
                session_id=session_id,
                latent=latent,
                transcript=turns,
                metadata={
                    "turn_count": len(turns),
                    "coherence": float(coherence) if coherence is not None else None,
                    "flush_partial": True,  # flagged as a within-session flush
                    # emotion_cats required by resonance.py for blend computation.
                    # Zeros = neutral; a future encoder upgrade can populate real values.
                    "emotion_cats": [0.0] * 8,
                    "dominant_emotion": "neutral",
                    "dominant_archetype": "companion",
                },
                summary=summary,
            )

        # Invalidate the recall engine so it reloads with new episode next query.
        with self._engine_lock:
            self._engine = None

        logger.info(
            "Episodic flush: stored %d turns for session %s (coherence=%.3f)",
            len(turns), session_id, float(coherence) if coherence is not None else -1,
        )

    @staticmethod
    def _make_summary(turns: List[Dict[str, str]], max_chars: int = 800) -> str:
        """Build a compact plain-text digest without an LLM call.

        Format: interleaved "U: ..." / "A: ..." lines, truncated to max_chars.
        """
        lines = []
        prefix_map = {"user": "U", "assistant": "A"}
        for t in turns:
            role = prefix_map.get(t["role"], t["role"][0].upper())
            snippet = t["content"].strip().replace("\n", " ")[:160]
            lines.append(f"{role}: {snippet}")
        digest = "\n".join(lines)
        if len(digest) > max_chars:
            digest = digest[:max_chars].rsplit("\n", 1)[0] + "\n[...]"
        return digest

    # -- Internal: lazy-loaded objects ---------------------------------------

    def _get_engine(self) -> Optional[Any]:
        """Lazy-load RecallEngine (BGE model load ~2s, done once)."""
        if self._engine is not None:
            return self._engine
        if not self._active or self._store_path is None:
            return None

        # Only load if store files actually exist.
        db_path = self._store_path / "episodes.db"
        hot_path = self._store_path / "hot_metadata.json"
        if not db_path.exists() or not hot_path.exists():
            return None  # Store not yet initialised by first flush.

        # Self-diagnostic: if store_path is the fallback location, check run_agent.py patch
        if self._store_path is not None:
            # Extract agent_name from config or hermes_home
            hermes_home = kwargs.get("hermes_home") or str(Path.home() / ".hermes")
            config: Dict[str, Any] = kwargs.get("config", {}) or {}
            agent_name = config.get("agent_name") or Path(hermes_home).name
            fallback_pattern = f"/episodic_memory/{agent_name}"
            if str(self._store_path).endswith(fallback_pattern):
                logger.warning(
                    "WARNING: store_path looks like the fallback default (%s)."
                    " If you set store_path explicitly in config.yaml and still see this," 
                    "run_agent.py needs patching. See: https://github.com/f00stx/episodic-memory/blob/main/integrations/hermes/README.md#patching-run_agentpy",
                    self._store_path
                )

        with self._engine_lock:
            if self._engine is not None:
                return self._engine
            try:
                from episodic_memory import RecallEngine
                self._engine = RecallEngine(
                    store_path=str(self._store_path),
                    recall_threshold=self._recall_threshold,
                    filter_roleplay=self._filter_roleplay,
                    embedding_device="cpu",
                    embedding_model=self._embedding_model,
                )
                logger.info(
                    "Episodic RecallEngine loaded: %d episodes",
                    self._engine.n_episodes,
                )
            except Exception as e:
                logger.error("EpisodicMemoryError: BGE model '%s' not found in HF cache.\nRun: python -c \"from huggingface_hub import snapshot_download; snapshot_download('%s')\"\nOr set embedding_model: BAAI/bge-small-en-v1.5 in config for a smaller download (133MB).",
                             self._embedding_model, self._embedding_model)
                raise

        return self._engine

    def _get_embed_client(self) -> Optional[Any]:
        """Return the embed client from the RecallEngine (reuses loaded model).

        Falls back to constructing a minimal SentenceTransformer wrapper if
        the engine isn't available yet (e.g. store is empty).
        """
        engine = self._get_engine()
        if engine is not None:
            try:
                return engine._get_embed_client()
            except Exception:
                pass

        # Standalone fallback (first flush before store exists).
        try:
            from sentence_transformers import SentenceTransformer

            class _STClient:
                def __init__(self, model_name: str):
                    self._st = SentenceTransformer(model_name, device="cpu")

                def embed(self, texts):
                    return self._st.encode(
                        texts,
                        batch_size=32,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )

                def embed_one(self, text: str):
                    return self.embed([text])[0]

            return _STClient(self._embedding_model)
        except Exception as e:
            logger.warning("Could not build standalone embed client: %s", e)
            return None

    def _get_encoder(self) -> Optional[Any]:
        """Lazy-load EpisodicEncoder with default config (no custom checkpoint needed)."""
        if self._encoder is not None:
            return self._encoder
        try:
            from episodic_memory import EpisodicEncoder, EpisodicEncoderConfig
            cfg = EpisodicEncoderConfig()
            self._encoder = EpisodicEncoder(cfg).eval()
        except Exception as e:
            logger.warning("EpisodicEncoder init failed: %s", e)
        return self._encoder

    def _get_store(self, store_path: Path) -> Optional[Any]:
        """Lazy-load EpisodicMemoryStore (creates db + hot tier if absent)."""
        if self._store is not None:
            return self._store
        try:
            from episodic_memory import EpisodicMemoryStore
            self._store = EpisodicMemoryStore(store_path)
        except Exception as e:
            logger.warning("EpisodicMemoryStore init failed: %s", e)
        return self._store


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Called by Hermes plugin loader."""
    ctx.register_memory_provider(EpisodicMemoryProvider())
