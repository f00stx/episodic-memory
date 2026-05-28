#!/usr/bin/env python3
"""
Episodic Memory MCP Server -- stdio transport.

Exposes the episodic-memory RecallEngine as MCP tools for Claude Code
(and any other MCP client).

Tools:
  - memory_query        : semantic recall → summary + metadata
  - memory_query_fast   : fast-path resonance only (no LLM summary)
  - memory_stats        : episode count, emotion distribution
  - memory_get          : fetch full transcript by session_id
  - memory_list         : list all session_ids with basic metadata

Environment:
  EPISODIC_MEMORY_STORE_PATH  -- path to the memory store directory
                                 (default: ~/.ctm/memory)

Usage with Claude Code:
    claude mcp add episodic-memory -- python /path/to/mcp_server.py

Or via .mcp.json (project scope):
    {
      "episodic-memory": {
        "command": "python",
        "args": ["/home/richard/projects/episodic-memory/mcp_server.py"],
        "env": {
          "EPISODIC_MEMORY_STORE_PATH": "/home/richard/.ctm/memory"
        }
      }
    }
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging: stderr only so stdio JSON-RPC isn't polluted
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("episodic-memory-mcp")

# ---------------------------------------------------------------------------
# Lazy imports -- defer heavy BGE model load until first tool call
# ---------------------------------------------------------------------------
_engine: Any = None


def _get_engine() -> Any:
    global _engine
    if _engine is None:
        store_path = os.environ.get("EPISODIC_MEMORY_STORE_PATH", "~/.ctm/memory")
        store_path = Path(store_path).expanduser()
        logger.info("Initialising RecallEngine with store_path=%s", store_path)

        from episodic_memory import RecallEngine

        _engine = RecallEngine(
            store_path=store_path,
            recall_threshold=0.55,
            resonance_threshold=0.45,
            top_k=5,
            filter_roleplay=True,
            embedding_device="cpu",
            embedding_model="BAAI/bge-large-en-v1.5",  # matches cached embeddings (1024-dim)
        )
        # Defer n_episodes access -- it triggers BGE load. Just log path.
        logger.info("RecallEngine initialised at %s", store_path)
    return _engine


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

app = Server("episodic-memory")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="memory_query",
            description=(
                "Query episodic memory for relevant past conversations. "
                "Returns a summary, similarity score, emotion/archetype labels, "
                "and a ready-to-inject context block. "
                "Returns null if no relevant memory exceeds the recall threshold."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The query text -- current utterance or topic to search for.",
                    },
                    "exclude_session_id": {
                        "type": "string",
                        "description": "Optional session_id to exclude (e.g. the current live session).",
                    },
                    "exclude_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags to exclude (e.g. ['hardware', 'speculation']).",
                    },
                    "only_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags to require (e.g. ['completed']).",
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="memory_query_fast",
            description=(
                "Fast-path memory resonance query. Returns top-k matching session_ids "
                "with cosine similarities and emotional resonance vector. "
                "No LLM summary generation -- sub-5ms. Useful for emotional colouring "
                "without the latency of cold-tier retrieval."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Query text.",
                    },
                    "exclude_session_id": {
                        "type": "string",
                        "description": "Optional session_id to exclude.",
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="memory_stats",
            description=(
                "Return statistics about the episodic memory store: "
                "total episode count, emotion distribution, archetype distribution, "
                "and tag vocabulary with counts."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="memory_get",
            description=(
                "Fetch the full conversation transcript and metadata for a given session_id."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The session_id to look up.",
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="memory_list",
            description=(
                "List all stored session_ids with basic metadata "
                "(dominant_emotion, dominant_archetype, turn_count, stored_at)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of sessions to return (default: 100).",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Offset for pagination (default: 0).",
                    },
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    logger.debug("call_tool: %s args=%s", name, arguments)

    try:
        if name == "memory_query":
            result = _get_engine().query(
                text=arguments["text"],
                exclude_session_id=arguments.get("exclude_session_id"),
                exclude_tags=arguments.get("exclude_tags"),
                only_tags=arguments.get("only_tags"),
            )
            if result is None:
                return [TextContent(type="text", text="null")]

            payload = {
                "session_id": result.session_id,
                "summary": result.summary,
                "similarity": result.similarity,
                "turn_count": result.turn_count,
                "stored_at": result.stored_at,
                "dominant_emotion": result.dominant_emotion,
                "dominant_archetype": result.dominant_archetype,
                "is_superseded": result.is_superseded,
                "superseded_by": result.superseded_by,
                "supersession_age_gap_str": result.supersession_age_gap_str,
                "context_injection": result.context_injection(),
            }
            return [TextContent(type="text", text=json.dumps(payload, indent=2))]

        elif name == "memory_query_fast":
            res = _get_engine().query_resonance(
                text=arguments["text"],
                exclude_session_id=arguments.get("exclude_session_id"),
            )
            payload = {
                "max_similarity": res.max_similarity,
                "resonance_strength": res.resonance_strength,
                "triggered_recall": res.triggered_recall,
                "top_k_ids": res.top_k_ids,
                "top_k_similarities": res.top_k_similarities,
                "resonance_vector": res.resonance_vector.tolist(),
            }
            return [TextContent(type="text", text=json.dumps(payload, indent=2))]

        elif name == "memory_stats":
            engine = _get_engine()
            # Build emotion / archetype distributions from hot metadata
            emotions: dict[str, int] = {}
            archetypes: dict[str, int] = {}
            tags: dict[str, int] = {}
            from episodic_memory.store import EpisodicMemoryStore

            store = EpisodicMemoryStore(engine.store_path)
            for meta in store._hot_metadata:
                if meta.get("_removed"):
                    continue
                emo = meta.get("dominant_emotion", "neutral")
                emotions[emo] = emotions.get(emo, 0) + 1
                arch = meta.get("dominant_archetype", "sage")
                archetypes[arch] = archetypes.get(arch, 0) + 1
                for t in meta.get("tags", []) or []:
                    tags[t] = tags.get(t, 0) + 1

            payload = {
                "n_episodes": engine.n_episodes,
                "store_path": str(engine.store_path),
                "emotion_distribution": emotions,
                "archetype_distribution": archetypes,
                "tags": tags,
            }
            return [TextContent(type="text", text=json.dumps(payload, indent=2))]

        elif name == "memory_get":
            from episodic_memory.store import EpisodicMemoryStore

            store = EpisodicMemoryStore(_get_engine().store_path)
            sid = arguments["session_id"]
            transcript = store.fetch_transcript(sid)
            summary = store.fetch_summary(sid)
            latent = store.get_latent(sid)
            meta = {}
            if sid in store._session_index:
                meta = dict(store._hot_metadata[store._session_index[sid]])

            payload = {
                "session_id": sid,
                "found": transcript is not None,
                "transcript": transcript,
                "summary": summary,
                "latent": latent.tolist() if latent is not None else None,
                "metadata": meta,
            }
            return [TextContent(type="text", text=json.dumps(payload, indent=2))]

        elif name == "memory_list":
            from episodic_memory.store import EpisodicMemoryStore

            store = EpisodicMemoryStore(_get_engine().store_path)
            limit = arguments.get("limit", 100)
            offset = arguments.get("offset", 0)

            items = []
            for i, meta in enumerate(store._hot_metadata):
                if meta.get("_removed"):
                    continue
                if i < offset:
                    continue
                if len(items) >= limit:
                    break
                items.append(
                    {
                        "session_id": meta.get("session_id", ""),
                        "dominant_emotion": meta.get("dominant_emotion", "neutral"),
                        "dominant_archetype": meta.get("dominant_archetype", "sage"),
                        "turn_count": meta.get("turn_count", 0),
                        "stored_at": meta.get("stored_at", 0.0),
                        "tags": meta.get("tags", []),
                    }
                )

            payload = {
                "total": store.n_episodes,
                "offset": offset,
                "limit": limit,
                "sessions": items,
            }
            return [TextContent(type="text", text=json.dumps(payload, indent=2))]

        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
