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
import contextlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

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
# Event telemetry -- fire-and-forget JSONL + optional Redpanda/Kafka
#
# Events are appended to <store_path>/mcp_events.jsonl always.
# If EPISODIC_KAFKA_BROKERS is set (e.g. "localhost:19092") and kafka-python
# is installed, events are also produced to the Kafka topic EPISODIC_KAFKA_TOPIC
# (default: "episodic-memory-events").  All errors are swallowed so telemetry
# never blocks a tool call.
# ---------------------------------------------------------------------------
_event_log_path: Path | None = None
_kafka_producer: Any = None
_kafka_topic: str = ""
_kafka_init_done: bool = False


def _init_telemetry() -> None:
    global _event_log_path, _kafka_producer, _kafka_topic, _kafka_init_done
    if _kafka_init_done:
        return
    _kafka_init_done = True

    store_path = Path(
        os.environ.get("EPISODIC_MEMORY_STORE_PATH", "~/.ctm/memory")
    ).expanduser()
    _event_log_path = store_path / "mcp_events.jsonl"

    brokers = os.environ.get("EPISODIC_KAFKA_BROKERS", "").strip()
    if brokers:
        _kafka_topic = os.environ.get("EPISODIC_KAFKA_TOPIC", "episodic-memory-events")
        try:
            from kafka import KafkaProducer  # type: ignore
            _kafka_producer = KafkaProducer(
                bootstrap_servers=brokers.split(","),
                value_serializer=lambda v: json.dumps(v).encode(),
                acks=0,
                retries=0,
                api_version=(2, 0, 0),   # skip auto-detection (prevents blocking on connect)
                request_timeout_ms=2000,
            )
            logger.info("Kafka telemetry connected: brokers=%s topic=%s", brokers, _kafka_topic)
        except Exception as exc:
            logger.warning("Kafka telemetry unavailable (%s) -- file log only", exc)


def _emit_event(event: dict) -> None:
    """Append event to JSONL log and optionally produce to Kafka. Never raises."""
    event.setdefault("ts", time.time())

    # File write first -- always works, never blocks
    store_path = os.environ.get("EPISODIC_MEMORY_STORE_PATH", "")
    if store_path:
        try:
            path = Path(store_path).expanduser() / "mcp_events.jsonl"
            with open(path, "a") as fh:
                fh.write(json.dumps(event) + "\n")
        except Exception:
            pass

    # Kafka init (may block briefly on first call -- bounded by socket_timeout_ms)
    _init_telemetry()
    if _kafka_producer is not None:
        try:
            _kafka_producer.send(_kafka_topic, event)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Claude Code session file helpers
# ---------------------------------------------------------------------------

def _extract_text_from_content(content: Any) -> str:
    """Extract text from message.content (string or list of blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts)
    return ""


def _parse_claude_jsonl(filepath: Path) -> dict | None:
    """
    Parse a Claude Code JSONL session file.
    Returns dict with: session_id, project_dir, messages (list of {role, content, ts}), timestamp
    """
    try:
        messages = []
        session_id = filepath.stem
        project_dir = filepath.parent.name
        first_ts = None

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = entry.get("type")
                if msg_type not in ("user", "assistant"):
                    continue

                msg = entry.get("message", {})
                content = _extract_text_from_content(msg.get("content"))
                if not content.strip():
                    continue

                ts = entry.get("timestamp")
                if first_ts is None and ts:
                    first_ts = ts

                messages.append({
                    "role": msg_type,
                    "content": content,
                    "ts": ts,
                })

        if not messages:
            return None

        return {
            "session_id": session_id,
            "project_dir": project_dir,
            "messages": messages,
            "timestamp": first_ts,
        }
    except Exception as e:
        logger.warning("Failed to parse session file %s: %s", filepath, e)
        return None


def _compress_conversation(messages: list[dict], max_chars: int = 6000) -> str:
    """Build compressed conversation string, capped at max_chars."""
    lines = []
    total = 0
    for msg in messages:
        role = msg["role"]
        # Truncate individual messages to 1500 chars
        content = msg["content"][:1500]
        line = f"[{role}]: {content}\n"
        if total + len(line) > max_chars:
            # Add truncated indicator
            remaining = max_chars - total - 20
            if remaining > 50:
                lines.append(f"[{role}]: {content[:remaining]}... [truncated]\n")
            else:
                lines.append("\n[Conversation truncated due to length]\n")
            break
        lines.append(line)
        total += len(line)
    return "".join(lines)


async def _call_llm_for_extraction(query: str, compressed_conversation: str) -> str:
    """
    Call local LLM to extract relevant facts from conversation.
    Tries qwen-coder (8003) first, falls back to qwen-main (8001).
    Returns LLM response, 'NOT_RELEVANT', or 'LLM_UNAVAILABLE'.
    """
    prompt = f"""Given the search query: "{query}"

Extract any relevant facts or decisions from the following conversation.
Be concise -- 3-6 sentences. If the conversation contains nothing relevant, respond
with exactly: NOT_RELEVANT

CRITICAL HALLUCINATION GUARD: Only extract identifiers, model numbers, version numbers,
and technical specifications that appear VERBATIM in the provided text. If a specific
value is not explicitly stated, omit it entirely rather than inferring or generating it.
If you are uncertain about a specific value, tag it as [uncertain] rather than presenting
it as fact.

Conversation:
{compressed_conversation}"""

    endpoints = [
        ("http://localhost:8003/v1/chat/completions", "qwen-coder"),
        ("http://localhost:8001/v1/chat/completions", "qwen3-30b-a3b-q4_k_s.gguf"),
    ]

    for url, model in endpoints:
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    url,
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 400,
                        "temperature": 0.1,
                    },
                )
                response.raise_for_status()
                data = response.json()
                result = data["choices"][0]["message"]["content"].strip()
                return result
        except Exception as e:
            logger.warning("LLM extraction failed at %s: %s", url, e)

    return "LLM_UNAVAILABLE"


def _find_session_file(session_id: str) -> Path | None:
    """Search for {session_id}.jsonl across ~/.claude/projects/"""
    projects_dir = Path("~/.claude/projects").expanduser()
    if not projects_dir.exists():
        return None
    for jsonl_file in projects_dir.rglob("*.jsonl"):
        if jsonl_file.stem == session_id:
            return jsonl_file
    return None


def _keyword_score(text: str, query_terms: list[str]) -> int:
    """Score session by how many query terms appear in it."""
    text_lower = text.lower()
    score = 0
    for term in query_terms:
        score += len(re.findall(rf'\b{re.escape(term)}\b', text_lower, re.IGNORECASE))
    return score


# ---------------------------------------------------------------------------
# Lazy imports -- defer heavy BGE model load until first tool call
# ---------------------------------------------------------------------------
_engine: Any = None

# Path to the standalone ingest module (integrations/claude_code/ingest_sessions.py)
_INTEGRATIONS_PATH = Path(__file__).parent / "integrations" / "claude_code"

# Cached ingest deps -- loaded once on first ingest pass that has work
_ingest_embed_client: Any = None
_ingest_encoder: Any = None


def _get_engine() -> Any:
    global _engine
    if _engine is None:
        store_path = os.environ.get("EPISODIC_MEMORY_STORE_PATH", "~/.ctm/memory")
        store_path = Path(store_path).expanduser()

        llm_base_url = os.environ.get("EPISODIC_LLM_BASE_URL", "http://localhost:11434/v1")
        llm_model    = os.environ.get("EPISODIC_LLM_MODEL",    "llama3")
        llm_api_key  = os.environ.get("EPISODIC_LLM_API_KEY",  "none")

        logger.info("Initialising RecallEngine with store_path=%s llm=%s %s", store_path, llm_base_url, llm_model)

        from episodic_memory import RecallEngine

        _engine = RecallEngine(
            store_path=store_path,
            recall_threshold=0.55,
            resonance_threshold=0.45,
            top_k=5,
            filter_roleplay=True,
            embedding_device="cpu",
            embedding_model="BAAI/bge-large-en-v1.5",  # matches cached embeddings (1024-dim)
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
        )
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
        Tool(
            name="fetch_session_detail",
            description=(
                "Fetch and summarise the content of a specific Claude Code session file. "
                "Use this when episodic_recall or search_sessions_raw returns a session_id "
                "and you need the actual detailed content. Searches ~/.claude/projects/ "
                "for the JSONL file and uses a local LLM to extract facts relevant to your query."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The Claude Code session UUID (e.g. '253ddbb5-f4dd-4a39-a28e-5e2d81a86e95')",
                    },
                    "query": {
                        "type": "string",
                        "description": "What you are looking for in this session -- the LLM will extract facts relevant to this query.",
                    },
                },
                "required": ["session_id", "query"],
            },
        ),
        Tool(
            name="search_sessions_raw",
            description=(
                "Fallback: keyword-scan all Claude Code session files in ~/.claude/projects/, "
                "then use a local LLM to extract relevant content. Sequential, not parallel. "
                "Use ONLY when episodic_recall and session_search return no useful hits. "
                "Slower than episodic recall -- prefer episodic tools first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for across all Claude Code session files.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Maximum number of relevant results to return (default: 3).",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    logger.debug("call_tool: %s args=%s", name, arguments)
    _t0 = time.time()

    # Emit tool-invocation event immediately (before execution)
    _emit_event({
        "event":     "mcp_tool_called",
        "tool":      name,
        "args_keys": sorted(arguments.keys()),
        "query_len": len(arguments.get("text", "") or arguments.get("session_id", "")),
    })

    try:
        if name == "memory_query":
            result = _get_engine().query(
                text=arguments["text"],
                exclude_session_id=arguments.get("exclude_session_id"),
                exclude_tags=arguments.get("exclude_tags"),
                only_tags=arguments.get("only_tags"),
            )
            if result is None:
                _emit_event({
                    "event":       "memory_query_miss",
                    "tool":        name,
                    "elapsed_ms":  round((time.time() - _t0) * 1000, 1),
                })
                return [TextContent(type="text", text="null")]

            _emit_event({
                "event":            "memory_query_hit",
                "tool":             name,
                "session_id":       result.session_id,
                "similarity":       round(result.similarity, 4),
                "dominant_emotion": result.dominant_emotion,
                "is_superseded":    result.is_superseded,
                "elapsed_ms":       round((time.time() - _t0) * 1000, 1),
            })

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

            sorted_meta = sorted(
                [m for m in store._hot_metadata if not m.get("_removed")],
                key=lambda m: m.get("stored_at", 0.0),
                reverse=True,
            )

            items = []
            for meta in sorted_meta[offset : offset + limit]:
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

        elif name == "fetch_session_detail":
            session_id = arguments["session_id"]
            query = arguments["query"]

            # Emit start event
            _emit_event({
                "event": "fetch_session_detail_start",
                "tool": name,
                "session_id": session_id,
            })

            # Find the session file
            filepath = _find_session_file(session_id)
            if filepath is None:
                _emit_event({
                    "event": "fetch_session_detail_not_found",
                    "tool": name,
                    "session_id": session_id,
                    "elapsed_ms": round((time.time() - _t0) * 1000, 1),
                })
                return [TextContent(type="text", text=json.dumps({
                    "error": "session_file_missing",
                    "note": "Index entry exists but source file is unavailable -- summary cannot be generated",
                    "session_id": session_id,
                }))]

            # Parse the JSONL
            session_data = _parse_claude_jsonl(filepath)
            if session_data is None:
                _emit_event({
                    "event": "fetch_session_detail_parse_error",
                    "tool": name,
                    "session_id": session_id,
                    "elapsed_ms": round((time.time() - _t0) * 1000, 1),
                })
                return [TextContent(type="text", text=json.dumps({
                    "error": f"Failed to parse session file: {filepath}",
                    "session_id": session_id,
                }))]

            # Compress conversation and call LLM
            compressed = _compress_conversation(session_data["messages"])
            llm_result = await _call_llm_for_extraction(query, compressed)

            _emit_event({
                "event": "fetch_session_detail_complete",
                "tool": name,
                "session_id": session_id,
                "llm_not_relevant": llm_result == "NOT_RELEVANT",
                "elapsed_ms": round((time.time() - _t0) * 1000, 1),
            })

            if llm_result == "NOT_RELEVANT":
                return [TextContent(type="text", text=json.dumps({
                    "session_id": session_id,
                    "query": query,
                    "result": "NOT_RELEVANT",
                    "message": "The conversation does not contain content relevant to the query.",
                }))]

            # Parse approximate date from timestamp
            approx_date = None
            if session_data.get("timestamp"):
                try:
                    from datetime import datetime
                    ts = session_data["timestamp"]
                    if isinstance(ts, (int, float)):
                        approx_date = datetime.fromtimestamp(ts).isoformat()
                except Exception:
                    pass

            if llm_result == "LLM_UNAVAILABLE":
                # Return raw excerpt so caller has something useful even without LLM
                return [TextContent(type="text", text=json.dumps({
                    "session_id": session_id,
                    "approximate_date": approx_date,
                    "project_dir": session_data["project_dir"],
                    "summary": None,
                    "llm_unavailable": True,
                    "raw_excerpt": compressed[:3000],
                }, indent=2))]

            return [TextContent(type="text", text=json.dumps({
                "session_id": session_id,
                "approximate_date": approx_date,
                "project_dir": session_data["project_dir"],
                "summary": llm_result,
            }, indent=2))]

        elif name == "search_sessions_raw":
            query = arguments["query"]
            n_results = arguments.get("n_results", 3)
            n_results = max(1, min(n_results, 10))  # Cap between 1-10

            # Calculate candidate limit
            candidate_limit = min(n_results * 3, 15)

            # Emit start event
            _emit_event({
                "event": "search_sessions_raw_start",
                "tool": name,
                "query": query,
                "n_results": n_results,
            })

            # Walk ~/.claude/projects/ for all JSONL files
            projects_dir = Path("~/.claude/projects").expanduser()
            if not projects_dir.exists():
                _emit_event({
                    "event": "search_sessions_raw_no_projects",
                    "tool": name,
                    "elapsed_ms": round((time.time() - _t0) * 1000, 1),
                })
                return [TextContent(type="text", text=json.dumps({
                    "error": "Claude projects directory not found: ~/.claude/projects/",
                    "results": [],
                }))]

            # Parse all session files
            jsonl_files = list(projects_dir.rglob("*.jsonl"))
            if not jsonl_files:
                _emit_event({
                    "event": "search_sessions_raw_no_files",
                    "tool": name,
                    "elapsed_ms": round((time.time() - _t0) * 1000, 1),
                })
                return [TextContent(type="text", text=json.dumps({
                    "message": "No session files found in ~/.claude/projects/",
                    "results": [],
                }))]

            # Tokenize query for keyword scoring
            query_terms = [t.lower() for t in query.split() if len(t) >= 3]
            if not query_terms:
                # If all terms are short, use them anyway
                query_terms = [t.lower() for t in query.split()]

            # Score each session and build candidate list
            scored_sessions = []
            for filepath in jsonl_files:
                session_data = _parse_claude_jsonl(filepath)
                if session_data is None:
                    continue

                # Build full text for scoring
                full_text = " ".join([m["content"] for m in session_data["messages"]])
                score = _keyword_score(full_text, query_terms)

                if score > 0:
                    scored_sessions.append((score, session_data, filepath))

            if not scored_sessions:
                _emit_event({
                    "event": "search_sessions_raw_no_matches",
                    "tool": name,
                    "elapsed_ms": round((time.time() - _t0) * 1000, 1),
                })
                return [TextContent(type="text", text=json.dumps({
                    "message": "No sessions matched the query keywords.",
                    "results": [],
                }))]

            # Sort by score descending and take top candidates
            scored_sessions.sort(key=lambda x: x[0], reverse=True)
            candidates = scored_sessions[:candidate_limit]

            # Process candidates sequentially with LLM
            results = []
            for score, session_data, filepath in candidates:
                if len(results) >= n_results:
                    break

                compressed = _compress_conversation(session_data["messages"])
                llm_result = await _call_llm_for_extraction(query, compressed)

                if llm_result == "NOT_RELEVANT":
                    continue

                # Parse approximate date
                approx_date = None
                if session_data.get("timestamp"):
                    try:
                        from datetime import datetime
                        ts = session_data["timestamp"]
                        if isinstance(ts, (int, float)):
                            approx_date = datetime.fromtimestamp(ts).isoformat()
                    except Exception:
                        pass

                if llm_result == "LLM_UNAVAILABLE":
                    # Include raw excerpt so caller has something useful
                    results.append({
                        "session_id": session_data["session_id"],
                        "project_dir": session_data["project_dir"],
                        "approximate_date": approx_date,
                        "keyword_score": score,
                        "summary": None,
                        "llm_unavailable": True,
                        "raw_excerpt": compressed[:2000],
                    })
                else:
                    results.append({
                        "session_id": session_data["session_id"],
                        "project_dir": session_data["project_dir"],
                        "approximate_date": approx_date,
                        "keyword_score": score,
                        "summary": llm_result,
                    })

            _emit_event({
                "event": "search_sessions_raw_complete",
                "tool": name,
                "query": query,
                "candidates_checked": len(candidates),
                "results_found": len(results),
                "elapsed_ms": round((time.time() - _t0) * 1000, 1),
            })

            if not results:
                return [TextContent(type="text", text=json.dumps({
                    "message": f"Checked {len(candidates)} candidate sessions but found no relevant content after LLM analysis.",
                    "query": query,
                    "results": [],
                }))]

            return [TextContent(type="text", text=json.dumps({
                "query": query,
                "results": results,
            }, indent=2))]

        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    except Exception as exc:
        logger.exception("Tool %s failed", name)
        _emit_event({
            "event":      "mcp_tool_error",
            "tool":       name,
            "error":      str(exc),
            "elapsed_ms": round((time.time() - _t0) * 1000, 1),
        })
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]


# ---------------------------------------------------------------------------
# Passive session ingest (background task)
# ---------------------------------------------------------------------------

_ingest_lock = asyncio.Lock()


def _sync_ingest_pass(
    projects_dir: Path,
    store_path: Path,
    ledger_path: Path,
    min_age_secs: int,
    min_turns: int,
    bge_model: str,
) -> tuple[int, int, int]:
    """
    Scan for uningested Claude Code sessions and write them to the store.
    Runs in a thread via asyncio.to_thread -- safe to block.
    Returns (ingested, skipped, errors).
    """
    global _ingest_embed_client, _ingest_encoder

    if str(_INTEGRATIONS_PATH) not in sys.path:
        sys.path.insert(0, str(_INTEGRATIONS_PATH))

    from ingest_sessions import IngestLedger, _infer_project, encode_and_store, parse_session

    cutoff = time.time() - min_age_secs
    jsonl_files = sorted(projects_dir.rglob("*.jsonl"))
    ledger = IngestLedger(ledger_path)

    pending = [
        f for f in jsonl_files
        if not ledger.has(f.stem) and f.stat().st_mtime < cutoff
    ]

    if not pending:
        ledger.close()
        return 0, 0, 0

    # Lazy-load heavy deps on first pass that actually has work
    if _ingest_embed_client is None:
        logger.info("Ingest: loading BGE model %s...", bge_model)
        try:
            from sentence_transformers import SentenceTransformer
            _st = SentenceTransformer(bge_model, device="cpu")

            class _EmbedClient:
                def embed(self, texts):
                    return _st.encode(texts, normalize_embeddings=True, show_progress_bar=False)

            _ingest_embed_client = _EmbedClient()
        except Exception as e:
            logger.error("Ingest: failed to load BGE: %s; skipping pass", e)
            ledger.close()
            return 0, 0, 0

    if _ingest_encoder is None:
        try:
            from episodic_memory import EpisodicEncoder, EpisodicEncoderConfig
            _ingest_encoder = EpisodicEncoder(EpisodicEncoderConfig()).eval()
        except Exception as e:
            logger.error("Ingest: failed to load EpisodicEncoder: %s; skipping pass", e)
            ledger.close()
            return 0, 0, 0

    try:
        from episodic_memory import EpisodicMemoryStore
        store = EpisodicMemoryStore(str(store_path))
    except Exception as e:
        logger.error("Ingest: failed to open store: %s; skipping pass", e)
        ledger.close()
        return 0, 0, 0

    ingested = skipped = errors = 0
    for f in pending:
        sid, turns, stored_at = parse_session(f)
        project = _infer_project(f)

        if len(turns) < min_turns:
            skipped += 1
            ledger.mark(sid, f, len(turns))
            continue

        try:
            ok = encode_and_store(
                sid, turns, stored_at, project, store,
                _ingest_embed_client, _ingest_encoder,
            )
            if ok:
                ingested += 1
            else:
                skipped += 1
            ledger.mark(sid, f, len(turns))
        except Exception as e:
            logger.error("Ingest: error on session %s: %s", sid[:8], e)
            errors += 1

    ledger.close()
    return ingested, skipped, errors


async def _ingest_loop() -> None:
    """
    Background coroutine: ingest new Claude Code sessions periodically.

    Env vars:
        EPISODIC_PROJECTS_PATH    Claude projects dir  (default: ~/.claude/projects)
        EPISODIC_INGEST_MIN_AGE   Minutes idle before a session is considered done (default: 10)
        EPISODIC_INGEST_INTERVAL  Seconds between passes (default: 1800)
        EPISODIC_INGEST_MIN_TURNS Skip sessions shorter than N turns (default: 3)
        EPISODIC_INGEST_BGE_MODEL BGE model name (default: BAAI/bge-large-en-v1.5)
    """
    projects_dir = Path(
        os.environ.get("EPISODIC_PROJECTS_PATH", "~/.claude/projects")
    ).expanduser()
    store_path = Path(
        os.environ.get("EPISODIC_MEMORY_STORE_PATH", "~/.ctm/memory")
    ).expanduser()
    min_age_mins = int(os.environ.get("EPISODIC_INGEST_MIN_AGE", "10"))
    interval = int(os.environ.get("EPISODIC_INGEST_INTERVAL", "1800"))
    min_turns = int(os.environ.get("EPISODIC_INGEST_MIN_TURNS", "3"))
    bge_model = os.environ.get("EPISODIC_INGEST_BGE_MODEL", "BAAI/bge-large-en-v1.5")
    ledger_path = store_path / "claude_ingested.db"
    min_age_secs = min_age_mins * 60

    if not _INTEGRATIONS_PATH.exists():
        logger.warning(
            "Ingest integrations path missing: %s -- passive ingest disabled",
            _INTEGRATIONS_PATH,
        )
        return
    if not projects_dir.exists():
        logger.info(
            "EPISODIC_PROJECTS_PATH %s not found -- passive ingest disabled",
            projects_dir,
        )
        return

    logger.info(
        "Passive ingest loop started: projects=%s interval=%ds min_age=%dm",
        projects_dir, interval, min_age_mins,
    )

    while True:
        async with _ingest_lock:
            try:
                ingested, skipped, errors = await asyncio.to_thread(
                    _sync_ingest_pass,
                    projects_dir, store_path, ledger_path,
                    min_age_secs, min_turns, bge_model,
                )
                if ingested or errors:
                    logger.info(
                        "Ingest pass complete: ingested=%d skipped=%d errors=%d",
                        ingested, skipped, errors,
                    )
            except Exception as e:
                logger.error("Ingest pass failed: %s", e)

        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
async def main() -> None:
    ingest_task = asyncio.create_task(_ingest_loop())
    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options(),
            )
    finally:
        ingest_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ingest_task


# Module-level startup probe -- fires the instant this file is loaded.
# If this does NOT appear in mcp_events.jsonl, the subprocess is not running
# this version of the file (wrong path, stale bytecode, or startup crash).
_startup_store = os.environ.get("EPISODIC_MEMORY_STORE_PATH", "")
if _startup_store:
    try:
        _startup_path = Path(_startup_store).expanduser() / "mcp_events.jsonl"
        with open(_startup_path, "a") as _f:
            _f.write(json.dumps({"event": "mcp_server_started", "ts": time.time(),
                                  "pid": os.getpid(), "file": __file__}) + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
