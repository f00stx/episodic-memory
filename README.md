# episodic-memory

Standalone episodic memory system for AI agents - semantic recall, roleplay filtering, and temporal contradiction detection. Lightweight enough to run on a laptop; designed to plug into any OpenAI-compatible agent pipeline.

## TL;DR - why this exists

Most agent memory systems are either a vector database bolted to a retriever, or a raw context window that grows until it breaks. Neither handles the messy reality of long-running agents well:

- **Contradiction** - if the user's setup changed last month, the old episode is worse than useless. It silently poisons the context. Most systems return it anyway.
- **Mixed sessions** - agents that do both factual work and creative/roleplay sessions will hallucinate across the boundary if you don't filter. Vector similarity doesn't distinguish "we debugged a deployment" from "we roleplayed a deployment."
- **Retrieval latency** - a full embedding lookup on every turn is overkill. Most turns don't need episodic context at all.

This library came out of building a persistent AI agent designed to maintain genuine continuity across hundreds of sessions. The three features above - temporal supersession, roleplay filtering, and two-tier fast/slow retrieval - were the hard-won lessons. They're packaged here as a standalone drop-in with no required external services.

**What makes it different from LangChain memory / Mem0 / etc:**
- Temporal supersession is explicit and injected into the prompt - the agent *knows* a memory may be outdated, not just that it exists
- Roleplay filter is heuristic (O(1), no embeddings) - prevents fiction bleed without adding a classifier
- Two-tier store (numpy hot path + SQLite cold) means sub-5ms retrieval up to ~100K episodes without a vector DB
- No managed service dependency - runs fully local, SQLite is the only required storage backend

## What it does

- **Semantic recall** - `RecallEngine.query(text)` returns the most relevant past episode using cosine similarity over BGE embeddings. Sub-5ms for stores up to ~100K episodes.
- **Two-tier store** - hot tier (numpy + JSON) for fast similarity search; cold tier (SQLite) for full transcripts and metadata. No external database required.
- **Roleplay filter** - 50+ keyword tells exclude fiction/RP sessions from factual recall. Prevents hallucination-amplification when your agent has mixed session types.
- **Temporal supersession** - `ContradictionDetector` flags older episodes on the same topic as outdated when a newer one covers the same ground (sim >= 0.75, >1 day newer). Injected context leads with `[POSSIBLY OUTDATED - N weeks later: ...]`.
- **Semantic tagging** - zero-cost heuristic tags (`hardware`, `speculation`, `person`, `config`, `completed`, `error`, ...) with TTL-based auto-expiry. Filter recall by tag.

## Quick start

```bash
pip install git+https://github.com/f00stx/episodic-memory
```

```python
from episodic_memory import RecallEngine

engine = RecallEngine(store_path="~/.my_agent/memory")
result = engine.query("what microphone setup did we use?")
if result:
    print(result.context_injection())  # ready-to-inject system prompt block
    if result.is_superseded:
        print(f"Superseded {result.supersession_age_gap_str} later")
```

`context_injection()` returns a formatted block like:

```
[Memory - 0.82 similarity, 3 weeks ago]
The user and agent worked through an API endpoint configuration issue.
The session ended with a working setup confirmed against the staging environment.
Tone: collaborative, methodical.
```

Drop that into your system prompt before generating a response.

## Building a store

Use `EpisodicMemoryStore` to write episodes programmatically:

```python
import numpy as np
from episodic_memory import EpisodicMemoryStore

store = EpisodicMemoryStore("~/.my_agent/memory")
store.add(
    session_id="2024_11_15_001",
    latent=np.zeros(256, dtype=np.float32),   # your encoder output, or zeros
    transcript=[
        {"role": "user",      "content": "How do I set up the microphone?"},
        {"role": "assistant", "content": "Let's start with the input routing..."},
    ],
    summary="User and agent debugged microphone input routing. Ended successfully.",
    metadata={
        "dominant_emotion":    "trust",
        "dominant_archetype":  "sage",
        "turn_count":          12,
    },
)
```

The `latent` vector can be zeros if you don't have a custom encoder - `RecallEngine` re-embeds summaries via BGE at query time regardless.

## Architecture

```
RecallEngine              <- start here (high-level API)
+-- DirectTextResonance   <- BGE cosine search over cached summary embeddings
|   +-- RoleplayFilter    <- keyword triage, O(1), no embeddings
|   +-- ContradictionDetector <- temporal supersession check
+-- EpisodicRecall        <- fetches full episode + LLM-generated summary
+-- EpisodicMemoryStore   <- two-tier hot (numpy) / cold (SQLite) store
```

Two-tier design mirrors neuroscience:
- **Fast path** (amygdala-style, <5ms) - cosine similarity over pre-embedded summaries, blended emotional resonance
- **Slow path** (hippocampal, 100-500ms) - triggered only when similarity exceeds `recall_threshold`; fetches transcript + generates/caches a natural-language gist via local LLM

## Embedding model

The default model is `BAAI/bge-small-en-v1.5` (133MB, loads in ~5s, good quality):

```python
# Default - small and fast, good for most use cases
engine = RecallEngine(store_path="~/.my_agent/memory")

# Higher quality - 1.3GB, ~30s load time
engine = RecallEngine(
    store_path="~/.my_agent/memory",
    embedding_model="BAAI/bge-large-en-v1.5",
)

# OpenAI embeddings - zero local model, costs money
# Pass a custom embed_fn instead (see Advanced usage)
```

Models are downloaded from HuggingFace Hub on first use. Set `HF_HOME` or `TRANSFORMERS_CACHE` to control the download location.

## Docker

For multi-agent or server deployments:

```bash
cp .env.example .env
# Edit .env: set STORE_PATH to your episodes.db directory
docker compose up -d
curl http://localhost:8099/health
```

The service exposes:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness + episode count |
| `/query` | POST | Semantic recall -> `RecallResult` or null |
| `/query_resonance` | POST | Fast path only -> `ResonanceResult` |
| `/stats` | GET | Episode count, emotion distribution |
| `/tags` | GET | Tag vocabulary with counts and expiry stats |

```bash
curl -X POST http://localhost:8099/query \
  -H "Content-Type: application/json" \
  -d '{"text": "what did we work on last week?"}'
```

## Summary generation

When a match is found but no summary is cached, `EpisodicRecall` calls a local LLM to generate one. Default endpoint is `http://localhost:11434/v1` (Ollama). Pass any OpenAI-compatible endpoint:

```python
from episodic_memory.recall import EpisodicRecall

recall = EpisodicRecall(
    store=store,
    llm_base_url="http://localhost:11434/v1",   # Ollama
    llm_model="llama3",
    llm_api_key="none",                          # local endpoints don't need a key
)
```

Summaries are persisted to SQLite after generation - the LLM is only called once per episode.

When using the MCP server, the endpoint is configurable via environment variables (no source edits required):

The affective summary and technical index use separate LLM endpoints, so you can route each to the model best suited to the task (e.g. a general chat model for affect, a code-trained model for technical extraction).

**Affective summary** (emotional tone, relational mode):

| Variable | Default | Description |
|---|---|---|
| `EPISODIC_LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible base URL |
| `EPISODIC_LLM_MODEL` | `llama3` | Model name |
| `EPISODIC_LLM_API_KEY` | `none` | API key (`none` for local endpoints) |

**Technical index** (file paths, metrics, flags, decisions):

| Variable | Default | Description |
|---|---|---|
| `EPISODIC_TECHNICAL_LLM_BASE_URL` | `http://localhost:8003/v1` | Base URL for technical index LLM |
| `EPISODIC_TECHNICAL_LLM_MODEL` | `qwen2.5-coder-32b-instruct` | Model for technical extraction |
| `EPISODIC_TECHNICAL_LLM_API_KEY` | inherits `EPISODIC_LLM_API_KEY` | API key |

If `EPISODIC_TECHNICAL_LLM_*` are unset, they fall back to the affective summary endpoint. Example -- explicit split configuration:

```bash
EPISODIC_LLM_BASE_URL=http://localhost:8001/v1 \
EPISODIC_LLM_MODEL=qwen3-30b-a3b \
EPISODIC_TECHNICAL_LLM_BASE_URL=http://localhost:8003/v1 \
EPISODIC_TECHNICAL_LLM_MODEL=qwen2.5-coder-32b-instruct \
python mcp_server.py
```

## Semantic tags

Episodes are tagged automatically at store time - zero LLM cost:

```python
from episodic_memory.tagger import EpisodicTagger

tagger = EpisodicTagger()
result = tagger.tag("GPU driver updated, NVLink confirmed working")
# result.tags == ["hardware", "completed"]
# result.expires_at == stored_at + 90 days  (hardware TTL)
```

Filter recall by tag:

```python
# Only return non-hardware, non-speculation episodes
result = engine.query(
    "setup process",
    exclude_tags=["hardware", "speculation"],
)

# Only return completed-task episodes
result = engine.query(
    "microphone config",
    only_tags=["completed"],
)
```

The `_PERSON_PAT` regex in `tagger.py` contains a generic placeholder name list - replace it with the names relevant to your agent and the people it interacts with.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- sentence-transformers >= 2.2 (pulls in the embedding model)
- numpy >= 1.24
- faiss-cpu >= 1.7 (optional - used if installed, falls back to numpy dot-product)

For summary generation: any OpenAI-compatible LLM endpoint (Ollama, LM Studio, vLLM, OpenAI API).

## Integrations

### Hermes Agent

A drop-in memory provider plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent) (NousResearch) is included in `integrations/hermes/`. It wires episodic recall into the Hermes turn loop via the `prefetch`/`queue_prefetch` pattern - zero added turn latency - and exposes `hermes episodic-memory status/stats/search` CLI commands.

See [`integrations/hermes/README.md`](integrations/hermes/README.md) for setup.

## Known limitations

These are architectural decisions with known tradeoffs, not bugs. Understanding them upfront saves a lot of head-scratching.

### The affective summary discards technical facts

`EpisodicRecall`'s LLM summary prompt is designed for *emotional and relational* context injection -- it explicitly instructs the model not to reproduce quotes or list facts. This is correct for the use case of priming an agent's "mood" and relational stance. It is the wrong tool if you need the agent to remember "we used `--limit 500` and got AUROC 0.6866 on the domain corpus."

**Practical consequence:** Precise technical details (metric values, file paths, flag combinations, model checkpoint names, version numbers) do not survive the compression path. Only the emotional tone, relational mode, and broad thematic summary are retained.

**Workarounds:**
- Write precise facts to a separate project state file and load it explicitly at session start.
- Use `fetch_session_detail` (MCP) or `memory_get` to pull the raw transcript for a specific session -- the transcript itself is stored verbatim in SQLite, only the *summary* is lossy.
- Add a parallel "technical index" summarisation pass with a fact-preserving prompt -- the system doesn't do this out of the box.

### LLM summary is generated lazily, not at ingest time

Summaries are generated on first recall (when `recall_threshold` is crossed), not when the episode is first stored. This means:
- First recall of any cold session adds 100-500ms LLM latency.
- If the LLM endpoint is unavailable at recall time, the fallback summary is a bare metadata stub ("A 34-turn conversation from 2024-11-15. Dominant tone: trust.") -- no content.
- Pre-generation is possible via `EpisodicRecall.precompute_summaries()`, but it is not automatic.

### MCP server: LLM endpoint is not configurable via environment variable

The MCP server (`mcp_server.py`) hardcodes `llm_base_url="http://localhost:11434/v1"` and `llm_model="llama3"` when constructing the `RecallEngine`. There is no env var to override these without editing the source file. If your Ollama instance is on a different host or port, or you want to use a different model, you need to edit `_get_engine()` directly.

### Passive ingest delay

The background ingest loop (`_ingest_loop` in `mcp_server.py`) only picks up a Claude Code session after it has been idle for 10 minutes AND the loop's 30-minute interval has elapsed. In the worst case, a session you just closed takes up to 40 minutes to become recallable. If you need immediate availability, run `ingest_sessions.py` manually.

### Cosine similarity over summary embeddings, not full transcripts

The hot-tier resonance search operates over BGE embeddings of the *summary text*, not the full transcript. This is what gives sub-5ms performance. It also means that a session only becomes searchable once it has a summary (see lazy generation above), and that the retrieval quality is bounded by summary quality.

---

## License

MIT
