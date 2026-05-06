# episodic-memory

Standalone episodic memory system for AI agents -- semantic recall, roleplay filtering, and temporal contradiction detection. Lightweight enough to run on a laptop; designed to plug into any OpenAI-compatible agent pipeline.

## What it does

- **Semantic recall** -- `RecallEngine.query(text)` returns the most relevant past episode using cosine similarity over BGE embeddings. Sub-5ms for stores up to ~100K episodes.
- **Two-tier store** -- hot tier (numpy + JSON) for fast similarity search; cold tier (SQLite) for full transcripts and metadata. No external database required.
- **Roleplay filter** -- 50+ keyword tells exclude fiction/RP sessions from factual recall. Prevents hallucination-amplification when your agent has mixed session types.
- **Temporal supersession** -- `ContradictionDetector` flags older episodes on the same topic as outdated when a newer one covers the same ground (sim ≥ 0.75, >1 day newer). Injected context leads with `[POSSIBLY OUTDATED -- N weeks later: ...]`.
- **Semantic tagging** -- zero-cost heuristic tags (`hardware`, `speculation`, `person`, `config`, `completed`, `error`, ...) with TTL-based auto-expiry. Filter recall by tag.

## Quick start

```bash
pip install episodic-memory
```

```python
from episodic_memory import RecallEngine

engine = RecallEngine(store_path="~/.my_agent/memory")
result = engine.query("what microphone setup did we use?")
if result:
    print(result.context_injection())  # ready-to-inject system prompt block
    if result.is_superseded:
        print(f"⚠ Superseded {result.supersession_age_gap_str} later")
```

`context_injection()` returns a formatted block like:

```
[Memory -- 0.82 similarity, 3 weeks ago]
The user and agent worked through a microphone input routing issue on an XMOS
eval board. The session ended with a successful test signal confirmed on the
LINE IN jack. Tone: collaborative, methodical.
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

The `latent` vector can be zeros if you don't have a custom encoder -- `RecallEngine` re-embeds summaries via BGE at query time regardless.

## Architecture

```
RecallEngine              ← start here (high-level API)
├── DirectTextResonance   ← BGE cosine search over cached summary embeddings
│   ├── RoleplayFilter    ← keyword triage, O(1), no embeddings
│   └── ContradictionDetector ← temporal supersession check
├── EpisodicRecall        ← fetches full episode + LLM-generated summary
└── EpisodicMemoryStore   ← two-tier hot (numpy) / cold (SQLite) store
```

Two-tier design mirrors neuroscience:
- **Fast path** (amygdala-style, <5ms) -- cosine similarity over pre-embedded summaries, blended emotional resonance
- **Slow path** (hippocampal, 100-500ms) -- triggered only when similarity exceeds `recall_threshold`; fetches transcript + generates/caches a natural-language gist via local LLM

## Embedding model

The default model is `BAAI/bge-small-en-v1.5` (133MB, loads in ~5s, good quality):

```python
# Default -- small and fast, good for most use cases
engine = RecallEngine(store_path="~/.my_agent/memory")

# Higher quality -- 1.3GB, ~30s load time
engine = RecallEngine(
    store_path="~/.my_agent/memory",
    embedding_model="BAAI/bge-large-en-v1.5",
)

# OpenAI embeddings -- zero local model, costs money
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
| `/query` | POST | Semantic recall → `RecallResult` or null |
| `/query_resonance` | POST | Fast path only → `ResonanceResult` |
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

Summaries are persisted to SQLite after generation -- the LLM is only called once per episode.

## Semantic tags

Episodes are tagged automatically at store time -- zero LLM cost:

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

The `_PERSON_PAT` regex in `tagger.py` contains a generic placeholder name list -- replace it with the names relevant to your agent and the people it interacts with.

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- sentence-transformers ≥ 2.2 (pulls in the embedding model)
- numpy ≥ 1.24
- faiss-cpu ≥ 1.7 (optional -- used if installed, falls back to numpy dot-product)

For summary generation: any OpenAI-compatible LLM endpoint (Ollama, LM Studio, vLLM, OpenAI API).

## License

MIT
