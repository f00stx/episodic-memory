# episodic-memory

Standalone episodic memory system with **roleplay filtering** and **temporal contradiction detection**. Extracted from [Project Aura / CTM](https://gitea.rickamai.com/richard/ctm).

## What it does

- **Two-tier store** -- hot (numpy + JSON) for fast similarity search, cold (SQLite) for full transcripts and metadata.
- **BGE-backed recall** -- `RecallEngine` queries the store with cosine similarity, returns the most relevant past episode above a configurable threshold.
- **Roleplay filter** -- 50+ keyword tells (explicit and summary-abstract) exclude fiction/RP sessions from factual recall. Prevents hallucination-amplification.
- **Temporal supersession** -- `ContradictionDetector` flags older episodes on the same topic as outdated when a newer one covers the same ground (sim ≥ 0.75, >1 day newer). The injected recall text leads with `[POSSIBLY OUTDATED -- N weeks later: ...]`.

## Quick start

```python
from episodic_memory import RecallEngine

engine = RecallEngine(store_path="~/.my_agent/memory")
result = engine.query("what microphone setup did we use?")
if result:
    print(result.context_injection())  # ready-to-inject system prompt block
    if result.is_superseded:
        print(f"⚠ Superseded {result.supersession_age_gap_str} later")
```

## Architecture

```
RecallEngine          ← start here (high-level API)
├── RoleplayFilter    ← keyword triage, O(1), no embeddings
├── DirectTextResonance ← BGE cosine search over summary embeddings
├── EpisodicRecall    ← fetches full episode + builds RecallResult
├── ContradictionDetector ← temporal supersession check
└── EpisodicMemoryStore   ← two-tier hot/cold store (SQLite + numpy)
```

### Optional: CTM encoder

`RecallEngine` defaults to BGE (`BAAI/bge-large-en-v1.5`) for embeddings. If you have a CTM `EpisodicEncoder` checkpoint, pass it via:

```python
engine = RecallEngine(
    store_path="~/.my_agent/memory",
    embedding_model="BAAI/bge-large-en-v1.5",  # or path to local model
)
```

## Building a store

Use `CTMSession.on_turn()` + `on_session_end()` (from the CTM project) to encode sessions, or write directly to `EpisodicMemoryStore`:

```python
from episodic_memory import EpisodicMemoryStore

store = EpisodicMemoryStore("~/.my_agent/memory")
store.store_episode(
    session_id="2026_05_04_001",
    transcript=[{"role": "user", "content": "..."}, ...],
    summary="Brief LLM-generated summary of the session",
    agent_emb=my_768_dim_vector,   # numpy float32
    emotion_cats=my_8_dim_vector,
    dominant_emotion="joy",
    dominant_archetype="companion",
)
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- sentence-transformers ≥ 2.2
- faiss-cpu ≥ 1.7

## License

MIT
