# episodic-memory - Hermes Agent Plugin

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) memory provider plugin that gives your agent persistent episodic memory across sessions - with zero turn latency, temporal contradiction detection, and roleplay filtering.

## What it adds

- **Automatic recall** - the most relevant past episode is injected into the system prompt before each turn. Uses a background prefetch (fires after the previous turn ends) so there's no added latency on the next turn.
- **Supersession detection** - if a newer episode covers the same topic, the older one is annotated `[POSSIBLY OUTDATED - N weeks later: ...]` rather than silently injected as current fact.
- **Roleplay filter** - fiction/RP sessions are excluded from factual recall automatically. BGE cosine similarity alone doesn't distinguish "we debugged a deploy" from "we roleplayed debugging a deploy" - the filter does.
- **`episodic_recall` tool** - your agent can deliberately search any topic at any time, beyond what was auto-prefetched.

## Install

### 1. Clone the repo

```bash
git clone https://github.com/f00stx/episodic-memory
cd episodic-memory
```

### 2. Run the unified install script

```bash
# Usage: ./scripts/install.sh <profile_path>
# Example:
./scripts/install.sh ~/.hermes/profiles/myprofile
```

This script:
- Installs the library (`uv pip install --force-reinstall`)
- Copies the plugin integration file into `~/.hermes/hermes-agent/plugins/`
- Creates a symlink in your profile's plugins dir
- Removes stale `.pyc` files

### 3. Download the BGE embedding model

```bash
# Downloads bge-small-en-v1.5 (133MB)
python scripts/download_model.py --small

# Or download bge-large-en-v1.5 (2.5GB)
python scripts/download_model.py --large
```

> **Note:** You can set `embedding_model: BAAI/bge-small-en-v1.5` in `config.yaml` to use the smaller model.

### 4. Configure your profile (`~/.hermes/profiles/myprofile/config.yaml`)

```yaml
memory:
  provider: episodic_memory
  flush_min_turns: 6
  episodic_memory:
    store_path: ~/.ctm/memory/myprofile
    embedding_model: BAAI/bge-small-en-v1.5
    recall_threshold: 0.50
```

See [Configuration reference](#configuration-reference) for all options.

### 5. Build a memory store for your profile

The store is built offline from your past sessions. The episodic-memory library ships an `EpisodicMemoryStore` API - use it to encode past sessions into the store, or use the Docker REST service to build it programmatically.

A minimal store-builder for Hermes session history:

```python
from episodic_memory import EpisodicMemoryStore
import numpy as np

store = EpisodicMemoryStore("~/.hermes/episodic_memory/my_profile")
store.add(
    session_id="2024_11_15_001",
    latent=np.zeros(256, dtype=np.float32),  # zeros if you don't have a custom encoder
    transcript=[
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."},
    ],
    summary="One-paragraph gist of the session.",
    metadata={"turn_count": 12},
)
```

See the [episodic-memory README](https://github.com/f00stx/episodic-memory) for the full store API and Docker REST service.

> **Note — the store starts empty.** On a fresh install, `episodes.db` has no entries and recall is a no-op. The plugin encodes sessions automatically as you use the agent (`flush_min_turns` controls how often — default 6 turns). After a handful of conversations the store will have enough content for recall to kick in. You can check progress at any time with `hermes episodic-memory status`. If you want to seed the store immediately from existing Hermes session history, see the store-builder example above.

### 6. Verify

In `~/.hermes/profiles/<your_profile>/config.yaml`:

```yaml
memory:
  provider: episodic_memory
  episodic_memory:
    store_path: ~/.hermes/episodic_memory/my_profile   # path containing episodes.db + hot_metadata.json
    embedding_model: BAAI/bge-small-en-v1.5            # or bge-large-en-v1.5 for higher quality
    recall_threshold: 0.55                              # lower = more recalls, more noise
    filter_roleplay: true
```

Or use the setup wizard:

```bash
hermes memory setup
```

### 7. Verify

```bash
hermes episodic-memory status    # episode count + store health
hermes episodic-memory stats     # emotion/topic distribution
hermes episodic-memory search "what did we work on last week?"
```

## Configuration reference

| Key | Default | Description |
|-----|---------|-------------|
| `store_path` | `$HERMES_HOME/episodic_memory/<profile>/` | Path to the store directory |
| `embedding_model` | `BAAI/bge-small-en-v1.5` | Model name or absolute local path |
| `recall_threshold` | `0.55` | Minimum cosine similarity to trigger recall injection |
| `filter_roleplay` | `true` | Exclude RP/fiction sessions from factual recall |
| `agent_name` | Profile directory name | Store subdirectory when `store_path` is unset |

## How it works

```
Hermes turn loop
    |
    +- queue_prefetch(last_user_msg)    -> background BGE query (fires after each turn)
    |                                      costs ~5-50ms CPU, zero turn latency
    |
    +- prefetch(current_user_msg)       -> returns cached recall block
    |   +- RecallEngine.query()             injected as "## Episodic Memory Context"
    |       +- DirectTextResonance          fast BGE cosine + roleplay filter
    |       +- EpisodicRecall              slow SQLite lookup (fires on sim >= 0.55)
    |
    +- episodic_recall tool             -> deliberate agent-triggered search
```

BGE runs on **CPU** - does not compete with your main LLM for VRAM.

First query lazily loads the embedding model (~2-5s for bge-small, ~10-30s for bge-large). Subsequent queries: ~5-50ms.

## One provider per profile

Hermes enforces one active memory provider per profile. If your profile already has another provider configured (e.g. `ctm_kg`), the episodic_memory plugin will be displaced. Check `memory.provider` in your config before activating.

## Notes

- The store is **read-only** from the plugin side - the plugin never writes to it during a conversation. Build/update the store offline.
- Profile isolation is automatic - each profile gets its own store subdirectory via the `hermes_home` path Hermes provides at init time.
- The `run_agent.py` config injection must be present for `store_path` and other settings to reach the plugin's `initialize()`. This is included in recent Hermes versions - if settings aren't loading, check that `kwargs.get("config")` is wired through in your version's `initialize_all()` call.
