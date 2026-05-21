# HN Post

## Title

Show HN: episodic-memory - persistent memory for AI agents with temporal contradiction detection

## Body

I built this while working on a long-running personal AI agent that needed to remember things across hundreds of sessions without either bloating the context window or hallucinating about its own past.

The three problems I kept hitting that existing solutions didn't solve well:

**Contradiction** - if the user's setup changed last month, the old memory is worse than useless. It silently poisons the context. Most vector DB approaches return it anyway because similarity doesn't encode recency.

**Fiction bleed** - agents that do both factual work and roleplay/creative sessions will hallucinate across the boundary. "We debugged the deployment" and "we roleplayed debugging the deployment" have near-identical embeddings. The library ships a heuristic filter (50+ keyword tells, O(1), no embeddings) that excludes RP sessions from factual recall.

**Retrieval latency** - a full embedding lookup on every turn is overkill. Most turns don't need episodic context. The library uses a two-tier store: a numpy/JSON hot path for fast cosine similarity (<5ms up to ~100K episodes) and SQLite cold storage for full transcripts. A background prefetch fires after each turn so the next turn's context is ready with zero added latency.

The temporal supersession piece is the bit I haven't seen elsewhere - when a newer episode covers the same topic (sim >= 0.75, >1 day newer), the older one is flagged and the injected context leads with "[POSSIBLY OUTDATED - N weeks later: ...]" so the agent knows to treat it with skepticism rather than as current fact.

Stack: Python, sentence-transformers (BGE), SQLite, numpy. Optional faiss. No required external services - SQLite is the only storage backend. Docker REST service included for multi-agent deployments.

Also includes a drop-in plugin for Hermes Agent (NousResearch) if you're using that.

https://github.com/f00stx/episodic-memory

Happy to answer questions about the architecture, particularly the two-tier store design or the contradiction detection approach.
