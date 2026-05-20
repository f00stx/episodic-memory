"""CLI commands for the episodic_memory provider.

Registered as: hermes episodic-memory <subcommand>
Only active when episodic_memory is the configured memory provider.

Subcommands:
    status      — show store health, episode count, last recall
    stats       — emotion/archetype distribution, roleplay ratio
    search <q>  — query the store and print the top result
    path        — print the resolved store path
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _get_engine(store_path: str | None, model: str = "BAAI/bge-large-en-v1.5"):
    """Load a RecallEngine from the store path.  Exits with message if unavailable."""
    try:
        from episodic_memory import RecallEngine
    except ImportError:
        print("ERROR: episodic-memory package not installed.")
        print("  pip install git+https://github.com/f00stx/episodic-memory")
        sys.exit(1)

    if not store_path:
        print("ERROR: no store_path configured. Set memory.episodic_memory.store_path in config.yaml")
        sys.exit(1)

    p = Path(store_path).expanduser()
    if not (p / "episodes.db").exists():
        print(f"ERROR: No episodes.db at {p}")
        print("  See the episodic-memory README for store build instructions.")
        sys.exit(1)

    return RecallEngine(
        store_path=str(p),
        embedding_device="cpu",
        embedding_model=model,
    )


def _load_config(hermes_home: str | None) -> dict:
    """Read episodic_memory.json config, falling back to empty dict."""
    if not hermes_home:
        return {}
    cfg_path = Path(hermes_home) / "episodic_memory.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text())
        except Exception:
            return {}
    return {}


def episodic_memory_cmd(args):
    """Dispatch handler called by argparse."""
    import os

    hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    config = _load_config(hermes_home)
    store_path = config.get("store_path") or str(Path(hermes_home) / "episodic_memory")
    model = config.get("embedding_model", "BAAI/bge-large-en-v1.5")

    sub = getattr(args, "episodic_memory_cmd", None)

    if sub == "path":
        print(f"Store path: {Path(store_path).expanduser().resolve()}")
        return

    if sub == "status":
        engine = _get_engine(store_path, model)
        p = Path(store_path).expanduser()
        db_size = (p / "episodes.db").stat().st_size // 1024
        hot_size = (p / "hot_metadata.json").stat().st_size // 1024
        print(f"Store:     {p}")
        print(f"Episodes:  {engine.n_episodes}")
        print(f"episodes.db:       {db_size} KB")
        print(f"hot_metadata.json: {hot_size} KB")
        print(f"Model:     {model}")
        print(f"Threshold: {engine._recall_threshold}")
        print(f"Status:    ✓ healthy")
        return

    if sub == "stats":
        # Load hot metadata and summarise emotion/archetype distribution
        p = Path(store_path).expanduser()
        hot_path = p / "hot_metadata.json"
        if not hot_path.exists():
            print("ERROR: hot_metadata.json not found")
            sys.exit(1)

        hot = json.loads(hot_path.read_text())
        episodes = list(hot.values()) if isinstance(hot, dict) else hot

        emotion_counts: dict = {}
        arch_counts: dict = {}
        roleplay_count = 0
        annotated = 0

        for ep in episodes:
            dom_e = ep.get("dominant_emotion")
            dom_a = ep.get("dominant_archetype")
            is_rp = ep.get("is_roleplay", False)
            if dom_e:
                emotion_counts[dom_e] = emotion_counts.get(dom_e, 0) + 1
                annotated += 1
            if dom_a:
                arch_counts[dom_a] = arch_counts.get(dom_a, 0) + 1
            if is_rp:
                roleplay_count += 1

        total = len(episodes)
        print(f"Total episodes: {total}  |  Annotated: {annotated}  |  Roleplay: {roleplay_count} ({roleplay_count*100//max(total,1)}%)\n")

        print("Dominant emotion distribution:")
        for e, c in sorted(emotion_counts.items(), key=lambda x: -x[1]):
            bar = "█" * (c * 30 // max(emotion_counts.values(), default=1))
            print(f"  {e:<14} {c:>4}  {bar}")

        print("\nDominant archetype distribution:")
        for a, c in sorted(arch_counts.items(), key=lambda x: -x[1]):
            bar = "█" * (c * 30 // max(arch_counts.values(), default=1))
            print(f"  {a:<14} {c:>4}  {bar}")
        return

    if sub == "search":
        query = " ".join(args.query_terms or [])
        if not query:
            print("Usage: hermes episodic-memory search <query terms>")
            sys.exit(1)

        engine = _get_engine(store_path, model)
        print(f"Querying: \"{query}\"")
        print(f"Store:    {Path(store_path).expanduser()}\n")

        result = engine.query(query)
        if result is None:
            print("No relevant episode found (below recall threshold).")
            print(f"Threshold: {engine._recall_threshold}")
        else:
            print(result.context_injection())
            print(f"\nSimilarity:       {result.similarity:.3f}")
            print(f"Dominant emotion: {result.dominant_emotion}")
            print(f"Is superseded:    {result.is_superseded}")
        return

    # Default: show help
    print("Usage: hermes episodic-memory <status|stats|search|path>")
    print()
    print("  status       Show store health and episode count")
    print("  stats        Emotion/archetype distribution and roleplay ratio")
    print("  search <q>   Semantic search and print the top recalled episode")
    print("  path         Print the resolved store path")


def register_cli(subparser) -> None:
    """Build the hermes episodic-memory argparse tree."""
    subs = subparser.add_subparsers(dest="episodic_memory_cmd")

    subs.add_parser("status", help="Show store health and episode count")
    subs.add_parser("stats", help="Emotion/archetype distribution and roleplay ratio")

    search_p = subs.add_parser("search", help="Semantic search and print the top result")
    search_p.add_argument("query_terms", nargs="+", help="Query terms (natural language)")

    subs.add_parser("path", help="Print the resolved store path")

    subparser.set_defaults(func=episodic_memory_cmd)
