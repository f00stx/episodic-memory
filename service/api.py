"""episodic-memory REST API service.

Thin Flask wrapper around RecallEngine.  Used when agents want to query
episodic memory over HTTP (e.g. Docker-based deployments, multi-agent setups)
rather than importing the library directly.

Endpoints:
    GET  /health               -- liveness + episode count
    POST /query                -- semantic recall (returns RecallResult or null)
    POST /query_resonance      -- fast path only (returns ResonanceResult)
    GET  /stats                -- store statistics (episode count, emotion dist)

Configuration via environment variables:
    STORE_PATH          Path to episodes.db + hot_metadata.json (required)
    EMBEDDING_MODEL     BGE model name or path (default: BAAI/bge-large-en-v1.5)
    RECALL_THRESHOLD    Float, default 0.55
    FILTER_ROLEPLAY     "true"/"false", default "true"
    PORT                HTTP port, default 8099

The service is intentionally read-only -- the store is built offline by
encode_*_memories.py scripts and mounted as a volume.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STORE_PATH = os.environ.get("STORE_PATH", "")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
RECALL_THRESHOLD = float(os.environ.get("RECALL_THRESHOLD", "0.55"))
FILTER_ROLEPLAY = os.environ.get("FILTER_ROLEPLAY", "true").lower() == "true"
PORT = int(os.environ.get("PORT", "8099"))

# ---------------------------------------------------------------------------
# RecallEngine -- lazy singleton
# ---------------------------------------------------------------------------

_engine = None


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine

    if not STORE_PATH:
        raise RuntimeError("STORE_PATH environment variable is required")

    p = Path(STORE_PATH).expanduser()
    if not (p / "episodes.db").exists():
        raise RuntimeError(f"episodes.db not found at {p}")

    from episodic_memory import RecallEngine

    _engine = RecallEngine(
        store_path=str(p),
        recall_threshold=RECALL_THRESHOLD,
        filter_roleplay=FILTER_ROLEPLAY,
        embedding_device="cpu",
        embedding_model=EMBEDDING_MODEL,
    )
    logger.info("RecallEngine loaded: %d episodes at %s", _engine.n_episodes, p)
    return _engine


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/health")
def health():
    try:
        engine = _get_engine()
        return jsonify({"status": "ok", "n_episodes": engine.n_episodes, "store": STORE_PATH})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 503


@app.route("/query", methods=["POST"])
def query():
    """Semantic recall -- returns the top RecallResult or null.

    Request body:
        {
          "query": "text to search for",
          "exclude_session_id": "optional session id to exclude"
        }

    Response:
        {
          "result": null | {
            "summary": "...",
            "context_injection": "...",    // formatted for system-prompt injection
            "similarity": 0.72,
            "dominant_emotion": "trust",
            "is_superseded": false,
            "superseded_by_summary": null,
            "supersession_age_gap_str": null
          }
        }
    """
    body = request.get_json(force=True, silent=True) or {}
    text = (body.get("query") or "").strip()
    if not text:
        return jsonify({"error": "query is required"}), 400

    exclude = body.get("exclude_session_id")

    try:
        engine = _get_engine()
        result = engine.query(text[:500], exclude_session_id=exclude)
        if result is None:
            return jsonify({"result": None})
        return jsonify({
            "result": {
                "summary":                 result.summary,
                "context_injection":       result.context_injection(),
                "similarity":              float(result.similarity),
                "dominant_emotion":        result.dominant_emotion,
                "is_superseded":           result.is_superseded,
                "superseded_by_summary":   result.superseded_by_summary,
                "supersession_age_gap_str": result.supersession_age_gap_str,
            }
        })
    except Exception as e:
        logger.exception("Query error")
        return jsonify({"error": str(e)}), 500


@app.route("/query_resonance", methods=["POST"])
def query_resonance():
    """Fast path only -- emotional resonance without slow-path recall.

    Request body: {"query": "text"}
    Response: {"triggered_recall": bool, "blend": {...}, "top_k_ids": [...]}
    """
    body = request.get_json(force=True, silent=True) or {}
    text = (body.get("query") or "").strip()
    if not text:
        return jsonify({"error": "query is required"}), 400

    try:
        engine = _get_engine()
        res = engine.query_resonance(text[:500])
        return jsonify({
            "triggered_recall":     res.triggered_recall,
            "max_similarity":       float(res.max_similarity) if res.max_similarity else 0.0,
            "top_k_ids":            res.top_k_ids or [],
            "top_k_similarities":   [float(s) for s in (res.top_k_similarities or [])],
            "blend": {
                "dominant_emotion":   res.blend.dominant_emotion   if res.blend else None,
                "dominant_archetype": res.blend.dominant_archetype if res.blend else None,
            } if res.blend else None,
        })
    except Exception as e:
        logger.exception("Resonance query error")
        return jsonify({"error": str(e)}), 500


@app.route("/stats")
def stats():
    """Episode count + emotion/archetype distribution."""
    import json

    store_p = Path(STORE_PATH).expanduser()
    hot_path = store_p / "hot_metadata.json"
    if not hot_path.exists():
        return jsonify({"error": "hot_metadata.json not found"}), 404

    hot = json.loads(hot_path.read_text())
    episodes = list(hot.values()) if isinstance(hot, dict) else hot

    emotion_counts: dict = {}
    arch_counts: dict = {}
    roleplay_count = 0

    for ep in episodes:
        dom_e = ep.get("dominant_emotion")
        dom_a = ep.get("dominant_archetype")
        if dom_e:
            emotion_counts[dom_e] = emotion_counts.get(dom_e, 0) + 1
        if dom_a:
            arch_counts[dom_a] = arch_counts.get(dom_a, 0) + 1
        if ep.get("is_roleplay"):
            roleplay_count += 1

    return jsonify({
        "total_episodes":    len(episodes),
        "roleplay_episodes": roleplay_count,
        "emotion_distribution":   emotion_counts,
        "archetype_distribution": arch_counts,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Eagerly load engine at startup (warm BGE model before first request)
    try:
        _get_engine()
    except Exception as e:
        logger.warning("Engine pre-warm failed (will retry on first request): %s", e)

    app.run(host="0.0.0.0", port=PORT, debug=False)
