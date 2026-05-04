"""episodic-memory REST API service.

Thin Flask wrapper around RecallEngine.  Used when agents want to query
episodic memory over HTTP (e.g. Docker-based deployments, multi-agent setups)
rather than importing the library directly.

Endpoints:
    GET  /health               -- liveness + episode count
    POST /query                -- semantic recall (returns RecallResult or null)
    POST /query_resonance      -- fast path only (returns ResonanceResult)
    GET  /stats                -- store statistics (episode count, emotion dist)
    GET  /tags                 -- tag vocabulary with counts and expiry stats

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

import json
import logging
import os
import sqlite3
import time
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
          "exclude_session_id": "optional session id to exclude",
          "exclude_tags": ["roleplay", "speculation"],
          "only_tags": ["hardware"],
          "include_expired": false
        }

    Response: RecallResult dict or null
    """
    body = request.get_json(force=True, silent=True) or {}
    text = (body.get("query") or body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "query is required"}), 400

    exclude_session_id = body.get("exclude_session_id")
    exclude_tags       = body.get("exclude_tags") or None
    only_tags          = body.get("only_tags") or None
    include_expired    = bool(body.get("include_expired", False))

    try:
        engine = _get_engine()
        result = engine.query(
            text[:500],
            exclude_session_id=exclude_session_id,
            exclude_tags=exclude_tags,
            only_tags=only_tags,
            include_expired=include_expired,
        )
        if result is None:
            return jsonify(None)
        return jsonify({
            "session_id":             result.session_id,
            "summary":                result.summary,
            "context_injection":      result.context_injection(),
            "similarity":             float(result.similarity),
            "dominant_emotion":       result.dominant_emotion,
            "dominant_archetype":     result.dominant_archetype,
            "turn_count":             result.turn_count,
            "stored_at":              result.stored_at,
            "is_superseded":          result.is_superseded,
            "superseded_by":          result.superseded_by,
            "superseded_by_summary":  result.superseded_by_summary,
            "supersession_age_gap":   result.supersession_age_gap_str,
        })
    except Exception as e:
        logger.exception("query error")
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
            "triggered_recall":    res.triggered_recall,
            "max_similarity":      float(res.max_similarity),
            "resonance_strength":  float(res.resonance_strength),
            "top_k_ids":           res.top_k_ids,
            "top_k_similarities":  [float(s) for s in res.top_k_similarities],
        })
    except Exception as e:
        logger.exception("query_resonance error")
        return jsonify({"error": str(e)}), 500


@app.route("/stats")
def stats():
    """Store statistics -- episode count, emotion distribution, tag distribution."""
    try:
        engine = _get_engine()
        p = Path(STORE_PATH).expanduser()

        conn = sqlite3.connect(str(p / "episodes.db"))
        emotion_rows = conn.execute(
            "SELECT dominant_emotion, COUNT(*) FROM episodes GROUP BY dominant_emotion ORDER BY 2 DESC"
        ).fetchall()
        arch_rows = conn.execute(
            "SELECT dominant_archetype, COUNT(*) FROM episodes GROUP BY dominant_archetype ORDER BY 2 DESC"
        ).fetchall()
        conn.close()

        return jsonify({
            "n_episodes":           engine.n_episodes,
            "emotion_distribution": dict(emotion_rows),
            "archetype_distribution": dict(arch_rows),
        })
    except Exception as e:
        logger.exception("stats error")
        return jsonify({"error": str(e)}), 500


@app.route("/tags")
def tags():
    """Tag vocabulary with counts and expiry statistics.

    Returns:
        {
          "tag_counts": {"hardware": 42, "project": 120, ...},
          "with_ttl": 55,
          "expired": 3,
          "vocabulary": ["completed", "config", "date_sensitive", ...]
        }
    """
    try:
        from episodic_memory.tagger import ALL_TAGS
        p = Path(STORE_PATH).expanduser()
        conn = sqlite3.connect(str(p / "episodes.db"))

        rows = conn.execute("SELECT tags, expires_at FROM episodes").fetchall()
        conn.close()

        tag_counts: dict[str, int] = {t: 0 for t in ALL_TAGS}
        with_ttl   = 0
        expired    = 0
        now        = time.time()

        for row in rows:
            try:
                ep_tags = json.loads(row[0] or "[]")
            except json.JSONDecodeError:
                ep_tags = []
            for t in ep_tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1
            exp = row[1]
            if exp is not None:
                with_ttl += 1
                if now > float(exp):
                    expired += 1

        # Drop zero-count tags from output
        tag_counts = {k: v for k, v in tag_counts.items() if v > 0}

        return jsonify({
            "tag_counts": tag_counts,
            "with_ttl":   with_ttl,
            "expired":    expired,
            "vocabulary": sorted(ALL_TAGS),
        })
    except Exception as e:
        logger.exception("tags error")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
