import json
import sqlite3
import os
from pathlib import Path
from typing import Dict, Optional
from http.server import BaseHTTPRequestHandler, HTTPServer

# Check if sqlite3 is available
SQLITE_AVAILABLE = False
def _import_sqlite3():
    global SQLITE_AVAILABLE
    try:
        import sqlite3
        SQLITE_AVAILABLE = True
    except ImportError:
        SQLITE_AVAILABLE = False
_import_sqlite3()

def _gather_stats_sync(store: Optional[Path] = None) -> Dict:
    """Gather episode and ledger stats synchronously."""
    if store is None:
        store = Path.home() / ".ctm" / "memory" / "tars"
    stats = {}

    # -- episodes.db stats --
    db_path = store / "episodes.db"
    if SQLITE_AVAILABLE and db_path.exists():
        try:
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*), MIN(stored_at), MAX(stored_at) FROM episodes"
                )
                row = cursor.fetchone()
                stats.update({
                    "total_episodes": row[0] or 0,
                    "oldest_episode": row[1],
                    "newest_episode": row[2],
                    "store_size_mb": round(db_path.stat().st_size / 1_048_576, 2)
                })
        except Exception as e:
            print(f"Error querying episodes.db: {e}")

    # -- ingest ledger stats --
    ledger_paths = [
        store / "claude_ingested.db",
        store / "ingested.db",
    ]
    ledger = next((p for p in ledger_paths if p.exists()), None)
    if SQLITE_AVAILABLE and ledger:
        try:
            with sqlite3.connect(str(ledger)) as conn:
                # Discover the table name
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                table_name = tables[0][0] if tables else None
                if table_name:
                    lrow = conn.execute(f"SELECT COUNT(*), MAX(ingested_at) FROM {table_name}").fetchone()
                    stats["total_ingested"] = lrow[0] or 0
                    stats["last_ingest"] = lrow[1]
        except Exception as e:
            print(f"Error querying ingest ledger: {e}")

    # -- retrieval counts from mcp_events.jsonl --
    events_path = store / "mcp_events.jsonl"
    if events_path.exists():
        try:
            with open(events_path) as f:
                counts = {"get_recent_sessions": 0, "episodic_recall": 0, "session_search": 0}
                for line in f:
                    try:
                        event = json.loads(line)
                        event_name = event.get("event", "")
                        if event_name in counts:
                            counts[event_name] += 1
                    except json.JSONDecodeError:
                        continue
                stats.update({"retrieval_counts": counts})
        except Exception as e:
            print(f"Error reading mcp_events.jsonl: {e}")

    return stats

def _format_stats(stats: Dict) -> str:
    """Format stats into a human-readable string."""
    ep_total = stats.get("total_episodes", "—")
    size_mb = stats.get("store_size_mb", "—")
    oldest = stats.get("oldest_episode", "—")
    newest = stats.get("newest_episode", "—")
    last_ingest = stats.get("last_ingest", "—")
    total_ingested = stats.get("total_ingested", "—")
    retrieval_counts = stats.get("retrieval_counts", {"get_recent_sessions": "—", "episodic_recall": "—", "session_search": "—"})
    
    srch = retrieval_counts.get("session_search", "—")
    recall = retrieval_counts.get("episodic_recall", "—")
    recent = retrieval_counts.get("get_recent_sessions", "—")
    
    return f"""Episodic Memory Stats (TARS)
  Episodes:      {ep_total}  ({size_mb} MB)
  Date range:    {oldest} → {newest}
  Last ingest:   {last_ingest}
  Total ingested: {total_ingested} sessions
  Retrievals:    get_recent_sessions={recent}  episodic_recall={recall}  session_search={srch}"""

class MetricsHandler(BaseHTTPRequestHandler):
    def _send_response(self, content, content_type="text/html"):
        self.send_response(200)
        self.send_header("Content-type", content_type)
        self.end_headers()
        self.wfile.write(content.encode())

    def do_GET(self):
        store_path = Path(self.server.store_path)
        if self.path == "/":
            stats = _gather_stats_sync(store_path)
            summary = _format_stats(stats)
            
            # Get recent episodes
            recent_episodes = []
            if SQLITE_AVAILABLE and (store_path / "episodes.db").exists():
                try:
                    with sqlite3.connect(str(store_path / "episodes.db")) as conn:
                        rows = conn.execute("""
                            SELECT session_id, stored_at, dominant_emotion, 
                                   dominant_archetype, summary 
                            FROM episodes 
                            ORDER BY stored_at DESC 
                            LIMIT 10
                        """).fetchall()
                        for row in rows:
                            recent_episodes.append({
                                "session_id": row[0][:-12] + "..." if len(row[0]) > 12 else row[0],
                                "stored_at": row[1],
                                "emotion": row[2],
                                "archetype": row[3],
                                "summary": (row[4] or "")[0:120]
                            })
                except Exception as e:
                    print(f"Error fetching recent episodes: {e}")
            
            html = f"""<!DOCTYPE html>
<html><head><title>Episodic Memory Metrics</title></head>
<body><pre>{summary}\n\nRecent Episodes (max 10, newest first)\n{chr(10).join([f"  {e['session_id']} | {e['stored_at']} | {e['emotion']} | {e['archetype']} | {e['summary']}" for e in recent_episodes])}</pre></body>
</html>"""
            self._send_response(html)
        
        elif self.path == "/stats":
            stats = _gather_stats_sync(store_path)
            self._send_response(json.dumps(stats, indent=2), "application/json")
        
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8101, help="Port to run on")
    parser.add_argument("--store", type=str, default="~/.ctm/memory/tars", help="Store path")
    args = parser.parse_args()
    
    store_path = Path(args.store).expanduser()
    
    server = HTTPServer(("localhost", args.port), MetricsHandler)
    server.store_path = store_path
    print(f"Serving on http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()