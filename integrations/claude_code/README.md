# Claude Code Integration

Ingests Claude Code session transcripts into the episodic memory store.
Replaces the Hermes `on_session_end` write path for users who have moved
to Claude Code without Hermes.

## How it works

Claude Code stores each session as a JSONL file under
`~/.claude/projects/<sanitised-cwd>/<session-uuid>.jsonl`.

`ingest_sessions.py` scans these files, parses the conversation turns,
embeds them with BGE, encodes via EpisodicEncoder, and writes to the
episodic store -- the same pipeline the Hermes plugin uses.

A ledger (SQLite) tracks which sessions have been ingested so re-runs
are safe and idempotent.

## Usage

```bash
# Dry run -- see what would be ingested
python3 ingest_sessions.py --dry-run

# Ingest everything new into the default TARS store
python3 ingest_sessions.py

# Custom store path
python3 ingest_sessions.py --store ~/.ctm/memory/thufir

# Verbose per-session output
python3 ingest_sessions.py --verbose

# Smaller model (133MB vs 1.3GB -- slightly lower quality)
python3 ingest_sessions.py --model BAAI/bge-small-en-v1.5
```

## Automating with cron

Run once per day (or on login) to keep the store current:

```bash
# crontab -e
0 3 * * * /path/to/venv/bin/python /path/to/ingest_sessions.py >> ~/.ctm/memory/ingest.log 2>&1
```

Or trigger manually at the end of a work session.

## Dependencies

Same as the main episodic-memory library:

```bash
pip install episodic-memory sentence-transformers
```

`sentence-transformers` is needed to load the BGE model. The episodic-memory
library itself pulls in numpy; torch is required by EpisodicEncoder.

## Notes

- Sessions with fewer than 3 turns are skipped by default (`--min-turns`).
  They are still recorded in the ledger so they are not revisited.
- The `source: claude_code` metadata field distinguishes these episodes
  from Hermes-ingested ones in the store.
- The `project` field is inferred from the Claude projects folder name
  (e.g. `-home-richard-projects-gimli` → `gimli`).
- Emotion and archetype fields are set to neutral/companion defaults.
  A future upgrade could run the emotion probe over the turn embeddings.
