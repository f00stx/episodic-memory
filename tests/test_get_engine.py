"""
Tests for EpisodicMemoryProvider._get_engine().

Covers:
- Returns None when _active=False
- Returns None when store files don't exist
- Returns None when store_path is None
- Does NOT raise NameError (regression for kwargs bug in self-diagnostic block)
- Logs warning when store_path looks like fallback default
- Returns a RecallEngine when store files exist (mocked)
- Engine is cached after first successful load (same object returned twice)
"""

import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

# Match the PYTHONPATH used by the existing test suite:
# dummy_agent/src provides the 'agent' stub, integrations/hermes provides the plugin
_repo_root = Path("/home/richard/projects/episodic-memory")
sys.path.insert(0, str(_repo_root / "dummy_agent" / "src"))
sys.path.insert(0, str(_repo_root / "integrations" / "hermes"))
import __init__ as hermes_plugin

EpisodicMemoryProvider = hermes_plugin.EpisodicMemoryProvider


def _make_provider(store_path=None, active=True):
    """Create a provider with minimal initialisation for _get_engine() testing."""
    provider = EpisodicMemoryProvider()
    provider._active = active
    provider._store_path = Path(store_path) if store_path else None
    provider._engine = None
    provider._embedding_model = "BAAI/bge-large-en-v1.5"
    provider._recall_threshold = 0.55
    provider._filter_roleplay = True
    return provider


# --- Inactive / missing state ---

def test_get_engine_returns_none_when_inactive():
    """_get_engine() must return None immediately when _active=False."""
    provider = _make_provider(store_path="/tmp/fake_store", active=False)
    result = provider._get_engine()
    assert result is None


def test_get_engine_returns_none_when_store_path_is_none():
    """_get_engine() must return None when _store_path is None."""
    provider = _make_provider(store_path=None, active=True)
    result = provider._get_engine()
    assert result is None


def test_get_engine_returns_none_when_store_files_missing(tmp_path):
    """_get_engine() must return None when episodes.db or hot_metadata.json don't exist."""
    provider = _make_provider(store_path=str(tmp_path), active=True)
    # tmp_path is empty — no episodes.db, no hot_metadata.json
    result = provider._get_engine()
    assert result is None


def test_get_engine_returns_none_when_only_db_exists(tmp_path):
    """_get_engine() must return None when only episodes.db exists (hot_metadata missing)."""
    (tmp_path / "episodes.db").touch()
    provider = _make_provider(store_path=str(tmp_path), active=True)
    result = provider._get_engine()
    assert result is None


def test_get_engine_returns_none_when_only_hot_metadata_exists(tmp_path):
    """_get_engine() must return None when only hot_metadata.json exists (db missing)."""
    (tmp_path / "hot_metadata.json").write_text("{}")
    provider = _make_provider(store_path=str(tmp_path), active=True)
    result = provider._get_engine()
    assert result is None


# --- Regression: NameError on kwargs ---

def test_get_engine_no_kwargs_nameerror(tmp_path):
    """
    Regression test: _get_engine() must not raise NameError for 'kwargs'.

    Previously the self-diagnostic block referenced kwargs.get(...) but _get_engine()
    has no kwargs parameter. This test ensures the bug does not regress.
    """
    # Store path that matches the fallback pattern to trigger the diagnostic block
    fallback_store = tmp_path / "episodic_memory" / tmp_path.name
    fallback_store.mkdir(parents=True)
    (fallback_store / "episodes.db").touch()
    (fallback_store / "hot_metadata.json").write_text("{}")

    provider = _make_provider(store_path=str(fallback_store), active=True)

    # Should not raise NameError — if kwargs bug is present this will throw
    with patch("episodic_memory.RecallEngine") as mock_engine_cls:
        mock_engine_cls.return_value = MagicMock()
        try:
            provider._get_engine()
        except NameError as e:
            raise AssertionError(f"NameError regression: {e}") from e


# --- Fallback path warning ---

def test_get_engine_warns_on_fallback_store_path(tmp_path, caplog):
    """_get_engine() should log a WARNING when store_path matches the fallback pattern."""
    # Construct a path that ends in /episodic_memory/<name>
    agent_name = tmp_path.name
    fallback_store = tmp_path / "episodic_memory" / agent_name
    fallback_store.mkdir(parents=True)
    (fallback_store / "episodes.db").touch()
    (fallback_store / "hot_metadata.json").write_text("{}")

    provider = _make_provider(store_path=str(fallback_store), active=True)

    with patch("episodic_memory.RecallEngine") as mock_engine_cls:
        mock_engine_cls.return_value = MagicMock()
        with caplog.at_level(logging.WARNING):
            provider._get_engine()

    assert any("fallback" in r.message.lower() or "store_path" in r.message.lower()
               for r in caplog.records), "Expected a WARNING about fallback store_path"


def test_get_engine_no_warning_for_explicit_store_path(tmp_path, caplog):
    """_get_engine() should NOT warn when store_path doesn't match the fallback pattern."""
    explicit_store = tmp_path / "my_custom_store"
    explicit_store.mkdir()
    (explicit_store / "episodes.db").touch()
    (explicit_store / "hot_metadata.json").write_text("{}")

    provider = _make_provider(store_path=str(explicit_store), active=True)

    with patch("episodic_memory.RecallEngine") as mock_engine_cls:
        mock_engine_cls.return_value = MagicMock()
        with caplog.at_level(logging.WARNING):
            provider._get_engine()

    fallback_warnings = [r for r in caplog.records
                         if "fallback" in r.message.lower() and r.levelno == logging.WARNING]
    assert not fallback_warnings, f"Unexpected fallback warning: {fallback_warnings}"


# --- Successful load and caching ---

def test_get_engine_returns_engine_when_store_exists(tmp_path):
    """_get_engine() should return a RecallEngine when both store files exist."""
    (tmp_path / "episodes.db").touch()
    (tmp_path / "hot_metadata.json").write_text("{}")

    provider = _make_provider(store_path=str(tmp_path), active=True)
    mock_engine = MagicMock()

    with patch("episodic_memory.RecallEngine", return_value=mock_engine):
        result = provider._get_engine()

    assert result is mock_engine


def test_get_engine_caches_engine(tmp_path):
    """_get_engine() should return the same engine instance on repeated calls."""
    (tmp_path / "episodes.db").touch()
    (tmp_path / "hot_metadata.json").write_text("{}")

    provider = _make_provider(store_path=str(tmp_path), active=True)
    mock_engine = MagicMock()

    with patch("episodic_memory.RecallEngine", return_value=mock_engine) as mock_cls:
        first = provider._get_engine()
        second = provider._get_engine()

    assert first is second, "Engine should be cached — same object on second call"
    assert mock_cls.call_count == 1, "RecallEngine should only be instantiated once"
