from unittest.mock import Mock, patch

# Import the plugin module directly by loading the file
import sys
from pathlib import Path

# Ensure the integrations/hermes dir is in sys.path
sys.path.insert(0, str(Path("/home/richard/projects/episodic-memory/integrations/hermes").absolute()))
import __init__ as hermes_plugin


def test_get_tool_schemas_when_inactive():
    """Confirm get_tool_schemas() returns non-empty list even when _active=False."""
    provider = hermes_plugin.EpisodicMemoryProvider()
    provider._active = False  # Manually deactivate
    schemas = provider.get_tool_schemas()
    assert len(schemas) > 0, "get_tool_schemas() should return schemas even when inactive"
    assert schemas[0]["name"] == "episodic_recall", "Expected episodic_recall tool schema"

def test_get_tool_schemas_when_active():
    """Confirm get_tool_schemas() returns non-empty list when _active=True."""
    provider = hermes_plugin.EpisodicMemoryProvider()
    provider._active = True
    schemas = provider.get_tool_schemas()
    assert len(schemas) > 0, "get_tool_schemas() should return schemas when active"
    assert schemas[0]["name"] == "episodic_recall", "Expected episodic_recall tool schema"