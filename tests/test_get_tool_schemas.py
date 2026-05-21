from unittest.mock import Mock, patch

import importlib.util
import sys
from unittest.mock import Mock, patch

# Add the integrations/hermes dir to sys.path
sys.path.insert(0, "/home/richard/projects/episodic-memory/integrations/hermes")
# Import the plugin module directly from the path
def load_module_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

hermes_plugin = load_module_from_path("hermes_plugin", "/home/richard/projects/episodic-memory/integrations/hermes/__init__.py")

def test_get_tool_schemas_when_inactive():
    """Confirm get_tool_schemas() returns non-empty list even when _active=False."""
    with patch.object(hermes_plugin.EpisodicMemoryPlugin, '_active', False):
        plugin = hermes_plugin.EpisodicMemoryPlugin(Mock(), None)
        schemas = plugin.get_tool_schemas()
        assert len(schemas) > 0, "get_tool_schemas() should return tools even when inactive"
        assert schemas[0]['name'] in ['episodic_recall', 'episodic_search'], "Expected a known tool name"