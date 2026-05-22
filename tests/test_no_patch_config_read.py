import os
import tempfile
import yaml
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

def test_initialize_reads_config_directly_when_kwargs_empty():
    # Create a temporary directory with a config.yaml
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create config.yaml with episodic_memory settings
        config_data = {
            "memory": {
                "provider": "episodic_memory",
                "episodic_memory": {
                    "store_path": "/tmp/test-store",
                    "recall_threshold": 0.42,
                    "embedding_model": "BAAI/bge-small-en-v1.5"
                }
            }
        }
        
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        
        # Import the provider and initialize it
        from integrations.hermes import EpisodicMemoryProvider
        provider = EpisodicMemoryProvider()
        
        # Initialize with no config in kwargs, but hermes_home pointing to our temp dir
        provider.initialize(session_id="test", hermes_home=str(tmp_path))
        
        # Assert that the provider picked up the config values
        assert provider._store_path == Path("/tmp/test-store")
        assert provider._recall_threshold == 0.42
        assert provider._embedding_model == "BAAI/bge-small-en-v1.5"


def test_kwargs_config_takes_precedence_over_direct_read():
    # Create a temporary directory with a config.yaml
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create config.yaml with episodic_memory settings
        config_data = {
            "memory": {
                "provider": "episodic_memory",
                "episodic_memory": {
                    "store_path": "/tmp/test-store",
                    "recall_threshold": 0.42,
                    "embedding_model": "BAAI/bge-small-en-v1.5"
                }
            }
        }
        
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        
        # Import the provider and initialize it with config in kwargs
        from integrations.hermes import EpisodicMemoryProvider
        provider = EpisodicMemoryProvider()
        
        provider.initialize(
            session_id="test", 
            hermes_home=str(tmp_path), 
            config={
                "store_path": "/tmp/kwargs-store", 
                "recall_threshold": 0.99
            }
        )
        
        # Assert that kwargs config takes precedence
        assert provider._store_path == Path("/tmp/kwargs-store")
        assert provider._recall_threshold == 0.99
        # embedding_model should come from the default since it wasn't in kwargs
        assert provider._embedding_model == "BAAI/bge-large-en-v1.5"