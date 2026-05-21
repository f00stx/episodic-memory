"""
Safely patch Hermes run_agent.py to inject episodic-memory config.

This adds the config block extraction to the initialize_all() call so that
store_path, recall_threshold, etc. are passed to the plugin's initialize().

Usage:
    python scripts/patch_run_agent.py <hermes_root>
    Example: python scripts/patch_run_agent.py ~/.hermes/hermes-agent

The patch is idempotent — safe to run multiple times.
"""

import os
import argparse
from pathlib import Path
import re

def main():
    parser = argparse.ArgumentParser(description="Patch Hermes run_agent.py for episodic-memory config injection")
    parser.add_argument("hermes_root", help="Path to Hermes root (e.g. ~/.hermes/hermes-agent)")
    args = parser.parse_args()

    hermes_path = Path(args.hermes_root)
    run_agent_py = hermes_path / "run_agent.py"

    if not run_agent_py.exists():
        print(f"Error: run_agent.py not found at {run_agent_py}")
        return 1

    # Read file content
    content = run_agent_py.read_text(encoding="utf-8")

    # Check if already patched
    if "memory[\"episodic_memory\"]" in content:
        print("run_agent.py is already patched for episodic_memory config injection.")
        return 0

    # Pattern to find initialize_all() call
    initialize_pattern = re.compile(
        r"(initialize_all\(\s*[^)]*config=([^)]*))\)"
    )
    match = initialize_pattern.search(content)

    if not match:
        print("Error: Could not find initialize_all() call in run_agent.py")
        return 1

    # Build the patch
    config_arg = match.group(2)
    if config_arg.strip() == "":
        # No config arg, add it
        insert_pos = match.end() - 1
        new_config_arg = "config"
        patch = f", memory[\"episodic_memory\"]={new_config_arg}"
    else:
        # Config arg exists, append to it
        insert_pos = match.end() - 1
        patch = f", memory[\"episodic_memory\"]={config_arg}"

    # Insert the patch
    new_content = content[:insert_pos] + patch + content[insert_pos:]
    run_agent_py.write_text(new_content, encoding="utf-8")

    print(f"Patched {run_agent_py} successfully.")
    print("Config block will now be injected into episodic_memory plugin.")
    return 0


if __name__ == "__main__":
    exit(main())
