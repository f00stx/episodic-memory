#!/bin/bash
#
# Unified install script for episodic-memory Hermes plugin
# Usage: ./scripts/install.sh <profile_path>
# Example: ./scripts/install.sh ~/.hermes/profiles/myprofile

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <profile_path>"
    echo "Example: $0 ~/.hermes/profiles/myprofile"
    exit 1
fi

PROFILE_PATH="$1"
PROFILE_NAME="$(basename "$PROFILE_PATH")"
HERMES_DIR="${PROFILE_PATH%/profiles/*}"
PLUGIN_DIR="$HERMES_DIR/hermes-agent/plugins/memory/episodic_memory"
VENV_DIR="$HERMES_DIR/hermes-agent/venv"

# Validate profile path
if [ ! -d "$PROFILE_PATH" ]; then
    echo "Error: Profile directory not found: $PROFILE_PATH"
    exit 1
fi

echo "Installing episodic-memory for profile: $PROFILE_NAME"
echo "Hermes root: $HERMES_DIR"
echo "Virtual environment: $VENV_DIR"

echo -n "1/4 Installing package... "
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Hermes venv not found at $VENV_DIR"
    exit 1
fi
# uv is a system binary, not inside the venv
if command -v uv &>/dev/null; then
    uv pip install --python "$VENV_DIR" --force-reinstall .
else
    "$VENV_DIR/bin/pip" install --force-reinstall .
fi
echo "OK"

echo "2/4 Copying plugin integration..."
mkdir -p "$PLUGIN_DIR"
cp -r integrations/hermes/* "$PLUGIN_DIR/"

echo "3/4 Cleaning stale bytecode..."
find "$PLUGIN_DIR" -name "*.pyc" -delete || true
find "$PLUGIN_DIR" -name "__pycache__" -type d -exec rm -rf {} + || true

echo "4/4 Creating plugin symlink..."
PLUGIN_SYMLINK="$PROFILE_PATH/plugins/episodic_memory"
mkdir -p "$(dirname "$PLUGIN_SYMLINK")"
ln -sf "$PLUGIN_DIR" "$PLUGIN_SYMLINK"

echo "Installation complete for profile '$PROFILE_NAME'."
echo "Run 'python scripts/download_model.py' to download the embedding model."
echo "See README for config and run_agent.py patch instructions."
