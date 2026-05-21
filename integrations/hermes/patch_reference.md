# run_agent.py Patch Reference

This document explains the patch needed to enable config injection for the episodic-memory plugin in Hermes Agent.

## Why is this patch needed?

Hermes Agent has a security design where plugin configuration blocks (like `episodic_memory:`) are NOT passed by default into the plugin's `initialize()` method. They are only passed if the main `initialize_all()` function explicitly extracts them from the `config` and includes them in the `**kwargs` passed to the plugin.

If this patch is missing, any settings under `episodic_memory:` in your `config.yaml` (like `store_path`, `recall_threshold`) will be silently ignored, and the plugin will fall back to its defaults.

## The patch

In `/home/richard/.hermes/hermes-agent/src/hermes/plugins/memory/__init__.py`, find the `initialize_all()` function and ensure it extracts the `episodic_memory` block from the config, like this:

```python
# Inside initialize_all(config: dict) -> dict:

# Extract episodic_memory config block to pass to plugin
episodic_memory_config = {
    k: v for k, v in config.items()
    if k in ("provider", "flush_min_turns")  # Hermes standard keys
} | {
    k: v for k, v in config.get("episodic_memory", {}).items()
}

# Later, in the plugin init call:
plugin_instance.initialize(storage_path=storage_path, **episodic_memory_config)
```

## Verification

After applying the patch and restarting the gateway, check:

1. Your `store_path` from `config.yaml` appears in the logs
2. Run `hermes episodic-memory status` — it should show your store path and episode count
3. If the path still looks like `~/.hermes/episodic_memory/<profile>`, the patch is likely missing

## Safety

The `patch_run_agent.py` script provided in this repo is idempotent. It checks if the patch is already applied before modifying the file.