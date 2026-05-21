"""
Download BGE embedding models for episodic-memory.

Usage:
    python scripts/download_model.py          # downloads bge-large-en-v1.5 (2.5GB)
    python scripts/download_model.py --small    # downloads bge-small-en-v1.5 (133MB)
"""

import os
import argparse
from huggingface_hub import snapshot_download, HfApi

# Model identifiers
LARGE_MODEL = "BAAI/bge-large-en-v1.5"
SMALL_MODEL = "BAAI/bge-small-en-v1.5"

# Cache directory (same as transformers)
HF_HOME = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
MODEL_CACHE_DIR = os.path.join(HF_HOME, "hub")


def verify_model_cached(model_id: str) -> bool:
    """
    Check if a model is already downloaded in the HF cache.
    """
    # snapshot_download uses a deterministic cache key based on model_id
    # The actual path is models--{author}--{model} in HF_HOME/hub
    org, model = model_id.split("/")
    expected_dir = f"models--{org}--{model}"
    cache_path = os.path.join(MODEL_CACHE_DIR, expected_dir)
    return os.path.isdir(cache_path)

def main():
    parser = argparse.ArgumentParser(description="Download BGE embedding model for episodic-memory")
    parser.add_argument(
        "--small",
        action="store_true",
        help="Download bge-small-en-v1.5 instead of bge-large"
    )
    args = parser.parse_args()

    model_id = SMALL_MODEL if args.small else LARGE_MODEL
    model_size = "small" if args.small else "large"
    expected_size = "133MB" if args.small else "2.5GB"

    print(f"Downloading {model_size} BGE model ({expected_size}): {model_id}")

    # Check if already cached
    if verify_model_cached(model_id):
        print(f"Model already downloaded: {model_id}")
        print(f"Cache location: {MODEL_CACHE_DIR}")
        return 0

    # Download model
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir_use_symlinks=False,  # Copy files
            tqdm_class=None,  # No progress bar (clean logs)
        )  # type: ignore
        print(f"Model downloaded successfully: {model_id}")
        print(f"Cache location: {MODEL_CACHE_DIR}")
        return 0
    except Exception as e:
        print(f"Error downloading model {model_id}: {e}")
        print(f"You may need to run 'pip install huggingface-hub' first.")
        return 1

if __name__ == "__main__":
    exit(main())
