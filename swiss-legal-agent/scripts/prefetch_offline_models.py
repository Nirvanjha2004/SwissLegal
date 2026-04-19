from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_MODELS = [
    # Multilingual embedding model for legal retrieval experiments.
    "intfloat/multilingual-e5-small",
    # Lightweight reranker baseline.
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
]

DEFAULT_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.safetensors",
    "*.bin",
    "*.py",
]


def prefetch_model(model_id: str, destination_root: Path) -> Path:
    safe_name = model_id.replace("/", "--")
    target_dir = destination_root / safe_name
    target_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        allow_patterns=DEFAULT_ALLOW_PATTERNS,
    )
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face models for offline Kaggle inference.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models") / "hf",
        help="Destination directory for downloaded models.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help="Model id to prefetch. Repeat flag for multiple models.",
    )

    args = parser.parse_args()
    models = args.models if args.models else DEFAULT_MODELS

    args.models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Prefetching {len(models)} model(s) into {args.models_dir}...")
    for model_id in models:
        downloaded_path = prefetch_model(model_id, args.models_dir)
        print(f"Downloaded: {model_id} -> {downloaded_path}")

    print("Done. You can now run inference with internet disabled.")


if __name__ == "__main__":
    main()
