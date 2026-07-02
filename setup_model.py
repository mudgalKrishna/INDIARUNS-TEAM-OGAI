#!/usr/bin/env python3
"""
setup_model.py - One-time model download for offline SLM reasoning rewriter
============================================================================
Run this ONCE to download the model weights locally.
After this, rewrite_reasoning.py works fully offline forever.

Usage:
    py setup_model.py

What this downloads:
    Model : Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
    Size  : ~986 MB (~1 GB)
    Source: Hugging Face (bartowski/Qwen2.5-1.5B-Instruct-GGUF)
    Dest  : ./models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf

Why Qwen2.5-1.5B:
    - Best 1 GB model for instruction-following + paraphrasing
    - Handles Indian company/tech jargon well
    - Apache 2.0 license (commercial use OK)
    - Runs on CPU, no GPU required

After download, install the runtime if not done:
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

Then run the rewriter:
    py rewrite_reasoning.py --input top100_reasoned.jsonl --out submission.csv
"""

import os
import sys
import urllib.request
import urllib.error
import shutil
from pathlib import Path

# -- Model config -------------------------------------------------------------

MODEL_REPO  = "bartowski/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILE  = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
MODEL_URL   = (
    "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF"
    "/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
)
EXPECTED_MB = 986
MODELS_DIR  = Path(__file__).parent / "models"
MODEL_PATH  = MODELS_DIR / MODEL_FILE


# -- Progress bar -------------------------------------------------------------

class ProgressBar:
    def __init__(self, total_mb):
        self.total    = total_mb * 1024 * 1024
        self.done     = 0
        self.last_pct = -1

    def __call__(self, block_num, block_size, total_size):
        if total_size > 0:
            self.total = total_size
        self.done = min(self.done + block_size, self.total)
        pct = int(self.done * 100 / self.total)
        if pct != self.last_pct and pct % 2 == 0:
            filled = pct // 5
            bar = "#" * filled + "." * (20 - filled)
            mb_done  = self.done  / 1024 / 1024
            mb_total = self.total / 1024 / 1024
            print(f"\r  [{bar}] {pct:3d}%  {mb_done:.0f}/{mb_total:.0f} MB",
                  end="", flush=True)
            self.last_pct = pct


# -- Download helpers ---------------------------------------------------------

def try_huggingface_hub():
    """Try huggingface_hub if installed (resumable downloads)."""
    try:
        from huggingface_hub import hf_hub_download
        print("  Using huggingface_hub (resumable)...")
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"  huggingface_hub failed: {e}. Falling back to urllib...")
        return False


def download_urllib():
    print(f"  From: {MODEL_URL}")
    print(f"  To:   {MODEL_PATH}")
    print(f"  Size: ~{EXPECTED_MB} MB\n")
    bar = ProgressBar(EXPECTED_MB)
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=bar)
        print()
    except urllib.error.URLError as e:
        print(f"\n  ERROR: Download failed: {e}")
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        sys.exit(1)


# -- Main ---------------------------------------------------------------------

def main():
    print("""
+===================================================+
|  REDROB SLM SETUP - One-time model download      |
|  Qwen2.5-1.5B-Instruct-Q4_K_M.gguf (~1 GB)     |
+===================================================+
    """)

    MODELS_DIR.mkdir(exist_ok=True)

    # Already downloaded?
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
        if size_mb > 900:
            print(f"  [OK] Model already at: {MODEL_PATH}")
            print(f"       Size: {size_mb:.0f} MB")
            _check_llama_cpp()
            return
        else:
            print(f"  Partial download ({size_mb:.0f} MB). Re-downloading...")
            MODEL_PATH.unlink()

    # Disk space check
    free_gb = shutil.disk_usage(MODELS_DIR).free / 1024 ** 3
    if free_gb < 1.2:
        print(f"  ERROR: Need 1.2 GB free, only {free_gb:.1f} GB available.")
        sys.exit(1)
    print(f"  Free disk: {free_gb:.1f} GB  [OK]")
    print(f"  Downloading {MODEL_FILE} ...")
    print()

    downloaded = try_huggingface_hub()
    if not downloaded:
        download_urllib()

    # Verify size
    size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
    if size_mb < 900:
        print(f"  ERROR: File too small ({size_mb:.0f} MB) — may be corrupted.")
        sys.exit(1)

    print(f"\n  [OK] Model ready: {MODEL_PATH}  ({size_mb:.0f} MB)")
    _check_llama_cpp()

    print("\n  NEXT STEP:")
    print("  py rewrite_reasoning.py --input top100_reasoned.jsonl --out submission.csv\n")


def _check_llama_cpp():
    print()
    try:
        import llama_cpp
        print(f"  [OK] llama-cpp-python: {llama_cpp.__version__}")
    except ImportError:
        print("  [!] llama-cpp-python not installed. Run ONE of:")
        print()
        print("  # CPU only (recommended if no NVIDIA GPU):")
        print("  pip install llama-cpp-python \\")
        print("    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu")
        print()
        print("  # CUDA 12.1 (if you have an NVIDIA GPU):")
        print("  pip install llama-cpp-python \\")
        print("    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")


if __name__ == "__main__":
    main()
