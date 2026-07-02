# Redrob Intelligent Candidate Ranking Pipeline

> **Hackathon:** India Runs Data & AI Challenge — Candidate Ranking Track  
> **Task:** Rank top 100 candidates from a 100,000-candidate pool for a senior AI/ML retrieval engineer role.  
> **Approach:** 5-stage offline CPU-only pipeline using a LightGBM regressor trained via teacher distillation + deterministic reasoning + optional SLM rewrite.

---

## Quick Start (Single Command)

```bash
python run_pipeline.py --candidates candidates.jsonl --ranker regressor_no_prescore.pkl
```

This produces `submission.csv` with the top 100 ranked candidates. **Runtime: ~31 seconds on CPU.**

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation — Windows](#installation--windows)
3. [Installation — macOS](#installation--macos)
4. [Installation — Linux](#installation--linux)
5. [One-time Model Download](#one-time-model-download)
6. [Running the Pipeline](#running-the-pipeline)
7. [Validating Your Submission](#validating-your-submission)
8. [Pipeline Flags Reference](#pipeline-flags-reference)
9. [File Structure](#file-structure)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | >= 3.10 |
| RAM | >= 8 GB (16 GB recommended) |
| Disk Space | >= 3 GB free |
| CPU | Any modern x86-64 CPU with **AVX2** support (for SLM stage) |
| Internet | Required once for model download only |

---

## Installation — Windows

### Step 1: Verify Python Version
```powershell
python --version
# Should show Python 3.10 or higher
```

If Python is not installed, download it from [python.org](https://python.org) or via Windows Store.

### Step 2: Create a Virtual Environment (Recommended)
```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Core Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Install llama-cpp-python (SLM Engine)

> **CPU-only (works on any machine):**
```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

> **With NVIDIA GPU (optional, ~10x faster for SLM stage):**
```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

### Step 5: Download the SLM Model (One-Time)
```powershell
python setup_model.py
```
Downloads `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` (~986 MB) into the `models/` folder. Resumable if interrupted.

---

## Installation — macOS

### Step 1: Install Python via Homebrew
```bash
brew install python@3.11
python3 --version
```

### Step 2: Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Core Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install llama-cpp-python

> **Apple Silicon (M1/M2/M3) — Metal GPU acceleration:**
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

> **Intel Mac — CPU only:**
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### Step 5: Download the SLM Model (One-Time)
```bash
python setup_model.py
```

---

## Installation — Linux

### Step 1: Install Python
```bash
# Ubuntu / Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip -y

# Fedora / RHEL
sudo dnf install python3.11 -y

# Verify
python3 --version
```

### Step 2: Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Build Tools (required for llama-cpp-python compilation)
```bash
# Ubuntu / Debian
sudo apt install build-essential cmake -y

# Fedora / RHEL
sudo dnf groupinstall "Development Tools" -y
sudo dnf install cmake -y
```

### Step 4: Install Core Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Install llama-cpp-python

> **CPU-only build:**
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

> **CUDA 12.1 (NVIDIA GPU):**
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

### Step 6: Download the SLM Model (One-Time)
```bash
python setup_model.py
```

---

## One-time Model Download

The SLM model (`Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`) is approximately **986 MB** and needs to be downloaded once before running the full pipeline.

```bash
python setup_model.py
```

**What it does:**
- Checks for 2 GB free disk space
- Downloads the GGUF quantized model from HuggingFace
- Saves it to `models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`
- Download is resumable if interrupted

> **Note:** If you prefer to skip the SLM rewrite stage (e.g., for testing), you can use `--no-rewrite` flag and skip this step entirely.

---

## Running the Pipeline

### Full Pipeline (All Stages including SLM rewrite)
```bash
python run_pipeline.py \
  --candidates candidates.jsonl \
  --ranker regressor_no_prescore.pkl
```

### Skip SLM Rewrite (Faster, uses deterministic reasoning)
```bash
python run_pipeline.py \
  --candidates candidates.jsonl \
  --ranker regressor_no_prescore.pkl \
  --no-rewrite
```

### Resume from a Specific Stage (if pipeline was interrupted)
```bash
# Resume from Stage 2 (LightGBM) onwards
python run_pipeline.py \
  --candidates candidates.jsonl \
  --ranker regressor_no_prescore.pkl \
  --start-stage 2

# Resume from Stage 4 (SLM Rewrite) onwards
python run_pipeline.py \
  --candidates candidates.jsonl \
  --ranker regressor_no_prescore.pkl \
  --start-stage 4
```

### Expected Runtime
| Stage | Operation | Time |
|---|---|---|
| Stage 1 | Hard filter + prescore (100k → 22k) | ~17 seconds |
| Stage 2 | LightGBM regression (22k → top 100) | ~4.6 seconds |
| Stage 3 | Deterministic reasoning (top 100) | < 0.1 seconds |
| Stage 4 | SLM rewrite — Qwen2.5-1.5B (top 100) | ~2–4 minutes (optional) |
| Stage 5 | Write submission.csv | < 0.1 seconds |
| **Total (without SLM)** | | **~31 seconds** |
| **Total (with SLM)** | | **~3–5 minutes** |

---

## Validating Your Submission

After the pipeline runs, validate your output against the official submission spec:

```bash
python validate_submission.py submission.csv
```

Expected output:
```
Submission is valid.
```

---

## Pipeline Flags Reference

| Flag | Default | Description |
|---|---|---|
| `--candidates` | required | Path to `candidates.jsonl` |
| `--ranker` | `regressor_no_prescore.pkl` | Path to LightGBM `.pkl` model |
| `--filter-top-k` | `22000` | Number of candidates after Stage 1 |
| `--lgbm-top-k` | `15000` | Number of candidates fed to LightGBM |
| `--top-final` | `100` | Final candidates in submission |
| `--no-rewrite` | False | Skip SLM rewrite (Stage 4) |
| `--start-stage` | `1` | Resume pipeline from a specific stage |
| `--threads` | `auto` | CPU threads for LLM inference |
| `--temperature` | `0.35` | SLM sampling temperature |

---

## File Structure

```
github_repo/
├── run_pipeline.py                    # Main orchestrator — run this
├── filter_candidates.py               # Stage 1: Hard filter + feature extraction
├── fix_reasoning.py                   # Stage 3: Deterministic reasoning builder
├── rewrite_reasoning.py               # Stage 4: Qwen2.5 SLM rewrite engine
├── score_candidates.py                # Stage 3.5: Composite scoring
├── setup_model.py                     # One-time model downloader
├── validate_submission.py             # Official submission validator
├── train_lightgbm_regressor.py        # (Offline) Training script
├── regressor_no_prescore.pkl          # Pre-trained LightGBM model weights
├── regressor_no_prescore_features.json# Feature names for the model
├── requirements.txt                   # Python dependencies
├── models/
│   └── Qwen2.5-1.5B-Instruct-Q4_K_M.gguf  # SLM weights (download via setup_model.py)
├── submission.csv                     # Final output (generated by pipeline)
├── filtered_22k.jsonl                 # Stage 1 output (cached)
├── lgbm_top15k.jsonl                  # Stage 2 output (cached)
└── top100_reasoned.jsonl              # Stage 3 output (cached)
```

---

## Troubleshooting

### `OSError: [WinError -1073741795] Windows Error 0xc000001d`
Your CPU does not support AVX2 instructions required by `llama-cpp-python`. Use the `--no-rewrite` flag to skip Stage 4:
```bash
python run_pipeline.py --candidates candidates.jsonl --ranker regressor_no_prescore.pkl --no-rewrite
```

### `ModuleNotFoundError: No module named 'lightgbm'`
Install dependencies:
```bash
pip install -r requirements.txt
```

### `Model not found` error during Stage 4
Run the one-time download:
```bash
python setup_model.py
```

### Download interrupted / incomplete GGUF file
`setup_model.py` uses `huggingface_hub` which supports **resumable downloads**. Simply run `python setup_model.py` again — it will continue from where it left off.

### `PermissionError` on Windows
Run your terminal as Administrator, or ensure no other process has the output files open (e.g., close Excel if `submission.csv` is open).

### Stage 2 LightGBM model mismatch
Ensure you are using the exact `regressor_no_prescore.pkl` provided in the repo. Do not mix pkl files from different training runs — the feature list must exactly match `regressor_no_prescore_features.json`.

---

## Reproduction Command (for Stage 3 Code Review)

For the hackathon Stage 3 code reproduction check, the exact command to produce `submission.csv` from scratch is:

```bash
pip install -r requirements.txt
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python setup_model.py  # one-time model download
python run_pipeline.py --candidates candidates.jsonl --ranker regressor_no_prescore.pkl
```

---

*Pipeline runtime: ~31 seconds (CPU-only, without SLM) | ~4 minutes (with SLM rewrite)*  
*Peak RAM: ~1.65 GB | Disk footprint: ~1.54 GB*
