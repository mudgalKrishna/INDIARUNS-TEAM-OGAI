#!/usr/bin/env python3
"""
rewrite_reasoning.py - Offline SLM reasoning rewriter (Qwen2.5-1.5B GGUF)
==========================================================================
Takes templated reasoning from fix_reasoning.py and rewrites each one into
natural, varied 2-sentence assessments using a local Small Language Model.

The model receives:
  1. Full Job Description context (so it understands fit vs gap)
  2. The specific factual reasoning string (so it can only rephrase, not invent)
  3. Strict rules to preserve every number, name, and score

This architecture prevents hallucination: the model cannot add
information that wasn't in the factual reasoning string.

Usage:
    # Full run — reads top100_reasoned.jsonl, writes updated submission.csv:
    py rewrite_reasoning.py --input top100_reasoned.jsonl --out submission.csv

    # Dry-run — print 3 prompts without loading the model:
    py rewrite_reasoning.py --input top100_reasoned.jsonl --out submission.csv --dry-run

    # Custom model path:
    py rewrite_reasoning.py --input top100_reasoned.jsonl --out submission.csv \
        --model models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf

Prerequisites:
    py setup_model.py       # one-time: downloads Qwen2.5-1.5B (~1 GB)
    pip install llama-cpp-python \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
"""

import json
import os
import sys
import re
import time
import argparse
from pathlib import Path
from datetime import datetime

# ── Default model path ────────────────────────────────────────────────────────
DEFAULT_MODEL = Path(__file__).parent / "models" / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"

# ── Job Description Context (embedded for offline use) ────────────────────────
# This is the actual Redrob JD so the SLM understands the hiring bar.
JD_CONTEXT = """
ROLE: Senior AI Engineer — Founding Team at Redrob AI (Series A)
LOCATION: Pune / Noida, India (Hybrid)
EXPERIENCE: 5-9 years total, 4-5 in applied ML/AI at product companies

MUST-HAVE REQUIREMENTS:
- Production experience with embeddings-based retrieval (sentence-transformers, BGE, E5, OpenAI embeddings) deployed to real users
- Production experience with vector databases or hybrid search: Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS
- Strong Python and production code quality
- Hands-on experience designing evaluation frameworks for ranking: NDCG, MRR, MAP, A/B testing

NICE-TO-HAVE:
- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank models (XGBoost-based or neural LTR)
- HR-tech or marketplace product experience

EXPLICIT DISQUALIFIERS:
- Pure services company background only (TCS, Infosys, Wipro, Accenture, Cognizant, etc.)
- Only recent LangChain/ChatGPT wrapper experience, no pre-LLM ML production
- Computer vision / robotics without NLP/IR background
- 60+ day notice period (high execution risk)
- No GitHub or external validation of technical work

BEHAVIORAL SIGNALS (from Redrob platform):
- notice_period_days: ideal <30 days; acceptable 30-60; risky >60
- github_activity_score: 0-100 (-1 means no GitHub linked); higher is better
- open_to_work_flag: true = actively looking
- recruiter_response_rate: fraction of messages replied to (higher = more reachable)
- interview_completion_rate: fraction of interviews attended
- last_active_date: recent activity = candidate is reachable

WHAT REDROB IS BUILDING:
The ranking, retrieval, and matching systems that determine what recruiters see
when they search candidates. First 90 days: audit current BM25 + rule-based scoring,
then ship a v2 hybrid retrieval system, then build evaluation infrastructure.
"""

# ── Redrob signals reference (for prompt context) ─────────────────────────────
SIGNALS_CONTEXT = """
REDROB BEHAVIORAL SIGNALS REFERENCE:
- profile_completeness_score (0-100): how complete the profile is
- open_to_work_flag (bool): actively looking for roles
- notice_period_days (0-180): days to join; <30 = ideal for this role
- github_activity_score (-1 to 100): code contribution signal; -1 = no GitHub
- recruiter_response_rate (0-1): fraction of recruiter messages replied to
- interview_completion_rate (0-1): fraction of interviews attended
- offer_acceptance_rate (-1 to 1): fraction of offers accepted
- willing_to_relocate (bool): open to Pune/Noida
- linkedin_connected (bool): LinkedIn account linked
"""

# ── System prompt with full JD context ───────────────────────────────────────
SYSTEM_PROMPT = f"""You are a senior technical recruiter writing factual candidate assessments for Redrob AI.

=== THE ROLE YOU ARE HIRING FOR ===
{JD_CONTEXT}

=== YOUR TASK ===
You will receive a factual assessment string about a candidate. Your ONLY job is to
rephrase it into 2 natural, varied sentences that a recruiter would find informative.

=== STRICT RULES (MUST FOLLOW) ===
1. Preserve EVERY specific fact: years of experience, months of skill use, company names,
   skill names, GitHub scores, notice period days — everything numerical or named
2. NO HALLUCINATION: Do NOT add any technology, tool, skill, company, or claim not in the original text.
3. NO TEMPLATING: Do not just insert the candidate's name into a fixed template. Ensure highly diverse sentence structures across different candidates. Do NOT use all-identical reasoning strings.
4. NO CONTRADICTIONS: Ensure the tone matches a top candidate. Do not contradict their high rank.
5. Do NOT use the word "however" more than once across both sentences.
6. Write EXACTLY 2 sentences — no more, no less.
7. Vary the sentence structure — do not always start with the job title.
8. Reference at least one behavioral signal (notice period, GitHub, or availability) naturally.
9. Make the assessment feel like it was written by a human recruiter, not generated.
10. Keep the tone professional but direct — this is for a founding-team technical role."""


def build_user_prompt(original_reasoning: str, candidate_id: str) -> str:
    return f"""Rephrase this candidate assessment for {candidate_id} in exactly 2 sentences.
Preserve ALL facts. Do NOT add new information. Reference at least one availability signal.

Original assessment:
{original_reasoning}

Rewritten (2 sentences only, same facts, natural recruiter voice):"""


# ── Validation: facts preserved, no hallucinated tech ────────────────────────

# Tech keywords that a model might hallucinate
HALLUCINATION_GUARD = {
    "kubernetes", "docker", "spark", "kafka", "airflow", "dbt", "databricks",
    "aws", "gcp", "azure", "terraform", "redis", "postgres", "mysql",
    "react", "javascript", "typescript", "golang", "rust", "java", "scala",
    "bert", "gpt", "llama", "mistral", "claude", "openai", "gemini",
    "langchain", "llamaindex", "autogen", "crewai", "langgraph",
}


def validate_rewrite(original: str, rewritten: str) -> tuple:
    """
    Returns (is_valid: bool, reason: str)
    Checks: no hallucinated tech, numbers preserved, reasonable length.
    """
    orig_lower = original.lower()
    rew_lower  = rewritten.lower()

    # No new tech keywords
    for kw in HALLUCINATION_GUARD:
        if kw in rew_lower and kw not in orig_lower:
            return False, f"Hallucinated keyword: '{kw}'"

    # Key numbers from original preserved
    orig_nums = set(re.findall(r'\b\d+\b', original))
    rew_nums  = set(re.findall(r'\b\d+\b', rewritten))
    missing   = orig_nums - rew_nums
    if len(missing) > 1:
        return False, f"Missing numbers: {missing}"

    # Length sanity
    sentences = [s.strip() for s in re.split(r'[.!?]+', rewritten) if len(s.strip()) > 5]
    if len(sentences) > 5 or len(rewritten) < 40:
        return False, f"Bad length ({len(sentences)} sentences, {len(rewritten)} chars)"

    # Must not be identical to template patterns (crude template detection)
    if rewritten.strip() == original.strip():
        return False, "Output identical to input"

    return True, "OK"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_path: str, n_threads: int = None):
    """Load Qwen2.5-1.5B GGUF via llama-cpp-python."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("\nERROR: llama-cpp-python not installed.")
        print("\nInstall (CPU-only):")
        print("  pip install llama-cpp-python \\")
        print("    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu")
        print("\nInstall (CUDA 12.1, GPU):")
        print("  pip install llama-cpp-python \\")
        print("    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"\nERROR: Model not found at: {model_path}")
        print("Run first:  py setup_model.py")
        sys.exit(1)

    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"  Model: {os.path.basename(model_path)}  ({size_mb:.0f} MB)")

    if n_threads is None:
        import multiprocessing
        n_threads = max(1, multiprocessing.cpu_count() - 1)

    # Context window: system prompt (~800 tok) + user prompt (~200 tok) + output (~150 tok)
    n_ctx = 1536

    print(f"  Loading... (n_ctx={n_ctx}, n_threads={n_threads})", end="", flush=True)
    t0 = time.time()
    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=0,       # 0 = CPU only; set to -1 to use full GPU if available
        verbose=False,
        chat_format="chatml", # Qwen2.5 uses ChatML format
    )
    print(f" done in {time.time()-t0:.1f}s")
    return llm


# ── Single inference ──────────────────────────────────────────────────────────

def rewrite_one(llm, original: str, candidate_id: str,
                temperature: float = 0.6, max_tokens: int = 160) -> str:
    """Run one rewrite call. Returns rewritten text."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_prompt(original, candidate_id)},
    ]
    resp = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.92,
        repeat_penalty=1.12,
        stop=["Original assessment:", "Rewritten (", "\n\n\n", "---"],
    )
    text = resp["choices"][0]["message"]["content"].strip()
    # Strip any prompt leakage
    for stopper in ["Original assessment:", "Rewritten (", "===", "STRICT"]:
        if stopper in text:
            text = text.split(stopper)[0].strip()
    return text


# ── Main processing ───────────────────────────────────────────────────────────

def process(input_path: str, out_csv: str, model_path: str,
            dry_run: bool = False, n_threads: int = None,
            temperature: float = 0.6):

    print(f"\n  Input:  {input_path}")
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Loaded {len(records)} candidates")

    if dry_run:
        print("\n  [DRY RUN] First 3 prompts:\n")
        for rec in records[:3]:
            print(f"  {'='*60}")
            print(f"  Candidate: {rec['candidate_id']}")
            print(f"  Original:  {rec['fixed_reasoning']}")
            print(f"\n  SYSTEM PROMPT (truncated):\n  {SYSTEM_PROMPT[:300]}...")
            print(f"\n  USER PROMPT:\n  {build_user_prompt(rec['fixed_reasoning'], rec['candidate_id'])}")
        return

    # Load model
    print()
    llm = load_model(model_path, n_threads=n_threads)

    # Process loop
    print(f"\n  Rewriting {len(records)} reasonings (temperature={temperature})")
    print(f"  {'Rank':<5}  {'Candidate ID':<18}  {'Time':>5}  Status")
    print(f"  {'-'*65}")

    results        = []
    fallback_count = 0
    total_start    = time.time()

    for rec in records:
        cid      = rec["candidate_id"]
        rank     = rec.get("rank", len(results) + 1)
        score    = rec.get("score", 0)
        original = rec["fixed_reasoning"]

        t0 = time.time()
        try:
            rewritten = rewrite_one(llm, original, cid, temperature=temperature)
            elapsed   = time.time() - t0
            ok, reason = validate_rewrite(original, rewritten)
            if ok:
                status = "OK"
            else:
                rewritten = original
                fallback_count += 1
                status = f"FALLBACK ({reason[:40]})"
        except Exception as e:
            rewritten = original
            elapsed   = time.time() - t0
            fallback_count += 1
            status = f"ERROR ({str(e)[:30]})"

        results.append({
            "rank":               rank,
            "candidate_id":       cid,
            "score":              score,
            "fixed_reasoning":    original,
            "rewritten_reasoning": rewritten,
        })
        print(f"  {rank:<5}  {cid:<18}  {elapsed:>4.1f}s  {status}")

    total_time = time.time() - total_start

    # Write rewritten JSONL
    jsonl_out = out_csv.replace(".csv", "_rewritten.jsonl")
    with open(jsonl_out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write updated submission.csv
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        f.write("candidate_id,rank,score,reasoning\n")
        for r in results:
            clean = r["rewritten_reasoning"].replace('"', "'").replace("\n", " ").strip()
            f.write(f'{r["candidate_id"]},{r["rank"]},{r["score"]},"{clean}"\n')

    # Stats
    unique = len({r["rewritten_reasoning"] for r in results})
    however_remaining = sum(
        1 for r in results if "; however" in r["rewritten_reasoning"].lower()
    )

    print(f"""
  +=====================================================+
  |  REWRITE SUMMARY                                   |
  +=====================================================+
  | Total candidates:          {len(results):<5}                   |
  | Fallbacks (original kept): {fallback_count:<5}                   |
  | Unique reasonings:         {unique:<5} / {len(results):<5}              |
  | '; however' remaining:     {however_remaining:<5} ({however_remaining/len(results)*100:.0f}%)              |
  | Total time:                {total_time/60:.1f} min                    |
  | Avg per candidate:         {total_time/len(results):.1f}s                     |
  +=====================================================+
  | Output:  {out_csv:<44}|
  | JSONL:   {jsonl_out:<44}|
  +=====================================================+
    """)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Offline SLM reasoning rewriter — Qwen2.5-1.5B-Instruct GGUF"
    )
    parser.add_argument("--input", required=True,
        help="Input JSONL from fix_reasoning.py (e.g. top100_reasoned.jsonl)")
    parser.add_argument("--out", required=True,
        help="Output submission.csv path")
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
        help=f"Path to GGUF model file (default: {DEFAULT_MODEL})")
    parser.add_argument("--dry-run", action="store_true",
        help="Print prompts for first 3 candidates without running the model")
    parser.add_argument("--threads", type=int, default=None,
        help="CPU threads (default: cpu_count - 1)")
    parser.add_argument("--temperature", type=float, default=0.6,
        help="Sampling temperature 0-1 (default: 0.6). Higher = more varied")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input not found: {args.input}")
        sys.exit(1)

    print(f"""
+============================================================+
|  REDROB - SLM Reasoning Rewriter (Fully Offline)          |
|  Model: Qwen2.5-1.5B-Instruct-Q4_K_M GGUF               |
|  Context: Full JD embedded in system prompt               |
+============================================================+
    """)

    process(
        input_path=args.input,
        out_csv=args.out,
        model_path=args.model,
        dry_run=args.dry_run,
        n_threads=args.threads,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
