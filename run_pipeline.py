#!/usr/bin/env python3
"""
run_pipeline.py - Redrob Intelligent Candidate Ranking Pipeline
===============================================================
MASTER ORCHESTRATOR — runs the full 5-stage pipeline and reports
wall-clock time, peak RAM, GPU VRAM, and disk usage per stage.

PIPELINE:
  Stage 1 | filter_candidates.py  | 100k → ~22k  (hard rules + honeypot + prescore)
  Stage 2 | LightGBM inference    | 22k  → 15k   (pre-trained ranker.pkl)
  Stage 3 | fix_reasoning.py      | top 100       (deterministic fact-grounded reasoning)
  Stage 4 | rewrite_reasoning.py  | top 100       (SLM paraphrase, JD-aware, offline)
  Stage 5 | submission.csv        | final output

USAGE:
  # Full pipeline from scratch:
  py run_pipeline.py --candidates candidates.jsonl --ranker ranker.pkl

  # Skip to stage 2 (filter already done):
  py run_pipeline.py --candidates candidates.jsonl --ranker ranker.pkl --start-stage 2

  # Skip SLM rewrite (deterministic reasoning only):
  py run_pipeline.py --candidates candidates.jsonl --ranker ranker.pkl --no-rewrite

  # Check disk/file status without running:
  py run_pipeline.py --candidates candidates.jsonl --ranker ranker.pkl --status

PREREQUISITES:
  pip install -r requirements.txt
  py setup_model.py                  # downloads Qwen2.5-1.5B GGUF (~1 GB)
  pip install llama-cpp-python \\
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
"""

import json
import os
import sys
import time
import pickle
import shutil
import threading
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ── Resource monitoring ───────────────────────────────────────────────────────

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    _PROC = None


def _gpu_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=3
        ).decode().strip().splitlines()[0]
        used, total = map(int, out.split(","))
        return used, total
    except Exception:
        return None, None


class ResourceMonitor:
    def __init__(self):
        self.peak_ram_mb = 0.0
        self.peak_gpu_mb = 0.0
        self._running    = False
        self._thread     = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self):
        while self._running:
            if HAS_PSUTIL and _PROC:
                try:
                    ram = _PROC.memory_info().rss / 1024 / 1024
                    self.peak_ram_mb = max(self.peak_ram_mb, ram)
                except Exception:
                    pass
            gpu_used, _ = _gpu_mb()
            if gpu_used:
                self.peak_gpu_mb = max(self.peak_gpu_mb, float(gpu_used))
            time.sleep(0.5)

    def snapshot(self):
        ram = 0.0
        if HAS_PSUTIL and _PROC:
            try:
                ram = _PROC.memory_info().rss / 1024 / 1024
            except Exception:
                pass
        gpu_u, gpu_t = _gpu_mb()
        return {
            "current_ram_mb": round(ram, 1),
            "peak_ram_mb":    round(self.peak_ram_mb, 1),
            "gpu_used_mb":    gpu_u,
            "gpu_total_mb":   gpu_t,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def ts():
    return datetime.now().strftime("%H:%M:%S")

def log(msg, level="OK"):
    icons = {"OK": "[OK] ", "WARN": "[WRN]", "ERR": "[ERR]", "INFO": "[-->]"}
    print(f"  {icons.get(level,'     ')} [{ts()}] {msg}", flush=True)

def sep(title):
    print(f"\n{'='*68}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*68}", flush=True)

def fmttime(s):
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s)//60}m {int(s)%60}s"

def fmtmb(path):
    if path and os.path.exists(path):
        return round(os.path.getsize(path) / 1024 / 1024, 2)
    return 0.0

def check_disk(path, needed_mb):
    free_mb = shutil.disk_usage(os.path.dirname(os.path.abspath(path))).free / 1024 / 1024
    if free_mb < needed_mb:
        log(f"Low disk space! Need ~{needed_mb:.0f} MB, have {free_mb:.0f} MB", "WARN")
    else:
        log(f"Disk free: {free_mb:.0f} MB  (need ~{needed_mb:.0f} MB) [OK]")


# ── Stage 1: Hard filter ──────────────────────────────────────────────────────

def stage1_filter(candidates_path, out_path, top_k, report, monitor):
    sep(f"STAGE 1  |  HARD FILTER + PRESCORE  (100k -> ~{top_k//1000}k)")

    # Disk estimate: filtered output ~ 60% of input
    in_mb = fmtmb(candidates_path)
    check_disk(out_path, in_mb * 0.6 + 50)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        import filter_candidates as fc
    except ImportError:
        log("filter_candidates.py not found in same directory", "ERR")
        sys.exit(1)

    t0 = time.time()
    n_out = fc.run_filter(
        candidates_path=candidates_path,
        top_k=top_k,
        out_path=out_path,
        report=report,
        report_path=out_path.replace(".jsonl", "_report.txt"),
    )
    elapsed = time.time() - t0
    snap = monitor.snapshot()
    log(f"Done in {fmttime(elapsed)} | {n_out:,} candidates | RAM: {snap['peak_ram_mb']:.0f} MB | Disk: {fmtmb(out_path):.1f} MB")
    return {"stage": "1_filter", "n_in": "100k", "n_out": n_out,
            "time_s": round(elapsed, 1), **snap, "disk_mb": fmtmb(out_path)}


# ── Stage 2: LightGBM inference ───────────────────────────────────────────────

def stage2_lgbm(filtered_path, ranker_pkl, out_path, top_k, monitor):
    sep(f"STAGE 2  |  LIGHTGBM INFERENCE  (~22k -> top {top_k//1000}k)")
    check_disk(out_path, fmtmb(filtered_path) * 0.7)

    try:
        import numpy as np
    except ImportError:
        log("numpy not installed: pip install numpy", "ERR"); sys.exit(1)

    log(f"Loading ranker: {ranker_pkl}")
    with open(ranker_pkl, "rb") as f:
        raw = pickle.load(f)

    best_iter = -1
    if isinstance(raw, dict):
        booster      = raw["booster"]
        best_iter    = raw.get("best_iteration", -1)
        feature_names = raw.get("feat_keys", [])
        log(f"Ranker dict: {len(feature_names)} features, best_iter={best_iter}")
        if feature_names:
            # Print ranker training metrics if available
            metrics = raw.get("metrics", {})
            if metrics:
                m_str = "  ".join(f"{k}={v:.4f}" for k, v in list(metrics.items())[:4])
                log(f"Training metrics: {m_str}")
    else:
        booster = raw
        feature_names = []

    # If feat_keys not in pickle, load from JSON
    if not feature_names:
        ranker_dir  = os.path.dirname(os.path.abspath(ranker_pkl))
        for fname in ["ranker_features.json", "ranker_features (1).json"]:
            fp = os.path.join(ranker_dir, fname)
            if os.path.exists(fp):
                with open(fp, "r", encoding="utf-8") as f:
                    feature_names = json.load(f)
                log(f"Loaded feature names from {fp}")
                break
    if not feature_names:
        log("Cannot find feature names (ranker_features.json). Pass --ranker-features.", "ERR")
        sys.exit(1)

    t0 = time.time()
    log(f"Loading {filtered_path}")
    candidates = []
    with open(filtered_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    log(f"Loaded {len(candidates):,} candidates")

    X = np.array([
        [float((c.get("features") or {}).get(fn, 0.0) or 0.0) for fn in feature_names]
        for c in candidates
    ], dtype=np.float32)
    log(f"Feature matrix: {X.shape}")

    t_inf = time.time()
    pred_kw = {"num_iteration": best_iter} if best_iter > 0 else {}
    scores  = booster.predict(X, **pred_kw)
    log(f"Inference in {time.time()-t_inf:.3f}s  range [{scores.min():.4f} - {scores.max():.4f}]")

    top_idx = scores.argsort()[::-1][:top_k]
    with open(out_path, "w", encoding="utf-8") as f:
        for rank, idx in enumerate(top_idx, 1):
            c = candidates[idx]
            c["lgbm_score"] = float(scores[idx])
            c["lgbm_rank"]  = rank
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    snap    = monitor.snapshot()
    top_s   = scores[top_idx]
    log(f"Done in {fmttime(elapsed)} | top-1={top_s[0]:.4f} top-100={top_s[min(99,len(top_s)-1)]:.4f} | RAM: {snap['peak_ram_mb']:.0f} MB | Disk: {fmtmb(out_path):.1f} MB")
    return {"stage": "2_lgbm", "n_in": len(candidates), "n_out": len(top_idx),
            "time_s": round(elapsed, 1), **snap, "disk_mb": fmtmb(out_path)}


# ── Stage 3: Deterministic reasoning ─────────────────────────────────────────

def stage3_fix_reasoning(lgbm_path, out_path, top_k, monitor):
    sep(f"STAGE 3  |  DETERMINISTIC REASONING  (top {top_k})")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        import fix_reasoning as fr
    except ImportError:
        log("fix_reasoning.py not found in same directory", "ERR"); sys.exit(1)

    t0 = time.time()
    records = []
    with open(lgbm_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= top_k:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log(f"Loaded {len(records):,} candidates for reasoning")

    stats = fr.process_records(records, out_path, top_k=top_k)
    elapsed = time.time() - t0
    snap    = monitor.snapshot()
    log(f"Done in {fmttime(elapsed)} | unique={stats['unique_reasonings']}/{stats['total']} | hallucinations_replaced={stats['hallucinations_replaced']}")
    return {"stage": "3_reasoning", "n_in": len(records), "n_out": stats["total"],
            "hallucinations_replaced": stats["hallucinations_replaced"],
            "unique_reasonings":       stats["unique_reasonings"],
            "time_s": round(elapsed, 1), **snap, "disk_mb": fmtmb(out_path)}


# ── Stage 3.5: JD-aligned composite scorer (0-100) ───────────────────────────

def stage35_composite_score(lgbm_path, reasoned_path, submission_path, top_k, monitor):
    sep(f"STAGE 3.5  |  COMPOSITE SCORING  (JD-aligned, 0-100)")

    # Check if a Regressor was used (scores are already 0-100)
    with open(lgbm_path, "r", encoding="utf-8") as f:
        first = json.loads(f.readline())
        if first.get("lgbm_score", 0) > 10.0:
            log("Stage 3.5 skipped: LightGBM output is already 0-100 (Regressor used).")
            return None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        import score_candidates as sc
    except ImportError:
        log("score_candidates.py not found — skipping composite scoring", "WARN")
        return None

    t0 = time.time()
    final_scores = sc.run(
        input_path=lgbm_path,
        top_k=top_k,
        reasoned_path=reasoned_path,
        submission_path=submission_path,
    )
    elapsed = time.time() - t0
    snap = monitor.snapshot()
    lo, hi = min(final_scores), max(final_scores)
    log(f"Done in {fmttime(elapsed)} | score range {lo:.1f}-{hi:.1f} (spread {hi-lo:.1f} pts)")
    return {"stage": "3.5_scoring", "n_out": len(final_scores),
            "time_s": round(elapsed, 1), **snap, "disk_mb": fmtmb(reasoned_path)}


# ── Stage 4: SLM rewrite ──────────────────────────────────────────────────────

def stage4_slm_rewrite(reasoned_path, out_path, model_path, temperature, threads, monitor):
    sep(f"STAGE 4  |  SLM REWRITE  (Qwen2.5-1.5B, JD-aware, offline)")

    if not os.path.exists(model_path):
        log(f"Model not found: {model_path}", "WARN")
        log("Run: py setup_model.py  to download (~1 GB)", "WARN")
        log("Skipping SLM rewrite. Using deterministic reasoning for submission.", "WARN")
        return None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        import rewrite_reasoning as rr
    except ImportError:
        log("rewrite_reasoning.py not found", "ERR"); sys.exit(1)

    t0 = time.time()
    rr.process(
        input_path=reasoned_path,
        out_csv=out_path,
        model_path=model_path,
        dry_run=False,
        n_threads=threads,
        temperature=temperature,
    )
    elapsed = time.time() - t0
    snap    = monitor.snapshot()
    log(f"Done in {fmttime(elapsed)} | RAM: {snap['peak_ram_mb']:.0f} MB | Disk: {fmtmb(out_path):.1f} MB")
    return {"stage": "4_slm_rewrite", "time_s": round(elapsed, 1), **snap, "disk_mb": fmtmb(out_path)}


# ── Stage 5: Submission CSV (deterministic fallback) ─────────────────────────

def stage5_submission(reasoned_path, out_path, top_k, monitor):
    """Write submission.csv from deterministic reasoning (no SLM)."""
    sep(f"STAGE 5  |  SUBMISSION CSV  (top {top_k})")

    records = []
    with open(reasoned_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: (-r.get("score", 0), r.get("candidate_id", "")))
    records = records[:top_k]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("candidate_id,rank,score,reasoning\n")
        for rank, r in enumerate(records, 1):
            rsn = r.get("fixed_reasoning", "").replace('"', "'").replace("\n", " ").strip()
            f.write(f'{r["candidate_id"]},{rank},{r["score"]},"{rsn}"\n')

    # Top-10 preview
    print(f"\n  -- TOP-10 --")
    print(f"  {'Rank':<5}  {'Candidate ID':<18}  {'Score':>7}  Reasoning preview")
    print(f"  {'-'*75}")
    for rank, r in enumerate(records[:10], 1):
        rsn = r.get("fixed_reasoning", "")[:65]
        print(f"  {rank:<5}  {r['candidate_id']:<18}  {r['score']:>7.4f}  {rsn}...")

    snap = monitor.snapshot()
    return {"stage": "5_submission", "n_out": len(records), **snap, "disk_mb": fmtmb(out_path)}


# ── Resource report ───────────────────────────────────────────────────────────

def print_report(stage_stats, total_start, paths):
    total = time.time() - total_start
    W = 68

    def box(title):
        print(f"\n+{'='*W}+")
        print(f"| {title:<{W-1}}|")
        print(f"+{'-'*W}+")

    def row(label, val):
        print(f"| {label:<35} {val:<{W-36}}|")

    def divider():
        print(f"+{'-'*W}+")

    box("RESOURCE & TIMING REPORT")
    row("Stage", f"{'Time':>8}  {'In':>8}  {'Out':>8}  {'Peak RAM':>10}")
    divider()

    stage_names = {
        "1_filter":    "Stage 1: Hard filter + prescore",
        "2_lgbm":      "Stage 2: LightGBM inference",
        "3_reasoning": "Stage 3: Deterministic reasoning",
        "4_slm_rewrite": "Stage 4: SLM rewrite (Qwen2.5)",
        "5_submission":  "Stage 5: Submission CSV",
    }
    for s in stage_stats:
        if s is None:
            continue
        nm   = stage_names.get(s["stage"], s["stage"])
        t    = fmttime(s.get("time_s", 0))
        n_in = f"{s['n_in']:,}" if isinstance(s.get("n_in"), int) else str(s.get("n_in", ""))
        n_out = f"{s['n_out']:,}" if isinstance(s.get("n_out"), int) else str(s.get("n_out", ""))
        ram  = f"{s.get('peak_ram_mb', 0):.0f} MB"
        row(nm, f"{t:>8}  {n_in:>8}  {n_out:>8}  {ram:>10}")
    divider()
    row("TOTAL PIPELINE TIME", fmttime(total))

    box("PEAK RAM USAGE")
    peak = max((s.get("peak_ram_mb", 0) for s in stage_stats if s), default=0)
    row(f"Peak RAM (this process)", f"{peak:.0f} MB  ({peak/1024:.2f} GB)")

    box("GPU USAGE")
    gpu_u = next((s.get("gpu_used_mb") for s in reversed(stage_stats) if s and s.get("gpu_used_mb")), None)
    gpu_t = next((s.get("gpu_total_mb") for s in reversed(stage_stats) if s and s.get("gpu_total_mb")), None)
    if gpu_u is not None:
        row("GPU VRAM used", f"{gpu_u} MB / {gpu_t} MB")
    else:
        row("GPU", "Not used (CPU-only pipeline)")

    box("DISK USAGE")
    labels = [
        ("candidates.jsonl (input)",       paths.get("candidates")),
        ("filtered_22k.jsonl",             paths.get("filtered")),
        ("lgbm_top15k.jsonl",              paths.get("lgbm_out")),
        ("top100_reasoned.jsonl",          paths.get("reasoned_out")),
        ("submission.csv (final)",         paths.get("submission")),
        ("SLM model (Qwen2.5-1.5B GGUF)", paths.get("model")),
    ]
    total_disk = 0
    for label, path in labels:
        mb = fmtmb(path)
        total_disk += mb
        status = "[OK]  " if (path and os.path.exists(path)) else "[n/a] "
        row(f"{status}{label}", f"{mb:>8.2f} MB")
    divider()
    row("TOTAL DISK (all files)", f"{total_disk:>8.2f} MB  ({total_disk/1024:.2f} GB)")
    print(f"+{'='*W}+\n")


# ── Status check ──────────────────────────────────────────────────────────────

def status_check(paths):
    print("\n  -- PIPELINE FILE STATUS --")
    for label, path in paths.items():
        if not path:
            continue
        if os.path.exists(path):
            mb = fmtmb(path)
            print(f"  [EXISTS]  {label:<30}  {path}  ({mb:.1f} MB)")
        else:
            print(f"  [MISSING] {label:<30}  {path}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Redrob end-to-end ranking pipeline (100k -> submission.csv)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = parser.add_argument_group("Input files")
    g.add_argument("--candidates", required=True, help="Raw candidates.jsonl (100k pool)")
    g.add_argument("--ranker",     required=True, help="Pre-trained ranker.pkl (LightGBM)")
    g.add_argument("--ranker-features", default=None,
        help="ranker_features.json (auto-detected beside ranker.pkl if omitted)")

    g = parser.add_argument_group("Stage control")
    g.add_argument("--start-stage", type=int, default=1, choices=[1,2,3,4,5],
        help="Start from stage N (1=filter, 2=lgbm, 3=reasoning, 4=slm, 5=csv)")
    g.add_argument("--no-rewrite", action="store_true",
        help="Skip Stage 4 SLM rewrite, go straight to submission.csv from deterministic reasoning")

    g = parser.add_argument_group("Intermediate paths")
    g.add_argument("--filtered",    default="filtered_22k.jsonl")
    g.add_argument("--lgbm-out",    default="lgbm_top15k.jsonl")
    g.add_argument("--reasoned-out",default="top100_reasoned.jsonl")
    g.add_argument("--submission",  default="submission.csv")
    g.add_argument("--model",
        default=str(Path(__file__).parent / "models" / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"),
        help="Path to Qwen2.5 GGUF model (default: models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf)")

    g = parser.add_argument_group("Top-K params")
    g.add_argument("--filter-top-k",  type=int, default=22000)
    g.add_argument("--lgbm-top-k",    type=int, default=15000)
    g.add_argument("--top-final",     type=int, default=100)

    g = parser.add_argument_group("SLM settings")
    g.add_argument("--temperature", type=float, default=0.6)
    g.add_argument("--threads",     type=int,   default=None)

    g = parser.add_argument_group("Misc")
    g.add_argument("--report",    action="store_true", help="Write Stage 1 filter report")
    g.add_argument("--status",    action="store_true", help="Show file status and exit")

    args = parser.parse_args()

    # Auto-resolve ranker features
    if not args.ranker_features:
        rd = os.path.dirname(os.path.abspath(args.ranker))
        for n in ["ranker_features.json", "ranker_features (1).json"]:
            p = os.path.join(rd, n)
            if os.path.exists(p):
                args.ranker_features = p
                break

    paths = {
        "candidates":   args.candidates,
        "filtered":     args.filtered,
        "lgbm_out":     args.lgbm_out,
        "reasoned_out": args.reasoned_out,
        "submission":   args.submission,
        "model":        args.model,
    }

    print("""
+====================================================================+
|  REDROB INTELLIGENT CANDIDATE RANKING PIPELINE                    |
|  Stage 1:   Hard filter + prescore  (100k -> ~22k)               |
|  Stage 2:   LightGBM LambdaRank     (22k  -> 15k)               |
|  Stage 3:   Deterministic reasoning (top 100)                    |
|  Stage 3.5: JD-aligned composite score (0-100 with spread)      |
|  Stage 4:   SLM rewrite - Qwen2.5-1.5B (JD-aware, offline)     |
|  Stage 5:   submission.csv                                       |
+====================================================================+
    """)

    if args.status:
        status_check(paths)
        model_exists = os.path.exists(args.model)
        print(f"  SLM model: {'EXISTS' if model_exists else 'MISSING - run: py setup_model.py'}")
        print(f"  Model path: {args.model}")
        return

    # Validate inputs
    if not os.path.exists(args.candidates):
        log(f"candidates file not found: {args.candidates}", "ERR"); sys.exit(1)
    if args.start_stage <= 2 and not os.path.exists(args.ranker):
        log(f"ranker not found: {args.ranker}", "ERR"); sys.exit(1)

    print(f"  Config:")
    print(f"    candidates:     {args.candidates}  ({fmtmb(args.candidates):.0f} MB)")
    print(f"    ranker:         {args.ranker}")
    print(f"    filter_top_k:   {args.filter_top_k:,}")
    print(f"    lgbm_top_k:     {args.lgbm_top_k:,}")
    print(f"    top_final:      {args.top_final}")
    print(f"    no_rewrite:     {args.no_rewrite}")
    print(f"    model:          {args.model}  ({'EXISTS' if os.path.exists(args.model) else 'run setup_model.py'})")
    print()

    monitor     = ResourceMonitor()
    monitor.start()
    stage_stats = []
    t_total     = time.time()

    try:
        # STAGE 1
        if args.start_stage <= 1:
            stage_stats.append(stage1_filter(args.candidates, args.filtered, args.filter_top_k, args.report, monitor))
        else:
            log(f"Skipping Stage 1 -- using: {args.filtered}")

        # STAGE 2
        if args.start_stage <= 2:
            stage_stats.append(stage2_lgbm(args.filtered, args.ranker, args.lgbm_out, args.lgbm_top_k, monitor))
        else:
            log(f"Skipping Stage 2 -- using: {args.lgbm_out}")

        # STAGE 3
        if args.start_stage <= 3:
            stage_stats.append(stage3_fix_reasoning(args.lgbm_out, args.reasoned_out, args.top_final, monitor))
        else:
            log(f"Skipping Stage 3 -- using: {args.reasoned_out}")

        # STAGE 3.5 — JD-aligned composite score (0-100)
        stage_stats.append(stage35_composite_score(
            args.lgbm_out, args.reasoned_out, args.submission, args.top_final, monitor
        ))

        # STAGE 4 - SLM rewrite (updates submission.csv with natural language)
        if not args.no_rewrite and args.start_stage <= 4:
            result = stage4_slm_rewrite(
                args.reasoned_out, args.submission,
                args.model, args.temperature, args.threads, monitor
            )
            stage_stats.append(result)
            if result is None:
                log("SLM skipped -- submission.csv already written by Stage 3.5")
        else:
            # STAGE 5 - write final submission CSV from reasoned JSONL
            stage_stats.append(stage5_submission(args.reasoned_out, args.submission, args.top_final, monitor))

    finally:
        monitor.stop()

    print_report([s for s in stage_stats if s], t_total, paths)
    print(f"  Submission: {args.submission}")
    print(f"  Reasoning:  {args.reasoned_out}\n")


if __name__ == "__main__":
    main()
