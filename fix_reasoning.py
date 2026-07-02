#!/usr/bin/env python3
"""
fix_reasoning.py — Deterministic reasoning generator (v2)
==========================================================
Builds factual, JD-aligned 2-sentence reasonings grounded in actual
profile data. No LLM required. Fully offline.

Key fixes over v1:
  - Reads from correct fields: profile, career_history, features, skills
  - Scores normalized to 0-100 (min-max across the pool being scored)
  - Reasoning is coherent with rank (top candidates get positive framing)
  - No "limited match" for candidates with 5 must-have skills
  - Highlights the actual signal that drove the LightGBM score

Usage:
    py fix_reasoning.py --input lgbm_top15k.jsonl --output top100_reasoned.jsonl --top-k 100
"""

import json
import re
import argparse
import os
import sys
from datetime import datetime

# ── JD constants ──────────────────────────────────────────────────────────────

PRODUCT_COMPANIES = {
    "swiggy", "razorpay", "cred", "zomato", "flipkart", "meesho", "nykaa",
    "inmobi", "byju's", "policybazaar", "ola", "zoho", "vedantu", "paytm",
    "unacademy", "pharmeasy", "upgrad", "freshworks", "phonepe", "dream11",
    "haptik", "yellow.ai", "sarvam ai", "mad street den", "wysa", "aganitha",
    "niramai", "observe.ai", "krutrim", "verloop.io", "locobuzz",
    "rephrase.ai", "saarthi.ai", "google", "microsoft", "amazon", "apple",
    "netflix", "uber", "meta", "adobe", "salesforce", "linkedin", "glance",
    "genpact ai"
}

SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "mindtree", "tech mahindra", "mphasis"
}

# Must-have JD skills (from JD: "Things you absolutely need")
MUST_HAVE_SKILLS = {
    'python', 'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus',
    'elasticsearch', 'opensearch', 'pgvector', 'bm25', 'vector search',
    'information retrieval', 'sentence transformers', 'embeddings',
    'bge', 'e5', 'rag', 'hybrid search', 'ndcg', 'mrr', 'map',
    'learning to rank', 'ltr', 'reranking', 'vector database'
}

# Nice-to-have JD skills
NICE_HAVE_SKILLS = {
    'lora', 'qlora', 'peft', 'fine-tuning', 'pytorch', 'tensorflow',
    'hugging face transformers', 'recommendation systems', 'nlp',
    'langchain', 'llamaindex', 'haystack', 'opensearch', 'xgboost',
    'lightgbm', 'mlops', 'weights & biases', 'kubeflow', 'ray'
}

# JD retrieval-core titles
RETRIEVAL_TITLES = {
    'search engineer', 'retrieval engineer', 'ranking engineer',
    'recommendation systems engineer', 'recsys engineer',
    'nlp engineer', 'applied ml engineer', 'senior ai engineer',
    'ai engineer', 'machine learning engineer', 'staff ml engineer',
    'lead ai engineer', 'senior ml engineer', 'applied scientist',
    'senior applied scientist', 'research engineer', 'senior data scientist'
}


# ── Profile extraction helpers ────────────────────────────────────────────────

def get_profile(rec):
    return rec.get('profile') or {}

def get_career(rec):
    return rec.get('career_history') or rec.get('work_experience') or []

def get_features(rec):
    return rec.get('features') or {}

def get_signals(rec):
    return rec.get('redrob_signals') or {}

def get_skills(rec):
    return rec.get('skills') or []

def get_score(rec):
    """Return best available score field."""
    for field in ('lgbm_score', 'teacher_score', 'prescore'):
        v = rec.get(field)
        if v is not None:
            return float(v)
    return float(rec.get('features', {}).get('prescore', 0))


def latest_company(rec):
    """Get the current/most recent company name."""
    p = get_profile(rec)
    cc = p.get('current_company', '')
    if cc:
        return cc
    ch = get_career(rec)
    for exp in ch:
        if exp.get('is_current'):
            return exp.get('company', '') or exp.get('company_name', '')
    if ch:
        return ch[0].get('company', '') or ch[0].get('company_name', '')
    return ''


def latest_title(rec):
    """Get current job title."""
    p = get_profile(rec)
    t = p.get('current_title', '')
    if t:
        return t
    ch = get_career(rec)
    for exp in ch:
        if exp.get('is_current'):
            return exp.get('title', '')
    if ch:
        return ch[0].get('title', '')
    return ''


def get_yoe(rec):
    p = get_profile(rec)
    yoe = p.get('years_of_experience')
    if yoe is not None:
        return float(yoe)
    return float(get_features(rec).get('yoe', 0))


def top_relevant_skills(rec, n=4):
    """Return top N relevant skill names (must-have first, then nice-have)."""
    skills = get_skills(rec)
    must, nice, other = [], [], []
    for s in skills:
        name  = (s.get('name') or '').strip()
        name_l = name.lower()
        prof  = (s.get('proficiency') or '').lower()
        if name_l in MUST_HAVE_SKILLS:
            must.append((name, s.get('duration_months', 0)))
        elif name_l in NICE_HAVE_SKILLS:
            nice.append((name, s.get('duration_months', 0)))
        elif prof in ('expert', 'advanced'):
            other.append((name, s.get('duration_months', 0)))

    must.sort(key=lambda x: -x[1])
    nice.sort(key=lambda x: -x[1])
    ordered = [x[0] for x in must[:n]] + [x[0] for x in nice[:n]]
    if len(ordered) < n:
        ordered += [x[0] for x in other if x[0] not in ordered]
    return ordered[:n]


def longest_relevant_skill(rec):
    """Return (name, duration_months) of the must-have skill with most experience."""
    skills = get_skills(rec)
    best = None
    for s in skills:
        name_l = (s.get('name') or '').lower()
        if name_l in MUST_HAVE_SKILLS:
            dur = s.get('duration_months', 0) or 0
            if best is None or dur > best[1]:
                best = (s.get('name'), dur)
    return best or (None, 0)


def is_product_company(company_name):
    return company_name.lower() in PRODUCT_COMPANIES


def is_services_company(company_name):
    return company_name.lower() in SERVICES_COMPANIES


def notice_label(days):
    if days == 0:
        return "immediately available"
    if days <= 15:
        return f"{days}-day notice (excellent)"
    if days <= 30:
        return f"{days}-day notice (strong)"
    if days <= 60:
        return f"{days}-day notice (acceptable)"
    if days <= 90:
        return f"{days}-day notice (high risk)"
    return f"{days}-day notice (very high risk — flag for recruiter)"


def github_label(score):
    if score < 0:
        return None
    if score >= 80:
        return f"strong GitHub presence (score {score:.0f}/100)"
    if score >= 50:
        return f"active GitHub (score {score:.0f}/100)"
    if score >= 20:
        return f"some GitHub activity (score {score:.0f}/100)"
    return None


# ── Reasoning quality tiers ───────────────────────────────────────────────────

def classify_candidate(rec):
    """
    Returns one of: 'excellent', 'strong', 'good', 'marginal', 'weak'
    Based on feature signals aligned with the JD.
    """
    feats = get_features(rec)
    sig   = get_signals(rec)

    n_must    = feats.get('n_must_have_skills', 0) or 0
    yoe       = feats.get('yoe', 0) or 0
    n_prod    = feats.get('n_product_cos', 0) or 0
    notice    = sig.get('notice_period_days', 90) or 90
    github    = sig.get('github_activity_score', -1)
    open_w    = sig.get('open_to_work_flag', False)
    title_core = feats.get('title_is_ai_core', 0) or 0
    prescore  = feats.get('prescore', 0) or 0
    n_roles   = feats.get('n_ai_roles_in_career', 0) or 0

    score = 0
    score += min(n_must * 12, 48)          # up to 48 pts for must-have skills
    score += min(n_prod * 8, 24)           # up to 24 pts for product co tenure
    score += 10 if title_core else 0       # 10 pts if title matches JD role
    score += 8 if 5 <= yoe <= 9 else (5 if yoe > 3 else 0)
    score += 6 if notice <= 30 else (3 if notice <= 60 else 0)
    score += 5 if github >= 50 else (2 if github >= 20 else 0)
    score += 4 if open_w else 0

    if score >= 70:
        return 'excellent'
    if score >= 55:
        return 'strong'
    if score >= 40:
        return 'good'
    if score >= 25:
        return 'marginal'
    return 'weak'


# ── Core reasoning builder ────────────────────────────────────────────────────

def build_reasoning(rec):
    """
    Build a 2-sentence factual assessment aligned with the Redrob JD.
    Sentence 1: what they bring (title, company, experience, key skills)
    Sentence 2: availability + primary concern or why they're a strong pick
    """
    p       = get_profile(rec)
    feats   = get_features(rec)
    sig     = get_signals(rec)
    ch      = get_career(rec)

    title    = latest_title(rec) or p.get('current_title', 'ML Engineer')
    company  = latest_company(rec) or 'a product company'
    yoe      = get_yoe(rec)
    notice   = int(sig.get('notice_period_days', 90) or 90)
    github   = float(sig.get('github_activity_score', -1) or -1)
    open_w   = sig.get('open_to_work_flag', False)
    response = float(sig.get('recruiter_response_rate', 0.5) or 0.5)
    n_must   = int(feats.get('n_must_have_skills', 0) or 0)
    n_prod   = int(feats.get('n_product_cos', 0) or 0)
    n_svc    = int(feats.get('n_services_cos', 0) or 0)
    prescore = float(feats.get('prescore', 0) or 0)

    top_skills = top_relevant_skills(rec, n=4)
    best_skill, best_skill_dur = longest_relevant_skill(rec)

    tier = classify_candidate(rec)
    is_prod = is_product_company(company)
    is_svc  = is_services_company(company)

    # ── Sentence 1: strengths ──────────────────────────────────────────────
    yoe_str = f"{yoe:.1f}" if yoe != int(yoe) else str(int(yoe))
    skill_str = ", ".join(top_skills[:3]) if top_skills else "applied ML"

    if best_skill and best_skill_dur >= 24:
        skill_anchor = f"{best_skill_dur}mo of production {best_skill}"
    elif top_skills:
        skill_anchor = f"hands-on work in {', '.join(top_skills[:2])}"
    else:
        skill_anchor = "applied ML experience"

    if tier == 'excellent':
        s1 = (
            f"{title} at {company} with {yoe_str} years of experience "
            f"and {skill_anchor}; strong alignment with this retrieval-focused role "
            f"including {n_must} of the JD's must-have technical requirements."
        )
    elif tier == 'strong':
        s1 = (
            f"{title} at {company} with {yoe_str} years of experience "
            f"and {skill_anchor}; covers {n_must} JD must-haves "
            f"({skill_str}) making them a solid candidate for this role."
        )
    elif tier == 'good':
        if n_must >= 3:
            s1 = (
                f"{title} at {company} ({yoe_str}yr); demonstrates {skill_anchor} "
                f"and meets {n_must} of the JD's core requirements — "
                f"reasonable fit for this Senior AI Engineer role."
            )
        else:
            s1 = (
                f"{title} at {company} ({yoe_str}yr) with skills in {skill_str}; "
                f"partial match — covers {n_must} of the JD's must-have retrieval/ranking "
                f"requirements, with some gaps remaining."
            )
    elif tier == 'marginal':
        s1 = (
            f"{title} at {company} ({yoe_str}yr); meets only {n_must} of the JD's "
            f"core requirements — limited direct match for this retrieval-engineering role."
        )
    else:
        s1 = (
            f"{title} at {company} ({yoe_str}yr); profile does not strongly match "
            f"this Senior AI Engineer role's retrieval and ranking requirements."
        )

    # ── Sentence 2: availability + primary signal ──────────────────────────
    gh_label  = github_label(github)
    open_str  = "actively looking" if open_w else "not flagged as actively looking"

    if tier in ('excellent', 'strong'):
        if notice <= 30:
            avail = f"Available within {notice} days"
        elif notice <= 60:
            avail = f"Notice period is {notice} days (manageable)"
        else:
            avail = f"Caution: {notice}-day notice raises timeline risk"

        extras = []
        if gh_label:
            extras.append(gh_label)
        if response >= 0.8:
            extras.append(f"highly responsive ({int(response*100)}% reply rate)")
        if open_w:
            extras.append("actively job-hunting")

        if extras:
            s2 = f"{avail}; {', '.join(extras[:2])} — recommend prioritising outreach."
        else:
            s2 = f"{avail} — recommend prioritising outreach."

    elif tier == 'good':
        concern = []
        if notice > 60:
            concern.append(f"{notice}-day notice is a risk")
        if n_svc > 0 and n_prod == 0:
            concern.append("services-only background")
        if github < 20 and github >= 0:
            concern.append("limited open-source visibility")

        if not open_w:
            concern.append("not actively looking")

        if concern:
            s2 = (
                f"Availability: {notice_label(notice)}; "
                f"primary concern — {'; '.join(concern[:2])}."
            )
        elif gh_label:
            s2 = f"{notice_label(notice).capitalize()}; {gh_label} — worth a screening call."
        else:
            s2 = f"{notice_label(notice).capitalize()} — worth a screening call."

    else:  # marginal / weak
        concerns = []
        if n_must < 2:
            concerns.append(f"only {n_must} must-have retrieval skills")
        if notice > 60:
            concerns.append(f"{notice}-day notice adds risk")
        if not open_w:
            concerns.append("not actively looking")
        if n_svc > 0 and n_prod == 0:
            concerns.append("no product-company experience")

        if concerns:
            s2 = f"Key gaps: {'; '.join(concerns[:3])}."
        else:
            s2 = f"Notice period: {notice_label(notice)}."

    return f"{s1} {s2}"


# ── Score normalisation (0-100) ───────────────────────────────────────────────

def normalize_scores(records, score_field='lgbm_score'):
    """Normalize lgbm_score (or fallback) to 0-100 using min-max of THIS pool, 
    UNLESS it's already a 0-100 regression score."""
    raw = [get_score(r) for r in records]
    lo, hi = min(raw), max(raw)
    
    # If the max score is > 10, it's a Regressor outputting 0-100 directly. Don't rescale.
    if hi > 10.0:
        return raw
        
    span = hi - lo if hi != lo else 1.0
    return [(s - lo) / span * 100 for s in raw]


# ── Main processing ───────────────────────────────────────────────────────────

def process_records(records, out_path, top_k=100):
    """
    Called by run_pipeline.py (Stage 3).
    Returns stats dict.
    """
    subset = records[:top_k]
    norm_scores = normalize_scores(subset)

    hallucinations_replaced = 0
    unique_reasonings = set()
    output_records = []

    for i, rec in enumerate(subset):
        cid  = rec.get('candidate_id', f'CAND_{i}')
        rank = rec.get('lgbm_rank', i + 1)

        reasoning = build_reasoning(rec)

        unique_reasonings.add(reasoning)
        output_records.append({
            'rank':          rank,
            'candidate_id':  cid,
            'score':         round(norm_scores[i], 2),
            'raw_score':     round(get_score(rec), 6),
            'fixed_reasoning': reasoning,
        })

    with open(out_path, 'w', encoding='utf-8') as f:
        for r in output_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f'  [reasoning] {len(output_records)} done | unique={len(unique_reasonings)}/{len(output_records)}')

    return {
        'total':                  len(output_records),
        'hallucinations_replaced': hallucinations_replaced,
        'unique_reasonings':       len(unique_reasonings),
    }


# ── Submission CSV writer ─────────────────────────────────────────────────────

def write_submission_csv(records, out_path):
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        f.write('rank,candidate_id,score,reasoning\n')
        for r in records:
            rsn = r.get('fixed_reasoning', '').replace('"', "'").replace('\n', ' ').strip()
            f.write(f'{r["rank"]},{r["candidate_id"]},{r["score"]},"{rsn}"\n')
    print(f'  Written {len(records)} rows to {out_path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Deterministic reasoning generator (v2)')
    parser.add_argument('--input',  required=True, help='Input JSONL (lgbm_top15k.jsonl or similar)')
    parser.add_argument('--output', required=True, help='Output JSONL path')
    parser.add_argument('--submission', default=None, help='Also write submission.csv')
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--show', type=int, default=10, help='Print top N reasoning samples')
    args = parser.parse_args()

    print(f'\n  Loading {args.input}...')
    records = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f'  Loaded {len(records):,} records')

    stats = process_records(records, args.output, top_k=args.top_k)

    # Print preview
    print(f'\n  ── TOP-{args.show} REASONING PREVIEW ──')
    print(f'  {"Rank":<5}  {"Candidate ID":<18}  {"Score":>6}  Reasoning')
    print(f'  {"-"*90}')
    with open(args.output, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= args.show:
                break
            r = json.loads(line)
            print(f'  {r["rank"]:<5}  {r["candidate_id"]:<18}  {r["score"]:>6.1f}  {r["fixed_reasoning"][:80]}...')

    if args.submission:
        out_records = []
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                out_records.append(json.loads(line.strip()))
        write_submission_csv(out_records, args.submission)

    print(f'\n  Stats:')
    print(f'    total:                  {stats["total"]}')
    print(f'    unique_reasonings:      {stats["unique_reasonings"]}')
    print(f'  Output: {args.output}\n')


if __name__ == '__main__':
    main()
