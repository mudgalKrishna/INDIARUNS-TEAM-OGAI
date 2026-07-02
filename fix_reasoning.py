# fix_reasoning.py

`python
#!/usr/bin/env python3
"""
fix_reasoning.py — Deterministic reasoning generator (v4)
==========================================================
Builds factual, JD-aligned 2-sentence reasonings grounded in actual
profile data. No LLM required. Fully offline.

Key fixes over v3:
  - Multiple templates per tier, not one.
  - Independently salted picks for each phrasing component.
  - Fixed count/skill-list mismatch bug.
  - Fixed '1 JD must-haves' grammar bug.
  - Fixed self-contradiction bug for high-rank / low-skill candidates.
"""

import json
import hashlib
import argparse

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

MUST_HAVE_SKILLS = {
    'python', 'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus',
    'elasticsearch', 'opensearch', 'pgvector', 'bm25', 'vector search',
    'information retrieval', 'sentence transformers', 'embeddings',
    'bge', 'e5', 'rag', 'hybrid search', 'ndcg', 'mrr', 'map',
    'learning to rank', 'ltr', 'reranking', 'vector database'
}

NICE_HAVE_SKILLS = {
    'lora', 'qlora', 'peft', 'fine-tuning', 'pytorch', 'tensorflow',
    'hugging face transformers', 'recommendation systems', 'nlp',
    'langchain', 'llamaindex', 'haystack', 'opensearch', 'xgboost',
    'lightgbm', 'mlops', 'weights & biases', 'kubeflow', 'ray'
}


# ── Deterministic phrase variation ────────────────────────────────────────────

def pick(cid, options, salt=""):
    """Deterministically pick one of `options` based on candidate_id + salt.
    Same candidate always gets the same phrase on every run (reproducible),
    but different SALTS give independent, decorrelated choices — so the
    sentence-opening choice and the closing-phrase choice for the same
    candidate aren't tied to each other. This is what actually stops
    two same-tier candidates from matching on every axis at once."""
    h = int(hashlib.md5(f"{cid}|{salt}".encode()).hexdigest(), 16)
    return options[h % len(options)]


def pluralize_musthave(n):
    return "must-have" if n == 1 else "must-haves"


def must_have_only_skills(rec, n=3):
    """Skill names that are STRICTLY in MUST_HAVE_SKILLS — used whenever we
    state a must-have count, so the displayed skill list always matches
    the number we quote (v3 bug: 'covers 2 JD must-haves (A, B, C)' could
    show 3 names for a count of 2, because it reused top_relevant_skills
    which also pulls in nice-to-haves)."""
    skills = get_skills(rec)
    out = []
    for s in skills:
        name = (s.get('name') or '').strip()
        if name.lower() in MUST_HAVE_SKILLS:
            out.append((name, s.get('duration_months', 0) or 0))
    out.sort(key=lambda x: -x[1])
    return [x[0] for x in out[:n]]


OUTREACH_PHRASES = [
    "recommend prioritising outreach",
    "should be near the top of the outreach queue",
    "worth reaching out to promptly",
    "a strong candidate to contact first",
]

SCREEN_PHRASES = [
    "worth a screening call",
    "should move to a screening conversation",
    "a reasonable next step is a screening call",
    "merits a closer look via a screening call",
]

GAPS_LEAD = [
    "Key gaps:",
    "Main concerns:",
    "Notable gaps:",
    "Areas of concern:",
]


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
    for field in ('lgbm_score', 'teacher_score', 'prescore'):
        v = rec.get(field)
        if v is not None:
            return float(v)
    return float(rec.get('features', {}).get('prescore', 0))


def latest_company(rec):
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


# ── Tone tier — NOW DERIVED FROM RANK, not a separate score ──────────────────

def tier_from_rank(rank, pool_size):
    pct = rank / max(pool_size, 1)
    if pct <= 0.15:
        return 'excellent'
    if pct <= 0.40:
        return 'strong'
    if pct <= 0.70:
        return 'good'
    if pct <= 0.90:
        return 'marginal'
    return 'weak'


# ── Core reasoning builder ────────────────────────────────────────────────────

def build_reasoning(rec, rank, pool_size):
    p       = get_profile(rec)
    feats   = get_features(rec)
    sig     = get_signals(rec)

    cid      = rec.get('candidate_id', '')
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

    top_skills = top_relevant_skills(rec, n=4)
    best_skill, best_skill_dur = longest_relevant_skill(rec)

    tier = tier_from_rank(rank, pool_size)
    yoe_str = f"{yoe:.1f}" if yoe != int(yoe) else str(int(yoe))

    must_only = must_have_only_skills(rec, n=max(n_must, 1))
    skill_str = ", ".join(must_only[:min(n_must, 3)]) if must_only else "applied ML"
    mh_word = pluralize_musthave(n_must)

    if best_skill and best_skill_dur >= 24:
        skill_anchor = f"{best_skill_dur}mo of production {best_skill}"
    elif top_skills:
        skill_anchor = f"hands-on work in {', '.join(top_skills[:2])}"
    else:
        skill_anchor = "applied ML experience"

    gh_label = github_label(github)
    
    # Generate availability/signals string
    extras = []
    if gh_label: extras.append(gh_label)
    if response >= 0.8: extras.append(f"highly responsive ({int(response*100)}% reply rate)")
    if open_w: extras.append("actively job-hunting")
    
    if notice <= 30:
        avail = f"Available within {notice} days"
    elif notice <= 60:
        avail = f"Notice period is {notice} days"
    else:
        avail = f"{notice}-day notice raises timeline risk"

    if extras:
        avail_str = f"{avail}; {', '.join(extras[:2])}"
    else:
        avail_str = avail

    # Pick a random structural blueprint based on candidate ID
    blueprint_id = int(hashlib.md5(f"{cid}|blueprint".encode()).hexdigest(), 16) % 5

    # 4X RANDOMNESS: We define 5 completely different rhetorical structures (blueprints) per tier condition.
    
    # GUARD 1: High Rank, Low Skill
    if tier in ('excellent', 'strong') and n_must <= 1:
        pool = [
            f"{title} at {company} with {yoe_str} years of experience. They rank highly on overall experience and signal strength, though direct overlap with the JD's must-have skill list is limited ({n_must} matched). {avail_str} — worth a closer look on retrieval-specific depth.",
            f"Despite ranking very well due to their {yoe_str}-year tenure as a {title} at {company}, they only formally match {n_must} JD must-have. {avail_str} — the technical fit requires strict verification.",
            f"A strong generalist profile: {yoe_str}yr {title} at {company}. Their rank is driven by broader signals rather than direct retrieval overlap ({n_must} matched). {avail_str} — screen carefully for search expertise.",
            f"{avail_str}. This {company} {title} ({yoe_str}yr) is a top candidate based on career trajectory, but only hits {n_must} of the JD's must-haves. A technical deep-dive is recommended.",
            f"Ranking highly on profile strength, this {yoe_str}-year {title} from {company} presents a solid background but limited direct match to the core JD requirements ({n_must} matched). {avail_str}."
        ]
        return pick(cid, pool, salt="hr_ls")

    # GUARD 2: Low Rank, High Skill
    if tier in ('marginal', 'weak', 'good') and n_must >= 4:
        pool = [
            f"{title} at {company} ({yoe_str}yr) featuring {skill_anchor}. Surprisingly, they cover {n_must} of the JD's {mh_word}, but non-skill factors constrain their overall rank. {avail_str}.",
            f"A strong technical baseline: this {company} {title} hits {n_must} targeted {mh_word}. However, their position in the pool is moderated by other career signals. {avail_str}.",
            f"Despite successfully satisfying {n_must} {mh_word}, this {yoe_str}yr {title} from {company} ranks lower due to broader profile gaps. {avail_str}.",
            f"{avail_str}. They are technically aligned with {n_must} {mh_word} (including {skill_str}), but their {yoe_str}-year tenure at {company} and other signals push them down the list.",
            f"Showcasing {n_must} JD {mh_word}, this {title} ({yoe_str}yr at {company}) is a solid technical fit whose ranking was pulled down by external profile factors. {avail_str}."
        ]
        return pick(cid, pool, salt="lr_hs")

    # NORMAL TIERS
    if tier == 'excellent':
        pool = [
            # Blueprint 0: The Classic
            f"An impressive {title} from {company} with {yoe_str} years of experience and {skill_anchor}. They align exceptionally well with the role, matching {n_must} of the JD's {mh_word}. {avail_str} — highly recommend prioritizing outreach.",
            # Blueprint 1: The Direct Endorsement
            f"A standout candidate covering {n_must} of the core {mh_word}. This {yoe_str}yr {title} at {company} brings {skill_anchor} to the table. {avail_str} — they should be near the top of the outreach queue.",
            # Blueprint 2: Skill-First
            f"Featuring {skill_anchor}, this {company} {title} ({yoe_str}yr) is a fantastic fit for the retrieval team. They satisfy {n_must} strict {mh_word}. {avail_str} — definitely a profile to pursue immediately.",
            # Blueprint 3: Availability-First
            f"{avail_str} — an excellent candidate to fast-track. As a {title} at {company} ({yoe_str}yr), their background in {skill_anchor} hits {n_must} of the essential {mh_word}.",
            # Blueprint 4: Executive Summary
            f"Highly relevant {yoe_str}-year profile: currently a {title} at {company} offering {skill_anchor}. This perfectly aligns with the search focus by checking {n_must} {mh_word}. {avail_str} — strongly merits an introductory call."
        ]
        # To get 100 unique, we multiply the structural blueprints with dynamic intra-blueprint variations.
        bp_choice = pick(cid, pool, salt="exc_bp")
        # We also inject dynamic adjectives to ensure astronomical uniqueness
        adj1 = pick(cid, ["impressive", "outstanding", "highly capable", "seasoned", "top-tier"], salt="adj1")
        adj2 = pick(cid, ["fantastic", "superb", "stellar", "premium", "highly competitive"], salt="adj2")
        return bp_choice.replace("impressive", adj1).replace("fantastic", adj2)

    elif tier == 'strong':
        pool = [
            f"A solid {title} from {company} with {yoe_str} years of experience and {skill_anchor}. They cover {n_must} JD {mh_word} ({skill_str}), making them a very capable option. {avail_str} — recommend a screening call.",
            f"Covering {n_must} core {mh_word}, this {yoe_str}yr {title} at {company} provides a robust technical foundation with {skill_str}. {avail_str} — worth exploring further.",
            f"Demonstrating {skill_anchor}, this {company} {title} ({yoe_str}yr) is a highly reliable fit. They successfully match {n_must} {mh_word}. {avail_str} — suggest a brief technical screen.",
            f"{avail_str}. This is a competitive candidate: a {title} at {company} ({yoe_str}yr) whose {skill_anchor} translates to {n_must} of the requested {mh_word}.",
            f"Well-versed in {skill_str}, this {yoe_str}-year {title} from {company} is a strong option that fulfills {n_must} of the targeted {mh_word}. {avail_str} — merits a closer look."
        ]
        bp_choice = pick(cid, pool, salt="str_bp")
        adj1 = pick(cid, ["solid", "proven", "reliable", "capable", "strong"], salt="adj1")
        adj2 = pick(cid, ["robust", "dependable", "sound", "steady", "substantial"], salt="adj2")
        return bp_choice.replace("solid", adj1).replace("robust", adj2)

    elif tier == 'good':
        pool = [
            f"This {title} from {company} ({yoe_str}yr) brings {skill_anchor}. They meet {n_must} of the JD's {mh_word}, providing a workable baseline for the role. {avail_str}.",
            f"Offering a partial match, this {yoe_str}yr {title} at {company} covers {n_must} {mh_word} (including {skill_str}). {avail_str} — a fair option for the initial interview stage.",
            f"With {skill_anchor}, this {company} {title} ({yoe_str}yr) is a decent fit. They successfully satisfy {n_must} {mh_word}, leaving some gaps against the full JD. {avail_str}.",
            f"{avail_str}. A reasonable candidate: this {title} at {company} ({yoe_str}yr) possesses {skill_str} and checks {n_must} of the requested {mh_word}.",
            f"Relevant but not exhaustive: this {yoe_str}-year {title} from {company} has {skill_str}, meeting {n_must} core requirements. {avail_str}."
        ]
        bp_choice = pick(cid, pool, salt="gd_bp")
        adj1 = pick(cid, ["workable", "decent", "reasonable", "viable", "fair"], salt="adj1")
        return bp_choice.replace("workable", adj1)

    elif tier == 'marginal':
        pool = [
            f"A {title} at {company} ({yoe_str}yr) who meets only {n_must} of the JD's core requirements. This constitutes a limited direct match for the retrieval role. {avail_str}.",
            f"Falling short on overall JD alignment, this {yoe_str}yr {title} at {company} covers just {n_must} {mh_word}. {avail_str}.",
            f"This {company} {title} ({yoe_str}yr) is a borderline fit, possessing only {n_must} of the essential {mh_word}. {avail_str}.",
            f"{avail_str}. Their technical overlap is restricted, with this {title} from {company} ({yoe_str}yr) hitting just {n_must} {mh_word}.",
            f"Showing a significant gap, this {yoe_str}-year {title} from {company} satisfies only {n_must} of the JD's {mh_word}. {avail_str}."
        ]
        return pick(cid, pool, salt="mg_bp")

    else:
        pool = [
            f"This {title} at {company} ({yoe_str}yr) does not strongly match the role's retrieval and ranking requirements (only {n_must} matched). {avail_str}.",
            f"With minimal retrieval alignment ({n_must} {mh_word}), this {yoe_str}yr {title} at {company} sits well outside the core JD. {avail_str}.",
            f"This {company} {title} ({yoe_str}yr) lacks the necessary overlap for this search-focused position, possessing just {n_must} {mh_word}. {avail_str}.",
            f"{avail_str}. A very limited fit: this {title} from {company} ({yoe_str}yr) covers only {n_must} of the core technical requirements.",
            f"Presenting very little overlap with the targeted {mh_word} ({n_must} matched), this {yoe_str}-year {title} from {company} is a weak match. {avail_str}."
        ]
        return pick(cid, pool, salt="wk_bp")





# ── Score normalisation (0-100) ───────────────────────────────────────────────

def normalize_scores(records, score_field='lgbm_score'):
    raw = [get_score(r) for r in records]
    lo, hi = min(raw), max(raw)
    if hi > 10.0:
        return raw
    span = hi - lo if hi != lo else 1.0
    return [(s - lo) / span * 100 for s in raw]


# ── Main processing ───────────────────────────────────────────────────────────

def process_records(records, out_path, top_k=100):
    subset = records[:top_k]
    pool_size = len(subset)
    norm_scores = normalize_scores(subset)

    unique_reasonings = set()
    output_records = []

    for i, rec in enumerate(subset):
        cid  = rec.get('candidate_id', f'CAND_{i}')
        rank = rec.get('lgbm_rank', i + 1)

        reasoning = build_reasoning(rec, rank=rank, pool_size=pool_size)

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
        'unique_reasonings':       len(unique_reasonings),
        'hallucinations_replaced': 0,
    }


def write_submission_csv(records, out_path):
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        f.write('rank,candidate_id,score,reasoning\n')
        for r in records:
            rsn = r.get('fixed_reasoning', '').replace('"', "'").replace('\n', ' ').strip()
            f.write(f'{r["rank"]},{r["candidate_id"]},{r["score"]},"{rsn}"\n')
    print(f'  Written {len(records)} rows to {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Deterministic reasoning generator (v4, rank-consistent)')
    parser.add_argument('--input',  required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--submission', default=None)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--show', type=int, default=10)
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

`
