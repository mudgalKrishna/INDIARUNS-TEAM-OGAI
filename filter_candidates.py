#!/usr/bin/env python3

"""

filter_candidates.py — Redrob Hackathon Candidate Filtering Pipeline

=====================================================================

Produces a ranked, filtered JSONL of candidates ready to feed into a

teacher model for score + reasoning generation.



Target output: top 50 000 candidates (or --top-k N), ordered by a

multi-signal pre-score so the teacher sees the most promising ones first.



Usage:

    python filter_candidates.py --candidates candidates.jsonl --out filtered_50k.jsonl

    python filter_candidates.py --candidates candidates.jsonl.gz --top-k 30000 --out filtered.jsonl

    python filter_candidates.py --candidates candidates.jsonl --top-k 50000 --out filtered.jsonl --report



What this script does:

    1. Hard-disqualification filters  (eliminate ~70-75 % of pool instantly)

    2. Honeypot detection             (flag & exclude impossible profiles)

    3. Per-candidate pre-score        (rank the survivors before teacher sees them)

    4. Output  top-K JSONL  +  a human-readable filter report



Design principle: every decision here is traceable to either

  (a) an explicit JD statement, or

  (b) an observation from the data-analysis transcript in the conversation.

"""



import json

import gzip

import argparse

import sys

import os

import collections

from datetime import date, datetime



# ─────────────────────────────────────────────────────────────────────────────

#  REFERENCE DATE  (contest reference date from analyse_dataset.py)

# ─────────────────────────────────────────────────────────────────────────────

TODAY = date(2026, 6, 9)



# ─────────────────────────────────────────────────────────────────────────────

#  COMPANY UNIVERSE  (derived from full data scan — 63 unique companies total)

# ─────────────────────────────────────────────────────────────────────────────



# JD explicitly disqualifies: "people who have only worked at consulting firms"

SERVICES_COMPANIES = {

    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',

    'mindtree', 'hcl', 'tech mahindra', 'mphasis', 'hexaware',

    'ltimindtree', 'genpact',

}



# Injected synthetic noise — fictional company names from pop culture

FICTIONAL_COMPANIES = {

    'pied piper', 'initech', 'hooli', 'wayne enterprises',

    'acme corp', 'stark industries', 'globex inc', 'dunder mifflin',

    'umbrella corporation', 'soylent corp',

}



# Real Indian product / startup companies — strong positive signal

PRODUCT_COMPANIES_TIER1 = {

    # FAANG / Big Tech India

    'google', 'meta', 'amazon', 'microsoft', 'apple', 'netflix',

    'adobe', 'salesforce', 'uber', 'linkedin',

}

PRODUCT_COMPANIES_TIER2 = {

    # Indian unicorns / large product companies

    'swiggy', 'zomato', 'razorpay', 'cred', 'flipkart', 'phonepe',

    'paytm', 'meesho', 'nykaa', 'inmobi', "byju's", 'zoho',

    'unacademy', 'upgrad', 'policybazaar', 'ola', 'dream11',

    'freshworks', 'vedantu', 'pharmeasy',

}

PRODUCT_COMPANIES_TIER3 = {

    # Indian AI-native / ML-focused companies

    'glance', 'rephrase.ai', 'aganitha', 'niramai', 'saarthi.ai',

    'sarvam ai', 'mad street den', 'observe.ai', 'krutrim',

    'wysa', 'haptik', 'verloop.io', 'yellow.ai', 'locobuzz',

    'genpact ai',

}

ALL_PRODUCT_COMPANIES = PRODUCT_COMPANIES_TIER1 | PRODUCT_COMPANIES_TIER2 | PRODUCT_COMPANIES_TIER3



# ─────────────────────────────────────────────────────────────────────────────

#  TITLE UNIVERSE  (all 47 unique profile titles + 48 career role titles from data)

# ─────────────────────────────────────────────────────────────────────────────



# Titles that are unambiguously NOT what the JD wants — used for hard filter

HARD_DISQUALIFY_TITLES = {

    'hr manager', 'business analyst', 'accountant', 'marketing manager',

    'operations manager', 'civil engineer', 'mechanical engineer',

    'content writer', 'customer support', 'sales executive',

    'graphic designer', 'project manager',

}



# Titles that are ambiguous — software engineers can do ML, data engineers matter

AMBIGUOUS_TITLES = {

    'software engineer', 'senior software engineer', 'full stack developer',

    'backend engineer', 'cloud engineer', 'devops engineer', 'qa engineer',

    'frontend engineer', 'java developer', '.net developer', 'mobile developer',

    'data analyst', 'analytics engineer', 'data engineer', 'senior data engineer',

}



# Core AI/ML titles — strongest positive title signal

AI_CORE_TITLES = {

    'ml engineer', 'machine learning engineer', 'senior machine learning engineer',

    'staff machine learning engineer', 'lead ai engineer', 'senior ml engineer — search & ranking',

    'ai engineer', 'senior ai engineer',

    'ai research engineer', 'applied ml engineer', 'senior applied scientist',

    'nlp engineer', 'senior nlp engineer',

    'data scientist', 'senior data scientist',

    'ai specialist', 'computer vision engineer',

    'recommendation systems engineer', 'search engineer',

    'senior software engineer (ml)',

    'junior ml engineer',

}



# ─────────────────────────────────────────────────────────────────────────────

#  SKILL UNIVERSE  (from full data scan — 133 unique skill names)

# ─────────────────────────────────────────────────────────────────────────────



# Skills directly mentioned in the JD as "absolutely need"

JD_MUST_HAVE_SKILLS = {

    'sentence transformers', 'embeddings', 'vector search', 'semantic search',

    'pinecone', 'weaviate', 'qdrant', 'milvus', 'faiss',

    'elasticsearch', 'opensearch',

    'information retrieval', 'python',

    'learning to rank',

}



# Skills mentioned in JD "would like to have"

JD_NICE_TO_HAVE_SKILLS = {

    'fine-tuning llms', 'qlora', 'lora', 'peft',

    'hugging face transformers', 'llms', 'langchain', 'llamaindex', 'haystack',

    'rag',

    'machine learning', 'deep learning', 'nlp',

    'pytorch', 'tensorflow', 'scikit-learn',

    'mlops', 'mlflow', 'weights & biases', 'kubeflow',

    'recommendation systems',

    'bm25', 'pgvector',

}



# ML skills from data that are broadly positive even if not JD-specific

ML_BROAD_SKILLS = {

    'feature engineering', 'time series', 'forecasting', 'statistical modeling',

    'reinforcement learning', 'data science',

    'computer vision', 'opencv', 'yolo', 'gans', 'diffusion models',

    'image classification', 'object detection',

    'asr', 'speech recognition', 'tts',

    'prompt engineering',

}



ALL_RELEVANT_SKILLS = JD_MUST_HAVE_SKILLS | JD_NICE_TO_HAVE_SKILLS | ML_BROAD_SKILLS



# Skills that are neutral/non-ML — having only these signals keyword stuffing

GENERIC_TECH_SKILLS = {

    'html', 'css', 'javascript', 'typescript', 'react', 'vue.js', 'angular',

    'next.js', 'webpack', 'tailwind', 'redux', 'graphql', 'rest apis',

    'java', 'go', 'rust', 'spring boot', 'django', 'flask', 'fastapi',

    'docker', 'kubernetes', 'terraform', 'ci/cd', 'aws', 'gcp', 'azure',

    'kafka', 'spark', 'airflow', 'apache beam', 'apache flink', 'hadoop',

    'bigquery', 'snowflake', 'dbt', 'databricks', 'etl', 'data pipelines',

    'sql', 'postgresql', 'mongodb', 'redis',

    'figma', 'photoshop', 'illustrator',

    'excel', 'powerpoint', 'agile', 'scrum', 'project management',

    'sales', 'marketing', 'accounting', 'content writing', 'seo',

    'tally', 'sap', 'salesforce crm', 'six sigma',

}



# ─────────────────────────────────────────────────────────────────────────────

#  HELPERS

# ─────────────────────────────────────────────────────────────────────────────



def _load(path, max_cands=None):

    candidates = []

    opener = gzip.open if path.endswith('.gz') else open

    mode   = 'rt' if path.endswith('.gz') else 'r'

    with opener(path, mode, encoding='utf-8') as f:

        for line in f:

            line = line.strip()

            if not line:

                continue

            candidates.append(json.loads(line))

            if max_cands and len(candidates) >= max_cands:

                break

    return candidates





def _days_since(date_str):

    """Return days between TODAY and a YYYY-MM-DD string."""

    try:

        d = datetime.strptime(date_str, '%Y-%m-%d').date()

        return (TODAY - d).days

    except Exception:

        return 9999





def _normalize_co(name):

    return name.strip().lower()





def _classify_company(name):

    n = _normalize_co(name)

    if n in FICTIONAL_COMPANIES:

        return 'fictional'

    if n in SERVICES_COMPANIES:

        return 'services'

    if n in PRODUCT_COMPANIES_TIER1:

        return 'tier1_product'

    if n in PRODUCT_COMPANIES_TIER2:

        return 'tier2_product'

    if n in PRODUCT_COMPANIES_TIER3:

        return 'tier3_product'

    return 'unknown'





def _career_company_types(career_history):

    types = [_classify_company(r['company']) for r in career_history]

    return types





def _has_relevant_skill(skills):

    names = {s['name'].lower() for s in skills}

    return bool(names & ALL_RELEVANT_SKILLS)





def _relevant_skill_score(skills):

    """0-100 skill quality score based on relevance, proficiency, and duration."""

    score = 0.0

    for s in skills:

        name = s['name'].lower()

        prof = s['proficiency']          # beginner / intermediate / advanced / expert

        dur  = s.get('duration_months', 0)

        endorse = s.get('endorsements', 0)



        if name not in ALL_RELEVANT_SKILLS:

            continue



        # Base weight by relevance tier

        if name in JD_MUST_HAVE_SKILLS:

            base = 4.0

        elif name in JD_NICE_TO_HAVE_SKILLS:

            base = 2.5

        else:

            base = 1.5  # ML_BROAD_SKILLS



        # Proficiency multiplier

        prof_mult = {'beginner': 0.4, 'intermediate': 0.7,

                     'advanced': 1.0, 'expert': 1.3}.get(prof, 0.7)



        # Duration multiplier — logarithmic so 0 months ≠ nothing but high duration matters

        import math

        dur_mult = min(1.5, 0.5 + 0.15 * math.log1p(dur)) if dur > 0 else 0.3



        # Small endorsement bonus (capped)

        endorse_bonus = min(0.3, endorse * 0.01)



        score += base * prof_mult * dur_mult + endorse_bonus



    return min(100.0, score)





def _career_quality_score(career_history):

    """

    Score based on company type, role relevance, and description richness.

    Returns 0-100.

    """

    score = 0.0

    total_weight = 0.0



    for role in career_history:

        ctype = _classify_company(role['company'])

        dur   = role.get('duration_months', 0)

        title = role['title'].lower()

        desc  = role.get('description', '')



        # Weight recent / longer roles more

        weight = max(1.0, dur / 12.0)

        total_weight += weight



        # Company type value

        co_val = {

            'tier1_product': 10.0,

            'tier2_product': 8.0,

            'tier3_product': 7.0,

            'unknown':       4.0,   # Could be legitimate small company

            'services':      1.5,   # Not zero — people move on from services

            'fictional':     0.0,   # Pure noise

        }.get(ctype, 4.0)



        # Role title bonus

        if title in AI_CORE_TITLES:

            role_val = 5.0

        elif title in AMBIGUOUS_TITLES:

            role_val = 2.0

        elif title in HARD_DISQUALIFY_TITLES:

            role_val = 0.0

        else:

            role_val = 1.5  # Unknown title — neutral



        # Description richness bonus (rich descriptions = someone wrote real content)

        desc_val = 0.0

        if len(desc) >= 300:

            desc_val = 2.0

        elif len(desc) >= 100:

            desc_val = 1.0



        score += weight * (co_val + role_val + desc_val)



    if total_weight == 0:

        return 0.0



    # Normalize to 0-100

    raw = score / total_weight

    return min(100.0, raw * 5.0)





def _behavioral_score(sig):

    """Score behavioral signals 0-100."""

    score = 0.0



    # Open to work — binary, high value

    if sig.get('open_to_work_flag'):

        score += 15.0



    # Recency — exponential decay

    days = _days_since(sig.get('last_active_date', '2020-01-01'))

    if days <= 7:

        score += 15.0

    elif days <= 30:

        score += 12.0

    elif days <= 90:

        score += 8.0

    elif days <= 180:

        score += 3.0

    # else 0



    # Notice period (JD: loves sub-30d, can buy out 30d, 30+ still in scope but higher bar)

    notice = sig.get('notice_period_days', 90)

    if notice == 0:

        score += 12.0

    elif notice <= 30:

        score += 10.0

    elif notice <= 60:

        score += 6.0

    elif notice <= 90:

        score += 3.0

    # >90 = 0



    # GitHub activity

    gh = sig.get('github_activity_score', -1)

    if gh > 70:

        score += 10.0

    elif gh > 40:

        score += 7.0

    elif gh > 0:

        score += 3.0

    # -1 = no github = 0



    # Recruiter responsiveness

    rr = sig.get('recruiter_response_rate', 0)

    score += rr * 8.0  # 0-8 pts



    # Applications submitted — active job seeker

    apps = sig.get('applications_submitted_30d', 0)

    if apps >= 5:

        score += 5.0

    elif apps >= 2:

        score += 3.0

    elif apps >= 1:

        score += 1.5



    # Interview completion

    icr = sig.get('interview_completion_rate', 0)

    score += icr * 5.0  # 0-5 pts



    # Profile completeness

    pc = sig.get('profile_completeness_score', 0)

    score += pc * 0.05   # 0-5 pts



    # Verifications

    if sig.get('verified_email'):

        score += 2.0

    if sig.get('verified_phone'):

        score += 2.0

    if sig.get('linkedin_connected'):

        score += 2.0



    # Willing to relocate — JD is Pune/Noida, values this

    if sig.get('willing_to_relocate'):

        score += 3.0



    return min(100.0, score)





def _education_score(education):

    """0-100 education quality score."""

    if not education:

        return 20.0  # neutral, not penalized



    tier_vals = {

        'tier_1':  100.0,

        'tier_2':   70.0,

        'tier_3':   45.0,

        'tier_4':   25.0,

        'unknown':  35.0,

    }

    # Take best tier across all qualifications

    best = max(tier_vals.get(e.get('tier', 'unknown'), 35.0) for e in education)



    # Bonus for CS/ML-adjacent field

    relevant_fields = {

        'computer science', 'computer engineering', 'information technology',

        'machine learning', 'artificial intelligence', 'data science',

        'electronics', 'electrical engineering', 'mathematics', 'statistics',

        'information science', 'computational mathematics',

    }

    field_bonus = 0.0

    for e in education:

        field = e.get('field_of_study', '').lower()

        if any(f in field for f in relevant_fields):

            field_bonus = 10.0

            break



    return min(100.0, best + field_bonus)





def _yoe_score(yoe):

    """Score years of experience against JD target of 5-9 years."""

    if 5 <= yoe <= 9:

        return 100.0

    elif 4 <= yoe < 5:

        return 80.0

    elif 9 < yoe <= 12:

        return 75.0

    elif 3 <= yoe < 4:

        return 50.0

    elif 12 < yoe <= 15:

        return 55.0

    elif 2 <= yoe < 3:

        return 30.0

    elif yoe > 15:

        return 35.0

    else:

        return 5.0





def _location_score(profile, sig):

    """Score based on JD location preferences (Pune/Noida, India)."""

    country = profile.get('country', '').strip()

    location = profile.get('location', '').lower()

    relocate = sig.get('willing_to_relocate', False)



    if country == 'India':

        # JD preferred cities

        preferred = {'noida', 'pune', 'delhi', 'gurgaon', 'bengaluru', 'bangalore',

                     'hyderabad', 'mumbai', 'chennai', 'ncr', 'new delhi'}

        if any(p in location for p in preferred):

            return 100.0

        return 70.0   # India but not preferred city

    elif relocate:

        return 50.0

    else:

        return 20.0   # Outside India, not willing to relocate





def _title_score(title):

    """Score current profile title."""

    t = title.lower().strip()

    if t in AI_CORE_TITLES:

        return 100.0

    elif t in AMBIGUOUS_TITLES:

        return 40.0

    elif t in HARD_DISQUALIFY_TITLES:

        return 0.0

    else:

        return 20.0  # Unknown title





def _has_ai_role_in_career(career_history):

    """True if candidate ever held an AI/ML role title, even if current title is not."""

    for role in career_history:

        if role['title'].lower() in AI_CORE_TITLES:

            return True

    return False





def _product_company_tenure_months(career_history):

    """Total months spent at product companies."""

    total = 0

    for role in career_history:

        if _classify_company(role['company']) in ('tier1_product', 'tier2_product', 'tier3_product'):

            total += role.get('duration_months', 0)

    return total





# ─────────────────────────────────────────────────────────────────────────────

#  HONEYPOT DETECTION

# ─────────────────────────────────────────────────────────────────────────────



def _detect_honeypot(c):

    """

    Return (is_honeypot: bool, flags: list[str]).

    Based on spec section 7: ~80 profiles with subtly impossible signals.

    """

    flags = []

    p    = c['profile']

    sig  = c['redrob_signals']



    # ── Flag 1: Expert skills with 0 months duration

    expert_zero = [s['name'] for s in c['skills']

                   if s['proficiency'] == 'expert' and s.get('duration_months', 0) == 0]

    if len(expert_zero) >= 3:

        flags.append(f'expert_skill_0months:count={len(expert_zero)}')



    # ── Flag 2: Career history months wildly exceed stated YoE

    total_career_months = sum(r.get('duration_months', 0) for r in c['career_history'])

    yoe_months = p['years_of_experience'] * 12

    # Allow 18-month slack (gap years, overlap labelling)

    if total_career_months > yoe_months + 24:

        flags.append(f'career_months({total_career_months})>>yoe_months({yoe_months:.0f})')



    # ── Flag 3: Non-tech title + 5+ expert/advanced AI skills

    is_hard_disq = p['current_title'].lower() in HARD_DISQUALIFY_TITLES

    expert_ai = [s for s in c['skills']

                 if s['name'].lower() in ALL_RELEVANT_SKILLS

                 and s['proficiency'] in ('expert', 'advanced')]

    if is_hard_disq and len(expert_ai) >= 5:

        flags.append(f'nontechTitle+expertAI:count={len(expert_ai)}')



    # ── Flag 4: No GitHub but 4+ expert ML skills (impossible credibility gap)

    gh = sig.get('github_activity_score', -1)

    expert_ml = [s for s in c['skills']

                 if s['name'].lower() in ALL_RELEVANT_SKILLS

                 and s['proficiency'] == 'expert']

    if gh == -1 and len(expert_ml) >= 5:

        flags.append(f'no_github+expert_ml:count={len(expert_ml)}')



    # ── Flag 5: Future graduation dates

    for edu in c['education']:

        if edu.get('end_year', 0) > 2026:

            flags.append(f'future_graduation:{edu["end_year"]}')



    # ── Flag 6: Headline claims "expert" but all skills are beginner

    headline = p.get('headline', '').lower()

    if 'expert' in headline and c['skills'] and all(

            s['proficiency'] == 'beginner' for s in c['skills']):

        flags.append('headline_expert_all_skills_beginner')



    # ── Flag 7: Worked at company founded after stated start date

    # (heuristic: check if role at a very new AI startup has impossibly long duration)

    # E.g. Sarvam AI was founded ~2023, so any role there with 60+ months is fake

    YOUNG_COMPANIES = {'sarvam ai', 'krutrim', 'rephrase.ai', 'saarthi.ai',

                       'observe.ai', 'genpact ai'}

    for role in c['career_history']:

        cn = _normalize_co(role['company'])

        if cn in YOUNG_COMPANIES and role.get('duration_months', 0) > 48:

            flags.append(f'impossibly_long_tenure_at_young_co:{role["company"]}({role["duration_months"]}mo)')



    # ── Flag 8: All skills identical proficiency = suspicious uniformity

    if len(c['skills']) >= 8:

        profs = [s['proficiency'] for s in c['skills']]

        if len(set(profs)) == 1 and profs[0] == 'expert':

            flags.append('all_skills_expert:suspicious_uniformity')



    is_honeypot = len(flags) >= 2  # Multiple flags = very likely honeypot

    return is_honeypot, flags





# ─────────────────────────────────────────────────────────────────────────────

#  HARD FILTER — disqualifies candidates that cannot make the top 100

# ─────────────────────────────────────────────────────────────────────────────



def _hard_filter(c):

    """

    Returns (passes: bool, reason: str).

    Eliminates obvious non-fits without scoring.

    """

    p    = c['profile']

    sig  = c['redrob_signals']

    title = p['current_title'].lower().strip()

    yoe   = p['years_of_experience']

    career = c['career_history']



    # ── Rule H1: Extreme YoE out of range

    if yoe < 1.5:

        return False, 'yoe_too_low(<1.5y)'

    if yoe > 18:

        return False, 'yoe_too_high(>18y)'



    # ── Rule H2: Non-tech title AND no AI/ML role in career history

    if title in HARD_DISQUALIFY_TITLES:

        if not _has_ai_role_in_career(career):

            return False, f'nontechTitle_no_ai_career:{title}'



    # ── Rule H3: Entire career at fictional companies only

    co_types = _career_company_types(career)

    if all(ct == 'fictional' for ct in co_types):

        return False, 'all_fictional_career'



    # ── Rule H4: Entire career at services+fictional with non-tech title

    if title in HARD_DISQUALIFY_TITLES:

        non_product = all(ct in ('fictional', 'services') for ct in co_types)

        if non_product:

            return False, 'nontechTitle_services_only'



    # ── Rule H5: Zero relevant skills AND non-tech title

    if title in HARD_DISQUALIFY_TITLES:

        if not _has_relevant_skill(c['skills']):

            return False, 'nontechTitle_zero_relevant_skills'



    # ── Rule H6: Outside India, not willing to relocate, non-AI title

    # (generous — we keep all who might relocate or are India-based)

    if p.get('country') not in ('India',) and not sig.get('willing_to_relocate'):

        # Still pass non-India if they have strong AI title + skills

        if title not in AI_CORE_TITLES and not _has_relevant_skill(c['skills']):

            return False, 'non_india_no_relocate_no_ai'



    # ── Rule H7: Ghost profile — last active > 2 years ago AND not open to work

    days_inactive = _days_since(sig.get('last_active_date', '2020-01-01'))

    if days_inactive > 730 and not sig.get('open_to_work_flag'):

        return False, f'ghost_profile:inactive_{days_inactive}d'



    return True, 'pass'





# ─────────────────────────────────────────────────────────────────────────────

#  COMPOSITE PRE-SCORE

# ─────────────────────────────────────────────────────────────────────────────



# Weights calibrated against JD priorities:

#   - JD cares most about skills + career (the "intelligence layer" role)

#   - Behavioral signals are tiebreakers (JD says "down-weight unavailable")

#   - Education is secondary

#   - Location matters (Pune/Noida-preferred)

WEIGHTS = {

    'title':     0.10,

    'skills':    0.28,

    'career':    0.28,

    'yoe':       0.10,

    'behavioral':0.14,

    'education': 0.06,

    'location':  0.04,

}





def _compute_prescore(c):

    """Returns a float 0-100 pre-score."""

    p   = c['profile']

    sig = c['redrob_signals']



    scores = {

        'title':     _title_score(p['current_title']),

        'skills':    _relevant_skill_score(c['skills']),

        'career':    _career_quality_score(c['career_history']),

        'yoe':       _yoe_score(p['years_of_experience']),

        'behavioral': _behavioral_score(sig),

        'education': _education_score(c['education']),

        'location':  _location_score(p, sig),

    }



    composite = sum(WEIGHTS[k] * v for k, v in scores.items())

    return round(composite, 4), scores





# ─────────────────────────────────────────────────────────────────────────────

#  FEATURE EXTRACTION  (37 structured features for LightGBM training)

# ─────────────────────────────────────────────────────────────────────────────



def _extract_features(c):

    """

    Extract all structured features for model training.

    Returns a dict of {feature_name: value}.

    These become the input features for the LightGBM ranker.

    """

    import math

    p   = c['profile']

    sig = c['redrob_signals']



    # ── Profile features

    yoe   = p['years_of_experience']

    title = p['current_title'].lower().strip()



    # ── Career features

    career = c['career_history']

    co_types = _career_company_types(career)

    n_roles = len(career)

    total_career_months = sum(r.get('duration_months', 0) for r in career)



    prod_tenure = _product_company_tenure_months(career)

    n_fictional = sum(1 for t in co_types if t == 'fictional')

    n_services  = sum(1 for t in co_types if t == 'services')

    n_product   = sum(1 for t in co_types if 'product' in t)

    n_tier1_prod= sum(1 for t in co_types if t == 'tier1_product')

    n_ai_roles  = sum(1 for r in career if r['title'].lower() in AI_CORE_TITLES)



    desc_lengths = [len(r.get('description', '')) for r in career]

    avg_desc_len = sum(desc_lengths) / max(1, len(desc_lengths))



    # ── Skill features

    skills = c['skills']

    n_skills = len(skills)

    n_relevant = sum(1 for s in skills if s['name'].lower() in ALL_RELEVANT_SKILLS)

    n_must_have = sum(1 for s in skills if s['name'].lower() in JD_MUST_HAVE_SKILLS)

    n_nice_have = sum(1 for s in skills if s['name'].lower() in JD_NICE_TO_HAVE_SKILLS)

    n_expert_relevant = sum(1 for s in skills

                            if s['name'].lower() in ALL_RELEVANT_SKILLS

                            and s['proficiency'] == 'expert')

    n_advanced_relevant = sum(1 for s in skills

                              if s['name'].lower() in ALL_RELEVANT_SKILLS

                              and s['proficiency'] in ('advanced', 'expert'))



    relevant_dur = [s.get('duration_months', 0) for s in skills

                    if s['name'].lower() in ALL_RELEVANT_SKILLS]

    max_relevant_dur = max(relevant_dur) if relevant_dur else 0

    avg_relevant_dur = sum(relevant_dur) / max(1, len(relevant_dur))



    total_endorsements = sum(s.get('endorsements', 0) for s in skills)

    relevant_endorsements = sum(s.get('endorsements', 0) for s in skills

                                if s['name'].lower() in ALL_RELEVANT_SKILLS)



    # ── Behavioral features

    days_inactive    = _days_since(sig.get('last_active_date', '2020-01-01'))

    notice_days      = sig.get('notice_period_days', 90)

    github_score     = sig.get('github_activity_score', -1)

    has_github       = 1 if github_score >= 0 else 0

    github_normalized= max(0.0, github_score)   # -1 → 0

    response_rate    = sig.get('recruiter_response_rate', 0.0)

    interview_compl  = sig.get('interview_completion_rate', 0.0)

    offer_accept     = max(0.0, sig.get('offer_acceptance_rate', 0.0))  # -1→0

    profile_complete = sig.get('profile_completeness_score', 0.0)

    apps_30d         = sig.get('applications_submitted_30d', 0)

    open_to_work     = 1 if sig.get('open_to_work_flag') else 0

    willing_relocate = 1 if sig.get('willing_to_relocate') else 0

    verified_email   = 1 if sig.get('verified_email') else 0

    verified_phone   = 1 if sig.get('verified_phone') else 0

    linkedin         = 1 if sig.get('linkedin_connected') else 0

    connections      = sig.get('connection_count', 0)

    endorsements_rcvd= sig.get('endorsements_received', 0)

    n_assessments    = len(sig.get('skill_assessment_scores', {}))

    avg_assessment   = (sum(sig.get('skill_assessment_scores', {}).values()) /

                        max(1, n_assessments)) if n_assessments > 0 else 0.0



    # Salary (useful signal for realistic expectations)

    sal = sig.get('expected_salary_range_inr_lpa', {})

    sal_mid = (sal.get('min', 0) + sal.get('max', 0)) / 2.0



    # ── Education features

    edus = c['education']

    tier_map = {'tier_1': 4, 'tier_2': 3, 'tier_3': 2, 'tier_4': 1, 'unknown': 2}

    best_edu_tier = max((tier_map.get(e.get('tier', 'unknown'), 2) for e in edus), default=2)



    # ── Location feature

    country = p.get('country', '')

    location = p.get('location', '').lower()

    preferred_locs = {'noida', 'pune', 'delhi', 'gurgaon', 'bengaluru',

                      'bangalore', 'hyderabad', 'mumbai', 'ncr'}

    is_india = 1 if country == 'India' else 0

    is_preferred_loc = 1 if any(loc in location for loc in preferred_locs) else 0



    # ── Composite sub-scores (reuse scoring functions)

    prescore, subscores = _compute_prescore(c)



    return {

        # Profile

        'yoe':                    yoe,

        'yoe_score':              subscores['title'],          # title-based sub-score

        'title_is_ai_core':       1 if title in AI_CORE_TITLES else 0,

        'title_is_ambiguous':     1 if title in AMBIGUOUS_TITLES else 0,

        'title_is_disqualify':    1 if title in HARD_DISQUALIFY_TITLES else 0,



        # Career

        'n_roles':                n_roles,

        'total_career_months':    total_career_months,

        'n_ai_roles_in_career':   n_ai_roles,

        'prod_tenure_months':     prod_tenure,

        'n_product_cos':          n_product,

        'n_tier1_product_cos':    n_tier1_prod,

        'n_services_cos':         n_services,

        'n_fictional_cos':        n_fictional,

        'frac_fictional':         n_fictional / max(1, n_roles),

        'career_quality_score':   subscores['career'],

        'avg_desc_length':        avg_desc_len,



        # Skills

        'n_skills':               n_skills,

        'n_relevant_skills':      n_relevant,

        'n_must_have_skills':     n_must_have,

        'n_nice_have_skills':     n_nice_have,

        'n_expert_relevant':      n_expert_relevant,

        'n_advanced_relevant':    n_advanced_relevant,

        'max_relevant_dur_months':max_relevant_dur,

        'avg_relevant_dur_months':avg_relevant_dur,

        'skill_quality_score':    subscores['skills'],

        'total_endorsements':     total_endorsements,

        'relevant_endorsements':  relevant_endorsements,



        # Behavioral

        'days_inactive':          days_inactive,

        'notice_days':            notice_days,

        'github_score':           github_normalized,

        'has_github':             has_github,

        'response_rate':          response_rate,

        'interview_completion':   interview_compl,

        'offer_acceptance':       offer_accept,

        'profile_completeness':   profile_complete,

        'apps_30d':               apps_30d,

        'open_to_work':           open_to_work,

        'willing_to_relocate':    willing_relocate,

        'n_assessments':          n_assessments,

        'avg_assessment_score':   avg_assessment,

        'connections':            math.log1p(connections),

        'endorsements_received':  math.log1p(endorsements_rcvd),

        'verified_email':         verified_email,

        'verified_phone':         verified_phone,

        'linkedin_connected':     linkedin,

        'sal_mid_lpa':            sal_mid,



        # Education

        'best_edu_tier':          best_edu_tier,

        'education_score':        subscores['education'],



        # Location

        'is_india':               is_india,

        'is_preferred_location':  is_preferred_loc,

        'location_score':         subscores['location'],



        # Composite

        'prescore':               prescore,

        'behavioral_score':       subscores['behavioral'],

        'yoe_fit_score':          subscores['yoe'],

    }





# ─────────────────────────────────────────────────────────────────────────────

#  MAIN PIPELINE

# ─────────────────────────────────────────────────────────────────────────────



def run_filter(candidates_path, top_k=50000, out_path='filtered_candidates.jsonl',

               report=False, report_path='filter_report.txt'):



    print(f"[1/5] Loading candidates from {candidates_path} ...")

    candidates = _load(candidates_path)

    total = len(candidates)

    print(f"      Loaded {total:,} candidates.")



    # ── Stage 1: Hard filter

    print(f"[2/5] Applying hard filters ...")

    passed    = []

    rejected  = collections.Counter()

    for c in candidates:

        ok, reason = _hard_filter(c)

        if ok:

            passed.append(c)

        else:

            rejected[reason] += 1



    print(f"      Passed:   {len(passed):,}")

    print(f"      Rejected: {total - len(passed):,}")



    # ── Stage 2: Honeypot detection

    print(f"[3/5] Running honeypot detection ...")

    clean   = []

    pots    = []

    pot_flags_counter = collections.Counter()

    for c in passed:

        is_hp, flags = _detect_honeypot(c)

        if is_hp:

            pots.append((c, flags))

            for f in flags:

                pot_flags_counter[f.split(':')[0]] += 1

        else:

            clean.append(c)



    print(f"      Clean:    {len(clean):,}")

    print(f"      Honeypots:{len(pots):,}")



    # ── Stage 3: Pre-score all clean candidates

    print(f"[4/5] Computing pre-scores + extracting features ...")

    scored = []

    for c in clean:

        prescore, subscores = _compute_prescore(c)

        features = _extract_features(c)

        scored.append({

            'candidate': c,

            'prescore':  prescore,

            'subscores': subscores,

            'features':  features,

        })



    scored.sort(key=lambda x: x['prescore'], reverse=True)



    # ── Stage 4: Write top-K

    top = scored[:top_k]

    print(f"[5/5] Writing top {len(top):,} candidates to {out_path} ...")

    with open(out_path, 'w', encoding='utf-8') as f:

        for item in top:

            # Enrich candidate with pre-score + features before writing

            record = {

                'candidate_id':  item['candidate']['candidate_id'],

                'prescore':      item['prescore'],

                'subscores':     item['subscores'],

                'features':      item['features'],

                'profile':       item['candidate']['profile'],

                'career_history':item['candidate']['career_history'],

                'education':     item['candidate']['education'],

                'skills':        item['candidate']['skills'],

                'certifications':item['candidate'].get('certifications', []),

                'languages':     item['candidate'].get('languages', []),

                'redrob_signals':item['candidate']['redrob_signals'],

            }

            f.write(json.dumps(record, ensure_ascii=False) + '\n')



    print(f"      Done. {len(top):,} candidates written.")



    # ── Report

    if report:

        _write_report(

            out_file       = report_path,

            total          = total,

            hard_filtered  = total - len(passed),

            hard_reasons   = rejected,

            honeypots      = len(pots),

            pot_flags      = pot_flags_counter,

            pot_examples   = pots[:20],

            clean          = len(clean),

            top_k          = len(top),

            top_scored     = scored[:50],

        )

        print(f"      Report saved to {report_path}")



    # ── Print quick score distribution summary

    print("\n── Pre-score distribution of top-K ──")

    buckets = [(90, 100, '90-100'), (80, 90, '80-90'), (70, 80, '70-80'),

               (60, 70, '60-70'), (50, 60, '50-60'), (0, 50, '<50')]

    for lo, hi, label in buckets:

        cnt = sum(1 for s in top if lo <= s['prescore'] < hi)

        print(f"  {label}: {cnt:,}")



    if top:

        print(f"\n  Top-5 pre-ranked candidates:")

        for item in top[:5]:

            c = item['candidate']

            p = c['profile']

            sig = c['redrob_signals']

            print(f"  #{top.index(item)+1:>3}  {c['candidate_id']}  "

                  f"{p['current_title']:<35}  YoE={p['years_of_experience']:.1f}  "

                  f"co={p['current_company']:<18}  "

                  f"notice={sig['notice_period_days']}d  "

                  f"gh={sig['github_activity_score']:.0f}  "

                  f"pre={item['prescore']:.2f}")



    return len(top)





def _write_report(out_file, total, hard_filtered, hard_reasons, honeypots,

                  pot_flags, pot_examples, clean, top_k, top_scored):

    lines = []

    div = '=' * 72



    def L(s=''):

        lines.append(str(s))



    L(div)

    L('  REDROB — CANDIDATE FILTERING REPORT')

    L(f'  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    L(div)



    L('\n── PIPELINE SUMMARY ──')

    L(f'  Input candidates:        {total:>8,}')

    L(f'  After hard filter:       {total - hard_filtered:>8,}  ({hard_filtered:,} removed)')

    L(f'  After honeypot removal:  {clean:>8,}  ({honeypots:,} removed)')

    L(f'  Output top-K:            {top_k:>8,}')



    L('\n── HARD FILTER REJECTION REASONS ──')

    for reason, cnt in hard_reasons.most_common():

        L(f'  {reason:<50s} {cnt:>7,}')



    L('\n── HONEYPOT FLAG TYPES ──')

    for flag, cnt in pot_flags.most_common():

        L(f'  {flag:<50s} {cnt:>7,}')



    L('\n── SAMPLE HONEYPOT PROFILES ──')

    for c, flags in pot_examples:

        L(f'  {c["candidate_id"]}  {c["profile"]["current_title"]:<30}  {flags}')



    L('\n── TOP 50 PRE-SCORED CANDIDATES ──')

    L(f'  {"Rank":<5}  {"ID":<15}  {"Title":<35}  {"YoE":>4}  '

      f'{"Co":<20}  {"pre":>6}  {"sk":>5}  {"car":>5}  {"beh":>5}')

    L('  ' + '-' * 110)

    for i, item in enumerate(top_scored):

        c  = item['candidate']

        p  = c['profile']

        ss = item['subscores']

        L(f'  {i+1:<5}  {c["candidate_id"]:<15}  '

          f'{p["current_title"]:<35}  {p["years_of_experience"]:>4.1f}  '

          f'{p["current_company"]:<20}  {item["prescore"]:>6.2f}  '

          f'{ss["skills"]:>5.1f}  {ss["career"]:>5.1f}  {ss["behavioral"]:>5.1f}')



    L('\n── FEATURE NAMES FOR LIGHTGBM ──')

    if top_scored:

        for feat_name in top_scored[0]['features']:

            L(f'  {feat_name}')



    L('')

    L(div)

    L('  END OF REPORT')

    L(div)



    with open(out_file, 'w', encoding='utf-8') as f:

        f.write('\n'.join(lines))





# ─────────────────────────────────────────────────────────────────────────────

#  CLI

# ─────────────────────────────────────────────────────────────────────────────



if __name__ == '__main__':

    parser = argparse.ArgumentParser(

        description='Redrob candidate filtering pipeline — produces top-K JSONL for teacher model scoring'

    )

    parser.add_argument(

        '--candidates', required=True,

        help='Path to candidates.jsonl or candidates.jsonl.gz'

    )

    parser.add_argument(

        '--top-k', type=int, default=50000,

        help='Number of candidates to output (default: 50000)'

    )

    parser.add_argument(

        '--out', default='filtered_candidates.jsonl',

        help='Output JSONL path (default: filtered_candidates.jsonl)'

    )

    parser.add_argument(

        '--report', action='store_true',

        help='Write a human-readable filter_report.txt alongside the output'

    )

    parser.add_argument(

        '--report-path', default='filter_report.txt',

        help='Path for the report file (default: filter_report.txt)'

    )

    args = parser.parse_args()



    if not os.path.exists(args.candidates):

        print(f'ERROR: File not found: {args.candidates}')

        sys.exit(1)



    run_filter(
        candidates_path = args.candidates,
        top_k           = args.top_k,
        out_path        = args.out,
        report          = args.report,
        report_path     = args.report_path
    )

if __name__ == '__main__':
    main()
