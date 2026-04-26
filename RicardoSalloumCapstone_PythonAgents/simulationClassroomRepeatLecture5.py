#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClassroomTest.py
================
Union of studentSimulation2.py + AutoTest2.py.

Simulates a 3/3/3/3 classroom (low / medium / high / perfect) using the
real Markov attention model from BaseStudent.py, then sends each student's
DEGRADED context to the live API and runs the full 25-question MCQ exam.

Each student gets the correct_lecture passed through their own attention
filter before being sent — so the API sees exactly what a low/medium/high/
perfect student would actually retain, not the raw lecture.

Flow per student:
  1. BaseStudent.set_lecture_context(correct_lecture, use_blanking=True)
       → Markov masking applied locally (same as Unity backend would do)
  2. student.get_clean_context()
       → degraded text extracted
  3. POST /send_lecture  with degraded text + use_blanking=False
       → backend stores it as-is (masking already done, no double-masking)
  4. POST /answer_mcq  × 25
       → results collected and scored

Tier expectations:
  perfect  → ~90-100%   (full lecture, no loss)
  high     → ~70-85%    (minor lapses)
  medium   → ~45-60%    (noticeable gaps)
  low      → ~20-35%    (heavy loss, near chance)
"""

import sys
import random
import requests
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

try:
    from BaseStudent import BaseStudent, ATTENTION_PROFILES
except ImportError:
    print("ERROR: BaseStudent.py not found in current directory.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE_URL           = "http://127.0.0.1:5002"
CONFIDENCE_THRESH  = 0.25      # below this → client-side random guess
TOTAL              = 25        # number of exam questions
CHANCE_SCORE       = TOTAL / 4 # 6.25 — 4-option random baseline
PASS_THRESHOLD     = 15

# 3 students per tier, IDs 0-11
CLASSROOM = [
    # (student_id, attention_type)
    (0,  "low"),
    (1,  "low"),
    (2,  "low"),
    (3,  "medium"),
    (4,  "medium"),
    (5,  "medium"),
    (6,  "high"),
    (7,  "high"),
    (8,  "high"),
    (9,  "perfect"),
    (10, "perfect"),
    (11, "perfect"),
]

# Expected score ranges per tier (used in assertions)
SCORE_RANGES = {
    "low":     (0,  int(TOTAL * 0.40)),
    "medium":  (int(TOTAL * 0.35), int(TOTAL * 0.70)),
    "high":    (int(TOTAL * 0.60), TOTAL),
    "perfect": (int(TOTAL * 0.80), TOTAL),
}


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURE
# ══════════════════════════════════════════════════════════════════════════════

correct_lecture = """
Sustainable Food Systems and Nutrition: A Lecture

A food system encompasses everything involved in feeding a population — from
growing, harvesting, and processing food, to distributing, consuming, and
disposing of it. A sustainable food system meets current nutritional needs
without compromising the ability of future generations to meet theirs.
It balances three pillars: environmental health, economic viability, and
social equity.

MACRONUTRIENTS AND THEIR ROLES
The human body requires three macronutrients in large quantities. Proteins are
composed of amino acids, of which 9 are essential — meaning the body cannot
synthesize them and must obtain them through diet. Proteins are critical for
tissue repair, enzyme production, and immune function. Carbohydrates are the
body's primary and preferred energy source; they are broken down into glucose,
which fuels cellular respiration and ATP production. Dietary fibre, a type of
complex carbohydrate, is indigestible but feeds gut microbiota and regulates
bowel function. Fats serve as long-term energy stores, support cell membrane
integrity, and are required for the absorption of fat-soluble vitamins A, D,
E, and K.

THE FOOD ENVIRONMENT AND BIODIVERSITY
Soil health is foundational to sustainable agriculture. Healthy soil contains
organic matter, beneficial microorganisms, and adequate mineral content. The
practice of monoculture — growing a single crop repeatedly on the same land —
degrades soil, reduces biodiversity, and increases vulnerability to pests and
disease. Crop rotation and intercropping are strategies that restore nutrients
and support biodiversity. Agricultural biodiversity, meaning the variety of
crops and livestock breeds in use, is essential for food system resilience.
Loss of biodiversity narrows the genetic base of our food supply, increasing
the risk of large-scale crop failure.

FOOD SECURITY
Food security exists when all people, at all times, have physical, social, and
economic access to sufficient, safe, and nutritious food. The four pillars of
food security are: Availability (food is produced or imported in sufficient
quantities), Access (people can afford and physically reach food), Utilization
(the body can absorb and use nutrients from food effectively), and Stability
(access is consistent over time, not disrupted by shocks like drought or
conflict). Approximately 733 million people face hunger globally, driven by
poverty, conflict, climate change, and inequality — not a shortage of total
food production.

THE NITROGEN CYCLE AND FOOD PRODUCTION
Nitrogen is essential for plant growth, forming the backbone of amino acids
and nucleic acids. The nitrogen cycle converts atmospheric nitrogen into
forms usable by plants, primarily through nitrogen fixation by soil bacteria
such as Rhizobium, which form symbiotic relationships with legume roots.
Synthetic nitrogen fertilizers, developed via the Haber-Bosch process, have
dramatically increased crop yields but also cause significant environmental
harm. Excess nitrogen runs off into waterways, causing eutrophication — algal
blooms that deplete oxygen and kill aquatic life. This is a major negative
externality of industrial agriculture.

FOOD PROCESSING AND THE NOVA CLASSIFICATION
Not all food processing is harmful. The NOVA classification system groups
foods into four categories based on the extent and purpose of processing.
Group 1 consists of unprocessed or minimally processed foods such as fruits,
vegetables, eggs, and plain meat. Group 2 includes processed culinary
ingredients like oils, flour, and sugar. Group 3 covers processed foods such
as canned vegetables or artisan cheese. Group 4, ultra-processed foods,
includes products formulated from industrial ingredients with additives for
flavour, colour, and shelf life — such as packaged snacks, soft drinks, and
reconstituted meat products. High consumption of Group 4 foods is strongly
associated with obesity, type 2 diabetes, cardiovascular disease, and
poor diet quality.

GREENHOUSE GAS EMISSIONS AND FOOD
The global food system is responsible for approximately 26 to 34 percent of
total greenhouse gas emissions. Livestock farming is the single largest
contributor within the food sector, accounting for roughly 14.5 percent of
global emissions, primarily through methane from enteric fermentation in
ruminants and nitrous oxide from manure. Red meat — especially beef — has
the highest carbon footprint per kilogram of protein. Plant-based diets
have significantly lower emissions; shifting consumption toward legumes,
whole grains, fruits, and vegetables is considered one of the most impactful
individual actions for reducing food-related emissions. Food loss and waste
account for around 8 to 10 percent of global greenhouse gas emissions.

"""
correct_lecture = correct_lecture * 5

# ══════════════════════════════════════════════════════════════════════════════
#  EXAM QUESTIONS
# ══════════════════════════════════════════════════════════════════════════════

exam_questions = [
    {"text": "What are the three pillars of a sustainable food system?",
     "options": ["Environmental health, economic viability, and social equity",
                 "Destruction, packaging, and refrigeration",
                 "Production, transport, and consumption",
                 "Soil, water, and sunlight"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "How many essential amino acids must be obtained through diet?",
     "options": ["9", "4", "20", "12"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What is the body's primary and preferred energy source?",
     "options": ["Carbohydrates", "Fats", "Proteins", "Oxygen"],
     "correctIndex": 0, "corruptedIndex": 3},
    {"text": "What do dietary fibres primarily feed in the body?",
     "options": ["Gut microbiota", "The stomach lining",
                 "Fat-soluble vitamins", "Intestinal blockages"],
     "correctIndex": 0, "corruptedIndex": 3},
    {"text": "Which vitamins require fat for absorption?",
     "options": ["Vitamins A, D, E, and K", "Vitamins B and C",
                 "Vitamins B6 and B12 only", "All vitamins equally"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What agricultural practice involves growing a single crop repeatedly on the same land?",
     "options": ["Monoculture", "Intercropping", "Crop rotation", "Permaculture"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What are two strategies that restore soil nutrients and support biodiversity?",
     "options": ["Crop rotation and intercropping",
                 "Synthetic fertilizers and concrete planting",
                 "Monoculture and irrigation",
                 "Reverse rotting and carbon capture"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What are the four pillars of food security?",
     "options": ["Availability, Access, Utilization, and Stability",
                 "Availability, Accessibility, Utilization, and Stability",
                 "Production, Distribution, Consumption, and Waste",
                 "Quantity, Quality, Proximity, and Continuity"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "Approximately how many people globally face hunger?",
     "options": ["733 million", "1 billion", "200 million", "2 billion"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What is the primary driver of global hunger according to the lecture?",
     "options": ["Poverty, conflict, climate change, and inequality",
                 "An excess of agriculture",
                 "Lack of synthetic fertilizers",
                 "Insufficient food miles"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "Which bacteria form symbiotic relationships with legume roots to fix nitrogen?",
     "options": ["Rhizobium", "Protoxide", "E. coli", "Nitrosomonas"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What industrial process is used to produce synthetic nitrogen fertilizers?",
     "options": ["The Haber-Bosch process", "The Scarcity Dividend process",
                 "Bovine Carbon Capture", "Reverse Rotting"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "What environmental problem is caused by excess nitrogen runoff into waterways?",
     "options": ["Eutrophication", "Improved water flavour",
                 "Bovine Carbon Capture", "Soil compaction"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What does eutrophication cause in aquatic environments?",
     "options": ["Algal blooms that deplete oxygen and kill aquatic life",
                 "Increased fish populations",
                 "Improved water clarity",
                 "Nitrogen fixation underwater"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What does the NOVA classification system classify foods by?",
     "options": ["The extent and purpose of processing",
                 "Danger level from 1 (vegetables) to 4 (snacks)",
                 "Carbon footprint per kilogram",
                 "Amino acid content"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "Which NOVA group includes ultra-processed foods?",
     "options": ["Group 4", "Group 1", "Group 2", "Group 3"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "Which NOVA group includes unprocessed or minimally processed foods?",
     "options": ["Group 1", "Group 4", "Group 2", "Group 3"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "High consumption of ultra-processed foods is strongly associated with which conditions?",
     "options": ["Obesity, type 2 diabetes, and cardiovascular disease",
                 "Improved gut microbiota diversity",
                 "Increased essential amino acid intake",
                 "Better fat-soluble vitamin absorption"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What percentage of global greenhouse gas emissions does the food system account for?",
     "options": ["26 to 34 percent", "0 percent", "50 percent", "10 percent"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What is the single largest contributor to emissions within the food sector?",
     "options": ["Livestock farming", "Food transport",
                 "Crop irrigation", "Food packaging"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What percentage of global emissions does livestock farming account for?",
     "options": ["14.5 percent", "0 percent", "50 percent", "5 percent"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What is the primary greenhouse gas produced by ruminants through enteric fermentation?",
     "options": ["Methane", "Carbon dioxide", "Nitrous oxide", "Protoxide"],
     "correctIndex": 0, "corruptedIndex": 3},
    {"text": "Which food type has the highest carbon footprint per kilogram of protein?",
     "options": ["Beef", "Legumes", "Whole grains", "Vegetables"],
     "correctIndex": 0, "corruptedIndex": 0},
    {"text": "What percentage of global greenhouse gas emissions comes from food loss and waste?",
     "options": ["8 to 10 percent", "0 percent", "26 percent", "50 percent"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What dietary shift is considered one of the most impactful actions to reduce food emissions?",
     "options": ["Shifting toward legumes, whole grains, fruits, and vegetables",
                 "Eating exclusively beef",
                 "Eliminating all carbohydrates",
                 "Increasing ultra-processed food consumption"],
     "correctIndex": 0, "corruptedIndex": 1},
]

assert len(exam_questions) == TOTAL, f"Expected {TOTAL} questions, got {len(exam_questions)}"


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def bar(value: float, total: float, width: int = 20) -> str:
    filled = round(value / total * width) if total else 0
    return "█" * filled + "░" * (width - filled)


def simulate_masking(sid: int, attention_type: str, seed: int) -> Tuple[str, dict]:
    """
    Run the Markov attention model locally and return:
      - degraded_text : what the student actually retained
      - stats         : full BaseStudent statistics dict
    """
    random.seed(seed)
    student = BaseStudent(student_id=sid, attention_type=attention_type)
    student.set_lecture_context(correct_lecture, use_blanking=True)
    return student.get_clean_context(), student.get_statistics()


def send_lecture(sid: int, attention_type: str, degraded_text: str) -> bool:
    """
    Send the already-degraded context to the backend.
    use_blanking=False — masking was already applied locally, don't double-mask.
    """
    r = requests.post(f"{BASE_URL}/send_lecture", json={
        "student_id":     sid,
        "attention_type": attention_type,
        "lecture":        degraded_text,
        "use_blanking":   False,
    })
    return r.status_code == 200


def run_exam(sid: int) -> dict:
    correct = guessed = other = 0
    rows = []

    for i, q in enumerate(exam_questions, 1):
        r = requests.post(f"{BASE_URL}/answer_mcq", json={
            "student_id": sid,
            "question":   q["text"],
            "options":    q["options"],
        })
        if r.status_code != 200:
            rows.append({"q": i, "choice": -1, "correct": False,
                         "guessed": True, "conf": 0.0})
            continue

        res    = r.json()
        conf   = res["confidence"]
        no_ctx = res.get("no_context", False)

        # Client-side low-confidence gate (mirrors AutoTest logic)
        if conf < CONFIDENCE_THRESH or no_ctx:
            cidx    = random.randrange(len(q["options"]))
            guessed += 1
            tag     = f"[Guessed] (conf {conf:.2f})"
        else:
            cidx = res["choice_index"]

        is_correct = (cidx == q["correctIndex"])
        if is_correct and not (conf < CONFIDENCE_THRESH or no_ctx):
            correct += 1
        elif not is_correct and not (conf < CONFIDENCE_THRESH or no_ctx):
            other += 1

        mark = "[Correct]" if is_correct else ("[Guessed]" if (conf < CONFIDENCE_THRESH or no_ctx) else "[Wrong]  ")
        print(f"    Q{i:>2}: {mark} — {q['options'][cidx][:55]:<55}  (conf:{conf:.2f})")
        rows.append({"q": i, "choice": cidx, "correct": is_correct,
                     "guessed": conf < CONFIDENCE_THRESH or no_ctx, "conf": conf})

    # Recount properly from rows
    correct = sum(1 for r in rows if r["correct"] and not r["guessed"])
    guessed = sum(1 for r in rows if r["guessed"])
    other   = sum(1 for r in rows if not r["correct"] and not r["guessed"])

    return {"correct": correct, "guessed": guessed, "other": other, "rows": rows}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  CLASSROOM TEST — Markov Masking → Live API → MCQ Exam")
    print("  Lecture: Sustainable Food Systems and Nutrition")
    print(f"  Classroom: 3 low / 3 medium / 3 high / 3 perfect  (12 students)")
    print(f"  Confidence threshold: {CONFIDENCE_THRESH}  |  Pass: {PASS_THRESHOLD}/{TOTAL}")
    print("=" * 80)

    # ── Phase 1: Local Markov simulation ──────────────────────────────────────
    print("\n── PHASE 1: Markov Attention Simulation ────────────────────────────────────")
    print(f"  {'ID':<4} {'Type':<8} {'Tier'} {'Retained':>10} {'Blanked':>8} "
          f"{'Zoned':>7} {'Episodes':>9} {'AvgBurst':>9} {'Retention':>10}")
    print(f"  {'─'*4} {'─'*8} {'─'*4} {'─'*10} {'─'*8} {'─'*7} {'─'*9} {'─'*9} {'─'*10}")

    sim_results: Dict[int, dict] = {}
    for sid, atype in CLASSROOM:
        degraded, stats = simulate_masking(sid, atype, seed=sid * 7 + 42)
        sim_results[sid] = {
            "attention_type": atype,
            "degraded_text":  degraded,
            "stats":          stats,
        }
        s = stats
        print(f"  {sid:<4} {atype:<8} {s['tier']:<4} "
              f"{s['retained_words']:>4}/{s['original_words']:<5} "
              f"{s['blanked_sentences']:>3}/{s['total_sentences']:<4} "
              f"{s['zoned_words']:>5}w  "
              f"{s['zone_out_episodes']:>6}ep  "
              f"{s['avg_zoneout_length']:>7.1f}w  "
              f"{s['effective_retention']*100:>8.1f}%")

    # ── Phase 2: Send degraded lectures to API ────────────────────────────────
    print("\n── PHASE 2: Sending Degraded Lectures to API ───────────────────────────────")
    for sid, atype in CLASSROOM:
        degraded = sim_results[sid]["degraded_text"]
        word_count = len(degraded.split())
        ok = send_lecture(sid, atype, degraded)
        status = "[OK]  " if ok else "[FAIL]"
        print(f"  {status} Student {sid:<3} [{atype:<8}]  {word_count} words sent")
        if not ok:
            print(f"    ERROR: backend rejected lecture for student {sid}")
        time.sleep(0.3)

    # ── Phase 3: Run exams ────────────────────────────────────────────────────
    print("\n── PHASE 3: MCQ Exams ──────────────────────────────────────────────────────")
    exam_results: Dict[int, dict] = {}

    for sid, atype in CLASSROOM:
        tier = sim_results[sid]["stats"]["tier"]
        print(f"\n  {'─'*78}")
        print(f"  Student {sid} [{atype:<8} / Tier {tier}]")
        print(f"  {'─'*78}")
        time.sleep(0.5)
        exam_results[sid] = run_exam(sid)

    # ── Phase 4: Results by student ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  RESULTS BY STUDENT")
    print("=" * 80)

    tier_scores: Dict[str, List[int]] = {"A": [], "B": [], "C": []}

    for sid, atype in CLASSROOM:
        res   = exam_results[sid]
        stats = sim_results[sid]["stats"]
        tier  = stats["tier"]
        pct   = res["correct"] / TOTAL * 100
        grade = "PASS" if res["correct"] >= PASS_THRESHOLD else "FAIL"
        color = "[PASS]" if grade == "PASS" else "[FAIL]"

        tier_scores[tier].append(res["correct"])

        print(f"\n  Student {sid} [{atype:<8} / Tier {tier}]  "
              f"Retention: {stats['effective_retention']*100:.0f}%")
        print(f"    Score    : {res['correct']}/{TOTAL} ({pct:.0f}%)  {color} {grade}")
        print(f"    Correct  : {bar(res['correct'], TOTAL)}  {res['correct']}")
        print(f"    Guessed  : {bar(res['guessed'], TOTAL)}  {res['guessed']}")
        print(f"    Other    : {bar(res['other'],   TOTAL)}  {res['other']}")
        print(f"    {'─'*74}")
        print(f"    {'Q':>3}  {'Result':<10}  {'Question':<46}  Answer")
        print(f"    {'─'*3}  {'─'*10}  {'─'*46}  {'─'*30}")
        for row in res["rows"]:
            qi = row["q"] - 1
            q_text  = exam_questions[qi]["text"][:46]
            if row["choice"] < 0:
                chosen = "NO RESPONSE"
            else:
                chosen = exam_questions[qi]["options"][row["choice"]][:30]
            if row["guessed"]:
                tag = "[Guessed] "
            elif row["correct"]:
                tag = "[Correct] "
            else:
                tag = "[Wrong]   "
            print(f"    {row['q']:>3}  {tag:<10}  {q_text:<46}  {chosen}")

    # ── Phase 5: Tier summary ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  TIER SUMMARY")
    print("=" * 80)
    print(f"  {'Tier':<6} {'AType':<8} {'Scores':<22} {'Mean':>6} {'Min':>5} "
          f"{'Max':>5} {'Expected range':>16}")
    print(f"  {'─'*6} {'─'*8} {'─'*22} {'─'*6} {'─'*5} {'─'*5} {'─'*16}")

    tier_to_atype = {"C": "low", "B": "medium", "A": "high/perfect"}
    all_passed = True

    for tier in ["C", "B", "A"]:
        scores = tier_scores[tier]
        if not scores:
            continue
        mean_s = sum(scores) / len(scores)
        lo, hi = SCORE_RANGES.get(
            "low" if tier == "C" else "medium" if tier == "B" else "high", (0, TOTAL))
        scores_str = " ".join(str(s) for s in scores)
        in_range = lo <= mean_s <= hi
        flag = "[OK]  " if in_range else "[WARN] "
        if not in_range:
            all_passed = False
        print(f"  {tier:<6} {tier_to_atype[tier]:<8} {scores_str:<22} "
              f"{mean_s:>6.1f} {min(scores):>5} {max(scores):>5} "
              f"  {flag} [{lo}–{hi}]")

    # ── Phase 6: Ordering check ───────────────────────────────────────────────
    print("\n── ORDERING CHECK ──────────────────────────────────────────────────────────")
    mean_c = sum(tier_scores["C"]) / len(tier_scores["C"]) if tier_scores["C"] else 0
    mean_b = sum(tier_scores["B"]) / len(tier_scores["B"]) if tier_scores["B"] else 0
    mean_a = sum(tier_scores["A"]) / len(tier_scores["A"]) if tier_scores["A"] else 0

    check_cb = mean_c < mean_b
    check_ba = mean_b < mean_a
    print(f"  Tier C (low)    mean: {mean_c:.1f}/{TOTAL}")
    print(f"  Tier B (medium) mean: {mean_b:.1f}/{TOTAL}")
    print(f"  Tier A (hi/prf) mean: {mean_a:.1f}/{TOTAL}")
    print(f"  C < B : {'[OK]' if check_cb else '[BROKEN]'}")
    print(f"  B < A : {'[OK]' if check_ba else '[BROKEN]'}")

    if check_cb and check_ba:
        print("\n  [OK] Tier ordering correct — attention model is working end-to-end")
    else:
        print("\n  [WARN] Tier ordering violated — check Markov params or API context handling")
        all_passed = False

    # ── Phase 7: Masking assertions ───────────────────────────────────────────
    print("\n── MASKING ASSERTIONS ──────────────────────────────────────────────────────")

    @dataclass
    class Check:
        name:    str
        passed:  bool
        message: str = ""

    checks: List[Check] = []
    original_words = len(correct_lecture.split())

    for sid, atype in CLASSROOM:
        stats = sim_results[sid]["stats"]
        degraded = sim_results[sid]["degraded_text"]
        ratio = stats["effective_retention"]
        tier  = stats["tier"]

        # No mask tokens leaked into clean context
        has_tokens = any(t in degraded for t in ("[MASK]", "[BLANKED]", "[ZONED]", "[DISRUPTED]"))
        checks.append(Check(
            f"  S{sid} [{atype}] no mask tokens in degraded text",
            not has_tokens,
            "degraded text still contains mask tokens"))

        # Perfect students keep everything
        if atype == "perfect":
            checks.append(Check(
                f"  S{sid} [perfect] retention ≥ 95%",
                ratio >= 0.95,
                f"retention was {ratio*100:.1f}%"))

        # Non-perfect students lost something
        else:
            checks.append(Check(
                f"  S{sid} [{atype}] words were lost",
                stats["retained_words"] < original_words,
                f"retained {stats['retained_words']}/{original_words} — no loss"))

        # Tier field is correct
        expected_tier = ATTENTION_PROFILES[atype]["tier"]
        checks.append(Check(
            f"  S{sid} [{atype}] tier == {expected_tier}",
            tier == expected_tier,
            f"got tier={tier}"))

    passed = sum(1 for c in checks if c.passed)
    failed = sum(1 for c in checks if not c.passed)
    for c in checks:
        print(f"  {'[OK]  ' if c.passed else '[FAIL] '} {c.name}" +
              (f"\n      → {c.message}" if not c.passed else ""))

    # ── Final verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    total_checks = passed + failed
    if failed == 0 and check_cb and check_ba:
        print(f"  [ALL PASS] {total_checks} checks passed. "
              f"Markov model + API pipeline working correctly.")
    else:
        print(f"  [ISSUES]   {failed}/{total_checks} checks failed. "
              f"Review output above for details.")
    print(f"  Chance baseline: {CHANCE_SCORE:.1f}/{TOTAL} (25%)")
    print("=" * 80)


if __name__ == "__main__":
    main()