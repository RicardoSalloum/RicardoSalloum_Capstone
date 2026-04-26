#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulationClassroomUnrelated.py
================================
Union of studentSimulation2.py + AutoTest2.py — with an entirely unrelated lecture.

Simulates a 3/3/3/3 classroom (low / medium / high / perfect) using the
real Markov attention model from BaseStudent.py, then sends each student's
DEGRADED context to the live API and runs the full 25-question MCQ exam.

Lecture topic: The Roman Republic and Empire — Politics, Society, and Legacy

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
#  LECTURE  (topic: The Roman Republic and Empire)
# ══════════════════════════════════════════════════════════════════════════════

correct_lecture = """
The Roman Republic and Empire: Politics, Society, and Legacy

Rome's rise from a small city-state on the banks of the Tiber River to the
master of the Mediterranean world is one of history's most remarkable stories.
The city, according to Roman tradition, was founded in 753 BCE. Its political
evolution passed through three broad phases: the Monarchy (753–509 BCE), the
Republic (509–27 BCE), and the Empire (27 BCE–476 CE in the West).

THE ROMAN REPUBLIC: STRUCTURE AND GOVERNANCE
The Roman Republic was governed by a complex system of elected magistrates,
deliberative bodies, and unwritten constitutional norms. At its heart sat the
Senate — a body of roughly three hundred patrician and later plebeian elders —
which controlled finances, foreign policy, and provincial administration.
Executive power was held by two consuls elected annually, each possessing
imperium, the legal authority to command armies and enforce law. This dual
consulship was designed explicitly to prevent any individual from accumulating
unchecked power; each consul held the right of veto, known as intercessio,
over the other's decisions.

Below the consuls, a hierarchy of magistracies existed: praetors administered
justice and commanded legions; quaestors managed state finances; censors
conducted the census and managed public morality; and aediles oversaw public
buildings, games, and grain supply. In times of acute crisis, the Senate could
appoint a dictator — a single magistrate invested with supreme power — for a
maximum term of six months.

The struggle between the patricians (the hereditary aristocracy) and the
plebeians (the common citizens) defined much of the Republic's early social
history. Through a series of conflicts known as the Conflict of the Orders
(494–287 BCE), plebeians gradually won the right to hold high offices, have
their own tribunes with the power to veto Senate legislation, and see their
laws recorded in the Twelve Tables (450 BCE) — Rome's first written legal code.

THE ROMAN LEGION AND MILITARY EXPANSION
Rome's military was the engine of its expansion. The fundamental unit was the
legion, comprising approximately five thousand to six thousand heavy infantry
soldiers called legionaries. Legionaries were Roman citizens who supplied their
own equipment; later reforms by Gaius Marius in 107 BCE opened the legions to
the landless poor and created a professional standing army loyal to its general
rather than to the state. The Marian reforms are widely considered a key factor
in the eventual collapse of the Republic, as ambitious commanders such as Sulla,
Pompey, and Julius Caesar could now use loyal personal armies to pursue
political ends.

Rome conquered the Italian peninsula by 265 BCE, then fought the three Punic
Wars against Carthage (264–146 BCE). The Second Punic War (218–201 BCE) saw
the Carthaginian general Hannibal Barca cross the Alps with war elephants and
devastate Roman forces at the Battle of Cannae in 216 BCE — one of the worst
military defeats in Roman history, with an estimated fifty thousand soldiers
killed in a single day. Despite this catastrophe Rome recovered, ultimately
defeating Carthage and razing the city to the ground in 146 BCE.

THE FALL OF THE REPUBLIC
The late Republic was torn apart by social conflict, slave revolts, and the
ambitions of powerful generals. The Gracchi brothers — Tiberius (tribune 133
BCE) and Gaius (tribune 123–122 BCE) — attempted land reform to address
the displacement of small farmers by wealthy landowners using slave labour from
conquered territories. Both were assassinated. Their deaths inaugurated a
century of political violence.

The Social War (91–87 BCE) erupted when Rome's Italian allies demanded
citizenship; Rome granted it to end the war, transforming the citizen body
enormously. The First Triumvirate (60 BCE) united Julius Caesar, Pompey, and
Crassus in an unofficial alliance. After Crassus died at the Battle of Carrhae
(53 BCE), Caesar's conquest of Gaul (58–50 BCE) made him immensely popular
and wealthy. When the Senate ordered him to disband his army, Caesar crossed
the Rubicon River in 49 BCE — a legally forbidden act — triggering civil war.
Caesar defeated Pompey, became dictator perpetuo (dictator in perpetuity), and
was assassinated on the Ides of March (15 March) 44 BCE by a group of senators
led by Brutus and Cassius, who feared the restoration of monarchy.

THE PRINCIPATE: AUGUSTUS AND THE EARLY EMPIRE
The chaos following Caesar's death ended when his adopted heir Octavian
defeated Mark Antony and Cleopatra at the Battle of Actium in 31 BCE.
The Senate awarded Octavian the honorific title Augustus in 27 BCE, marking
the start of the Principate — a political system that preserved the outward
forms of the Republic while concentrating real power in the hands of a single
ruler called the princeps (first citizen). Augustus held tribunician power,
proconsular imperium, and the title pontifex maximus (chief priest) simultaneously,
making him effectively an autocrat. He ruled for forty-one years, the longest
reign of any Roman emperor.

The Julio-Claudian dynasty (27 BCE–68 CE) included Augustus, Tiberius,
Caligula, Claudius, and Nero. It was followed by the Year of the Four Emperors
(69 CE), the Flavian dynasty, and then the so-called Five Good Emperors of the
Antonine dynasty: Nerva, Trajan, Hadrian, Antoninus Pius, and Marcus Aurelius.
Under Trajan (98–117 CE), the Empire reached its maximum territorial extent,
stretching from Britain in the northwest to Mesopotamia in the east.

ROMAN SOCIETY, LAW, AND LEGACY
Roman society was stratified by legal status, class, and gender. At the apex
stood Roman citizens, subdivided into the senatorial order, the equestrian
order, and the plebs. Below citizens were freed slaves (liberti) and, at the
bottom, enslaved people, who could constitute up to thirty percent of the
population in some regions. Slavery was integral to the Roman economy and
household. The institution of the paterfamilias — the male head of household —
held legal power of life and death over family members and slaves alike.

Roman law is perhaps Rome's most enduring legacy. Concepts such as the
presumption of innocence, the right to face one's accuser, and the distinction
between public and private law originated in Roman legal tradition. The
codification of Roman law under Emperor Justinian I in the 6th century CE —
the Corpus Juris Civilis — became the foundation of civil law systems across
continental Europe and Latin America.

Rome also left indelible marks on language, architecture, and religion. Latin
evolved into the Romance languages: Italian, French, Spanish, Portuguese, and
Romanian. Roman engineering produced the aqueduct system, concrete construction,
and the arch-and-vault architectural tradition. The adoption of Christianity as
the state religion under Emperor Constantine I (Edict of Milan, 313 CE) and its
formal establishment under Theodosius I in 380 CE reshaped the spiritual
landscape of the Western world permanently.

THE FALL OF THE WESTERN EMPIRE
Historians debate the causes of Rome's fall, but key factors include military
overextension, economic strain from debasement of currency and heavy taxation,
political instability with dozens of emperors in the third century alone, and
increasing pressure from migrating peoples along the frontiers. The Visigoths
sacked Rome in 410 CE — an event that shocked the ancient world — and in
476 CE the Germanic chieftain Odoacer deposed the last Western emperor,
Romulus Augustulus, conventionally marking the end of the Western Roman Empire.
The Eastern Empire, known as the Byzantine Empire, survived for nearly another
thousand years until the fall of Constantinople in 1453 CE.
"""


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
    print("  Lecture: The Roman Republic and Empire — Politics, Society, and Legacy")
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