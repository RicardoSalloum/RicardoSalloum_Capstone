#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulationClassroom3.py
=======================
Union of studentSimulation2.py + AutoTest2.py — with an entirely unrelated lecture.

Simulates a 3/3/3/3 classroom (low / medium / high / perfect) using the
real Markov attention model from BaseStudent.py, then sends each student's
DEGRADED context to the live API and runs the full 25-question MCQ exam.

Lecture topic: Quantum Mechanics and Atomic Theory

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
#  LECTURE  (topic: Quantum Mechanics and Atomic Theory)
# ══════════════════════════════════════════════════════════════════════════════

correct_lecture = """
Quantum Mechanics and Atomic Theory: A Lecture

Quantum mechanics is the branch of physics that describes the behaviour of
matter and energy at the scale of atoms and subatomic particles. It replaced
classical Newtonian mechanics for phenomena at this scale, revealing a world
governed by probability, wave-particle duality, and discrete energy levels.

HISTORICAL DEVELOPMENT OF ATOMIC MODELS
Early models of the atom evolved dramatically over a century of experimentation.
John Dalton proposed the first modern atomic theory in 1803: atoms are
indivisible solid spheres, each element consists of identical atoms, and
chemical reactions are rearrangements of atoms. J.J. Thomson discovered the
electron in 1897 via cathode ray experiments and proposed the plum-pudding
model — a diffuse positive sphere with electrons embedded throughout.

Ernest Rutherford overturned this in 1911 with his gold foil experiment:
alpha particles fired at thin gold foil mostly passed through, but some
deflected at large angles. Rutherford concluded that atoms contain a tiny,
dense, positively charged nucleus surrounded by mostly empty space. Niels
Bohr refined this in 1913, proposing that electrons orbit the nucleus in
discrete energy levels, or shells, and emit or absorb photons of specific
energy when jumping between levels. The modern quantum mechanical model,
developed through the 1920s, replaces fixed orbits with probability clouds
called orbitals, described by the Schrödinger equation.

PLANCK'S QUANTUM THEORY AND THE PHOTOELECTRIC EFFECT
Max Planck resolved the ultraviolet catastrophe in 1900 by proposing that
energy is quantised — emitted and absorbed in discrete packets called quanta.
The energy of a quantum is E = hf, where h is Planck's constant
(6.626 × 10⁻³⁴ J·s) and f is the frequency of the radiation.

Albert Einstein extended this in 1905 to explain the photoelectric effect:
when light strikes a metal surface, electrons are ejected only if the light
exceeds a threshold frequency, regardless of intensity. Einstein proposed that
light itself travels in discrete packets called photons, each carrying energy
E = hf. This was radical because it showed light behaves as a particle, not
just a wave. Einstein received the Nobel Prize in Physics in 1921 for this work.

WAVE-PARTICLE DUALITY AND THE DE BROGLIE HYPOTHESIS
Louis de Broglie proposed in 1924 that matter also exhibits wave-like
properties. The de Broglie wavelength of a particle is λ = h / mv, where m
is mass and v is velocity. This means fast, heavy objects have imperceptibly
small wavelengths, but electrons have wavelengths comparable to atomic
distances. The wave nature of electrons was confirmed by the Davisson-Germer
experiment (1927), which observed electron diffraction — a wave phenomenon —
from a crystalline nickel surface.

THE HEISENBERG UNCERTAINTY PRINCIPLE
Werner Heisenberg formulated the uncertainty principle in 1927: it is
fundamentally impossible to simultaneously know both the exact position and
exact momentum of a particle. Mathematically, Δx · Δp ≥ ℏ/2, where ℏ is the
reduced Planck constant. This is not a limitation of measurement instruments —
it is an intrinsic feature of nature. The more precisely position is known, the
more uncertain the momentum, and vice versa. An analogous relationship holds for
energy and time: ΔE · Δt ≥ ℏ/2.

QUANTUM NUMBERS AND ELECTRON CONFIGURATION
The state of an electron in an atom is described by four quantum numbers.
The principal quantum number n (1, 2, 3, …) defines the energy level and
approximate distance from the nucleus. The angular momentum quantum number
ℓ (0 to n−1) defines the shape of the orbital: ℓ = 0 is an s orbital
(spherical), ℓ = 1 is a p orbital (dumbbell), ℓ = 2 is a d orbital. The
magnetic quantum number mₗ (from −ℓ to +ℓ) defines the orientation of the
orbital in space. The spin quantum number mₛ is either +½ or −½. The Pauli
exclusion principle states that no two electrons in the same atom can have
identical sets of all four quantum numbers, limiting each orbital to two
electrons of opposite spin.

RADIOACTIVE DECAY
Atomic nuclei that are unstable undergo radioactive decay to reach a more
stable configuration. Alpha decay emits a helium nucleus (2 protons + 2
neutrons), reducing atomic number by 2 and mass number by 4. Beta-minus decay
converts a neutron into a proton, emitting an electron and an antineutrino,
increasing atomic number by 1. Gamma decay emits high-energy photons without
changing atomic or mass number — it often follows alpha or beta decay. The
half-life of a radioactive isotope is the time required for half of a sample
to decay; it is constant and unaffected by temperature, pressure, or chemical
state. Carbon-14, with a half-life of approximately 5,730 years, is used in
radiocarbon dating to determine the age of organic materials.
"""


# ══════════════════════════════════════════════════════════════════════════════
#  EXAM QUESTIONS
# ══════════════════════════════════════════════════════════════════════════════

exam_questions = [
    {"text": "Who proposed the first modern atomic theory in 1803, describing atoms as indivisible solid spheres?",
     "options": ["John Dalton", "Ernest Rutherford", "Niels Bohr", "J.J. Thomson"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "What model of the atom did J.J. Thomson propose after discovering the electron?",
     "options": ["The plum-pudding model", "The nuclear model", "The Bohr shell model", "The probability cloud model"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What experiment led Rutherford to conclude atoms have a tiny dense nucleus?",
     "options": ["The gold foil experiment", "The cathode ray experiment", "The Davisson-Germer experiment", "The double-slit experiment"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What does Bohr's atomic model propose about electrons?",
     "options": ["Electrons orbit the nucleus in discrete energy levels", "Electrons are embedded in a diffuse positive sphere", "Electrons have no fixed position", "Electrons orbit randomly at any energy"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What does the modern quantum mechanical model use to describe electron positions?",
     "options": ["Probability clouds called orbitals", "Fixed circular orbits", "A positive diffuse sphere", "Discrete rings at set distances"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What was the problem that Planck's quantum theory resolved?",
     "options": ["The ultraviolet catastrophe", "The photoelectric threshold paradox", "The double-slit interference puzzle", "The cathode ray deflection problem"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "According to Planck, what is the energy of a quantum?",
     "options": ["E = hf, where h is Planck's constant and f is frequency", "E = mv², where m is mass and v is velocity", "E = mc², where m is mass and c is the speed of light", "E = λ/h, where λ is wavelength and h is Planck's constant"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "What phenomenon did Einstein explain using the concept of photons?",
     "options": ["The photoelectric effect", "Radioactive decay", "Electron diffraction", "The ultraviolet catastrophe"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "For what work did Einstein receive the Nobel Prize in Physics in 1921?",
     "options": ["Explaining the photoelectric effect", "Developing the theory of relativity", "Discovering the electron", "Formulating the uncertainty principle"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What does the de Broglie hypothesis state?",
     "options": ["Matter also exhibits wave-like properties", "Light travels exclusively as a wave", "Energy is always conserved in particle collisions", "Electrons orbit the nucleus at fixed radii"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What is the de Broglie wavelength formula?",
     "options": ["λ = h / mv", "λ = mv / h", "λ = hf", "λ = E / c"],
     "correctIndex": 0, "corruptedIndex": 3},
    {"text": "Which experiment confirmed the wave nature of electrons through diffraction?",
     "options": ["The Davisson-Germer experiment", "The gold foil experiment", "The cathode ray experiment", "The double-slit experiment with photons"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What does the Heisenberg uncertainty principle state?",
     "options": ["It is fundamentally impossible to simultaneously know both exact position and exact momentum of a particle", "Measurement instruments always introduce error into observations", "Energy cannot be created or destroyed", "Electrons cannot occupy the same orbital simultaneously"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What is the mathematical expression of the Heisenberg uncertainty principle for position and momentum?",
     "options": ["Δx · Δp ≥ ℏ/2", "Δx · Δp = 0", "Δx + Δp ≥ h", "Δx · Δp ≤ ℏ/2"],
     "correctIndex": 0, "corruptedIndex": 3},
    {"text": "What does the principal quantum number n define?",
     "options": ["The energy level and approximate distance from the nucleus", "The shape of the orbital", "The orientation of the orbital in space", "The spin direction of the electron"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What shape does an orbital with angular momentum quantum number ℓ = 0 have?",
     "options": ["Spherical (s orbital)", "Dumbbell-shaped (p orbital)", "Complex four-lobed (d orbital)", "Toroidal (f orbital)"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What are the two possible values of the spin quantum number mₛ?",
     "options": ["+½ or −½", "+1 or −1", "0 or 1", "+½ or 0"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What does the Pauli exclusion principle state?",
     "options": ["No two electrons in the same atom can have identical sets of all four quantum numbers", "No two atoms can occupy the same space simultaneously", "Electrons always fill the lowest energy orbital first", "An electron can only have one value of spin at a time"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "What is emitted during alpha decay?",
     "options": ["A helium nucleus (2 protons + 2 neutrons)", "An electron and an antineutrino", "A high-energy photon", "A neutron only"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What happens to atomic number during beta-minus decay?",
     "options": ["It increases by 1", "It decreases by 1", "It stays the same", "It decreases by 2"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "Which type of radioactive decay emits high-energy photons without changing atomic or mass number?",
     "options": ["Gamma decay", "Alpha decay", "Beta-minus decay", "Beta-plus decay"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What is the half-life of a radioactive isotope?",
     "options": ["The time required for half of a sample to decay", "The time for a sample to fully decay", "The energy released per decay event", "The rate of decay per unit mass"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What is the approximate half-life of Carbon-14?",
     "options": ["5,730 years", "1,600 years", "14,000 years", "730 years"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "What is radiocarbon dating used to determine?",
     "options": ["The age of organic materials", "The age of inorganic rocks", "The energy content of a fuel source", "The mass of a radioactive nucleus"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What is the value of Planck's constant h?",
     "options": ["6.626 × 10⁻³⁴ J·s", "3.00 × 10⁸ m/s", "1.602 × 10⁻¹⁹ C", "9.109 × 10⁻³¹ kg"],
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
    print("  Lecture: Quantum Mechanics and Atomic Theory")
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