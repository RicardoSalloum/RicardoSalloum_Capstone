#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulationClassroom4.py
=======================
Union of studentSimulation2.py + AutoTest2.py — with an entirely unrelated lecture.

Simulates a 3/3/3/3 classroom (low / medium / high / perfect) using the
real Markov attention model from BaseStudent.py, then sends each student's
DEGRADED context to the live API and runs the full 25-question MCQ exam.

Lecture topic: Psychology of Memory, Learning, and Behaviour

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
#  LECTURE  (topic: Psychology of Memory, Learning, and Behaviour)
# ══════════════════════════════════════════════════════════════════════════════

correct_lecture = """
Psychology of Memory, Learning, and Behaviour: A Lecture

Psychology is the scientific study of mind and behaviour. It seeks to
understand how people perceive, think, feel, and act — both as individuals
and in groups. The field draws on biology, neuroscience, philosophy, and
social science to explain the causes and consequences of human experience.

CLASSICAL CONDITIONING
Classical conditioning, discovered by Ivan Pavlov in experiments with dogs,
is a form of associative learning in which a neutral stimulus becomes associated
with a stimulus that naturally produces a response. In Pavlov's experiments, the
sound of a bell (neutral stimulus) was repeatedly paired with food (unconditioned
stimulus), which naturally caused salivation (unconditioned response). After
repeated pairings, the bell alone elicited salivation — now called the conditioned
response. Key processes include acquisition (the initial learning of the
association), extinction (the weakening of the conditioned response when the
conditioned stimulus is presented without the unconditioned stimulus), and
spontaneous recovery (the reappearance of the conditioned response after a
rest period following extinction).

OPERANT CONDITIONING
B.F. Skinner developed operant conditioning, in which behaviour is shaped by
its consequences. Reinforcement increases the likelihood of a behaviour being
repeated. Positive reinforcement adds a desirable stimulus (e.g., a reward
after correct behaviour). Negative reinforcement removes an unpleasant stimulus
(e.g., taking painkillers to stop pain — the relief reinforces the behaviour).
Punishment decreases the likelihood of behaviour: positive punishment adds an
aversive stimulus; negative punishment removes a desirable one. Skinner used
a Skinner box — an operant conditioning chamber — to study these effects
systematically in rats and pigeons. Fixed-ratio schedules (reward after every
nth response) produce high, steady rates of responding; variable-ratio schedules
(reward after unpredictable numbers of responses) produce the highest and most
resistant-to-extinction response rates, which explains the addictiveness of
slot machines and social media notifications.

MEMORY SYSTEMS
Atkinson and Shiffrin's multi-store model proposes that memory consists of
three components. Sensory memory holds raw perceptual input for a fraction of
a second — iconic memory for vision (about 0.5 seconds) and echoic memory for
sound (about 3–4 seconds). Short-term memory (STM), or working memory in
Baddeley's more refined model, holds approximately 7 ± 2 items (Miller's
Law) for up to about 30 seconds without rehearsal. Long-term memory (LTM)
has essentially unlimited capacity and duration. LTM divides into explicit
(declarative) memory — which includes episodic memory (personal experiences
and autobiographical events) and semantic memory (general knowledge and
facts) — and implicit (non-declarative) memory, which includes procedural
memory (skills and habits) and conditioned responses.

ENCODING, STORAGE, AND RETRIEVAL
Memory formation involves three stages. Encoding converts sensory input into
a memory trace; elaborative encoding (linking new information to existing
knowledge) is more effective than maintenance rehearsal (mere repetition).
Storage retains the encoded information, consolidated from short-term to
long-term memory partly during sleep. Retrieval is the process of accessing
stored memories. The encoding specificity principle states that retrieval is
best when the context at retrieval matches the context at encoding —
explaining why people recall information better in the same room where they
learned it. Retrieval failure, not storage failure, is the most common cause
of forgetting; the tip-of-the-tongue phenomenon illustrates information being
stored but temporarily inaccessible.

COGNITIVE BIASES AND HEURISTICS
People frequently use mental shortcuts called heuristics, which are efficient
but can produce systematic errors known as cognitive biases. The availability
heuristic leads people to judge the likelihood of events by how easily examples
come to mind — causing overestimation of dramatic but rare events (plane
crashes) and underestimation of common but mundane ones (car accidents).
The representativeness heuristic involves judging probability based on how
well something resembles a prototype, often ignoring base rates. Confirmation
bias is the tendency to search for, interpret, and recall information in a way
that confirms pre-existing beliefs. The anchoring effect occurs when initial
information disproportionately influences subsequent judgements.

PIAGET'S STAGES OF COGNITIVE DEVELOPMENT
Jean Piaget proposed that children develop through four qualitatively distinct
stages. The sensorimotor stage (birth to 2 years) is characterised by learning
through sensory experience and physical action; object permanence — understanding
that objects continue to exist when out of sight — develops in this stage.
The preoperational stage (2–7 years) involves symbolic thinking and language
but lacks logical operations; children are egocentric, unable to take others'
perspectives. The concrete operational stage (7–11 years) brings logical
thinking about concrete events; conservation — understanding that quantity
does not change despite changes in shape or appearance — is mastered.
The formal operational stage (12 years and beyond) enables abstract and
hypothetical reasoning.

NEUROPLASTICITY AND LEARNING
The brain's ability to reorganise itself by forming new neural connections
in response to experience is called neuroplasticity. Hebb's rule summarises
synaptic learning: neurons that fire together wire together — repeated
co-activation strengthens the synaptic connection between neurons. Long-term
potentiation (LTP) is the cellular mechanism underlying this strengthening and
is considered the physiological basis of learning and memory. Sleep plays a
critical role: during slow-wave sleep, the hippocampus replays daytime
experiences, consolidating them into long-term cortical storage.
"""


# ══════════════════════════════════════════════════════════════════════════════
#  EXAM QUESTIONS
# ══════════════════════════════════════════════════════════════════════════════

exam_questions = [
    {"text": "Who discovered classical conditioning through experiments with dogs?",
     "options": ["Ivan Pavlov", "B.F. Skinner", "Jean Piaget", "William James"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "In classical conditioning, what is the conditioned response?",
     "options": ["A learned response to the conditioned stimulus alone", "The natural response to the unconditioned stimulus", "A punishment that reduces unwanted behaviour", "The spontaneous recovery of a reflex"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What is extinction in classical conditioning?",
     "options": ["The weakening of the conditioned response when the conditioned stimulus is presented without the unconditioned stimulus", "The complete loss of all learned associations permanently", "The transfer of conditioning to a new stimulus", "The strengthening of the conditioned response over time"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What is spontaneous recovery?",
     "options": ["The reappearance of a conditioned response after a rest period following extinction", "The permanent loss of an unconditioned response", "Learning a new response to replace an extinguished one", "A reflex appearing without any conditioning"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "Who developed operant conditioning?",
     "options": ["B.F. Skinner", "Ivan Pavlov", "Jean Piaget", "Atkinson and Shiffrin"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What is positive reinforcement?",
     "options": ["Adding a desirable stimulus to increase the likelihood of a behaviour being repeated", "Removing an unpleasant stimulus to increase behaviour", "Adding an aversive stimulus to decrease behaviour", "Removing a desirable stimulus to decrease behaviour"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "Which reinforcement schedule produces the highest and most resistant-to-extinction response rates?",
     "options": ["Variable-ratio schedule", "Fixed-ratio schedule", "Fixed-interval schedule", "Variable-interval schedule"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What does Miller's Law state about short-term memory capacity?",
     "options": ["Short-term memory holds approximately 7 ± 2 items", "Short-term memory holds approximately 20 items", "Short-term memory has unlimited capacity", "Short-term memory holds items for up to 10 minutes"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "What is the duration of iconic (visual) sensory memory?",
     "options": ["About 0.5 seconds", "About 3–4 seconds", "About 30 seconds", "About 2 minutes"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "Which type of long-term memory stores personal experiences and autobiographical events?",
     "options": ["Episodic memory", "Semantic memory", "Procedural memory", "Implicit memory"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "Which type of long-term memory stores skills and habits?",
     "options": ["Procedural memory", "Episodic memory", "Semantic memory", "Echoic memory"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "What does the encoding specificity principle state?",
     "options": ["Retrieval is best when the context at retrieval matches the context at encoding", "Information encoded during sleep is retained best", "Elaborative encoding is always superior to any other form", "Repetition is the most reliable way to encode information"],
     "correctIndex": 0, "corruptedIndex": 3},
    {"text": "What is the most common cause of forgetting?",
     "options": ["Retrieval failure, not storage failure", "Complete erasure from long-term storage", "Sensory memory decay within seconds", "Interference from procedural memory"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What cognitive bias causes people to overestimate dramatic but rare events?",
     "options": ["The availability heuristic", "Confirmation bias", "The anchoring effect", "The representativeness heuristic"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What is confirmation bias?",
     "options": ["The tendency to search for and interpret information in a way that confirms pre-existing beliefs", "Judging probability based on how well something matches a prototype", "Over-relying on the first piece of information encountered", "Assuming others share one's own perspective"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "What is the anchoring effect?",
     "options": ["Initial information disproportionately influencing subsequent judgements", "Recalling recent events more vividly than older ones", "Attributing outcomes to external rather than internal causes", "Overestimating one's ability to predict past events"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What cognitive ability develops during Piaget's sensorimotor stage?",
     "options": ["Object permanence", "Conservation of quantity", "Hypothetical reasoning", "Perspective-taking"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What cognitive characteristic defines Piaget's preoperational stage?",
     "options": ["Egocentrism — inability to take others' perspectives", "Mastery of conservation", "Abstract and hypothetical reasoning", "Understanding of object permanence"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "At what age range does the concrete operational stage occur, according to Piaget?",
     "options": ["7–11 years", "2–7 years", "Birth to 2 years", "12 years and beyond"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What cognitive skill is mastered during the concrete operational stage?",
     "options": ["Conservation — understanding that quantity does not change despite changes in appearance", "Object permanence", "Abstract reasoning about hypothetical situations", "Language acquisition and symbolic thinking"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "What is neuroplasticity?",
     "options": ["The brain's ability to reorganise itself by forming new neural connections in response to experience", "The fixed hardwiring of neural circuits after early childhood", "The loss of neurons due to ageing or injury", "The physical increase in brain size during adolescence"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What does Hebb's rule state?",
     "options": ["Neurons that fire together wire together", "Neurons that fire apart grow apart", "Synaptic connections weaken with repeated use", "Learning requires the formation of entirely new neurons"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "What is long-term potentiation (LTP)?",
     "options": ["The strengthening of a synaptic connection through repeated co-activation, considered the cellular basis of learning", "The permanent suppression of a neural pathway after punishment", "The gradual weakening of unused synapses over time", "A form of sensory memory lasting several seconds"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "What role does sleep play in memory consolidation?",
     "options": ["During slow-wave sleep, the hippocampus replays daytime experiences, consolidating them into long-term cortical storage", "Sleep erases short-term memory to free capacity for new learning", "REM sleep transfers semantic memories to procedural storage", "Sleep has no proven role in memory consolidation"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What is elaborative encoding?",
     "options": ["Linking new information to existing knowledge, making encoding more effective than simple repetition", "Repeating information aloud until it is memorised", "Writing information down to reinforce visual memory", "Encoding information across multiple sensory channels simultaneously"],
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
    print("  Lecture: Psychology of Memory, Learning, and Behaviour")
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