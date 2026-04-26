#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulationClassroom5.py
=======================
Union of studentSimulation2.py + AutoTest2.py — with an entirely unrelated lecture.

Simulates a 3/3/3/3 classroom (low / medium / high / perfect) using the
real Markov attention model from BaseStudent.py, then sends each student's
DEGRADED context to the live API and runs the full 25-question MCQ exam.

Lecture topic: Genetics and Molecular Biology

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
#  LECTURE  (topic: Genetics and Molecular Biology)
# ══════════════════════════════════════════════════════════════════════════════

correct_lecture = """
Genetics and Molecular Biology: A Lecture

Genetics is the branch of biology concerned with heredity — how traits are
passed from parents to offspring — and variation, meaning the differences
between individuals. Molecular biology investigates the molecular mechanisms
underlying these processes, particularly the role of DNA, RNA, and proteins.

DNA STRUCTURE
The carrier of genetic information in all living organisms is
deoxyribonucleic acid, or DNA. Its structure was determined in 1953 by James
Watson and Francis Crick, building on X-ray crystallography data produced by
Rosalind Franklin. DNA is a double helix: two antiparallel strands wound
around each other, held together by hydrogen bonds between complementary
nitrogenous bases. The four bases are adenine (A), thymine (T), guanine (G),
and cytosine (C). Base pairing is specific: A pairs with T (two hydrogen bonds)
and G pairs with C (three hydrogen bonds). The backbone of each strand is
composed of alternating sugar (deoxyribose) and phosphate groups. The sequence
of bases along a strand encodes genetic information.

DNA REPLICATION
Before a cell divides, it must copy its DNA so that each daughter cell
receives a complete genome. Replication is semiconservative: each new double
helix consists of one original strand and one newly synthesised strand. The
process begins at origins of replication, where the enzyme helicase unwinds
and separates the two strands, forming a replication fork. Primase synthesises
a short RNA primer to provide a starting point. DNA polymerase III then adds
new nucleotides in the 5′ to 3′ direction, reading the template strand in the
3′ to 5′ direction. Because only one strand (the leading strand) can be
synthesised continuously, the other strand (the lagging strand) is built in
short segments called Okazaki fragments, later joined by DNA ligase. Proofreading
by DNA polymerase reduces the error rate to approximately one mistake per
billion base pairs copied.

THE CENTRAL DOGMA
Francis Crick formulated the central dogma of molecular biology in 1958:
genetic information flows from DNA to RNA to protein, and not in reverse
(with some exceptions such as retroviruses). Transcription is the first step:
RNA polymerase reads the template strand of DNA in the 3′ to 5′ direction and
synthesises a complementary messenger RNA (mRNA) strand in the 5′ to 3′
direction. In eukaryotes, the pre-mRNA is processed before leaving the nucleus:
a 5′ cap and a poly-A tail are added for stability and translation initiation,
and non-coding introns are spliced out, leaving only protein-coding exons.

Translation is the second step, occurring at ribosomes in the cytoplasm.
Transfer RNA (tRNA) molecules carry specific amino acids and possess a
three-base anticodon that pairs with the complementary codon on the mRNA.
The ribosome moves along the mRNA in triplet codons (each codon specifying one
amino acid) from the start codon AUG (methionine) to a stop codon (UAA, UAG,
or UGA). The chain of amino acids is then folded into a functional protein.
The genetic code is degenerate (multiple codons can encode the same amino
acid) but universal (the same code is used by virtually all organisms on Earth).

MENDELIAN GENETICS
Gregor Mendel, experimenting with pea plants in the 1860s, established the
foundational principles of inheritance. The law of segregation states that
each organism carries two alleles for each gene (one from each parent), and
these alleles separate during the formation of gametes so that each gamete
carries only one allele. The law of independent assortment states that alleles
of different genes are distributed to gametes independently of one another,
provided the genes are on different chromosomes. Dominant alleles mask the
expression of recessive alleles. An organism's genetic makeup is its genotype;
the observable characteristics resulting from the genotype are the phenotype.
A Punnett square is a grid used to predict the probability of offspring
genotypes and phenotypes from a given cross.

MUTATIONS
A mutation is a heritable change in the DNA sequence. Point mutations affect
a single nucleotide: a substitution replaces one base with another. A silent
mutation changes a codon but not the amino acid it encodes (due to degeneracy).
A missense mutation changes a codon to one encoding a different amino acid.
A nonsense mutation converts a codon to a stop codon, truncating the protein.
Insertions and deletions of nucleotides cause frameshift mutations, which
alter the reading frame of all codons downstream and typically disrupt the
protein completely. Mutations can be caused by errors in DNA replication,
exposure to mutagens such as ultraviolet radiation, or chemical agents.
Not all mutations are harmful; some are neutral and others are the raw
material for evolution by natural selection.

CRISPR-CAS9 GENE EDITING
CRISPR-Cas9 is a revolutionary gene-editing technology derived from a natural
bacterial immune system. The system uses a guide RNA (gRNA) designed to match
a specific DNA sequence in the genome. The Cas9 protein, guided by the gRNA,
binds to the matching DNA and makes a double-strand break at that precise
location. The cell's own repair mechanisms then either introduce small
insertions or deletions (disabling the gene via non-homologous end joining)
or incorporate a supplied DNA template with the desired sequence (via
homology-directed repair). CRISPR-Cas9 is faster, cheaper, and more precise
than previous gene-editing tools. It has applications in research, agriculture
(engineering disease-resistant crops), and medicine (potential therapies for
genetic diseases such as sickle-cell anaemia). Jennifer Doudna and Emmanuelle
Charpentier were awarded the Nobel Prize in Chemistry in 2020 for its
development.
"""


# ══════════════════════════════════════════════════════════════════════════════
#  EXAM QUESTIONS
# ══════════════════════════════════════════════════════════════════════════════

exam_questions = [
    {"text": "Who determined the double-helix structure of DNA in 1953?",
     "options": ["James Watson and Francis Crick", "Gregor Mendel and Charles Darwin", "Rosalind Franklin and Maurice Wilkins", "Jennifer Doudna and Emmanuelle Charpentier"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "What technique did Rosalind Franklin use to produce data that was critical to determining DNA's structure?",
     "options": ["X-ray crystallography", "Electron microscopy", "Gel electrophoresis", "Polymerase chain reaction"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "Which nitrogenous bases pair together in DNA?",
     "options": ["Adenine with thymine, and guanine with cytosine", "Adenine with cytosine, and guanine with thymine", "Adenine with uracil, and guanine with cytosine", "Adenine with guanine, and thymine with cytosine"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "How many hydrogen bonds form between guanine and cytosine?",
     "options": ["Three", "Two", "One", "Four"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What is the backbone of each DNA strand composed of?",
     "options": ["Alternating sugar (deoxyribose) and phosphate groups", "Alternating nitrogenous bases and amino acids", "Alternating ribose and phosphate groups", "Alternating codons and anticodons"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "What does semiconservative replication mean?",
     "options": ["Each new double helix consists of one original strand and one newly synthesised strand", "Both strands of the original DNA are preserved and two completely new helices are made", "One entirely new double helix is made while the original is discarded", "DNA is copied only in the region currently being transcribed"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What enzyme unwinds and separates the two DNA strands at the replication fork?",
     "options": ["Helicase", "DNA polymerase III", "Primase", "DNA ligase"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "In which direction does DNA polymerase synthesise new DNA?",
     "options": ["5′ to 3′", "3′ to 5′", "Both directions simultaneously", "Direction depends on which strand is being copied"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What are Okazaki fragments?",
     "options": ["Short DNA segments synthesised on the lagging strand during replication", "Small RNA primers used to initiate leading-strand synthesis", "Non-coding sequences spliced out during mRNA processing", "Segments of DNA cleaved by the Cas9 protein"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "What is the approximate error rate of DNA replication after proofreading?",
     "options": ["One mistake per billion base pairs", "One mistake per million base pairs", "One mistake per thousand base pairs", "One mistake per ten thousand base pairs"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What does the central dogma of molecular biology state?",
     "options": ["Genetic information flows from DNA to RNA to protein", "Genetic information flows from protein to RNA to DNA", "Genes are inherited in pairs, one from each parent", "All cells in an organism contain identical DNA"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "Which enzyme carries out transcription?",
     "options": ["RNA polymerase", "DNA polymerase III", "Helicase", "Ribosome"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "What processing steps occur on pre-mRNA before it leaves the nucleus?",
     "options": ["Addition of a 5′ cap and poly-A tail, and splicing out of introns", "Addition of a stop codon and removal of exons", "Addition of tRNA anticodons and removal of promoters", "Methylation of the coding sequence only"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What is the start codon for translation, and which amino acid does it encode?",
     "options": ["AUG, which encodes methionine", "UAA, which encodes alanine", "GCU, which encodes glycine", "UGA, which encodes tryptophan"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What does 'degenerate' mean in the context of the genetic code?",
     "options": ["Multiple codons can encode the same amino acid", "Some codons do not encode any amino acid", "The code differs between different species", "A single codon can encode multiple amino acids"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "What does Mendel's law of segregation state?",
     "options": ["Two alleles for each gene separate during gamete formation so each gamete carries only one allele", "Alleles of different genes are distributed to gametes independently of each other", "Dominant alleles always completely mask recessive ones", "Genes are inherited in groups linked to the same chromosome"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What is the difference between genotype and phenotype?",
     "options": ["Genotype is the genetic makeup; phenotype is the observable characteristics", "Genotype is the observable trait; phenotype is the underlying gene sequence", "Genotype refers to dominant alleles only; phenotype includes all alleles", "Genotype is inherited; phenotype is entirely determined by environment"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What is a missense mutation?",
     "options": ["A substitution that changes a codon to one encoding a different amino acid", "A substitution that changes a codon but not the amino acid it encodes", "A substitution that converts a codon to a stop codon", "An insertion or deletion that shifts the reading frame"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What is a nonsense mutation?",
     "options": ["A substitution that converts an amino acid codon to a stop codon, truncating the protein", "A mutation with no effect on the protein sequence", "An insertion that extends the protein beyond its normal stop codon", "A deletion of an entire exon from the gene"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "Why do frameshift mutations typically disrupt a protein completely?",
     "options": ["They alter the reading frame of all codons downstream, changing every amino acid after the mutation", "They always introduce a stop codon immediately after the mutation site", "They prevent RNA polymerase from transcribing past the mutation", "They change only one amino acid but that amino acid is always critical"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What component of the CRISPR-Cas9 system directs Cas9 to the correct DNA location?",
     "options": ["A guide RNA (gRNA) designed to match a specific DNA sequence", "A restriction enzyme that recognises a specific cut site", "A DNA template that is inserted at the target location", "A promoter sequence that activates the Cas9 gene"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What does Cas9 do when it binds to the target DNA?",
     "options": ["It makes a double-strand break at that precise location", "It methylates the target sequence to silence gene expression", "It transcribes the target gene into mRNA", "It copies the target sequence into a new location in the genome"],
     "correctIndex": 0, "corruptedIndex": 2},
    {"text": "Which repair pathway can incorporate a supplied DNA template to make a specific edit?",
     "options": ["Homology-directed repair", "Non-homologous end joining", "Base excision repair", "Mismatch repair"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "Who received the Nobel Prize in Chemistry in 2020 for developing CRISPR-Cas9?",
     "options": ["Jennifer Doudna and Emmanuelle Charpentier", "James Watson and Francis Crick", "Gregor Mendel and Rosalind Franklin", "Frederick Sanger and Kary Mullis"],
     "correctIndex": 0, "corruptedIndex": 1},
    {"text": "What is a silent mutation?",
     "options": ["A base substitution that changes a codon but not the amino acid it encodes, due to degeneracy of the genetic code", "A mutation that occurs in a non-coding region and has no phenotypic effect", "A mutation inherited from one parent that is masked by the dominant allele from the other", "A deletion that removes only non-essential intron sequences"],
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
    print("  Lecture: Genetics and Molecular Biology")
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