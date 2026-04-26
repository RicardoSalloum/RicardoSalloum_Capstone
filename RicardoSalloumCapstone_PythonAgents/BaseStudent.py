# -*- coding: utf-8 -*-
"""
BaseStudent.py - Core student simulation with attention modeling

ATTENTION MODEL: Two-state Markov chain
=======================================
Each student has exactly two attention states: ATTENTIVE and ZONED_OUT.
On every word the student either stays in their current state or switches.

    ATTENTIVE  --p_dropout-->  ZONED_OUT
    ZONED_OUT  --p_recover-->  ATTENTIVE

Why this works:
  - A zone-out affects a coherent burst of words, not random isolated ones —
    just like real attention loss.
  - Concepts mentioned multiple times get multiple independent chances to
    be captured, which correctly benefits weaker students more in relative terms.

Steady-state retention formula:
    P(attentive) = p_recover / (p_dropout + p_recover)

Expected effective retention by tier:
    perfect: ~100%    high: ~80-88%    medium: ~48-58%    low: ~16-26%
    Ranges do NOT overlap — tier separation is built into the math.
"""

import random
import re
from pydantic import BaseModel
from typing import List


# ==================== PYDANTIC MODELS ====================
# These are used by FastAPI in Agent.py to validate incoming request bodies

class LectureInput(BaseModel):
    lecture: str
    student_id: int
    attention_type: str
    use_blanking: bool = True   # whether sentence-level dropout is enabled

class QuestionInput(BaseModel):
    studentName: str
    studentID: int
    question: str

class MCQInput(BaseModel):
    student_id: int
    question: str
    options: List[str]


# ==================== ATTENTION PROFILES ====================
# Each profile controls how often a student zones out and how quickly they recover.
#
# Steady-state P(attentive) = p_recover / (p_dropout + p_recover):
#   perfect:  100%   (never zones out)
#   high:     0.35 / 0.38 = 92%  → ~83% after blanking
#   medium:   0.18 / 0.26 = 69%  → ~52% after blanking
#   low:      0.08 / 0.23 = 35%  → ~19% after blanking
#
# Average zone-out burst length = 1 / p_recover (words):
#   perfect:  0 words    (never zones out)
#   high:     2.9 words  (brief lapses)
#   medium:   5.6 words  (noticeable gaps)
#   low:     12.5 words  (long coherent losses)

ATTENTION_PROFILES = {
    "perfect": {
        "p_dropout":  0.000,   # probability of entering zone-out per word
        "p_recover":  1.000,   # probability of recovering per word while zoned out
        "blank_prob": 0.000,   # probability of blanking an entire sentence
        "tier": "A"
    },
    "high": {
        "p_dropout":  0.035,
        "p_recover":  0.700,
        "blank_prob": 0.025,
        "tier": "A"
    },
    "medium": {
        "p_dropout":  0.100,
        "p_recover":  0.340,
        "blank_prob": 0.250,
        "tier": "B"
    },
    "low": {
        "p_dropout":  0.190,
        "p_recover":  0.220,
        "blank_prob": 0.390,
        "tier": "C"
    }
}


# ==================== BASE STUDENT CLASS ====================

class BaseStudent:
    """
    Simulates one student's lecture experience through two degradation passes:

    Pass 1 — Sentence blanking (_apply_blanking):
        Randomly drops entire sentences and replaces them with [BLANKED].
        Simulates a student being completely distracted during a passage.

    Pass 2 — Markov word masking (_apply_markov_mask):
        Runs a two-state machine word-by-word.
        While ATTENTIVE words are kept; while ZONED_OUT words become [ZONED].

    The clean context (everything that survived) is what the LLM gets via
    get_clean_context(). No extra information is ever injected — what the
    student missed is genuinely missing from their knowledge.
    """

    def __init__(self, student_id: int, attention_type: str):
        self.student_id     = student_id
        self.attention_type = attention_type.lower()

        # Look up the profile; default to "medium" if the type is unrecognised
        profile = ATTENTION_PROFILES.get(self.attention_type, ATTENTION_PROFILES["medium"])
        self.p_dropout  = profile["p_dropout"]
        self.p_recover  = profile["p_recover"]
        self.blank_prob = profile["blank_prob"]
        self.tier       = profile["tier"]

        # The masked context stored after processing a lecture
        self.context     = ""
        self.raw_lecture = ""

        # Counters used to build the statistics report
        self.blanked_sentences = 0
        self.zoned_words       = 0
        self.total_sentences   = 0
        self.total_words       = 0
        self.zone_out_episodes = 0

    # ==================== LECTURE PROCESSING ====================

    def set_lecture_context(self, lecture_text: str, use_blanking: bool = True):
        """
        Entry point called by Agent.py when a lecture is sent to a student.
        Runs both degradation passes and stores the result in self.context.
        """
        self.raw_lecture       = lecture_text
        self.blanked_sentences = 0
        self.zoned_words       = 0
        self.total_sentences   = 0
        self.total_words       = 0
        self.zone_out_episodes = 0

        # Pass 1: sentence-level blanking (only if enabled and the tier has blanking)
        if use_blanking and self.blank_prob > 0:
            lecture_text = self._apply_blanking(lecture_text)
        else:
            # Still count sentences for the statistics even if we skip blanking
            self.total_sentences = len(self._split_sentences(lecture_text))

        # Pass 2: Markov word masking (only if there is a non-zero dropout rate)
        if self.p_dropout > 0:
            lecture_text = self._apply_markov_mask(lecture_text)
        else:
            # Count non-blanked words for statistics
            self.total_words = len([
                w for w in lecture_text.split()
                if w.strip() and w != "[BLANKED]"
            ])

        self.context = lecture_text
        return self.context

    def _split_sentences(self, text: str) -> List[str]:
        """Split text on sentence-ending punctuation and return non-empty chunks."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _apply_blanking(self, text: str) -> str:
        """
        Drops entire sentences with probability self.blank_prob.
        Dropped sentences are replaced with the [BLANKED] token so the
        Markov pass can see the boundary without reading the content.
        """
        sentences = self._split_sentences(text)
        self.total_sentences = len(sentences)

        result = []
        for sentence in sentences:
            if random.random() < self.blank_prob:
                # This sentence is completely missed
                self.blanked_sentences += 1
                result.append("[BLANKED]")
            else:
                result.append(sentence)

        return ". ".join(result) + "."

    def _apply_markov_mask(self, text: str) -> str:
        """
        Two-state Markov attention model applied word-by-word.

        While ATTENTIVE:
          - The word is kept in the output.
          - With probability p_dropout, the student transitions to ZONED_OUT.

        While ZONED_OUT:
          - The word is lost (not added to output).
          - One [ZONED] marker is emitted for the whole burst (not per-word).
          - With probability p_recover, the student transitions back to ATTENTIVE.

        The state machine runs across sentence boundaries — a zone-out started
        in one sentence continues into the next, mimicking real attention behavior.
        [BLANKED] tokens from Pass 1 are passed through unchanged.
        """
        words = text.split()
        # Count only real words, not the [BLANKED] placeholders
        self.total_words = len([w for w in words if w != "[BLANKED]"])

        result     = []
        attentive  = True
        in_zoneout = False   # tracks whether we already emitted a [ZONED] for this burst

        for word in words:
            # Pass [BLANKED] markers straight through without changing attention state
            if word == "[BLANKED]":
                result.append(word)
                in_zoneout = False   # treat sentence boundary as a burst reset
                continue

            if attentive:
                # Student is paying attention — keep the word
                result.append(word)
                in_zoneout = False

                # Random chance to start a zone-out episode after this word
                if random.random() < self.p_dropout:
                    attentive = False
                    self.zone_out_episodes += 1
            else:
                # Student is zoned out — this word is lost
                self.zoned_words += 1

                # Only emit one [ZONED] marker at the start of each burst
                if not in_zoneout:
                    result.append("[ZONED]")
                    in_zoneout = True

                # Random chance to recover and start paying attention again
                if random.random() < self.p_recover:
                    attentive  = True
                    in_zoneout = False

        return " ".join(result)

    # ==================== CONTEXT RETRIEVAL ====================

    def get_clean_context(self) -> str:
        """
        Strips all internal tokens ([BLANKED], [ZONED], [MASK]) and returns
        the plain text of everything the student actually retained.
        This is the ONLY method that should feed the LLM or RAG pipeline —
        everything else is internal bookkeeping.
        """
        clean = (self.context
                 .replace("[BLANKED]", "")
                 .replace("[ZONED]",   "")
                 .replace("[MASK]",    "")   # backwards-compat alias
                 .replace("  ", " ")
                 .strip())
        return clean

    def get_recent_context(self, num_sentences: int = 10) -> str:
        """
        Returns the last N sentences from the student's clean context.
        Used for in-lecture questioning where only recent material is relevant.
        """
        clean     = self.get_clean_context()
        sentences = self._split_sentences(clean)

        if len(sentences) > num_sentences:
            recent = sentences[-num_sentences:]
        else:
            recent = sentences

        return ". ".join(recent) + "."

    def chunk_context(self, chunk_size: int = 150, overlap: int = 30) -> List[str]:
        """
        Splits the clean context into overlapping word-level chunks for RAG retrieval.
        Overlap lets adjacent chunks share context so no information falls
        exactly on a chunk boundary.
        """
        clean = self.get_clean_context()
        words = clean.split()

        # If the whole context fits in one chunk, just return it
        if len(words) <= chunk_size:
            return [clean]

        chunks = []
        start  = 0
        while start < len(words):
            end         = min(start + chunk_size, len(words))
            chunk_start = max(0, start - overlap)
            chunk_end   = min(len(words), end + overlap)
            chunks.append(" ".join(words[chunk_start:chunk_end]))
            start = end

            if end >= len(words):
                break

        return chunks

    # ==================== STATISTICS ====================

    def get_statistics(self) -> dict:
        """
        Returns a dictionary of retention metrics that Agent.py sends back
        to Unity after every lecture. Unity uses these to colour-code students
        and populate the inspector.
        """
        clean_word_count    = len(self.get_clean_context().split())
        original_word_count = len(self.raw_lecture.split())

        # Analytical steady-state retention from the Markov parameters
        if self.p_dropout > 0:
            theoretical_retention = self.p_recover / (self.p_dropout + self.p_recover)
        else:
            theoretical_retention = 1.0

        # Average words lost per zone-out episode
        if self.zone_out_episodes > 0:
            avg_zoneout_length = round(self.zoned_words / self.zone_out_episodes, 1)
        else:
            avg_zoneout_length = 0.0

        return {
            "student_id":            self.student_id,
            "attention_type":        self.attention_type,
            "tier":                  self.tier,
            "blanked_sentences":     self.blanked_sentences,
            "total_sentences":       self.total_sentences,
            "blank_rate":            round(self.blanked_sentences / max(self.total_sentences, 1), 3),
            "zoned_words":           self.zoned_words,
            "total_words":           self.total_words,
            "zone_rate":             round(self.zoned_words / max(self.total_words, 1), 3),
            "zone_out_episodes":     self.zone_out_episodes,
            "avg_zoneout_length":    avg_zoneout_length,
            "p_dropout":             self.p_dropout,
            "p_recover":             self.p_recover,
            "theoretical_retention": round(theoretical_retention, 3),
            "retained_words":        clean_word_count,
            "original_words":        original_word_count,
            "effective_retention":   round(clean_word_count / max(original_word_count, 1), 3),
            "context_length":        len(self.context),
        }

    # ==================== ISOLATION ====================

    def verify_isolation(self, expected_student_id: int) -> bool:
        """
        Safety check — confirms this student object belongs to the expected ID.
        Prevents one student's context from leaking into another's answers.
        """
        return self.student_id == expected_student_id
