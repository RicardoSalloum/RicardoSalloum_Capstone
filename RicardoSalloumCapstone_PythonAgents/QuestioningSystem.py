import random
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel
from datetime import datetime
import json

"""
QuestioningSystem.py - Real-time student questioning with TTS
Handles question generation, prioritisation, and text-to-speech output.

The system simulates students raising their hands during a lecture:
  1. Periodically, students check whether they can answer a test question.
  2. If they can't (due to blanked/zoned context), they generate a question.
  3. Questions are queued by priority and delivered via TTS.
"""
class QuestionRequest(BaseModel):
    #Represents one student's request to ask a question.
    student_id:    int
    student_name:  str
    question:      str
    question_type: str    # "early", "clarification", or "off_topic"
    confidence:    float  # how sure the student is that they need to ask
    timestamp:     float
    is_off_topic:  bool = False


class TeacherResponse(BaseModel):
    #Records a teacher's answer to a student question.
    question_id:         str
    response_text:       str
    timestamp:           float
    addressed_student_id: int


class QuestioningSystem:
    """
    Manages the flow of real-time student questions during a lecture.

    Workflow:
    1. Students periodically attempt to answer a random test question.
    2. If they fail (low confidence, missing context), they raise their hand.
    3. Questions are queued and served to the teacher in priority order.
    4. Teacher responses are recorded for the session report.
    """

    def __init__(self, test_bank: List[dict], max_questions_per_minute: int = 3):
        """
        Args:
            test_bank:                List of exam questions students will attempt.
            max_questions_per_minute: Rate limit to avoid flooding the teacher with questions.
        """
        self.test_bank = test_bank
        self.max_questions_per_minute = max_questions_per_minute

        # Separate lists for pending (not yet asked) and already-asked questions
        self.pending_questions: List[QuestionRequest] = []
        self.asked_questions:   List[QuestionRequest] = []
        self.teacher_responses: List[TeacherResponse] = []

        # Track timing so we can enforce the rate limit
        self.last_question_time = 0
        self.question_interval  = 60.0 / max_questions_per_minute   # seconds between questions

        # Running counters for the statistics report
        self.stats = {
            "early_questions":         0,
            "clarification_questions": 0,
            "off_topic_questions":     0,
            "questions_answered":      0,
            "average_response_time":   0.0
        }

    def student_attempts_question(
        self,
        student_id:    int,
        student_name:  str,
        student_context: str,
        qa_pipeline,
        is_disruptive:  bool  = False,
        off_topic_prob: float = 0.0
    ) -> Optional[QuestionRequest]:
        """
        Simulates one student's attempt to answer a random test question.
        If the attempt fails (or is disruptive), returns a QuestionRequest
        representing the student raising their hand. Returns None otherwise.

        Args:
            student_id:      The student's numeric ID.
            student_name:    Display name for logging.
            student_context: The student's degraded lecture notes (after Markov masking).
            qa_pipeline:     QA model — currently passed as None; evaluation uses heuristics.
            is_disruptive:   Whether the student has a chance to ask off-topic questions.
            off_topic_prob:  Probability of an off-topic question if is_disruptive is True.
        """
        # Disruptive students occasionally ask something completely unrelated
        if is_disruptive and random.random() < off_topic_prob:
            return self._generate_off_topic_question(student_id, student_name)

        # No test bank means we can't pick a question for the student to attempt
        if not self.test_bank:
            return None

        # Pick a random question from the test bank to simulate
        test_q        = random.choice(self.test_bank)
        question_text = test_q.get("text", test_q.get("question", ""))

        try:
            result     = qa_pipeline(question=question_text, context=student_context)
            confidence = result.get("score", 0.0)
            answer     = result.get("answer", "")

            # Decide whether the student needs to ask for help
            needs_help, question_type = self._evaluate_answer_quality(
                answer, confidence, question_text, student_context
            )

            if needs_help:
                return QuestionRequest(
                    student_id=student_id,
                    student_name=student_name,
                    question=question_text,
                    question_type=question_type,
                    confidence=confidence,
                    timestamp=datetime.now().timestamp(),
                    is_off_topic=False
                )

        except Exception as e:
            print(f"Error in student_attempts_question: {e}")

        return None

    def _evaluate_answer_quality(
        self,
        answer:     str,
        confidence: float,
        question:   str,
        context:    str
    ) -> Tuple[bool, str]:
        """
        Heuristically judges whether a student's answer is good enough or
        whether they need to ask a clarifying question.

        Returns:
            (needs_help: bool, question_type: str)
            question_type is one of "clarification", "early", or "none".
        """
        # A [BLANKED] marker means the student missed the relevant sentence entirely
        if "[BLANKED]" in context or "____" in answer:
            return (True, "clarification")

        # Low confidence suggests the student is unsure about their answer
        if confidence < 0.3:
            return (True, "clarification")

        # A very short answer likely means the context was missing key content
        if len(answer.split()) < 2:
            return (True, "clarification")

        # Check if the question's keywords appear in the context at all.
        # If they don't, the topic probably hasn't been covered yet — "early" question.
        stop_words    = {"what", "is", "the", "a", "an", "how", "why", "when", "where"}
        question_keywords = set(question.lower().split()) - stop_words
        context_lower = context.lower()

        covered_keywords = sum(1 for kw in question_keywords if kw in context_lower)
        keyword_coverage = covered_keywords / max(len(question_keywords), 1)

        if keyword_coverage < 0.3:
            return (True, "early")

        # Student seems to have enough information — no need to raise hand
        return (False, "none")

    def _generate_off_topic_question(self, student_id: int, student_name: str) -> QuestionRequest:
        """
        Generates a disruptive, off-topic question (e.g. 'Can I go to the bathroom?').
        Used to simulate students who are not engaged with the lecture.
        """
        off_topic_questions = [
            "Can I go to the bathroom?",
            "When is lunch?",
            "Is this going to be on the test?",
            "Can we watch a movie instead?",
            "What time does class end?",
            "Do we have homework tonight?",
            "Can I borrow a pencil?",
            "Why do we have to learn this?",
            "Can I get a drink of water?",
            "What are we doing tomorrow?"
        ]

        self.stats["off_topic_questions"] += 1

        return QuestionRequest(
            student_id=student_id,
            student_name=student_name,
            question=random.choice(off_topic_questions),
            question_type="off_topic",
            confidence=1.0,    # they're completely sure they want to ask this!
            timestamp=datetime.now().timestamp(),
            is_off_topic=True
        )

    def add_question(self, question: QuestionRequest):
        """Adds a question to the pending queue and updates the stat counters."""
        self.pending_questions.append(question)

        if question.question_type == "early":
            self.stats["early_questions"] += 1
        elif question.question_type == "clarification":
            self.stats["clarification_questions"] += 1

    def get_next_question(self, current_time: float) -> Optional[QuestionRequest]:
        """
        Pops and returns the highest-priority pending question, respecting
        the per-minute rate limit. Returns None if the rate limit is active
        or there are no pending questions.

        Priority order:
          1. clarification — student is confused about something already covered
          2. early         — student is asking about content not covered yet
          3. off_topic     — disruptive, lowest priority
        """
        # Enforce the rate limit: check if enough time has passed since the last question
        if current_time - self.last_question_time < self.question_interval:
            return None

        if not self.pending_questions:
            return None

        # Sort by (priority_number, descending_confidence) so the most urgent question
        # with the highest confidence floats to the top
        priority_order = {"clarification": 0, "early": 1, "off_topic": 2}
        self.pending_questions.sort(
            key=lambda q: (priority_order.get(q.question_type, 3), -q.confidence)
        )

        # Pop the top question, move it to the asked list, and reset the timer
        question = self.pending_questions.pop(0)
        self.asked_questions.append(question)
        self.last_question_time = current_time

        return question

    def record_teacher_response(
        self,
        question_id: str,
        response_text: str,
        student_id: int
    ):
        """Stores a teacher's answer and increments the questions-answered counter."""
        response = TeacherResponse(
            question_id=question_id,
            response_text=response_text,
            timestamp=datetime.now().timestamp(),
            addressed_student_id=student_id
        )

        self.teacher_responses.append(response)
        self.stats["questions_answered"] += 1

    def get_statistics(self) -> dict:
        """Returns a summary dict that Agent.py includes in the /get_statistics response."""
        total_questions = len(self.asked_questions)

        return {
            **self.stats,
            "total_questions_asked": total_questions,
            "pending_questions":     len(self.pending_questions),
            # question_rate here means how many questions were asked per teacher response
            "question_rate":         total_questions / max(1, len(self.teacher_responses))
        }

    def generate_question_timeline(self) -> List[dict]:
        """
        Builds a chronological list of all question and response events.
        Used for the session report exported at the end of a classroom session.
        """
        timeline = []

        for q in self.asked_questions:
            timeline.append({
                "timestamp":     q.timestamp,
                "event_type":    "question",
                "student_id":    q.student_id,
                "student_name":  q.student_name,
                "question":      q.question,
                "question_type": q.question_type
            })

        for r in self.teacher_responses:
            timeline.append({
                "timestamp":  r.timestamp,
                "event_type": "response",
                "student_id": r.addressed_student_id,
                "response":   r.response_text
            })

        # Sort all events by timestamp so the timeline is in order
        timeline.sort(key=lambda x: x["timestamp"])

        return timeline

    def export_session_data(self, filepath: str):
        """Dumps the full session data (stats, timeline, questions, responses) to JSON."""
        data = {
            "statistics":    self.get_statistics(),
            "timeline":      self.generate_question_timeline(),
            "all_questions": [q.dict() for q in self.asked_questions],
            "all_responses": [r.dict() for r in self.teacher_responses]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def Stop(self):
        """Clears the pending queue — called by VoiceController when the lecture ends."""
        self.pending_questions.clear()


class TTSWrapper:
    """
    Thin wrapper around several TTS backends (Kokoro, gTTS, pyttsx3).
    Tries to use Kokoro first since it gives the highest quality audio;
    falls back to gTTS if Kokoro isn't installed, and to pyttsx3 if gTTS isn't either.
    """

    def __init__(self, engine: str = "kokoro", voice: str = "af_bella"):
        """
        Args:
            engine: Which TTS backend to use ("kokoro", "gtts", or "pyttsx3").
            voice:  Kokoro voice ID (ignored for gTTS and pyttsx3).
        """
        self.engine       = engine
        self.voice        = voice
        self.kokoro_model = None

        if engine == "kokoro":
            try:
                import kokoro
                import torch

                # Using af_bella because it sounds like a friendly student voice
                self.kokoro_model = kokoro.Kokoro(voice, lang='en-us')
                print(f"✓ Kokoro TTS initialized with voice: {voice}")

            except ImportError:
                print("⚠ Kokoro not installed. Install with: pip install kokoro-onnx")
                print("  Falling back to gTTS")
                self.engine = "gtts"
            except Exception as e:
                print(f"⚠ Kokoro initialization failed: {e}")
                print("  Falling back to gTTS")
                self.engine = "gtts"

        elif engine == "pyttsx3":
            try:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)   # words per minute
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    self.tts_engine.setProperty('voice', voices[0].id)
            except:
                print("pyttsx3 not available, falling back to gtts")
                self.engine = "gtts"

    def speak(self, text: str, student_name: str = "Student",
              output_path: Optional[str] = None):
        """
        Generates speech for the given text.
        If output_path is provided, saves the audio to that file.
        Otherwise plays it immediately using the system's default audio player.

        Args:
            text:         The text to speak.
            student_name: Prepended to the text so it sounds like a student asking.
            output_path:  Optional file path to save the audio instead of playing it.
        """
        full_text = f"{student_name} asks: {text}"

        if self.engine == "kokoro" and self.kokoro_model is not None:
            try:
                import numpy as np
                import soundfile as sf
                import os

                samples, sample_rate = self.kokoro_model.create(full_text)

                # Decide where to save the WAV file
                if output_path:
                    audio_file = output_path
                else:
                    audio_file = "/tmp/student_question_kokoro.wav"

                sf.write(audio_file, samples, sample_rate)

                # Only play it automatically when no output_path was given
                if not output_path:
                    if os.name == 'posix':
                        os.system(
                            f'aplay {audio_file} 2>/dev/null || '
                            f'ffplay -nodisp -autoexit {audio_file} 2>/dev/null'
                        )
                    else:
                        os.system(f'start {audio_file}')

                print(f"🔊 [Kokoro TTS] {full_text}")

            except Exception as e:
                print(f"⚠ Kokoro TTS error: {e}")
                print(f"[TTS Fallback] {full_text}")

        elif self.engine == "gtts":
            try:
                from gtts import gTTS
                import os

                tts = gTTS(text=full_text, lang='en', slow=False)

                if output_path:
                    tts.save(output_path)
                else:
                    temp_file = "/tmp/student_question.mp3"
                    tts.save(temp_file)

                    if os.name == 'posix':
                        os.system(
                            f'mpg123 {temp_file} 2>/dev/null || '
                            f'ffplay -nodisp -autoexit {temp_file} 2>/dev/null'
                        )
                    else:
                        os.system(f'start {temp_file}')

            except Exception as e:
                print(f"gTTS error: {e}")
                print(f"[TTS Fallback] {full_text}")

        elif self.engine == "pyttsx3":
            try:
                if output_path:
                    self.tts_engine.save_to_file(full_text, output_path)
                    self.tts_engine.runAndWait()
                else:
                    self.tts_engine.say(full_text)
                    self.tts_engine.runAndWait()

            except Exception as e:
                print(f"pyttsx3 error: {e}")
                print(f"[TTS Fallback] {full_text}")

        else:
            # No TTS backend available — just print so we can at least see what would be spoken
            print(f"[TTS] {full_text}")
