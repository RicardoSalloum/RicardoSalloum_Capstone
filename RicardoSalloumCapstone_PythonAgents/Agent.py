import uvicorn
import torch
import random
import threading
import time
import requests
from typing import List, Dict, Optional, Tuple

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM

from BaseStudent import BaseStudent, LectureInput, QuestionInput, MCQInput
from QuestioningSystem import QuestioningSystem

# ==================== CONFIG ====================
# Using the smaller 0.5B Qwen model so it fits on student machines without a big GPU
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# RAG chunk settings — controls how we split the lecture for retrieval
CHUNK_SIZE    = 128 * 1   # words per chunk
CHUNK_OVERLAP = 32 * 1    # shared words between adjacent chunks (reduces boundary blind-spots)
RETRIEVE_TOP_K = 8        # how many chunks FAISS returns before the cross-encoder reranks them

# Experimentally determined threshold: cross-encoder scores below this mean the
# retrieved chunks are unrelated to the question, so we random-guess instead
CROSS_ENCODER_THRESHOLD = -10.5

# Confidence thresholds per tier — lower-tier students need stronger context evidence
# before we trust their answers; otherwise they lean more on random guessing
CONFIDENCE_THRESHOLDS = {"A": 0.20, "B": 0.30, "C": 0.40}

# ==================== STARTUP ====================
print("=" * 60)
print("Student Agent  —  Qwen2.5-0.5B-Instruct")
print("=" * 60)

# Pick GPU if available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device.upper()}")
if device == "cuda":
    print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ==================== LOAD MODEL ====================
print(f"\nLoading {MODEL_NAME} ...")

# Tokenizer converts text to token IDs the model understands
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

# Load the LLM in float16 to halve VRAM usage compared to float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,   # explicit fp16 — "auto" would choose bfloat16
    device_map="auto",           # let Transformers decide which layers go on GPU/CPU
    local_files_only=True,       # don't download; model must already be cached
)
model.eval()   # disable dropout and other training-only behaviour
print("  ✓ Model loaded")

# ==================== EMBEDDING + RERANKING ====================
# The embedding model converts text chunks to vectors for FAISS similarity search
print("Loading embedding model ...")
embed_model      = SentenceTransformer("all-MiniLM-L6-v2", device="cpu", local_files_only=True)
embed_model_lock = threading.Lock()   # FAISS/embedding calls are not thread-safe

# The cross-encoder scores (question, chunk) pairs; much more accurate than embedding
# distance alone, but too slow to run on all chunks — so we pre-filter with FAISS first
print("Loading cross-encoder ...")
cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", local_files_only=True
)
print("  ✓ Ready")
print("=" * 60)

# ==================== FASTAPI ====================
app = FastAPI(title="VR Classroom — Qwen2.5-1.5B Student Agent")

# In-memory store: student_id → BaseStudent object
students_memory: Dict[int, BaseStudent] = {}

# The questioning system is initialized later when Unity sends the test bank
questioning_system: Optional[QuestioningSystem] = None


# ==================== CACHE ====================

class StudentCache:
    """
    Per-student FAISS index that is rebuilt only when the student's lecture
    context changes. Avoids re-embedding the same text on every question.
    Thread-safe via an internal lock so concurrent HTTP requests don't corrupt it.
    """

    def __init__(self):
        self.chunks       = None          # list of chunk dicts with 'text' and index info
        self.index        = None          # FAISS IndexFlatL2 over the chunk embeddings
        self.context_hash = None          # hash of the context that produced this index
        self._lock        = threading.Lock()

    def is_valid(self, context: str) -> bool:
        """Returns True if the cached index was built from the same context."""
        with self._lock:
            return self.chunks is not None and self.context_hash == hash(context)

    def build(self, context: str):
        """
        Chunks the context, embeds every chunk, and loads them into a new FAISS index.
        Skips rebuilding if another thread already did it for the same context.
        """
        with self._lock:
            h = hash(context)

            # Another thread may have built the index while we waited for the lock
            if self.chunks is not None and self.context_hash == h:
                return

            # Remove leftover [BLANKED] tokens before embedding
            clean = (context
                     .replace("_", " ")
                     .replace("[BLANKED]", ""))

            self.chunks = _make_chunks(clean, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            texts = [c["text"] for c in self.chunks]

            # Embed all chunks in one batch; lock prevents concurrent embedding calls
            with embed_model_lock:
                embeddings = embed_model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )

            # IndexFlatL2 does exact L2-distance search (no approximation)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings.astype("float32"))
            self.context_hash = h

            print(f"  [Cache] ✓ {len(self.chunks)} chunks indexed "
                  f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")


# ==================== RAG ====================

def _make_chunks(text: str, size: int = CHUNK_SIZE,
                 overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Splits a text into overlapping word-level windows.
    Each chunk is a dict with the text and its word-index range so we can
    trace back where in the lecture it came from.
    """
    words  = text.split()
    chunks = []

    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i: i + size])

        if chunk.strip():
            chunks.append({
                "text":      chunk.strip(),
                "start_idx": i,
                "end_idx":   min(i + size, len(words)),
            })

    return chunks


def _expand_query(q: str) -> List[str]:
    """
    Creates a small set of query variants by substituting common question
    words with synonyms. More query variants improve FAISS recall because
    the embedding model may associate different words with the same chunks.
    Returns at most 3 variants to keep retrieval fast.
    """
    syns = {
        "what":  ["which", "describe"],
        "how":   ["method", "process"],
        "why":   ["reason", "cause"],
        "where": ["location", "place"],
        "when":  ["time", "period"]
    }

    out  = [q]
    ql   = q.lower()

    for kw, alts in syns.items():
        if ql.startswith(kw):
            for a in alts:
                out.append(ql.replace(kw, a, 1))

    # Also try the question without its first word (drops "what", "how", etc.)
    words = q.split()
    if words and words[0].lower() in syns:
        out.append(" ".join(words[1:]))

    return out[:3]


def _retrieve(student, queries: List[str],
              top_k: int = RETRIEVE_TOP_K) -> List[Dict]:
    """
    Runs all query variants through FAISS and returns the union of the
    top-k results, keeping only the highest score for each chunk that
    appears in multiple query results.
    """
    # Lazily attach a cache to the student object if it doesn't have one yet
    if not hasattr(student, "_cache"):
        student._cache = StudentCache()

    cache = student._cache

    # Rebuild the index only if the student's context has changed
    if not cache.is_valid(student.context):
        cache.build(student.context)

    # Take a snapshot under the lock so we can release it before the loop
    with cache._lock:
        chunks = list(cache.chunks)
        index  = cache.index

    # seen deduplicates chunks that appear in multiple query results
    seen: Dict[str, Dict] = {}

    for q in queries:
        with embed_model_lock:
            qemb = embed_model.encode([q], convert_to_numpy=True)

        dists, idxs = index.search(qemb.astype("float32"), min(top_k, len(chunks)))

        for d, i in zip(dists[0], idxs[0]):
            if i < len(chunks):
                # Convert L2 distance to a 0-1 similarity score
                score = 1.0 / (1.0 + float(d))
                key   = chunks[i]["text"]

                # Keep the best score if this chunk appeared in multiple queries
                if key not in seen or score > seen[key]["score"]:
                    seen[key] = {**chunks[i], "score": score}

    return list(seen.values())


def _rerank_and_gate(question: str, chunks: List[Dict],
                     top_k: int = RETRIEVE_TOP_K) -> List[Dict]:
    """
    Uses the cross-encoder to score every (question, chunk) pair, then
    applies a score threshold gate. If the best chunk is below the threshold,
    the lecture doesn't contain relevant information for this question and
    we return an empty list — which tells the MCQ function to random-guess.
    """
    if not chunks:
        return []

    # Cross-encoder gives a relevance score for each (question, chunk) pair
    scores = cross_encoder.predict([(question, c["text"]) for c in chunks])

    for c, s in zip(chunks, scores):
        c["rerank_score"] = float(s)

    ranked     = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
    best_score = ranked[0]["rerank_score"]

    # If even the best chunk is irrelevant, return empty so the caller guesses
    if best_score < CROSS_ENCODER_THRESHOLD:
        print(f"  [Gate] ✗ best={best_score:.2f} < {CROSS_ENCODER_THRESHOLD} → random guess")
        return []

    print(f"  [Gate] ✓ best={best_score:.2f}")
    return ranked


def _build_context(chunks: List[Dict], max_words: int = 600) -> str:
    """
    Joins the reranked chunks into a single string for the LLM prompt.
    Stops adding chunks once we hit max_words to avoid overflowing the
    model's context window.
    """
    parts = []
    total = 0

    for c in chunks:
        n = len(c["text"].split())

        if total + n > max_words:
            break

        parts.append(c["text"])
        total += n

    return "\n\n".join(parts)   # double newline between chunks for readability


# ==================== INFERENCE ====================

def _generate(messages: List[Dict], max_new_tokens: int = 50,
              deterministic: bool = False) -> str:
    """
    Formats a conversation into the model's chat template, runs generation,
    and returns just the newly generated text (not the prompt).

    deterministic=True disables sampling and uses greedy decoding.
    We use this for MCQ so the model always picks one letter reliably.
    """
    # Apply the model's chat template to get a single formatted string
    text   = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens)

    if deterministic:
        # Greedy decoding — used for MCQ letter selection
        gen_kwargs["do_sample"] = False
    else:
        # Sampling with temperature for open-ended answers
        gen_kwargs["do_sample"]   = True
        gen_kwargs["temperature"] = 0.7
        gen_kwargs["top_p"]       = 0.9

    # torch.no_grad() skips gradient tracking; we only need forward passes here
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    # Strip the prompt tokens from the output — we only want the new tokens
    new_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def answer_question(question: str, context: str) -> Tuple[str, float]:
    """
    Generates a free-text answer for the /ask_question endpoint.
    The system prompt instructs the model to stay strictly within the
    provided lecture context, even if that context contains intentionally
    wrong facts (to simulate what a student with wrong notes would say).
    Returns (answer_text, confidence_score).
    """
    messages = [
        {"role": "system", "content": (
            "You are a student sitting an exam. "
            "Answer ONLY using the lecture text provided. "
            "Do NOT use outside knowledge. "
            "You may relate topics together within the lecture. "
            "You may use synonyms, match terms by sound, or connect ideas "
            "that clearly refer to the same concept within the lecture. "
            "Do not introduce any information that is not present in the lecture. "
            "You must only output the letter and nothing else"

            "Below are examples.\n\n"

            "----- EXAMPLES -----\n\n"

            "Context: Photosynthesis occurs in the chloroplast and produces glucose.\n"
            "Question: What does photosynthesis produce?\n"
            "Options:\nA. Oxygen\nB. Glucose\nC. Nitrogen\nD. Water\n"
            "Answer: B\n\n"

            "Context: The mitochondria is the powerhouse of the cell and produces ATP.\n"
            "Question: What organelle produces ATP?\n"
            "Options:\nA. Ribosome\nB. Nucleus\nC. Mitochondria\nD. Lysosome\n"
            "Answer: C\n\n"

            "Context: The heart pumps blood through arteries and veins.\n"
            "Question: What does the heart pump?\n"
            "Options:\nA. Oxygen\nB. Blood\nC. Hormones\nD. Water\n"
            "Answer: B\n\n"

            "Context: DNA contains genetic information in living organisms.\n"
            "Question: What molecule carries genetic information?\n"
            "Options:\nA. RNA\nB. Protein\nC. DNA\nD. Lipid\n"
            "Answer: C\n\n"

            # --- MISINFORMATION EXAMPLES ---
            # These teach the model to follow the lecture's logic even when it's wrong
            "Context: Photosynthesis happens in the mitochondria and produces nitrogen.\n"
            "Question: What does photosynthesis produce?\n"
            "Options:\nA. Oxygen\nB. Glucose\nC. Nitrogen\nD. Water\n"
            "Answer: C\n\n"

            "Context: The nucleus creates ATP for the cell.\n"
            "Question: What organelle produces ATP?\n"
            "Options:\nA. Ribosome\nB. Nucleus\nC. Mitochondria\nD. Lysosome\n"
            "Answer: B\n\n"

            "Context: The heart pumps oxygen directly through the body.\n"
            "Question: What does the heart pump?\n"
            "Options:\nA. Oxygen\nB. Blood\nC. Hormones\nD. Water\n"
            "Answer: A\n\n"

            "Context: RNA is the main carrier of genetic information in cells.\n"
            "Question: What molecule carries genetic information?\n"
            "Options:\nA. RNA\nB. Protein\nC. DNA\nD. Lipid\n"
            "Answer: A\n\n"

            # --- UNRELATED EXAMPLES ---
            # When the context has nothing to do with the question, output a sentinel
            "Context: The Eiffel Tower is located in Paris and was built in 1889.\n"
            "Question: What organelle produces ATP?\n"
            "Options:\nA. Ribosome\nB. Nucleus\nC. Mitochondria\nD. Lysosome\n"
            "Answer: NOTHING RELEVANT FOUND\n\n"

            "Context: Basketball is played with five players on each team.\n"
            "Question: What does photosynthesis produce?\n"
            "Options:\nA. Oxygen\nB. Glucose\nC. Nitrogen\nD. Water\n"
            "Answer: NOTHING RELEVANT FOUND\n\n"

            "Context: The Pacific Ocean is the largest ocean on Earth.\n"
            "Question: What molecule carries genetic information?\n"
            "Options:\nA. RNA\nB. Protein\nC. DNA\nD. Lipid\n"
            "Answer: NOTHING RELEVANT FOUND\n\n"

            "Context: Cars use gasoline engines to convert fuel into motion.\n"
            "Question: What does the heart pump?\n"
            "Options:\nA. Oxygen\nB. Blood\nC. Hormones\nD. Water\n"
            "Answer: NOTHING RELEVANT FOUND\n\n"

            "----- END EXAMPLES -----\n"
        )},
        {"role": "user",
         "content": f"Lecture:\n{context}\n\nQuestion: {question}"},
    ]

    answer = _generate(messages, max_new_tokens=150, deterministic=False)
    low    = answer.lower()

    # Assign a rough confidence score based on answer length and hedging phrases
    if len(answer) > 20 and "don't know" not in low:
        conf = 0.8
    else:
        conf = 0.2

    # Slight confidence boost if the first word appears in the context
    if answer.split() and answer.split()[0].lower() in context.lower():
        conf = min(0.95, conf + 0.1)

    return answer, conf


def _option_support_score(option: str, context: str) -> float:
    """
    Measures how strongly one MCQ option is supported by the retrieved context.
    Used as a fallback signal when the model output is ambiguous.

    Returns 1.0 for a direct substring match, or a token-overlap ratio otherwise.
    """
    option_low  = option.lower()
    context_low = context.lower()

    # Direct match is the strongest signal
    if option_low in context_low:
        return 1.0

    # Fall back to Jaccard-style token overlap
    opt_tokens = set(option_low.split())
    ctx_tokens = set(context_low.split())

    if not opt_tokens:
        return 0.0

    overlap = len(opt_tokens & ctx_tokens) / len(opt_tokens)
    return overlap


def answer_mcq(question: str, options: List[str], context: str,
               tier: str, no_context: bool = False) -> Tuple[int, float]:
    """
    Picks one MCQ option given the student's retrieved context.

    Step 1 — Hard gate: if there is no relevant context, guess randomly.
    Step 2 — Strict LLM inference: the model reads the context and picks a letter.
              If it can't find anything relevant, it outputs "NOTHING RELEVANT FOUND"
              and we fall back to a random guess.
    Step 3 — Fallback: if the model output contains no recognisable letter, pick 0.
    """
    # No context → random guess with low confidence
    if no_context or not context.strip():
        idx  = random.randint(0, len(options) - 1)
        conf = round(random.uniform(0.10, 0.25), 3)
        return int(idx), conf

    # Format the options as "A. text", "B. text", etc. for the prompt
    opts_text = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))

    messages = [
        {"role": "system", "content": (
            "You are a student taking a test based ONLY on a specific provided lecture. "
            "The lecture may contain unconventional or 'anti-scientific' facts. "
            "You MUST follow the lecture's logic exactly, even if it contradicts common sense. "
            "If the lecture says 'Oxygen is toxic', then Oxygen is toxic. "
            "Do not introduce any information that is not present in the lecture. "
            "You are forbidden from accessing knowledge outside the provided context. "
            "Respond with ONLY the letter (A, B, C, or D). "
            "If no info exists, say 'NOTHING RELEVANT FOUND'."
        )},
        {"role": "user", "content": (
            f"LECTURE CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n"
            f"OPTIONS:\n{opts_text}"
        )},
    ]

    # Deterministic (greedy) decoding so the model reliably picks one letter
    raw       = _generate(messages, max_new_tokens=5, deterministic=True)
    raw_upper = raw.upper()

    print(f"  [MCQ] Model Output: {repr(raw)}")

    # If the model explicitly says it has no relevant info, guess randomly
    if "NOTHING RELEVANT" in raw_upper:
        idx = random.randint(0, len(options) - 1)
        return int(idx), round(random.uniform(0.05, 0.15), 3)

    # Extract the first A/B/C/D letter from the model's output
    letter = None
    for c in raw_upper:
        if c in "ABCD":
            letter = c
            break

    if letter is not None:
        idx = ord(letter) - ord("A")

        # Tier A students answer more confidently than lower tiers
        if tier == "A":
            conf = 0.85
        else:
            conf = 0.70

        return int(idx), round(conf, 3)

    # Nothing parseable — default to option 0 with very low confidence
    return 0, 0.1


# ==================== ENDPOINTS ====================

@app.post("/send_lecture")
def send_lecture(data: LectureInput):
    """
    Unity posts the full lecture text here for each student.
    The student's attention model runs immediately and the resulting
    statistics are returned so Unity can display them in the inspector.
    """
    try:
        sid = int(data.student_id)

        # Create a new BaseStudent if this is the first time we see this ID
        if sid not in students_memory:
            students_memory[sid] = BaseStudent(
                student_id=sid, attention_type=data.attention_type
            )
            students_memory[sid]._cache = StudentCache()

        s = students_memory[sid]
        s.set_lecture_context(data.lecture, use_blanking=data.use_blanking)

        return {"status": "success", "statistics": s.get_statistics()}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_lecture")   # Unity uses this alias interchangeably
def process_lecture(data: LectureInput):
    """Alias for /send_lecture kept for backwards compatibility with older Unity builds."""
    return send_lecture(data)


@app.post("/ask_question")
def ask_question_endpoint(data: QuestionInput):
    """
    Unity asks a specific student a free-text question.
    Retrieves relevant lecture chunks for that student and runs the LLM.
    """
    s = students_memory.get(data.studentID)

    if not s:
        # Student hasn't received the lecture yet — return a polite placeholder
        return {
            "student":    data.studentName,
            "question":   data.question,
            "answer":     "I need lecture content first.",
            "confidence": 0.0
        }

    # Full RAG pipeline: expand query → FAISS → cross-encoder → build context → LLM
    chunks  = _rerank_and_gate(data.question, _retrieve(s, _expand_query(data.question)))
    context = _build_context(chunks)
    ans, conf = answer_question(data.question, context)

    return {
        "student":    data.studentName,
        "question":   data.question,
        "answer":     ans,
        "confidence": float(round(conf, 3))
    }


@app.post("/answer_mcq")
def answer_mcq_endpoint(data: MCQInput):
    """
    Unity submits an MCQ question and options for one student.
    Returns the chosen option index, a confidence score, and a flag
    indicating whether the student had relevant context or had to guess.
    """
    try:
        s = students_memory.get(data.student_id)

        if not s:
            return {"choice_index": 0, "confidence": 0.0, "no_context": False}

        raw_chunks = _retrieve(s, _expand_query(data.question))
        ranked     = _rerank_and_gate(data.question, raw_chunks)
        no_context = len(ranked) == 0

        if no_context:
            context = ""
        else:
            context = _build_context(ranked)

        idx, conf = answer_mcq(
            data.question, data.options, context, s.tier, no_context=no_context
        )

        return {"choice_index": idx, "confidence": conf, "no_context": no_context}

    except Exception as e:
        print(f"  [ERROR] answer_mcq: {e}")
        return {"choice_index": 0, "confidence": 0.0, "no_context": False}


@app.post("/generate_question")
def generate_question(data: dict):
    """
    Unity asks the backend to generate a question on behalf of a student.
    Uses the QuestioningSystem to decide whether the student would raise
    their hand, then synthesizes TTS audio for the question text.
    """
    sid   = data.get("student_id")
    sname = data.get("student_name", f"Student_{sid}")
    s     = students_memory.get(int(sid))

    if not s or questioning_system is None:
        return {"has_question": False}

    # QuestioningSystem decides if this student would ask a question right now
    qreq = questioning_system.student_attempts_question(
        student_id=sid,
        student_name=sname,
        student_context=s.context,
        qa_pipeline=None,   # not used in the current implementation
    )

    if qreq:
        questioning_system.add_question(qreq)

        # Fire-and-forget TTS request — if it fails we still return the question text
        try:
            requests.post(
                "http://127.0.0.1:5003/synthesize",
                json={"text": qreq.question, "voice": "af_bella"},
                timeout=5
            )
        except Exception:
            pass

        return {
            "has_question":   True,
            "question":       qreq.question,
            "question_type":  qreq.question_type,
            "student_name":   sname,
            "confidence":     float(qreq.confidence)
        }

    return {"has_question": False}


@app.post("/answer_student_question")
def answer_student_question(data: dict):
    """
    Teacher mode: Unity sends a student's question so the teacher agent
    can answer it using the lecture content. Returns the answer + metadata.
    """
    qtext = data.get("question")
    sid   = data.get("student_id")

    if not qtext:
        return {"status": "error", "message": "No question provided"}

    # Find the student — if no ID given, use any loaded student as context
    if sid:
        s = students_memory.get(int(sid))
    else:
        s = next(iter(students_memory.values()), None)

    if not s:
        return {"status": "error", "message": "No students initialized"}

    t0      = time.time()
    chunks  = _rerank_and_gate(qtext, _retrieve(s, _expand_query(qtext)))
    context = _build_context(chunks)
    ans, conf = answer_question(qtext, context)

    return {
        "status":          "success",
        "question":        qtext,
        "answer":          ans,
        "confidence":      float(round(conf, 3)),
        "processing_time": float(round(time.time() - t0, 3)),
        "model":           MODEL_NAME
    }


@app.get("/get_statistics")
def get_statistics():
    """Returns attention and retention statistics for every loaded student."""
    stats = {"total_students": len(students_memory), "students": {}}

    for sid, s in students_memory.items():
        stats["students"][sid] = s.get_statistics()

    if questioning_system:
        stats["questioning"] = questioning_system.get_statistics()

    return stats


@app.post("/initialize_questioning")
def initialize_questioning(data: dict):
    """
    Unity sends the exam test bank here at the start of a session so the
    QuestioningSystem knows which questions students might struggle with.
    """
    global questioning_system
    questioning_system = QuestioningSystem(
        test_bank=data.get("test_bank", []),
        max_questions_per_minute=data.get("max_questions_per_minute", 3),
    )

    return {
        "status":         "initialized",
        "test_bank_size": len(data.get("test_bank", []))
    }


@app.get("/health")
def health_check():
    """Basic health check — Unity pings this to confirm the agent is running."""
    return {
        "status":          "healthy",
        "model":           MODEL_NAME,
        "device":          device.upper(),
        "students_loaded": len(students_memory),
        "chunk_size":      CHUNK_SIZE,
        "chunk_overlap":   CHUNK_OVERLAP,
    }


class LectureQuestionInput(BaseModel):
    snippet: str = ""

    # Guard: if Unity sends {"snippet": {}} due to a JsonUtility serialization bug,
    # coerce it to an empty string instead of crashing
    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        if isinstance(obj, dict) and not isinstance(obj.get("snippet", ""), str):
            obj = {**obj, "snippet": ""}
        return super().model_validate(obj, *args, **kwargs)


@app.post("/generate_lecture_question")
def generate_lecture_question(data: LectureQuestionInput):
    """
    Given a recent lecture snippet, uses the LLM to generate one natural
    student-style question — as if a student raised their hand mid-lecture.
    Returns the question text if one was successfully generated.
    """
    snippet = data.snippet.strip()

    if not snippet:
        return {"has_question": False, "question": ""}

    messages = [
        {"role": "system", "content": (
            "You are a curious student sitting in a university lecture. "
            "Based on the lecture excerpt provided, generate ONE genuine question "
            "you might ask the professor to clarify or explore the topic further. "
            "The question must be directly grounded in the lecture content — "
            "do not ask about anything not mentioned. "
            "Output ONLY the question itself, no preamble, no quotation marks, "
            "no explanation. The question must end with a question mark."
        )},
        {"role": "user", "content": f"Lecture excerpt:\n{snippet}\n\nYour question:"},
    ]

    try:
        question = _generate(messages, max_new_tokens=80, deterministic=False)

        # Trim to just the first complete question (ends at the first '?')
        if "?" in question:
            question = question[:question.index("?") + 1].strip()
        else:
            # No question mark means the model didn't produce a valid question
            return {"has_question": False, "question": ""}

        print(f"  [LLM Question] Generated: {question}")
        return {"has_question": True, "question": question}

    except Exception as e:
        print(f"  [ERROR] generate_lecture_question: {e}")
        return {"has_question": False, "question": ""}


class TestBankSimilarityInput(BaseModel):
    snippet: str
    questions: List[str]
    threshold: float = 0.35   # cosine-similarity cutoff; tune this to get more/fewer matches


@app.post("/testbank_similarity")
def testbank_similarity(data: TestBankSimilarityInput):
    """
    Computes cosine similarity between a live lecture snippet and every
    question in the test bank. Returns only the questions whose similarity
    score meets or exceeds the threshold, sorted from most to least relevant.

    Unity uses the returned matches to pick a contextually relevant question
    to ask students — similar to how a teacher might choose what to quiz on
    based on what they just covered.
    """
    if not data.snippet.strip() or not data.questions:
        return {"matches": []}

    with embed_model_lock:
        # Encode the snippet and all questions in one batch for efficiency
        texts      = [data.snippet] + data.questions
        embeddings = embed_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True   # L2-normalised so dot product == cosine similarity
        )

    snippet_vec   = embeddings[0]    # shape: (dim,)
    question_vecs = embeddings[1:]   # shape: (N, dim)

    # Dot product of L2-normalised vectors equals cosine similarity
    scores = (question_vecs @ snippet_vec).tolist()

    # Collect every question that meets the similarity threshold
    matches = []
    for i in range(len(data.questions)):
        if scores[i] >= data.threshold:
            matches.append({
                "index":    i,
                "question": data.questions[i],
                "score":    round(scores[i], 4)
            })

    # Sort so the most relevant question is first
    matches.sort(key=lambda x: x["score"], reverse=True)

    print(
        f"  [TestBank Similarity] snippet={len(data.snippet.split())}w  "
        f"questions={len(data.questions)}  "
        f"threshold={data.threshold}  matches={len(matches)}"
    )

    return {"matches": matches}


# ==================== MAIN ====================
if __name__ == "__main__":
    print(f"\nStarting on http://127.0.0.1:5002\n")
    uvicorn.run(app, host="127.0.0.1", port=5002)
