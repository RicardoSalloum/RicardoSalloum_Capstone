#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent_Voice.py - Voice Transcription Service
Port: 5005

Receives WAV audio from Unity (either a 30-second snippet or the full lecture),
runs it through OpenAI Whisper, removes filler words, and returns the cleaned text.
"""

import os
import re
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
import whisper
import tempfile
import warnings

# Whisper warns about FP16 not being supported on CPU; suppress it to keep logs clean
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# ==================== SETUP ====================
print("="*60)
print("Loading Whisper Voice Transcription Agent")
print("="*60)

# Whisper model size controls the speed/accuracy tradeoff.
# Options: tiny, base, small, medium, large — "base" is fast enough for this project
MODEL_SIZE = "base"
print(f"Loading Whisper model: {MODEL_SIZE}")
print("This may take a minute on first run...")

# Load the model once at startup; it stays in memory for all requests
model = whisper.load_model(MODEL_SIZE)
print(f"✓ Whisper {MODEL_SIZE} model loaded successfully")

print("="*60)

# ==================== FASTAPI APP ====================
app = FastAPI(title="Voice Transcription Agent")

# ==================== FILLER WORDS ====================
# Words that Whisper often picks up but carry no actual content.
# We strip these out to give the student agents cleaner lecture text.
FILLER_WORDS = [
    "uh", "uhh", "um", "umm", "uh-huh", "mm-hmm",
    "er", "err", "ah", "ahh", "eh", "ehh",
    "like", "you know", "I mean", "sort of", "kind of",
    "actually", "basically", "literally", "honestly",
    "okay", "ok", "alright", "right", "so", "well"
]


def clean_transcription(text: str) -> str:
    """
    Removes filler words from Whisper output while preserving the
    original casing of everything else (proper nouns, acronyms, etc.).

    Steps:
      1. Strip each filler word using a case-insensitive word-boundary regex.
      2. Collapse any double spaces left behind.
      3. Re-capitalize the first letter after sentence-ending punctuation.
      4. Ensure the very first character of the whole string is capitalized.
    """
    cleaned = text

    # Remove each filler word wherever it appears, case-insensitively.
    # \b ensures we match whole words only (e.g. "uh" won't match "uhh").
    for filler in FILLER_WORDS:
        pattern = r'(?i)\b' + re.escape(filler) + r'\b'
        cleaned = re.sub(pattern, '', cleaned)

    # Collapsing double spaces that were left when filler words were removed
    cleaned = re.sub(r'  +', ' ', cleaned).strip()

    # After '.', '!' or '?', the next letter should be capitalized
    # The lambda keeps the punctuation + space and uppercases only the letter
    cleaned = re.sub(
        r'([.!?]\s+)([a-z])',
        lambda m: m.group(1) + m.group(2).upper(),
        cleaned
    )

    # Make sure the very first character of the result is uppercase
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned


# ==================== ENDPOINTS ====================

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    is_snippet: str = Form(...)
):
    """
    Transcribes an uploaded WAV file and returns both the raw and
    filler-stripped versions of the text.

    Args:
        file:       WAV audio uploaded from Unity
        is_snippet: "true" for 30-second live snippets, "false" for the
                    full lecture at the end of a session
    """
    # Convert the form string to an actual boolean
    is_snippet_bool = is_snippet.lower() == "true"

    try:
        # Save the uploaded bytes to a temporary file so Whisper can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name

        file_size_mb = len(content) / (1024 * 1024)
        print(f"\n{'[SNIPPET]' if is_snippet_bool else '[FULL LECTURE]'} Processing audio...")
        print(f"File size: {file_size_mb:.2f} MB")

        # Base transcription parameters — FP16 is disabled because we run on CPU
        transcribe_params = {
            "language": "en",
            "task":     "transcribe",
            "fp16":     False,
            "verbose":  False
        }

        # For large files, reduce beam_size to keep transcription time reasonable
        if file_size_mb > 2:
            print("Large file detected - using faster transcription settings")
            transcribe_params["beam_size"] = 1   # faster but slightly less accurate
            transcribe_params["best_of"]   = 1
        else:
            transcribe_params["beam_size"] = 5   # default accuracy for short clips

        import time
        start_time = time.time()

        # Run Whisper — this is the main blocking call
        result = model.transcribe(temp_audio_path, **transcribe_params)

        elapsed_time = time.time() - start_time

        original_text = result["text"].strip()
        cleaned_text  = clean_transcription(original_text)

        # Remove the temp file now that Whisper is done with it
        os.unlink(temp_audio_path)

        print(f"✓ Transcription complete in {elapsed_time:.1f}s")
        print(f"Original ({len(original_text)} chars): {original_text[:100]}...")
        print(f"Cleaned  ({len(cleaned_text)} chars):  {cleaned_text[:100]}...")
        print(f"Removed {len(original_text) - len(cleaned_text)} characters")

        return JSONResponse({
            "original":                original_text,
            "cleaned":                 cleaned_text,
            "is_snippet":              is_snippet_bool,
            "processing_time_seconds": round(elapsed_time, 2),
            "file_size_mb":            round(file_size_mb, 2)
        })

    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error":    str(e),
                "original": "",
                "cleaned":  ""
            }
        )


@app.get("/health")
def health_check():
    """Unity calls this on startup to confirm the voice service is running."""
    return {
        "status": "healthy",
        "model":  MODEL_SIZE,
        "ready":  True
    }


@app.get("/")
def root():
    """Root endpoint with basic usage info for debugging."""
    return {
        "service":  "Voice Transcription Agent",
        "model":    f"Whisper {MODEL_SIZE}",
        "endpoint": "/transcribe",
        "usage":    "POST audio file with 'file' and 'is_snippet' fields"
    }


# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Voice Transcription Server")
    print("Port: 5005")
    print("Model: Whisper " + MODEL_SIZE)
    print("="*60 + "\n")

    uvicorn.run(app, host="127.0.0.1", port=5005)
