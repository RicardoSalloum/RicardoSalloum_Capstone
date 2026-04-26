#!/usr/bin/env python3
"""
Agent_TTS.py - TTS Service for VR Classroom
Port: 5003

Wraps the Kokoro neural TTS model in a FastAPI server so Unity can request
spoken audio by POSTing text to /synthesize. The response is a raw WAV file.
"""

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
import io
import sys
import wave
import numpy as np
import os

# Create the FastAPI app that will handle HTTP requests coming from Unity
app = FastAPI(title="TTS Service for VR Classroom")

print("\n" + "="*60)
print("Initializing TTS Engine...")
print("="*60)

# Try to load Kokoro; if the package isn't installed we can't run at all,
# so we exit immediately with a clear error message
try:
    from kokoro import KPipeline
    print("Loading Kokoro PyTorch model...")

    # 'a' is Kokoro's language code for American English
    kokoro_model = KPipeline(lang_code='a')
    TTS_ENGINE = "kokoro_pytorch"
    print("✓ Using Kokoro PyTorch (High-quality neural TTS)")

    # Try to print how many voices the loaded model exposes
    try:
        voices = kokoro_model.get_voices()
        if hasattr(voices, '__len__'):
            print(f"  Available voices: {len(voices)}")
        else:
            print(f"  Multiple voices available")
    except:
        # Voice listing is optional info, so we just skip it on failure
        print(f"  Voice information unavailable")

except ImportError as e:
    print(f"✗ Kokoro PyTorch not available: {e}")
    print("  Install with: pip install kokoro")
    sys.exit(1)
except Exception as e:
    print(f"✗ Kokoro PyTorch failed to load: {e}")
    sys.exit(1)

print("="*60 + "\n")


# Pydantic validates incoming JSON from Unity against this model
class TTSRequest(BaseModel):
    text: str
    voice: str = "af_bella"   # default voice if Unity doesn't send one


def kokoro_pytorch_generate(text: str, voice: str = "af_bella") -> bytes:
    """
    Runs the Kokoro model to synthesize speech for the given text,
    then packs the output float samples into a proper WAV file.
    Returns the entire WAV as bytes so FastAPI can stream it back to Unity.
    """
    global kokoro_model

    # Kokoro uses short internal codes like "af_bella".
    # This map lets Unity send readable names and we translate them here.
    voice_map = {
        "af_bella":    "af_bella",
        "af_sarah":    "af_sarah",
        "af":          "af",
        "am_adam":     "am_adam",
        "am_michael":  "am_michael",
        "am":          "am",
        "bf_emma":     "bf_emma",
        "bf_isabella": "bf_isabella",
        "bf":          "bf",
        "bm_george":   "bm_george",
        "bm_lewis":    "bm_lewis",
        "bm":          "bm"
    }

    # If the requested voice isn't in the map, fall back to the generic American voice
    kokoro_voice = voice_map.get(voice, "af")

    # Kokoro returns a generator; each iteration gives one audio chunk
    generator = kokoro_model(text, voice=kokoro_voice)

    # Accumulate every chunk so we can concatenate them at the end
    all_audio = []
    for graphemes, phonemes, audio_chunk in generator:
        all_audio.append(audio_chunk)

    # Merge all chunks into one contiguous array of float samples
    audio_samples = np.concatenate(all_audio)

    # If any sample falls outside [-1, 1], normalize the whole array
    # so we don't clip when converting to 16-bit PCM
    if audio_samples.max() > 1.0 or audio_samples.min() < -1.0:
        audio_samples = audio_samples / np.abs(audio_samples).max()

    # Scale floats to the 16-bit integer range (-32768 to 32767)
    audio_data = (audio_samples * 32767).astype(np.int16)

    # Write a complete WAV file into a memory buffer instead of disk
    sample_rate = 24000   # Kokoro outputs at 24 kHz
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)       # mono audio
        wav_file.setsampwidth(2)       # 2 bytes per sample (16-bit)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    # Rewind so the caller reads from the start
    wav_buffer.seek(0)
    return wav_buffer.read()


@app.post("/synthesize")
def synthesize_speech(data: TTSRequest):
    """
    Unity POSTs { text, voice } here and receives a WAV file in the body.
    The X-TTS-Engine header tells Unity which backend produced the audio.
    """
    try:
        audio_data = kokoro_pytorch_generate(data.text, data.voice)
        print(f"[Kokoro PyTorch] \"{data.text[:40]}...\" ({data.voice})")

        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "X-TTS-Engine": TTS_ENGINE
            }
        )
    except Exception as e:
        print(f"[TTS ERROR] {e}")
        import traceback
        traceback.print_exc()

        # Return an empty body with status 500 so Unity knows synthesis failed
        return Response(
            content=b"",
            status_code=500,
            headers={"X-Error": str(e)}
        )


@app.get("/health")
def health_check():
    """Ping endpoint — Unity calls this to confirm the TTS service is alive."""
    voices_list = [
        "af_bella", "af_sarah", "af",
        "am_adam", "am_michael", "am",
        "bf_emma", "bf_isabella", "bf",
        "bm_george", "bm_lewis", "bm"
    ]

    return {
        "status": "healthy",
        "tts_engine": TTS_ENGINE,
        "available_voices": voices_list,
        "voice_count": len(voices_list)
    }


@app.get("/voices")
def list_voices():
    """Returns the full voice catalogue grouped by accent and gender."""
    return {
        "engine": TTS_ENGINE,
        "voices": {
            "American Female": ["af_bella", "af_sarah", "af"],
            "American Male":   ["am_adam", "am_michael", "am"],
            "British Female":  ["bf_emma", "bf_isabella", "bf"],
            "British Male":    ["bm_george", "bm_lewis", "bm"]
        }
    }


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("TTS Service for VR Classroom | Port: 5003")
    print(f"Engine: {TTS_ENGINE.upper()}")
    print("✓ High-quality neural TTS ready!")
    print("  Multiple voices available")
    print("  No internet required")
    print("="*60 + "\n")

    # Bind to localhost only; Unity on the same machine connects to this port
    uvicorn.run(app, host="127.0.0.1", port=5003)
