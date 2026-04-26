@echo off
echo Launching VR Classroom Agents...

:: Agent.py - Student Agent (Port 5002)
start "Agent - Student (5002)" cmd /k "cd /d "%~dp0" && call .\venv\Scripts\activate && python Agent.py"

:: Agent_TTS.py - TTS Service (Port 5003)
start "Agent TTS (5003)" cmd /k "cd /d "%~dp0" && call .\venv\Scripts\activate && python Agent_TTS.py"

:: Agent_Voice.py - Voice Transcription (Port 5005)
start "Agent Voice - Whisper (5005)" cmd /k "cd /d "%~dp0" && call .\venv\Scripts\activate && python Agent_Voice.py"

echo All 3 agents launched.

:: ============================================================
:: FIRST-TIME SETUP (run these once manually in your venv):
::
::   python -m venv venv
::   .\venv\Scripts\activate
::   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
::   pip install -r requirements.txt
:: ============================================================