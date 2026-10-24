
from __future__ import annotations

import secrets
import subprocess
import threading
import wave
import io
from collections import OrderedDict
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import ZentroMartAgent
from .voice import voice_turn

app = FastAPI(title="VoiceOps 2026 - ZentroMart")
app.mount("/static", StaticFiles(directory="static"), name="static")
VOICE_OUTPUTS = Path("outputs/voice")
VOICE_OUTPUTS.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
SESSION_AGENTS: OrderedDict[str, ZentroMartAgent] = OrderedDict()
SESSION_LOCK = threading.Lock()
MAX_SESSIONS = 256
MAX_AUDIO_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_AUDIO_SECONDS = 60
STT_LANGUAGES = {"bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN", "pa-IN", "ta-IN", "te-IN"}
TTS_LANGUAGES = STT_LANGUAGES | {"hi-en"}
TTS_VOICES = {"Kaveri", "Trupti", "Pranav", "Shlok"}


def _webm_to_wav(source: Path, destination: Path) -> None:
    """Convert browser-recorded WebM audio into a Gnani STT input format."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "WebM conversion is unavailable. Install the project dependencies and try again."
        ) from exc
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", str(destination)],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError("I could not convert that WebM recording. Please record it again.") from exc


class ChatRequest(BaseModel):
    message: str


def _session_agent(request: Request, response: Response) -> ZentroMartAgent:
    session_id = request.cookies.get("voiceops_session", "")
    if len(session_id) > 128:
        session_id = ""

    with SESSION_LOCK:
        agent = SESSION_AGENTS.get(session_id)
        if agent is None:
            session_id = secrets.token_urlsafe(32)
            agent = ZentroMartAgent()
            SESSION_AGENTS[session_id] = agent
            while len(SESSION_AGENTS) > MAX_SESSIONS:
                SESSION_AGENTS.popitem(last=False)
        else:
            SESSION_AGENTS.move_to_end(session_id)

    response.set_cookie(
        "voiceops_session",
        session_id,
        max_age=3600,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return agent


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/api/chat")
def chat(payload: ChatRequest, http_request: Request, response: Response):
    result = _session_agent(http_request, response).respond(payload.message)
    return {"reply": result.reply, "sources": result.sources, "tool_calls": result.tool_calls}


@app.post("/api/reset")
def reset(http_request: Request, response: Response):
    _session_agent(http_request, response).reset()
    return {"ok": True}


@app.post("/api/voice")
async def voice(
    http_request: Request,
    response: Response,
    audio: UploadFile = File(...),
    language: str = "en-IN",
    reply_language: str | None = None,
    voice: str = "Kaveri",
):
    """Gnani STT -> the same conversational agent -> Gnani TTS."""
    if language not in STT_LANGUAGES:
        raise HTTPException(status_code=422, detail="Unsupported speech-recognition language.")
    if reply_language is not None and reply_language not in TTS_LANGUAGES:
        raise HTTPException(status_code=422, detail="Unsupported speech-output language.")
    if voice not in TTS_VOICES:
        raise HTTPException(status_code=422, detail="Unsupported Gnani voice.")
    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    # Chromium's MediaRecorder emits WebM, but Gnani STT accepts WAV rather
    # than WebM. Convert it locally before it reaches the speech provider.
    supported_audio = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}
    if suffix.lower() not in supported_audio:
        raise HTTPException(
            status_code=422,
            detail="Unsupported audio format. Upload WAV, MP3, M4A, OGG, AAC, FLAC, or WebM.",
        )
    token = uuid4().hex
    incoming = VOICE_OUTPUTS / f"{token}{suffix}"
    converted_audio = VOICE_OUTPUTS / f"{token}_input.wav"
    reply_audio = VOICE_OUTPUTS / f"{token}_reply.wav"
    audio_data = await audio.read(MAX_AUDIO_UPLOAD_BYTES + 1)
    if len(audio_data) > MAX_AUDIO_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio file exceeds the 20 MB upload limit.")
    if suffix.lower() == ".wav":
        try:
            with wave.open(io.BytesIO(audio_data), "rb") as wav_file:
                duration = wav_file.getnframes() / wav_file.getframerate()
        except (wave.Error, EOFError, ZeroDivisionError) as exc:
            raise HTTPException(status_code=422, detail="The WAV audio file is invalid.") from exc
        if duration > MAX_AUDIO_SECONDS:
            raise HTTPException(status_code=422, detail="Audio must be no longer than 60 seconds.")
    with incoming.open("wb") as stream:
        stream.write(audio_data)
    try:
        speech_input = incoming
        if suffix.lower() == ".webm":
            _webm_to_wav(incoming, converted_audio)
            speech_input = converted_audio
        heard, result = voice_turn(
            str(speech_input), _session_agent(http_request, response), str(reply_audio), language, reply_language, voice
        )
    except Exception as exc:
        incoming.unlink(missing_ok=True)
        raise HTTPException(
            status_code=503,
            detail=str(exc) if isinstance(exc, RuntimeError) else f"Voice provider request failed ({type(exc).__name__}).",
        ) from exc
    finally:
        if incoming.exists():
            incoming.unlink()
        if converted_audio.exists():
            converted_audio.unlink()
    return {"transcript": heard, "reply": result.reply, "sources": result.sources,
            "audio_url": f"/outputs/voice/{reply_audio.name}"}
