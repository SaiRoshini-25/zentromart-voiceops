
"""Gnani-only speech adapter. Requires GNANI_API_KEY and gnani-vachana."""
from __future__ import annotations

from pathlib import Path


def voice_turn(
    audio_path: str,
    agent,
    output_path: str,
    language_code: str = "en-IN",
    reply_language: str | None = None,
    voice: str = "Kaveri",
):
    """Transcribe in ``language_code`` and synthesize in ``reply_language``."""
    try:
        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient
    except ImportError as exc:
        raise RuntimeError("Install gnani-vachana and set GNANI_API_KEY to use voice mode.") from exc
    try:
        heard = GnaniSTTClient().transcribe(audio_path, language_code=language_code).get("transcript", "")
    except Exception as exc:
        raise RuntimeError(
            f"Speech-to-text failed ({type(exc).__name__}): {exc}. Check your Gnani API key and audio format."
        ) from exc
    result = agent.respond(heard) if heard.strip() else agent.respond("")
    try:
        GnaniTTSClient().synthesize(result.reply, voice=voice, model="timbre-v2.5",
                                    language=reply_language or language_code, output_file=str(Path(output_path)))
    except Exception as exc:
        raise RuntimeError(
            f"Speech synthesis failed ({type(exc).__name__}): {exc}. Check your Gnani API key and voice-service access."
        ) from exc
    return heard, result
