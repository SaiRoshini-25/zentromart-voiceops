"""Batch runner required by the capstone: writes text, audio and RAG evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from src.agent import ZentroMartAgent
from src.voice import voice_turn

parser = argparse.ArgumentParser()
parser.add_argument("audio_folder", help="Folder containing customer audio files")
parser.add_argument("--output", default="outputs")
parser.add_argument("--language", default="en-IN")
parser.add_argument("--reply-language", default=None, help="Gnani TTS language; defaults to --language")
parser.add_argument("--voice", default="Kaveri", choices=("Kaveri", "Trupti", "Pranav", "Shlok"))
args = parser.parse_args()
out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
for audio in Path(args.audio_folder).glob("*.*"):
    agent = ZentroMartAgent()
    heard, result = voice_turn(
        str(audio), agent, str(out / f"{audio.stem}_reply.wav"), args.language, args.reply_language, args.voice
    )
    chunks = [
        {
            "section_id": chunk.section_id,
            "title": chunk.title,
            "source_document": chunk.source,
            "text": chunk.text,
        }
        for chunk in agent.kb.chunks
        if chunk.section_id in result.sources
    ]
    artifact = {
        "input_audio": audio.name,
        "transcript": heard,
        "reply": result.reply,
        "reply_audio": f"{audio.stem}_reply.wav",
        "source_section_ids": result.sources,
        "retrieved_chunks": chunks,
        "tool_calls": result.tool_calls,
    }
    (out / f"{audio.stem}.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
