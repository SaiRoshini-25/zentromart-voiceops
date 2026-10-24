
# VoiceOps 2026 - ZentroMart support agent

A grounded customer-support agent for the VoiceOps capstone. It searches the supplied current policy documents, preserves section IDs, verifies callers before revealing order data, creates tickets for human-only decisions, and supports Gnani STT/TTS.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
uvicorn src.main:app --reload
```

Open `http://127.0.0.1:8000`. Chat works without credentials and gives concise answers grounded in the current support documents. Add a real `GEMINI_API_KEY` for synthesized answers and a `GNANI_API_KEY` before using voice.

## Submission configuration

Include `.env.example` in the submission ZIP, never `.env`. The example contains placeholder values only; each evaluator should copy it to a local, untracked `.env` and provide their own keys.

Run the offline regression checks from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m pytest --cov=src --cov-report=term-missing
.\.venv\Scripts\python.exe tools\evaluate_practice.py
.\.venv\Scripts\python.exe Capstone_Project_Dataset_FTTT_2026\validate_dataset.py
```

The practice evaluator saves one row per single question and conversation turn, with expected/actual decisions, evidence, tool calls, pass status, and failure reasons under `outputs/evaluation/`. It replays each conversation in turn order and mocks ticket creation to avoid writing evaluation tickets to the live ticket CSV. No prompt-only LLM baseline is included unless one is measured separately.

See [report.pdf](report.pdf) for the capstone report, validation results, and limitations. The voice endpoint requires Gnani credentials; hosted STT/TTS was not exercised during local validation.

The voice UI caps microphone recordings at 60 seconds, checks uploaded audio duration in the browser when available, and supports Gnani STT in Bengali, English/Hindi-English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Punjabi, Tamil, and Telugu. Its independent TTS selector supports those languages plus Hinglish (`hi-en`), so callers can speak one supported language and receive another. WAV duration is validated server-side, and all voice uploads are limited to 20 MB. Hosted STT/TTS requires valid Gnani credentials.

## Batch audio run

```powershell
python run.py path\to\audio-folder --output sample_outputs --language en-IN --reply-language hi-en --voice Kaveri
```

This is the submission batch command. It accepts a folder of customer audio files and, for every input, writes a spoken WAV reply plus a JSON artifact containing the transcript, text reply, tool calls, source section IDs, and the full retrieved document chunks (section ID, title, source document, and text). Use `--reply-language` only when the spoken reply should differ from the STT language; otherwise omit it.

## Design notes

- `src/rag.py` indexes current Markdown policy sections and favors policies over secondary documents.
- `src/rules.py` contains deterministic business rules for verified order facts, including refund-record interpretation; a missing completion date is always treated as pending.
- `src/agent.py` keeps short-lived conversation state, enforces caller verification, and only confirms ticket creation after the tool succeeds.
- `src/main.py` isolates browser conversations with an HTTP-only session cookie; session state is in-memory and intended for local demonstration only.
- `src/voice.py` uses Gnani only; there is no alternate speech provider.
- `run.py` accepts an audio folder and writes a transcript/reply/evidence JSON file and a spoken WAV reply for each audio file.
- Dataset files are synthetic and are included locally under `data/` for reproducible development.
