# VoiceOps 2026: ZentroMart Customer Support Agent

**Capstone report | 24 October 2026 | Status: local prototype**
**Author:** Not supplied

## Abstract

This project builds a text-and-voice customer-support agent for ZentroMart, a fictional Indian retailer. It retrieves current policy sections, looks up synthetic orders through verification-gated tools, maintains short-lived conversation state, and creates human-handoff tickets. Chat works locally without paid API calls; an optional Gemini integration synthesizes answers, while Gnani STT/TTS powers the voice flow. The latest local regression and supplied practice evaluations pass; hosted model and live speech-provider behavior remain unverified.

## Steps 1–2: LLM and grounded retrieval

The agent uses a system prompt to constrain optional Gemini responses to retrieved company evidence and verified order facts. The RAG index chunks Markdown by `[SECTION-ID]`, filters superseded documents, and weights current policies above agent guides, help articles, FAQs, and marketing pages. If no Gemini response is available, chat returns guidance from retrieved sources instead of inventing an answer.

**Practice comparison (52 questions):** A Step 1 prompting-only LLM score was **not measured**: no valid baseline or Step 1-to-Step 2 accuracy comparison can be claimed. The offline Step 2 evaluation checks decisions, required and forbidden facts, every listed gold-evidence ID, and tool calls. **52/52 single questions passed all applicable checks (100%)**. All expected evidence IDs were retrieved for **47/47 cases with gold evidence (100%)**. This is a strict reproducible benchmark, not an LLM accuracy score or a hosted-LLM accuracy claim.

## Step 3: Conversation, tools, and evaluation

The agent remembers the active request, asks for an order ID and then one verification value, and limits failed identity checks to one retry. Browser sessions use separate opaque HTTP-only cookies. Ticket creation follows the mock tool result; the agent now escalates high-value damage claims, explicit human requests, third-or-later contacts, overdue refunds, unauthorized activity, legal complaints, and safety risks. Verification values are redacted from ticket summaries.

On the supplied **23 conversations / 96 turns**, **96/96 turns passed all applicable checks (100%)**. All listed expected evidence IDs were retrieved on **96/96 turns (100%)**. The expected tool type and order ID were present on **54/54 turns that specified a tool**. Ticket creation is mocked in evaluation so replay does not write to the project ticket CSV. These are strict deterministic checks against the supplied practice data, not overall conversation-accuracy scores. The regression suite also exercises the supplied manual smoke scenarios, including concise policy answers, headphone follow-ups, topic changes, order verification, privacy, and escalation.

## Step 4: Voice and batch run

The browser offers microphone recording and audio-file upload. `src/voice.py` uses Gnani STT and explicitly names the `timbre-v2.5` TTS model; no alternate speech provider is used. The required folder batch runner is:

```powershell
python run.py path\to\audio-folder --output outputs --language en-IN
```

For each audio file, it writes a JSON result containing transcript, reply, sources, and tool calls, plus a spoken WAV reply. `run.py --help` was verified. The hosted Gnani calls and a recorded-voice demo were **not run** in this validation; there are no supplied audio clips. A configured account/key and a self-recorded test clip are required to verify the complete speech path.

## Implementation, validation, and limits

The project uses FastAPI, Python, a static browser interface, Markdown policy retrieval, and CSV-backed mock tools. The supplied dataset validator passed: **9 CSVs, 21 knowledge-base documents, 1,000 orders, 1,783 order items, 52 questions, and 23 conversations**. **169 pytest cases pass**, with **86% overall source coverage**. Tests cover policy retrieval, typo-tolerant queries, concise manual-smoke responses, multi-turn follow-ups, privacy/verification, escalation, date handling, web validation, session isolation, voice-adapter behavior, evaluation output, and submission files.

The dataset is synthetic and uses **20 October 2026** as its reference date. The order checks are educational simulations, not production authentication. Session state is in-memory and limited to one process. The practice benchmark is deterministic and dataset-specific; no prompting-only baseline or live speech-provider run is available.

## What went wrong and next steps

The initial report exceeded the brief's four-page maximum and omitted the four-step rubric; this revised report is structured to the brief. The full evaluation artifacts are in `outputs/evaluation/questions.csv`, `outputs/evaluation/conversations.csv`, and `outputs/evaluation/summary.json`. Re-run with `python tools/evaluate_practice.py`; it disables Gemini for a deterministic offline evaluation and mocks ticket creation. Re-run tests with `python -m pytest -v` and coverage with `python -m pytest --cov=src --cov-report=term-missing`.

The next evaluation step is to measure a genuine prompting-only Step 1 baseline against the current Step 2 result. Also test Gnani STT/TTS using an authorized account and a self-recorded clip, record the requested two-minute demo, and verify the final public links.

## Submission status

The PDF deliverable is `report.pdf`; the runnable source and `.env.example` are in the project folder. The brief also requires a **public Git repository** and **demo video up to two minutes**. Neither has been published or supplied, and this workspace has no Git repository or remote. No public source, demo, or submission-portal link is available to claim. The supplied dataset link is [Capstone Project Dataset FTTT 2026](https://drive.google.com/file/d/1jeGjWgZJ4vdfmbB71kghovcCUYNeSHOY/view?usp=sharing); its public-sharing permission was not independently verified.
