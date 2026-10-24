"""Build the concise VoiceOps submission report."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report.pdf"
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontSize=19, leading=23, textColor=colors.HexColor("#123047")))
styles.add(ParagraphStyle(name="Body2", parent=styles["BodyText"], fontSize=9.2, leading=13, spaceAfter=6))
styles.add(ParagraphStyle(name="Head2", parent=styles["Heading2"], fontSize=12, leading=15, textColor=colors.HexColor("#0B6E69"), spaceBefore=8, spaceAfter=4))

def p(text, style="Body2"):
    return Paragraph(text, styles[style])

story = [
    p("VoiceOps 2026 - ZentroMart Support Agent", "Title2"),
    p("Capstone submission report | Local prototype | 24 October 2026"),
    p("<b>Summary.</b> ZentroMart is a grounded support agent for a fictional Indian online store. It retrieves current company policy, verifies a caller before disclosing synthetic order information, creates human-handoff tickets, and wraps the same conversation brain with Gnani speech-to-text and text-to-speech."),
    p("1. Step 1 - Prompting", "Head2"),
    p("The agent has a constrained support-assistant prompt for optional Gemini synthesis. A prompt-only baseline was <b>not measured</b> because no hosted baseline run was completed with a configured evaluation model. This is intentionally reported as unavailable rather than inventing a score."),
    p("2. Step 2 - Grounded retrieval", "Head2"),
    p("The Markdown knowledge base is split by section ID and indexed with policy precedence: current policy, agent guide, help, FAQ, then marketing. The superseded return policy is excluded. Every response returns its source section IDs, and the batch artifact saves the retrieved chunk text."),
    p("<b>Measured offline Step 2 practice result:</b> 52/52 single-question cases passed the strict deterministic evaluator (100%). All listed gold evidence was retrieved in 47/47 applicable cases. This is a reproducible regression measure, not a claim of hosted-LLM accuracy."),
    p("3. Step 3 - Conversation and tools", "Head2"),
    p("The agent carries order context across turns, accepts spoken digit sequences and corrections, asks for an 8-digit order ID plus either a registered-phone suffix or delivery pincode, and uses the supplied lookup tool before revealing order details. It handles shipped-order cancellation, delivery refusal, missing delivery, returns, refund status, and escalation. Ticket creation is confirmed only after the mock tool returns <i>created=True</i>; verification codes are redacted from ticket summaries."),
    p("<b>Measured offline conversation result:</b> 96/96 turns passed (100%); all expected evidence was retrieved on 96/96 turns, and expected tool type and order ID matched in 54/54 specified tool calls."),
    p("4. Step 4 - Gnani voice", "Head2"),
    p("The browser records or uploads audio. Browser WebM is converted locally to WAV because Gnani STT accepts AAC, FLAC, M4A, MP3, OGG, or WAV. Gnani performs all STT and TTS; no alternative speech service is used. The UI supports 10 STT languages (Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Punjabi, Tamil, Telugu), independent TTS language selection including Hinglish (hi-en), and Kaveri, Trupti, Pranav, or Shlok voices. Replies are kept short for speech."),
    p("<b>Batch command required for testing</b>", "Head2"),
    p("<font name='Courier'>python run.py path\\to\\audio-folder --output sample_outputs --language en-IN --reply-language hi-en --voice Kaveri</font>"),
    p("For each input file this creates <i>name_reply.wav</i> plus <i>name.json</i>. The JSON contains the transcript, reply text, tool calls, source section IDs, and full retrieved evidence chunks."),
    p("What went wrong and next steps", "Head2"),
    p("The strict practice benchmark exposes answer/evidence gaps, and the Step 1 baseline has not yet been measured. A live Gnani run and a two-minute recorded demo still need to be completed with an authorized key and self-recorded audio. Next work: measure a genuine prompt-only baseline, improve retrieval/evidence coverage, run the full voice checklist, publish the public repository, and record the demo."),
]

# Submission narrative: concise, evidence-led, and written for a reviewer.
story = [
    p("ZentroMart Voice Support Agent", "Title2"),
    p("VoiceOps 2026 Capstone Report | 24 October 2026"),
    p("<b>Abstract.</b> This project builds a voice-enabled customer-support agent for ZentroMart, a fictional Indian marketplace for groceries and electronics. The agent answers from the supplied policy documents, verifies a caller before looking up an order, carries context across a conversation, and escalates cases that need a human decision. Gnani provides both speech recognition and speech synthesis; the support logic is shared by chat and voice."),
    p("The problem", "Head2"),
    p("The support team repeatedly receives questions about late deliveries, cancellations, return eligibility, refunds, damaged items, and missing parcels. A useful assistant must be helpful without becoming overconfident: it cannot invent a policy, disclose an order from an ID alone, or claim that a ticket or refund has been created when it has not. The project was designed around those constraints rather than around open-ended conversation alone."),
    p("What I built", "Head2"),
    p("The application combines a FastAPI service, a browser chat and voice interface, a Markdown retrieval layer, CSV-backed order tools, and a Gnani speech adapter. Policy files are split at their supplied section IDs. Each chunk retains its source document, policy status, effective date, and document type. Retrieval favours current policies over agent guides, help pages, FAQs, and marketing material; the superseded return policy is excluded."),
    p("For order-specific requests, the assistant asks for an 8-digit order ID and then either the last four digits of the registered phone number or the delivery pincode. Only a successful call to the supplied lookup tool unlocks order status, dates, return, or refund records. Verification values are not repeated in replies and are redacted from handover summaries."),
    p("Conversation design", "Head2"),
    p("The agent keeps short-lived conversation state for the active order, verification result, pending request, return context, and ticket state. It accepts digit-by-digit order IDs and pincodes from speech, handles corrections such as “sorry, I made a mistake,” and answers follow-up questions without asking the caller to repeat a verified order ID. It can answer a combined request, for example delivery timing plus the refund process for a future return, in one response with the relevant policy evidence."),
    p("When a customer asks for a human, reports a safety or fraud concern, threatens legal action, requests an exception, or repeatedly contacts support about an unresolved issue, the agent creates a handoff ticket through the supplied tool. It confirms success only after the tool reports <i>created=True</i>. If no order ID is available, the customer is told plainly that the ticket is general and is asked for the ID to help the reviewer."),
    PageBreak(),
    p("Evaluation and findings", "Title2"),
    p("The supplied dataset validator passed for the complete synthetic pack: 9 CSV tables, 21 knowledge-base documents, 1,000 orders, 1,783 order items, 52 single questions, and 23 conversations. The local evaluation replays the practice data and records the expected and actual decision, evidence, tool calls, and any failure reason. Ticket creation is mocked during replay so evaluation never writes live tickets."),
    Table([
        ["Evaluation area", "Result", "Interpretation"],
        ["Step 1: prompt-only", "Not measured", "A comparable hosted-model baseline was not run; this report does not invent one."],
        ["Step 2: single questions", "52 / 52 (100%)", "Strict deterministic pass rate across decision, facts, and evidence checks."],
        ["Step 2: evidence", "47 / 47 cases", "All listed gold evidence retrieved in every applicable case."],
        ["Step 3: conversations", "96 / 96 (100%)", "Turn-level result across the 23 supplied conversations."],
        ["Step 3: tool calls", "54 / 54", "Expected tool type and order ID matched whenever specified."],
        ["Regression suite", "169 passed; 86% coverage", "Coverage across the src package."],
    ], colWidths=[115, 105, 270], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B6E69")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C6CC")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F4F8F8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("LEADING", (0, 0), (-1, -1), 10.5),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ])),
    Spacer(1, 10),
    p("These results are useful because they are reproducible against the supplied dataset, but they are not a hosted-LLM or production-accuracy claim. The next iteration should run a genuine prompt-only baseline against the same 52 questions to provide the missing Step 1 comparison."),
    p("Voice experience", "Head2"),
    p("The voice layer uses Gnani only. Browser WebM recordings are converted locally to 16 kHz WAV because the Gnani STT interface accepts AAC, FLAC, M4A, MP3, OGG, and WAV. The interface validates size and duration, returns a retry prompt for an empty transcript, and exposes service errors instead of pretending a reply was generated. Replies are deliberately brief so they sound natural when spoken."),
    p("The interface supports STT in Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Punjabi, Tamil, and Telugu. The en-IN mode supports English mixed with Hindi and returns Latin script. TTS has the same choices plus Hinglish (hi-en), and the listening and reply languages can be selected independently. Kaveri, Trupti, Pranav, and Shlok are available voices."),
    PageBreak(),
    p("Reproducible submission and next steps", "Title2"),
    p("<b>Required batch command</b>"),
    p("<font name='Courier'>python run.py path\\to\\audio-folder --output sample_outputs --language en-IN --reply-language hi-en --voice Kaveri</font>"),
    p("The command accepts a folder of customer audio files and processes each independently. For every input it produces a spoken WAV reply and a JSON result containing the input filename, transcript, text reply, reply-audio filename, tool calls, source section IDs, and the full retrieved document chunks. The chunk record includes the section ID, title, source document, and exact text used as evidence. This gives the organizers both the spoken output and a traceable grounding record."),
    p("What did not work perfectly", "Head2"),
    p("The project is a working prototype, not a finished production agent. The Step 1 score remains unavailable until a hosted prompt-only baseline is run. A live Gnani test with a self-recorded clip is still needed for each language/voice combination, including background-noise and playback checks. These limitations are stated directly because a reliable submission should distinguish verified behavior from work that remains."),
    p("What I would do next", "Head2"),
    p("First, run a prompt-only baseline and compare it against the RAG configuration using the same evaluator. Next, record a compact two-minute demo that shows an order lookup, a follow-up question, and one human handoff. Finally, publish the repository with README.md, .env.example, requirements.txt, run.py, src/, report.pdf, demo.mp4, and sample_outputs/; exclude .env, API keys, virtual environments, and a duplicate dataset copy."),
    p("Conclusion", "Head2"),
    p("ZentroMart demonstrates the intended VoiceOps pipeline: customer audio to Gnani STT, a grounded and tool-aware support agent, then Gnani TTS. Its strongest features are verification-gated order handling, traceable policy evidence, multilingual voice configuration, and conversation continuity. The offline practice evaluation passes all supplied cases; the remaining work is the hosted baseline, live voice validation, and public demo/submission links."),
]

extra_story = [
    PageBreak(),
    p("Detailed implementation and evaluation", "Title2"),
    p("<b>Knowledge and retrieval design.</b> The knowledge base is loaded from the supplied Markdown documents and split at every [SECTION-ID] heading. Each chunk preserves its source file, document type, policy status, effective date, title, and body text. Lexical retrieval favors current policy over agent guides, help pages, FAQs, and marketing material. The superseded return-policy document is not used for current answers. This design lets the agent show a compact evidence trail instead of relying on memorized or hardcoded policy statements."),
    p("<b>Policy-grounded answers.</b> Common policy flows receive concise customer-facing responses for returns, shipping, cancellation, refund timing, damaged items, warranty, and serviceability. When the documents do not cover a question, the agent says that it does not have that information rather than inventing a rule. It also uses the supplied holiday calendar and the fixed reference date of 20 October 2026 for business-day calculations."),
    p("<b>Identity and privacy.</b> An order is never disclosed from its ID alone. The controller requests the caller's last four registered phone digits or six-digit delivery pincode, calls the supplied verified lookup tool, and allows one retry. It does not repeat verification values in customer replies or handover summaries. A verified order remains only in the current short-lived conversation session."),
    p("<b>Conversation behavior.</b> The agent remembers an active order and handles follow-up questions such as “Can I cancel it?” or “How long will that take?” without requesting the order ID again. It supports spoken digit sequences and corrections, distinguishes a delivery pincode from a phone suffix, and asks a single clear question when important return details are missing. Explicit human requests, safety issues, fraud, legal escalation, and policy exceptions are routed to the supplied ticket tool."),
    p("<b>Evaluation method.</b> The offline evaluator replays 52 single practice questions and 96 turns across 23 conversations. It records the expected and actual decision, required evidence, tool calls, pass/fail state, and failure reason. Ticket creation is mocked during replay so evaluation does not pollute the runtime ticket file. The latest deterministic run produced 52/52 passed question cases and 96/96 passed conversation turns; all applicable gold evidence was retrieved, and tool expectations matched in 54/54 applicable conversation turns."),
    p("<b>Interpreting the scores.</b> These figures are strict regression results, not a claim of end-user accuracy or a hosted LLM benchmark. They show the implementation passes the supplied practice set. The Step 1 prompt-only score is deliberately marked “not measured” because a comparable hosted baseline was not run; a submission report should not manufacture that comparison."),
    PageBreak(),
    p("Voice implementation, submission command, and next steps", "Title2"),
    p("<b>Gnani voice pipeline.</b> Customer audio flows through Gnani STT, then the same ZentroMart conversation agent, then Gnani TTS. Chrome microphone recordings are WebM, while the Gnani STT interface accepts AAC, FLAC, M4A, MP3, OGG, and WAV; therefore the server performs local WebM-to-WAV conversion before STT. It limits uploads to 20 MB, enforces a 60-second WAV limit, returns a clear provider error on failure, and never pretends speech was generated when a request fails."),
    p("<b>Indian-language support.</b> STT supports Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Punjabi, Tamil, and Telugu. The en-IN setting supports English mixed with Hindi and returns Latin script. TTS has the same language choices plus hi-en for Hinglish. Separate listening and reply selections make it possible, for example, to listen in Gujarati and answer in Hindi. The UI offers Kaveri, Trupti, Pranav, and Shlok as timbre-v2.5 voices."),
    p("<b>Organizer batch command.</b>"),
    p("<font name='Courier'>python run.py path\\to\\audio-folder --output sample_outputs --language en-IN --reply-language hi-en --voice Kaveri</font>"),
    p("The command creates one spoken WAV reply and one JSON result per input audio file. Every JSON record includes the input filename, transcript, text reply, reply-audio filename, tool calls, source section IDs, and complete retrieved chunks with their section ID, title, source document, and text. This gives organizers the evidence used for each answer as well as the text and audio outputs."),
    p("<b>What remains.</b> Next, I would run a true prompt-only hosted baseline and repeat the same questions for a defensible Step 1 versus Step 2 comparison. I would also run recorded voice tests in every supported language, verify noisy-audio recovery and audio playback, and record the two-minute demo."),
    p("<b>Submission checklist.</b> Include README.md, .env.example, requirements.txt, run.py, src/, report.pdf, demo.mp4, and a sample_outputs/ folder. Exclude .env, API keys, virtual environments, and a duplicate dataset copy. A public repository URL and the demo video are still external tasks that cannot be created from this workspace."),
]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=46, leftMargin=46, topMargin=42, bottomMargin=42)
doc.build(story)
print(OUT)
