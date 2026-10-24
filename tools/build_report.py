"""Build the concise VoiceOps submission report."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


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
    p("<b>Measured offline Step 2 practice result:</b> 15/52 single-question cases passed the strict deterministic evaluator (28.8%). Evidence coverage was 18/47 cases with listed gold evidence. This is a reproducible regression measure, not a claim of hosted-LLM accuracy."),
    p("3. Step 3 - Conversation and tools", "Head2"),
    p("The agent carries order context across turns, accepts spoken digit sequences and corrections, asks for an 8-digit order ID plus either a registered-phone suffix or delivery pincode, and uses the supplied lookup tool before revealing order details. It handles shipped-order cancellation, delivery refusal, missing delivery, returns, refund status, and escalation. Ticket creation is confirmed only after the mock tool returns <i>created=True</i>; verification codes are redacted from ticket summaries."),
    p("<b>Measured offline conversation result:</b> 68/96 turns passed (70.8%); expected tool type and order ID matched in 54/54 specified tool calls. Evidence coverage was 73/96 turns."),
    p("4. Step 4 - Gnani voice", "Head2"),
    p("The browser records or uploads audio. Browser WebM is converted locally to WAV because Gnani STT accepts AAC, FLAC, M4A, MP3, OGG, or WAV. Gnani performs all STT and TTS; no alternative speech service is used. The UI supports 10 STT languages (Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Punjabi, Tamil, Telugu), independent TTS language selection including Hinglish (hi-en), and Kaveri, Trupti, Pranav, or Shlok voices. Replies are kept short for speech."),
    p("<b>Batch command required for testing</b>", "Head2"),
    p("<font name='Courier'>python run.py path\\to\\audio-folder --output sample_outputs --language en-IN --reply-language hi-en --voice Kaveri</font>"),
    p("For each input file this creates <i>name_reply.wav</i> plus <i>name.json</i>. The JSON contains the transcript, reply text, tool calls, source section IDs, and full retrieved evidence chunks."),
    p("What went wrong and next steps", "Head2"),
    p("The strict practice benchmark exposes answer/evidence gaps, and the Step 1 baseline has not yet been measured. A live Gnani run and a two-minute recorded demo still need to be completed with an authorized key and self-recorded audio. Next work: measure a genuine prompt-only baseline, improve retrieval/evidence coverage, run the full voice checklist, publish the public repository, and record the demo."),
]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=46, leftMargin=46, topMargin=42, bottomMargin=42)
doc.build(story)
print(OUT)
