"""Offline evaluation of the supplied single-question and conversation sets."""
from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

from .agent import ZentroMartAgent

PRACTICE_DIR = Path("Capstone_Project_Dataset_FTTT_2026") / "practice"
OUTPUT_FIELDS = (
    "case_id",
    "expected_decision",
    "actual_decision",
    "gold_evidence",
    "retrieved_section_ids",
    "tool_calls_expected",
    "tool_calls_actual",
    "passed",
    "failure_reason",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _evidence_ids(value: str) -> list[str]:
    return [
        evidence.strip()
        for evidence in value.split(";")
        if evidence.strip()
    ]


def _actual_decision(result) -> str:
    if "create_ticket()" in result.tool_calls:
        return "escalate"
    reply = result.reply.casefold().replace("’", "'").replace("‘", "'")
    if (
        "ESC-4" in result.sources
        or "can't share" in reply
        or "cannot share" in reply
        or "can't disclose" in reply
        or "can't comply" in reply
        or re.search(r"\b(?:i|we)\s+(?:can't|cannot)\s+(?:cancel|offer|approve|share)\b", reply)
    ):
        return "refuse"
    if "IDV-2" in result.sources and "verified" in reply:
        return "answer"
    if re.search(r"\bwhich\b|\bwhat .{0,30}\?", reply) and "?" in reply:
        return "clarify"
    if any(section in result.sources for section in ("IDV-1", "IDV-2", "STY-1", "STY-5")):
        return "clarify"
    return "answer"


def _contains_fact(reply: str, fact: str) -> bool:
    def normalize(value: str) -> str:
        return re.sub(r"[^\w]+", "", value.casefold(), flags=re.UNICODE)

    normalized_reply = normalize(reply)
    return any(
        normalize(option) in normalized_reply
        for option in fact.split("/")
        if normalize(option)
    )


def _tool_expectations_met(expected: str, actual: list[str]) -> bool:
    for invocation in filter(None, (part.strip() for part in expected.split(";"))):
        if "lookup_order" in invocation:
            order_match = re.search(r"order_id=(\d+)", invocation)
            matching_calls = [call for call in actual if "lookup_order" in call]
            if not matching_calls:
                return False
            if order_match and not any(order_match.group(1) in call for call in matching_calls):
                return False
        elif "create_ticket" in invocation and "create_ticket()" not in actual:
            return False
    return True


def _compare(row: dict[str, str], result, is_conversation: bool) -> dict:
    expected = row["expected_decision"]
    actual = _actual_decision(result)
    accepted = (
        {item.strip() for item in row.get("acceptable_decisions", "").split(";") if item.strip()}
        if not is_conversation
        else {expected}
    )
    failures = []
    if actual not in accepted:
        failures.append(f"decision expected {sorted(accepted)}, got {actual}")

    for fact in filter(None, (part.strip() for part in row.get("must_include", "").split(";"))):
        if not _contains_fact(result.reply, fact):
            failures.append(f"missing required fact: {fact}")
    for forbidden in filter(None, (part.strip() for part in row.get("must_not_include", "").split(";"))):
        if _contains_fact(result.reply, forbidden):
            failures.append(f"contains forbidden claim: {forbidden}")

    gold = _evidence_ids(row.get("gold_evidence", ""))
    missing_evidence = [
        evidence
        for evidence in gold
        if evidence not in result.sources
    ]
    if missing_evidence:
        failures.append("missing gold evidence: " + ", ".join(missing_evidence))

    expected_tools = row.get("expected_tool_calls", "")
    if not _tool_expectations_met(expected_tools, result.tool_calls):
        failures.append("expected tool call missing or order ID mismatched")

    case_id = row.get("id") or f"{row.get('conversation_id')}:{row.get('turn')}"
    return {
        "case_id": case_id,
        "expected_decision": expected,
        "actual_decision": actual,
        "gold_evidence": row.get("gold_evidence", ""),
        "retrieved_section_ids": "; ".join(result.sources),
        "tool_calls_expected": expected_tools,
        "tool_calls_actual": "; ".join(result.tool_calls),
        "passed": not failures,
        "failure_reason": "; ".join(failures),
    }


def _run_questions(rows: list[dict[str, str]], data_dir: str | Path) -> list[dict]:
    output = []
    for row in rows:
        agent = ZentroMartAgent(data_dir)
        result = agent.respond(row["question"])
        output.append(_compare(row, result, is_conversation=False))
    return output


def _run_conversations(rows: list[dict[str, str]], data_dir: str | Path) -> list[dict]:
    conversations = defaultdict(list)
    for row in rows:
        conversations[row["conversation_id"]].append(row)

    output = []
    for conversation_id, turns in conversations.items():
        agent = ZentroMartAgent(data_dir)
        for row in sorted(turns, key=lambda turn: int(turn["turn"])):
            result = agent.respond(row["customer_message"])
            output.append(_compare(row, result, is_conversation=True))
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_practice(
    data_dir: str | Path = "data",
    practice_dir: str | Path = PRACTICE_DIR,
    output_dir: str | Path = "outputs/evaluation",
) -> dict:
    """Run all 52 questions and 96 ordered turns, writing CSVs and a summary."""
    practice_dir, output_dir = Path(practice_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
        with patch(
            "src.agent.create_ticket",
            return_value={"created": True, "ticket_id": "EVAL"},
        ):
            questions = _run_questions(_read_rows(practice_dir / "questions.csv"), data_dir)
            conversations = _run_conversations(
                _read_rows(practice_dir / "conversations.csv"), data_dir
            )
    _write_csv(output_dir / "questions.csv", questions)
    _write_csv(output_dir / "conversations.csv", conversations)

    summary = {
        "mode": "offline deterministic agent; ticket creation mocked for replay",
        "prompt_only_baseline": "not measured; no baseline LLM run was performed",
        "questions": {
            "total": len(questions),
            "passed": sum(row["passed"] for row in questions),
            "failed": sum(not row["passed"] for row in questions),
        },
        "conversation_turns": {
            "total": len(conversations),
            "passed": sum(row["passed"] for row in conversations),
            "failed": sum(not row["passed"] for row in conversations),
            "conversations": len({row["case_id"].split(":")[0] for row in conversations}),
        },
        "decision_counts": dict(Counter(row["actual_decision"] for row in questions)),
        "question_gold_evidence": {
            "expected_ids": sum(bool(row["gold_evidence"]) for row in questions),
            "all_ids_retrieved": sum(
                bool(row["gold_evidence"])
                and all(
                    evidence in row["retrieved_section_ids"].split("; ")
                    for evidence in _evidence_ids(row["gold_evidence"])
                )
                for row in questions
            ),
        },
        "conversation_gold_evidence": {
            "expected_ids": sum(bool(row["gold_evidence"]) for row in conversations),
            "all_ids_retrieved": sum(
                bool(row["gold_evidence"])
                and all(
                    evidence in row["retrieved_section_ids"].split("; ")
                    for evidence in _evidence_ids(row["gold_evidence"])
                )
                for row in conversations
            ),
        },
        "conversation_tool_calls": {
            "expected": sum(bool(row["tool_calls_expected"]) for row in conversations),
            "matched": sum(
                bool(row["tool_calls_expected"])
                and _tool_expectations_met(
                    row["tool_calls_expected"],
                    row["tool_calls_actual"].split("; "),
                )
                for row in conversations
            ),
        },
        "artifacts": ["questions.csv", "conversations.csv"],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary
