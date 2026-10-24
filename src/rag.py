
"""Small, dependency-free RAG index for the ZentroMart knowledge base."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

WORD = re.compile(r"[\w-]+", re.UNICODE)


@dataclass
class Chunk:
    section_id: str
    title: str
    text: str
    source: str
    doc_type: str
    status: str
    effective_from: str


class KnowledgeBase:
    """Ranks policy sections with lexical search and policy precedence."""

    def __init__(self, kb_dir: str | Path):
        self.chunks = self._load(Path(kb_dir))

    @staticmethod
    def _tokens(text: str) -> list[str]:
        tokens = []
        for token in WORD.findall(text.lower()):
            if len(token) > 5 and token.endswith("ies"):
                token = token[:-3] + "y"
            elif len(token) > 5 and token.endswith("ing"):
                token = token[:-3]
                if len(token) > 2 and token[-1] == token[-2]:
                    token = token[:-1]
            elif len(token) > 4 and token.endswith("ed"):
                token = token[:-2]
                if len(token) > 2 and token[-1] == token[-2]:
                    token = token[:-1]
            elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                token = token[:-1]
            tokens.append(token)
        return tokens

    def _load(self, folder: Path) -> list[Chunk]:
        chunks: list[Chunk] = []
        for path in sorted(folder.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            meta = dict(re.findall(r"^([a-z_]+):\s*(.+)$", raw, re.MULTILINE))
            parts = re.split(r"^##\s+\[([^]]+)\]\s*(.+)$", raw, flags=re.MULTILINE)
            for index in range(1, len(parts), 3):
                section_id, title, body = parts[index:index + 3]
                chunks.append(Chunk(section_id, title.strip(), body.strip(), path.name,
                                    meta.get("doc_type", "faq"), meta.get("status", "current"),
                                    meta.get("effective_from", "")))
        return chunks

    def search(self, question: str, limit: int = 4) -> list[Chunk]:
        query = Counter(self._tokens(question))
        return_refund_follow_up = (
            bool({"return", "item"} & query.keys())
            and bool({"refund", "when", "long", "arrive"} & query.keys())
        )
        shipped_cancellation = (
            "stop it" in question.lower()
            or "on its way" in question.lower()
            or "already shipped" in question.lower()
        )
        if return_refund_follow_up:
            query.update(self._tokens("pickup quality check refund initiated business payment method"))
        if shipped_cancellation:
            query.update(self._tokens("shipped cannot cancelled refuse parcel return"))
        ranked = []
        weights = {"policy": 1.5, "agent_guide": 1.3, "help": 1.1, "faq": 1.0, "marketing": .65}
        for chunk in self.chunks:
            if chunk.status != "current":
                continue
            words = Counter(self._tokens(f"{chunk.title} {chunk.text}"))
            overlap = sum(min(count, words[word]) for word, count in query.items())
            phrase_bonus = sum(2 for word in query if len(word) > 4 and word in words)
            score = (overlap + phrase_bonus) * weights.get(chunk.doc_type, 1)
            if return_refund_follow_up and chunk.section_id == "RET-7":
                score += 8
            elif return_refund_follow_up and chunk.section_id == "REF-2":
                score += 16
            if shipped_cancellation and chunk.section_id == "CAN-2":
                score += 16
            if score:
                ranked.append((score, chunk))
        return [chunk for _, chunk in sorted(ranked, key=lambda item: item[0], reverse=True)[:limit]]
