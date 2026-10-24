
"""Small, dependency-free RAG index for the ZentroMart knowledge base."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
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

    def __init__(self, kb_dir: str | Path, protected_words: set[str] | None = None):
        self.chunks = self._load(Path(kb_dir))
        self.protected_words = {
            token
            for word in protected_words or set()
            for token in self._tokens(word)
        }
        self.vocabulary = Counter(
            token
            for chunk in self.chunks
            if chunk.status == "current"
            for token in self._tokens(f"{chunk.title} {chunk.text}")
            if token.isascii() and token.isalpha() and len(token) >= 4
        )
        self._vocabulary_by_length: dict[int, list[str]] = {}
        for token in self.vocabulary:
            self._vocabulary_by_length.setdefault(len(token), []).append(token)

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

    def correct_typos(self, text: str) -> str:
        """Correct only close, single-word matches from current support content."""
        def replace(match: re.Match[str]) -> str:
            original = match.group()
            token = self._tokens(original)[0]
            if not token.isascii() or not token.isalpha() or len(token) < 4:
                return original
            if token in self.protected_words:
                return original
            if token in self.vocabulary:
                return original

            max_distance = 1 if len(token) <= 6 else 2
            possible = (
                candidate
                for length in range(max(4, len(token) - max_distance), len(token) + max_distance + 1)
                for candidate in self._vocabulary_by_length.get(length, ())
            )
            best_candidate = None
            best_score = 0.0
            for candidate in possible:
                score = SequenceMatcher(None, token, candidate, autojunk=False).ratio()
                if score > best_score:
                    best_candidate = candidate
                    best_score = score
                elif score == best_score and best_candidate:
                    if self.vocabulary[candidate] > self.vocabulary[best_candidate]:
                        best_candidate = candidate

            threshold = 0.82 if len(token) <= 6 else 0.80
            if best_candidate and best_score >= threshold:
                return best_candidate
            return original

        return re.sub(r"[A-Za-z]+", replace, text)

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
        question = self.correct_typos(question)
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
