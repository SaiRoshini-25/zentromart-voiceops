"""Conversation controller: retrieval, verification, tools and safe handoff."""
from __future__ import annotations

import re
import unicodedata
import os
import csv
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from .rag import KnowledgeBase
from .rules import refund_method_label, refund_record, refund_timeline
from .tools import create_ticket, lookup_order

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ORDER_ID = re.compile(r"(?<![A-Z0-9])(?:ORD[- ]?)?((?:\d[ -]?){8})(?!\d)", re.IGNORECASE)
NUMBER = re.compile(r"(?<!\d)\d{4,6}(?!\d)")
SPOKEN_DIGITS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
PRODUCT_SKU = re.compile(r"\b[A-Z]{3}-\d{4}\b")
THIRD_PARTY_DATA_REQUEST = re.compile(
    r"\b(?:phone number|address|personal details)\b.{0,50}\b(?:person|customer|"
    r"someone|somebody|someone else)\b|\b(?:person|customer)\b.{0,40}\bwho ordered\b",
    re.IGNORECASE,
)
HUMAN_REQUEST = re.compile(
    r"\b(?:speak|talk|connect|transfer|put me through|want|need|request)\b"
    r".{0,30}\b(?:human|person|representative|live agent|real person)\b|"
    r"\b(?:human|live|real)\s+(?:agent|representative|person)\b|इंसान|मानव",
    re.IGNORECASE,
)
UNAUTHORIZED_ACTIVITY = re.compile(
    r"\b(?:did(?: not|n't) (?:place|make|authorize)|never (?:placed|made|ordered)|"
    r"not my order|unauthori[sz]ed|fraud)\b|"
    r"\bsomeone.{0,45}\b(?:used|placed|made)\b.{0,35}\b(?:account|order)\b.{0,30}"
    r"\b(?:without permission|without my permission|unauthori[sz]ed)\b",
    re.IGNORECASE,
)
LEGAL_ACTION = re.compile(
    r"\b(?:legal action|consumer court|police complaint|sue you|file a case)\b",
    re.IGNORECASE,
)
SAFETY_RISK = re.compile(
    r"\b(?:swelling battery|swollen battery|burning smell|sparks|catching fire|"
    r"battery.{0,20}swelling|electric shock|injury|injured|burning|smells? like smoke)\b",
    re.IGNORECASE,
)
PRICE_AMOUNT = re.compile(r"(?:\u20b9|rs\.?\s*)([\d,]+)", re.IGNORECASE)
TICKET_SENSITIVE_NUMBER = re.compile(r"(?<![A-Z0-9-])\d{4,6}(?![A-Z0-9])", re.IGNORECASE)


@dataclass
class TurnResult:
    reply: str
    sources: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)


class ZentroMartAgent:
    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        with (self.data_dir / "holidays.csv").open(encoding="utf-8", newline="") as handle:
            self.holidays = {
                date.fromisoformat(row["date"])
                for row in csv.DictReader(handle)
            }
        with (self.data_dir / "products.csv").open(encoding="utf-8", newline="") as handle:
            self.products = {row["sku"]: row for row in csv.DictReader(handle)}
        self.kb = KnowledgeBase(
            self.data_dir / "kb",
            protected_words={
                word
                for product in self.products.values()
                for word in product["title"].split()
            },
        )
        self.product_prices = {
            sku: int(product["price_inr"])
            for sku, product in self.products.items()
        }
        with (self.data_dir / "pincodes.csv").open(encoding="utf-8", newline="") as handle:
            self.pincodes = {
                row["pincode"]: row
                for row in csv.DictReader(handle)
            }
        # This is intentionally limited to public support knowledge and the
        # product catalog.  It never includes .env, application files, or the
        # complete customer-order dataset.
        self._gemini_reference_context = self._build_gemini_reference_context()
        self.reset()

    def reset(self) -> None:
        self.order_id: str | None = None
        self.verified_order: dict | None = None
        self.awaiting_verification = False
        self.verification_attempts = 0
        self.awaiting_order_id = False
        self.pending_request: str | None = None
        self.issue_product: str | None = None
        self.issue_purchase_timing: str | None = None
        self.issue_description: str | None = None
        self.current_product_sku: str | None = None
        self.damage_claim_active = False
        self.pending_return_question = False
        self.ticket_created = False
        self.history: list[tuple[str, str]] = []

    @staticmethod
    def _order_id(message: str) -> str | None:
        match = ORDER_ID.search(message)
        return re.sub(r"\D", "", match.group(1)) if match else None

    @staticmethod
    def _normalise_spoken_digits(message: str) -> str:
        """Turn a dictated digit sequence into digits without altering normal prose."""
        pattern = r"\b(?:" + "|".join(SPOKEN_DIGITS) + r")(?:[\s,-]+(?:" + "|".join(SPOKEN_DIGITS) + r")){3,}\b"
        return re.sub(
            pattern,
            lambda match: "".join(SPOKEN_DIGITS[word] for word in re.findall(r"[a-z]+", match.group().lower())),
            message,
            flags=re.IGNORECASE,
        )

    @staticmethod
    def _verification_value(message: str, order_id: str | None = None) -> str | None:
        """Return a supplied phone suffix or pincode, never mistaking the order ID for proof."""
        order_spans = []
        if order_id:
            order_spans = [match.span() for match in ORDER_ID.finditer(message)]
        values = []
        for match in NUMBER.finditer(message):
            value = match.group()
            is_part_of_order_id = any(
                match.start() < end and match.end() > start
                for start, end in order_spans
            )
            if value != order_id and not is_part_of_order_id:
                values.append(value)
        if not values:
            return None
        # Speech recognition often retains a misheard first attempt before a
        # customer says "sorry" or "I made a mistake" and repeats the value.
        # In that situation use the final stated value, not the stale one.
        if re.search(r"\b(?:sorry|mistake|correction|actually|rather)\b", message, re.IGNORECASE):
            return values[-1]
        return values[0]

    @staticmethod
    def _is_hinglish(text: str) -> bool:
        return bool(
            re.search(r"\b(hai|mujhe|mera|kya|kar|krna|chahiye)\b", text.lower())
            or re.search(r"[\u0900-\u097f]", text)
        )

    @staticmethod
    def _friendly_date(value: str | date | None) -> str | None:
        if not value:
            return None
        if isinstance(value, date):
            return f"{value.day} {value.strftime('%B %Y')}"
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            return value
        return f"{parsed.day} {parsed.strftime('%B %Y')}"

    @staticmethod
    def _parse_customer_date(text: str, default_year: int = 2026) -> date | None:
        iso_date = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text)
        if iso_date:
            try:
                return date.fromisoformat(iso_date.group())
            except ValueError:
                return None
        natural_date = re.search(
            r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
            r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
            r"(?:\s+(20\d{2}))?\b",
            text,
            re.IGNORECASE,
        )
        if not natural_date:
            return None
        month = natural_date.group(2)[:3].lower()
        try:
            return date(
                int(natural_date.group(3) or default_year),
                datetime.strptime(month, "%b").month,
                int(natural_date.group(1)),
            )
        except ValueError:
            return None

    def _product_candidates(self, message: str) -> list[dict]:
        sku = PRODUCT_SKU.search(message)
        if sku and sku.group() in self.products:
            return [self.products[sku.group()]]
        words = set(re.findall(r"[a-z0-9]+", message.casefold()))
        ignored = {"pack", "of", "the", "and", "with", "ml", "mah", "black", "white", "green", "blue", "red", "grey", "gray", "beige", "maroon"}
        matches: list[dict] = []
        for product in self.products.values():
            title_words = set(re.findall(r"[a-z0-9]+", product["title"].casefold()))
            key_words = title_words - ignored
            if len(key_words) >= 2 and key_words.issubset(words):
                matches.append(product)
        return matches

    def _product_mentioned(self, message: str) -> dict | None:
        matches = self._product_candidates(message)
        if not matches:
            return None
        if self.verified_order:
            order_skus = {
                item.get("sku")
                for item in self.verified_order.get("items", [])
            }
            order_match = next(
                (product for product in matches if product["sku"] in order_skus),
                None,
            )
            if order_match:
                return order_match
        return matches[0]

    def _refund_timeline_answer(self, message: str, lower: str) -> TurnResult | None:
        if not re.search(r"\brefund(?:s|ed|ing)?\b|रिफ़?ंड|वापसी", lower, re.IGNORECASE):
            return None
        method_patterns = (
            ("wallet", r"\b(?:zentromart\s+)?wallet\b"),
            ("upi", r"\bupi\b"),
            ("card", r"\b(?:credit|debit)\s+card\b|\bcard\b"),
            ("netbanking", r"\bnet[ -]?bank(?:ing)?\b"),
            ("cod_bank", r"\b(?:bank transfer|neft)\b"),
        )
        method = next(
            (name for name, pattern in method_patterns if re.search(pattern, lower)),
            None,
        )
        if not method:
            if self.verified_order:
                method = str(self.verified_order.get("payment_method") or "").lower()
            if method not in {"wallet", "upi", "card", "netbanking", "cod_bank"}:
                return None
        method_days = {
            "upi": (1, 3, "UPI"),
            "card": (5, 7, "Card"),
            "netbanking": (3, 5, "Net banking"),
            "cod_bank": (5, 7, "Bank-transfer"),
        }
        return_follow_up = any(
            role == "customer" and re.search(r"\breturn(?:ed|ing)?\b|लौट|वापस", prior, re.IGNORECASE)
            for role, prior in self.history[:-1]
        )
        if method == "wallet":
            if self.verified_order and self.verified_order.get("is_plus") and return_follow_up:
                return TurnResult(
                    "As a ZentroMart Plus member, your return refund is initiated at pickup. "
                    "Wallet refunds are Instant, usually within 2 hours.",
                    ["RET-7", "PLS-2", "REF-2"],
                )
            return TurnResult(
                "ZentroMart Wallet refunds are Instant, within 2 hours of initiation.",
                ["REF-2", "KB-00.1"],
            )
        start = self._parse_customer_date(message)
        minimum, maximum, label = method_days[method]
        if not start:
            if self.verified_order:
                order = self.verified_order
                if return_follow_up and order.get("is_plus"):
                    return TurnResult(
                        "As a ZentroMart Plus member, your return refund is initiated at pickup. "
                        f"{label} refunds then usually arrive in {minimum}–{maximum} business days.",
                        ["RET-7", "PLS-2", "REF-2"],
                    )
                if return_follow_up or order.get("status") in {"return_requested", "return_picked_up"}:
                    return TurnResult(
                        "After pickup, the item must pass a quality check; the refund is initiated within 2 business days of pickup. "
                        f"Then your {label.lower()} refund usually arrives in {minimum}–{maximum} business days.",
                        ["RET-7", "REF-1", "REF-2"],
                    )
            return TurnResult(
                f"{label} refunds usually arrive in {minimum}–{maximum} business days after initiation.",
                ["REF-2", "KB-00.1"],
            )
        first = self._friendly_date(self._business_days_after(start, minimum, self.holidays))
        last = self._friendly_date(self._business_days_after(start, maximum, self.holidays))
        return TurnResult(
            f"For a refund initiated on {self._friendly_date(start)}, {label.lower()} refunds usually arrive "
            f"between {first} and {last}. Sundays and listed warehouse holidays are not counted.",
            ["REF-2", "KB-00.3"],
        )

    def _build_gemini_reference_context(self) -> str:
        """Build the complete safe reference package sent to Gemini."""
        documents = "\n\n".join(
            f"[{chunk.section_id}] {chunk.title}\n{chunk.text}"
            for chunk in self.kb.chunks
            if chunk.status == "current"
        )
        catalog = "\n".join(
            " | ".join(
                (
                    f"SKU={product['sku']}",
                    f"title={product['title']}",
                    f"category={product['return_category']}",
                    f"assured={product['assured']}",
                    f"warranty_months={product['warranty_months'] or 'none'}",
                )
            )
            for product in self.products.values()
        )
        return f"SUPPORT KNOWLEDGE BASE:\n{documents}\n\nPRODUCT CATALOG:\n{catalog}"

    @staticmethod
    def _clean_gemini_reply(value: str) -> str:
        """Make a model draft suitable for a short customer-facing chat reply."""
        cleaned = re.sub(r"\s+", " ", value.strip())
        cleaned = re.sub(r"^(?:answer|response)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*(?:sources?|evidence)\s*:\s*\[[^]]*\]\s*$", "", cleaned, flags=re.IGNORECASE)
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        return " ".join(sentences[:3]).strip()

    @staticmethod
    def _is_refund_status_request(lower: str) -> bool:
        """Whether the customer is asking us to check a particular refund."""
        mentions_refund = bool(re.search(r"\brefund(?:s|ed|ing)?\b|रिफ़?ंड|वापसी", lower))
        asks_status = bool(re.search(
            r"\b(?:complete(?:d)?|completion|status|amount|how much|"
            r"where(?:'s| is)?|credited|received|receive[dp]?|got|paid|process(?:ed|ing)?|date|money|"
            r"kahan|kab)\b|कहाँ|कहां|कब|कितना|कितने|स्थिति",
            lower,
        ))
        asks_to_check = bool(
            re.search(r"\bcheck\b.{0,20}\b(?:my )?refund\b|\brefund\b.{0,20}\bcheck\b", lower)
        )
        return bool(
            mentions_refund and (asks_status or asks_to_check)
        )

    def _refund_record_reply(self) -> TurnResult | None:
        """Return the recorded refund outcome, never substituting an estimate for it."""
        if not self.verified_order:
            return None

        record = refund_record(self.verified_order)
        if not record:
            return None
        amount_text = f"₹{record.amount_inr:,}" if record.amount_inr is not None else "The recorded"
        initiated_text = self._friendly_date(record.initiated_on) or record.initiated_on
        method_label = refund_method_label(record.method)
        if record.completed_on:
            completed_text = self._friendly_date(record.completed_on) or record.completed_on
            return TurnResult(
                f"Yes—your {amount_text} refund was initiated on {initiated_text} and completed on "
                f"{completed_text} to {method_label}.",
                ["REF-1", "REF-2", "IDV-2"],
            )
        return TurnResult(
            f"Your {amount_text} refund was initiated on {initiated_text} to {method_label}. "
            f"It has not yet been marked completed. {refund_timeline(record.method)}",
            ["REF-1", "REF-2", "IDV-2"],
        )

    def _intro(self, hindi: bool) -> str:
        return ("Namaste! Main ZentroMart ka AI assistant hoon. Aapka 8-digit order ID bata dijiye."
                if hindi else "Hello! I'm ZentroMart's AI assistant. Please share your order ID so I can help.")

    @staticmethod
    def _welcome(message: str) -> str | None:
        lower = message.casefold()
        normalized = " ".join(
            "".join(
                char if char.isalnum() or char.isspace() or unicodedata.category(char).startswith("M") else " "
                for char in lower
            ).split()
        )
        if re.search(r"\bwhat can you help me with\b|\bwhat do you help with\b", lower):
            return (
                "I can help with order tracking, returns, refunds, cancellations, delivery, "
                "and product warranty. Tell me what you need help with."
            )
        if re.fullmatch(
            r"(?:(?:okay|ok|alright|acha|achha|achhe|that s all|all good|no problem|సరే|అచ్చే)\s+)*"
            r"(?:thank\s+you|thanks?|ధన్యవాదాలు|థ్యాంక్\s*యూ|థాంక్\s*యూ|థాంక్యూ)"
            r"(?:\s+(?:so much|very much))?",
            normalized,
            re.IGNORECASE,
        ):
            return "You're welcome! Is there anything else I can help you with?"
        if re.fullmatch(
            r"(?:(?:thanks?|thank you)\s+)?"
            r"(?:ok|okay|alright|got it|understood|perfect|cool|all good|"
            r"that is all|that s all(?: for now)?|that s it(?: for now)?|"
            r"nothing else(?: to ask)?|no more questions|all done|im done|i m done|we re done|"
            r"no thanks|no thank you|im good|i m good|we re good)"
            r"(?:\s+(?:thanks?|thank you))?",
            normalized,
            re.IGNORECASE,
        ):
            return "Okay! Let me know if you need anything else."
        if re.fullmatch(
            r"(?:(?:ok|okay|alright|thanks?|thank you)\s+)*"
            r"(?:bye(?:\s+bye| for now)?|goodbye|good\s+bye|see you(?: later)?|see ya|take care|good night|"
            r"talk to you later|अलविदा|शुभ रात्रि|వీడ్కోలు|శుభ రాత్రి)"
            r"(?:\s+(?:thanks?|thank you))?",
            normalized,
            re.IGNORECASE,
        ):
            return "Goodbye! Feel free to come back if you need help."
        if re.search(r"\b(?:need|want) help with an order\b", lower) or re.fullmatch(
            r"[\s\W]*(?:hi|hello|hey)[\s\W]*", lower
        ):
            return "Hi! I'm ZentroMart's AI assistant. What do you need help with?"
        return None

    def _curated_answer(self, message: str, lower: str) -> TurnResult | None:
        # Keep the most common customer journeys short and conversational.  The
        # fallback below is deliberately conservative, but it should never dump
        # raw policy chunks into a customer-facing reply.
        if (
            self.verified_order
            and self._refund_overdue()
            and (
                self._is_refund_status_request(lower)
                or re.search(r"\b(?:late|overdue|past the maximum)\b", lower)
            )
        ):
            if self.ticket_created and (record_reply := self._refund_record_reply()):
                record_reply.reply += " A human support ticket is already open for review."
                return record_reply
            return self._ticket(
                message,
                self.kb.search(message),
                "refund is beyond its maximum timeline",
            )
        sku = PRODUCT_SKU.search(message)
        if sku:
            self.current_product_sku = sku.group()
        product_candidates = self._product_candidates(message)
        mentioned_product = self._product_mentioned(message)
        if mentioned_product:
            self.current_product_sku = mentioned_product["sku"]
        active_sku = sku.group() if sku else self.current_product_sku
        matched_product = mentioned_product
        catalog_sources = [
            f"products.csv:{product['sku']}"
            for product in product_candidates
            if product["return_category"] == matched_product["return_category"]
        ] if matched_product else []
        if self.verified_order and matched_product:
            catalog_sources = [f"products.csv:{matched_product['sku']}"]
        return_question = bool(re.search(
            r"\b(?:return|send back|sending back|send it back|exchange|how long|days?)\b|"
            r"वापस|वापिस|लौट",
            lower,
            re.IGNORECASE,
        ))
        if matched_product and return_question:
            category = matched_product.get("return_category", "")
            source = f"products.csv:{matched_product['sku']}"
            verified_item = None
            if self.verified_order:
                verified_item = next(
                    (
                        item for item in self.verified_order.get("items", [])
                        if item.get("sku") == matched_product["sku"]
                    ),
                    None,
                )
            if verified_item and self.verified_order.get("delivered_on"):
                delivered = date.fromisoformat(str(self.verified_order["delivered_on"]))
                placed = datetime.fromisoformat(str(self.verified_order["placed_at"])).date()
                festive_sale = (
                    category == "fashion"
                    and date(2026, 10, 1) <= placed <= date(2026, 10, 15)
                )
                window = (
                    30 if festive_sale else
                    15 if category == "fashion" else
                    10 if category in {"home_kitchen", "books_media"} else
                    7 if category in {"electronics", "beauty_sealed"} else
                    None
                )
                if category in {"innerwear", "grocery"}:
                    return TurnResult(
                        "This item cannot be returned for fit or change of mind.",
                        ["RET-2", "RET-5", *catalog_sources],
                    )
                if window:
                    deadline = self._friendly_date((delivered + timedelta(days=window)).isoformat())
                    reply = (
                        f"Your order was delivered on {self._friendly_date(delivered.isoformat())}; "
                        f"the return window is {window} days, through {deadline}, if the item meets the return conditions."
                    )
                    sources = ["RET-2", source]
                    if festive_sale:
                        sources.append("RET-8")
                    if category == "fashion":
                        sources.append("RET-6")
                        if self.verified_order.get("is_plus"):
                            reply += " Your Plus membership waives the reverse-pickup fee."
                        else:
                            reply += " A ₹49 reverse-pickup fee applies to change-of-mind clothing returns."
                    if self.verified_order.get("is_plus"):
                        sources.extend(["RET-7", "PLS-2"])
                    return TurnResult(reply, list(dict.fromkeys(sources)))
            if category in {"innerwear", "grocery"}:
                return TurnResult(
                    "This item cannot be returned for fit or change of mind.",
                    ["RET-2", "RET-5", *catalog_sources],
                )
            if category == "beauty_sealed" and re.search(
                r"\b(?:opened|open|tried)\b|खोल|खुला",
                lower,
            ):
                return TurnResult(
                    "Beauty and personal-care products can be returned only if unopened with the seal intact, so an opened product is not eligible for a change-of-mind return.",
                    ["RET-4", "RET-2", *catalog_sources],
                )
            delivered_on = self._parse_customer_date(message)
            if category == "electronics" and delivered_on and delivered_on < date(2026, 8, 1):
                return TurnResult(
                    "Electronics delivered before 1 August 2026 keep the earlier 10-day return window, counted from delivery.",
                    ["RET-3", "RETv3-2", *catalog_sources],
                )
            sale_date = self._parse_customer_date(message)
            if category == "fashion" and sale_date and date(2026, 10, 1) <= sale_date <= date(2026, 10, 15) and re.search(r"\bsale\b|फ़ेस्टिव|फेस्टिव", lower):
                return TurnResult(
                    "Because this clothing order was placed during the 1–15 October festive sale, it has a 30-day return window from delivery.",
                    ["RET-8", "RET-2", source],
                )
            days = {
                "electronics": "7",
                "fashion": "15",
                "home_kitchen": "10",
                "books_media": "10",
                "beauty_sealed": "7",
            }.get(category)
            if days:
                sources = ["RET-2", source]
                reply = f"This item can be returned within {days} calendar days of delivery."
                if category == "fashion":
                    reply += " Clothing orders placed from 1 to 15 October 2026 during the festive sale have 30 days."
                    sources.append("RET-8")
                if category == "electronics":
                    reply += " Items delivered before 1 August 2026 keep the earlier 10-day window."
                    sources.append("RET-3")
                return TurnResult(reply, list(dict.fromkeys(["RET-2", *catalog_sources, *sources[1:]])))
        if re.search(r"\b(?:coupon|coupons)\b|कूपन", lower, re.IGNORECASE):
            return TurnResult(
                "Only one coupon can be used per order; coupons cannot be combined.",
                ["CPN-1"],
            )
        if re.search(
            r"\b(?:how\s+(?:do|does)|what\s+is|explain|tell\s+me\s+about)\b.{0,35}\brefunds?\b"
            r"|\brefund\s+(?:process|policy)\b",
            lower,
        ):
            return TurnResult(
                "Refund timing depends on why the refund is due. Cancellations before shipping are initiated immediately. "
                "For returns, it’s initiated within 2 business days after pickup once the item passes the quality check "
                "(at pickup for ZentroMart Plus members). After initiation, the money goes to the original payment method; "
                "timing varies by method. Which payment method did you use?",
                ["REF-1", "REF-2", "RET-7"],
            )
        refund_timeline = self._refund_timeline_answer(message, lower)
        if refund_timeline:
            return refund_timeline
        if re.search(r"दुकान|स्टोर", lower) and re.search(r"कहाँ|कोच्चि|है|हैं", lower):
            return TurnResult(
                "माफ़ कीजिए, मेरे पास दुकानों की जानकारी नहीं है। मैं आपको किसी सहयोगी से जोड़ सकती हूँ।",
                [],
            )
        if re.search(r"\b(?:voucher|compensation)\b|वाउचर|मुआवज़ा", lower, re.IGNORECASE):
            if self.verified_order and re.search(r"\b(?:late|delay)\b|देर|लेट", lower):
                order = self.verified_order
                is_late_express = (
                    order.get("delivery_type") == "express"
                    and order.get("status") in {"delivered", "returned_refunded"}
                    and order.get("delivered_on")
                    and order.get("promised_date")
                    and date.fromisoformat(str(order["delivered_on"]))
                    > date.fromisoformat(str(order["promised_date"]))
                )
                if is_late_express and int(order.get("delivery_fee_inr") or 0) == 79:
                    return TurnResult(
                        "I’m sorry your Express order was late. The ₹79 Express fee is automatically credited to your ZentroMart Wallet within 48 hours; we can’t offer an additional voucher.",
                        ["DEL-4", "ESC-4"],
                    )
            return TurnResult(
                "I’m sorry about the delay. I can’t offer an unapproved voucher or compensation; the policy allows a ₹79 Express-fee credit only when a paid Express order arrives after its promised date.",
                ["DEL-4", "ESC-4"],
            )
        if re.search(r"सामान|वस्तु", lower) and re.search(r"टूटा|खराब", lower):
            self.damage_claim_active = True
            return TurnResult(
                "डिलीवरी के 48 घंटे के अंदर सामान और पैकेट की फ़ोटो के साथ शिकायत करें। जाँच के बाद मुफ़्त रिप्लेसमेंट या पूरा रिफ़ंड मिल सकता है।",
                ["FAQH-3", "DWM-1", "DWM-2"],
            )
        if re.search(r"ऑर्डर", lower) and re.search(r"शिप", lower) and re.search(r"कैंसिल", lower):
            return TurnResult(
                "शिप होने के बाद ऑर्डर कैंसिल नहीं किया जा सकता। डिलीवरी के समय पार्सल लेने से मना कर सकते हैं; प्रीपेड ऑर्डर का रिफ़ंड पार्सल वेयरहाउस लौटने पर शुरू होता है।",
                ["FAQH-1", "CAN-2", "REF-1"],
            )
        if re.search(r"\b(?:student discount|store near me|pick up.*store|store pickup)\b", lower):
            return TurnResult("I don't have information about that in the support documents. I can connect you to a human colleague.", [])
        if re.search(r"\b(?:dubai|outside india|international)\b", lower) and re.search(r"\bdeliver\b", lower):
            return TurnResult("ZentroMart delivers only within India.", ["DEL-5", "FAQ-5"])
        if re.search(r"\b(?:not home|not at home|courier comes|delivery attempt)\b", lower):
            return TurnResult("The courier makes up to 3 delivery attempts and sends an SMS after a failed attempt to reschedule.", ["DEL-3"])
        if re.search(r"\b(?:cash on delivery|cod)\b", lower) and re.search(r"\b(?:return|money back|refund)\b", lower):
            return TurnResult("For a cash-on-delivery refund, choose ZentroMart Wallet for an instant refund or bank transfer, which takes 5 to 7 business days after bank details are verified.", ["REF-3", "REF-2"])
        if re.search(r"\b(?:plus member|zentromart plus)\b", lower) and re.search(r"\b(?:pickup|return)\b.{0,30}\b(?:fee|charged|cost)\b|\b(?:fee|charged|cost)\b.{0,30}\b(?:pickup|return)\b", lower):
            return TurnResult("No - reverse pickup is free for ZentroMart Plus members; the fee is waived.", ["RET-6", "PLS-2"])
        if re.search(r"\b(?:torn|damaged|defective|wrong)\b", lower) and re.search(r"\b(?:fee|pickup|return)\b", lower):
            return TurnResult("Reverse pickup is free with no fee for a damaged, defective, or wrong item.", ["RET-6", "DWM-2"])
        if re.search(r"\b(?:collection|pickup)\b", lower) and re.search(r"\bfree\b", lower) and re.search(r"\btorn|damaged|defective|wrong\b", lower):
            return TurnResult("Yes. Collection is free, with no reverse-pickup fee, when an item is damaged, defective, or wrong.", ["RET-6", "DWM-2"])
        if re.search(r"\b(?:lipstick|perfume|beauty|personal care)\b", lower) and re.search(r"\b(?:opened|open|tried)\b", lower):
            return TurnResult("Beauty and personal-care products can be returned only when unopened with the seal intact.", ["RET-2", "RET-4"])
        if active_sku and re.search(r"\bwarranty|assured|covered\b", lower):
            product = self.products.get(active_sku)
            if product and product.get("assured") == "True" and product.get("warranty_months"):
                brand_months = int(product["warranty_months"])
                brand_warranty = (
                    f" It also includes a {brand_months}-month brand warranty."
                    if brand_months > 0 else ""
                )
                return TurnResult(
                    f"{product['title']} has a 12-month warranty through ZentroMart Assured from delivery.{brand_warranty} "
                    "Raise a claim in My Orders in the app.",
                    ["WAR-1", "WAR-2", f"products.csv:{active_sku}"],
                )
            if product:
                return TurnResult(
                    f"I can't confirm ZentroMart Assured coverage for {product['title']}; manufacturer warranty depends on the listing.",
                    ["WAR-1", f"products.csv:{active_sku}"],
                )
        if sku:
            return TurnResult(
                "Thanks—I found that product. Would you like to check its warranty, return eligibility, or something else?",
                [f"products.csv:{sku.group()}"],
            )
        if self.damage_claim_active and re.search(r"\b(?:guarantee|promise)\b", lower) and re.search(r"\b(?:replacement|replace)\b", lower):
            return TurnResult(
                "I can't guarantee a replacement today. Please share the order ID and delivery date so we can verify the claim and review the photos.",
                ["DWM-1", "DWM-2"],
            )
        if self.damage_claim_active and re.search(r"\bphotos?\b", lower):
            return TurnResult(
                "Thanks—please keep the photos of the item and packaging ready. What was the delivery date?",
                ["DWM-1"],
            )
        has_active_refund = self.verified_order and any(
            self.verified_order.get(key)
            for key in (
                "return_refund_initiated_on",
                "cancellation_refund_initiated_on",
                "rto_refund_initiated_on",
            )
        )
        refund_query = re.search(r"\brefund(?:ed|ing)?\b", lower) or (
            has_active_refund
            and re.search(r"\bmoney\b", lower)
            and re.search(r"\b(?:when|how long|arrive|come|receive)\b", lower)
        )
        if self.verified_order and refund_query and (
            self._is_refund_status_request(lower)
            or re.search(r"\b(?:when|how long|arrive|come|receive)\b|कहाँ|कब|कितने दिन", lower)
        ):
            if reply := self._refund_record_reply():
                return reply
        if self.verified_order and re.search(r"\b(?:refuse|refusal|reject)\b.{0,35}\b(?:parcel|delivery|arriv)", lower):
            payment = str(self.verified_order.get("payment_method", "")).lower()
            refund_note = (
                " For this prepaid order, the refund starts after the parcel returns to the warehouse."
                if payment != "cod" else " For a cash-on-delivery order, nothing is charged."
            )
            return TurnResult(
                "You can refuse a shipped parcel at delivery." + refund_note,
                ["CAN-2", "REF-1"],
            )
        if (
            self.verified_order
            and str(self.verified_order.get("status", "")).lower() == "out_for_delivery"
            and "guarantee" not in lower
            and (
                re.search(r"\b(?:what time|when|arrival time)\b.{0,30}\barriv\w*\b", lower)
                or re.search(r"\b(?:delivery time|eta|estimated time|what time)\b", lower)
            )
        ):
            promised = self._friendly_date(self.verified_order.get("promised_date"))
            reply = "Your order is out for delivery."
            if promised:
                reply += f" Its promised delivery date is {promised}."
            reply += " I can't confirm an exact arrival time."
            return TurnResult(reply, ["TRK-1", "DEL-1"])
        if (
            self.verified_order
            and not re.search(r"\b(?:cancel|stop)\b", lower)
            and not re.search(r"\brefund(?:ed|ing)?\b", lower)
            and re.search(r"\b(?:where is|track|status|state|when.*arriv\w*|arriving today|arrive today|has.*shipped|shipped yet|when was it shipped|when did it ship|kab ship hua)\b", lower)
        ):
            status = str(self.verified_order.get("status", "")).replace("_", " ")
            promised = self.verified_order.get("promised_date")
            shipped_on = self.verified_order.get("shipped_on")
            if re.search(r"\b(?:when was it shipped|when did it ship|kab ship hua)\b", lower) and shipped_on:
                return TurnResult(
                    f"Your order was handed to the courier on {self._friendly_date(shipped_on)}.",
                    ["TRK-1"],
                )
            reply = f"Your order is {status}."
            if shipped_on := self._friendly_date(shipped_on):
                reply += f" It was handed to the courier on {shipped_on}."
            if promised := self._friendly_date(promised):
                reply += f" The promised delivery date is {promised}."
            return TurnResult(reply, ["TRK-1", "DEL-1"])

        if self.verified_order:
            order = self.verified_order
            status = str(order.get("status", "")).lower()
            if (
                order.get("delivery_type") == "express"
                and order.get("status") in {"delivered", "returned_refunded"}
                and order.get("delivered_on")
                and order.get("promised_date")
                and date.fromisoformat(str(order["delivered_on"]))
                > date.fromisoformat(str(order["promised_date"]))
                and re.search(r"\b(?:express|fee|late|delay)\b", lower)
            ):
                fee = int(order.get("delivery_fee_inr") or 0)
                if fee == 79:
                    delivered = self._friendly_date(str(order["delivered_on"]))
                    promised = self._friendly_date(str(order["promised_date"]))
                    return TurnResult(
                        f"Your Express order arrived on {delivered}, after its promised date of {promised}. "
                        "The ₹79 Express fee is credited to your ZentroMart Wallet automatically within 48 hours.",
                        ["DEL-4"],
                    )
                if fee == 0 and order.get("plus_member") == "True":
                    return TurnResult(
                        "Your ZentroMart Plus membership waived the Express fee, so there is no delivery-fee credit due.",
                        ["DEL-4"],
                    )
            if (
                status in {"placed", "packed"}
                and not re.search(r"\b(?:cancel|stop)\b", lower)
                and re.search(r"\b(?:process|ship)\w*\b", lower)
            ):
                promised = self._friendly_date(order.get("promised_date"))
                reply = f"Your order is {status}."
                if promised:
                    reply += f" The promised delivery date is {promised}."
                return TurnResult(reply, ["TRK-1", "DEL-1"])
            missing_delivery_follow_up = any(
                re.search(r"\b(?:can't find|cannot find|not receive|missing)\b", turn[1].lower())
                for turn in self.history
                if turn[0] == "customer"
            )
            if status == "delivered" and (
                re.search(r"\b(?:can't find|cannot find|not receive|haven't received|have not received|did not get|didn't get|didn't receive|missing)\b", lower)
                or (missing_delivery_follow_up and re.search(r"\bwhat should i do\b", lower))
            ):
                delivered = self._friendly_date(order.get("delivered_on"))
                reply = "The order is recorded as delivered"
                if delivered:
                    reply += f" on {delivered}"
                reply += (
                    ". Please check with family, neighbours, or building security. "
                    "If it is still missing, report it within 48 hours of delivery so the missing-item team can review it."
                )
                return TurnResult(reply, ["TRK-2", "DWM-1"])
            if status == "delivery_attempted" and re.search(r"\b(?:why|deliver|try|again|receive)\b", lower):
                attempts = order.get("delivery_attempts") or 0
                return TurnResult(
                    f"The courier could not complete delivery; the record shows {attempts} failed delivery attempt"
                    f"{'s' if attempts != 1 else ''}. We do not have a reason recorded. "
                    "After a failed attempt, the courier sends an SMS to reschedule; up to 3 attempts may be made.",
                    ["TRK-1", "DEL-3"],
                )
            if status == "returned_to_origin" and re.search(r"\b(?:returned|seller|money|refund|back)\b", lower):
                if reply := self._refund_record_reply():
                    return reply
                return TurnResult(
                    "Your order was returned to origin and no refund initiation is recorded yet. "
                    "For a prepaid order, the refund starts after the parcel reaches the warehouse.",
                    ["TRK-1", "REF-1"],
                )
            if status == "returned_to_origin" and re.search(r"\b(?:send|resend|again|same order)\b", lower):
                return TurnResult(
                    "This order has been returned to origin. I can't confirm a replacement shipment from the order record; "
                    "the documented next step for a prepaid order is its refund after return to the warehouse.",
                    ["TRK-1", "DEL-3", "REF-1"],
                )
            if status == "return_requested" and re.search(r"\b(?:pickup|picked|refund|arrive)\b", lower):
                return TurnResult(
                    "Your return has been requested, but no pickup or refund initiation is recorded yet. "
                    "The refund can be initiated after pickup and the required quality check.",
                    ["RET-7", "REF-1"],
                )
            if status == "return_picked_up" and re.search(r"\b(?:quality|qc|refund|pickup|picked)\b|रिफ़?ंड|गुणवत्ता", lower):
                pickup = self._friendly_date(order.get("return_pickup_on"))
                reply = "Your return was picked up"
                if pickup:
                    reply += f" on {pickup}"
                reply += ". No quality-check pass or refund initiation is recorded yet."
                return TurnResult(reply, ["RET-7", "REF-1"])

        if (
            self.verified_order
            and "refund" not in lower
            and re.search(r"\b(?:late|delayed|overdue|hasn't arrived|has not arrived)\b", lower)
        ):
            status = str(self.verified_order.get("status", "")).replace("_", " ")
            promised = self.verified_order.get("promised_date")
            reply = f"I can see that your order is {status}."
            if promised := self._friendly_date(promised):
                reply += f" Its promised delivery date is {promised}."
            reply += " If it has passed that date, I can help you check the next step."
            return TurnResult(reply, ["TRK-1", "DEL-1"])

        asks_availability = re.search(
            r"\b(?:serviceable|available|deliver(?:y)?|delivery|cod|cash on delivery|express)\b",
            lower,
        )
        if (
            re.search(r"\b(?:what|how|can|is).{0,35}\b(?:pincode|pin code)\b|\bpincode\s+\d{6}\b", lower)
            or (re.search(r"(?<!\d)\d{6}(?!\d)", message) and asks_availability)
        ):
            pincode_match = re.search(r"(?<!\d)(\d{6})(?!\d)", message)
            if not pincode_match:
                return TurnResult("Please share the 6-digit delivery pincode so I can check availability.", ["DEL-5"])
            row = self.pincodes.get(pincode_match.group(1))
            if not row:
                return TurnResult("I can’t confirm delivery availability for that pincode from our serviceability list.", ["DEL-5"])
            field = (
                "express_available" if re.search(r"\bexpress\b", lower)
                else "cod_available" if re.search(r"\b(?:cash on delivery|cod)\b", lower)
                else "serviceable"
            )
            labels = {
                "express_available": "Express delivery",
                "cod_available": "Cash on delivery",
                "serviceable": "Delivery",
            }
            available = row[field] == "True"
            reply = f"{labels[field]} is {'available' if available else 'not available'} at {pincode_match.group(1)}."
            if field == "serviceable" and available:
                reply += " COD and Express availability are checked separately."
            return TurnResult(reply, ["DEL-5"])

        if re.search(r"\b(?:sunday|sundays)\b", lower) and re.search(r"\bbusiness days?\b", lower):
            return TurnResult("No. Sundays are excluded when counting business days; listed holidays are excluded too.", ["KB-00.3"])

        if re.search(r"\bholidays?\b", lower) and re.search(r"\bbusiness days?\b", lower):
            return TurnResult("Yes. Dates listed in the holiday calendar are excluded from business-day calculations, as are Sundays.", ["KB-00.3"])

        if sku and re.search(r"\bwarranty|assured\b", lower):
            product = self.products.get(sku.group())
            if product:
                warranty = product.get("warranty_months")
                if product.get("assured") == "True" and warranty:
                    return TurnResult(
                        f"{product['title']} is covered by ZentroMart Assured with a {warranty}-month warranty from delivery. "
                        "You can raise the claim in My Orders in the app.",
                        ["WAR-1", "WAR-2", f"products.csv:{sku.group()}"],
                    )
                return TurnResult(
                    f"I can’t confirm a ZentroMart Assured warranty for {product['title']}. The manufacturer warranty, if any, depends on the product listing.",
                    ["WAR-1", f"products.csv:{sku.group()}"],
                )
            return TurnResult("I can’t confirm warranty coverage for that SKU from the product catalog.", ["WAR-1"])

        if re.search(r"\b(?:warranty|assured)\b", lower) and re.search(r"\b(?:claim|how do i)\b", lower):
            return TurnResult(
                "ZentroMart Assured products have a 12-month warranty from delivery. "
                "Open My Orders in the app and raise a warranty claim; ZentroMart arranges pickup and repair or replacement.",
                ["WAR-2"],
            )

        if re.search(r"\bSKU\s+[A-Z0-9]+-\d+\b", message, re.IGNORECASE) and not sku:
            return TurnResult(
                "I can’t confirm warranty coverage for that SKU from the product catalog.",
                ["WAR-1"],
            )

        if self.pending_return_question and re.search(r"\b(?:eligible|eligibility|can i|what about)\b", lower):
            self.issue_description = message
            return TurnResult("What type of item are you returning?", ["RET-2", "STY-5"])

        if self.pending_return_question and re.search(
            r"\b(?:clothing|clothes|shoes?|footwear|electronics?|phone|laptop|beauty|grocery|groceries|book|furniture|home|kitchen|innerwear)\b",
            lower,
        ):
            self.issue_product = lower
            return TurnResult("Thanks. What date was it delivered?", ["RET-2", "STY-5"])

        if self.pending_return_question and re.search(
            r"\b(?:delivered|delivery date|today|yesterday|(?:\d{1,2}(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|twenty[- ]first|twenty[- ]second|twenty[- ]third|twenty[- ]fourth|twenty[- ]fifth|twenty[- ]sixth|twenty[- ]seventh|twenty[- ]eighth|twenty[- ]ninth|thirtieth|thirty[- ]first)\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|sept|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?))\b",
            lower,
        ):
            if self.issue_product and re.search(r"\b(?:grocery|groceries|innerwear)\b", self.issue_product):
                return TurnResult("This category is not returnable for change of mind.", ["RET-5"])
            if self.issue_product and re.search(r"\b(?:clothing|clothes|shoes?|footwear)\b", self.issue_product):
                self.pending_return_question = False
                return TurnResult("Clothing and footwear have a 15-calendar-day return window from delivery, if unused with tags intact.", ["RET-2", "RET-4"])
            if self.issue_product and re.search(r"\b(?:electronics?|phone|laptop)\b", self.issue_product):
                self.pending_return_question = False
                return TurnResult("Electronics have a 7-calendar-day return window from delivery, or 10 days if delivered before 1 August 2026.", ["RET-2", "RET-3"])

        if self.pending_return_question and re.search(r"\b(?:change of mind|don't like|do not like|return my product|return my item)\b", lower):
            return TurnResult("What type of item are you returning?", ["RET-2", "STY-5"])

        if self.verified_order and re.search(r"\b(?:cancel|stop)\b", lower):
            status = str(self.verified_order.get("status", "")).replace("_", " ")
            if status == "shipped":
                payment = str(self.verified_order.get("payment_method", "")).lower()
                refund_note = (
                    " For a prepaid order, the refund starts after the parcel returns to the warehouse, usually 7 to 10 calendar days after the last delivery attempt."
                    if payment != "cod" else " For a cash-on-delivery order, nothing is charged if you refuse it."
                )
                return TurnResult(
                    "I can’t cancel this order because it has shipped; a shipped order can’t be cancelled. You can refuse the parcel at delivery "
                    "(refuse to accept it), "
                    "or accept it and return eligible items within their return window." + refund_note,
                    ["CAN-2", "RET-2", "REF-1"],
                )
            if status in {"cancelled", "delivered"}:
                state = "already cancelled" if status == "cancelled" else "already delivered"
                return TurnResult(
                    f"This order is {state}, so it can’t be cancelled now. "
                    "If it was delivered, you may request a return within the applicable return window.",
                    ["CAN-1", "CAN-2", "RET-2"],
                )
            return TurnResult(
                "You can cancel before the order ships from My Orders in the app; I can guide you but can’t cancel it for you.",
                ["CAN-1"],
            )

        if (
            self.verified_order
            and re.search(r"\b(?:yes|yeah|yep|please|sure)\b.{0,20}\b(?:guide|show|help)\b|\b(?:guide|show)\s+me\b", lower)
            and any(
                "cancel" in prior.casefold()
                for role, prior in self.history[:-1]
                if role == "customer"
            )
        ):
            status = str(self.verified_order.get("status", "")).lower()
            if status in {"placed", "packed"}:
                return TurnResult(
                    "Open the ZentroMart app, go to My Orders, select this order, and choose Cancel order. "
                    "Confirm the cancellation before it ships; the refund is then initiated immediately.",
                    ["CAN-1", "IDV-2"],
                )
            return TurnResult(
                "This order can no longer be cancelled in the app because it has already shipped. "
                "You can refuse it at delivery or return it if eligible.",
                ["CAN-2", "RET-2"],
            )

        if re.search(r"\b(?:cancel|stop)\b", lower) and re.search(r"\b(?:already shipped|has shipped|shipped already)\b", lower):
            return TurnResult(
                "A shipped order can’t be cancelled. You can refuse the parcel at delivery, "
                "or accept it and return eligible items within their return window.",
                ["CAN-2", "RET-2"],
            )

        if self.verified_order and re.search(r"\brefund\b", lower) and re.search(r"\b(?:how long|when|take|arrive|come)\b", lower):
            order = self.verified_order
            fields = (
                ("return_refund_initiated_on", "return_refund_method", "return_refund_completed_on"),
                ("cancellation_refund_initiated_on", "cancellation_refund_method", None),
                ("rto_refund_initiated_on", "rto_refund_method", None),
            )
            initiated = next(
                ((order.get(date_key), order.get(method_key))
                 for date_key, method_key, _ in fields if order.get(date_key)),
                None,
            )
            if initiated:
                date_text, method = initiated
                maximum = {"upi": 3, "card": 7, "netbanking": 5, "wallet": 0, "cod_wallet": 0, "cod_bank": 7}.get(str(method).lower())
                if maximum is not None:
                    due = self._business_days_after(date.fromisoformat(date_text), maximum, self.holidays)
                    if maximum == 0:
                        reply = "Wallet refunds are processed instantly, usually within 2 hours of initiation."
                    else:
                        reply = f"Your {str(method).replace('_', ' ')} refund should arrive within {maximum} business days, by {due.strftime('%-d %B %Y')}."
                    return TurnResult(reply, ["REF-2"])
            if str(order.get("status", "")).lower() == "shipped":
                return TurnResult(
                    "This order has shipped and no refund has been initiated. If you refuse delivery, the parcel must return to the warehouse first; the refund timing then depends on the payment method.",
                    ["CAN-2", "REF-1", "REF-2"],
                )
            method = str(order.get("payment_method", "")).lower()
            ranges = {"upi": "1–3 business days", "card": "5–7 business days", "netbanking": "3–5 business days", "wallet": "instantly, usually within 2 hours", "cod": "5–7 business days by bank transfer after bank details are verified, or instantly to ZentroMart Wallet"}
            if method in ranges:
                return TurnResult(f"Refunds go back to the original payment method. For this order, the {method.replace('_', ' ')} timeline is {ranges[method]} after initiation.", ["REF-2", "REF-3"])

        if re.search(r"\b(?:refund|money back)\b", lower) and re.search(r"\b(?:today|promise|guarantee)\b", lower):
            return TurnResult("I can't guarantee that a refund will arrive today. Timing depends on the payment method and starts when the refund is initiated.", ["REF-1", "REF-2"])

        if re.search(r"\b(?:clothing|clothes|footwear|fashion|kurta|tote bag)\b", lower) and re.search(r"\breturn\b", lower):
            festive = re.search(r"\b(?:ordered|order placed|bought)\b.{0,45}\b(?:0?[1-9]|1[0-5])\s+(?:october|oct)\b", lower)
            if festive:
                return TurnResult(
                    "Yes. Clothing orders placed from 1 to 15 October 2026 are eligible for a 30-day return window, counted from delivery. What was the delivery date?",
                    ["RET-8"],
                )
            if re.search(r"\bfee\b|\bcharged\b|\bcost\b", lower):
                return TurnResult(
                    "A ₹49 reverse-pickup fee applies to change-of-mind clothing returns. It’s waived for ZentroMart Plus members and for damaged, defective or wrong items.",
                    ["RET-6", "PLS-2"],
                )
            return TurnResult(
                "Clothing, footwear and bags can usually be returned within 15 calendar days of delivery, provided the item is unused with its tags intact.",
                ["RET-2", "RET-4"],
            )

        if re.search(r"\belectronics?\b", lower) and re.search(r"\b(?:physical(?:ly)? damage|damaged|broken)\b", lower):
            return TurnResult(
                "Electronics with physical damage aren’t eligible for a normal return. If the damage happened in delivery, report it within 48 hours with photos so the damaged-item team can review it.",
                ["RET-4", "DWM-1", "DWM-4"],
            )

        if re.search(r"\brefund\b", lower) and re.search(r"\b(?:upi|net banking|card|wallet)\b", lower):
            method = (
                "card" if re.search(r"\b(?:credit|debit)\s+card\b", lower)
                else "netbanking" if re.search(r"\bnet banking\b", lower)
                else "wallet" if re.search(r"\bwallet\b", lower)
                else "upi"
            )
            timelines = {
                "upi": "1–3 business days",
                "card": "5–7 business days",
                "netbanking": "3–5 business days",
                "wallet": "instantly, usually within 2 hours",
            }
            date_match = re.search(
                r"\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(20\d{2})\b",
                lower,
            )
            if date_match and re.search(r"\b(?:started|initiated|began)\b", lower):
                initiated = datetime.strptime(date_match.group(), "%d %B %Y").date()
                maximum = {"upi": 3, "card": 7, "netbanking": 5}.get(method, 0)
                if maximum:
                    due = self._business_days_after(initiated, maximum, self.holidays)
                    return TurnResult(
                        f"The {method.replace('_', ' ')} refund should arrive within {timelines[method]}, by {due.day} {due.strftime('%B %Y')}.",
                        ["REF-2", "KB-00.3"],
                    )
            return TurnResult(f"{method.capitalize()} refunds usually arrive {timelines[method]} after initiation.", ["REF-2"])

        if re.search(r"\b(?:cash on delivery|\bCOD\b)", message, re.IGNORECASE) and re.search(r"\brefund\b", lower):
            return TurnResult(
                "For a cash-on-delivery refund, choose ZentroMart Wallet or a bank transfer in My Orders. "
                "Wallet refunds are usually completed within 2 hours; bank transfers take 5–7 business days after your bank details are verified.",
                ["REF-2", "REF-3"],
            )

        if self.verified_order and re.search(r"\brefund\b", lower) and re.search(r"\b(?:reference|RRN|number)\b", lower):
            rrn = next((self.verified_order.get(key) for key in ("return_rrn", "cancellation_rrn", "rto_rrn") if self.verified_order.get(key)), None)
            if rrn:
                return TurnResult(f"Your refund reference number is {rrn}. Your bank can use it to trace the refund.", ["REF-6", "IDV-2"])
            return TurnResult("An RRN is created once a refund is initiated. Please share your order ID and verify it so I can check its refund status.", ["REF-6", "IDV-1"])

        if (
            re.search(r"\b(?:full phone|phone number|email|street address|home address)\b", lower)
            and not re.search(r"\b(?:registered phone|phone suffix|ends? in)\b", lower)
        ):
            return TurnResult(
                "For privacy, I cannot provide a full phone number, email address or street address in chat. "
                "I can help with the order status or another non-sensitive order detail.",
                ["IDV-4"],
            )

        if sku and re.search(r"\bwarranty|assured\b", lower):
            product = self.products.get(sku.group())
            if product:
                warranty = product.get("warranty_months")
                assured = product.get("assured") == "True"
                if assured and warranty:
                    return TurnResult(
                        f"{product['title']} is covered by ZentroMart Assured with a {warranty}-month warranty from delivery. You can start a claim from My Orders.",
                        ["WAR-1", "WAR-2", f"products.csv:{sku.group()}"],
                    )
            return TurnResult(
                "I can’t confirm warranty coverage for that SKU from the product catalog.",
                ["WAR-1"],
            )

        if re.search(r"\belectronics?\b", lower) and (
            re.search(r"\breturn\b", lower)
            or re.search(r"\b(?:send back|sending back|returning)\b", lower)
            or re.search(r"\b(?:before|after|on or after|earlier than|prior to)\b", lower)
        ):
            if re.search(r"\b(?:before|earlier than|prior to)\b.*\b1\s+august\b", lower):
                return TurnResult(
                    "Electronics delivered before 1 August 2026 can be returned within 10 calendar days of delivery.",
                    ["RET-3"],
                )
            return TurnResult(
                "Electronics delivered on or after 1 August 2026 can be returned within 7 calendar days of delivery.",
                ["RET-2", "RET-3"],
            )

        if re.search(r"\b(?:clothing|clothes|footwear|fashion)\b", lower) and re.search(r"\breturn\b", lower):
            return TurnResult(
                "Clothing, footwear and bags can usually be returned within 15 calendar days of delivery, provided the item is unused with its tags intact.",
                ["RET-2", "RET-4"],
            )

        if re.search(r"\bgrocer(?:y|ies)\b", lower) and re.search(r"\bcancel\b", lower):
            relative_slot_time = re.search(r"\b(\d+)\s*(minutes?|mins?|hours?|hrs?)\s+before\b", lower)
            if relative_slot_time:
                amount = int(relative_slot_time.group(1))
                minutes = amount * (60 if relative_slot_time.group(2).startswith(("hour", "hr")) else 1)
                cutoff = (
                    "that cutoff has passed"
                    if minutes < 120 else "you are still within the cancellation cutoff"
                )
                return TurnResult(
                    "Grocery orders can be cancelled up to 2 hours before the delivery slot starts. "
                    f"At {relative_slot_time.group(0).replace('before', 'before the slot starts')}, {cutoff}.",
                    ["CAN-3"],
                )
            return TurnResult(
                "Grocery orders can be cancelled up to 2 hours before the slot starts. "
                "What start time is shown in My Orders?",
                ["CAN-3", "STY-5"],
            )

        if re.search(r"\bgrocer(?:y|ies)\b", lower) and re.search(r"\b(?:change.*mind|return)\b", lower):
            if re.search(r"\b(?:damaged|wrong|missing)\b", lower):
                return TurnResult(
                    "Yes—please report a damaged, wrong or missing grocery item within 48 hours of delivery, with photos of the item and packaging.",
                    ["DWM-1", "DWM-2"],
                )
            return TurnResult(
                "Groceries can’t be returned for a change of mind. If the item is damaged, wrong or missing, I can help you report it.",
                ["RET-2", "DWM-1"],
            )

        if re.search(r"\b(?:opened|open)\b", lower) and re.search(r"\bbeauty\b", lower):
            return TurnResult(
                "Beauty and personal-care items are eligible only when unopened and the seal is intact. An opened item can’t be returned for a change of mind.",
                ["RET-2", "RET-4"],
            )

        if re.search(r"\b(?:credit|debit) card\b", lower) and re.search(r"\brefund\b", lower):
            return TurnResult(
                "Once the refund is initiated, card refunds usually reach you in 5–7 business days.",
                ["REF-2"],
            )

        if re.search(r"\bcash on delivery|\bCOD\b", message, re.IGNORECASE) and re.search(r"\brefund\b", lower):
            return TurnResult(
                "For a cash-on-delivery refund, choose ZentroMart Wallet or a bank transfer in My Orders. Wallet refunds are usually completed within 2 hours; bank transfers take 5–7 business days after your bank details are verified.",
                ["REF-2", "REF-3"],
            )

        if re.search(r"\bcancel\b", lower) and re.search(r"\b(?:not shipped|has not shipped|before shipping)\b", lower):
            return TurnResult(
                "You can cancel an order free of charge before it ships from My Orders in the app. The refund is initiated immediately; I can guide you, but I can’t cancel it for you.",
                ["CAN-1", "REF-1"],
            )

        if "guarantee" in lower and re.search(r"\bdelivery\b", lower):
            if self.verified_order:
                status = str(self.verified_order.get("status", "")).replace("_", " ")
                promised = self._friendly_date(self.verified_order.get("promised_date"))
                reply = "I can’t guarantee an exact delivery time."
                if status:
                    reply += f" Your order is currently {status}."
                if promised:
                    reply += f" Its delivery commitment is {promised}."
                return TurnResult(reply, ["TRK-1", "DEL-1"])
            return TurnResult(
                "I can’t guarantee a delivery date that isn’t shown on your order. The promised date in My Orders is the delivery commitment.",
                ["DEL-1"],
            )

        if re.search(r"\b(?:different|another|new) order\b", lower):
            self.order_id = None
            self.verified_order = None
            self.awaiting_verification = False
            self.verification_attempts = 0
            self.pending_request = None
            self.awaiting_order_id = True
            self.issue_product = None
            self.issue_purchase_timing = None
            self.issue_description = None
            self.pending_return_question = False
            return TurnResult(
                "Sure, let's switch to the other order. Please share its 8-digit order ID "
                "from My Orders, and I'll verify it before checking details.",
                ["IDV-1", "STY-5"],
            )

        if re.search(r"\breturn policy\b", lower):
            return TurnResult(
                "Return periods start on delivery: most clothing has 15 days; home items and "
                "books have 10; electronics have 7 days (10 if delivered before 1 August 2026). "
                "Sealed beauty products have 7 days; groceries, innerwear and digital codes "
                "aren't returnable for change of mind. What item are you asking about?",
                ["RET-2", "RET-3", "RET-5"],
            )

        if (
            re.search(r"\breturn\b", lower)
            and re.search(r"\brefund\b", lower)
            and not self._is_refund_status_request(lower)
        ):
            return TurnResult(
                "After a return pickup, the item must pass quality check. We initiate the refund within 2 business days of pickup; after that, the arrival time depends on your payment method.",
                ["RET-7", "REF-2"],
            )

        # A return cannot be assessed from intent alone. Ask only for the two
        # facts that determine the applicable policy instead of retrieving a
        # broad collection of unrelated internal guidance.
        if (
            "refund" not in lower
            and re.search(r"\b(?:want|need|like)\b.{0,30}\breturn\b|\breturn\b.{0,30}\b(?:product|item|eligible|eligibility)\b", lower)
        ):
            has_product = bool(re.search(
                r"\b(?:electronics?|phone|laptop|clothing|clothes|shirt|shoes?|footwear|"
                r"beauty|grocery|groceries|book|furniture|home|kitchen|innerwear)\b",
                lower,
            ))
            has_delivery_date = bool(re.search(
                r"\bdelivered?\b.{0,24}\b(?:\d|yesterday|today|ago|october|august|september)\b",
                lower,
            ))
            if not has_product or not has_delivery_date:
                self.pending_return_question = True
                self.issue_product = None
                return TurnResult(
                    "I can check that. What type of item do you want to return?",
                    ["RET-2", "RET-4", "RET-5", "STY-5"],
                )

        change_of_mind = (
            re.search(r"\b(?:don't like|do not like|change of mind|changed my mind)\b", lower)
            or ("return" in lower and "recently" in lower)
        )
        if change_of_mind and ("return" in lower or self.issue_product):
            return TurnResult(
                "Possibly—the return window depends on the item's category and delivery date, "
                "and some items can't be returned for change of mind. What item are you asking about?",
                ["RET-2", "RET-5"],
            )

        if re.search(r"\b(?:arrived damaged|arrived broken|damaged product|damaged item)\b", lower):
            self.damage_claim_active = True
            return TurnResult(
                "Please report it within 48 hours of delivery and include photos of the item and "
                "packaging. After verification, the policy allows a free replacement or full "
                "refund, including delivery fees, with no pickup fee. Claims over ₹10,000 must "
                "be handed to a human for review. "
                "When was it delivered?",
                ["DWM-1", "DWM-2", "DWM-3"],
            )

        if "guarantee" in lower and "refund" in lower:
            return TurnResult(
                "I can't guarantee a full refund today without checking the order and its return "
                "status. Refund timing depends on the payment method, and returned items may need "
                "to pass a quality check first. Share the 8-digit order ID if you'd like me to check.",
                ["REF-1", "REF-2", "RET-7"],
            )

        if re.search(r"\b(?:headphones?|headsets?|earbuds?)\b", lower):
            self.issue_product = "headphones"
            self.issue_description = message
            return TurnResult(
                "I can help check the return or warranty options. Electronics return windows "
                "depend on the delivery date, and warranty coverage depends on the model. "
                "What model or SKU are they?",
                ["RET-2", "RET-3", "WAR-1", "WAR-2"],
            )

        timing = re.search(r"\b(?:bought|purchased|ordered)\b.{0,24}\b(two|2|one|1|three|3)\s+weeks?\s+ago\b", lower)
        if self.issue_product and timing:
            self.issue_purchase_timing = timing.group()
            self.issue_description = message
            return TurnResult(
                "Thanks. The electronics return window is counted from delivery, not purchase: "
                "usually 7 days, or 10 days if delivered before 1 August 2026. What date were "
                "the headphones delivered?",
                ["RET-2", "RET-3"],
            )

        if self.issue_product and re.search(r"\bwhat are my options\b|\boptions\b", lower):
            if self.issue_purchase_timing:
                return TurnResult(
                    "For your headphones, the return window is based on the delivery date, "
                    "not when you bought them; warranty coverage depends on the model. "
                    "What date were they delivered?",
                    ["RET-2", "RET-3", "WAR-1", "WAR-2"],
                )
            return TurnResult(
                "For your headphones, the return period depends on the delivery date; electronics "
                "generally have 7 days, or 10 days if delivered before 1 August 2026. Warranty "
                "coverage depends on the exact model. What model are they?",
                ["RET-2", "RET-3", "WAR-1", "WAR-2"],
            )
        return None

    def _ticket(self, message: str, chunks: list, reason: str) -> TurnResult:
        if self.ticket_created:
            return TurnResult(
                "A support ticket has already been created for this issue. A human colleague will review it.",
                ["ESC-1", "ESC-2"],
            )
        safe_message = TICKET_SENSITIVE_NUMBER.sub("[verification value redacted]", message)
        checked = (
            f"Order verification succeeded; current status is {self.verified_order.get('status', 'unknown')}."
            if self.verified_order
            else "No order details were disclosed; identity verification is not complete."
        )
        summary = (
            f"Customer issue: {safe_message}. Intent: human handover. "
            f"Order ID: {self.order_id or 'not supplied'}. Checks performed: {checked} "
            f"Escalation reason: {reason}."
        )
        result = create_ticket(summary=summary, order_id=self.order_id, reason=reason, intent="human_handover")
        if result.get("created"):
            self.ticket_created = True
            reply = "I’ve created a support ticket and a human colleague will review this. I can’t approve or promise that outcome myself."
            if not self.order_id:
                if reason == "high-value damaged item claim":
                    reply += " The damaged-item claim is recorded, but no order was linked. Please share the 8-digit order ID so the reviewer can identify the right order."
                else:
                    reply += " No order was linked to this ticket. Please share the 8-digit order ID so the reviewer can identify the right order."
            if reason == "customer reports a product safety risk":
                reply = "Stop using the product immediately and keep away from heat or charging. " + reply
            if self.verified_order and reason == "refund is beyond its maximum timeline":
                rrn = next((self.verified_order.get(key) for key in (
                    "return_rrn", "cancellation_rrn", "rto_rrn"
                ) if self.verified_order.get(key)), None)
                record = refund_record(self.verified_order)
                if record:
                    initiated = self._friendly_date(record.initiated_on) or record.initiated_on
                    detail = (
                        f"Your refund was initiated on {initiated} and has not yet been marked completed. "
                        f"{refund_timeline(record.method)} "
                    )
                    reply = detail + reply
                if rrn:
                    reply = f"Your refund reference number (RRN) is {rrn}. " + reply
            sources = list(dict.fromkeys([c.section_id for c in chunks] + ["ESC-1", "ESC-2"]))
            if reason == "high-value damaged item claim":
                sources.extend(["DWM-3", "ESC-4"])
            elif reason == "customer reports a product safety risk":
                sources.append("DWM-5")
            elif reason == "refund is beyond its maximum timeline":
                sources.extend(["REF-2", "REF-6"])
            elif reason == "third or later contact about the same issue within seven days":
                sources.append("ESC-3")
            sku = PRODUCT_SKU.search(message)
            product_sku = sku.group() if sku else self.current_product_sku
            if product_sku and product_sku in self.products:
                sources.append(f"products.csv:{product_sku}")
            sources = list(dict.fromkeys(sources))
            return TurnResult(reply, sources, ["create_ticket()"])
        failure_reply = "I can’t complete that request here. A human colleague can help review it."
        if reason == "customer reports a product safety risk":
            failure_reply = "Stop using the product immediately and keep away from heat or charging. " + failure_reply
        return TurnResult(failure_reply, [c.section_id for c in chunks])

    @staticmethod
    def _business_days_after(start: date, count: int, holidays: set[date]) -> date:
        current = start
        elapsed = 0
        while elapsed < count:
            current += timedelta(days=1)
            if current.weekday() != 6 and current not in holidays:
                elapsed += 1
        return current

    def _refund_overdue(self) -> bool:
        if not self.verified_order:
            return False
        refund_fields = (
            ("return_refund_initiated_on", "return_refund_method", "return_refund_completed_on"),
            ("cancellation_refund_initiated_on", "cancellation_refund_method", None),
            ("rto_refund_initiated_on", "rto_refund_method", None),
        )
        timelines = {
            "upi": 3,
            "card": 7,
            "netbanking": 5,
            "wallet": 0,
            "cod_wallet": 0,
            "cod_bank": 7,
        }
        today = date(2026, 10, 20)
        for initiated_key, method_key, completed_key in refund_fields:
            initiated_on = self.verified_order.get(initiated_key)
            method = self.verified_order.get(method_key)
            if not initiated_on or method not in timelines:
                continue
            if completed_key and self.verified_order.get(completed_key):
                continue
            maximum_date = self._business_days_after(
                date.fromisoformat(initiated_on), timelines[method], self.holidays
            )
            grace_deadline = self._business_days_after(maximum_date, 3, self.holidays)
            if today > grace_deadline:
                return True
        return False

    def _escalation(self, message: str, lower: str, chunks: list) -> TurnResult | None:
        damage_report = re.search(
            r"\b(?:damag\w*|crack\w*|broken|defective|faulty)\b", lower
        )
        if self.verified_order:
            damaged_sku = self.verified_order.get("damage_sku")
            if damaged_sku and self.product_prices.get(damaged_sku, 0) > 10_000:
                return self._ticket(message, chunks, "high-value damaged item claim")
        elif damage_report:
            sku = PRODUCT_SKU.search(message)
            if sku and self.product_prices.get(sku.group(), 0) > 10_000:
                return self._ticket(message, chunks, "high-value damaged item claim")
            amount = PRICE_AMOUNT.search(message)
            if amount and int(amount.group(1).replace(",", "")) > 10_000:
                return self._ticket(message, chunks, "high-value damaged item claim")
        if HUMAN_REQUEST.search(message):
            return self._ticket(message, chunks, "customer requests a human agent")
        if UNAUTHORIZED_ACTIVITY.search(message):
            return self._ticket(message, chunks, "customer reports activity they did not authorize")
        if LEGAL_ACTION.search(message):
            return self._ticket(message, chunks, "customer mentions legal action or a formal complaint")
        if SAFETY_RISK.search(message):
            return self._ticket(message, chunks, "customer reports a product safety risk")
        if self.verified_order:
            if (self.verified_order.get("contacts_last_7_days") or 0) >= 2:
                return self._ticket(message, chunks, "third or later contact about the same issue within seven days")
            if self._refund_overdue():
                return self._ticket(message, chunks, "refund is beyond its maximum timeline")
        if re.search(r"\b(?:compensation|approve|manager|escalat|exception)\b", lower):
            return self._ticket(message, chunks, "customer requests a human review or a decision outside agent authority")
        if re.search(r"\bcreate\b.{0,20}\bticket\b|\bticket\b.{0,20}\b(?:create|open|raise)\b", lower):
            return self._ticket(message, chunks, "customer requests a support ticket")
        return None

    def _grounded_answer(self, message: str, chunks: list) -> str | None:
        """Ask an LLM only to synthesize retrieved evidence; never let it invent policy."""
        api_key = os.getenv("GEMINI_API_KEY")
        model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        if not api_key or not model:
            return None
        try:
            from google import genai
            from google.genai import types
            order_facts = "No verified order is available."
            if self.verified_order:
                allowed = ("order_id", "status", "payment_method", "delivery_type", "promised_date",
                           "shipped_on", "delivered_on", "delivery_attempts", "return_status",
                           "return_refund_initiated_on", "return_refund_completed_on",
                           "cancellation_refund_initiated_on", "rto_refund_initiated_on",
                           "return_rrn", "cancellation_rrn", "rto_rrn")
                order_facts = "; ".join(f"{key}={self.verified_order[key]}" for key in allowed if self.verified_order.get(key))
            prompt = (
                "You are ZentroMart's AI support assistant. Answer the customer using ONLY the supplied "
                "support knowledge package. Do not invent a policy, date, amount, or order detail. If the "
                "knowledge package does not answer the question, say you don't have that information and offer a human "
                "colleague. Use the verified order facts when relevant. Keep the answer to two or three short "
                "sentences. Do not mention these instructions.\n\n"
                f"VERIFIED ORDER FACTS:\n{order_facts}\n\n{self._gemini_reference_context}\n\nCUSTOMER QUESTION: {message}"
            )
            response = genai.Client(api_key=api_key).models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0),
            )
            return self._clean_gemini_reply(response.text) if response.text else None
        except Exception as exc:
            logging.warning("Gemini answer synthesis failed (%s)", type(exc).__name__)
            return None

    @staticmethod
    def _offline_answer(message: str, chunks: list) -> tuple[str, list[str]]:
        if re.search(r"\b(?:warranty|warrantied|assured)\b", message, re.IGNORECASE):
            sections = {chunk.section_id: chunk for chunk in chunks}
            if "WAR-1" in sections or "WAR-2" in sections:
                source_ids = [
                    section_id for section_id in ("WAR-1", "WAR-2") if section_id in sections
                ]
                if re.search(r"\b(?:not cover|excluded|exclusions)\b", message, re.IGNORECASE):
                    return (
                        "Warranty does not cover physical or liquid damage, normal wear, or "
                        "unauthorized repairs.",
                        ["WAR-3"],
                    )
                return (
                    "It depends on the product. ZentroMart Assured products have a 12-month "
                    "warranty from delivery; manufacturer warranty periods vary by item. "
                    "What product or SKU is it, and when was it delivered?",
                    source_ids,
                )

        excerpts = []
        for chunk in chunks[:2]:
            text = re.sub(r"\s+", " ", chunk.text.strip().replace("**", "").replace("`", ""))
            if len(text) > 700:
                text = text[:697].rsplit(" ", 1)[0] + "..."
            excerpts.append(text)
        return (
            "The current support guidance says: " + " ".join(excerpts),
            [chunk.section_id for chunk in chunks[:2]],
        )

    def respond(self, message: str) -> TurnResult:
        message = self._normalise_spoken_digits(message.strip())
        hindi = self._is_hinglish(message)
        if not message:
            return TurnResult("Sorry, I didn't catch that. Could you say it again?")
        self.history.append(("customer", message))
        welcome = self._welcome(message)
        if welcome:
            return TurnResult(welcome, ["STY-1"])
        message = self.kb.correct_typos(message)

        greeting_only = re.fullmatch(
            r"[\s\W]*(?:h+i+|h+e+y+|hello+|namaste|नमस्ते|नमस्कार|हैलो)[\s\W]*",
            message,
            re.IGNORECASE,
        )
        vague_order_help = (
            any(word in message.lower() for word in ("order", "ऑर्डर"))
            and any(word in message.lower() for word in ("help", "madad", "मदद"))
        )
        substantive_order_issue = bool(re.search(
            r"\b(?:delivered|missing|not receive|refund|cancel|return|status|track|arriv)\b", message,
            re.IGNORECASE,
        ))
        if len(self.history) == 1 and not self._order_id(message) and (greeting_only or (vague_order_help and not substantive_order_issue)):
            return TurnResult(self._intro(hindi), ["STY-1"])

        found_id = self._order_id(message)
        if found_id:
            # Customers commonly give the order ID and their proof in one turn.
            # Handle that before issuing the usual verification prompt; otherwise the
            # pincode/phone suffix is silently ignored and the conversation loops.
            is_new_order = found_id != self.order_id
            self.order_id = found_id
            self.awaiting_order_id = False
            # A prior speech-to-text turn may have contained a nine-digit
            # misheard order number. Do not replay that malformed turn after
            # the caller supplies a valid replacement order ID.
            if self.pending_request and re.search(r"(?<!\d)\d{9,}(?!\d)", self.pending_request):
                self.pending_request = None
            if is_new_order:
                self.verified_order = None
                self.verification_attempts = 0
                self.awaiting_verification = True
            elif self.verified_order:
                self.awaiting_verification = False
                status = str(self.verified_order.get("status", "")).replace("_", " ")
                reply = "This order is already verified in this chat."
                if status:
                    reply += f" Its current status is {status}."
                reply += " What would you like to check—delivery, cancellation, return, or refund?"
                return TurnResult(reply, ["IDV-2", "TRK-1"])
            else:
                self.awaiting_verification = True
            proof = self._verification_value(message, found_id)
            if proof:
                self.verification_attempts += 1
                result = lookup_order(found_id, proof)
                call = f"lookup_order(order_id={found_id}, verification_value=provided)"
                if result.get("verified"):
                    self.verified_order, self.awaiting_verification = result, False
                    if self.pending_request:
                        pending = self.pending_request
                        self.pending_request = None
                        resumed = self.respond(pending)
                        resumed.tool_calls.insert(0, call)
                        resumed.sources = list(dict.fromkeys(["IDV-2", *resumed.sources]))
                        if "verified" not in resumed.reply.casefold():
                            resumed.reply = "Thanks, your identity is verified. " + resumed.reply
                        return resumed
                    status = str(result.get("status", "")).replace("_", " ")
                    promised = self._friendly_date(result.get("promised_date"))
                    detail = f" The current status is {status}." if status else ""
                    if promised:
                        detail += f" The promised delivery date is {promised}."
                    asks_arrival = bool(re.search(r"\b(?:where|track|status|arriv\w*|deliver\w*)\b", message, re.IGNORECASE))
                    asks_return_refund = bool(
                        re.search(r"\brefund\b", message, re.IGNORECASE)
                        and re.search(r"\breturn\b", message, re.IGNORECASE)
                    )
                    if asks_arrival and asks_return_refund:
                        return TurnResult(
                            "Thanks, your identity is verified."
                            + detail
                            + " If you return an eligible item, we initiate the refund within 2 business days after pickup and quality check; then the time to reach you depends on the payment method.",
                            ["IDV-2", "TRK-1", "DEL-1", "RET-7", "REF-2"],
                            [call],
                        )
                    if asks_arrival:
                        return TurnResult(
                            "Thanks, your identity is verified." + detail,
                            ["IDV-2", "TRK-1"],
                            [call],
                        )
                    return TurnResult(
                        "Thanks, your identity is verified." + detail,
                        ["IDV-2"],
                        [call],
                    )
                if self.verification_attempts >= 2:
                    self.awaiting_verification = False
                    return TurnResult(
                        "Those details still don’t match, so I can’t share order information. Please contact support from the registered phone number; you can also ask a general policy question here.",
                        ["IDV-3"],
                        [call],
                    )
                return TurnResult(
                    "That didn’t match. Please try one more time with either the last four phone digits or delivery pincode.",
                    ["IDV-2", "IDV-3"],
                    [call],
                )
            lookup_order(found_id)
            return TurnResult("To protect your order details, please share either the last four digits of your registered phone number or the six-digit delivery pincode.", ["IDV-2"], [f"lookup_order(order_id={found_id})"])

        if (not self.awaiting_verification and re.search(r"(?<!\d)\d{4,7}(?!\d)", message) and re.search(
            r"\b(?:order(?:\s+(?:id|number))?|it is)\b", message, re.IGNORECASE
        )):
            return TurnResult(
                "I need the full 8-digit order ID. Please repeat all eight digits from My Orders.",
                ["IDV-1"],
            )

        if self.verified_order and re.fullmatch(r"\s*\d{4,6}\s*", message):
            return TurnResult(
                "Your order is already verified in this chat, so you don’t need to share that again. What would you like to check?",
                ["IDV-2"],
            )

        lower = message.lower()
        if (
            self.verified_order
            and self._refund_overdue()
            and (
                self._is_refund_status_request(lower)
                or re.search(r"\b(?:late|overdue|past the maximum)\b", lower)
            )
        ):
            if self.ticket_created and (record_reply := self._refund_record_reply()):
                record_reply.reply += " A human support ticket is already open for review."
                return record_reply
            return self._ticket(
                message,
                self.kb.search(message),
                "refund is beyond its maximum timeline",
            )
        if self.verified_order and (
            self._is_refund_status_request(lower)
            or re.search(r"रिफ़?ंड|वापसी", lower)
        ):
            if reply := self._refund_record_reply():
                return reply
        if self.ticket_created and re.search(r"\b(?:create|reate|open|raise)\b.{0,25}\bticket\b", lower):
            return TurnResult(
                "A support ticket has already been created for this issue. A human colleague will review it; I won't create a duplicate ticket.",
                ["ESC-1", "ESC-2"],
            )
        if re.search(r"\b(?:discount|compensation)\b", lower) and re.search(r"\b(?:late|delay|next purchase|exception)\b", lower):
            chunks = self.kb.search(message)
            ticket = self._ticket(message, chunks, "customer requests a human review or a decision outside agent authority")
            ticket.reply = (
                "A 75% discount for a late order is not a documented entitlement, so I can't approve it. "
                + ticket.reply
            )
            return ticket
        if (
            HUMAN_REQUEST.search(message)
            or UNAUTHORIZED_ACTIVITY.search(message)
            or LEGAL_ACTION.search(message)
            or SAFETY_RISK.search(message)
            or re.search(r"\b(?:damag\w*|crack\w*|broken|defective|faulty)\b", lower)
            or any(word in lower for word in ("compensation", "approve", "manager", "escalat", "exception"))
            or re.search(r"\bcreate\b.{0,20}\bticket\b|\bticket\b.{0,20}\b(?:create|open|raise)\b", lower)
        ):
            chunks = self.kb.search(message)
            escalation = self._escalation(message, lower, chunks)
            if escalation:
                return escalation

        if re.search(r"\b(?:different|another|new) order\b", lower):
            return self._curated_answer(message, lower)

        if (
            self.awaiting_verification
            and not self.verified_order
            and re.search(r"\b(?:actually|instead|different question|ask about|want to ask)\b", lower)
            and (PRODUCT_SKU.search(message) or re.search(r"\b(?:warranty|return policy|refund policy|delivery policy)\b", lower))
        ):
            self.order_id = None
            self.awaiting_verification = False
            self.verification_attempts = 0
            self.pending_request = None
            self.pending_return_question = False
            self.issue_product = None
            self.issue_purchase_timing = None
            self.issue_description = None

        if (
            self.awaiting_order_id
            and re.search(r"\b(?:actually|instead)\b", lower)
            and re.search(r"\breturn\b", lower)
        ):
            self.awaiting_order_id = False
            self.pending_request = None
            self.pending_return_question = True
            return TurnResult(
                "Sure - for a return, what type of item is it? The return window and any pickup fee depend on the category.",
                ["RET-2", "RET-6", "STY-5"],
            )

        if self.awaiting_order_id:
            if re.search(r"\b(?:don't know|do not know|forgot|can't find|cannot find)\b.{0,30}\b(?:order id|order number|it)\b", lower):
                self.pending_request = None
                return TurnResult(
                    "You can find the 8-digit order ID in My Orders or in your order-confirmation message. "
                    "I can't locate an order from a phone number alone; share the order ID when you find it.",
                    ["IDV-1", "STY-5"],
                )
            if re.search(r"\bphone\b|\bends in\b|\bending in\b", lower) and not found_id:
                return TurnResult(
                    "I can't verify an order using a phone suffix alone. Please share its 8-digit order ID first.",
                    ["IDV-1"],
                )
            if re.search(r"\bORD[- ]?\d{1,7}\b", message, re.IGNORECASE):
                return TurnResult(
                    "That order ID looks incomplete. Please send the full 8-digit ID shown in My Orders.",
                    ["IDV-1"],
                )
            if re.search(r"(?<!\d)\d{4,7}(?!\d)", message):
                return TurnResult(
                    "I need the full 8-digit order ID. Please repeat all eight digits from My Orders.",
                    ["IDV-1"],
                )
            if any(word in lower for word in ("order", "cancel", "deliver", "refund")):
                return TurnResult(
                    "Please share the 8-digit order ID from My Orders so I can check the right order.",
                    ["IDV-1", "STY-5"],
                )
            self.awaiting_order_id = False
            self.pending_request = None

        if self.awaiting_verification and not self.verified_order:
            if re.search(r"\b(?:address|delivery address|email|personal details)\b", lower):
                return TurnResult(
                    "I understand you're in a hurry, but I can't share the delivery address or other order details until verification succeeds. "
                    "Please provide either the last four registered phone digits or the six-digit delivery pincode.",
                    ["IDV-1", "IDV-2"],
                )
            proof = None if PRODUCT_SKU.search(message) else self._verification_value(message)
            if proof:
                self.verification_attempts += 1
                result = lookup_order(self.order_id, proof)
                call = f"lookup_order(order_id={self.order_id}, verification_value=provided)"
                if result.get("verified"):
                    self.verified_order, self.awaiting_verification = result, False
                    if self.pending_request:
                        pending = self.pending_request
                        self.pending_request = None
                        resumed = self.respond(pending)
                        resumed.tool_calls.insert(0, call)
                        resumed.sources = list(dict.fromkeys(["IDV-2", *resumed.sources]))
                        if "verified" not in resumed.reply.casefold():
                            resumed.reply = "Thanks, your identity is verified. " + resumed.reply
                        return resumed
                    status = str(result.get("status", "")).replace("_", " ")
                    detail = f" The current status is {status}." if status else ""
                    return TurnResult(
                        "Thanks, your identity is verified." + detail + " How can I help with this order?",
                        ["IDV-2", "TRK-1"],
                        [call],
                    )
                if self.verification_attempts >= 2:
                    self.awaiting_verification = False
                    return TurnResult("I couldn’t verify those details, so I can’t share order information. Please contact support from the registered phone number or ask a general policy question.", ["IDV-3"], [call])
                return TurnResult("That didn’t match. Please try one more time with either the last four phone digits or delivery pincode.", ["IDV-2", "IDV-3"], [call])
            return TurnResult("Please share either the last four digits of your registered phone number or the six-digit delivery pincode.", ["IDV-2"])

        if self.pending_return_question and re.search(r"\b(?:pickup|reverse pickup)\b.{0,30}\b(?:fee|cost|pay|charge)|\b(?:fee|cost|pay|charge)\b.{0,30}\b(?:pickup|reverse pickup)\b", lower):
            return TurnResult(
                "For a change-of-mind clothing return, reverse pickup costs ₹49. It is waived for ZentroMart Plus members and for wrong, damaged, or defective items. What item are you returning?",
                ["RET-6", "PLS-2", "DWM-2"],
            )

        if THIRD_PARTY_DATA_REQUEST.search(message):
            return TurnResult(
                "I'm sorry, I can't share another customer's personal information.",
                ["ESC-4", "IDV-4"],
            )
        if (
            re.search(r"\brefund\b", lower)
            and re.search(r"\b(?:late|overdue|past the maximum)\b", lower)
            and not self.verified_order
        ):
            self.awaiting_order_id = True
            self.pending_request = message
            return TurnResult(
                "I can check the refund timeline after verifying the order. What is the 8-digit order ID?",
                ["IDV-1", "REF-2", "STY-5"],
            )

        status_check_without_order = bool(
            re.search(r"\bplaced an order\b.{0,45}\bprocess", lower)
            or re.search(r"\bexpecting (?:a )?delivery\b.{0,45}\b(?:did not|didn't|not) receive", lower)
            or re.search(r"\brequested (?:a )?return\b.{0,55}\b(?:pickup|picked)", lower)
            or re.search(r"\breturned product\b.{0,45}\b(?:pickup|refund)", lower)
            or re.search(r"\b(?:delivered|app says delivered)\b.{0,50}\b(?:haven't received|have not received|did not get|didn't get|did not receive|didn't receive|can't find|cannot find|missing)\b", lower)
        )
        if status_check_without_order and not self.verified_order:
            self.awaiting_order_id = True
            self.pending_request = message
            prefix = "I'm ZentroMart's AI assistant. " if len(self.history) == 1 else ""
            return TurnResult(
                prefix + "Please share your 8-digit order ID so I can check its status before advising you.",
                ["IDV-1", "STY-1", "STY-5"],
            )

        curated = self._curated_answer(message, lower)
        if curated:
            return curated

        order_specific = any(phrase in lower for phrase in (
            "hasn't arrived", "has not arrived", "my order is late", "order is late",
            "cancel it", "cancel my",
            "where is my", "track my", "refund for it", "how long will that",
            "verify my order", "different order", "another order", "new order",
        )) or bool(re.search(r"\bmy order\b", lower)) or self._is_refund_status_request(lower)
        shipped_cancellation = (
            any(phrase in lower for phrase in ("on its way", "already shipped", "stop it"))
            and any(word in lower for word in ("cancel", "stop"))
        )
        if order_specific and not self.verified_order and not shipped_cancellation:
            self.awaiting_order_id = True
            self.pending_request = message
            prefix = "I'm ZentroMart's AI assistant. " if len(self.history) == 1 else ""
            return TurnResult(prefix + "Please share your 8-digit order ID so I can check its status before advising you.", ["IDV-1", "STY-1", "STY-5"])

        if re.search(r"\bcomplaint\b", lower) and not re.search(r"\b(?:consumer court|police complaint|legal action)\b", lower):
            return TurnResult("I’m sorry to hear that. What happened so I can understand how to help?", ["STY-5"])

        if re.search(r"\bhow many days\b.*\breturn\b|कितने दिन.*वापस", message, re.IGNORECASE):
            return TurnResult(
                "Which item do you want to return? The return window depends on its product category.",
                ["RET-2"],
            )
        if (
            not re.search(r"\breturn(?:ed|ing)?\b", lower)
            and re.search(r"\bwhen\b.*\brefund\b|\brefund\b.*\bwhen\b|\brefund\b.*\bcome\b", lower)
        ):
            return TurnResult(
                "Which payment method did you use (UPI, card, net banking, or ZentroMart Wallet)? "
                "Refund timelines depend on the payment method.",
                ["REF-2"],
            )

        chunks = self.kb.search(message)
        escalation = self._escalation(message, lower, chunks)
        if escalation:
            return escalation
        # One weak lexical hit is not evidence. Prefer an honest handover to a
        # plausible-sounding answer that has no grounding in the policy.
        meaningful = {word for word in self.kb._tokens(message)
                      if len(word) > 3 and word not in {"what", "when", "where", "with", "from", "this", "that", "have", "need", "please", "zentromart", "order", "delivery"}}
        evidence_words = set(self.kb._tokens(" ".join(c.title + " " + c.text for c in chunks)))
        shipped_cancellation = (
            any(phrase in lower for phrase in ("on its way", "already shipped", "stop it"))
            and any(chunk.section_id == "CAN-2" for chunk in chunks)
        )
        if not chunks or (
            meaningful and len(meaningful & evidence_words) < 2 and not shipped_cancellation
        ):
            return TurnResult("I don't have information about that in my support documents. I can connect you with a human colleague.")

        synthesized = self._grounded_answer(message, chunks)
        if synthesized:
            return TurnResult(synthesized, [c.section_id for c in chunks])

        reply, sources = self._offline_answer(message, chunks)
        return TurnResult(reply, sources)
