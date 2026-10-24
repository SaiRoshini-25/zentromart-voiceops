"""Deterministic business rules used by the dialogue layer.

These functions interpret verified order data only.  They intentionally do not
decide whether an order may be disclosed; that remains the verification
state-machine's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RefundRecord:
    initiated_on: str
    method: str
    amount_inr: int | None
    completed_on: str | None
    kind: str


_REFUND_FIELDS = (
    ("return", "return_refund_initiated_on", "return_refund_method", "return_refund_amount_inr", "return_refund_completed_on"),
    ("cancellation", "cancellation_refund_initiated_on", "cancellation_refund_method", "cancellation_refund_amount_inr", None),
    ("rto", "rto_refund_initiated_on", "rto_refund_method", "rto_refund_amount_inr", None),
)

REFUND_METHOD_LABELS = {
    "upi": "UPI",
    "card": "credit or debit card",
    "netbanking": "net banking",
    "wallet": "ZentroMart Wallet",
    "cod_wallet": "ZentroMart Wallet",
    "cod_bank": "bank transfer (NEFT)",
}

REFUND_TIMELINES = {
    "upi": "UPI refunds usually reach you within 1–3 business days after initiation.",
    "card": "Card refunds usually reach you within 5–7 business days after initiation.",
    "netbanking": "Net banking refunds usually reach you within 3–5 business days after initiation.",
    "wallet": "Wallet refunds are usually completed within 2 hours of initiation.",
    "cod_wallet": "Wallet refunds are usually completed within 2 hours of initiation.",
    "cod_bank": "Bank-transfer refunds usually reach you within 5–7 business days after initiation.",
}


def refund_record(order: Mapping[str, object]) -> RefundRecord | None:
    """Get the recorded refund, preserving a missing completion date as pending."""
    for kind, initiated_key, method_key, amount_key, completed_key in _REFUND_FIELDS:
        initiated = order.get(initiated_key)
        if not initiated:
            continue
        raw_amount = order.get(amount_key)
        try:
            amount = int(raw_amount) if raw_amount not in (None, "") else None
        except (TypeError, ValueError):
            amount = None
        return RefundRecord(
            initiated_on=str(initiated),
            method=str(order.get(method_key) or "").lower(),
            amount_inr=amount,
            completed_on=str(order[completed_key]) if completed_key and order.get(completed_key) else None,
            kind=kind,
        )
    return None


def refund_method_label(method: str) -> str:
    return REFUND_METHOD_LABELS.get(method, method.replace("_", " "))


def refund_timeline(method: str) -> str:
    return REFUND_TIMELINES.get(method, "The arrival time depends on the refund method.")
