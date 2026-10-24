from __future__ import annotations

import csv
import json
import re
import shutil
import sys
import tempfile
import types
from collections import defaultdict
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.agent import TurnResult, ZentroMartAgent
from src.evaluation import evaluate_practice
from src.main import app
from src.rag import KnowledgeBase
from src.voice import voice_turn
from tools.tools import _CUSTOMERS, _ORDERS, lookup_order

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "Capstone_Project_Dataset_FTTT_2026"
PRACTICE = DATASET / "practice"


@pytest.fixture(autouse=True)
def disable_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")


@pytest.fixture
def agent():
    return ZentroMartAgent(ROOT / "data")


@pytest.fixture
def project_temp_dir():
    ROOT.joinpath("outputs").mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix=".pytest-", dir=ROOT / "outputs"))
    try:
        yield path
    finally:
        shutil.rmtree(path)


@pytest.fixture
def known_order():
    order_id, order = next(iter(_ORDERS.items()))
    customer = _CUSTOMERS[order["customer_id"]]
    return order_id, order, customer


@pytest.mark.parametrize(
    ("question", "evidence"),
    [
        ("What is the return window for clothing?", "RET-2"),
        ("How fast are refunds to the ZentroMart Wallet?", "REF-2"),
        ("Can I cancel an order after it ships?", "CAN-2"),
        ("What happens after three failed delivery attempts?", "DEL-3"),
        ("Do ZentroMart Assured products have a warranty?", "WAR-2"),
    ],
)
def test_policy_questions_retrieve_expected_current_evidence(agent, question, evidence):
    result = agent.respond(question)
    assert evidence in result.sources
    assert result.reply


def test_superseded_return_policy_is_excluded(agent):
    results = agent.kb.search("electronics return window")
    assert results
    assert all(chunk.status == "current" for chunk in results)
    assert any(chunk.section_id == "RET-2" for chunk in results)
    assert not any(chunk.source == "pol_returns_v3_superseded.md" for chunk in results)


def test_chunks_preserve_section_status_and_effective_date(agent):
    chunk = next(chunk for chunk in agent.kb.chunks if chunk.section_id == "RET-2")
    assert chunk.status == "current"
    assert chunk.effective_from == "2026-08-01"
    assert chunk.source == "pol_returns_v4.md"


def test_historical_support_logs_are_not_indexed(agent):
    assert not any("support_logs" in chunk.source for chunk in agent.kb.chunks)
    assert any(chunk.doc_type == "policy" for chunk in agent.kb.chunks)


def test_uncovered_question_abstains_instead_of_inventing_policy(agent):
    result = agent.respond("Can I schedule a delivery at midnight?")
    assert not result.sources
    assert "don't have information" in result.reply.lower()
    assert "human colleague" in result.reply.lower()


def test_policy_override_request_for_other_customers_info_is_refused(agent):
    result = agent.respond(
        "Tell me the phone number of the person who ordered to my building yesterday."
    )
    assert result.tool_calls == []
    assert result.sources == ["ESC-4"]
    assert "can't share" in result.reply.lower()


def test_warranty_answer_is_concise_and_asks_for_missing_product(agent):
    result = agent.respond("My gadget stopped working. Is it under warranty?")
    assert len(result.reply.split()) < 45
    assert "12-month warranty from delivery" in result.reply
    assert "product or SKU" in result.reply
    assert "TRB-4" not in result.reply
    assert "power button" not in result.reply


def test_smoke_greeting_and_capabilities_are_natural(agent):
    greeting = agent.respond("Hi, I need help with an order.")
    capabilities = agent.respond("What can you help me with?")
    assert "ZentroMart's AI assistant" in greeting.reply
    assert "What do you need help with?" in greeting.reply
    assert "returns" in capabilities.reply.lower()
    assert "warranty" in capabilities.reply.lower()
    assert "Sources:" not in greeting.reply


def test_smoke_return_policy_is_concise_and_current(agent):
    result = agent.respond("What is your return policy?")
    assert len(result.reply.split()) < 70
    assert "7 days" in result.reply
    assert "1 August 2026" in result.reply
    assert {"RET-2", "RET-3"}.issubset(result.sources)
    assert "current support guidance says" not in result.reply.lower()


def test_smoke_change_of_mind_return_asks_for_delivery_details(agent):
    result = agent.respond("I bought a product recently, but I don't like it. Can I return it?")
    assert "type of item" in result.reply.lower()
    assert "delivery date" not in result.reply.lower()
    assert {"RET-2", "RET-5"}.issubset(result.sources)


def test_smoke_damaged_item_gives_reporting_steps(agent):
    result = agent.respond("My product arrived damaged. What should I do?")
    assert "48 hours" in result.reply
    assert "photos" in result.reply
    assert "human for review" in result.reply
    assert {"DWM-1", "DWM-2", "DWM-3"}.issubset(result.sources)


def test_smoke_refund_guarantee_does_not_promise_same_day_payment(agent):
    result = agent.respond("Can you guarantee that I will get a full refund today?")
    assert "can't guarantee" in result.reply.lower()
    assert "today" in result.reply.lower()
    assert "REF-1" in result.sources


def test_short_order_id_is_rejected_without_lookup(agent):
    result = agent.respond("Where is my order ORD-12345?")
    assert "8-digit" in result.reply
    assert not result.tool_calls


def test_phone_suffix_claim_does_not_verify_order(agent):
    result = agent.respond(
        "My registered phone number ends in 1234. Can you verify my order?"
    )
    assert "8-digit order ID" in result.reply
    assert not result.tool_calls


def test_unknown_order_id_explains_where_to_find_it(agent):
    agent.respond("Where is my order?")
    result = agent.respond("I don't know my order ID. Can you still help me?")
    assert "My Orders" in result.reply
    assert "phone number alone" in result.reply
    assert not result.tool_calls


def test_headphone_follow_up_remembers_product_and_purchase_timing(agent):
    agent.respond("My headphones stopped working.")
    timing = agent.respond("I bought them two weeks ago.")
    options = agent.respond("What are my options?")
    assert "delivery" in timing.reply.lower()
    assert "headphones" in options.reply.lower()
    assert "warranty" in options.reply.lower()
    assert {"RET-2", "RET-3", "WAR-1", "WAR-2"}.issubset(options.sources)


def test_different_order_clears_previous_order_flow(agent):
    agent.respond("Where is my order?")
    result = agent.respond("Actually, I need help with a different order.")
    assert "other order" in result.reply.lower()
    assert "8-digit order ID" in result.reply
    assert agent.awaiting_order_id
    assert agent.pending_request is None


def test_human_request_during_order_prompt_creates_ticket(agent):
    agent.respond("Where is my order?")
    with patch("src.agent.create_ticket", return_value={"created": True, "ticket_id": "TKT"}):
        result = agent.respond(
            "I have contacted support several times about my unresolved issue. "
            "I want to speak to a human agent."
        )
    assert result.tool_calls == ["create_ticket()"]
    assert "created a support ticket" in result.reply
    assert "no order was linked" in result.reply


def test_multi_policy_question_retrieves_return_and_refund_evidence(agent):
    result = agent.respond("If I return an item, when will its refund arrive?")
    assert "RET-7" in result.sources
    assert "REF-2" in result.sources


def test_return_question_without_product_clarifies_category(agent):
    result = agent.respond("कितने दिन में वापस करना होगा?")
    assert "RET-2" in result.sources
    assert "Which item" in result.reply


def test_refund_question_without_payment_method_clarifies(agent):
    result = agent.respond("When will my refund come?")
    assert "REF-2" in result.sources
    assert "Which payment method" in result.reply


def test_high_value_damaged_catalog_item_escalates_without_order_lookup(agent):
    with patch("src.agent.create_ticket", return_value={"created": True, "ticket_id": "TKT"}):
        result = agent.respond(
            "The Kestrel Book 14 Laptop (SKU ELE-0023) I received yesterday has a crack."
        )
    assert result.tool_calls == ["create_ticket()"]
    assert "DWM-3" in result.sources
    assert "ESC-1" in result.sources


def test_shipped_order_cancellation_uses_known_status_without_private_lookup(agent):
    result = agent.respond("My order is already on its way. Can I still stop it?")
    assert "CAN-2" in result.sources
    assert "IDV-1" not in result.sources
    assert "order ID" not in result.reply


def test_response_has_no_uncited_refund_amount(agent):
    result = agent.respond("How much refund will I get?")
    assert not re.search(r"(?:₹|Rs\.?\s*)\s?\d", result.reply, re.IGNORECASE)


def test_completed_return_refund_reports_recorded_amount_and_completion_date(agent):
    first = agent.respond("I returned my product. Has the refund been completed?")
    assert "8-digit order ID" in first.reply

    order = agent.respond("My order ID is 72630112.")
    assert "last four digits" in order.reply

    verified = agent.respond("My phone ends in 4217.")
    assert "₹21,919" in verified.reply
    assert "initiated on 15 October 2026" in verified.reply
    assert "completed on 15 October 2026" in verified.reply
    assert "ZentroMart Wallet" in verified.reply

    repeat = agent.respond("Can you tell me the refund amount and date?")
    assert "₹21,919" in repeat.reply
    assert "completed on 15 October 2026" in repeat.reply


def test_refund_check_request_resumes_with_actual_record_after_verification_retry(agent):
    agent.respond("I returned my order. Can you check my refund?")
    agent.respond("My order ID is 72630112.")
    retry = agent.respond("4127")
    assert "try one more time" in retry.reply

    verified = agent.respond("4217")
    assert "₹21,919" in verified.reply
    assert "initiated on 15 October 2026" in verified.reply
    assert "completed on 15 October 2026" in verified.reply
    assert "ZentroMart Wallet" in verified.reply
    assert "current support guidance" not in verified.reply


def test_initiated_return_refund_is_not_described_as_completed(agent):
    first = agent.respond("My return was picked up a few days ago. Has my refund been processed?")
    assert "8-digit order ID" in first.reply

    agent.respond("My order ID is 16640894.")
    verified = agent.respond("My phone ends in 2108.")
    assert "initiated on 20 July 2026" in verified.reply
    assert "not yet been marked completed" in verified.reply
    assert "1–3 business days" in verified.reply

    follow_up = agent.respond("I paid with UPI. When will I receive the money?")
    assert "initiated on 20 July 2026" in follow_up.reply
    assert "not yet been marked completed" in follow_up.reply
    assert "UPI refunds" in follow_up.reply


def test_cancelled_order_refund_status_uses_verified_refund_record(agent):
    first = agent.respond("I cancelled my order. Has my money been refunded?")
    assert "8-digit order ID" in first.reply

    agent.respond("My order ID is 54239550.")
    verified = agent.respond("0986")
    assert "₹5,278" in verified.reply
    assert "initiated on 17 October 2026" in verified.reply
    assert "not yet been marked completed" in verified.reply
    assert "UPI refunds" in verified.reply

    follow_up = agent.respond("I paid using UPI. What is the refund status?")
    assert "₹5,278" in follow_up.reply
    assert "not yet been marked completed" in follow_up.reply


@pytest.mark.parametrize(
    ("messages", "required"),
    [
        (
            ["Hi, I placed an order. Has it been processed?", "My order ID is 35061733.", "My phone number ends in 2743.", "When will it arrive?"],
            ["placed", "27 October 2026"],
        ),
        (
            ["Has my order been shipped yet?", "The order ID is 71577152.", "My phone ends in 1684.", "Can I cancel it before it ships?"],
            ["packed", "cancel before the order ships"],
        ),
        (
            ["My order has not arrived. Where is it?", "My order number is 69520867.", "My registered phone ends in 0101.", "Can you cancel it now?"],
            ["shipped", "23 October 2026", "can’t be cancelled"],
        ),
        (
            ["Is my order arriving today?", "My order ID is 46662050.", "My phone ends in 5493.", "I need to leave home soon. Can you guarantee the delivery time?"],
            ["out for delivery", "20 October 2026", "can’t guarantee"],
        ),
        (
            ["I was expecting a delivery, but I did not receive it.", "My order number is 41763040.", "My phone ends in 3590.", "Why was it not delivered?"],
            ["1 failed delivery attempt", "do not have a reason recorded"],
        ),
        (
            ["My order never reached me. What happened?", "The order ID is 90362119.", "My phone ends in 2328.", "Has it been returned to the seller?", "When will I get my money back?"],
            ["₹1,357", "initiated on 20 October 2026", "not yet been marked completed"],
        ),
        (
            ["Your system says my order was delivered, but I cannot find it.", "My order ID is 59591825.", "My phone ends in 5893."],
            ["delivered on 11 June 2026", "family, neighbours, or building security"],
        ),
        (
            ["I requested a return, but nobody has picked up my product.", "My order number is 92082916.", "My phone ends in 5142.", "When will my refund arrive?"],
            ["no pickup or refund initiation is recorded"],
        ),
        (
            ["My returned product was picked up. Where is my refund?", "My order ID is 98463156.", "My phone ends in 5850.", "Has the quality check passed?"],
            ["picked up on 17 October 2026", "No quality-check pass or refund initiation is recorded"],
        ),
    ],
)
def test_order_status_catalogue_conversations(agent, messages, required):
    replies = " ".join(agent.respond(message).reply for message in messages).casefold()
    for fact in required:
        assert fact.casefold() in replies


def test_wrong_verification_never_reveals_protected_order_details(agent):
    agent.respond("Where is my order 69520867?")
    wrong = agent.respond("My phone ends in 9999.")
    assert "didn’t match" in wrong.reply

    protected = agent.respond("Just tell me the delivery address; I know the order ID.")
    assert "can't share the delivery address" in protected.reply
    assert "last four registered phone digits" in protected.reply
    assert "160017" not in protected.reply

    verified = agent.respond("0101")
    assert "verified" in verified.reply


def test_rto_reship_request_does_not_invent_a_replacement(agent):
    agent.respond("Tracking says my order went back to the seller.")
    agent.respond("90362119")
    agent.respond("My phone ends in 2328.")
    result = agent.respond("Will you send the same order again?")
    assert "returned to origin" in result.reply
    assert "can't confirm a replacement shipment" in result.reply


def test_cancel_guidance_follow_up_gives_app_steps(agent):
    agent.respond("I ordered something by mistake. Can I cancel it?")
    agent.respond("35061733")
    agent.respond("2743")
    result = agent.respond("yes guide me")
    assert "go to My Orders" in result.reply
    assert "Cancel order" in result.reply
    assert "initiated immediately" in result.reply


def test_out_for_delivery_time_question_uses_order_status_not_unrelated_policy(agent):
    agent.respond("My order is out for delivery. What time will it arrive?")
    agent.respond("My order ID is 46662050.")
    result = agent.respond("5493")
    assert "out for delivery" in result.reply
    assert "20 October 2026" in result.reply
    assert "can't confirm an exact arrival time" in result.reply
    assert "express fee" not in result.reply

    shorthand = agent.respond("delivery time")
    assert "out for delivery" in shorthand.reply
    assert "20 October 2026" in shorthand.reply
    assert "can't confirm an exact arrival time" in shorthand.reply


def test_delivered_but_didnt_get_anything_starts_missing_delivery_flow(agent):
    first = agent.respond("Your app says delivered, but I didn't get anything.")
    assert "8-digit order ID" in first.reply

    agent.respond("59591825")
    result = agent.respond("5893")
    assert "delivered on 11 June 2026" in result.reply
    assert "family, neighbours, or building security" in result.reply


def test_duplicate_ticket_request_does_not_create_a_second_ticket(agent):
    with patch("src.agent.create_ticket", return_value={"created": True, "ticket_id": "TKT"}) as create:
        agent.respond("My issue is unresolved. Please create a ticket.")
        duplicate = agent.respond("reate another ticket for exactly the same issue.")
    assert create.call_count == 1
    assert "already been created" in duplicate.reply
    assert "duplicate" in duplicate.reply


def test_warranty_sku_is_remembered_across_turns(agent):
    agent.respond("My electronics product stopped working.")
    agent.respond("The product SKU is ELE-0012.")
    result = agent.respond("Is it covered by warranty?")
    assert "Kestrel K7 Smartphone" in result.reply
    assert "12-month warranty" in result.reply
    assert "products.csv:ELE-0012" in result.sources


def test_damage_follow_up_and_replacement_guarantee_are_concise(agent):
    agent.respond("My product arrived damaged.")
    photos = agent.respond("I have photos of the damage.")
    guarantee = agent.respond("Can you guarantee a replacement today?")
    assert "item and packaging" in photos.reply
    assert "can't guarantee a replacement today" in guarantee.reply


def test_unlisted_late_delivery_discount_escalates_without_inventing_policy(agent):
    with patch("src.agent.create_ticket", return_value={"created": True, "ticket_id": "TKT"}):
        result = agent.respond("Can I get a 75% discount on my next purchase because my order was late?")
    assert result.tool_calls == ["create_ticket()"]
    assert "not a documented entitlement" in result.reply
    assert "support ticket" in result.reply


def test_gemini_reference_package_contains_complete_safe_knowledge_not_order_dataset(agent):
    context = agent._gemini_reference_context
    assert "[REF-2]" in context
    assert "[RET-7]" in context
    assert "SKU=ELE-0012" in context
    assert "54239550" not in context
    assert "GEMINI_API_KEY" not in context


def test_gemini_reply_is_cleaned_for_customer_chat():
    cleaned = ZentroMartAgent._clean_gemini_reply(
        "Answer: Your order is shipped. It should arrive tomorrow. Please check the app. Sources: [TRK-1]"
    )
    assert cleaned == "Your order is shipped. It should arrive tomorrow. Please check the app."


def test_general_refund_answer_includes_relevant_source_in_api_result():
    response = TestClient(app).post(
        "/api/chat", json={"message": "How fast are refunds to the ZentroMart Wallet?"}
    )
    assert response.status_code == 200
    assert "REF-2" in response.json()["sources"]
    assert "Instant" in response.json()["reply"]


def test_order_id_only_lookup_reveals_no_order_details(known_order):
    order_id, order, _ = known_order
    result = lookup_order(order_id)
    assert result["verification_required"] is True
    assert not any(key in result for key in ("status", "total_inr", "customer_id", "items"))
    assert order["status"] not in result.values()


def test_nonexistent_order_does_not_disclose_details():
    result = lookup_order("00000000", "1234")
    assert result.get("verified") is False
    assert "found" not in result
    assert "status" not in result


def test_valid_registered_phone_suffix_verifies_order(known_order):
    order_id, order, customer = known_order
    result = lookup_order(order_id, customer["phone"][-4:])
    assert result["verified"] is True
    assert result["order_id"] == order_id
    assert result["status"] == order["status"]
    assert "phone" not in result


def test_valid_delivery_pincode_verifies_order(known_order):
    order_id, order, _ = known_order
    result = lookup_order(order_id, order["delivery_pincode"])
    assert result["verified"] is True
    assert result["delivery_pincode"] == order["delivery_pincode"]


def test_wrong_phone_suffix_or_pincode_does_not_verify(known_order):
    order_id, _, customer = known_order
    proof = "0000" if customer["phone"][-4:] != "0000" else "9999"
    result = lookup_order(order_id, proof)
    assert result["verified"] is False
    assert "status" not in result


def test_order_id_and_pincode_are_preserved_as_strings(known_order):
    order_id, order, _ = known_order
    result = lookup_order(order_id, order["delivery_pincode"])
    assert isinstance(result["order_id"], str)
    assert isinstance(result["delivery_pincode"], str)


def test_verified_order_contains_correct_item_and_product_join(known_order):
    order_id, _, customer = known_order
    result = lookup_order(order_id, customer["phone"][-4:])
    assert result["verified"] is True
    assert result["items"]
    assert all(item["order_id"] == order_id for item in result["items"])
    catalog_skus = {
        row["sku"]
        for row in csv.DictReader((ROOT / "data" / "products.csv").open(encoding="utf-8"))
    }
    assert all(item["sku"] in catalog_skus for item in result["items"])


def test_agent_denies_order_data_after_two_failed_proofs(agent, known_order):
    order_id, order, customer = known_order
    agent.respond(f"My order is {order_id}")
    wrong = "0000" if customer["phone"][-4:] != "0000" else "9999"
    agent.respond(wrong)
    result = agent.respond(wrong)
    assert agent.verified_order is None
    assert "share order information" in result.reply.lower()
    assert str(order.get("total_inr")) not in result.reply


def test_order_specific_request_asks_for_order_id(agent):
    result = agent.respond("Where is my order?")
    assert "IDV-1" in result.sources
    assert "order ID" in result.reply


def test_hindi_greeting_identifies_assistant_and_asks_for_order_id(agent):
    result = agent.respond("नमस्ते, मुझे अपने ऑर्डर के बारे में मदद चाहिए।")
    assert "AI assistant" in result.reply
    assert "STY-1" in result.sources
    assert "order ID" in result.reply or "ऑर्डर ID" in result.reply


def test_follow_up_uses_verified_order_context(agent, known_order):
    order_id, _, customer = known_order
    agent.respond(f"My order is {order_id}")
    verified = agent.respond(customer["phone"][-4:])
    assert "verified" in verified.reply.lower()
    follow_up = agent.respond("Where is my order?")
    assert "IDV-1" not in follow_up.sources


def test_repeating_verified_order_id_does_not_restart_verification(agent):
    agent.respond("My order is 69520867.")
    agent.respond("0101")
    repeated = agent.respond("I need help with order 69520867.")
    assert "already verified" in repeated.reply.lower()
    assert not agent.awaiting_verification
    assert agent.verified_order is not None


def test_shipping_question_resumes_after_pincode_verification(agent):
    agent.respond("Has my order been shipped yet?")
    agent.respond("71577152")
    result = agent.respond("560103")
    assert "packed" in result.reply.lower()
    assert "27 October 2026" in result.reply
    assert {"IDV-2", "TRK-1", "DEL-1"}.issubset(result.sources)


def test_arriving_today_question_resumes_after_phone_verification(agent):
    agent.respond("Is my order arriving today?")
    agent.respond("46662050")
    result = agent.respond("5493")
    assert "out for delivery" in result.reply.lower()
    assert "20 October 2026" in result.reply
    assert {"IDV-2", "TRK-1", "DEL-1"}.issubset(result.sources)


def test_order_id_and_pincode_in_one_turn_verify_and_show_status(agent):
    result = agent.respond(
        "Where is my order 69520867? My delivery pincode is 160017."
    )
    assert agent.verified_order is not None
    assert agent.verified_order["order_id"] == "69520867"
    assert "shipped" in result.reply.lower()
    assert "IDV-2" in result.sources
    assert result.tool_calls == [
        "lookup_order(order_id=69520867, verification_value=provided)"
    ]


def test_reset_clears_verified_and_pending_conversation_state(agent, known_order):
    order_id, _, customer = known_order
    agent.respond(f"My order is {order_id}")
    agent.respond(customer["phone"][-4:])
    agent.reset()
    assert agent.verified_order is None
    assert agent.order_id is None
    assert not agent.history
    assert agent.awaiting_verification is False


def test_missing_verification_requests_only_one_proof(agent, known_order):
    agent.respond(f"My order is {known_order[0]}")
    result = agent.respond("Can you just tell me its status?")
    assert "last four digits" in result.reply
    assert "pincode" in result.reply
    assert "status" not in result.reply.lower()


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ("Can I speak to a person?", "human agent"),
        ("I never placed this order.", "did not authorize"),
        ("I will take this to consumer court.", "legal action"),
        ("My phone battery is swelling.", "safety risk"),
    ],
)
def test_explicit_escalation_triggers_create_ticket(agent, message, reason):
    with patch("src.agent.create_ticket", return_value={"created": True, "ticket_id": "TKT"}) as create:
        result = agent.respond(message)
    assert result.tool_calls == ["create_ticket()"]
    assert reason in create.call_args.kwargs["reason"]
    assert "ESC-1" in result.sources


def test_ticket_failure_never_claims_ticket_was_created(agent):
    with patch("src.agent.create_ticket", return_value={"created": False, "error": "disk full"}):
        result = agent.respond("Can I speak to a human?")
    assert "created a support ticket" not in result.reply.lower()
    assert "human colleague" in result.reply.lower()
    assert not result.tool_calls


def test_ticket_summary_redacts_customer_verification_code(agent):
    with patch("src.agent.create_ticket", return_value={"created": True, "ticket_id": "TKT"}) as create:
        agent._ticket("Please escalate; my code is 1234", [], "customer request")
    summary = create.call_args.kwargs["summary"]
    assert "1234" not in summary
    assert "[verification value redacted]" in summary


def test_ticket_summary_preserves_product_sku_while_redacting_verification_value(agent):
    message = "SKU ELE-0023 is damaged; verification suffix is 1234."
    with patch("src.agent.create_ticket", return_value={"created": True, "ticket_id": "TKT"}) as create:
        agent._ticket(message, [], "high-value damaged item claim")
    summary = create.call_args.kwargs["summary"]
    assert "ELE-0023" in summary
    assert "1234" not in summary
    assert "[verification value redacted]" in summary


@pytest.mark.parametrize(
    ("start", "days", "expected"),
    [
        (date(2026, 10, 16), 1, date(2026, 10, 17)),
        (date(2026, 10, 17), 1, date(2026, 10, 20)),
        (date(2026, 10, 18), 1, date(2026, 10, 20)),
        (date(2026, 10, 16), 2, date(2026, 10, 20)),
        (date(2026, 10, 16), 3, date(2026, 10, 21)),
    ],
)
def test_business_days_skip_sundays_and_holidays(agent, start, days, expected):
    assert agent._business_days_after(start, days, agent.holidays) == expected


def test_return_windows_are_calendar_days_not_business_days(agent):
    ret = next(chunk for chunk in agent.kb.chunks if chunk.section_id == "RET-2")
    assert "calendar days" in ret.text
    assert "7 days" in ret.text
    assert "15 days" in ret.text


def test_refund_deadline_skips_sunday_and_holiday(agent):
    agent.verified_order = {
        "return_refund_initiated_on": "2026-10-16",
        "return_refund_method": "upi",
        "return_refund_completed_on": None,
    }
    assert agent._refund_overdue() is False


@pytest.mark.parametrize(
    ("message", "section"),
    [
        ("How can I cancel a shipped order?", "CAN-2"),
        ("What happens if courier misses me three times?", "DEL-3"),
        ("How are wallet refunds processed?", "REF-2"),
    ],
)
def test_policy_retrieval_for_cancellation_delivery_and_refund(agent, message, section):
    assert section in agent.respond(message).sources


def test_web_rejects_malformed_chat_payload():
    response = TestClient(app).post("/api/chat", json={"wrong": "shape"})
    assert response.status_code == 422


def test_web_rejects_non_string_chat_message():
    response = TestClient(app).post("/api/chat", json={"message": 123})
    assert response.status_code == 422


def test_web_empty_message_is_safe():
    response = TestClient(app).post("/api/chat", json={"message": ""})
    assert response.status_code == 200
    assert "didn't catch" in response.json()["reply"]


def test_web_issues_http_only_session_cookie():
    response = TestClient(app).post("/api/chat", json={"message": "Hello"})
    cookie = response.cookies.get("voiceops_session")
    assert cookie
    assert "httponly" in response.headers["set-cookie"].lower()
    assert "samesite=lax" in response.headers["set-cookie"].lower()


def test_web_session_reset_is_scoped_to_one_client(known_order):
    first, second = TestClient(app), TestClient(app)
    first.post("/api/chat", json={"message": f"My order is {known_order[0]}"})
    first.post("/api/chat", json={"message": known_order[2]["phone"][-4:]})
    first.post("/api/reset")
    result = first.post("/api/chat", json={"message": "Where is my order?"})
    assert "IDV-1" in result.json()["sources"]
    assert not second.cookies.get("voiceops_session")


def test_voice_endpoint_surfaces_adapter_failure_as_503(project_temp_dir, monkeypatch):
    import io
    import wave

    monkeypatch.setattr("src.main.VOICE_OUTPUTS", project_temp_dir)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\0\0" * 8000)
    with patch("src.main.voice_turn", side_effect=RuntimeError("Gnani unavailable")):
        response = TestClient(app).post(
            "/api/voice", files={"audio": ("sample.wav", buffer.getvalue(), "audio/wav")}
        )
    assert response.status_code == 503
    assert response.json()["detail"] == "Gnani unavailable"
    assert list(project_temp_dir.iterdir()) == []


def test_voice_endpoint_rejects_wav_longer_than_sixty_seconds(project_temp_dir, monkeypatch):
    import io
    import wave

    monkeypatch.setattr("src.main.VOICE_OUTPUTS", project_temp_dir)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\0\0" * 8000 * 61)
    response = TestClient(app).post(
        "/api/voice",
        files={"audio": ("long.wav", buffer.getvalue(), "audio/wav")},
    )
    assert response.status_code == 422
    assert "60 seconds" in response.json()["detail"]
    assert list(project_temp_dir.iterdir()) == []


def test_voice_ui_exposes_required_language_codes_and_upload_limit():
    page = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    for language in ("bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN", "pa-IN", "ta-IN", "te-IN", "hi-en"):
        assert f'value="{language}"' in page
    assert "reply_language" in script
    assert 'id="tts-voice"' in page
    assert "voice=" in script
    assert "60000" in script
    assert "60 seconds" in script
    assert "voice-message.webm" in script
    assert '".webm"' in (ROOT / "src" / "main.py").read_text(encoding="utf-8")


def test_voice_turn_synthesizes_retry_prompt_for_empty_transcript(monkeypatch, project_temp_dir):
    gnani = types.ModuleType("gnani")
    gnani.__path__ = []
    stt_module = types.ModuleType("gnani.stt")
    tts_module = types.ModuleType("gnani.tts")
    stt_client, tts_client = Mock(), Mock()
    stt_client.transcribe.return_value = {"transcript": ""}
    stt_module.GnaniSTTClient = Mock(return_value=stt_client)
    tts_module.GnaniTTSClient = Mock(return_value=tts_client)
    monkeypatch.setitem(sys.modules, "gnani", gnani)
    monkeypatch.setitem(sys.modules, "gnani.stt", stt_module)
    monkeypatch.setitem(sys.modules, "gnani.tts", tts_module)
    agent = Mock()
    agent.respond.return_value = TurnResult("Sorry, I didn't catch that. Could you say it again?")

    heard, result = voice_turn("input.webm", agent, str(project_temp_dir / "reply.wav"))

    assert heard == ""
    assert "didn't catch" in result.reply
    agent.respond.assert_called_once_with("")
    tts_client.synthesize.assert_called_once()
    assert tts_client.synthesize.call_args.kwargs["model"] == "timbre-v2.5"


def test_voice_turn_surfaces_stt_failure(monkeypatch):
    gnani = types.ModuleType("gnani")
    gnani.__path__ = []
    stt_module = types.ModuleType("gnani.stt")
    tts_module = types.ModuleType("gnani.tts")
    stt_client = Mock()
    stt_client.transcribe.side_effect = RuntimeError("STT unavailable")
    stt_module.GnaniSTTClient = Mock(return_value=stt_client)
    tts_module.GnaniTTSClient = Mock()
    monkeypatch.setitem(sys.modules, "gnani", gnani)
    monkeypatch.setitem(sys.modules, "gnani.stt", stt_module)
    monkeypatch.setitem(sys.modules, "gnani.tts", tts_module)
    with pytest.raises(RuntimeError, match="STT unavailable"):
        voice_turn("input.webm", Mock(), "reply.wav")


def test_voice_turn_surfaces_tts_failure(monkeypatch):
    gnani = types.ModuleType("gnani")
    gnani.__path__ = []
    stt_module = types.ModuleType("gnani.stt")
    tts_module = types.ModuleType("gnani.tts")
    stt_client, tts_client = Mock(), Mock()
    stt_client.transcribe.return_value = {"transcript": "Hello"}
    tts_client.synthesize.side_effect = RuntimeError("TTS unavailable")
    stt_module.GnaniSTTClient = Mock(return_value=stt_client)
    tts_module.GnaniTTSClient = Mock(return_value=tts_client)
    monkeypatch.setitem(sys.modules, "gnani", gnani)
    monkeypatch.setitem(sys.modules, "gnani.stt", stt_module)
    monkeypatch.setitem(sys.modules, "gnani.tts", tts_module)
    with pytest.raises(RuntimeError, match="TTS unavailable"):
        voice_turn("input.webm", Mock(), "reply.wav")


def test_ticket_tool_creates_csv_artifact_and_rejects_invalid_summary(
    project_temp_dir, monkeypatch
):
    from tools import tools as mock_tools

    ticket_file = project_temp_dir / "tickets.csv"
    monkeypatch.setattr(mock_tools, "_TICKETS", str(ticket_file))
    failure = mock_tools.create_ticket(summary="too short")
    assert failure["created"] is False
    assert not ticket_file.exists()

    success = mock_tools.create_ticket(
        summary="Customer reports a delayed refund and needs a human review.",
        order_id="12345678",
        reason="refund_delay",
    )
    assert success["created"] is True
    with ticket_file.open(encoding="utf-8", newline="") as handle:
        ticket = next(csv.DictReader(handle))
    assert ticket["ticket_id"] == success["ticket_id"]
    assert ticket["order_id"] == "12345678"


@pytest.mark.parametrize(
    ("message", "required"),
    [
        ("What about electronics delivered before 1 August 2026?", "10 calendar days"),
        ("I ordered clothing on 10 October 2026. Do I get an extended return window?", "30-day"),
        ("Can I return groceries because I changed my mind?", "can’t be returned"),
        ("Can I return electronics with physical damage?", "physical damage"),
        ("Is there a reverse-pickup fee for a change-of-mind clothing return?", "₹49"),
        ("How long does a net banking refund take?", "3–5 business days"),
        ("Can you promise that my refund will arrive today?", "can't guarantee"),
        ("Can I cancel my grocery delivery 30 minutes before the slot?", "cutoff has passed"),
        ("Do you count Sunday as a business day?", "Sundays are excluded"),
        ("Do you count holidays when calculating refund business days?", "holiday calendar"),
    ],
)
def test_attached_policy_scenarios_return_specific_grounded_answers(agent, message, required):
    result = agent.respond(message)
    assert required.casefold() in result.reply.casefold()
    assert result.sources


@pytest.mark.parametrize(
    ("message", "field", "expected"),
    [
        ("Is delivery available at pincode 560034?", "serviceable", "available"),
        ("Is cash on delivery available at 560034?", "cod_available", "available"),
        ("Is express delivery available at 560034?", "express_available", "available"),
        ("Can you deliver to pincode 000000?", None, "can’t confirm"),
    ],
)
def test_attached_pincode_cases_use_csv_data(agent, message, field, expected):
    result = agent.respond(message)
    assert expected in result.reply.casefold()
    assert "DEL-5" in result.sources
    if field:
        assert agent.pincodes["560034"][field] == "True"


@pytest.mark.parametrize(
    ("message", "status"),
    [
        ("My order is 89782837. My phone ends in 4533.", "delivered"),
        ("My order is 54239550. My pincode is 641004.", "cancelled"),
    ],
)
def test_attached_order_examples_show_actual_status_after_verification(agent, message, status):
    result = agent.respond(message)
    assert agent.verified_order["status"] == status
    assert status in result.reply.casefold()
    assert result.tool_calls == ["lookup_order(order_id=" + agent.order_id + ", verification_value=provided)"]


def test_attached_order_follow_up_resumes_original_delivery_question(agent):
    first = agent.respond("My order still hasn't arrived.")
    agent.respond("My order number is 69520867.")
    verified = agent.respond("My phone ends in 0101.")
    assert "IDV-1" in first.sources
    assert agent.verified_order["status"] == "shipped"
    assert "shipped" in verified.reply
    cancelled = agent.respond("Can I cancel it?")
    assert "CAN-2" in cancelled.sources
    assert "can’t be cancelled" in cancelled.reply
    refund = agent.respond("How long will the refund take?")
    assert "REF-2" in refund.sources


def test_spoken_verification_correction_keeps_the_active_order(agent):
    agent.respond("My order number is six nine five two zero eight six seven.")
    result = agent.respond(
        "My phone ends nine nine nine nine sorry, I made a mistake, it is zero one zero one."
    )
    assert agent.order_id == "69520867"
    assert agent.verified_order and agent.verified_order["status"] == "shipped"
    assert "identity is verified" in result.reply.casefold()
    assert result.tool_calls == ["lookup_order(order_id=69520867, verification_value=provided)"]


def test_misheard_nine_digit_order_then_spoken_correction_stays_on_track(agent):
    agent.respond("Can you check my order number six nine five two zero eight seven six seven?")
    agent.respond("six nine five two zero eight six seven")
    result = agent.respond(
        "My phone number ends in nine nine nine and sorry it ends with zero one zero one"
    )
    assert agent.verified_order and agent.verified_order["status"] == "shipped"
    assert "current status is shipped" in result.reply.casefold()
    assert "current support guidance says" not in result.reply.casefold()


def test_audio_checklist_shipped_refusal_follow_up_uses_refusal_policy(agent):
    agent.respond("Check order 69520867")
    agent.respond("0101")
    result = agent.respond("What if I refuse the parcel when it arrives?")
    assert "refuse" in result.reply.casefold()
    assert "CAN-2" in result.sources


def test_audio_checklist_missing_delivery_is_not_treated_as_a_greeting(agent):
    result = agent.respond("My order says delivered, but I haven't received anything. Can you help?")
    assert "8-digit order id" in result.reply.casefold()
    assert agent.awaiting_order_id
    agent.respond("59591825")
    resolved = agent.respond("5893")
    assert "recorded as delivered" in resolved.reply.casefold()
    assert "DWM-1" in resolved.sources


def test_audio_checklist_refund_record_and_generic_return_fee_followups(agent):
    agent.respond("I cancelled my order. Has my refund been credited yet?")
    agent.respond("54239550")
    refund = agent.respond("0986")
    assert "refund" in refund.reply.casefold()
    assert "REF-" in " ".join(refund.sources)

    second = ZentroMartAgent(agent.data_dir)
    second.respond("Can I cancel my order?")
    second.respond("Actually, I want to return it after delivery.")
    fee = second.respond("Will I have to pay a pickup fee?")
    assert "₹49" in fee.reply
    assert "RET-6" in fee.sources


def test_combined_spoken_order_verification_answers_arrival_and_return_refund(agent):
    result = agent.respond(
        "Check my order number six nine five two zero eight six seven and my phone number ends "
        "with zero one zero one. When will it arrive and how long will my refund take if I return it?"
    )
    assert agent.verified_order and agent.verified_order["status"] == "shipped"
    assert "promised delivery date" in result.reply.casefold()
    assert "within 2 business days after pickup and quality check" in result.reply.casefold()
    assert {"TRK-1", "RET-7", "REF-2"}.issubset(result.sources)


def test_attached_return_eligibility_follow_up_retains_missing_information(agent):
    first = agent.respond("I want to return my product.")
    second = agent.respond("Can you tell me if it is eligible?")
    assert "type of item" in first.reply
    assert "type of item" in second.reply
    assert agent.pending_return_question


def test_attached_warranty_topic_change_does_not_use_sku_as_verification(agent):
    agent.respond("I need help with order 69520867.")
    result = agent.respond("Actually, I want to ask about the warranty on ELE-0012.")
    assert "12-month warranty" in result.reply
    assert "products.csv:ELE-0012" in result.sources
    assert agent.verified_order is None
    assert not agent.awaiting_verification


def test_attached_repeat_contact_and_overdue_refund_escalate_after_verification(agent):
    order_id = "39141531"
    order = _ORDERS[order_id]
    customer = _CUSTOMERS[order["customer_id"]]
    agent.respond(f"My order is {order_id}")
    agent.respond(customer["phone"][-4:])
    with patch("src.agent.create_ticket", return_value={"created": True, "ticket_id": "TKT"}) as create:
        result = agent.respond("This is my third contact about the same issue this week.")
    assert result.tool_calls == ["create_ticket()"]
    assert "third or later contact" in create.call_args.kwargs["reason"]


@pytest.mark.parametrize(
    ("message", "expected_reason"),
    [
        ("Someone placed an order using my account without permission.", "did not authorize"),
        ("My device is burning and smells like smoke.", "safety risk"),
        ("My product arrived damaged and cost ₹15,000.", "high-value damaged item"),
        ("I want a refund exception that is not allowed by your policy.", "human review"),
        ("Please create a ticket for my unresolved issue.", "support ticket"),
    ],
)
def test_attached_escalation_triggers_create_ticket(agent, message, expected_reason):
    with patch("src.agent.create_ticket", return_value={"created": True, "ticket_id": "TKT"}) as create:
        result = agent.respond(message)
    assert result.tool_calls == ["create_ticket()"]
    assert expected_reason in create.call_args.kwargs["reason"]


def test_attached_human_request_after_ticket_does_not_create_duplicate(agent):
    with patch("src.agent.create_ticket", return_value={"created": True, "ticket_id": "TKT"}) as create:
        first = agent.respond("Give me ₹500 compensation immediately.")
        second = agent.respond("I want a human agent.")
    assert first.tool_calls == ["create_ticket()"]
    assert second.tool_calls == []
    assert create.call_count == 1
    assert "already been created" in second.reply


def test_attached_vague_complaint_asks_for_details(agent):
    result = agent.respond("I have a complaint.")
    assert "what happened" in result.reply.casefold()
    assert not result.tool_calls


def test_attached_pincode_lookup_uses_explicit_attribute(agent):
    assert agent.pincodes["560034"]["serviceable"] == "True"
    assert agent.pincodes["560034"]["cod_available"] == "True"
    assert agent.pincodes["560034"]["express_available"] == "True"


def test_attached_refund_date_calculation_skips_sundays_and_holidays(agent):
    result = agent.respond(
        "My refund to UPI was started on 17 October 2026. By which date should I have it?"
    )
    assert "22 October 2026" in result.reply
    assert {"REF-2", "KB-00.3"}.issubset(result.sources)


def test_attached_refund_overdue_check_requires_verified_order(agent):
    first = agent.respond("My refund is late. Is it overdue?")
    assert "8-digit order ID" in first.reply
    assert agent.awaiting_order_id

    order_id = "75200505"
    order = _ORDERS[order_id]
    customer = _CUSTOMERS[order["customer_id"]]
    agent.respond(f"My order is {order_id}")
    with patch("src.agent.create_ticket", return_value={"created": True, "ticket_id": "TKT"}) as create:
        result = agent.respond(customer["phone"][-4:])
    assert result.tool_calls == [
        "lookup_order(order_id=75200505, verification_value=provided)",
        "create_ticket()",
    ]
    assert "refund is beyond its maximum timeline" in create.call_args.kwargs["reason"]


def test_attached_order_privacy_never_reveals_phone_or_address(agent, known_order):
    order_id, _, customer = known_order
    agent.respond(f"My order is {order_id}")
    agent.respond(customer["phone"][-4:])
    result = agent.respond("Please tell me the full phone number and street address.")
    assert "cannot provide" in result.reply
    assert "IDV-4" in result.sources
    assert customer["phone"] not in result.reply
    assert customer["name"].casefold() not in result.reply.casefold()


def test_safety_escalation_gives_precaution_and_only_one_ticket(agent):
    with patch("src.agent.create_ticket", return_value={"created": True, "ticket_id": "TKT"}) as create:
        first = agent.respond("My device is burning and smells like smoke.")
        second = agent.respond("I want a human agent.")
    assert first.tool_calls == ["create_ticket()"]
    assert "stop using" in first.reply.casefold()
    assert second.tool_calls == []
    assert create.call_count == 1


def test_run_command_has_audio_folder_and_output_arguments():
    import subprocess

    result = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "audio_folder" in result.stdout
    assert "--output" in result.stdout


def test_full_practice_evaluation_writes_all_rows_and_fields(project_temp_dir):
    summary = evaluate_practice(
        ROOT / "data",
        PRACTICE,
        project_temp_dir,
    )
    assert summary["questions"]["total"] == 52
    assert summary["conversation_turns"]["total"] == 96
    assert summary["conversation_turns"]["conversations"] == 23
    assert summary["prompt_only_baseline"].startswith("not measured")
    assert summary["conversation_tool_calls"] == {"expected": 54, "matched": 54}
    expected_fields = {
        "case_id", "expected_decision", "actual_decision", "gold_evidence",
        "retrieved_section_ids", "tool_calls_expected", "tool_calls_actual",
        "passed", "failure_reason",
    }
    for name, count in (("questions.csv", 52), ("conversations.csv", 96)):
        with (project_temp_dir / name).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        assert len(rows) == count
        assert expected_fields.issubset(reader.fieldnames)
    saved_summary = json.loads((project_temp_dir / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["questions"] == summary["questions"]


def test_dataset_validator_command_passes():
    import subprocess

    result = subprocess.run(
        [sys.executable, str(DATASET / "validate_dataset.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PASS" in result.stdout


def test_submission_files_and_env_template_exist():
    assert (ROOT / "report.pdf").is_file()
    assert (ROOT / "README.md").is_file()
    assert (ROOT / ".env.example").is_file()
    assert (ROOT / "requirements.txt").is_file()
    assert (ROOT / "run.py").is_file()
    assert (ROOT / "src").is_dir()

    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "GNANI_API_KEY=" in example
    assert "GEMINI_API_KEY=" in example
    assert "replace_with_your_gemini_key" in example
    assert "replace_with_your_gnani_key" in example
