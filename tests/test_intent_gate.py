"""The intent gate and the fixed replies.

The gate is deterministic on purpose, so these are exhaustive rather than
representative. Its two failure modes are both worth a demo: REFUSING a
consultancy message (catastrophic) and ANSWERING a pizza question (worse, and
a judge will try it first).
"""

from __future__ import annotations

import pytest

from edushield import core


def gate(text: str) -> dict:
    return core.intent_gate(core.sanitise_and_redact({"text": text}))


def intent_of(text: str) -> str:
    return gate(text)["_intent"]


# --- fraud_check routing ---------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Is this university real or fake?",
        "My agent guarantees my visa and wants full payment upfront",
        "Is this consultancy legit?",
        "Can I trust https://some-university-admissions.com",
        "what are the red flags here",
        "This offer letter says 100% admission guarantee, verify please",
    ],
)
def test_trust_questions_route_to_the_investigation(text):
    assert intent_of(text) == "fraud_check"


def test_a_bare_url_with_no_question_is_still_a_submission():
    assert intent_of("https://some-university-admissions.com") == "fraud_check"


def test_study_abroad_prose_with_no_question_is_treated_as_a_submission():
    assert intent_of("my consultancy says they can get me admission and a visa") == "fraud_check"


def test_unclear_question_shape_is_investigated_not_guessed():
    assert intent_of("I have an offer from an agent, what now?") == "fraud_check"


def test_asking_for_alternatives_is_a_profile_question():
    """Alternatives are served by the profile branch, from the same verified
    snapshot, so this must not fall through to the fraud branch and produce an
    empty section."""
    assert intent_of("Give me 3 other similar universities in the UK") == "profile"


# --- profile routing -------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "what is University of Bristol known for",
        "how long is an MSc at Imperial",
        "what are the entry requirements for Manchester",
        "tell me about Cardiff University",
        "what scholarships does Bristol offer",
        "how much does it cost to study at Manchester",
        "what credit system does Edinburgh use",
    ],
)
def test_facts_about_a_college_route_to_the_profile_branch(text):
    assert intent_of(text) == "profile"


def test_a_question_that_is_both_routes_to_profile_and_says_so():
    ctx = gate("is University of Bristol legit and what scholarships do they have")
    assert ctx["_intent"] == "profile"
    assert "legitimacy" in ctx["_intent_reason"]


# --- the refusals ----------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "what is the weather in Delhi tomorrow",
        "tell me a joke about students",
        "write a poem about exams",
        "who is the president of France",
    ],
)
def test_out_of_scope_topics_are_refused(text):
    assert intent_of(text) == "other"


def test_a_mixed_message_refuses_rather_than_guessing():
    """The safe failure mode. Answering half of a mixed message means the model
    picked a side of an ambiguity nobody asked it to resolve."""
    ctx = gate("what is the weather in Delhi, and is University of Manchester legit")
    assert ctx["_intent"] == "other"
    assert "mixed" in ctx["_intent_reason"]


def test_a_greeting_never_triggers_an_investigation():
    for text in ("hi", "Hello!", "thanks", "good morning", "bye"):
        assert intent_of(text) == "other"


def test_an_empty_message_never_triggers_an_investigation():
    assert intent_of("") == "other"


# --- fixed replies ---------------------------------------------------------

def test_a_refusal_carries_no_evidence_and_says_no_source_was_queried():
    out = core.fixed_reply(gate("what is the weather in Delhi"))
    assert out["refused"] is True
    assert "no external source was queried" in out["output"].lower()
    assert out["intent_handled"] == "other"


def test_a_greeting_reply_is_fixed_and_makes_no_investigation_claim():
    out = core.fixed_reply(gate("hello"))
    assert out["refused"] is False
    assert "Varosha" in out["output"]


def test_the_other_envelope_carries_no_risk_assessment():
    """A refusal must not render as a clean bill of health. A student who asked
    about a pizza must not be shown "risk: Low"."""
    from edushield import pipeline

    env = pipeline.investigate({"text": "what is the weather in Delhi"}, entry="test")
    assert env["intent"] == "other"
    assert env.get("risk") is None
    assert env.get("evidence") is None
    assert env["trace"][0]["agent"] == "Intent gate"
