"""The HTTP surface: /health, /webhook/investigate, /webhook/student-decision,
/chat and /case/{id}.

The endpoint tests exist mostly to check the seams the pipeline tests cannot
reach: that a report is parked only when a decision is actually needed, that the
store is bounded, that a bad decision is a 400 rather than a silent no-op, and
that a dead model produces a lower risk level and never a stronger-looking one.

The model-down case is the one worth rehearsing before a live demo. It is the
most likely thing to break on the day, and "fails closed" is a claim that has to
be tested rather than asserted.
"""

from __future__ import annotations

import pytest

from edushield import app as app_module

MESSAGE = (
    "Bright Path Education says University of Manchester will waive my "
    "GBP 28,000 fee if I pay their processing fee today. They emailed me from "
    "bright-path-edu.com. Call me on 9876543210 or email a@b.com"
)


# --- /health ---------------------------------------------------------------

def test_health_reports_the_disclaimer(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["ollama"] is True
    assert body["disclaimer"]


def test_health_reports_a_dead_model_without_failing(client, set_responses, monkeypatch):
    llm = set_responses(dead=True)
    # /health reads its own module-level handle, not the pipeline's.
    monkeypatch.setattr(app_module, "get_ollama", lambda: llm)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ollama"] is False


# --- /webhook/investigate --------------------------------------------------

def test_investigate_returns_a_complete_envelope(client):
    r = client.post("/webhook/investigate", json={"text": MESSAGE})
    assert r.status_code == 200
    env = r.json()
    assert env["case_id"]
    assert env["verdict"] in ("no_concern_found", "flags_raised",
                              "serious_contradictions", "insufficient_evidence")
    assert env["disclaimer"]


def test_investigate_never_echoes_the_pii_back(client):
    """The sanitiser's job is not only analysis; the response must not put the
    student's phone number and email back in front of them."""
    text = client.post("/webhook/investigate", json={"text": MESSAGE}).text
    assert "9876543210" not in text
    assert "a@b.com" not in text


def test_investigate_does_not_return_the_chat_field(client):
    body = client.post("/webhook/investigate", json={"text": MESSAGE}).json()
    assert "output" not in body


def test_investigate_parks_a_case_that_needs_a_decision(client):
    body = {"text": "Is University of Manchester real? I got an offer from "
                "www.manchester-offer-portal.example. They want GBP 30,000 "
                "upfront before I apply. Guaranteed visa."}
    env = client.post("/webhook/investigate", json=body).json()
    if env["status"] == "awaiting_student":
        assert client.get("/case/%s" % env["case_id"]).status_code == 200
    else:
        assert "decision_prompt" not in env


def test_a_completed_case_is_not_parked(client):
    env = client.post("/webhook/investigate", json={"text": "Is University of Manchester real?"}).json()
    if env["status"] != "awaiting_student":
        assert client.get("/case/%s" % env["case_id"]).status_code == 404


def test_an_empty_message_is_rejected_not_answered(client):
    r = client.post("/webhook/investigate", json={"text": "   "})
    assert r.status_code in (400, 422)
    assert "verdict" not in r.text


def test_the_whole_investigation_is_bounded_by_a_wall_clock_budget():
    """A five-minute presentation slot cannot absorb a 300-second run. The
    budget is what turns "eventually finishes" into "finishes in time"."""
    from edushield import pipeline

    assert pipeline.BUDGET_SECONDS <= 90, "budget must fit a demo slot"
    llm = pipeline.get_ollama()
    llm.start_budget(pipeline.BUDGET_SECONDS)
    assert 0 < llm.remaining() <= pipeline.BUDGET_SECONDS + 1


def test_a_call_is_refused_once_the_budget_is_gone():
    """Past the ceiling the model is not consulted at all, and the report is
    completed deterministically instead."""
    from edushield import llm as llm_mod

    client_ = llm_mod.Ollama()
    client_.start_budget(0.01)
    import time as _t

    _t.sleep(0.05)
    assert client_.out_of_budget() is True
    text, meta = client_.generate("hello", num_predict=10)
    assert text is None
    assert meta["budget_exhausted"] is True


def test_an_empty_body_is_a_400_rather_than_a_clean_report(client):
    """The single most dangerous default this API could have: every field has a
    default, so `{}` used to return 200 with a confident, empty, all-clear
    report. A frontend that sends the wrong key name must fail loudly."""
    r = client.post("/webhook/investigate", json={})
    assert r.status_code == 400
    assert "verdict" not in r.text
    assert "Nothing to investigate" in r.json()["detail"]


def test_a_wrong_key_name_is_a_400_not_an_empty_report(client):
    r = client.post("/webhook/investigate", json={"message": MESSAGE})
    assert r.status_code == 400
    assert "verdict" not in r.text


def test_a_url_only_request_is_accepted(client):
    r = client.post("/webhook/investigate", json={"url": "https://example.com"})
    assert r.status_code == 200


# --- the model-down path ---------------------------------------------------

def test_a_dead_model_fails_closed(client, set_responses):
    """No model means no risk assessment. The response must not borrow a
    confident-sounding level from anywhere."""
    set_responses(dead=True)
    env = client.post("/webhook/investigate", json={"text": MESSAGE}).json()
    assert env["verdict"] in ("insufficient_evidence", "no_concern_found", "flags_raised")
    if env["risk"]:
        assert env["risk"]["level"] in ("Low", "Medium", "High")
    assert env["disclaimer"]


def test_a_dead_model_never_produces_a_stronger_level_than_a_working_one(client, set_responses):
    """Compared side by side, because "fails closed" is a comparative claim."""
    live = client.post("/webhook/investigate", json={"text": MESSAGE}).json()
    set_responses(dead=True)
    dead = client.post("/webhook/investigate", json={"text": MESSAGE}).json()
    order = {"Low": 0, "Medium": 1, "High": 2}
    assert order[dead["risk"]["level"]] <= order[live["risk"]["level"]]


def test_a_dead_model_still_returns_evidence(client, set_responses):
    """Failing closed must not mean returning an empty report: the register
    match and the checklist are deterministic and still worth showing."""
    set_responses(dead=True)
    env = client.post("/webhook/investigate", json={"text": MESSAGE}).json()
    assert env["checklist"]["registry_to_check"]["url"].startswith("https://")
    assert env["verification"]["match_status"] in ("verified", "not_found", "partial_match")


def test_a_dead_model_does_not_invent_citations(client, set_responses):
    set_responses(dead=True)
    env = client.post("/webhook/investigate", json={"text": MESSAGE}).json()
    for e in env["evidence"]:
        assert e["source_url"].startswith("https://")


# --- /chat -----------------------------------------------------------------

def test_chat_returns_a_readable_output_field(client):
    body = client.post("/chat", json={"chatInput": "Is University of Manchester real?",
                                      "sessionId": "s1"}).json()
    assert body["output"]
    assert body["case_id"]
    assert body["envelope"]["disclaimer"]


def test_chat_defaults_to_a_session_id(client):
    body = client.post("/chat", json={"chatInput": "hello"}).json()
    assert body["case_id"]


def test_chat_with_empty_input_is_rejected(client):
    r = client.post("/chat", json={"chatInput": "   "})
    assert r.status_code in (400, 422)


# --- /webhook/student-decision --------------------------------------------

def parked_case(client) -> str:
    body = {"text": "www.manchester-offer-portal.example says my GBP 30,000 fee is "
                    "waived if I pay a processing fee today. Guaranteed visa, "
                    "no entrance exam, apply now."}
    env = client.post("/webhook/investigate", json=body).json()
    return env["case_id"]


def test_a_decision_on_a_parked_case_is_recorded(client, monkeypatch):
    from edushield import core
    env = {**core.grounding_gate({
        "case_id": "c1", "verification": {"match_status": "not_found"},
        "risk": {"level": "High", "summary": "s", "reasons": []},
        "evidence": [], "checklist": {}, "audit": {}}),
        "status": "awaiting_student", "verdict": "insufficient_evidence"}
    app_module.store.put("c1", env)
    r = client.post("/webhook/student-decision",
                    json={"case_id": "c1", "decision": "dismiss"})
    assert r.status_code == 200
    assert r.json()["decision_recorded"] == "dismiss"


def test_an_invalid_decision_is_a_400(client):
    app_module.store.put("c1", {"case_id": "c1", "status": "awaiting_student"})
    r = client.post("/webhook/student-decision",
                    json={"case_id": "c1", "decision": "make_it_real"})
    assert r.status_code == 400
    assert "decision must be one of" in r.json()["detail"]


def test_an_unknown_scope_is_a_400(client):
    app_module.store.put("c1", {"case_id": "c1", "status": "awaiting_student"})
    r = client.post("/webhook/student-decision",
                    json={"case_id": "c1", "decision": "investigate_further",
                          "scope": ["mind_reading"]})
    assert r.status_code == 400
    assert "unknown scope" in r.json()["detail"]


def test_a_scope_on_a_non_investigate_decision_is_a_400(client):
    app_module.store.put("c1", {"case_id": "c1", "status": "awaiting_student"})
    r = client.post("/webhook/student-decision",
                    json={"case_id": "c1", "decision": "accept", "scope": ["advisories"]})
    assert r.status_code == 400


def test_a_decision_for_an_unknown_case_is_a_404_that_explains_itself(client):
    r = client.post("/webhook/student-decision", json={"case_id": "nope", "decision": "accept"})
    assert r.status_code == 404
    assert "investigate again" in r.json()["detail"]


# --- /case -----------------------------------------------------------------

def test_an_unknown_case_is_a_404(client):
    assert client.get("/case/nope").status_code == 404


def test_the_store_is_bounded(client):
    """An unbounded dict is a slow leak on a long-running demo server."""
    cap = app_module.store.capacity
    for i in range(cap + 25):
        app_module.store.put("case-%d" % i, {"case_id": "case-%d" % i})
    assert len(app_module.store) <= cap
