"""The grounding gate, the verdict derivation and the trace.

The grounding gate is the last thing that runs before a student sees anything.
It polices the PIPELINE, not just the model, because a bug in deterministic code
is just as capable of showing a student a confident lie as a hallucinating model
is.

It must also never raise. A gate that crashes on the last node turns a small
defect into a blank screen, and a blank screen is the worst thing this tool can
do to someone who trusted it. That is tested here directly.
"""

from __future__ import annotations

import pytest

from edushield import core

RDAP_URL = "https://rdap.org/domain/example-admissions.com"
OFS_URL = "https://www.officeforstudents.org.uk/so-providers/"


def base_ctx(**kw):
    c = {
        "case_id": "c1",
        "verification": {
            "match_status": "verified",
            "matched_name": "University of Manchester",
            "jurisdiction": "England, UK",
            "manual_check": {"authority": "OfS Register", "url": OFS_URL},
            "domain": {"domain": "manchester.ac.uk", "created": "2001-08-30",
                       "age_days": 9000, "source_url": RDAP_URL, "source_status": "live"},
            "contradicted_claims": [],
        },
        "extraction": {"institution": "University of Manchester",
                       "claims": [{"claim_id": "C1", "text": "Fee is 28000",
                                   "type": "fee", "status": "VERIFIED"}]},
        "evidence": [{"id": "E1", "label": "Register match", "value": "Matched: University of Manchester",
                      "source_name": "OfS", "source_url": OFS_URL,
                      "retrieved_at": "2026-09-27", "tier": "static"}],
        "risk": {"level": "Low", "summary": "No concern found.", "reasons": []},
        "checklist": {"questions_to_ask": ["Which university issued this?"],
                      "registry_to_check": {"name": "OfS Register", "url": OFS_URL},
                      "documents_before_paying": ["Offer letter"],
                      "what_a_real_offer_looks_like": "From the university's own domain."},
        "audit": {"passed": True, "attempts": 1, "gaps": []},
    }
    c.update(kw)
    return c


# --- evidence policing -----------------------------------------------------

def test_evidence_without_a_url_is_dropped():
    out = core.grounding_gate(base_ctx(evidence=[
        {"id": "E1", "label": "ok", "source_url": OFS_URL, "retrieved_at": "2026-09-27"},
        {"id": "E2", "label": "no url", "retrieved_at": "2026-09-27"},
    ]))
    assert [e["id"] for e in out["evidence"]] == ["E1"]
    assert any("Evidence records dropped" in r for r in out["_grounding_final"]["repairs"])


def test_evidence_with_a_non_http_url_is_dropped():
    out = core.grounding_gate(base_ctx(evidence=[
        {"id": "E1", "label": "x", "source_url": "javascript:alert(1)",
         "retrieved_at": "2026-09-27"},
    ]))
    assert out["evidence"] == []


def test_evidence_ids_are_renumbered_contiguously():
    """The UI and the model's evidence_ref values both depend on these being E1..En
    in order. Gaps would silently break the cross-references."""
    out = core.grounding_gate(base_ctx(evidence=[
        {"id": "E7", "label": "a", "source_url": OFS_URL, "retrieved_at": "2026-09-27"},
        {"id": "E8", "label": "b", "source_url": RDAP_URL, "retrieved_at": "2026-09-27"},
    ]))
    assert [e["id"] for e in out["evidence"]] == ["E1", "E2"]


# --- verification policing -------------------------------------------------

def test_an_invalid_match_status_is_corrected_not_passed_through():
    out = core.grounding_gate(base_ctx(verification=dict(
        base_ctx()["verification"], match_status="probably_fine")))
    assert out["verification"]["match_status"] == "not_found"
    assert out["_grounding_final"]["repair_count"]


def test_a_missing_manual_check_authority_is_supplied():
    """An empty manual_check renders as an empty instruction in the UI, which
    leaves the student with no next step on the one card that tells them what to
    do next."""
    v = base_ctx()["verification"]
    v["manual_check"] = {}
    out = core.grounding_gate(base_ctx(verification=v))
    assert out["verification"]["manual_check"]["authority"]
    assert out["verification"]["manual_check"]["url"].startswith("http")


def test_zero_domain_age_with_a_creation_date_is_nulled():
    """"0 days old" reads as "registered today", which is an assertion we cannot
    support. Unknown must look unknown."""
    v = base_ctx()["verification"]
    v["domain"] = {"domain": "x.com", "created": "2026-09-27", "age_days": 0}
    out = core.grounding_gate(base_ctx(verification=v))
    assert out["verification"]["domain"]["age_days"] is None
    assert any("implausible" in r for r in out["_grounding_final"]["repairs"])


def test_a_non_numeric_domain_age_is_nulled():
    v = base_ctx()["verification"]
    v["domain"] = {"domain": "x.com", "created": "2026-09-27", "age_days": "a while"}
    out = core.grounding_gate(base_ctx(verification=v))
    assert out["verification"]["domain"]["age_days"] is None


# --- claim policing --------------------------------------------------------

def test_a_claim_with_an_unknown_type_or_status_is_neutralised():
    out = core.grounding_gate(base_ctx(extraction={"institution": "X", "claims": [
        {"claim_id": "C1", "text": "a", "type": "vibes", "status": "definitely_true"}]}))
    c = out["extraction"]["claims"][0]
    assert c["type"] == "other"
    assert c["status"] == "UNVERIFIED"


def test_an_emptied_claim_is_removed_rather_than_rendered_blank():
    out = core.grounding_gate(base_ctx(extraction={"institution": "X", "claims": [
        {"claim_id": "C1", "text": "   ", "type": "fee"}]}))
    assert out["extraction"]["claims"] == []


# --- registry URL policing -------------------------------------------------

def test_an_invented_registry_url_is_replaced_with_the_real_authority():
    """A wrong regulator URL sends a student to the wrong regulator, which is
    worse than sending them nowhere."""
    out = core.grounding_gate(base_ctx(checklist={
        "questions_to_ask": ["q"],
        "registry_to_check": {"name": "Fake Council", "url": "https://not-a-regulator.example/"},
        "documents_before_paying": ["d"],
        "what_a_real_offer_looks_like": "o"}))
    reg = out["checklist"]["registry_to_check"]
    assert reg["url"] == OFS_URL
    assert out["_grounding_final"]["registry_is_official"] is True
    assert any("not a known official register" in r for r in out["_grounding_final"]["repairs"])


def test_a_missing_registry_url_is_replaced_too():
    out = core.grounding_gate(base_ctx(checklist={"questions_to_ask": ["q"]}))
    assert out["checklist"]["registry_to_check"]["url"].startswith("http")


def test_a_wrong_official_host_is_never_accepted():
    for bad in ["https://ugc.gov.in.evil.example/", "https://evil.example/?x=ugc.gov.in",
                "not a url at all"]:
        out = core.grounding_gate(base_ctx(checklist={"registry_to_check": {"name": "x", "url": bad}}))
        assert out["_grounding_final"]["registry_is_official"] is True


# --- fallbacks -------------------------------------------------------------

def test_a_missing_checklist_gets_a_correct_default():
    out = core.grounding_gate(base_ctx(checklist={}))
    ck = out["checklist"]
    assert ck["questions_to_ask"]
    assert ck["documents_before_paying"]
    assert ck["what_a_real_offer_looks_like"]
    assert ck["registry_to_check"]["url"].startswith("http")


def test_a_judgement_phrased_checklist_item_is_dropped():
    out = core.grounding_gate(base_ctx(checklist={
        "questions_to_ask": ["This is a scam, what do you think?", "Which university issued this?"],
        "registry_to_check": {"name": "OfS", "url": OFS_URL}}))
    assert out["checklist"]["questions_to_ask"] == ["Which university issued this?"]


def test_checklist_lists_are_length_capped():
    out = core.grounding_gate(base_ctx(checklist={
        "questions_to_ask": ["q%d" % i for i in range(50)],
        "documents_before_paying": ["d%d" % i for i in range(50)],
        "registry_to_check": {"name": "OfS", "url": OFS_URL}}))
    assert len(out["checklist"]["questions_to_ask"]) <= 5
    assert len(out["checklist"]["documents_before_paying"]) <= 5


# --- never raises ----------------------------------------------------------

@pytest.mark.parametrize("payload", [
    {},
    {"verification": None},
    {"evidence": None},
    {"extraction": {"claims": "not a list"}},
    {"risk": "not a dict"},
    {"checklist": {"questions_to_ask": "not a list"}},
    {"audit": {"attempts": "many"}},
    {"evidence": [{"label": "no url or date"}]},
    {"verification": {"match_status": 42, "manual_check": None}},
    {"risk": {"level": None, "reasons": [None, {}, {"why": ""}]}},
])
def test_the_gate_never_raises_on_hostile_input(payload):
    """A blank screen is the worst thing this tool can do to a student who
    trusted it, so a small defect must degrade to a repaired envelope."""
    out = core.grounding_gate(base_ctx(**payload))
    assert isinstance(out, dict)
    assert out["verification"]["match_status"] in core.MATCH_STATUS
    assert out["risk"]["level"] in ("Low", "Medium", "High")
    assert out["checklist"]["registry_to_check"]["url"].startswith("http")
    assert isinstance(out["evidence"], list)
    assert isinstance(out["extraction"]["claims"], list)


def test_audit_attempts_are_clamped():
    out = core.grounding_gate(base_ctx(audit={"passed": True, "attempts": 99}))
    assert 1 <= out["audit"]["attempts"] <= 3


# --- the level cap is re_applied -------------------------------------------

def test_the_high_cap_is_re_applied_at_the_final_gate():
    """Cheap, and it means a later edit upstream cannot quietly reopen the hole
    the citation gate closed."""
    out = core.grounding_gate(base_ctx(
        verification={"match_status": "verified", "contradicted_claims": [],
                      "manual_check": {"authority": "OfS", "url": OFS_URL},
                      "domain": None, "contradicted_claims": []},
        risk={"level": "High", "summary": "s",
              "reasons": [{"flag": "f", "why": "w", "source_url": OFS_URL}]}))
    assert out["risk"]["level"] == "Medium"
    assert any("downgraded to Medium" in r for r in out["_grounding_final"]["repairs"])


def test_no_evidence_at_all_forces_low_even_at_the_final_gate():
    out = core.grounding_gate(base_ctx(evidence=[], risk={
        "level": "High", "summary": "s",
        "reasons": [{"flag": "f", "why": "w", "source_url": OFS_URL}]}))
    assert out["risk"]["level"] == "Low"


# --- verdict derivation ----------------------------------------------------

def test_no_evidence_means_insufficient_evidence():
    out = core.grounding_gate(base_ctx(evidence=[], risk={"level": "Low", "summary": "s",
                                                          "reasons": []}))
    assert core.derive_verdict(out) == "insufficient_evidence"


def test_a_known_bad_entry_means_serious_contradictions():
    out = core.grounding_gate(base_ctx(
        evidence=[{"id": "E1", "label": "Register match", "value": "MATCHED A KNOWN-BAD ENTRY",
                   "source_name": "UGC", "source_url": "https://www.ugc.gov.in/",
                   "retrieved_at": "2026-09-27", "tier": "static"}],
        risk={"level": "High", "summary": "s", "reasons": []}))
    assert core.derive_verdict(out) == "serious_contradictions"


def test_a_contradicted_claim_means_serious_contradictions():
    v = base_ctx()["verification"]
    v["contradicted_claims"] = ["C1"]
    out = core.grounding_gate(base_ctx(verification=v, risk={
        "level": "High", "summary": "s",
        "reasons": [{"flag": "f", "why": "w", "source_url": OFS_URL}]}))
    assert core.derive_verdict(out) == "serious_contradictions"


def test_a_medium_level_with_supporting_evidence_is_flags_raised():
    out = core.grounding_gate(base_ctx(risk={
        "level": "Medium", "summary": "s",
        "reasons": [{"flag": "f", "why": "w", "source_url": OFS_URL}]}))
    assert core.derive_verdict(out) == "flags_raised"


def test_a_low_level_backed_by_an_independent_source_is_no_concern_found():
    out = core.grounding_gate(base_ctx(risk={"level": "Low", "summary": "s", "reasons": []}))
    assert core.derive_verdict(out) == "no_concern_found"


def test_a_low_level_with_only_self_reports_is_not_a_clean_bill_of_health():
    """"No concern" is only honest if something independent supported it. With
    nothing retrievable we did not clear anyone, we simply could not check."""
    out = core.grounding_gate(base_ctx(
        evidence=[{"id": "E1", "label": "Provider's own website (self-report, NOT verification)",
                   "value": "we are accredited", "source_name": "them",
                   "source_url": "https://example-admissions.com",
                   "retrieved_at": "2026-09-27", "tier": "self_report"}],
        risk={"level": "Low", "summary": "s", "reasons": []}))
    assert core.derive_verdict(out) == "insufficient_evidence"


# --- the trace -------------------------------------------------------------

def test_the_trace_names_every_agent_and_carries_real_timings():
    out = core.grounding_gate(base_ctx(_t0=1000.0, _t_plan=1010.0, _t_retrieval=1100.0,
                                       _t_extract=1200.0, _t_verify=1210.0, _t_reason=1300.0,
                                       _t_gate=1310.0, _t_audit=1400.0, _t_action=1500.0,
                                       _t_ground=1510.0, pii_kinds_redacted=["email"],
                                       injection_detected=False, retrieval_plan=[{"step": "x"}]))
    env = core.assemble_fraud_envelope(out)
    agents = [t["agent"] for t in env["trace"]]
    for a in ["Sanitiser", "Planner", "Extractor", "Register match", "Risk reasoner",
              "Citation gate", "Self-audit", "Action planner", "Grounding gate"]:
        assert a in agents
    assert any("email" in t["did"] for t in env["trace"])


def test_a_missing_timing_is_null_rather_than_fabricated():
    """A fabricated trace is the kind of thing that unravels in Q&A."""
    out = core.grounding_gate(base_ctx())
    env = core.assemble_fraud_envelope(out)
    assert all(t["ms"] is None for t in env["trace"])


def test_the_layer_2_timings_are_reported_rather_than_dropped():
    """Register match, Domain cross-check and Risk reasoner all sit inside one
    block, and an earlier refactor dropped the stamps on the floor, so three
    agents showed ms: null on the live demo path. A null reads as "we did not
    measure this", which is exactly the doubt a measured trace exists to remove.
    """
    env = core.assemble_fraud_envelope(base_ctx(
        verification=dict(base_ctx()["verification"], official_domain_check={
            "checked": True, "submitted_domain": "manchester-offer-portal.example",
            "official_domain": "manchester.ac.uk", "match": "MISMATCH"}),
        _t_start=1000.0, _t0=1001.0, _t_plan=1002.0, _t_retrieval=1003.0, _t_extract=1020.0,
        _t_verify_block=1021.0, _t_match=1023.0, _t_domain=1024.0, _t_verify=1025.0,
        _t_reason=1040.0, _t_gate=1041.0, _t_audit=1045.0, _t_action=1050.0, _t_ground=1051.0))
    by_agent = {t["agent"]: t["ms"] for t in env["trace"]}
    assert by_agent["Register match"] == 2
    assert by_agent["Domain cross-check"] == 1
    assert by_agent["Risk reasoner"] == 15


def test_the_layer_2_timings_do_not_double_count():
    """Register match and Domain cross-check are adjacent slices of one block, so
    together they must not exceed the block they were measured inside."""
    env = core.assemble_fraud_envelope(base_ctx(
        verification=dict(base_ctx()["verification"], official_domain_check={
            "checked": True, "submitted_domain": "x.example",
            "official_domain": "manchester.ac.uk", "match": "MISMATCH"}),
        _t_start=1000.0, _t0=1001.0, _t_plan=1002.0, _t_retrieval=1003.0, _t_extract=1020.0,
        _t_verify_block=1021.0, _t_match=1030.0, _t_domain=1040.0, _t_verify=1045.0,
        _t_reason=1046.0, _t_gate=1047.0, _t_audit=1048.0, _t_action=1049.0, _t_ground=1050.0))
    by_agent = {t["agent"]: t["ms"] for t in env["trace"]}
    assert by_agent["Register match"] + by_agent["Domain cross-check"] <= 1045 - 1021


def test_the_profile_lookup_is_not_charged_for_the_extractor():
    """The profile branch reported one identical delta for both the model call
    and the dataset lookup, so the trace implied the lookup cost as much as a
    20-second LLM call. The register lookup is a dict search; it should say so.
    """
    built = core.build_profile("University of Manchester", "manchester.ac.uk")
    env = core.assemble_profile_envelope(
        "c1", "webhook", {"institution": "University of Manchester"}, built,
        "text", {"model_output_accepted": True},
        {"t_extract": 100.0, "t_profile_match": 300.0, "t_profile_ground": 400.0})
    by_agent = {t["agent"]: t["ms"] for t in env["trace"]}
    assert by_agent["Extractor"] == 200
    assert by_agent["Profile lookup"] < 5
    assert by_agent["Domain cross-check"] is not None


def test_the_trace_does_not_claim_there_was_no_domain_when_it_simply_failed_to_load():
    """"No domain in input" and "we could not reach the domain" are very different
    statements. Reporting the first when only the second is true understates our
    own check, and a judge reading the trace would believe it."""
    v = base_ctx()["verification"]
    v["official_domain_check"] = {"checked": True, "submitted_domain": "bright-path-edu.com",
                                  "official_domain": "manchester.ac.uk", "match": "MISMATCH"}
    v["domain"] = None
    out = core.grounding_gate(base_ctx(verification=v))
    env = core.assemble_fraud_envelope(out)
    line = [t for t in env["trace"] if t["agent"] == "Evidence retrieval"][0]["did"]
    assert "No domain in the input" not in line
    assert "bright-path-edu.com" in line


def test_the_trace_still_says_so_plainly_when_there_was_no_domain():
    v = base_ctx()["verification"]
    v["official_domain_check"] = None
    v["domain"] = None
    out = core.grounding_gate(base_ctx(verification=v))
    env = core.assemble_fraud_envelope(out)
    line = [t for t in env["trace"] if t["agent"] == "Evidence retrieval"][0]["did"]
    assert "No domain in the input" in line


def test_a_backwards_clock_cannot_produce_a_negative_duration():
    out = core.grounding_gate(base_ctx(_t0=2000.0, _t_plan=1000.0, _t_retrieval=1500.0,
                                       _t_extract=1200.0, _t_verify=1210.0, _t_reason=1300.0,
                                       _t_gate=1310.0, _t_audit=1400.0, _t_action=1500.0,
                                       _t_ground=1510.0))
    env = core.assemble_fraud_envelope(out)
    assert all((t["ms"] is None) or t["ms"] >= 0 for t in env["trace"])
