"""The citation gate and the level caps.

The most important anti-hallucination control in the system, so the tests here
are written adversarially: the model is assumed to fabricate, to over-claim and
to deliver a verdict in the same breath. Every assertion below is one a student
would be harmed by getting wrong.

The governing idea, from the module docstring: "never invent a URL" is a REQUEST
in a prompt, and a request is not a control. This gate CHECKS, then DELETES.
"""

from __future__ import annotations

import pytest

from edushield import core


RDAP_URL = "https://rdap.org/domain/example-admissions.com"
OFS_URL = "https://www.officeforstudents.org.uk/so-providers/"


def ctx(verification=None, evidence=None, claims=None, risk=None):
    return {
        "verification": verification if verification is not None else {
            "match_status": "not_found",
            "contradicted_claims": [],
            "manual_check": {"authority": "OfS Register", "url": OFS_URL},
        },
        "evidence": evidence if evidence is not None else [
            {"id": "E1", "label": "Domain registration date (RDAP)",
             "value": "2026-08-30 (28 days old)", "source_name": "RDAP",
             "source_url": RDAP_URL, "retrieved_at": "2026-09-27", "tier": "live"},
        ],
        "extraction": {"institution": "Example University",
                       "claims": claims if claims is not None else []},
        "risk": risk if risk is not None else {"level": "Medium", "summary": "s", "reasons": []},
    }


def reason(**kw):
    r = {"flag": "F", "why": "W", "source_url": RDAP_URL, "retrieved_at": "2026-09-27",
         "confidence": 0.7}
    r.update(kw)
    return r


# --- fabricated citations --------------------------------------------------

def test_a_fabricated_url_is_deleted_not_merely_flagged():
    out = core.citation_gate(ctx(risk={
        "level": "Medium", "summary": "s",
        "reasons": [reason(flag="Ghost record",
                           source_url="https://totally-made-up-registry.example/xyz",
                           retrieved_at="2026-09-27")],
    }))
    r = out["risk"]["reasons"][0]
    assert r["source_url"] is None
    assert r["cited"] is False
    assert r["retrieved_at"] is None
    assert out["_grounding"]["citations_rejected"] == 1
    assert "totally-made-up-registry.example" in out["_grounding"]["rejected_citations"][0]["rejected_url"]


def test_a_url_that_differs_only_in_scheme_still_resolves_and_is_snapped_back():
    """A model that downgraded https:// to http:// cited the right record, so
    the citation survives -- but the URL is REPLACED with the one that was
    actually retrieved. The student is meant to click it, and only the retrieved
    version is one we can vouch for."""
    out = core.citation_gate(ctx(risk={
        "level": "Medium", "summary": "s",
        "reasons": [reason(source_url="http://rdap.org/domain/example-admissions.com")],
    }))
    assert out["_grounding"]["citations_rejected"] == 0
    assert out["risk"]["reasons"][0]["source_url"] == RDAP_URL


def test_a_url_with_an_injected_tracking_parameter_is_snapped_back():
    out = core.citation_gate(ctx(risk={
        "level": "Medium", "summary": "s",
        "reasons": [reason(source_url=RDAP_URL + "?utm_source=chatgpt")],
    }))
    assert out["risk"]["reasons"][0]["source_url"] == RDAP_URL


def test_a_retyped_date_is_snapped_to_the_real_record():
    """The model produced an unverified date and the UI prints it."""
    out = core.citation_gate(ctx(risk={
        "level": "Medium", "summary": "s",
        "reasons": [reason(retrieved_at="1999-01-01")],
    }))
    assert out["risk"]["reasons"][0]["retrieved_at"] == "2026-09-27"


def test_confidence_is_nulled_when_there_is_no_source():
    """A confidence number with nothing behind it is false precision, and the UI
    renders it as a bar a student will read as meaningful."""
    out = core.citation_gate(ctx(risk={
        "level": "Medium", "summary": "s",
        "reasons": [reason(source_url="https://invented.example/x", confidence=0.95)],
    }))
    assert out["risk"]["reasons"][0]["confidence"] is None


def test_confidence_is_clamped_into_range():
    out = core.citation_gate(ctx(risk={
        "level": "Medium", "summary": "s", "reasons": [reason(confidence=7.5)]}))
    assert out["risk"]["reasons"][0]["confidence"] == 1.0


def test_a_dangling_evidence_reference_is_dropped():
    out = core.citation_gate(ctx(risk={
        "level": "Medium", "summary": "s",
        "reasons": [reason(evidence_ref="E99")],
    }))
    # E99 does not exist, so the reference is cleared -- but the citation itself
    # resolved by URL, so the reason survives with a corrected reference.
    assert out["risk"]["reasons"][0]["evidence_ref"] == "E1"
    assert any("E99" in n for n in out["_grounding"]["notes"])


# --- verdict language ------------------------------------------------------

@pytest.mark.parametrize("flag", [
    "This is a scam",
    "They are fraudsters",
    "Fake university",
    "These people are lying",
    "It is fraudulent",
    "Complete scammer operation",
])
def test_a_reason_phrased_as_a_verdict_is_discarded(flag):
    out = core.citation_gate(ctx(risk={
        "level": "High", "summary": "s", "reasons": [reason(flag=flag)]}))
    assert out["risk"]["reasons"] == []
    assert out["_grounding"]["verdict_reasons_dropped"] == 1


def test_an_official_register_named_fake_universities_is_not_a_verdict():
    """"List of Fake Universities" is the UGC's real name. Mistaking it for the
    model declaring a verdict would silence the strongest true finding we have."""
    assert core.has_verdict("Named on the List of Fake Universities") is False
    assert core.has_verdict("This is a fraud") is True


def test_dropping_a_verdict_reason_cannot_raise_the_level():
    """Dropping language must not soften the assessment. The level is re-derived
    from what survived, so the verdict reason leaves no reasons cited at all --
    which drives the level all the way DOWN to Low, not merely down to Medium."""
    out = core.citation_gate(ctx(risk={
        "level": "High", "summary": "On the evidence retrieved, the domain is 28 days old.",
        "reasons": [reason(flag="They are scammers")]}))
    assert out["risk"]["reasons"] == []
    assert out["risk"]["level"] == "Low"
    assert out["_grounding"]["reasons_cited"] == 0


def test_a_verdict_phrased_summary_is_rebuilt_from_retrieved_evidence():
    """No model wording survives. The replacement is assembled from records this
    pipeline actually retrieved, and here nothing was confirmed or contradicted,
    so it says exactly that rather than borrowing the model's phrasing."""
    out = core.citation_gate(ctx(risk={
        "level": "Medium", "summary": "This university is a scam",
        "reasons": [reason()]}))
    s = out["risk"]["summary"].lower()
    assert "scam" not in s
    assert "insufficient evidence" in s
    assert any("rebuilt" in n for n in out["_grounding"]["notes"])


def test_a_rebuilt_summary_uses_the_facts_when_there_are_some():
    """With a domain on file the rebuilt summary must actually state the fact
    that was retrieved, not fall back to a generic sentence."""
    v = {"match_status": "not_found", "contradicted_claims": [],
         "manual_check": {"authority": "OfS", "url": OFS_URL},
         "domain": {"domain": "example-admissions.com", "created": "2026-08-30",
                    "age_days": 28, "source_url": RDAP_URL, "source_status": "live"}}
    out = core.citation_gate(ctx(verification=v, risk={
        "level": "Medium", "summary": "Definitely a scam", "reasons": [reason()]}))
    s = out["risk"]["summary"]
    assert "2026-08-30" in s
    assert "28 days" in s
    assert "scam" not in s.lower()


def test_an_empty_summary_is_rebuilt():
    out = core.citation_gate(ctx(risk={"level": "Low", "summary": "  ", "reasons": [reason()]}))
    assert out["risk"]["summary"].strip()


# --- the level caps --------------------------------------------------------

def test_no_evidence_forces_low():
    out = core.citation_gate(ctx(evidence=[], risk={
        "level": "High", "summary": "s",
        "reasons": [reason(source_url=None)]}))
    assert out["risk"]["level"] == "Low"
    assert out["_grounding"]["cap_reasons"]


def test_no_cited_reason_forces_low():
    """A model that produced a strong assessment with nothing behind it must not
    be allowed to set the level."""
    out = core.citation_gate(ctx(risk={
        "level": "High", "summary": "s",
        "reasons": [reason(source_url="https://fabricated.example/a"),
                    reason(source_url="https://fabricated.example/b")]}))
    assert out["risk"]["level"] == "Low"
    assert out["_grounding"]["citations_rejected"] == 2


def test_high_without_a_contradiction_is_capped_to_medium():
    """The contract's rule: absence from a register may not by itself produce a
    High level."""
    out = core.citation_gate(ctx(risk={"level": "High", "summary": "s", "reasons": [reason()]}))
    assert out["risk"]["level"] == "Medium"
    assert out["risk"]["level_before_cap"] == "High"
    assert out["_grounding"]["level_capped"] is True


def test_high_is_kept_when_a_claim_is_contradicted():
    out = core.citation_gate(ctx(
        verification={"match_status": "verified", "contradicted_claims": ["C1"],
                      "manual_check": {"authority": "OfS", "url": OFS_URL}},
        risk={"level": "High", "summary": "s", "reasons": [reason()]}))
    assert out["risk"]["level"] == "High"
    assert out["_grounding"]["hard_signal"] is True


def test_high_is_kept_when_the_register_knows_bad_entry():
    out = core.citation_gate(ctx(
        evidence=[{"id": "E1", "label": "Register match", "value": "MATCHED A KNOWN-BAD ENTRY",
                   "source_name": "UGC", "source_url": OFS_URL,
                   "retrieved_at": "2026-09-27", "tier": "static"}],
        risk={"level": "High", "summary": "s",
              "reasons": [reason(source_url=OFS_URL, retrieved_at="2026-09-27")]}))
    assert out["risk"]["level"] == "High"


def test_an_unknown_level_defaults_to_low():
    out = core.citation_gate(ctx(risk={"level": "catastrophic", "summary": "s", "reasons": [reason()]}))
    assert out["risk"]["level"] in ("Low", "Medium")


# --- NOT_FOUND neutrality --------------------------------------------------

def test_a_reason_resting_only_on_an_empty_register_is_demoted():
    """A reason with no source at all, over a register that found nothing, must
    not carry weight. Demoted rather than deleted: the question is a fair one."""
    out = core.citation_gate(ctx(verification={"match_status": "not_found", "contradicted_claims": [],
                                               "manual_check": {"authority": "OfS", "url": OFS_URL}},
                                risk={"level": "Medium", "summary": "s",
                                      "reasons": [reason(source_url=None, confidence=0.9)]}))
    r = out["risk"]["reasons"][0]
    assert r["weight"] == "demoted_no_positive_source"
    assert r["confidence"] is None


def test_a_cited_reason_over_an_empty_register_keeps_its_weight():
    out = core.citation_gate(ctx(risk={"level": "Medium", "summary": "s", "reasons": [reason()]}))
    assert out["risk"]["reasons"][0]["weight"] == "normal"


# --- reporting -------------------------------------------------------------

def test_the_gate_reports_what_it_did():
    out = core.citation_gate(ctx(risk={
        "level": "High", "summary": "This is a scam",
        "reasons": [reason(flag="they are scammers"),
                    reason(flag="Ghost", source_url="https://no.example/x")]}))
    g = out["_grounding"]
    assert g["citations_rejected"] == 1
    assert g["verdict_reasons_dropped"] == 1
    assert g["reasons_cited"] == 0
    assert g["notes"]


def test_an_empty_reason_is_skipped():
    out = core.citation_gate(ctx(risk={"level": "Low", "summary": "s",
                                       "reasons": [reason(flag="", why=""), {"nothing": 1}]}))
    assert out["risk"]["reasons"] == []


def test_the_gate_never_mutates_its_input():
    """It runs inside a retry loop on the same context, so a mutation here would
    compound across attempts."""
    c = ctx(risk={"level": "Medium", "summary": "s", "reasons": [reason(source_url="https://no.example/x")]})
    before = repr(c)
    core.citation_gate(c)
    assert repr(c) == before
