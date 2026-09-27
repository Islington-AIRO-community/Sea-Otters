"""The human-in-the-loop layer: the student decides, we only record.

This is the part of the project that makes the whole thing defensible. The agent
may assemble evidence and phrase risk, but the student is the decider, and the
merge function has to hold that line even when a decision looks like a bad one.

Two failure modes are worth more than the rest:

  * the merge SILENTLY CHANGING a finding, or
  * a decision for case A being applied to case B.

The first is the tool lying to its own user. The second is one student's decision
being written onto another student's report, and it is the reason a mismatched
case_id is reported rather than merged.
"""

from __future__ import annotations

from edushield import core


def parked(case_id="c1", **kw):
    env = {
        "case_id": case_id,
        "status": "awaiting_student",
        "intent": "fraud_check",
        "verdict": "flags_raised",
        "risk": {"level": "Medium", "summary": "The fee quote is not published on the official site.",
                 "reasons": [{"flag": "fee_not_published", "why": "w", "source_url": "https://www.ugc.gov.in/"}]},
        "verification": {"match_status": "verified", "matched_name": "University of Manchester",
                         "domain": {"domain": "example-admissions.com", "created": "2026-08-01",
                                    "age_days": 57, "source_url": "https://rdap.org/domain/example-admissions.com"},
                         "contradicted_claims": []},
        "evidence": [{"id": "E1", "label": "Register match", "value": "Matched",
                      "source_url": "https://www.ugc.gov.in/", "retrieved_at": "2026-09-27"}],
        "decision_prompt": {"question": "What would you like to do?",
                            "options": ["investigate_further", "accept", "dismiss"]},
        "disclaimer": core.DISCLAIMER,
    }
    env.update(kw)
    return env


# --- the decision is recorded, never applied --------------------------------

def test_accepting_changes_no_finding():
    before = parked()
    out = core.merge_student_decision(before, {"case_id": "c1", "decision": "accept"})
    assert out["risk"] == before["risk"]
    assert out["verification"] == before["verification"]
    assert out["evidence"] == before["evidence"]
    assert out["verdict"] == "flags_raised"


def test_dismissing_does_not_clear_the_concern():
    """The most important assertion in this file. "Dismiss" is a student's
    prerogative; it must not become a way to make a warning disappear."""
    before = parked()
    out = core.merge_student_decision(before, {"case_id": "c1", "decision": "dismiss"})
    assert out["verdict"] == "flags_raised"
    assert out["risk"]["level"] == "Medium"
    assert out["evidence"] == before["evidence"]
    assert "evidence stays exactly as it was" in out["decision_note"].lower()


def test_dismissing_never_downgrades_the_risk_level():
    for decision in ("accept", "dismiss", "investigate_further"):
        out = core.merge_student_decision(parked(), {"case_id": "c1", "decision": decision})
        assert out["risk"]["level"] == "Medium", decision


def test_the_decision_and_a_timestamp_are_always_recorded():
    out = core.merge_student_decision(parked(), {"case_id": "c1", "decision": "accept"})
    assert out["decision_recorded"] == "accept"
    assert out["decided_at"]
    assert out["status"] == "complete"


def test_an_unknown_decision_records_rather_than_raises():
    """The endpoint validates the vocabulary; the merge records whatever it is
    handed so a bad call is visible in the record rather than swallowed."""
    out = core.merge_student_decision(parked(), {"case_id": "c1", "decision": "whatever"})
    assert out["decision_recorded"] == "whatever"
    assert out["status"] == "complete"


def test_a_missing_decision_defaults_rather_than_crashing():
    out = core.merge_student_decision(parked(), {"case_id": "c1"})
    assert out["decision_recorded"] == "accept"


# --- case isolation --------------------------------------------------------

def test_a_decision_for_a_different_case_is_refused_loudly():
    out = core.merge_student_decision(parked("c1"), {"case_id": "c2", "decision": "dismiss"})
    assert out["decision_recorded"] == "rejected_case_id_mismatch"
    assert "was NOT applied" in out["decision_note"]


def test_a_refused_decision_still_leaves_the_findings_intact():
    before = parked("c1")
    out = core.merge_student_decision(before, {"case_id": "c2", "decision": "dismiss"})
    assert out["risk"] == before["risk"]
    assert out["verdict"] == "flags_raised"


def test_a_matching_case_id_is_merged():
    out = core.merge_student_decision(parked("c1"), {"case_id": "c1", "decision": "accept"})
    assert out["decision_recorded"] == "accept"


# --- investigate_further ---------------------------------------------------

def test_investigate_further_with_no_scope_says_so_instead_of_inventing():
    out = core.merge_student_decision(parked(), {"case_id": "c1", "decision": "investigate_further"})
    assert out["followup_findings"]
    assert "No scope was selected" in out["followup_findings"][0]["finding"]


def test_domain_history_scope_reports_the_retrieved_date():
    out = core.merge_student_decision(parked(), {
        "case_id": "c1", "decision": "investigate_further", "scope": ["domain_history"]})
    f = out["followup_findings"][0]
    assert f["scope"] == "domain_history"
    assert "2026-08-01" in f["finding"]
    assert f["source_url"] == "https://rdap.org/domain/example-admissions.com"


def test_an_unverifiable_domain_is_reported_as_unknown_not_as_fresh():
    """No RDAP answer must not become "registered recently, beware"."""
    p = parked()
    p["verification"]["domain"] = {"domain": "x.com", "created": None, "age_days": None}
    out = core.merge_student_decision(p, {
        "case_id": "c1", "decision": "investigate_further", "scope": ["domain_history"]})
    f = out["followup_findings"][0]
    assert "cannot be confirmed" in f["finding"]
    assert "not proof of anything" in f["finding"]


def test_company_registry_scope_never_claims_to_have_checked():
    p = parked()
    p["verification"]["domain"] = None
    out = core.merge_student_decision(p, {
        "case_id": "c1", "decision": "investigate_further", "scope": ["company_registry"]})
    f = out["followup_findings"][0]
    assert "not checked in this build" in f["finding"]
    assert f["source_url"].startswith("https://")


def test_every_valid_scope_is_honoured():
    scopes = ["domain_history", "company_registry", "fee_verification", "advisories"]
    out = core.merge_student_decision(parked(), {
        "case_id": "c1", "decision": "investigate_further", "scope": scopes})
    got = sorted(f["scope"] for f in out["followup_findings"])
    assert got == sorted(scopes)


def test_an_unknown_scope_is_ignored_rather_than_echoed():
    out = core.merge_student_decision(parked(), {
        "case_id": "c1", "decision": "investigate_further",
        "scope": ["domain_history", "not_a_scope"]})
    assert [f["scope"] for f in out["followup_findings"]] == ["domain_history"]


def test_a_scope_that_is_not_a_list_is_ignored():
    out = core.merge_student_decision(parked(), {
        "case_id": "c1", "decision": "investigate_further", "scope": "domain_history"})
    assert out["followup_findings"][0]["scope"] == "none"


def test_findings_are_kept_separate_from_the_original_evidence():
    """A student must be able to see what was known when the report was first
    shown, separately from what the follow-up added."""
    before = parked()
    out = core.merge_student_decision(before, {
        "case_id": "c1", "decision": "investigate_further", "scope": ["advisories"]})
    assert out["evidence"] == before["evidence"]
    assert out["followup_findings"]


def test_accepting_adds_no_findings():
    out = core.merge_student_decision(parked(), {"case_id": "c1", "decision": "accept"})
    assert "followup_findings" not in out


def test_the_disclaimer_survives_the_merge():
    for decision in ("accept", "dismiss", "investigate_further"):
        out = core.merge_student_decision(parked(), {"case_id": "c1", "decision": decision})
        assert out["disclaimer"] == core.DISCLAIMER


def test_the_parked_envelope_is_not_mutated_in_place():
    """The store holds the same dict; an in-place edit would rewrite history for
    anyone reloading the case."""
    before = parked()
    core.merge_student_decision(before, {"case_id": "c1", "decision": "dismiss"})
    assert "decision_recorded" not in before
    assert before["status"] == "awaiting_student"
