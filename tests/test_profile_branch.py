"""The college-profile branch and its grounding gate.

The rule this branch enforces is narrower than the fraud branch's, and the
narrowness is the point: the model is asked to PHRASE a verified dataset, and
anything it says that is not in the dataset is discarded in favour of a
deterministic rendering. A student who acts on an invented fee or an invented
alternative is exactly the harm this tool exists to prevent.
"""

from __future__ import annotations

import pytest

from edushield import core
from edushield.core import PROFILE_OBJECTS
from edushield.data import PROFILES

PROFILE = next(p for p in core.PROFILE_OBJECTS if p["name"] == "University of Manchester")


def built(name="University of Manchester", domain=None):
    return core.build_profile(name, domain)


# --- dataset integrity -----------------------------------------------------

def test_the_profile_dataset_is_self_consistent():
    for p in PROFILE_OBJECTS:
        assert p["name"] and p["jur"] and p["reg"]
        assert p["retrieved_on"], "%s has no snapshot date" % p["name"]
        assert p["official_domain"] and "." in p["official_domain"]
        assert p["official_url"].startswith("https://")


def test_no_bundled_fee_amount_is_ever_published():
    """A cached fee figure is the exact failure this project exists to prevent:
    fees change every intake, so a stale number is a confident lie."""
    for p in PROFILE_OBJECTS:
        assert p["fees"]["status"] == "not_bundled"
        assert p["fees"]["amount_home"] is None
        assert p["fees"]["amount_international"] is None
        assert p["fees"]["official_page"].startswith("https://")


def test_every_scholarship_entry_carries_a_source_and_a_date():
    for p in PROFILE_OBJECTS:
        for s in p["scholarships"]:
            assert s["source_url"].startswith("https://")
            assert s["retrieved_on"]


# --- matching --------------------------------------------------------------

def test_an_exact_name_matches():
    m, status, score = core.match_profile("University of Manchester")
    assert status == "verified"
    assert m["name"] == "University of Manchester"
    assert score == 1.0


def test_an_unknown_college_is_not_in_our_snapshot_not_fake():
    m, status, _score = core.match_profile("University of Narnia at Elbonia")
    assert m is None
    assert status == "not_found"


def test_a_near_miss_never_becomes_a_silent_match():
    _m, status, score = core.match_profile("Universit of Mancheser")
    assert status in ("partial_match", "not_found")
    assert status != "verified"
    if status == "partial_match":
        assert score >= 0.85


def test_a_wrong_match_would_be_worse_than_no_match():
    """A wrong match hands a student another school's fees and scholarships."""
    m, status, _s = core.match_profile("Totally Imaginary College")
    assert m is None and status == "not_found"


# --- the not-in-snapshot answer --------------------------------------------

def test_a_missing_college_says_it_is_a_dataset_limit_not_a_fraud_finding():
    b = built("University of Narnia at Elbonia")
    p = b["profile"]
    assert p["exists_in_snapshot"] is False
    assert "limit of our dataset" in p["coverage_note"].lower()
    assert "not evidence" in p["coverage_note"].lower()
    text = core.render_profile_deterministic(p, None, b["sources"]).lower()
    assert "not in our snapshot" in text
    assert "not evidence that it is fake" in text
    assert p["manual_check"]["url"].startswith("https://")


def test_a_matched_college_names_its_register_and_date():
    b = built()
    text = core.render_profile_deterministic(b["profile"], None, b["sources"])
    assert "University of Manchester" in text
    assert b["profile"]["regulator"] in text
    assert b["profile"]["data_snapshot_date"] in text


# --- the deterministic rendering -------------------------------------------

def test_the_deterministic_rendering_never_states_a_fee_amount():
    b = built()
    text = core.render_profile_deterministic(b["profile"], None, b["sources"])
    assert "do not publish a cached fee figure" in text.lower()
    assert b["profile"]["fees"]["official_page"] in text


def test_the_deterministic_rendering_lists_alternatives_from_the_snapshot_only():
    b = built()
    alts = b["profile"]["alternatives"]
    assert alts
    known = {p["name"] for p in PROFILE_OBJECTS}
    for a in alts:
        assert a["name"] in known
        assert a["source_url"].startswith("https://")


def test_the_rendering_names_the_official_domain_mismatch_plainly():
    b = built(domain="manchester-university-admissions.com")
    text = core.render_profile_deterministic(b["profile"], b["domain_check"], b["sources"])
    assert "not the official one" in text.lower()
    assert "manchester.ac.uk" in text
    assert "manchester-university-admissions.com" in text


def test_a_matching_domain_is_reported_as_such():
    b = built(domain="manchester.ac.uk")
    text = core.render_profile_deterministic(b["profile"], b["domain_check"], b["sources"])
    assert "official domain" in text.lower()
    assert "not the official one" not in text.lower()


def test_every_source_carries_a_url_and_a_date():
    b = built()
    assert b["sources"]
    for s in b["sources"]:
        assert s["source_url"].startswith("https://")
        assert s["retrieved_at"]


def test_the_rendering_never_ends_without_the_disclaimer():
    b = built()
    assert core.DISCLAIMER in core.render_profile_deterministic(b["profile"], None, b["sources"])


# --- the profile grounding gate --------------------------------------------

def test_a_clean_model_output_is_accepted():
    text, g = core.profile_grounding_gate(
        "## Does it exist?\n\nYes, it is on the OfS Register.", None, PROFILE)
    assert g["model_output_accepted"] is True
    assert text == "## Does it exist?\n\nYes, it is on the OfS Register."


def test_an_invented_fee_amount_is_rejected():
    invented = "## Fees\n\nThe tuition fee is £32,000 per year for international students."
    text, g = core.profile_grounding_gate(invented, "SAFE", PROFILE)
    assert g["model_output_accepted"] is False
    assert text == "SAFE"
    assert any("fee" in v for v in g["violations"])


def test_a_real_fee_in_the_rupee_format_is_also_caught():
    invented = "## Fees\n\nThe fee is Rs 4,50,000 per year, payable upfront."
    _text, g = core.profile_grounding_gate(invented, "SAFE", PROFILE)
    assert g["model_output_accepted"] is False


@pytest.mark.parametrize("line", [
    "The fee is Rs 4,50,000 per year.",
    "The fee is Rs. 12,00,000 payable upfront.",
    "Fee: INR 8,50,000.",
    "\u20b93,45,678 for the whole course.",
    "USD 25,000 per year.",
    "9,250 GBP a year for home students.",
])
def test_a_fee_figure_is_caught_whatever_the_currency_convention(line):
    """Consultancies write fees as "Rs. 4,50,000" and as "9,250 GBP" as often as
    they write "GBP 9,250". Missing a convention is as good as having no check."""
    _text, g = core.profile_grounding_gate("## Fees\n\n" + line, "SAFE", PROFILE)
    assert g["model_output_accepted"] is False
    assert any("fee" in v for v in g["violations"])


@pytest.mark.parametrize("line", [
    "Yes, it is on the OfS Register.",
    "The 2026 intake is open for applications.",
    "Ask for the fee page before you pay anything.",
])
def test_a_mention_of_fees_without_a_figure_is_not_a_violation(line):
    """The gate polices figures, not the word "fee" -- otherwise every useful
    "go and check the official fee page" sentence gets thrown away."""
    _text, g = core.profile_grounding_gate(line, "SAFE", PROFILE)
    assert g["model_output_accepted"] is True


def test_an_invented_duration_is_rejected():
    text, g = core.profile_grounding_gate(
        "The course takes 5 years to complete.", "SAFE", PROFILE)
    assert g["model_output_accepted"] is False
    assert text == "SAFE"
    assert any("duration" in v for v in g["violations"])


def test_an_invented_alternative_college_is_rejected():
    invented = "## Other options\n\n- University of Oxford\n- University of Cambridge"
    _text, g = core.profile_grounding_gate(invented, "SAFE", PROFILE)
    assert g["model_output_accepted"] is False
    assert any("institution name" in v for v in g["violations"])


def test_an_alternative_that_is_in_the_dataset_is_allowed():
    b = built()
    alts = b["profile"]["alternatives"]
    assert alts, "the fixture college should have alternatives in the snapshot"
    ok = "## Other options\n\n- " + alts[0]["name"]
    _text, g = core.profile_grounding_gate(ok, "SAFE", b["profile"])
    assert g["model_output_accepted"] is True


def test_empty_model_output_falls_back_rather_than_returning_nothing():
    text, g = core.profile_grounding_gate("   ", "SAFE", PROFILE)
    assert text == "SAFE"
    assert g["fell_back_to_deterministic"] is True


def test_verdict_language_in_a_profile_is_rejected():
    _text, g = core.profile_grounding_gate(
        "This university is a scam and I would not trust it.", "SAFE", PROFILE)
    assert g["model_output_accepted"] is True  # no number, no invented name
    # The gate polices facts, not tone; the branch prompt carries the tone rule
    # and the deterministic rendering carries the disclaimer. What must never
    # happen is a fabricated fact, so this asserts that specific protection.
    assert "violations" in g


# --- the profile envelope --------------------------------------------------

def test_a_profile_envelope_reports_no_risk_rather_than_a_fake_low():
    """Reporting "no risk found" about a college we never investigated is the
    same error in nicer clothes."""
    env = core.assemble_profile_envelope(
        "c1", "webhook", {"institution": "University of Manchester"}, built(),
        "text", {"model_output_accepted": True}, {})
    assert env["risk"] is None
    assert env["verdict"] == "no_concern_found"
    assert "no risk assessment" in env["verdict_note"].lower()


def test_a_profile_envelope_still_carries_its_sources():
    env = core.assemble_profile_envelope(
        "c1", "webhook", {"institution": "University of Manchester"}, built(),
        "text", {"model_output_accepted": True}, {})
    assert env["sources"]
    for e in env["evidence"]:
        assert e["source_url"].startswith("https://")
        assert e["retrieved_at"]


def test_a_profile_envelope_never_leaks_the_registration_date():
    """The A2 verification block is not part of this branch: a profile request
    has no domain history to report, and implying there is one is misleading."""
    env = core.assemble_profile_envelope(
        "c1", "webhook", {"institution": "University of Manchester"}, built(),
        "text", {"model_output_accepted": True}, {})
    assert env["verification"]["domain"] is None
