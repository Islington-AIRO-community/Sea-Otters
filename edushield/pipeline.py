"""The orchestrator. This file replaces the n8n graph.

Same order, same agents, same gates, same loop. The only difference is that a
function call replaces a node edge, so `awaiting_student` is served from an
in-process store instead of a Wait node that holds an execution open.

The pipeline is deliberately written so that **every** model call has a
deterministic fallback. A local 7B model is weaker than the n8n deployment's
model and a live demo cannot depend on it behaving; if A2 returns nothing
parseable the citation gate still enforces the level cap, and if A4 returns
nothing the grounding gate still supplies a correct checklist. The system fails
closed, never open.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from . import core, prompts, tools
from .llm import get_ollama

# One initial pass plus one retry. A third pass costs a full risk generation
# (~30s) to re-word a report the deterministic gates have already policed, which
# is the wrong thing to spend a five-minute demo slot on.
MAX_AUDIT_ATTEMPTS = 2

# Wall-clock ceiling for ONE investigation, in seconds of model time. Every
# model call derives its own timeout from what is left, so the total is bounded
# no matter how the model behaves; once the budget is spent, calls return
# immediately and the deterministic fallbacks complete the report. Override on
# the demo machine with EDUSHIELD_BUDGET if the hardware is faster or slower.
BUDGET_SECONDS = float(os.environ.get("EDUSHIELD_BUDGET", "45"))


def _fallback_extraction(agent_text: str) -> dict:
    """If A1 fails entirely, the pipeline still returns a shaped extraction with
    zero claims. A null is a correct answer; an invented value is a failure."""
    return {
        "institution": None,
        "program": None,
        "consultancy": None,
        "claimed_fees": None,
        "claimed_timeline": None,
        "claimed_guarantees": [],
        "claims": [],
    }


def _fallback_risk(level: str = "Low") -> dict:
    return {
        "level": level,
        "summary": "The automated reasoner did not return a usable assessment, so no risk level above Low is claimed. The evidence list below was still checked deterministically.",
        "reasons": [],
    }


def _fallback_audit(gaps: list) -> dict:
    return {"passed": False, "gaps": gaps or ["The automated auditor did not return a usable result."], "followup_queries": []}


def _fallback_checklist(verification: dict) -> dict:
    mc = verification.get("manual_check") or core.DEFAULT_AUTHORITY
    return {
        "questions_to_ask": [
            "Which university issued this, and can I see the offer letter on their own domain?",
            "What is the written refund policy if the visa is refused?",
            "Are these fees taken from the university's published fee page?",
        ],
        "registry_to_check": {"name": mc["authority"], "url": mc["url"]},
        "documents_before_paying": [
            "Written offer letter on university letterhead",
            "Signed refund / cancellation policy",
            "Bank details matching the registered company name exactly",
        ],
        "what_a_real_offer_looks_like": "A genuine offer is issued from the university's own domain, references a CAS number verifiable with the university, and its fee figures match the university's published fee page.",
    }


def _coerce_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _link_claims_to_evidence(claims, evidence):
    """Attach the evidence ids that actually bear on each claim, deterministically.

    A2 is a risk WEIGHING step, and weighing needs to know which record speaks to
    which claim. Without this the model is handed one flat list and free to
    attach a fee record to a guarantee claim -- a false citation even though both
    records are real. The link is a fact about the data, so it is computed here
    and never asked for.

    A claim that no record bears on keeps an empty list. Inventing a link would
    be worse than having none, because the citation gate would then find a real
    record at the other end of it and believe it.
    """
    ids = {e.get("label", ""): e.get("id") for e in evidence}

    def find(*needles):
        return sorted(
            {eid for label, eid in ids.items() if eid and any(n in label.lower() for n in needles)}
        )

    for c in claims:
        t = c.get("type")
        if t == "guarantee":
            # A guarantee is a claim ABOUT the provider, so the provider's own
            # words are the record that speaks to it. Tier self_report, which is
            # never treated as verification -- but it is the right thing to weigh.
            c["evidence_ids"] = find("self-report", "website")
        elif t in ("fee", "accreditation", "deadline"):
            # State recognition and accreditation are different checks, so the
            # register is what bears on both, and the blacklist when it hits.
            c["evidence_ids"] = find("blacklist") or find("register")
        elif t == "timeline":
            c["evidence_ids"] = find("blacklist") or find("domain registration date")
        else:
            c["evidence_ids"] = find("official domain", "blacklist", "register", "domain registration date")


def run_profile_branch(ctx: dict) -> Dict[str, Any]:
    """College profile lookup. The dataset is the source of truth; the model is
    only allowed to phrase it, and its phrasing is checked back against the data."""
    llm = get_ollama()
    t_extract = core.now_ms()

    agent_text = core.assemble_agent_text(ctx, (ctx.get("retrieval") or {}).get("site") or {})
    parsed, _meta = llm.json_call(
        prompts.a1(agent_text), system="You output strict JSON only.", temperature=0.0
    )
    extraction = parsed if isinstance(parsed, dict) else _fallback_extraction(agent_text)

    t_profile_match = core.now_ms()
    built = core.build_profile(
        (extraction or {}).get("institution"), ctx.get("domain")
    )
    deterministic = core.render_profile_deterministic(
        built["profile"], built.get("domain_check"), built["sources"]
    )

    text_call, _ = llm.text_call(
        prompts.profile_formatter(built["profile"], built.get("domain_check"), built["sources"]),
        temperature=0.2,
        num_predict=330,
    )
    t_profile_ground = core.now_ms()
    text, grounding = core.profile_grounding_gate(
        text_call or "", deterministic, built["profile"]
    )

    return core.assemble_profile_envelope(
        case_id=ctx["case_id"],
        entry=ctx.get("_entry", "webhook"),
        extraction=extraction,
        built=built,
        text=text,
        grounding=grounding,
        timings={
            "t_extract": t_extract,
            "t_profile_match": t_profile_match,
            "t_profile_ground": t_profile_ground,
        },
    )


def run_fraud_branch(ctx: dict) -> Dict[str, Any]:
    """LAYER 1-6: the full multi-agent investigation."""
    llm = get_ollama()
    site = (ctx.get("retrieval") or {}).get("site") or {}
    rdap = (ctx.get("retrieval") or {}).get("rdap") or {}

    # ---- A1: Extractor ----------------------------------------------------
    agent_text = core.assemble_agent_text(ctx, site)
    t_before_extract = core.now_ms()
    parsed, a1_meta = llm.json_call(
        prompts.a1(agent_text), system="You output strict JSON only.", temperature=0.0
    )
    ctx["_t_extract"] = core.now_ms()
    extraction = parsed if isinstance(parsed, dict) else _fallback_extraction(agent_text)
    # A model can hand back a JSON envelope nested under a key; normalise once
    # here rather than teaching three downstream nodes about it.
    if not extraction.get("claims") and isinstance(extraction.get("extraction"), dict):
        extraction = extraction["extraction"]
    extraction["claims"] = [
        {
            "claim_id": str(c.get("claim_id") or "C%d" % (i + 1)),
            "text": str(c.get("text") or "")[:300],
            "type": c.get("type") if c.get("type") in core.CLAIM_TYPES else "other",
        }
        for i, c in enumerate(_coerce_list(extraction.get("claims"))[:10])
        if isinstance(c, dict) and c.get("text")
    ]

    # ---- LAYER 2: Verify + Build Evidence (deterministic) -----------------
    verified = core.verify_and_build_evidence(ctx, extraction, site, rdap)
    ctx.update(
        {
            "extraction": verified["extraction"],
            "verification": verified["verification"],
            "evidence": verified["evidence"],
            "evidence_count": verified["evidence_count"],
        }
    )
    # The trace is only as honest as the stamps we actually carry forward, and
    # dropping these made three agents report ms: null on the live demo path.
    for k in ("_t_verify_block", "_t_match", "_t_domain", "_t_verify"):
        if verified.get(k) is not None:
            ctx[k] = verified[k]

    _link_claims_to_evidence(ctx["extraction"]["claims"], ctx["evidence"])

    # ---- A2 -> Citation Gate -> A3 -> (retry) -> A4 ------------------------
    audit = _fallback_audit([])
    attempts = 0
    gaps: list = []
    ctx["_grounding"] = {}

    for attempt in range(1, MAX_AUDIT_ATTEMPTS + 1):
        attempts = attempt

        t_before_reason = core.now_ms()
        if attempt == 1:
            risk_prompt = prompts.a2(
                ctx["verification"], ctx["evidence"], ctx["extraction"]["claims"],
                site, ctx.get("pii_kinds_redacted"), ctx.get("injection_detected"),
            )
        else:
            risk_prompt = prompts.a2_retry(
                ctx["verification"], ctx["evidence"], ctx["extraction"]["claims"],
                site, ctx.get("pii_kinds_redacted"), ctx.get("injection_detected"), gaps,
            )
        risk_parsed, a2_meta = llm.json_call(
            risk_prompt, system="You output strict JSON only.", temperature=0.2, num_predict=400
        )
        ctx["_t_reason"] = core.now_ms()

        risk_in = risk_parsed if isinstance(risk_parsed, dict) else _fallback_risk()
        ctx["risk"] = {
            "level": risk_in.get("level") if risk_in.get("level") in ("Low", "Medium", "High") else "Low",
            "summary": str(risk_in.get("summary") or ""),
            "reasons": [r for r in _coerce_list(risk_in.get("reasons")) if isinstance(r, dict)],
        }

        # The citation gate runs unconditionally, even when the model failed.
        # That is the point of it: an empty assessment is already capped at Low.
        ctx = core.citation_gate(ctx)
        ctx["_t_gate"] = core.now_ms()

        t_before_audit = core.now_ms()
        audit_parsed, _ = llm.json_call(
            prompts.a3(
                ctx["risk"], ctx["evidence"], ctx["verification"].get("match_status"),
                ctx["verification"].get("contradicted_claims"),
                ctx["verification"].get("manual_check"), ctx.get("_grounding"),
            ),
            system="You output strict JSON only.",
            temperature=0.0,
            num_predict=220,
        )
        ctx["_t_audit"] = core.now_ms()

        if isinstance(audit_parsed, dict) and "pass" in audit_parsed:
            passed = bool(audit_parsed.get("pass"))
            gaps = [str(g)[:300] for g in _coerce_list(audit_parsed.get("gaps"))]
            audit = {
                "passed": passed,
                "gaps": gaps,
                "followup_queries": [str(q)[:300] for q in _coerce_list(audit_parsed.get("followup_queries"))],
            }
        else:
            # The auditor is unavailable. Treating silence as a pass would let an
            # unexamined assessment through; treating it as a fail would loop
            # forever. The deterministic gates already ran, so this is recorded
            # as a not-run and the loop stops on attempt 1.
            audit = _fallback_audit(["The automated auditor did not return a usable result."])
            gaps = audit["gaps"]
            ctx["_audit_unanalyzable"] = True
            break

        if passed or attempt >= MAX_AUDIT_ATTEMPTS - 1:
            break

    ctx["audit"] = {**audit, "attempts": attempts}
    if ctx.pop("_audit_unanalyzable", None):
        ctx["audit"]["passed"] = False
        ctx["audit"]["gaps"] = list(ctx["audit"].get("gaps") or []) + [
            "The self-audit step could not run. The deterministic citation and grounding gates did run, "
            "so the citation contract and the level cap were still enforced, but no model critic reviewed the wording."
        ]

    # ---- A4: Action Planner ----------------------------------------------
    t_before_action = core.now_ms()
    ck_parsed, _ = llm.json_call(
        prompts.a4(
            ctx["risk"], ctx["evidence"], ctx["verification"].get("match_status"),
            ctx["verification"].get("manual_check"), ctx["extraction"]["claims"],
            ctx["verification"].get("domain"),
        ),
        system="You output strict JSON only.",
        temperature=0.3,
        num_predict=460,
    )
    ctx["_t_action"] = core.now_ms()
    ctx["checklist"] = ck_parsed if isinstance(ck_parsed, dict) else _fallback_checklist(ctx["verification"])

    # ---- GROUNDING GATE (final, deterministic) ---------------------------
    ctx = core.grounding_gate(ctx)
    ctx["_t_ground"] = core.now_ms()

    # ---- attach a profile if we happen to know the college ---------------
    profile_attached = None
    institution = (ctx.get("extraction") or {}).get("institution")
    if institution:
        built = core.build_profile(institution, ctx.get("domain"))
        if built["profile"].get("exists_in_snapshot"):
            profile_attached = built["profile"]

    return core.assemble_fraud_envelope(ctx, profile_attached)



def investigate(body: Dict[str, Any], entry: str = "webhook") -> Dict[str, Any]:
    """The whole pipeline, one request in, one envelope out.

    The run budget is armed here, before anything that can block, so that every
    model call below derives its own timeout from what is left. A model that
    wedges or rambles can then no longer stretch a five-minute presentation
    slot: once the budget is spent the calls return immediately and the
    deterministic fallbacks finish the report through the same fail-closed
    path the gates already enforce.
    """
    get_ollama().start_budget(BUDGET_SECONDS)

    ctx = core.sanitise_and_redact(body, entry)
    ctx = core.intent_gate(ctx)

    if ctx["_intent"] == "other":
        return core.assemble_other_envelope(core.fixed_reply(ctx))

    # The plan is built before any retrieval so the UI can show it first, and it
    # only lists steps that will actually run.
    ctx = core.build_retrieval_plan(ctx)

    # ---- the two live calls, in parallel ---------------------------------
    domain = ctx.get("domain")
    html = None
    rdap_raw: dict = {}
    site_meta: dict = {}
    rdap_meta: dict = {}
    if domain:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_site = pool.submit(tools.fetch_website, domain)
            f_rdap = pool.submit(tools.fetch_rdap, domain)
            html, site_meta = f_site.result()
            rdap_raw, rdap_meta = f_rdap.result()

    site = core.read_site(html or "", domain)
    rdap = tools.normalise_rdap(rdap_raw, domain)
    ctx["retrieval"] = {"site": site, "rdap": rdap, "site_meta": site_meta, "rdap_meta": rdap_meta}
    ctx["_t_retrieval"] = core.now_ms()

    if ctx["_intent"] == "profile":
        return run_profile_branch(ctx)
    return run_fraud_branch(ctx)

def chat(message: str, case_id: Optional[str] = None) -> Dict[str, Any]:
    """Same pipeline, chat-shaped input. One front door, two shapes."""
    return investigate(
        {"text": message, "case_id": case_id or "", "input_type": "text", "source": "chat"},
        entry="chat",
    )
