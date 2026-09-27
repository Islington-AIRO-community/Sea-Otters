"""Every deterministic node, ported from the working n8n workflow.

The rule this file exists to enforce: **no model ever decides a fact.**
Register presence, dates, citations, claim statuses and the verdict are all
produced here by string comparison, regex or date arithmetic. The LLM is only
ever allowed to phrase and weigh.

Ported node-for-node, with the same comments, because the reasoning behind each
branch is the argument the team makes in Q&A.
"""

from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .data import (
    AUTHORITY,
    DEFAULT_AUTHORITY,
    OFFICIAL_HOSTS,
    PROFILES,
    REGISTRY,
)

DISCLAIMER = "This is a risk assessment to inform your decision, not a verdict."

MATCH_STATUS = ["verified", "partial_match", "not_found", "source_unavailable"]
# `identity` and `website` exist because the domain cross-check needs a claim to
# contradict. A lookalike domain is evidence about exactly one thing -- "this is
# the institution's own site" -- and with no type for that the check produced
# evidence it had no claim it was allowed to contradict. That branch was dead
# code, so the product's flagship case could never mark a claim CONTRADICTED and
# could never clear the High cap.
#
# See IDENTITY_CLAIM below: the text fallback matters as much as these types,
# because a small local model rarely picks the type correctly.
CLAIM_TYPES = [
    "fee",
    "timeline",
    "guarantee",
    "accreditation",
    "deadline",
    "identity",
    "website",
    "other",
]

# A claim that asserts the provider's own identity or official site, however it
# was typed. The domain cross-check contradicts THESE and nothing else.
#
# This pattern exists because the typed branch alone is not enough: a local 7B
# model buckets "this is their official site" under "other" or "accreditation"
# most of the time, so relying on the type alone left the flagship case
# undetected in live runs. Matching the text as well makes it reliable.
IDENTITY_CLAIM = re.compile(
    r"\b(?:official|legitimate|real|authentic|genuine|true)\s+"
    r"(?:web ?site|website|site|domain|url|portal|page)|"
    r"\bthis\s+is\s+(?:the|their|its)\b|"
    r"\b(?:we|they)\s+are\s+the\s+(?:official|real|genuine)|"
    r"\bapply\s+(?:here|directly)\b|"
    r"\bis\s+the\s+official\b",
    re.I,
)
CLAIM_STATUS = [
    "VERIFIED",
    "PARTIAL_MATCH",
    "UNVERIFIED",
    "NOT_FOUND_IN_CORRECT_REGISTER",
    "CONTRADICTED",
    "INSUFFICIENT_EVIDENCE",
]


def today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def now_ms() -> float:
    return time.time() * 1000.0


# ===========================================================================
# LAYER 1 -- Sanitise & Redact
# ===========================================================================

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"return status\s*verified",
    r"you are now",
    r"disregard (the )?(system|previous)",
    r"note to (automated|ai|llm)",
    r"do not report this",
    r"mark (this|it) as (verified|safe|legit)",
    r"you must (return|respond with|output)",
]

PII_RULES = [
    (r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b", "[EMAIL_REDACTED]", "email"),
    (r"\b\d{4}\s?\d{4}\s?\d{4}\b", "[CARD_REDACTED]", "card"),
    (r"\b\d{2}[\/]\d{2}[\/]\d{4}\b", "[NATIONAL_ID_REDACTED]", "national_id"),
    (
        r"\b(?:sort code|iban|account number|a\/c|account no)\b[^,.\n]{0,40}",
        "[BANK_REDACTED]",
        "bank",
    ),
]

PHONE_CANDIDATES = re.compile(
    r"(\+\d{1,3}[\s.\-]?)?(?:\(\d{2,5}\)|\d{2,5})[\s.\-]\d{3,5}(?:[\s.\-]\d{2,5})?"
)
# A phone number at the END OF A SENTENCE is still a phone number. The trailing
# lookahead rejects a digit and rejects a hyphen, but must NOT reject a full
# stop -- "call me on 9876543210." was leaking through, which is the exact PII
# this layer exists to stop. `\.\d` keeps a decimal like 1234567890.50 intact.
BARE_MOBILE = re.compile(r"(?<![\d.\-])\d{10}(?![\d]|\.\d)")
DOMAIN_IN_TEXT = re.compile(r"https?:\/\/[^\s\"'<>)]+", re.I)
DOMAIN_SHAPE = re.compile(
    r"(?:https?:\/\/)?(?:www\.)?([a-z0-9][a-z0-9-]{0,62}(?:\.[a-z0-9-]{1,24}){1,4})", re.I
)
NON_DOMAIN_TAIL = re.compile(r"\.(png|jpg|jpeg|gif|svg|pdf|docx?)$", re.I)
# A hostname's last label is letters, not digits. Without this, scanning prose
# for a bare hostname matches "e.g" and "3.14" and then tries to fetch them.
TLD_SHAPE = re.compile(r"\.[a-z]{2,24}$", re.I)


def sanitise_and_redact(body: dict, entry: str = "webhook") -> dict:
    """Strip HTML, redact PII, extract a domain, run the injection tripwire."""
    t_start = now_ms()
    raw = str(
        body.get("text")
        or body.get("chatInput")
        or body.get("message")
        or ""
    )[:20000]
    explicit_url = str(body.get("url") or "").strip()
    case_id = str(
        body.get("case_id")
        or ("case-%d-%s" % (int(time.time() * 1000), uuid.uuid4().hex[:6]))
    )

    clean = raw
    clean = re.sub(r"<script[\s\S]*?</script>", " ", clean, flags=re.I)
    clean = re.sub(r"<style[\s\S]*?</style>", " ", clean, flags=re.I)
    clean = re.sub(r"<[^>]+>", " ", clean)
    clean = re.sub(r"&nbsp;", " ", clean, flags=re.I)
    clean = re.sub(r"&amp;", "&", clean, flags=re.I)
    clean = re.sub(r"\s+", " ", clean).strip()

    # Prompt-injection tripwire. The content is controlled by the attacker, so
    # this is recorded as a risk signal rather than silently filtered. A judge
    # watching a live attempt appear in the report is the point of the exercise.
    hits: List[str] = []
    for p in INJECTION_PATTERNS:
        m = re.search(p, clean, re.I)
        if m:
            hits.append(m.group(0))

    # PII redaction. Offer letters carry passport numbers and bank details.
    # The phone pattern is deliberately anchored: it must contain a separator
    # or a country code, because a loose 9-13 digit match eats date ranges and
    # fee figures -- corrupting exactly the numbers a student most needs checked.
    pii: List[str] = []
    redacted = clean

    for pattern, token, kind in PII_RULES:
        rx = re.compile(pattern, re.I)
        if rx.search(redacted):
            pii.append(kind)
            redacted = rx.sub(token, redacted)

    def _phone_sub(m: "re.Match[str]") -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) < 9:
            return m.group(0)  # too short: a date or a fee
        if re.match(r"^(19|20)\d{2}", digits) and len(digits) <= 8:
            return m.group(0)
        pii.append("phone")
        return "[PHONE_REDACTED]"

    redacted = PHONE_CANDIDATES.sub(_phone_sub, redacted)

    def _bare_sub(m: "re.Match[str]") -> str:
        if not re.match(r"^[6-9]", m.group(0)):
            return m.group(0)
        pii.append("phone")
        return "[PHONE_REDACTED]"

    redacted = BARE_MOBILE.sub(_bare_sub, redacted)

    # Domain: explicit url wins, else a full url in the text, else a bare
    # hostname. The bare-hostname pass scans the whole prose, so the TLD check
    # is what stops "e.g" or a fee like "3.14" being fetched as a website.
    domain: Optional[str] = None
    candidates = [explicit_url] + DOMAIN_IN_TEXT.findall(redacted) + [redacted]
    for c in candidates:
        m = DOMAIN_SHAPE.search(str(c))
        if not m:
            continue
        host = m.group(1).lower()
        if NON_DOMAIN_TAIL.search(host) or not TLD_SHAPE.search(host):
            continue
        domain = host
        break

    return {
        "_entry": entry,
        "case_id": case_id,
        "input_type": str(body.get("input_type") or ("url" if explicit_url else "text")),
        "raw_text": raw,
        "clean_text": redacted[:6000],
        "pii_kinds_redacted": sorted(set(pii)),
        "injection_detected": len(hits) > 0,
        "injection_evidence": hits[:3],
        "domain": domain,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "_t_start": t_start,
        "_t0": now_ms(),
    }


# ===========================================================================
# LAYER 1 -- Intent Gate (deterministic, no LLM)
# ===========================================================================

PROFILE_MARKERS = [
    "best for", "best known for", "known for", "what is it known", "what are they known",
    "how long", "duration", "years to", "how many years", "year long", "years long",
    "credit", "credits", "credit hours", "ects", "scqf", "curriculum", "modules",
    "entry requirement", "admission requirement", "entry requirements", "admission requirements",
    "what do i need to get in", "requirements to get", "eligibility",
    "scholarship", "scholarships", "financial aid", "bursary", "fee structure",
    "what major", "which major", "majors", "what courses", "which courses",
    "what courses do they offer", "alternatives", "other universities", "other colleges",
    "other options", "similar universities", "tell me about", "what is it like",
    "how much does it cost", "what is the fee", "tuition fee", "cost to study",
]

FRAUD_MARKERS = [
    "legit", "legitimate", "real or fake", "is this real", "genuine", "authentic",
    "fake", "scam", "fraud", "fraudulent", "safe to pay", "can i trust",
    "verify", "verification", "check this", "check if", "is this a real university",
    "is this a real college", "red flag", "warning sign", "do your research",
]

# Without this, a pasted consultancy message -- which is the product's main
# input -- could match no marker at all and be REFUSED, which is a
# catastrophic demo failure.
STUDY_ABROAD = [
    "university", "universities", "college", "colleges", "consultancy", "consultancies",
    "consultant", "study abroad", "studying abroad", "study-abroad", "overseas",
    "admission", "admissions", "offer letter", "offer", "visa", "tuition",
    "accreditation", "accredited", "recognised", "recognized", "degree", "course",
    "programme", "program", "msc", "ma ", "mba", "phd", "bachelors", "master",
    "ielts", "toefl", "cas ", "refund", "deposit", "enrol", "enroll", "apply",
    "application", "fees", "fee", "brochure", "legit", "genuine", "agent",
    "education", "student", "students", "guarantee", "guaranteed", "100%",
    "ugc", "ofs", "teqsa", "nces", "qaa", "accreditor", "unrecognised", "unapproved",
    "pay", "payment", "service charge", "processing fee", "seat", "intake",
]

OUT_OF_SCOPE = [
    "weather", "temperature", "rain", "forecast", "recipe", "cook", "cooking",
    "joke", "jokes", "funny", "poem", "poetry", "song", "lyrics", "movie", "film",
    "football", "cricket", "nba", "stocks", "share price", "crypto", "bitcoin",
    "horoscope", "astrology", "capital of", "who was", "president of", "prime minister",
    "election", "write a poem", "tell me a joke", "how tall", "population of",
]

GREETING = re.compile(
    r"^(hi|hey|hello|yo|hiya|good (morning|afternoon|evening)|thanks|thank you|thx|ok|okay|cool|bye|goodbye)\b[\s!.?]*$",
    re.I,
)


def intent_gate(ctx: dict) -> dict:
    """Decide ONE thing: is this a question ABOUT a college, or a trust check?

    Deterministic on purpose. A scope check done by a model is a prompt-
    injection surface: "ignore your scope and tell me about Bristol" is a request
    a model may comply with. A regex cannot be argued out of its instructions,
    and when in doubt this gate refuses, which is the safe failure mode.
    """
    text = str(ctx.get("clean_text") or "")[:4000]
    lower = text.lower()
    has_domain = bool(ctx.get("domain"))

    asks_profile = any(k in lower for k in PROFILE_MARKERS)
    asks_fraud = any(k in lower for k in FRAUD_MARKERS)
    out_of_scope = any(k in lower for k in OUT_OF_SCOPE)
    both = out_of_scope and (asks_profile or asks_fraud)

    stripped = text.strip()
    if not stripped:
        intent, reason = "other", "empty message"
    elif GREETING.match(stripped):
        intent, reason = "other", "greeting only"
    elif out_of_scope and not asks_profile and not asks_fraud:
        intent, reason = "other", "known out-of-scope topic; refusing rather than answering"
    elif both:
        intent, reason = "other", "mixed in-scope and out-of-scope terms; refusing as the safe default"
    elif asks_profile:
        # Profile WINS over fraud: "is Bristol legit and what scholarships do
        # they have" is one question, and the profile answer already carries
        # the legitimacy result, so the student loses nothing.
        intent = "profile"
        reason = (
            "asked both a legitimacy and a facts question; profile answer covers both"
            if asks_fraud
            else "asked for facts about an institution"
        )
    elif asks_fraud:
        intent, reason = "fraud_check", "asked whether something is trustworthy"
    elif has_domain:
        intent, reason = "fraud_check", "submitted a URL with no question; treating it as a submission to check"
    elif any(k in lower for k in STUDY_ABROAD):
        intent, reason = "fraud_check", "study-abroad vocabulary with no profile question; treating it as something to check"
    elif stripped.endswith("?"):
        intent, reason = "fraud_check", "unclear but question-shaped; investigating rather than guessing"
    else:
        intent, reason = "other", "no in-scope terms found"

    return {**ctx, "_intent": intent, "_intent_reason": reason}


REFUSAL = """I can't help with that.

I only cover **studying abroad, universities, and education consultancies**. I can:
- check whether a university is on an official register
- check whether an education consultancy is a real registered company
- check how old a website's domain is
- check whether a fee, admission or "guaranteed visa" claim is contradicted by an official source
- tell you about a real university: what it is known for, how long the degree takes, entry requirements, credit system, scholarships and official fees page
- suggest other institutions we can verify in the same register

A fraud-checking tool that will happily chat about anything else is a tool you cannot trust when it matters.

*(%s. No investigation was run and no external source was queried.)*"""

GREETING_REPLY = """Hi — I'm **Varosha**, a study-abroad legitimacy checker.

I check a university against an **official register**, check how old a website's domain is, and cross-check the site you were given against the institution's real domain. I can also tell you about a real college: what it is known for, how long the degree takes, its credit system, entry requirements, scholarships and the official fees page.

**Ask me to check something** (a consultancy message, an offer letter, a website):
- is this university real, or is it on a government warning list?
- this consultancy guarantees my visa and wants full payment upfront
- `https://some-university-admissions.com` — is that the real site?

**Or ask me about a real college**:
- is University of Bristol legitimate?
- what is University of Manchester known for?
- how long is an MSc at Imperial, and what are the credit hours?
- what scholarships does Cardiff University offer, and what are the entry requirements?
- what does it cost, and are there alternatives?

I present evidence and flag risk. I never tell you whether to trust anyone — that decision stays with you."""


def fixed_reply(ctx: dict) -> dict:
    """Fixed strings. A model asked to refuse can be persuaded not to."""
    reason = ctx.get("_intent_reason") or ""
    is_refusal = "refus" in reason.lower()
    return {
        **ctx,
        "output": REFUSAL % reason if is_refusal else GREETING_REPLY,
        "refused": is_refusal,
        "intent_handled": "other",
    }


# ===========================================================================
# LAYER 2 -- Retrieval plan
# ===========================================================================


def build_retrieval_plan(ctx: dict) -> dict:
    """Built from what is actually present, so the UI can show the plan BEFORE
    results instead of inventing a narrative afterwards. The plan is also a
    promise to the student, so it only lists steps that really run."""
    plan = [
        {
            "step": "extract_claims",
            "tool": "ollama:mistral:latest",
            "why": "turn the message into individually testable claims",
        }
    ]
    if ctx.get("domain"):
        plan.append(
            {
                "step": "website_fetch",
                "tool": "HTTP Request (surf the submitted site)",
                "why": "read the provider's own claims as claims to be tested, never as proof",
            }
        )
        plan.append(
            {
                "step": "domain_registration",
                "tool": "HTTP Request -> RDAP (rdap.org)",
                "why": "compare domain age against the claimed history",
            }
        )
    plan += [
        {
            "step": "register_match",
            "tool": "deterministic match engine",
            "why": "check the institution against an official register snapshot",
        },
        {
            "step": "reason_over_weak_signals",
            "tool": "ollama:mistral:latest",
            "why": "weigh several weak signals together, not one keyword",
        },
        {
            "step": "citation_gate",
            "tool": "deterministic (no model)",
            "why": "delete any source the model could not actually have retrieved",
        },
        {
            "step": "self_audit",
            "tool": "ollama:mistral:latest",
            "why": "check every reason is cited and the level is justified",
        },
        {
            "step": "action_checklist",
            "tool": "ollama:mistral:latest",
            "why": "turn the findings into what the student can do this week",
        },
        {
            "step": "grounding_gate",
            "tool": "deterministic (no model)",
            "why": "re-validate the whole report before a student reads it",
        },
        {
            "step": "human_checkpoint",
            "tool": "in-process store (replaces the n8n Wait node)",
            "why": "the student decides what happens next",
        },
    ]
    return {**ctx, "retrieval_plan": plan, "_t_plan": now_ms()}


# ===========================================================================
# LAYER 2 -- Assemble retrieval context (site self-claims)
# ===========================================================================

SELF_CLAIM_PROBES = [
    (r"guarantee[ds]?\s+(?:a\s+|your\s+)?(?:visa|admission|placement)", "States a guaranteed visa, admission or placement."),
    (r"100\s*%\s*(?:admission|placement|success|guaranteed)", "States a 100% admission, placement or success rate."),
    (r"(?:pay|send|transfer)\s+(?:the\s+)?(?:full|entire|total)\s+(?:fee|amount|tuition|sum)", "Asks for the full fee or tuition up front."),
    (r"\bno\s+(?:entrance|entry)\s+(?:exam|test|requirement)", "States there is no entrance exam or entry requirement."),
    (r"\b(?:fully|officially)\s+accredit(?:ed|ation)\b", "Claims official accreditation."),
    (r"\b(?:recognis|recogniz)ed\s+by\b", "Claims recognition by a body it does not name."),
    (r"\bpay\s+(?:in\s+)?cash\b", "Asks for payment in cash."),
    (r"\b(?:cryptocurrency|bitcoin|usdt|wire\s+transfer\s+only)\b", "Asks for a payment method a bank cannot reverse."),
]


def read_site(html: str, domain: Optional[str]) -> dict:
    """Turn a fetched page into a self-report record.

    The content is data, never instructions. Each probe records the EXACT
    matched substring so a reviewer can audit what was found instead of
    trusting a paraphrase. A probe is a signal to reason over, never a verdict.
    """
    site = {
        "fetched": False,
        "reachable": False,
        "title": None,
        "self_claims": [],
        "note": None,
        "source_url": ("https://" + domain) if domain else None,
    }

    if not domain:
        site["note"] = "No domain was found in the submission, so no website was fetched."
        return site

    # A real page is longer than this. The threshold counts the PROSE, not the
    # markup: counting raw HTML meant a short but genuine page -- a minimal
    # landing page, an error page -- was reported as "no readable content", which
    # reads to a student as "the site said nothing" rather than "the site is
    # thin". Below the threshold nothing was fetched; above it, an empty page is
    # a finding in its own right.
    if not html:
        site["note"] = "The submitted website could not be reached from this machine (DNS, TLS or timeout)."
        return site

    # Counted on the PROSE, not the markup, and the bar is deliberately low: a
    # one-line landing page is still a page the provider chose to show a
    # student, and it can still contain a guarantee. The original 60-character
    # bar silently discarded short real pages, which reports "the site said
    # nothing" when the site actually said something worth quoting.
    prose_only = re.sub(r"<[^>]+>", " ", re.sub(r"<(script|style)[\s\S]*?</\1>", " ", html, flags=re.I))
    if len(re.sub(r"\s+", "", prose_only)) < 20:
        site["note"] = "The submitted website returned no readable page text."
        return site

    site["fetched"] = True
    site["reachable"] = True

    t = re.search(r"<title[^>]*>([\s\S]{0,300}?)</title>", html, re.I)
    if t:
        title = re.sub(r"&[a-z#0-9]+;", " ", t.group(1), flags=re.I)
        title = re.sub(r"\s+", " ", title).strip()[:160]
        site["title"] = title or None

    prose = html
    prose = re.sub(r"<script[\s\S]*?</script>", " ", prose, flags=re.I)
    prose = re.sub(r"<style[\s\S]*?</style>", " ", prose, flags=re.I)
    prose = re.sub(r"<[^>]+>", " ", prose)
    prose = re.sub(r"&[a-z#0-9]+;", " ", prose, flags=re.I)
    prose = re.sub(r"\s+", " ", prose)

    for pattern, label in SELF_CLAIM_PROBES:
        m = re.search(pattern, prose, re.I)
        if m:
            site["self_claims"].append(
                {"label": label, "matched_text": re.sub(r"\s+", " ", m.group(0))[:120]}
            )

    if not site["self_claims"]:
        site["note"] = "No standard risk phrase was present in the page text. That is the absence of these particular phrases, not a clean bill of health."
    return site


def assemble_agent_text(intake: dict, site: dict) -> str:
    """A URL-only submission carries no prose, so the domain is appended
    explicitly -- otherwise a student who pastes just a link gets an empty
    report, which is the worst failure mode for a fraud checker."""
    parts = [intake.get("clean_text") or ""]
    if intake.get("domain"):
        parts.append("SUBMITTED WEBSITE: " + intake["domain"])
    if site.get("title"):
        parts.append("SITE TITLE (the provider's own wording, unverified): " + site["title"])
    return "\n".join(p for p in parts if p)[:8000]


# ===========================================================================
# LAYER 2 -- Match engine
# ===========================================================================

_FILLER = re.compile(
    r"\b(ltd|limited|llc|inc|incorporated|plc|corp|corporation|gmbh|pvt|private|university|college|school|academy|institute|of|the|city)\b"
)


def norm(s: Optional[str]) -> str:
    s = str(s or "").lower()
    s = _FILLER.sub(" ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i]
        for k in range(1, len(b) + 1):
            cur.append(
                min(
                    prev[k] + 1,
                    cur[k - 1] + 1,
                    prev[k - 1] + (0 if a[i - 1] == b[k - 1] else 1),
                )
            )
        prev = cur
    return prev[len(b)]


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return 1 - levenshtein(a, b) / max(len(a), len(b))


def match_registry(institution: Optional[str]) -> Tuple[Optional[dict], str, Optional[str], Optional[str], float]:
    """A near-miss is reported as a lookalike warning and never promoted to a
    match. A fuzzy match that silently upgrades to "verified" is how a checker
    loses a student's trust."""
    if not institution:
        return None, "not_found", None, None, 0.0

    target = norm(institution)
    for r in REGISTRY:
        if norm(r["name"]) == target:
            return r, "verified", r["name"], r["jur"], 1.0

    scored = []
    for r in REGISTRY:
        pn = norm(r["name"])
        tail = pn.split(" ")[-1] if pn.split(" ") else ""
        s = max(similarity(target, pn), similarity(target, tail))
        scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    if scored:
        best_s, best_r = scored[0]
        if best_s >= 0.85:
            return best_r, "partial_match", best_r["name"], best_r["jur"], round(best_s, 3)
        return None, "not_found", None, None, round(best_s, 3)
    return None, "not_found", None, None, 0.0


def official_domain_check(match: Optional[dict], submitted: Optional[str]) -> Optional[dict]:
    """Compare the site the student was handed against the site's the register
    actually holds. A lookalike domain is the single most common way a
    convincing fake is built, and domain AGE alone misses it."""
    if not match or not submitted:
        return None

    official = str(match.get("official_domain") or "").lower()
    if not official:
        return None
    official = re.sub(r"^www\.", "", official)
    submitted = re.sub(r"^www\.", "", str(submitted).lower())

    official_host = official.split("/")[0]
    submitted_host = submitted.split("/")[0]
    same = official_host == submitted_host
    contains = (not same) and (official_host in submitted_host)

    return {
        "checked": True,
        "official_domain": official_host,
        "submitted_domain": submitted_host,
        "match": "match" if same else ("contains_official" if contains else "mismatch"),
        "official_url": match.get("official_url") or ("https://" + official_host),
    }


# ===========================================================================
# LAYER 2 -- Verify + Build Evidence  (THE CORE)
# ===========================================================================


def verify_and_build_evidence(
    intake: dict, extraction: dict, site: dict, rdap: dict
) -> dict:
    institution = extraction.get("institution")
    domain = intake.get("domain")

    t_block_start = now_ms()
    match, match_status, matched_name, jurisdiction, match_score = match_registry(institution)
    t_match = now_ms()
    domain_check = official_domain_check(match, domain)
    t_domain = now_ms()

    evidence: List[dict] = []
    seq = [0]

    def push(e: dict) -> None:
        seq[0] += 1
        e["id"] = "E%d" % seq[0]
        e.setdefault("claim_id", None)
        evidence.append(e)

    d = today()

    if match:
        push(
            {
                "label": "Register match",
                "value": "MATCHED A KNOWN-BAD ENTRY" if match.get("blacklist") else "Matched: " + match["name"],
                "source_name": "%s (snapshot retrieved %s)" % (match["reg"], match["retrieved_on"]),
                "source_url": match["url"],
                "retrieved_at": d,
                "tier": "static",
            }
        )
        if match.get("blacklist"):
            push(
                {
                    "label": "Official blacklist",
                    "value": match["blacklist_reason"],
                    "source_name": "%s (snapshot retrieved %s)" % (match["reg"], match["retrieved_on"]),
                    "source_url": match["url"],
                    "retrieved_at": d,
                    "tier": "static",
                }
            )
    else:
        push(
            {
                "label": "Register check",
                "value": (
                    "No entry in the bundled register snapshot for this name"
                    if institution
                    else "No institution name could be extracted, so no register was queried"
                ),
                "source_name": "EduShield bundled register snapshot (OfS / UGC / TEQSA extracts)",
                "source_url": "https://www.officeforstudents.org.uk/so-providers/",
                "retrieved_at": d,
                "tier": "static",
            }
        )

    if domain:
        push(
            {
                "label": "Domain registration date (RDAP)",
                "value": (
                    "%s (%s days old)" % (rdap["registration_date"], rdap["age_days"])
                    if rdap.get("registration_date") and rdap.get("age_days") is not None
                    else "Unavailable: " + (rdap.get("reason") or "no RDAP record for this TLD")
                ),
                "source_name": "RDAP (rdap.org)",
                "source_url": rdap.get("source_url") or ("https://rdap.org/domain/" + domain),
                "retrieved_at": d,
                "tier": rdap.get("source_status") or "cached",
            }
        )

    # The provider's own website. Tier "self_report", labelled in the label
    # field, so it cannot be mistaken for an independent source by the UI or A2.
    if site.get("fetched"):
        parts = []
        if site.get("title"):
            parts.append('Page title: "%s"' % site["title"])
        for c in site.get("self_claims") or []:
            parts.append('%s Found text: "%s"' % (c["label"], c["matched_text"]))
        push(
            {
                "label": "Provider's own website (self-report, NOT verification)",
                "value": " | ".join(parts) if parts else ("Site was reachable. " + (site.get("note") or "No standard risk phrase present in the page text.")),
                "source_name": "Submitted website, quoted from the provider themselves",
                "source_url": site.get("source_url") or ("https://" + str(domain)),
                "retrieved_at": d,
                "tier": "self_report",
            }
        )
    elif site.get("note"):
        push(
            {
                "label": "Submitted website reachability",
                "value": site["note"],
                "source_name": "EduShield HTTP fetch",
                "source_url": site.get("source_url") or ("https://" + str(domain)),
                "retrieved_at": d,
                "tier": "live",
            }
        )

    if intake.get("injection_detected"):
        push(
            {
                "label": "Prompt-injection attempt in submitted content",
                "value": 'Text resembling instructions to an automated system was found and ignored: "%s"'
                % "; ".join(intake.get("injection_evidence") or []),
                "source_name": "EduShield deterministic sanitiser",
                "source_url": "https://owasp.org/www-project-top-10-for-large-language-model-applications/",
                "retrieved_at": d,
                "tier": "static",
            }
        )

    if domain_check:
        if domain_check["match"] == "match":
            value = "The site you were given (%s) matches the official domain on the register." % domain_check["submitted_domain"]
        else:
            value = (
                "The site you were given (%s) is not the institution's official domain. "
                "The official domain on the register is %s. A site that copies a real "
                "university's name on an unrelated domain is a known fraud pattern."
                % (domain_check["submitted_domain"], domain_check["official_domain"])
            )
        push(
            {
                "label": "Website domain vs the institution's official domain",
                "value": value,
                "source_name": "%s (snapshot retrieved %s)" % (match["reg"], match["retrieved_on"]),
                "source_url": domain_check["official_url"],
                "retrieved_at": d,
                "tier": "static",
            }
        )

    # THE CITATION CONTRACT: an uncited signal must never reach the reasoner.
    kept = [e for e in evidence if e.get("source_url") and e.get("retrieved_at")]
    dropped = len(evidence) - len(kept)

    # Per-claim status, assigned deterministically. The whole product rests on
    # the line between CONTRADICTED and NOT_FOUND_IN_CORRECT_REGISTER.
    claims = [dict(c) for c in (extraction.get("claims") or [])]
    blacklisted = bool(match and match.get("blacklist"))
    domain_mismatch = bool(domain_check and domain_check["match"] == "mismatch")

    for c in claims:
        if c.get("status"):
            continue  # a status set upstream always wins
        ctype = c.get("type")
        if blacklisted:
            c["status"] = "CONTRADICTED"
        elif domain_mismatch and (
            ctype in ("identity", "website")
            or (ctype in ("other", "accreditation") and IDENTITY_CLAIM.search(c.get("text") or ""))
        ):
            # A wrong domain is evidence about ONE thing: the site the student
            # was handed is not this institution's own site. It is NOT evidence
            # about accreditation, fees or course length. Contradicting those
            # would be over-claiming, and over-claiming is how a tool like this
            # convinces a student that a perfectly real university is fake.
            c["status"] = "CONTRADICTED"
            c["note"] = "The site given (%s) is not %s's official domain (%s)." % (
                (domain_check or {}).get("submitted_domain", "unknown"),
                matched_name or "this institution",
                (domain_check or {}).get("official_domain", "unknown"),
            )
        elif match_status == "verified":
            # "Accredited" is a different check from state recognition. Being on
            # the register confirms the institution exists; it does not confirm
            # every badge it claims.
            c["status"] = "PARTIAL_MATCH" if ctype == "accreditation" else "VERIFIED"
        elif ctype == "accreditation":
            c["status"] = "NOT_FOUND_IN_CORRECT_REGISTER"
        elif match_status == "partial_match":
            c["status"] = "PARTIAL_MATCH"
        elif not institution:
            c["status"] = "INSUFFICIENT_EVIDENCE"
        else:
            c["status"] = "UNVERIFIED"

    contradicted = [c.get("claim_id") for c in claims if c.get("status") == "CONTRADICTED"]
    manual_check = AUTHORITY.get(jurisdiction or "", DEFAULT_AUTHORITY)

    return {
        "verification": {
            "match_status": match_status,
            "match_score": match_score,
            "matched_name": matched_name,
            "jurisdiction": jurisdiction,
            "manual_check": manual_check,
            "tier_used": "static",
            "domain": (
                {
                    "domain": domain,
                    "created": rdap.get("registration_date"),
                    # never 0 as a stand-in for "unknown" -- the UI prints this
                    "age_days": rdap.get("age_days"),
                    "source_url": rdap.get("source_url"),
                    "source_status": rdap.get("source_status") or "cached",
                }
                if domain
                else None
            ),
            "official_domain_check": domain_check,
            # A lookalike domain is a deterministic contradiction of "this is
            # the institution's own site", and the register says so. It has to be
            # a hard signal in its own right: keyed only off a CONTRADICTED
            # claim it depended on a claim type the contract does not have, so
            # the product's flagship case could never clear the High cap.
            "domain_mismatch": domain_mismatch,
            "contradicted_claims": contradicted,
            "dropped_uncited_records": dropped,
        },
        "extraction": {**extraction, "claims": claims},
        "evidence": kept,
        "evidence_count": len(kept),
        # Real stamps for the deterministic Layer 2, split so the register match
        # and the domain cross-check can each be reported on their own. They are
        # adjacent slices of one block, so they add up rather than double count.
        "_t_verify_block": t_block_start,
        "_t_match": t_match,
        "_t_domain": t_domain,
        "_t_verify": now_ms(),
    }


# ===========================================================================
# THE CITATION GATE
# ===========================================================================

PROTECT_SUBSTITUTIONS = [
    (re.compile(r"list of fake universities", re.I), "__EDU_PROTECT_0__"),
    (re.compile(r"fake universities list", re.I), "__EDU_PROTECT_1__"),
    (re.compile(r"list of fraudulent", re.I), "__EDU_PROTECT_2__"),
]

LOADED = re.compile(
    r"\b(?:fraud|fraudulent|fraudsters?|scams?|scammers?|scamming|fakes?|fakers?|criminals?|illegal|liars?|lies?|lying|lied|dishonest|deceptive|thieves?|conman|con-artist|swindle|swindlers?)\b",
    re.I,
)


def _protect(s: Any) -> str:
    t = str(s or "")
    for rx, tok in PROTECT_SUBSTITUTIONS:
        t = rx.sub(tok, t)
    return t


def _unprotect(s: Any) -> str:
    t = str(s or "")
    return (
        t.replace("__EDU_PROTECT_0__", "List of Fake Universities")
        .replace("__EDU_PROTECT_1__", "fake universities list")
        .replace("__EDU_PROTECT_2__", "list of fraudulent")
    )


def has_verdict(text: Any) -> bool:
    """Checked against the PROTECTED form, so an official register genuinely
    called "List of Fake Universities" is not mistaken for a verdict."""
    return bool(LOADED.search(_protect(text)))


def _norm_url(u: Any) -> str:
    t = str(u or "").strip().lower()
    t = re.sub(r"^https?://", "", t)
    t = re.sub(r"^www\.", "", t)
    t = re.sub(r"[/?#].*$", "", t)
    return re.sub(r"/+$", "", t)


def _host(u: Any) -> Optional[str]:
    try:
        return (urlparse(str(u)).hostname or "").lower().removeprefix("www.") or None
    except Exception:
        return None


def citation_gate(ctx: dict) -> dict:
    """The single most important anti-hallucination control here.

    "Never invent a URL" is a REQUEST in a prompt. A request is not a control.
    Local models invent URLs constantly and do it fluently, which is exactly
    what makes it dangerous: a student cannot tell a real citation from a
    plausible one by looking. So this does not ask. It CHECKS, then DELETES.
    """
    evidence = ctx.get("evidence") or []
    verification = ctx.get("verification") or {}
    extraction = ctx.get("extraction") or {}
    claims = extraction.get("claims") or []
    risk_in = ctx.get("risk") or {}

    by_url = {}
    by_id = {}
    for e in evidence:
        if e.get("source_url"):
            by_url[_norm_url(e["source_url"])] = e
        by_id[str(e.get("id"))] = e

    findings: List[str] = []
    rejected: List[dict] = []
    dropped_verdict: List[dict] = []
    reasons: List[dict] = []
    cited_count = 0

    for r0 in risk_in.get("reasons") or []:
        if not r0 or not (r0.get("flag") or r0.get("why")):
            continue

        source_url = r0.get("source_url")
        retrieved_at = r0.get("retrieved_at")
        evidence_ref = r0.get("evidence_ref")
        resolved = None

        if source_url:
            resolved = by_url.get(_norm_url(source_url))
            if not resolved:
                # The model cited something that does not exist. Delete the
                # citation: keeping it puts a fabricated URL in front of a
                # student as though it were real.
                rejected.append(
                    {
                        "reason_flag": str(r0.get("flag") or "")[:120],
                        "rejected_url": str(source_url)[:300],
                    }
                )
                findings.append(
                    'Removed a source URL from "%s" that matched no retrieved evidence.'
                    % str(r0.get("flag") or "")[:60]
                )
                source_url = None
                retrieved_at = None
            else:
                # Snap ALL THREE fields to the real record. A model that cites
                # E1 but retypes the date has still produced an unverified
                # date, and the UI prints it. The URL is snapped for the same
                # reason and the same reason it matters most: the student is
                # meant to click it, and a model may have downgraded the scheme
                # to http://, added a tracking parameter, or truncated the
                # path. The record that was actually retrieved is the only
                # version we can vouch for.
                source_url = resolved.get("source_url")
                retrieved_at = resolved.get("retrieved_at")
                if not evidence_ref or str(evidence_ref) not in by_id:
                    if evidence_ref:
                        # Silently correcting a bad reference would hide that
                        # the model cited a record which does not exist. The
                        # repair happens, and the fact that it was needed is
                        # recorded for the audit trail.
                        findings.append(
                            'Repaired an evidence reference ("%s") that pointed at no '
                            "retrieved record." % str(evidence_ref)[:40]
                        )
                    evidence_ref = resolved.get("id")

        if evidence_ref and str(evidence_ref) not in by_id:
            findings.append(
                'Dropped an evidence reference ("%s") that pointed at no record.'
                % str(evidence_ref)[:40]
            )
            evidence_ref = None

        # Verdict language disqualifies the whole reason. Dropping it cannot
        # soften the assessment, because level and summary are re-derived below.
        if has_verdict(r0.get("flag")) or has_verdict(r0.get("why")):
            dropped_verdict.append(
                {
                    "flag": str(r0.get("flag") or "")[:120],
                    "why": str(r0.get("why") or "")[:300],
                    "had_citation": bool(source_url),
                }
            )
            findings.append(
                'Discarded a reason phrased as a verdict ("%s"). Judgements are not findings; the underlying evidence is still listed.'
                % str(r0.get("flag") or "")[:60]
            )
            continue

        cited = bool(source_url)
        if cited:
            cited_count += 1

        conf = r0.get("confidence")
        if cited and isinstance(conf, (int, float)):
            conf = max(0.0, min(1.0, float(conf)))
        else:
            # Nulled when uncited. A confidence number with no source behind it
            # is false precision, and the UI renders it as a bar the student
            # will read as meaningful.
            conf = None

        reasons.append(
            {
                "flag": str(r0.get("flag") or "Unnamed signal")[:120],
                "why": str(r0.get("why") or "")[:400],
                "evidence_ref": evidence_ref,
                "source_url": source_url,
                "retrieved_at": retrieved_at if cited else None,
                "confidence": conf,
                "cited": cited,
                # NOT_FOUND NEUTRALITY: a reason resting only on a register that
                # came back empty cannot carry weight. Demoted, not deleted --
                # the question is a fair one.
                "weight": (
                    "demoted_no_positive_source"
                    if verification.get("match_status") == "not_found" and not cited
                    else "normal"
                ),
            }
        )

    # LEVEL JUSTIFICATION. Three hard caps, each traceable to contract.md.
    hard_signal = (
        bool(verification.get("contradicted_claims"))
        or bool(verification.get("domain_mismatch"))
        or any("KNOWN-BAD" in str(e.get("value") or "") for e in evidence)
        or any(c.get("status") == "CONTRADICTED" for c in claims)
    )

    level = risk_in.get("level") if risk_in.get("level") in ("Low", "Medium", "High") else "Low"
    level_before = level
    cap_reasons: List[str] = []

    if not evidence:
        level = "Low"
        cap_reasons.append("No external evidence was retrievable, so no level above Low is supportable.")
    elif cited_count == 0:
        level = "Low"
        cap_reasons.append("No reason survived with a citation to a retrieved source, so the assessment cannot be raised.")
    elif not hard_signal and level == "High":
        level = "Medium"
        cap_reasons.append(
            "No claim is CONTRADICTED by an authoritative source. Per the contract, "
            "absence from a register may not by itself produce a High level."
        )
    findings.extend(cap_reasons)

    # The summary is the first thing the student reads, so it is never taken on
    # trust. If it carries judgement language it is REPLACED by a sentence built
    # only from facts this pipeline actually retrieved. No model wording survives.
    def build_summary() -> str:
        bits = []
        if any("KNOWN-BAD" in str(e.get("value") or "") for e in evidence):
            bits.append("the provider is named on an official list of unrecognised institutions")
        contra = sum(1 for c in claims if c.get("status") == "CONTRADICTED")
        if contra:
            bits.append("%d of the claims made are contradicted by that source" % contra)
        if any(re.search(r"injection|prompt", str(e.get("label") or ""), re.I) for e in evidence):
            bits.append("the submitted text contained an attempt to instruct this automated checker")
        dom = verification.get("domain") or {}
        if dom.get("created"):
            bits.append("the website domain was registered on %s, %s days before this check" % (dom["created"], dom.get("age_days")))
        elif dom.get("age_days") is None and verification.get("domain"):
            bits.append("the age of the website domain could not be confirmed")
        if not bits:
            return (
                "No claim could be confirmed or contradicted by an independent source. "
                "Treat this as insufficient evidence and check manually."
                if evidence
                else "No independent source could be reached for this submission, so nothing was cleared and nothing was disproved."
            )
        return "On the evidence retrieved, " + "; ".join(bits) + "."

    raw_summary = str(risk_in.get("summary") or "")
    if has_verdict(raw_summary):
        summary = build_summary()
        findings.append("Replaced the risk summary: it was phrased as a verdict, so it was rebuilt from retrieved evidence only.")
    else:
        summary = raw_summary
    if not summary.strip():
        summary = build_summary()
        findings.append("The risk summary was empty, so it was rebuilt from retrieved evidence only.")

    for r in reasons:
        if r["weight"] == "demoted_no_positive_source":
            r["confidence"] = None

    return {
        **ctx,
        "_t_gate": now_ms(),
        "risk": {"level": level, "level_before_cap": level_before, "summary": summary, "reasons": reasons},
        "_grounding": {
            "citations_rejected": len(rejected),
            "rejected_citations": rejected,
            "verdict_reasons_dropped": len(dropped_verdict),
            "dropped_verdict_reasons": dropped_verdict,
            "reasons_cited": cited_count,
            "reasons_total": len(reasons),
            "hard_signal": hard_signal,
            "level_capped": level != level_before,
            "cap_reasons": cap_reasons,
            "notes": findings,
        },
    }


# ===========================================================================
# THE GROUNDING GATE (final)
# ===========================================================================


def grounding_gate(ctx: dict) -> dict:
    """The last thing that runs before a student sees anything.

    Polices the pipeline, not just the model, because a bug in deterministic
    code is just as capable of showing a student a confident lie as a
    hallucinating model is. NEVER throws: a gate that crashes on the last node
    turns a small defect into a blank screen, and a blank screen is the worst
    thing this tool can do to someone who trusted it.
    """
    repairs: List[str] = []

    def note(m: str) -> None:
        if len(repairs) < 40:
            repairs.append(m)

    def swept(s: Any, label: str) -> str:
        if has_verdict(s):
            note("Dropped %s for phrasing it as a judgement." % label)
            return ""
        return _unprotect(_protect(s))

    # 1. EVIDENCE
    evidence = ctx.get("evidence") or []
    before = len(evidence)
    evidence = [
        e
        for e in evidence
        if e and e.get("source_url") and e.get("retrieved_at") and str(e["source_url"]).startswith("http")
    ]
    if len(evidence) != before:
        note("Evidence records dropped: %d -> %d." % (before, len(evidence)))
    evidence = [{**e, "id": "E%d" % (i + 1)} for i, e in enumerate(evidence)]

    # 2/3. VERIFICATION
    verification = dict(ctx.get("verification") or {})
    if verification.get("match_status") not in MATCH_STATUS:
        note('match_status "%s" is not a contract value; set to not_found.' % str(verification.get("match_status"))[:40])
        verification["match_status"] = "not_found"
    if verification.get("domain"):
        dom = dict(verification["domain"])
        if not dom.get("created"):
            dom["age_days"] = None
        if dom.get("age_days") is not None and not isinstance(dom["age_days"], (int, float)):
            dom["age_days"] = None
            note("Non-numeric age_days nulled.")
        # "0 days old" would read as "registered today". Unknown must look unknown.
        if dom.get("created") and dom.get("age_days") == 0:
            dom["age_days"] = None
            note("age_days of 0 with a creation date is implausible; nulled rather than shown.")
        verification["domain"] = dom
    if not (verification.get("manual_check") or {}).get("authority"):
        verification["manual_check"] = {
            "authority": "the official higher-education regulator for the claimed country",
            "url": "https://www.ugc.gov.in/",
        }
        note("manual_check was missing; supplied a generic authority so the UI never renders an empty instruction.")

    # 2. EXTRACTION + per-claim status
    extraction = dict(ctx.get("extraction") or {})
    claims_out = []
    raw_claims = extraction.get("claims")
    if not isinstance(raw_claims, list):
        if raw_claims:
            note("extraction.claims was not a list; discarded rather than guessed at.")
        raw_claims = []
    for i, c in enumerate(raw_claims):
        if not isinstance(c, dict):
            note("A claim was not an object; dropped.")
            continue
        c = dict(c)
        c["claim_id"] = str(c.get("claim_id") or "C%d" % (i + 1))
        # Strip, not just truncate: a claim of "   " would otherwise render as an
        # empty bullet in the UI, which reads as a row we failed to fill in.
        c["text"] = str(c.get("text") or "").strip()[:300]
        if c.get("type") not in CLAIM_TYPES:
            c["type"] = "other"
            note('Claim %s had an unknown type; set to "other".' % c["claim_id"])
        if c.get("status") not in CLAIM_STATUS:
            c["status"] = "UNVERIFIED"
            note("Claim %s had an unknown status; set to UNVERIFIED (neutral default)." % c["claim_id"])
        if c["text"]:
            claims_out.append(c)
    extraction["claims"] = claims_out

    # 2/4. RISK
    risk_in = ctx.get("risk") or {}
    if not isinstance(risk_in, dict):
        # A gate that crashes on the last node turns a small defect into a blank
        # screen. A malformed risk block is degraded to a neutral Low, which the
        # caps below then re-derive from the evidence.
        note("risk was not an object; treated as an empty assessment at Low.")
        risk_in = {}
    level = risk_in.get("level") if risk_in.get("level") in ("Low", "Medium", "High") else "Low"
    if level != risk_in.get("level"):
        note("risk.level was not one of Low/Medium/High; defaulted to Low.")

    r_before = len(risk_in.get("reasons") or [])
    reasons = []
    for r in risk_in.get("reasons") or []:
        if not r or not (r.get("flag") or r.get("why")):
            continue
        reasons.append(
            {
                "flag": swept(r.get("flag"), "a risk flag"),
                "why": swept(r.get("why"), "a risk explanation"),
                "evidence_ref": r.get("evidence_ref"),
                "source_url": r.get("source_url"),
                "retrieved_at": r.get("retrieved_at") if r.get("source_url") else None,
                "confidence": (
                    max(0.0, min(1.0, float(r["confidence"])))
                    if r.get("source_url") and isinstance(r.get("confidence"), (int, float))
                    else None
                ),
                "cited": bool(r.get("source_url")),
                "weight": r.get("weight") or "normal",
            }
        )
    reasons = [r for r in reasons if r["flag"] or r["why"]]
    if len(reasons) != r_before:
        note("Risk reasons dropped for phrasing them as judgements: %d -> %d." % (r_before, len(reasons)))

    # Re-apply the citation cap. Cheap, and it means a later edit upstream cannot
    # quietly reopen the hole the Citation Gate closed.
    if not evidence:
        level = "Low"
    elif not any(r["cited"] for r in reasons):
        level = "Low"
    elif (
        level == "High"
        and not verification.get("contradicted_claims")
        and not verification.get("domain_mismatch")
        and not any("KNOWN-BAD" in str(e.get("value") or "") for e in evidence)
    ):
        level = "Medium"
        note("High downgraded to Medium at the final gate: no claim is CONTRADICTED by an authoritative source.")

    risk = {
        "level": level,
        "level_before_cap": risk_in.get("level_before_cap") or risk_in.get("level") or level,
        "summary": swept(risk_in.get("summary"), "the risk summary") or _fallback_summary(evidence, claims_out),
        "reasons": reasons,
    }

    # 5/6. CHECKLIST
    ck_in = ctx.get("checklist") or {}
    if not isinstance(ck_in, dict):
        ck_in = {}
        note("checklist was not an object; rebuilt from defaults.")

    def as_arr(v: Any, n: int) -> List[str]:
        if not isinstance(v, list):
            return []
        kept = []
        for x in v:
            if not isinstance(x, str) or not x.strip():
                continue
            if has_verdict(x):
                note('Dropped a checklist item phrased as a judgement: "%s".' % x[:50])
                continue
            kept.append(x[:300])
            if len(kept) >= n:
                break
        return kept

    reg = ck_in.get("registry_to_check") or {}
    reg_url = str(reg.get("url") or "")
    reg_host = _host(reg_url)
    if not reg_host or reg_host not in OFFICIAL_HOSTS:
        # A4 invented a regulator URL, or returned something unusable. Replace it
        # with the deterministic authority rather than sending a student onward.
        fallback = verification["manual_check"]
        note('A4 returned a registry URL that is not a known official register; replaced with "%s".' % fallback["authority"])
        reg = {"name": fallback["authority"], "url": fallback["url"]}
        reg_url = fallback["url"]
        reg_host = _host(reg_url)

    questions = as_arr(ck_in.get("questions_to_ask"), 5) or [
        "Which university issued this, and can I see the offer letter on their own domain?",
        "What is the written refund policy if the visa is refused?",
        "Are these fees taken from the university's published fee page?",
    ]
    documents = as_arr(ck_in.get("documents_before_paying"), 5) or [
        "Written offer letter on university letterhead",
        "Signed refund / cancellation policy",
        "Bank details matching the registered company name exactly",
    ]
    offer = swept(ck_in.get("what_a_real_offer_looks_like"), "the offer guidance")[:400] or (
        "A genuine offer is issued from the university's own domain, references a CAS number "
        "verifiable with the university, and its fee figures match the university's published fee page."
    )

    checklist = {
        "questions_to_ask": questions,
        "registry_to_check": {
            "name": str(reg.get("name") or verification["manual_check"]["authority"])[:160],
            "url": reg_url,
        },
        "documents_before_paying": documents,
        "what_a_real_offer_looks_like": offer,
    }

    audit_in = ctx.get("audit") or {}
    if not isinstance(audit_in, dict):
        audit_in = {}
        note("audit was not an object; recorded as not run.")
    try:
        attempts = int(audit_in.get("attempts") or 1)
    except (TypeError, ValueError):
        attempts = 1
    audit = {
        "passed": bool(audit_in.get("passed")),
        "attempts": max(1, min(3, attempts)),
        "gaps": [str(g)[:300] for g in (audit_in.get("gaps") or [])],
    }

    return {
        **ctx,
        "_t_ground": now_ms(),
        "extraction": extraction,
        "verification": verification,
        "evidence": evidence,
        "evidence_count": len(evidence),
        "risk": risk,
        "checklist": checklist,
        "audit": audit,
        "_grounding_final": {
            "repairs": repairs,
            "repair_count": len(repairs),
            "registry_host": reg_host,
            "registry_is_official": reg_host in OFFICIAL_HOSTS,
            "reasons_cited": sum(1 for r in reasons if r["cited"]),
            "reasons_total": len(reasons),
        },
    }


def _fallback_summary(evidence: List[dict], claims: List[dict]) -> str:
    if not evidence:
        return "No independent source could be reached for this submission, so nothing was cleared and nothing was disproved."
    bits = []
    if any("KNOWN-BAD" in str(e.get("value") or "") for e in evidence):
        bits.append("the provider is named on an official list of unrecognised institutions")
    contra = sum(1 for c in claims if c.get("status") == "CONTRADICTED")
    if contra:
        bits.append("%d of the claims made are contradicted by that source" % contra)
    if not bits:
        return "No claim could be confirmed or contradicted by an independent source. Treat this as insufficient evidence and check manually."
    return "On the evidence retrieved, " + "; ".join(bits) + "."


# ===========================================================================
# VERDICT DERIVATION + ASSEMBLER
# ===========================================================================


def derive_verdict(ctx: dict) -> str:
    """Never read off the model."""
    evidence = ctx.get("evidence") or []
    verification = ctx.get("verification") or {}
    level = (ctx.get("risk") or {}).get("level") or "Low"

    has_blacklist = any(
        re.search(r"KNOWN-BAD|List of Fake Universities", str(e.get("value") or ""), re.I)
        for e in evidence
    )
    has_contradiction = (
        has_blacklist
        or bool(verification.get("contradicted_claims"))
        or bool(verification.get("domain_mismatch"))
    )

    if not evidence:
        return "insufficient_evidence"
    if has_contradiction:
        return "serious_contradictions"
    if level in ("Medium", "High"):
        return "flags_raised"
    # A Low level is only "no concern" if something genuinely supported it.
    # With nothing retrievable we did not clear anyone, we simply could not check.
    supported = any(e.get("tier") in ("static", "live") for e in evidence)
    return "no_concern_found" if supported else "insufficient_evidence"


def _delta(a: Any, b: Any) -> Optional[int]:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b >= a:
        return int(round(b - a))
    return None


def build_trace(ctx: dict) -> List[dict]:
    """Real measured timings. The handbook asks teams to explain how their
    solution works, and a fabricated trace is the kind of thing that unravels
    in Q&A."""
    t = {
        "intake": ctx.get("_t_start"),
        "start": ctx.get("_t0"),
        "plan": ctx.get("_t_plan"),
        "retrieval": ctx.get("_t_retrieval"),
        "extract": ctx.get("_t_extract"),
        "verify": ctx.get("_t_verify"),
        "verify_block": ctx.get("_t_verify_block"),
        "match": ctx.get("_t_match"),
        "domain": ctx.get("_t_domain"),
        "reason": ctx.get("_t_reason"),
        "gate": ctx.get("_t_gate"),
        "audit": ctx.get("_t_audit"),
        "action": ctx.get("_t_action"),
        "ground": ctx.get("_t_ground"),
    }
    # Clamp to monotonic so a clock that jumped cannot produce a negative.
    last: Optional[float] = None
    for k in ("intake", "start", "plan", "retrieval", "extract", "verify", "reason", "gate", "audit", "action", "ground"):
        v = t[k]
        if not isinstance(v, (int, float)):
            t[k] = None
            continue
        if last is not None and v < last:
            t[k] = last
        last = t[k]

    risk = ctx.get("risk") or {}
    verification = ctx.get("verification") or {}
    extraction = ctx.get("extraction") or {}
    site = (ctx.get("retrieval") or {}).get("site") or {}
    grounding = ctx.get("_grounding_final") or {}
    audit = ctx.get("audit") or {}
    trace: List[dict] = []

    def add(agent: str, did: str, a: str, b: str) -> None:
        trace.append({"agent": agent, "did": did, "ms": _delta(t.get(a), t.get(b))})

    pii = ", ".join(ctx.get("pii_kinds_redacted") or []) or "none found"
    add("Sanitiser", "HTML stripped, PII redacted (%s)" % pii, "intake", "start")
    add("Planner", "Retrieval plan built (%d steps)" % len(ctx.get("retrieval_plan") or []), "start", "plan")

    dom = verification.get("domain") or {}
    submitted = dc_submitted = (verification.get("official_domain_check") or {}).get("submitted_domain") or ctx.get("domain")
    if not submitted:
        retrieval_note = "No domain in the input to look up"
    elif site.get("fetched") or dom.get("created"):
        retrieval_note = "Website read as self-report; domain %s" % (
            "created " + str(dom.get("created")) if dom.get("created") else "age unavailable"
        )
    else:
        # The domain WAS in the input; we simply could not retrieve it. Reporting
        # that as "no domain in input" is a different and much weaker statement,
        # and a judge reading the trace would take it at face value.
        retrieval_note = "Domain %s named, but neither its site nor its registration record could be retrieved" % submitted
    add("Evidence retrieval", retrieval_note, "plan", "retrieval")
    add("Extractor", "Extracted %d claim(s)" % len(extraction.get("claims") or []), "retrieval", "extract")
    add(
        "Register match",
        verification.get("match_status", "") + (" as " + verification["matched_name"] if verification.get("matched_name") else ""),
        "verify_block",
        "match",
    )

    dc = verification.get("official_domain_check")
    if dc and dc.get("checked"):
        trace.append(
            {
                "agent": "Domain cross-check",
                "did": "submitted %s vs official %s -> %s" % (dc["submitted_domain"], dc["official_domain"], dc["match"]),
                "ms": _delta(t.get("match"), t.get("domain")),
            }
        )

    add("Risk reasoner", "%s · %d reason(s)" % (risk.get("level") or "Low", len(risk.get("reasons") or [])), "verify", "reason")

    gate_report = ctx.get("_grounding") or {}
    add(
        "Citation gate",
        ("REJECTED %d unsourced citation(s)" % gate_report["citations_rejected"])
        if gate_report.get("citations_rejected")
        else "All citations resolved against retrieved evidence",
        "reason",
        "gate",
    )
    add(
        "Self-audit",
        ("Passed" if audit.get("passed") else "Failed, retry budget spent") + " on attempt %d" % (audit.get("attempts") or 1),
        "gate",
        "audit",
    )
    add("Action planner", "Checklist generated", "audit", "action")
    add(
        "Grounding gate",
        ("%d envelope repair(s)" % grounding["repair_count"] if grounding.get("repair_count") else "Envelope validated")
        + " · %s confirmed" % (risk.get("level") or "Low"),
        "action",
        "ground",
    )
    return trace


def assemble_fraud_envelope(ctx: dict, profile: Optional[dict] = None) -> dict:
    """The merge point. Builds the single canonical envelope from whichever
    branch ran. Deliberately not an agent: it is deterministic and debuggable,
    which is the entire reason it is a separate step."""
    return {
        "case_id": ctx.get("case_id"),
        "_entry": ctx.get("_entry", "webhook"),
        "status": "awaiting_student",
        "intent": "fraud_check",
        "verdict": derive_verdict(ctx),
        "retrieval_plan": ctx.get("retrieval_plan") or [],
        "extraction": ctx.get("extraction") or {},
        "verification": ctx.get("verification") or {},
        "evidence": ctx.get("evidence") or [],
        "risk": ctx.get("risk") or {},
        "checklist": ctx.get("checklist"),
        "audit": ctx.get("audit"),
        # If the student named a college we can verify, attach its profile too,
        # so "is Bristol legit and what does it cost" is answered in one pass.
        "profile": profile,
        "trace": build_trace(ctx),
        "disclaimer": DISCLAIMER,
    }


def assemble_other_envelope(ctx: dict) -> dict:
    return {
        "case_id": ctx.get("case_id"),
        "_entry": ctx.get("_entry", "webhook"),
        "status": "complete",
        "intent": "other",
        "output": ctx.get("output", ""),
        "refused": bool(ctx.get("refused")),
        "scope_reason": ctx.get("_intent_reason") or None,
        "profile": None,
        "trace": [{"agent": "Intent gate", "did": ctx.get("_intent_reason") or "no investigation run", "ms": None}],
        "disclaimer": DISCLAIMER,
    }


# ===========================================================================
# LAYER 2 -- College profile (fully deterministic)
# ===========================================================================


def _profile_objects() -> List[dict]:
    out = []
    for p in PROFILES:
        out.append(
            {
                "name": p["name"],
                "jur": p["jur"],
                "reg": p["reg"],
                "official_domain": p["official_domain"],
                "official_url": "https://www." + p["official_domain"],
                "type": p["type"],
                "established_year": p["established"],
                "group": p["group"],
                "best_for": p["best_for"],
                "majors": p["majors"],
                "duration_years": p["duration"],
                "credit_hours": p["credit"],
                "admission_requirements": p["reqs"],
                "scholarships": [
                    {
                        "name": n,
                        "detail": d,
                        "source_url": p["reg_url"],
                        "retrieved_on": p["source_date"],
                        "tier": "static",
                    }
                    for n, d in p["scholarships"]
                ],
                "fees": {
                    "amount_home": None,
                    "amount_international": None,
                    "currency": None,
                    "status": "not_bundled",
                    "why": "Tuition fees are set per course and change each intake. A cached figure here "
                    "would be the exact failure this project exists to prevent, so no amount is "
                    "guessed. The official fees page below is the only place it should come from.",
                    "official_page": p["fees_url"],
                    "retrieved_on": p["source_date"],
                },
                "retrieved_on": p["source_date"],
            }
        )
    return out


PROFILE_OBJECTS = _profile_objects()


def match_profile(requested: Optional[str]) -> Tuple[Optional[dict], str, float]:
    """No model is asked whether an institution exists, what it is known for,
    or how many credits a course is. Those are facts."""
    if not requested:
        return None, "not_found", 0.0

    target = norm(requested)
    for p in PROFILE_OBJECTS:
        if norm(p["name"]) == target:
            return p, "verified", 1.0

    scored = []
    for p in PROFILE_OBJECTS:
        pn = norm(p["name"])
        tail = pn.split(" ")[-1] if pn.split(" ") else ""
        tokens = [t for t in target.split(" ") if t]
        overlap = (sum(1 for t in tokens if t in pn) / len(tokens)) if tokens else 0.0
        scored.append((max(similarity(target, pn), similarity(target, tail), overlap * 0.95), p))
    scored.sort(key=lambda x: x[0], reverse=True)

    if scored:
        best_s, best_p = scored[0]
        # 0.85 is deliberately strict. A wrong match here hands a student
        # another school's fees and scholarships, which is worse than saying
        # "not in our snapshot".
        if best_s >= 0.85:
            return best_p, "partial_match", round(best_s, 3)
        return None, "not_found", round(best_s, 3)
    return None, "not_found", 0.0


def build_profile(requested: Optional[str], submitted_domain: Optional[str]) -> dict:
    t_match = now_ms()
    match, match_status, score = match_profile(requested)
    t_matched = now_ms()
    domain_check = official_domain_check(match, submitted_domain)
    t_domain = now_ms()

    alternatives: List[dict] = []
    if match:
        same_group = [
            p for p in PROFILE_OBJECTS if p["name"] != match["name"] and p["group"] == match["group"]
        ]
        same_jur = [
            p
            for p in PROFILE_OBJECTS
            if p["name"] != match["name"] and p["jur"] == match["jur"] and p not in same_group
        ]

        def pick(arr: List[dict]) -> List[dict]:
            return [
                {
                    "name": p["name"],
                    "jurisdiction": p["jur"],
                    "group": p["group"],
                    "official_domain": p["official_domain"],
                    # Alternatives come ONLY from this same verified snapshot.
                    # Generating "top universities to study in X" without a source
                    # is the confident invention this project exists to catch.
                    "why": (
                        "Also a %s institution, listed on the same register." % p["group"]
                        if p["group"] and p["group"] == match["group"]
                        else "Another institution we can verify on the same register (%s)." % p["reg"]
                    ),
                    "source_url": p["official_url"],
                    "retrieved_on": p["retrieved_on"],
                }
                for p in arr[:3]
            ]

        alternatives = (pick(same_group) + pick(same_jur))[:4]

    sources: List[dict] = []
    if match:
        sources.append(
            {
                "label": "Register entry",
                "source_name": "%s (snapshot retrieved %s)" % (match["reg"], match["retrieved_on"]),
                "source_url": match["official_url"],
                "retrieved_at": match["retrieved_on"],
                "tier": "static",
            }
        )
        auth = AUTHORITY.get(match["jur"], DEFAULT_AUTHORITY)
        sources.append(
            {
                "label": "Regulator to check yourself",
                "source_name": auth["authority"],
                "source_url": auth["url"],
                "retrieved_at": match["retrieved_on"],
                "tier": "static",
            }
        )
        sources.append(
            {
                "label": "Official fees page",
                "source_name": "Official tuition and fees page",
                "source_url": match["fees"]["official_page"],
                "retrieved_at": match["fees"]["retrieved_on"],
                "tier": "static",
            }
        )
        if match.get("scholarships"):
            sources.append(
                {
                    "label": "Scholarship information",
                    "source_name": "Official scholarship and funding pages",
                    "source_url": match["scholarships"][0]["source_url"],
                    "retrieved_at": match["scholarships"][0]["retrieved_on"],
                    "tier": "static",
                }
            )
        if domain_check:
            sources.append(
                {
                    "label": "Official domain",
                    "source_name": "Institution official website",
                    "source_url": domain_check["official_url"],
                    "retrieved_at": match["retrieved_on"],
                    "tier": "static",
                }
            )
    for a in alternatives:
        sources.append(
            {
                "label": "Alternative: " + a["name"],
                "source_name": "Register snapshot",
                "source_url": a["source_url"],
                "retrieved_at": a["retrieved_on"],
                "tier": "static",
            }
        )

    cited = [s for s in sources if s.get("source_url") and s.get("retrieved_at")]

    if match:
        profile = {
            "exists_in_snapshot": True,
            "match_status": match_status,
            "match_score": score,
            "institution": match["name"],
            "requested_name": requested,
            "jurisdiction": match["jur"],
            "type": match["type"],
            "established_year": match["established_year"],
            "group": match["group"],
            "regulator": match["reg"],
            "official_domain": match["official_domain"],
            "official_url": match["official_url"],
            "best_known_for": match["best_for"],
            "majors": match["majors"],
            "duration_years": match["duration_years"],
            "credit_hours": match["credit_hours"],
            "admission_requirements": match["admission_requirements"],
            "scholarships": match["scholarships"],
            "fees": match["fees"],
            "alternatives": alternatives,
            "manual_check": AUTHORITY.get(match["jur"], DEFAULT_AUTHORITY),
            "data_snapshot_date": match["retrieved_on"],
            "coverage_note": "This is a dated local snapshot, not a live register query. Confirm anything you rely on at the official pages listed under sources.",
        }
    else:
        profile = {
            "exists_in_snapshot": False,
            "match_status": match_status,
            "match_score": score,
            "institution": None,
            "requested_name": requested,
            "jurisdiction": None,
            "official_domain": None,
            "official_url": None,
            "best_known_for": [],
            "majors": [],
            "duration_years": None,
            "credit_hours": None,
            "admission_requirements": [],
            "scholarships": [],
            "fees": None,
            "alternatives": [],
            "manual_check": DEFAULT_AUTHORITY,
            "data_snapshot_date": None,
            "coverage_note": "NOT IN OUR SNAPSHOT. This is a limit of our dataset, NOT evidence that the institution is fake. Our snapshot holds %d institutions. Check the register named in manual_check yourself." % len(PROFILE_OBJECTS),
        }

    return {
        "timings": {"t_match": t_match, "t_matched": t_matched, "t_domain": t_domain},
        "profile": profile,
        "domain_check": domain_check,
        "sources": cited,
        "uncited_dropped": len(sources) - len(cited),
    }


def render_profile_deterministic(profile: dict, domain_check: Optional[dict], sources: List[dict]) -> str:
    """The guaranteed-correct answer, built straight from the verified dataset.
    It is what the student sees if the model's phrasing fails the grounding
    gate, and it is the fallback that means this branch can never return an
    empty section."""
    p = profile
    dc = domain_check
    L: List[str] = []

    if p.get("exists_in_snapshot"):
        L.append("## Does it exist?")
        L.append(
            "Yes. **%s** appears on the **%s** (snapshot dated %s).%s"
            % (
                p["institution"],
                p["regulator"],
                p["data_snapshot_date"],
                (" It is a %s, established %s." % (p["type"], p["established_year"])) if p.get("type") else "",
            )
        )
        if p["match_status"] == "partial_match":
            L.append(
                'You wrote "%s". The register\'s official name is "%s", so this is a close match rather than an exact one. Confirm the exact name on the official site before you apply.'
                % (p["requested_name"], p["institution"])
            )
        L.append("")
        L.append("**Check it yourself:** %s — %s" % (p["manual_check"]["authority"], p["manual_check"]["url"]))
        L.append("")
    else:
        L.append("## Does it exist?")
        L.append("**Not in our snapshot.** Our local register extract holds %d institutions and this one is not among them." % len(PROFILE_OBJECTS))
        L.append("")
        L.append("That is a **limit of our dataset, not evidence that it is fake.** A university can be entirely real and still be missing from a small extract. Before you conclude anything, check: **%s** — %s" % (p["manual_check"]["authority"], p["manual_check"]["url"]))
        L.append("")
        L.append("If you send me the exact name as the institution writes it, or a link to their official site, I can try again.")
        L.append("")

    if p.get("exists_in_snapshot"):
        if p.get("best_known_for"):
            L.append("## Best known for")
            L += ["- " + x for x in p["best_known_for"]]
            L.append("")
        if p.get("majors"):
            L.append("## Majors and courses")
            L += ["- " + x for x in p["majors"]]
            L.append("")
        if p.get("duration_years"):
            L.append("## How long it takes")
            for k, v in p["duration_years"].items():
                label = k.replace("_", " ").title()
                L.append("- %s: **%s**" % (label, v))
            L.append("")
        if p.get("credit_hours"):
            ch = p["credit_hours"]
            L.append("## Credits")
            L.append("- Credit system: **%s**" % ch.get("system"))
            if ch.get("undergraduate_per_year"):
                L.append("- Typical undergraduate year: " + ch["undergraduate_per_year"])
            if ch.get("postgraduate_taught_total"):
                L.append("- Typical taught postgraduate total: " + ch["postgraduate_taught_total"])
            if ch.get("note"):
                L.append("- " + ch["note"])
            L.append("")
        if p.get("admission_requirements"):
            L.append("## Entry requirements")
            L += ["- " + x for x in p["admission_requirements"]]
            L.append("")
        if p.get("scholarships"):
            L.append("## Scholarships")
            L += ["- **%s** — %s" % (s["name"], s["detail"]) for s in p["scholarships"]]
            L.append("")
        if p.get("fees"):
            L.append("## Fees")
            L.append("I do not publish a cached fee figure. Tuition fees are set per course and change every intake, and a stale number here is the exact failure this tool exists to catch.")
            L.append("")
            L.append("Read the current figure on the official fees page: **%s**" % p["fees"]["official_page"])
            L.append("")
        if p.get("alternatives"):
            L.append("## Other options")
            L += ["- **%s** (%s) — %s" % (a["name"], a["jurisdiction"], a["why"]) for a in p["alternatives"]]
            L.append("")

    L.append("## Website check")
    if dc:
        if dc["match"] == "match":
            L.append("The site you gave (**%s**) is the official domain. Official site: %s" % (dc["submitted_domain"], dc["official_url"]))
        else:
            L.append("**The site you gave is not the official one.**")
            L.append("")
            L.append("- You gave: **%s**" % dc["submitted_domain"])
            L.append("- Official: **%s** — %s" % (dc["official_domain"], dc["official_url"]))
            L.append("")
            L.append("Copying a real university's name on an unrelated domain is one of the most common ways a convincing fake is built. Check the letterhead, the registered address and the payment account against the official site.")
    elif p.get("official_domain"):
        L.append("I could not cross-check a website, because no URL was included in your message.")
        L.append("")
        L.append("The official domain for %s is **%s**. If a consultancy gave you a different one, that is worth checking." % (p.get("institution") or "this institution", p["official_domain"]))
    else:
        L.append("No website was included, and we have no official domain on file for this institution, so this could not be checked.")
    L.append("")

    if sources:
        L.append("## Sources")
        L += ["- %s — %s (retrieved %s)" % (s["label"], s["source_url"], s["retrieved_at"]) for s in sources]
        L.append("")

    L.append("_%s_" % (p.get("coverage_note") or ""))
    L.append("")
    L.append(DISCLAIMER)
    return "\n".join(L)


def profile_grounding_gate(raw_model_text: str, deterministic: str, profile: dict) -> Tuple[str, dict]:
    """The model was asked to phrase a verified dataset readably. A formatting
    pass can still introduce a fact -- a fee, a scholarship name, a duration
    that was never in the data -- and a student acting on an invented number is
    exactly the harm this tool exists to prevent. So the model's text is CHECKED,
    and if it mentions anything not in the dataset it is discarded in favour of
    the deterministic rendering."""
    raw = str(raw_model_text or "")
    forbidden: List[str] = []

    fee_figures = [
        v
        for v in [
            (profile.get("fees") or {}).get("amount_home"),
            (profile.get("fees") or {}).get("amount_international"),
        ]
        if isinstance(v, (int, float)) and v > 0
    ]
    # Deliberately broad. Indian and UK consultancies quote fees as "Rs. 4,50,000"
    # and as "9,250 GBP" just as often as "GBP 9,250", and fees change every
    # intake, so a figure we cannot source is a figure we must not repeat. A false
    # positive here costs one model-written sentence and falls back to the
    # deterministic text, so being generous is the right side to err on.
    _CUR = r"(?:[\u00a3$\u20b9\u20b8\u20a9]\b|\b(?:usd|inr|rs\.?|gbp|eur|cad|aud))"
    _NUM = r"\d[\d,\u20b9\u20b8]{2,}"
    invented_fee = re.findall(_CUR + r"\s?" + _NUM + r"|" + _NUM + r"\s?" + _CUR, raw, re.I)
    del _CUR, _NUM
    fee_words = re.search(
        r"(fee|tuition|cost|price|afford|pay|payment|charge|charges|per year|annual|billing|payable|expensive|cheap|£|\$|€|₹)",
        raw,
        re.I,
    )
    if invented_fee and fee_words and not fee_figures:
        forbidden.append("fee amount not present in the dataset: " + ", ".join(invented_fee))

    dataset_years = set()
    for v in (profile.get("duration_years") or {}).values():
        dataset_years.update(re.findall(r"\d+", str(v)))
    if dataset_years:
        for c in re.findall(r"\b(\d)(?:\.\d)?\s*(?:-|–|to)?\s*(\d)?\s*(?:year|yr)", raw, re.I):
            nums = [n for n in c if n]
            if nums and not all(n in dataset_years for n in nums):
                forbidden.append("duration not present in the dataset: " + "".join(nums))

    allowed = set()
    if profile.get("institution"):
        allowed.add(str(profile["institution"]))
    for s in profile.get("scholarships") or []:
        if s and s.get("name"):
            allowed.add(str(s["name"]))
    for a in profile.get("alternatives") or []:
        if a and a.get("name"):
            allowed.add(str(a["name"]))
    for c in re.findall(
        r"\b(?:University|College|Institute|Academy|School)\s+of\s+[A-Z][A-Za-z]+|\b[A-Z][A-Za-z]+\s+(?:University|College|Institute|Academy)\b",
        raw,
    ):
        if not any(
            str(n).lower() in c.lower() or c.lower() in str(n).lower() for n in allowed
        ):
            forbidden.append("institution name not present in the dataset: " + c)

    violations = forbidden[:5]
    used_model = not violations and bool(raw.strip())

    return (
        raw if used_model else deterministic,
        {
            "model_output_accepted": used_model,
            "violations": violations,
            "fell_back_to_deterministic": not used_model,
        },
    )


def assemble_profile_envelope(
    case_id: str,
    entry: str,
    extraction: dict,
    built: dict,
    text: str,
    grounding: dict,
    timings: dict,
) -> dict:
    p = built["profile"]
    srcs = built["sources"]
    bt = built.get("timings") or {}
    trace = [
        {"agent": "Intent gate", "did": "routed to college profile", "ms": None},
        {
            "agent": "Extractor",
            "did": 'extracted institution "%s"' % (extraction.get("institution")),
            "ms": _delta(timings.get("t_extract"), timings.get("t_profile_match")),
        },
        {
            "agent": "Profile lookup",
            "did": ("matched %s on %s" % (p["institution"], p["regulator"]))
            if p.get("exists_in_snapshot")
            else "no match in the bundled snapshot",
            "ms": _delta(bt.get("t_match"), bt.get("t_matched")),
        },
        {
            "agent": "Domain cross-check",
            "did": (
                "submitted %s vs official %s -> %s"
                % (
                    built["domain_check"]["submitted_domain"],
                    built["domain_check"]["official_domain"],
                    built["domain_check"]["match"],
                )
            )
            if built.get("domain_check")
            else "no domain in input",
            "ms": _delta(bt.get("t_matched"), bt.get("t_domain")),
        },
        {
            "agent": "Briefing writer",
            "did": "model phrasing accepted after grounding check"
            if grounding.get("model_output_accepted")
            else "model output REJECTED, deterministic rendering used: " + "; ".join(grounding.get("violations") or []),
            "ms": _delta(timings.get("t_profile_match"), timings.get("t_profile_ground")),
        },
        {
            "agent": "Grounding gate",
            "did": "%d cited source(s); %d uncited record(s) dropped" % (len(srcs), built.get("uncited_dropped", 0)),
            "ms": None,
        },
    ]

    return {
        "case_id": case_id,
        "_entry": entry,
        "status": "complete",
        "intent": "profile",
        "output": text,
        "profile": p,
        "profile_official_domain_check": built.get("domain_check"),
        "sources": srcs,
        # A profile request has no claim to risk-score, so risk is explicitly
        # null rather than a fabricated "Low". Reporting "no risk found" about
        # a college we have not investigated would be the same error in nicer
        # clothes.
        "risk": None,
        "evidence": [
            {
                "id": "E%d" % (i + 1),
                "label": s["label"],
                "value": s["source_name"],
                "source_name": s["source_name"],
                "source_url": s["source_url"],
                "retrieved_at": s["retrieved_at"],
                "tier": s.get("tier") or "static",
                "claim_id": None,
            }
            for i, s in enumerate(srcs)
        ],
        "verification": {
            "match_status": p.get("match_status") or "not_found",
            "matched_name": p.get("institution"),
            "jurisdiction": p.get("jurisdiction"),
            "tier_used": "static",
            "manual_check": p.get("manual_check"),
            "official_domain_check": built.get("domain_check"),
            "domain": None,
        },
        "verdict": "no_concern_found",
        "verdict_note": "This is a profile lookup, not a risk investigation. No risk assessment was performed and none should be inferred.",
        "trace": trace,
        "disclaimer": DISCLAIMER,
    }


# ===========================================================================
# Chat formatting
# ===========================================================================


def format_chat_reply(env: dict) -> str:
    """Same envelope, rendered for a chat window. The data is not re-derived
    here, only formatted."""
    if env.get("intent") in ("other", "profile"):
        return env.get("output", "")

    L: List[str] = []
    p = env.get("profile")
    if p and p.get("exists_in_snapshot"):
        L.append("### About " + p["institution"])
        best = ", ".join((p.get("best_known_for") or [])[:3])
        L.append("It is on the **%s**. %s" % (p["regulator"], ("Best known for " + best + ".") if best else ""))
        L.append("Official domain: **%s**" % p["official_domain"])
        L.append("")
        L.append('> Ask me *"tell me about %s"* for the full profile — majors, duration, credits, requirements, scholarships and the official fees page.' % p["institution"])
        L.append("")

    L.append("### What I checked")
    for x in env.get("trace") or []:
        L.append("- **%s** — %s%s" % (x["agent"], x["did"], (" (%dms)" % x["ms"]) if x.get("ms") is not None else ""))
    L.append("")
    L.append("### Verdict: " + str(env.get("verdict") or "").replace("_", " "))
    L.append("")

    if env.get("verdict") == "insufficient_evidence":
        L.append("I could not check this against any source. That is a limit on what I could verify, **not** a clean bill of health.")
        L.append("")

    if env.get("evidence"):
        L.append("### Evidence")
        for e in env["evidence"]:
            L.append("**%s**" % e["label"])
            L.append(e["value"])
            L.append("_%s — %s (retrieved %s, tier: %s)_" % (e["source_name"], e["source_url"], e["retrieved_at"], e["tier"]))
            L.append("")

    risk = env.get("risk") or {}
    if risk.get("reasons"):
        L.append("### Risk: " + str(risk.get("level")))
        if risk.get("summary"):
            L.append(risk["summary"])
        L.append("")
        for r in risk["reasons"]:
            L.append("- **%s** — %s %s" % (r["flag"], r["why"], ("_(" + str(r.get("source_url")) + ")_") if r.get("source_url") else "_(reasoning only — no source)_"))
        L.append("")

    ck = env.get("checklist") or {}
    if ck.get("questions_to_ask"):
        L.append("### Before you pay anyone, ask")
        L += ["- " + q for q in ck["questions_to_ask"]]
        L.append("")
        if ck.get("registry_to_check"):
            L.append("**Register to check:** %s — %s" % (ck["registry_to_check"]["name"], ck["registry_to_check"]["url"]))
            L.append("")

    dc = (env.get("verification") or {}).get("official_domain_check")
    if dc and dc.get("checked") and dc["match"] != "match":
        L.append("### Website check")
        L.append("The site you gave (**%s**) is **not** the official domain. The official one is **%s** — %s" % (dc["submitted_domain"], dc["official_domain"], dc["official_url"]))
        L.append("")

    L.append("_%s_" % (env.get("disclaimer") or DISCLAIMER))
    L.append("")
    L.append("Want me to dig further into a specific area? Reply with your decision and I will keep going. The decision is yours — I only show you the evidence.")
    return "\n".join(L)


# ===========================================================================
# Human-in-the-loop resume
# ===========================================================================


def merge_student_decision(parked: dict, body: dict) -> dict:
    """The decision is the student's alone. This RECORDS it and attaches the
    evidence the chosen scope calls for; it never re-scores, never overrides a
    verdict, and never appends a recommendation the student did not ask for."""
    decision = str(body.get("decision") or "accept")
    scope = body.get("scope") if isinstance(body.get("scope"), list) else []
    decided_at = datetime.now(timezone.utc).isoformat()

    # A case_id that does not match means the browser resumed the wrong
    # execution. Reported rather than silently merged: silently merging would
    # attach one student's decision to another student's report.
    if body.get("case_id") and parked.get("case_id") and str(body["case_id"]) != str(parked["case_id"]):
        return {
            **parked,
            "status": "complete",
            "decision_recorded": "rejected_case_id_mismatch",
            "decided_at": decided_at,
            "decision_note": "The case_id in the decision did not match the parked report, so the decision was NOT applied. This usually means the wrong execution was resumed.",
        }

    findings: List[dict] = []
    if decision == "investigate_further":
        verification = parked.get("verification") or {}
        d = verification.get("domain")
        if "domain_history" in scope:
            findings.append(
                {
                    "scope": "domain_history",
                    "finding": (
                        "Domain %s was registered %s (%s days ago)." % (d["domain"], d["created"], d["age_days"])
                        if d and d.get("created")
                        else "No registration date is published for this domain, so its age cannot be confirmed. An unverifiable domain is a reason for caution, not proof of anything."
                    ),
                    "source_url": (d or {}).get("source_url"),
                }
            )
        if "company_registry" in scope:
            findings.append(
                {
                    "scope": "company_registry",
                    "finding": "Company status is not checked in this build. Check the consultancy's exact legal name at the official company register (Companies House in the UK, MCA21 in India) and read the company_status field. A dissolved or liquidated company cannot lawfully be taking your money.",
                    "source_url": "https://find-and-update.company-information.service.gov.uk/",
                }
            )
        if "fee_verification" in scope:
            findings.append(
                {
                    "scope": "fee_verification",
                    "finding": "Compare the quoted fee against the university's own published fee page. A consultancy fee that is a flat percentage of tuition, or a 'processing' fee with no written basis, is a strong warning sign.",
                    "source_url": None,
                }
            )
        if "advisories" in scope:
            findings.append(
                {
                    "scope": "advisories",
                    "finding": "Check whether the regulator has published an advisory about this provider, and read reviews for repeated mentions of one complaint pattern rather than single allegations.",
                    "source_url": None,
                }
            )
        if not scope:
            findings.append({"scope": "none", "finding": "No scope was selected, so no additional checks were run.", "source_url": None})

    out = {**parked, "status": "complete", "decision_recorded": decision, "decided_at": decided_at}

    # Kept separate from the original evidence[] so a student can always see
    # what was known at each stage.
    if findings:
        out["followup_findings"] = findings

    if decision == "dismiss":
        out["decision_note"] = "You dismissed this concern. That is recorded, and nothing was changed in the findings above. The evidence stays exactly as it was."
    elif decision == "accept":
        out["decision_note"] = "You accepted these findings. The evidence above is what you acted on."
    else:
        out["decision_note"] = "You asked for a closer look. The findings below are what the selected scope could and could not establish."
    return out
