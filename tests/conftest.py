"""Test fixtures.

Two rules govern this suite:

1. **No test may touch the network or a real model.** Every live call is
   replaced. A test that passes only because Ollama happened to be running, or
   because rdap.org answered, tells you nothing about the code and fails on the
   demo machine.
2. **A model that lies is the default, not the exception.** The fake LLM used
   here deliberately returns a fabricated citation, a verdict-worded reason and
   an invented fee, because that is the only way to prove the deterministic
   gates actually catch those things.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edushield import pipeline  # noqa: E402
from edushield.app import app as fastapi_app  # noqa: E402


# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------

# Markers to identify which prompt is being answered. Each matches a unique
# phrase from the corresponding prompt template in prompts.py.
PROMPT_MARKERS = {
    "a1": "convert a message from an education consultancy",
    "a2": "You assess RISK for a student",
    "a3": "You are a quality auditor",
    "a4": "You turn a risk assessment into a checklist",
    "profile": "You are writing a short, factual briefing",
}

DEFAULT_EXTRACTION = {
    "institution": "University of Manchester",
    "program": "MSc Data Science",
    "consultancy": "Bright Path Education",
    "claimed_fees": "GBP 28,000",
    "claimed_timeline": "2 years",
    "claimed_guarantees": ["Guaranteed visa"],
    "claims": [
        {"claim_id": "C1", "text": "Fee is GBP 28,000", "type": "fee"},
        {"claim_id": "C2", "text": "Visa is guaranteed", "type": "guarantee"},
    ],
}

DEFAULT_RISK = {
    "level": "Medium",
    "summary": "The domain is recent and the fee claim has not been confirmed.",
    "reasons": [
        {
            "flag": "Domain age",
            "why": "The domain was registered recently relative to the claimed history.",
            "evidence_ref": "E2",
            "source_url": "https://rdap.org/domain/example-university-admissions.com",
            "retrieved_at": "2026-09-27",
            "confidence": 0.6,
        }
    ],
}

DEFAULT_AUDIT = {"pass": True, "gaps": [], "followup_queries": []}

DEFAULT_CHECKLIST = {
    "questions_to_ask": ["Is there a written refund policy if the visa is refused?"],
    "registry_to_check": {
        "name": "Office for Students (OfS) Register of Regulated Providers",
        "url": "https://www.officeforstudents.org.uk/so-providers/",
    },
    "documents_before_paying": ["Written offer letter on university letterhead"],
    "what_a_real_offer_looks_like": "A real offer comes from the university's own domain.",
}


class FakeLLM:
    """Scriptable stand-in for Ollama.

    ``responses`` maps a prompt marker name to a value, or to a list consumed one
    per call so a retry loop can be exercised. An unexpected prompt is a test
    failure rather than a silent default, because a silent default is how a
    regression hides.
    """

    def __init__(self, responses: Optional[dict] = None, default_text: str = ""):
        self.responses = responses or {}
        self.default_text = default_text
        self.calls: list = []
        self._consumed: dict = {}

    def _lookup(self, prompt: str) -> tuple:
        kind = "unknown"
        for name, marker in PROMPT_MARKERS.items():
            if marker in prompt:
                kind = name
                break
        self.calls.append(kind)
        value = self.responses.get(kind)
        if isinstance(value, list):
            n = self._consumed.get(kind, 0)
            self._consumed[kind] = n + 1
            if n < len(value):
                value = value[n]
            else:
                value = value[-1]
        return kind, value

    def json_call(self, prompt: str, **kwargs: Any):
        kind, value = self._lookup(prompt)
        if value is None:
            return None, {"ok": True, "model": "fake", "ms": 1, "parsed": False, "kind": kind}
        return value, {"ok": True, "model": "fake", "ms": 1, "parsed": True, "kind": kind}

    def text_call(self, prompt: str, **kwargs: Any):
        kind, value = self._lookup(prompt)
        if value is None:
            return None, {"ok": True, "model": "fake", "ms": 1, "kind": kind}
        return str(value), {"ok": True, "model": "fake", "ms": 1, "kind": kind}

    def start_budget(self, seconds: float) -> None:
        """Mirrors Ollama.start_budget. The fake answers instantly, so the
        budget only needs to exist for the pipeline to arm it."""
        self.budget = seconds

    def available(self) -> bool:
        return True

    def models(self) -> list:
        return ["fake-model:latest"]


class DeadLLM(FakeLLM):
    """Every call fails. Used to prove the pipeline fails CLOSED rather than
    open: a dead model must lower the level and empty the reasons, never let a
    strong-looking assessment through."""

    def json_call(self, prompt: str, **kwargs: Any):
        self.calls.append("dead")
        return None, {"ok": False, "model": "fake", "ms": 1, "error": "connection refused"}

    def text_call(self, prompt: str, **kwargs: Any):
        self.calls.append("dead")
        return None, {"ok": False, "model": "fake", "ms": 1, "error": "connection refused"}

    def available(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_live_network(monkeypatch):
    """Stub both outbound calls in every test.

    RDAP is stubbed with a real, well-formed response so the date arithmetic is
    genuinely exercised; the site fetch is stubbed with a page containing a risk
    phrase, so the self-report probes run for real.
    """
    from edushield import tools

    def fake_fetch_website(domain):
        html = (
            "<html><head><title>Example University Admissions</title></head>"
            "<body>We guarantee your visa admission. No entrance exam required. "
            "Pay the full fee upfront by bank transfer. We are fully accredited.</body></html>"
        )
        return html, {"url": "https://" + domain, "ok": True, "ms": 5, "status": 200}

    def fake_fetch_rdap(domain):
        return (
            {
                "events": [
                    {"eventAction": "registration", "eventDate": "2026-08-30T00:00:00Z"},
                    {"eventAction": "expiration", "eventDate": "2027-08-30T00:00:00Z"},
                ]
            },
            {"ok": True, "ms": 5, "status": 200},
        )

    monkeypatch.setattr(tools, "fetch_website", fake_fetch_website)
    monkeypatch.setattr(tools, "fetch_rdap", fake_fetch_rdap)


@pytest.fixture
def fake_llm(monkeypatch):
    """The default well-behaved model. Individual tests override via `set_responses`."""
    llm = FakeLLM(
        {
            "a1": DEFAULT_EXTRACTION,
            "a2": DEFAULT_RISK,
            "a3": DEFAULT_AUDIT,
            "a4": DEFAULT_CHECKLIST,
            "profile": "## Does it exist?\n\nYes.",
        }
    )
    monkeypatch.setattr(pipeline, "get_ollama", lambda: llm)
    return llm


@pytest.fixture
def set_responses(monkeypatch):
    """Install a custom FakeLLM. `responses` uses the keys in PROMPT_MARKERS."""

    def _install(responses=None, dead=False):
        llm = DeadLLM() if dead else FakeLLM(responses or {}, default_text="")
        monkeypatch.setattr(pipeline, "get_ollama", lambda: llm)
        return llm

    return _install


@pytest.fixture
def client(fake_llm, monkeypatch):
    from fastapi.testclient import TestClient

    from edushield import app as app_module

    monkeypatch.setattr(app_module, "get_ollama", lambda: fake_llm)
    app_module.store._items.clear()
    with TestClient(app_module.app) as c:
        yield c
    app_module.store._items.clear()


@pytest.fixture(autouse=True)
def _no_env_leak(monkeypatch):
    """A stray env var in the developer's shell must not change a test result."""
    for var in ("EDUSHIELD_MODEL", "EDUSHIELD_FALLBACK_MODEL", "EDUSHIELD_CTX", "OLLAMA_URL"):
        monkeypatch.delenv(var, raising=False)
    os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_investigate(body: dict) -> dict:
    """Run the pipeline synchronously and return the envelope."""
    return pipeline.investigate(body, entry="test")
