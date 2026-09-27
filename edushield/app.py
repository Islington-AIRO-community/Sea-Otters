"""FastAPI app. The 2 contract endpoints, plus a chat alias and a health check.

CORS is fully open because this is a demo backend: the frontend is a static
file the judges open from a different origin than this server. If you ever put
this behind a real domain, pin `allow_origins` to your frontend's origin.

The case store is what replaces n8n's Wait node. It holds the parked report so
`POST /webhook/student-decision` can resume it without an execution object. It
is in-process and bounded, which is fine for a live demo and is the honest
scope of what it is: not a database.
"""

from __future__ import annotations

import os
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import core, pipeline
from .llm import get_ollama

app = FastAPI(
    title="Varosha backend (no n8n)",
    description="Study-abroad fraud verification. Local free llama models via Ollama, live RDAP, bundled register snapshot.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("EDUSHIELD_CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DISCLAIMER = core.DISCLAIMER


# ---------------------------------------------------------------------------
# Case store -- the replacement for the Wait node
# ---------------------------------------------------------------------------

class CaseStore:
    """Bounded LRU of parked reports. A Wait node holds one execution open for
    24h; this holds the same envelope for TTL_SECONDS."""

    def __init__(self, capacity: int = 200, ttl_seconds: int = 86_400):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self._items: "OrderedDict[str, tuple]" = OrderedDict()

    def put(self, case_id: str, envelope: Dict[str, Any]) -> None:
        self._items[case_id] = (time.time(), envelope)
        self._items.move_to_end(case_id)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)

    def get(self, case_id: str) -> Optional[Dict[str, Any]]:
        item = self._items.get(case_id)
        if not item:
            return None
        stored_at, envelope = item
        if time.time() - stored_at > self.ttl:
            self._items.pop(case_id, None)
            return None
        self._items.move_to_end(case_id)
        return envelope

    def drop(self, case_id: str) -> None:
        self._items.pop(case_id, None)

    def __len__(self) -> int:
        return len(self._items)


store = CaseStore()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class InvestigateRequest(BaseModel):
    case_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input_type: str = "text"
    text: str = ""
    url: str = ""
    source: str = "web"


class ChatRequest(BaseModel):
    chatInput: str = Field(default="", alias="chatInput")
    case_id: Optional[str] = None

    model_config = {"populate_by_name": True}


class StudentDecisionRequest(BaseModel):
    case_id: str
    decision: str = "accept"
    scope: List[str] = Field(default_factory=list)


VALID_DECISIONS = {"investigate_further", "accept", "dismiss"}
VALID_SCOPE = {"domain_history", "company_registry", "fee_verification", "advisories"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

_UI = os.path.join(os.path.dirname(__file__), "static")


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    """The student-facing page. Served from this origin on purpose: the browser
    then talks to /webhook/investigate same-origin, so the demo needs no CORS
    setup, no second terminal and no separate dev server."""
    return FileResponse(os.path.join(_UI, "index.html"), media_type="text/html")


@app.get("/health")
def health() -> dict:
    """Lets the frontend grey out its submit button before the demo instead of
    failing on the first click."""
    llm = get_ollama()
    return {
        "status": "ok",
        "ollama": llm.available(),
        "models": llm.models(),
        "parked_cases": len(store),
        "disclaimer": DISCLAIMER,
    }


def _require_input(text: str, url: str) -> None:
    """An investigation with nothing to investigate must not return 200.

    Every field on the request has a default, so a frontend that sends the wrong
    key name -- or a body that fails to parse at all -- used to produce a clean,
    confident, EMPTY report. That is the worst possible failure for this tool:
    it looks like a successful check. A loud 400 is the safe outcome.
    """
    if not (text or "").strip() and not (url or "").strip():
        raise HTTPException(
            400,
            "Nothing to investigate: send the student's message in 'text' "
            "(or the page in 'url'). An empty request is refused rather than "
            "answered, so it cannot be mistaken for a clean report.",
        )

@app.post("/webhook/investigate")
def investigate(req: InvestigateRequest = Body(...)) -> dict:
    _require_input(req.text, req.url)

    env = pipeline.investigate(req.model_dump(), entry="webhook")

    if env.get("status") == "awaiting_student":
        # Park it. This is what the Wait node did in n8n.
        store.put(env["case_id"], env)

    # The `output` field is chat-only formatting; the frontend ignores it here.
    env.pop("output", None)
    return env


@app.post("/webhook/student-decision")
def student_decision(req: StudentDecisionRequest = Body(...)) -> dict:
    if req.decision not in VALID_DECISIONS:
        raise HTTPException(400, "decision must be one of %s" % sorted(VALID_DECISIONS))
    bad_scope = [s for s in req.scope if s not in VALID_SCOPE]
    if bad_scope:
        raise HTTPException(400, "unknown scope value(s): %s" % bad_scope)
    if req.decision != "investigate_further" and req.scope:
        raise HTTPException(400, "scope is only valid when decision is investigate_further")

    parked = store.get(req.case_id)
    if parked is None:
        raise HTTPException(
            404,
            "No parked report for that case_id. The report may have expired, or the server restarted. "
            "Run /webhook/investigate again.",
        )

    out = core.merge_student_decision(parked, req.model_dump())
    store.put(req.case_id, out)
    out.pop("output", None)
    return out


@app.post("/chat")
def chat(req: ChatRequest = Body(...)) -> dict:
    """Same pipeline, chat-shaped input, chat-shaped output. The frontend can
    point at this for the chat demo and the other two for the report view."""
    _require_input(req.chatInput, "")
    env = pipeline.chat(req.chatInput or "", req.case_id)
    if env.get("status") == "awaiting_student":
        store.put(env["case_id"], env)
    return {
        "case_id": env.get("case_id"),
        "intent": env.get("intent"),
        "status": env.get("status"),
        "output": env.get("output") or core.format_chat_reply(env),
        "envelope": env,
    }


@app.get("/case/{case_id}")
def get_case(case_id: str) -> dict:
    """Lets a reloaded browser recover a parked report. n8n needed this to be
    the Wait node's own resume URL; here it is a lookup."""
    env = store.get(case_id)
    if env is None:
        raise HTTPException(404, "no parked report for that case_id")
    return env


    if env.get("status") == "awaiting_student":
        # Park it. This is what the Wait node did in n8n.
        store.put(env["case_id"], env)

    # The `output` field is chat-only formatting; the frontend ignores it here.
    env.pop("output", None)
    return env


@app.post("/webhook/student-decision")
def student_decision(req: StudentDecisionRequest = Body(...)) -> dict:
    if req.decision not in VALID_DECISIONS:
        raise HTTPException(400, "decision must be one of %s" % sorted(VALID_DECISIONS))
    bad_scope = [s for s in req.scope if s not in VALID_SCOPE]
    if bad_scope:
        raise HTTPException(400, "unknown scope value(s): %s" % bad_scope)
    if req.decision != "investigate_further" and req.scope:
        raise HTTPException(400, "scope is only valid when decision is investigate_further")

    parked = store.get(req.case_id)
    if parked is None:
        raise HTTPException(
            404,
            "No parked report for that case_id. The report may have expired, or the server restarted. "
            "Run /webhook/investigate again.",
        )

    out = core.merge_student_decision(parked, req.model_dump())
    store.put(req.case_id, out)
    out.pop("output", None)
    return out


@app.post("/chat")
def chat(req: ChatRequest = Body(...)) -> dict:
    """Same pipeline, chat-shaped input, chat-shaped output. The frontend can
    point at this for the chat demo and the other two for the report view."""
    env = pipeline.chat(req.chatInput or "", req.case_id)
    if env.get("status") == "awaiting_student":
        store.put(env["case_id"], env)
    return {
        "case_id": env.get("case_id"),
        "intent": env.get("intent"),
        "status": env.get("status"),
        "output": env.get("output") or core.format_chat_reply(env),
        "envelope": env,
    }


@app.get("/case/{case_id}")
def get_case(case_id: str) -> dict:
    """Lets a reloaded browser recover a parked report. n8n needed this to be
    the Wait node's own resume URL; here it is a lookup."""
    env = store.get(case_id)
    if env is None:
        raise HTTPException(404, "no parked report for that case_id")
    return env
