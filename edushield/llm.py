"""Local Ollama client.

Free, offline, no API key. The only external dependency of the whole system.

Every call returns ``(payload, meta)`` and NEVER raises. A model that times out,
returns prose where JSON was asked for, or emits half an object must not take
the pipeline down with it: callers get ``None`` and fall back to deterministic
logic, which is the same contract the gates enforce on model output anyway.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional, Tuple

import httpx

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")

# mistral first: 32k context and markedly better at holding a JSON schema than
# llama2's 4k. Both are free local models. Override with the env var below.
DEFAULT_MODEL = os.environ.get("EDUSHIELD_MODEL", "mistral:latest")
FALLBACK_MODEL = os.environ.get("EDUSHIELD_FALLBACK_MODEL", "llama2:latest")

# Measured on this machine: ~19 generated tokens/sec, so a 180s timeout is not
# safety, it is a 3-minute stall. A call that has not produced usable output in
# this long is not going to, and every caller has a deterministic fallback.
DEFAULT_TIMEOUT = float(os.environ.get("EDUSHIELD_LLM_TIMEOUT", "35"))

# Wall-clock ceiling for one whole investigation. Every model call derives its
# own timeout from what is left, so the total is bounded no matter how the model
# behaves. Past the ceiling, calls return immediately and the deterministic
# layer finishes the report.
TOTAL_BUDGET = float(os.environ.get("EDUSHIELD_BUDGET", "45"))
DEFAULT_CTX = int(os.environ.get("EDUSHIELD_CTX", "8192"))


class LLMUnavailable(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# JSON repair
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)


def _strip_fence(text: str) -> str:
    m = _FENCE.search(text)
    return m.group(1) if m else text


def _balanced_slice(text: str, opener: str, closer: str) -> Optional[str]:
    """Return the first balanced opener..closer span, ignoring braces in strings."""
    start = text.find(opener)
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    # Unclosed: return the tail so the repair pass has something to close.
    return text[start:]


def _repair(candidate: str) -> str:
    """Best-effort fixes for the malformed JSON small models actually emit."""
    s = candidate
    # Python literals leaking out of a non-constrained decode.
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    # Control characters inside string literals.
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", s)
    # Trailing commas before } or ].
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # Unescaped newlines inside string literals: close, escape, reopen.
    s = re.sub(r"\\?\n(?=[^}\]\"]*\")", " ", s)
    return s


def parse_json_loose(text: str) -> Optional[Any]:
    """Parse JSON out of model output, tolerating fences, prose and small damage.

    Local models wrap JSON in explanation, fence it, or emit a trailing comma.
    All of that is recoverable, so it is recovered rather than discarded.
    """
    if text is None:
        return None
    raw = _strip_fence(str(text)).strip()
    if not raw:
        return None

    for attempt in (raw, _balanced_slice(raw, "{", "}") or "", _balanced_slice(raw, "[", "[") or ""):
        if not attempt:
            continue
        for candidate in (attempt, _repair(attempt)):
            try:
                return json.loads(candidate)
            except (ValueError, TypeError):
                continue
    return None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class Ollama:
    def __init__(self, url: str = OLLAMA_URL, timeout: float = DEFAULT_TIMEOUT):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None
        self._available: Optional[bool] = None
        self._deadline: Optional[float] = None

    # -- run budget --------------------------------------------------------
    def start_budget(self, seconds: float = TOTAL_BUDGET) -> None:
        """Start the clock for one investigation. Called at the top of each run."""
        self._deadline = time.time() + seconds

    def remaining(self) -> float:
        """Seconds left in the run budget, or inf when no budget is running."""
        if self._deadline is None:
            return float("inf")
        return self._deadline - time.time()

    def out_of_budget(self) -> bool:
        return self.remaining() <= 0.5

    # -- plumbing ----------------------------------------------------------
    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            r = self.client.get(f"{self.url}/api/tags", timeout=5.0)
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def models(self) -> list:
        try:
            r = self.client.get(f"{self.url}/api/tags", timeout=5.0)
            r.raise_for_status()
            return [m.get("name") for m in r.json().get("models", [])]
        except Exception:
            return []

    # -- generation --------------------------------------------------------
    def _pick_model(self, requested: Optional[str]) -> str:
        installed = self.models()
        if requested and (not installed or requested in installed):
            return requested
        if installed:
            for pref in (DEFAULT_MODEL, FALLBACK_MODEL):
                if pref in installed:
                    return pref
            return installed[0]
        return requested or DEFAULT_MODEL

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        json_mode: bool = False,
        num_predict: int = 1400,
        num_ctx: Optional[int] = None,
    ) -> Tuple[Optional[str], dict]:
        """One completion. Returns ``(text_or_None, meta)``; never raises.

        On failure the fallback model is tried once, because a 4k-context model
        blowing up on a long prompt is the single most likely local failure and
        the larger one usually handles it.
        """
        payload: dict = {
            "model": self._pick_model(model),
            "prompt": prompt,
            "stream": False,
            # Explicit, because the 5-minute default can expire between demo
            # steps and every call would then pay a full model load.
            "keep_alive": os.environ.get("EDUSHIELD_KEEP_ALIVE", "30m"),
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx or DEFAULT_CTX,
                "num_predict": num_predict,
            },
        }
        if system:
            payload["system"] = system
        if json_mode:
            # Ollama's grammar constraint. This is the cheapest large win for a
            # small local model: it stops prose leaking into a JSON field.
            payload["format"] = "json"

        attempts = [payload]
        if payload["model"] != FALLBACK_MODEL and FALLBACK_MODEL in self.models():
            second = dict(payload)
            second["model"] = FALLBACK_MODEL
            attempts.append(second)

        # A second model is only worth trying if the first attempt failed
        # QUICKLY. Retrying after a timeout does not rescue the run, it doubles
        # the wait: that is how a single checklist call once took 187 seconds.
        budget_left = self.remaining()
        if budget_left < float("inf") and not (0 < budget_left <= 12):
            attempts = attempts[:1]

        last_err = None
        for body in attempts:
            remaining = self.remaining()
            if remaining <= 0.5:
                return None, {"model": body["model"], "ok": False, "ms": 0,
                              "error": "run budget exhausted", "budget_exhausted": True}
            call_timeout = self.timeout
            if remaining != float("inf"):
                call_timeout = max(2.0, min(call_timeout, remaining))
            started = time.time()
            try:
                r = self.client.post(f"{self.url}/api/generate", json=body, timeout=call_timeout)
                r.raise_for_status()
                data = r.json()
                text = data.get("response")
                if text is None:
                    last_err = "no response field"
                    continue
                return text, {
                    "model": body["model"],
                    "ok": True,
                    "ms": int((time.time() - started) * 1000),
                    "eval_count": data.get("eval_count"),
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "used_fallback_model": body["model"] != payload["model"],
                }
            except Exception as exc:  # noqa: BLE001 - a dead model must not kill the run
                last_err = f"{type(exc).__name__}: {exc}"
                continue

        return None, {
            "model": payload["model"],
            "ok": False,
            "ms": int((time.time() - started) * 1000) if "started" in dir() else 0,
            "error": last_err or "unknown",
        }

    def json_call(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        num_predict: int = 1400,
        num_ctx: Optional[int] = None,
    ) -> Tuple[Optional[Any], dict]:
        """Constrained JSON completion. Returns ``(parsed_or_None, meta)``.

        Falls back to an unconstrained call when the grammar constraint yields
        nothing parseable, because a repairable object beats no object.
        """
        text, meta = self.generate(
            prompt,
            system=system,
            model=model,
            temperature=temperature,
            json_mode=True,
            num_predict=num_predict,
            num_ctx=num_ctx,
        )
        parsed = parse_json_loose(text) if text else None
        if parsed is None and meta.get("ok") and not self.out_of_budget():
            # The unconstrained retry doubles this call's cost, so it only runs
            # while there is budget to pay for it. A repairable object is still
            # worth a second pass; two minutes of waiting is not.
            text2, meta2 = self.generate(
                prompt,
                system=system,
                model=model,
                temperature=temperature,
                json_mode=False,
                num_predict=num_predict,
                num_ctx=num_ctx,
            )
            parsed = parse_json_loose(text2) if text2 else None
            meta2["degraded_to_unconstrained"] = True
            meta2["constrained_attempt_ok"] = True
            return parsed, meta2
        meta["parsed"] = parsed is not None
        return parsed, meta

    def text_call(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        num_predict: int = 1200,
        num_ctx: Optional[int] = None,
    ) -> Tuple[Optional[str], dict]:
        return self.generate(
            prompt,
            system=system,
            model=model,
            temperature=temperature,
            json_mode=False,
            num_predict=num_predict,
            num_ctx=num_ctx,
        )


_ollama: Optional[Ollama] = None


def get_ollama() -> Ollama:
    global _ollama
    if _ollama is None:
        _ollama = Ollama()
    return _ollama
