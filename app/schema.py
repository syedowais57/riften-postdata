"""The shape of a router trace, and the rules for reading one.

A trace is one request through the router. The router sees the request as the
client sent it, so `messages` is the full transcript the client replayed, not just
the newest user turn. That single fact drives most of the export logic later on:
the last request in a session already contains every earlier turn, which is why
the exporter keeps the longest conversation per session instead of stitching turns
together itself.

Quality signals arrive from two different places and it matters which is which.
`feedback` is explicit, a human pressed weak / ok / strong. `retry_of` and
`continuation` are implicit, inferred from what the client did next. Implicit
signals are noisier but far more plentiful, so both are kept and labelled.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

Feedback = Literal["weak", "ok", "strong"]
Continuation = Literal["accepted", "rejected"]
FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "error"]


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    # Tools fail in production. A trace with a failed tool call is still a
    # legitimate trace, it just should not become a training example.
    ok: bool = True
    error: str | None = None


@dataclass
class Trace:
    id: str
    session_id: str
    ts: str                      # ISO 8601, UTC
    model: str

    system_prompt: str
    messages: list[dict[str, Any]]      # full transcript as received
    response: str | None                # assistant text, None when the call failed

    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: FinishReason = "stop"
    status: int = 200                   # HTTP status the upstream returned

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0

    feedback: Feedback | None = None
    retry_of: str | None = None         # this trace re-ran an earlier trace id
    continuation: Continuation | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def truncated(self) -> bool:
        """The model ran out of room. The answer is cut mid-thought."""
        return self.finish_reason == "length"

    @property
    def failed(self) -> bool:
        return self.status >= 300 or self.finish_reason == "error"

    @property
    def tool_error(self) -> bool:
        return any(not t.ok for t in self.tool_calls)

    @property
    def turn_count(self) -> int:
        return len(self.messages)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        d["truncated"] = self.truncated
        d["failed"] = self.failed
        d["tool_error"] = self.tool_error
        return d


def from_dict(d: dict[str, Any]) -> Trace:
    tools = [ToolCall(**t) for t in d.get("tool_calls") or []]
    # Derived fields are written into the jsonl for convenience, but they are
    # recomputed from source on read so a hand-edited file cannot lie to us.
    known = {
        "id", "session_id", "ts", "model", "system_prompt", "messages", "response",
        "finish_reason", "status", "prompt_tokens", "completion_tokens",
        "cost_usd", "latency_ms", "feedback", "retry_of", "continuation",
    }
    return Trace(tool_calls=tools, **{k: v for k, v in d.items() if k in known})


# Per-million-token pricing, used by the generator and shown in the UI. Rounded
# published rates; exact figures do not matter for a synthetic corpus, the point
# is that models cost visibly different amounts.
MODEL_PRICING = {
    "gpt-4o":            {"in": 2.50, "out": 10.00},
    "gpt-4o-mini":       {"in": 0.15, "out": 0.60},
    "claude-sonnet-4":   {"in": 3.00, "out": 15.00},
    "claude-haiku-3.5":  {"in": 0.80, "out": 4.00},
    "llama-3.3-70b":     {"in": 0.23, "out": 0.40},
}


def price(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = MODEL_PRICING.get(model)
    if not p:
        return 0.0
    return round(prompt_tokens / 1e6 * p["in"] + completion_tokens / 1e6 * p["out"], 6)
