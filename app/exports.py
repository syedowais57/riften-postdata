"""Turn router traffic into training data, and account for everything dropped.

Two exports, and one report explaining the difference between what went in and
what came out. The report is not decoration. Post-training data is mostly a
filtering problem, and a filter you cannot inspect is a filter you cannot trust,
so every excluded row is counted against a named reason.

SFT
---
One conversation per line, OpenAI chat format. Three rules from the spec:

  * drop rejected answers, non-2xx, and truncated rows
  * agent clients replay the full transcript each turn, so the last request in a
    session already contains every earlier turn
  * therefore keep one conversation per session, the longest

The second point is the one that bites. Exporting every trace would emit the same
conversation four or five times with a growing tail, which over-weights the early
turns of long sessions and teaches the model that openings matter more than they
do. Picking the longest surviving trace per session gives one clean copy.

"Rejected" is read as three things, because the router records rejection three
ways: an explicit weak rating, a continuation the client refused, and a turn that
was superseded by a retry. A retried answer is by definition the one the user did
not want.

Preference
----------
Pairs of (chosen, rejected) for the same prompt. Three sources, in descending
order of how much I trust them:

  * retry     the client re-ran the same prompt and kept the second answer
  * feedback  a weak turn paired against a strong or ok turn in the same session
  * continuation  the client refused to continue from the answer

Emitted DPO-style, since the spec fixed the SFT format but not this one. Each line
carries prompt, chosen, rejected and the metadata for both sides, so a trainer
that wants a different layout can remap without going back to the traces.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from app.schema import Trace

EXPORT_DIR = Path("exports")


# Reasons are constants rather than free text so the report cannot drift from
# the code that produces it.
class Reason:
    NON_2XX = "non-2xx response from upstream"
    TRUNCATED = "truncated, hit the token ceiling"
    NO_RESPONSE = "no assistant text in the trace"
    TOOL_ERROR = "a tool call failed during the turn"
    FEEDBACK_WEAK = "explicitly rated weak"
    CONTINUATION_REJECTED = "client rejected the continuation"
    SUPERSEDED = "superseded by a retry of the same prompt"
    NOT_LONGEST = "shorter duplicate of another trace in the same session"
    NO_PAIR = "no counterpart answer to pair it against"
    IDENTICAL = "retry returned the same text, no preference to learn"


@dataclass
class Exclusions:
    """Counts by reason, plus the ids, so a claim can be checked."""
    counts: Counter = field(default_factory=Counter)
    ids: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def drop(self, trace_id: str, reason: str) -> None:
        self.counts[reason] += 1
        # Cap the stored ids. The counts are the point; the ids are for spot
        # checks, and a few hundred per reason is already more than anyone reads.
        if len(self.ids[reason]) < 200:
            self.ids[reason].append(trace_id)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_excluded": self.total,
            "by_reason": [
                {"reason": r, "count": c, "sample_ids": self.ids[r][:10]}
                for r, c in self.counts.most_common()
            ],
        }


def _meta(t: Trace) -> dict[str, Any]:
    return {
        "trace_id": t.id,
        "session_id": t.session_id,
        "model": t.model,
        "prompt_tokens": t.prompt_tokens,
        "completion_tokens": t.completion_tokens,
        "total_tokens": t.total_tokens,
        "cost_usd": round(t.cost_usd, 6),
        "latency_ms": t.latency_ms,
        "feedback": t.feedback,
        "finish_reason": t.finish_reason,
        "ts": t.ts,
    }


def _superseded_ids(traces: Iterable[Trace]) -> set[str]:
    """Ids that a later trace re-ran. Those answers lost."""
    return {t.retry_of for t in traces if t.retry_of}


def build_sft(traces: list[Trace]) -> tuple[list[dict], Exclusions]:
    ex = Exclusions()
    superseded = _superseded_ids(traces)

    kept_per_session: dict[str, Trace] = {}
    for t in traces:
        if t.status >= 300 or t.finish_reason == "error":
            ex.drop(t.id, Reason.NON_2XX); continue
        if t.truncated:
            ex.drop(t.id, Reason.TRUNCATED); continue
        if not t.response:
            ex.drop(t.id, Reason.NO_RESPONSE); continue
        if t.id in superseded:
            ex.drop(t.id, Reason.SUPERSEDED); continue
        if t.feedback == "weak":
            ex.drop(t.id, Reason.FEEDBACK_WEAK); continue
        if t.continuation == "rejected":
            ex.drop(t.id, Reason.CONTINUATION_REJECTED); continue
        if t.tool_error:
            # A failed tool call means the assistant answered around a hole. It
            # is not wrong exactly, but it is not a example worth imitating.
            ex.drop(t.id, Reason.TOOL_ERROR); continue

        best = kept_per_session.get(t.session_id)
        if best is None or t.turn_count > best.turn_count:
            if best is not None:
                ex.drop(best.id, Reason.NOT_LONGEST)
            kept_per_session[t.session_id] = t
        else:
            ex.drop(t.id, Reason.NOT_LONGEST)

    lines = []
    for t in sorted(kept_per_session.values(), key=lambda x: x.ts):
        # messages already begins with the system prompt as the client sent it.
        msgs = [dict(m) for m in t.messages]
        if not msgs or msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": t.system_prompt}] + msgs
        msgs.append({"role": "assistant", "content": t.response})
        lines.append({"messages": msgs, "metadata": _meta(t)})
    return lines, ex


def _norm(text: str | None) -> str:
    """Whitespace-insensitive compare, so formatting noise is not a difference."""
    return " ".join((text or "").split())


def _prompt_of(t: Trace) -> str:
    """The last user turn. Two answers to the same prompt are comparable."""
    for m in reversed(t.messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def build_preference(traces: list[Trace]) -> tuple[list[dict], Exclusions]:
    ex = Exclusions()
    by_id = {t.id: t for t in traces}
    usable = [t for t in traces if t.response and t.status < 300]
    for t in traces:
        if t not in usable:
            ex.drop(t.id, Reason.NON_2XX if t.status >= 300 else Reason.NO_RESPONSE)

    pairs: list[dict] = []
    paired: set[str] = set()

    # 1. Retries. The strongest signal available: same prompt, the client threw
    #    the first answer away and kept the second.
    for t in usable:
        if not t.retry_of:
            continue
        loser = by_id.get(t.retry_of)
        if not loser or not loser.response:
            continue
        # A user can reject a perfectly good answer and retry, and the second
        # call can return the same text. That is a real thing the router sees,
        # and it is not a preference: training on it teaches nothing and the
        # identical pair would quietly skew the loss. Drop it, and say so.
        if _norm(t.response) == _norm(loser.response):
            ex.drop(t.id, Reason.IDENTICAL)
            paired.update({t.id, loser.id})
            continue
        pairs.append(_pair(_prompt_of(t), t, loser, "retry"))
        paired.update({t.id, loser.id})

    # 2. Explicit feedback within a session. Pair a weak answer against a strong
    #    or ok one. Same session keeps the system prompt and context comparable.
    by_session: dict[str, list[Trace]] = defaultdict(list)
    for t in usable:
        by_session[t.session_id].append(t)
    for sess in by_session.values():
        weak = [t for t in sess if t.feedback == "weak" and t.id not in paired]
        good = [t for t in sess if t.feedback in ("strong", "ok") and t.id not in paired]
        for w in weak:
            match = next((g for g in good if g.id not in paired
                          and _norm(g.response) != _norm(w.response)), None)
            if not match:
                ex.drop(w.id, Reason.NO_PAIR); continue
            pairs.append(_pair(_prompt_of(w), match, w, "feedback"))
            paired.update({w.id, match.id})

    # 3. Continuation. Weaker than the above, so it only fires on what is left.
    for sess in by_session.values():
        rejected = [t for t in sess if t.continuation == "rejected" and t.id not in paired]
        accepted = [t for t in sess if t.continuation == "accepted" and t.id not in paired]
        for r in rejected:
            match = next((a for a in accepted if a.id not in paired
                          and _norm(a.response) != _norm(r.response)), None)
            if not match:
                ex.drop(r.id, Reason.NO_PAIR); continue
            pairs.append(_pair(_prompt_of(r), match, r, "continuation"))
            paired.update({r.id, match.id})

    pairs.sort(key=lambda p: p["metadata"]["chosen"]["ts"])
    return pairs, ex


def _pair(prompt: str, chosen: Trace, rejected: Trace, source: str) -> dict:
    return {
        "prompt": prompt,
        "chosen": chosen.response,
        "rejected": rejected.response,
        "metadata": {
            "signal": source,
            "session_id": chosen.session_id,
            "chosen": _meta(chosen),
            "rejected": _meta(rejected),
        },
    }


def write_all(traces: list[Trace], out_dir: Path = EXPORT_DIR) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sft, sft_ex = build_sft(traces)
    pref, pref_ex = build_preference(traces)

    with (out_dir / "sft.jsonl").open("w", encoding="utf-8") as fh:
        for line in sft:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    with (out_dir / "preference.jsonl").open("w", encoding="utf-8") as fh:
        for line in pref:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")

    report = {
        "input_traces": len(traces),
        "input_sessions": len({t.session_id for t in traces}),
        "sft": {
            "written": len(sft),
            "excluded": sft_ex.as_dict(),
        },
        "preference": {
            "written": len(pref),
            "by_signal": dict(Counter(p["metadata"]["signal"] for p in pref)),
            "excluded": pref_ex.as_dict(),
        },
    }
    (out_dir / "exclusions.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report
