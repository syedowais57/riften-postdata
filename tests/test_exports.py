"""Tests for the filtering rules, because the filtering is the product.

The UI is easy to eyeball. The exports are not: a wrong rule here silently ships
bad training data, and nothing downstream complains until a model gets worse. So
every exclusion rule gets a test that constructs the exact trace it is meant to
catch, and the two rules with real logic behind them, longest-per-session and the
preference trust order, get tests for the behaviour rather than the mechanism.

    python -m pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.exports import Reason, build_preference, build_sft          # noqa: E402
from app.schema import ToolCall, Trace                               # noqa: E402


def trace(tid: str, session: str = "s1", *, response: str | None = "an answer",
          turns: int = 2, status: int = 200, finish: str = "stop",
          feedback: str | None = None, retry_of: str | None = None,
          continuation: str | None = None, tools: list[ToolCall] | None = None,
          ts: str = "2026-08-10T09:00:00+00:00") -> Trace:
    """A trace with sensible defaults, so each test names only what it is about."""
    messages = [{"role": "system", "content": "sys"}]
    for i in range(turns):
        messages.append({"role": "user", "content": f"question {i}"})
    return Trace(
        id=tid, session_id=session, ts=ts, model="gpt-4o-mini",
        system_prompt="sys", messages=messages, response=response,
        tool_calls=tools or [], finish_reason=finish, status=status,
        prompt_tokens=100, completion_tokens=20, cost_usd=0.001, latency_ms=900,
        feedback=feedback, retry_of=retry_of, continuation=continuation,
    )


def reasons(ex) -> dict[str, int]:
    return dict(ex.counts)


# --------------------------------------------------------------------------
# SFT exclusions, one test per rule
# --------------------------------------------------------------------------

def test_non_2xx_is_excluded():
    rows, ex = build_sft([trace("a", status=500, finish="error", response=None)])
    assert rows == []
    assert reasons(ex)[Reason.NON_2XX] == 1


def test_truncated_is_excluded():
    rows, ex = build_sft([trace("a", finish="length")])
    assert rows == []
    assert reasons(ex)[Reason.TRUNCATED] == 1


def test_weak_feedback_is_excluded():
    rows, ex = build_sft([trace("a", feedback="weak")])
    assert rows == []
    assert reasons(ex)[Reason.FEEDBACK_WEAK] == 1


def test_rejected_continuation_is_excluded():
    rows, ex = build_sft([trace("a", continuation="rejected")])
    assert rows == []
    assert reasons(ex)[Reason.CONTINUATION_REJECTED] == 1


def test_failed_tool_call_is_excluded():
    bad = ToolCall(name="run_sql", arguments={}, ok=False, error="timeout")
    rows, ex = build_sft([trace("a", tools=[bad])])
    assert rows == []
    assert reasons(ex)[Reason.TOOL_ERROR] == 1


def test_superseded_answer_is_excluded_but_the_retry_survives():
    first = trace("a", turns=2)
    retry = trace("b", turns=2, retry_of="a", ts="2026-08-10T09:01:00+00:00")
    rows, ex = build_sft([first, retry])
    assert [r["metadata"]["trace_id"] for r in rows] == ["b"]
    assert reasons(ex)[Reason.SUPERSEDED] == 1


# --------------------------------------------------------------------------
# The rule that shapes the output most
# --------------------------------------------------------------------------

def test_only_the_longest_conversation_per_session_is_kept():
    """Clients replay the transcript, so the longest trace contains the rest."""
    short = trace("a", session="s1", turns=2)
    mid = trace("b", session="s1", turns=4, ts="2026-08-10T09:01:00+00:00")
    long = trace("c", session="s1", turns=6, ts="2026-08-10T09:02:00+00:00")
    rows, ex = build_sft([short, mid, long])
    assert len(rows) == 1
    assert rows[0]["metadata"]["trace_id"] == "c"
    assert reasons(ex)[Reason.NOT_LONGEST] == 2


def test_sessions_do_not_compete_with_each_other():
    rows, _ = build_sft([
        trace("a", session="s1", turns=6),
        trace("b", session="s2", turns=2),
    ])
    assert {r["metadata"]["session_id"] for r in rows} == {"s1", "s2"}


def test_longest_is_chosen_from_survivors_not_from_everything():
    """A long truncated trace must not beat a shorter clean one."""
    long_but_truncated = trace("a", session="s1", turns=8, finish="length")
    short_but_clean = trace("b", session="s1", turns=3)
    rows, _ = build_sft([long_but_truncated, short_but_clean])
    assert [r["metadata"]["trace_id"] for r in rows] == ["b"]


def test_exported_conversation_ends_with_the_assistant_reply():
    rows, _ = build_sft([trace("a", turns=2, response="the answer")])
    msgs = rows[0]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[-1] == {"role": "assistant", "content": "the answer"}


def test_metadata_is_attached_to_every_row():
    rows, _ = build_sft([trace("a")])
    meta = rows[0]["metadata"]
    for key in ("model", "total_tokens", "cost_usd", "latency_ms", "feedback"):
        assert key in meta


# --------------------------------------------------------------------------
# Preference pairs
# --------------------------------------------------------------------------

def test_retry_makes_a_pair_with_the_retry_as_chosen():
    losing = trace("a", response="vague non-answer")
    winning = trace("b", response="specific answer", retry_of="a")
    pairs, _ = build_preference([losing, winning])
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "specific answer"
    assert pairs[0]["rejected"] == "vague non-answer"
    assert pairs[0]["metadata"]["signal"] == "retry"


def test_identical_retry_is_not_a_preference_pair():
    """Rejecting a good answer and getting it back again teaches nothing."""
    first = trace("a", response="same text")
    retry = trace("b", response="same text", retry_of="a")
    pairs, ex = build_preference([first, retry])
    assert pairs == []
    assert reasons(ex)[Reason.IDENTICAL] == 1


def test_identical_check_ignores_whitespace_differences():
    first = trace("a", response="same   text\n")
    retry = trace("b", response="same text", retry_of="a")
    pairs, ex = build_preference([first, retry])
    assert pairs == []
    assert reasons(ex)[Reason.IDENTICAL] == 1


def test_weak_feedback_pairs_against_a_strong_answer():
    weak = trace("a", response="hedge", feedback="weak")
    strong = trace("b", response="real answer", feedback="strong")
    pairs, _ = build_preference([weak, strong])
    assert len(pairs) == 1
    assert pairs[0]["metadata"]["signal"] == "feedback"
    assert pairs[0]["chosen"] == "real answer"


def test_a_trace_is_used_in_only_one_pair():
    """Otherwise a single strong answer would anchor every weak one."""
    losing = trace("a", response="bad")
    winning = trace("b", response="good", retry_of="a", feedback="strong")
    other_weak = trace("c", response="also bad", feedback="weak")
    pairs, _ = build_preference([losing, winning, other_weak])
    assert len(pairs) == 1
    assert pairs[0]["metadata"]["signal"] == "retry"


def test_retry_wins_over_continuation_for_the_same_trace():
    losing = trace("a", response="bad", continuation="rejected")
    winning = trace("b", response="good", retry_of="a", continuation="accepted")
    pairs, _ = build_preference([losing, winning])
    assert [p["metadata"]["signal"] for p in pairs] == ["retry"]


def test_unpairable_weak_answer_is_reported_not_dropped_silently():
    lonely = trace("a", response="hedge", feedback="weak")
    pairs, ex = build_preference([lonely])
    assert pairs == []
    assert reasons(ex)[Reason.NO_PAIR] == 1


def test_pair_carries_metadata_for_both_sides():
    pairs, _ = build_preference([
        trace("a", response="bad"),
        trace("b", response="good", retry_of="a"),
    ])
    meta = pairs[0]["metadata"]
    assert meta["chosen"]["trace_id"] == "b"
    assert meta["rejected"]["trace_id"] == "a"
    assert "cost_usd" in meta["chosen"] and "cost_usd" in meta["rejected"]


# --------------------------------------------------------------------------
# The accounting itself
# --------------------------------------------------------------------------

def test_every_input_trace_is_either_exported_or_explained():
    """The whole point of the exclusion report. Nothing may vanish quietly."""
    traces = [
        trace("a", session="s1", turns=4),
        trace("b", session="s1", turns=2),
        trace("c", session="s2", status=503, finish="error", response=None),
        trace("d", session="s3", finish="length"),
        trace("e", session="s4", feedback="weak"),
    ]
    rows, ex = build_sft(traces)
    assert len(rows) + ex.total == len(traces)
