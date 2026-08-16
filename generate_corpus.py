"""Generate a synthetic router corpus.

No production traffic or keys were available, so this fabricates traffic in the
shape the router captures. Two goals, in order:

  1. Every branch the exporter has to handle actually occurs: retries, tool
     errors, truncated answers, non-2xx responses, mixed models, and sessions
     that grow a turn at a time because the client replays the transcript.
  2. It reads like real traffic. Questions get answers about the thing that was
     asked, weak answers are fluent and empty rather than nonsense, and a
     truncated answer stops mid-sentence in a plausible place.

The second goal is why the content lives in app/scenarios.py as whole scenarios
rather than as parallel lists of questions and answers. Drawing them separately
produced traces where someone asked about inference spend and got told about
retry logic, which makes the inspector useless to read.

Seeded, so two runs give the same corpus and any number in the README can be
reproduced.

    python generate_corpus.py --sessions 130 --out data/traces.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.schema import MODEL_PRICING, price
from app.scenarios import SCENARIOS, TOOL_ERRORS

SEED = 11

MODELS = list(MODEL_PRICING)
# Weighted so the corpus looks like routed traffic: a cheap model carries most of
# it, the expensive ones show up on the harder turns and on retries.
MODEL_WEIGHTS = [0.22, 0.34, 0.18, 0.16, 0.10]
STRONG_MODELS = ["gpt-4o", "claude-sonnet-4"]


def make_id() -> str:
    return uuid.uuid4().hex[:12]


def build(rng: random.Random, n_sessions: int) -> list[dict]:
    traces: list[dict] = []
    clock = datetime(2026, 8, 9, 8, 0, tzinfo=timezone.utc)

    for _ in range(n_sessions):
        sc = rng.choice(SCENARIOS)
        session_id = make_id()
        model = rng.choices(MODELS, MODEL_WEIGHTS)[0]
        transcript: list[dict] = [{"role": "system", "content": sc.system}]
        clock += timedelta(minutes=rng.randint(2, 90))

        # Turn 0 is the scenario's own question; later turns come from its
        # follow-ups, so a session stays about one thing.
        turns: list[tuple[str, str, str]] = [(sc.question, sc.good, sc.weak)]
        n_follow = rng.randint(0, len(sc.follow_ups))
        turns += sc.follow_ups[:n_follow]

        for idx, (question, good, weak) in enumerate(turns):
            transcript = transcript + [{"role": "user", "content": question}]
            clock += timedelta(seconds=rng.randint(5, 240))

            # A router switches model mid-session. That is the whole point of it.
            if idx and rng.random() < 0.20:
                model = rng.choices(MODELS, MODEL_WEIGHTS)[0]

            t = emit_turn(rng, session_id, clock, model, sc, transcript, good, weak)
            traces.append(t)
            if t["response"]:
                transcript = transcript + [
                    {"role": "assistant", "content": t["response"]}]

            # A bad answer usually gets retried straight away on a stronger model.
            # That retry is the cleanest preference signal in the corpus.
            bad = (t["finish_reason"] != "stop" or t["feedback"] == "weak"
                   or t["continuation"] == "rejected")
            if bad and rng.random() < 0.62:
                clock += timedelta(seconds=rng.randint(3, 40))
                retry = emit_turn(rng, session_id, clock, rng.choice(STRONG_MODELS),
                                  sc, transcript, good, weak, force_good=True)
                retry["retry_of"] = t["id"]
                traces.append(retry)
                transcript = transcript + [
                    {"role": "assistant", "content": retry["response"]}]
    return traces


def emit_turn(rng: random.Random, session_id: str, ts: datetime, model: str,
              sc, transcript: list[dict], good: str, weak: str,
              force_good: bool = False) -> dict:
    """One request through the router, with the failure modes real traffic has."""
    roll = 1.0 if force_good else rng.random()

    status, finish = 200, "stop"
    response: str | None
    tool_calls: list[dict] = []
    feedback = None
    continuation = None

    if roll < 0.05:
        status = rng.choice([429, 500, 502, 503])
        finish, response = "error", None
    elif roll < 0.12:
        finish, response = "length", sc.truncated
    elif roll < 0.34:
        response = weak
    else:
        response = good

    if status == 200 and sc.tool and rng.random() < 0.45:
        name, args = sc.tool
        ok = rng.random() > 0.18
        tool_calls.append({"name": name, "arguments": args, "ok": ok,
                           "error": None if ok else rng.choice(TOOL_ERRORS)})
        if finish == "stop" and rng.random() < 0.22:
            finish = "tool_calls"

    # Explicit feedback is sparse, as it always is in practice, and correlates
    # with answer quality rather than being random.
    if status == 200 and rng.random() < 0.34:
        if response == weak:
            feedback = rng.choices(["weak", "ok"], [0.78, 0.22])[0]
        elif finish == "length":
            feedback = "weak"
        else:
            feedback = rng.choices(["strong", "ok", "weak"], [0.55, 0.36, 0.09])[0]

    # Implicit signal: did the client accept the answer and carry on from it?
    if status == 200 and rng.random() < 0.55:
        poor = (response == weak or finish == "length"
                or any(not c["ok"] for c in tool_calls))
        continuation = rng.choices(["rejected", "accepted"],
                                   [0.72, 0.28] if poor else [0.11, 0.89])[0]

    prompt_tokens = sum(len(m["content"]) for m in transcript) // 4 + rng.randint(20, 120)
    completion_tokens = (len(response) // 4 + rng.randint(5, 60)) if response else 0
    latency = int(rng.gauss(1400, 520) + completion_tokens * rng.uniform(4, 11))

    return {
        "id": make_id(),
        "session_id": session_id,
        "ts": ts.isoformat(),
        "model": model,
        "system_prompt": sc.system,
        "messages": [dict(m) for m in transcript],
        "response": response,
        "tool_calls": tool_calls,
        "finish_reason": finish,
        "status": status,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": price(model, prompt_tokens, completion_tokens),
        "latency_ms": max(120, latency),
        "feedback": feedback,
        "retry_of": None,
        "continuation": continuation,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=130)
    ap.add_argument("--out", default="data/traces.jsonl")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    traces = build(random.Random(args.seed), args.sessions)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t) + "\n")

    sessions = len({t["session_id"] for t in traces})
    print(f"wrote {len(traces)} traces across {sessions} sessions -> {out}")
    print("  retries {}   non-2xx {}   truncated {}   tool errors {}".format(
        sum(1 for t in traces if t["retry_of"]),
        sum(1 for t in traces if t["status"] >= 300),
        sum(1 for t in traces if t["finish_reason"] == "length"),
        sum(1 for t in traces if any(not c["ok"] for c in t["tool_calls"]))))


if __name__ == "__main__":
    main()
