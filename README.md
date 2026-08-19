# Post-training data platform on router traces

Takes raw router traffic, lets you look at it properly, and turns it into SFT and
preference data with a full account of everything that was thrown away.

Built for the Riften project brief. No production traffic or keys, so the corpus
is synthetic but shaped like the real thing.

```bash
pip install -r requirements.txt
python generate_corpus.py --sessions 150   # writes data/traces.jsonl
python manage.py ingest                    # jsonl -> sqlite
python manage.py serve                     # http://127.0.0.1:8000
```

Exports come from the Exports tab, or the CLI:

```bash
python manage.py export     # writes sft.jsonl, preference.jsonl, exclusions.json
python manage.py stats      # per-model cost, latency, truncation, feedback
python -m pytest -q         # 21 tests over the filtering rules
```

## What it does

**Traces.** A filterable table of every request through the router. Filter by
model, feedback, finish reason, minimum cost, minimum latency, free text, and
toggles for truncated, errors, tool errors and retries. Sort by cost, latency,
tokens or turn count. Click any trace to see the transcript exactly as the client
sent it, the tool calls with their failures, and the rest of that session.

**Exports.** Two files plus a report.

**Exclusions.** Every dropped row counted against a named reason, with sample ids
so any claim can be checked against the traces.

## The exports

### SFT — `exports/sft.jsonl`

OpenAI chat format, one conversation per line, with a metadata block carrying
model, tokens, cost, latency and feedback.

Dropped: non-2xx, truncated, no response body, explicitly weak, continuation
rejected, superseded by a retry, and turns where a tool call failed.

The rule that matters most is the last one from the brief: agent clients replay
the full transcript each turn, so the final request in a session already contains
every earlier turn. Exporting every trace would emit the same conversation four or
five times with a growing tail, which over-weights the openings of long sessions.
So the exporter keeps **one conversation per session, the longest that survives
filtering**, and counts the rest as shorter duplicates.

I also drop turns where a tool call failed. Those answers are not wrong exactly,
the model answered around a hole, but they are not behaviour worth imitating.

### Preference — `exports/preference.jsonl`

The brief fixed the SFT format and not this one, so it is DPO-style: `prompt`,
`chosen`, `rejected`, and a metadata block for both sides. Anything wanting a
different layout can remap without going back to the traces.

Three signals, in the order they are trusted:

| Signal | What happened | Why it is ranked there |
|---|---|---|
| `retry` | Client re-ran the same prompt and kept the second answer | Strongest. Same prompt, same session, an explicit do-over |
| `feedback` | A turn rated weak, paired against a strong or ok turn in the same session | Explicit, but the two turns are not the same prompt |
| `continuation` | Client refused to continue from the answer | Weakest. Silence is ambiguous, so it only fires on what is left |

Pairs are formed greedily in that order and a trace is used once, so a strong
signal is never spent on a weak pairing.

**One thing worth calling out.** A user can reject a perfectly good answer and
retry, and the retry can come back with the same text. That looked like a
preference pair and it is not: training on identical chosen and rejected teaches
nothing and quietly skews the loss. They are detected on a whitespace-normalised
compare and counted under *"retry returned the same text, no preference to
learn"*. On the seeded corpus that is 25 of 66 retry candidates, which is enough
to matter.

## Numbers on the seeded corpus

`--sessions 150 --seed 11`, reproducible:

```
308 traces across 150 sessions
  66 retries   10 non-2xx   23 truncated   28 tool errors

SFT            133 rows, 175 excluded
  shorter duplicate of another trace in the same session   63
  superseded by a retry of the same prompt                 44
  truncated, hit the token ceiling                         23
  client rejected the continuation                         15
  a tool call failed during the turn                       14
  non-2xx response from upstream                           10
  explicitly rated weak                                     6

Preference      41 pairs   retry 35, feedback 4, continuation 2
  retry returned the same text, no preference to learn     25
  no counterpart answer to pair it against                 15
```

133 SFT rows from 308 traces is a 43% yield, and that is the honest number. Most
of the loss is structural rather than quality: the duplicate rule alone accounts
for 63 rows, because sessions with five turns contribute one conversation, not
five.

## Tests

```
python -m pytest -q     # 21 passing
```

The UI is easy to eyeball. The exports are not: a wrong rule here ships bad
training data silently, and nothing downstream complains until a model gets
worse. So every exclusion rule has a test that builds the exact trace it is meant
to catch, and the rules with real logic behind them are tested for behaviour
rather than mechanism. The last test asserts the property the whole exclusion
report exists to guarantee: every input trace is either exported or explained,
never quietly dropped.

## Layout

```
generate_corpus.py     synthetic traffic, seeded
manage.py              ingest / export / stats / serve
app/schema.py          what a trace is, and how to read one
app/scenarios.py       corpus content, grouped so answers match questions
app/store.py           SQLite ingest and queries
app/exports.py         SFT, preference, exclusion accounting
app/main.py            FastAPI routes
app/templates/         server-rendered pages
app/static/app.css     the design
tests/test_exports.py  the filtering rules
```

## Choices

**SQLite, not a dataframe.** The UI needs filtering, sorting and counts over a few
thousand rows, which is what SQL is for. It is one file, so there is nothing to
start.

**Server-rendered.** There is no client state worth synchronising. Filters are
query parameters. A page of HTML arrives faster than a bundle plus a fetch, and
the URL stays shareable, which matters for a tool where someone wants to send a
colleague a filtered view.

**Scenarios, not random pairs.** The first generator drew questions and answers
from separate lists and paired them randomly, which produced traces where someone
asked about inference spend and was told about retry logic. Fine for exercising
the export code, useless to read, and the whole point of an inspector is that a
human reads it. Content now lives in whole scenarios, each with its own good
answer, weak answer and truncated variant. The weak answers are deliberately
fluent and empty rather than nonsense, because that is what a preference pair has
to be able to separate.

**Reasons as constants.** Exclusion reasons are class attributes, not strings at
the call site, so the report cannot drift from the code that produces it.

## What I would do next

- Inter-signal agreement: when explicit feedback and continuation disagree on the
  same turn, that turn is worth a look. Right now the stronger signal just wins.
- Dedupe near-identical SFT rows across sessions, not just within one. The
  identical-text check is exact after whitespace normalisation; real traffic will
  need something fuzzier.
- Hold out a validation split by session rather than by row, so the same
  conversation cannot appear on both sides.
- The exclusion report counts rows. It should also report the token and cost mass
  being dropped, since a hundred cheap rows and a hundred expensive ones are not
  the same loss.
