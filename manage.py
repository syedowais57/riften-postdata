"""Small CLI for the things you do outside the browser.

    python manage.py ingest                  # jsonl -> sqlite
    python manage.py export                  # write sft, preference, exclusions
    python manage.py stats                   # what is in the corpus
    python manage.py serve                   # run the app

Kept as one file because there are four commands and none of them need options
beyond a path. A package of subcommand modules would be more structure than the
problem has.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app import exports, store


def cmd_ingest(args) -> None:
    conn = store.connect()
    traces = store.read_jsonl(Path(args.traces))
    n = store.ingest(conn, traces)
    print(f"ingested {n} traces from {args.traces}")


def cmd_export(args) -> None:
    conn = store.connect()
    if not _has_rows(conn):
        cmd_ingest(args)
    report = exports.write_all(store.all_traces(conn), Path(args.out))
    sft, pref = report["sft"], report["preference"]
    print(f"input      {report['input_traces']} traces / "
          f"{report['input_sessions']} sessions")
    print(f"sft        {sft['written']} rows, "
          f"{sft['excluded']['total_excluded']} excluded")
    for row in sft["excluded"]["by_reason"]:
        print(f"             {row['count']:>4}  {row['reason']}")
    print(f"preference {pref['written']} pairs  {pref['by_signal']}")
    for row in pref["excluded"]["by_reason"]:
        print(f"             {row['count']:>4}  {row['reason']}")
    print(f"written to {args.out}/")


def cmd_stats(args) -> None:
    conn = store.connect()
    if not _has_rows(conn):
        cmd_ingest(args)
    s = store.summary(conn, {})
    print(f"{s['n']} traces across {s['sessions']} sessions")
    print(f"  cost ${s['cost']:.2f}   tokens {s['tokens']:,}   "
          f"avg latency {s['latency']:.0f} ms")
    print(f"  {s['failed']} non-2xx   {s['truncated']} truncated   "
          f"{s['tool_errors']} tool errors")
    print()
    print(f"  {'model':<20}{'n':>6}{'cost':>10}{'latency':>10}"
          f"{'trunc':>8}{'err':>6}{'strong':>8}{'weak':>6}")
    for m in store.by_model(conn):
        print(f"  {m['model']:<20}{m['n']:>6}{m['cost']:>10.2f}"
              f"{m['latency']:>10.0f}{m['truncated']:>8}{m['failed']:>6}"
              f"{m['strong']:>8}{m['weak']:>6}")


def cmd_serve(args) -> None:
    import uvicorn
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


def _has_rows(conn) -> bool:
    try:
        return conn.execute("SELECT 1 FROM traces LIMIT 1").fetchone() is not None
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traces", default="data/traces.jsonl")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest").set_defaults(fn=cmd_ingest)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)

    e = sub.add_parser("export")
    e.add_argument("--out", default="exports")
    e.set_defaults(fn=cmd_export)

    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(fn=cmd_serve)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
