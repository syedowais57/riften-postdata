"""FastAPI app: browse traces, inspect one, run the exports.

Server rendered. There is no client state worth synchronising here, filters are
just query parameters, and a page of HTML arrives faster than a bundle plus a
fetch. The only JavaScript is a keyboard shortcut and the filter form auto-submit.

    python -m app.main            # or: uvicorn app.main:app --reload
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import exports, store

BASE = Path(__file__).resolve().parent
app = FastAPI(title="Riften post-training data platform")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")

conn = store.connect()


def _ensure_data() -> None:
    """Ingest on first boot so a fresh clone shows something."""
    try:
        conn.execute("SELECT 1 FROM traces LIMIT 1").fetchone()
    except Exception:
        src = Path("data/traces.jsonl")
        if src.exists():
            n = store.ingest(conn, store.read_jsonl(src))
            print(f"ingested {n} traces from {src}")


_ensure_data()


def _fmt(value: float, places: int = 2) -> str:
    return f"{value:,.{places}f}"


templates.env.filters["fmt"] = _fmt
templates.env.filters["fromjson"] = json.loads
templates.env.filters["shorten"] = lambda s, n=90: (
    (s[: n - 1] + "…") if s and len(s) > n else (s or ""))


@app.get("/", response_class=HTMLResponse)
def traces_view(request: Request):
    p = dict(request.query_params)
    page = max(1, int(p.get("page", 1)))
    per = 50
    rows, total = store.query(conn, p, limit=per, offset=(page - 1) * per)
    return templates.TemplateResponse("traces.html", {
        "request": request,
        "rows": rows,
        "total": total,
        "page": page,
        "pages": max(1, (total + per - 1) // per),
        "params": p,
        "summary": store.summary(conn, p),
        "models": store.models(conn),
        "by_model": store.by_model(conn),
        "qs": "&".join(f"{k}={v}" for k, v in p.items() if k != "page" and v),
    })


@app.get("/trace/{trace_id}", response_class=HTMLResponse)
def trace_view(request: Request, trace_id: str):
    row = store.get(conn, trace_id)
    if row is None:
        return HTMLResponse("<h1>404</h1><p>No trace with that id.</p>", status_code=404)
    d = dict(row)
    d["messages"] = json.loads(d["messages"])
    d["tool_calls"] = json.loads(d["tool_calls"])
    siblings = store.session_traces(conn, d["session_id"])
    return templates.TemplateResponse("trace.html", {
        "request": request, "t": d, "siblings": siblings,
    })


@app.get("/exports", response_class=HTMLResponse)
def exports_view(request: Request):
    report_path = exports.EXPORT_DIR / "exclusions.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) \
        if report_path.exists() else None
    return templates.TemplateResponse("exports.html", {
        "request": request, "report": report,
    })


@app.post("/exports/run")
def run_exports():
    exports.write_all(store.all_traces(conn))
    return RedirectResponse("/exports", status_code=303)


@app.get("/exports/{name}")
def download(name: str):
    allowed = {"sft.jsonl", "preference.jsonl", "exclusions.json"}
    if name not in allowed:
        return HTMLResponse("not found", status_code=404)
    path = exports.EXPORT_DIR / name
    if not path.exists():
        return HTMLResponse("run the exports first", status_code=404)
    return FileResponse(path, filename=name, media_type="application/octet-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
