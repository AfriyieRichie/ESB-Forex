"""Setup Dashboard — FastAPI backend.

Signal list -> click a signal -> the system draws the setup (zone, the two
touches, the trigger candle, entry/stop/target) and shows the checklist -> you
apply your read and log it. The detector encodes the mechanical checklist; the
chart is drawn server-side by the same renderer used everywhere in the project.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import scanner

BASE = Path(__file__).resolve().parent
CHARTS = BASE / "charts"
CHARTS.mkdir(exist_ok=True)
JOURNAL = BASE / "journal.json"
SETTINGS = BASE / "settings.json"

app = FastAPI(title="Setup Dashboard")


def _read(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return default
    return default


def _write(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@app.get("/api/signals")
def signals(rescan: bool = False):
    if rescan or scanner.cache()["scanned_at"] is None:
        scanner.scan()
    c = scanner.cache()
    return {
        "signals": scanner.signal_dicts(),
        "latest": c["latest"].isoformat() if c["latest"] else None,
        "scanned_at": c["scanned_at"].isoformat() if c["scanned_at"] else None,
    }


@app.get("/api/chart/{sid}.png")
def chart(sid: str):
    if scanner.cache()["scanned_at"] is None:
        scanner.scan()
    if sid not in scanner.cache()["by_id"]:
        return JSONResponse({"error": "unknown signal"}, status_code=404)
    path = CHARTS / f"{sid}.png"
    if not path.exists():
        scanner.render_chart(sid, path)
    return FileResponse(path, media_type="image/png")


@app.get("/api/settings")
def get_settings():
    return _read(SETTINGS, {"equity": 10000, "risk": 1.0})


@app.post("/api/settings")
async def set_settings(request: Request):
    body = await request.json()
    cur = _read(SETTINGS, {"equity": 10000, "risk": 1.0})
    cur.update({k: body[k] for k in ("equity", "risk") if k in body})
    _write(SETTINGS, cur)
    return cur


@app.get("/api/journal")
def get_journal():
    return _read(JOURNAL, [])


@app.post("/api/log")
async def log_trade(request: Request):
    trade = await request.json()
    trade["id"] = int(time.time() * 1000)
    trade.setdefault("status", "open")
    trade.setdefault("r", None)
    j = _read(JOURNAL, [])
    j.insert(0, trade)
    _write(JOURNAL, j)
    return trade


@app.post("/api/close/{tid}")
async def close_trade(tid: int, request: Request):
    body = await request.json()
    outcome = body.get("outcome")
    j = _read(JOURNAL, [])
    for t in j:
        if t["id"] == tid:
            if outcome == "win":
                t["status"], t["r"] = "win", t.get("rr") or 1
            elif outcome == "loss":
                t["status"], t["r"] = "loss", -1
            elif outcome == "be":
                t["status"], t["r"] = "be", 0
            elif outcome == "reopen":
                t["status"], t["r"] = "open", None
    _write(JOURNAL, j)
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


app.mount("/", StaticFiles(directory=BASE / "static"), name="static")
