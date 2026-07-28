"""Live scan engine — the cron's payload.

One command does the whole cycle: refresh the live edge from free Dukascopy data,
run the checklist detector, publish the static dashboard artifacts (signals.json +
pre-rendered chart PNGs), and fire a Telegram alert for any setup that is NEW since
the last run. The GitHub Action runs this on a schedule.

Env (set as GitHub Action secrets):
    TELEGRAM_BOT_TOKEN   from @BotFather
    TELEGRAM_CHAT_ID     your chat id (from @userinfobot)

    uv run python scripts/live_scan.py               # refresh + scan + publish + alert
    uv run python scripts/live_scan.py --no-alert    # everything except sending alerts
    uv run python scripts/live_scan.py --no-refresh  # use data already on disk
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "setup-dashboard"))
import scanner  # noqa: E402

from fxlab import instruments  # noqa: E402
from fxlab.data import Client, download_h1, resample, save_bars  # noqa: E402
from fxlab.data.store import RAW_DIR  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# Local runs read the Telegram creds from a gitignored .env; in GitHub Actions
# the same names come from repo secrets, so this is a no-op there.
load_dotenv(ROOT / ".env")

PUBLIC = ROOT / "vercel-deploy" / "public"
CHARTS = PUBLIC / "charts"
SEEN = ROOT / "vercel-deploy" / "seen.json"
LOOKBACK_YEARS = 2


def refresh_data() -> None:
    """Top up the live edge from free Dukascopy data.

    History months are cached on disk and never re-fetched. The current and
    previous month DO keep filling in as bars close, so we drop just those two
    from the cache to force a fresh pull — the rest stays cached, so this is a
    handful of requests, not a full re-download.
    """
    today = dt.date.today()
    start = dt.date(today.year - LOOKBACK_YEARS, 1, 1)
    prev = (today.replace(day=1) - dt.timedelta(days=1))  # last day of prev month
    client = Client(RAW_DIR)
    for inst in instruments.BASKET:
        for d in (today, prev):
            edge = RAW_DIR / inst.symbol / str(d.year) / f"{d.month:02d}_hour.bi5"
            if edge.exists():
                edge.unlink()
        h1 = download_h1(client, inst, start, today, progress=False)
        if h1.is_empty():
            continue
        save_bars(h1, inst.symbol, "H1")
        for tf in ("H4", "D1"):
            save_bars(resample(h1, tf), inst.symbol, tf)


def send_telegram(text: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat):
        print("  (telegram not configured — skipping alert)")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": text, "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=20,
        )
        r.raise_for_status()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  telegram send failed: {e}")
        return False


def format_alert(s: dict) -> str:
    arrow = "🟢 LONG" if s["direction"] == "long" else "🔴 SHORT"
    return (
        f"<b>{s['setup'].upper()}</b>  {arrow}\n"
        f"<b>{s['symbol']}</b> · {s['zone_tier']}{' +confluence' if s['confluent'] else ''}\n"
        f"entry <code>{s['entry']}</code>  stop <code>{s['stop']}</code>  target <code>{s['target']}</code>\n"
        f"{s['rr']}R · {s['risk_pips']} pip risk · {s['ts'][:16].replace('T', ' ')} UTC\n"
        f"<i>Radar only — apply your read + checklist before taking.</i>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-alert", action="store_true")
    parser.add_argument("--no-refresh", action="store_true",
                        help="skip the Dukascopy top-up (use data already on disk)")
    args = parser.parse_args()

    if not args.no_refresh:
        print("refreshing live edge from Dukascopy...", flush=True)
        refresh_data()

    print("scanning live market...", flush=True)
    c = scanner.scan()
    sigs = scanner.signal_dicts()
    latest = c["latest"].isoformat() if c["latest"] else None
    data_through = c["data_through"].isoformat() if c["data_through"] else None
    print(f"  {len(sigs)} setups · newest setup {latest} · data through {data_through}", flush=True)

    # publish static artifacts
    if CHARTS.exists():
        shutil.rmtree(CHARTS)
    CHARTS.mkdir(parents=True, exist_ok=True)
    for s in sigs:
        scanner.render_chart(s["id"], CHARTS / f"{s['id']}.png")
    PUBLIC.joinpath("signals.json").write_text(
        json.dumps({"signals": sigs, "latest": latest, "data_through": data_through,
                    "built_at": dt.datetime.now(dt.timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )
    print(f"  published {len(sigs)} setups + charts", flush=True)

    # Alert on genuinely new setups. A setup is only marked "seen" once it has
    # actually been alerted (or when alerts are disabled) — so a failed send is
    # retried next run instead of being silently swallowed. Seen ids are pruned
    # to the ones still present, keeping the file bounded.
    current = {s["id"] for s in sigs}
    seen = (set(json.loads(SEEN.read_text())) if SEEN.exists() else set()) & current
    new = [s for s in sigs if s["id"] not in seen]
    print(f"  {len(new)} new since last run", flush=True)
    for s in new:
        if args.no_alert:
            seen.add(s["id"])
        elif send_telegram(format_alert(s)):
            seen.add(s["id"])
            print(f"    alerted: {s['symbol']} {s['setup']}")
        # else: send failed -> leave unseen so it retries next run
    SEEN.write_text(json.dumps(sorted(seen)), encoding="utf-8")


if __name__ == "__main__":
    main()
