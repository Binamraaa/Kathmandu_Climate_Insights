"""Write the app's forecast file: docs/forecast.json.

Two runs a day (the GitHub Action in .github/workflows/forecast.yml does this):
  * morning run (before 12:00 Nepal time): uses data up to yesterday and forecasts
    TODAY, tomorrow and the day after. Today's forecast comes from MOS v2.1.
  * evening run (from 12:00): uses today's data and forecasts TOMORROW onward.
The mode follows the Nepal clock automatically, so a delayed run still does the right
thing. Today's forecast from the morning run is kept in the file all day, so the app
can show "today" in the daytime and "tomorrow" in the evening.

    python -m src.export_forecast                 # mode from the clock
    python -m src.export_forecast --mode morning  # force a mode
    python -m src.export_forecast --replay 2025-03-15   # offline test, not logged
"""
import argparse
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.model_io import load_rain_models, load_temp_models
from src.predict import ROOT, forecast, load_mos_v2

OUT = ROOT / "docs" / "forecast.json"
NPT = timezone(timedelta(hours=5, minutes=45))
MAX_DAYS = 4


def merge_with_previous(fc: dict, today: pd.Timestamp, prev: dict | None) -> dict:
    """Keep earlier forecasts for days the new run doesn't cover (e.g. today, from the
    morning run) as long as they are not in the past."""
    if not prev:
        return fc
    new_dates = {d["date"] for d in fc["days"]}
    keep = [d for d in prev.get("days", []) if d["date"] >= str(today.date()) and d["date"] not in new_dates]
    for d in keep:  # files written by older versions lack these fields
        d.setdefault("run", "evening")
        d.setdefault("updated_at", prev.get("generated_at"))
    fc["days"] = sorted(keep + fc["days"], key=lambda d: d["date"])[:MAX_DAYS]
    return fc


def build(daily: pd.DataFrame, mode: str, now: datetime, temp, rain, mos, live: bool) -> dict:
    today = pd.Timestamp(now.date())
    daily = daily[daily.index < today] if mode == "morning" else daily[daily.index <= today]
    fc = forecast(daily, temp, rain, mos, live=live)
    stamp = now.isoformat(timespec="minutes")
    for d in fc["days"]:
        d["run"], d["updated_at"] = mode, stamp
    fc["mode"], fc["generated_at"] = mode, stamp
    return fc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["morning", "evening"], help="default: morning before 12:00 Nepal time, else evening")
    ap.add_argument("--replay", help="YYYY-MM-DD: build from recorded data instead of live data (not logged)")
    args = ap.parse_args()
    temp, rain, mos = load_temp_models("champion_v2"), load_rain_models("rain_v1"), load_mos_v2()

    if args.replay:
        from src.features import load_station_csv
        from src.train_mos import load_obs
        try:
            obs = load_obs(verbose=False)
        except SystemExit:  # no GSOD file yet: DHM record only
            obs = load_station_csv(ROOT / "data" / "df_analysis.csv")
        now = datetime.combine(pd.Timestamp(args.replay).date(), datetime.min.time(), NPT).replace(hour=19)
        mode = args.mode or "evening"
        fc = build(obs.loc[: pd.Timestamp(args.replay)].tail(60), mode, now, temp, rain, mos, live=False)
        fc["input_source"] = "recorded station data (replay)"
    else:
        from src.live_input import apply_calibration, fetch_recent, load_calibration
        now = datetime.now(NPT)
        mode = args.mode or ("morning" if now.hour < 12 else "evening")
        cal = load_calibration()
        fc = build(apply_calibration(fetch_recent(45), cal), mode, now, temp, rain, mos, live=True)
        fc["input_source"] = "Open-Meteo" + (" (bias-corrected)" if cal else " (raw)")
        try:
            from src.extras import attach, fetch_extras
            fc = attach(fc, fetch_extras())
        except Exception as e:  # noqa: BLE001 — extras refine the advice; the forecast stands without them
            print(f"  (UV / wind / air-quality extras unavailable: {e})")
        prev = None
        if OUT.exists():
            try:
                prev = json.loads(OUT.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                prev = None
        fc = merge_with_previous(fc, pd.Timestamp(now.date()), prev)

    fc["location"] = "Kathmandu (Tribhuvan Airport)"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(fc, indent=2, ensure_ascii=False), encoding="utf-8")
    shown = ", ".join(f"{d['date']} {d['max_temp_c']}/{d['min_temp_c']}°C [{d.get('run', '?')}]" for d in fc["days"])
    print(f"Wrote {OUT} ({mode} run): {shown}")

    if not args.replay:  # replays are tests, not real forecasts: keep them out of the track record
        try:
            from src.verify import fetch_reference, log_forecast
            new_only = {**fc, "days": [d for d in fc["days"] if d.get("updated_at") == fc["generated_at"]]}
            log_forecast(new_only, fetch_reference(), run=mode)
        except Exception as e:  # noqa: BLE001 — logging must never break the app update
            print(f"  (verification log not updated: {e})")


if __name__ == "__main__":
    main()