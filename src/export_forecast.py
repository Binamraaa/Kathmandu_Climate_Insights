"""Write the app's forecast file: docs/forecast.json.

Run every evening (the GitHub Action in .github/workflows/forecast.yml does this):
    python -m src.export_forecast
Test offline from recorded data:
    python -m src.export_forecast --replay 2025-03-15
"""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.model_io import load_rain_models, load_temp_models
from src.predict import ROOT, forecast, load_mos_v2

OUT = ROOT / "docs" / "forecast.json"
NPT = timezone(timedelta(hours=5, minutes=45))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", help="YYYY-MM-DD: build from recorded data instead of live data")
    args = ap.parse_args()
    temp, rain, mos = load_temp_models("champion_v2"), load_rain_models("rain_v1"), load_mos_v2()

    if args.replay:
        from src.features import load_station_csv
        from src.train_mos import load_obs
        try:
            obs = load_obs(verbose=False)
        except SystemExit:  # no GSOD file yet: DHM record only
            obs = load_station_csv(ROOT / "data" / "df_analysis.csv")
        daily = obs.loc[: pd.Timestamp(args.replay)].tail(60)
        fc, source = forecast(daily, temp, rain, mos, live=False), "recorded station data (replay)"
    else:
        from src.live_input import apply_calibration, fetch_recent, load_calibration
        cal = load_calibration()
        daily = apply_calibration(fetch_recent(45), cal)
        fc = forecast(daily, temp, rain, mos, live=True)
        source = "Open-Meteo" + (" (bias-corrected)" if cal else " (raw)")

    fc["input_source"] = source
    fc["generated_at"] = datetime.now(NPT).isoformat(timespec="minutes")
    fc["location"] = "Kathmandu (Tribhuvan Airport)"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(fc, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.replay:  # replays are tests, not real forecasts: keep them out of the track record
        try:
            from src.verify import fetch_reference, log_forecast
            log_forecast(fc, fetch_reference())
        except Exception as e:  # noqa: BLE001 — logging must never break the app update
            print(f"  (verification log not updated: {e})")
    d = fc["days"][0]
    print(f"Wrote {OUT}: {d['date']} max {d['max_temp_c']}°C, min {d['min_temp_c']}°C, "
          f"rain {d['rain_prob']:.0%} [{d['model']}]")


if __name__ == "__main__":
    main()
