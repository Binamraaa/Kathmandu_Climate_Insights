"""Compare Open-Meteo with the DHM station record and save a bias correction.

Run once (needs internet):  python -m src.calibrate_openmeteo
Writes models/openmeteo_calibration.json, which predict.py uses automatically.
"""
import json

import numpy as np
import pandas as pd

from src.features import load_station_csv, WET_DAY_MM
from src.live_input import CALIBRATION_PATH, fetch_history

ROOT = CALIBRATION_PATH.parents[1]
START = "2022-01-01"  # historical-forecast data is most complete from ~2022


def main():
    station = load_station_csv(ROOT / "data" / "df_analysis.csv")
    end = station.index.max().strftime("%Y-%m-%d")
    print(f"Fetching Open-Meteo {START} → {end} …")
    om = fetch_history(START, end)
    both = station.join(om, rsuffix="_om", how="inner").dropna()
    print(f"{len(both)} overlapping days")

    cal = {"period": [START, end], "n_days": int(len(both))}
    for k, col in [("max", "Max_Temp_C"), ("min", "Min_Temp_C")]:
        diff = both[col] - both[f"{col}_om"]
        by_month = diff.groupby(both.index.month).mean()
        cal[f"{k}_bias_by_month"] = {str(m): round(float(v), 3) for m, v in by_month.items()}
        corrected = both[f"{col}_om"] + both.index.month.map(by_month).values
        print(f"{col}: MAE raw {diff.abs().mean():.2f} °C → corrected {(both[col] - corrected).abs().mean():.2f} °C "
              f"| mean bias {diff.mean():+.2f} °C")

    # Precip: pick the Open-Meteo amount whose exceedance frequency matches the station's wet-day frequency
    station_wet_rate = (both["Precipitation_mm"] >= WET_DAY_MM).mean()
    om_threshold = float(np.quantile(both["Precipitation_mm_om"], 1 - station_wet_rate))
    cal["precip_scale"] = round(WET_DAY_MM / om_threshold, 3) if om_threshold > 0 else 1.0
    agree = ((both["Precipitation_mm_om"] * cal["precip_scale"] >= WET_DAY_MM) == (both["Precipitation_mm"] >= WET_DAY_MM)).mean()
    print(f"Precip: station wet-day rate {station_wet_rate:.0%}, scale {cal['precip_scale']}, wet/dry agreement {agree:.0%}")

    CALIBRATION_PATH.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(f"Saved {CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
