"""Download NOAA GSOD daily data for Kathmandu Airport (WMO 44454, ICAO VNKT).

This is the same airport station as the DHM record, published free by NOAA and
updated to within a few days of today. Run (needs internet):

    python -m src.fetch_gsod            # 1999 → this year, plus comparison with DHM
    python -m src.fetch_gsod --from 2022

Saves data/gsod_kathmandu.csv with the same columns as the station data:
Max_Temp_C, Min_Temp_C, Precipitation_mm (NaN where NOAA marks values missing).
"""
import argparse
import io
from datetime import date

import numpy as np
import pandas as pd
import requests

from src.features import WET_DAY_MM, load_station_csv
from src.model_io import MODELS_DIR

STATION_ID = "44454099999"
URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{sid}.csv"
ROOT = MODELS_DIR.parent
OUT = ROOT / "data" / "gsod_kathmandu.csv"


def parse_gsod(csv_text: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(csv_text), dtype=str)
    raw.columns = [c.strip().upper() for c in raw.columns]
    num = lambda c: pd.to_numeric(raw[c].str.strip(), errors="coerce")
    mx, mn, pr = num("MAX"), num("MIN"), num("PRCP")
    mx[mx >= 9999] = np.nan
    mn[mn >= 9999] = np.nan
    pr[pr >= 99.9] = np.nan
    # PRCP flag "I" = incomplete/no report; treat as missing rather than zero
    if "PRCP_ATTRIBUTES" in raw.columns:
        pr[raw["PRCP_ATTRIBUTES"].str.strip().eq("I")] = np.nan
    df = pd.DataFrame(
        {
            "Max_Temp_C": ((mx - 32) * 5 / 9).round(1).values,
            "Min_Temp_C": ((mn - 32) * 5 / 9).round(1).values,
            "Precipitation_mm": (pr * 25.4).round(1).values,
        },
        index=pd.to_datetime(raw["DATE"].str.strip()),
    )
    df.index.name = "Date"
    return df


def fetch_years(start: int, end: int) -> pd.DataFrame:
    frames = []
    for y in range(start, end + 1):
        r = requests.get(URL.format(year=y, sid=STATION_ID), timeout=60)
        if r.status_code == 404:
            print(f"  {y}: no file")
            continue
        r.raise_for_status()
        d = parse_gsod(r.text)
        print(f"  {y}: {len(d)} days")
        frames.append(d)
    return pd.concat(frames).sort_index()


def compare_with_dhm(gsod: pd.DataFrame):
    dhm = load_station_csv(ROOT / "data" / "df_analysis.csv")
    both = dhm.join(gsod, rsuffix="_gsod", how="inner")
    print(f"\nOverlap with DHM: {len(both)} days ({both.index.min().date()} → {both.index.max().date()})")
    for col in ["Max_Temp_C", "Min_Temp_C"]:
        d = (both[col] - both[f"{col}_gsod"]).dropna()
        print(f"  {col}: MAE {d.abs().mean():.2f} °C | bias (DHM − GSOD) {d.mean():+.2f} °C | "
              f"within 1 °C {np.mean(d.abs() <= 1):.0%} | n={len(d)}")
    p = both[["Precipitation_mm", "Precipitation_mm_gsod"]].dropna()
    agree = ((p.iloc[:, 0] >= WET_DAY_MM) == (p.iloc[:, 1] >= WET_DAY_MM)).mean()
    print(f"  Rain: GSOD has a value on {len(p)/len(both):.0%} of days | wet/dry agreement {agree:.0%}")
    print("\nRule of thumb: temperature MAE under ~0.7 °C means GSOD can extend the DHM record.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1999)
    args = ap.parse_args()
    print(f"Downloading GSOD {STATION_ID} {args.start} → {date.today().year} …")
    gsod = fetch_years(args.start, date.today().year)
    gsod.to_csv(OUT)
    print(f"Saved {OUT} — {len(gsod)} days, last date {gsod.index.max().date()}")
    compare_with_dhm(gsod)


if __name__ == "__main__":
    main()
