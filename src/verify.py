"""Verification log: how good are K Laune's forecasts in real life?

Every time `export_forecast` runs, it appends the 3-day forecast (plus a raw
Open-Meteo forecast as a "typical weather app" reference) to
data/verification/forecast_log.csv.

You add what actually happened (MFD/DHM daily bulletin, your own notes):
    python -m src.verify add 2026-10-13 --max 27.5 --min 14.8 --rain 0
    python -m src.verify add 2026-10-14 --max 26 --min 15 --rain 3.2

Then see the scorecard:
    python -m src.verify report
"""
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
VDIR = ROOT / "data" / "verification"
FLOG = VDIR / "forecast_log.csv"
OBS = VDIR / "observations.csv"
WET_MM = 1.0
BAND_EDGES = [10, 15, 20, 25, 30]


# ------------------------------------------------------------------ logging
def fetch_reference(days: int = 4) -> pd.DataFrame | None:
    """Raw Open-Meteo forecast for the airport: what a typical weather app shows."""
    from src.live_input import FORECAST_URL, KTM_AIRPORT, TZ
    try:
        r = requests.get(FORECAST_URL, timeout=30, params={**KTM_AIRPORT, "timezone": TZ, "forecast_days": days,
                         "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max"})
        r.raise_for_status()
        d = r.json()["daily"]
        return pd.DataFrame({"date": d["time"], "max": d["temperature_2m_max"], "min": d["temperature_2m_min"],
                             "rain_prob": [None if p is None else p / 100 for p in d["precipitation_probability_max"]]})
    except Exception as e:  # noqa: BLE001
        print(f"  (reference forecast unavailable: {e})")
        return None


def log_forecast(fc: dict, reference: pd.DataFrame | None = None, live: bool = True):
    """Append one row per forecast day (and per reference day) to the log."""
    VDIR.mkdir(parents=True, exist_ok=True)
    issued = fc["issued_for"]
    rows = []
    for d in fc["days"]:
        rows.append({"issued": issued, "target": d["date"], "horizon": d["horizon"], "source": d["model"],
                     "max": d["max_temp_c"], "min": d["min_temp_c"],
                     "max_lo": (d.get("max_range_c") or [None, None])[0], "max_hi": (d.get("max_range_c") or [None, None])[1],
                     "rain_prob": d["rain_prob"], "umbrella": d["umbrella"], "live": live})
    if reference is not None:
        for _, r in reference.iterrows():
            h = (pd.Timestamp(r["date"]) - pd.Timestamp(issued)).days
            if 1 <= h <= 3:
                rows.append({"issued": issued, "target": r["date"], "horizon": h, "source": "Open-Meteo raw (reference)",
                             "max": r["max"], "min": r["min"], "max_lo": None, "max_hi": None,
                             "rain_prob": r["rain_prob"], "umbrella": None if pd.isna(r["rain_prob"]) else bool(r["rain_prob"] >= 0.4),
                             "live": live})
    new = pd.DataFrame(rows)
    if FLOG.exists():
        old = pd.read_csv(FLOG)
        # one forecast per (issued, target, source): the latest run wins
        new = pd.concat([old, new]).drop_duplicates(["issued", "target", "source"], keep="last")
    new.sort_values(["target", "source", "horizon"]).to_csv(FLOG, index=False)
    print(f"Logged {len(rows)} forecast rows to data/verification/{FLOG.name}")


# ------------------------------------------------------------------ observations
def add_observation(day: str, tmax: float, tmin: float, rain_mm: float | None, note: str = ""):
    VDIR.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"date": pd.Timestamp(day).strftime("%Y-%m-%d"), "max": tmax, "min": tmin,
                         "rain_mm": rain_mm, "note": note}])
    if OBS.exists():
        row = pd.concat([pd.read_csv(OBS), row]).drop_duplicates("date", keep="last")
    row.sort_values("date").to_csv(OBS, index=False)
    print(f"Saved observation for {day}: max {tmax}, min {tmin}, rain {rain_mm if rain_mm is not None else '–'} mm")


# ------------------------------------------------------------------ scorecard
def report():
    if not FLOG.exists() or not OBS.exists():
        raise SystemExit("Need both a forecast log and observations. Run export_forecast, then `verify add` for past days.")
    f = pd.read_csv(FLOG)
    o = pd.read_csv(OBS).rename(columns={"date": "target", "max": "obs_max", "min": "obs_min"})
    m = f.merge(o, on="target")
    if m.empty:
        raise SystemExit("No forecast days have observations yet.")
    m["source_group"] = np.where(m.source.str.contains("reference"), "Typical app (raw Open-Meteo)",
                         np.where(m.source.str.startswith("MOS"), "K Laune: MOS", "K Laune: station model"))
    m["k_laune"] = ~m.source.str.contains("reference")
    m["wet"] = np.where(m.rain_mm.isna(), np.nan, (m.rain_mm >= WET_MM).astype(float))
    band = lambda t: np.digitize(t, BAND_EDGES)

    print(f"Verified days: {m.target.nunique()} ({m.target.min()} → {m.target.max()})\n")
    rows = []
    for (h, k), g in m.groupby(["horizon", "k_laune"]):
        r = {"Horizon": f"Day {h}", "Source": "K Laune" if k else "Typical app", "Days": len(g),
             "Max MAE": (g["max"] - g.obs_max).abs().mean(), "Max bias": (g["max"] - g.obs_max).mean(),
             "Min MAE": (g["min"] - g.obs_min).abs().mean(), "Min bias": (g["min"] - g.obs_min).mean(),
             "Band right": np.mean(band(g["max"]) == band(g.obs_max))}
        rng = g.dropna(subset=["max_lo", "max_hi"])
        r["Range hit"] = np.mean((rng.obs_max >= rng.max_lo) & (rng.obs_max <= rng.max_hi)) if len(rng) else np.nan
        w = g.dropna(subset=["wet", "umbrella"])
        if len(w):
            u = w.umbrella.astype(str).str.lower().isin(["true", "1"])
            wet = w.wet == 1
            r["Rain caught"] = (u & wet).sum() / wet.sum() if wet.sum() else np.nan
            r["False alarms"] = (u & ~wet).sum() / u.sum() if u.sum() else np.nan
        rows.append(r)
    res = pd.DataFrame(rows)
    fmt = res.copy()
    for c in ["Max MAE", "Min MAE"]:
        fmt[c] = res[c].map("{:.2f}".format)
    for c in ["Max bias", "Min bias"]:
        fmt[c] = res[c].map("{:+.2f}".format)
    for c in ["Band right", "Range hit", "Rain caught", "False alarms"]:
        if c in res:
            fmt[c] = res[c].map(lambda v: f"{v:.0%}" if pd.notna(v) else "–")
    print("Errors in °C (lower is better). Bias > 0 means the forecast ran warm.\n")
    print(fmt.to_string(index=False))

    d1 = res[res.Horizon == "Day 1"].set_index("Source")
    if {"K Laune", "Typical app"} <= set(d1.index):
        diff = d1.loc["Typical app", "Max MAE"] - d1.loc["K Laune", "Max MAE"]
        print(f"\nTomorrow's high: K Laune is {abs(diff):.2f} °C {'more' if diff > 0 else 'less'} accurate than a typical app.")
    n = m[m.k_laune & (m.horizon == 1)].target.nunique()
    if n < 14:
        print(f"Only {n} verified days for tomorrow's forecast so far. Treat these numbers as early signs; ~30 days gives a fair picture.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="record what actually happened on a day")
    a.add_argument("date", help="YYYY-MM-DD")
    a.add_argument("--max", type=float, required=True, help="observed high, °C")
    a.add_argument("--min", type=float, required=True, help="observed low, °C")
    a.add_argument("--rain", type=float, help="observed rain, mm (0 if dry; leave out if unknown)")
    a.add_argument("--note", default="", help="e.g. 'MFD bulletin' or 'sunny, cool breeze'")
    sub.add_parser("report", help="print the accuracy scorecard")
    args = ap.parse_args()
    if args.cmd == "add":
        if pd.Timestamp(args.date).date() > date.today():
            raise SystemExit("That date is in the future: observations are for days that already happened.")
        add_observation(args.date, args.max, args.min, args.rain, args.note)
    else:
        report()


if __name__ == "__main__":
    main()
