"""Kathmandu 3-day forecast.

Tomorrow (t+1): MOS v2 if trained (weather models + station, with a 10–90% range).
Days 2–3: Champion v2 (temperature) and Rain v1 (rain), station history only.

Live (needs internet):   python -m src.predict
Replay a past date:      python -m src.predict --replay 2025-03-15
JSON output for an app:  python -m src.predict --json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features import VARS, build_features, load_station_csv
from src.model_io import MODELS_DIR, _read_meta, load_rain_models, load_temp_models
from src.mos_features import BAND_NAMES, assemble, outfit

ROOT = Path(__file__).resolve().parents[1]


MOS_CANDIDATES = ["mos_v2_1", "mos_v2"]


def _mos_score(meta: dict) -> float:
    """Max MAE + Min MAE of the trained MOS model (first row of its saved results)."""
    first = next(iter(meta.get("results", {}).values()), {})
    return float(first.get("Max MAE", np.inf)) + float(first.get("Min MAE", np.inf))


def load_mos_v2(name: str | None = None):
    """Load a MOS model. By default picks whichever trained version scored best."""
    names = [name] if name else MOS_CANDIDATES
    found = []
    for n in names:
        folder = MODELS_DIR / n
        if (folder / "meta.json").exists():
            meta, clim = _read_meta(folder)
            found.append((_mos_score(meta), n, folder, meta, clim))
    if not found:
        return None
    _, n, folder, meta, clim = min(found, key=lambda t: t[0])
    m = {"meta": meta, "climatology": clim, "name": n}
    for k in ["max", "min"]:
        m[k] = xgb.XGBRegressor()
        m[k].load_model(folder / f"{k}_t1.json")
    m["rain"] = xgb.XGBClassifier()
    m["rain"].load_model(folder / "rain_t1.json")
    return m


def nwp_for_day(mos, day: pd.Timestamp, live: bool) -> dict:
    """Daily weather-model forecasts for `day`: live Forecast API, or the training cache for replays."""
    out = {}
    for model in mos["meta"]["nwp_models"]:
        try:
            if live:
                from src.nwp_hourly import fetch_live, to_daily
                d = to_daily(fetch_live(model))
            else:
                f = ROOT / "data" / "nwp_cache_v2" / f"{model}.csv"
                d = pd.read_csv(f, index_col="Date", parse_dates=True) if f.exists() else pd.DataFrame()
            if day in d.index:
                out[model] = d.loc[[day]]
        except Exception as e:  # noqa: BLE001
            print(f"  (weather model {model} unavailable: {e})")
    return out


def outfit_text(lo_band: int, hi_band: int, main_band: int, layer: bool) -> str:
    if not layer:
        return BAND_NAMES[main_band]
    other = hi_band if main_band == lo_band else lo_band
    return f"{BAND_NAMES[main_band]}, bring a layer for {BAND_NAMES[other].split(' (')[0].lower()}"


def forecast(daily: pd.DataFrame, temp: dict, rain: dict, mos=None, live=False) -> dict:
    clim = rain["climatology"]
    feats = build_features(daily, clim)
    today = feats.index[-1]
    row = feats.iloc[[-1]]
    missing = [c for c in temp["features"] if pd.isna(row[c].iloc[0]) and c.startswith("anom_")]
    if missing:
        raise ValueError(f"Not enough temperature history (missing: {missing[:5]} …). Need ~40 days.")

    out = {"issued_for": str(today.date()), "days": []}
    for h in (1, 2, 3):
        date = today + pd.Timedelta(days=h)
        day = {"date": str(date.date()), "horizon": h, "model": "Champion v2 / Rain v1"}
        for k, col in VARS.items():
            anom = float(temp["models"][(k, h)].predict(row[temp["features"]])[0])
            day[f"{k.lower()}_temp_c"] = round(anom + float(clim.loc[date.dayofyear, col]), 1)
        p = float(rain["models"][h].predict_proba(row[rain["features"]])[0, 1])
        day["rain_prob"], day["umbrella"] = round(p, 2), bool(p >= rain["umbrella_threshold"])
        b = int(np.digitize(day["max_temp_c"], [10, 15, 20, 25, 30]))
        day["outfit"], day["layer"] = BAND_NAMES[b], False
        out["days"].append(day)

    if mos is not None:
        D = today + pd.Timedelta(days=1)
        nwp = nwp_for_day(mos, D, live)
        if len(nwp) >= 2:
            meta = mos["meta"]
            prev = row.copy()
            prev.index = pd.DatetimeIndex([D])
            X = assemble(prev, nwp, clim, meta["station_cols"], meta["nwp_models"])[meta["features"]]
            d = out["days"][0]
            for k, col in VARS.items():
                q = np.sort(mos[k.lower()].predict(X)[0]) + float(clim.loc[D.dayofyear, col])
                c = meta["range_margin"][k]
                d[f"{k.lower()}_temp_c"] = round(float(q[1]), 1)
                d[f"{k.lower()}_range_c"] = [round(float(q[0] - c), 1), round(float(q[2] + c), 1)]
            p = float(mos["rain"].predict_proba(X)[0, 1])
            d["rain_prob"], d["umbrella"] = round(p, 2), bool(p >= meta["umbrella_threshold"])
            m, lo, hi, layer = outfit(*[np.array([v]) for v in (d["max_range_c"][0], d["max_temp_c"], d["max_range_c"][1])],
                                      scale=meta.get("layer_scale", 0.526))
            d["outfit"] = outfit_text(int(lo[0]), int(hi[0]), int(m[0]), bool(layer[0]))
            d["layer"] = bool(layer[0])
            d["model"] = f"{meta.get('label', mos['name'])} ({len(nwp)} weather models)"
        else:
            print("  MOS v2 skipped for tomorrow: fewer than two weather models available for that date.")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", help="YYYY-MM-DD: forecast from recorded data as of this date (offline)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    ap.add_argument("--no-mos", action="store_true", help="use the station-only models for every day")
    ap.add_argument("--mos", choices=MOS_CANDIDATES, help="force a MOS version (default: best-scoring one)")
    args = ap.parse_args()

    temp, rain = load_temp_models("champion_v2"), load_rain_models("rain_v1")
    mos = None if args.no_mos else load_mos_v2(args.mos)
    actual = None
    if args.replay:
        from src.train_mos import load_obs
        try:
            station = load_obs(verbose=False)
        except SystemExit:
            station = load_station_csv(ROOT / "data" / "df_analysis.csv")
        t = pd.Timestamp(args.replay)
        daily = station.loc[:t].tail(60)
        actual = station.loc[t + pd.Timedelta(days=1): t + pd.Timedelta(days=3)]
        source = "recorded station data (replay)"
        fc = forecast(daily, temp, rain, mos, live=False)
    else:
        from src.live_input import apply_calibration, fetch_recent, load_calibration
        cal = load_calibration()
        if cal is None:
            print("⚠ No calibration file — run `python -m src.calibrate_openmeteo` first. Using raw Open-Meteo values.\n")
        daily = apply_calibration(fetch_recent(45), cal)
        source = "Open-Meteo" + (" (bias-corrected)" if cal else " (raw)")
        fc = forecast(daily, temp, rain, mos, live=True)
    fc["input_source"] = source

    if args.json:
        print(json.dumps(fc, indent=2))
        return
    print(f"\nKathmandu forecast issued for {fc['issued_for']} — input: {source}\n")
    for d in fc["days"]:
        rng = lambda k: f" ({d[k][0]}–{d[k][1]})" if k in d else ""
        print(f"{d['date']}  Max {d['max_temp_c']}°C{rng('max_range_c')}  Min {d['min_temp_c']}°C{rng('min_range_c')}  "
              f"Rain {d['rain_prob']*100:.0f}% {'☂ umbrella' if d['umbrella'] else ''}")
        print(f"            Wear: {d['outfit']}   [{d['model']}]")
        if actual is not None and pd.Timestamp(d["date"]) in actual.index:
            a = actual.loc[pd.Timestamp(d["date"])]
            rain_txt = "no report" if pd.isna(a.Precipitation_mm) else f"{a.Precipitation_mm:.1f} mm"
            print(f"            Actual: max {a.Max_Temp_C:.1f} / min {a.Min_Temp_C:.1f} / rain {rain_txt}")


if __name__ == "__main__":
    main()
