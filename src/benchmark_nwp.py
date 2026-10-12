"""Do free weather-model forecasts beat our station-only model?

Run (needs internet):  python -m src.benchmark_nwp

Compares, on the same days, against the DHM station record:
  * Champion v2 / Rain v1 (station history only)
  * Raw Open-Meteo forecasts from several weather models
  * The same forecasts with a simple monthly bias correction (fit 2022-2023, tested 2024)

Caveat: Open-Meteo's historical-forecast archive is stitched from the most recent
model runs, so it is slightly kinder to the weather models than a true day-ahead
forecast. Treat their scores as an upper bound. If they still don't beat Champion v2,
stop here; if they win clearly, the next step (MOS) is worth building.
"""
import numpy as np
import pandas as pd

from src.features import VARS, WET_DAY_MM, build_features, load_station_csv
from src.live_input import fetch_history_model
from src.model_io import MODELS_DIR, load_rain_models, load_temp_models

START, FIT_END, TEST_START = "2022-01-01", "2023-12-31", "2024-01-01"
WEATHER_MODELS = ["best_match", "ecmwf_ifs025", "gfs_seamless", "icon_seamless"]


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def main():
    station = load_station_csv(MODELS_DIR.parent / "data" / "df_analysis.csv")
    end = station.index.max().strftime("%Y-%m-%d")
    T, R = load_temp_models(), load_rain_models()

    # Our models' t+1 forecasts, indexed by the day being forecast
    feats = build_features(station, R["climatology"]).dropna(subset=R["features"])
    ours = pd.DataFrame(index=feats.index + pd.Timedelta(days=1))
    for k, col in VARS.items():
        fut_doy = ours.index.dayofyear
        ours[col] = T["models"][(k, 1)].predict(feats[T["features"]]) + R["climatology"].loc[fut_doy, col].values
    ours["rain_prob"] = R["models"][1].predict_proba(feats[R["features"]])[:, 1]

    test_days = station.loc[TEST_START:end].index.intersection(ours.index)
    obs = station.loc[test_days]
    wet = (obs["Precipitation_mm"] >= WET_DAY_MM).astype(int)
    rows = [{
        "Source": "Champion v2 / Rain v1 (ours)",
        "Max MAE": (obs.Max_Temp_C - ours.loc[test_days, "Max_Temp_C"]).abs().mean(),
        "Min MAE": (obs.Min_Temp_C - ours.loc[test_days, "Min_Temp_C"]).abs().mean(),
        "Rain Brier": brier(wet, ours.loc[test_days, "rain_prob"]),
    }]
    rows.append({"Source": "Persistence (tomorrow = today)",
                 "Max MAE": (obs.Max_Temp_C - station.Max_Temp_C.shift(1).loc[test_days]).abs().mean(),
                 "Min MAE": (obs.Min_Temp_C - station.Min_Temp_C.shift(1).loc[test_days]).abs().mean(),
                 "Rain Brier": brier(wet, (station.Precipitation_mm.shift(1).loc[test_days] >= WET_DAY_MM).astype(int))})

    for wm in WEATHER_MODELS:
        print(f"Fetching {wm} …")
        try:
            nwp = fetch_history_model(START, end, wm)
        except Exception as e:  # noqa: BLE001
            print(f"  skipped ({e})")
            continue
        both = station.join(nwp, rsuffix="_nwp", how="inner").dropna()
        fit, test = both.loc[:FIT_END], both.loc[TEST_START:].reindex(test_days).dropna()
        if len(test) < 100 or len(fit) < 200:
            print(f"  skipped (only {len(fit)} fit / {len(test)} test days)")
            continue
        row_raw, row_cor = {"Source": f"{wm} (raw)"}, {"Source": f"{wm} (bias-corrected)"}
        for col in VARS.values():
            bias = (fit[col] - fit[f"{col}_nwp"]).groupby(fit.index.month).mean()
            corrected = test[f"{col}_nwp"] + test.index.month.map(bias).fillna(0).values
            key = "Max MAE" if col == "Max_Temp_C" else "Min MAE"
            row_raw[key] = (test[col] - test[f"{col}_nwp"]).abs().mean()
            row_cor[key] = (test[col] - corrected).abs().mean()
        y = (test["Precipitation_mm"] >= WET_DAY_MM).astype(int)
        row_raw["Rain Brier"] = brier(y, (test["Precipitation_mm_nwp"] >= WET_DAY_MM).astype(int))
        # turn forecast rain amount into a probability using 2022-23 data (wet-day frequency per amount bin)
        bins = [-0.01, 0.1, 0.5, 1, 2, 5, 10, 25, 1e9]
        fb = pd.cut(fit["Precipitation_mm_nwp"], bins)
        prob_map = (fit["Precipitation_mm"] >= WET_DAY_MM).groupby(fb, observed=False).mean()
        p = pd.cut(test["Precipitation_mm_nwp"], bins).map(prob_map).astype(float).fillna(y.mean())
        row_cor["Rain Brier"] = brier(y, p)
        rows += [row_raw, row_cor]

    res = pd.DataFrame(rows).set_index("Source").round(3)
    print(f"\nTest period {test_days.min().date()} → {test_days.max().date()} ({len(test_days)} days). Lower is better.\n")
    print(res.to_string())
    out = MODELS_DIR.parent / "benchmark_nwp_results.csv"
    res.to_csv(out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
