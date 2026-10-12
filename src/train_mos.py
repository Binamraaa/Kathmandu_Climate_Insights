"""MOS v1: correct weather-model forecasts with station data (t+1 only).

Run (needs internet, ~2-5 min first time; weather-model data is cached):
    python -m src.train_mos

Inputs
  * Observations: DHM record to Oct 2024, extended with NOAA GSOD
    (data/gsod_kathmandu.csv, from `python -m src.fetch_gsod`), bias-adjusted
    to DHM by month using the 1999-2024 overlap.
  * Weather models: Open-Meteo historical forecasts (ECMWF, ICON, GFS, best match),
    cached in data/nwp_cache/. Missing values (e.g. ECMWF before its archive starts)
    are left as NaN; XGBoost handles them natively.
  * Station features as of the day before (same as Champion v2).

Test = the most recent 365 days; train = everything from 2022 before that.
Reports MAE, clothing-band accuracy, rain Brier score and umbrella hit rates against
Champion v2 / Rain v1, the best bias-corrected weather model, and persistence.

Caveat: the historical-forecast archive is a slightly flattering stand-in for a true
day-ahead forecast, so live accuracy will be a little worse than reported here.
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.features import VARS, WET_DAY_MM, build_features, load_station_csv
from src.live_input import fetch_nwp_full
from src.model_io import MODELS_DIR, _write_meta, load_rain_models, load_temp_models

ROOT = MODELS_DIR.parent
CACHE = ROOT / "data" / "nwp_cache"
NWP_MODELS = ["ecmwf_ifs025", "icon_seamless", "gfs_seamless", "best_match"]
START = "2022-01-01"
TEST_DAYS = 365
BANDS = [-99, 10, 15, 20, 25, 30, 99]  # clothing bands on max temp (°C)
SEED = 42


# ---------------------------------------------------------------- data
def load_obs(verbose: bool = True) -> pd.DataFrame:
    dhm = load_station_csv(ROOT / "data" / "df_analysis.csv")
    gsod_path = ROOT / "data" / "gsod_kathmandu.csv"
    if not gsod_path.exists():
        raise SystemExit("data/gsod_kathmandu.csv not found. Run `python -m src.fetch_gsod` first.")
    gsod = pd.read_csv(gsod_path, index_col="Date", parse_dates=True)
    both = dhm.join(gsod, rsuffix="_g", how="inner")
    adj = gsod.copy()
    for col in VARS.values():
        bias = (both[col] - both[f"{col}_g"]).groupby(both.index.month).mean()
        adj[col] = gsod[col] + gsod.index.month.map(bias).values
        if verbose:
            print(f"GSOD {col}: monthly bias correction {bias.min():+.2f} … {bias.max():+.2f} °C")
    obs = pd.concat([dhm, adj[adj.index > dhm.index.max()]])
    obs = obs[~obs.index.duplicated()].asfreq("D")
    if verbose:
        print(f"Observations: {obs.index.min().date()} → {obs.index.max().date()} "
              f"(DHM to {dhm.index.max().date()}, GSOD after). Missing temp days since 2022: "
              f"{obs.loc[START:, 'Max_Temp_C'].isna().sum()}, missing rain days: "
              f"{obs.loc[START:, 'Precipitation_mm'].isna().mean():.0%}")
    return obs


def get_nwp(model: str, end: pd.Timestamp) -> pd.DataFrame | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{model}.csv"
    if f.exists():
        df = pd.read_csv(f, index_col="Date", parse_dates=True)
        if df.index.max() >= end - pd.Timedelta(days=3):
            return df
    print(f"Fetching {model} {START} → {end.date()} …")
    try:
        df = fetch_nwp_full(model, start_date=START, end_date=end.strftime("%Y-%m-%d"))
    except Exception as e:  # noqa: BLE001
        print(f"  skipped {model}: {e}")
        return None
    df.to_csv(f)
    first = df["tmax"].first_valid_index()
    print(f"  {model}: data from {first.date() if first is not None else 'none'}")
    return df


def build_dataset(obs, clim, T, R):
    end = obs.index.max()
    feats = build_features(obs, clim)
    prev = feats.shift(1)  # everything known at the end of the day before
    D = obs.index
    doy = D.dayofyear
    cmax, cmin = clim.loc[doy, "Max_Temp_C"].values, clim.loc[doy, "Min_Temp_C"].values

    X = pd.DataFrame(index=D)
    station_cols = list(dict.fromkeys(T["features"] + R["features"]))
    for c in station_cols:
        X[f"st_{c}"] = prev[c]
    X["clim_rain_prob"] = clim.loc[doy, "rain_prob"].values
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)

    nwp_used = []
    for m in NWP_MODELS:
        df = get_nwp(m, end)
        if df is None:
            continue
        df = df.reindex(D)
        df["tmax_anom"], df["tmin_anom"] = df["tmax"] - cmax, df["tmin"] - cmin
        for c in df.columns:
            X[f"{m}_{c}"] = df[c]
        nwp_used.append(m)
    for v in ["tmax_anom", "tmin_anom", "precip"]:
        cols = [f"{m}_{v}" for m in nwp_used]
        X[f"ens_{v}"] = X[cols].mean(axis=1)
        X[f"ens_{v}_spread"] = X[cols].std(axis=1)

    Y = pd.DataFrame(index=D)
    Y["Max_Temp_C"], Y["Min_Temp_C"] = obs["Max_Temp_C"], obs["Min_Temp_C"]
    Y["anom_Max"], Y["anom_Min"] = obs["Max_Temp_C"] - cmax, obs["Min_Temp_C"] - cmin
    Y["clim_Max"], Y["clim_Min"] = cmax, cmin
    Y["wet"] = np.where(obs["Precipitation_mm"].isna(), np.nan, (obs["Precipitation_mm"] >= WET_DAY_MM).astype(float))

    # Champion v2 / Rain v1 forecasts for day D (issued on D-1) as the comparison
    # temperature history must exist; missing rain reports are left as NaN (XGBoost handles them)
    ok = prev[[c for c in T["features"] if c.startswith("anom_")]].notna().all(axis=1)
    Y["ours_Max"] = Y["ours_Min"] = Y["ours_rain"] = np.nan
    if ok.any():
        Y.loc[ok, "ours_Max"] = T["models"][("Max", 1)].predict(prev.loc[ok, T["features"]]) + cmax[ok.values]
        Y.loc[ok, "ours_Min"] = T["models"][("Min", 1)].predict(prev.loc[ok, T["features"]]) + cmin[ok.values]
        Y.loc[ok, "ours_rain"] = R["models"][1].predict_proba(prev.loc[ok, R["features"]])[:, 1]
    Y["persist_Max"], Y["persist_Min"] = obs["Max_Temp_C"].shift(1), obs["Min_Temp_C"].shift(1)

    keep = (D >= pd.Timestamp(START) + pd.Timedelta(days=40)) & X[[f"{m}_tmax" for m in nwp_used]].notna().any(axis=1)
    return X[keep], Y[keep], nwp_used


# ---------------------------------------------------------------- models
def tune(est, X, y, scoring):
    grid = {"n_estimators": [200, 400, 700], "learning_rate": [0.02, 0.05], "max_depth": [2, 3, 4],
            "min_child_weight": [3, 10, 20], "subsample": [0.8, 1.0], "colsample_bytree": [0.5, 0.8],
            "reg_lambda": [1, 5, 20]}
    s = RandomizedSearchCV(est, grid, n_iter=15, cv=TimeSeriesSplit(4), scoring=scoring, random_state=SEED)
    s.fit(X, y)
    return s.best_estimator_, s.best_params_


def band_acc(actual, pred):
    a, p = pd.cut(actual, BANDS).cat.codes.values, pd.cut(pred, BANDS).cat.codes.values
    return float(np.mean(a == p))


def main():
    T, R = load_temp_models(), load_rain_models()
    clim = R["climatology"]
    obs = load_obs()
    X, Y, nwp_used = build_dataset(obs, clim, T, R)
    if not nwp_used:
        raise SystemExit("No weather-model data could be fetched.")
    split = X.index.max() - pd.Timedelta(days=TEST_DAYS)
    tr, te = X.index <= split, X.index > split
    print(f"\nWeather models used: {nwp_used}")
    print(f"Train {X.index[tr].min().date()} → {X.index[tr].max().date()} ({tr.sum()} days) | "
          f"test {X.index[te].min().date()} → {X.index[te].max().date()} ({te.sum()} days)\n")

    models, preds = {}, {}
    for k in VARS:
        m_tr = tr & Y[f"anom_{k}"].notna().values
        est, params = tune(xgb.XGBRegressor(objective="reg:absoluteerror", random_state=SEED),
                           X[m_tr], Y.loc[m_tr, f"anom_{k}"], "neg_mean_absolute_error")
        models[k] = est
        preds[k] = pd.Series(est.predict(X[te]), index=X.index[te]) + Y.loc[te, f"clim_{k}"]
        print(f"MOS {k}: {params}")
    r_tr = tr & Y["wet"].notna().values
    est, params = tune(xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=SEED),
                       X[r_tr], Y.loc[r_tr, "wet"], "neg_brier_score")
    models["rain"] = est
    preds["rain"] = pd.Series(est.predict_proba(X[te])[:, 1], index=X.index[te])
    p_tr = est.predict_proba(X[r_tr])[:, 1]
    cands = np.arange(0.2, 0.71, 0.05)
    thr = float(round(cands[np.argmax([f1_score(Y.loc[r_tr, "wet"], p_tr >= t) for t in cands])], 2))
    print(f"MOS rain: {params} | umbrella threshold {thr}\n")

    # Best single bias-corrected weather model (reference)
    Yte = Y[te]
    best = {}
    for k, v in [("Max", "tmax"), ("Min", "tmin")]:
        scores = {}
        for m in nwp_used:
            col = f"{m}_{v}"
            fit = pd.DataFrame({"o": Y.loc[tr, VARS[k]], "f": X.loc[tr, col]}).dropna()
            if len(fit) < 200:
                continue
            bias = (fit.o - fit.f).groupby(fit.index.month).mean()
            p = X.loc[te, col] + X.index[te].month.map(bias).values
            ok = p.notna() & Yte[VARS[k]].notna()
            scores[m] = (p, (Yte[VARS[k]][ok] - p[ok]).abs().mean())
        if scores:
            name = min(scores, key=lambda m: scores[m][1])
            best[k] = (name, scores[name][0])

    # ------------------------------------------------------------ report
    rows = []
    def add(name, pmax, pmin, prain):
        r = {"Source": name}
        for k, p in [("Max", pmax), ("Min", pmin)]:
            if p is None:
                r[f"{k} MAE"] = np.nan
                continue
            ok = p.notna() & Yte[VARS[k]].notna()
            r[f"{k} MAE"] = (Yte[VARS[k]][ok] - p[ok]).abs().mean()
            if k == "Max":
                r["Clothing band ✓"] = band_acc(Yte[VARS[k]][ok], p[ok])
        if prain is not None:
            ok = prain.notna() & Yte["wet"].notna()
            r["Rain Brier"] = brier_score_loss(Yte["wet"][ok], prain[ok])
        rows.append(r)

    add("MOS v1 (weather models + station)", preds["Max"], preds["Min"], preds["rain"])
    add("Champion v2 / Rain v1", Yte["ours_Max"], Yte["ours_Min"], Yte["ours_rain"])
    if best:
        add(f"Best weather model, bias-corrected ({best.get('Max', ('–',))[0]} / {best.get('Min', ('–',))[0]})",
            best.get("Max", (None, None))[1], best.get("Min", (None, None))[1], None)
    add("Persistence (tomorrow = today)", Yte["persist_Max"], Yte["persist_Min"], None)
    res = pd.DataFrame(rows).set_index("Source")
    fmt = res.copy()
    for c in ["Max MAE", "Min MAE"]:
        fmt[c] = res[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "–")
    fmt["Clothing band ✓"] = res["Clothing band ✓"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "–")
    fmt["Rain Brier"] = res["Rain Brier"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "–")
    print(f"Test set: last {TEST_DAYS} days. MAE in °C, lower is better; clothing band = right 5 °C band for max temp.\n")
    print(fmt.to_string())

    ok = Yte["wet"].notna()
    for name, p, t in [("MOS v1", preds["rain"], thr), ("Rain v1", Yte["ours_rain"], R["umbrella_threshold"])]:
        okp = ok & p.notna()
        flag = p[okp] >= t
        y = Yte["wet"][okp]
        print(f"Umbrella flag, {name}: catches {recall_score(y, flag):.0%} of wet days, "
              f"right {precision_score(y, flag):.0%} of the time it says yes")

    imp = pd.Series(models["Max"].feature_importances_, index=X.columns).sort_values(ascending=False).head(8)
    print("\nTop drivers of MOS Max forecast:", ", ".join(imp.index))

    folder = MODELS_DIR / "mos_v1"
    folder.mkdir(parents=True, exist_ok=True)
    models["Max"].save_model(folder / "max_t1.json")
    models["Min"].save_model(folder / "min_t1.json")
    models["rain"].save_model(folder / "rain_t1.json")
    _write_meta(folder, {"features": list(X.columns), "nwp_models": nwp_used, "umbrella_threshold": thr,
                         "wet_day_mm": WET_DAY_MM, "train_end": str(split.date()),
                         "test_results": json.loads(res.to_json(orient="index"))}, clim)
    res.to_csv(ROOT / "mos_v1_results.csv")
    print(f"\nSaved models to {folder} and results to mos_v1_results.csv")


if __name__ == "__main__":
    main()
