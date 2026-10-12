"""MOS v2: honest day-ahead MOS with forecast ranges and an app-tuned umbrella flag.

Run (needs internet; first run downloads hourly weather-model archives, ~5-15 min):
    python -m src.train_mos_v2

What's different from MOS v1
  * Weather-model inputs come from the Previous Runs API (`_previous_day1`): what
    each model predicted 24 h ahead. No near-0-hour flattery. Archive mostly from Jan 2024.
  * More models (ECMWF IFS + AIFS, ICON, GFS, JMA, UK Met Office, CMA, best match);
    models the API can't serve are skipped automatically.
  * Max/Min are quantile models: 10th, 50th and 90th percentile in one model.
  * Evaluation is blocked cross-validation (5 contiguous time blocks, 7-day gaps), so
    every day from 2024 on gets an out-of-sample forecast covering all seasons.
  * Reports what the app needs: clothing-band accuracy with soft boundaries, and an
    umbrella threshold set to catch ≥ 85% of wet days.
"""
import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV

from src.features import VARS, WET_DAY_MM, build_features
from src.model_io import MODELS_DIR, _write_meta, load_rain_models, load_temp_models
from src.mos_features import BAND_EDGES, LAYER_SCALE, assemble, band, choose_layer_scale, core_columns, outfit
from src.nwp_hourly import fetch_previous_runs, to_daily
from src.train_mos import load_obs

ROOT = MODELS_DIR.parent
CACHE = ROOT / "data" / "nwp_cache_v2"
NWP_MODELS = ["best_match", "ecmwf_ifs025", "ecmwf_aifs025_single", "icon_seamless", "gfs_seamless",
              "jma_seamless", "ukmo_seamless", "cma_grapes_global"]
START = "2024-01-01"
QUANTILES = np.array([0.1, 0.5, 0.9])
RECALL_TARGET = 0.85
TARGET_COVERAGE = 0.80  # 10–90% range
LAYER_TARGET = 0.28     # aim for "bring a layer" on ~28% of days
PRUNE_DROP = ["ecmwf_aifs025_single", "ukmo_seamless"]  # short archive history (< ~8 months)
N_FOLDS, PURGE_DAYS, SEED = 5, 7, 42
GRID = {"n_estimators": [200, 400, 700], "learning_rate": [0.02, 0.05], "max_depth": [2, 3, 4],
        "min_child_weight": [3, 10, 20], "subsample": [0.8, 1.0], "colsample_bytree": [0.4, 0.7],
        "reg_lambda": [1, 5, 20]}


def get_daily_nwp(model: str, end: pd.Timestamp) -> pd.DataFrame | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{model}.csv"
    if f.exists():
        df = pd.read_csv(f, index_col="Date", parse_dates=True)
        if df.index.max() >= end - pd.Timedelta(days=3):
            return df
    print(f"Fetching {model} (day-ahead, hourly) {START} → {end.date()} …")
    try:
        df = to_daily(fetch_previous_runs(model, START, end.strftime("%Y-%m-%d")))
    except Exception as e:  # noqa: BLE001
        print(f"  skipped {model}: {e}")
        return None
    n = int(df["tmax"].notna().sum())
    if n < 150:
        print(f"  skipped {model}: only {n} days with data")
        return None
    df.to_csv(f)
    print(f"  {model}: {n} days, from {df['tmax'].first_valid_index().date()}")
    return df


def blocked_folds(n: int, idx: pd.DatetimeIndex):
    """Contiguous time blocks; training excludes PURGE_DAYS either side of the test block."""
    bounds = np.linspace(0, n, N_FOLDS + 1).astype(int)
    folds = []
    for i in range(N_FOLDS):
        te = np.arange(bounds[i], bounds[i + 1])
        t0, t1 = idx[te[0]] - pd.Timedelta(days=PURGE_DAYS), idx[te[-1]] + pd.Timedelta(days=PURGE_DAYS)
        tr = np.where((idx < t0) | (idx > t1))[0]
        folds.append((tr, te))
    return folds


def median_mae_scorer(est, X, y):
    return -np.mean(np.abs(np.asarray(y) - est.predict(X)[:, 1]))


def oof_predict(est, X, y, folds, proba=False):
    out = np.full((len(X), 3) if not proba else len(X), np.nan)
    for tr, te in folds:
        m = clone(est).fit(X.iloc[tr], y.iloc[tr])
        out[te] = m.predict_proba(X.iloc[te])[:, 1] if proba else np.sort(m.predict(X.iloc[te]), axis=1)
    return out


def fit_task(base, X, y, folds, scoring, proba=False):
    ok = y.notna().values
    Xk, yk = X[ok], y[ok]
    pos = pd.Series(np.arange(len(X)))[ok].values
    remap = {p: i for i, p in enumerate(pos)}
    fk = [(np.array([remap[p] for p in tr if p in remap]), np.array([remap[p] for p in te if p in remap])) for tr, te in folds]
    fk = [(tr, te) for tr, te in fk if len(tr) and len(te)]
    s = RandomizedSearchCV(base, GRID, n_iter=15, cv=fk, scoring=scoring, random_state=SEED)
    s.fit(Xk, yk)
    oof = oof_predict(s.best_estimator_, Xk, yk, fk, proba)
    full = np.full((len(X), 3) if not proba else len(X), np.nan)
    full[ok] = oof
    final = clone(s.best_estimator_).fit(Xk, yk)
    return final, full, s.best_params_


def conformal_widen(q: pd.DataFrame, y: pd.Series, folds):
    """Quantile trees tend to give ranges that are too narrow. Widen them by a margin c
    so the 10–90% range really contains ~80% of outcomes (conformalised quantile regression).
    Cross-fitted: each fold's margin comes from the other folds' errors."""
    err = np.maximum(q.q10 - y, y - q.q90)  # >0 when the actual falls outside the range
    out = q.copy()
    for _, te in folds:
        other = err.drop(err.index[te]).dropna()
        c = np.quantile(other, TARGET_COVERAGE) if len(other) else 0.0
        out.iloc[te, out.columns.get_loc("q10")] -= c
        out.iloc[te, out.columns.get_loc("q90")] += c
    c_final = float(np.quantile(err.dropna(), TARGET_COVERAGE))
    return out, c_final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune", action="store_true",
                    help="MOS v2.1: drop short-history weather models and use the core feature set")
    args = ap.parse_args()
    out_name = "mos_v2_1" if args.prune else "mos_v2"
    label = "MOS v2.1 (pruned)" if args.prune else "MOS v2"
    T, R = load_temp_models(), load_rain_models()
    clim = R["climatology"]
    obs = load_obs()
    end = obs.index.max()

    wanted = [m for m in NWP_MODELS if not (args.prune and m in PRUNE_DROP)]
    nwp = {m: d for m in wanted if (d := get_daily_nwp(m, end)) is not None}
    if len(nwp) < 2:
        raise SystemExit("Fewer than two weather models available; cannot train MOS v2.")
    models_used = list(nwp)

    feats = build_features(obs, clim)
    prev = feats.shift(1)
    station_cols = list(dict.fromkeys(T["features"] + R["features"]))
    D = obs.loc[START:].index
    X = assemble(prev.loc[D], nwp, clim, station_cols, models_used)
    X = X[X[[f"{m}_tmax" for m in models_used]].notna().sum(axis=1) >= 2]
    X_all = X  # full set, used for the weather-model baseline
    if args.prune:
        X = X[core_columns(models_used)]
    D = X.index
    doy = D.dayofyear
    cmax, cmin = clim.loc[doy, "Max_Temp_C"].values, clim.loc[doy, "Min_Temp_C"].values
    Y = pd.DataFrame({"Max_Temp_C": obs.loc[D, "Max_Temp_C"], "Min_Temp_C": obs.loc[D, "Min_Temp_C"]}, index=D)
    Y["anom_Max"], Y["anom_Min"] = Y.Max_Temp_C - cmax, Y.Min_Temp_C - cmin
    P = obs.loc[D, "Precipitation_mm"]
    Y["wet"] = np.where(P.isna(), np.nan, (P >= WET_DAY_MM).astype(float))
    folds = blocked_folds(len(X), D)
    print(f"\nWeather models used: {models_used}")
    print(f"{len(X)} days ({D.min().date()} → {D.max().date()}), {X.shape[1]} features, "
          f"{N_FOLDS}-fold blocked CV with {PURGE_DAYS}-day gaps\n")

    preds, final, params, margin = {}, {}, {}, {}
    for k, c in [("Max", cmax), ("Min", cmin)]:
        base = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=QUANTILES, random_state=SEED)
        final[k], q, params[k] = fit_task(base, X, Y[f"anom_{k}"], folds, median_mae_scorer)
        raw = pd.DataFrame(q + c[:, None], index=D, columns=["q10", "q50", "q90"])
        preds[k], margin[k] = conformal_widen(raw, Y[f"{k}_Temp_C"], folds)
        y_ok = Y[f"{k}_Temp_C"].notna()
        raw_cov = np.mean((Y[f"{k}_Temp_C"][y_ok] >= raw.q10[y_ok]) & (Y[f"{k}_Temp_C"][y_ok] <= raw.q90[y_ok]))
        print(f"{label} {k}: {params[k]}\n  raw range coverage {raw_cov:.0%} → widened by ±{margin[k]:.2f} °C")
    base = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=SEED)
    final["rain"], pr, params["rain"] = fit_task(base, X, Y["wet"], folds, "neg_brier_score", proba=True)
    preds["rain"] = pd.Series(pr, index=D)
    print(f"{label} rain: {params['rain']}\n")

    # Comparisons on the same days
    ok_t = prev.loc[D, [c for c in T["features"] if c.startswith("anom_")]].notna().all(axis=1).values
    ours = pd.DataFrame(index=D, columns=["Max", "Min", "rain"], dtype=float)
    ours.loc[ok_t, "Max"] = T["models"][("Max", 1)].predict(prev.loc[D[ok_t], T["features"]]) + cmax[ok_t]
    ours.loc[ok_t, "Min"] = T["models"][("Min", 1)].predict(prev.loc[D[ok_t], T["features"]]) + cmin[ok_t]
    ours.loc[ok_t, "rain"] = R["models"][1].predict_proba(prev.loc[D[ok_t], R["features"]])[:, 1]
    persist = pd.DataFrame({"Max": obs.Max_Temp_C.shift(1).loc[D], "Min": obs.Min_Temp_C.shift(1).loc[D]})
    best_nwp = {}
    for k, v in [("Max", "tmax"), ("Min", "tmin")]:
        scores = {}
        for m in models_used:
            p = pd.Series(np.nan, index=D)
            for tr, te in folds:
                fit = pd.DataFrame({"o": Y[f"{k}_Temp_C"].iloc[tr], "f": X_all[f"{m}_{v}"].iloc[tr]}).dropna()
                if len(fit) < 60:
                    continue
                bias = (fit.o - fit.f).groupby(fit.index.month).mean()
                p.iloc[te] = X_all[f"{m}_{v}"].iloc[te].values + D[te].month.map(bias).fillna((fit.o - fit.f).mean()).values
            okp = p.notna() & Y[f"{k}_Temp_C"].notna()
            if okp.sum() > 0.8 * len(D):
                scores[m] = (p, (Y[f"{k}_Temp_C"][okp] - p[okp]).abs().mean())
        if scores:
            name = min(scores, key=lambda m: scores[m][1])
            best_nwp[k] = (name, scores[name][0])

    def mae(k, p):
        okp = p.notna() & Y[f"{k}_Temp_C"].notna()
        return (Y[f"{k}_Temp_C"][okp] - p[okp]).abs().mean(), okp

    # Clothing bands (on max temp)
    a = band(Y.Max_Temp_C)
    valid = Y.Max_Temp_C.notna().values
    qx = preds["Max"]
    layer_scale = choose_layer_scale(qx.q10[valid], qx.q50[valid], qx.q90[valid], LAYER_TARGET)
    _, lo0, hi0, layer0 = outfit(qx.q10, qx.q50, qx.q90, LAYER_SCALE)
    m_band, lo, hi, layer = outfit(qx.q10, qx.q50, qx.q90, layer_scale)
    rows = []
    mx, _ = mae("Max", preds["Max"].q50)
    mn, _ = mae("Min", preds["Min"].q50)
    rows.append({"Source": f"{label} (day-ahead weather models + station)", "Max MAE": mx, "Min MAE": mn,
                 "Band (hard)": np.mean((a == m_band)[valid]),
                 "Band (soft)": np.mean(((a >= lo) & (a <= hi))[valid]),
                 "Rain Brier": brier_score_loss(Y.wet.dropna(), preds["rain"][Y.wet.notna()])})
    for name, pmx, pmn, prn in [("Champion v2 / Rain v1", ours["Max"], ours["Min"], ours["rain"]),
                                (f"Best weather model, bias-corrected ({best_nwp.get('Max', ('–',))[0]} / "
                                 f"{best_nwp.get('Min', ('–',))[0]})",
                                 best_nwp.get("Max", (None, pd.Series(np.nan, index=D)))[1],
                                 best_nwp.get("Min", (None, pd.Series(np.nan, index=D)))[1], None),
                                ("Persistence (tomorrow = today)", persist["Max"], persist["Min"], None)]:
        mx, okx = mae("Max", pmx)
        mn, _ = mae("Min", pmn)
        hard = np.mean(band(Y.Max_Temp_C[okx]) == band(pmx[okx])) if okx.any() else np.nan
        r = {"Source": name, "Max MAE": mx, "Min MAE": mn, "Band (hard)": hard, "Band (soft)": hard}
        if prn is not None:
            okr = prn.notna() & Y.wet.notna()
            r["Rain Brier"] = brier_score_loss(Y.wet[okr], prn[okr])
        rows.append(r)
    res = pd.DataFrame(rows).set_index("Source")
    show = res.copy()
    for c in ["Max MAE", "Min MAE"]:
        show[c] = res[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "–")
    for c in ["Band (hard)", "Band (soft)"]:
        show[c] = res[c].map(lambda v: f"{v:.0%}" if pd.notna(v) else "–")
    show["Rain Brier"] = res["Rain Brier"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "–")
    print("Out-of-sample results. MAE in °C (lower is better). Band = right 5 °C clothing band for max temp;")
    print("soft = advice covers both bands when the forecast range crosses a boundary ('bring a layer').\n")
    print(show.to_string())

    # Forecast ranges
    print("\nForecast range (10–90%) — should contain the actual value ~80% of the time:")
    for k in ["Max", "Min"]:
        q, y = preds[k], Y[f"{k}_Temp_C"]
        okq = y.notna()
        cov = np.mean((y[okq] >= q.q10[okq]) & (y[okq] <= q.q90[okq]))
        print(f"  {k}: contains actual {cov:.0%} of days | average width {(q.q90 - q.q10)[okq].mean():.1f} °C")
    soft0 = np.mean(((a >= lo0) & (a <= hi0))[valid])
    print(f"  'Bring a layer' rule tuned: {np.mean(layer[valid]):.0%} of days (was {np.mean(layer0[valid]):.0%} "
          f"with the default rule; soft band accuracy {res.loc[res.index[0], 'Band (soft)']:.0%} vs {soft0:.0%})")

    # Umbrella threshold for the app
    okr = Y.wet.notna().values
    y, p = Y.wet[okr].values, preds["rain"][okr].values
    cands = np.round(np.arange(0.05, 0.81, 0.01), 2)
    rec = np.array([recall_score(y, p >= t) for t in cands])
    thr = float(cands[rec >= RECALL_TARGET].max()) if (rec >= RECALL_TARGET).any() else float(cands[0])
    print(f"\nUmbrella flag (threshold {thr}, set to catch ≥ {RECALL_TARGET:.0%} of wet days):")
    for name, pp, t in [(label, p, thr), ("Rain v1", ours["rain"][okr].values, R["umbrella_threshold"])]:
        okp = ~np.isnan(pp)
        f = pp[okp] >= t
        print(f"  {name}: catches {recall_score(y[okp], f):.0%} of wet days, right {precision_score(y[okp], f):.0%} "
              f"of the time it says yes, flags {f.mean():.0%} of days")

    # Seasons
    season = pd.Series(np.select([D.month.isin([12, 1, 2]), D.month.isin([3, 4, 5]), D.month.isin([6, 7, 8, 9])],
                                 ["Winter", "Pre-monsoon", "Monsoon"], "Post-monsoon"), index=D)
    print(f"\nMax MAE by season ({label} vs Champion v2):")
    for s in ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]:
        i = (season == s) & Y.Max_Temp_C.notna() & ours["Max"].notna()
        if i.sum():
            print(f"  {s:<13} {(Y.Max_Temp_C[i] - preds['Max'].q50[i]).abs().mean():.2f} vs "
                  f"{(Y.Max_Temp_C[i] - ours['Max'][i]).abs().mean():.2f}  ({int(i.sum())} days)")

    imp = pd.Series(final["Max"].feature_importances_, index=X.columns).sort_values(ascending=False).head(8)
    print(f"\nTop drivers of {label} Max:", ", ".join(imp.index))

    folder = MODELS_DIR / out_name
    folder.mkdir(parents=True, exist_ok=True)
    final["Max"].save_model(folder / "max_t1.json")
    final["Min"].save_model(folder / "min_t1.json")
    final["rain"].save_model(folder / "rain_t1.json")
    _write_meta(folder, {"features": list(X.columns), "station_cols": station_cols, "nwp_models": models_used,
                         "quantiles": QUANTILES.tolist(), "range_margin": margin, "layer_scale": layer_scale, "label": label,
                         "umbrella_threshold": thr, "wet_day_mm": WET_DAY_MM,
                         "band_edges": BAND_EDGES, "trained_on": [str(D.min().date()), str(D.max().date())],
                         "results": json.loads(res.to_json(orient="index"))}, clim)
    res.to_csv(ROOT / f"{out_name}_results.csv")
    print(f"\nSaved models to {folder} and results to {out_name}_results.csv")


if __name__ == "__main__":
    main()
