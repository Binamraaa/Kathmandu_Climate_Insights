"""Feature assembly shared by MOS v2 training and live prediction."""
import numpy as np
import pandas as pd

NWP_DAILY_COLS = ["tmax", "tmin", "t_mean", "precip", "precip_hours", "cloud", "cloud_afternoon", "rh", "wind_max", "pmsl"]
BAND_EDGES = [10, 15, 20, 25, 30]                     # clothing bands on max temp (°C)
BAND_NAMES = ["Cold (<10°C)", "Cool (10–15°C)", "Mild (15–20°C)", "Warm (20–25°C)", "Hot (25–30°C)", "Very hot (≥30°C)"]


def assemble(prev: pd.DataFrame, nwp: dict, clim: pd.DataFrame, station_cols: list, nwp_models: list) -> pd.DataFrame:
    """prev: station features known the day before, indexed by target day D.
    nwp: {model: daily weather-model forecast for D}. Returns one row per D."""
    D = prev.index
    doy = D.dayofyear
    cmax, cmin = clim.loc[doy, "Max_Temp_C"].values, clim.loc[doy, "Min_Temp_C"].values
    cols = {}
    for c in station_cols:
        cols[f"st_{c}"] = prev[c].values
    cols["clim_rain_prob"] = clim.loc[doy, "rain_prob"].values
    cols["doy_sin"], cols["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    for m in nwp_models:
        df = nwp.get(m, pd.DataFrame(index=D)).reindex(D)
        for c in NWP_DAILY_COLS:
            cols[f"{m}_{c}"] = df[c].values.astype(float) if c in df else np.full(len(D), np.nan)
        cols[f"{m}_tmax_anom"] = cols[f"{m}_tmax"] - cmax
        cols[f"{m}_tmin_anom"] = cols[f"{m}_tmin"] - cmin
    X = pd.DataFrame(cols, index=D)
    extra = {}
    for v in ["tmax_anom", "tmin_anom", "precip", "cloud"]:
        sub = X[[f"{m}_{v}" for m in nwp_models]]
        extra[f"ens_{v}"], extra[f"ens_{v}_spread"] = sub.mean(axis=1), sub.std(axis=1)
    pr = X[[f"{m}_precip" for m in nwp_models]]
    extra["ens_wet_share"] = (pr >= 1).sum(axis=1) / pr.notna().sum(axis=1).replace(0, np.nan)
    X = pd.concat([X, pd.DataFrame(extra, index=D)], axis=1)
    return X


def band(t):
    return np.digitize(np.asarray(t, float), BAND_EDGES)


LAYER_SCALE = 0.526  # shrinks a 10–90% range to ~25–75% (normal approx.: 0.674 / 1.282)


def outfit(q10, q50, q90, scale: float = LAYER_SCALE):
    """Clothing recommendation with soft boundaries.
    Returns (main band, lowest band covered, highest band covered, add-a-layer flag).
    If the middle ~50% of the forecast range (25–75%) crosses a boundary next to the
    median's band, the advice covers both bands ("wear X, bring a layer") instead of
    guessing one. Using the inner range keeps "bring a layer" for genuinely close calls."""
    q50 = np.asarray(q50, float)
    lo_t = q50 - scale * (q50 - np.asarray(q10, float))
    hi_t = q50 + scale * (np.asarray(q90, float) - q50)
    m, lo, hi = band(q50), band(lo_t), band(hi_t)
    cov_lo, cov_hi = np.maximum(lo, m - 1), np.minimum(hi, m + 1)
    return m, cov_lo, cov_hi, cov_hi > cov_lo


# ---- Pruned feature set (MOS v2.1): fewer, more robust inputs for ~600 days of data
CORE_STATION = ["anom_Max_lag0", "anom_Min_lag0", "anom_Max_mean3", "anom_Min_mean3", "anom_Max_mean7",
                "anom_Min_mean7", "dtr", "precip_lag0", "precip_sum7", "wet_days_7", "days_since_wet"]
CORE_NWP = ["tmax_anom", "tmin_anom", "precip", "precip_hours", "cloud"]


def core_columns(nwp_models: list) -> list:
    cols = [f"st_{c}" for c in CORE_STATION] + ["clim_rain_prob", "doy_sin", "doy_cos"]
    cols += [f"{m}_{c}" for m in nwp_models for c in CORE_NWP]
    cols += [f"ens_{v}{s}" for v in ["tmax_anom", "tmin_anom", "precip", "cloud"] for s in ["", "_spread"]]
    return cols + ["ens_wet_share"]


def choose_layer_scale(q10, q50, q90, target_rate: float) -> float:
    """Pick how much of the forecast range counts as a 'close call' so that
    'bring a layer' appears on about `target_rate` of days."""
    scales = np.round(np.arange(0.05, 1.001, 0.01), 2)
    rates = np.array([outfit(q10, q50, q90, s)[3].mean() for s in scales])
    return float(scales[np.argmin(np.abs(rates - target_rate))])
