"""Shared feature engineering for Kathmandu Climate Insights.

Used by the notebooks (training) and by predict.py (live forecasts), so the
model always sees features built exactly the same way.

Input: a daily DataFrame indexed by date with columns
    Max_Temp_C, Min_Temp_C, Precipitation_mm
"""
import numpy as np
import pandas as pd

VARS = {"Max": "Max_Temp_C", "Min": "Min_Temp_C"}
WET_DAY_MM = 1.0  # WMO wet-day threshold


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    doy = df.index.dayofyear
    df["Day_of_Year"] = doy
    df["Day_of_Year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["Day_of_Year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def _circular_smooth(s: pd.Series, pad: int = 15) -> pd.Series:
    s = s.reindex(range(1, 367)).interpolate().bfill().ffill()
    out = pd.concat([s.iloc[-pad:], s, s.iloc[:pad]]).rolling(2 * pad + 1, center=True).mean().iloc[pad:-pad]
    out.index = range(1, 367)
    return out


def fit_climatology(train: pd.DataFrame) -> pd.DataFrame:
    """Smoothed day-of-year means for Max/Min temp and wet-day probability (training data only)."""
    train = add_calendar(train)
    g = train.groupby("Day_of_Year")
    clim = pd.DataFrame({col: _circular_smooth(g[col].mean()) for col in VARS.values()})
    clim["rain_prob"] = _circular_smooth(g["Precipitation_mm"].apply(lambda x: (x >= WET_DAY_MM).mean()))
    return clim


def build_features(daily: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    """Add every model feature. All features use data up to and including day t only."""
    df = add_calendar(daily)
    for k, col in VARS.items():
        df[f"clim_{k}"] = clim[col].reindex(df["Day_of_Year"]).values
        df[f"anom_{k}"] = df[col] - df[f"clim_{k}"]
    P = df["Precipitation_mm"]
    for k in VARS:
        a = df[f"anom_{k}"]
        for L in [0, 1, 2, 3, 7]:
            df[f"anom_{k}_lag{L}"] = a.shift(L)
        for w in [3, 7, 30]:
            df[f"anom_{k}_mean{w}"] = a.rolling(w).mean()
        df[f"anom_{k}_std7"] = a.rolling(7).std()
        df[f"anom_{k}_diff1"] = a.diff()
    for L in [0, 1, 2, 3]:
        df[f"precip_lag{L}"] = P.shift(L)
    for w in [3, 7, 30]:
        # mean × window = sum on complete data; tolerates gaps (e.g. GSOD missing rain reports)
        df[f"precip_sum{w}"] = P.rolling(w, min_periods=max(1, int(w * 0.6))).mean() * w
    df["rain_today"] = (P > 0.1).astype(int)
    df["dtr"] = df["Max_Temp_C"] - df["Min_Temp_C"]
    df["dtr_mean7"] = df["dtr"].rolling(7).mean()

    # Extra features for the rain classifier
    wet = (P >= WET_DAY_MM).astype(float).where(P.notna())
    df["wet_days_7"] = wet.rolling(7, min_periods=4).mean() * 7
    df["wet_days_30"] = wet.rolling(30, min_periods=18).mean() * 30
    wet = wet.fillna(0).astype(int)
    run = wet.groupby((wet == 1).cumsum()).cumcount()  # days since last wet day
    df["days_since_wet"] = np.where(wet == 1, 0, run).clip(0, 60)
    df["rain_clim_prob"] = clim["rain_prob"].reindex(df["Day_of_Year"]).values
    df["rain_clim_prob_t1"] = clim["rain_prob"].reindex(((df["Day_of_Year"]) % 366) + 1).values
    return df


TEMP_FEATURES_PREFIXES = ("anom_Max_", "anom_Min_", "precip_", "rain_today", "dtr")


def temp_feature_list(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith(TEMP_FEATURES_PREFIXES)] + ["Day_of_Year_sin", "Day_of_Year_cos"]


RAIN_EXTRA = ["wet_days_7", "wet_days_30", "days_since_wet", "rain_clim_prob", "rain_clim_prob_t1"]


def load_station_csv(path) -> pd.DataFrame:
    """Load df_analysis.csv as a clean daily frame."""
    raw = pd.read_csv(path)
    raw["Date"] = pd.to_datetime(raw["Year"].astype(str) + "-" + raw["Day_of_Year"].astype(str), format="%Y-%j")
    return raw.set_index("Date").sort_index()[["Max_Temp_C", "Min_Temp_C", "Precipitation_mm"]]
