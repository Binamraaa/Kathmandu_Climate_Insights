"""Hourly weather-model data from Open-Meteo, aggregated to Kathmandu days.

Training uses the Previous Runs API: `<var>_previous_day1` is the value each model
predicted 24 hours before the valid time, i.e. an honest day-ahead forecast
(archive mostly from January 2024). Live use takes tomorrow from the regular
Forecast API. Both are aggregated to identical daily features.
"""
import pandas as pd
import requests

from src.live_input import FORECAST_URL, KTM_AIRPORT, TZ

PREV_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
BASE_VARS = ["temperature_2m", "precipitation", "cloud_cover", "relative_humidity_2m", "wind_speed_10m", "pressure_msl"]
CORE_VARS = ["temperature_2m", "precipitation"]


def _request(url: str, variables: list[str], **params) -> pd.DataFrame:
    r = requests.get(url, params={**KTM_AIRPORT, "timezone": TZ, "hourly": ",".join(variables), **params}, timeout=90)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({k: v for k, v in h.items() if k != "time"}, index=pd.to_datetime(h["time"]))
    return df.astype(float)


def _with_fallback(url, variables_for, **params):
    """Try all variables, then temperature + precipitation only (some models lack some fields)."""
    last = None
    for base in (BASE_VARS, CORE_VARS):
        try:
            df = _request(url, variables_for(base), **params)
            return df, base
        except requests.HTTPError as e:
            last = e
    raise last


def fetch_previous_runs(model: str, start: str, end: str, lead_day: int = 1) -> pd.DataFrame:
    """Hourly day-ahead forecasts for [start, end], fetched in ~6-month chunks."""
    suffix = f"_previous_day{lead_day}"
    chunks = []
    for s, e in _date_chunks(start, end, months=6):
        df, base = _with_fallback(PREV_RUNS_URL, lambda b: [v + suffix for v in b],
                                  models=model, start_date=s, end_date=e)
        chunks.append(df.rename(columns={v + suffix: v for v in base}))
    return pd.concat(chunks).sort_index()


def fetch_live(model: str) -> pd.DataFrame:
    """Hourly latest forecast covering today and tomorrow."""
    df, _ = _with_fallback(FORECAST_URL, lambda b: b, models=model, past_days=1, forecast_days=3)
    return df


def to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly values to local calendar days (needs ≥ 20 hours that day)."""
    g = hourly.groupby(hourly.index.normalize())
    n = g["temperature_2m"].count()
    d = pd.DataFrame(index=n.index)
    d["tmax"], d["tmin"] = g["temperature_2m"].max(), g["temperature_2m"].min()
    d["t_mean"] = g["temperature_2m"].mean()
    d["precip"] = g["precipitation"].sum(min_count=20)
    d["precip_hours"] = g["precipitation"].apply(lambda x: (x >= 0.1).sum())
    if "cloud_cover" in hourly:
        d["cloud"] = g["cloud_cover"].mean()
        aft = hourly[(hourly.index.hour >= 11) & (hourly.index.hour <= 16)]
        d["cloud_afternoon"] = aft.groupby(aft.index.normalize())["cloud_cover"].mean()
    if "relative_humidity_2m" in hourly:
        d["rh"] = g["relative_humidity_2m"].mean()
    if "wind_speed_10m" in hourly:
        d["wind_max"] = g["wind_speed_10m"].max()
    if "pressure_msl" in hourly:
        d["pmsl"] = g["pressure_msl"].mean()
    d[n < 20] = float("nan")
    d.index.name = "Date"
    return d


def _date_chunks(start: str, end: str, months: int):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    while s <= e:
        c = min(s + pd.DateOffset(months=months) - pd.Timedelta(days=1), e)
        yield s.strftime("%Y-%m-%d"), c.strftime("%Y-%m-%d")
        s = c + pd.Timedelta(days=1)
