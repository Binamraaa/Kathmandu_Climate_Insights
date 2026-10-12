"""Live weather input from Open-Meteo (free, no API key).

The models were trained on DHM station data (Kathmandu Airport). Open-Meteo
values are model/gridded estimates, so they are bias-corrected with a
calibration file built by `python -m src.calibrate_openmeteo`.
"""
import json
from pathlib import Path

import pandas as pd
import requests

# Tribhuvan International Airport (DHM Kathmandu Airport station)
KTM_AIRPORT = {"latitude": 27.6966, "longitude": 85.3591, "elevation": 1338}
TZ = "Asia/Kathmandu"
DAILY = "temperature_2m_max,temperature_2m_min,precipitation_sum"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "models" / "openmeteo_calibration.json"


def _get(url: str, **params) -> pd.DataFrame:
    params = {**KTM_AIRPORT, "daily": DAILY, "timezone": TZ, **params}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame(
        {
            "Max_Temp_C": d["temperature_2m_max"],
            "Min_Temp_C": d["temperature_2m_min"],
            "Precipitation_mm": d["precipitation_sum"],
        },
        index=pd.to_datetime(d["time"]),
    )
    df.index.name = "Date"
    return df.astype(float)


def fetch_recent(past_days: int = 45) -> pd.DataFrame:
    """Last `past_days` days up to and including today (today is partly forecast)."""
    return _get(FORECAST_URL, past_days=past_days, forecast_days=1)


EXTRA_DAILY = "cloud_cover_mean,relative_humidity_2m_mean,precipitation_hours,wind_speed_10m_max,shortwave_radiation_sum"
NWP_RENAME = {"temperature_2m_max": "tmax", "temperature_2m_min": "tmin", "precipitation_sum": "precip",
              "cloud_cover_mean": "cloud", "relative_humidity_2m_mean": "rh", "precipitation_hours": "precip_hours",
              "wind_speed_10m_max": "wind_max", "shortwave_radiation_sum": "sw_rad"}


def fetch_nwp_full(model: str, url: str = HISTORICAL_FORECAST_URL, **params) -> pd.DataFrame:
    """All useful daily variables from one weather model. Falls back to the core
    variables if the extra ones aren't available for that model."""
    base = {**KTM_AIRPORT, "timezone": TZ, "models": model, **params}
    for daily in (DAILY + "," + EXTRA_DAILY, DAILY):
        r = requests.get(url, params={**base, "daily": daily}, timeout=60)
        if r.ok:
            d = r.json()["daily"]
            df = pd.DataFrame({NWP_RENAME[k]: v for k, v in d.items() if k in NWP_RENAME},
                              index=pd.to_datetime(d["time"]))
            df.index.name = "Date"
            return df.astype(float)
    r.raise_for_status()


def fetch_history_model(start: str, end: str, model: str) -> pd.DataFrame:
    """Past forecasts from one weather model (e.g. ecmwf_ifs025, gfs_seamless, icon_seamless, best_match)."""
    return _get(HISTORICAL_FORECAST_URL, start_date=start, end_date=end, models=model)


def fetch_history(start: str, end: str) -> pd.DataFrame:
    """Past values from the same weather models the forecast API uses.
    Falls back to the reanalysis archive if the historical-forecast API fails."""
    try:
        return _get(HISTORICAL_FORECAST_URL, start_date=start, end_date=end)
    except Exception as e:  # noqa: BLE001
        print(f"Historical-forecast API failed ({e}); using reanalysis archive instead.")
        return _get(ARCHIVE_URL, start_date=start, end_date=end)


def load_calibration(path: Path = CALIBRATION_PATH):
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def apply_calibration(df: pd.DataFrame, cal: dict | None) -> pd.DataFrame:
    """Shift temps by the monthly station-minus-Open-Meteo bias; rescale precip so the
    wet-day threshold matches the station's."""
    if cal is None:
        return df
    out = df.copy()
    m = out.index.month.astype(str)
    out["Max_Temp_C"] += m.map(cal["max_bias_by_month"]).astype(float).values
    out["Min_Temp_C"] += m.map(cal["min_bias_by_month"]).astype(float).values
    out["Precipitation_mm"] *= cal["precip_scale"]
    return out
