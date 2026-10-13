"""Extra weather details for the clothing advice: UV, wind, humidity, air quality,
and the shape of the day (how warm it is at 8 am and 7 pm).

These come straight from Open-Meteo (weather models, and CAMS for air quality); they
are not corrected by our MOS model. Temperatures stay MOS; the extras only refine
the advice (sun protection, wind, mugginess, mask, evening layer).
"""
import numpy as np
import pandas as pd
import requests

from src.live_input import FORECAST_URL, KTM_AIRPORT, TZ

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
MORNING_HOUR, EVENING_HOUR = 8, 19


def _get(url, **params):
    r = requests.get(url, params={**params, "timezone": TZ}, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_extras(days: int = 4) -> dict:
    """{ 'YYYY-MM-DD': {uv_max, gust_max, rh_mean, feels_max_adj, feels_min_adj,
                        morning_frac, evening_frac, aqi_max, pm25_max} }"""
    out = {}
    w = _get(FORECAST_URL, **KTM_AIRPORT, forecast_days=days,
             daily="uv_index_max,wind_gusts_10m_max,temperature_2m_max,temperature_2m_min,"
                   "apparent_temperature_max,apparent_temperature_min,relative_humidity_2m_mean",
             hourly="temperature_2m")
    dd = w["daily"]
    hourly = pd.Series(w["hourly"]["temperature_2m"], index=pd.to_datetime(w["hourly"]["time"]), dtype=float)
    for i, day in enumerate(dd["time"]):
        tx, tn = dd["temperature_2m_max"][i], dd["temperature_2m_min"][i]
        e = {"uv_max": dd["uv_index_max"][i], "gust_max": dd["wind_gusts_10m_max"][i],
             "rh_mean": (dd.get("relative_humidity_2m_mean") or [None] * len(dd["time"]))[i]}
        if None not in (tx, tn, dd["apparent_temperature_max"][i], dd["apparent_temperature_min"][i]):
            # how much warmer/colder it feels than the thermometer says (humidity, wind, sun)
            e["feels_max_adj"] = round(float(np.clip(dd["apparent_temperature_max"][i] - tx, -3, 3)), 1)
            e["feels_min_adj"] = round(float(np.clip(dd["apparent_temperature_min"][i] - tn, -3, 3)), 1)
        h = hourly[hourly.index.normalize() == pd.Timestamp(day)]
        if len(h) >= 20 and tx is not None and tn is not None and tx - tn > 0.5:
            # where 8 am and 7 pm sit between the day's low and high, from the models' daily curve
            for k, hr in [("morning_frac", MORNING_HOUR), ("evening_frac", EVENING_HOUR)]:
                v = h[h.index.hour == hr]
                if len(v):
                    e[k] = round(float(np.clip((v.iloc[0] - tn) / (tx - tn), 0, 1)), 2)
        out[day] = e
    try:
        aq = _get(AQ_URL, latitude=KTM_AIRPORT["latitude"], longitude=KTM_AIRPORT["longitude"],
                  hourly="us_aqi,pm2_5", forecast_days=days)
        a = pd.DataFrame({"aqi": aq["hourly"]["us_aqi"], "pm25": aq["hourly"]["pm2_5"]},
                         index=pd.to_datetime(aq["hourly"]["time"]), dtype=float)
        a = a[(a.index.hour >= 7) & (a.index.hour <= 20)]          # hours people are out
        for day, g in a.groupby(a.index.normalize()):
            key = day.strftime("%Y-%m-%d")
            if key in out and g["aqi"].notna().any():
                out[key]["aqi_max"] = int(g["aqi"].max())
                out[key]["pm25_max"] = round(float(g["pm25"].max()), 1)
    except Exception as e:  # noqa: BLE001 — air quality is a bonus, never a blocker
        print(f"  (air quality unavailable: {e})")
    return out


def attach(fc: dict, extras: dict) -> dict:
    for d in fc["days"]:
        if d["date"] in extras:
            d["extras"] = {k: v for k, v in extras[d["date"]].items() if v is not None}
    return fc
