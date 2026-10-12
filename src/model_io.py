"""Save/load models in XGBoost's native JSON format plus plain metadata files.

Avoids pickle (.joblib), which breaks across xgboost versions, OS builds and
file transfers ("input stream corrupted"). JSON files are plain text and load on
other versions too, but predictions are only guaranteed identical on the xgboost
version they were trained with (recorded in meta.json), so a mismatch warns.
"""
import warnings
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _write_meta(folder: Path, meta: dict, clim: pd.DataFrame):
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"xgboost_version": xgb.__version__, **meta}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    clim.to_csv(folder / "climatology.csv", index_label="Day_of_Year")


def _read_meta(folder: Path):
    meta = json.loads((folder / "meta.json").read_text())
    clim = pd.read_csv(folder / "climatology.csv", index_col="Day_of_Year")
    trained = meta.get("xgboost_version")
    if trained and trained.split(".")[:2] != xgb.__version__.split(".")[:2]:
        warnings.warn(f"{folder.name} was trained with xgboost {trained} but you have {xgb.__version__}. "
                      f"Forecasts may differ. Fix: pip install xgboost=={trained}", stacklevel=2)
    return meta, clim


def save_temp_models(models: dict, features: list, clim: pd.DataFrame, extra: dict | None = None, name="champion_v2"):
    folder = MODELS_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    for (k, h), m in models.items():
        m.save_model(folder / f"{k.lower()}_t{h}.json")
    _write_meta(folder, {"features": features, "keys": [[k, h] for k, h in models], **(extra or {})}, clim)
    print(f"Saved {len(models)} models to {folder}")


def load_temp_models(name="champion_v2") -> dict:
    folder = MODELS_DIR / name
    meta, clim = _read_meta(folder)
    models = {}
    for k, h in meta["keys"]:
        m = xgb.XGBRegressor()
        m.load_model(folder / f"{k.lower()}_t{h}.json")
        models[(k, h)] = m
    return {"models": models, "features": meta["features"], "climatology": clim, **meta}


def save_rain_models(models: dict, features: list, clim: pd.DataFrame, umbrella_threshold: float,
                     wet_day_mm: float, extra: dict | None = None, name="rain_v1"):
    folder = MODELS_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    for h, m in models.items():
        m.save_model(folder / f"t{h}.json")
    _write_meta(folder, {"features": features, "horizons": list(models), "umbrella_threshold": umbrella_threshold,
                         "wet_day_mm": wet_day_mm, **(extra or {})}, clim)
    print(f"Saved {len(models)} models to {folder}")


def load_rain_models(name="rain_v1") -> dict:
    folder = MODELS_DIR / name
    meta, clim = _read_meta(folder)
    models = {}
    for h in meta["horizons"]:
        m = xgb.XGBClassifier()
        m.load_model(folder / f"t{h}.json")
        models[int(h)] = m
    return {"models": models, "features": meta["features"], "climatology": clim, **meta}
