# K Laune / Kathmandu Climate Insights: setup and daily use

Windows + Anaconda. Project folder: `C:\Users\ALiEN\Kathmandu_climate_history`.
**Always run commands from this folder** (the one that contains `src\`), after `conda activate ktm-climate`.

## Where things stand

| Piece | Status |
| --- | --- |
| **MOS v2.1 (pruned)** | Live model for tomorrow. Max error 1.09 °C, Min 0.81 °C, rain Brier 0.117. Picked automatically. |
| MOS v2 | Backup (Max 1.06, Min 0.85, rain 0.129). Use with `--mos mos_v2`. |
| Champion v2 / Rain v1 | Days 2 and 3, and fallback for tomorrow if weather-model data is missing. |
| App (`docs/index.html`) | Avatar builder, outfit ideas, English / Roman Nepali, scooter mode. |
| Verification log | Logs every live forecast; you add observed weather; `verify report` scores it. |

## Daily routine

| When | What | Command |
| --- | --- | --- |
| Every evening, 20:00 | Forecast + app update + log (automatic once scheduled, section 4) | `python -m src.export_forecast` |
| Next day, after the MFD/DHM bulletin | Record yesterday's actual weather | `python -m src.verify add 2026-10-13 --max 27.5 --min 14.8 --rain 0` |
| Any time | See the app | `python -m http.server 8000 -d docs`, then open http://localhost:8000 |
| Weekly | Accuracy scorecard | `python -m src.verify report` |

---

## 1. Repo layout

```
Kathmandu_climate_history/
├── data/
│   ├── df_analysis.csv           DHM station record, 1999 → Oct 2024
│   ├── gsod_kathmandu.csv        NOAA station data (section 7a)
│   ├── nwp_cache/, nwp_cache_v2/ weather-model downloads (not committed)
│   └── verification/
│       ├── forecast_log.csv      every live forecast, written automatically
│       ├── observations.csv      what actually happened, added by you
│       └── export_log.txt        output of the scheduled 20:00 run
├── models/
│   ├── champion_v2/, rain_v1/    station-only models (days 2–3)
│   ├── mos_v2/, mos_v2_1/        MOS models (tomorrow); predict.py picks the better one
│   └── openmeteo_calibration.json   live-input bias correction (section 3)
├── notebooks/                    champion_v2.ipynb, rain_classifier.ipynb
├── src/
│   ├── features.py, model_io.py, mos_features.py   shared code
│   ├── live_input.py, nwp_hourly.py                live Open-Meteo data
│   ├── calibrate_openmeteo.py                      builds the live-input correction
│   ├── predict.py                                  3-day forecast in the terminal
│   ├── export_forecast.py                          writes docs/forecast.json + logs it
│   ├── verify.py                                   observations + scorecard
│   ├── fetch_gsod.py, benchmark_nwp.py             data and benchmark checks
│   └── train_mos.py, train_mos_v2.py               MOS training
├── docs/
│   ├── index.html                the K Laune app
│   └── forecast.json             tonight's forecast
├── .github/workflows/forecast.yml   daily job once on GitHub
├── run_forecast.bat              scheduled 20:00 run on your laptop
├── requirements.txt, requirements-app.txt
├── SETUP.md
└── .gitignore
```

Updating from a new zip: extract it into the project folder and allow overwrites. Zips never include `data\` or your `mos_v2*` model folders, so those are safe. Don't delete the `models` folder before extracting.

## 2. One-time environment setup

```
conda create -n ktm-climate python=3.12 -y
conda activate ktm-climate
pip install -r requirements.txt
pip show xgboost
```

`pip show xgboost` must say **3.4.1** (otherwise `pip install xgboost==3.4.1`). Other versions load the models but shift the forecasts slightly; `predict.py` warns you.

Offline check: `python -m src.predict --replay 2023-07-15` should print about 29.5 °C max and 79% rain for 16 July 2023, with the actual values beside it.

## 3. Live forecast

Once, with internet, build the live-input correction:

```
python -m src.calibrate_openmeteo
```

Then:

```
python -m src.predict                 # forecast in the terminal
python -m src.predict --json          # same, as JSON
python -m src.export_forecast         # update the app + log the forecast
```

The `export_forecast` line should end with `[MOS v2.1 (pruned) (6 weather models)]`. Run it after about 6 pm, when today's high and rain total are settled.

## 4. Run it automatically every evening (Windows)

`run_forecast.bat` goes to the project folder, activates `ktm-climate`, and runs `export_forecast`, appending all output to `data\verification\export_log.txt`. If Anaconda is not in `C:\Users\ALiEN\anaconda3`, edit that line in the file.

Schedule it once:

```
schtasks /create /tn "K Laune forecast" /tr "C:\Users\ALiEN\Kathmandu_climate_history\run_forecast.bat" /sc daily /st 20:00
schtasks /run /tn "K Laune forecast"
```

The second line runs it straight away as a test; check `export_log.txt` afterwards. The laptop must be on at 20:00. In Task Scheduler, open "K Laune forecast" → Settings and tick **"Run task as soon as possible after a scheduled start is missed"**. Remove it with `schtasks /delete /tn "K Laune forecast" /f` once GitHub takes over (section 6).

## 5. The app (K Laune)

`docs/index.html` is a single file: no build step. It reads `docs/forecast.json` (shows "Sample data" without it).

- **First visit:** a 7-question avatar builder (age, man / woman / rather not say, skin tone, hair, usual style, walk or scooter, how much the cold bothers them). "My avatar" in the header reopens it.
- **Main view:** morning and afternoon avatars in the recommended outfits, a one-line headline, rain chip (umbrella, or raincoat for riders), and the next two days.
- **Outfit ideas:** a row of four mini avatars under the main look; the first is the suggestion, the rest are alternatives for the same weather. Tap the morning avatar to see morning ideas.
- **Language:** the नेपाली / EN button switches between English and Roman Nepali.

View locally: `python -m http.server 8000 -d docs`, then http://localhost:8000. On your phone (same Wi-Fi): `http://<laptop IPv4 from ipconfig>:8000`. Double-clicking `index.html` won't load the forecast; browsers block that for local files.

## 6. Push to GitHub and go live

`.gitignore` should contain:

```
.ipynb_checkpoints/
__pycache__/
*.~lock*
data/nwp_cache/
data/nwp_cache_v2/
```

Commit and push:

```
git add src notebooks models docs data/verification requirements.txt requirements-app.txt .github run_forecast.bat SETUP.md .gitignore
git commit -m "K Laune: MOS v2.1 live, outfit ideas, verification log"
git push
```

`data/gsod_kathmandu.csv` is NOAA public data and fine to publish. Check DHM's terms before publishing `df_analysis.csv`.

Then on GitHub:

1. **Settings → Pages → Deploy from a branch**, branch `main`, folder `/docs`. Save.
2. **Settings → Actions → General → Workflow permissions → Read and write permissions.**
3. **Actions → "Daily forecast" → Run workflow** to create the first live `forecast.json`.
4. The app is at `https://<your-username>.github.io/<repo-name>/`. The job runs every evening at 18:15 Nepal time and commits the forecast and log. On a phone, "Add to Home Screen" makes it feel like an app.

Once the Action runs, delete the Windows task (section 4), and `git pull` before adding observations.

## 7. Data checks, benchmarks and training (reference)

**a) NOAA station data:** `python -m src.fetch_gsod` (add `--from 2022` for speed). Downloads GSOD for station 444540 (VNKT), saves `data/gsod_kathmandu.csv`, compares with DHM. Last check: MAE 0.50 °C (Max) and 0.49 °C (Min), good enough to extend the record.

**b) Weather models vs our model:** `python -m src.benchmark_nwp`. Saves `benchmark_nwp_results.csv`.

**c) MOS v1:** `python -m src.train_mos`. Historical-forecast archive; scores flatter than reality. Kept for comparison only.

**d) MOS v2 and v2.1:** honest day-ahead training (Previous Runs API, January 2024 onward), forecast ranges, tuned umbrella and layer rules.

```
python -m src.train_mos_v2 --prune    # MOS v2.1 → models/mos_v2_1/ (current live model)
python -m src.train_mos_v2            # MOS v2   → models/mos_v2/
```

First run downloads hourly data (10–20 min); later runs reuse `data/nwp_cache_v2/`. Both tune "bring a layer" to about 28% of days (`LAYER_TARGET` in `train_mos_v2.py` changes this).

**e) When to retrain:** every ~3 months or when NOAA publishes newer GSOD data:

```
python -m src.fetch_gsod
python -m src.train_mos_v2 --prune
```

Compare the new table with the current one (Max 1.09, Min 0.81, rain 0.117) before trusting it. The notebooks (Champion v2 / Rain v1) only need rerunning if something breaks: open them in Jupyter or VS Code with the `ktm-climate` kernel and Run All.

## 8. Track real-world accuracy

Every live `export_forecast` run (laptop or GitHub Action) appends the 3-day forecast, plus a raw Open-Meteo forecast as a "typical weather app" reference, to `data/verification/forecast_log.csv`. Replays are never logged.

Record what actually happened (observed high/low from the MFD/DHM daily bulletin):

```
python -m src.verify add 2026-10-13 --max 27.5 --min 14.8 --rain 0 --note "MFD bulletin"
```

`--rain` is in mm (0 if dry; leave out if unknown). Re-entering a date replaces it. Future dates are refused.

Scorecard:

```
python -m src.verify report
```

For days 1–3: error and warm/cold bias for high and low, clothing-band accuracy, how often the actual high fell inside the forecast range, rain caught vs false alarms, side by side with the typical-app reference. About 30 verified days gives a fair picture; that's also the point to consider a rolling bias correction.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `No module named 'src'` | You're not in the project folder. `cd` to the folder that contains `src\`. |
| `No module named src.<something>` | That file is missing from `src\`. Re-extract the latest zip into the project folder. |
| `XGBoostError: input stream corrupted` | Old `.joblib` files in `models\`. Delete them. |
| Warning: "trained with xgboost 3.4.1 but you have …" | `pip install xgboost==3.4.1` |
| "No calibration file" warning | `python -m src.calibrate_openmeteo` |
| "weather model … unavailable" | Fine if at least two still work; otherwise tomorrow falls back to Champion v2. |
| `Not enough history to build features` | Open-Meteo returned gaps. Run again later. |
| Scheduled run did nothing | Check `data\verification\export_log.txt`; check the Anaconda path in `run_forecast.bat`; was the laptop on at 20:00? |
| App shows "Sample data" | No `docs\forecast.json` yet. Run `export_forecast` (or the GitHub Action). |
| GitHub Action fails at `git push` | Settings → Actions → General → Read and write permissions. |
| `verify report` says no observations | Add some with `verify add` for days that already have logged forecasts. |
| `requests.exceptions.ConnectionError` | No internet, or a firewall/VPN blocking the API. |
