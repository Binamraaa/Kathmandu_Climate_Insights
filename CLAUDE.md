# CLAUDE.md: के लाउने? (K Laune) / Kathmandu Climate Insights

Day-ahead weather forecasting for Kathmandu Airport (DHM station), turned into a
bilingual "what to wear" web app. Owner: Binamra Bhusal (Binamraaa on GitHub).
Live app: https://binamraaa.github.io/Kathmandu_Climate_Insights/

## Environment

- Windows + Anaconda, env `ktm-climate`, Python 3.12. Repo folder: `C:\Users\ALiEN\Kathmandu_climate_history`.
- Run everything from the repo root as modules: `python -m src.<module>`.
- `xgboost==3.4.1` is pinned. Other versions load the JSON models but shift predictions; `model_io` warns.
- Full install: `requirements.txt`. The GitHub Action uses the lighter `requirements-app.txt`.

## Commands

```
python -m src.predict --replay 2023-07-15       # offline sanity check (≈29.5 °C max, 79% rain)
python -m src.export_forecast                    # live forecast → docs/forecast.json (+ verification log)
python -m src.export_forecast --mode morning     # force mode (default follows the Nepal clock)
python -m http.server 8000 -d docs               # app at http://localhost:8000
python -m src.fetch_observations --dry-run       # today's MFD bulletin (the Action saves it automatically)
python -m src.verify add 2026-10-14 --min 14.9   # observations (fields can be added in separate steps)
python -m unittest tests.test_fetch_observations # bulletin parser tests
python -m src.verify report                      # accuracy scorecard
python -m src.train_mos_v2 --prune               # retrain the live model (MOS v2.1)
node --check docs/app.js                         # quick JS syntax check
```

## Architecture

**Models** (`models/`, XGBoost JSON + `meta.json` + `climatology.csv`, saved/loaded via `src/model_io.py`):
- `mos_v2_1/`: **live model for the next forecast day.** Quantile XGBoost (q10/q50/q90, conformally widened) for Max/Min plus a rain classifier. Inputs: day-ahead forecasts from 6 Open-Meteo weather models (best_match, ECMWF IFS 0.25, ICON, GFS, JMA, CMA) and station features from the day before. Out-of-sample: Max 1.09 °C, Min 0.81 °C, rain Brier 0.117.
- `mos_v2/`: backup (8 models, 145 features). `predict.py` picks whichever saved model scores better.
- `champion_v2/`, `rain_v1/`: station-only models, used for days 2–3 and as fallback.

**Pipeline**
- `src/export_forecast.py`: mode from the Nepal clock. **Before 12:00 = morning**: uses data up to yesterday, forecasts today + 2 days. **From 12:00 = evening**: forecasts tomorrow onward. Merges with the existing `forecast.json` so today's morning forecast stays all day. Attaches extras, writes UTF-8 JSON, appends to the verification log (never in `--replay`).
- `src/extras.py`: UV, wind gusts, humidity, apparent-temperature offsets, 8 am / 7 pm position on the daily curve, AQI (CAMS). Raw Open-Meteo, **not** MOS-corrected.
- `src/live_input.py`, `src/nwp_hourly.py`: Open-Meteo access (Forecast, Previous Runs `_previous_day1`, historical-forecast, archive APIs).
- `src/features.py`, `src/mos_features.py`: shared feature code. Training and live prediction must use the same functions.
- `src/verify.py`: forecast log + observations + scorecard. Tags each row `run` = morning/evening.
- `src/fetch_observations.py`: reads the MFD bulletin from ashesh.com.np/weather (today's or yesterday's), applies the date rule, never overwrites recorded values without `--force`.
- `.github/workflows/forecast.yml`: morning crons at 05:25, 06:55, 08:25 NPT and evening crons at 15:32, 16:32, 17:32 NPT. Scheduled runs pass `--skip-if-done`, so the first to land does the work and the backups exit quietly; manual dispatch (with a mode input) always runs. Every run then tries `fetch_observations` (non-fatal). Commits as "K Laune forecast" (neutral author, so no bot shows up as a contributor) with "Forecast" / "Observations" / "Forecast + observations" messages, then `git pull --rebase` + push.

**App** (`docs/`, served by GitHub Pages from `/docs`, no build step)
- `index.html`, `style.css`, `app.js` (avatar engine + translations + app logic in one file). PWA files: `manifest.webmanifest`, `sw.js` (network-first service worker), `icons/` (192, 512, 512 maskable, 180 apple-touch).
- **`docs/app.js` is the source of truth.** Edit it directly; there is no separate source.
- Flow: first visit opens a 7-step avatar builder (age, man/woman/rather not say, skin tone, hair + facial hair + spectacles, style, walk/scooter, cold sensitivity). Main view: morning + afternoon looks (one "All day" look when both fall in the same clothing tier), "Other ideas" row, advice chips, per-look protection switches, day tabs.
- Clothing: 6 warmth tiers (<8, 8–13, 13–17, 17–21, 21–30, ≥30 °C) × styles (street, smart, dress, kurta, daura, sari) in `WARDROBE`; alternatives in `IDEAS`.
- Feels-like = temp + apparent offset − ride wind chill (4° morning, 2° afternoon) ± cold preference (2°).
- Protection: `need` comes from the day's weather; the avatar wears an item only if `wearPrefs(P)` allows it. Per look: `am` {cap, mask}, `pm` {cap, sunglasses, mask}. Sunglasses are never drawn on the morning look. Beanie beats cap on cold mornings; no cap with a dhaka topi. Advice chips always show regardless of prefs.
- Builder preview is a steady portrait (tier 4, no weather accessories; helmet if rider), so answers never cause flicker.
- Day selection: hide past days; open on today before 17:00 local, tomorrow after.
- Languages: `I18N.en` / `I18N.ne`. Nepali mode uses Devanagari for Nepali words and keeps English garment names as people say them (बिहान hoodie, दिउँसो T-shirt।). Keep it simple; avoid formal or bookish Nepali.

## Conventions and gotchas

- **Always write text files with `encoding="utf-8"`.** Windows defaults to cp1252 and once broke `forecast.json`.
- **Save models as XGBoost JSON** via `model_io`, never pickle/joblib ("input stream corrupted" history).
- **`docs/forecast.json` and `data/verification/forecast_log.csv` belong to the bot.** Don't commit laptop runs of `export_forecast`; discard with `git checkout -- docs/forecast.json data/verification/forecast_log.csv`.
- **`data/verification/observations.csv` is shared:** the bot adds bulletin values, people add the rest by hand. `git pull` before `verify add`, so hand edits don't conflict with the bot's rows.
- **Git routine: commit, then pull, then push.** The bot commits twice a day. Global config `pull.rebase=true`, `rebase.autoStash=true` is set. On conflicts in bot files, keep GitHub's version (`git checkout --ours …` during a rebase).
- **GitHub cron delays can be hours** (an evening run once landed at 00:16 NPT), and runs can be dropped (16 Oct 2026 morning). The clock-based mode handles correctness; backup crons + `--skip-if-done` handle coverage.
- **Never train MOS on Open-Meteo's historical-forecast archive.** It's stitched from near-0-hour forecasts (look-ahead leak). Use the Previous Runs API (`_previous_day1`, from Jan 2024).
- **Service worker is network-first.** If you change caching, keep it that way to avoid stale versions on phones; bump `CACHE` in `sw.js` when changing the cached file list.
- **MFD observation dates:** a morning bulletin's *min* belongs to that day; its *max* and *rain* (24 h to 08:45) belong to the previous day. Record min for D from D's bulletin, max and rain from D+1's. Source: dhm.gov.np/mfd (JS app) or the ashesh.com.np/weather listing.
- **NOAA GSOD** (station 444540) extends the DHM record (bias-adjusted monthly) but lags by months.
- **Keep Open-Meteo attribution (CC BY 4.0)** visible in the app and README.
- **Avatars never infer ethnicity or religion.** Users choose skin tone, hair and clothing; no religious markers.
- `.gitignore` excludes `data/nwp_cache*/`, `data/verification/export_log.txt`, `*_results.csv`, `data/gsod_kathmandu.csv`. `.gitattributes` makes GitHub count `docs/` in language stats and fixes line endings.

## Testing approach

- Python: `python -m src.predict --replay <date>` and `python -m src.export_forecast --replay <date>` (offline, not logged).
- App: `node --check docs/app.js`, then the local server; test on a phone in a private tab (caches).
- Live APIs can't be mocked by the Action; when changing fetch code, test locally first, then run the workflow manually.

## Open items

- Observation gaps: 15 Oct max/rain and 16–17 Oct (no archive of those bulletins); fill from NOAA GSOD when it updates.
- `verify report`: the "Days" column counts days with any observation; make it per metric.
- Daily: check the bot recorded the MFD observations (add by hand if the site was down). After ~30 verified days: consider a rolling bias correction; publish results in the README.
- Retrain MOS every ~3 months (or when GSOD updates): `fetch_gsod`, then `train_mos_v2 --prune`; compare with Max 1.09 / Min 0.81 / rain 0.117 before trusting it.
- Roadmap ideas: comfort feedback ("too cold / just right / too warm"), neighbourhood elevation adjustment (~0.6 °C per 100 m), more valley stations.
