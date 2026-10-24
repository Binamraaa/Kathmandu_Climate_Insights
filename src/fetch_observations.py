"""Record the daily observations for Kathmandu Airport automatically.

Reads the station's own SYNOP reports (WMO 44454, sent by DHM to the WMO network) from
OGIMET (https://www.ogimet.com) and saves them via `verify.add_observation`. These are
the same measurements as the MFD bulletin, but machine-readable, reachable from GitHub's
servers, and kept for days, so every run also fills gaps left by missed runs.

Which report gives what (times UTC, Nepal = UTC+5:45):
    max  of day D: 12 UTC report of D (17:45 NPT), section 3 group 1snTxTxTx
    min  of day D: 03 UTC report of D (08:45 NPT), section 3 group 2snTnTnTn
    rain of day D: 03 UTC report of D+1, group 6RRRt with t=4 (24 h to 08:45), the
                   same window as the MFD bulletin's rainfall

    python -m src.fetch_observations --dry-run    # show what would be saved
    python -m src.fetch_observations              # save (never overwrites existing values)
    python -m src.fetch_observations --force      # overwrite existing values
    python -m src.fetch_observations --days 30    # look further back (default 10 days)
"""
import argparse
import os
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from src.verify import OBS, add_observation

URL = "https://www.ogimet.com/cgi-bin/getsynop"
STATION = "44454"   # Kathmandu Airport (Tribhuvan International), DHM index 1030
USER_AGENT = "KLaune-observations/2.0 (+https://github.com/Binamraaa/Kathmandu_Climate_Insights)"
NPT = timezone(timedelta(hours=5, minutes=45))
LOOKBACK_DAYS = 10
TRACE_MM = 0.1
# A missing value is only worth a warning once its report should have arrived.
WARN_GAPS_DAYS = 7


# ------------------------------------------------------------------ SYNOP decoding
def section3(report: str) -> list[str]:
    """Groups of section 3 (between '333' and '555' or the end); [] for NIL reports."""
    groups = report.strip().rstrip("=").split()
    if "333" not in groups:
        return []
    out = []
    for g in groups[groups.index("333") + 1:]:
        if g == "555":
            break
        out.append(g)
    return out


def temperature(group: str) -> float | None:
    """'10276' → 27.6, '21015' → -1.5 (sign digit 1 = negative); '/' anywhere → None."""
    if len(group) != 5 or "/" in group or group[1] not in "01":
        return None
    v = int(group[2:]) / 10
    return -v if group[1] == "1" else v


def precipitation(group: str) -> tuple[float | None, int | None]:
    """'6RRRt' → (mm, period code t). RRR is whole mm; 990 = trace, 991–999 = 0.1–0.9 mm, 989 = 989 mm or more."""
    if len(group) != 5 or "/" in group:
        return None, None
    rrr, t = int(group[1:4]), int(group[4])
    if rrr == 990:
        return TRACE_MM, t
    if rrr > 990:
        return (rrr - 990) / 10, t
    if rrr == 989:
        return None, t
    return float(rrr), t


def first(groups: list[str], indicator: str) -> str | None:
    return next((g for g in groups if g.startswith(indicator)), None)


def parse_synops(text: str, station: str = STATION) -> dict[str, dict]:
    """OGIMET getsynop CSV → {day: {"max"/"min"/"rain_mm": value}} using the rules above."""
    by_day: dict[str, dict] = {}
    seen = False
    for line in text.splitlines():
        parts = line.strip().split(",", 6)
        if len(parts) < 7 or parts[0] != station:
            continue
        seen = True
        y, m, d, hh = (int(p) for p in parts[1:5])
        day = date(y, m, d)
        s3 = section3(parts[6])
        if hh == 12:
            g = first(s3, "1")
            v = temperature(g) if g else None
            if v is not None:
                by_day.setdefault(day.isoformat(), {})["max"] = v
        elif hh == 3:
            g = first(s3, "2")
            v = temperature(g) if g else None
            if v is not None:
                by_day.setdefault(day.isoformat(), {})["min"] = v
            g = first(s3, "6")
            mm, t = precipitation(g) if g else (None, None)
            if mm is not None and t == 4:
                by_day.setdefault((day - timedelta(days=1)).isoformat(), {})["rain_mm"] = mm
    if not seen:
        raise ValueError(f"No {station} reports in the OGIMET reply (starts: {text.strip()[:200]!r})")
    return dict(sorted(by_day.items()))


# ------------------------------------------------------------------ checks and saving
def check(vals: dict) -> list[str]:
    """Sanity problems with one day's values (empty list = fine)."""
    probs = []
    for k in ("max", "min"):
        v = vals.get(k)
        if v is not None and not -5 <= v <= 40:
            probs.append(f"{k} {v} °C is outside -5…40 °C")
    if vals.get("max") is not None and vals.get("min") is not None and vals["max"] <= vals["min"]:
        probs.append(f"max {vals['max']} is not above min {vals['min']}")
    if vals.get("rain_mm") is not None and not 0 <= vals["rain_mm"] <= 400:
        probs.append(f"rain {vals['rain_mm']} mm is outside 0…400 mm")
    return probs


def plan_updates(by_day: dict[str, dict], obs: pd.DataFrame, force: bool = False) -> tuple[dict, list[str]]:
    """Split into values to save and messages about values already recorded."""
    save, skipped = {}, []
    for day, fields in by_day.items():
        old = obs[obs["date"] == day]
        for k, v in fields.items():
            prev = old.iloc[0][k] if len(old) else None
            if prev is not None and pd.notna(prev):
                if float(prev) == v:
                    continue
                if not force:
                    skipped.append(f"{day} {k}: keeping {prev} (station report says {v}; use --force to overwrite)")
                    continue
            save.setdefault(day, {})[k] = v
    return save, skipped


def gaps(obs: pd.DataFrame, by_day: dict[str, dict], today: date, days: int = WARN_GAPS_DAYS) -> list[str]:
    """Recent days still missing a value after this run (today's max and yesterday's rain aren't due yet)."""
    out = []
    for i in range(days, 0, -1):
        d = (today - timedelta(days=i)).isoformat()
        old = obs[obs["date"] == d]
        have = {k for k in ("max", "min", "rain_mm") if len(old) and pd.notna(old.iloc[0][k])}
        have |= set(by_day.get(d, {}))
        due = ["min", "max"] + (["rain_mm"] if i >= 2 else [])
        missing = [k for k in due if k not in have]
        if missing:
            out.append(f"{d}: {', '.join(missing)}")
    return out


def fetch(begin: datetime, end: datetime, url: str = URL) -> str:
    params = {"block": STATION, "begin": begin.strftime("%Y%m%d%H%M"), "end": end.strftime("%Y%m%d%H%M")}
    r = requests.get(url, params=params, timeout=60, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.text


def warn(msg: str):
    """Plain print locally; a yellow annotation on the run page in GitHub Actions."""
    print(f"::warning title=Observations::{msg}" if os.environ.get("GITHUB_ACTIONS") else f"Warning: {msg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print what would be saved, save nothing")
    ap.add_argument("--force", action="store_true", help="overwrite values that are already recorded")
    ap.add_argument("--days", type=int, default=LOOKBACK_DAYS, help="how many days back to look")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    begin = (now - timedelta(days=args.days)).replace(hour=0, minute=0, second=0, microsecond=0)
    by_day = parse_synops(fetch(begin, now))
    print(f"SYNOP {STATION} via OGIMET, {begin:%Y-%m-%d} to {now:%Y-%m-%d %H:%M} UTC: values for {len(by_day)} days")

    bad = {d: p for d, v in by_day.items() if (p := check(v))}
    for d, p in bad.items():
        warn(f"{d} skipped, failed the sanity check: {'; '.join(p)}")
    by_day = {d: v for d, v in by_day.items() if d not in bad}

    obs = pd.read_csv(OBS) if OBS.exists() else pd.DataFrame(columns=["date", "max", "min", "rain_mm", "note"])
    save, skipped = plan_updates(by_day, obs, args.force)
    for s in skipped:
        print(f"  skip {s}")
    missing = gaps(obs, by_day, datetime.now(NPT).date())
    if missing:
        warn("no station report yet for " + "; ".join(missing) + " (add from the MFD bulletin if it stays missing)")
    if not save:
        print("Nothing new to save.")
        return
    note = "auto: SYNOP 44454 via OGIMET"
    for day, f in save.items():
        if args.dry_run:
            print(f"  would save {day}: " + ", ".join(f"{k} {v}" for k, v in f.items()))
        else:
            add_observation(day, f.get("max"), f.get("min"), f.get("rain_mm"), note)


if __name__ == "__main__":
    main()
