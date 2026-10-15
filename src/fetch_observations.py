"""Record the daily MFD observations for Kathmandu automatically.

Reads https://www.ashesh.com.np/weather/ (a plain-HTML mirror of the MFD/DHM daily
observations) and saves the Kathmandu row via `verify.add_observation`.

Date rule: in a bulletin dated D, the MIN belongs to day D; the MAX and RAIN
(24 h to 08:45) belong to day D-1.

    python -m src.fetch_observations --dry-run    # show what would be saved
    python -m src.fetch_observations              # save (never overwrites existing values)
    python -m src.fetch_observations --force      # overwrite existing values
"""
import argparse
import re
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser

import pandas as pd
import requests

from src.verify import OBS, add_observation

URL = "https://www.ashesh.com.np/weather/"
USER_AGENT = "KLaune-observations/1.0 (+https://github.com/Binamraaa/Kathmandu_Climate_Insights)"
NPT = timezone(timedelta(hours=5, minutes=45))
STATION = "Kathmandu"
# The page is a list of div blocks styled from a currency template, not an HTML table.
# Header reads "Max°C Min°C Rain-㎜"; the cells are float:right, so the DOM order is reversed.
COLUMNS = {"unit": "max", "rate_buying": "min", "rate_selling": "rain_mm"}
TRACE_MM = 0.1
MISSING = {"", "-", "--"}


class _StationParser(HTMLParser):
    """Collects the text of each child div of every <div class="country"> block, by class."""

    def __init__(self):
        super().__init__()
        self.rows = {}
        self._row = None
        self._field = None
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        if tag != "div":
            return
        cls = (dict(attrs).get("class") or "").strip()
        if self._row is None:
            if cls == "country":
                self._row, self._depth = {}, 1
            return
        self._depth += 1
        self._field = cls or None
        if cls:
            self._row.setdefault(cls, "")

    def handle_endtag(self, tag):
        if tag != "div" or self._row is None:
            return
        self._depth -= 1
        self._field = None
        if self._depth == 0:
            name = self._row.get("name", "").strip()
            if name:
                self.rows[name] = {k: v.strip() for k, v in self._row.items()}
            self._row = None

    def handle_data(self, data):
        if self._row is not None and self._field:
            self._row[self._field] += data


def parse_value(text: str, rain: bool = False) -> float | None:
    """'27.5' → 27.5; 'Trace' → 0.1 (rain only); '-', '--' or '*' (previous record) → None."""
    t = text.strip()
    if t in MISSING or t.endswith("*"):
        return None
    if t.lower() == "trace":
        return TRACE_MM if rain else None
    return float(t)


def parse_page(html: str, station: str = STATION) -> tuple[date, dict]:
    """Bulletin date and {"max", "min", "rain_mm"} for one station (None where missing)."""
    m = re.search(r"updated on\.?\s*(\d{4}-\d{2}-\d{2})", html, re.IGNORECASE)
    if not m:
        raise ValueError("No 'Weather data updated on' date found: has the page layout changed?")
    p = _StationParser()
    p.feed(html)
    row = p.rows.get(station)
    if row is None:
        raise ValueError(f"No {station} row found (stations seen: {', '.join(p.rows) or 'none'})")
    missing = [c for c in COLUMNS if c not in row]
    if missing:
        raise ValueError(f"{station} row lacks columns {missing}: has the page layout changed?")
    vals = {k: parse_value(row[c], rain=(k == "rain_mm")) for c, k in COLUMNS.items()}
    return date.fromisoformat(m.group(1)), vals


def check(vals: dict) -> list[str]:
    """Sanity problems with the parsed values (empty list = fine)."""
    probs = []
    for k in ("max", "min"):
        v = vals[k]
        if v is not None and not -5 <= v <= 40:
            probs.append(f"{k} {v} °C is outside -5…40 °C")
    if vals["max"] is not None and vals["min"] is not None and vals["max"] <= vals["min"]:
        probs.append(f"max {vals['max']} is not above min {vals['min']}")
    if vals["rain_mm"] is not None and vals["rain_mm"] < 0:
        probs.append(f"rain {vals['rain_mm']} mm is negative")
    return probs


def assign_days(bulletin: date, vals: dict) -> dict[str, dict]:
    """Apply the MFD date rule: {day: {field: value}}, leaving out missing values."""
    prev = (bulletin - timedelta(days=1)).isoformat()
    out = {prev: {"max": vals["max"], "rain_mm": vals["rain_mm"]}, bulletin.isoformat(): {"min": vals["min"]}}
    return {d: {k: v for k, v in f.items() if v is not None} for d, f in out.items() if any(v is not None for v in f.values())}


def plan_updates(by_day: dict[str, dict], obs: pd.DataFrame, force: bool = False) -> tuple[dict, list[str]]:
    """Split into values to save and messages about values already recorded."""
    save, skipped = {}, []
    for day, fields in by_day.items():
        old = obs[obs["date"] == day]
        for k, v in fields.items():
            prev = old.iloc[0][k] if len(old) else None
            if prev is not None and pd.notna(prev):
                if float(prev) == v:
                    skipped.append(f"{day} {k}: already {prev}")
                    continue
                if not force:
                    skipped.append(f"{day} {k}: keeping {prev} (bulletin says {v}; use --force to overwrite)")
                    continue
            save.setdefault(day, {})[k] = v
    return save, skipped


def fetch(url: str = URL) -> str:
    r = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print what would be saved, save nothing")
    ap.add_argument("--force", action="store_true", help="overwrite values that are already recorded")
    args = ap.parse_args()

    bulletin, vals = parse_page(fetch())
    today = datetime.now(NPT).date()
    if bulletin != today:
        print(f"Page shows the {bulletin} bulletin, not today's ({today} Nepal time). Nothing saved; try later.")
        return
    show = lambda v: "–" if v is None else v
    print(f"MFD bulletin {bulletin}, {STATION}: max {show(vals['max'])}, min {show(vals['min'])}, rain {show(vals['rain_mm'])} mm")
    probs = check(vals)
    if probs:
        raise SystemExit("Values failed the sanity check, nothing saved: " + "; ".join(probs))

    obs = pd.read_csv(OBS) if OBS.exists() else pd.DataFrame(columns=["date", "max", "min", "rain_mm", "note"])
    save, skipped = plan_updates(assign_days(bulletin, vals), obs, args.force)
    for s in skipped:
        print(f"  skip {s}")
    if not save:
        print("Nothing new to save.")
        return
    note = f"auto: MFD via ashesh.com.np {bulletin}"
    for day, f in save.items():
        if args.dry_run:
            print(f"  would save {day}: " + ", ".join(f"{k} {v}" for k, v in f.items()) + f" (note: {note})")
        else:
            add_observation(day, f.get("max"), f.get("min"), f.get("rain_mm"), note)


if __name__ == "__main__":
    main()
