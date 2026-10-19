"""Run from the repo root: python -m unittest tests.test_fetch_observations"""
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from src.fetch_observations import assign_days, check, is_recent, parse_page, parse_value, plan_updates

FIXTURE = Path(__file__).parent / "fixtures" / "ashesh_weather.html"


def page(rain: str, tmin: str, tmax: str, updated: str = "2026-10-15") -> str:
    """Minimal page with the same markup as ashesh.com.np (cells in DOM order rain, min, max)."""
    return f"""<div class="main"><div class="exchange-rate">Max°C Min°C Rain-㎜</div>
    <div class="country"><div class="flag">☀</div><div class="name">Pokhara</div>
      <div class="rate_selling">9.9</div><div class="rate_buying">9.9</div><div class="unit">9.9</div><hr></div>
    <div class="country"><div class="flag">☀</div><div class="name"> Kathmandu </div>
      <div class="rate_selling"> {rain} </div><div class="rate_buying"> {tmin} </div><div class="unit"> {tmax} </div><hr></div>
    <div class="privacy">Weather data updated on. {updated}</div></div>"""


class ParseTest(unittest.TestCase):
    def test_fixture_kathmandu_row(self):
        bulletin, vals = parse_page(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(bulletin, date(2026, 10, 15))
        self.assertEqual(vals, {"max": 27.5, "min": 13.5, "rain_mm": 0.0})
        self.assertEqual(check(vals), [])

    def test_columns_follow_classes_not_order(self):
        _, vals = parse_page(page("2.5", "15.0", "24.2"))
        self.assertEqual(vals, {"max": 24.2, "min": 15.0, "rain_mm": 2.5})

    def test_trace_and_missing(self):
        self.assertEqual(parse_value("Trace", rain=True), 0.1)
        self.assertIsNone(parse_value("-"))
        self.assertIsNone(parse_value("--"))
        _, vals = parse_page(page("Trace", "--", "26.0"))
        self.assertEqual(vals, {"max": 26.0, "min": None, "rain_mm": 0.1})

    def test_missing_station_or_date(self):
        with self.assertRaises(ValueError):
            parse_page(page("0.0", "13.5", "27.5").replace("Kathmandu", "Jiri"))
        with self.assertRaises(ValueError):
            parse_page(page("0.0", "13.5", "27.5").replace("updated on", "as of"))


class DateRuleTest(unittest.TestCase):
    def test_min_today_max_rain_yesterday(self):
        days = assign_days(date(2026, 10, 15), {"max": 27.5, "min": 13.5, "rain_mm": 0.0})
        self.assertEqual(days, {"2026-10-14": {"max": 27.5, "rain_mm": 0.0}, "2026-10-15": {"min": 13.5}})

    def test_bulletin_age(self):
        today = date(2026, 10, 19)
        self.assertTrue(is_recent(date(2026, 10, 19), today))
        self.assertTrue(is_recent(date(2026, 10, 18), today))
        self.assertFalse(is_recent(date(2026, 10, 17), today))
        self.assertFalse(is_recent(date(2026, 10, 20), today))

    def test_missing_values_are_skipped(self):
        days = assign_days(date(2026, 10, 15), {"max": 27.5, "min": None, "rain_mm": None})
        self.assertEqual(days, {"2026-10-14": {"max": 27.5}})


class SanityAndOverwriteTest(unittest.TestCase):
    def test_sanity(self):
        self.assertTrue(check({"max": 13.0, "min": 13.5, "rain_mm": 0.0}))
        self.assertTrue(check({"max": 45.0, "min": 13.5, "rain_mm": 0.0}))
        self.assertTrue(check({"max": 27.5, "min": -6.0, "rain_mm": 0.0}))
        self.assertTrue(check({"max": 27.5, "min": 13.5, "rain_mm": -1.0}))

    def test_never_overwrite_without_force(self):
        obs = pd.DataFrame([{"date": "2026-10-14", "max": None, "min": 14.9, "rain_mm": 1.0, "note": "MFD"}])
        by_day = {"2026-10-14": {"max": 27.5, "rain_mm": 0.0}, "2026-10-15": {"min": 13.5}}
        save, skipped = plan_updates(by_day, obs)
        self.assertEqual(save, {"2026-10-14": {"max": 27.5}, "2026-10-15": {"min": 13.5}})
        self.assertEqual(len(skipped), 1)
        save, _ = plan_updates(by_day, obs, force=True)
        self.assertEqual(save["2026-10-14"], {"max": 27.5, "rain_mm": 0.0})


if __name__ == "__main__":
    unittest.main()
