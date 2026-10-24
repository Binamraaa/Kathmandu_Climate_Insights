"""Run from the repo root: python -m unittest tests.test_fetch_observations"""
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from src.fetch_observations import check, gaps, parse_synops, plan_updates, precipitation, section3, temperature

# Real OGIMET replies for 44454, 5–24 Oct 2026 (includes NIL reports and the Dashain gap).
FIXTURE = Path(__file__).parent / "fixtures" / "ogimet_44454.csv"


def line(day: int, hour: int, report: str) -> str:
    return f"44454,2026,10,{day:02d},{hour:02d},00,AAXX {day:02d}{hour:02d}4 44454 {report}="


class DecodeTest(unittest.TestCase):
    def test_groups(self):
        self.assertEqual(temperature("10276"), 27.6)
        self.assertEqual(temperature("21015"), -1.5)
        self.assertIsNone(temperature("2////"))
        self.assertEqual(precipitation("60344"), (34.0, 4))
        self.assertEqual(precipitation("69904"), (0.1, 4))     # trace
        self.assertEqual(precipitation("69934"), (0.3, 4))
        self.assertEqual(precipitation("60004"), (0.0, 4))
        self.assertEqual(section3("22456 20000 10196 333 20140 59009 60004 555 10123="), ["20140", "59009", "60004"])
        self.assertEqual(section3("NIL="), [])

    def test_fixture_matches_mfd(self):
        days = parse_synops(FIXTURE.read_text(encoding="utf-8"))
        # Maxima and rain checked against the MFD bulletins
        self.assertEqual(days["2026-10-13"]["max"], 27.6)
        self.assertEqual(days["2026-10-14"], {"max": 27.5, "min": 14.5, "rain_mm": 0.0})
        self.assertEqual(days["2026-10-18"]["max"], 25.0)
        self.assertEqual(days["2026-10-07"]["rain_mm"], 34.0)
        self.assertEqual(days["2026-10-08"]["rain_mm"], 0.3)
        # 21 Oct: morning report only, the evening one is NIL
        self.assertEqual(days["2026-10-21"], {"min": 14.0})
        self.assertNotIn("2026-10-22", days)
        # 24 Oct 03 UTC gives that morning's low and the rain for the 23rd
        self.assertEqual(days["2026-10-24"], {"min": 14.7})
        self.assertEqual(days["2026-10-23"], {"rain_mm": 0.0})
        self.assertTrue(all(check(v) == [] for v in days.values()))

    def test_date_rules(self):
        text = "\n".join([line(15, 3, "22456 20000 333 20135 60124"), line(15, 12, "22456 20000 333 10266 60008 70000")])
        self.assertEqual(parse_synops(text), {"2026-10-14": {"rain_mm": 12.0}, "2026-10-15": {"min": 13.5, "max": 26.6}})

    def test_only_24h_rain_counts(self):
        # 9-hour totals (t=8) at 12 UTC must not be taken as daily rain
        self.assertEqual(parse_synops(line(15, 12, "22456 20000 333 10266 60138")), {"2026-10-15": {"max": 26.6}})

    def test_reply_without_reports(self):
        with self.assertRaises(ValueError):
            parse_synops("Status: 501 Sorry, the query is too large")


class SanityAndOverwriteTest(unittest.TestCase):
    def test_sanity(self):
        self.assertTrue(check({"max": 13.0, "min": 13.5}))
        self.assertTrue(check({"max": 45.0}))
        self.assertTrue(check({"min": -6.0}))
        self.assertTrue(check({"rain_mm": -1.0}))
        self.assertEqual(check({"rain_mm": 0.0}), [])

    def test_never_overwrite_without_force(self):
        obs = pd.DataFrame([{"date": "2026-10-14", "max": None, "min": 14.9, "rain_mm": 0.0, "note": "MFD"}])
        by_day = {"2026-10-14": {"max": 27.5, "min": 14.5, "rain_mm": 0.0}, "2026-10-15": {"min": 13.5}}
        save, skipped = plan_updates(by_day, obs)
        self.assertEqual(save, {"2026-10-14": {"max": 27.5}, "2026-10-15": {"min": 13.5}})
        self.assertEqual(len(skipped), 1)   # the differing min; the equal rain is silent
        save, _ = plan_updates(by_day, obs, force=True)
        self.assertEqual(save["2026-10-14"], {"max": 27.5, "min": 14.5})

    def test_gaps(self):
        obs = pd.DataFrame([{"date": "2026-10-22", "max": 26.0, "min": 14.0, "rain_mm": 0.0, "note": ""}])
        by_day = {"2026-10-23": {"min": 14.2, "max": 25.9}}
        # yesterday's rain isn't due until today's 08:45 report
        self.assertEqual(gaps(obs, by_day, date(2026, 10, 24), days=3), ["2026-10-21: min, max, rain_mm"])


if __name__ == "__main__":
    unittest.main()
