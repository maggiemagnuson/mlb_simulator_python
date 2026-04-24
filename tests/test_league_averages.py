from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from mlb_simulator.league_averages import (
    LeagueAverages,
    cache_path_for_date,
    cache_path_for_year,
    load_cached_league_averages,
    resolve_league_averages_for_year,
    save_cached_league_averages,
)


class LeagueAveragesCacheTest(unittest.TestCase):
    def test_save_and_load_cached_league_averages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            averages = LeagueAverages(hitter_on_base=0.333)
            saved_path = save_cached_league_averages(2019, averages, cache_dir=cache_dir)

            self.assertEqual(saved_path, cache_path_for_year(2019, cache_dir))
            loaded = load_cached_league_averages(2019, cache_dir=cache_dir)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertAlmostEqual(loaded.hitter_on_base, 0.333)

    def test_save_and_load_daily_current_season_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            averages = LeagueAverages(hitter_on_base=0.321)
            saved_path = save_cached_league_averages(2026, averages, cache_dir=cache_dir, as_of_date=date(2026, 4, 23))

            self.assertEqual(saved_path, cache_path_for_date(2026, date(2026, 4, 23), cache_dir))
            loaded = load_cached_league_averages(2026, cache_dir=cache_dir, as_of_date=date(2026, 4, 23))
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertAlmostEqual(loaded.hitter_on_base, 0.321)

    def test_resolve_fetches_and_caches_when_missing_for_past_year(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            fetched = LeagueAverages(hitter_on_base=0.301)
            with patch("mlb_simulator.league_averages.LeagueAverages.from_baseball_reference", return_value=fetched) as mock_fetch:
                resolution = resolve_league_averages_for_year(2021, cache_dir=cache_dir, reference_date=date(2026, 4, 23))

            self.assertEqual(resolution.source, "fetched")
            self.assertAlmostEqual(resolution.averages.hitter_on_base, 0.301)
            self.assertTrue(cache_path_for_year(2021, cache_dir).exists())
            mock_fetch.assert_called_once_with(2021, session=None)

    def test_current_year_uses_daily_cache_and_blends_with_prior_years(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            current = LeagueAverages(hitter_on_base=0.340)
            prior_one = LeagueAverages(hitter_on_base=0.320)
            prior_two = LeagueAverages(hitter_on_base=0.300)

            save_cached_league_averages(2026, current, cache_dir=cache_dir, as_of_date=date(2026, 4, 23))
            save_cached_league_averages(2025, prior_one, cache_dir=cache_dir)
            save_cached_league_averages(2024, prior_two, cache_dir=cache_dir)

            resolution = resolve_league_averages_for_year(2026, cache_dir=cache_dir, reference_date=date(2026, 4, 23))

            self.assertEqual(resolution.source, "blended")
            self.assertEqual(resolution.cache_path, cache_path_for_date(2026, date(2026, 4, 23), cache_dir))
            self.assertAlmostEqual(resolution.averages.hitter_on_base, 0.327, places=3)
            self.assertIn("blended with weights 50%/35%/15%", resolution.message)

    def test_resolve_uses_defaults_when_fetch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            with patch(
                "mlb_simulator.league_averages.LeagueAverages.from_baseball_reference",
                side_effect=RuntimeError("network down"),
            ):
                resolution = resolve_league_averages_for_year(2022, cache_dir=cache_dir, reference_date=date(2026, 4, 23))

            self.assertEqual(resolution.source, "default")
            self.assertIsNone(resolution.cache_path)
            self.assertIn("using bundled defaults", resolution.message.lower())


if __name__ == "__main__":
    unittest.main()
