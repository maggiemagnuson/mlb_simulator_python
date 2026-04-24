from __future__ import annotations

import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mlb_simulator.cli import _build_projection_table_lines, _parse_year_from_date, _resolve_league_averages
from mlb_simulator.league_averages import LeagueAverages


class CliLeagueAverageResolutionTest(unittest.TestCase):
    def test_parse_year_from_date(self) -> None:
        self.assertEqual(_parse_year_from_date("2016-05-01"), 2016)

    def test_resolve_league_averages_uses_year_from_date(self) -> None:
        args = argparse.Namespace(
            date="2017-08-12",
            league_year=None,
            league_averages_file=None,
            league_cache_dir=None,
            refresh_league_averages=False,
            no_fetch_league_averages=True,
        )
        with patch("mlb_simulator.cli.resolve_league_averages_for_year") as mock_resolve:
            mock_resolve.return_value = type(
                "Resolution",
                (),
                {"averages": LeagueAverages(), "message": "ok"},
            )()
            averages, message = _resolve_league_averages(args)

        self.assertIsInstance(averages, LeagueAverages)
        self.assertEqual(message, "ok")
        mock_resolve.assert_called_once_with(
            2017,
            cache_dir=None,
            refresh=False,
            allow_fetch=False,
            fallback_to_defaults=True,
        )


class CliProjectionFormattingTest(unittest.TestCase):
    def test_projection_table_uses_fixed_width_columns(self) -> None:
        projections = [
            SimpleNamespace(
                player_name="Geraldo Perdomo",
                team_name="Arizona Diamondbacks",
                at_bats=4.37,
                runs=0.62,
                hits=1.16,
                hit_1_plus_probability=0.724,
                hit_2_plus_probability=0.258,
                doubles=0.14,
                triples=0.00,
                home_runs=0.17,
                rbi=0.53,
                walks=0.34,
                batting_average=0.259,
                on_base_percentage=0.328,
                slugging_percentage=0.391,
            ),
            SimpleNamespace(
                player_name="Ketel Marte",
                team_name="Arizona Diamondbacks",
                at_bats=4.46,
                runs=0.72,
                hits=1.33,
                hit_1_plus_probability=0.801,
                hit_2_plus_probability=0.354,
                doubles=0.24,
                triples=0.00,
                home_runs=0.33,
                rbi=0.79,
                walks=0.23,
                batting_average=0.294,
                on_base_percentage=0.333,
                slugging_percentage=0.574,
            ),
        ]

        lines = _build_projection_table_lines(projections)

        self.assertGreaterEqual(len(lines), 4)
        self.assertIn("1+H%", lines[0])
        self.assertIn("2+H%", lines[0])
        self.assertTrue(all("	" not in line for line in lines))
        self.assertEqual(len(lines[0]), len(lines[1]))
        self.assertEqual(len(lines[0]), len(lines[2]))
        self.assertEqual(len(lines[0]), len(lines[3]))
        self.assertIn("Arizona Diamondbacks", lines[2])
        self.assertIn("72.4%", lines[2])
        self.assertIn("35.4%", lines[3])


if __name__ == "__main__":
    unittest.main()
