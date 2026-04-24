from __future__ import annotations

import unittest

from mlb_simulator.league_averages import LeagueAverages
from mlb_simulator.models import BattingStats, PitchingStats, SimulationInput
from mlb_simulator.simulator import MonteCarloSimulator


class SmokeTest(unittest.TestCase):
    def test_simulator_runs_with_basic_input(self) -> None:
        lineup = [
            BattingStats(
                player_id=100 + idx,
                player_name=f"Away {idx}",
                on_base=0.33,
                at_bats=500,
                hits=150,
                doubles=30,
                triples=3,
                home_runs=20,
                singles=97,
                walks=50,
                strikeouts=120,
                grounded_into_double_plays=10,
                stolen_bases=10,
                total_plate_appearances=550,
            )
            for idx in range(9)
        ]
        home_lineup = [
            BattingStats(
                player_id=200 + idx,
                player_name=f"Home {idx}",
                on_base=0.32,
                at_bats=500,
                hits=140,
                doubles=25,
                triples=2,
                home_runs=18,
                singles=95,
                walks=48,
                strikeouts=130,
                grounded_into_double_plays=11,
                stolen_bases=8,
                total_plate_appearances=548,
            )
            for idx in range(9)
        ]
        game = SimulationInput(
            game_date="2016-05-01",
            away_team="Away",
            home_team="Home",
            away_lineup=lineup,
            home_lineup=home_lineup,
            away_pitching=[PitchingStats(player_id=1, player_name="A Pitcher", on_base=0.30, innings_pitched=180, hits_allowed=150, walks_allowed=45, strikeouts=180, home_runs_allowed=20, batters_faced=720, at_bats_against=620)],
            home_pitching=[PitchingStats(player_id=2, player_name="H Pitcher", on_base=0.31, innings_pitched=170, hits_allowed=160, walks_allowed=50, strikeouts=170, home_runs_allowed=22, batters_faced=710, at_bats_against=610)],
        )
        summary = MonteCarloSimulator(game, seed=7, league_averages=LeagueAverages()).run(number_of_games=50)
        self.assertEqual(len(summary.away_projections), 9)
        self.assertEqual(len(summary.home_projections), 9)
        self.assertAlmostEqual(summary.away_win_probability + summary.home_win_probability, 1.0, places=6)
        self.assertGreater(summary.average_total_runs, 0.0)
        self.assertTrue(all(0.0 <= projection.hit_1_plus_probability <= 1.0 for projection in summary.away_projections))
        self.assertTrue(all(0.0 <= projection.hit_2_plus_probability <= 1.0 for projection in summary.home_projections))

    def test_stolen_base_speed_reduces_double_play_risk(self) -> None:
        league = LeagueAverages()
        slow = BattingStats(player_id=1, player_name="Slow", stolen_bases=0, total_plate_appearances=600)
        fast = BattingStats(player_id=2, player_name="Fast", stolen_bases=30, total_plate_appearances=600)

        self.assertLess(fast.double_play_speed_multiplier(league), slow.double_play_speed_multiplier(league))


if __name__ == "__main__":
    unittest.main()
