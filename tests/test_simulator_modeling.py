from __future__ import annotations

import unittest

from mlb_simulator.league_averages import LeagueAverages
from mlb_simulator.models import (
    BattingStats,
    PitchingStats,
    ResolvedBattingProfile,
    ResolvedPitchingProfile,
    SimulationInput,
)
from mlb_simulator.simulator import MonteCarloSimulator


class SimulatorModelingTest(unittest.TestCase):
    def _simulator(self) -> MonteCarloSimulator:
        lineup = [BattingStats(player_id=idx + 1, player_name=f"B{idx + 1}") for idx in range(9)]
        pitchers = [PitchingStats.default(LeagueAverages())]
        game = SimulationInput(
            game_date="2026-04-29",
            away_team="Away",
            home_team="Home",
            away_lineup=lineup,
            home_lineup=list(lineup),
            away_pitching=pitchers,
            home_pitching=pitchers,
        )
        return MonteCarloSimulator(game, league_averages=LeagueAverages(), seed=7)

    def test_plate_appearance_probabilities_form_valid_distribution(self) -> None:
        league = LeagueAverages()
        batter = BattingStats(
            player_id=1,
            player_name="Batter",
            resolved_profile=ResolvedBattingProfile(
                walk_rate=0.12,
                hit_by_pitch_rate=0.01,
                strikeout_rate=0.25,
                home_run_rate=0.07,
                non_home_run_hit_rate=0.21,
            ),
        )
        pitcher = PitchingStats(
            player_id=2,
            player_name="Pitcher",
            resolved_profile=ResolvedPitchingProfile(
                walk_rate_allowed=0.10,
                hit_by_pitch_rate_allowed=0.01,
                strikeout_rate=0.24,
                home_run_rate_allowed=0.06,
                non_home_run_hit_rate_allowed=0.20,
                single_share_allowed=league.single_share_of_non_home_run_hits,
                double_share_allowed=league.double_share_of_non_home_run_hits,
                triple_share_allowed=league.triple_share_of_non_home_run_hits,
            ),
        )

        probabilities = self._simulator()._plate_appearance_probabilities(batter, pitcher)
        total = (
            probabilities.walk
            + probabilities.hit_by_pitch
            + probabilities.strikeout
            + probabilities.home_run
            + probabilities.non_home_run_hit
            + probabilities.in_play_out
        )

        self.assertLessEqual(total, 1.0 + 1e-9)
        self.assertGreaterEqual(probabilities.in_play_out, 0.0)
        self.assertGreater(probabilities.walk, 0.0)
        self.assertGreater(probabilities.non_home_run_hit, 0.0)

    def test_pitcher_hit_type_mix_changes_non_home_run_hit_distribution(self) -> None:
        batter = BattingStats(
            player_id=1,
            player_name="Batter",
            resolved_profile=ResolvedBattingProfile(
                non_home_run_hit_rate=0.20,
                single_share=0.55,
                double_share=0.40,
                triple_share=0.05,
            ),
        )
        single_heavy_pitcher = PitchingStats(
            player_id=2,
            player_name="Single Heavy",
            resolved_profile=ResolvedPitchingProfile(
                non_home_run_hit_rate_allowed=0.20,
                single_share_allowed=0.80,
                double_share_allowed=0.17,
                triple_share_allowed=0.03,
            ),
        )
        double_heavy_pitcher = PitchingStats(
            player_id=3,
            player_name="Double Heavy",
            resolved_profile=ResolvedPitchingProfile(
                non_home_run_hit_rate_allowed=0.20,
                single_share_allowed=0.40,
                double_share_allowed=0.55,
                triple_share_allowed=0.05,
            ),
        )

        simulator = self._simulator()
        single_heavy_mix = simulator._combined_non_home_run_hit_probabilities(batter, single_heavy_pitcher)
        double_heavy_mix = simulator._combined_non_home_run_hit_probabilities(batter, double_heavy_pitcher)

        self.assertGreater(single_heavy_mix["single"], double_heavy_mix["single"])
        self.assertGreater(double_heavy_mix["double"], single_heavy_mix["double"])

    def test_platoon_adjustment_changes_matchup_rates_when_handedness_is_available(self) -> None:
        league = LeagueAverages()
        batter = BattingStats(
            player_id=1,
            player_name="Lefty Batter",
            bats="L",
            resolved_profile=ResolvedBattingProfile(
                walk_rate=0.10,
                hit_by_pitch_rate=0.01,
                strikeout_rate=0.22,
                home_run_rate=0.05,
                non_home_run_hit_rate=0.19,
            ),
        )
        righty = PitchingStats(
            player_id=2,
            player_name="Righty",
            throws="R",
            resolved_profile=ResolvedPitchingProfile(
                walk_rate_allowed=0.09,
                hit_by_pitch_rate_allowed=0.01,
                strikeout_rate=0.23,
                home_run_rate_allowed=0.04,
                non_home_run_hit_rate_allowed=0.18,
                single_share_allowed=league.single_share_of_non_home_run_hits,
                double_share_allowed=league.double_share_of_non_home_run_hits,
                triple_share_allowed=league.triple_share_of_non_home_run_hits,
            ),
        )
        lefty = PitchingStats(
            player_id=3,
            player_name="Lefty",
            throws="L",
            resolved_profile=ResolvedPitchingProfile(
                walk_rate_allowed=0.09,
                hit_by_pitch_rate_allowed=0.01,
                strikeout_rate=0.23,
                home_run_rate_allowed=0.04,
                non_home_run_hit_rate_allowed=0.18,
                single_share_allowed=league.single_share_of_non_home_run_hits,
                double_share_allowed=league.double_share_of_non_home_run_hits,
                triple_share_allowed=league.triple_share_of_non_home_run_hits,
            ),
        )

        simulator = self._simulator()
        opposite_side = simulator._plate_appearance_probabilities(batter, righty)
        same_side = simulator._plate_appearance_probabilities(batter, lefty)

        self.assertGreater(opposite_side.home_run, same_side.home_run)
        self.assertGreater(opposite_side.non_home_run_hit, same_side.non_home_run_hit)
        self.assertLess(opposite_side.strikeout, same_side.strikeout)


if __name__ == "__main__":
    unittest.main()
