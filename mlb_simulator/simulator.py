from __future__ import annotations

import random
import statistics
from collections import defaultdict
from dataclasses import dataclass

from .league_averages import LeagueAverages
from .models import (
    BaseState,
    BattingStats,
    PitchingStats,
    PlayerGameStats,
    PlayerKey,
    PlayerProjection,
    ResolvedPitchingProfile,
    ScoreBoard,
    SimulationInput,
    SimulationSummary,
    TeamSide,
)


@dataclass(slots=True)
class GameConfig:
    innings: int = 9
    histogram_bins: int = 200
    max_plate_appearance_event_total: float = 0.94
    single_second_to_home_no_two_outs: float = 0.32
    single_second_to_home_two_outs: float = 0.55
    single_first_to_third_no_two_outs: float = 0.22
    single_first_to_third_two_outs: float = 0.45
    double_first_to_home_no_two_outs: float = 0.40
    double_first_to_home_two_outs: float = 0.62
    double_play_context_multiplier: float = 3.0
    sac_fly_context_multiplier: float = 4.0
    estimated_pitches_per_plate_appearance: float = 3.95
    default_starter_batters_faced: float = 24.0
    min_starter_batters_faced: int = 18
    max_starter_batters_faced: int = 30
    bullpen_inherits_starter_strength_fraction: float = 0.25
    bullpen_walk_rate_multiplier: float = 0.99
    bullpen_hit_by_pitch_rate_multiplier: float = 1.00
    bullpen_strikeout_rate_multiplier: float = 1.05
    bullpen_home_run_rate_multiplier: float = 0.96
    bullpen_non_home_run_hit_rate_multiplier: float = 0.96


@dataclass(slots=True)
class PlateAppearanceProbabilities:
    walk: float
    hit_by_pitch: float
    strikeout: float
    home_run: float
    non_home_run_hit: float

    @property
    def in_play_out(self) -> float:
        remainder = 1.0 - (
            self.walk + self.hit_by_pitch + self.strikeout + self.home_run + self.non_home_run_hit
        )
        return max(remainder, 0.0)




@dataclass(slots=True)
class PitchingPlan:
    starter: PitchingStats
    bullpen: PitchingStats
    starter_batter_limit: int
    batters_faced: int = 0

    @property
    def current_pitcher(self) -> PitchingStats:
        return self.bullpen if self.batters_faced >= self.starter_batter_limit else self.starter

    def note_plate_appearance(self) -> None:
        self.batters_faced += 1

class ProjectionAccumulator:
    def __init__(self) -> None:
        self._values: dict[PlayerKey, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def record(self, key: PlayerKey, stats: PlayerGameStats) -> None:
        bucket = self._values[key]
        bucket["at_bats"].append(float(stats.at_bats))
        bucket["runs"].append(float(stats.runs))
        bucket["hits"].append(float(stats.hits))
        bucket["hit_1_plus_probability"].append(1.0 if stats.hits >= 1 else 0.0)
        bucket["hit_2_plus_probability"].append(1.0 if stats.hits >= 2 else 0.0)
        bucket["doubles"].append(float(stats.doubles))
        bucket["triples"].append(float(stats.triples))
        bucket["home_runs"].append(float(stats.home_runs))
        bucket["rbi"].append(float(stats.rbi))
        bucket["walks"].append(float(stats.walks))
        bucket["batting_average"].append(float(stats.batting_average))
        bucket["on_base_percentage"].append(float(stats.on_base_percentage))
        bucket["slugging_percentage"].append(float(stats.slugging_percentage))

    def mean(self, key: PlayerKey, metric: str) -> float:
        values = self._values.get(key, {}).get(metric, [])
        return statistics.mean(values) if values else 0.0


class MonteCarloSimulator:
    def __init__(
        self,
        game: SimulationInput,
        *,
        league_averages: LeagueAverages | None = None,
        seed: int | None = None,
        config: GameConfig | None = None,
    ) -> None:
        self.game = game
        self.league_averages = league_averages or LeagueAverages()
        self.rng = random.Random(seed)
        self.config = config or GameConfig()

    def run(self, number_of_games: int = 10000, score_bias: float = 0.0) -> SimulationSummary:
        self._validate_inputs()

        histogram = [0] * self.config.histogram_bins
        total_runs_samples: list[float] = []
        away_wins = 0
        home_wins = 0
        projections = ProjectionAccumulator()

        for _ in range(number_of_games):
            scoreboard, box_score = self._simulate_game()
            for key, stats in box_score.items():
                projections.record(key, stats)

            total_runs = scoreboard.away_runs + scoreboard.home_runs + round(score_bias)
            total_runs = int(total_runs)
            total_runs_samples.append(float(total_runs))
            histogram[min(total_runs, self.config.histogram_bins - 1)] += 1

            if scoreboard.home_runs > scoreboard.away_runs:
                home_wins += 1
            else:
                away_wins += 1

        return SimulationSummary(
            game_date=self.game.game_date,
            away_team=self.game.away_team,
            home_team=self.game.home_team,
            away_win_probability=away_wins / number_of_games,
            home_win_probability=home_wins / number_of_games,
            average_total_runs=statistics.mean(total_runs_samples),
            median_total_runs=statistics.median(total_runs_samples),
            stddev_total_runs=statistics.stdev(total_runs_samples) if len(total_runs_samples) > 1 else 0.0,
            total_runs_samples=total_runs_samples,
            total_runs_histogram=histogram,
            away_projections=self._build_team_projections("away", projections),
            home_projections=self._build_team_projections("home", projections),
        )

    def _validate_inputs(self) -> None:
        if len(self.game.away_lineup) < 9 or len(self.game.home_lineup) < 9:
            raise ValueError("Both lineups must contain at least 9 hitters.")
        if not self.game.away_pitching or not self.game.home_pitching:
            raise ValueError("Both teams must include at least one pitcher.")

    def _simulate_game(self) -> tuple[ScoreBoard, dict[PlayerKey, PlayerGameStats]]:
        scoreboard = ScoreBoard()
        scoreboard.add_frame()
        box_score = self._initialize_box_score()

        away_batting_index = 0
        home_batting_index = 0
        away_pitching_plan = self._build_pitching_plan(self.game.away_pitching)
        home_pitching_plan = self._build_pitching_plan(self.game.home_pitching)
        game_completed = False

        while not game_completed:
            if scoreboard.away_is_batting:
                away_batting_index, game_completed = self._play_half_inning(
                    team_side="away",
                    batting_index=away_batting_index,
                    lineup=self.game.away_lineup,
                    pitching_plan=home_pitching_plan,
                    scoreboard=scoreboard,
                    box_score=box_score,
                )
            else:
                home_batting_index, game_completed = self._play_half_inning(
                    team_side="home",
                    batting_index=home_batting_index,
                    lineup=self.game.home_lineup,
                    pitching_plan=away_pitching_plan,
                    scoreboard=scoreboard,
                    box_score=box_score,
                )

            if game_completed:
                break

            completed_home_half = (scoreboard.current_frame % 2) == 1
            if completed_home_half and scoreboard.current_frame >= (self.config.innings * 2) - 1 and scoreboard.away_runs != scoreboard.home_runs:
                break

            scoreboard.add_frame()

        return scoreboard, box_score

    def _initialize_box_score(self) -> dict[PlayerKey, PlayerGameStats]:
        box_score: dict[PlayerKey, PlayerGameStats] = {}
        for side, lineup in (("away", self.game.away_lineup), ("home", self.game.home_lineup)):
            for batter in lineup:
                key: PlayerKey = (side, batter.player_id)
                box_score[key] = PlayerGameStats(
                    team_side=side,
                    player_id=batter.player_id,
                    player_name=batter.player_name,
                )
        return box_score

    def _build_pitching_plan(self, pitchers: list[PitchingStats]) -> PitchingPlan:
        starter = pitchers[0]
        bullpen = pitchers[1] if len(pitchers) > 1 else self._build_bullpen_pitcher(starter)
        starter_batter_limit = self._estimate_starter_batter_limit(starter)
        return PitchingPlan(starter=starter, bullpen=bullpen, starter_batter_limit=starter_batter_limit)

    def _estimate_starter_batter_limit(self, pitcher: PitchingStats) -> int:
        profile_limit = 0.0
        if pitcher.resolved_profile is not None:
            profile_limit = max(pitcher.resolved_profile.average_batters_faced_per_start, 0.0)
        if profile_limit <= 0.0 and pitcher.games_started > 0.0 and pitcher.estimated_batters_faced > 0.0:
            profile_limit = pitcher.estimated_batters_faced / max(pitcher.games_started, 1.0)
        if profile_limit <= 0.0 and pitcher.average_pitches > 0.0:
            profile_limit = pitcher.average_pitches / max(self.config.estimated_pitches_per_plate_appearance, 1.0)
        if profile_limit <= 0.0:
            profile_limit = self.config.default_starter_batters_faced
        return int(round(min(max(profile_limit, self.config.min_starter_batters_faced), self.config.max_starter_batters_faced)))

    def _build_bullpen_pitcher(self, starter: PitchingStats) -> PitchingStats:
        league = self.league_averages
        starter_walk = starter.walk_rate_allowed(league)
        starter_hbp = starter.hit_by_pitch_rate_allowed(league)
        starter_strikeout = starter.strikeout_rate(league)
        starter_home_run = starter.home_run_rate_allowed(league)
        starter_non_hr_hit = starter.non_home_run_hit_rate_allowed(league)

        inherit = self.config.bullpen_inherits_starter_strength_fraction

        def inherited_rate(starter_rate: float, league_rate: float, multiplier: float) -> float:
            base = league_rate * multiplier
            if league_rate <= 0.0:
                return max(base, 0.0)
            deviation = (starter_rate / league_rate) - 1.0
            value = base * (1.0 + (deviation * inherit))
            return max(min(value, 0.999999), 0.0)

        walk_rate = inherited_rate(starter_walk, league.walk_rate_per_pa, self.config.bullpen_walk_rate_multiplier)
        hbp_rate = inherited_rate(starter_hbp, league.hit_by_pitch_rate_per_pa, self.config.bullpen_hit_by_pitch_rate_multiplier)
        strikeout_rate = inherited_rate(starter_strikeout, league.strikeout_rate_per_pa, self.config.bullpen_strikeout_rate_multiplier)
        home_run_rate = inherited_rate(starter_home_run, league.home_run_rate_per_pa, self.config.bullpen_home_run_rate_multiplier)
        non_hr_hit_rate = inherited_rate(starter_non_hr_hit, league.non_home_run_hit_rate_per_pa, self.config.bullpen_non_home_run_hit_rate_multiplier)
        on_base_rate = walk_rate + hbp_rate + home_run_rate + non_hr_hit_rate

        bullpen = PitchingStats.default(league)
        bullpen.player_id = -starter.player_id if starter.player_id else -1
        bullpen.player_name = f"{starter.player_name} Bullpen" if starter.player_name else "Bullpen"
        bullpen.average_pitches = 18.0
        bullpen.games_started = 0.0
        bullpen.resolved_profile = ResolvedPitchingProfile(
            effective_batters_faced=league.prior_batters_faced_for_pitcher_prior,
            on_base_allowed_rate=on_base_rate,
            walk_rate_allowed=walk_rate,
            hit_by_pitch_rate_allowed=hbp_rate,
            strikeout_rate=strikeout_rate,
            home_run_rate_allowed=home_run_rate,
            non_home_run_hit_rate_allowed=non_hr_hit_rate,
            average_pitches=18.0,
            average_batters_faced_per_start=9.0,
        )
        return bullpen

    def _play_half_inning(
        self,
        *,
        team_side: TeamSide,
        batting_index: int,
        lineup: list[BattingStats],
        pitching_plan: PitchingPlan,
        scoreboard: ScoreBoard,
        box_score: dict[PlayerKey, PlayerGameStats],
    ) -> tuple[int, bool]:
        bases = BaseState()
        outs = 0

        while outs < 3:
            if team_side == "home" and scoreboard.current_frame >= (self.config.innings * 2) - 1 and scoreboard.home_runs > scoreboard.away_runs:
                return batting_index, True

            batter = lineup[batting_index]
            current_pitcher = pitching_plan.current_pitcher
            outs = self._at_bat(
                team_side=team_side,
                batter=batter,
                pitcher=current_pitcher,
                outs=outs,
                bases=bases,
                scoreboard=scoreboard,
                box_score=box_score,
            )
            pitching_plan.note_plate_appearance()

            batting_index = (batting_index + 1) % len(lineup)

        return batting_index, False

    def _at_bat(
        self,
        *,
        team_side: TeamSide,
        batter: BattingStats,
        pitcher: PitchingStats,
        outs: int,
        bases: BaseState,
        scoreboard: ScoreBoard,
        box_score: dict[PlayerKey, PlayerGameStats],
    ) -> int:
        batter_key: PlayerKey = (team_side, batter.player_id)
        batter_stats = box_score[batter_key]
        batter_stats.plate_appearances += 1

        probabilities = self._plate_appearance_probabilities(batter, pitcher)
        outcome = self._select_plate_appearance_outcome(probabilities)

        if outcome == "walk":
            batter_stats.walks += 1
            self._force_runner_advance(batter_key, batter_stats, bases, scoreboard, box_score)
            return outs
        if outcome == "hit_by_pitch":
            batter_stats.hit_by_pitch += 1
            self._force_runner_advance(batter_key, batter_stats, bases, scoreboard, box_score)
            return outs
        if outcome == "strikeout":
            batter_stats.at_bats += 1
            batter_stats.strikeouts += 1
            return outs + 1
        if outcome == "home_run":
            self._handle_home_run(batter_key, batter_stats, bases, scoreboard, box_score)
            return outs
        if outcome == "non_home_run_hit":
            self._handle_non_home_run_hit(batter, batter_key, batter_stats, outs, bases, scoreboard, box_score)
            return outs
        if outcome == "in_play_out":
            return self._handle_in_play_out(
                batter=batter,
                batter_stats=batter_stats,
                outs=outs,
                bases=bases,
                scoreboard=scoreboard,
                box_score=box_score,
            )
        raise RuntimeError(f"Unhandled outcome: {outcome}")

    def _plate_appearance_probabilities(self, batter: BattingStats, pitcher: PitchingStats) -> PlateAppearanceProbabilities:
        league = self.league_averages
        walk = self._log5_probability(
            batter.walk_rate(league),
            pitcher.walk_rate_allowed(league),
            league.walk_rate_per_pa,
        )
        hit_by_pitch = self._log5_probability(
            batter.hit_by_pitch_rate(league),
            pitcher.hit_by_pitch_rate_allowed(league),
            league.hit_by_pitch_rate_per_pa,
        )
        strikeout = self._log5_probability(
            batter.strikeout_rate(league),
            pitcher.strikeout_rate(league),
            league.strikeout_rate_per_pa,
        )
        home_run = self._log5_probability(
            batter.home_run_rate(league),
            pitcher.home_run_rate_allowed(league),
            league.home_run_rate_per_pa,
        )
        non_home_run_hit = self._log5_probability(
            batter.non_home_run_hit_rate(league),
            pitcher.non_home_run_hit_rate_allowed(league),
            league.non_home_run_hit_rate_per_pa,
        )

        total = walk + hit_by_pitch + strikeout + home_run + non_home_run_hit
        if total > self.config.max_plate_appearance_event_total:
            scale = self.config.max_plate_appearance_event_total / total
            walk *= scale
            hit_by_pitch *= scale
            strikeout *= scale
            home_run *= scale
            non_home_run_hit *= scale

        return PlateAppearanceProbabilities(
            walk=walk,
            hit_by_pitch=hit_by_pitch,
            strikeout=strikeout,
            home_run=home_run,
            non_home_run_hit=non_home_run_hit,
        )

    @staticmethod
    def _log5_probability(batter_rate: float, pitcher_rate: float, league_rate: float) -> float:
        batter_rate = max(min(batter_rate, 0.999999), 0.000001)
        pitcher_rate = max(min(pitcher_rate, 0.999999), 0.000001)
        league_rate = max(min(league_rate, 0.999999), 0.000001)
        numerator = (batter_rate * pitcher_rate) / league_rate
        denominator = numerator + (((1.0 - batter_rate) * (1.0 - pitcher_rate)) / (1.0 - league_rate))
        if denominator <= 0.0:
            return league_rate
        return max(min(numerator / denominator, 0.999999), 0.0)

    def _select_plate_appearance_outcome(self, probabilities: PlateAppearanceProbabilities) -> str:
        draw = self.rng.random()
        cumulative = probabilities.walk
        if draw <= cumulative:
            return "walk"
        cumulative += probabilities.hit_by_pitch
        if draw <= cumulative:
            return "hit_by_pitch"
        cumulative += probabilities.strikeout
        if draw <= cumulative:
            return "strikeout"
        cumulative += probabilities.home_run
        if draw <= cumulative:
            return "home_run"
        cumulative += probabilities.non_home_run_hit
        if draw <= cumulative:
            return "non_home_run_hit"
        return "in_play_out"

    def _handle_non_home_run_hit(
        self,
        batter: BattingStats,
        batter_key: PlayerKey,
        batter_stats: PlayerGameStats,
        outs: int,
        bases: BaseState,
        scoreboard: ScoreBoard,
        box_score: dict[PlayerKey, PlayerGameStats],
    ) -> None:
        hit_type_probabilities = batter.non_home_run_hit_probabilities(self.league_averages)
        draw = self.rng.random()
        cumulative = hit_type_probabilities["single"]
        if draw <= cumulative:
            self._handle_single(batter_key, batter_stats, outs, bases, scoreboard, box_score)
            return
        cumulative += hit_type_probabilities["double"]
        if draw <= cumulative:
            self._handle_double(batter_key, batter_stats, outs, bases, scoreboard, box_score)
            return
        self._handle_triple(batter_key, batter_stats, bases, scoreboard, box_score)

    def _handle_single(
        self,
        batter_key: PlayerKey,
        batter_stats: PlayerGameStats,
        outs: int,
        bases: BaseState,
        scoreboard: ScoreBoard,
        box_score: dict[PlayerKey, PlayerGameStats],
    ) -> None:
        old_first = bases.first
        old_second = bases.second
        old_third = bases.third
        bases.clear()

        if old_third is not None:
            self._score_runner(old_third, batter_stats, scoreboard, box_score)

        second_scores = False
        if old_second is not None:
            threshold = (
                self.config.single_second_to_home_two_outs
                if outs == 2
                else self.config.single_second_to_home_no_two_outs
            )
            second_scores = self.rng.random() < threshold
            if second_scores:
                self._score_runner(old_second, batter_stats, scoreboard, box_score)
            else:
                bases.third = old_second

        if old_first is not None:
            if second_scores:
                threshold = (
                    self.config.single_first_to_third_two_outs
                    if outs == 2
                    else self.config.single_first_to_third_no_two_outs
                )
                if self.rng.random() < threshold:
                    bases.third = old_first
                else:
                    bases.second = old_first
            else:
                bases.second = old_first

        bases.first = batter_key
        batter_stats.at_bats += 1
        batter_stats.hits += 1
        batter_stats.singles += 1
        scoreboard.increment_hit()

    def _handle_double(
        self,
        batter_key: PlayerKey,
        batter_stats: PlayerGameStats,
        outs: int,
        bases: BaseState,
        scoreboard: ScoreBoard,
        box_score: dict[PlayerKey, PlayerGameStats],
    ) -> None:
        old_first = bases.first
        old_second = bases.second
        old_third = bases.third
        bases.clear()

        if old_third is not None:
            self._score_runner(old_third, batter_stats, scoreboard, box_score)
        if old_second is not None:
            self._score_runner(old_second, batter_stats, scoreboard, box_score)
        if old_first is not None:
            threshold = (
                self.config.double_first_to_home_two_outs
                if outs == 2
                else self.config.double_first_to_home_no_two_outs
            )
            if self.rng.random() < threshold:
                self._score_runner(old_first, batter_stats, scoreboard, box_score)
            else:
                bases.third = old_first

        bases.second = batter_key
        batter_stats.at_bats += 1
        batter_stats.hits += 1
        batter_stats.doubles += 1
        scoreboard.increment_hit()

    def _handle_triple(
        self,
        batter_key: PlayerKey,
        batter_stats: PlayerGameStats,
        bases: BaseState,
        scoreboard: ScoreBoard,
        box_score: dict[PlayerKey, PlayerGameStats],
    ) -> None:
        for runner in (bases.third, bases.second, bases.first):
            if runner is not None:
                self._score_runner(runner, batter_stats, scoreboard, box_score)
        bases.clear()
        bases.third = batter_key
        batter_stats.at_bats += 1
        batter_stats.hits += 1
        batter_stats.triples += 1
        scoreboard.increment_hit()

    def _handle_home_run(
        self,
        batter_key: PlayerKey,
        batter_stats: PlayerGameStats,
        bases: BaseState,
        scoreboard: ScoreBoard,
        box_score: dict[PlayerKey, PlayerGameStats],
    ) -> None:
        for runner in (bases.third, bases.second, bases.first):
            if runner is not None:
                self._score_runner(runner, batter_stats, scoreboard, box_score)
        bases.clear()
        batter_stats.at_bats += 1
        batter_stats.hits += 1
        batter_stats.home_runs += 1
        batter_stats.runs += 1
        batter_stats.rbi += 1
        scoreboard.increment_hit()
        scoreboard.increment_run()

    def _handle_in_play_out(
        self,
        *,
        batter: BattingStats,
        batter_stats: PlayerGameStats,
        outs: int,
        bases: BaseState,
        scoreboard: ScoreBoard,
        box_score: dict[PlayerKey, PlayerGameStats],
    ) -> int:
        batter_stats.at_bats += 1

        if bases.first is not None and outs < 2:
            double_play_rate = batter.in_play_double_play_rate(self.league_averages)
            double_play_rate *= self.config.double_play_context_multiplier
            double_play_rate *= batter.double_play_speed_multiplier(self.league_averages)
            double_play_rate = max(min(double_play_rate, 0.35), 0.0)
            if self.rng.random() < double_play_rate:
                bases.first = None
                return outs + 2

        if bases.third is not None and outs < 2:
            sac_fly_rate = max(min(batter.in_play_sac_fly_rate(self.league_averages) * self.config.sac_fly_context_multiplier, 0.25), 0.0)
            if self.rng.random() < sac_fly_rate:
                batter_stats.at_bats -= 1
                batter_stats.sac_flies += 1
                self._score_runner(bases.third, batter_stats, scoreboard, box_score)
                bases.third = None
                return outs + 1

        return outs + 1

    def _force_runner_advance(
        self,
        batter_key: PlayerKey,
        batter_stats: PlayerGameStats,
        bases: BaseState,
        scoreboard: ScoreBoard,
        box_score: dict[PlayerKey, PlayerGameStats],
    ) -> None:
        if bases.first is not None and bases.second is not None and bases.third is not None:
            self._score_runner(bases.third, batter_stats, scoreboard, box_score)
            bases.third = None
        if bases.first is not None and bases.second is not None:
            bases.third = bases.second
            bases.second = None
        if bases.first is not None:
            bases.second = bases.first
            bases.first = None
        bases.first = batter_key

    def _score_runner(
        self,
        runner_key: PlayerKey,
        batter_stats: PlayerGameStats,
        scoreboard: ScoreBoard,
        box_score: dict[PlayerKey, PlayerGameStats],
    ) -> None:
        box_score[runner_key].runs += 1
        batter_stats.rbi += 1
        scoreboard.increment_run()

    def _build_team_projections(self, team_side: TeamSide, projections: ProjectionAccumulator) -> list[PlayerProjection]:
        team_name = self.game.away_team if team_side == "away" else self.game.home_team
        lineup = self.game.away_lineup if team_side == "away" else self.game.home_lineup
        result: list[PlayerProjection] = []
        for batter in lineup:
            key: PlayerKey = (team_side, batter.player_id)
            result.append(
                PlayerProjection(
                    team_side=team_side,
                    team_name=team_name,
                    player_id=batter.player_id,
                    player_name=batter.player_name,
                    at_bats=projections.mean(key, "at_bats"),
                    runs=projections.mean(key, "runs"),
                    hits=projections.mean(key, "hits"),
                    hit_1_plus_probability=projections.mean(key, "hit_1_plus_probability"),
                    hit_2_plus_probability=projections.mean(key, "hit_2_plus_probability"),
                    doubles=projections.mean(key, "doubles"),
                    triples=projections.mean(key, "triples"),
                    home_runs=projections.mean(key, "home_runs"),
                    rbi=projections.mean(key, "rbi"),
                    walks=projections.mean(key, "walks"),
                    batting_average=projections.mean(key, "batting_average"),
                    on_base_percentage=projections.mean(key, "on_base_percentage"),
                    slugging_percentage=projections.mean(key, "slugging_percentage"),
                )
            )
        return result
