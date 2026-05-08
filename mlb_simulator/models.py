from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Literal

from .league_averages import LeagueAverages

TeamSide = Literal["away", "home"]
PlayerKey = tuple[TeamSide, int]


def _lookup(mapping: Mapping[str, Any], *keys: str, default: Any = 0) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _clean_name(value: Any) -> str:
    return str(value or "").strip().strip('"')


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return 0.0
    return float(text)


def _to_int(value: Any) -> int:
    return int(round(_to_float(value)))


def _clamp_rate(value: float) -> float:
    return max(min(float(value), 0.999999), 0.0)


def _normalize_handedness(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"L", "R", "S"}:
        return text
    return ""


def _normalize_shares(weighted: Mapping[str, float], fallback: Mapping[str, float]) -> dict[str, float]:
    normalized = {name: max(float(value), 0.0) for name, value in weighted.items()}
    total = sum(normalized.values())
    if total <= 0.0:
        return {name: max(float(value), 0.0) for name, value in fallback.items()}
    return {name: value / total for name, value in normalized.items()}


def _smoothed_rate_from_counts(observed_events: float, sample: float, *, prior_rate: float, prior_sample: float) -> float:
    sample = max(sample, 0.0)
    prior_sample = max(prior_sample, 0.0)
    total_sample = sample + prior_sample
    if total_sample <= 0.0:
        return max(prior_rate, 0.0)
    numerator = max(observed_events, 0.0) + (max(prior_rate, 0.0) * prior_sample)
    return numerator / total_sample


@dataclass(slots=True)
class ResolvedBattingProfile:
    effective_plate_appearances: float = 0.0
    effective_non_home_run_hits: float = 0.0
    effective_balls_in_play_outs: float = 0.0
    on_base_rate: float = 0.0
    walk_rate: float = 0.0
    hit_by_pitch_rate: float = 0.0
    strikeout_rate: float = 0.0
    home_run_rate: float = 0.0
    non_home_run_hit_rate: float = 0.0
    single_share: float = 1.0
    double_share: float = 0.0
    triple_share: float = 0.0
    double_play_rate_on_in_play_out: float = 0.0
    sac_fly_rate_on_in_play_out: float = 0.0
    stolen_bases_per_600_pa: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "effective_plate_appearances": self.effective_plate_appearances,
            "effective_non_home_run_hits": self.effective_non_home_run_hits,
            "effective_balls_in_play_outs": self.effective_balls_in_play_outs,
            "on_base_rate": self.on_base_rate,
            "walk_rate": self.walk_rate,
            "hit_by_pitch_rate": self.hit_by_pitch_rate,
            "strikeout_rate": self.strikeout_rate,
            "home_run_rate": self.home_run_rate,
            "non_home_run_hit_rate": self.non_home_run_hit_rate,
            "single_share": self.single_share,
            "double_share": self.double_share,
            "triple_share": self.triple_share,
            "double_play_rate_on_in_play_out": self.double_play_rate_on_in_play_out,
            "sac_fly_rate_on_in_play_out": self.sac_fly_rate_on_in_play_out,
            "stolen_bases_per_600_pa": self.stolen_bases_per_600_pa,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResolvedBattingProfile":
        return cls(
            effective_plate_appearances=_to_float(payload.get("effective_plate_appearances")),
            effective_non_home_run_hits=_to_float(payload.get("effective_non_home_run_hits")),
            effective_balls_in_play_outs=_to_float(payload.get("effective_balls_in_play_outs")),
            on_base_rate=_to_float(payload.get("on_base_rate")),
            walk_rate=_to_float(payload.get("walk_rate")),
            hit_by_pitch_rate=_to_float(payload.get("hit_by_pitch_rate")),
            strikeout_rate=_to_float(payload.get("strikeout_rate")),
            home_run_rate=_to_float(payload.get("home_run_rate")),
            non_home_run_hit_rate=_to_float(payload.get("non_home_run_hit_rate")),
            single_share=_to_float(payload.get("single_share")),
            double_share=_to_float(payload.get("double_share")),
            triple_share=_to_float(payload.get("triple_share")),
            double_play_rate_on_in_play_out=_to_float(payload.get("double_play_rate_on_in_play_out")),
            sac_fly_rate_on_in_play_out=_to_float(payload.get("sac_fly_rate_on_in_play_out")),
            stolen_bases_per_600_pa=_to_float(payload.get("stolen_bases_per_600_pa")),
        )

    def normalized_non_home_run_hit_shares(self, league: LeagueAverages) -> dict[str, float]:
        return _normalize_shares(
            {
                "single": self.single_share,
                "double": self.double_share,
                "triple": self.triple_share,
            },
            {
                "single": league.single_share_of_non_home_run_hits,
                "double": league.double_share_of_non_home_run_hits,
                "triple": league.triple_share_of_non_home_run_hits,
            },
        )


@dataclass(slots=True)
class ResolvedPitchingProfile:
    effective_batters_faced: float = 0.0
    on_base_allowed_rate: float = 0.0
    walk_rate_allowed: float = 0.0
    hit_by_pitch_rate_allowed: float = 0.0
    strikeout_rate: float = 0.0
    home_run_rate_allowed: float = 0.0
    non_home_run_hit_rate_allowed: float = 0.0
    single_share_allowed: float = 0.0
    double_share_allowed: float = 0.0
    triple_share_allowed: float = 0.0
    average_pitches: float = 0.0
    average_batters_faced_per_start: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "effective_batters_faced": self.effective_batters_faced,
            "on_base_allowed_rate": self.on_base_allowed_rate,
            "walk_rate_allowed": self.walk_rate_allowed,
            "hit_by_pitch_rate_allowed": self.hit_by_pitch_rate_allowed,
            "strikeout_rate": self.strikeout_rate,
            "home_run_rate_allowed": self.home_run_rate_allowed,
            "non_home_run_hit_rate_allowed": self.non_home_run_hit_rate_allowed,
            "single_share_allowed": self.single_share_allowed,
            "double_share_allowed": self.double_share_allowed,
            "triple_share_allowed": self.triple_share_allowed,
            "average_pitches": self.average_pitches,
            "average_batters_faced_per_start": self.average_batters_faced_per_start,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResolvedPitchingProfile":
        return cls(
            effective_batters_faced=_to_float(payload.get("effective_batters_faced")),
            on_base_allowed_rate=_to_float(payload.get("on_base_allowed_rate")),
            walk_rate_allowed=_to_float(payload.get("walk_rate_allowed")),
            hit_by_pitch_rate_allowed=_to_float(payload.get("hit_by_pitch_rate_allowed")),
            strikeout_rate=_to_float(payload.get("strikeout_rate")),
            home_run_rate_allowed=_to_float(payload.get("home_run_rate_allowed")),
            non_home_run_hit_rate_allowed=_to_float(payload.get("non_home_run_hit_rate_allowed")),
            single_share_allowed=_to_float(payload.get("single_share_allowed")),
            double_share_allowed=_to_float(payload.get("double_share_allowed")),
            triple_share_allowed=_to_float(payload.get("triple_share_allowed")),
            average_pitches=_to_float(payload.get("average_pitches")),
            average_batters_faced_per_start=_to_float(payload.get("average_batters_faced_per_start")),
        )

    def normalized_non_home_run_hit_shares(self, league: LeagueAverages) -> dict[str, float]:
        return _normalize_shares(
            {
                "single": self.single_share_allowed,
                "double": self.double_share_allowed,
                "triple": self.triple_share_allowed,
            },
            {
                "single": league.single_share_of_non_home_run_hits,
                "double": league.double_share_of_non_home_run_hits,
                "triple": league.triple_share_of_non_home_run_hits,
            },
        )


@dataclass(slots=True)
class BattingStats:
    player_id: int
    player_name: str = ""
    bats: str = ""
    on_base: float = 0.0
    hit_by_pitch: float = 0.0
    sac_flies: float = 0.0
    at_bats: float = 0.0
    hits: float = 0.0
    doubles: float = 0.0
    triples: float = 0.0
    home_runs: float = 0.0
    singles: float = 0.0
    walks: float = 0.0
    strikeouts: float = 0.0
    grounded_into_double_plays: float = 0.0
    stolen_bases: float = 0.0
    total_plate_appearances: float = 0.0
    resolved_profile: ResolvedBattingProfile | None = None

    @classmethod
    def from_api(cls, payload: Mapping[str, Any]) -> "BattingStats":
        hits = _to_float(_lookup(payload, "hits", "H"))
        doubles = _to_float(_lookup(payload, "doubles", "2B"))
        triples = _to_float(_lookup(payload, "triples", "3B"))
        home_runs = _to_float(_lookup(payload, "hr", "homeRuns", "HR"))
        singles = _to_float(_lookup(payload, "singles"))
        if singles == 0.0 and hits:
            singles = max(hits - doubles - triples - home_runs, 0.0)
        return cls(
            player_id=_to_int(_lookup(payload, "playerID", "playerId", "id")),
            player_name=_clean_name(_lookup(payload, "playerName", "name")),
            throws=_normalize_handedness(_lookup(payload, "throws", "pitchHand", default="")),
            on_base=_to_float(_lookup(payload, "onBase", "obp", "OBP")),
            hit_by_pitch=_to_float(_lookup(payload, "hitByPitch", "hbp", "HBP")),
            sac_flies=_to_float(_lookup(payload, "sacFlies", "sf", "SF")),
            at_bats=_to_float(_lookup(payload, "atBats", "ab", "AB")),
            hits=hits,
            doubles=doubles,
            triples=triples,
            home_runs=home_runs,
            singles=singles,
            walks=_to_float(_lookup(payload, "bb", "walks", "BB", "baseOnBalls")),
            strikeouts=_to_float(_lookup(payload, "strikeOuts", "so", "SO")),
            grounded_into_double_plays=_to_float(_lookup(payload, "groundIntoDoublePlay", "gidp", "GDP")),
            stolen_bases=_to_float(_lookup(payload, "stolenBases", "sb", "SB")),
            total_plate_appearances=_to_float(_lookup(payload, "tpa", "plateAppearances", "PA")),
        )

    @property
    def non_home_run_hits(self) -> float:
        return max(self.hits - self.home_runs, 0.0)

    @property
    def estimated_balls_in_play_outs(self) -> float:
        return max(self.at_bats - self.hits - self.strikeouts, 0.0)

    def weighted_on_base(self, league: LeagueAverages) -> float:
        if self.resolved_profile is not None:
            return _clamp_rate(self.resolved_profile.on_base_rate)
        observed_events = self.on_base * max(self.total_plate_appearances, 0.0)
        return _smoothed_rate_from_counts(
            observed_events,
            self.total_plate_appearances,
            prior_rate=league.hitter_on_base,
            prior_sample=league.min_plate_appearances,
        )

    def walk_rate(self, league: LeagueAverages) -> float:
        if self.resolved_profile is not None:
            return _clamp_rate(self.resolved_profile.walk_rate)
        return _smoothed_rate_from_counts(
            self.walks,
            self.total_plate_appearances,
            prior_rate=league.walk_rate_per_pa,
            prior_sample=league.min_plate_appearances,
        )

    def hit_by_pitch_rate(self, league: LeagueAverages) -> float:
        if self.resolved_profile is not None:
            return _clamp_rate(self.resolved_profile.hit_by_pitch_rate)
        return _smoothed_rate_from_counts(
            self.hit_by_pitch,
            self.total_plate_appearances,
            prior_rate=league.hit_by_pitch_rate_per_pa,
            prior_sample=league.min_plate_appearances,
        )

    def strikeout_rate(self, league: LeagueAverages) -> float:
        if self.resolved_profile is not None:
            return _clamp_rate(self.resolved_profile.strikeout_rate)
        return _smoothed_rate_from_counts(
            self.strikeouts,
            self.total_plate_appearances,
            prior_rate=league.strikeout_rate_per_pa,
            prior_sample=league.min_plate_appearances,
        )

    def home_run_rate(self, league: LeagueAverages) -> float:
        if self.resolved_profile is not None:
            return _clamp_rate(self.resolved_profile.home_run_rate)
        return _smoothed_rate_from_counts(
            self.home_runs,
            self.total_plate_appearances,
            prior_rate=league.home_run_rate_per_pa,
            prior_sample=league.min_plate_appearances,
        )

    def non_home_run_hit_rate(self, league: LeagueAverages) -> float:
        if self.resolved_profile is not None:
            return _clamp_rate(self.resolved_profile.non_home_run_hit_rate)
        return _smoothed_rate_from_counts(
            self.non_home_run_hits,
            self.total_plate_appearances,
            prior_rate=league.non_home_run_hit_rate_per_pa,
            prior_sample=league.min_plate_appearances,
        )

    def in_play_double_play_rate(self, league: LeagueAverages) -> float:
        if self.resolved_profile is not None:
            return _clamp_rate(self.resolved_profile.double_play_rate_on_in_play_out)
        return _smoothed_rate_from_counts(
            self.grounded_into_double_plays,
            self.estimated_balls_in_play_outs,
            prior_rate=league.double_play_rate_on_in_play_out,
            prior_sample=league.prior_balls_in_play_outs_for_hitter_prior,
        )

    def in_play_sac_fly_rate(self, league: LeagueAverages) -> float:
        if self.resolved_profile is not None:
            return _clamp_rate(self.resolved_profile.sac_fly_rate_on_in_play_out)
        return _smoothed_rate_from_counts(
            self.sac_flies,
            self.estimated_balls_in_play_outs,
            prior_rate=league.sac_fly_rate_on_in_play_out,
            prior_sample=league.prior_balls_in_play_outs_for_hitter_prior,
        )

    def stolen_bases_per_600_pa(self, league: LeagueAverages) -> float:
        if self.resolved_profile is not None:
            return max(self.resolved_profile.stolen_bases_per_600_pa, 0.0)
        smoothed_rate = _smoothed_rate_from_counts(
            self.stolen_bases,
            self.total_plate_appearances,
            prior_rate=5.0 / 600.0,
            prior_sample=league.min_plate_appearances,
        )
        return max(smoothed_rate * 600.0, 0.0)

    def effective_batting_side(self, opposing_throws: str = "") -> str:
        bats = _normalize_handedness(self.bats)
        throws = _normalize_handedness(opposing_throws)
        if bats == "S" and throws == "L":
            return "R"
        if bats == "S" and throws == "R":
            return "L"
        return bats

    def double_play_speed_multiplier(self, league: LeagueAverages) -> float:
        baseline = 5.0
        elite = 30.0
        max_discount = 0.40
        max_penalty = 0.10
        speed = self.stolen_bases_per_600_pa(league)
        if speed >= baseline:
            share = min((speed - baseline) / max(elite - baseline, 1.0), 1.0)
            return 1.0 - (share * max_discount)
        share = min((baseline - speed) / baseline, 1.0)
        return 1.0 + (share * max_penalty)

    def on_base_event_probabilities(self, league: LeagueAverages) -> dict[str, float]:
        if self.resolved_profile is not None:
            non_hr_shares = self.resolved_profile.normalized_non_home_run_hit_shares(league)
            weighted = {
                "single": max(self.resolved_profile.non_home_run_hit_rate, 0.0) * non_hr_shares["single"],
                "double": max(self.resolved_profile.non_home_run_hit_rate, 0.0) * non_hr_shares["double"],
                "triple": max(self.resolved_profile.non_home_run_hit_rate, 0.0) * non_hr_shares["triple"],
                "home_run": max(self.resolved_profile.home_run_rate, 0.0),
                "walk": max(self.resolved_profile.walk_rate, 0.0),
                "hit_by_pitch": max(self.resolved_profile.hit_by_pitch_rate, 0.0),
            }
            total = sum(weighted.values())
            if total <= 0.0:
                return {
                    "single": 1.0,
                    "double": 0.0,
                    "triple": 0.0,
                    "home_run": 0.0,
                    "walk": 0.0,
                    "hit_by_pitch": 0.0,
                }
            return {name: value / total for name, value in weighted.items()}
        sample_on_base_events = max(self.hits + self.walks + self.hit_by_pitch, 0.0)
        prior_on_base_events = league.min_plate_appearances * league.hitter_on_base
        weighted = {
            "single": _smoothed_rate_from_counts(
                self.singles,
                sample_on_base_events,
                prior_rate=league.prob_single_on_base,
                prior_sample=prior_on_base_events,
            ),
            "double": _smoothed_rate_from_counts(
                self.doubles,
                sample_on_base_events,
                prior_rate=league.prob_double_on_base,
                prior_sample=prior_on_base_events,
            ),
            "triple": _smoothed_rate_from_counts(
                self.triples,
                sample_on_base_events,
                prior_rate=league.prob_triple_on_base,
                prior_sample=prior_on_base_events,
            ),
            "home_run": _smoothed_rate_from_counts(
                self.home_runs,
                sample_on_base_events,
                prior_rate=league.prob_home_run_on_base,
                prior_sample=prior_on_base_events,
            ),
            "walk": _smoothed_rate_from_counts(
                self.walks,
                sample_on_base_events,
                prior_rate=league.prob_walk_on_base,
                prior_sample=prior_on_base_events,
            ),
            "hit_by_pitch": _smoothed_rate_from_counts(
                self.hit_by_pitch,
                sample_on_base_events,
                prior_rate=league.prob_hit_by_pitch_on_base,
                prior_sample=prior_on_base_events,
            ),
        }
        total = sum(max(value, 0.0) for value in weighted.values())
        if total <= 0.0:
            return {
                "single": 1.0,
                "double": 0.0,
                "triple": 0.0,
                "home_run": 0.0,
                "walk": 0.0,
                "hit_by_pitch": 0.0,
            }
        return {name: max(value, 0.0) / total for name, value in weighted.items()}

    def non_home_run_hit_probabilities(self, league: LeagueAverages) -> dict[str, float]:
        if self.resolved_profile is not None:
            return self.resolved_profile.normalized_non_home_run_hit_shares(league)
        sample_non_home_run_hits = self.non_home_run_hits
        prior_non_home_run_hits = league.non_home_run_hit_rate_per_pa * league.min_plate_appearances
        weighted = {
            "single": _smoothed_rate_from_counts(
                self.singles,
                sample_non_home_run_hits,
                prior_rate=league.single_share_of_non_home_run_hits,
                prior_sample=prior_non_home_run_hits,
            ),
            "double": _smoothed_rate_from_counts(
                self.doubles,
                sample_non_home_run_hits,
                prior_rate=league.double_share_of_non_home_run_hits,
                prior_sample=prior_non_home_run_hits,
            ),
            "triple": _smoothed_rate_from_counts(
                self.triples,
                sample_non_home_run_hits,
                prior_rate=league.triple_share_of_non_home_run_hits,
                prior_sample=prior_non_home_run_hits,
            ),
        }
        total = sum(max(value, 0.0) for value in weighted.values())
        if total <= 0.0:
            return {"single": 1.0, "double": 0.0, "triple": 0.0}
        return {name: max(value, 0.0) / total for name, value in weighted.items()}


@dataclass(slots=True)
class PitchingStats:
    player_id: int
    player_name: str = ""
    throws: str = ""
    on_base: float = 0.0
    average_pitches: float = 0.0
    games_started: float = 0.0
    innings_pitched: float = 0.0
    hits_allowed: float = 0.0
    doubles_allowed: float = 0.0
    triples_allowed: float = 0.0
    has_hit_type_detail: bool = False
    walks_allowed: float = 0.0
    hit_batsmen: float = 0.0
    strikeouts: float = 0.0
    home_runs_allowed: float = 0.0
    batters_faced: float = 0.0
    at_bats_against: float = 0.0
    resolved_profile: ResolvedPitchingProfile | None = None

    @classmethod
    def from_api(cls, payload: Mapping[str, Any]) -> "PitchingStats":
        has_hit_type_detail = any(key in payload for key in ("doublesAllowed", "doubles", "2B", "triplesAllowed", "triples", "3B"))
        return cls(
            player_id=_to_int(_lookup(payload, "playerID", "playerId", "id")),
            player_name=_clean_name(_lookup(payload, "playerName", "name")),
            throws=_normalize_handedness(_lookup(payload, "throws", "pitchHand", default="")),
            on_base=_to_float(_lookup(payload, "onBase", "obp", "OBP")),
            average_pitches=_to_float(_lookup(payload, "aveNumOfPitches", "averagePitches")),
            games_started=_to_float(_lookup(payload, "gamesStarted", "GS")),
            innings_pitched=_to_float(_lookup(payload, "inningsPitched", "IP")),
            hits_allowed=_to_float(_lookup(payload, "hitsAllowed", "hits", "H")),
            doubles_allowed=_to_float(_lookup(payload, "doublesAllowed", "doubles", "2B")),
            triples_allowed=_to_float(_lookup(payload, "triplesAllowed", "triples", "3B")),
            has_hit_type_detail=has_hit_type_detail,
            walks_allowed=_to_float(_lookup(payload, "bb", "walksAllowed", "walks", "BB", "baseOnBalls")),
            hit_batsmen=_to_float(_lookup(payload, "hitBatsmen", "hitByPitch", "HBP")),
            strikeouts=_to_float(_lookup(payload, "strikeOuts", "so", "SO")),
            home_runs_allowed=_to_float(_lookup(payload, "homeRunsAllowed", "homeRuns", "HR")),
            batters_faced=_to_float(_lookup(payload, "battersFaced", "BF")),
            at_bats_against=_to_float(_lookup(payload, "atBatsAgainst", "atBats", "AB")),
        )

    @classmethod
    def default(cls, league: LeagueAverages | None = None) -> "PitchingStats":
        league = league or LeagueAverages()
        return cls(
            player_id=0,
            player_name="League Average Pitcher",
            on_base=league.pitcher_on_base,
            average_pitches=league.average_pitches_per_start,
            games_started=max(league.min_innings_pitched / 5.5, 1.0),
            innings_pitched=league.min_innings_pitched,
            hits_allowed=league.hits_total,
            doubles_allowed=league.doubles_total,
            triples_allowed=league.triples_total,
            has_hit_type_detail=True,
            walks_allowed=league.walks_total,
            hit_batsmen=league.hit_by_pitch_total,
            strikeouts=league.strikeouts_total,
            home_runs_allowed=league.home_runs_total,
            batters_faced=league.total_plate_appearances,
            at_bats_against=league.at_bats_per_lineup_spot * 9.0,
        )

    @property
    def has_outcome_detail(self) -> bool:
        return any(
            value > 0.0
            for value in (
                self.batters_faced,
                self.at_bats_against,
                self.hits_allowed,
                self.walks_allowed,
                self.hit_batsmen,
                self.strikeouts,
                self.home_runs_allowed,
            )
        )

    @property
    def estimated_batters_faced(self) -> float:
        if self.resolved_profile is not None and self.resolved_profile.effective_batters_faced > 0.0:
            return self.resolved_profile.effective_batters_faced
        if not self.has_outcome_detail:
            return 0.0
        return max(
            self.batters_faced,
            self.at_bats_against + self.walks_allowed + self.hit_batsmen,
            self.innings_pitched * 4.25,
        )

    @property
    def non_home_run_hits_allowed(self) -> float:
        return max(self.hits_allowed - self.home_runs_allowed, 0.0)

    @property
    def singles_allowed(self) -> float:
        return max(self.non_home_run_hits_allowed - self.doubles_allowed - self.triples_allowed, 0.0)

    def non_home_run_hit_probabilities_allowed(self, league: LeagueAverages) -> dict[str, float]:
        if self.resolved_profile is not None:
            return self.resolved_profile.normalized_non_home_run_hit_shares(league)
        sample_non_home_run_hits = self.non_home_run_hits_allowed if self.has_hit_type_detail else 0.0
        prior_non_home_run_hits = league.non_home_run_hit_rate_per_pa * league.prior_batters_faced_for_pitcher_prior
        return _normalize_shares(
            {
                "single": _smoothed_rate_from_counts(
                    self.singles_allowed,
                    sample_non_home_run_hits,
                    prior_rate=league.single_share_of_non_home_run_hits,
                    prior_sample=prior_non_home_run_hits,
                ),
                "double": _smoothed_rate_from_counts(
                    self.doubles_allowed,
                    sample_non_home_run_hits,
                    prior_rate=league.double_share_of_non_home_run_hits,
                    prior_sample=prior_non_home_run_hits,
                ),
                "triple": _smoothed_rate_from_counts(
                    self.triples_allowed,
                    sample_non_home_run_hits,
                    prior_rate=league.triple_share_of_non_home_run_hits,
                    prior_sample=prior_non_home_run_hits,
                ),
            },
            {
                "single": league.single_share_of_non_home_run_hits,
                "double": league.double_share_of_non_home_run_hits,
                "triple": league.triple_share_of_non_home_run_hits,
            },
        )

    def _fallback_allowed_rate(self, league_rate: float, league: LeagueAverages, *, scale_to_on_base: bool) -> float:
        if not scale_to_on_base:
            return league_rate
        scale = self.weighted_on_base(league) / max(league.hitter_on_base, 1e-6)
        return max(league_rate * scale, 0.0)

    def _smoothed_allowed_rate(
        self,
        observed_events: float,
        sample: float,
        *,
        league_rate: float,
        league: LeagueAverages,
        scale_to_on_base: bool = False,
    ) -> float:
        if sample <= 0.0 and not self.has_outcome_detail:
            return self._fallback_allowed_rate(league_rate, league, scale_to_on_base=scale_to_on_base)
        return _smoothed_rate_from_counts(
            observed_events,
            sample,
            prior_rate=league_rate,
            prior_sample=league.prior_batters_faced_for_pitcher_prior,
        )

    def weighted_on_base(self, league: LeagueAverages) -> float:
        if self.resolved_profile is not None:
            return _clamp_rate(self.resolved_profile.on_base_allowed_rate)
        sample = max(self.estimated_batters_faced, self.innings_pitched * 4.25, 0.0)
        observed_events = max(self.hits_allowed + self.walks_allowed + self.hit_batsmen, self.on_base * sample)
        return _smoothed_rate_from_counts(
            observed_events,
            sample,
            prior_rate=league.pitcher_on_base,
            prior_sample=league.prior_batters_faced_for_pitcher_prior,
        )

    def walk_rate_allowed(self, league: LeagueAverages) -> float:
        if self.resolved_profile is not None:
            return _clamp_rate(self.resolved_profile.walk_rate_allowed)
        return self._smoothed_allowed_rate(
            self.walks_allowed,
            self.estimated_batters_faced,
            league_rate=league.walk_rate_per_pa,
            league=league,
            scale_to_on_base=True,
        )

    def hit_by_pitch_rate_allowed(self, league: LeagueAverages) -> float:
        if self.resolved_profile is not None:
            return _clamp_rate(self.resolved_profile.hit_by_pitch_rate_allowed)
        return self._smoothed_allowed_rate(
            self.hit_batsmen,
            self.estimated_batters_faced,
            league_rate=league.hit_by_pitch_rate_per_pa,
            league=league,
            scale_to_on_base=True,
        )

    def strikeout_rate(self, league: LeagueAverages) -> float:
        if self.resolved_profile is not None:
            return _clamp_rate(self.resolved_profile.strikeout_rate)
        return self._smoothed_allowed_rate(
            self.strikeouts,
            self.estimated_batters_faced,
            league_rate=league.strikeout_rate_per_pa,
            league=league,
            scale_to_on_base=False,
        )

    def home_run_rate_allowed(self, league: LeagueAverages) -> float:
        if self.resolved_profile is not None:
            return _clamp_rate(self.resolved_profile.home_run_rate_allowed)
        return self._smoothed_allowed_rate(
            self.home_runs_allowed,
            self.estimated_batters_faced,
            league_rate=league.home_run_rate_per_pa,
            league=league,
            scale_to_on_base=True,
        )

    def non_home_run_hit_rate_allowed(self, league: LeagueAverages) -> float:
        if self.resolved_profile is not None:
            return _clamp_rate(self.resolved_profile.non_home_run_hit_rate_allowed)
        return self._smoothed_allowed_rate(
            max(self.hits_allowed - self.home_runs_allowed, 0.0),
            self.estimated_batters_faced,
            league_rate=league.non_home_run_hit_rate_per_pa,
            league=league,
            scale_to_on_base=True,
        )


@dataclass(slots=True)
class MatchUp:
    game_id: str
    date: str = ""
    away_name: str = ""
    home_name: str = ""

    @classmethod
    def from_api(cls, payload: Mapping[str, Any]) -> "MatchUp":
        return cls(
            game_id=str(_lookup(payload, "gameid", "gameId", "id", default="")),
            date=str(_lookup(payload, "date", default="")),
            away_name=_clean_name(_lookup(payload, "visName", "awayTeam", "awayName")),
            home_name=_clean_name(_lookup(payload, "homeName", "homeTeam")),
        )


@dataclass(slots=True)
class SimulationInput:
    game_date: str
    away_team: str
    home_team: str
    away_lineup: list[BattingStats]
    home_lineup: list[BattingStats]
    away_pitching: list[PitchingStats]
    home_pitching: list[PitchingStats]

    @classmethod
    def from_api(cls, payload: Mapping[str, Any]) -> "SimulationInput":
        return cls(
            game_date=str(_lookup(payload, "gamedate", "gameDate", default="")),
            away_team=_clean_name(_lookup(payload, "awayTeam", default="Away")),
            home_team=_clean_name(_lookup(payload, "homeTeam", default="Home")),
            away_lineup=[BattingStats.from_api(item) for item in _lookup(payload, "awayLineUp", "away_lineup", default=[])],
            home_lineup=[BattingStats.from_api(item) for item in _lookup(payload, "homeLineUp", "home_lineup", default=[])],
            away_pitching=[PitchingStats.from_api(item) for item in _lookup(payload, "awayPitching", "away_pitching", default=[])],
            home_pitching=[PitchingStats.from_api(item) for item in _lookup(payload, "homePitching", "home_pitching", default=[])],
        )


@dataclass(slots=True)
class PlayerGameStats:
    team_side: TeamSide
    player_id: int
    player_name: str = ""
    at_bats: int = 0
    runs: int = 0
    hits: int = 0
    rbi: int = 0
    walks: int = 0
    strikeouts: int = 0
    pitches_seen: int = 0
    hit_by_pitch: int = 0
    sac_flies: int = 0
    doubles: int = 0
    triples: int = 0
    home_runs: int = 0
    singles: int = 0
    plate_appearances: int = 0

    @property
    def batting_average(self) -> float:
        return self.hits / self.at_bats if self.at_bats else 0.0

    @property
    def on_base_percentage(self) -> float:
        numerator = self.hits + self.walks + self.hit_by_pitch
        return numerator / self.plate_appearances if self.plate_appearances else 0.0

    @property
    def slugging_percentage(self) -> float:
        if not self.at_bats:
            return 0.0
        total_bases = self.singles + (2 * self.doubles) + (3 * self.triples) + (4 * self.home_runs)
        return total_bases / self.at_bats


@dataclass(slots=True)
class BaseState:
    first: PlayerKey | None = None
    second: PlayerKey | None = None
    third: PlayerKey | None = None

    def clear(self) -> None:
        self.first = None
        self.second = None
        self.third = None


@dataclass(slots=True)
class ScoreBoard:
    frames: list[int] = field(default_factory=list)
    away_hits: int = 0
    home_hits: int = 0
    away_errors: int = 0
    home_errors: int = 0
    current_frame: int = -1

    def add_frame(self) -> None:
        self.frames.append(0)
        self.current_frame += 1

    @property
    def away_is_batting(self) -> bool:
        return self.current_frame % 2 == 0

    def increment_run(self) -> None:
        self.frames[self.current_frame] += 1

    def increment_hit(self) -> None:
        if self.away_is_batting:
            self.away_hits += 1
        else:
            self.home_hits += 1

    @property
    def away_runs(self) -> int:
        return sum(self.frames[::2])

    @property
    def home_runs(self) -> int:
        return sum(self.frames[1::2])


@dataclass(slots=True)
class PlayerProjection:
    team_side: TeamSide
    team_name: str
    player_id: int
    player_name: str
    at_bats: float
    runs: float
    hits: float
    hit_1_plus_probability: float
    hit_2_plus_probability: float
    doubles: float
    triples: float
    home_runs: float
    rbi: float
    walks: float
    batting_average: float
    on_base_percentage: float
    slugging_percentage: float


@dataclass(slots=True)
class SimulationSummary:
    game_date: str
    away_team: str
    home_team: str
    away_win_probability: float
    home_win_probability: float
    average_total_runs: float
    median_total_runs: float
    stddev_total_runs: float
    total_runs_samples: list[float]
    total_runs_histogram: list[int]
    away_projections: list[PlayerProjection]
    home_projections: list[PlayerProjection]

    def probability_over(self, line: float) -> float:
        denominator = max(len(self.total_runs_samples), 1)
        total = sum(1 for value in self.total_runs_samples if value > line)
        return total / denominator

    def probability_under(self, line: float) -> float:
        denominator = max(len(self.total_runs_samples), 1)
        total = sum(1 for value in self.total_runs_samples if value <= line)
        return total / denominator
