from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .league_averages import LeagueAverages
from .models import (
    BattingStats,
    MatchUp,
    PitchingStats,
    ResolvedBattingProfile,
    ResolvedPitchingProfile,
    SimulationInput,
    _smoothed_rate_from_counts,
)


_PLAYER_ID_RE = re.compile(r"/player/[^/]*-(\d+)")
_DEFAULT_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
)
_LINEUP_LABEL_RE = re.compile(r"^[A-Z]{2,4} Lineup$")
_LINEUP_ROW_RE = re.compile(r"^(\d+)\.\s+(.+?)(?:\s+\([LRS]\).*)?$")

_TEAM_ALIASES = {
    "athletics": "athletics",
    "as": "athletics",
    "a s": "athletics",
    "diamondbacks": "diamondbacks",
    "dbacks": "diamondbacks",
    "d backs": "diamondbacks",
    "redsox": "redsox",
    "red sox": "redsox",
    "whitesox": "whitesox",
    "white sox": "whitesox",
    "bluejays": "bluejays",
    "blue jays": "bluejays",
}

_OFFICIAL_TEAM_NAMES = {
    "orioles": "Baltimore Orioles",
    "redsox": "Boston Red Sox",
    "yankees": "New York Yankees",
    "rays": "Tampa Bay Rays",
    "bluejays": "Toronto Blue Jays",
    "whitesox": "Chicago White Sox",
    "guardians": "Cleveland Guardians",
    "tigers": "Detroit Tigers",
    "royals": "Kansas City Royals",
    "twins": "Minnesota Twins",
    "astros": "Houston Astros",
    "athletics": "Athletics",
    "angels": "Los Angeles Angels",
    "mariners": "Seattle Mariners",
    "rangers": "Texas Rangers",
    "braves": "Atlanta Braves",
    "marlins": "Miami Marlins",
    "mets": "New York Mets",
    "phillies": "Philadelphia Phillies",
    "nationals": "Washington Nationals",
    "cubs": "Chicago Cubs",
    "reds": "Cincinnati Reds",
    "brewers": "Milwaukee Brewers",
    "pirates": "Pittsburgh Pirates",
    "cardinals": "St. Louis Cardinals",
    "diamondbacks": "Arizona Diamondbacks",
    "rockies": "Colorado Rockies",
    "dodgers": "Los Angeles Dodgers",
    "padres": "San Diego Padres",
    "giants": "San Francisco Giants",
}
_TEAM_STATS_TOKEN_RE = re.compile(r"\d+\.\d+|\.\d+|\d+")
_TEAM_BULLPEN_ROW_RE = re.compile(r"^(AL|NL)\b")
_TEAM_BULLPEN_CACHE_FORMAT_VERSION = 2
_BATTING_SIDE_RE = re.compile(r"\(([LRS])\)")
_THROWING_HAND_RE = re.compile(r"\b([LRS])HP\b")

_DEFAULT_PLAYER_CACHE_DIRNAME = "player_cache"
_PRIOR_SEASON_WEIGHTS = (5.0, 4.0, 3.0)
_PLAYER_CACHE_FORMAT_VERSION = 6
_PROFILE_CACHE_MODEL_VERSION = 7

_HITTER_PRIOR_PA_CAPS = {
    "walk": 120.0,
    "hit_by_pitch": 240.0,
    "strikeout": 150.0,
    "home_run": 170.0,
    "non_home_run_hit": 260.0,
    "stolen_bases": 180.0,
}
_HITTER_PRIOR_NON_HOME_RUN_HIT_CAP = 95.0
_HITTER_PRIOR_BIP_OUT_CAP = 170.0
_PITCHER_PRIOR_BF_CAPS = {
    "walk": 170.0,
    "hit_by_pitch": 320.0,
    "strikeout": 170.0,
    "home_run": 260.0,
    "non_home_run_hit": 320.0,
}
_PITCHER_PRIOR_STARTS_CAP = 10.0



class MlbDataError(RuntimeError):
    pass




def _cap_prior_sample(sample: float, cap: float) -> float:
    if cap <= 0.0:
        return max(sample, 0.0)
    return min(max(sample, 0.0), cap)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _normalize_batting_side(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"L", "R", "S"}:
        return text
    return ""


def _normalize_throwing_hand(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"L", "R", "S"}:
        return text
    return ""


def _extract_batting_side(value: Any) -> str:
    match = _BATTING_SIDE_RE.search(str(value or "").upper())
    if match is None:
        return ""
    return _normalize_batting_side(match.group(1))


def _extract_throwing_hand(value: Any) -> str:
    match = _THROWING_HAND_RE.search(str(value or "").upper())
    if match is None:
        return ""
    return _normalize_throwing_hand(match.group(1))


def _merge_handedness_into_record(
    record: dict[str, Any],
    *,
    bats: str = "",
    throws: str = "",
) -> dict[str, Any]:
    merged = dict(record or {})
    normalized_bats = _normalize_batting_side(bats or merged.get("bats") or merged.get("batSide"))
    normalized_throws = _normalize_throwing_hand(throws or merged.get("throws") or merged.get("pitchHand"))
    if normalized_bats:
        merged["bats"] = normalized_bats
    if normalized_throws:
        merged["throws"] = normalized_throws
    return merged
@dataclass(slots=True)
class LineupPlayer:
    player_id: int
    player_name: str
    bats: str = ""
    throws: str = ""


@dataclass(slots=True)
class LineupCard:
    away_team: str
    home_team: str
    away_lineup: list[LineupPlayer]
    home_lineup: list[LineupPlayer]
    away_pitcher: LineupPlayer | None = None
    home_pitcher: LineupPlayer | None = None

    @property
    def away_team_key(self) -> str:
        return canonical_team_key(self.away_team)

    @property
    def home_team_key(self) -> str:
        return canonical_team_key(self.home_team)


@dataclass(slots=True)
class LineupInspection:
    game_id: str
    game_date: str
    away_team: str
    home_team: str
    source: str
    away_lineup: list[LineupPlayer] = field(default_factory=list)
    home_lineup: list[LineupPlayer] = field(default_factory=list)
    away_pitcher: LineupPlayer | None = None
    home_pitcher: LineupPlayer | None = None
    note: str = ""

    @property
    def is_set(self) -> bool:
        return len(self.away_lineup) == 9 and len(self.home_lineup) == 9


@dataclass(slots=True)
class MlbDataClient:
    statsapi_base_url: str = "https://statsapi.mlb.com/api/v1"
    starting_lineups_base_url: str = "https://www.mlb.com/starting-lineups"
    team_pitching_stats_base_url: str = "https://www.mlb.com/stats/team/pitching"
    timeout: int = 30
    debug_dir: str | None = None
    player_cache_dir: str | None = None
    profile_fallback_league: LeagueAverages | None = None
    session: requests.Session = field(init=False, repr=False)
    _player_stat_cache: dict[tuple[int, str, int], dict[str, Any]] = field(init=False, repr=False, default_factory=dict)
    _resolved_batting_stats_cache: dict[tuple[int, int], BattingStats] = field(init=False, repr=False, default_factory=dict)
    _resolved_pitching_stats_cache: dict[tuple[int, int], PitchingStats] = field(init=False, repr=False, default_factory=dict)
    _team_bullpen_stats_cache: dict[tuple[int, int], PitchingStats] = field(init=False, repr=False, default_factory=dict)
    _team_bullpen_table_cache: dict[int, dict[str, dict[str, Any]]] = field(init=False, repr=False, default_factory=dict)
    _schedule_cache: dict[str, dict[str, Any]] = field(init=False, repr=False, default_factory=dict)
    _lineups_page_cache: dict[str, list[LineupCard]] = field(init=False, repr=False, default_factory=dict)
    _lineups_page_debug: dict[str, dict[str, Any]] = field(init=False, repr=False, default_factory=dict)
    _live_feed_debug: dict[str, dict[str, Any]] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        session = requests.Session()
        session.headers.update({
            "User-Agent": _DEFAULT_BROWSER_USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self.session = session
        if self.profile_fallback_league is None:
            self.profile_fallback_league = LeagueAverages()

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "MlbDataClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _debug_root(self) -> Path | None:
        if not self.debug_dir:
            return None
        root = Path(self.debug_dir)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _debug_write_text(self, relative_path: str, content: str) -> None:
        root = self._debug_root()
        if root is None:
            return
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _debug_write_json(self, relative_path: str, payload: dict[str, Any]) -> None:
        root = self._debug_root()
        if root is None:
            return
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _player_cache_root(self) -> Path:
        root = Path(self.player_cache_dir) if self.player_cache_dir else default_player_cache_dir()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _starting_lineups_request_headers(self) -> dict[str, str]:
        return {
            "User-Agent": _DEFAULT_BROWSER_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Referer": "https://www.mlb.com/",
        }

    def _team_stats_request_headers(self) -> dict[str, str]:
        return {
            "User-Agent": _DEFAULT_BROWSER_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Referer": "https://www.mlb.com/stats/",
        }

    def _statsapi_url(self, path: str) -> str:
        return urljoin(self.statsapi_base_url.rstrip("/") + "/", path.lstrip("/"))

    def _record_live_feed_debug(self, game_id: str, *, url: str, response: requests.Response | None, error: str = "") -> None:
        payload: dict[str, Any] = {
            "game_id": str(game_id),
            "requested_url": url,
            "status_code": getattr(response, "status_code", None),
            "final_url": getattr(response, "url", url),
            "content_type": (response.headers.get("content-type") if response is not None else "") or "",
            "error": error,
        }
        if response is not None and response.text:
            payload["body_preview_lines"] = _extract_text_preview(response.text, limit=20)
        self._live_feed_debug[str(game_id)] = payload
        self._debug_write_json(f"statsapi/game_{game_id}_feed_live.json", payload)

    def get_lineups_debug(self, game_date: str) -> dict[str, Any]:
        self._fetch_starting_lineups_cards(game_date)
        return self._lineups_page_debug.get(
            game_date,
            {"game_date": game_date, "total_cards": 0, "successful_fetches": 0, "sources": [], "matchups": []},
        )

    def _get_json(self, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise MlbDataError(f"Expected JSON object from {url}, got {type(payload)!r}")
        return payload

    def _statsapi_get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self._statsapi_url(path)
        return self._get_json(url, params=params)

    def _starting_lineups_url(self, game_date: str) -> str:
        base = self.starting_lineups_base_url.rstrip("/")
        return f"{base}/{game_date}"

    def fetch_daily_games(self, game_date: date | str) -> list[MatchUp]:
        payload = self._statsapi_get(
            "schedule",
            params={
                "sportId": 1,
                "date": str(game_date),
                "hydrate": "probablePitcher,linescore,team",
            },
        )
        games = []
        for game in _iter_schedule_games(payload):
            game_pk = str(game.get("gamePk", ""))
            if game_pk:
                self._schedule_cache[game_pk] = game
            games.append(MatchUp.from_api(_schedule_game_to_matchup_payload(game)))
        return games

    def fetch_game(self, game_id: str, *, lineup_source: str = "auto") -> SimulationInput:
        schedule_game = self._get_schedule_game(game_id)
        official_date = str(schedule_game.get("officialDate", ""))
        season = _year_from_date_text(official_date)

        if lineup_source not in {"auto", "boxscore", "starting-lineups"}:
            raise MlbDataError(f"Unsupported lineup source: {lineup_source}")

        live_feed: dict[str, Any] = {}
        live_feed_unavailable = False
        if lineup_source in {"auto", "boxscore"}:
            live_feed, live_feed_unavailable = self._fetch_live_feed_if_available(game_id)
            from_live_feed = self._build_simulation_input_from_live_feed(schedule_game, live_feed, season=season)
            if from_live_feed is not None:
                return from_live_feed
            if lineup_source == "boxscore":
                if live_feed_unavailable:
                    raise MlbDataError(
                        f"The MLB live feed for game {game_id} is not available yet. Try again later or use "
                        f"--lineup-source starting-lineups if lineups have been posted on MLB.com."
                    )
                raise MlbDataError(
                    f"Live boxscore did not contain a complete lineup for {schedule_game_team_name(schedule_game, 'away')} at "
                    f"{schedule_game_team_name(schedule_game, 'home')} on {official_date}."
                )

        from_starting_lineups = self._build_simulation_input_from_starting_lineups_page(
            schedule_game,
            live_feed,
            season=season,
            game_date=official_date,
        )
        if from_starting_lineups is not None:
            return from_starting_lineups

        inspection = self.inspect_game_lineup(game_id, lineup_source=lineup_source)
        note = f" {inspection.note}" if inspection.note else ""
        raise MlbDataError(
            f"Could not resolve a full 9-player lineup for {schedule_game_team_name(schedule_game, 'away')} at "
            f"{schedule_game_team_name(schedule_game, 'home')} on {official_date}."
            f" Resolved {len(inspection.away_lineup)} away hitters and {len(inspection.home_lineup)} home hitters "
            f"from {inspection.source}.{note}"
        )

    def inspect_game_lineup(self, game_id: str, *, lineup_source: str = "auto") -> LineupInspection:
        schedule_game = self._get_schedule_game(game_id)
        official_date = str(schedule_game.get("officialDate", ""))

        if lineup_source not in {"auto", "boxscore", "starting-lineups"}:
            raise MlbDataError(f"Unsupported lineup source: {lineup_source}")

        live_feed: dict[str, Any] = {}
        live_feed_unavailable = False
        if lineup_source in {"auto", "boxscore"}:
            live_feed, live_feed_unavailable = self._fetch_live_feed_if_available(game_id)
            inspection = self._inspect_lineup_from_live_feed(schedule_game, live_feed)
            if inspection.is_set:
                return inspection
            if lineup_source == "boxscore":
                note = inspection.note
                if live_feed_unavailable:
                    note = f"{note} Live feed endpoint returned 404.".strip()
                inspection.note = note or "Live boxscore did not contain a complete lineup."
                return inspection

        inspection = self._inspect_lineup_from_starting_lineups_page(
            schedule_game,
            live_feed,
            game_date=official_date,
        )
        if inspection is not None:
            return inspection

        lineups_debug = self.get_lineups_debug(official_date)
        total_cards = int(lineups_debug.get("total_cards", 0))
        successful_fetches = int(lineups_debug.get("successful_fetches", 0))
        if total_cards > 0:
            sample_matchups = ", ".join(lineups_debug.get("matchups", [])[:6])
            note = (
                f"No matching lineup card was found on MLB.com. Parsed {total_cards} matchup cards across "
                f"{successful_fetches} successful lineup page fetches."
            )
            if sample_matchups:
                note = f"{note} Sample parsed matchups: {sample_matchups}."
        else:
            note = (
                "No matching lineup card was found on MLB.com. Parsed 0 matchup cards from the fetched lineup pages."
            )
            first_source = next(iter(lineups_debug.get("sources", [])), None)
            if first_source:
                title = first_source.get("title") or ""
                note = (
                    f"{note} First fetch: status {first_source.get('status_code')} from {first_source.get('final_url') or first_source.get('requested_url')}"
                    f" with title {title!r}."
                )
        if live_feed_unavailable:
            note = f"{note} The MLB live feed endpoint also returned 404."
        if self.debug_dir:
            note = f"{note} Debug files written to {Path(self.debug_dir).resolve()}."
        return LineupInspection(
            game_id=str(game_id),
            game_date=official_date,
            away_team=schedule_game_team_name(schedule_game, "away"),
            home_team=schedule_game_team_name(schedule_game, "home"),
            source="starting-lineups",
            note=note,
        )

    def _fetch_live_feed_if_available(self, game_id: str) -> tuple[dict[str, Any], bool]:
        url = self._statsapi_url(f"game/{game_id}/feed/live")
        try:
            payload = self._statsapi_get(f"game/{game_id}/feed/live")
            self._record_live_feed_debug(game_id, url=url, response=None)
            return payload, False
        except requests.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            if status_code == 404:
                self._record_live_feed_debug(game_id, url=url, response=exc.response, error="404 Not Found")
                return {}, True
            self._record_live_feed_debug(game_id, url=url, response=exc.response, error=str(exc))
            raise

    def _get_schedule_game(self, game_id: str) -> dict[str, Any]:
        cached = self._schedule_cache.get(str(game_id))
        if cached is not None:
            return cached
        payload = self._statsapi_get(
            "schedule",
            params={
                "sportId": 1,
                "gamePk": str(game_id),
                "hydrate": "probablePitcher,linescore,team",
            },
        )
        for game in _iter_schedule_games(payload):
            game_pk = str(game.get("gamePk", ""))
            if game_pk:
                self._schedule_cache[game_pk] = game
        cached = self._schedule_cache.get(str(game_id))
        if cached is None:
            raise MlbDataError(f"Game {game_id} was not found in the MLB schedule feed.")
        return cached

    def _build_simulation_input_from_live_feed(
        self,
        schedule_game: dict[str, Any],
        live_feed: dict[str, Any],
        *,
        season: int,
    ) -> SimulationInput | None:
        away_lineup = self._extract_lineup_from_live_feed(live_feed, side="away", season=season)
        home_lineup = self._extract_lineup_from_live_feed(live_feed, side="home", season=season)
        if len(away_lineup) != 9 or len(home_lineup) != 9:
            return None

        away_pitcher = self._extract_starting_pitcher_from_live_feed(schedule_game, live_feed, side="away", season=season)
        home_pitcher = self._extract_starting_pitcher_from_live_feed(schedule_game, live_feed, side="home", season=season)
        away_bullpen = self._fetch_team_bullpen_from_schedule_game(schedule_game, side="away", season=season)
        home_bullpen = self._fetch_team_bullpen_from_schedule_game(schedule_game, side="home", season=season)

        return SimulationInput(
            game_date=str(schedule_game.get("officialDate", "")),
            away_team=schedule_game_team_name(schedule_game, "away"),
            home_team=schedule_game_team_name(schedule_game, "home"),
            away_lineup=away_lineup,
            home_lineup=home_lineup,
            away_pitching=[pitcher for pitcher in [away_pitcher, away_bullpen] if pitcher is not None],
            home_pitching=[pitcher for pitcher in [home_pitcher, home_bullpen] if pitcher is not None],
        )

    def _build_simulation_input_from_starting_lineups_page(
        self,
        schedule_game: dict[str, Any],
        live_feed: dict[str, Any],
        *,
        season: int,
        game_date: str,
    ) -> SimulationInput | None:
        cards = self._fetch_starting_lineups_cards(game_date)
        card = self._find_lineup_card(
            cards,
            away_team=schedule_game_team_name(schedule_game, "away"),
            home_team=schedule_game_team_name(schedule_game, "home"),
        )
        if card is None:
            return None
        if len(card.away_lineup) != 9 or len(card.home_lineup) != 9:
            return None

        away_lineup = [
            self._fetch_batter_from_player_id(player.player_id, player.player_name, season=season, bats=player.bats)
            for player in card.away_lineup
        ]
        home_lineup = [
            self._fetch_batter_from_player_id(player.player_id, player.player_name, season=season, bats=player.bats)
            for player in card.home_lineup
        ]

        away_pitcher_id = self._preferred_pitcher_id(schedule_game, live_feed, side="away", fallback=card.away_pitcher)
        home_pitcher_id = self._preferred_pitcher_id(schedule_game, live_feed, side="home", fallback=card.home_pitcher)

        away_pitcher_name = card.away_pitcher.player_name if card.away_pitcher is not None else ""
        home_pitcher_name = card.home_pitcher.player_name if card.home_pitcher is not None else ""

        away_pitcher = self._fetch_pitcher_from_player_id(
            away_pitcher_id,
            away_pitcher_name,
            season=season,
            throws=card.away_pitcher.throws if card.away_pitcher is not None else "",
        )
        home_pitcher = self._fetch_pitcher_from_player_id(
            home_pitcher_id,
            home_pitcher_name,
            season=season,
            throws=card.home_pitcher.throws if card.home_pitcher is not None else "",
        )
        away_bullpen = self._fetch_team_bullpen_from_schedule_game(schedule_game, side="away", season=season)
        home_bullpen = self._fetch_team_bullpen_from_schedule_game(schedule_game, side="home", season=season)

        return SimulationInput(
            game_date=game_date,
            away_team=schedule_game_team_name(schedule_game, "away"),
            home_team=schedule_game_team_name(schedule_game, "home"),
            away_lineup=away_lineup,
            home_lineup=home_lineup,
            away_pitching=[pitcher for pitcher in [away_pitcher, away_bullpen] if pitcher is not None],
            home_pitching=[pitcher for pitcher in [home_pitcher, home_bullpen] if pitcher is not None],
        )

    def _inspect_lineup_from_live_feed(
        self,
        schedule_game: dict[str, Any],
        live_feed: dict[str, Any],
    ) -> LineupInspection:
        away_lineup = self._extract_lineup_players_from_live_feed(live_feed, side="away")
        home_lineup = self._extract_lineup_players_from_live_feed(live_feed, side="home")

        away_pitcher = None
        home_pitcher = None
        try:
            away_pitcher = self._lineup_pitcher_from_source(schedule_game, live_feed, side="away")
        except MlbDataError:
            pass
        try:
            home_pitcher = self._lineup_pitcher_from_source(schedule_game, live_feed, side="home")
        except MlbDataError:
            pass

        note = ""
        if len(away_lineup) != 9 or len(home_lineup) != 9:
            note = "Live boxscore did not contain a full 9-player lineup."

        return LineupInspection(
            game_id=str(schedule_game.get("gamePk", "")),
            game_date=str(schedule_game.get("officialDate", "")),
            away_team=schedule_game_team_name(schedule_game, "away"),
            home_team=schedule_game_team_name(schedule_game, "home"),
            source="boxscore",
            away_lineup=away_lineup,
            home_lineup=home_lineup,
            away_pitcher=away_pitcher,
            home_pitcher=home_pitcher,
            note=note,
        )

    def _inspect_lineup_from_starting_lineups_page(
        self,
        schedule_game: dict[str, Any],
        live_feed: dict[str, Any],
        *,
        game_date: str,
    ) -> LineupInspection | None:
        cards = self._fetch_starting_lineups_cards(game_date)
        card = self._find_lineup_card(
            cards,
            away_team=schedule_game_team_name(schedule_game, "away"),
            home_team=schedule_game_team_name(schedule_game, "home"),
        )
        if card is None:
            return None

        away_pitcher = card.away_pitcher or self._safe_lineup_pitcher_from_source(schedule_game, live_feed, side="away")
        home_pitcher = card.home_pitcher or self._safe_lineup_pitcher_from_source(schedule_game, live_feed, side="home")

        note = ""
        if len(card.away_lineup) != 9 or len(card.home_lineup) != 9:
            note = (
                f"Starting Lineups page matched this game, but only resolved {len(card.away_lineup)} away hitters and "
                f"{len(card.home_lineup)} home hitters."
            )

        return LineupInspection(
            game_id=str(schedule_game.get("gamePk", "")),
            game_date=game_date,
            away_team=schedule_game_team_name(schedule_game, "away"),
            home_team=schedule_game_team_name(schedule_game, "home"),
            source="starting-lineups",
            away_lineup=card.away_lineup,
            home_lineup=card.home_lineup,
            away_pitcher=away_pitcher,
            home_pitcher=home_pitcher,
            note=note,
        )

    def _extract_lineup_players_from_live_feed(
        self,
        live_feed: dict[str, Any],
        *,
        side: str,
    ) -> list[LineupPlayer]:
        team_box = _lookup(live_feed, "liveData", "boxscore", "teams", side, default={})
        players = team_box.get("players", {}) or {}
        game_players = _lookup(live_feed, "gameData", "players", default={})
        lineup_rows: list[tuple[int, LineupPlayer]] = []

        for batter_id in team_box.get("batters", []) or []:
            player_key = f"ID{batter_id}"
            player_payload = players.get(player_key, {})
            batting_order = str(player_payload.get("battingOrder", ""))
            if not batting_order.endswith("0"):
                continue
            name = _lookup(game_players, player_key, "fullName", default="") or _lookup(player_payload, "person", "fullName", default="")
            bats = (
                _normalize_batting_side(_lookup(game_players, player_key, "batSide", "code", default=""))
                or _normalize_batting_side(_lookup(player_payload, "person", "batSide", "code", default=""))
            )
            lineup_rows.append((int(batting_order or 0), LineupPlayer(int(batter_id), str(name), bats=bats)))

        lineup_rows.sort(key=lambda item: item[0])
        return [player for _, player in lineup_rows[:9]]

    def _extract_lineup_from_live_feed(
        self,
        live_feed: dict[str, Any],
        *,
        side: str,
        season: int,
    ) -> list[BattingStats]:
        team_box = _lookup(live_feed, "liveData", "boxscore", "teams", side, default={})
        players = team_box.get("players", {}) or {}
        game_players = _lookup(live_feed, "gameData", "players", default={})
        lineup_rows: list[tuple[int, BattingStats]] = []

        for batter_id in team_box.get("batters", []) or []:
            player_key = f"ID{batter_id}"
            player_payload = players.get(player_key, {})
            batting_order = str(player_payload.get("battingOrder", ""))
            if not batting_order.endswith("0"):
                continue
            bats = (
                _normalize_batting_side(_lookup(game_players, player_key, "batSide", "code", default=""))
                or _normalize_batting_side(_lookup(player_payload, "person", "batSide", "code", default=""))
            )
            season_stats = _merge_handedness_into_record(
                (player_payload.get("seasonStats", {}) or {}).get("batting") or {},
                bats=bats,
            )
            name = _lookup(game_players, player_key, "fullName", default="") or _lookup(player_payload, "person", "fullName", default="")
            lineup_rows.append(
                (
                    int(batting_order or 0),
                    self._resolve_batter_stats(
                        int(batter_id),
                        str(name),
                        season=season,
                        current_record=season_stats if _batting_stats_are_complete(season_stats) else None,
                        bats=bats,
                    ),
                )
            )

        lineup_rows.sort(key=lambda item: item[0])
        return [stats for _, stats in lineup_rows[:9]]

    def _extract_starting_pitcher_from_live_feed(
        self,
        schedule_game: dict[str, Any],
        live_feed: dict[str, Any],
        *,
        side: str,
        season: int,
    ) -> PitchingStats:
        preferred = self._preferred_pitcher_id(schedule_game, live_feed, side=side)
        player_key = f"ID{preferred}"
        team_box = _lookup(live_feed, "liveData", "boxscore", "teams", side, default={})
        players = team_box.get("players", {}) or {}
        throws = (
            _normalize_throwing_hand(_lookup(live_feed, "gameData", "players", player_key, "pitchHand", "code", default=""))
            or _normalize_throwing_hand(_lookup(players.get(player_key, {}) or {}, "person", "pitchHand", "code", default=""))
        )
        season_stats = _merge_handedness_into_record(
            (players.get(player_key, {}) or {}).get("seasonStats", {}).get("pitching") or {},
            throws=throws,
        )
        name = _lookup(live_feed, "gameData", "players", player_key, "fullName", default="")
        return self._resolve_pitcher_stats(
            preferred,
            str(name),
            season=season,
            current_record=season_stats if _pitching_stats_are_complete(season_stats) else None,
            throws=throws,
        )

    def _preferred_pitcher_id(
        self,
        schedule_game: dict[str, Any],
        live_feed: dict[str, Any],
        *,
        side: str,
        fallback: LineupPlayer | None = None,
    ) -> int:
        schedule_probable = _lookup(schedule_game, "teams", side, "probablePitcher", "id", default=0)
        if schedule_probable:
            return int(schedule_probable)
        if fallback is not None:
            return fallback.player_id
        pitchers = _lookup(live_feed, "liveData", "boxscore", "teams", side, "pitchers", default=[])
        if pitchers:
            return int(pitchers[0])
        raise MlbDataError(
            f"Could not determine the probable starting pitcher for {schedule_game_team_name(schedule_game, side)}."
        )

    def _safe_lineup_pitcher_from_source(
        self,
        schedule_game: dict[str, Any],
        live_feed: dict[str, Any],
        *,
        side: str,
    ) -> LineupPlayer | None:
        try:
            return self._lineup_pitcher_from_source(schedule_game, live_feed, side=side)
        except MlbDataError:
            return None

    def _lineup_pitcher_from_source(
        self,
        schedule_game: dict[str, Any],
        live_feed: dict[str, Any],
        *,
        side: str,
    ) -> LineupPlayer:
        pitcher_id = self._preferred_pitcher_id(schedule_game, live_feed, side=side)
        player_key = f"ID{pitcher_id}"
        name = (
            _lookup(live_feed, "gameData", "players", player_key, "fullName", default="")
            or _lookup(schedule_game, "teams", side, "probablePitcher", "fullName", default="")
            or _lookup(schedule_game, "teams", side, "probablePitcher", "lastInitName", default="")
            or _lookup(schedule_game, "teams", side, "probablePitcher", "lastName", default="")
        )
        throws = _normalize_throwing_hand(_lookup(live_feed, "gameData", "players", player_key, "pitchHand", "code", default=""))
        return LineupPlayer(int(pitcher_id), _clean_name(name) or schedule_game_team_name(schedule_game, side), throws=throws)

    def _fetch_batter_from_player_id(self, player_id: int, player_name: str, *, season: int, bats: str = "") -> BattingStats:
        return self._resolve_batter_stats(player_id, player_name, season=season, bats=bats)

    def _fetch_pitcher_from_player_id(self, player_id: int, player_name: str, *, season: int, throws: str = "") -> PitchingStats:
        return self._resolve_pitcher_stats(player_id, player_name, season=season, throws=throws)

    def _fetch_team_bullpen_from_schedule_game(
        self,
        schedule_game: dict[str, Any],
        *,
        side: str,
        season: int,
    ) -> PitchingStats | None:
        team_id = schedule_game_team_id(schedule_game, side)
        team_name = schedule_game_team_name(schedule_game, side)
        if team_id <= 0 or not team_name:
            return None
        try:
            return self._resolve_team_bullpen_stats(team_id, team_name, season=season)
        except Exception as exc:  # pragma: no cover - network failures fall back to synthetic bullpen
            self._debug_write_json(
                f"bullpen/{season}/{side}_{team_id}_error.json",
                {
                    "team_id": int(team_id),
                    "team_name": team_name,
                    "season": int(season),
                    "error": str(exc),
                },
            )
            return None

    def _resolve_team_bullpen_stats(self, team_id: int, team_name: str, *, season: int) -> PitchingStats:
        cache_key = (int(team_id), int(season))
        cached = self._team_bullpen_stats_cache.get(cache_key)
        if cached is not None:
            return cached

        current_record = self._fetch_team_bullpen_record(team_id, team_name, season=season)
        current_stats = team_bullpen_stats_from_record(team_id, team_name, current_record)

        historical: list[tuple[PitchingStats, float]] = []
        prior_season = int(season) - 1
        if prior_season > 0:
            try:
                prior_record = self._fetch_team_bullpen_record(team_id, team_name, season=prior_season)
            except Exception:
                prior_record = {}
            if prior_record:
                prior_stats = team_bullpen_stats_from_record(team_id, team_name, prior_record)
                if prior_stats.estimated_batters_faced > 0.0:
                    historical.append((prior_stats, 1.0))

        profile = build_resolved_pitching_profile(
            current=current_stats,
            historical=historical,
            league=self.profile_fallback_league or LeagueAverages(),
        )
        current_stats.resolved_profile = ResolvedPitchingProfile(
            effective_batters_faced=profile.effective_batters_faced,
            on_base_allowed_rate=profile.on_base_allowed_rate,
            walk_rate_allowed=profile.walk_rate_allowed,
            hit_by_pitch_rate_allowed=profile.hit_by_pitch_rate_allowed,
            strikeout_rate=profile.strikeout_rate,
            home_run_rate_allowed=profile.home_run_rate_allowed,
            non_home_run_hit_rate_allowed=profile.non_home_run_hit_rate_allowed,
            single_share_allowed=profile.single_share_allowed,
            double_share_allowed=profile.double_share_allowed,
            triple_share_allowed=profile.triple_share_allowed,
            average_pitches=18.0,
            average_batters_faced_per_start=9.0,
        )
        current_stats.average_pitches = 18.0
        current_stats.games_started = 0.0
        self._team_bullpen_stats_cache[cache_key] = current_stats
        return current_stats

    def _fetch_team_bullpen_record(self, team_id: int, team_name: str, *, season: int) -> dict[str, Any]:
        table = self._load_or_fetch_team_bullpen_table(int(season))
        canonical = canonical_team_key(team_name)
        record = dict(table.get(canonical) or {})
        if not record:
            raise MlbDataError(f"Could not find team reliever stats for {team_name} in season {season}.")
        return record

    def _load_or_fetch_team_bullpen_table(self, season: int) -> dict[str, dict[str, Any]]:
        cached = self._team_bullpen_table_cache.get(int(season))
        if cached is not None:
            return cached

        current_year = date.today().year
        today_text = date.today().isoformat()
        if int(season) == current_year:
            cache_path = team_bullpen_current_cache_path(self._player_cache_root(), int(season))
            payload = read_team_bullpen_cache_file(cache_path)
            if payload is not None and int(payload.get("version") or 0) == _TEAM_BULLPEN_CACHE_FORMAT_VERSION and int(payload.get("season") or 0) == int(season) and str(payload.get("fetched_on") or "") == today_text:
                records = {str(key): dict(value or {}) for key, value in (payload.get("records") or {}).items()}
                self._team_bullpen_table_cache[int(season)] = records
                return records
            try:
                records = self._fetch_team_bullpen_table_from_web(int(season))
            except Exception:
                if payload is not None and int(payload.get("version") or 0) == _TEAM_BULLPEN_CACHE_FORMAT_VERSION and int(payload.get("season") or 0) == int(season):
                    records = {str(key): dict(value or {}) for key, value in (payload.get("records") or {}).items()}
                else:
                    raise
            write_team_bullpen_cache_file(cache_path, season=int(season), kind="current", records=records)
        else:
            cache_path = team_bullpen_season_cache_path(self._player_cache_root(), int(season))
            payload = read_team_bullpen_cache_file(cache_path)
            if payload is not None and int(payload.get("version") or 0) == _TEAM_BULLPEN_CACHE_FORMAT_VERSION and int(payload.get("season") or 0) == int(season):
                records = {str(key): dict(value or {}) for key, value in (payload.get("records") or {}).items()}
                self._team_bullpen_table_cache[int(season)] = records
                return records
            records = self._fetch_team_bullpen_table_from_web(int(season))
            write_team_bullpen_cache_file(cache_path, season=int(season), kind="season", records=records)

        self._team_bullpen_table_cache[int(season)] = records
        return records

    def _candidate_team_bullpen_urls(self, season: int) -> list[str]:
        base = self.team_pitching_stats_base_url.rstrip("/")
        candidates = [
            f"{base}/innings-pitched/{int(season)}?split=rp",
            f"{base}/era/{int(season)}?split=rp",
            f"{base}/{int(season)}?split=rp",
            f"{base}?split=rp&season={int(season)}",
        ]
        if int(season) == date.today().year:
            candidates.extend([
                f"{base}?split=rp",
                f"{base}/innings-pitched?split=rp",
                f"{base}/era?split=rp",
            ])
        deduped: list[str] = []
        for url in candidates:
            if url not in deduped:
                deduped.append(url)
        return deduped

    def _fetch_team_bullpen_table_from_web(self, season: int) -> dict[str, dict[str, Any]]:
        last_http_error: requests.HTTPError | None = None
        summaries: list[dict[str, Any]] = []
        for index, url in enumerate(self._candidate_team_bullpen_urls(season), start=1):
            try:
                response = self.session.get(url, headers=self._team_stats_request_headers(), timeout=self.timeout)
                response.raise_for_status()
            except requests.HTTPError as exc:
                last_http_error = exc
                summary = {
                    "requested_url": url,
                    "final_url": getattr(exc.response, "url", url),
                    "status_code": getattr(exc.response, "status_code", None),
                    "parsed_teams": 0,
                }
                summaries.append(summary)
                self._debug_write_json(f"bullpen/{season}/page_{index:02d}_summary.json", summary)
                if exc.response is not None and exc.response.text:
                    self._debug_write_text(f"bullpen/{season}/page_{index:02d}.html", exc.response.text)
                continue

            records = parse_team_bullpen_stats_html(response.text)
            summary = {
                "requested_url": url,
                "final_url": response.url,
                "status_code": response.status_code,
                "parsed_teams": len(records),
                "teams": sorted(records.keys()),
                "title": _extract_html_title(response.text),
            }
            summaries.append(summary)
            self._debug_write_json(f"bullpen/{season}/page_{index:02d}_summary.json", summary)
            self._debug_write_text(f"bullpen/{season}/page_{index:02d}.html", response.text)
            if records:
                self._debug_write_json(f"bullpen/{season}/summary.json", {"season": int(season), "sources": summaries, "parsed_teams": sorted(records.keys())})
                return records

        self._debug_write_json(f"bullpen/{season}/summary.json", {"season": int(season), "sources": summaries, "parsed_teams": []})
        if last_http_error is not None:
            raise last_http_error
        raise MlbDataError(f"Could not parse any team reliever stats pages for season {season}.")

    def _resolve_batter_stats(
        self,
        player_id: int,
        player_name: str,
        *,
        season: int,
        current_record: dict[str, Any] | None = None,
        bats: str = "",
    ) -> BattingStats:
        cache_key = (int(player_id), int(season))
        cached = self._resolved_batting_stats_cache.get(cache_key)
        if cached is not None:
            normalized_bats = _normalize_batting_side(bats)
            if normalized_bats and not cached.bats:
                cached.bats = normalized_bats
            return cached

        base_record = self._resolve_base_record(
            int(player_id),
            _clean_name(player_name),
            group="hitting",
            season=int(season),
            current_record=current_record,
        )
        resolved_record = _merge_handedness_into_record(base_record, bats=bats)
        if resolved_record != base_record:
            self._persist_player_record(
                int(player_id),
                _clean_name(player_name),
                group="hitting",
                season=int(season),
                stat_record=resolved_record,
            )
        stats = batter_stats_from_record(player_id, player_name, resolved_record)
        profile = self._load_materialized_profile(player_id, group="hitting", season=season)
        if profile is None:
            historical_prior = self._build_historical_batting_prior(player_id, player_name, season=season)
            profile = build_resolved_batting_profile(
                current=stats,
                historical=historical_prior,
                league=self.profile_fallback_league or LeagueAverages(),
            )
            self._save_materialized_profile(player_id, _clean_name(player_name), group="hitting", season=season, profile=profile.to_dict())
        stats.resolved_profile = profile
        self._resolved_batting_stats_cache[cache_key] = stats
        return stats

    def _resolve_pitcher_stats(
        self,
        player_id: int,
        player_name: str,
        *,
        season: int,
        current_record: dict[str, Any] | None = None,
        throws: str = "",
    ) -> PitchingStats:
        cache_key = (int(player_id), int(season))
        cached = self._resolved_pitching_stats_cache.get(cache_key)
        if cached is not None:
            normalized_throws = _normalize_throwing_hand(throws)
            if normalized_throws and not cached.throws:
                cached.throws = normalized_throws
            return cached

        base_record = self._resolve_base_record(
            int(player_id),
            _clean_name(player_name),
            group="pitching",
            season=int(season),
            current_record=current_record,
        )
        resolved_record = _merge_handedness_into_record(base_record, throws=throws)
        if resolved_record != base_record:
            self._persist_player_record(
                int(player_id),
                _clean_name(player_name),
                group="pitching",
                season=int(season),
                stat_record=resolved_record,
            )
        stats = pitcher_stats_from_record(player_id, player_name, resolved_record)
        profile = self._load_materialized_profile(player_id, group="pitching", season=season)
        if profile is None:
            historical_prior = self._build_historical_pitching_prior(player_id, player_name, season=season)
            profile = build_resolved_pitching_profile(
                current=stats,
                historical=historical_prior,
                league=self.profile_fallback_league or LeagueAverages(),
            )
            self._save_materialized_profile(player_id, _clean_name(player_name), group="pitching", season=season, profile=profile.to_dict())
        stats.resolved_profile = profile
        self._resolved_pitching_stats_cache[cache_key] = stats
        return stats

    def _resolve_base_record(
        self,
        player_id: int,
        player_name: str,
        *,
        group: str,
        season: int,
        current_record: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if current_record is not None:
            normalized = dict(current_record)
            if season == date.today().year:
                self._write_player_cache_file(
                    player_current_cache_path(self._player_cache_root(), group, player_id),
                    player_id=player_id,
                    player_name=player_name,
                    group=group,
                    season=season,
                    kind="current",
                    stat_record=normalized,
                )
                self._player_stat_cache[(player_id, group, season)] = normalized
            return normalized
        return self._fetch_player_stat_record(player_id, player_name, group=group, season=season)

    def _persist_player_record(
        self,
        player_id: int,
        player_name: str,
        *,
        group: str,
        season: int,
        stat_record: dict[str, Any],
    ) -> None:
        if int(season) == date.today().year:
            path = player_current_cache_path(self._player_cache_root(), group, player_id)
            kind = "current"
        else:
            path = player_season_cache_path(self._player_cache_root(), group, season, player_id)
            kind = "season"
        normalized = dict(stat_record)
        self._write_player_cache_file(
            path,
            player_id=player_id,
            player_name=player_name,
            group=group,
            season=season,
            kind=kind,
            stat_record=normalized,
        )
        self._player_stat_cache[(player_id, group, season)] = normalized

    def _build_historical_batting_prior(self, player_id: int, player_name: str, *, season: int) -> list[tuple[BattingStats, float]]:
        weighted_stats: list[tuple[BattingStats, float]] = []
        for offset, weight in enumerate(_PRIOR_SEASON_WEIGHTS, start=1):
            prior_season = int(season) - offset
            if prior_season <= 0:
                continue
            record = self._fetch_player_stat_record(player_id, player_name, group="hitting", season=prior_season)
            stats = batter_stats_from_record(player_id, player_name, record)
            if stats.total_plate_appearances <= 0.0 and stats.at_bats <= 0.0:
                continue
            weighted_stats.append((stats, weight))
        return weighted_stats

    def _build_historical_pitching_prior(self, player_id: int, player_name: str, *, season: int) -> list[tuple[PitchingStats, float]]:
        weighted_stats: list[tuple[PitchingStats, float]] = []
        for offset, weight in enumerate(_PRIOR_SEASON_WEIGHTS, start=1):
            prior_season = int(season) - offset
            if prior_season <= 0:
                continue
            record = self._fetch_player_stat_record(player_id, player_name, group="pitching", season=prior_season)
            stats = pitcher_stats_from_record(player_id, player_name, record)
            if stats.estimated_batters_faced <= 0.0 and stats.innings_pitched <= 0.0:
                continue
            weighted_stats.append((stats, weight))
        return weighted_stats

    def _fetch_player_stat_record(self, player_id: int, player_name: str, *, group: str, season: int) -> dict[str, Any]:
        cache_key = (int(player_id), group, int(season))
        cached = self._player_stat_cache.get(cache_key)
        if cached is not None:
            return cached
        if int(season) == date.today().year:
            record = self._load_or_fetch_current_player_record(int(player_id), _clean_name(player_name), group=group, season=int(season))
        else:
            record = self._load_or_fetch_static_player_record(int(player_id), _clean_name(player_name), group=group, season=int(season))
        self._player_stat_cache[cache_key] = record
        return record

    def _load_or_fetch_current_player_record(self, player_id: int, player_name: str, *, group: str, season: int) -> dict[str, Any]:
        current_path = player_current_cache_path(self._player_cache_root(), group, player_id)
        cached_payload = read_player_cache_file(current_path)
        today_text = date.today().isoformat()

        if cached_payload is not None:
            cached_version = int(cached_payload.get("version") or 0)
            cached_season = int(cached_payload.get("season") or 0)
            cached_record = dict(cached_payload.get("stat_record") or {})
            fetched_on = str(cached_payload.get("fetched_on") or "")
            if cached_version == _PLAYER_CACHE_FORMAT_VERSION and cached_season == int(season) and fetched_on == today_text:
                return cached_record
            if cached_season > 0 and cached_season != int(season) and cached_record:
                self._promote_cached_current_record_to_static(
                    player_id,
                    player_name=str(cached_payload.get("player_name") or player_name),
                    group=group,
                    cached_season=cached_season,
                    fallback_record=cached_record,
                )

        try:
            record = self._fetch_player_stat_record_from_api(player_id, group=group, season=season)
        except Exception:
            if cached_payload is not None and int(cached_payload.get("version") or 0) == _PLAYER_CACHE_FORMAT_VERSION and int(cached_payload.get("season") or 0) == int(season):
                fallback = dict(cached_payload.get("stat_record") or {})
                if fallback:
                    return fallback
            raise

        self._write_player_cache_file(
            current_path,
            player_id=player_id,
            player_name=player_name,
            group=group,
            season=season,
            kind="current",
            stat_record=record,
        )
        return record

    def _load_or_fetch_static_player_record(self, player_id: int, player_name: str, *, group: str, season: int) -> dict[str, Any]:
        season_path = player_season_cache_path(self._player_cache_root(), group, season, player_id)
        cached_payload = read_player_cache_file(season_path)
        if cached_payload is not None and int(cached_payload.get("version") or 0) == _PLAYER_CACHE_FORMAT_VERSION and int(cached_payload.get("season") or 0) == int(season):
            cached_record = dict(cached_payload.get("stat_record") or {})
            if cached_record:
                return cached_record
        record = self._fetch_player_stat_record_from_api(player_id, group=group, season=season)
        self._write_player_cache_file(
            season_path,
            player_id=player_id,
            player_name=player_name,
            group=group,
            season=season,
            kind="season",
            stat_record=record,
        )
        return record

    def _promote_cached_current_record_to_static(
        self,
        player_id: int,
        *,
        player_name: str = "",
        group: str,
        cached_season: int,
        fallback_record: dict[str, Any] | None = None,
    ) -> None:
        season_path = player_season_cache_path(self._player_cache_root(), group, cached_season, player_id)
        if season_path.exists():
            return
        record: dict[str, Any]
        try:
            record = self._fetch_player_stat_record_from_api(player_id, group=group, season=cached_season)
        except Exception:
            if fallback_record is None:
                return
            record = dict(fallback_record)
        self._write_player_cache_file(
            season_path,
            player_id=player_id,
            player_name=player_name or str((fallback_record or {}).get("playerName") or ""),
            group=group,
            season=cached_season,
            kind="season",
            stat_record=record,
        )

    def _load_materialized_profile(
        self,
        player_id: int,
        *,
        group: str,
        season: int,
    ) -> ResolvedBattingProfile | ResolvedPitchingProfile | None:
        path = player_profile_cache_path(self._player_cache_root(), group, season, player_id)
        payload = read_player_cache_file(path)
        if payload is None:
            return None
        if int(payload.get("version") or 0) != _PLAYER_CACHE_FORMAT_VERSION:
            return None
        if int(payload.get("season") or 0) != int(season):
            return None
        if str(payload.get("kind") or "") != "profile":
            return None
        if int(payload.get("model_version") or 0) != _PROFILE_CACHE_MODEL_VERSION:
            return None
        if int(season) == date.today().year and str(payload.get("as_of_date") or "") != date.today().isoformat():
            return None
        profile_payload = dict(payload.get("profile") or payload.get("stat_record") or {})
        if not profile_payload:
            return None
        if group == "hitting":
            return ResolvedBattingProfile.from_dict(profile_payload)
        return ResolvedPitchingProfile.from_dict(profile_payload)

    def _save_materialized_profile(
        self,
        player_id: int,
        player_name: str,
        *,
        group: str,
        season: int,
        profile: dict[str, Any],
    ) -> None:
        path = player_profile_cache_path(self._player_cache_root(), group, season, player_id)
        self._write_player_cache_file(
            path,
            player_id=player_id,
            player_name=player_name,
            group=group,
            season=season,
            kind="profile",
            stat_record={},
            extra={
                "as_of_date": date.today().isoformat(),
                "model_version": _PROFILE_CACHE_MODEL_VERSION,
                "profile": profile,
            },
        )

    def _write_player_cache_file(
        self,
        path: Path,
        *,
        player_id: int,
        player_name: str = "",
        group: str,
        season: int,
        kind: str,
        stat_record: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> None:
        write_player_cache_file(
            path,
            player_id=player_id,
            player_name=player_name,
            group=group,
            season=season,
            kind=kind,
            stat_record=stat_record,
            extra=extra,
        )

    def _fetch_player_stat_record_from_api(self, player_id: int, *, group: str, season: int) -> dict[str, Any]:
        primary_params = {
            "stats": "season",
            "group": group,
            "season": int(season),
            "sportId": 1,
            "gameType": "R",
        }
        try:
            payload = self._statsapi_get(f"people/{int(player_id)}/stats", params=primary_params)
            record = extract_first_stat_split(payload)
            if record:
                return record
        except requests.HTTPError:
            pass

        hydrate = f"stats(group=[{group}],type=[season],season={int(season)},sportId=1,gameType=[R])"
        payload = self._statsapi_get(
            "people",
            params={
                "personIds": int(player_id),
                "hydrate": hydrate,
            },
        )
        return extract_first_people_stat_split(payload, int(player_id))

    def _candidate_starting_lineups_urls(self, game_date: str) -> list[str]:
        base = self.starting_lineups_base_url.rstrip("/")
        candidates = [f"{base}/{game_date}", base, f"{base}?date={game_date}"]
        deduped: list[str] = []
        for url in candidates:
            if url not in deduped:
                deduped.append(url)
        return deduped

    def _fetch_starting_lineups_cards(self, game_date: str) -> list[LineupCard]:
        cached = self._lineups_page_cache.get(game_date)
        if cached is not None:
            return cached

        best_by_matchup: dict[tuple[str, str], LineupCard] = {}
        last_http_error: requests.HTTPError | None = None
        successful_fetch = False
        source_summaries: list[dict[str, Any]] = []

        for index, url in enumerate(self._candidate_starting_lineups_urls(game_date), start=1):
            try:
                response = self.session.get(
                    url,
                    headers=self._starting_lineups_request_headers(),
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except requests.HTTPError as exc:
                last_http_error = exc
                summary = {
                    "requested_url": url,
                    "final_url": getattr(exc.response, "url", url),
                    "status_code": getattr(exc.response, "status_code", None),
                    "content_type": (exc.response.headers.get("content-type") if exc.response is not None else "") or "",
                    "card_count": 0,
                    "matchups": [],
                    "title": "",
                    "text_preview_lines": _extract_text_preview(exc.response.text if exc.response is not None else "", limit=20),
                }
                source_summaries.append(summary)
                self._debug_write_json(f"starting-lineups/{game_date}/page_{index:02d}_summary.json", summary)
                if exc.response is not None and exc.response.text:
                    self._debug_write_text(f"starting-lineups/{game_date}/page_{index:02d}.html", exc.response.text)
                continue

            successful_fetch = True
            cards = parse_starting_lineups_html(response.text)
            summary = {
                "requested_url": url,
                "final_url": response.url,
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "card_count": len(cards),
                "matchups": [f"{card.away_team} at {card.home_team}" for card in cards[:25]],
                "title": _extract_html_title(response.text),
                "text_preview_lines": _extract_text_preview(response.text, limit=20),
            }
            source_summaries.append(summary)
            self._debug_write_json(f"starting-lineups/{game_date}/page_{index:02d}_summary.json", summary)
            self._debug_write_text(f"starting-lineups/{game_date}/page_{index:02d}.html", response.text)

            for card in cards:
                key = (card.away_team_key, card.home_team_key)
                current = best_by_matchup.get(key)
                if current is None or lineup_card_score(card) > lineup_card_score(current):
                    best_by_matchup[key] = card

        resolved_cards = list(best_by_matchup.values())
        debug_payload = {
            "game_date": game_date,
            "successful_fetches": sum(1 for source in source_summaries if source.get("status_code") == 200),
            "total_cards": len(resolved_cards),
            "matchups": [f"{card.away_team} at {card.home_team}" for card in resolved_cards],
            "sources": source_summaries,
        }
        self._lineups_page_debug[game_date] = debug_payload
        self._debug_write_json(f"starting-lineups/{game_date}/summary.json", debug_payload)

        if resolved_cards:
            self._lineups_page_cache[game_date] = resolved_cards
            return resolved_cards

        if last_http_error is not None and not successful_fetch:
            raise last_http_error

        self._lineups_page_cache[game_date] = []
        return []

    @staticmethod
    def _find_lineup_card(cards: Iterable[LineupCard], *, away_team: str, home_team: str) -> LineupCard | None:
        away_key = canonical_team_key(away_team)
        home_key = canonical_team_key(home_team)
        for card in cards:
            if card.away_team_key == away_key and card.home_team_key == home_key:
                return card
        return None


# Backwards-compatible aliases for older imports.
SnoozleApiClient = MlbDataClient
SnoozleApiError = MlbDataError


def _iter_schedule_games(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for day in payload.get("dates", []) or []:
        for game in day.get("games", []) or []:
            if isinstance(game, dict):
                yield game


def _lookup(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _year_from_date_text(value: str) -> int:
    text = str(value or "")
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    raise MlbDataError(f"Could not infer season year from date {value!r}")


def _clean_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\xa0", " ")).strip()


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return 0.0
    if text.startswith("."):
        text = f"0{text}"
    return float(text)


def _innings_to_decimal(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    if "." not in text:
        return _safe_float(text)
    innings, partial = text.split(".", 1)
    return _safe_float(innings) + (_safe_float(partial) / 3.0)


def _slug_key(value: str) -> str:
    text = _clean_name(value).lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonical_team_key(team_name: str) -> str:
    slug = _slug_key(team_name)
    if slug in _TEAM_ALIASES:
        return _TEAM_ALIASES[slug]

    alias_checks = [
        ("athletics", "athletics"),
        ("diamondbacks", "diamondbacks"),
        ("d backs", "diamondbacks"),
        ("red sox", "redsox"),
        ("white sox", "whitesox"),
        ("blue jays", "bluejays"),
    ]
    for needle, canonical in alias_checks:
        if needle in slug:
            return canonical

    words = slug.split()
    if len(words) >= 2 and " ".join(words[-2:]) in _TEAM_ALIASES:
        return _TEAM_ALIASES[" ".join(words[-2:])]
    return words[-1] if words else slug


def schedule_game_team_name(game: dict[str, Any], side: str) -> str:
    return _clean_name(_lookup(game, "teams", side, "team", "name", default=""))


def schedule_game_team_id(game: dict[str, Any], side: str) -> int:
    team_id = _lookup(game, "teams", side, "team", "id", default=0)
    return int(team_id or 0)


def _schedule_game_to_matchup_payload(game: dict[str, Any]) -> dict[str, Any]:
    return {
        "gameId": str(game.get("gamePk", "")),
        "date": str(game.get("officialDate", "")),
        "awayTeam": schedule_game_team_name(game, "away"),
        "homeTeam": schedule_game_team_name(game, "home"),
    }


def extract_first_stat_split(payload: dict[str, Any]) -> dict[str, Any]:
    for stats_group in payload.get("stats", []) or []:
        splits = stats_group.get("splits", []) or []
        if not splits:
            continue
        stat = splits[0].get("stat")
        if isinstance(stat, dict):
            return stat
    return {}


def extract_first_people_stat_split(payload: dict[str, Any], player_id: int) -> dict[str, Any]:
    people = payload.get("people", []) or []
    for person in people:
        if int(person.get("id") or 0) != int(player_id):
            continue
        for stats_group in person.get("stats", []) or []:
            splits = stats_group.get("splits", []) or []
            if not splits:
                continue
            stat = splits[0].get("stat")
            if isinstance(stat, dict):
                return stat
    return {}


def lineup_card_score(card: LineupCard) -> tuple[int, int, int, int]:
    return (
        len(card.away_lineup) + len(card.home_lineup),
        int(card.away_pitcher is not None) + int(card.home_pitcher is not None),
        len(card.away_lineup),
        len(card.home_lineup),
    )


def _extract_html_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    return _clean_name(title)


def _extract_text_preview(text: str, *, limit: int = 20) -> list[str]:
    if not text:
        return []
    cleaned = [_clean_name(line) for line in text.splitlines()]
    cleaned = [line for line in cleaned if line]
    return cleaned[:limit]


def _batting_stats_are_complete(stats: dict[str, Any]) -> bool:
    required = ["hits", "doubles", "triples", "homeRuns", "baseOnBalls"]
    return bool(stats) and any(key in stats for key in ("plateAppearances", "atBats")) and all(key in stats for key in required)


def _pitching_stats_are_complete(stats: dict[str, Any]) -> bool:
    return bool(stats) and ("inningsPitched" in stats or "gamesStarted" in stats) and any(
        key in stats for key in ("obp", "hits", "baseOnBalls", "whip")
    )


def batter_stats_from_record(player_id: int, player_name: str, record: dict[str, Any]) -> BattingStats:
    hits = _safe_float(record.get("hits"))
    doubles = _safe_float(record.get("doubles"))
    triples = _safe_float(record.get("triples"))
    home_runs = _safe_float(record.get("homeRuns"))
    singles = max(hits - doubles - triples - home_runs, 0.0)
    at_bats = _safe_float(record.get("atBats"))
    walks = _safe_float(record.get("baseOnBalls"))
    hit_by_pitch = _safe_float(record.get("hitByPitch"))
    sac_flies = _safe_float(record.get("sacFlies"))
    strikeouts = _safe_float(record.get("strikeOuts"))
    grounded_into_double_plays = _safe_float(record.get("groundIntoDoublePlay"))
    stolen_bases = _safe_float(record.get("stolenBases"))
    plate_appearances = _safe_float(record.get("plateAppearances"))
    if plate_appearances <= 0.0:
        plate_appearances = at_bats + walks + hit_by_pitch + sac_flies
    on_base = _safe_float(record.get("obp"))
    if on_base <= 0.0 and plate_appearances > 0.0:
        on_base = (hits + walks + hit_by_pitch) / plate_appearances
    return BattingStats(
        player_id=int(player_id),
        player_name=_clean_name(player_name),
        bats=_normalize_batting_side(record.get("bats") or record.get("batSide")),
        on_base=on_base,
        hit_by_pitch=hit_by_pitch,
        sac_flies=sac_flies,
        at_bats=at_bats,
        hits=hits,
        doubles=doubles,
        triples=triples,
        home_runs=home_runs,
        singles=singles,
        walks=walks,
        strikeouts=strikeouts,
        grounded_into_double_plays=grounded_into_double_plays,
        stolen_bases=stolen_bases,
        total_plate_appearances=plate_appearances,
    )


def pitcher_stats_from_record(player_id: int, player_name: str, record: dict[str, Any]) -> PitchingStats:
    innings_pitched = _innings_to_decimal(record.get("inningsPitched"))
    hits = _safe_float(record.get("hits"))
    doubles_allowed = _safe_float(record.get("doubles"))
    triples_allowed = _safe_float(record.get("triples"))
    has_hit_type_detail = any(key in record for key in ("doubles", "triples", "2B", "3B"))
    walks = _safe_float(record.get("baseOnBalls"))
    hit_batsmen = _safe_float(record.get("hitBatsmen") or record.get("hitByPitch"))
    strikeouts = _safe_float(record.get("strikeOuts"))
    home_runs_allowed = _safe_float(record.get("homeRuns"))
    sac_flies = _safe_float(record.get("sacFlies"))
    batters_faced = _safe_float(record.get("battersFaced"))
    at_bats = _safe_float(record.get("atBats"))

    on_base = max(
        _safe_float(record.get("obp")),
        _safe_float(record.get("obpAgainst")),
        _safe_float(record.get("opponentOnBasePercentage")),
    )
    if on_base <= 0.0:
        denom = batters_faced or (at_bats + walks + hit_batsmen + sac_flies)
        if denom > 0.0:
            on_base = (hits + walks + hit_batsmen) / denom
    if on_base <= 0.0:
        whip = _safe_float(record.get("whip"))
        if whip > 0.0 and innings_pitched > 0.0:
            baserunners = whip * innings_pitched
            estimated_bf = batters_faced or ((innings_pitched * 2.82) + hits + walks)
            if estimated_bf > 0.0:
                on_base = baserunners / estimated_bf

    total_pitches = max(_safe_float(record.get("pitchesThrown")), _safe_float(record.get("numberOfPitches")))
    games_started = max(_safe_float(record.get("gamesStarted")), 0.0)
    average_pitches = 0.0
    if total_pitches > 0.0 and games_started > 0.0:
        average_pitches = total_pitches / games_started
    elif total_pitches > 0.0 and _safe_float(record.get("gamesPitched")) > 0.0:
        average_pitches = total_pitches / _safe_float(record.get("gamesPitched"))
    else:
        pitches_per_inning = max(_safe_float(record.get("pitchesPerInning")), _safe_float(record.get("pitchesThrownPerInning")))
        if pitches_per_inning > 0.0 and innings_pitched > 0.0 and games_started > 0.0:
            average_pitches = (pitches_per_inning * innings_pitched) / games_started

    return PitchingStats(
        player_id=int(player_id),
        player_name=_clean_name(player_name),
        throws=_normalize_throwing_hand(record.get("throws") or record.get("pitchHand")),
        on_base=on_base,
        average_pitches=average_pitches,
        games_started=games_started,
        innings_pitched=innings_pitched,
        hits_allowed=hits,
        doubles_allowed=doubles_allowed,
        triples_allowed=triples_allowed,
        has_hit_type_detail=has_hit_type_detail,
        walks_allowed=walks,
        hit_batsmen=hit_batsmen,
        strikeouts=strikeouts,
        home_runs_allowed=home_runs_allowed,
        batters_faced=batters_faced,
        at_bats_against=at_bats,
    )



def team_bullpen_stats_from_record(team_id: int, team_name: str, record: dict[str, Any]) -> PitchingStats:
    return pitcher_stats_from_record(-(90000 + int(team_id)), f"{_clean_name(team_name)} Bullpen", record)


def _team_canonical_from_text(value: str) -> str | None:
    slug = _slug_key(value)
    if not slug:
        return None
    best_match: tuple[int, str] | None = None
    for canonical, official in _OFFICIAL_TEAM_NAMES.items():
        variants = {
            _slug_key(official),
            _slug_key(official.split()[-1]),
            canonical,
        }
        if canonical == "diamondbacks":
            variants.update({_slug_key("Arizona D-backs"), _slug_key("D-backs")})
        if canonical == "athletics":
            variants.update({_slug_key("Oakland Athletics"), _slug_key("A's")})
        for variant in variants:
            if variant and variant in slug:
                score = len(variant)
                if best_match is None or score > best_match[0]:
                    best_match = (score, canonical)
    return best_match[1] if best_match is not None else None


def _parse_team_bullpen_stats_blob(value: str) -> dict[str, Any] | None:
    cleaned = _clean_name(str(value).replace("‌", " ").replace("​", " "))
    if not _TEAM_BULLPEN_ROW_RE.match(cleaned):
        return None

    match = re.match(r"^(AL|NL)\s+(.+)$", cleaned)
    if match is None:
        return None

    league = match.group(1)
    remainder = match.group(2)
    tokens = _TEAM_STATS_TOKEN_RE.findall(remainder)
    if len(tokens) < 19:
        return None
    tokens = tokens[:19]
    return {
        "league": league,
        "wins": int(float(tokens[0])),
        "losses": int(float(tokens[1])),
        "era": float(tokens[2]),
        "games": int(float(tokens[3])),
        "gamesStarted": int(float(tokens[4])),
        "completeGames": int(float(tokens[5])),
        "shutouts": int(float(tokens[6])),
        "saves": int(float(tokens[7])),
        "saveOpportunities": int(float(tokens[8])),
        "inningsPitched": tokens[9],
        "hits": float(tokens[10]),
        "runs": float(tokens[11]),
        "earnedRuns": float(tokens[12]),
        "homeRuns": float(tokens[13]),
        "hitBatsmen": float(tokens[14]),
        "baseOnBalls": float(tokens[15]),
        "strikeOuts": float(tokens[16]),
        "whip": float(tokens[17]),
        "avg": float(tokens[18]),
    }


def _team_bullpen_record_from_stats(team_name: str, stats_blob: dict[str, Any]) -> dict[str, Any]:
    innings_pitched = _innings_to_decimal(stats_blob["inningsPitched"])
    outs = innings_pitched * 3.0
    hits = float(stats_blob["hits"])
    walks = float(stats_blob["baseOnBalls"])
    hit_batsmen = float(stats_blob["hitBatsmen"])
    estimated_batters_faced = max(outs + hits + walks + hit_batsmen, 0.0)
    batting_average = float(stats_blob["avg"])
    at_bats = (hits / batting_average) if batting_average > 0.0 else max(estimated_batters_faced - walks - hit_batsmen, 0.0)
    on_base = _safe_divide(hits + walks + hit_batsmen, estimated_batters_faced)
    canonical = canonical_team_key(team_name)
    official_name = _OFFICIAL_TEAM_NAMES.get(canonical, team_name)
    return {
        "teamKey": canonical,
        "teamName": official_name,
        "league": stats_blob["league"],
        "obp": on_base,
        "whip": float(stats_blob["whip"]),
        "avg": float(stats_blob["avg"]),
        "era": float(stats_blob["era"]),
        "inningsPitched": stats_blob["inningsPitched"],
        "hits": hits,
        "runs": float(stats_blob["runs"]),
        "earnedRuns": float(stats_blob["earnedRuns"]),
        "homeRuns": float(stats_blob["homeRuns"]),
        "hitBatsmen": hit_batsmen,
        "baseOnBalls": walks,
        "strikeOuts": float(stats_blob["strikeOuts"]),
        "battersFaced": estimated_batters_faced,
        "atBats": at_bats,
        "gamesStarted": 0,
        "gamesPitched": float(stats_blob["games"]),
        "saves": int(stats_blob["saves"]),
        "saveOpportunities": int(stats_blob["saveOpportunities"]),
    }



def _extract_team_bullpen_table_header(cell: Any) -> str:
    abbr = cell.find("abbr")
    if abbr is not None:
        text = _clean_name(abbr.get_text(" ", strip=True))
    else:
        text = _clean_name(cell.get_text(" ", strip=True))
    return text.upper()



def _extract_team_bullpen_table_team_name(cell: Any) -> str:
    team_link = cell.find("a", attrs={"aria-label": True})
    if team_link is not None:
        label = _clean_name(team_link.get("aria-label"))
        if label:
            return label

    team_link = cell.find("a")
    if team_link is not None:
        team_text = _clean_name(team_link.get_text(" ", strip=True))
        canonical = _team_canonical_from_text(team_text)
        if canonical is not None:
            return _OFFICIAL_TEAM_NAMES.get(canonical, team_text)
        if team_text:
            return team_text

    text = _clean_name(cell.get_text(" ", strip=True).replace("‌", " ").replace("​", " "))
    canonical = _team_canonical_from_text(text)
    if canonical is not None:
        return _OFFICIAL_TEAM_NAMES.get(canonical, text)
    return text



def _parse_team_bullpen_stats_table(soup: BeautifulSoup) -> dict[str, dict[str, Any]]:
    required_headers = [
        "TEAM",
        "LEAGUE",
        "W",
        "L",
        "ERA",
        "G",
        "GS",
        "CG",
        "SHO",
        "SV",
        "SVO",
        "IP",
        "H",
        "R",
        "ER",
        "HR",
        "HB",
        "BB",
        "SO",
        "WHIP",
        "AVG",
    ]

    for table in soup.find_all("table"):
        header_row = None
        thead = table.find("thead")
        if thead is not None:
            header_rows = thead.find_all("tr", recursive=False)
            if header_rows:
                header_row = header_rows[-1]
        if header_row is None:
            header_row = table.find("tr")
        if header_row is None:
            continue

        header_cells = header_row.find_all(["th", "td"], recursive=False)
        headers = [_extract_team_bullpen_table_header(cell) for cell in header_cells]
        if headers[: len(required_headers)] != required_headers:
            continue

        tbody = table.find("tbody") or table
        records: dict[str, dict[str, Any]] = {}
        for row in tbody.find_all("tr", recursive=False):
            cells = row.find_all(["th", "td"], recursive=False)
            if len(cells) < len(required_headers):
                continue

            team_name = _extract_team_bullpen_table_team_name(cells[0])
            if not team_name:
                continue

            league = _clean_name(cells[1].get_text(" ", strip=True)).upper()
            if league not in {"AL", "NL"}:
                continue

            values = {headers[index]: _clean_name(cells[index].get_text(" ", strip=True)) for index in range(len(required_headers))}
            values["TEAM"] = team_name
            stats_blob = {
                "league": values["LEAGUE"],
                "wins": int(float(values["W"])),
                "losses": int(float(values["L"])),
                "era": _safe_float(values["ERA"]),
                "games": int(float(values["G"])),
                "gamesStarted": int(float(values["GS"])),
                "completeGames": int(float(values["CG"])),
                "shutouts": int(float(values["SHO"])),
                "saves": int(float(values["SV"])),
                "saveOpportunities": int(float(values["SVO"])),
                "inningsPitched": values["IP"],
                "hits": _safe_float(values["H"]),
                "runs": _safe_float(values["R"]),
                "earnedRuns": _safe_float(values["ER"]),
                "homeRuns": _safe_float(values["HR"]),
                "hitBatsmen": _safe_float(values["HB"]),
                "baseOnBalls": _safe_float(values["BB"]),
                "strikeOuts": _safe_float(values["SO"]),
                "whip": _safe_float(values["WHIP"]),
                "avg": _safe_float(values["AVG"]),
            }
            record = _team_bullpen_record_from_stats(team_name, stats_blob)
            records[record["teamKey"]] = record

        if records:
            return records

    return {}



def parse_team_bullpen_stats_html(html: str) -> dict[str, dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")

    table_records = _parse_team_bullpen_stats_table(soup)
    if table_records:
        return table_records

    lines = [_clean_name(line) for line in soup.get_text("\n").splitlines()]
    lines = [line for line in lines if line]

    table_start = 0
    for index in range(max(len(lines) - 3, 0)):
        if lines[index] == "TEAM" and lines[index + 1] == "TEAM" and "LEAGUE" in lines[index + 2:index + 6]:
            table_start = index
            break
    lines_to_scan = lines[table_start:]

    records: dict[str, dict[str, Any]] = {}
    for index, line in enumerate(lines_to_scan):
        canonical = _team_canonical_from_text(line)
        if canonical is None:
            continue
        stats_blob: dict[str, Any] | None = None
        for offset in range(1, 8):
            if index + offset >= len(lines_to_scan):
                break
            stats_blob = _parse_team_bullpen_stats_blob(lines_to_scan[index + offset])
            if stats_blob is not None:
                break
        if stats_blob is None:
            continue
        record = _team_bullpen_record_from_stats(_OFFICIAL_TEAM_NAMES.get(canonical, line), stats_blob)
        records[record["teamKey"]] = record
    return records


def team_bullpen_current_cache_path(cache_dir: Path, season: int) -> Path:
    return cache_dir / "bullpen" / "current" / f"{int(season)}.json"


def team_bullpen_season_cache_path(cache_dir: Path, season: int) -> Path:
    return cache_dir / "bullpen" / "seasons" / f"{int(season)}.json"


def read_team_bullpen_cache_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def write_team_bullpen_cache_file(path: Path, *, season: int, kind: str, records: dict[str, dict[str, Any]]) -> None:
    payload = {
        "version": _TEAM_BULLPEN_CACHE_FORMAT_VERSION,
        "season": int(season),
        "kind": kind,
        "fetched_on": date.today().isoformat(),
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def default_player_cache_dir() -> Path:
    return Path.cwd() / _DEFAULT_PLAYER_CACHE_DIRNAME


def player_current_cache_path(cache_dir: Path, group: str, player_id: int) -> Path:
    return cache_dir / "current" / group / f"{int(player_id)}.json"


def player_season_cache_path(cache_dir: Path, group: str, season: int, player_id: int) -> Path:
    return cache_dir / "seasons" / group / str(int(season)) / f"{int(player_id)}.json"


def player_profile_cache_path(cache_dir: Path, group: str, season: int, player_id: int) -> Path:
    return cache_dir / "profiles" / group / str(int(season)) / f"{int(player_id)}.json"


# Backward-compatible alias for older imports/tests.
def player_prior_cache_path(cache_dir: Path, group: str, season: int, player_id: int) -> Path:
    return player_profile_cache_path(cache_dir, group, season, player_id)


def read_player_cache_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def write_player_cache_file(
    path: Path,
    *,
    player_id: int,
    player_name: str = "",
    group: str,
    season: int,
    kind: str,
    stat_record: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "version": _PLAYER_CACHE_FORMAT_VERSION,
        "player_id": int(player_id),
        "player_name": _clean_name(player_name),
        "group": group,
        "season": int(season),
        "kind": kind,
        "fetched_on": date.today().isoformat(),
        "stat_record": stat_record,
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def weighted_average_batting_stats(
    player_id: int,
    player_name: str,
    weighted_stats: list[tuple[BattingStats, float]],
) -> BattingStats:
    total_weight = sum(weight for _, weight in weighted_stats if weight > 0.0)
    if total_weight <= 0.0:
        return BattingStats(player_id=player_id, player_name=player_name)

    def avg(attr: str) -> float:
        return sum(getattr(stats, attr) * weight for stats, weight in weighted_stats) / total_weight

    averaged = BattingStats(
        player_id=int(player_id),
        player_name=_clean_name(player_name),
        on_base=0.0,
        hit_by_pitch=avg("hit_by_pitch"),
        sac_flies=avg("sac_flies"),
        at_bats=avg("at_bats"),
        hits=avg("hits"),
        doubles=avg("doubles"),
        triples=avg("triples"),
        home_runs=avg("home_runs"),
        singles=avg("singles"),
        walks=avg("walks"),
        strikeouts=avg("strikeouts"),
        grounded_into_double_plays=avg("grounded_into_double_plays"),
        stolen_bases=avg("stolen_bases"),
        total_plate_appearances=avg("total_plate_appearances"),
    )
    if averaged.total_plate_appearances > 0.0:
        averaged.on_base = (averaged.hits + averaged.walks + averaged.hit_by_pitch) / averaged.total_plate_appearances
    return averaged


def weighted_average_pitching_stats(
    player_id: int,
    player_name: str,
    weighted_stats: list[tuple[PitchingStats, float]],
) -> PitchingStats:
    total_weight = sum(weight for _, weight in weighted_stats if weight > 0.0)
    if total_weight <= 0.0:
        return PitchingStats(player_id=player_id, player_name=player_name)

    def avg(attr: str) -> float:
        return sum(getattr(stats, attr) * weight for stats, weight in weighted_stats) / total_weight

    averaged = PitchingStats(
        player_id=int(player_id),
        player_name=_clean_name(player_name),
        on_base=0.0,
        average_pitches=avg("average_pitches"),
        games_started=avg("games_started"),
        innings_pitched=avg("innings_pitched"),
        hits_allowed=avg("hits_allowed"),
        doubles_allowed=avg("doubles_allowed"),
        triples_allowed=avg("triples_allowed"),
        has_hit_type_detail=any(stats.has_hit_type_detail for stats, _ in weighted_stats),
        walks_allowed=avg("walks_allowed"),
        hit_batsmen=avg("hit_batsmen"),
        strikeouts=avg("strikeouts"),
        home_runs_allowed=avg("home_runs_allowed"),
        batters_faced=avg("batters_faced"),
        at_bats_against=avg("at_bats_against"),
    )
    denom = averaged.batters_faced or (averaged.at_bats_against + averaged.walks_allowed + averaged.hit_batsmen)
    if denom > 0.0:
        averaged.on_base = (averaged.hits_allowed + averaged.walks_allowed + averaged.hit_batsmen) / denom
    return averaged


def combine_batting_stats_with_prior(current: BattingStats, prior: BattingStats | None) -> BattingStats:
    if prior is None or (prior.total_plate_appearances <= 0.0 and prior.at_bats <= 0.0):
        return current
    combined = BattingStats(
        player_id=current.player_id,
        player_name=current.player_name,
        on_base=0.0,
        hit_by_pitch=current.hit_by_pitch + prior.hit_by_pitch,
        sac_flies=current.sac_flies + prior.sac_flies,
        at_bats=current.at_bats + prior.at_bats,
        hits=current.hits + prior.hits,
        doubles=current.doubles + prior.doubles,
        triples=current.triples + prior.triples,
        home_runs=current.home_runs + prior.home_runs,
        singles=current.singles + prior.singles,
        walks=current.walks + prior.walks,
        strikeouts=current.strikeouts + prior.strikeouts,
        grounded_into_double_plays=current.grounded_into_double_plays + prior.grounded_into_double_plays,
        stolen_bases=current.stolen_bases + prior.stolen_bases,
        total_plate_appearances=current.total_plate_appearances + prior.total_plate_appearances,
    )
    if combined.total_plate_appearances > 0.0:
        combined.on_base = (combined.hits + combined.walks + combined.hit_by_pitch) / combined.total_plate_appearances
    return combined


def combine_pitching_stats_with_prior(current: PitchingStats, prior: PitchingStats | None) -> PitchingStats:
    if prior is None or (prior.estimated_batters_faced <= 0.0 and prior.innings_pitched <= 0.0):
        return current
    weight_current = max(current.estimated_batters_faced, current.innings_pitched * 4.25, 1.0)
    weight_prior = max(prior.estimated_batters_faced, prior.innings_pitched * 4.25, 1.0)
    combined = PitchingStats(
        player_id=current.player_id,
        player_name=current.player_name,
        on_base=0.0,
        average_pitches=((current.average_pitches * weight_current) + (prior.average_pitches * weight_prior)) / (weight_current + weight_prior),
        games_started=current.games_started + prior.games_started,
        innings_pitched=current.innings_pitched + prior.innings_pitched,
        hits_allowed=current.hits_allowed + prior.hits_allowed,
        doubles_allowed=current.doubles_allowed + prior.doubles_allowed,
        triples_allowed=current.triples_allowed + prior.triples_allowed,
        has_hit_type_detail=current.has_hit_type_detail or prior.has_hit_type_detail,
        walks_allowed=current.walks_allowed + prior.walks_allowed,
        hit_batsmen=current.hit_batsmen + prior.hit_batsmen,
        strikeouts=current.strikeouts + prior.strikeouts,
        home_runs_allowed=current.home_runs_allowed + prior.home_runs_allowed,
        batters_faced=current.batters_faced + prior.batters_faced,
        at_bats_against=current.at_bats_against + prior.at_bats_against,
    )
    denom = combined.batters_faced or (combined.at_bats_against + combined.walks_allowed + combined.hit_batsmen)
    if denom > 0.0:
        combined.on_base = (combined.hits_allowed + combined.walks_allowed + combined.hit_batsmen) / denom
    return combined


def batting_stats_to_record(stats: BattingStats) -> dict[str, Any]:
    return {
        "obp": stats.on_base,
        "hitByPitch": stats.hit_by_pitch,
        "sacFlies": stats.sac_flies,
        "atBats": stats.at_bats,
        "hits": stats.hits,
        "doubles": stats.doubles,
        "triples": stats.triples,
        "homeRuns": stats.home_runs,
        "baseOnBalls": stats.walks,
        "strikeOuts": stats.strikeouts,
        "groundIntoDoublePlay": stats.grounded_into_double_plays,
        "stolenBases": stats.stolen_bases,
        "plateAppearances": stats.total_plate_appearances,
    }


def pitching_stats_to_record(stats: PitchingStats) -> dict[str, Any]:
    return {
        "obp": stats.on_base,
        "inningsPitched": stats.innings_pitched,
        "hits": stats.hits_allowed,
        "doubles": stats.doubles_allowed,
        "triples": stats.triples_allowed,
        "baseOnBalls": stats.walks_allowed,
        "hitBatsmen": stats.hit_batsmen,
        "strikeOuts": stats.strikeouts,
        "homeRuns": stats.home_runs_allowed,
        "battersFaced": stats.batters_faced,
        "atBats": stats.at_bats_against,
        "pitchesThrown": stats.average_pitches,
    }


def _weighted_rate_and_sample(weighted_values: list[tuple[float, float, float]], *, fallback_rate: float, fallback_sample: float) -> tuple[float, float]:
    weighted_events = 0.0
    weighted_sample = 0.0
    total_weight = 0.0
    for events, sample, weight in weighted_values:
        if sample <= 0.0 or weight <= 0.0:
            continue
        weighted_events += max(events, 0.0) * weight
        weighted_sample += sample * weight
        total_weight += weight
    if weighted_sample <= 0.0 or total_weight <= 0.0:
        return max(fallback_rate, 0.0), max(fallback_sample, 0.0)
    return weighted_events / weighted_sample, weighted_sample


def _weighted_average_value(weighted_values: list[tuple[float, float]], *, fallback_value: float) -> float:
    numerator = 0.0
    denominator = 0.0
    for value, weight in weighted_values:
        if weight <= 0.0:
            continue
        numerator += value * weight
        denominator += weight
    if denominator <= 0.0:
        return fallback_value
    return numerator / denominator


def build_resolved_batting_profile(
    *,
    current: BattingStats,
    historical: list[tuple[BattingStats, float]],
    league: LeagueAverages,
) -> ResolvedBattingProfile:
    current_pa = max(current.total_plate_appearances, 0.0)
    prior_walk_rate, prior_walk_sample = _weighted_rate_and_sample(
        [(stats.walks, stats.total_plate_appearances, weight) for stats, weight in historical],
        fallback_rate=league.walk_rate_per_pa,
        fallback_sample=_HITTER_PRIOR_PA_CAPS["walk"],
    )
    prior_hbp_rate, prior_hbp_sample = _weighted_rate_and_sample(
        [(stats.hit_by_pitch, stats.total_plate_appearances, weight) for stats, weight in historical],
        fallback_rate=league.hit_by_pitch_rate_per_pa,
        fallback_sample=_HITTER_PRIOR_PA_CAPS["hit_by_pitch"],
    )
    prior_strikeout_rate, prior_strikeout_sample = _weighted_rate_and_sample(
        [(stats.strikeouts, stats.total_plate_appearances, weight) for stats, weight in historical],
        fallback_rate=league.strikeout_rate_per_pa,
        fallback_sample=_HITTER_PRIOR_PA_CAPS["strikeout"],
    )
    prior_home_run_rate, prior_home_run_sample = _weighted_rate_and_sample(
        [(stats.home_runs, stats.total_plate_appearances, weight) for stats, weight in historical],
        fallback_rate=league.home_run_rate_per_pa,
        fallback_sample=_HITTER_PRIOR_PA_CAPS["home_run"],
    )
    prior_non_hr_hit_rate, prior_non_hr_hit_sample = _weighted_rate_and_sample(
        [(stats.non_home_run_hits, stats.total_plate_appearances, weight) for stats, weight in historical],
        fallback_rate=league.non_home_run_hit_rate_per_pa,
        fallback_sample=_HITTER_PRIOR_PA_CAPS["non_home_run_hit"],
    )

    walk_rate = _smoothed_rate_from_counts(current.walks, current_pa, prior_rate=prior_walk_rate, prior_sample=_cap_prior_sample(prior_walk_sample, _HITTER_PRIOR_PA_CAPS["walk"]))
    hit_by_pitch_rate = _smoothed_rate_from_counts(current.hit_by_pitch, current_pa, prior_rate=prior_hbp_rate, prior_sample=_cap_prior_sample(prior_hbp_sample, _HITTER_PRIOR_PA_CAPS["hit_by_pitch"]))
    strikeout_rate = _smoothed_rate_from_counts(current.strikeouts, current_pa, prior_rate=prior_strikeout_rate, prior_sample=_cap_prior_sample(prior_strikeout_sample, _HITTER_PRIOR_PA_CAPS["strikeout"]))
    home_run_rate = _smoothed_rate_from_counts(current.home_runs, current_pa, prior_rate=prior_home_run_rate, prior_sample=_cap_prior_sample(prior_home_run_sample, _HITTER_PRIOR_PA_CAPS["home_run"]))
    non_home_run_hit_rate = _smoothed_rate_from_counts(current.non_home_run_hits, current_pa, prior_rate=prior_non_hr_hit_rate, prior_sample=_cap_prior_sample(prior_non_hr_hit_sample, _HITTER_PRIOR_PA_CAPS["non_home_run_hit"]))

    current_non_hr_hits = max(current.non_home_run_hits, 0.0)
    prior_single_share, prior_non_hr_sample = _weighted_rate_and_sample(
        [(stats.singles, stats.non_home_run_hits, weight) for stats, weight in historical],
        fallback_rate=league.single_share_of_non_home_run_hits,
        fallback_sample=_HITTER_PRIOR_NON_HOME_RUN_HIT_CAP,
    )
    prior_double_share, prior_double_sample = _weighted_rate_and_sample(
        [(stats.doubles, stats.non_home_run_hits, weight) for stats, weight in historical],
        fallback_rate=league.double_share_of_non_home_run_hits,
        fallback_sample=_HITTER_PRIOR_NON_HOME_RUN_HIT_CAP,
    )
    prior_triple_share, prior_triple_sample = _weighted_rate_and_sample(
        [(stats.triples, stats.non_home_run_hits, weight) for stats, weight in historical],
        fallback_rate=league.triple_share_of_non_home_run_hits,
        fallback_sample=_HITTER_PRIOR_NON_HOME_RUN_HIT_CAP,
    )
    single_share = _smoothed_rate_from_counts(current.singles, current_non_hr_hits, prior_rate=prior_single_share, prior_sample=_cap_prior_sample(prior_non_hr_sample, _HITTER_PRIOR_NON_HOME_RUN_HIT_CAP))
    double_share = _smoothed_rate_from_counts(current.doubles, current_non_hr_hits, prior_rate=prior_double_share, prior_sample=_cap_prior_sample(prior_double_sample, _HITTER_PRIOR_NON_HOME_RUN_HIT_CAP))
    triple_share = _smoothed_rate_from_counts(current.triples, current_non_hr_hits, prior_rate=prior_triple_share, prior_sample=_cap_prior_sample(prior_triple_sample, _HITTER_PRIOR_NON_HOME_RUN_HIT_CAP))
    share_total = max(single_share + double_share + triple_share, 1e-6)
    single_share /= share_total
    double_share /= share_total
    triple_share /= share_total

    current_bip_outs = max(current.estimated_balls_in_play_outs, 0.0)
    prior_gidp_rate, prior_bip_outs = _weighted_rate_and_sample(
        [(stats.grounded_into_double_plays, stats.estimated_balls_in_play_outs, weight) for stats, weight in historical],
        fallback_rate=league.double_play_rate_on_in_play_out,
        fallback_sample=_HITTER_PRIOR_BIP_OUT_CAP,
    )
    prior_sac_fly_rate, prior_sac_fly_sample = _weighted_rate_and_sample(
        [(stats.sac_flies, stats.estimated_balls_in_play_outs, weight) for stats, weight in historical],
        fallback_rate=league.sac_fly_rate_on_in_play_out,
        fallback_sample=_HITTER_PRIOR_BIP_OUT_CAP,
    )
    capped_bip_prior = _cap_prior_sample(prior_bip_outs, _HITTER_PRIOR_BIP_OUT_CAP)
    double_play_rate = _smoothed_rate_from_counts(
        current.grounded_into_double_plays,
        current_bip_outs,
        prior_rate=prior_gidp_rate,
        prior_sample=capped_bip_prior,
    )
    sac_fly_rate = _smoothed_rate_from_counts(
        current.sac_flies,
        current_bip_outs,
        prior_rate=prior_sac_fly_rate,
        prior_sample=_cap_prior_sample(prior_sac_fly_sample, _HITTER_PRIOR_BIP_OUT_CAP),
    )

    prior_sb_rate, prior_sb_sample = _weighted_rate_and_sample(
        [(stats.stolen_bases, stats.total_plate_appearances, weight) for stats, weight in historical],
        fallback_rate=5.0 / 600.0,
        fallback_sample=_HITTER_PRIOR_PA_CAPS["stolen_bases"],
    )
    stolen_bases_rate = _smoothed_rate_from_counts(
        current.stolen_bases,
        current_pa,
        prior_rate=prior_sb_rate,
        prior_sample=_cap_prior_sample(prior_sb_sample, _HITTER_PRIOR_PA_CAPS["stolen_bases"]),
    )
    effective_pa = current_pa + max(
        _cap_prior_sample(prior_walk_sample, _HITTER_PRIOR_PA_CAPS["walk"]),
        _cap_prior_sample(prior_home_run_sample, _HITTER_PRIOR_PA_CAPS["home_run"]),
        _cap_prior_sample(prior_non_hr_hit_sample, _HITTER_PRIOR_PA_CAPS["non_home_run_hit"]),
        _cap_prior_sample(prior_strikeout_sample, _HITTER_PRIOR_PA_CAPS["strikeout"]),
    )

    on_base_rate = walk_rate + hit_by_pitch_rate + home_run_rate + non_home_run_hit_rate

    return ResolvedBattingProfile(
        effective_plate_appearances=effective_pa,
        effective_non_home_run_hits=current_non_hr_hits + _cap_prior_sample(prior_non_hr_sample, _HITTER_PRIOR_NON_HOME_RUN_HIT_CAP),
        effective_balls_in_play_outs=current_bip_outs + capped_bip_prior,
        on_base_rate=on_base_rate,
        walk_rate=walk_rate,
        hit_by_pitch_rate=hit_by_pitch_rate,
        strikeout_rate=strikeout_rate,
        home_run_rate=home_run_rate,
        non_home_run_hit_rate=non_home_run_hit_rate,
        single_share=single_share,
        double_share=double_share,
        triple_share=triple_share,
        double_play_rate_on_in_play_out=double_play_rate,
        sac_fly_rate_on_in_play_out=sac_fly_rate,
        stolen_bases_per_600_pa=max(stolen_bases_rate * 600.0, 0.0),
    )


def build_resolved_pitching_profile(
    *,
    current: PitchingStats,
    historical: list[tuple[PitchingStats, float]],
    league: LeagueAverages,
) -> ResolvedPitchingProfile:
    current_bf = max(current.estimated_batters_faced, 0.0)
    prior_walk_rate, prior_walk_sample = _weighted_rate_and_sample(
        [(stats.walks_allowed, stats.estimated_batters_faced, weight) for stats, weight in historical],
        fallback_rate=league.walk_rate_per_pa,
        fallback_sample=_PITCHER_PRIOR_BF_CAPS["walk"],
    )
    prior_hbp_rate, prior_hbp_sample = _weighted_rate_and_sample(
        [(stats.hit_batsmen, stats.estimated_batters_faced, weight) for stats, weight in historical],
        fallback_rate=league.hit_by_pitch_rate_per_pa,
        fallback_sample=_PITCHER_PRIOR_BF_CAPS["hit_by_pitch"],
    )
    prior_strikeout_rate, prior_strikeout_sample = _weighted_rate_and_sample(
        [(stats.strikeouts, stats.estimated_batters_faced, weight) for stats, weight in historical],
        fallback_rate=league.strikeout_rate_per_pa,
        fallback_sample=_PITCHER_PRIOR_BF_CAPS["strikeout"],
    )
    prior_home_run_rate, prior_home_run_sample = _weighted_rate_and_sample(
        [(stats.home_runs_allowed, stats.estimated_batters_faced, weight) for stats, weight in historical],
        fallback_rate=league.home_run_rate_per_pa,
        fallback_sample=_PITCHER_PRIOR_BF_CAPS["home_run"],
    )
    prior_non_hr_hit_rate, prior_non_hr_hit_sample = _weighted_rate_and_sample(
        [(stats.non_home_run_hits_allowed, stats.estimated_batters_faced, weight) for stats, weight in historical],
        fallback_rate=league.non_home_run_hit_rate_per_pa,
        fallback_sample=_PITCHER_PRIOR_BF_CAPS["non_home_run_hit"],
    )

    hit_type_prior_cap = max(league.non_home_run_hit_rate_per_pa * _PITCHER_PRIOR_BF_CAPS["non_home_run_hit"], 1.0)
    current_non_hr_hits_allowed = current.non_home_run_hits_allowed
    current_hit_type_sample = current_non_hr_hits_allowed if current.has_hit_type_detail else 0.0
    prior_single_share, prior_single_share_sample = _weighted_rate_and_sample(
        [(stats.singles_allowed, stats.non_home_run_hits_allowed, weight) for stats, weight in historical if stats.has_hit_type_detail],
        fallback_rate=league.single_share_of_non_home_run_hits,
        fallback_sample=hit_type_prior_cap,
    )
    prior_double_share, prior_double_share_sample = _weighted_rate_and_sample(
        [(stats.doubles_allowed, stats.non_home_run_hits_allowed, weight) for stats, weight in historical if stats.has_hit_type_detail],
        fallback_rate=league.double_share_of_non_home_run_hits,
        fallback_sample=hit_type_prior_cap,
    )
    prior_triple_share, prior_triple_share_sample = _weighted_rate_and_sample(
        [(stats.triples_allowed, stats.non_home_run_hits_allowed, weight) for stats, weight in historical if stats.has_hit_type_detail],
        fallback_rate=league.triple_share_of_non_home_run_hits,
        fallback_sample=hit_type_prior_cap,
    )

    prior_pitches = _weighted_average_value(
        [(stats.average_pitches, weight) for stats, weight in historical if stats.average_pitches > 0.0],
        fallback_value=league.average_pitches_per_start,
    )
    current_bf_per_start = _safe_divide(current_bf, current.games_started)
    prior_bf_per_start = _weighted_average_value(
        [(_safe_divide(stats.estimated_batters_faced, stats.games_started), weight) for stats, weight in historical if stats.games_started > 0.0],
        fallback_value=24.0,
    )
    current_starts = max(current.games_started, 0.0)
    prior_starts = max(sum(max(stats.games_started, 0.0) * weight for stats, weight in historical) / max(sum(weight for _, weight in historical if weight > 0.0), 1.0), 0.0)

    walk_rate_allowed = _smoothed_rate_from_counts(current.walks_allowed, current_bf, prior_rate=prior_walk_rate, prior_sample=_cap_prior_sample(prior_walk_sample, _PITCHER_PRIOR_BF_CAPS["walk"]))
    hit_by_pitch_rate_allowed = _smoothed_rate_from_counts(current.hit_batsmen, current_bf, prior_rate=prior_hbp_rate, prior_sample=_cap_prior_sample(prior_hbp_sample, _PITCHER_PRIOR_BF_CAPS["hit_by_pitch"]))
    strikeout_rate = _smoothed_rate_from_counts(current.strikeouts, current_bf, prior_rate=prior_strikeout_rate, prior_sample=_cap_prior_sample(prior_strikeout_sample, _PITCHER_PRIOR_BF_CAPS["strikeout"]))
    home_run_rate_allowed = _smoothed_rate_from_counts(current.home_runs_allowed, current_bf, prior_rate=prior_home_run_rate, prior_sample=_cap_prior_sample(prior_home_run_sample, _PITCHER_PRIOR_BF_CAPS["home_run"]))
    non_home_run_hit_rate_allowed = _smoothed_rate_from_counts(
        current_non_hr_hits_allowed,
        current_bf,
        prior_rate=prior_non_hr_hit_rate,
        prior_sample=_cap_prior_sample(prior_non_hr_hit_sample, _PITCHER_PRIOR_BF_CAPS["non_home_run_hit"]),
    )

    single_share_allowed = _smoothed_rate_from_counts(
        current.singles_allowed,
        current_hit_type_sample,
        prior_rate=prior_single_share,
        prior_sample=_cap_prior_sample(prior_single_share_sample, hit_type_prior_cap),
    )
    double_share_allowed = _smoothed_rate_from_counts(
        current.doubles_allowed,
        current_hit_type_sample,
        prior_rate=prior_double_share,
        prior_sample=_cap_prior_sample(prior_double_share_sample, hit_type_prior_cap),
    )
    triple_share_allowed = _smoothed_rate_from_counts(
        current.triples_allowed,
        current_hit_type_sample,
        prior_rate=prior_triple_share,
        prior_sample=_cap_prior_sample(prior_triple_share_sample, hit_type_prior_cap),
    )
    share_total = max(single_share_allowed + double_share_allowed + triple_share_allowed, 1e-6)
    single_share_allowed /= share_total
    double_share_allowed /= share_total
    triple_share_allowed /= share_total

    average_pitches = _weighted_average_value(
        [
            (current.average_pitches, max(current_bf, 1.0)) if current.average_pitches > 0.0 else (0.0, 0.0),
            (prior_pitches, max(sum(weight for _, weight in historical if weight > 0.0), 1.0)),
        ],
        fallback_value=league.average_pitches_per_start,
    )
    average_batters_faced_per_start = _weighted_average_value(
        [
            (current_bf_per_start, min(current_starts, _PITCHER_PRIOR_STARTS_CAP)) if current_bf_per_start > 0.0 else (0.0, 0.0),
            (prior_bf_per_start, min(prior_starts, _PITCHER_PRIOR_STARTS_CAP)) if prior_bf_per_start > 0.0 else (0.0, 0.0),
        ],
        fallback_value=24.0,
    )
    on_base_allowed_rate = walk_rate_allowed + hit_by_pitch_rate_allowed + home_run_rate_allowed + non_home_run_hit_rate_allowed
    effective_bf = current_bf + max(
        _cap_prior_sample(prior_walk_sample, _PITCHER_PRIOR_BF_CAPS["walk"]),
        _cap_prior_sample(prior_home_run_sample, _PITCHER_PRIOR_BF_CAPS["home_run"]),
        _cap_prior_sample(prior_non_hr_hit_sample, _PITCHER_PRIOR_BF_CAPS["non_home_run_hit"]),
        _cap_prior_sample(prior_strikeout_sample, _PITCHER_PRIOR_BF_CAPS["strikeout"]),
    )

    return ResolvedPitchingProfile(
        effective_batters_faced=effective_bf,
        on_base_allowed_rate=on_base_allowed_rate,
        walk_rate_allowed=walk_rate_allowed,
        hit_by_pitch_rate_allowed=hit_by_pitch_rate_allowed,
        strikeout_rate=strikeout_rate,
        home_run_rate_allowed=home_run_rate_allowed,
        non_home_run_hit_rate_allowed=non_home_run_hit_rate_allowed,
        single_share_allowed=single_share_allowed,
        double_share_allowed=double_share_allowed,
        triple_share_allowed=triple_share_allowed,
        average_pitches=average_pitches,
        average_batters_faced_per_start=average_batters_faced_per_start,
    )


def parse_starting_lineups_html(html: str) -> list[LineupCard]:
    soup = BeautifulSoup(html, "html.parser")

    dom_cards = _parse_starting_lineups_from_dom(soup)
    if dom_cards:
        return dom_cards

    return _parse_starting_lineups_from_text_fallback(soup)


def _parse_starting_lineups_from_dom(soup: BeautifulSoup) -> list[LineupCard]:
    cards: list[LineupCard] = []

    for matchup in soup.select("div.starting-lineups__matchup"):
        away_anchor = matchup.select_one(
            ".starting-lineups__team-name--away .starting-lineups__team-name--link"
        )
        home_anchor = matchup.select_one(
            ".starting-lineups__team-name--home .starting-lineups__team-name--link"
        )
        if away_anchor is None or home_anchor is None:
            continue

        away_team = _clean_name(away_anchor.get_text(" ", strip=True))
        home_team = _clean_name(home_anchor.get_text(" ", strip=True))
        if not away_team or not home_team:
            continue

        pitchers: list[LineupPlayer] = []
        for anchor in matchup.select("a.starting-lineups__pitcher--link[href]"):
            player = _lineup_player_from_anchor(anchor, context_text=anchor.parent.get_text(" ", strip=True) if anchor.parent is not None else "")
            if player is not None and not player.throws:
                player.throws = _extract_throwing_hand(matchup.get_text(" ", strip=True))
            if player is None:
                continue
            if any(existing.player_id == player.player_id for existing in pitchers):
                continue
            pitchers.append(player)
            if len(pitchers) == 2:
                break

        best_away: list[LineupPlayer] = []
        best_home: list[LineupPlayer] = []
        for teams_block in matchup.select("div.starting-lineups__teams"):
            away_ol = teams_block.select_one("ol.starting-lineups__team--away")
            home_ol = teams_block.select_one("ol.starting-lineups__team--home")
            away_lineup = _extract_lineup_from_ol(away_ol)
            home_lineup = _extract_lineup_from_ol(home_ol)
            if (len(away_lineup) + len(home_lineup)) > (len(best_away) + len(best_home)):
                best_away = away_lineup
                best_home = home_lineup

        cards.append(
            LineupCard(
                away_team=away_team,
                home_team=home_team,
                away_lineup=best_away,
                home_lineup=best_home,
                away_pitcher=pitchers[0] if len(pitchers) >= 1 else None,
                home_pitcher=pitchers[1] if len(pitchers) >= 2 else None,
            )
        )

    return cards


def _lineup_player_from_anchor(anchor: Any, *, context_text: str = "") -> LineupPlayer | None:
    href = anchor.get("href") or ""
    match = _PLAYER_ID_RE.search(href)
    if not match:
        return None
    name = _clean_name(anchor.get_text(" ", strip=True))
    if not name:
        return None
    surrounding_text = f"{context_text} {anchor.get_text(' ', strip=True)}"
    return LineupPlayer(
        int(match.group(1)),
        name,
        bats=_extract_batting_side(surrounding_text),
        throws=_extract_throwing_hand(surrounding_text),
    )


def _extract_lineup_from_ol(ol: Any) -> list[LineupPlayer]:
    if ol is None:
        return []

    lineup: list[LineupPlayer] = []
    for item in ol.find_all("li", recursive=False):
        anchor = item.select_one("a.starting-lineups__player--link[href]")
        if anchor is not None:
            player = _lineup_player_from_anchor(anchor, context_text=item.get_text(" ", strip=True))
            if player is not None:
                lineup.append(player)
        else:
            text = _clean_name(item.get_text(" ", strip=True))
            if text:
                name = re.sub(r"\s+\([LRS]\).*$", "", text).strip()
                lineup.append(LineupPlayer(0, name, bats=_extract_batting_side(text)))
        if len(lineup) >= 9:
            break

    return [player for player in lineup if player.player_id > 0]


def _parse_starting_lineups_from_text_fallback(soup: BeautifulSoup) -> list[LineupCard]:
    player_ids_by_name: dict[str, int] = {}
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href") or ""
        match = _PLAYER_ID_RE.search(href)
        if not match:
            continue
        name = _clean_name(anchor.get_text(" ", strip=True))
        if not name:
            continue
        player_ids_by_name.setdefault(name, int(match.group(1)))

    lines = [_clean_name(line) for line in soup.get_text("\n").splitlines()]
    lines = [line for line in lines if line]

    cards: list[LineupCard] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if " @ " not in line:
            index += 1
            continue
        away_team, home_team = [part.strip() for part in line.split(" @ ", 1)]

        first_label = None
        for probe in range(index + 1, len(lines)):
            if _LINEUP_LABEL_RE.match(lines[probe]):
                first_label = probe
                break
            if " @ " in lines[probe]:
                break
        if first_label is None:
            index += 1
            continue

        probable_pitchers: list[LineupPlayer] = []
        probable_section = lines[index + 1:first_label]
        for candidate_index, candidate in enumerate(probable_section):
            if candidate in player_ids_by_name:
                nearby = " ".join(probable_section[max(candidate_index - 1, 0):candidate_index + 2])
                probable_pitchers.append(
                    LineupPlayer(
                        player_ids_by_name[candidate],
                        candidate,
                        throws=_extract_throwing_hand(nearby),
                    )
                )
            if len(probable_pitchers) == 2:
                break

        away_pitcher = None
        home_pitcher = None
        if probable_pitchers:
            away_pitcher = probable_pitchers[0]
        if len(probable_pitchers) > 1:
            home_pitcher = probable_pitchers[1]

        away_lineup, home_lineup, consumed_until = _parse_lineups_from_text_lines(
            lines,
            start_label_index=first_label,
            player_ids_by_name=player_ids_by_name,
        )
        cards.append(
            LineupCard(
                away_team=away_team,
                home_team=home_team,
                away_lineup=away_lineup,
                home_lineup=home_lineup,
                away_pitcher=away_pitcher,
                home_pitcher=home_pitcher,
            )
        )
        index = consumed_until

    return cards


def _parse_lineups_from_text_lines(
    lines: list[str],
    *,
    start_label_index: int,
    player_ids_by_name: dict[str, int],
) -> tuple[list[LineupPlayer], list[LineupPlayer], int]:
    away_label_index = start_label_index
    home_label_index = None
    for probe in range(away_label_index + 1, len(lines)):
        if _LINEUP_LABEL_RE.match(lines[probe]):
            home_label_index = probe
            break
    if home_label_index is None:
        return [], [], away_label_index + 1

    away_lineup, away_end = _collect_lineup_rows(lines, home_label_index + 1, player_ids_by_name)
    home_lineup, home_end = _collect_lineup_rows(lines, away_end, player_ids_by_name)
    return away_lineup, home_lineup, max(away_end, home_end)


def _collect_lineup_rows(
    lines: list[str],
    start_index: int,
    player_ids_by_name: dict[str, int],
) -> tuple[list[LineupPlayer], int]:
    lineup: list[LineupPlayer] = []
    index = start_index
    while index < len(lines):
        line = lines[index]
        if _LINEUP_LABEL_RE.match(line) or " @ " in line:
            break

        name = None
        match = _LINEUP_ROW_RE.match(line)
        batting_side = ""
        if match:
            name = match.group(2).strip()
            batting_side = _extract_batting_side(line)
            index += 1
        elif re.fullmatch(r"\d+\.", line) and index + 1 < len(lines):
            name = lines[index + 1].strip()
            batting_side = _extract_batting_side(name)
            index += 2
            if index < len(lines) and lines[index].startswith("("):
                batting_side = batting_side or _extract_batting_side(lines[index])
                index += 1
        else:
            index += 1

        if name is None:
            continue
        if name != "TBD" and name in player_ids_by_name:
            lineup.append(LineupPlayer(player_ids_by_name[name], name, bats=batting_side))
        else:
            lineup.append(LineupPlayer(0, name, bats=batting_side))
        if len(lineup) == 9:
            break

    lineup = [player for player in lineup if player.player_id > 0]
    return lineup, index
