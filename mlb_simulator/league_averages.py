from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import requests
from bs4 import BeautifulSoup


LeagueAveragesSource = Literal["cache", "fetched", "default", "file", "blended"]


DEFAULT_CACHE_DIRNAME = "mlb-simulator-python"
DEFAULT_CACHE_FILENAME_TEMPLATE = "league-averages-{year}.json"
DEFAULT_DAILY_CACHE_FILENAME_TEMPLATE = "league-averages-{year}-asof-{as_of_date}.json"
CACHE_FORMAT_VERSION = 2


@dataclass(slots=True)
class LeagueAverages:
    hitter_on_base: float = 0.31680500779601944
    pitcher_on_base: float = 0.31680500779601944
    average_pitches_per_start: float = 144.7754087628866
    min_plate_appearances: float = 100.0
    min_innings_pitched: float = 50.0
    hit_by_pitch_total: float = 32.0
    sac_flies_total: float = 19.0
    strikeouts_total: float = 650.0
    grounded_into_double_play_total: float = 45.0
    at_bats_per_lineup_spot: float = 294.44444444444446
    hits_total: float = 664.0
    doubles_total: float = 134.0
    triples_total: float = 12.0
    home_runs_total: float = 105.0
    singles_total: float = 413.0
    walks_total: float = 257.0
    total_plate_appearances: float = 2972.0

    @property
    def walk_rate_per_pa(self) -> float:
        return self.walks_total / max(self.total_plate_appearances, 1.0)

    @property
    def hit_by_pitch_rate_per_pa(self) -> float:
        return self.hit_by_pitch_total / max(self.total_plate_appearances, 1.0)

    @property
    def hit_rate_per_pa(self) -> float:
        return self.hits_total / max(self.total_plate_appearances, 1.0)

    @property
    def home_run_rate_per_pa(self) -> float:
        return self.home_runs_total / max(self.total_plate_appearances, 1.0)

    @property
    def non_home_run_hit_rate_per_pa(self) -> float:
        return max(self.hits_total - self.home_runs_total, 0.0) / max(self.total_plate_appearances, 1.0)

    @property
    def strikeout_rate_per_pa(self) -> float:
        return self.strikeouts_total / max(self.total_plate_appearances, 1.0)

    @property
    def estimated_balls_in_play_outs(self) -> float:
        return max(
            self.total_plate_appearances - self.walks_total - self.hit_by_pitch_total - self.hits_total - self.strikeouts_total,
            1.0,
        )

    @property
    def prior_balls_in_play_outs_for_hitter_prior(self) -> float:
        return self.estimated_balls_in_play_outs / max(self.total_plate_appearances, 1.0) * self.min_plate_appearances

    @property
    def prior_batters_faced_for_pitcher_prior(self) -> float:
        return max(self.min_innings_pitched * 4.25, 1.0)

    @property
    def prior_balls_in_play_outs_for_pitcher_prior(self) -> float:
        return self.estimated_balls_in_play_outs / max(self.total_plate_appearances, 1.0) * self.prior_batters_faced_for_pitcher_prior

    @property
    def single_share_of_non_home_run_hits(self) -> float:
        return self.singles_total / max(self.singles_total + self.doubles_total + self.triples_total, 1.0)

    @property
    def double_share_of_non_home_run_hits(self) -> float:
        return self.doubles_total / max(self.singles_total + self.doubles_total + self.triples_total, 1.0)

    @property
    def triple_share_of_non_home_run_hits(self) -> float:
        return self.triples_total / max(self.singles_total + self.doubles_total + self.triples_total, 1.0)

    @property
    def sac_fly_rate_on_in_play_out(self) -> float:
        return self.sac_flies_total / self.estimated_balls_in_play_outs

    @property
    def double_play_rate_on_in_play_out(self) -> float:
        return self.grounded_into_double_play_total / self.estimated_balls_in_play_outs

    @property
    def prob_single_on_base(self) -> float:
        return self.singles_total / max(self.total_plate_appearances * self.hitter_on_base, 1.0)

    @property
    def prob_double_on_base(self) -> float:
        return self.doubles_total / max(self.total_plate_appearances * self.hitter_on_base, 1.0)

    @property
    def prob_triple_on_base(self) -> float:
        return self.triples_total / max(self.total_plate_appearances * self.hitter_on_base, 1.0)

    @property
    def prob_home_run_on_base(self) -> float:
        return self.home_runs_total / max(self.total_plate_appearances * self.hitter_on_base, 1.0)

    @property
    def prob_walk_on_base(self) -> float:
        return self.walks_total / max(self.total_plate_appearances * self.hitter_on_base, 1.0)

    @property
    def prob_hit_by_pitch_on_base(self) -> float:
        return self.hit_by_pitch_total / max(self.total_plate_appearances * self.hitter_on_base, 1.0)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def to_python_constants(self) -> str:
        return "\n".join(
            [
                f"hitter_on_base = {self.hitter_on_base}",
                f"pitcher_on_base = {self.pitcher_on_base}",
                f"average_pitches_per_start = {self.average_pitches_per_start}",
                f"min_plate_appearances = {self.min_plate_appearances}",
                f"min_innings_pitched = {self.min_innings_pitched}",
                f"hit_by_pitch_total = {self.hit_by_pitch_total}",
                f"sac_flies_total = {self.sac_flies_total}",
                f"strikeouts_total = {self.strikeouts_total}",
                f"grounded_into_double_play_total = {self.grounded_into_double_play_total}",
                f"at_bats_per_lineup_spot = {self.at_bats_per_lineup_spot}",
                f"hits_total = {self.hits_total}",
                f"doubles_total = {self.doubles_total}",
                f"triples_total = {self.triples_total}",
                f"home_runs_total = {self.home_runs_total}",
                f"singles_total = {self.singles_total}",
                f"walks_total = {self.walks_total}",
                f"total_plate_appearances = {self.total_plate_appearances}",
            ]
        )

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write_json(self, path: str | os.PathLike[str] | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json() + "\n", encoding="utf-8")
        return destination

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LeagueAverages":
        allowed = cls.__dataclass_fields__.keys()
        filtered = {key: payload[key] for key in allowed if key in payload}
        return cls(**filtered)

    @classmethod
    def from_json_file(cls, path: str | os.PathLike[str] | Path) -> "LeagueAverages":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "averages" in payload and isinstance(payload["averages"], dict):
            payload = payload["averages"]
        return cls.from_dict(payload)

    @classmethod
    def blend(cls, weighted_averages: list[tuple["LeagueAverages", float]]) -> "LeagueAverages":
        normalized = [(averages, float(weight)) for averages, weight in weighted_averages if averages is not None and weight > 0.0]
        if not normalized:
            return cls()
        total_weight = sum(weight for _, weight in normalized)
        fields = cls.__dataclass_fields__.keys()
        blended = {
            field_name: sum(getattr(averages, field_name) * weight for averages, weight in normalized) / total_weight
            for field_name in fields
        }
        return cls(**blended)

    @classmethod
    def from_baseball_reference(cls, year: int, session: requests.Session | None = None) -> "LeagueAverages":
        session = session or requests.Session()
        batting_soup = _fetch_soup(session, f"https://www.baseball-reference.com/leagues/MLB/{year}.shtml")
        pitching_soup = _fetch_soup(session, f"https://www.baseball-reference.com/leagues/MLB/{year}-standard-pitching.shtml")
        pitches_soup = _fetch_soup(session, f"https://www.baseball-reference.com/leagues/MLB/{year}-pitches-pitching.shtml")

        batting_total = _extract_total_row(batting_soup, "teams_standard_batting")
        pitching_total = _extract_total_row(pitching_soup, "teams_standard_pitching")
        pitches_total = _extract_total_row(pitches_soup, "teams_pitches_pitching")

        games = max(_safe_float(_pick(batting_total, "G", "g", "games")), 1.0)
        team_hits = _safe_float(_pick(batting_total, "H", "hits"))
        team_doubles = _safe_float(_pick(batting_total, "2B", "doubles"))
        team_triples = _safe_float(_pick(batting_total, "3B", "triples"))
        team_home_runs = _safe_float(_pick(batting_total, "HR", "home_runs"))
        team_walks = _safe_float(_pick(batting_total, "BB", "bases_on_balls", "walks"))
        team_plate_appearances = _safe_float(_pick(batting_total, "PA", "plate_appearances"))
        team_at_bats = _safe_float(_pick(batting_total, "AB", "at_bats"))
        team_hit_by_pitch = _safe_float(_pick(batting_total, "HBP", "hit_by_pitch"))
        team_sac_flies = _safe_float(_pick(batting_total, "SF", "sacrifice_flies"))
        team_strikeouts = _safe_float(_pick(batting_total, "SO", "strikeouts"))
        team_gidp = _safe_float(_pick(batting_total, "GDP", "GIDP", "grounded_into_double_play"))
        singles = team_hits - team_doubles - team_triples - team_home_runs

        innings_pitched = _innings_to_decimal(_pick(pitching_total, "IP", "innings_pitched"))
        pitcher_hits = _safe_float(_pick(pitching_total, "H", "hits"))
        pitcher_walks = _safe_float(_pick(pitching_total, "BB", "bases_on_balls", "walks"))
        batters_faced = ((innings_pitched * 2.82) + pitcher_hits + pitcher_walks) / games
        pitches_per_pa = _safe_float(
            _pick(
                pitches_total,
                "PitPerPlateAppearance",
                "Pit/PA",
                "pit_per_plate_app",
                "pitches_per_plate_appearance",
            )
        )
        average_pitches_per_start = batters_faced * pitches_per_pa
        on_base = (pitcher_hits + pitcher_walks) / max(team_at_bats + team_walks, 1.0)

        defaults = cls()
        return cls(
            hitter_on_base=on_base,
            pitcher_on_base=on_base,
            average_pitches_per_start=average_pitches_per_start,
            hit_by_pitch_total=team_hit_by_pitch,
            sac_flies_total=team_sac_flies,
            strikeouts_total=team_strikeouts or defaults.strikeouts_total,
            grounded_into_double_play_total=team_gidp or defaults.grounded_into_double_play_total,
            at_bats_per_lineup_spot=team_at_bats / 9.0,
            hits_total=team_hits,
            doubles_total=team_doubles,
            triples_total=team_triples,
            home_runs_total=team_home_runs,
            singles_total=singles,
            walks_total=team_walks,
            total_plate_appearances=team_plate_appearances,
        )


@dataclass(slots=True)
class LeagueAveragesResolution:
    year: int
    averages: LeagueAverages
    source: LeagueAveragesSource
    cache_path: Path | None = None
    message: str = ""


@dataclass(slots=True)
class _SingleSeasonResolution:
    year: int
    averages: LeagueAverages
    source: Literal["cache", "fetched", "default"]
    cache_path: Path | None = None
    cache_kind: Literal["yearly", "daily"] = "yearly"
    as_of_date: str | None = None


def _safe_float(value: str | int | float | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return 0.0
    return float(text)


def _innings_to_decimal(value: str | int | float | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        value = str(value)
    innings, dot, partial = value.partition(".")
    decimal = float(innings or 0)
    if dot:
        decimal += _safe_float(partial) / 3.0
    return decimal


def _current_date() -> date:
    return date.today()


def default_cache_dir() -> Path:
    override = os.environ.get("MLB_SIMULATOR_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / DEFAULT_CACHE_DIRNAME
    return Path.home() / ".cache" / DEFAULT_CACHE_DIRNAME


def cache_path_for_year(year: int, cache_dir: str | os.PathLike[str] | Path | None = None) -> Path:
    base = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
    return base / DEFAULT_CACHE_FILENAME_TEMPLATE.format(year=year)


def cache_path_for_date(
    year: int,
    as_of_date: str | date,
    cache_dir: str | os.PathLike[str] | Path | None = None,
) -> Path:
    base = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
    if isinstance(as_of_date, date):
        as_of = as_of_date.isoformat()
    else:
        as_of = str(as_of_date)
    return base / DEFAULT_DAILY_CACHE_FILENAME_TEMPLATE.format(year=year, as_of_date=as_of)


def _read_cache_payload(path: Path) -> tuple[LeagueAverages, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "averages" in payload and isinstance(payload["averages"], dict):
        metadata = {key: value for key, value in payload.items() if key != "averages"}
        return LeagueAverages.from_dict(payload["averages"]), metadata
    return LeagueAverages.from_dict(payload), {"version": 1, "cache_kind": "legacy", "season_year": None}


def _write_cache_payload(
    path: Path,
    year: int,
    averages: LeagueAverages,
    *,
    cache_kind: Literal["yearly", "daily"],
    as_of_date: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "version": CACHE_FORMAT_VERSION,
        "season_year": year,
        "cache_kind": cache_kind,
        "fetched_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "averages": averages.to_dict(),
    }
    if as_of_date is not None:
        payload["as_of_date"] = as_of_date
    if metadata:
        payload.update(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def save_cached_league_averages(
    year: int,
    averages: LeagueAverages,
    *,
    cache_dir: str | os.PathLike[str] | Path | None = None,
    as_of_date: str | date | None = None,
) -> Path:
    if as_of_date is None:
        path = cache_path_for_year(year, cache_dir)
        return _write_cache_payload(path, year, averages, cache_kind="yearly")
    if isinstance(as_of_date, date):
        as_of = as_of_date.isoformat()
    else:
        as_of = str(as_of_date)
    path = cache_path_for_date(year, as_of, cache_dir)
    return _write_cache_payload(path, year, averages, cache_kind="daily", as_of_date=as_of)


def load_cached_league_averages(
    year: int,
    *,
    cache_dir: str | os.PathLike[str] | Path | None = None,
    as_of_date: str | date | None = None,
) -> LeagueAverages | None:
    if as_of_date is None:
        path = cache_path_for_year(year, cache_dir)
    else:
        path = cache_path_for_date(year, as_of_date, cache_dir)
    if not path.exists():
        return None
    averages, _ = _read_cache_payload(path)
    return averages


def _resolve_single_season(
    year: int,
    *,
    cache_dir: str | os.PathLike[str] | Path | None,
    refresh: bool,
    allow_fetch: bool,
    fallback_to_defaults: bool,
    session: requests.Session | None,
    daily_as_of: date | None = None,
) -> _SingleSeasonResolution:
    is_daily = daily_as_of is not None
    if is_daily:
        cache_path = cache_path_for_date(year, daily_as_of, cache_dir)
        cache_kind: Literal["yearly", "daily"] = "daily"
        as_of_text = daily_as_of.isoformat()
    else:
        cache_path = cache_path_for_year(year, cache_dir)
        cache_kind = "yearly"
        as_of_text = None

    if not refresh and cache_path.exists():
        averages, _ = _read_cache_payload(cache_path)
        return _SingleSeasonResolution(
            year=year,
            averages=averages,
            source="cache",
            cache_path=cache_path,
            cache_kind=cache_kind,
            as_of_date=as_of_text,
        )

    if allow_fetch:
        try:
            fetched = LeagueAverages.from_baseball_reference(year, session=session)
        except Exception:
            if not fallback_to_defaults:
                raise
        else:
            saved_path = save_cached_league_averages(year, fetched, cache_dir=cache_dir, as_of_date=daily_as_of)
            return _SingleSeasonResolution(
                year=year,
                averages=fetched,
                source="fetched",
                cache_path=saved_path,
                cache_kind=cache_kind,
                as_of_date=as_of_text,
            )

    if fallback_to_defaults:
        return _SingleSeasonResolution(
            year=year,
            averages=LeagueAverages(),
            source="default",
            cache_path=None,
            cache_kind=cache_kind,
            as_of_date=as_of_text,
        )

    raise FileNotFoundError(f"No cached league averages for {year} found at {cache_path}")


def _blend_weights_for_reference_date(reference_date: date) -> tuple[float, float, float]:
    if reference_date.month < 5:
        return (0.50, 0.35, 0.15)
    if reference_date.month < 6 or (reference_date.month == 6 and reference_date.day < 15):
        return (0.70, 0.25, 0.05)
    return (0.85, 0.15, 0.0)


def _describe_component(resolution: _SingleSeasonResolution) -> str:
    if resolution.cache_kind == "daily" and resolution.as_of_date:
        return f"{resolution.year} daily as-of {resolution.as_of_date} ({resolution.source})"
    return f"{resolution.year} yearly ({resolution.source})"


def resolve_league_averages_for_year(
    year: int,
    *,
    cache_dir: str | os.PathLike[str] | Path | None = None,
    refresh: bool = False,
    allow_fetch: bool = True,
    fallback_to_defaults: bool = True,
    session: requests.Session | None = None,
    reference_date: date | None = None,
    prior_years_to_blend: int = 2,
) -> LeagueAveragesResolution:
    reference_date = reference_date or _current_date()

    if year != reference_date.year:
        single = _resolve_single_season(
            year,
            cache_dir=cache_dir,
            refresh=refresh,
            allow_fetch=allow_fetch,
            fallback_to_defaults=fallback_to_defaults,
            session=session,
            daily_as_of=None,
        )
        if single.source == "cache":
            message = f"Loaded cached league averages for {year} from {single.cache_path}"
        elif single.source == "fetched":
            message = f"Fetched league averages for {year} and cached them at {single.cache_path}"
        else:
            message = f"Could not fetch league averages for {year}; using bundled defaults instead."
        return LeagueAveragesResolution(
            year=year,
            averages=single.averages,
            source=single.source,
            cache_path=single.cache_path,
            message=message,
        )

    current = _resolve_single_season(
        year,
        cache_dir=cache_dir,
        refresh=refresh,
        allow_fetch=allow_fetch,
        fallback_to_defaults=fallback_to_defaults,
        session=session,
        daily_as_of=reference_date,
    )

    weighted_components: list[tuple[LeagueAverages, float]] = []
    component_descriptions: list[str] = [_describe_component(current)]
    weights = _blend_weights_for_reference_date(reference_date)
    if weights[0] > 0.0:
        weighted_components.append((current.averages, weights[0]))

    for index in range(1, prior_years_to_blend + 1):
        if index >= len(weights):
            break
        prior_weight = weights[index]
        if prior_weight <= 0.0:
            continue
        prior_year = year - index
        prior = _resolve_single_season(
            prior_year,
            cache_dir=cache_dir,
            refresh=False,
            allow_fetch=allow_fetch,
            fallback_to_defaults=fallback_to_defaults,
            session=session,
            daily_as_of=None,
        )
        component_descriptions.append(_describe_component(prior))
        weighted_components.append((prior.averages, prior_weight))

    if len(weighted_components) == 1:
        only = weighted_components[0][0]
        source = current.source
        if current.source == "cache":
            message = f"Loaded current-season daily league averages for {year} as of {reference_date.isoformat()} from {current.cache_path}"
        elif current.source == "fetched":
            message = (
                f"Fetched current-season daily league averages for {year} as of {reference_date.isoformat()} "
                f"and cached them at {current.cache_path}"
            )
        else:
            message = f"Could not resolve current-season league averages for {year}; using bundled defaults instead."
        return LeagueAveragesResolution(
            year=year,
            averages=only,
            source=source,
            cache_path=current.cache_path,
            message=message,
        )

    blended = LeagueAverages.blend(weighted_components)
    message = (
        f"Resolved current-season league averages for {year} using {', '.join(component_descriptions)}; "
        f"blended with weights {weights[0]:.0%}/{weights[1]:.0%}/{weights[2]:.0%}."
    )
    return LeagueAveragesResolution(
        year=year,
        averages=blended,
        source="blended",
        cache_path=current.cache_path,
        message=message,
    )


def _fetch_soup(session: requests.Session, url: str) -> BeautifulSoup:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def _find_table(soup: BeautifulSoup, table_id: str):
    table = soup.find("table", {"id": table_id})
    if table is not None:
        return table

    for comment in soup.find_all(string=lambda s: isinstance(s, str) and "<table" in s):
        inner = BeautifulSoup(comment, "html.parser")
        table = inner.find("table", {"id": table_id})
        if table is not None:
            return table
    raise ValueError(f"Could not locate table {table_id!r}")


def _extract_total_row(soup: BeautifulSoup, table_id: str) -> dict[str, str]:
    table = _find_table(soup, table_id)
    header_names = _header_names(table)
    body = table.find("tbody") or table
    for row in body.find_all("tr"):
        label_cell = row.find(["th", "td"])
        if label_cell is None:
            continue
        label = label_cell.get_text(strip=True).upper()
        if label in {"TOTAL", "TOT", "LEAGUE TOTALS"}:
            return _row_to_mapping(row, header_names)

    rows = body.find_all("tr")
    if not rows:
        raise ValueError(f"No rows found in table {table_id!r}")
    return _row_to_mapping(rows[-1], header_names)


def _header_names(table) -> list[str]:
    thead = table.find("thead")
    if thead is None:
        return []
    rows = thead.find_all("tr")
    if not rows:
        return []
    last = rows[-1]
    return [cell.get_text(strip=True) for cell in last.find_all(["th", "td"])]


def _row_to_mapping(row, header_names: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    cells = row.find_all(["th", "td"])
    for idx, cell in enumerate(cells):
        value = cell.get_text(strip=True)
        data_stat = cell.get("data-stat")
        if data_stat:
            mapping[data_stat] = value
        if idx < len(header_names):
            header = header_names[idx]
            if header:
                mapping[header] = value
    return mapping


def _pick(mapping: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    normalized = {str(k).lower(): v for k, v in mapping.items()}
    for key in keys:
        if key.lower() in normalized:
            return normalized[key.lower()]
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate or cache league-average constants from Baseball Reference.")
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--format", choices=["json", "python"], default="python")
    parser.add_argument("--output", help="Optional output file. If omitted, prints to stdout.")
    parser.add_argument("--cache-dir", help="Optional cache directory for year-based JSON files.")
    parser.add_argument("--refresh", action="store_true", help="Refresh from Baseball Reference even if a cache file exists.")
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Do not fetch from Baseball Reference. Use cached values if present, otherwise fall back to bundled defaults.",
    )
    parser.add_argument(
        "--print-status",
        action="store_true",
        help="Print a short status line to stderr describing whether cached, fetched, blended, or default values were used.",
    )
    args = parser.parse_args(argv)

    resolution = resolve_league_averages_for_year(
        args.year,
        cache_dir=args.cache_dir,
        refresh=args.refresh,
        allow_fetch=not args.no_fetch,
        fallback_to_defaults=True,
    )
    if args.print_status:
        print(resolution.message, file=os.sys.stderr)

    averages = resolution.averages
    rendered = averages.to_json() if args.format == "json" else averages.to_python_constants()
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
