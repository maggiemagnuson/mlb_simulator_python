from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from .api import MlbDataClient, MlbDataError
from .league_averages import LeagueAverages, resolve_league_averages_for_year
from .simulator import MonteCarloSimulator


TABLE_SEPARATOR = "  "
NUMERIC_HEADERS = ("AB", "R", "H", "1+H%", "2+H%", "2B", "3B", "HR", "RBI", "BB", "AVG", "OBP", "SLG")


def _parse_year_from_date(date_text: str) -> int:
    return datetime.strptime(date_text, "%Y-%m-%d").year


def _format_hit_probability(value: float) -> str:
    return f"{value * 100:.1f}%"


def _write_projection_rows(path: Path, projection_rows) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Player", "Team", "AB", "R", "H", "1+H%", "2+H%", "2B", "3B", "HR", "RBI", "BB", "AVG", "OBP", "SLG"])
        for projection in projection_rows:
            writer.writerow(
                [
                    projection.player_name,
                    projection.team_name,
                    f"{projection.at_bats:.2f}",
                    f"{projection.runs:.2f}",
                    f"{projection.hits:.2f}",
                    _format_hit_probability(projection.hit_1_plus_probability),
                    _format_hit_probability(projection.hit_2_plus_probability),
                    f"{projection.doubles:.2f}",
                    f"{projection.triples:.2f}",
                    f"{projection.home_runs:.2f}",
                    f"{projection.rbi:.2f}",
                    f"{projection.walks:.2f}",
                    f"{projection.batting_average:.3f}",
                    f"{projection.on_base_percentage:.3f}",
                    f"{projection.slugging_percentage:.3f}",
                ]
            )


def _projection_to_row(projection) -> list[str]:
    return [
        projection.player_name,
        projection.team_name,
        f"{projection.at_bats:.2f}",
        f"{projection.runs:.2f}",
        f"{projection.hits:.2f}",
        _format_hit_probability(projection.hit_1_plus_probability),
        _format_hit_probability(projection.hit_2_plus_probability),
        f"{projection.doubles:.2f}",
        f"{projection.triples:.2f}",
        f"{projection.home_runs:.2f}",
        f"{projection.rbi:.2f}",
        f"{projection.walks:.2f}",
        f"{projection.batting_average:.3f}",
        f"{projection.on_base_percentage:.3f}",
        f"{projection.slugging_percentage:.3f}",
    ]


def _compute_column_widths(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[int]:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    return widths


def _format_table_line(values: Sequence[str], widths: Sequence[int], *, left_align_columns: Iterable[int]) -> str:
    left_align = set(left_align_columns)
    cells = []
    for index, value in enumerate(values):
        if index in left_align:
            cells.append(f"{value:<{widths[index]}}")
        else:
            cells.append(f"{value:>{widths[index]}}")
    return TABLE_SEPARATOR.join(cells)


def _build_projection_table_lines(projections) -> list[str]:
    headers = ["Player", "Team", *NUMERIC_HEADERS]
    rows = [_projection_to_row(projection) for projection in projections]
    widths = _compute_column_widths(headers, rows)

    header_line = _format_table_line(headers, widths, left_align_columns=(0, 1))
    separator_line = "-" * len(header_line)
    table_lines = [header_line, separator_line]
    for row in rows:
        table_lines.append(_format_table_line(row, widths, left_align_columns=(0, 1)))
    return table_lines


def _print_summary(summary, away_pitcher: str, home_pitcher: str) -> None:
    matchup_title = f"{summary.game_date} {summary.away_team} at {summary.home_team}"
    print("=" * max(70, len(matchup_title)))
    print(matchup_title)
    print(f"Prob Away Team Wins: {summary.away_win_probability * 100:.2f}%")
    print(f"Prob Home Team Wins: {summary.home_win_probability * 100:.2f}%")
    print(f"Average Comb Score: {summary.average_total_runs:.2f}")
    print(f"Median Comb Score: {summary.median_total_runs:.2f}")
    print(f"Std Comb Score: {summary.stddev_total_runs:.2f}")

    for label, pitcher, projections in (
        (summary.away_team, away_pitcher, summary.away_projections),
        (summary.home_team, home_pitcher, summary.home_projections),
    ):
        print(f"\n{label} Starting Pitcher: {pitcher}")
        print(f"{label} Hitters Projected Performance")
        for line in _build_projection_table_lines(projections):
            print(line)


def _print_lineup_inspection(inspection) -> None:
    status = "SET" if inspection.is_set else "TBD"
    print(
        f"{status} {inspection.away_team} at {inspection.home_team} "
        f"({len(inspection.away_lineup)}/9 away, {len(inspection.home_lineup)}/9 home) via {inspection.source}"
    )
    if inspection.away_pitcher or inspection.home_pitcher:
        away_pitcher = inspection.away_pitcher.player_name if inspection.away_pitcher else ""
        home_pitcher = inspection.home_pitcher.player_name if inspection.home_pitcher else ""
        print(f"  Pitchers: {away_pitcher} vs {home_pitcher}")
    if inspection.away_lineup:
        print("  Away: " + ", ".join(player.player_name for player in inspection.away_lineup))
    if inspection.home_lineup:
        print("  Home: " + ", ".join(player.player_name for player in inspection.home_lineup))
    if inspection.note:
        print(f"  Note: {inspection.note}")


def _resolve_league_averages(args) -> tuple[LeagueAverages, str]:
    if args.league_averages_file:
        averages = LeagueAverages.from_json_file(args.league_averages_file)
        message = f"Loaded league averages from file: {args.league_averages_file}"
        return averages, message

    league_year = args.league_year or _parse_year_from_date(args.date)
    resolution = resolve_league_averages_for_year(
        league_year,
        cache_dir=args.league_cache_dir,
        refresh=args.refresh_league_averages,
        allow_fetch=not args.no_fetch_league_averages,
        fallback_to_defaults=True,
    )
    return resolution.averages, resolution.message


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Python MLB Monte Carlo simulator.")
    parser.add_argument("--date", required=True, help="Game date in YYYY-MM-DD format.")
    parser.add_argument("--games", type=int, default=10000, help="Number of simulations to run per game.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--game-id", help="Optional single MLB gamePk. If omitted, all games on the date are simulated.")
    parser.add_argument("--output-csv", help="Optional CSV path for hitter projection rows.")
    parser.add_argument(
        "--lineup-source",
        choices=["auto", "boxscore", "starting-lineups"],
        default="auto",
        help=(
            "Where to source batting orders from. 'auto' uses MLB StatsAPI live boxscore first, then MLB's public "
            "Starting Lineups page as a fallback if the official lineup is not yet attached to the game feed."
        ),
    )
    parser.add_argument(
        "--statsapi-base-url",
        default="https://statsapi.mlb.com/api/v1",
        help="Override for the public MLB StatsAPI base URL.",
    )
    parser.add_argument(
        "--starting-lineups-base-url",
        default="https://www.mlb.com/starting-lineups",
        help="Override for the MLB Starting Lineups page base URL.",
    )
    parser.add_argument(
        "--league-year",
        type=int,
        help="League-average season to use. Defaults to the year from --date.",
    )
    parser.add_argument(
        "--league-averages-file",
        help="Optional JSON file with league averages. Overrides cache and auto-fetch.",
    )
    parser.add_argument(
        "--league-cache-dir",
        help="Optional cache directory for auto-generated league averages.",
    )
    parser.add_argument(
        "--refresh-league-averages",
        action="store_true",
        help="Refresh league averages from Baseball Reference even if a cache file exists.",
    )
    parser.add_argument(
        "--no-fetch-league-averages",
        action="store_true",
        help="Do not fetch league averages. Use cached values if present, otherwise fall back to bundled defaults.",
    )
    parser.add_argument(
        "--quiet-league-averages",
        action="store_true",
        help="Suppress the startup message describing how league averages were resolved.",
    )
    parser.add_argument(
        "--check-lineups",
        action="store_true",
        help="Show whether each game has a full 9-player lineup resolved, then exit without simulating.",
    )
    parser.add_argument(
        "--debug-dir",
        help=(
            "Optional directory where raw MLB fetch artifacts are written, including lineup-page HTML, "
            "fetch summaries, and live-feed debug JSON."
        ),
    )
    parser.add_argument(
        "--player-cache-dir",
        help=(
            "Optional cache directory for player current-season files, historical season files, and resolved player profiles. "
            "Defaults to ./player_cache under the project root / current working directory."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    league_averages = None
    league_message = ""
    if not args.check_lineups:
        league_averages, league_message = _resolve_league_averages(args)

    with MlbDataClient(
        statsapi_base_url=args.statsapi_base_url,
        starting_lineups_base_url=args.starting_lineups_base_url,
        debug_dir=args.debug_dir,
        player_cache_dir=args.player_cache_dir,
        profile_fallback_league=league_averages,
    ) as client:
        if args.game_id:
            games = [args.game_id]
        else:
            daily_games = client.fetch_daily_games(args.date)
            if not daily_games:
                raise SystemExit(f"No games for {args.date}")
            games = [matchup.game_id for matchup in daily_games]

        if args.check_lineups:
            for game_id in games:
                inspection = client.inspect_game_lineup(game_id, lineup_source=args.lineup_source)
                _print_lineup_inspection(inspection)
            return 0

        if not args.quiet_league_averages:
            print(league_message)

        all_rows = []
        for game_id in games:
            try:
                game = client.fetch_game(game_id, lineup_source=args.lineup_source)
            except MlbDataError as exc:
                print(f"Skipping game {game_id}: {exc}")
                continue

            if len(game.away_lineup) != 9 or len(game.home_lineup) != 9:
                print(f"Skipping {game.away_team} at {game.home_team}: incomplete lineup")
                continue
            simulator = MonteCarloSimulator(game, seed=args.seed, league_averages=league_averages)
            summary = simulator.run(number_of_games=args.games)
            away_pitcher = game.away_pitching[0].player_name if game.away_pitching else ""
            home_pitcher = game.home_pitching[0].player_name if game.home_pitching else ""
            _print_summary(summary, away_pitcher, home_pitcher)
            all_rows.extend(summary.away_projections)
            all_rows.extend(summary.home_projections)

    if args.output_csv:
        output_path = Path(args.output_csv)
        _write_projection_rows(output_path, all_rows)
        print(f"\nWrote projections to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
