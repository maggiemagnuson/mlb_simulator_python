# MLB Simulator Python (public MLB data rewrite)

Version 0.9.0

This is a Python rewrite of the original Java Monte Carlo MLB simulator.

The old Snoozle dependency has been removed. The simulator now pulls data from public MLB sources:

- **MLB StatsAPI** for schedules, game metadata, official batting orders when available, probable pitchers, and player season stats
- **MLB Starting Lineups** as a fallback when a scheduled game's official lineup has not yet been attached to the StatsAPI game feed
- **MLB Team Pitching Leaderboards (reliever split)** for team-specific bullpen profiles
- **Baseball Reference** for league-average constants used by the simulator

The goal of this rewrite is to keep the original simulator easy to run while making the data layer easier to maintain and the game model more realistic than the original straight port.

---

## What changed from the earlier Snoozle-based versions

- Removed the Snoozle API client entirely
- Added a new `MlbDataClient` built on public MLB endpoints
- Added lineup resolution modes:
  - `auto` (default): use StatsAPI boxscore first, then MLB Starting Lineups fallback
  - `boxscore`: only use the official StatsAPI game feed
  - `starting-lineups`: always use the MLB Starting Lineups page for batting orders and the StatsAPI season stats endpoint for player totals
- Added a plain `requirements.txt` for no-install usage
- Kept `run_sim.py` so you can launch the simulator from the repo root without installing the package
- Added `run_league_averages.py` for the same no-install workflow for league averages

---

## Requirements

- Python 3.10+
- Network access to:
  - `statsapi.mlb.com`
  - `www.mlb.com`
  - `www.baseball-reference.com` (only if you want fresh league-average generation)

Install runtime libraries:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` contains:

- `requests`
- `beautifulsoup4`

---

## Quick start without installing the package

From the project root:

```bash
python run_sim.py --date 2026-04-23 --games 100 --output-csv projections.csv
```

Console output now uses fixed-width columns so long team names do not push the rate stats out of alignment.

That is the recommended path if you do **not** want to run `pip install -e .`.

You can also run the module directly:

```bash
python -m mlb_simulator.cli --date 2026-04-23 --games 100 --output-csv projections.csv
```

---

## Typical commands

### Simulate all games on a date

```bash
python run_sim.py --date 2026-04-23 --games 1000
```

### Simulate a single game by MLB gamePk

```bash
python run_sim.py --date 2026-04-23 --game-id 777123 --games 5000
```

### Force official boxscore lineups only

```bash
python run_sim.py --date 2026-04-23 --lineup-source boxscore --games 1000
```

### Force MLB Starting Lineups page for batting orders

```bash
python run_sim.py --date 2026-04-23 --lineup-source starting-lineups --games 1000
```

### Write hitter projections to CSV

```bash
python run_sim.py --date 2026-04-23 --games 1000 --output-csv projections.csv
```

### Check whether lineups are posted

```bash
python run_sim.py --date 2026-04-23 --check-lineups
```

This prints whether each game resolved to a full 9-player lineup, plus the current hitters and probable pitchers when available.

---

## League averages

The simulator uses league-average constants as the run-environment baseline, while player talent priors now come from cached historical player seasons.

By default:

1. **past seasons** use one static yearly cache file per season, for example `league-averages-2025.json`
2. the **current season** uses a daily cache file keyed by the fetch date, for example `league-averages-2026-asof-2026-04-23.json`
3. when the requested season is the current season, the simulator blends the current-season daily values with the two prior yearly baselines
4. if any required cache file is missing, the simulator fetches it from Baseball Reference and writes the missing cache automatically
5. if fetching fails, it falls back to bundled defaults

This means the current season refreshes naturally once per day, while prior seasons stay static unless you force a refresh or the file does not exist yet.

Current-season blend weights are:

- April: `50% current season + 35% previous year + 15% two years prior`
- May through mid-June: `70% current season + 25% previous year + 5% two years prior`
- After mid-June: `85% current season + 15% previous year`

The resolver also handles when to create a new yearly file. Once a season is no longer the current year, the next request for that season uses the static yearly cache path and creates the yearly file automatically if it is missing.

---

## Player priors and player cache

The simulator now builds batter and pitcher profiles from the previous three MLB regular seasons using recency weights of `5 / 4 / 3`, then combines them with current-season stats through event-specific pseudo-count smoothing.

Those player caches are stored in the project / working directory by default rather than under `~/.cache`.

Default layout:

```text
./player_cache/
  current/
    hitting/<player_id>.json
    pitching/<player_id>.json
  seasons/
    hitting/<year>/<player_id>.json
    pitching/<year>/<player_id>.json
  profiles/
    hitting/<season>/<player_id>.json
    pitching/<season>/<player_id>.json
```

How it works:

1. for the current season, each player keeps **one current file per group** (`current/hitting/<id>.json` or `current/pitching/<id>.json`)
2. that file is refreshed at most once per day and overwritten in place, so the current season does not accumulate date-stamped files
3. when a new year starts, the resolver checks whether last season's static file exists under `seasons/<group>/<year>/<id>.json`
4. if it does not exist yet, it fetches the full prior-season stat line and writes it before overwriting the current-season file
5. the resolved player profile under `profiles/<group>/<season>/<id>.json` is then regenerated from the current season plus the previous three seasons
6. those profile files store resolved event rates and effective samples, not merged counting stat lines, so player skill differences are preserved better

You can override the default location with:

```bash
python run_sim.py --date 2026-04-23 --player-cache-dir ./my_player_cache
```

This player-cache system is separate from the league-averages cache. League averages still use the existing league cache path unless you override `--league-cache-dir`.

Team bullpen tables are cached under the same project-local cache root:

```text
./player_cache/
  bullpen/
    current/<season>.json
    seasons/<season>.json
```

The current-season bullpen table is refreshed at most once per day. Prior-season bullpen tables are static once written.

---

## Simulation model notes

Compared with the earlier ports, the run engine now uses a more realistic plate-appearance model:

- uses cached historical batter and pitcher priors from the previous three seasons, then blends current-season stats into those priors

- separates walks, HBP, strikeouts, home runs, and non-home-run hits instead of treating everything as one on-base bucket
- lets the opposing pitcher influence hit, walk, strikeout, and home-run rates
- adds double-play and sac-fly logic on balls in play
- uses more conservative baserunner advancement on singles and doubles

The CLI still prints **expected hits**, not the probability that a hitter records at least one hit in a single game. So a line like `1.05` means the hitter averaged 1.05 hits across all simulations; it does **not** mean the hitter gets a hit in every simulated game.

This is still a simplified simulator. It now uses a starter-to-bullpen handoff with team-specific reliever-split bullpen stats, but it still does not model exact reliever usage, platoon splits, defense, park factors, or detailed base-running data.

### Generate league averages manually

```bash
python run_league_averages.py --year 2026 --format json
```

### Use a specific league-average file

```bash
python run_sim.py \
  --date 2026-04-23 \
  --league-averages-file ./league-averages-2026.json \
  --games 1000
```

### Disable fetching and use only cache/defaults

```bash
python run_sim.py --date 2026-04-23 --no-fetch-league-averages
```

---

## CLI reference

### Core options

- `--date YYYY-MM-DD` (required)
- `--games N`
- `--seed N`
- `--game-id GAME_PK`
- `--output-csv PATH`

### Data-source options

- `--lineup-source {auto,boxscore,starting-lineups}`
- `--statsapi-base-url URL`
- `--starting-lineups-base-url URL`

### League-average options

- `--league-year YEAR`
- `--league-averages-file PATH`
- `--league-cache-dir PATH`
- `--player-cache-dir PATH`
- `--refresh-league-averages`
- `--no-fetch-league-averages`
- `--quiet-league-averages`
- `--check-lineups`

---

## Project layout

```text
mlb_simulator_python_0_7_0_resolved_profiles/
├── requirements.txt
├── run_sim.py
├── run_league_averages.py
├── README.md
├── MIGRATION_NOTES.md
├── pyproject.toml
├── mlb_simulator/
│   ├── __init__.py
│   ├── api.py
│   ├── cli.py
│   ├── league_averages.py
│   ├── models.py
│   └── simulator.py
└── tests/
```

---

## How the new data flow works

### 1. Daily schedule

`MlbDataClient.fetch_daily_games()` pulls the date's games from MLB StatsAPI schedule data.

### 2. Game input resolution

`MlbDataClient.fetch_game()` tries, in order:

1. MLB StatsAPI live game boxscore / feed data
2. MLB Starting Lineups page fallback

If the official game feed already has a complete 9-player batting order for both teams, that is used directly.

If not, the client scrapes MLB's public Starting Lineups page for the date, extracts player IDs from player links, and then pulls season stats from the StatsAPI `stats` endpoint.

### 3. Simulation

The simulator converts those player stats into the same normalized batting/pitching structures used by the Monte Carlo engine.

---


## Cache notes

- Player stat caches live under `./player_cache/` by default.
- Each cache file now includes `player_id` and `player_name` for easier inspection.
- Version `0.7.1` refreshes stale `current/`, `seasons/`, and `profiles/` files automatically when older cache formats are detected.
- Player season stats are now fetched from player-specific MLB StatsAPI endpoints instead of the generic stats endpoint, which fixes an issue where multiple players could incorrectly receive the same cached stat line.

## Limitations

- **Pregame lineup availability is still a real-world constraint.** If neither the official boxscore feed nor MLB's public starting-lineups page has a full 9-player order, the simulator will skip that game.
- **The fallback path depends on MLB.com page structure.** If MLB changes the Starting Lineups page markup substantially, the fallback parser may need a small update.
- **Historical backtesting may differ from Snoozle.** For games resolved from the fallback lineup page, player stats come from StatsAPI season totals rather than Snoozle's old custom daily cumulative feed.
- **Pitcher on-base allowed is reconstructed from available public stats when necessary.** Public MLB endpoints do not expose the same exact Snoozle pitcher payload.
- **Bullpen usage is still simplified.** Like the original project, this simulator is primarily starter-vs-lineup driven.

---

## Running tests

```bash
python -m unittest discover -s tests -v
```

---

## Optional package install

If you do want the console scripts:

```bash
python -m pip install -e .
```

Then you can run:

```bash
mlb-sim --date 2026-04-23 --games 1000
mlb-league-averages --year 2026 --format json
```

---

## Notes for future extension

The data layer is now isolated enough that you can swap in a different provider later.

The easiest extension points are:

- replace `MlbDataClient` with a paid feed provider
- add a persistent player stat cache
- add better bullpen modeling
- add park factors and platoon splits
- add explicit historical date-bounded stat resolution



## Troubleshooting

### `404 Client Error` from `.../game/<gamePk>/feed/live`

This can happen for scheduled games before MLB has published the richer live game payload. In `0.3.1`, the simulator treats that as a missing live feed instead of a hard crash and falls back to the MLB Starting Lineups page when possible.

If you still see a skip for a game, first run `python run_sim.py --date YYYY-MM-DD --check-lineups`. That will tell you whether the game really has a lineup posted and how many hitters were resolved for each team.

As of `0.3.6`, the Starting Lineups fallback uses the actual matchup DOM cards on MLB.com instead of flattening the whole page into text. This fixes a bug where posted lineups could be present in the saved HTML but still be reported as `0` parsed cards.


## Recent output additions

- The hitter table and CSV now include `1+H%` and `2+H%`, which are the simulated probabilities that a batter records at least one hit or at least two hits in a game.
- Double-play risk now includes a speed adjustment based on the batter's stolen-base rate, so faster runners are less likely to be turned into double plays than slower runners.

## Version 0.9.0 notes

- Adds starter-to-bullpen transitions so games are no longer simulated as if one pitcher faces the lineup all night.
- Uses event-specific prior sample caps for hitter and pitcher profiles, which preserves proven skill while letting April hot and cold starts move the probabilities more.
- Bumps the resolved profile cache model version, so old `profiles/` files are ignored and rebuilt.
