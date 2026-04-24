# Migration notes

## 0.3.0

This release replaces the legacy Snoozle API dependency with public MLB data sources.

### Data source changes

Old flow:

- `dailygames` from Snoozle
- `montecarlostats` from Snoozle

New flow:

- schedules from MLB StatsAPI
- official lineups and player season stats from MLB StatsAPI live game data when available
- fallback batting orders from MLB Starting Lineups page
- fallback player stats from MLB StatsAPI `stats` endpoint
- league averages from Baseball Reference cache/fetch flow

### CLI changes

Removed:

- `--base-url`

Added:

- `--lineup-source {auto,boxscore,starting-lineups}`
- `--statsapi-base-url`
- `--starting-lineups-base-url`
- `requirements.txt`
- `run_sim.py`
- `run_league_averages.py`

### Compatibility notes

- `SnoozleApiClient` is kept as a backwards-compatible alias of `MlbDataClient`
- `SnoozleApiError` is kept as a backwards-compatible alias of `MlbDataError`
- the Monte Carlo engine itself is unchanged in spirit; the main rewrite is the upstream data layer

### Behavioral differences from Snoozle

- if MLB has not published a complete lineup, the simulator now skips that game instead of relying on the old Snoozle projected feed
- fallback player stats are sourced from public StatsAPI season stats, not Snoozle's custom daily payload
- pitcher on-base allowed is reconstructed from public pitching stats when necessary
