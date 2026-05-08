from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import requests

from mlb_simulator.api import LineupCard, LineupPlayer, MlbDataClient, parse_starting_lineups_html, parse_team_bullpen_stats_html


class MlbDataClientTest(unittest.TestCase):
    def test_client_initializes_session(self) -> None:
        fake_session = type("FakeSession", (), {"headers": {}, "close": lambda self: None})()
        with patch("mlb_simulator.api.requests.Session", return_value=fake_session) as mock_session:
            client = MlbDataClient()

        self.assertIs(client.session, fake_session)
        self.assertIn("Mozilla/5.0", client.session.headers["User-Agent"])
        mock_session.assert_called_once_with()

    def test_parse_starting_lineups_html_extracts_players_and_pitchers(self) -> None:
        html = """
        <html><body>
          <div>Pirates @ Mets</div>
          <a href="/player/paul-skenes-694973">Paul Skenes</a>
          <div>RHP</div>
          <a href="/player/freddy-peralta-642547">Freddy Peralta</a>
          <div>RHP</div>
          <div>PIT Lineup</div>
          <div>NYM Lineup</div>
          <div>1. <a href="/player/oneil-cruz-665833">Oneil Cruz</a> (L) CF</div>
          <div>2. <a href="/player/bryan-reynolds-668804">Bryan Reynolds</a> (S) LF</div>
          <div>3. <a href="/player/marcell-ozuna-542303">Marcell Ozuna</a> (R) DH</div>
          <div>4. <a href="/player/ryan-ohearn-656811">Ryan O'Hearn</a> (L) RF</div>
          <div>5. <a href="/player/jared-triolo-669707">Jared Triolo</a> (R) SS</div>
          <div>6. <a href="/player/spencer-horwitz-687462">Spencer Horwitz</a> (L) 1B</div>
          <div>7. <a href="/player/nick-gonzales-663853">Nick Gonzales</a> (R) 2B</div>
          <div>8. <a href="/player/henry-davis-680779">Henry Davis</a> (R) C</div>
          <div>9. <a href="/player/kebryan-hayes-663647">Ke'Bryan Hayes</a> (R) 3B</div>
          <div>1. <a href="/player/francisco-lindor-596019">Francisco Lindor</a> (S) SS</div>
          <div>2. <a href="/player/juan-soto-665742">Juan Soto</a> (L) LF</div>
          <div>3. <a href="/player/bo-bichette-666182">Bo Bichette</a> (R) 3B</div>
          <div>4. <a href="/player/jorge-polanco-593871">Jorge Polanco</a> (S) 1B</div>
          <div>5. <a href="/player/luis-robert-jr-673357">Luis Robert Jr.</a> (R) CF</div>
          <div>6. <a href="/player/brett-baty-683146">Brett Baty</a> (L) DH</div>
          <div>7. <a href="/player/marcus-semien-543760">Marcus Semien</a> (R) 2B</div>
          <div>8. <a href="/player/carson-benge-123456">Carson Benge</a> (L) RF</div>
          <div>9. <a href="/player/francisco-alvarez-682626">Francisco Alvarez</a> (R) C</div>
        </body></html>
        """

        cards = parse_starting_lineups_html(html)
        self.assertEqual(len(cards), 1)
        card = cards[0]
        self.assertEqual(card.away_team, "Pirates")
        self.assertEqual(card.home_team, "Mets")
        self.assertEqual(card.away_pitcher.player_id, 694973)
        self.assertEqual(card.home_pitcher.player_id, 642547)
        self.assertEqual(card.away_pitcher.throws, "R")
        self.assertEqual(len(card.away_lineup), 9)
        self.assertEqual(card.away_lineup[0].player_name, "Oneil Cruz")
        self.assertEqual(card.away_lineup[0].bats, "L")
        self.assertEqual(card.home_lineup[-1].player_id, 682626)

    def test_parse_starting_lineups_html_extracts_dom_matchup_cards(self) -> None:
        html = """
        <html><body>
          <div class="starting-lineups__matchup" data-gamePk="822745">
            <div class="starting-lineups__game">
              <div class="starting-lineups__team-names">
                <span class="starting-lineups__team-name starting-lineups__team-name--away">
                  <a class="starting-lineups__team-name--link" href="/braves" data-tri-code="ATL">Braves</a>
                </span>
                <span class="starting-lineups__team-name starting-lineups__team-name--at">@</span>
                <span class="starting-lineups__team-name starting-lineups__team-name--home">
                  <a class="starting-lineups__team-name--link" href="/nationals" data-tri-code="WSH">Nationals</a>
                </span>
              </div>
              <div class="starting-lineups__pitcher-name"><a class="starting-lineups__pitcher--link" href="/player/jr-ritchie-702275">JR Ritchie</a></div>
              <div class="starting-lineups__pitcher-name"><a class="starting-lineups__pitcher--link" href="/player/cade-cavalli-676917">Cade Cavalli</a></div>
            </div>
            <div class="starting-lineups__teams starting-lineups__teams--sm starting-lineups__teams--xl">
              <div class="starting-lineups__teams--header">
                <div class="starting-lineups__teams--away-head">ATL Lineup</div>
                <div class="starting-lineups__teams--home-head">WSH Lineup</div>
              </div>
              <ol class="starting-lineups__team starting-lineups__team--away">
                <li class="starting-lineups__player"><a class="starting-lineups__player--link" href="/player/ronald-acuna-jr-660670">Ronald Acuna Jr.</a><span class="starting-lineups__player--position"> (R) RF</span></li>
                <li class="starting-lineups__player"><a class="starting-lineups__player--link" href="/player/drake-baldwin-686948">Drake Baldwin</a></li>
                <li class="starting-lineups__player"><a class="starting-lineups__player--link" href="/player/matt-olson-621566">Matt Olson</a></li>
                <li class="starting-lineups__player"><a class="starting-lineups__player--link" href="/player/austin-riley-663586">Austin Riley</a></li>
                <li class="starting-lineups__player"><a class="starting-lineups__player--link" href="/player/ozzie-albies-645277">Ozzie Albies</a></li>
                <li class="starting-lineups__player"><a class="starting-lineups__player--link" href="/player/michael-harris-ii-669720">Michael Harris II</a></li>
                <li class="starting-lineups__player"><a class="starting-lineups__player--link" href="/player/sean-murphy-669221">Sean Murphy</a></li>
                <li class="starting-lineups__player"><a class="starting-lineups__player--link" href="/player/orlando-arcia-606115">Orlando Arcia</a></li>
                <li class="starting-lineups__player"><a class="starting-lineups__player--link" href="/player/jarred-kelenic-672284">Jarred Kelenic</a></li>
              </ol>
              <ol class="starting-lineups__team starting-lineups__team--home">
                <li class="starting-lineups__player"><a class="starting-lineups__player--link" href="/player/james-wood-695578">James Wood</a></li>
                <li class="starting-lineups__player"><a class="starting-lineups__player--link" href="/player/luis-garcia-jr-671277">Luis Garcia Jr.</a></li>
                <li class="starting-lineups__player"><a class="starting-lineups__player--link" href="/player/jose-tena-677950">Jose Tena</a></li>
                <li class="starting-lineups__player"><a class="starting-lineups__player--link" href="/player/cj-abrams-682928">CJ Abrams</a></li>
                <li class="starting-lineups__player"><a class="starting-lineups__player--link" href="/player/jacob-young-696285">Jacob Young</a></li>
                <li class="starting-lineups__player"><a class="starting-lineups__player--link" href="/player/daylen-lile-694336">Daylen Lile</a></li>
                <li class="starting-lineups__player"><a class="starting-lineups__player--link" href="/player/nasim-nunez-682812">Nasim Nunez</a></li>
                <li class="starting-lineups__player"><a class="starting-lineups__player--link" href="/player/jorbit-vivas-694815">Jorbit Vivas</a></li>
                <li class="starting-lineups__player"><a class="starting-lineups__player--link" href="/player/keibert-ruiz-660688">Keibert Ruiz</a></li>
              </ol>
            </div>
          </div>
        </body></html>
        """

        cards = parse_starting_lineups_html(html)
        self.assertEqual(len(cards), 1)
        card = cards[0]
        self.assertEqual(card.away_team, "Braves")
        self.assertEqual(card.home_team, "Nationals")
        self.assertEqual(card.away_pitcher.player_id, 702275)
        self.assertEqual(card.home_pitcher.player_id, 676917)
        self.assertEqual(len(card.away_lineup), 9)
        self.assertEqual(len(card.home_lineup), 9)
        self.assertEqual(card.away_lineup[0].player_name, "Ronald Acuna Jr.")
        self.assertEqual(card.away_lineup[0].bats, "R")
        self.assertEqual(card.home_lineup[-1].player_id, 660688)

    def test_parse_team_bullpen_stats_html_extracts_team_rows(self) -> None:
        html = """
        <html><body>
          <div>Reliever</div>
          <div>TEAM</div>
          <div>1</div>
          <div>Houston Astros Astros</div>
          <div>AL 3 8 5.66 24 0 0 0 5 5 105.0 102 70 66 22 7 66 113 1.60.254</div>
          <div>2</div>
          <div>Washington Nationals Nationals</div>
          <div>NL 7 5 5.57 23 0 0 0 5 13 114.2 116 74 71 22 7 57 90 1.51.259</div>
        </body></html>
        """

        records = parse_team_bullpen_stats_html(html)

        self.assertIn("astros", records)
        self.assertIn("nationals", records)
        self.assertEqual(records["astros"]["teamName"], "Houston Astros")
        self.assertEqual(records["astros"]["strikeOuts"], 113.0)
        self.assertAlmostEqual(records["astros"]["avg"], 0.254)
        self.assertGreater(records["nationals"]["battersFaced"], 0.0)

    def test_fetch_daily_games_parses_schedule_payload(self) -> None:
        payload = {
            "dates": [
                {
                    "games": [
                        {
                            "gamePk": 123,
                            "officialDate": "2026-04-23",
                            "teams": {
                                "away": {"team": {"name": "Chicago Cubs"}},
                                "home": {"team": {"name": "St. Louis Cardinals"}},
                            },
                        }
                    ]
                }
            ]
        }
        client = MlbDataClient()
        with patch.object(MlbDataClient, "_statsapi_get", return_value=payload) as mock_get:
            games = client.fetch_daily_games("2026-04-23")

        self.assertEqual(len(games), 1)
        self.assertEqual(games[0].game_id, "123")
        self.assertEqual(games[0].away_name, "Chicago Cubs")
        self.assertEqual(games[0].home_name, "St. Louis Cardinals")
        mock_get.assert_called_once()

    def test_fetch_game_uses_live_feed_when_official_lineups_exist(self) -> None:
        client = MlbDataClient()
        schedule_game = {
            "gamePk": 123,
            "officialDate": "2026-04-23",
            "teams": {
                "away": {"team": {"name": "Away Team"}, "probablePitcher": {"id": 1001}},
                "home": {"team": {"name": "Home Team"}, "probablePitcher": {"id": 2001}},
            },
        }
        live_feed = make_live_feed_with_lineups()

        with patch.object(MlbDataClient, "_get_schedule_game", return_value=schedule_game):
            with patch.object(MlbDataClient, "_statsapi_get", return_value=live_feed):
                game = client.fetch_game("123", lineup_source="auto")

        self.assertEqual(game.away_team, "Away Team")
        self.assertEqual(len(game.away_lineup), 9)
        self.assertEqual(game.away_lineup[0].player_name, "Away 1")
        self.assertEqual(game.away_lineup[0].bats, "L")
        self.assertEqual(game.away_pitching[0].player_name, "Away Pitcher")
        self.assertEqual(game.away_pitching[0].throws, "R")
        self.assertEqual(game.home_pitching[0].player_name, "Home Pitcher")

    def test_fetch_game_falls_back_to_starting_lineups_page(self) -> None:
        client = MlbDataClient()
        schedule_game = {
            "gamePk": 123,
            "officialDate": "2026-04-23",
            "teams": {
                "away": {"team": {"name": "Away Team"}},
                "home": {"team": {"name": "Home Team"}},
            },
        }
        live_feed = make_live_feed_without_lineups()
        card = LineupCard(
            away_team="Away Team",
            home_team="Home Team",
            away_lineup=[LineupPlayer(idx, f"Away {idx}", bats="L" if idx % 2 else "R") for idx in range(1, 10)],
            home_lineup=[LineupPlayer(100 + idx, f"Home {idx}", bats="R" if idx % 2 else "L") for idx in range(1, 10)],
            away_pitcher=LineupPlayer(1001, "Away Pitcher", throws="R"),
            home_pitcher=LineupPlayer(2001, "Home Pitcher", throws="L"),
        )

        def fake_statsapi_get(path, *, params=None):
            if path == "game/123/feed/live":
                return live_feed
            if path.endswith("/stats") and path.startswith("people/"):
                person_id = int(path.split("/")[1])
                if params["group"] == "hitting":
                    return {"stats": [{"splits": [{"stat": hitting_stat_record(person_id)}]}]}
                return {"stats": [{"splits": [{"stat": pitching_stat_record(person_id)}]}]}
            if path == "people":
                person_id = int(params["personIds"])
                group = "hitting" if "group=[hitting]" in str(params.get("hydrate")) else "pitching"
                if group == "hitting":
                    record = hitting_stat_record(person_id)
                else:
                    record = pitching_stat_record(person_id)
                return {"people": [{"id": person_id, "stats": [{"splits": [{"stat": record}]}]}]}
            raise AssertionError(f"Unexpected path: {path}")

        with patch.object(MlbDataClient, "_get_schedule_game", return_value=schedule_game):
            with patch.object(MlbDataClient, "_statsapi_get", side_effect=fake_statsapi_get):
                with patch.object(MlbDataClient, "_fetch_starting_lineups_cards", return_value=[card]):
                    game = client.fetch_game("123", lineup_source="auto")

        self.assertEqual(len(game.away_lineup), 9)
        self.assertEqual(game.away_lineup[0].player_id, 1)
        self.assertEqual(game.away_lineup[0].bats, "L")
        self.assertEqual(game.home_lineup[-1].player_id, 109)
        self.assertEqual(game.away_pitching[0].player_id, 1001)
        self.assertEqual(game.away_pitching[0].throws, "R")
        self.assertGreater(game.away_pitching[0].on_base, 0.0)

    def test_fetch_game_falls_back_when_live_feed_404s(self) -> None:
        client = MlbDataClient()
        schedule_game = {
            "gamePk": 123,
            "officialDate": "2026-04-23",
            "teams": {
                "away": {"team": {"name": "Away Team"}, "probablePitcher": {"id": 1001}},
                "home": {"team": {"name": "Home Team"}, "probablePitcher": {"id": 2001}},
            },
        }
        card = LineupCard(
            away_team="Away Team",
            home_team="Home Team",
            away_lineup=[LineupPlayer(idx, f"Away {idx}", bats="L" if idx % 2 else "R") for idx in range(1, 10)],
            home_lineup=[LineupPlayer(100 + idx, f"Home {idx}", bats="R" if idx % 2 else "L") for idx in range(1, 10)],
            away_pitcher=LineupPlayer(1001, "Away Pitcher", throws="R"),
            home_pitcher=LineupPlayer(2001, "Home Pitcher", throws="L"),
        )

        def fake_statsapi_get(path, *, params=None):
            if path == "game/123/feed/live":
                response = requests.Response()
                response.status_code = 404
                raise requests.HTTPError(response=response)
            if path.endswith("/stats") and path.startswith("people/"):
                person_id = int(path.split("/")[1])
                if params["group"] == "hitting":
                    return {"stats": [{"splits": [{"stat": hitting_stat_record(person_id)}]}]}
                return {"stats": [{"splits": [{"stat": pitching_stat_record(person_id)}]}]}
            if path == "people":
                person_id = int(params["personIds"])
                group = "hitting" if "group=[hitting]" in str(params.get("hydrate")) else "pitching"
                if group == "hitting":
                    record = hitting_stat_record(person_id)
                else:
                    record = pitching_stat_record(person_id)
                return {"people": [{"id": person_id, "stats": [{"splits": [{"stat": record}]}]}]}
            raise AssertionError(f"Unexpected path: {path}")

        with patch.object(MlbDataClient, "_get_schedule_game", return_value=schedule_game):
            with patch.object(MlbDataClient, "_statsapi_get", side_effect=fake_statsapi_get):
                with patch.object(MlbDataClient, "_fetch_starting_lineups_cards", return_value=[card]):
                    game = client.fetch_game("123", lineup_source="auto")

        self.assertEqual(len(game.away_lineup), 9)
        self.assertEqual(game.home_lineup[-1].player_id, 109)
        self.assertEqual(game.away_pitching[0].player_name, "Away Pitcher")
        self.assertEqual(game.home_pitching[0].player_name, "Home Pitcher")

    def test_fetch_player_stat_record_uses_player_specific_endpoint(self) -> None:
        client = MlbDataClient()

        with patch.object(MlbDataClient, "_statsapi_get", return_value={"stats": [{"splits": [{"stat": {"hits": 7}}]}]}) as mock_get:
            record = client._fetch_player_stat_record_from_api(695578, group="hitting", season=2026)

        self.assertEqual(record["hits"], 7)
        mock_get.assert_called_once_with(
            "people/695578/stats",
            params={
                "stats": "season",
                "group": "hitting",
                "season": 2026,
                "sportId": 1,
                "gameType": "R",
            },
        )

    def test_inspect_game_lineup_reports_set_from_starting_lineups(self) -> None:
        client = MlbDataClient()
        schedule_game = {
            "gamePk": 123,
            "officialDate": "2026-04-23",
            "teams": {
                "away": {"team": {"name": "Away Team"}, "probablePitcher": {"id": 1001, "fullName": "Away Pitcher"}},
                "home": {"team": {"name": "Home Team"}, "probablePitcher": {"id": 2001, "fullName": "Home Pitcher"}},
            },
        }
        card = LineupCard(
            away_team="Away Team",
            home_team="Home Team",
            away_lineup=[LineupPlayer(idx, f"Away {idx}", bats="L" if idx % 2 else "R") for idx in range(1, 10)],
            home_lineup=[LineupPlayer(100 + idx, f"Home {idx}", bats="R" if idx % 2 else "L") for idx in range(1, 10)],
            away_pitcher=LineupPlayer(1001, "Away Pitcher", throws="R"),
            home_pitcher=LineupPlayer(2001, "Home Pitcher", throws="L"),
        )

        with patch.object(MlbDataClient, "_get_schedule_game", return_value=schedule_game):
            with patch.object(MlbDataClient, "_statsapi_get", side_effect=requests.HTTPError(response=requests.Response())):
                with patch.object(MlbDataClient, "_fetch_live_feed_if_available", return_value=({}, True)):
                    with patch.object(MlbDataClient, "_fetch_starting_lineups_cards", return_value=[card]):
                        inspection = client.inspect_game_lineup("123", lineup_source="auto")

        self.assertTrue(inspection.is_set)
        self.assertEqual(inspection.source, "starting-lineups")
        self.assertEqual(len(inspection.away_lineup), 9)
        self.assertEqual(inspection.home_lineup[0].player_name, "Home 1")


    def test_fetch_starting_lineups_cards_merges_multiple_page_variants(self) -> None:
        client = MlbDataClient()
        partial_html = """
        <html><body>
          <div>Braves @ Nationals</div>
          <a href="/player/jr-ritchie-1">JR Ritchie</a>
          <a href="/player/cade-cavalli-2">Cade Cavalli</a>
          <div>ATL Lineup</div>
          <div>WSH Lineup</div>
          <div>1. TBD</div>
          <div>1. TBD</div>
        </body></html>
        """
        richer_html = """
        <html><body>
          <div>Braves @ Nationals</div>
          <a href="/player/jr-ritchie-1">JR Ritchie</a>
          <a href="/player/cade-cavalli-2">Cade Cavalli</a>
          <div>ATL Lineup</div>
          <div>WSH Lineup</div>
          <div>1. <a href="/player/ronald-acuna-jr-10">Ronald Acuna Jr.</a> (R) RF</div>
          <div>2. <a href="/player/drake-baldwin-11">Drake Baldwin</a> (L) C</div>
          <div>3. <a href="/player/matt-olson-12">Matt Olson</a> (L) 1B</div>
          <div>4. <a href="/player/austin-riley-13">Austin Riley</a> (R) 3B</div>
          <div>5. <a href="/player/ozzie-albies-14">Ozzie Albies</a> (S) 2B</div>
          <div>6. <a href="/player/michael-harris-ii-15">Michael Harris II</a> (L) CF</div>
          <div>7. <a href="/player/dominic-smith-16">Dominic Smith</a> (L) DH</div>
          <div>8. <a href="/player/mauricio-dubon-17">Mauricio Dubon</a> (R) SS</div>
          <div>9. <a href="/player/mike-yastrzemski-18">Mike Yastrzemski</a> (L) LF</div>
          <div>1. <a href="/player/james-wood-19">James Wood</a> (L) RF</div>
          <div>2. <a href="/player/luis-garcia-jr-20">Luis Garcia Jr.</a> (L) 1B</div>
          <div>3. <a href="/player/jose-tena-21">Jose Tena</a> (L) DH</div>
          <div>4. <a href="/player/cj-abrams-22">CJ Abrams</a> (L) SS</div>
          <div>5. <a href="/player/jacob-young-23">Jacob Young</a> (R) CF</div>
          <div>6. <a href="/player/daylen-lile-24">Daylen Lile</a> (L) LF</div>
          <div>7. <a href="/player/nasim-nunez-25">Nasim Nunez</a> (S) 2B</div>
          <div>8. <a href="/player/jorbit-vivas-26">Jorbit Vivas</a> (L) 3B</div>
          <div>9. <a href="/player/keibert-ruiz-27">Keibert Ruiz</a> (S) C</div>
        </body></html>
        """

        responses = [
            type('Response', (), {'text': partial_html, 'raise_for_status': lambda self: None, 'url': 'https://www.mlb.com/starting-lineups/2026-04-23', 'status_code': 200, 'headers': {'content-type': 'text/html'}})(),
            type('Response', (), {'text': richer_html, 'raise_for_status': lambda self: None, 'url': 'https://www.mlb.com/starting-lineups', 'status_code': 200, 'headers': {'content-type': 'text/html'}})(),
            type('Response', (), {'text': '', 'raise_for_status': lambda self: None, 'url': 'https://www.mlb.com/starting-lineups?date=2026-04-23', 'status_code': 200, 'headers': {'content-type': 'text/html'}})(),
        ]

        def fake_get(url, timeout=None, headers=None):
            return responses.pop(0)

        with patch.object(client.session, 'get', side_effect=fake_get):
            cards = client._fetch_starting_lineups_cards('2026-04-23')

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].away_team_key, 'braves')
        self.assertEqual(cards[0].home_team_key, 'nationals')
        self.assertEqual(len(cards[0].away_lineup), 9)
        self.assertEqual(len(cards[0].home_lineup), 9)


    def test_inspect_game_lineup_debug_note_includes_fetch_summary(self) -> None:
        client = MlbDataClient(debug_dir=None)
        schedule_game = {
            "gamePk": 123,
            "officialDate": "2026-04-23",
            "teams": {
                "away": {"team": {"name": "Atlanta Braves"}},
                "home": {"team": {"name": "Washington Nationals"}},
            },
        }
        debug_payload = {
            "game_date": "2026-04-23",
            "successful_fetches": 1,
            "total_cards": 0,
            "matchups": [],
            "sources": [
                {
                    "requested_url": "https://www.mlb.com/starting-lineups",
                    "final_url": "https://www.mlb.com/starting-lineups",
                    "status_code": 200,
                    "title": "Starting Lineups",
                }
            ],
        }

        with patch.object(MlbDataClient, "_get_schedule_game", return_value=schedule_game):
            with patch.object(MlbDataClient, "_fetch_live_feed_if_available", return_value=({}, True)):
                with patch.object(MlbDataClient, "_fetch_starting_lineups_cards", return_value=[]):
                    with patch.object(MlbDataClient, "get_lineups_debug", return_value=debug_payload):
                        inspection = client.inspect_game_lineup("123", lineup_source="auto")

        self.assertFalse(inspection.is_set)
        self.assertIn("Parsed 0 matchup cards", inspection.note)
        self.assertIn("status 200", inspection.note)
        self.assertIn("returned 404", inspection.note)



    def test_current_player_cache_rolls_previous_season_into_static_cache(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            client = MlbDataClient(player_cache_dir=str(cache_dir))
            stale_path = cache_dir / "current" / "hitting" / "42.json"
            stale_path.parent.mkdir(parents=True, exist_ok=True)
            stale_path.write_text(
                """{
  "version": 3,
  "player_id": 42,
  "group": "hitting",
  "season": 2025,
  "kind": "current",
  "fetched_on": "2025-09-30",
  "stat_record": {"hits": 50, "atBats": 200, "plateAppearances": 220, "baseOnBalls": 15, "homeRuns": 5, "doubles": 10, "triples": 1}
}
""",
                encoding="utf-8",
            )

            def fake_fetch(player_id, *, group, season):
                self.assertEqual(player_id, 42)
                self.assertEqual(group, "hitting")
                if season == 2025:
                    return hitting_stat_record(player_id)
                if season == 2026:
                    record = hitting_stat_record(player_id)
                    record = dict(record)
                    record["hits"] = 12
                    record["atBats"] = 40
                    record["plateAppearances"] = 45
                    return record
                raise AssertionError(f"Unexpected season {season}")

            with patch.object(MlbDataClient, "_fetch_player_stat_record_from_api", side_effect=fake_fetch):
                current = client._fetch_player_stat_record(42, "Test Batter", group="hitting", season=2026)

            static_path = cache_dir / "seasons" / "hitting" / "2025" / "42.json"
            self.assertTrue(static_path.exists())
            self.assertEqual(current["hits"], 12)
            current_payload = json.loads(stale_path.read_text(encoding="utf-8"))
            self.assertEqual(current_payload["season"], 2026)

    def test_resolved_batter_stats_uses_project_root_player_cache(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            client = MlbDataClient(player_cache_dir=str(cache_dir))

            def fake_fetch(player_id, *, group, season):
                self.assertEqual(group, "hitting")
                record = dict(hitting_stat_record(player_id))
                if season == 2026:
                    record["hits"] = 8
                    record["atBats"] = 32
                    record["plateAppearances"] = 36
                elif season == 2025:
                    record["hits"] = 150
                    record["atBats"] = 550
                    record["plateAppearances"] = 620
                elif season == 2024:
                    record["hits"] = 140
                    record["atBats"] = 530
                    record["plateAppearances"] = 600
                elif season == 2023:
                    record["hits"] = 130
                    record["atBats"] = 520
                    record["plateAppearances"] = 590
                return record

            with patch.object(MlbDataClient, "_fetch_player_stat_record_from_api", side_effect=fake_fetch):
                stats = client._resolve_batter_stats(99, "Test Batter", season=2026)

            profile_path = cache_dir / "profiles" / "hitting" / "2026" / "99.json"
            self.assertTrue(profile_path.exists())
            current_path = cache_dir / "current" / "hitting" / "99.json"
            self.assertTrue(current_path.exists())
            self.assertEqual(json.loads(current_path.read_text(encoding="utf-8"))["player_name"], "Test Batter")
            self.assertEqual(json.loads(profile_path.read_text(encoding="utf-8"))["player_name"], "Test Batter")
            self.assertIsNotNone(stats.resolved_profile)
            self.assertGreater(stats.resolved_profile.home_run_rate, 0.0)
            self.assertGreater(stats.resolved_profile.non_home_run_hit_rate, 0.0)



    def test_resolved_profiles_preserve_player_power_differences(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            client = MlbDataClient(player_cache_dir=str(cache_dir))

            def fake_fetch(player_id, *, group, season):
                self.assertEqual(group, "hitting")
                if player_id == 1:
                    if season == 2026:
                        return {
                            "hits": 6,
                            "doubles": 1,
                            "triples": 0,
                            "homeRuns": 2,
                            "baseOnBalls": 4,
                            "strikeOuts": 10,
                            "hitByPitch": 0,
                            "sacFlies": 0,
                            "groundIntoDoublePlay": 1,
                            "stolenBases": 1,
                            "atBats": 28,
                            "plateAppearances": 32,
                        }
                    return {
                        "hits": 150,
                        "doubles": 28,
                        "triples": 1,
                        "homeRuns": 42,
                        "baseOnBalls": 75,
                        "strikeOuts": 160,
                        "hitByPitch": 4,
                        "sacFlies": 4,
                        "groundIntoDoublePlay": 10,
                        "stolenBases": 5,
                        "atBats": 560,
                        "plateAppearances": 640,
                    }
                if season == 2026:
                    return {
                        "hits": 8,
                        "doubles": 2,
                        "triples": 1,
                        "homeRuns": 0,
                        "baseOnBalls": 2,
                        "strikeOuts": 4,
                        "hitByPitch": 0,
                        "sacFlies": 1,
                        "groundIntoDoublePlay": 1,
                        "stolenBases": 14,
                        "atBats": 32,
                        "plateAppearances": 35,
                    }
                return {
                    "hits": 165,
                    "doubles": 30,
                    "triples": 8,
                    "homeRuns": 6,
                    "baseOnBalls": 35,
                    "strikeOuts": 60,
                    "hitByPitch": 2,
                    "sacFlies": 5,
                    "groundIntoDoublePlay": 4,
                    "stolenBases": 28,
                    "atBats": 590,
                    "plateAppearances": 630,
                }

            with patch.object(MlbDataClient, "_fetch_player_stat_record_from_api", side_effect=fake_fetch):
                slugger = client._resolve_batter_stats(1, "Slugger", season=2026)
                contact = client._resolve_batter_stats(2, "Contact", season=2026)

            self.assertIsNotNone(slugger.resolved_profile)
            self.assertIsNotNone(contact.resolved_profile)
            self.assertGreater(slugger.resolved_profile.home_run_rate, contact.resolved_profile.home_run_rate)
            self.assertGreater(contact.resolved_profile.stolen_bases_per_600_pa, slugger.resolved_profile.stolen_bases_per_600_pa)

    def test_resolved_stats_persist_explicit_handedness_to_cache(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            client = MlbDataClient(player_cache_dir=str(cache_dir))

            with patch.object(MlbDataClient, "_fetch_player_stat_record_from_api", side_effect=lambda player_id, *, group, season: hitting_stat_record(player_id) if group == "hitting" else pitching_stat_record(player_id)):
                batter = client._resolve_batter_stats(77, "Switch Hitter", season=2026, bats="S")
                pitcher = client._resolve_pitcher_stats(88, "Lefty", season=2026, throws="L")

            self.assertEqual(batter.bats, "S")
            self.assertEqual(pitcher.throws, "L")
            hitter_payload = json.loads((cache_dir / "current" / "hitting" / "77.json").read_text(encoding="utf-8"))
            pitcher_payload = json.loads((cache_dir / "current" / "pitching" / "88.json").read_text(encoding="utf-8"))
            self.assertEqual(hitter_payload["stat_record"]["bats"], "S")
            self.assertEqual(pitcher_payload["stat_record"]["throws"], "L")

def make_live_feed_with_lineups() -> dict:
    away_players = {}
    home_players = {}
    game_players = {}
    away_batters = []
    home_batters = []

    for idx in range(1, 10):
        batter_id = idx
        away_batters.append(batter_id)
        away_players[f"ID{batter_id}"] = {
            "battingOrder": f"{idx}00",
            "seasonStats": {"batting": hitting_stat_record(batter_id)},
        }
        game_players[f"ID{batter_id}"] = {"fullName": f"Away {idx}", "batSide": {"code": "L" if idx % 2 else "R"}}

    for idx in range(1, 10):
        batter_id = 100 + idx
        home_batters.append(batter_id)
        home_players[f"ID{batter_id}"] = {
            "battingOrder": f"{idx}00",
            "seasonStats": {"batting": hitting_stat_record(batter_id)},
        }
        game_players[f"ID{batter_id}"] = {"fullName": f"Home {idx}", "batSide": {"code": "R" if idx % 2 else "L"}}

    away_players["ID1001"] = {"seasonStats": {"pitching": pitching_stat_record(1001)}}
    home_players["ID2001"] = {"seasonStats": {"pitching": pitching_stat_record(2001)}}
    game_players["ID1001"] = {"fullName": "Away Pitcher", "pitchHand": {"code": "R"}}
    game_players["ID2001"] = {"fullName": "Home Pitcher", "pitchHand": {"code": "L"}}

    return {
        "gameData": {"players": game_players},
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {"batters": away_batters, "pitchers": [1001], "players": away_players},
                    "home": {"batters": home_batters, "pitchers": [2001], "players": home_players},
                }
            }
        },
    }


def make_live_feed_without_lineups() -> dict:
    return {
        "gameData": {"players": {"ID1001": {"fullName": "Away Pitcher", "pitchHand": {"code": "R"}}, "ID2001": {"fullName": "Home Pitcher", "pitchHand": {"code": "L"}}}},
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {"batters": [], "pitchers": [1001], "players": {"ID1001": {"seasonStats": {"pitching": {}}}}},
                    "home": {"batters": [], "pitchers": [2001], "players": {"ID2001": {"seasonStats": {"pitching": {}}}}},
                }
            }
        },
    }


def hitting_stat_record(player_id: int) -> dict:
    return {
        "obp": ".333",
        "hitByPitch": 3,
        "sacFlies": 2,
        "atBats": 120,
        "hits": 36,
        "doubles": 8,
        "triples": 1,
        "homeRuns": 5,
        "baseOnBalls": 14,
        "plateAppearances": 140,
    }


def pitching_stat_record(player_id: int) -> dict:
    return {
        "obp": ".301",
        "inningsPitched": "51.2",
        "pitchesThrown": 924,
        "gamesStarted": 10,
        "hits": 42,
        "baseOnBalls": 17,
        "hitBatsmen": 2,
        "battersFaced": 213,
    }


if __name__ == "__main__":
    unittest.main()
