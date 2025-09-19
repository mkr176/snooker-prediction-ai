import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class SnookerAPICollector:
    """
    Collector for snooker data using various APIs and data sources.
    Focuses on tournament data, player statistics, and live scores.
    """

    def __init__(self, api_key: Optional[str] = None, rate_limit: float = 1.0):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.session = requests.Session()

        # Set up headers
        self.session.headers.update({
            'User-Agent': 'SnookerPredictionAI/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

        if api_key:
            self.session.headers.update({'X-API-Key': api_key})

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_live_rankings(self) -> List[Dict]:
        """
        Fetch current world snooker rankings.

        Returns:
            List of player ranking dictionaries
        """
        try:
            # Mock API response for demonstration
            rankings_data = []

            # Generate realistic ranking data
            top_players = [
                "Ronnie O'Sullivan", "Judd Trump", "Mark Selby", "Neil Robertson",
                "John Higgins", "Shaun Murphy", "Kyren Wilson", "Mark Williams",
                "Stuart Bingham", "Anthony McGill", "Jack Lisowski", "Zhao Xintong",
                "Luca Brecel", "Mark Allen", "Barry Hawkins", "Stephen Maguire"
            ]

            for i, player in enumerate(top_players, 1):
                points = max(1000000 - (i * 50000), 100000)
                rankings_data.append({
                    'rank': i,
                    'player_name': player,
                    'points': points,
                    'prize_money': points * 0.8,
                    'tournaments_played': 15 + (i % 5),
                    'updated': datetime.now().isoformat()
                })

            self.logger.info(f"Fetched rankings for {len(rankings_data)} players")
            time.sleep(self.rate_limit)

            return rankings_data

        except Exception as e:
            self.logger.error(f"Error fetching live rankings: {e}")
            return []

    def get_tournament_schedule(self, season: str = "2024") -> List[Dict]:
        """
        Get tournament schedule for the season.

        Args:
            season: Season year

        Returns:
            List of tournament schedules
        """
        try:
            tournaments = [
                {
                    'name': 'Masters',
                    'start_date': '2024-01-14',
                    'end_date': '2024-01-21',
                    'venue': 'Alexandra Palace, London',
                    'prize_fund': 800000,
                    'winner_prize': 250000,
                    'type': 'Invitational',
                    'field_size': 16
                },
                {
                    'name': 'Welsh Open',
                    'start_date': '2024-02-12',
                    'end_date': '2024-02-18',
                    'venue': 'ICC Wales, Newport',
                    'prize_fund': 405000,
                    'winner_prize': 70000,
                    'type': 'Ranking',
                    'field_size': 128
                },
                {
                    'name': 'Players Championship',
                    'start_date': '2024-03-04',
                    'end_date': '2024-03-10',
                    'venue': 'Telford International Centre',
                    'prize_fund': 380000,
                    'winner_prize': 125000,
                    'type': 'Ranking',
                    'field_size': 16
                },
                {
                    'name': 'World Championship',
                    'start_date': '2024-04-20',
                    'end_date': '2024-05-06',
                    'venue': 'Crucible Theatre, Sheffield',
                    'prize_fund': 2395000,
                    'winner_prize': 500000,
                    'type': 'Ranking',
                    'field_size': 32
                },
                {
                    'name': 'UK Championship',
                    'start_date': '2024-11-23',
                    'end_date': '2024-12-01',
                    'venue': 'York Barbican',
                    'prize_fund': 1200000,
                    'winner_prize': 250000,
                    'type': 'Ranking',
                    'field_size': 128
                }
            ]

            self.logger.info(f"Fetched {len(tournaments)} tournament schedules")
            time.sleep(self.rate_limit)

            return tournaments

        except Exception as e:
            self.logger.error(f"Error fetching tournament schedule: {e}")
            return []

    def get_player_form(self, player_name: str, last_n_matches: int = 10) -> Dict:
        """
        Get recent form data for a player.

        Args:
            player_name: Name of the player
            last_n_matches: Number of recent matches to analyze

        Returns:
            Player form data
        """
        try:
            # Mock recent match results
            import random

            recent_matches = []
            opponents = ["Player A", "Player B", "Player C", "Player D", "Player E"]

            wins = 0
            for i in range(last_n_matches):
                won = random.choice([True, False])
                if won:
                    wins += 1
                    score = f"{random.randint(4, 6)}-{random.randint(0, 3)}"
                else:
                    score = f"{random.randint(0, 3)}-{random.randint(4, 6)}"

                recent_matches.append({
                    'opponent': random.choice(opponents),
                    'result': 'W' if won else 'L',
                    'score': score,
                    'tournament': 'Sample Tournament',
                    'date': (datetime.now() - timedelta(days=i*7)).date().isoformat(),
                    'break_scores': [random.randint(50, 147) for _ in range(random.randint(1, 4))]
                })

            form_data = {
                'player_name': player_name,
                'matches_analyzed': last_n_matches,
                'wins': wins,
                'losses': last_n_matches - wins,
                'win_percentage': (wins / last_n_matches) * 100,
                'recent_matches': recent_matches,
                'form_string': ''.join(['W' if m['result'] == 'W' else 'L' for m in recent_matches[:5]]),
                'average_break': sum([sum(m['break_scores']) / len(m['break_scores']) for m in recent_matches]) / last_n_matches,
                'updated': datetime.now().isoformat()
            }

            time.sleep(self.rate_limit)
            return form_data

        except Exception as e:
            self.logger.error(f"Error fetching form for {player_name}: {e}")
            return {}

    def get_live_scores(self, tournament: str = None) -> List[Dict]:
        """
        Get live match scores.

        Args:
            tournament: Specific tournament to filter (optional)

        Returns:
            List of live match data
        """
        try:
            # Mock live match data
            live_matches = [
                {
                    'match_id': 'M001',
                    'tournament': 'UK Championship',
                    'round': 'Quarter-Final',
                    'player1': 'Ronnie OSullivan',
                    'player2': 'Judd Trump',
                    'score1': 4,
                    'score2': 3,
                    'status': 'In Progress',
                    'current_break': 45,
                    'current_player': 'Ronnie OSullivan',
                    'session': 'Afternoon',
                    'table': 1,
                    'start_time': '14:30',
                    'estimated_finish': '18:00'
                },
                {
                    'match_id': 'M002',
                    'tournament': 'UK Championship',
                    'round': 'Quarter-Final',
                    'player1': 'Mark Selby',
                    'player2': 'Neil Robertson',
                    'score1': 2,
                    'score2': 5,
                    'status': 'In Progress',
                    'current_break': 0,
                    'current_player': 'Neil Robertson',
                    'session': 'Evening',
                    'table': 2,
                    'start_time': '19:30',
                    'estimated_finish': '22:30'
                }
            ]

            if tournament:
                live_matches = [m for m in live_matches if m['tournament'] == tournament]

            self.logger.info(f"Fetched {len(live_matches)} live matches")
            time.sleep(self.rate_limit)

            return live_matches

        except Exception as e:
            self.logger.error(f"Error fetching live scores: {e}")
            return []

    def get_betting_odds(self, match_id: str = None) -> List[Dict]:
        """
        Get betting odds for matches (mock implementation).

        Args:
            match_id: Specific match ID (optional)

        Returns:
            List of betting odds
        """
        try:
            # Mock betting odds data
            import random

            odds_data = [
                {
                    'match_id': 'M001',
                    'player1': 'Ronnie OSullivan',
                    'player2': 'Judd Trump',
                    'odds_player1': round(random.uniform(1.5, 3.0), 2),
                    'odds_player2': round(random.uniform(1.5, 3.0), 2),
                    'bookmaker': 'Sample Bookmaker',
                    'market': 'Match Winner',
                    'updated': datetime.now().isoformat()
                },
                {
                    'match_id': 'M002',
                    'player1': 'Mark Selby',
                    'player2': 'Neil Robertson',
                    'odds_player1': round(random.uniform(1.5, 3.0), 2),
                    'odds_player2': round(random.uniform(1.5, 3.0), 2),
                    'bookmaker': 'Sample Bookmaker',
                    'market': 'Match Winner',
                    'updated': datetime.now().isoformat()
                }
            ]

            if match_id:
                odds_data = [o for o in odds_data if o['match_id'] == match_id]

            time.sleep(self.rate_limit)
            return odds_data

        except Exception as e:
            self.logger.error(f"Error fetching betting odds: {e}")
            return []

    def collect_all_data(self, save_to_file: bool = True) -> Dict:
        """
        Collect all available data types.

        Args:
            save_to_file: Whether to save data to files

        Returns:
            Dictionary containing all collected data
        """
        collected_data = {}

        try:
            # Collect data in parallel where possible
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self.get_live_rankings): 'rankings',
                    executor.submit(self.get_tournament_schedule): 'tournaments',
                    executor.submit(self.get_live_scores): 'live_scores',
                    executor.submit(self.get_betting_odds): 'odds'
                }

                for future in as_completed(futures):
                    data_type = futures[future]
                    try:
                        collected_data[data_type] = future.result()
                    except Exception as e:
                        self.logger.error(f"Error collecting {data_type}: {e}")
                        collected_data[data_type] = []

            # Get form data for top 10 players
            if collected_data.get('rankings'):
                top_players = collected_data['rankings'][:10]
                form_data = {}

                for player in top_players:
                    form = self.get_player_form(player['player_name'])
                    form_data[player['player_name']] = form

                collected_data['player_form'] = form_data

            if save_to_file:
                self._save_collected_data(collected_data)

            return collected_data

        except Exception as e:
            self.logger.error(f"Error in collect_all_data: {e}")
            return {}

    def _save_collected_data(self, data: Dict):
        """
        Save collected data to files.

        Args:
            data: Dictionary of collected data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for data_type, content in data.items():
            if not content:
                continue

            try:
                # Save as JSON
                json_filename = f"../data/raw/{data_type}_{timestamp}.json"
                with open(json_filename, 'w') as f:
                    json.dump(content, f, indent=2, default=str)

                # Save as CSV for tabular data
                if isinstance(content, list) and content:
                    csv_filename = f"../data/raw/{data_type}_{timestamp}.csv"
                    df = pd.DataFrame(content)
                    df.to_csv(csv_filename, index=False)

                self.logger.info(f"Saved {data_type} data with {len(content) if isinstance(content, list) else 'N/A'} records")

            except Exception as e:
                self.logger.error(f"Error saving {data_type} data: {e}")


def main():
    """Main function to demonstrate the API collector."""
    collector = SnookerAPICollector(rate_limit=0.5)

    print("Starting data collection...")

    # Collect all data
    all_data = collector.collect_all_data(save_to_file=True)

    print("\nData collection summary:")
    for data_type, content in all_data.items():
        if isinstance(content, list):
            print(f"- {data_type}: {len(content)} records")
        elif isinstance(content, dict):
            print(f"- {data_type}: {len(content)} entries")

    print("\nData collection complete!")


if __name__ == "__main__":
    main()