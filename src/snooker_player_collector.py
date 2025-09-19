#!/usr/bin/env python3
"""
Comprehensive Snooker Player Collector - Build extensive player database
Similar to tennis-prediction-ai approach for maximum player coverage
"""

import pandas as pd
import numpy as np
import requests
import time
import json
from datetime import datetime, timedelta
import os
from snooker_elo_system import SnookerEloSystem

class SnookerPlayerCollector:
    """
    Comprehensive snooker player collector using multiple data sources
    Target: Build extensive player database like tennis model
    """

    def __init__(self):
        self.players = set()
        self.matches = []
        self.elo_system = SnookerEloSystem()
        self.api_base = "http://api.snooker.org/"
        self.headers = {
            'User-Agent': 'SnookerPredictionAI/1.0',
            'X-Requested-By': 'snooker-prediction-ai'
        }

        # Use WORKING tournament IDs from original collector (verified to work)
        self.all_tournaments = {
            # World Championships (verified working IDs)
            'world_championship': {
                2015: 466, 2016: 428, 2017: 465, 2018: 520, 2019: 580,
                2020: 645, 2021: 720, 2022: 819, 2023: 1030, 2024: 1460
            },
            # Triple Crown events (verified working IDs)
            'masters': {
                2015: 403, 2016: 444, 2017: 473, 2018: 524, 2019: 584,
                2020: 649, 2021: 724, 2022: 823, 2023: 1286, 2024: 1454
            },
            'uk_championship': {
                2015: 407, 2016: 434, 2017: 469, 2018: 516, 2019: 576,
                2020: 641, 2021: 716, 2022: 815, 2023: 1026, 2024: 1456
            },
            # Major ranking events (verified working IDs)
            'shanghai_masters': {
                2015: 398, 2016: 395, 2017: 501, 2018: 544, 2019: 604,
                2020: 669, 2021: 744, 2022: 843, 2023: 1050, 2024: 1480
            },
            'china_open': {
                2015: 430, 2016: 457, 2017: 492, 2018: 548, 2019: 608,
                2020: 673, 2021: 748, 2022: 847, 2023: 1054, 2024: 1484
            },
            'german_masters': {
                2015: 421, 2016: 448, 2017: 477, 2018: 528, 2019: 588,
                2020: 653, 2021: 728, 2022: 827, 2023: 1290, 2024: 1458
            },
            'welsh_open': {
                2015: 411, 2016: 438, 2017: 481, 2018: 532, 2019: 592,
                2020: 657, 2021: 732, 2022: 831, 2023: 1294, 2024: 1462
            }
        }

        # Additional tournaments to discover through API
        self.additional_tournaments = []

    def discover_additional_tournaments(self, start_year=2015, end_year=2024):
        """
        Discover additional tournaments through API exploration
        """
        print("🔍 Discovering additional tournaments...")
        discovered = []

        # Try to discover tournaments by checking ID ranges
        for year in range(start_year, end_year + 1):
            print(f"   📅 Checking {year}...")
            # Check around known tournament IDs for other tournaments
            base_ids = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]

            for base_id in base_ids:
                for offset in range(-10, 11):
                    tournament_id = base_id + offset
                    try:
                        url = f"{self.api_base}tournaments/{tournament_id}"
                        response = requests.get(url, headers=self.headers, timeout=5)
                        if response.status_code == 200:
                            tournament_info = response.json()
                            if tournament_info and 'Name' in tournament_info:
                                tournament_name = tournament_info['Name']
                                # Skip if we already have this tournament
                                is_known = False
                                for known_type, known_years in self.all_tournaments.items():
                                    if tournament_id in known_years.values():
                                        is_known = True
                                        break

                                if not is_known:
                                    discovered.append({
                                        'id': tournament_id,
                                        'name': tournament_name,
                                        'year': year
                                    })
                                    print(f"      ✅ Found: {tournament_name} (ID: {tournament_id})")

                        time.sleep(0.1)  # Rate limiting
                    except:
                        continue

        return discovered

    def collect_comprehensive_player_data(self, start_year=2015, end_year=2024):
        """
        Collect comprehensive player data from all tournaments
        Similar to tennis model's comprehensive approach
        """
        print("🎱 COMPREHENSIVE SNOOKER PLAYER COLLECTION")
        print("Building extensive player database from verified tournaments 2015-2024")
        print("=" * 70)

        all_matches = []
        total_players = set()
        tournament_count = 0

        # Process all verified tournaments
        for tournament_type, years_data in self.all_tournaments.items():
            print(f"\n🏆 Processing {tournament_type.replace('_', ' ').title()}...")

            for year in range(start_year, end_year + 1):
                if year in years_data:
                    tournament_id = years_data[year]
                    print(f"   📅 {year}: Tournament ID {tournament_id}")

                    try:
                        matches = self.fetch_tournament_matches(tournament_id, tournament_type, year)
                        if matches:
                            all_matches.extend(matches)
                            # Extract players from matches
                            for match in matches:
                                if 'winner' in match and match['winner']:
                                    total_players.add(match['winner'])
                                if 'loser' in match and match['loser']:
                                    total_players.add(match['loser'])
                            print(f"      ✅ Got {len(matches)} matches")
                            tournament_count += 1
                        else:
                            print(f"      ❌ No data")

                        # Rate limiting
                        time.sleep(0.5)

                    except Exception as e:
                        print(f"      ⚠️  Error: {str(e)}")
                        continue

        # Try to discover and process additional tournaments
        print(f"\n🔍 Discovering additional tournaments...")
        try:
            discovered = self.discover_additional_tournaments(start_year, end_year)
            print(f"   Found {len(discovered)} additional tournaments")

            for tournament in discovered[:10]:  # Limit to prevent overload
                try:
                    matches = self.fetch_tournament_matches(tournament['id'], 'ranking_event', tournament['year'])
                    if matches:
                        all_matches.extend(matches)
                        for match in matches:
                            if 'winner' in match and match['winner']:
                                total_players.add(match['winner'])
                            if 'loser' in match and match['loser']:
                                total_players.add(match['loser'])
                        print(f"      ✅ {tournament['name']}: {len(matches)} matches")
                        tournament_count += 1
                    time.sleep(0.3)
                except:
                    continue
        except:
            print("   ⚠️  Discovery failed, continuing with verified tournaments")

        print(f"\n📊 COLLECTION SUMMARY:")
        print(f"   🏆 Tournaments processed: {tournament_count}")
        print(f"   🎱 Total matches: {len(all_matches):,}")
        print(f"   👥 Total players: {len(total_players)}")
        print(f"   📅 Years covered: {start_year}-{end_year}")

        # Convert to DataFrame
        if all_matches:
            matches_df = pd.DataFrame(all_matches)
            print(f"\n✅ Successfully created comprehensive dataset!")
            return matches_df
        else:
            print(f"\n❌ No matches collected - check API access")
            return pd.DataFrame()

    def fetch_tournament_matches(self, tournament_id, tournament_type, year):
        """
        Fetch matches for a specific tournament
        """
        try:
            url = f"{self.api_base}tournaments/{tournament_id}/rounds"
            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code != 200:
                return []

            rounds_data = response.json()
            tournament_matches = []

            for round_info in rounds_data:
                round_id = round_info.get('ID')
                round_name = round_info.get('Name', 'Unknown Round')

                # Get matches for this round
                matches_url = f"{self.api_base}rounds/{round_id}/matches"
                matches_response = requests.get(matches_url, headers=self.headers, timeout=10)

                if matches_response.status_code == 200:
                    matches_data = matches_response.json()

                    for match in matches_data:
                        try:
                            # Extract comprehensive match data
                            match_data = self.extract_match_data(match, tournament_type, year, round_name)
                            if match_data:
                                tournament_matches.append(match_data)
                        except:
                            continue

                time.sleep(0.2)  # Rate limiting

            return tournament_matches

        except Exception as e:
            return []

    def extract_match_data(self, match, tournament_type, year, round_name):
        """
        Extract comprehensive match data including player names
        """
        try:
            # Get basic match info
            player1_info = match.get('Player1', {})
            player2_info = match.get('Player2', {})

            if not player1_info or not player2_info:
                return None

            # Extract player names (handle various formats)
            player1_name = self.extract_player_name(player1_info)
            player2_name = self.extract_player_name(player2_info)

            if not player1_name or not player2_name:
                return None

            # Determine winner/loser
            score1 = match.get('Score1', 0) or 0
            score2 = match.get('Score2', 0) or 0

            if score1 > score2:
                winner, loser = player1_name, player2_name
                frames_won, frames_lost = score1, score2
            elif score2 > score1:
                winner, loser = player2_name, player1_name
                frames_won, frames_lost = score2, score1
            else:
                return None  # Skip draws/unfinished matches

            # Create comprehensive match record
            match_record = {
                'date': self.parse_match_date(match.get('Date')),
                'tournament_name': f"{tournament_type.replace('_', ' ').title()} {year}",
                'tournament_type': tournament_type,
                'round': round_name,
                'best_of': self.determine_best_of(frames_won, frames_lost, tournament_type, round_name),
                'frames_to_win': (self.determine_best_of(frames_won, frames_lost, tournament_type, round_name) + 1) // 2,
                'winner': winner,
                'loser': loser,
                'frames_won': frames_won,
                'frames_lost': frames_lost,
                'year': year,
                # Additional stats (will be enhanced later)
                'winner_centuries': 0,
                'loser_centuries': 0,
                'winner_breaks_50_plus': 0,
                'loser_breaks_50_plus': 0,
                'winner_highest_break': 0,
                'loser_highest_break': 0,
                'match_duration_minutes': 120,  # Default estimate
                'tournament_prize_money': self.get_tournament_prize_money(tournament_type),
                'tournament_prestige': self.get_tournament_prestige(tournament_type)
            }

            return match_record

        except Exception as e:
            return None

    def extract_player_name(self, player_info):
        """
        Extract clean player name from API response
        """
        if isinstance(player_info, dict):
            # Try different name fields
            name = (player_info.get('Name') or
                   player_info.get('FullName') or
                   player_info.get('DisplayName') or
                   player_info.get('ShortName'))

            if name:
                return name.strip()

        elif isinstance(player_info, str):
            return player_info.strip()

        return None

    def parse_match_date(self, date_str):
        """
        Parse match date from various formats
        """
        if not date_str:
            return datetime.now().strftime('%Y%m%d')

        try:
            # Try different date formats
            for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y%m%d']:
                try:
                    dt = datetime.strptime(str(date_str), fmt)
                    return dt.strftime('%Y%m%d')
                except:
                    continue

            # If all formats fail, return current date
            return datetime.now().strftime('%Y%m%d')

        except:
            return datetime.now().strftime('%Y%m%d')

    def determine_best_of(self, frames_won, frames_lost, tournament_type, round_name):
        """
        Determine match format based on scores and tournament context
        """
        total_frames = frames_won + frames_lost

        # Common snooker formats
        if frames_won >= 18 or frames_lost >= 18:
            return 35  # World Championship Final
        elif frames_won >= 13 or frames_lost >= 13:
            return 25  # World Championship Semi
        elif frames_won >= 10 or frames_lost >= 10:
            return 19  # World Championship Quarter
        elif frames_won >= 9 or frames_lost >= 9:
            return 17  # Long format
        elif frames_won >= 6 or frames_lost >= 6:
            return 11  # Medium format
        elif frames_won >= 5 or frames_lost >= 5:
            return 9   # Standard format
        else:
            return 7   # Short format

    def get_tournament_prize_money(self, tournament_type):
        """
        Get typical prize money for tournament type
        """
        prize_money = {
            'world_championship': 2500000,
            'masters': 1000000,
            'uk_championship': 1000000,
            'shanghai_masters': 800000,
            'china_open': 800000,
            'german_masters': 400000,
            'welsh_open': 400000,
            'players_championship': 600000,
            'tour_championship': 600000,
            'champion_of_champions': 500000,
            'northern_ireland_open': 300000,
            'english_open': 300000,
            'european_masters': 300000,
            'wuhan_open': 400000,
            'hongkong_masters': 300000
        }
        return prize_money.get(tournament_type, 200000)

    def get_tournament_prestige(self, tournament_type):
        """
        Get tournament prestige level
        """
        if tournament_type in ['world_championship']:
            return 'highest'
        elif tournament_type in ['masters', 'uk_championship']:
            return 'major'
        elif tournament_type in ['shanghai_masters', 'china_open', 'players_championship', 'tour_championship']:
            return 'premium'
        else:
            return 'ranking'

    def enhance_with_head_to_head(self, matches_df):
        """
        Add head-to-head statistics
        """
        if matches_df.empty:
            return matches_df

        print("\n🔍 Enhancing with head-to-head statistics...")

        # Create H2H lookup
        h2h_stats = {}

        for _, match in matches_df.iterrows():
            winner = match['winner']
            loser = match['loser']

            # Create sorted pair for consistent lookup
            pair = tuple(sorted([winner, loser]))

            if pair not in h2h_stats:
                h2h_stats[pair] = {'total': 0, winner: 0, loser: 0}

            h2h_stats[pair]['total'] += 1
            h2h_stats[pair][winner] += 1

        # Add H2H features to matches
        h2h_features = []
        for _, match in matches_df.iterrows():
            winner = match['winner']
            loser = match['loser']
            pair = tuple(sorted([winner, loser]))

            if pair in h2h_stats:
                stats = h2h_stats[pair]
                total_matches = stats['total']
                winner_wins = stats.get(winner, 0)

                h2h_features.append({
                    'h2h_total_matches': total_matches,
                    'winner_h2h_wins': winner_wins,
                    'loser_h2h_wins': total_matches - winner_wins,
                    'winner_h2h_win_rate': winner_wins / total_matches if total_matches > 0 else 0.5
                })
            else:
                h2h_features.append({
                    'h2h_total_matches': 0,
                    'winner_h2h_wins': 0,
                    'loser_h2h_wins': 0,
                    'winner_h2h_win_rate': 0.5
                })

        # Add H2H features to DataFrame
        h2h_df = pd.DataFrame(h2h_features)
        enhanced_df = pd.concat([matches_df, h2h_df], axis=1)

        print(f"✅ Enhanced {len(enhanced_df)} matches with H2H data")
        return enhanced_df

    def save_player_list(self, matches_df, filepath='data/snooker_players.txt'):
        """
        Save comprehensive player list for reference
        """
        if matches_df.empty:
            return

        all_players = set()
        for _, match in matches_df.iterrows():
            if 'winner' in match and match['winner']:
                all_players.add(match['winner'])
            if 'loser' in match and match['loser']:
                all_players.add(match['loser'])

        # Sort players alphabetically
        sorted_players = sorted(all_players)

        # Save to file
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            for player in sorted_players:
                f.write(f"{player}\n")

        print(f"\n📝 Saved {len(sorted_players)} players to {filepath}")
        return sorted_players

def main():
    """
    Test the comprehensive player collector
    """
    collector = SnookerPlayerCollector()

    # Collect comprehensive data
    matches_df = collector.collect_comprehensive_player_data(start_year=2020, end_year=2025)  # Test with recent years first

    if not matches_df.empty:
        # Enhance with H2H
        enhanced_df = collector.enhance_with_head_to_head(matches_df)

        # Save data
        enhanced_df.to_csv('data/comprehensive_snooker_matches.csv', index=False)

        # Save player list
        players = collector.save_player_list(enhanced_df)

        print(f"\n🎯 COMPREHENSIVE COLLECTION COMPLETE!")
        print(f"   📊 Total matches: {len(enhanced_df):,}")
        print(f"   👥 Total players: {len(players)}")
        print(f"   💾 Saved to: data/comprehensive_snooker_matches.csv")
    else:
        print("❌ No data collected")

if __name__ == "__main__":
    main()