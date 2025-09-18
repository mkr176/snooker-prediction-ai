#!/usr/bin/env python3
"""
Snooker Data Collector - Generate professional snooker match data
Adapted from tennis system for snooker-specific features and tournaments
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import json
from datetime import datetime, timedelta
import os
from snooker_elo_system import SnookerEloSystem

class SnookerDataCollector:
    """
    Collect and generate snooker match data for prediction model training
    Features snooker-specific statistics and tournament structures
    """

    def __init__(self):
        self.matches = []
        self.players = set()
        self.elo_system = SnookerEloSystem()

        # Professional snooker players (current and recent)
        self.top_players = [
            'Ronnie O\'Sullivan', 'Judd Trump', 'Mark Selby', 'Neil Robertson',
            'Kyren Wilson', 'Mark Williams', 'John Higgins', 'Stuart Bingham',
            'Shaun Murphy', 'Mark Allen', 'Barry Hawkins', 'Joe Perry',
            'Anthony McGill', 'Luca Brecel', 'Zhao Xintong', 'Jack Lisowski',
            'Gary Wilson', 'Tom Ford', 'Matthew Selt', 'David Gilbert',
            'Yan Bingtao', 'Ali Carter', 'Stephen Maguire', 'Ryan Day',
            'Zhou Yuelong', 'Ricky Walden', 'Matthew Stevens', 'Martin Gould',
            'Liang Wenbo', 'Scott Donaldson', 'Ding Junhui', 'Thepchaiya Un-Nooh',
            'Robert Milkins', 'Sam Craigie', 'Chris Wakelin', 'Jordan Brown',
            'Noppon Saengkham', 'Andrew Higginson', 'Ben Woollaston', 'Jimmy Robertson',
            'Hossein Vafaei', 'Yuan Sijun', 'Tian Pengfei', 'Jamie Jones',
            'Mitchell Mann', 'Lyu Haotian', 'Oliver Lines', 'Jamie Clarke',
            'Graeme Dott', 'Ken Doherty', 'Marco Fu', 'Michael Holt'
        ]

        # Snooker tournament types and their characteristics
        self.tournaments = {
            'world_championship': {
                'best_of': [35, 33, 31, 25, 19],  # Different rounds
                'weight': 50,
                'frequency': 1,  # Once per year
                'prestige': 'highest'
            },
            'masters': {
                'best_of': [19, 17, 11, 11, 9],
                'weight': 35,
                'frequency': 1,
                'prestige': 'very_high'
            },
            'uk_championship': {
                'best_of': [19, 17, 13, 11, 9, 7],
                'weight': 35,
                'frequency': 1,
                'prestige': 'very_high'
            },
            'champion_of_champions': {
                'best_of': [11, 9, 7],
                'weight': 30,
                'frequency': 1,
                'prestige': 'high'
            },
            'players_championship': {
                'best_of': [11, 9, 7],
                'weight': 25,
                'frequency': 1,
                'prestige': 'high'
            },
            'tour_championship': {
                'best_of': [19, 11, 9, 7],
                'weight': 25,
                'frequency': 1,
                'prestige': 'high'
            },
            'ranking_event': {
                'best_of': [11, 9, 7, 5],
                'weight': 20,
                'frequency': 15,  # Multiple ranking events per year
                'prestige': 'medium'
            },
            'invitational': {
                'best_of': [9, 7, 5],
                'weight': 15,
                'frequency': 8,
                'prestige': 'medium'
            }
        }

    def generate_snooker_dataset(self, num_matches=25000):
        """Generate comprehensive snooker dataset"""
        print("🎱 GENERATING SNOOKER DATASET")
        print("Professional snooker prediction data generation")
        print("=" * 60)

        matches = []
        players_used = set()

        # Generate matches for each tournament type
        for tournament_type, config in self.tournaments.items():
            matches_for_tournament = int(num_matches * config['frequency'] / sum(t['frequency'] for t in self.tournaments.values()))

            print(f"📊 Generating {matches_for_tournament:,} matches for {tournament_type}")

            for _ in range(matches_for_tournament):
                match = self.generate_single_match(tournament_type, config)
                if match:
                    matches.append(match)
                    players_used.add(match['winner'])
                    players_used.add(match['loser'])

        self.matches = matches
        self.players = players_used

        print(f"\n✅ SNOOKER DATASET GENERATED!")
        print(f"   📊 Total matches: {len(matches):,}")
        print(f"   🎱 Players: {len(players_used):,}")
        print(f"   🏆 Tournament types: {len(self.tournaments)}")

        return pd.DataFrame(matches)

    def generate_single_match(self, tournament_type, config):
        """Generate a single snooker match with realistic statistics"""

        # Select two players
        player1 = np.random.choice(self.top_players)
        player2 = np.random.choice([p for p in self.top_players if p != player1])

        # Determine match format
        best_of = np.random.choice(config['best_of'])
        frames_to_win = (best_of + 1) // 2

        # Generate realistic ELO-based outcome
        elo1 = self.elo_system.get_player_rating(player1, tournament_type)
        elo2 = self.elo_system.get_player_rating(player2, tournament_type)

        win_prob = 1 / (1 + 10 ** ((elo2 - elo1) / 400))

        # Determine winner
        if np.random.random() < win_prob:
            winner = player1
            loser = player2
            winner_elo = elo1
            loser_elo = elo2
        else:
            winner = player2
            loser = player1
            winner_elo = elo2
            loser_elo = elo1

        # Generate frame scores (snooker-specific)
        frames_won, frames_lost = self.generate_frame_score(frames_to_win, best_of, win_prob)

        # Generate snooker-specific statistics
        match_stats = self.generate_match_statistics(winner, loser, frames_won, frames_lost, tournament_type)

        # Create match record
        match = {
            'date': self.generate_random_date(),
            'tournament_name': self.get_tournament_name(tournament_type),
            'tournament_type': tournament_type,
            'round': self.generate_round(tournament_type),
            'best_of': best_of,
            'frames_to_win': frames_to_win,

            # Players
            'winner': winner,
            'loser': loser,

            # Score
            'frames_won': frames_won,
            'frames_lost': frames_lost,

            # Match statistics
            **match_stats,

            # ELO ratings (at time of match)
            'winner_elo': winner_elo,
            'loser_elo': loser_elo,
            'elo_difference': winner_elo - loser_elo,
        }

        # Update ELO ratings
        self.elo_system.update_ratings(winner, loser, tournament_type, frames_won, frames_lost)

        return match

    def generate_frame_score(self, frames_to_win, best_of, win_prob):
        """Generate realistic frame scores for snooker match"""

        # Adjust win probability based on format (longer matches are more predictable)
        if best_of >= 25:  # World Championship later rounds
            win_prob = 0.5 + (win_prob - 0.5) * 1.3
        elif best_of >= 17:  # Semi-finals, finals
            win_prob = 0.5 + (win_prob - 0.5) * 1.2
        else:  # Shorter matches
            win_prob = 0.5 + (win_prob - 0.5) * 1.1

        win_prob = max(0.1, min(0.9, win_prob))

        # Simulate frame by frame
        winner_frames = 0
        loser_frames = 0
        total_frames = 0

        while winner_frames < frames_to_win and loser_frames < frames_to_win:
            if np.random.random() < win_prob:
                winner_frames += 1
            else:
                loser_frames += 1
            total_frames += 1

        return winner_frames, loser_frames

    def generate_match_statistics(self, winner, loser, frames_won, frames_lost, tournament_type):
        """Generate realistic snooker match statistics"""

        total_frames = frames_won + frames_lost

        # Break statistics (snooker-specific)
        winner_stats = self.generate_player_stats(winner, frames_won, total_frames, True)
        loser_stats = self.generate_player_stats(loser, frames_lost, total_frames, False)

        return {
            # Break statistics
            'winner_centuries': winner_stats['centuries'],
            'loser_centuries': loser_stats['centuries'],
            'winner_breaks_50_plus': winner_stats['breaks_50_plus'],
            'loser_breaks_50_plus': loser_stats['breaks_50_plus'],
            'winner_highest_break': winner_stats['highest_break'],
            'loser_highest_break': loser_stats['highest_break'],

            # Pot success rates
            'winner_pot_success': winner_stats['pot_success'],
            'loser_pot_success': loser_stats['pot_success'],
            'winner_long_pot_success': winner_stats['long_pot_success'],
            'loser_long_pot_success': loser_stats['long_pot_success'],

            # Safety play
            'winner_safety_success': winner_stats['safety_success'],
            'loser_safety_success': loser_stats['safety_success'],

            # Frame control
            'winner_avg_frame_time': winner_stats['avg_frame_time'],
            'loser_avg_frame_time': loser_stats['avg_frame_time'],
            'winner_first_visit_clearance': winner_stats['first_visit_clearance'],
            'loser_first_visit_clearance': loser_stats['first_visit_clearance'],

            # Match duration
            'match_duration_minutes': total_frames * np.random.normal(25, 5),

            # Tournament context
            'tournament_prize_money': self.get_tournament_prize_money(tournament_type),
            'tournament_prestige': self.tournaments[tournament_type]['prestige']
        }

    def generate_player_stats(self, player, frames_won, total_frames, is_winner):
        """Generate individual player statistics for the match"""

        # Base stats influenced by player skill (simplified)
        base_skill = 0.75 if player in self.top_players[:16] else 0.65
        if is_winner:
            base_skill += 0.1  # Winner bonus

        # Break building
        centuries = max(0, int(np.random.poisson(frames_won * base_skill * 0.3)))
        breaks_50_plus = centuries + max(0, int(np.random.poisson(frames_won * base_skill * 0.8)))

        if centuries > 0:
            highest_break = np.random.randint(100, 147)
        elif breaks_50_plus > 0:
            highest_break = np.random.randint(50, 99)
        else:
            highest_break = np.random.randint(20, 49)

        return {
            'centuries': centuries,
            'breaks_50_plus': breaks_50_plus,
            'highest_break': highest_break,
            'pot_success': np.random.normal(base_skill * 85, 5),
            'long_pot_success': np.random.normal(base_skill * 65, 8),
            'safety_success': np.random.normal(base_skill * 80, 6),
            'avg_frame_time': np.random.normal(20 + (1-base_skill) * 10, 3),
            'first_visit_clearance': np.random.normal(base_skill * 40, 8)
        }

    def generate_random_date(self):
        """Generate random date for matches (last 3 years)"""
        start_date = datetime.now() - timedelta(days=3*365)
        end_date = datetime.now()

        random_date = start_date + timedelta(
            days=np.random.randint(0, (end_date - start_date).days)
        )

        return random_date.strftime('%Y%m%d')

    def get_tournament_name(self, tournament_type):
        """Get realistic tournament name based on type"""
        tournament_names = {
            'world_championship': 'World Snooker Championship',
            'masters': 'Masters',
            'uk_championship': 'UK Championship',
            'champion_of_champions': 'Champion of Champions',
            'players_championship': 'Players Championship',
            'tour_championship': 'Tour Championship',
            'ranking_event': np.random.choice([
                'Shanghai Masters', 'UK Open', 'European Masters', 'German Masters',
                'World Open', 'China Open', 'International Championship', 'Northern Ireland Open'
            ]),
            'invitational': np.random.choice([
                'Championship League', 'Shoot Out', 'Six Red World Championship',
                'Paul Hunter Classic', 'Gibraltar Open'
            ])
        }
        return tournament_names[tournament_type]

    def generate_round(self, tournament_type):
        """Generate appropriate round for tournament type"""
        if tournament_type == 'world_championship':
            return np.random.choice(['Qualifying', 'First Round', 'Second Round', 'Quarter-Final', 'Semi-Final', 'Final'])
        elif tournament_type in ['masters', 'uk_championship']:
            return np.random.choice(['First Round', 'Quarter-Final', 'Semi-Final', 'Final'])
        else:
            return np.random.choice(['First Round', 'Second Round', 'Quarter-Final', 'Semi-Final', 'Final'])

    def get_tournament_prize_money(self, tournament_type):
        """Get typical prize money for tournament type"""
        prize_money = {
            'world_championship': 2500000,  # £2.5M total
            'masters': 725000,              # £725K total
            'uk_championship': 1000000,     # £1M total
            'champion_of_champions': 440000, # £440K total
            'players_championship': 380000,  # £380K total
            'tour_championship': 375000,    # £375K total
            'ranking_event': 400000,        # £400K average
            'invitational': 200000          # £200K average
        }
        return prize_money.get(tournament_type, 200000)

    def enhance_with_head_to_head(self, matches_df):
        """Add head-to-head statistics"""
        print("📊 CALCULATING HEAD-TO-HEAD STATISTICS")
        print("Building historical H2H records...")

        h2h_records = {}
        enhanced_matches = []

        for idx, match in matches_df.iterrows():
            winner = match['winner']
            loser = match['loser']

            # Create H2H key
            h2h_key = tuple(sorted([winner, loser]))

            if h2h_key not in h2h_records:
                h2h_records[h2h_key] = {'total': 0, winner: 0, loser: 0}

            # Add H2H features to match
            match_enhanced = match.copy()
            h2h_data = h2h_records[h2h_key]

            match_enhanced['h2h_total_matches'] = h2h_data['total']
            match_enhanced['winner_h2h_wins'] = h2h_data.get(winner, 0)
            match_enhanced['loser_h2h_wins'] = h2h_data.get(loser, 0)

            if h2h_data['total'] > 0:
                match_enhanced['winner_h2h_win_rate'] = h2h_data.get(winner, 0) / h2h_data['total']
            else:
                match_enhanced['winner_h2h_win_rate'] = 0.5

            enhanced_matches.append(match_enhanced)

            # Update H2H record
            h2h_records[h2h_key]['total'] += 1
            h2h_records[h2h_key][winner] = h2h_records[h2h_key].get(winner, 0) + 1

        enhanced_df = pd.DataFrame(enhanced_matches)
        print(f"✅ H2H statistics added for {len(h2h_records):,} player pairs")

        return enhanced_df

    def save_snooker_data(self, matches_df):
        """Save snooker dataset"""
        print(f"\n💾 SAVING SNOOKER DATASET")

        # Save main dataset
        matches_df.to_csv('../data/snooker_matches.csv', index=False)
        print(f"✅ Snooker matches: ../data/snooker_matches.csv")

        # Build and save ELO system
        print("🏆 Building ELO system from snooker data...")
        self.elo_system.build_from_match_data(matches_df)

        # Save ELO system
        os.makedirs('../models', exist_ok=True)
        self.elo_system.save_system('../models/snooker_elo_system.pkl')
        print(f"✅ Snooker ELO system: ../models/snooker_elo_system.pkl")

        # Show top players
        print(f"\n🏆 TOP 10 SNOOKER PLAYERS BY ELO:")
        top_players = self.elo_system.get_top_players(10)
        for i, (player, elo) in enumerate(top_players, 1):
            print(f"   {i:2d}. {player:<25} {elo:.0f}")

        return True

def main():
    """Generate comprehensive snooker dataset"""
    print("🎱 SNOOKER PREDICTION DATA GENERATION")
    print("Professional snooker match dataset creation")
    print("Adapted from tennis system for snooker features")
    print("=" * 60)

    collector = SnookerDataCollector()

    # Generate dataset (reduced size for faster completion)
    matches_df = collector.generate_snooker_dataset(5000)

    # Enhance with head-to-head
    enhanced_df = collector.enhance_with_head_to_head(matches_df)

    # Save data
    collector.save_snooker_data(enhanced_df)

    print(f"\n🚀 SNOOKER DATASET GENERATION COMPLETE!")
    print(f"✅ Ready to train snooker prediction model!")
    print(f"📊 Dataset: {len(enhanced_df):,} professional snooker matches")
    print(f"🎱 Players: {len(collector.players):,} professional players")
    print(f"🎯 Features: Break building, pot success, safety play, tournament prestige")

if __name__ == "__main__":
    main()