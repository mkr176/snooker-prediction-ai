#!/usr/bin/env python3
"""
Snooker ELO Rating System - Adapted from Tennis for Snooker Predictions
Handles tournament-specific ratings and player statistics for professional snooker
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict
import joblib

class SnookerEloSystem:
    """
    ELO Rating System for professional snooker players
    Adapted from tennis system for snooker-specific features
    """

    def __init__(self, initial_rating=1500, k_factor=32):
        self.initial_rating = initial_rating
        self.k_factor = k_factor

        # Player ratings and statistics
        self.player_elo = {}
        self.player_stats = {}

        # Tournament-specific ratings (similar to tennis surfaces)
        self.tournament_elo = {}

        # Tournament weightings (snooker equivalents)
        self.tournament_weights = {
            'world_championship': 50,    # World Championship
            'masters': 35,               # Masters
            'uk_championship': 35,       # UK Championship
            'champion_of_champions': 30, # Champion of Champions
            'players_championship': 25,  # Players Championship
            'tour_championship': 25,     # Tour Championship
            'ranking_event': 20,         # Other ranking events
            'invitational': 15,          # Invitational tournaments
            'qualifying': 10             # Qualifying events
        }

        # Historical ELO progression
        self.elo_history = {}

    def get_player_rating(self, player, tournament_type='ranking_event'):
        """Get current ELO rating for a player in specific tournament type"""
        if player not in self.player_elo:
            self.player_elo[player] = self.initial_rating

        if tournament_type not in self.tournament_elo:
            self.tournament_elo[tournament_type] = {}

        if player not in self.tournament_elo[tournament_type]:
            self.tournament_elo[tournament_type][player] = self.initial_rating

        return self.tournament_elo[tournament_type][player]

    def calculate_expected_score(self, rating1, rating2):
        """Calculate expected score using ELO formula"""
        return 1 / (1 + 10 ** ((rating2 - rating1) / 400))

    def update_ratings(self, winner, loser, tournament_type='ranking_event',
                      frames_won=None, frames_lost=None, match_date=None):
        """
        Update ELO ratings after a match
        """
        # Initialize ratings if needed
        if winner not in self.player_elo:
            self.player_elo[winner] = self.initial_rating
        if loser not in self.player_elo:
            self.player_elo[loser] = self.initial_rating

        # Get current ratings
        winner_rating = self.get_player_rating(winner, tournament_type)
        loser_rating = self.get_player_rating(loser, tournament_type)

        # Calculate expected scores
        winner_expected = self.calculate_expected_score(winner_rating, loser_rating)
        loser_expected = 1 - winner_expected

        # Determine K-factor based on tournament importance
        k = self.k_factor * (self.tournament_weights.get(tournament_type, 20) / 20)

        # Frame difference bonus (snooker-specific)
        frame_diff_bonus = 1.0
        if frames_won and frames_lost:
            frame_diff = abs(frames_won - frames_lost)
            # Bonus for convincing wins (similar to tennis set difference)
            if frame_diff >= 3:
                frame_diff_bonus = 1.2
            elif frame_diff >= 2:
                frame_diff_bonus = 1.1

        k *= frame_diff_bonus

        # Update ratings
        winner_new = winner_rating + k * (1 - winner_expected)
        loser_new = loser_rating + k * (0 - loser_expected)

        # Store new ratings
        self.tournament_elo[tournament_type][winner] = winner_new
        self.tournament_elo[tournament_type][loser] = loser_new

        # Update overall ELO (weighted average)
        winner_ratings = []
        loser_ratings = []
        for t in self.tournament_elo.keys():
            if winner in self.tournament_elo[t]:
                winner_ratings.append(self.tournament_elo[t][winner])
            if loser in self.tournament_elo[t]:
                loser_ratings.append(self.tournament_elo[t][loser])

        if winner_ratings:
            self.player_elo[winner] = np.mean(winner_ratings)
        if loser_ratings:
            self.player_elo[loser] = np.mean(loser_ratings)

        # Record history
        if match_date:
            if winner not in self.elo_history:
                self.elo_history[winner] = []
            if loser not in self.elo_history:
                self.elo_history[loser] = []
            self.elo_history[winner].append((match_date, winner_new))
            self.elo_history[loser].append((match_date, loser_new))

        # Update statistics
        self.update_player_stats(winner, loser, frames_won, frames_lost, tournament_type)

    def update_player_stats(self, winner, loser, frames_won, frames_lost, tournament_type):
        """Update player statistics"""
        # Initialize stats if not exists
        if winner not in self.player_stats:
            self.player_stats[winner] = {
                'matches_played': 0, 'matches_won': 0, 'frames_won': 0,
                'frames_lost': 0, 'centuries': 0, 'breaks_50_plus': 0,
                'tournament_wins': 0, 'ranking_events': 0, 'prize_money': 0
            }
        if loser not in self.player_stats:
            self.player_stats[loser] = {
                'matches_played': 0, 'matches_won': 0, 'frames_won': 0,
                'frames_lost': 0, 'centuries': 0, 'breaks_50_plus': 0,
                'tournament_wins': 0, 'ranking_events': 0, 'prize_money': 0
            }

        # Winner stats
        self.player_stats[winner]['matches_played'] += 1
        self.player_stats[winner]['matches_won'] += 1
        if frames_won:
            self.player_stats[winner]['frames_won'] += frames_won
        if frames_lost:
            self.player_stats[winner]['frames_lost'] += frames_lost

        # Loser stats
        self.player_stats[loser]['matches_played'] += 1
        if frames_lost:
            self.player_stats[loser]['frames_won'] += frames_lost
        if frames_won:
            self.player_stats[loser]['frames_lost'] += frames_won

        # Tournament-specific stats
        if tournament_type in ['world_championship', 'masters', 'uk_championship']:
            self.player_stats[winner]['ranking_events'] += 1
            self.player_stats[loser]['ranking_events'] += 1

    def predict_match_outcome(self, player1, player2, tournament_type='ranking_event'):
        """
        Predict match outcome between two players
        """
        rating1 = self.get_player_rating(player1, tournament_type)
        rating2 = self.get_player_rating(player2, tournament_type)

        player1_win_prob = self.calculate_expected_score(rating1, rating2)

        return {
            'player1_win_prob': player1_win_prob,
            'player2_win_prob': 1 - player1_win_prob,
            'player1_rating': rating1,
            'player2_rating': rating2,
            'rating_difference': rating1 - rating2
        }

    def get_player_form(self, player, last_n_matches=10):
        """Get recent form for a player"""
        if player not in self.player_stats:
            return 0.5

        stats = self.player_stats[player]
        if stats['matches_played'] < 3:
            return 0.5

        # Calculate form based on recent performance
        win_rate = stats['matches_won'] / stats['matches_played']
        frame_rate = stats['frames_won'] / (stats['frames_won'] + stats['frames_lost']) if (stats['frames_won'] + stats['frames_lost']) > 0 else 0.5

        # Combine win rate and frame rate
        form = (win_rate * 0.7) + (frame_rate * 0.3)
        return min(max(form, 0.1), 0.9)

    def build_from_match_data(self, matches_df):
        """
        Build ELO system from historical match data
        """
        print("Building snooker ELO system from historical data...")

        # Sort matches by date
        if 'date' in matches_df.columns:
            matches_df = matches_df.sort_values('date')

        processed_matches = 0

        for _, match in matches_df.iterrows():
            try:
                winner = match.get('winner', '')
                loser = match.get('loser', '')
                tournament_type = match.get('tournament_type', 'ranking_event')
                frames_won = match.get('frames_won', None)
                frames_lost = match.get('frames_lost', None)
                match_date = match.get('date', None)

                if winner and loser:
                    self.update_ratings(
                        winner=winner,
                        loser=loser,
                        tournament_type=tournament_type,
                        frames_won=frames_won,
                        frames_lost=frames_lost,
                        match_date=match_date
                    )
                    processed_matches += 1

            except Exception as e:
                continue

        print(f"ELO ratings calculated for {len(self.player_elo)} players")
        print(f"Processing {processed_matches} matches...")

    def get_top_players(self, limit=20, tournament_type=None):
        """Get top players by ELO rating"""
        if tournament_type:
            # Tournament-specific rankings
            ratings = [(player, rating) for player, rating in
                      self.tournament_elo[tournament_type].items()]
        else:
            # Overall rankings
            ratings = [(player, rating) for player, rating in self.player_elo.items()]

        # Filter to players with meaningful match history
        filtered_ratings = [
            (player, rating) for player, rating in ratings
            if self.player_stats[player]['matches_played'] >= 3
        ]

        return sorted(filtered_ratings, key=lambda x: x[1], reverse=True)[:limit]

    def get_all_players(self):
        """Get all players in the system"""
        return list(self.player_elo.keys())

    def player_exists(self, player_name):
        """Check if a player exists in the system"""
        return player_name in self.player_elo

    def has_played_matches(self, player_name):
        """Check if a player has played any matches (has match history)"""
        return (player_name in self.player_stats and
                self.player_stats[player_name]['matches_played'] > 0)

    def get_player_tournament_performance(self, player, tournament_type):
        """Get player's performance in specific tournament type"""
        if player not in self.tournament_elo[tournament_type]:
            return self.initial_rating
        return self.tournament_elo[tournament_type][player]

    def plot_elo_progression(self, players, tournament_type=None, save_path=None):
        """
        Plot ELO progression over time (like tennis model visualization)
        """
        plt.figure(figsize=(12, 8))

        for player in players:
            if player in self.elo_history:
                dates, ratings = zip(*self.elo_history[player])
                plt.plot(dates, ratings, label=player, linewidth=2)

        plt.title(f"Snooker ELO Progression{' - ' + tournament_type if tournament_type else ''}")
        plt.xlabel("Date")
        plt.ylabel("ELO Rating")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def save_system(self, filepath):
        """Save the ELO system to file"""
        joblib.dump(self, filepath)

    @classmethod
    def load_system(cls, filepath):
        """Load ELO system from file"""
        return joblib.load(filepath)