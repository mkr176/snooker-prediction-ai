import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

class SnookerPredictionUtils:
    """
    Utility functions for snooker match predictions.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_head_to_head_features(self, player1: str, player2: str,
                                      match_history: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate head-to-head features between two players.

        Args:
            player1: First player name
            player2: Second player name
            match_history: Historical match data

        Returns:
            Dictionary of head-to-head features
        """
        # Filter matches between these players
        h2h_matches = match_history[
            ((match_history['player1'] == player1) & (match_history['player2'] == player2)) |
            ((match_history['player1'] == player2) & (match_history['player2'] == player1))
        ]

        if h2h_matches.empty:
            return {
                'h2h_total_matches': 0,
                'h2h_player1_wins': 0,
                'h2h_player1_win_rate': 0.5,
                'h2h_avg_frames_per_match': 10.0,
                'h2h_recent_form_player1': 0.5
            }

        # Calculate head-to-head statistics
        total_matches = len(h2h_matches)

        # Count wins for player1
        player1_wins = len(h2h_matches[
            ((h2h_matches['player1'] == player1) & (h2h_matches['winner'] == player1)) |
            ((h2h_matches['player2'] == player1) & (h2h_matches['winner'] == player1))
        ])

        win_rate = player1_wins / total_matches if total_matches > 0 else 0.5

        # Average frames per match
        avg_frames = h2h_matches['total_frames'].mean() if 'total_frames' in h2h_matches.columns else 10.0

        # Recent form (last 5 matches)
        recent_matches = h2h_matches.tail(5)
        recent_wins = len(recent_matches[
            ((recent_matches['player1'] == player1) & (recent_matches['winner'] == player1)) |
            ((recent_matches['player2'] == player1) & (recent_matches['winner'] == player1))
        ])
        recent_form = recent_wins / len(recent_matches) if len(recent_matches) > 0 else 0.5

        return {
            'h2h_total_matches': total_matches,
            'h2h_player1_wins': player1_wins,
            'h2h_player1_win_rate': win_rate,
            'h2h_avg_frames_per_match': avg_frames,
            'h2h_recent_form_player1': recent_form
        }

    def calculate_recent_form(self, player: str, match_history: pd.DataFrame,
                            window_size: int = 10) -> Dict[str, float]:
        """
        Calculate recent form statistics for a player.

        Args:
            player: Player name
            match_history: Historical match data
            window_size: Number of recent matches to consider

        Returns:
            Dictionary of form features
        """
        # Filter player's matches
        player_matches = match_history[
            (match_history['player1'] == player) | (match_history['player2'] == player)
        ].sort_values('date').tail(window_size)

        if player_matches.empty:
            return {
                'recent_matches_played': 0,
                'recent_win_rate': 0.5,
                'recent_avg_score': 8.0,
                'recent_centuries_per_match': 2.0,
                'recent_break_average': 45.0
            }

        # Calculate wins
        wins = len(player_matches[player_matches['winner'] == player])
        win_rate = wins / len(player_matches)

        # Calculate average score
        player1_scores = player_matches[player_matches['player1'] == player]['score1']
        player2_scores = player_matches[player_matches['player2'] == player]['score2']
        all_scores = pd.concat([player1_scores, player2_scores])
        avg_score = all_scores.mean() if len(all_scores) > 0 else 8.0

        # Calculate centuries per match (if data available)
        centuries_per_match = 2.0  # Default value

        # Break average (if data available)
        break_average = 45.0  # Default value

        return {
            'recent_matches_played': len(player_matches),
            'recent_win_rate': win_rate,
            'recent_avg_score': avg_score,
            'recent_centuries_per_match': centuries_per_match,
            'recent_break_average': break_average
        }

    def calculate_tournament_features(self, tournament: str, venue: str = None) -> Dict[str, float]:
        """
        Calculate tournament-specific features.

        Args:
            tournament: Tournament name
            venue: Venue name

        Returns:
            Dictionary of tournament features
        """
        # Tournament importance weights
        importance_weights = {
            'World Championship': 10.0,
            'UK Championship': 8.0,
            'Masters': 9.0,
            'Champion of Champions': 7.0,
            'Welsh Open': 6.0,
            'Players Championship': 7.5,
            'Gibraltar Open': 5.0,
            'German Masters': 6.0,
            'European Masters': 6.0,
            'English Open': 5.5
        }

        # Prize money (in thousands)
        prize_money = {
            'World Championship': 2395,
            'UK Championship': 1200,
            'Masters': 800,
            'Champion of Champions': 440,
            'Welsh Open': 405,
            'Players Championship': 380,
            'Gibraltar Open': 380,
            'German Masters': 400,
            'European Masters': 428,
            'English Open': 405
        }

        importance = importance_weights.get(tournament, 5.0)
        prize = prize_money.get(tournament, 300)

        # Venue-specific features (if available)
        home_advantage = 0.0
        if venue and 'England' in venue:
            home_advantage = 0.1

        return {
            'tournament_importance': importance,
            'tournament_prize_money': prize,
            'venue_home_advantage': home_advantage,
            'is_major_tournament': 1.0 if importance >= 8.0 else 0.0,
            'is_world_championship': 1.0 if tournament == 'World Championship' else 0.0
        }

    def calculate_ranking_features(self, player1_rank: int, player2_rank: int) -> Dict[str, float]:
        """
        Calculate ranking-based features.

        Args:
            player1_rank: Player 1's world ranking
            player2_rank: Player 2's world ranking

        Returns:
            Dictionary of ranking features
        """
        # Handle missing rankings
        if player1_rank is None or player1_rank <= 0:
            player1_rank = 128
        if player2_rank is None or player2_rank <= 0:
            player2_rank = 128

        ranking_diff = player2_rank - player1_rank  # Positive if player1 ranked higher
        rank_advantage = ranking_diff / max(player1_rank, player2_rank)

        # Calculate ranking-based probabilities
        elo_diff = (player2_rank - player1_rank) * 10
        expected_score_p1 = 1 / (1 + 10 ** (elo_diff / 400))

        return {
            'player1_ranking': player1_rank,
            'player2_ranking': player2_rank,
            'ranking_difference': abs(ranking_diff),
            'ranking_advantage_player1': ranking_diff,
            'rank_ratio': player2_rank / player1_rank,
            'both_top_16': 1.0 if player1_rank <= 16 and player2_rank <= 16 else 0.0,
            'expected_score_p1': expected_score_p1
        }

    def prepare_prediction_features(self, player1: str, player2: str, tournament: str,
                                  player1_rank: int = None, player2_rank: int = None,
                                  venue: str = None, match_history: pd.DataFrame = None) -> np.ndarray:
        """
        Prepare feature vector for prediction.

        Args:
            player1: First player name
            player2: Second player name
            tournament: Tournament name
            player1_rank: Player 1's ranking
            player2_rank: Player 2's ranking
            venue: Venue name
            match_history: Historical match data

        Returns:
            Feature vector for prediction
        """
        features = {}

        # Tournament features
        tournament_features = self.calculate_tournament_features(tournament, venue)
        features.update(tournament_features)

        # Ranking features
        ranking_features = self.calculate_ranking_features(
            player1_rank or 50, player2_rank or 50
        )
        features.update(ranking_features)

        # Head-to-head features (if match history available)
        if match_history is not None and not match_history.empty:
            h2h_features = self.calculate_head_to_head_features(player1, player2, match_history)
            features.update(h2h_features)

            # Recent form features
            form1 = self.calculate_recent_form(player1, match_history)
            form2 = self.calculate_recent_form(player2, match_history)

            for key, value in form1.items():
                features[f'player1_{key}'] = value
            for key, value in form2.items():
                features[f'player2_{key}'] = value

        else:
            # Default values when no history available
            default_features = {
                'h2h_total_matches': 0,
                'h2h_player1_wins': 0,
                'h2h_player1_win_rate': 0.5,
                'h2h_avg_frames_per_match': 10.0,
                'h2h_recent_form_player1': 0.5,
                'player1_recent_matches_played': 10,
                'player1_recent_win_rate': 0.6,
                'player1_recent_avg_score': 8.0,
                'player1_recent_centuries_per_match': 2.0,
                'player1_recent_break_average': 45.0,
                'player2_recent_matches_played': 10,
                'player2_recent_win_rate': 0.6,
                'player2_recent_avg_score': 8.0,
                'player2_recent_centuries_per_match': 2.0,
                'player2_recent_break_average': 45.0
            }
            features.update(default_features)

        # Convert to numpy array (ensure consistent order)
        feature_names = sorted(features.keys())
        feature_vector = np.array([features[name] for name in feature_names])

        return feature_vector.reshape(1, -1)

    def interpret_prediction(self, prediction_result: Dict, player1: str, player2: str) -> Dict[str, str]:
        """
        Interpret prediction results into human-readable format.

        Args:
            prediction_result: Raw prediction results
            player1: First player name
            player2: Second player name

        Returns:
            Interpreted prediction
        """
        prob = prediction_result['probability'][0]
        confidence = prediction_result['confidence'][0]

        # Determine winner
        if prob > 0.5:
            predicted_winner = player1
            win_probability = prob
        else:
            predicted_winner = player2
            win_probability = 1 - prob

        # Confidence levels
        if confidence < 0.3:
            confidence_level = "Low"
        elif confidence < 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "High"

        # Betting recommendation
        if confidence > 0.6 and win_probability > 0.65:
            betting_rec = f"Strong recommendation for {predicted_winner}"
        elif confidence > 0.4 and win_probability > 0.60:
            betting_rec = f"Moderate recommendation for {predicted_winner}"
        else:
            betting_rec = "No clear betting recommendation"

        return {
            'predicted_winner': predicted_winner,
            'win_probability': f"{win_probability:.1%}",
            'confidence_level': confidence_level,
            'confidence_score': f"{confidence:.3f}",
            'betting_recommendation': betting_rec,
            'match_summary': f"{predicted_winner} favored to win with {win_probability:.1%} probability"
        }

    def calculate_value_bet(self, prediction_prob: float, bookmaker_odds: float,
                          stake: float = 10.0) -> Dict[str, float]:
        """
        Calculate value betting opportunities.

        Args:
            prediction_prob: Model's predicted probability
            bookmaker_odds: Bookmaker's decimal odds
            stake: Stake amount

        Returns:
            Value betting analysis
        """
        # Convert odds to implied probability
        implied_prob = 1 / bookmaker_odds

        # Calculate expected value
        expected_value = (prediction_prob * (bookmaker_odds - 1) * stake) - ((1 - prediction_prob) * stake)

        # Value bet if our probability is higher than implied probability
        is_value_bet = prediction_prob > implied_prob

        # Calculate edge
        edge = prediction_prob - implied_prob

        return {
            'implied_probability': implied_prob,
            'model_probability': prediction_prob,
            'expected_value': expected_value,
            'edge': edge,
            'is_value_bet': is_value_bet,
            'recommended_stake': stake if is_value_bet else 0.0
        }