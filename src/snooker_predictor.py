#!/usr/bin/env python3
"""
Snooker Match Predictor - Professional snooker prediction system
Adapted from tennis system with enhanced name matching and snooker-specific features
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from snooker_elo_system import SnookerEloSystem
import warnings
warnings.filterwarnings('ignore')

class SnookerPredictor:
    """
    Snooker Match Predictor using advanced machine learning and ELO ratings.

    Features snooker-specific analysis:
    - ELO ratings by tournament type
    - Break building statistics
    - Safety play analysis
    - Tournament prestige weighting
    - Enhanced player name matching
    """

    def __init__(self):
        self.model = None
        self.elo_system = None
        self.feature_columns = None
        self.confidence_threshold = 0.75

    def load_model(self):
        """Load the trained snooker prediction model"""
        try:
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(os.path.dirname(current_dir), 'models')

            self.model = joblib.load(os.path.join(models_dir, 'snooker_prediction_model.pkl'))
            self.feature_columns = joblib.load(os.path.join(models_dir, 'snooker_features.pkl'))
            self.elo_system = SnookerEloSystem.load_system(os.path.join(models_dir, 'snooker_elo_system.pkl'))

            print("✅ Snooker prediction model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please train the model first using train_snooker_model.py")
            return False

    def normalize_name(self, name):
        """
        Normalize a name for better matching (handle apostrophes, accents, etc.)
        Same as tennis system
        """
        import re
        normalized = name.lower().strip()
        variations = [normalized]

        # Add version without apostrophes
        no_apostrophe = re.sub(r"['']", "", normalized)
        if no_apostrophe != normalized:
            variations.append(no_apostrophe)

        # Add version with different apostrophe styles
        if "'" in normalized:
            variations.append(normalized.replace("'", "'"))
        if "'" in normalized:
            variations.append(normalized.replace("'", "'"))

        # Handle common accent removal
        accent_map = {
            'á': 'a', 'à': 'a', 'â': 'a', 'ä': 'a', 'ã': 'a',
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
            'ó': 'o', 'ò': 'o', 'ô': 'o', 'ö': 'o', 'õ': 'o',
            'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
            'ñ': 'n', 'ç': 'c'
        }

        for original in variations.copy():
            no_accent = original
            for accented, plain in accent_map.items():
                no_accent = no_accent.replace(accented, plain)
            if no_accent != original:
                variations.append(no_accent)

        return list(set(variations))

    def find_player_match(self, player_name):
        """
        Find the best match for a snooker player name
        Same enhanced matching as tennis system
        """
        if not self.elo_system:
            return None, 0

        all_players = self.elo_system.get_all_players()
        search_variations = self.normalize_name(player_name)

        # Try exact match first
        for search_variant in search_variations:
            for player in all_players:
                player_variations = self.normalize_name(player)
                for player_variant in player_variations:
                    if search_variant == player_variant:
                        if self.elo_system.has_played_matches(player):
                            return player, 1.0

        best_match = None
        best_score = 0

        # Fuzzy matching with normalized names
        for player in all_players:
            if not self.elo_system.has_played_matches(player):
                continue

            player_variations = self.normalize_name(player)
            score = 0

            for search_variant in search_variations:
                search_parts = search_variant.split()

                for player_variant in player_variations:
                    player_parts = player_variant.split()
                    current_score = 0

                    # Full substring match
                    if search_variant in player_variant or player_variant in search_variant:
                        current_score = 0.85

                    # Name parts matching
                    exact_matches = 0
                    partial_matches = 0

                    for search_part in search_parts:
                        if len(search_part) >= 2:
                            for player_part in player_parts:
                                if search_part == player_part:
                                    exact_matches += 1
                                elif search_part in player_part or player_part in search_part:
                                    partial_matches += 1
                                elif (len(search_part) >= 4 and len(player_part) >= 4 and
                                      search_part[:3] == player_part[:3]):
                                    partial_matches += 0.7

                    if exact_matches > 0 or partial_matches > 0:
                        part_score = 0.4 + (exact_matches * 0.3) + (partial_matches * 0.15)
                        current_score = max(current_score, part_score)

                        if exact_matches + partial_matches >= len(search_parts):
                            current_score += 0.2

                        if len(search_parts) >= 2 and len(player_parts) >= 2:
                            if search_parts[0] == player_parts[0]:
                                current_score += 0.15
                            if search_parts[-1] == player_parts[-1]:
                                current_score += 0.15

                    length_diff = abs(len(search_variant) - len(player_variant))
                    if length_diff <= 3:
                        current_score += 0.1
                    elif length_diff <= 6:
                        current_score += 0.05

                    score = max(score, current_score)

            if score > best_score and score >= 0.5:
                best_match = player
                best_score = score

        return best_match, best_score

    def suggest_similar_players(self, player_name, limit=5):
        """Suggest similar player names with snooker-specific matching"""
        if not self.elo_system:
            if not self.load_model():
                return []

        all_players = self.get_available_players()
        search_variations = self.normalize_name(player_name)

        scored_players = []
        surname_matches = []
        similar_surnames = []

        for player in all_players:
            player_variations = self.normalize_name(player)
            score = 0

            # Check for surname matches
            for search_variant in search_variations:
                search_parts = search_variant.split()
                for player_variant in player_variations:
                    player_parts = player_variant.split()

                    if len(search_parts) >= 2 and len(player_parts) >= 2:
                        search_lastname = search_parts[-1]
                        player_lastname = player_parts[-1]

                        if search_lastname == player_lastname:
                            surname_matches.append(player)
                            score += 25
                        elif (len(search_lastname) >= 3 and len(player_lastname) >= 3 and
                              (search_lastname[:3] == player_lastname[:3] or
                               search_lastname in player_lastname or player_lastname in search_lastname)):
                            similar_surnames.append(player)

            # Score calculation (same as tennis)
            for search_variant in search_variations:
                search_parts = search_variant.split()
                for player_variant in player_variations:
                    player_parts = player_variant.split()

                    if search_variant == player_variant:
                        score = 100
                    elif search_variant in player_variant or player_variant in search_variant:
                        score = max(score, 80)
                    else:
                        matches = 0
                        partial_matches = 0

                        for search_part in search_parts:
                            if len(search_part) >= 2:
                                for player_part in player_parts:
                                    if search_part == player_part:
                                        matches += 2
                                    elif search_part in player_part or player_part in search_part:
                                        partial_matches += 1
                                    elif (len(search_part) >= 3 and len(player_part) >= 3 and
                                          search_part[:2] == player_part[:2]):
                                        partial_matches += 0.5

                        if matches > 0 or partial_matches > 0:
                            current_score = 30 + (matches * 15) + (partial_matches * 5)
                            if abs(len(search_variant) - len(player_variant)) <= 5:
                                current_score += 5
                            score = max(score, current_score)

            if score > 0:
                scored_players.append((player, score))

        scored_players.sort(key=lambda x: x[1], reverse=True)

        suggestions = []
        for player in surname_matches:
            if player not in suggestions:
                suggestions.append(player)
                if len(suggestions) >= limit:
                    break

        for player, score in scored_players:
            if player not in suggestions:
                suggestions.append(player)
                if len(suggestions) >= limit:
                    break

        if len(suggestions) < limit:
            for player in similar_surnames:
                if player not in suggestions:
                    suggestions.append(player)
                    if len(suggestions) >= limit:
                        break

        return suggestions[:limit] if suggestions else all_players[:limit]

    def validate_players(self, player1, player2):
        """
        Validate that both players exist with enhanced matching
        Same as tennis system
        """
        if not self.elo_system:
            if not self.load_model():
                return False, "Model not loaded", player1, player2

        matched_player1, score1 = self.find_player_match(player1)
        matched_player2, score2 = self.find_player_match(player2)

        if matched_player1 and matched_player2:
            message = "Both players validated"
            if matched_player1.lower() != player1.lower():
                message += f" ('{player1}' -> '{matched_player1}')"
            if matched_player2.lower() != player2.lower():
                message += f" ('{player2}' -> '{matched_player2}')"
            return True, message, matched_player1, matched_player2

        elif matched_player1 and not matched_player2:
            suggestions = self.suggest_similar_players(player2, 3)
            message = f"Player '{player2}' not found in the system."
            if matched_player1.lower() != player1.lower():
                message += f" ('{player1}' matched to '{matched_player1}')"
            if suggestions:
                message += f"\n   💡 Did you mean: {', '.join(suggestions)}?"
            return False, message, matched_player1, player2

        elif not matched_player1 and matched_player2:
            suggestions = self.suggest_similar_players(player1, 3)
            message = f"Player '{player1}' not found in the system."
            if matched_player2.lower() != player2.lower():
                message += f" ('{player2}' matched to '{matched_player2}')"
            if suggestions:
                message += f"\n   💡 Did you mean: {', '.join(suggestions)}?"
            return False, message, player1, matched_player2

        else:
            suggestions1 = self.suggest_similar_players(player1, 3)
            suggestions2 = self.suggest_similar_players(player2, 3)
            message = f"Neither '{player1}' nor '{player2}' found in the system.\n"
            if suggestions1:
                message += f"   💡 Similar to '{player1}': {', '.join(suggestions1)}\n"
            if suggestions2:
                message += f"   💡 Similar to '{player2}': {', '.join(suggestions2)}"
            return False, message, player1, player2

    def create_prediction_features(self, player1, player2, tournament_type='ranking_event',
                                 best_of=7, match_date=None):
        """
        Create prediction features for snooker match
        """
        if match_date is None:
            match_date = datetime.now()

        # Get ELO ratings
        elo1 = self.elo_system.get_player_rating(player1, tournament_type)
        elo2 = self.elo_system.get_player_rating(player2, tournament_type)

        # Get player statistics
        stats1 = self.elo_system.player_stats[player1]
        stats2 = self.elo_system.player_stats[player2]

        # Calculate derived features
        features = {
            # ELO features
            'player1_elo': elo1,
            'player2_elo': elo2,
            'elo_difference': elo1 - elo2,
            'elo_ratio': elo1 / elo2 if elo2 > 0 else 1.0,

            # Match format
            'best_of': best_of,
            'frames_to_win': (best_of + 1) // 2,

            # Tournament features
            'tournament_weight': self.elo_system.tournament_weights.get(tournament_type, 20),
            'is_major_tournament': 1 if tournament_type in ['world_championship', 'masters', 'uk_championship'] else 0,

            # Player experience
            'player1_matches_played': stats1['matches_played'],
            'player2_matches_played': stats2['matches_played'],
            'experience_difference': stats1['matches_played'] - stats2['matches_played'],

            # Win rates
            'player1_win_rate': stats1['matches_won'] / max(stats1['matches_played'], 1),
            'player2_win_rate': stats2['matches_won'] / max(stats2['matches_played'], 1),
            'win_rate_difference': (stats1['matches_won'] / max(stats1['matches_played'], 1)) -
                                 (stats2['matches_won'] / max(stats2['matches_played'], 1)),

            # Frame statistics
            'player1_frame_win_rate': stats1['frames_won'] / max(stats1['frames_won'] + stats1['frames_lost'], 1),
            'player2_frame_win_rate': stats2['frames_won'] / max(stats2['frames_won'] + stats2['frames_lost'], 1),
            'frame_rate_difference': (stats1['frames_won'] / max(stats1['frames_won'] + stats1['frames_lost'], 1)) -
                                   (stats2['frames_won'] / max(stats2['frames_won'] + stats2['frames_lost'], 1)),

            # Tournament success
            'player1_tournament_wins': stats1['tournament_wins'],
            'player2_tournament_wins': stats2['tournament_wins'],
            'tournament_wins_difference': stats1['tournament_wins'] - stats2['tournament_wins'],

            # Ranking events
            'player1_ranking_events': stats1['ranking_events'],
            'player2_ranking_events': stats2['ranking_events'],
            'ranking_events_difference': stats1['ranking_events'] - stats2['ranking_events'],

            # Prize money (career earnings indicator)
            'player1_prize_money': stats1['prize_money'],
            'player2_prize_money': stats2['prize_money'],
            'prize_money_ratio': stats1['prize_money'] / max(stats2['prize_money'], 1),

            # Break building (snooker-specific)
            'player1_centuries': stats1['centuries'],
            'player2_centuries': stats2['centuries'],
            'centuries_difference': stats1['centuries'] - stats2['centuries'],

            'player1_breaks_50_plus': stats1['breaks_50_plus'],
            'player2_breaks_50_plus': stats2['breaks_50_plus'],
            'breaks_50_plus_difference': stats1['breaks_50_plus'] - stats2['breaks_50_plus'],

            # Form indicators
            'player1_form': self.elo_system.get_player_form(player1),
            'player2_form': self.elo_system.get_player_form(player2),
            'form_difference': self.elo_system.get_player_form(player1) - self.elo_system.get_player_form(player2)
        }

        return features

    def predict_match(self, player1, player2, tournament_type='ranking_event', best_of=7):
        """
        Predict snooker match outcome using machine learning model
        """
        if not self.model:
            if not self.load_model():
                return None

        # Validate players with enhanced matching
        validation_result = self.validate_players(player1, player2)
        if len(validation_result) == 4:
            valid, message, matched_player1, matched_player2 = validation_result
        else:
            valid, message = validation_result
            matched_player1, matched_player2 = player1, player2

        if not valid:
            print(f"❌ Validation failed: {message}")
            available = self.get_available_players(10)
            if available:
                print("💡 Available players include:", ", ".join(available[:5]))
                if len(available) > 5:
                    print(f"   ...and {len(available) - 5} more players")
            return None

        print(f"✅ {message}")

        # Use matched player names
        player1, player2 = matched_player1, matched_player2

        # Create prediction features
        features = self.create_prediction_features(player1, player2, tournament_type, best_of)

        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        X = features_df[self.feature_columns].fillna(0)

        # Get prediction
        prediction_proba = self.model.predict_proba(X)[0]
        prediction_class = self.model.predict(X)[0]

        # Interpret results
        player1_win_prob = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
        player2_win_prob = 1 - player1_win_prob

        winner = player1 if prediction_class == 1 else player2
        confidence = max(player1_win_prob, player2_win_prob)

        # Get ELO-based prediction for comparison
        elo_prediction = self.elo_system.predict_match_outcome(player1, player2, tournament_type)

        return {
            'player1': player1,
            'player2': player2,
            'tournament_type': tournament_type,
            'best_of': best_of,
            'predicted_winner': winner,
            'confidence': confidence,
            'player1_win_prob': player1_win_prob,
            'player2_win_prob': player2_win_prob,
            'elo_prediction': elo_prediction,
            'model_vs_elo': {
                'model_winner': winner,
                'elo_favorite': player1 if elo_prediction['player1_win_prob'] > 0.5 else player2,
                'agreement': (winner == player1 and elo_prediction['player1_win_prob'] > 0.5) or
                           (winner == player2 and elo_prediction['player2_win_prob'] > 0.5)
            }
        }

    def get_available_players(self, limit=None):
        """Get list of all available players in the system"""
        if not self.elo_system:
            if not self.load_model():
                return []

        all_players = self.elo_system.get_all_players()
        active_players = [player for player in all_players
                         if self.elo_system.has_played_matches(player)]

        if limit:
            return active_players[:limit]
        return active_players

    def analyze_head_to_head(self, player1, player2, tournament_type=None):
        """Analyze head-to-head record between two players"""
        # This would analyze historical data if available
        # For now, return ELO-based analysis
        if not self.elo_system:
            return None

        prediction = self.elo_system.predict_match_outcome(player1, player2, tournament_type or 'ranking_event')

        return {
            'player1': player1,
            'player2': player2,
            'elo_advantage': prediction['rating_difference'],
            'predicted_favorite': player1 if prediction['player1_win_prob'] > 0.5 else player2,
            'confidence': max(prediction['player1_win_prob'], prediction['player2_win_prob']),
            'tournament_type': tournament_type
        }