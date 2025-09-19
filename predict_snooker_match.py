#!/usr/bin/env python3
"""
Interactive Snooker Match Predictor - Terminal Interface
Professional snooker prediction system with REAL data
"""

import sys
import os
sys.path.append('src')

import argparse
import pandas as pd
import joblib
from snooker_elo_system import SnookerEloSystem

class RealDataPredictor:
    """Real data snooker predictor using the new 85% accuracy model"""

    def __init__(self):
        try:
            self.model = joblib.load('models/snooker_85_percent_model.pkl')
            self.features = joblib.load('models/snooker_features.pkl')
            self.elo_system = joblib.load('models/snooker_elo_complete.pkl')
            print("✅ Real data model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)

        self.tournament_weights = {
            'world_championship': 50, 'masters': 35, 'uk_championship': 35,
            'ranking_event': 20, 'invitational': 15
        }

    def get_available_players(self, limit=None):
        """Get list of available players"""
        all_players = list(self.elo_system.player_elo.keys())
        return sorted(all_players)[:limit] if limit else sorted(all_players)

    def validate_player(self, player_name):
        """Check if player exists in dataset"""
        return player_name in self.elo_system.player_elo

    def predict_match(self, player1, player2, tournament_type='ranking_event', best_of=7):
        """Predict match outcome using real data model"""

        # Validate players
        if not self.validate_player(player1):
            return {'error': f'Player "{player1}" not found in dataset'}
        if not self.validate_player(player2):
            return {'error': f'Player "{player2}" not found in dataset'}

        # Get ELO features
        p1_elo = self.elo_system.get_player_elo_features(player1)
        p2_elo = self.elo_system.get_player_elo_features(player2)

        # Create features exactly like training
        match_features = {
            'player_elo_diff': p1_elo['overall_elo'] - p2_elo['overall_elo'],
            'total_elo': p1_elo['overall_elo'] + p2_elo['overall_elo'],
            'player1_elo': p1_elo['overall_elo'],
            'player2_elo': p2_elo['overall_elo'],
            'recent_form_diff': p1_elo['recent_form'] - p2_elo['recent_form'],
            'momentum_diff': p1_elo['recent_momentum'] - p2_elo['recent_momentum'],
            'elo_change_diff': p1_elo['recent_elo_change'] - p2_elo['recent_elo_change'],
            'experience_diff': p1_elo['matches_played'] - p2_elo['matches_played'],
            'win_rate_diff': p1_elo['career_win_rate'] - p2_elo['career_win_rate'],
            'centuries_diff': 0, 'highest_break_diff': 0, 'score_diff': 0,
            'total_frames': best_of, 'match_duration': 120,
            'tournament_weight': self.tournament_weights.get(tournament_type, 20),
            'is_world_championship': 1 if tournament_type == 'world_championship' else 0,
            'is_major_tournament': 1 if tournament_type in ['world_championship', 'masters', 'uk_championship'] else 0,
            'is_ranking_event': 1 if tournament_type == 'ranking_event' else 0,
            'h2h_total_matches': 0, 'h2h_win_rate': 0.5,
            'elo_x_form': (p1_elo['overall_elo'] - p2_elo['overall_elo']) * (p1_elo['recent_form'] - p2_elo['recent_form']),
            'form_x_momentum': (p1_elo['recent_form'] - p2_elo['recent_form']) * (p1_elo['recent_momentum'] - p2_elo['recent_momentum'])
        }

        # Create DataFrame and predict
        features_df = pd.DataFrame([match_features])
        X = features_df[self.features].fillna(0)
        prediction = self.model.predict_proba(X)[0]

        return {
            'player1': player1,
            'player2': player2,
            'player1_prob': prediction[1],
            'player2_prob': prediction[0],
            'winner': player1 if prediction[1] > prediction[0] else player2,
            'confidence': max(prediction),
            'player1_elo': p1_elo['overall_elo'],
            'player2_elo': p2_elo['overall_elo'],
            'elo_diff': p1_elo['overall_elo'] - p2_elo['overall_elo'],
            'tournament_type': tournament_type,
            'best_of': best_of
        }

def display_prediction_result(result):
    """Display prediction result in a nice format"""
    print(f"\n🏆 WINNER: {result['winner']}")
    print(f"📊 Confidence: {result['confidence']:.1%}")
    print(f"\n📈 DETAILED PROBABILITIES:")
    print(f"   {result['player1']}: {result['player1_prob']:.1%}")
    print(f"   {result['player2']}: {result['player2_prob']:.1%}")
    print(f"\n🎱 MATCH DETAILS:")
    print(f"   Tournament: {result['tournament_type'].replace('_', ' ').title()}")
    print(f"   Format: Best of {result['best_of']} (first to {(result['best_of'] + 1) // 2})")
    print(f"\n⚖️  ELO RATINGS:")
    print(f"   {result['player1']}: {result['player1_elo']:.0f}")
    print(f"   {result['player2']}: {result['player2_elo']:.0f}")
    print(f"   Difference: {result['elo_diff']:+.0f} points")

    # Analysis
    confidence_level = result['confidence']
    if confidence_level > 0.8:
        analysis = "🎯 Clear favorite - significant advantage"
    elif confidence_level > 0.65:
        analysis = "💪 Moderate favorite - good edge"
    else:
        analysis = "📈 Close match - slight edge"

    print(f"\n🤖 MODEL vs ELO: ✅ Real data prediction")
    print(f"   ML Model favors: {result['winner']}")
    print(f"   ELO favors: {result['player1'] if result['elo_diff'] > 0 else result['player2']}")
    print(f"\n{analysis}")

def interactive_prediction():
    """Interactive mode - ask user for input"""
    print("🎱 SNOOKER MATCH PREDICTOR")
    print("Professional snooker prediction system with REAL data")
    print("=" * 50)

    # Initialize predictor
    predictor = RealDataPredictor()

    while True:
        print("\n🔮 PREDICT A SNOOKER MATCH")
        print("-" * 25)
        print("💡 Type 'players' to see available players, or 'quit' to exit")

        # Get player names
        player1 = input("Enter Player 1 name: ").strip()
        if not player1:
            print("❌ Player name cannot be empty")
            continue

        if player1.lower() in ['quit', 'exit']:
            print("🎱 Thanks for using Snooker Match Predictor!")
            break

        if player1.lower() == 'players':
            available_players = predictor.get_available_players()
            if available_players:
                print(f"\n📋 Available players ({len(available_players)} total):")
                for i, player in enumerate(available_players[:20], 1):
                    print(f"  {i:2d}. {player}")
                if len(available_players) > 20:
                    print(f"  ... and {len(available_players)-20} more players")
            else:
                print("❌ No players found. Make sure the model is trained.")
            continue

        player2 = input("Enter Player 2 name: ").strip()
        if not player2:
            print("❌ Player name cannot be empty")
            continue

        # Tournament selection
        print("\n🏆 Select tournament type:")
        print("  1. World Championship")
        print("  2. Masters")
        print("  3. UK Championship")
        print("  4. Ranking Event (default)")
        print("  5. Invitational")

        tournament_choice = input("Tournament type (1-5, default=4): ").strip()
        tournament_map = {
            '1': 'world_championship',
            '2': 'masters',
            '3': 'uk_championship',
            '4': 'ranking_event',
            '5': 'invitational'
        }
        tournament_type = tournament_map.get(tournament_choice, 'ranking_event')

        # Format selection
        print("\n📏 Select match format:")
        print("  1. Best of 7 (first to 4)")
        print("  2. Best of 9 (first to 5)")
        print("  3. Best of 11 (first to 6)")
        print("  4. Best of 17 (first to 9)")
        print("  5. Best of 19 (first to 10)")
        print("  6. Best of 35 (first to 18) - World Championship Final")

        format_choice = input("Match format (1-6, default=1): ").strip()
        format_map = {
            '1': 7, '2': 9, '3': 11, '4': 17, '5': 19, '6': 35
        }
        best_of = format_map.get(format_choice, 7)

        # Make prediction
        print(f"\n🔮 Predicting {player1} vs {player2}...")
        print(f"   Tournament: {tournament_type.replace('_', ' ').title()}")
        print(f"   Format: Best of {best_of} (first to {(best_of + 1) // 2})")

        result = predictor.predict_match(player1, player2, tournament_type, best_of)

        if 'error' in result:
            print(f"\n❌ {result['error']}")
            # Suggest similar players
            all_players = predictor.get_available_players()
            player1_suggestions = [p for p in all_players if player1.lower() in p.lower() or p.lower() in player1.lower()]
            player2_suggestions = [p for p in all_players if player2.lower() in p.lower() or p.lower() in player2.lower()]
            if player1_suggestions:
                print(f"💡 Similar to '{player1}': {', '.join(player1_suggestions[:3])}")
            if player2_suggestions:
                print(f"💡 Similar to '{player2}': {', '.join(player2_suggestions[:3])}")
        else:
            print("\n" + "="*50)
            print("🎯 PREDICTION COMPLETE!")
            print("="*50)
            display_prediction_result(result)

        # Ask if user wants to continue
        continue_choice = input("\nMake another prediction? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("🎱 Thanks for using Snooker Match Predictor!")
            break

def predict_match_cli(player1, player2, tournament_type='ranking_event', best_of=7):
    """Predict a match via command line"""
    predictor = RealDataPredictor()

    result = predictor.predict_match(
        player1=player1,
        player2=player2,
        tournament_type=tournament_type,
        best_of=best_of
    )

    if 'error' in result:
        print(f"❌ {result['error']}")
        return False

    print("✅ Both players validated")
    display_prediction_result(result)
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Snooker Match Predictor')
    parser.add_argument('player1', nargs='?', help='First player name')
    parser.add_argument('player2', nargs='?', help='Second player name')
    parser.add_argument('--tournament', default='ranking_event',
                       choices=['world_championship', 'masters', 'uk_championship', 'ranking_event', 'invitational'],
                       help='Tournament type')
    parser.add_argument('--best-of', type=int, default=7, help='Match format (best of X)')
    parser.add_argument('--examples', action='store_true', help='Show example predictions')

    args = parser.parse_args()

    if args.examples:
        print("🎱 EXAMPLE PREDICTIONS")
        print("=" * 30)
        examples = [
            ("Zhou Yuelong", "Luca Brecel", "ranking_event", 7),
            ("Yan Bingtao", "Neil Robertson", "masters", 11),
            ("Mark Allen", "Michael Holt", "world_championship", 35)
        ]

        for player1, player2, tournament, best_of in examples:
            print(f"\n🔮 {player1} vs {player2} ({tournament.replace('_', ' ').title()})")
            predict_match_cli(player1, player2, tournament, best_of)
        return

    if args.player1 and args.player2:
        predict_match_cli(args.player1, args.player2, args.tournament, args.best_of)
    else:
        interactive_prediction()

if __name__ == "__main__":
    main()