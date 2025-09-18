#!/usr/bin/env python3
"""
Interactive Snooker Match Predictor - Terminal Interface
Professional snooker prediction system with enhanced name matching
"""

import sys
import os
sys.path.append('src')

import argparse
from snooker_predictor import SnookerPredictor

def interactive_prediction():
    """Interactive mode - ask user for input"""
    print("🎱 SNOOKER MATCH PREDICTOR")
    print("Professional snooker prediction system")
    print("=" * 50)

    # Initialize predictor
    predictor = SnookerPredictor()

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
            available_players = predictor.get_available_players(20)
            if available_players:
                print(f"\n📋 Available players ({len(available_players)} showing):")
                for i, player in enumerate(available_players[:20], 1):
                    print(f"  {i:2d}. {player}")
                if len(predictor.get_available_players()) > 20:
                    total = len(predictor.get_available_players())
                    print(f"  ... and {total-20} more players")
            else:
                print("❌ No players found. Make sure the model is trained.")
            continue

        player2 = input("Enter Player 2 name: ").strip()
        if not player2:
            print("❌ Player name cannot be empty")
            continue

        if player2.lower() in ['quit', 'exit']:
            print("🎱 Thanks for using Snooker Match Predictor!")
            break

        # Get tournament type
        print(f"\n🏆 Select tournament type:")
        print(f"  1. World Championship")
        print(f"  2. Masters")
        print(f"  3. UK Championship")
        print(f"  4. Ranking Event (default)")
        print(f"  5. Invitational")

        tournament_choice = input("Tournament type (1-5, default=4): ").strip()

        tournament_map = {
            '1': 'world_championship',
            '2': 'masters',
            '3': 'uk_championship',
            '4': 'ranking_event',
            '5': 'invitational'
        }
        tournament_type = tournament_map.get(tournament_choice, 'ranking_event')

        # Get match format
        print(f"\n📏 Select match format:")
        print(f"  1. Best of 7 (first to 4)")
        print(f"  2. Best of 9 (first to 5)")
        print(f"  3. Best of 11 (first to 6)")
        print(f"  4. Best of 17 (first to 9)")
        print(f"  5. Best of 19 (first to 10)")
        print(f"  6. Best of 35 (first to 18) - World Championship Final")

        format_choice = input("Match format (1-6, default=1): ").strip()

        format_map = {
            '1': 7, '2': 9, '3': 11, '4': 17, '5': 19, '6': 35
        }
        best_of = format_map.get(format_choice, 7)

        # Make prediction
        print(f"\n🔮 Predicting {player1} vs {player2}...")
        print(f"   Tournament: {tournament_type.replace('_', ' ').title()}")
        print(f"   Format: Best of {best_of} (first to {(best_of + 1) // 2})")

        result = predict_match_cli(player1, player2, tournament_type, best_of)

        if result:
            print("\n" + "="*50)
            print("🎯 PREDICTION COMPLETE!")
            print("="*50)
        else:
            print("\n❌ Prediction failed. Please check player names and try again.")

        # Ask if user wants to continue
        continue_choice = input("\nMake another prediction? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("🎱 Thanks for using Snooker Match Predictor!")
            break

def predict_match_cli(player1, player2, tournament_type='ranking_event', best_of=7):
    """Predict a match via command line"""
    predictor = SnookerPredictor()

    result = predictor.predict_match(
        player1=player1,
        player2=player2,
        tournament_type=tournament_type,
        best_of=best_of
    )

    if result:
        display_prediction_result(result)
        return True
    else:
        return False

def display_prediction_result(result):
    """Display prediction result in a nice format"""
    print(f"\n🏆 WINNER: {result['predicted_winner']}")
    print(f"📊 Confidence: {result['confidence']:.1%}")

    print(f"\n📈 DETAILED PROBABILITIES:")
    print(f"   {result['player1']}: {result['player1_win_prob']:.1%}")
    print(f"   {result['player2']}: {result['player2_win_prob']:.1%}")

    print(f"\n🎱 MATCH DETAILS:")
    print(f"   Tournament: {result['tournament_type'].replace('_', ' ').title()}")
    print(f"   Format: Best of {result['best_of']} (first to {(result['best_of'] + 1) // 2})")

    # ELO comparison
    elo_pred = result['elo_prediction']
    print(f"\n⚖️  ELO RATINGS:")
    print(f"   {result['player1']}: {elo_pred['player1_rating']:.0f}")
    print(f"   {result['player2']}: {elo_pred['player2_rating']:.0f}")
    print(f"   Difference: {abs(elo_pred['rating_difference']):.0f} points")

    # Model vs ELO agreement
    model_vs_elo = result['model_vs_elo']
    agreement = "✅ Agree" if model_vs_elo['agreement'] else "⚠️  Disagree"
    print(f"\n🤖 MODEL vs ELO: {agreement}")
    print(f"   ML Model favors: {model_vs_elo['model_winner']}")
    print(f"   ELO favors: {model_vs_elo['elo_favorite']}")

    # Confidence interpretation
    if result['confidence'] >= 0.8:
        print(f"\n💪 Strong prediction - Clear favorite identified")
    elif result['confidence'] >= 0.65:
        print(f"\n👍 Moderate prediction - Slight edge to winner")
    else:
        print(f"\n🤔 Close match - Could go either way")

def show_examples():
    """Show example snooker rivalries"""
    examples = [
        ("Ronnie O'Sullivan", "Judd Trump", "The two modern greats"),
        ("Mark Selby", "Neil Robertson", "Tactical vs attacking styles"),
        ("John Higgins", "Mark Williams", "Scottish vs Welsh rivalry"),
        ("Kyren Wilson", "Jack Lisowski", "Rising stars battle"),
        ("Shaun Murphy", "Stuart Bingham", "Experienced campaigners"),
        ("Barry Hawkins", "Joe Perry", "Solid professionals"),
    ]

    print(f"\n🎱 FAMOUS SNOOKER RIVALRIES:")
    print("-" * 40)
    for i, (p1, p2, desc) in enumerate(examples, 1):
        print(f"  {i}. {p1} vs {p2}")
        print(f"     {desc}")

def main():
    """Main function for snooker match prediction"""
    parser = argparse.ArgumentParser(description='Snooker Match Predictor')
    parser.add_argument('player1', nargs='?', help='First player name')
    parser.add_argument('player2', nargs='?', help='Second player name')
    parser.add_argument('--tournament', default='ranking_event',
                       choices=['world_championship', 'masters', 'uk_championship',
                               'ranking_event', 'invitational'],
                       help='Tournament type')
    parser.add_argument('--best-of', type=int, default=7,
                       choices=[7, 9, 11, 17, 19, 35],
                       help='Match format (best of X frames)')
    parser.add_argument('--examples', action='store_true',
                       help='Show example player matchups')

    args = parser.parse_args()

    # Show examples if requested
    if args.examples:
        show_examples()
        return

    # If no players specified, go to interactive mode
    if not args.player1 or not args.player2:
        interactive_prediction()
    else:
        # Direct prediction
        print("🎱 SNOOKER MATCH PREDICTOR")
        print("Professional snooker prediction system")
        print("=" * 50)

        result = predict_match_cli(
            args.player1,
            args.player2,
            args.tournament,
            args.best_of
        )

        if not result:
            print(f"\n💡 Try interactive mode: python predict_snooker_match.py")
            print(f"💡 Or see examples: python predict_snooker_match.py --examples")

if __name__ == "__main__":
    main()