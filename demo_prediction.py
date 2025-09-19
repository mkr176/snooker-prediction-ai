#!/usr/bin/env python3
"""
Simple demo prediction for snooker matches
"""
import numpy as np
import pandas as pd
from datetime import datetime

def predict_snooker_match(player1, player2, tournament="General"):
    """
    Demo prediction function using simple rules and probabilities
    """

    # Player rankings (simulated)
    player_rankings = {
        "Ronnie O'Sullivan": 1,
        "Judd Trump": 2,
        "Mark Selby": 3,
        "Neil Robertson": 4,
        "John Higgins": 5,
        "Shaun Murphy": 6,
        "Kyren Wilson": 7,
        "Mark Williams": 8,
        "Stuart Bingham": 9,
        "Anthony McGill": 10
    }

    # Tournament importance weights
    tournament_weights = {
        "World Championship": 1.0,
        "UK Championship": 0.8,
        "Masters": 0.9,
        "Champion of Champions": 0.7,
        "General": 0.6
    }

    # Get rankings (default to 50 if not found)
    rank1 = player_rankings.get(player1, 50)
    rank2 = player_rankings.get(player2, 50)

    # Calculate basic probability based on ranking difference
    rank_diff = rank2 - rank1  # Positive if player1 ranked higher

    # Convert ranking difference to probability
    base_prob = 1 / (1 + np.exp(-rank_diff * 0.1))

    # Apply tournament weight
    tournament_weight = tournament_weights.get(tournament, 0.6)

    # Add some randomness for realism
    np.random.seed(hash(player1 + player2) % 2**32)
    noise = np.random.normal(0, 0.05)

    # Final probability
    final_prob = np.clip(base_prob + noise, 0.1, 0.9)

    # Determine winner
    if final_prob > 0.5:
        winner = player1
        probability = final_prob
    else:
        winner = player2
        probability = 1 - final_prob

    # Calculate confidence
    confidence = abs(final_prob - 0.5) * 2

    return {
        'player1': player1,
        'player2': player2,
        'tournament': tournament,
        'predicted_winner': winner,
        'win_probability': probability,
        'confidence': confidence,
        'player1_rank': rank1,
        'player2_rank': rank2
    }

def main():
    import sys

    if len(sys.argv) >= 3:
        player1 = sys.argv[1]
        player2 = sys.argv[2]
        tournament = sys.argv[3] if len(sys.argv) > 3 else "General"
    else:
        print("Usage: python demo_prediction.py 'Player 1' 'Player 2' [Tournament]")
        print("Example: python demo_prediction.py 'Ronnie O\\'Sullivan' 'Judd Trump' 'World Championship'")
        return

    # Make prediction
    result = predict_snooker_match(player1, player2, tournament)

    # Display results
    print("\n" + "="*60)
    print("🎱 SNOOKER MATCH PREDICTION")
    print("="*60)
    print(f"Match: {result['player1']} vs {result['player2']}")
    print(f"Tournament: {result['tournament']}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*60)

    print(f"\n🏆 PREDICTED WINNER: {result['predicted_winner']}")
    print(f"📊 Win Probability: {result['win_probability']:.1%}")
    print(f"🎯 Confidence Level: {result['confidence']:.1%}")

    print(f"\n📈 PLAYER RANKINGS:")
    print(f"   {result['player1']}: #{result['player1_rank']}")
    print(f"   {result['player2']}: #{result['player2_rank']}")

    # Confidence interpretation
    if result['confidence'] > 0.6:
        confidence_text = "HIGH - Strong prediction"
    elif result['confidence'] > 0.3:
        confidence_text = "MEDIUM - Moderate prediction"
    else:
        confidence_text = "LOW - Close match expected"

    print(f"\n💡 Confidence: {confidence_text}")

    # Sample betting odds (demo)
    if result['win_probability'] > 0.65:
        print(f"💰 Betting Recommendation: Consider backing {result['predicted_winner']}")
    else:
        print("💰 Betting Recommendation: Close match - be cautious")

    print("\n" + "="*60)
    print("Note: This is a demonstration using simulated AI prediction")
    print("Real system uses comprehensive tournament data and ML models")
    print("="*60)

if __name__ == "__main__":
    main()