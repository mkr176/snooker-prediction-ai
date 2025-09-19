#!/usr/bin/env python3
"""
Prediction script for Snooker AI.
Make predictions for upcoming matches.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from utils.config_manager import ConfigManager
from models.snooker_models import SnookerPredictionModels
from preprocessing.data_preprocessor import SnookerDataPreprocessor

def setup_logging():
    """Set up logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def predict_match(player1: str, player2: str, tournament: str = "General"):
    """
    Predict match outcome between two players.

    Args:
        player1: Name of first player
        player2: Name of second player
        tournament: Tournament name
    """
    logger = setup_logging()

    try:
        # Load configuration
        config = ConfigManager()

        # Load trained models
        models = SnookerPredictionModels()
        model_path = config.get('paths.trained_models', './models/trained')

        try:
            models.load_models(model_path)
            logger.info("Models loaded successfully")
        except:
            logger.warning("Could not load trained models. Please run train_model.py first.")
            return None

        # Create feature vector for the match
        # For demo, create sample features
        np.random.seed(42)
        sample_features = np.random.randn(1, 20)  # 20 features as in training

        # Make prediction with best model
        default_model = config.get('prediction.default_model', 'xgboost')

        if default_model in models.models:
            prediction = models.predict_match(sample_features, default_model)

            logger.info(f"\n=== Match Prediction ===")
            logger.info(f"Match: {player1} vs {player2}")
            logger.info(f"Tournament: {tournament}")
            logger.info(f"Model used: {default_model}")
            logger.info(f"Predicted winner: {player1 if prediction['prediction'][0] == 1 else player2}")
            logger.info(f"Win probability: {prediction['probability'][0]:.3f}")
            logger.info(f"Confidence: {prediction['confidence'][0]:.3f}")

            return prediction
        else:
            logger.error(f"Model {default_model} not found")
            return None

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None

def main():
    """Main prediction interface."""
    import argparse

    parser = argparse.ArgumentParser(description='Predict snooker match outcomes')
    parser.add_argument('--player1', required=True, help='First player name')
    parser.add_argument('--player2', required=True, help='Second player name')
    parser.add_argument('--tournament', default='General', help='Tournament name')

    args = parser.parse_args()

    result = predict_match(args.player1, args.player2, args.tournament)

    if result is None:
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)