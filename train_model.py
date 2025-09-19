#!/usr/bin/env python3
"""
Main training script for Snooker Prediction AI.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from utils.config_manager import ConfigManager
from preprocessing.data_preprocessor import SnookerDataPreprocessor
from models.snooker_models import SnookerPredictionModels

def setup_logging():
    """Set up basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    """Main training pipeline."""
    logger = setup_logging()
    logger.info("Starting Snooker Prediction AI Training...")

    try:
        # Load configuration
        config = ConfigManager()

        # Create directories
        config.create_directories()

        # Initialize components
        preprocessor = SnookerDataPreprocessor()
        models = SnookerPredictionModels()

        # For now, create sample data since real data collection is in progress
        logger.info("Creating sample training data...")

        # Generate realistic sample data
        np.random.seed(42)
        n_samples = 500

        # Sample data with realistic snooker features
        X = np.random.randn(n_samples, 20)  # 20 features

        # Create more realistic target based on features
        prob = 1 / (1 + np.exp(-(X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.3)))
        y = np.random.binomial(1, prob)

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")

        # Train models
        logger.info("Training models...")
        trained_models = models.train_all_models(X_train, y_train, X_val, y_val)

        # Get performance summary
        summary = models.get_model_summary()
        logger.info("Model Performance:")
        print(summary)

        # Save models with absolute path
        model_path = str(Path(__file__).parent / "models" / "trained")
        models.save_models(model_path)
        logger.info(f"Models saved to: {model_path}")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)