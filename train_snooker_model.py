#!/usr/bin/env python3
"""
Snooker Prediction Model Training - Professional snooker machine learning system
Adapted from tennis system with snooker-specific features and optimizations
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys
sys.path.append('src')

from snooker_data_collector import SnookerDataCollector
from snooker_elo_system import SnookerEloSystem

class SnookerModelTrainer:
    """
    Train snooker prediction models using professional match data
    Features snooker-specific machine learning optimization
    """

    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()

    def prepare_training_data(self, matches_df):
        """
        Prepare snooker match data for machine learning training
        """
        print("🔄 PREPARING SNOOKER TRAINING DATA")
        print("Converting matches to machine learning format...")
        print("-" * 50)

        training_data = []

        for idx, match in matches_df.iterrows():
            try:
                # Create training example from winner's perspective (label = 1)
                winner_features = self.extract_match_features(match, perspective='winner')
                winner_features['target'] = 1
                training_data.append(winner_features)

                # Create training example from loser's perspective (label = 0)
                loser_features = self.extract_match_features(match, perspective='loser')
                loser_features['target'] = 0
                training_data.append(loser_features)

                if idx % 5000 == 0:
                    print(f"   Processed {idx:,} matches...")

            except Exception as e:
                continue

        training_df = pd.DataFrame(training_data)

        print(f"\n✅ TRAINING DATA PREPARED!")
        print(f"   📊 Training examples: {len(training_df):,}")
        print(f"   🎱 Features per example: {len(training_df.columns) - 1}")
        print(f"   ⚖️  Class balance: {training_df['target'].value_counts().to_dict()}")

        return training_df

    def extract_match_features(self, match, perspective):
        """
        Extract snooker-specific features from a match record
        """
        if perspective == 'winner':
            player_prefix = 'winner'
            opponent_prefix = 'loser'
        else:
            player_prefix = 'loser'
            opponent_prefix = 'winner'

        features = {
            # ELO features
            'player_elo': match.get(f'{player_prefix}_elo', 1500),
            'opponent_elo': match.get(f'{opponent_prefix}_elo', 1500),
            'elo_difference': match.get(f'{player_prefix}_elo', 1500) - match.get(f'{opponent_prefix}_elo', 1500),

            # Match format features
            'best_of': match.get('best_of', 7),
            'frames_to_win': match.get('frames_to_win', 4),

            # Tournament features
            'tournament_weight': self.get_tournament_weight(match.get('tournament_type', 'ranking_event')),
            'is_world_championship': 1 if match.get('tournament_type') == 'world_championship' else 0,
            'is_major_tournament': 1 if match.get('tournament_type') in ['world_championship', 'masters', 'uk_championship'] else 0,
            'is_ranking_event': 1 if 'ranking' in match.get('tournament_type', '') else 0,

            # Break building features (snooker-specific)
            'player_centuries': match.get(f'{player_prefix}_centuries', 0),
            'opponent_centuries': match.get(f'{opponent_prefix}_centuries', 0),
            'centuries_difference': match.get(f'{player_prefix}_centuries', 0) - match.get(f'{opponent_prefix}_centuries', 0),

            'player_breaks_50_plus': match.get(f'{player_prefix}_breaks_50_plus', 0),
            'opponent_breaks_50_plus': match.get(f'{opponent_prefix}_breaks_50_plus', 0),
            'breaks_50_plus_difference': match.get(f'{player_prefix}_breaks_50_plus', 0) - match.get(f'{opponent_prefix}_breaks_50_plus', 0),

            'player_highest_break': match.get(f'{player_prefix}_highest_break', 0),
            'opponent_highest_break': match.get(f'{opponent_prefix}_highest_break', 0),
            'highest_break_difference': match.get(f'{player_prefix}_highest_break', 0) - match.get(f'{opponent_prefix}_highest_break', 0),

            # Pot success features
            'player_pot_success': match.get(f'{player_prefix}_pot_success', 75),
            'opponent_pot_success': match.get(f'{opponent_prefix}_pot_success', 75),
            'pot_success_difference': match.get(f'{player_prefix}_pot_success', 75) - match.get(f'{opponent_prefix}_pot_success', 75),

            'player_long_pot_success': match.get(f'{player_prefix}_long_pot_success', 60),
            'opponent_long_pot_success': match.get(f'{opponent_prefix}_long_pot_success', 60),
            'long_pot_success_difference': match.get(f'{player_prefix}_long_pot_success', 60) - match.get(f'{opponent_prefix}_long_pot_success', 60),

            # Safety play features
            'player_safety_success': match.get(f'{player_prefix}_safety_success', 70),
            'opponent_safety_success': match.get(f'{opponent_prefix}_safety_success', 70),
            'safety_success_difference': match.get(f'{player_prefix}_safety_success', 70) - match.get(f'{opponent_prefix}_safety_success', 70),

            # Frame control features
            'player_avg_frame_time': match.get(f'{player_prefix}_avg_frame_time', 25),
            'opponent_avg_frame_time': match.get(f'{opponent_prefix}_avg_frame_time', 25),
            'frame_time_difference': match.get(f'{opponent_prefix}_avg_frame_time', 25) - match.get(f'{player_prefix}_avg_frame_time', 25),

            'player_first_visit_clearance': match.get(f'{player_prefix}_first_visit_clearance', 30),
            'opponent_first_visit_clearance': match.get(f'{opponent_prefix}_first_visit_clearance', 30),
            'first_visit_clearance_difference': match.get(f'{player_prefix}_first_visit_clearance', 30) - match.get(f'{opponent_prefix}_first_visit_clearance', 30),

            # Head-to-head features
            'h2h_total_matches': match.get('h2h_total_matches', 0),
            'player_h2h_wins': match.get(f'{player_prefix}_h2h_wins', 0),
            'player_h2h_win_rate': match.get(f'{player_prefix}_h2h_win_rate', 0.5),

            # Frame score analysis
            'frames_played': match.get('frames_won', 0) + match.get('frames_lost', 0),
            'frame_margin': abs(match.get('frames_won', 0) - match.get('frames_lost', 0)),

            # Tournament prestige
            'prize_money': match.get('tournament_prize_money', 200000),
            'prestige_score': self.get_prestige_score(match.get('tournament_prestige', 'medium'))
        }

        return features

    def get_tournament_weight(self, tournament_type):
        """Get tournament weight for feature engineering"""
        weights = {
            'world_championship': 50,
            'masters': 35,
            'uk_championship': 35,
            'champion_of_champions': 30,
            'players_championship': 25,
            'tour_championship': 25,
            'ranking_event': 20,
            'invitational': 15
        }
        return weights.get(tournament_type, 20)

    def get_prestige_score(self, prestige):
        """Convert prestige to numerical score"""
        scores = {
            'highest': 5,
            'very_high': 4,
            'high': 3,
            'medium': 2,
            'low': 1
        }
        return scores.get(prestige, 2)

    def train_models(self, training_df):
        """
        Train multiple snooker prediction models and select the best
        """
        print(f"\n🤖 TRAINING SNOOKER PREDICTION MODELS")
        print("Testing multiple algorithms for optimal performance...")
        print("-" * 50)

        # Prepare features and target
        feature_columns = [col for col in training_df.columns if col != 'target']
        X = training_df[feature_columns]
        y = training_df['target']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training set: {len(X_train):,} examples")
        print(f"Test set: {len(X_test):,} examples")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Models to train
        models_to_train = {
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                num_leaves=31,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                random_state=42,
                verbose=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        }

        results = {}

        for name, model in models_to_train.items():
            print(f"\n🔄 Training {name}...")

            # Use scaled data for some models
            if name in ['Random Forest']:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)

            # Evaluate model
            accuracy = accuracy_score(y_test, y_pred)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

            print(f"   ✅ {name}: {accuracy:.3f} accuracy (CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f})")

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(feature_columns, model.feature_importances_))

        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
        best_model = results[best_model_name]['model']

        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   📊 Accuracy: {results[best_model_name]['accuracy']:.3f}")
        print(f"   📈 Cross-validation: {results[best_model_name]['cv_mean']:.3f} ± {results[best_model_name]['cv_std']:.3f}")

        # Store results
        self.models = results
        self.best_model = best_model
        self.feature_columns = feature_columns

        return best_model, feature_columns, results

    def analyze_feature_importance(self, model_name=None):
        """Analyze which snooker features are most predictive"""
        if not self.feature_importance:
            return

        if model_name and model_name in self.feature_importance:
            importance = self.feature_importance[model_name]
        else:
            # Use the first available model's importance
            importance = list(self.feature_importance.values())[0]

        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        print(f"\n📊 TOP 15 MOST PREDICTIVE SNOOKER FEATURES:")
        print("-" * 60)
        for i, (feature, imp) in enumerate(sorted_features[:15], 1):
            print(f"  {i:2d}. {feature:<35} {imp:.0f}")

    def save_model(self, model, feature_columns):
        """Save the trained snooker model"""
        print(f"\n💾 SAVING SNOOKER PREDICTION MODEL")

        # Create models directory
        os.makedirs('models', exist_ok=True)

        # Save model
        joblib.dump(model, 'models/snooker_prediction_model.pkl')
        print(f"✅ Model saved: models/snooker_prediction_model.pkl")

        # Save feature columns
        joblib.dump(feature_columns, 'models/snooker_features.pkl')
        print(f"✅ Features saved: models/snooker_features.pkl")

        # Save scaler
        joblib.dump(self.scaler, 'models/snooker_scaler.pkl')
        print(f"✅ Scaler saved: models/snooker_scaler.pkl")

def main():
    """Train the snooker prediction model"""
    print("🎱 SNOOKER PREDICTION MODEL TRAINING")
    print("Professional snooker machine learning system")
    print("Adapted from tennis system with snooker features")
    print("=" * 60)

    trainer = SnookerModelTrainer()

    # Step 1: Generate snooker dataset
    print("📊 STEP 1: GENERATING SNOOKER DATASET")
    collector = SnookerDataCollector()

    # Check if data already exists
    if os.path.exists('data/snooker_matches.csv'):
        print("📂 Loading existing snooker dataset...")
        matches_df = pd.read_csv('data/snooker_matches.csv')
        print(f"✅ Loaded {len(matches_df):,} matches")
    else:
        print("🔄 Generating new snooker dataset...")
        matches_df = collector.generate_snooker_dataset(25000)
        enhanced_df = collector.enhance_with_head_to_head(matches_df)
        collector.save_snooker_data(enhanced_df)
        matches_df = enhanced_df

    # Step 2: Prepare training data
    print(f"\n📊 STEP 2: PREPARING TRAINING DATA")
    training_df = trainer.prepare_training_data(matches_df)

    # Step 3: Train models
    print(f"\n🤖 STEP 3: TRAINING MODELS")
    best_model, feature_columns, results = trainer.train_models(training_df)

    # Step 4: Analyze features
    print(f"\n📈 STEP 4: FEATURE ANALYSIS")
    trainer.analyze_feature_importance()

    # Step 5: Save model
    print(f"\n💾 STEP 5: SAVING MODEL")
    trainer.save_model(best_model, feature_columns)

    print(f"\n🚀 SNOOKER MODEL TRAINING COMPLETE!")
    print(f"✅ Model ready for snooker match predictions!")
    print(f"📊 Training data: {len(training_df):,} examples")
    print(f"🎱 Features: {len(feature_columns)} snooker-specific features")
    print(f"🏆 Best algorithm: {max(results.keys(), key=lambda k: results[k]['cv_mean'])}")
    print(f"🎯 Test accuracy: {max(results.values(), key=lambda v: v['cv_mean'])['accuracy']:.1%}")

    print(f"\n🎱 Ready to predict snooker matches!")
    print(f"Usage: python predict_snooker_match.py \"Ronnie O'Sullivan\" \"Judd Trump\"")

if __name__ == "__main__":
    main()