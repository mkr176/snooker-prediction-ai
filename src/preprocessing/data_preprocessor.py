import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from typing import Dict, List, Tuple, Optional
import json

class SnookerDataPreprocessor:
    """
    Preprocessor for snooker data to create features for machine learning models.
    Handles player statistics, match history, head-to-head records, and tournament data.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_and_merge_data(self, data_path: str = "../data/raw") -> pd.DataFrame:
        """
        Load and merge all data sources into a single DataFrame.

        Args:
            data_path: Path to raw data directory

        Returns:
            Merged DataFrame with all match and player data
        """
        try:
            # Load different data sources
            matches_df = self._load_matches_data(data_path)
            players_df = self._load_players_data(data_path)
            tournaments_df = self._load_tournaments_data(data_path)

            # Merge data
            merged_df = self._merge_datasets(matches_df, players_df, tournaments_df)

            self.logger.info(f"Loaded and merged data: {len(merged_df)} records")
            return merged_df

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return pd.DataFrame()

    def _load_matches_data(self, data_path: str) -> pd.DataFrame:
        """Load matches data from multiple sources."""
        matches_data = []

        # Sample match data for demonstration
        sample_matches = [
            {
                'match_id': 'WC2024_F001',
                'tournament': 'World Championship',
                'season': '2023-24',
                'round': 'Final',
                'player1': 'Ronnie OSullivan',
                'player2': 'Judd Trump',
                'score1': 18,
                'score2': 13,
                'winner': 'Ronnie OSullivan',
                'date': '2024-05-06',
                'venue': 'Crucible Theatre',
                'session_time': 'Evening',
                'match_duration_minutes': 380,
                'player1_breaks': [147, 100, 85, 92, 76, 68, 55],
                'player2_breaks': [134, 88, 71, 95, 62, 59],
                'player1_centuries': 5,
                'player2_centuries': 4,
                'player1_50_plus': 7,
                'player2_50_plus': 6,
                'total_points_player1': 1450,
                'total_points_player2': 1250
            },
            {
                'match_id': 'WC2024_SF001',
                'tournament': 'World Championship',
                'season': '2023-24',
                'round': 'Semi-Final',
                'player1': 'Judd Trump',
                'player2': 'Kyren Wilson',
                'score1': 17,
                'score2': 11,
                'winner': 'Judd Trump',
                'date': '2024-05-04',
                'venue': 'Crucible Theatre',
                'session_time': 'Afternoon',
                'match_duration_minutes': 320,
                'player1_breaks': [112, 95, 88, 73, 65],
                'player2_breaks': [101, 87, 69, 58],
                'player1_centuries': 4,
                'player2_centuries': 3,
                'player1_50_plus': 5,
                'player2_50_plus': 4,
                'total_points_player1': 1320,
                'total_points_player2': 1100
            }
        ]

        return pd.DataFrame(sample_matches)

    def _load_players_data(self, data_path: str) -> pd.DataFrame:
        """Load player statistics and rankings data."""
        players_data = [
            {
                'player_name': 'Ronnie OSullivan',
                'ranking': 1,
                'points': 1150000,
                'prize_money': 2800000,
                'country': 'England',
                'age': 48,
                'professional_since': 1992,
                'world_titles': 7,
                'ranking_titles': 39,
                'career_centuries': 1200,
                'career_147s': 15
            },
            {
                'player_name': 'Judd Trump',
                'ranking': 2,
                'points': 980000,
                'prize_money': 2200000,
                'country': 'England',
                'age': 34,
                'professional_since': 2005,
                'world_titles': 1,
                'ranking_titles': 25,
                'career_centuries': 850,
                'career_147s': 8
            },
            {
                'player_name': 'Kyren Wilson',
                'ranking': 7,
                'points': 520000,
                'prize_money': 1100000,
                'country': 'England',
                'age': 32,
                'professional_since': 2010,
                'world_titles': 0,
                'ranking_titles': 6,
                'career_centuries': 400,
                'career_147s': 2
            }
        ]

        return pd.DataFrame(players_data)

    def _load_tournaments_data(self, data_path: str) -> pd.DataFrame:
        """Load tournament information data."""
        tournaments_data = [
            {
                'tournament': 'World Championship',
                'venue': 'Crucible Theatre',
                'prize_fund': 2395000,
                'winner_prize': 500000,
                'type': 'Ranking',
                'field_size': 32,
                'prestige_factor': 10.0
            },
            {
                'tournament': 'UK Championship',
                'venue': 'York Barbican',
                'prize_fund': 1200000,
                'winner_prize': 250000,
                'type': 'Ranking',
                'field_size': 128,
                'prestige_factor': 8.0
            },
            {
                'tournament': 'Masters',
                'venue': 'Alexandra Palace',
                'prize_fund': 800000,
                'winner_prize': 250000,
                'type': 'Invitational',
                'field_size': 16,
                'prestige_factor': 9.0
            }
        ]

        return pd.DataFrame(tournaments_data)

    def _merge_datasets(self, matches_df: pd.DataFrame, players_df: pd.DataFrame,
                       tournaments_df: pd.DataFrame) -> pd.DataFrame:
        """Merge all datasets into a comprehensive DataFrame."""
        # Merge match data with player1 stats
        merged = matches_df.merge(
            players_df.add_suffix('_p1'),
            left_on='player1',
            right_on='player_name_p1',
            how='left'
        )

        # Merge with player2 stats
        merged = merged.merge(
            players_df.add_suffix('_p2'),
            left_on='player2',
            right_on='player_name_p2',
            how='left'
        )

        # Merge with tournament data
        merged = merged.merge(tournaments_df, on='tournament', how='left')

        return merged

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create machine learning features from raw data.

        Args:
            df: Merged DataFrame with all data

        Returns:
            DataFrame with engineered features
        """
        features_df = df.copy()

        # Basic match features
        features_df['total_frames'] = features_df['score1'] + features_df['score2']
        features_df['frame_difference'] = abs(features_df['score1'] - features_df['score2'])
        features_df['match_competitiveness'] = 1 - (features_df['frame_difference'] / features_df['total_frames'])

        # Player ranking features
        features_df['ranking_difference'] = abs(features_df['ranking_p1'] - features_df['ranking_p2'])
        features_df['ranking_advantage_p1'] = features_df['ranking_p2'] - features_df['ranking_p1']  # Positive if p1 ranked higher
        features_df['favorite_player'] = np.where(features_df['ranking_p1'] < features_df['ranking_p2'], 1, 2)

        # Experience features
        features_df['experience_difference'] = abs(
            (features_df['age_p1'] - features_df['professional_since_p1']) -
            (features_df['age_p2'] - features_df['professional_since_p2'])
        )

        # Title achievements features
        features_df['total_titles_p1'] = features_df['world_titles_p1'] + features_df['ranking_titles_p1']
        features_df['total_titles_p2'] = features_df['world_titles_p2'] + features_df['ranking_titles_p2']
        features_df['title_difference'] = features_df['total_titles_p1'] - features_df['total_titles_p2']

        # Break-making features
        features_df['centuries_per_frame_p1'] = features_df['player1_centuries'] / features_df['score1']
        features_df['centuries_per_frame_p2'] = features_df['player2_centuries'] / features_df['score2']
        features_df['century_advantage_p1'] = features_df['centuries_per_frame_p1'] - features_df['centuries_per_frame_p2']

        # Scoring efficiency features
        features_df['avg_points_per_frame_p1'] = features_df['total_points_player1'] / features_df['score1']
        features_df['avg_points_per_frame_p2'] = features_df['total_points_player2'] / features_df['score2']
        features_df['scoring_efficiency_p1'] = features_df['avg_points_per_frame_p1'] - features_df['avg_points_per_frame_p2']

        # Tournament importance features
        features_df['high_prestige'] = (features_df['prestige_factor'] >= 8.0).astype(int)
        features_df['world_championship'] = (features_df['tournament'] == 'World Championship').astype(int)

        # Date and time features
        features_df['date'] = pd.to_datetime(features_df['date'])
        features_df['day_of_week'] = features_df['date'].dt.dayofweek
        features_df['month'] = features_df['date'].dt.month
        features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)

        # Session timing
        features_df['evening_session'] = (features_df['session_time'] == 'Evening').astype(int)

        return features_df

    def add_historical_features(self, df: pd.DataFrame, window_size: int = 10) -> pd.DataFrame:
        """
        Add historical performance features for each player.

        Args:
            df: DataFrame with match data
            window_size: Number of previous matches to consider

        Returns:
            DataFrame with historical features
        """
        df_with_history = df.copy()
        df_with_history = df_with_history.sort_values(['player1', 'date']).reset_index(drop=True)

        # Calculate rolling statistics for each player
        for player_col in ['player1', 'player2']:
            score_col = 'score1' if player_col == 'player1' else 'score2'
            opponent_score_col = 'score2' if player_col == 'player1' else 'score1'

            # Recent form (wins in last N matches)
            df_with_history[f'recent_wins_{player_col}'] = (
                df_with_history.groupby(player_col)[f'{score_col}']
                .rolling(window=window_size, min_periods=1)
                .apply(lambda x: sum(x > df_with_history.loc[x.index, opponent_score_col]))
                .reset_index(0, drop=True)
            )

            # Recent win percentage
            df_with_history[f'recent_win_pct_{player_col}'] = (
                df_with_history[f'recent_wins_{player_col}'] / window_size
            ).fillna(0.5)

            # Recent average score
            df_with_history[f'recent_avg_score_{player_col}'] = (
                df_with_history.groupby(player_col)[score_col]
                .rolling(window=window_size, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )

        return df_with_history

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features for machine learning.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()

        categorical_columns = [
            'tournament', 'round', 'venue', 'session_time', 'country_p1', 'country_p2', 'type'
        ]

        for column in categorical_columns:
            if column in df_encoded.columns:
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()

                # Handle missing values
                df_encoded[column] = df_encoded[column].fillna('Unknown')
                df_encoded[f'{column}_encoded'] = self.label_encoders[column].fit_transform(df_encoded[column])

        return df_encoded

    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable for prediction.

        Args:
            df: DataFrame with match results

        Returns:
            DataFrame with target variable
        """
        df_with_target = df.copy()

        # Binary target: 1 if player1 wins, 0 if player2 wins
        df_with_target['player1_wins'] = (df_with_target['winner'] == df_with_target['player1']).astype(int)

        # Multi-class target for score prediction
        df_with_target['score_margin'] = abs(df_with_target['score1'] - df_with_target['score2'])
        df_with_target['margin_category'] = pd.cut(
            df_with_target['score_margin'],
            bins=[0, 2, 5, 10, float('inf')],
            labels=['Close', 'Moderate', 'Comfortable', 'Dominant'],
            include_lowest=True
        )

        return df_with_target

    def prepare_training_data(self, df: pd.DataFrame, feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for model training.

        Args:
            df: DataFrame with all features
            feature_columns: Specific columns to use as features

        Returns:
            Tuple of (X, y, feature_names)
        """
        if feature_columns is None:
            # Select numerical features automatically
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            # Remove target variables and identifiers
            exclude_columns = [
                'match_id', 'score1', 'score2', 'player1_wins', 'winner',
                'date', 'total_points_player1', 'total_points_player2'
            ]

            feature_columns = [col for col in numerical_columns if col not in exclude_columns]

        # Prepare features
        X = df[feature_columns].fillna(0).values

        # Prepare target
        y = df['player1_wins'].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        self.logger.info(f"Prepared training data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")

        return X_scaled, y, feature_columns

    def save_preprocessor(self, filepath: str):
        """Save the preprocessor state for later use."""
        preprocessor_state = {
            'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
            'label_encoders': {
                name: encoder.classes_.tolist()
                for name, encoder in self.label_encoders.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(preprocessor_state, f, indent=2)

        self.logger.info(f"Preprocessor state saved to {filepath}")

    def load_preprocessor(self, filepath: str):
        """Load preprocessor state from file."""
        with open(filepath, 'r') as f:
            preprocessor_state = json.load(f)

        if preprocessor_state['scaler_mean'] is not None:
            self.scaler.mean_ = np.array(preprocessor_state['scaler_mean'])
            self.scaler.scale_ = np.array(preprocessor_state['scaler_scale'])

        for name, classes in preprocessor_state['label_encoders'].items():
            encoder = LabelEncoder()
            encoder.classes_ = np.array(classes)
            self.label_encoders[name] = encoder

        self.logger.info(f"Preprocessor state loaded from {filepath}")


def main():
    """Main function to demonstrate the preprocessor."""
    preprocessor = SnookerDataPreprocessor()

    # Load and process data
    print("Loading and merging data...")
    df = preprocessor.load_and_merge_data()

    print("Creating features...")
    df_with_features = preprocessor.create_features(df)

    print("Adding historical features...")
    df_with_history = preprocessor.add_historical_features(df_with_features)

    print("Encoding categorical features...")
    df_encoded = preprocessor.encode_categorical_features(df_with_history)

    print("Creating target variable...")
    df_final = preprocessor.create_target_variable(df_encoded)

    print("Preparing training data...")
    X, y, feature_names = preprocessor.prepare_training_data(df_final)

    print(f"\nPreprocessing complete:")
    print(f"- Total samples: {len(df_final)}")
    print(f"- Features: {len(feature_names)}")
    print(f"- Training data shape: {X.shape}")

    # Save preprocessor
    preprocessor.save_preprocessor("../config/preprocessor_state.json")

    return df_final, X, y, feature_names


if __name__ == "__main__":
    main()