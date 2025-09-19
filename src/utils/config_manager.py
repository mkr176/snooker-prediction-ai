import yaml
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """
    Configuration manager for the Snooker Prediction AI system.
    Handles loading, validation, and access to configuration parameters.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the configuration file
        """
        if config_path is None:
            # Default path relative to this file
            current_dir = Path(__file__).parent
            config_path = current_dir.parent.parent / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_logging()

        self.logger = logging.getLogger(__name__)

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)

            # Resolve environment variables
            config = self._resolve_env_variables(config)

            # Validate configuration
            self._validate_config(config)

            return config

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")

    def _resolve_env_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve environment variables in configuration values.

        Args:
            config: Configuration dictionary

        Returns:
            Configuration with resolved environment variables
        """
        def resolve_value(value):
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                return os.getenv(env_var, value)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            else:
                return value

        return resolve_value(config)

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration parameters.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        required_sections = [
            'data_collection', 'preprocessing', 'training',
            'evaluation', 'prediction', 'paths'
        ]

        for section in required_sections:
            if section not in config:
                raise ValueError(f"Required configuration section missing: {section}")

        # Validate specific parameters
        if config['training']['validation_split'] <= 0 or config['training']['validation_split'] >= 1:
            raise ValueError("validation_split must be between 0 and 1")

        if config['preprocessing']['features']['form_window'] < 1:
            raise ValueError("form_window must be at least 1")

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_config = self.config.get('logging', {})

        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Create log directory if it doesn't exist
        log_dir = Path(self.get('paths.log_directory', './logs'))
        log_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_dir / 'snooker_prediction.log'),
                logging.StreamHandler()
            ] if log_config.get('console_logging', True) else [
                logging.FileHandler(log_dir / 'snooker_prediction.log')
            ]
        )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., 'training.models.xgboost.params')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.

        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Model configuration
        """
        model_config = self.get(f'training.models.{model_name}', {})
        if not model_config:
            raise ValueError(f"Configuration not found for model: {model_name}")

        return model_config

    def is_model_enabled(self, model_name: str) -> bool:
        """
        Check if a model is enabled for training.

        Args:
            model_name: Name of the model

        Returns:
            True if model is enabled
        """
        return self.get(f'training.models.{model_name}.enabled', False)

    def get_enabled_models(self) -> list:
        """
        Get list of enabled models.

        Returns:
            List of enabled model names
        """
        models = self.get('training.models', {})
        return [name for name, config in models.items() if config.get('enabled', False)]

    def get_data_paths(self) -> Dict[str, str]:
        """
        Get all data-related paths.

        Returns:
            Dictionary of data paths
        """
        paths = self.get('paths', {})
        return {
            'raw_data': paths.get('raw_data', './data/raw'),
            'processed_data': paths.get('processed_data', './data/processed'),
            'tournament_data': paths.get('tournament_data', './data/tournaments'),
            'trained_models': paths.get('trained_models', './models/trained'),
            'model_checkpoints': paths.get('model_checkpoints', './models/checkpoints')
        }

    def get_feature_config(self) -> Dict[str, Any]:
        """
        Get feature engineering configuration.

        Returns:
            Feature configuration
        """
        return self.get('preprocessing.features', {})

    def get_evaluation_metrics(self) -> list:
        """
        Get list of evaluation metrics to calculate.

        Returns:
            List of metric names
        """
        return self.get('evaluation.metrics', ['accuracy', 'roc_auc'])

    def get_tournament_config(self, tournament_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific tournament.

        Args:
            tournament_name: Name of the tournament

        Returns:
            Tournament configuration
        """
        # Convert tournament name to config key format
        config_key = tournament_name.lower().replace(' ', '_').replace("'", "")
        return self.get(f'tournaments.{config_key}', {})

    def save_config(self, output_path: str = None) -> None:
        """
        Save current configuration to file.

        Args:
            output_path: Path to save configuration (defaults to original path)
        """
        if output_path is None:
            output_path = self.config_path

        with open(output_path, 'w', encoding='utf-8') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)

    def create_directories(self) -> None:
        """Create all directories specified in the configuration."""
        paths = self.get_data_paths()

        for path_name, path_value in paths.items():
            Path(path_value).mkdir(parents=True, exist_ok=True)

        # Create additional directories
        additional_paths = [
            self.get('paths.log_directory', './logs'),
            self.get('paths.predictions_output', './output/predictions'),
            self.get('paths.reports_output', './output/reports')
        ]

        for path in additional_paths:
            Path(path).mkdir(parents=True, exist_ok=True)

    def validate_paths(self) -> Dict[str, bool]:
        """
        Validate that all configured paths exist.

        Returns:
            Dictionary mapping path names to existence status
        """
        paths = self.get_data_paths()
        validation_results = {}

        for path_name, path_value in paths.items():
            validation_results[path_name] = Path(path_value).exists()

        return validation_results

    def get_hyperparameter_grid(self, model_name: str) -> Dict[str, list]:
        """
        Get hyperparameter search grid for a model.

        Args:
            model_name: Name of the model

        Returns:
            Hyperparameter grid dictionary
        """
        # Default hyperparameter grids
        default_grids = {
            'xgboost': {
                'n_estimators': [200, 300, 500],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'lightgbm': {
                'n_estimators': [200, 300, 500],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'num_leaves': [31, 50, 70],
                'feature_fraction': [0.8, 0.9, 1.0]
            }
        }

        # Check if custom grid is defined in config
        custom_grid = self.get(f'hyperparameter_grids.{model_name}')
        if custom_grid:
            return custom_grid

        return default_grids.get(model_name, {})

    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"ConfigManager(config_path={self.config_path})"

    def __repr__(self) -> str:
        """Representation of the configuration."""
        return self.__str__()


# Global configuration instance
_config_instance = None

def get_config(config_path: str = None) -> ConfigManager:
    """
    Get the global configuration instance.

    Args:
        config_path: Path to configuration file

    Returns:
        ConfigManager instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = ConfigManager(config_path)

    return _config_instance


def load_config(config_path: str = None) -> ConfigManager:
    """
    Load configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)


def main():
    """Main function to demonstrate the configuration manager."""
    config = ConfigManager()

    print("Configuration loaded successfully!")
    print(f"Config path: {config.config_path}")

    print("\nEnabled models:")
    for model in config.get_enabled_models():
        print(f"  - {model}")

    print("\nData paths:")
    paths = config.get_data_paths()
    for name, path in paths.items():
        print(f"  {name}: {path}")

    print("\nCreating directories...")
    config.create_directories()

    print("\nValidating paths...")
    validation = config.validate_paths()
    for name, exists in validation.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {paths[name]}")

    # Example of accessing specific configuration values
    print(f"\nForm window size: {config.get('preprocessing.features.form_window')}")
    print(f"Default model: {config.get('prediction.default_model')}")
    print(f"Validation split: {config.get('training.validation_split')}")


if __name__ == "__main__":
    main()