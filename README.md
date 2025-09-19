# Snooker Prediction AI 🎱

A comprehensive machine learning system for predicting snooker match outcomes using player statistics, historical data, and tournament information.

## Features

- **Data Collection**: Automated scraping and API collection from snooker tournaments
- **Advanced ML Models**: XGBoost, LightGBM, Random Forest, Neural Networks, and ensemble methods
- **Feature Engineering**: Head-to-head records, recent form, ranking differences, tournament importance
- **Real-time Predictions**: Predict match outcomes with confidence scores
- **Value Betting**: Calculate expected value for betting opportunities
- **Tournament Analysis**: Comprehensive analysis of major tournaments

## Project Structure

```
snooker-prediction-ai/
├── config/
│   ├── config.yaml                 # Main configuration file
│   └── preprocessor_state.json     # Saved preprocessor state
├── data/
│   ├── raw/                        # Raw collected data
│   ├── processed/                  # Processed training data
│   └── tournaments/                # Tournament-specific data
├── src/
│   ├── data_collection/
│   │   ├── snooker_scraper.py      # Web scraping utilities
│   │   └── api_collector.py        # API data collection
│   ├── preprocessing/
│   │   └── data_preprocessor.py    # Data preprocessing and feature engineering
│   ├── models/
│   │   └── snooker_models.py       # ML models and training
│   └── utils/
│       ├── config_manager.py       # Configuration management
│       └── prediction_utils.py     # Prediction utilities
├── models/
│   ├── trained/                    # Saved trained models
│   └── checkpoints/                # Model checkpoints
├── notebooks/                      # Jupyter notebooks for analysis
├── tests/                          # Unit tests
├── requirements.txt                # Python dependencies
├── train_model.py                  # Main training script
└── predict.py                      # Prediction script
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd snooker-prediction-ai
```

2. **Create virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train Models

```bash
# Train all models with default configuration
python train_model.py

# Train specific models
python train_model.py --models xgboost lightgbm

# Skip data collection (use existing data)
python train_model.py --skip-data-collection
```

### 2. Make Predictions

```bash
# Predict match outcome
python predict.py --player1 "Ronnie O'Sullivan" --player2 "Judd Trump" --tournament "World Championship"

# Simple prediction
python predict.py --player1 "Mark Selby" --player2 "Neil Robertson"
```

### 3. Collect Data

```python
from src.data_collection.snooker_scraper import SnookerScraper
from src.data_collection.api_collector import SnookerAPICollector

# Initialize collectors
scraper = SnookerScraper()
api_collector = SnookerAPICollector()

# Collect tournament data
tournaments = scraper.get_tournaments("2023-24")
matches = scraper.get_match_results("World Championship", "2023-24")

# Collect live data
rankings = api_collector.get_live_rankings()
live_scores = api_collector.get_live_scores()
```

## Configuration

The system uses a YAML configuration file (`config/config.yaml`) to manage all settings:

### Key Configuration Sections

- **data_collection**: API settings, scraping parameters, data sources
- **preprocessing**: Feature engineering, scaling, missing value handling
- **training**: Model parameters, cross-validation, ensemble settings
- **evaluation**: Metrics, validation strategies
- **prediction**: Default models, confidence thresholds

### Example Configuration

```yaml
training:
  models:
    xgboost:
      enabled: true
      hyperparameter_tuning: true
      params:
        n_estimators: 300
        max_depth: 8
        learning_rate: 0.1

prediction:
  default_model: "xgboost"
  confidence_thresholds:
    low: 0.1
    medium: 0.3
    high: 0.5
```

## Models

### Supported Models

1. **XGBoost**: Gradient boosting with excellent performance on tabular data
2. **LightGBM**: Fast gradient boosting with categorical feature support
3. **Random Forest**: Ensemble of decision trees with feature importance
4. **Neural Network**: Deep learning model with dropout and batch normalization
5. **Logistic Regression**: Linear baseline model
6. **Ensemble**: Weighted combination of top-performing models

### Model Performance

The system automatically evaluates models using:
- Accuracy
- ROC-AUC
- Precision/Recall
- F1-Score
- Cross-validation

## Features

### Engineered Features

The system creates comprehensive features for prediction:

#### Player Features
- Current world ranking
- Recent form (win/loss record)
- Average break scores
- Century count statistics
- Career achievements

#### Head-to-Head Features
- Historical match record
- Recent head-to-head form
- Average frames per match
- Venue-specific performance

#### Tournament Features
- Tournament importance weight
- Prize money
- Venue characteristics
- Round significance

#### Temporal Features
- Day of week effects
- Session timing (afternoon/evening)
- Seasonal performance trends

## Usage Examples

### Advanced Prediction with Custom Features

```python
from src.utils.prediction_utils import SnookerPredictionUtils
from src.models.snooker_models import SnookerPredictionModels

# Initialize utilities
utils = SnookerPredictionUtils()
models = SnookerPredictionModels()

# Load trained models
models.load_models("./models/trained")

# Prepare features for prediction
features = utils.prepare_prediction_features(
    player1="Ronnie O'Sullivan",
    player2="Judd Trump",
    tournament="World Championship",
    player1_rank=1,
    player2_rank=2,
    venue="Crucible Theatre"
)

# Make prediction
prediction = models.predict_match(features, "xgboost")

# Interpret results
interpretation = utils.interpret_prediction(prediction, "Ronnie O'Sullivan", "Judd Trump")
print(interpretation['match_summary'])
```

### Value Betting Analysis

```python
# Calculate value betting opportunity
value_analysis = utils.calculate_value_bet(
    prediction_prob=0.65,    # Model probability
    bookmaker_odds=2.1,      # Bookmaker decimal odds
    stake=10.0               # Stake amount
)

if value_analysis['is_value_bet']:
    print(f"Value bet detected! Edge: {value_analysis['edge']:.3f}")
    print(f"Expected value: ${value_analysis['expected_value']:.2f}")
```

## Data Sources

### Supported Tournaments

- World Championship
- UK Championship
- Masters
- Champion of Champions
- Welsh Open
- Players Championship
- Gibraltar Open
- German Masters
- European Masters
- English Open

### Data Collection Methods

1. **Web Scraping**: Automated collection from official snooker websites
2. **API Integration**: Real-time data from snooker APIs
3. **Manual Data Entry**: Support for custom tournament data

## Model Training

### Training Pipeline

1. **Data Collection**: Gather tournament and player data
2. **Preprocessing**: Clean data, engineer features, encode categories
3. **Feature Selection**: Select most predictive features
4. **Model Training**: Train multiple models with cross-validation
5. **Hyperparameter Tuning**: Optimize model parameters
6. **Ensemble Creation**: Combine best models
7. **Evaluation**: Comprehensive model evaluation
8. **Model Saving**: Save trained models for prediction

### Training Commands

```bash
# Full training pipeline
python train_model.py

# Hyperparameter tuning for specific model
python train_model.py --models xgboost --tune-hyperparameters

# Evaluation only (requires existing models)
python train_model.py --evaluation-only
```

## Evaluation and Monitoring

### Performance Metrics

- **Accuracy**: Overall prediction accuracy
- **ROC-AUC**: Area under the ROC curve
- **Precision/Recall**: For both classes (player1 win/loss)
- **Calibration**: How well probabilities match outcomes
- **Feature Importance**: Most predictive features

### Model Monitoring

- Performance tracking over time
- Data drift detection
- Automatic retraining triggers
- Alert system for performance degradation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Dependencies

### Core Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library
- **xgboost**: Gradient boosting framework
- **lightgbm**: Gradient boosting framework
- **tensorflow**: Deep learning framework

### Data Collection

- **requests**: HTTP library
- **beautifulsoup4**: Web scraping
- **selenium**: Browser automation

### Visualization

- **matplotlib**: Plotting library
- **seaborn**: Statistical data visualization
- **plotly**: Interactive visualizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- World Snooker Tour for official data
- Snooker community for insights and feedback
- Open source ML libraries that make this project possible

## Support

For questions, issues, or feature requests:

1. Check the [Issues](../../issues) page
2. Create a new issue with detailed description
3. For urgent matters, contact the maintainers

---

**Disclaimer**: This system is for educational and research purposes. Please gamble responsibly and within your means. Past performance does not guarantee future results.