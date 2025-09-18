# 🎱 Snooker Prediction AI

Professional snooker match prediction system using machine learning and ELO ratings. Adapted from the tennis prediction system with snooker-specific features and optimizations.

## ✨ Features

- **🤖 Machine Learning Models**: LightGBM, XGBoost, and Random Forest for accurate predictions
- **⚖️ Snooker ELO System**: Tournament-specific ratings (World Championship, Masters, UK Championship, etc.)
- **🎯 Enhanced Name Matching**: Fuzzy matching with apostrophe and accent handling
- **📊 Snooker-Specific Features**: Break building, pot success, safety play, frame control
- **🏆 Tournament Analysis**: Different weightings for tournament prestige and importance
- **📈 Player Statistics**: Comprehensive tracking of centuries, breaks, and performance metrics

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn lightgbm xgboost matplotlib joblib
```

### Installation

1. Clone or download the snooker prediction system
2. Navigate to the project directory:
```bash
cd snooker-prediction-ai
```

### Usage

#### Interactive Prediction
```bash
python predict_snooker_match.py
```

#### Direct Prediction
```bash
python predict_snooker_match.py "Ronnie O'Sullivan" "Judd Trump"
```

#### With Tournament and Format Options
```bash
python predict_snooker_match.py "Mark Selby" "Neil Robertson" --tournament world_championship --best-of 35
```

#### See Example Rivalries
```bash
python predict_snooker_match.py --examples
```

## 🎯 Tournament Types

- **World Championship** - Highest prestige (weight: 50)
- **Masters** - Major tournament (weight: 35)
- **UK Championship** - Major tournament (weight: 35)
- **Champion of Champions** - Elite tournament (weight: 30)
- **Players Championship** - Top 16 tournament (weight: 25)
- **Tour Championship** - Season finale (weight: 25)
- **Ranking Event** - Standard ranking tournaments (weight: 20)
- **Invitational** - Non-ranking events (weight: 15)

## 📏 Match Formats

- **Best of 7** (first to 4) - Short format
- **Best of 9** (first to 5) - Standard format
- **Best of 11** (first to 6) - Extended format
- **Best of 17** (first to 9) - Semi-final format
- **Best of 19** (first to 10) - Quarter-final format
- **Best of 35** (first to 18) - World Championship Final

## 🎱 Featured Players

The system includes comprehensive data for professional players including:

**Current Top Players:**
- Ronnie O'Sullivan
- Judd Trump
- Mark Selby
- Neil Robertson
- John Higgins
- Mark Williams
- Kyren Wilson
- Jack Lisowski
- Shaun Murphy
- Stuart Bingham

**Legends & Veterans:**
- Stephen Hendry
- Jimmy White
- Ken Doherty
- Graeme Dott
- Peter Ebdon

## 🤖 Training the Model

Generate training data and train the prediction model:

```bash
python train_snooker_model.py
```

This will:
1. Generate comprehensive snooker dataset (25,000 matches)
2. Create snooker-specific features
3. Train multiple ML models
4. Save the best performing model
5. Analyze feature importance

## 📊 Snooker-Specific Features

### Break Building
- **Centuries**: 100+ breaks made
- **Breaks 50+**: All significant breaks
- **Highest Break**: Personal best break
- **Break Difference**: Comparison between players

### Pot Success
- **Overall Pot Success**: General potting accuracy (%)
- **Long Pot Success**: Long-range potting accuracy (%)
- **Pot Success Difference**: Advantage calculation

### Safety Play
- **Safety Success**: Defensive shot success rate (%)
- **Safety Advantage**: Tactical superiority metric

### Frame Control
- **Average Frame Time**: Speed of play (minutes)
- **First Visit Clearance**: Ability to clear from break-off (%)
- **Frame Control Difference**: Tempo advantage

### Tournament Performance
- **ELO Ratings**: Tournament-specific ratings
- **Tournament Weight**: Event importance scaling
- **Prize Money**: Financial incentive factor
- **Prestige Score**: Tournament status ranking

## 📈 Example Predictions

```bash
# Classic rivalry
python predict_snooker_match.py "Ronnie O'Sullivan" "Judd Trump"

# Tactical vs attacking
python predict_snooker_match.py "Mark Selby" "Neil Robertson" --tournament masters

# World Championship final
python predict_snooker_match.py "John Higgins" "Mark Williams" --tournament world_championship --best-of 35
```

## 🎯 Name Matching Features

The system includes advanced name matching:

- **Case Insensitive**: Works with any capitalization
- **Fuzzy Matching**: Handles typos and variations
- **Apostrophe Handling**: O'Sullivan, O'Connor, etc.
- **Accent Support**: Handles international characters
- **Smart Suggestions**: Provides closest matches when exact match fails

Example successful matches:
- "ronnie osullivan" → Ronnie O'Sullivan
- "JUDD TRUMP" → Judd Trump
- "mark selby" → Mark Selby

## 📁 Project Structure

```
snooker-prediction-ai/
├── src/
│   ├── snooker_predictor.py          # Main prediction engine
│   ├── snooker_data_collector.py     # Dataset generation
│   ├── snooker_elo_system.py         # ELO rating system
│   └── __init__.py
├── models/                           # Trained ML models
├── data/                             # Generated datasets
├── predict_snooker_match.py          # Interactive interface
├── train_snooker_model.py            # Model training script
└── README.md                         # This file
```

## 🔧 Advanced Configuration

### Custom Tournament Types
Modify `snooker_elo_system.py` to add new tournament types with custom weights.

### Feature Engineering
Extend `snooker_data_collector.py` to add new snooker-specific features like:
- Break-off success rates
- Crowd performance factors
- Head-to-head frame patterns
- Tournament venue performance

### Model Tuning
Adjust hyperparameters in `train_snooker_model.py` for optimal performance.

## 📊 Model Performance

The system typically achieves:
- **Accuracy**: 65-75% on test data
- **Cross-validation**: 5-fold CV for robust evaluation
- **Feature Importance**: ELO difference, break building, and tournament type are top predictors

## 🎱 How It Works

1. **Data Generation**: Creates realistic snooker match scenarios with professional player statistics
2. **ELO Calculation**: Updates tournament-specific ratings based on match results and importance
3. **Feature Extraction**: Converts matches into ML-ready format with snooker-specific metrics
4. **Model Training**: Tests multiple algorithms and selects the best performer
5. **Prediction**: Combines ML model with ELO ratings for comprehensive match prediction

## 🏆 Famous Rivalries to Try

```bash
# The two modern greats
python predict_snooker_match.py "Ronnie O'Sullivan" "Judd Trump"

# Tactical vs attacking styles
python predict_snooker_match.py "Mark Selby" "Neil Robertson"

# Scottish vs Welsh rivalry
python predict_snooker_match.py "John Higgins" "Mark Williams"

# Rising stars battle
python predict_snooker_match.py "Kyren Wilson" "Jack Lisowski"

# Experienced campaigners
python predict_snooker_match.py "Shaun Murphy" "Stuart Bingham"
```

## 📝 Sample Output

```
🎱 SNOOKER MATCH PREDICTOR
Professional snooker prediction system
==================================================

🔮 Predicting Ronnie O'Sullivan vs Judd Trump...
   Tournament: World Championship
   Format: Best of 35 (first to 18)

==================================================
🎯 PREDICTION COMPLETE!
==================================================

🏆 WINNER: Ronnie O'Sullivan
📊 Confidence: 64.2%

📈 DETAILED PROBABILITIES:
   Ronnie O'Sullivan: 64.2%
   Judd Trump: 35.8%

🎱 MATCH DETAILS:
   Tournament: World Championship
   Format: Best of 35 (first to 18)

⚖️  ELO RATINGS:
   Ronnie O'Sullivan: 1847
   Judd Trump: 1723
   Difference: 124 points

🤖 MODEL vs ELO: ✅ Agree
   ML Model favors: Ronnie O'Sullivan
   ELO favors: Ronnie O'Sullivan

👍 Moderate prediction - Slight edge to winner
```

## 🤝 Contributing

This snooker prediction system is adapted from the tennis prediction AI. To contribute:

1. Add new snooker-specific features
2. Improve player name matching
3. Enhance tournament modeling
4. Add historical match data integration

## 📄 License

Open source project - feel free to use and modify for snooker prediction analysis.

---

**🎱 Ready to predict snooker matches with professional-grade AI!**