# ScoreSense

A production-grade prediction system that implements advanced ensemble learning to determine if final values will meet or exceed predicted targets. Designed for high-stakes predictions where accuracy and statistical validation are critical.

## Core Functionality

- High-precision ensemble learning combining 6 machine learning techniques
- Rigorous statistical validation with bootstrap confidence intervals
- Performance analysis across different score ranges
- Automated model weight optimization
- Comprehensive metric evaluation

## Key Components

### Machine Learning Stack
- Regularized Logistic Regression (L1/L2)
- Decision Trees (optimized for interpretability)
- Custom Linear Predictor
- Stacked Ensemble with Cross-validation
- Performance-based Model Weighting
- Automated Feature Scaling

### Advanced Features
- Bootstrap confidence intervals for reliability metrics
- Range-based performance analysis
- Stratified k-fold cross validation
- Comprehensive metric suite (accuracy, precision, recall, F1)
- Built-in feature engineering

## When to Use ScoreSense

- High-Stakes Predictions: When accuracy matters more than speed
- Performance Validation: When statistical confidence is required
- Dynamic Environments: When data patterns might change over time
- Model Comparison: When choosing between different approaches

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from scoresense.models import ScoreSenseEnsemble
import numpy as np

# Prepare your prediction data
X = np.array([
    [midpoint_score, target_score, midpoint_to_target_ratio],
    # ... more samples ...
])
y = np.array([1 if final >= target else 0 for final, target in zip(finals, targets)])

# Initialize and train the ensemble
model = ScoreSenseEnsemble()
model.fit(X, y)

# Get predictions and probabilities
predictions = model.predict(X)
probabilities = model.predict_proba(X)
```

See `examples/example.py` for a complete implementation example.

## Project Structure

- `scoresense/`
  - `models.py`: Core ensemble implementation with advanced ML techniques
  - `utils.py`: Statistical analysis and validation utilities
- `examples/`: Production-ready implementation examples
- `tests/`: Comprehensive test suite for model validation

## Technical Details

### Base Models
1. Logistic Regression
   - L1/L2 regularization support
   - Automated hyperparameter optimization

2. Decision Tree
   - Optimized depth for interpretability
   - Built-in cross-validation

3. Custom Linear Formula
   - Domain-specific optimization
   - Enhanced for outlier handling

### Meta-Model
- Stacked ensemble with cross-validation
- Automated feature scaling
- Performance-based model weighting

## Performance Metrics

- Overall accuracy, precision, recall, and F1 scores
- Range-based performance analysis
- Bootstrap confidence intervals
- Cross-validation metrics

## License

Apache License 2.0

## Prediction Features

### Success Probability Prediction
ScoreSense now includes functionality to predict the probability of success given a midpoint value and a target prediction. This is useful for:
- Assessing the likelihood of reaching specific targets
- Evaluating different target scenarios
- Making data-driven decisions about goal setting

### Usage Examples

```python
from scoresense.models import ScoreSenseEnsemble

# Initialize and train the model
model = ScoreSenseEnsemble()
model.fit(X, y)

# Basic prediction
midpoint = 30
target = 55
probability = model.predict_success_probability(midpoint, target)
print(f"Probability of success: {probability:.2%}")

# Multiple scenario analysis
scenarios = [
    (25, 45),  # Conservative
    (25, 50),  # Moderate
    (25, 55)   # Ambitious
]
for midpoint, target in scenarios:
    prob = model.predict_success_probability(midpoint, target)
    print(f"Midpoint: {midpoint}, Target: {target}")
    print(f"Probability: {prob:.2%}")
```

## Running Examples

To run the example script:
```bash
python examples/example.py
```

The example script demonstrates:
1. Basic model training and evaluation
2. Single prediction examples
3. Multiple scenario analysis
4. Interactive prediction mode where you can input your own values