import numpy as np
import pytest
from scoresense.utils import (
    create_features,
    calculate_metrics,
    calculate_metrics_summary,
    bootstrap_confidence_interval,
    analyze_performance_by_range,
    calculate_performance_based_weights
)

@pytest.fixture
def sample_data():
    return [
        {'prediction': 58, 'midpoint': 35, 'final': 66},
        {'prediction': 53, 'midpoint': 31, 'final': 54},
        {'prediction': 47, 'midpoint': 24, 'final': 51},
        {'prediction': 48, 'midpoint': 25, 'final': 57}
    ]

@pytest.fixture
def processed_data(sample_data):
    return create_features(sample_data)

def test_create_features_valid_data(sample_data):
    result = create_features(sample_data)
    assert len(result) == len(sample_data)
    for row in result:
        assert 'success' in row
        assert 'growthRatio' in row
        assert 'midpointToPredictionRatio' in row
        assert isinstance(row['success'], int)
        assert row['success'] in [0, 1]

def test_create_features_invalid_input():
    assert create_features(None) == []
    assert create_features([]) == []
    assert create_features([{'invalid': 'data'}]) == []

def test_create_features_missing_keys():
    data = [{'prediction': 58, 'midpoint': 35}]  # missing 'final'
    result = create_features(data)
    assert result == []

def test_create_features_division_by_zero():
    data = [{'prediction': 58, 'midpoint': 0, 'final': 66}]
    result = create_features(data)
    assert len(result) == 1
    assert result[0]['growthRatio'] == 0
    assert result[0]['midpointToPredictionRatio'] == 0

def test_calculate_metrics():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])
    metrics = calculate_metrics(y_true, y_pred)
    
    assert set(metrics.keys()) == {'accuracy', 'precision', 'recall', 'f1'}
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1

def test_calculate_metrics_summary():
    results = [
        {'accuracy': 0.8, 'precision': 0.75, 'recall': 0.8, 'f1': 0.77},
        {'accuracy': 0.9, 'precision': 0.85, 'recall': 0.9, 'f1': 0.87}
    ]
    summary = calculate_metrics_summary(results)
    
    assert set(summary.keys()) == {'accuracy', 'precision', 'recall', 'f1'}
    for metric in summary.values():
        assert set(metric.keys()) == {'mean', 'std'}
        assert 0 <= metric['mean'] <= 1
        assert metric['std'] >= 0

def test_bootstrap_confidence_interval():
    predictions = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    actuals = np.array([1, 0, 1, 0, 0, 1, 1, 1])
    
    lower, upper = bootstrap_confidence_interval(predictions, actuals, n_bootstraps=100)
    assert lower <= upper
    assert 0 <= lower <= 1
    assert 0 <= upper <= 1

def test_bootstrap_confidence_interval_invalid_input():
    with pytest.raises(ValueError):
        bootstrap_confidence_interval(
            np.array([1, 0]), 
            np.array([1, 0, 1]),
            n_bootstraps=100
        )

def test_analyze_performance_by_range():
    predictions = np.array([1, 0, 1, 0])
    actuals = np.array([1, 0, 0, 0])
    features = np.array([[15, 1], [25, 1], [30, 1], [35, 1]])
    ranges = [(0, 20), (20, 30), (30, 40)]
    
    results = analyze_performance_by_range(predictions, actuals, features, ranges)
    assert isinstance(results, dict)
    
    for range_metrics in results.values():
        assert set(range_metrics.keys()) == {'accuracy', 'precision', 'recall', 'f1', 'sample_size'}
        assert range_metrics['sample_size'] > 0

def test_calculate_performance_based_weights():
    performances = [0.8, 0.9, 0.7]
    weights = calculate_performance_based_weights(performances)
    
    assert len(weights) == len(performances)
    assert np.isclose(sum(weights), 1)
    assert all(w >= 0 for w in weights)
    # Check that better performing models get higher weights
    performance_weight_pairs = list(zip(performances, weights))
    sorted_by_performance = sorted(performance_weight_pairs, key=lambda x: x[0], reverse=True)
    sorted_by_weight = sorted(performance_weight_pairs, key=lambda x: x[1], reverse=True)
    assert sorted_by_performance == sorted_by_weight

def test_calculate_performance_based_weights_edge_cases():
    assert calculate_performance_based_weights([]) == []
    weights = calculate_performance_based_weights([1.0])
    assert len(weights) == 1
    assert weights[0] == 1.0

def test_calculate_performance_based_weights_temperature():
    performances = [0.8, 0.9, 0.7]
    weights_t1 = calculate_performance_based_weights(performances, temperature=1.0)
    weights_t2 = calculate_performance_based_weights(performances, temperature=2.0)
    
    # Higher temperature should lead to more uniform weights
    assert max(weights_t2) - min(weights_t2) < max(weights_t1) - min(weights_t1)