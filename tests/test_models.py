import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from scoresense.models import ScoreSenseEnsemble

@pytest.fixture
def sample_data():
    X = np.array([
        [35, 58, 1.66],
        [31, 53, 1.71],
        [24, 47, 1.96],
        [25, 48, 1.92],
        [29, 55, 1.90],
        [24, 48, 2.00],
        [28, 52, 1.86],
        [32, 57, 1.78],
    ])
    y = np.array([1, 1, 1, 1, 0, 0, 1, 0])
    return X, y

def test_model_initialization():
    model = ScoreSenseEnsemble()
    assert model.final_model is None
    assert model.final_scaler is None
    assert isinstance(model.feature_names, list)
    assert len(model.models_to_evaluate) == 3

def test_model_fit(sample_data):
    X, y = sample_data
    model = ScoreSenseEnsemble()
    model.fit(X, y)
    assert model.final_model is not None
    assert model.final_scaler is not None

def test_model_predict(sample_data):
    X, y = sample_data
    model = ScoreSenseEnsemble()
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert all(isinstance(p, (int, np.integer)) for p in predictions)
    assert all(p in [0, 1] for p in predictions)

def test_model_predict_proba(sample_data):
    X, y = sample_data
    model = ScoreSenseEnsemble()
    model.fit(X, y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(y), 2)
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    assert np.allclose(np.sum(probabilities, axis=1), 1)

def test_predict_success_probability(sample_data):
    X, y = sample_data
    model = ScoreSenseEnsemble()
    model.fit(X, y)
    
    result = model.predict_success_probability(30, 55)
    assert isinstance(result, dict)
    assert 'probability_exceeds_target' in result
    assert 'binary_prediction' in result
    assert 'model_name' in result
    assert 0 <= result['probability_exceeds_target'] <= 1
    assert result['binary_prediction'] in [0, 1]

def test_empty_data():
    model = ScoreSenseEnsemble()
    with pytest.raises(ValueError, match="Empty dataset provided"):
        model.fit(np.array([]), np.array([]))

def test_single_class_data():
    X = np.array([[1, 2, 0.5], [2, 3, 0.6]])
    y = np.array([1, 1])  # Only one class
    model = ScoreSenseEnsemble()
    with pytest.raises(ValueError, match="Need at least two classes"):
        model.fit(X, y)

def test_predict_without_fit():
    model = ScoreSenseEnsemble()
    X = np.array([[1, 2, 0.5]])
    with pytest.raises(NotFittedError):
        model.predict(X)

def test_predict_proba_without_fit():
    model = ScoreSenseEnsemble()
    X = np.array([[1, 2, 0.5]])
    with pytest.raises(NotFittedError):
        model.predict_proba(X)

def test_save_model_without_fit(tmp_path):
    model = ScoreSenseEnsemble()
    with pytest.raises(NotFittedError):
        model.save_model(tmp_path / "model.joblib")

def test_invalid_probability_inputs(sample_data):
    X, y = sample_data  # Use the fixture that already has both classes
    model = ScoreSenseEnsemble()
    model.fit(X, y)
    
    with pytest.raises(ValueError):
        model.predict_success_probability("invalid", 55)
    with pytest.raises(ValueError):
        model.predict_success_probability(30, "invalid")