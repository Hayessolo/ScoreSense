import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
import warnings
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class ScoreSenseEnsemble:
    """An ensemble learning model for high-stakes predictions with statistical validation.

    This ensemble combines multiple machine learning models including Logistic Regression
    and Decision Trees to create robust predictions. It includes built-in statistical
    validation and performance analysis capabilities.

    Attributes:
        model_lr (LogisticRegression): Logistic regression model with balanced class weights
        model_dt (DecisionTreeClassifier): Decision tree model optimized for interpretability
        model_voting (VotingClassifier): Ensemble model combining LR and DT with soft voting
        models_to_evaluate (dict): Dictionary of individual models for evaluation

    Example:
        >>> model = ScoreSenseEnsemble()
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> probabilities = model.predict_proba(X_test)
    """
    def __init__(self):
        self.model_lr = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
        self.model_dt = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')
        self.model_voting = VotingClassifier(
            estimators=[('lr', self.model_lr), ('dt', self.model_dt)],
            voting='soft'
        )
        self.models_to_evaluate = {
            "Logistic Regression": self.model_lr,
            "Decision Tree": self.model_dt,
            "Voting Classifier (Soft)": self.model_voting
        }
        self.final_model = None
        self.final_scaler = None
        self.feature_names = ['midpoint', 'prediction', 'midpointToPredictionRatio']

    def fit(self, X, y):
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty dataset provided")
            
        if len(np.unique(y)) < 2:
            raise ValueError("Need at least two classes in the dataset")

        # Evaluate models
        evaluation_results = self._evaluate_models(X, y)
        if not evaluation_results:
            raise ValueError("Model evaluation failed")

        # Select best model
        best_model_name = self._select_best_model(evaluation_results)
        if not best_model_name:
            raise ValueError("Could not determine best model")

        # Train final model
        self._train_final_model(X, y, best_model_name)
        return self

    def _evaluate_models(self, X, y, n_splits=5):
        n_splits = min(n_splits, X.shape[0])
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0)
        }

        results = {}
        for name, model in self.models_to_evaluate.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])

            try:
                cv_results = cross_validate(pipeline, X, y, cv=kf, scoring=scoring, n_jobs=-1)
                results[name] = {
                    'Avg Fit Time (s)': np.mean(cv_results['fit_time']),
                    'Avg Score Time (s)': np.mean(cv_results['score_time']),
                    'Avg Accuracy': np.mean(cv_results['test_accuracy']),
                    'Avg Precision': np.mean(cv_results['test_precision']),
                    'Avg Recall': np.mean(cv_results['test_recall']),
                    'Avg F1-score': np.mean(cv_results['test_f1'])
                }
            except Exception as e:
                print(f"Error during cross-validation for {name}: {e}")
                continue

        return results

    def _select_best_model(self, evaluation_results):
        best_model_name = None
        best_f1_score = -1.0

        for name, metrics in evaluation_results.items():
            current_f1 = metrics.get('Avg F1-score', -1.0)
            if not np.isnan(current_f1) and current_f1 > best_f1_score:
                best_f1_score = current_f1
                best_model_name = name

        return best_model_name

    def _train_final_model(self, X, y, best_model_name):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', self.models_to_evaluate[best_model_name])
        ])
        pipeline.fit(X, y)
        self.final_model = pipeline.named_steps['model']
        self.final_scaler = pipeline.named_steps['scaler']

    def predict(self, X):
        if self.final_model is None or self.final_scaler is None:
            raise NotFittedError("Model is not fitted yet. Call 'fit' first.")
        
        X_scaled = self.final_scaler.transform(X)
        return self.final_model.predict(X_scaled)

    def predict_proba(self, X):
        if self.final_model is None or self.final_scaler is None:
            raise NotFittedError("Model is not fitted yet. Call 'fit' first.")
            
        X_scaled = self.final_scaler.transform(X)
        return self.final_model.predict_proba(X_scaled)

    def predict_success_probability(self, midpoint, target_prediction):
        """
        Predicts the probability and binary outcome for a given midpoint and target.
        
        Args:
            midpoint (float): The midpoint value
            target_prediction (float): The target value to achieve
            
        Returns:
            dict: Contains probability_exceeds_target and binary_prediction
        """
        if not isinstance(midpoint, (int, float)) or not isinstance(target_prediction, (int, float)):
            raise ValueError("Midpoint and target_prediction must be numeric")

        try:
            ratio = target_prediction / midpoint if midpoint != 0 else 0
            input_features = np.array([[midpoint, target_prediction, ratio]])
            
            X_scaled = self.final_scaler.transform(input_features)
            probability = self.final_model.predict_proba(X_scaled)[0, 1]
            binary_prediction = self.final_model.predict(X_scaled)[0]
            
            return {
                'probability_exceeds_target': probability,
                'binary_prediction': binary_prediction,
                'model_name': self.final_model.__class__.__name__
            }
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def save_model(self, filepath):
        """Save the model to disk"""
        if self.final_model is None or self.final_scaler is None:
            raise NotFittedError("Model is not fitted yet. Call 'fit' first.")
        
        model_data = {
            'model': self.final_model,
            'scaler': self.final_scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)