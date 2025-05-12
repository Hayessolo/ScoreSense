import numpy as np
import pandas as pd
from scoresense.models import ScoreSenseEnsemble
from scoresense.utils import create_features, analyze_performance_by_range

def load_and_prepare_data(file_path):
    """Load and prepare the dataset with error handling"""
    try:
        df = pd.read_csv(file_path)
        required_columns = ['Final Value >=', 'Midpoint Value', 'Actual Final Value']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        # Convert DataFrame to list of dictionaries
        data = []
        for _, row in df.iterrows():
            data.append({
                'prediction': row['Final Value >='],
                'midpoint': row['Midpoint Value'],
                'final': row['Actual Final Value']
            })
        return data
    except FileNotFoundError:
        print(f"Error: Could not find file {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def prepare_features(data):
    """Prepare features with error handling"""
    try:
        # Apply feature engineering
        processed_data = create_features(data)
        if not processed_data:
            raise ValueError("Feature engineering produced no valid data")

        # Prepare training data
        X = np.array([[row['midpoint'], row['prediction'], row['midpointToPredictionRatio']] 
                     for row in processed_data])
        y = np.array([row['success'] for row in processed_data])
        return X, y
    except Exception as e:
        print(f"Error preparing features: {str(e)}")
        return None, None

def train_and_evaluate_model(X, y):
    """Train and evaluate the model with error handling"""
    try:
        # Initialize and train the ensemble
        model = ScoreSenseEnsemble()
        model.fit(X, y)
        return model
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return None

def analyze_scenarios(model, scenarios):
    """Analyze different prediction scenarios"""
    print("\nAnalyzing Different Scenarios:")
    for midpoint, target in scenarios:
        try:
            result = model.predict_success_probability(midpoint, target)
            print(f"\nScenario: Midpoint={midpoint}, Target={target}")
            print(f"Probability of Success: {result['probability_exceeds_target']:.2%}")
            print(f"Binary Prediction: {'Success' if result['binary_prediction'] else 'Not Successful'}")
            print(f"Model Used: {result['model_name']}")
            
            # Calculate required improvement
            improvement_needed = target - midpoint
            improvement_percentage = (improvement_needed / midpoint) * 100 if midpoint != 0 else 0
            print(f"Required Improvement: {improvement_needed:.1f} points ({improvement_percentage:.1f}%)")
            
            # Provide interpretation
            if result['probability_exceeds_target'] >= 0.75:
                print("Interpretation: High probability of success")
            elif result['probability_exceeds_target'] >= 0.5:
                print("Interpretation: Moderate probability of success")
            else:
                print("Interpretation: Lower probability of success")
        except Exception as e:
            print(f"Error analyzing scenario: {str(e)}")

def analyze_model_usage(model, scenarios):
    """Track which models are being used most frequently"""
    model_usage = {}
    for midpoint, target in scenarios:
        result = model.predict_success_probability(midpoint, target)
        model_name = result['model_name']
        model_usage[model_name] = model_usage.get(model_name, 0) + 1
    
    print("\nModel Usage Statistics:")
    for model_name, count in model_usage.items():
        print(f"{model_name}: {count} times")

def interactive_predictions(model):
    """Interactive prediction mode with error handling"""
    print("\nInteractive Prediction Mode:")
    print("Enter -1 for midpoint to exit")
    
    while True:
        try:
            midpoint = float(input("\nEnter midpoint value (-1 to exit): "))
            if midpoint == -1:
                break
                
            target = float(input("Enter target value: "))
            
            result = model.predict_success_probability(midpoint, target)
            print(f"\nResults:")
            print(f"Probability of Success: {result['probability_exceeds_target']:.2%}")
            print(f"Prediction: {'Success' if result['binary_prediction'] else 'Not Successful'}")
            print(f"Model Used: {result['model_name']}")
            
            # Additional analysis
            improvement_needed = target - midpoint
            improvement_percentage = (improvement_needed / midpoint) * 100 if midpoint != 0 else 0
            print(f"\nAnalysis:")
            print(f"Required Improvement: {improvement_needed:.1f} points ({improvement_percentage:.1f}%)")
            
            # Interpretation
            print("\nInterpretation:")
            if result['probability_exceeds_target'] >= 0.75:
                print("High probability of success - Target appears achievable")
            elif result['probability_exceeds_target'] >= 0.5:
                print("Moderate probability of success - Consider additional support")
            else:
                print("Lower probability of success - Consider adjusting target or implementing support measures")
                
        except ValueError:
            print("Invalid input. Please enter numeric values.")
        except Exception as e:
            print(f"Error during prediction: {str(e)}")

def main():
    # Load and prepare data
    data = load_and_prepare_data('examples/sample_data.csv')
    if not data:
        print("Failed to load data. Exiting.")
        return

    # Prepare features
    X, y = prepare_features(data)
    if X is None or y is None:
        print("Failed to prepare features. Exiting.")
        return

    # Train model
    model = train_and_evaluate_model(X, y)
    if not model:
        print("Failed to train model. Exiting.")
        return

    # Save the trained model (optional)
    try:
        model.save_model('final_model.joblib')
        print("\nModel saved successfully to 'final_model.joblib'")
    except Exception as e:
        print(f"Warning: Could not save model: {str(e)}")

    # Analyze different scenarios
    scenarios = [
        (25, 45),  # Conservative
        (25, 50),  # Moderate
        (25, 55),  # Ambitious
        (30, 60),  # Very ambitious
    ]
    analyze_scenarios(model, scenarios)

    # Analyze model usage
    analyze_model_usage(model, scenarios)

    # Start interactive prediction mode
    interactive_predictions(model)

if __name__ == "__main__":
    main()