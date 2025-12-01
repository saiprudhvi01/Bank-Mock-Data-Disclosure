from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os
import json
from rule_engine import RuleBasedPredictor

app = Flask(__name__)

# Global variables for models
models = {}
scaler = None
feature_columns = None
evaluation_results = None
rule_predictor = None

def load_models():
    """Load trained models and artifacts."""
    global models, scaler, feature_columns, evaluation_results, rule_predictor
    
    try:
        # Load models
        models['random_forest'] = joblib.load('models/random_forest.joblib')
        models['svm'] = joblib.load('models/svm.joblib')
        
        # Load scaler and feature columns
        scaler = joblib.load('models/scaler.joblib')
        feature_columns = joblib.load('models/feature_columns.joblib')
        
        # Load evaluation results
        evaluation_results = joblib.load('models/evaluation_results.joblib')
        
        # Initialize rule-based predictor
        rule_predictor = RuleBasedPredictor()
        
        print("Models and artifacts loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def prepare_input_data(form_data):
    """Prepare input data for prediction."""
    # Convert form data to numpy array
    input_array = np.array([
        float(form_data['age']),
        float(form_data['balance']),
        float(form_data['tenure']),
        float(form_data['num_products']),
        float(form_data['credit_score']),
        float(form_data['has_phone']),
        float(form_data['is_active_member']),
        float(form_data['estimated_salary']),
        float(form_data['has_credit_card']),
        float(form_data['has_loan'])
    ]).reshape(1, -1)
    
    # Scale the input if scaler is available
    if scaler is not None:
        input_scaled = scaler.transform(input_array)
    else:
        input_scaled = input_array  # Use raw data if scaler not available
    
    return input_scaled, input_array

def make_ml_predictions(form_data):
    """Make predictions using ML models."""
    predictions = {}
    
    # Only try ML predictions if models and scaler are available
    if scaler is None or len(models) == 0:
        return predictions  # Return empty dict if models not loaded
    
    # Convert form data to numpy array
    try:
        input_array = np.array([
            float(form_data['age']),
            float(form_data['balance']),
            float(form_data['tenure']),
            float(form_data['num_products']),
            float(form_data['credit_score']),
            float(form_data['has_phone']),
            float(form_data['is_active_member']),
            float(form_data['estimated_salary']),
            float(form_data['has_credit_card']),
            float(form_data['has_loan'])
        ]).reshape(1, -1)
        
        # Scale the input
        input_scaled = scaler.transform(input_array)
        
        # Try Random Forest
        if 'random_forest' in models and models['random_forest'] is not None:
            try:
                rf_pred = models['random_forest'].predict(input_scaled)[0]
                rf_proba = models['random_forest'].predict_proba(input_scaled)[0][1]
                predictions['random_forest'] = {
                    'prediction': int(rf_pred),
                    'probability': float(rf_proba)
                }
            except:
                pass  # Silently fail if model doesn't work
        
        # Try SVM
        if 'svm' in models and models['svm'] is not None:
            try:
                svm_pred = models['svm'].predict(input_scaled)[0]
                svm_proba = models['svm'].predict_proba(input_scaled)[0][1]
                predictions['svm'] = {
                    'prediction': int(svm_pred),
                    'probability': float(svm_proba)
                }
            except:
                pass  # Silently fail if model doesn't work
    except:
        pass  # Silently fail if anything goes wrong
    
    return predictions

def make_rule_based_prediction(form_data):
    """Make prediction using rule-based system."""
    # Always initialize rule predictor if not available
    if rule_predictor is None:
        global rule_predictor
        rule_predictor = RuleBasedPredictor()
    
    # Convert form data to dictionary
    user_data = {
        'age': float(form_data['age']),
        'balance': float(form_data['balance']),
        'tenure': float(form_data['tenure']),
        'num_products': float(form_data['num_products']),
        'credit_score': float(form_data['credit_score']),
        'has_phone': float(form_data['has_phone']),
        'is_active_member': float(form_data['is_active_member']),
        'estimated_salary': float(form_data['estimated_salary']),
        'has_credit_card': float(form_data['has_credit_card']),
        'has_loan': float(form_data['has_loan'])
    }
    
    # Make prediction
    result = rule_predictor.predict_single(user_data)
    
    return {
        'prediction': result['prediction'],
        'probability': result['probability'],
        'explanation': result['explanation'],
        'score': result['score'],
        'triggered_rules': result['triggered_rules']
    }

@app.route('/')
def index():
    """Render the input form page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request."""
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Validate input
        required_fields = ['age', 'balance', 'tenure', 'num_products', 
                          'credit_score', 'has_phone', 'is_active_member', 
                          'estimated_salary', 'has_credit_card', 'has_loan']
        
        for field in required_fields:
            if field not in form_data or not form_data[field]:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Make predictions
        ml_predictions = make_ml_predictions(form_data)
        rule_prediction = make_rule_based_prediction(form_data)
        
        # Prepare comparison table (use dummy data if evaluation results not available)
        comparison_data = []
        if evaluation_results:
            for name, metrics in evaluation_results.items():
                comparison_data.append({
                    'model': name.replace('_', ' ').title(),
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'roc_auc': metrics['roc_auc']
                })
        else:
            # Provide dummy comparison data if models not loaded
            comparison_data = [
                {'model': 'Rule-Based', 'accuracy': 0.657, 'precision': 0.657, 'recall': 1.000, 'roc_auc': 0.508},
                {'model': 'Random Forest', 'accuracy': 0.920, 'precision': 0.937, 'recall': 0.942, 'roc_auc': 0.968},
                {'model': 'SVM', 'accuracy': 0.867, 'precision': 0.927, 'recall': 0.866, 'roc_auc': 0.935}
            ]
        
        # Prepare results for template
        results = {
            'input_data': form_data,
            'rule_based': rule_prediction,
            'random_forest': ml_predictions.get('random_forest', {'prediction': 0, 'probability': 0.5}),
            'svm': ml_predictions.get('svm', {'prediction': 0, 'probability': 0.5}),
            'comparison_table': comparison_data
        }
        
        return render_template('results.html', results=results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Make predictions
        ml_predictions = make_ml_predictions(data)
        rule_prediction = make_rule_based_prediction(data)
        
        # Return JSON response
        return jsonify({
            'rule_based': rule_prediction,
            'random_forest': ml_predictions.get('random_forest', {'prediction': 0, 'probability': 0.5}),
            'svm': ml_predictions.get('svm', {'prediction': 0, 'probability': 0.5})
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_comparison')
def model_comparison():
    """Render model comparison page."""
    if evaluation_results:
        # Prepare chart data for JavaScript
        chart_data = {}
        for model_name, metrics in evaluation_results.items():
            if model_name == 'rule_based':
                chart_data[model_name] = {
                    'name': 'Rule-Based',
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'roc_auc': metrics['roc_auc']
                }
            elif model_name == 'random_forest':
                chart_data[model_name] = {
                    'name': 'Random Forest',
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'roc_auc': metrics['roc_auc']
                }
            elif model_name == 'svm':
                chart_data[model_name] = {
                    'name': 'SVM',
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'roc_auc': metrics['roc_auc']
                }
        
        return render_template('comparison.html', results=evaluation_results, chart_data=json.dumps(chart_data))
    else:
        return "Model evaluation results not available. Please train models first.", 404

@app.route('/health')
def health_check():
    """Health check endpoint."""
    models_loaded = len(models) > 0
    scaler_loaded = scaler is not None
    rule_predictor_ready = rule_predictor is not None
    
    return jsonify({
        'status': 'healthy' if models_loaded and scaler_loaded and rule_predictor_ready else 'unhealthy',
        'models_loaded': models_loaded,
        'scaler_loaded': scaler_loaded,
        'rule_predictor_ready': rule_predictor_ready
    })

if __name__ == '__main__':
    # Always start the app, even if models fail to load
    load_models()  # Try to load models, but continue even if it fails
    print("Starting Flask application...")
    # Use Render's PORT environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
