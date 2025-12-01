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
    if scaler is None:
        raise ValueError("Scaler not loaded. Models may not be trained yet.")
    
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
    
    # Scale the input
    input_scaled = scaler.transform(input_array)
    
    return input_scaled, input_array

def make_ml_predictions(input_scaled):
    """Make predictions using ML models."""
    predictions = {}
    
    for name, model in models.items():
        # Get prediction and probability
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0, 1]
        
        predictions[name] = {
            'prediction': int(pred),
            'probability': float(proba)
        }
    
    return predictions

def make_rule_based_prediction(form_data):
    """Make prediction using rule-based system."""
    if rule_predictor is None:
        raise ValueError("Rule predictor not loaded. Models may not be trained yet.")
    
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
        
        # Prepare input data
        input_scaled, input_original = prepare_input_data(form_data)
        
        # Make predictions
        ml_predictions = make_ml_predictions(input_scaled)
        rule_prediction = make_rule_based_prediction(form_data)
        
        # Prepare comparison table
        comparison_data = []
        for name, metrics in evaluation_results.items():
            comparison_data.append({
                'model': name.replace('_', ' ').title(),
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'roc_auc': metrics['roc_auc']
            })
        
        # Prepare results for template
        results = {
            'input_data': form_data,
            'rule_based': rule_prediction,
            'random_forest': ml_predictions['random_forest'],
            'svm': ml_predictions['svm'],
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
        
        # Prepare input data
        input_scaled, input_original = prepare_input_data(data)
        
        # Make predictions
        ml_predictions = make_ml_predictions(input_scaled)
        rule_prediction = make_rule_based_prediction(data)
        
        # Return JSON response
        return jsonify({
            'rule_based': rule_prediction,
            'random_forest': ml_predictions['random_forest'],
            'svm': ml_predictions['svm']
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
    # Load models before starting the app
    if load_models():
        print("Starting Flask application...")
        # Use Render's PORT environment variable or default to 5000
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("Failed to load models. Please run train.py first to train and save models.")
        exit(1)
