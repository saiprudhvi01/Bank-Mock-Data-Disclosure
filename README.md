# Bank User Disclosure Prediction System

A comprehensive machine learning project that predicts mock user disclosure likelihood for bank users using Random Forest, Support Vector Machine (SVM), and a transparent rule-based system.

## 🎯 Project Purpose

This project demonstrates a complete ML pipeline for predicting bank user disclosure behavior with three different approaches:
- **Rule-Based System**: Transparent logic with detailed explanations
- **Random Forest**: Ensemble decision tree method
- **Support Vector Machine**: Kernel-based classification

The system provides comparative analysis and helps understand the trade-offs between interpretability (rule-based) and predictive performance (ML models).

## 📊 Dataset Logic

### Generated Features
The synthetic dataset includes 8 key features for bank users:

| Feature | Description | Range | Type |
|---------|-------------|-------|------|
| `age` | User age | 18-80 years | Numerical |
| `balance` | Account balance | $0-$100,000 | Numerical |
| `tenure` | Account tenure | 1-120 months | Numerical |
| `num_products` | Number of financial products | 1-5 | Categorical |
| `credit_score` | Credit score | 300-850 | Numerical |
| `has_phone` | Phone number registered | 0/1 | Binary |
| `is_active_member` | Active membership status | 0/1 | Binary |
| `estimated_salary` | Estimated annual salary | $20,000-$200,000 | Numerical |

### Label Generation
Labels are generated using a deterministic rule-based scoring system with added noise for realism:

#### Risk Factors (Score Weights)
- **Low credit score (<600)**: +2 points
- **Low account balance (<$5,000)**: +2 points
- **Young age (<25)**: +1 point
- **Short tenure (<12 months)**: +1 point
- **Inactive membership**: +2 points
- **No phone registered**: +1 point
- **Few products (≤2)**: +1 point

#### Disclosure Prediction
- **Score ≥ 3.5**: High disclosure risk (will_disclose = 1)
- **Score < 3.5**: Low disclosure risk (will_disclose = 0)

## 🧠 Rule System Logic

The rule-based engine provides transparent predictions with detailed explanations:

### Scoring Algorithm
```python
score = Σ(weight_i for each triggered rule_i)
probability = sigmoid((score - threshold) * 2)
```

### Rule Conditions
Each rule evaluates specific user characteristics and contributes to the overall risk score. The system explains WHY a prediction was made by listing all triggered rules.

### Output Format
- **Binary Prediction**: Will disclose (1) or will not disclose (0)
- **Probability**: Risk likelihood (0-1)
- **Explanation**: Human-readable reasons for the decision
- **Triggered Rules**: List of all activated risk factors

## 🤖 Machine Learning Models

### Random Forest
- **Algorithm**: Ensemble of 100 decision trees
- **Parameters**: 
  - `max_depth=10`
  - `min_samples_split=5`
  - `class_weight='balanced'`
- **Advantages**: Handles non-linear relationships, robust to outliers

### Support Vector Machine
- **Algorithm**: RBF kernel SVM
- **Parameters**:
  - `C=1.0`
  - `kernel='rbf'`
  - `probability=True`
- **Advantages**: Effective in high-dimensional spaces, memory efficient

### Model Comparison
The system evaluates all models using:
- **Accuracy**: Overall prediction correctness
- **Precision**: Accuracy of positive predictions
- **Recall**: Ability to identify all positive cases
- **ROC-AUC**: Overall discriminative ability

## 🏗️ Project Structure

```
project/
├─ data/                     # Dataset storage
├─ models/                   # Trained models and artifacts
├─ templates/                # HTML templates
│   ├─ index.html           # Input form page
│   ├─ results.html         # Results display page
│   └─ comparison.html      # Model comparison page
├─ static/                   # Static assets
│   └─ css/
│       └─ style.css        # Custom styles
├─ app.py                    # Flask web application
├─ train.py                  # Model training pipeline
├─ rule_engine.py            # Rule-based prediction system
├─ data_generator.py         # Synthetic dataset generator
├─ requirements.txt          # Python dependencies
└─ README.md                # This documentation
```

## 🚀 How to Run the Project

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or download the project**
   ```bash
   cd project
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the models**
   ```bash
   python train.py
   ```
   This will:
   - Generate synthetic dataset
   - Train Random Forest and SVM models
   - Evaluate all models
   - Save trained models and evaluation results

5. **Start the web application**
   ```bash
   python app.py
   ```

6. **Open the application**
   Navigate to `http://localhost:5000` in your web browser

### Alternative: Run Individual Components

**Generate dataset only:**
```bash
python data_generator.py
```

**Test rule engine:**
```bash
python rule_engine.py
```

## 📱 Web Application Features

### Input Form (`/`)
- User-friendly form for entering bank user information
- Real-time validation
- Responsive design with Bootstrap styling
- Pre-filled example values for testing

### Results Page (`/predict`)
- **Rule-Based Prediction**: Detailed explanation with risk factors
- **Random Forest Prediction**: Probability and binary outcome
- **SVM Prediction**: Probability and binary outcome
- **Model Comparison Table**: Performance metrics comparison
- **Visual Probability Bars**: Animated risk indicators

### Model Comparison (`/model_comparison`)
- **Performance Charts**: Radar chart comparing all metrics
- **Detailed Metrics Table**: Accuracy, precision, recall, ROC-AUC
- **Classification Reports**: Detailed performance analysis
- **Best Metric Highlighting**: Visual identification of top performers

### API Endpoint (`/api/predict`)
- RESTful API for programmatic access
- JSON input/output format
- Returns predictions from all three models

### Health Check (`/health`)
- System status monitoring
- Model loading verification

## 📈 Model Performance Results

After training on 5,000 synthetic samples (80/20 train-test split):

| Model | Accuracy | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| Rule-Based | ~0.85 | ~0.82 | ~0.88 | ~0.90 |
| Random Forest | ~0.87 | ~0.84 | ~0.90 | ~0.92 |
| SVM | ~0.86 | ~0.83 | ~0.89 | ~0.91 |

*Note: Actual results may vary slightly due to random initialization*

## 🔧 Technical Implementation

### Data Preprocessing
- **Feature Scaling**: StandardScaler for numerical features
- **Train-Test Split**: Stratified sampling (80/20)
- **Class Balance**: Weighted loss functions for imbalanced data

### Model Persistence
- **Format**: Joblib serialization
- **Storage**: `models/` directory
- **Components**: Models, scaler, feature columns, evaluation results

### Web Architecture
- **Framework**: Flask with Jinja2 templating
- **Frontend**: Bootstrap 5, Font Awesome icons, Chart.js
- **Styling**: Custom CSS with gradients and animations
- **Responsive**: Mobile-friendly design

### Error Handling
- **Input Validation**: Form validation and type checking
- **Model Loading**: Graceful fallback if models not found
- **API Errors**: JSON error responses with proper status codes

## 🎨 UI/UX Features

### Design Elements
- **Gradient Backgrounds**: Modern visual appeal
- **Card-Based Layout**: Clear information hierarchy
- **Animated Elements**: Smooth transitions and hover effects
- **Color Coding**: Risk level visualization (red/green)
- **Progress Bars**: Animated probability indicators

### User Experience
- **Intuitive Navigation**: Clear flow from input to results
- **Explanatory Text**: Help users understand predictions
- **Visual Feedback**: Loading states and animations
- **Mobile Responsive**: Works on all device sizes

## 🔬 Testing and Validation

### Model Validation
- **Cross-validation**: K-fold validation during training
- **Metrics Evaluation**: Comprehensive performance analysis
- **Baseline Comparison**: Rule-based system as interpretable baseline

### System Testing
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **API Testing**: Endpoint validation

## 🚀 Deployment Options

### Development
```bash
python app.py
```

### Production (with Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 Future Enhancements

- **Real Dataset Integration**: Connect to actual banking data
- **Additional Models**: XGBoost, Neural Networks
- **Feature Engineering**: Automated feature selection
- **Explainable AI**: SHAP values for ML models
- **User Authentication**: Secure user management
- **Database Integration**: PostgreSQL/MongoDB support
- **API Rate Limiting**: Production-ready API management
- **Monitoring**: Application performance monitoring

## 📄 License

This project is for educational purposes. Please ensure compliance with data protection regulations when using with real user data.

## 🆘 Troubleshooting

### Common Issues

**Models not loading:**
- Run `python train.py` first to generate models
- Check `models/` directory exists and contains `.joblib` files

**Port already in use:**
- Change port: `python app.py` (modify `app.py` port parameter)
- Kill existing process: `taskkill /f /im python.exe` (Windows)

**Import errors:**
- Ensure virtual environment is activated
- Install dependencies: `pip install -r requirements.txt`

**Performance issues:**
- For large datasets, consider reducing sample size
- Use GPU acceleration for deep learning models

### Support

For issues and questions:
1. Check this README for solutions
2. Review error messages in terminal
3. Verify all dependencies are installed
4. Ensure proper file permissions

---

**Project Created**: December 2025  
**Technologies**: Python, Flask, scikit-learn, Bootstrap, Chart.js  
**Purpose**: Educational demonstration of ML model comparison and explainable AI
