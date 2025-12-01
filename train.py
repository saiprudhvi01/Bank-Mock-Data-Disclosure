import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
import joblib
import os
from data_generator import generate_bank_user_dataset
from rule_engine import RuleBasedPredictor

class ModelTrainer:
    """
    Train and evaluate machine learning models for bank user disclosure prediction.
    """
    
    def __init__(self):
        self.feature_columns = [
            'age', 'balance', 'tenure', 'num_products', 
            'credit_score', 'has_phone', 'is_active_member', 
            'estimated_salary', 'has_credit_card', 'has_loan'
        ]
        self.target_column = 'will_disclose'
        self.scaler = StandardScaler()
        self.models = {}
        self.evaluation_results = {}
    
    def prepare_data(self, df=None):
        """
        Prepare data for training.
        
        Args:
            df (pd.DataFrame, optional): Input dataframe. If None, generates new data.
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, scaler)
        """
        if df is None:
            print("Generating synthetic dataset...")
            df = generate_bank_user_dataset(n_samples=5000, random_state=42)
        
        # Separate features and target
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Data prepared successfully:")
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        print(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(self, X_train, y_train):
        """
        Train Random Forest classifier.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            
        Returns:
            RandomForestClassifier: Trained model
        """
        print("Training Random Forest model...")
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        
        print("Random Forest model trained successfully")
        return rf
    
    def train_svm(self, X_train, y_train):
        """
        Train Support Vector Machine classifier.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            
        Returns:
            SVC: Trained model
        """
        print("Training SVM model...")
        
        svm = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        
        svm.fit(X_train, y_train)
        self.models['svm'] = svm
        
        print("SVM model trained successfully")
        return svm
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test target
            
        Returns:
            dict: Evaluation results
        """
        print("Evaluating models...")
        
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            results = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            self.evaluation_results[name] = results
            
            print(f"\n{name.upper()} Results:")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall: {results['recall']:.4f}")
            print(f"ROC-AUC: {results['roc_auc']:.4f}")
        
        return self.evaluation_results
    
    def evaluate_rule_based(self, X_test, y_test):
        """
        Evaluate rule-based system on test data.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test target
            
        Returns:
            dict: Rule-based evaluation results
        """
        print("Evaluating rule-based system...")
        
        predictor = RuleBasedPredictor()
        
        # Convert test data to dictionary format
        test_data = X_test.tolist()
        feature_dicts = []
        for i in range(len(test_data)):
            feature_dict = dict(zip(self.feature_columns, test_data[i]))
            feature_dicts.append(feature_dict)
        
        # Make predictions
        predictions = predictor.predict_batch(feature_dicts)
        y_pred = [p['prediction'] for p in predictions]
        y_pred_proba = [p['probability'] for p in predictions]
        
        # Calculate metrics
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        self.evaluation_results['rule_based'] = results
        
        print(f"\nRULE-BASED Results:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"ROC-AUC: {results['roc_auc']:.4f}")
        
        return results
    
    def save_models(self, model_dir='models'):
        """
        Save trained models and scaler.
        
        Args:
            model_dir (str): Directory to save models
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            model_path = os.path.join(model_dir, f'{name}.joblib')
            joblib.dump(model, model_path)
            print(f"Saved {name} model to {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")
        
        # Save feature columns
        feature_path = os.path.join(model_dir, 'feature_columns.joblib')
        joblib.dump(self.feature_columns, feature_path)
        print(f"Saved feature columns to {feature_path}")
        
        # Save evaluation results
        eval_path = os.path.join(model_dir, 'evaluation_results.joblib')
        joblib.dump(self.evaluation_results, eval_path)
        print(f"Saved evaluation results to {eval_path}")
    
    def generate_comparison_report(self):
        """
        Generate model comparison report.
        
        Returns:
            str: Formatted comparison report
        """
        if not self.evaluation_results:
            return "No evaluation results available. Run evaluate_models() first."
        
        report = []
        report.append("MODEL COMPARISON REPORT")
        report.append("=" * 50)
        
        # Create comparison table
        metrics = ['accuracy', 'precision', 'recall', 'roc_auc']
        report.append(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'ROC-AUC':<10}")
        report.append("-" * 60)
        
        for name, results in self.evaluation_results.items():
            row = f"{name:<15}"
            for metric in metrics:
                row += f" {results[metric]:<10.4f}"
            report.append(row)
        
        report.append("\n" + "=" * 50)
        report.append("DETAILED CLASSIFICATION REPORTS")
        report.append("=" * 50)
        
        for name, results in self.evaluation_results.items():
            report.append(f"\n{name.upper()} Classification Report:")
            report.append(results['classification_report'])
        
        return "\n".join(report)

def main():
    """
    Main training pipeline.
    """
    print("Starting model training pipeline...")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data()
    
    # Train models
    trainer.train_random_forest(X_train, y_train)
    trainer.train_svm(X_train, y_train)
    
    # Evaluate models
    trainer.evaluate_models(X_test, y_test)
    trainer.evaluate_rule_based(X_test, y_test)
    
    # Generate comparison report
    report = trainer.generate_comparison_report()
    print("\n" + report)
    
    # Save models and results
    trainer.save_models()
    
    # Save report to file
    with open('models/comparison_report.txt', 'w') as f:
        f.write(report)
    print("\nComparison report saved to models/comparison_report.txt")
    
    print("\nTraining pipeline completed successfully!")

if __name__ == "__main__":
    main()
