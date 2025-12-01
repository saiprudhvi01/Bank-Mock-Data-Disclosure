class RuleBasedPredictor:
    """
    Rule-based prediction system for bank user disclosure likelihood.
    Provides transparent predictions with detailed explanations.
    """
    
    def __init__(self):
        self.rules = {
            'credit_score': {
                'threshold': 600,
                'weight': 2,
                'condition': 'less_than',
                'reason': 'Low credit score (<600) increases disclosure risk'
            },
            'balance': {
                'threshold': 5000,
                'weight': 2,
                'condition': 'less_than',
                'reason': 'Low account balance (<$5,000) increases disclosure risk'
            },
            'age': {
                'threshold': 25,
                'weight': 1,
                'condition': 'less_than',
                'reason': 'Young age (<25) increases disclosure risk'
            },
            'tenure': {
                'threshold': 12,
                'weight': 1,
                'condition': 'less_than',
                'reason': 'Short account tenure (<12 months) increases disclosure risk'
            },
            'is_active_member': {
                'threshold': 1,
                'weight': 2,
                'condition': 'equals',
                'reason': 'Inactive membership increases disclosure risk'
            },
            'has_phone': {
                'threshold': 1,
                'weight': 1,
                'condition': 'equals',
                'reason': 'No phone number registered increases disclosure risk'
            },
            'num_products': {
                'threshold': 2,
                'weight': 1,
                'condition': 'less_than_or_equal',
                'reason': 'Few financial products (≤2) increases disclosure risk'
            },
            'has_credit_card': {
                'threshold': 1,
                'weight': 1,
                'condition': 'equals',
                'reason': 'No credit card increases disclosure risk'
            },
            'has_loan': {
                'threshold': 0,
                'weight': 1,
                'condition': 'greater_than',
                'reason': 'Active loan increases disclosure risk'
            }
        }
        self.prediction_threshold = 4.0  # Increased threshold due to more rules
    
    def predict_single(self, user_data):
        """
        Predict disclosure likelihood for a single user.
        
        Args:
            user_data (dict): Dictionary containing user features
            
        Returns:
            dict: Prediction with probability and explanation
        """
        score = 0
        triggered_rules = []
        
        # Apply each rule
        for feature, rule in self.rules.items():
            if feature not in user_data:
                continue
                
            value = user_data[feature]
            threshold = rule['threshold']
            condition = rule['condition']
            
            rule_triggered = False
            
            if condition == 'less_than' and value < threshold:
                rule_triggered = True
            elif condition == 'less_than_or_equal' and value <= threshold:
                rule_triggered = True
            elif condition == 'equals' and value == threshold:
                rule_triggered = True
            elif condition == 'greater_than' and value > threshold:
                rule_triggered = True
            elif condition == 'greater_than_or_equal' and value >= threshold:
                rule_triggered = True
            
            if rule_triggered:
                score += rule['weight']
                triggered_rules.append({
                    'feature': feature,
                    'value': value,
                    'threshold': threshold,
                    'weight': rule['weight'],
                    'reason': rule['reason']
                })
        
        # Convert score to probability
        probability = self._score_to_probability(score)
        
        # Make binary prediction
        will_disclose = score >= self.prediction_threshold
        
        # Generate explanation
        explanation = self._generate_explanation(triggered_rules, score, will_disclose)
        
        return {
            'prediction': int(will_disclose),
            'probability': probability,
            'score': score,
            'threshold': self.prediction_threshold,
            'triggered_rules': triggered_rules,
            'explanation': explanation
        }
    
    def predict_batch(self, user_data_list):
        """
        Predict disclosure likelihood for multiple users.
        
        Args:
            user_data_list (list): List of dictionaries containing user features
            
        Returns:
            list: List of prediction dictionaries
        """
        return [self.predict_single(user_data) for user_data in user_data_list]
    
    def _score_to_probability(self, score):
        """
        Convert rule score to probability using sigmoid function.
        
        Args:
            score (float): Rule-based score
            
        Returns:
            float: Probability (0-1)
        """
        # Use sigmoid function to convert score to probability
        # Shift and scale to get reasonable probability range
        import math
        adjusted_score = (score - self.prediction_threshold) * 2
        probability = 1 / (1 + math.exp(-adjusted_score))
        return max(0.01, min(0.99, probability))  # Clip to avoid extreme values
    
    def _generate_explanation(self, triggered_rules, score, will_disclose):
        """
        Generate human-readable explanation for the prediction.
        
        Args:
            triggered_rules (list): List of triggered rule dictionaries
            score (float): Total rule score
            will_disclose (bool): Binary prediction
            
        Returns:
            str: Human-readable explanation
        """
        if not triggered_rules:
            return "No risk factors detected. User appears to have low disclosure risk."
        
        explanation_parts = []
        
        # Start with overall assessment
        if will_disclose:
            explanation_parts.append("HIGH DISCLOSURE RISK DETECTED")
            explanation_parts.append(f"Risk Score: {score:.1f} (Threshold: {self.prediction_threshold})")
        else:
            explanation_parts.append("LOW DISCLOSURE RISK")
            explanation_parts.append(f"Risk Score: {score:.1f} (Threshold: {self.prediction_threshold})")
        
        explanation_parts.append("\nRisk Factors Identified:")
        
        # List triggered rules
        for rule in triggered_rules:
            if rule['feature'] in ['balance', 'estimated_salary']:
                value_str = f"${rule['value']:,.2f}"
                threshold_str = f"${rule['threshold']:,.2f}"
            elif rule['feature'] == 'credit_score':
                value_str = str(int(rule['value']))
                threshold_str = str(rule['threshold'])
            else:
                value_str = str(rule['value'])
                threshold_str = str(rule['threshold'])
            
            explanation_parts.append(f"• {rule['reason']} (Value: {value_str}, Weight: +{rule['weight']})")
        
        # Add recommendation
        if will_disclose:
            explanation_parts.append("\nRECOMMENDATION: Additional verification and monitoring advised.")
        else:
            explanation_parts.append("\nRECOMMENDATION: Standard processing acceptable.")
        
        return "\n".join(explanation_parts)
    
    def get_rule_summary(self):
        """
        Get summary of all rules used in the system.
        
        Returns:
            dict: Rule summary
        """
        return {
            'total_rules': len(self.rules),
            'prediction_threshold': self.prediction_threshold,
            'rules': self.rules
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize predictor
    predictor = RuleBasedPredictor()
    
    # Test with sample data
    test_user = {
        'age': 22,
        'balance': 3000.0,
        'tenure': 6,
        'num_products': 1,
        'credit_score': 550,
        'has_phone': 0,
        'is_active_member': 0,
        'estimated_salary': 35000.0
    }
    
    # Make prediction
    result = predictor.predict_single(test_user)
    
    print("Rule-Based Prediction System Test")
    print("=" * 50)
    print(f"Prediction: {'Will Disclose' if result['prediction'] else 'Will Not Disclose'}")
    print(f"Probability: {result['probability']:.3f}")
    print(f"Score: {result['score']:.1f}")
    print("\nExplanation:")
    print(result['explanation'])
    
    # Display rule summary
    print("\n" + "=" * 50)
    print("Rule System Summary:")
    summary = predictor.get_rule_summary()
    print(f"Total Rules: {summary['total_rules']}")
    print(f"Prediction Threshold: {summary['prediction_threshold']}")
