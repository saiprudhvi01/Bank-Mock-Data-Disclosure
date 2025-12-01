import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random

def generate_bank_user_dataset(n_samples=5000, random_state=42):
    """
    Generate synthetic bank user dataset with rule-based disclosure labels.
    
    Features:
    - age: User age (18-80)
    - balance: Account balance ($0-$100,000)
    - tenure: Account tenure in months (1-120)
    - num_products: Number of financial products (1-5)
    - credit_score: Credit score (300-850)
    - has_phone: Has phone number registered (0/1)
    - is_active_member: Is active member (0/1)
    - estimated_salary: Estimated annual salary ($20,000-$200,000)
    - has_credit_card: Has credit card (0/1)
    - has_loan: Has active loan (0/1)
    """
    np.random.seed(random_state)
    random.seed(random_state)
    
    # Generate features with realistic distributions
    data = {
        'age': np.random.normal(40, 15, n_samples).astype(int),
        'balance': np.random.lognormal(8, 1.5, n_samples),
        'tenure': np.random.gamma(2, 15, n_samples).astype(int),
        'num_products': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.3, 0.35, 0.2, 0.1, 0.05]),
        'credit_score': np.random.normal(650, 100, n_samples),
        'has_phone': np.random.choice([0, 1], n_samples, p=[0.15, 0.85]),
        'is_active_member': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'estimated_salary': np.random.lognormal(10.5, 0.5, n_samples),
        'has_credit_card': np.random.choice([0, 1], n_samples, p=[0.25, 0.75]),
        'has_loan': np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    }
    
    # Clean and bound the values
    df = pd.DataFrame(data)
    df['age'] = np.clip(df['age'], 18, 80)
    df['balance'] = np.clip(df['balance'], 0, 100000)
    df['tenure'] = np.clip(df['tenure'], 1, 120)
    df['credit_score'] = np.clip(df['credit_score'], 300, 850)
    df['estimated_salary'] = np.clip(df['estimated_salary'], 20000, 200000)
    
    # Apply rule-based scoring for disclosure likelihood
    df['disclosure_score'] = 0
    df['disclosure_reasons'] = [[] for _ in range(n_samples)]
    
    # Rule 1: Low credit score increases disclosure risk
    low_credit_mask = df['credit_score'] < 600
    df.loc[low_credit_mask, 'disclosure_score'] += 2
    for idx in df[low_credit_mask].index:
        df.at[idx, 'disclosure_reasons'].append('Low credit score (<600)')
    
    # Rule 2: Low balance increases disclosure risk
    low_balance_mask = df['balance'] < 5000
    df.loc[low_balance_mask, 'disclosure_score'] += 2
    for idx in df[low_balance_mask].index:
        df.at[idx, 'disclosure_reasons'].append('Low account balance (<$5,000)')
    
    # Rule 3: Young age increases disclosure risk
    young_age_mask = df['age'] < 25
    df.loc[young_age_mask, 'disclosure_score'] += 1
    for idx in df[young_age_mask].index:
        df.at[idx, 'disclosure_reasons'].append('Young age (<25)')
    
    # Rule 4: Short tenure increases disclosure risk
    short_tenure_mask = df['tenure'] < 12
    df.loc[short_tenure_mask, 'disclosure_score'] += 1
    for idx in df[short_tenure_mask].index:
        df.at[idx, 'disclosure_reasons'].append('Short account tenure (<12 months)')
    
    # Rule 5: Non-active membership increases disclosure risk
    inactive_mask = df['is_active_member'] == 0
    df.loc[inactive_mask, 'disclosure_score'] += 2
    for idx in df[inactive_mask].index:
        df.at[idx, 'disclosure_reasons'].append('Inactive membership')
    
    # Rule 6: No phone number increases disclosure risk
    no_phone_mask = df['has_phone'] == 0
    df.loc[no_phone_mask, 'disclosure_score'] += 1
    for idx in df[no_phone_mask].index:
        df.at[idx, 'disclosure_reasons'].append('No phone number registered')
    
    # Rule 7: Few financial products increases disclosure risk
    few_products_mask = df['num_products'] <= 2
    df.loc[few_products_mask, 'disclosure_score'] += 1
    for idx in df[few_products_mask].index:
        df.at[idx, 'disclosure_reasons'].append('Few financial products (≤2)')
    
    # Rule 8: No credit card increases disclosure risk
    no_credit_card_mask = df['has_credit_card'] == 0
    df.loc[no_credit_card_mask, 'disclosure_score'] += 1
    for idx in df[no_credit_card_mask].index:
        df.at[idx, 'disclosure_reasons'].append('No credit card increases disclosure risk')
    
    # Rule 9: Having active loan increases disclosure risk
    has_loan_mask = df['has_loan'] == 1
    df.loc[has_loan_mask, 'disclosure_score'] += 1
    for idx in df[has_loan_mask].index:
        df.at[idx, 'disclosure_reasons'].append('Active loan increases disclosure risk')
    
    # Add noise to make dataset more realistic
    noise = np.random.normal(0, 0.5, n_samples)
    df['disclosure_score'] = df['disclosure_score'] + noise
    
    # Create binary label based on threshold (score >= 3.5 = high disclosure risk)
    df['will_disclose'] = (df['disclosure_score'] >= 3.5).astype(int)
    
    # Drop intermediate columns
    df = df.drop(['disclosure_score'], axis=1)
    
    # Round numerical columns
    df['balance'] = df['balance'].round(2)
    df['estimated_salary'] = df['estimated_salary'].round(2)
    df['credit_score'] = df['credit_score'].round(0).astype(int)
    
    return df

def save_dataset(df, filepath):
    """Save dataset to CSV file."""
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")
    print(f"Shape: {df.shape}")
    print(f"Disclosure rate: {df['will_disclose'].mean():.2%}")

if __name__ == "__main__":
    # Generate dataset
    df = generate_bank_user_dataset(n_samples=5000, random_state=42)
    
    # Save to data directory
    save_dataset(df, 'data/bank_users.csv')
    
    # Display sample and statistics
    print("\nDataset Sample:")
    print(df.head())
    print("\nFeature Statistics:")
    print(df.describe())
    print("\nDisclosure Distribution:")
    print(df['will_disclose'].value_counts(normalize=True))
