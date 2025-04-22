"""
Script to generate synthetic credit scoring data for demonstration purposes.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification


def generate_credit_data(n_samples=10000, random_state=42, output_path=None):
    """
    Generate synthetic credit scoring data.
    
    Args:
        n_samples (int): Number of samples to generate
        random_state (int): Random seed for reproducibility
        output_path (str): Path to save the generated data
        
    Returns:
        pd.DataFrame: Generated credit data
    """
    # Set random seed
    np.random.seed(random_state)
    
    # Generate synthetic binary classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        weights=[0.85, 0.15],  # Imbalanced dataset (15% default rate)
        random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create a DataFrame
    df = pd.DataFrame(X_scaled, columns=[
        'income',
        'age',
        'employment_length',
        'debt_to_income',
        'credit_score',
        'num_credit_lines',
        'num_late_payments',
        'loan_amount',
        'loan_term',
        'interest_rate'
    ])
    
    # Transform the features to more realistic values
    
    # Income (annual income in dollars)
    df['income'] = np.exp(df['income'] * 0.4 + 11)  # Log-normal distribution centered around ~$60,000
    
    # Age (in years)
    df['age'] = df['age'] * 10 + 40  # Normal distribution centered around 40 years
    df['age'] = df['age'].clip(18, 90).astype(int)  # Clip to realistic age range and convert to integer
    
    # Employment length (in years)
    df['employment_length'] = (df['employment_length'] * 3 + 5).clip(0, 40).astype(int)
    
    # Debt-to-income ratio
    df['debt_to_income'] = (df['debt_to_income'] * 0.1 + 0.3).clip(0, 0.8)
    
    # Credit score
    df['credit_score'] = (df['credit_score'] * 100 + 650).clip(300, 850).astype(int)
    
    # Number of credit lines
    df['num_credit_lines'] = (df['num_credit_lines'] * 2 + 5).clip(0, 20).astype(int)
    
    # Number of late payments
    df['num_late_payments'] = (df['num_late_payments'] * 2 + 1).clip(0, 10).astype(int)
    
    # Loan amount
    df['loan_amount'] = np.exp(df['loan_amount'] * 0.5 + 9).clip(1000, 100000).astype(int)
    
    # Loan term (in months)
    loan_terms = np.array([12, 24, 36, 48, 60, 72, 84, 96, 120])
    df['loan_term'] = loan_terms[np.digitize(df['loan_term'], bins=np.linspace(-3, 3, len(loan_terms) - 1))]
    
    # Interest rate
    df['interest_rate'] = (df['interest_rate'] * 2 + 5).clip(1, 15)
    
    # Add some categorical features
    
    # Housing status
    housing_status = ['Own', 'Mortgage', 'Rent']
    housing_probs = [0.3, 0.4, 0.3]
    df['housing_status'] = np.random.choice(housing_status, size=n_samples, p=housing_probs)
    
    # Loan purpose
    loan_purposes = [
        'Debt Consolidation', 
        'Credit Card Refinancing', 
        'Home Improvement', 
        'Major Purchase', 
        'Medical Expenses',
        'Business',
        'Auto',
        'Other'
    ]
    loan_purpose_probs = [0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05]
    df['loan_purpose'] = np.random.choice(loan_purposes, size=n_samples, p=loan_purpose_probs)
    
    # Employment status
    employment_status = ['Full-time', 'Part-time', 'Self-employed', 'Retired', 'Other']
    employment_probs = [0.7, 0.1, 0.1, 0.05, 0.05]
    df['employment_status'] = np.random.choice(employment_status, size=n_samples, p=employment_probs)
    
    # Education level
    education_levels = [
        'High School', 
        'Associate Degree', 
        'Bachelor\'s Degree', 
        'Master\'s Degree',
        'Doctoral Degree',
        'Other'
    ]
    education_probs = [0.2, 0.2, 0.4, 0.1, 0.05, 0.05]
    df['education'] = np.random.choice(education_levels, size=n_samples, p=education_probs)
    
    # Add the target variable (loan default: 1 = default, 0 = no default)
    df['loan_default'] = y
    
    # Make some adjustments to create more realistic relationships
    
    # Higher income should reduce default probability
    default_adjustment = -0.2 * (df['income'] > df['income'].quantile(0.75)).astype(int)
    
    # Higher credit score should reduce default probability
    default_adjustment += -0.3 * (df['credit_score'] > df['credit_score'].quantile(0.75)).astype(int)
    
    # Higher debt-to-income should increase default probability
    default_adjustment += 0.2 * (df['debt_to_income'] > df['debt_to_income'].quantile(0.75)).astype(int)
    
    # More late payments should increase default probability
    default_adjustment += 0.3 * (df['num_late_payments'] > df['num_late_payments'].quantile(0.75)).astype(int)
    
    # Apply adjustments with some probability
    mask = np.random.rand(n_samples) < 0.7  # Apply to 70% of the samples
    df.loc[mask & (default_adjustment < 0) & (df['loan_default'] == 1), 'loan_default'] = 0
    df.loc[mask & (default_adjustment > 0) & (df['loan_default'] == 0), 'loan_default'] = 1
    
    # Save the data if an output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
    
    return df


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate synthetic credit scoring data')
    
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Number of samples to generate')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output-path', type=str, default='data/raw/credit_data.csv',
                        help='Path to save the generated data')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Generate and save credit data
    df = generate_credit_data(
        n_samples=args.n_samples, 
        random_state=args.random_seed, 
        output_path=args.output_path
    )
    
    # Display some information about the data
    print(f"Generated {len(df)} credit scoring records")
    print(f"Default rate: {df['loan_default'].mean():.2%}")
    print("\nFeature statistics:")
    print(df.describe().T[['mean', 'std', 'min', 'max']])
    
    # Display correlations with the target
    # Get only numeric columns for correlation calculation
    numeric_cols = df.select_dtypes(include=['number']).columns
    correlations = df[numeric_cols].corr()['loan_default'].sort_values(ascending=False)
    print("\nFeature correlations with loan_default:")
    print(correlations) 