"""
Data processing module for loading and preprocessing credit scoring data.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data(data_path, filename):
    """
    Load data from the specified path and filename.
    
    Args:
        data_path (str): Path to the data directory
        filename (str): Name of the data file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    file_path = os.path.join(data_path, filename)
    
    if filename.endswith('.csv'):
        return pd.read_csv(file_path)
    elif filename.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel files.")


def preprocess_data(df, target_column, categorical_features=None, numerical_features=None):
    """
    Preprocess the data for credit scoring models.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        categorical_features (list): List of categorical feature names
        numerical_features (list): List of numerical feature names
        
    Returns:
        tuple: Preprocessed X_train, X_test, y_train, y_test, and preprocessing pipeline
    """
    # Create a copy of the dataframe to avoid modifying the original
    data = df.copy()
    
    # Handle missing values
    data = handle_missing_values(data)
    
    # If features not explicitly provided, infer them
    if categorical_features is None and numerical_features is None:
        categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove target column from features if present
        if target_column in categorical_features:
            categorical_features.remove(target_column)
        if target_column in numerical_features:
            numerical_features.remove(target_column)
    
    # Feature preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop columns not specified
    )
    
    # Split data into features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit the preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Handle class imbalance with SMOTE (only on training data)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    
    return X_train_resampled, X_test_processed, y_train_resampled, y_test, preprocessor


def handle_missing_values(df):
    """
    Handle missing values in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    data = df.copy()
    
    # For numerical columns, fill with median
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_columns:
        data[col] = data[col].fillna(data[col].median())
    
    # For categorical columns, fill with mode
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        data[col] = data[col].fillna(data[col].mode()[0])
    
    return data


def create_feature_descriptions(categorical_features, numerical_features):
    """
    Create a dictionary of feature descriptions for explainability.
    
    Args:
        categorical_features (list): List of categorical feature names
        numerical_features (list): List of numerical feature names
        
    Returns:
        dict: Dictionary mapping feature names to descriptions
    """
    # This is just a template - in a real application, you would have
    # more detailed descriptions for each feature
    feature_descriptions = {}
    
    for feature in categorical_features:
        feature_descriptions[feature] = f"Categorical feature: {feature}"
    
    for feature in numerical_features:
        feature_descriptions[feature] = f"Numerical feature: {feature}"
    
    # Some example financial feature descriptions
    common_financial_features = {
        'income': 'Annual income of the applicant',
        'age': 'Age of the applicant in years',
        'employment_length': 'Length of current employment in years',
        'debt_to_income': 'Debt to income ratio',
        'loan_amount': 'Requested loan amount',
        'loan_term': 'Term of loan in months',
        'credit_score': 'Credit score of the applicant',
        'num_credit_lines': 'Number of credit lines',
        'num_late_payments': 'Number of late payments in the last 2 years',
        'housing_status': 'Housing status (own, mortgage, rent)',
        'loan_purpose': 'Purpose of the loan'
    }
    
    # Update with common financial features if they exist in our feature lists
    feature_descriptions.update({
        k: v for k, v in common_financial_features.items() 
        if k in categorical_features or k in numerical_features
    })
    
    return feature_descriptions


def save_processed_data(X_train, X_test, y_train, y_test, output_path):
    """
    Save processed data to disk.
    
    Args:
        X_train: Processed training features
        X_test: Processed test features
        y_train: Training target
        y_test: Test target
        output_path (str): Path to save the processed data
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Convert to pandas DataFrames if they're not already
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    
    # Save to csv
    X_train.to_csv(os.path.join(output_path, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_path, 'X_test.csv'), index=False)
    pd.Series(y_train).to_csv(os.path.join(output_path, 'y_train.csv'), index=False)
    pd.Series(y_test).to_csv(os.path.join(output_path, 'y_test.csv'), index=False) 