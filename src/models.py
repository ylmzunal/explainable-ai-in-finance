"""
Module for training, evaluating, and saving credit scoring models.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import optuna

# xgboost and lightgbm will be imported conditionally when needed
# import xgboost as xgb
# import lightgbm as lgb


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained model
    """
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        penalty='l2',
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained model
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    """
    Train a Gradient Boosting model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained model
    """
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """
    Train an XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained model
    """
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        return model
    except ImportError:
        print("XGBoost is not installed. Please install it with 'pip install xgboost'.")
        return None


def train_lightgbm(X_train, y_train):
    """
    Train a LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained model
    """
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
    except ImportError:
        print("LightGBM is not installed. Please install it with 'pip install lightgbm'.")
        return None


def optimize_model_hyperparameters(X_train, y_train, X_test, y_test, model_type='xgboost', n_trials=50):
    """
    Optimize model hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        model_type (str): Type of model to optimize ('xgboost', 'lightgbm', 'rf', 'gb')
        n_trials (int): Number of optimization trials
        
    Returns:
        Trained model with optimized hyperparameters
    """
    def objective(trial):
        if model_type == 'xgboost':
            try:
                import xgboost as xgb
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'random_state': 42,
                    'use_label_encoder': False,
                    'eval_metric': 'logloss'
                }
                model = xgb.XGBClassifier(**params)
            except ImportError:
                print("XGBoost is not installed. Skipping this optimization.")
                return float('inf')
        
        elif model_type == 'lightgbm':
            try:
                import lightgbm as lgb
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'random_state': 42
                }
                model = lgb.LGBMClassifier(**params)
            except ImportError:
                print("LightGBM is not installed. Skipping this optimization.")
                return float('inf')
        
        elif model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42,
                'class_weight': 'balanced'
            }
            model = RandomForestClassifier(**params)
        
        elif model_type == 'gb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42
            }
            model = GradientBoostingClassifier(**params)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict and calculate AUC
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        return auc
    
    # Create study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    
    # Train model with best parameters
    if model_type == 'xgboost':
        best_params.update({
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        })
        model = xgb.XGBClassifier(**best_params)
    
    elif model_type == 'lightgbm':
        best_params.update({'random_state': 42})
        model = lgb.LGBMClassifier(**best_params)
    
    elif model_type == 'rf':
        best_params.update({
            'random_state': 42,
            'class_weight': 'balanced'
        })
        model = RandomForestClassifier(**best_params)
    
    elif model_type == 'gb':
        best_params.update({'random_state': 42})
        model = GradientBoostingClassifier(**best_params)
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model, best_params


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        threshold (float): Decision threshold for binary classification
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    return metrics


def save_model(model, model_path, model_name, metadata=None):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model
        model_path (str): Path to save the model
        model_name (str): Name of the model file
        metadata (dict): Additional metadata to save with the model
    """
    os.makedirs(model_path, exist_ok=True)
    
    model_file = os.path.join(model_path, f"{model_name}.joblib")
    
    # Create a dictionary with the model and metadata
    model_data = {
        'model': model,
        'metadata': metadata or {}
    }
    
    # Save the model
    joblib.dump(model_data, model_file)
    
    print(f"Model saved to {model_file}")
    

def load_model(model_path, model_name):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the model
        model_name (str): Name of the model file
        
    Returns:
        tuple: (model, metadata)
    """
    model_file = os.path.join(model_path, f"{model_name}.joblib")
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    # Load the model data
    model_data = joblib.load(model_file)
    
    return model_data['model'], model_data['metadata'] 