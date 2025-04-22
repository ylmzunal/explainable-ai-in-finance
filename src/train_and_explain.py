"""
Main script for training credit scoring models and generating explanations.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from data_processing import (
    load_data, 
    preprocess_data, 
    create_feature_descriptions,
    save_processed_data
)
from models import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    train_gradient_boosting,
    train_lightgbm,
    optimize_model_hyperparameters,
    evaluate_model,
    save_model,
    load_model
)
from explainers import ShapExplainer, LimeExplainer
from visualization import (
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_feature_importances,
    plot_correlation_matrix,
    plot_distribution
)


def train_models(X_train, y_train, models_to_train=None, optimize=False, X_test=None, y_test=None):
    """
    Train multiple credit scoring models.
    
    Args:
        X_train: Training features
        y_train: Training target
        models_to_train: List of model names to train
        optimize: Whether to optimize hyperparameters
        X_test: Test features (needed for optimization)
        y_test: Test target (needed for optimization)
        
    Returns:
        dict: Dictionary of trained models
    """
    if models_to_train is None:
        models_to_train = ['logistic', 'rf', 'xgboost', 'gb', 'lightgbm']
    
    models = {}
    
    for model_name in models_to_train:
        print(f"Training {model_name} model...")
        
        if model_name == 'logistic':
            models[model_name] = train_logistic_regression(X_train, y_train)
        
        elif model_name == 'rf':
            if optimize and X_test is not None and y_test is not None:
                model_result, best_params = optimize_model_hyperparameters(
                    X_train, y_train, X_test, y_test, model_type='rf', n_trials=20
                )
                if model_result is not None:
                    models[model_name] = model_result
                    print(f"Best parameters for RF: {best_params}")
            else:
                models[model_name] = train_random_forest(X_train, y_train)
        
        elif model_name == 'gb':
            if optimize and X_test is not None and y_test is not None:
                model_result, best_params = optimize_model_hyperparameters(
                    X_train, y_train, X_test, y_test, model_type='gb', n_trials=20
                )
                if model_result is not None:
                    models[model_name] = model_result
                    print(f"Best parameters for GB: {best_params}")
            else:
                models[model_name] = train_gradient_boosting(X_train, y_train)
        
        elif model_name == 'xgboost':
            if optimize and X_test is not None and y_test is not None:
                model_result, best_params = optimize_model_hyperparameters(
                    X_train, y_train, X_test, y_test, model_type='xgboost', n_trials=20
                )
                if model_result is not None:
                    models[model_name] = model_result
                    print(f"Best parameters for XGBoost: {best_params}")
            else:
                model_result = train_xgboost(X_train, y_train)
                if model_result is not None:
                    models[model_name] = model_result
        
        elif model_name == 'lightgbm':
            if optimize and X_test is not None and y_test is not None:
                model_result, best_params = optimize_model_hyperparameters(
                    X_train, y_train, X_test, y_test, model_type='lightgbm', n_trials=20
                )
                if model_result is not None:
                    models[model_name] = model_result
                    print(f"Best parameters for LightGBM: {best_params}")
            else:
                model_result = train_lightgbm(X_train, y_train)
                if model_result is not None:
                    models[model_name] = model_result
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    if not models:
        raise ValueError("No models were successfully trained. Please check dependencies.")
    
    return models


def evaluate_models(models, X_test, y_test, output_dir=None):
    """
    Evaluate multiple models and save results.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test target
        output_dir: Directory to save evaluation results
        
    Returns:
        dict: Dictionary of evaluation metrics for each model
    """
    evaluation_results = {}
    model_predictions = {}
    
    for model_name, model in models.items():
        if model is None:
            print(f"Skipping evaluation of {model_name} model as it was not successfully trained.")
            continue
            
        print(f"Evaluating {model_name} model...")
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        evaluation_results[model_name] = metrics
        
        # Get predictions for ROC curve
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        model_predictions[model_name] = y_pred_proba
        
        # Print metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Confusion Matrix: {metrics['confusion_matrix']}")
        print(f"  Classification Report:\n{metrics['classification_report']}")
    
    # Save plots if output directory is provided
    if output_dir and model_predictions:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot ROC curves
        plot_roc_curve(
            y_test,
            model_predictions,
            output_path=os.path.join(output_dir, 'roc_curves.png'),
            title='ROC Curves for Credit Scoring Models'
        )
        
        # Plot Precision-Recall curves
        plot_precision_recall_curve(
            y_test,
            model_predictions,
            output_path=os.path.join(output_dir, 'precision_recall_curves.png'),
            title='Precision-Recall Curves for Credit Scoring Models'
        )
        
        # Plot confusion matrices for each model
        for model_name, model in models.items():
            if model is None:
                continue
            y_pred = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
            plot_confusion_matrix(
                y_test,
                y_pred,
                output_path=os.path.join(output_dir, f'confusion_matrix_{model_name}.png'),
                class_names=['No Default', 'Default'],
                title=f'Confusion Matrix - {model_name.upper()}'
            )
    
    return evaluation_results


def generate_shap_explanations(models, X_train, X_test, feature_names, output_dir=None, n_samples=100):
    """
    Generate SHAP explanations for trained models.
    
    Args:
        models: Dictionary of trained models
        X_train: Training features
        X_test: Test features
        feature_names: List of feature names
        output_dir: Directory to save explanations
        n_samples: Number of samples to use for explanations
        
    Returns:
        dict: Dictionary of SHAP explainers for each model
    """
    shap_explainers = {}
    
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for model_name, model in models.items():
        if model is None:
            print(f"Skipping SHAP explanations for {model_name} model as it was not successfully trained.")
            continue
            
        print(f"Generating SHAP explanations for {model_name} model...")
        
        # Initialize SHAP explainer
        if model_name in ['rf', 'xgboost', 'lightgbm', 'gb']:
            model_type = 'tree'
        elif model_name == 'logistic':
            model_type = 'linear'
        else:
            model_type = None
        
        explainer = ShapExplainer(model, X_train, feature_names=feature_names, model_type=model_type)
        shap_explainers[model_name] = explainer
        
        # Generate global explanations (summary plots)
        if output_dir:
            # Bar summary plot
            explainer.plot_summary(
                X_test,
                n_samples=n_samples,
                plot_type='bar',
                output_path=os.path.join(output_dir, f'shap_summary_bar_{model_name}.png'),
                title=f'SHAP Feature Importance - {model_name.upper()}'
            )
            
            # Beeswarm plot
            explainer.plot_summary(
                X_test,
                n_samples=n_samples,
                plot_type='violin',
                output_path=os.path.join(output_dir, f'shap_summary_violin_{model_name}.png'),
                title=f'SHAP Feature Values - {model_name.upper()}'
            )
            
            # Generate dependence plots for top 3 features
            shap_values, _ = explainer.explain_dataset(X_test, n_samples=n_samples)
            
            # Get top features by mean absolute SHAP value
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[-3:]
            
            for idx in top_indices:
                feature_name = feature_names[idx]
                explainer.plot_dependence(
                    X_test,
                    feature_idx=idx,
                    n_samples=n_samples,
                    output_path=os.path.join(output_dir, f'shap_dependence_{model_name}_{feature_name}.png'),
                    title=f'SHAP Dependence Plot - {feature_name} ({model_name.upper()})'
                )
    
    return shap_explainers


def generate_lime_explanations(models, X_train, X_test, y_test, feature_names, output_dir=None, n_explanations=5):
    """
    Generate LIME explanations for trained models.
    
    Args:
        models: Dictionary of trained models
        X_train: Training features
        X_test: Test features
        y_test: Test target
        feature_names: List of feature names
        output_dir: Directory to save explanations
        n_explanations: Number of explanations to generate
        
    Returns:
        dict: Dictionary of LIME explainers for each model
    """
    lime_explainers = {}
    
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get random indices for explanation examples
    indices = np.random.choice(len(X_test), size=min(n_explanations, len(X_test)), replace=False)
    
    for model_name, model in models.items():
        if model is None:
            print(f"Skipping LIME explanations for {model_name} model as it was not successfully trained.")
            continue
            
        print(f"Generating LIME explanations for {model_name} model...")
        
        explainer = LimeExplainer(
            model,
            X_train,
            feature_names=feature_names,
            class_names=['No Default', 'Default'],
            mode='classification'
        )
        lime_explainers[model_name] = explainer
        
        # Select instances to explain
        # Include some defaulted and non-defaulted loans
        defaulted_indices = np.where(y_test == 1)[0]
        non_defaulted_indices = np.where(y_test == 0)[0]
        
        # Sample min(n_explanations, len(defaulted_indices)) defaulted loans
        n_defaulted = min(n_explanations // 2, len(defaulted_indices))
        defaulted_sample = np.random.choice(defaulted_indices, n_defaulted, replace=False)
        
        # Sample remaining non-defaulted loans
        n_non_defaulted = n_explanations - n_defaulted
        non_defaulted_sample = np.random.choice(non_defaulted_indices, n_non_defaulted, replace=False)
        
        # Combine samples
        sample_indices = np.concatenate([defaulted_sample, non_defaulted_sample])
        
        # Generate explanations for sampled instances
        for i, idx in enumerate(sample_indices):
            # Get instance
            instance = X_test[idx]
            true_label = y_test[idx]
            
            # Generate explanation
            exp = explainer.explain_instance(instance, num_features=10, num_samples=5000)
            
            # Plot and save explanation
            if output_dir:
                # Save as image
                explainer.plot_explanation(
                    exp,
                    output_path=os.path.join(output_dir, f'lime_explanation_{model_name}_instance_{i}.png')
                )
                
                # Save as HTML
                explainer.save_explanation_as_html(
                    exp,
                    output_path=os.path.join(output_dir, 'html', f'lime_explanation_{model_name}_instance_{i}.html')
                )
                
                # Save as DataFrame
                exp_df = explainer.get_explanation_as_dataframe(exp)
                exp_df.to_csv(os.path.join(output_dir, f'lime_explanation_{model_name}_instance_{i}.csv'), index=False)
    
    return lime_explainers


def save_models(models, output_dir, preprocessor=None, feature_names=None):
    """
    Save trained models and metadata.
    
    Args:
        models: Dictionary of trained models
        output_dir: Directory to save models
        preprocessor: Data preprocessing pipeline
        feature_names: List of feature names
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, model in models.items():
        print(f"Saving {model_name} model...")
        
        # Create metadata
        metadata = {
            'feature_names': feature_names,
            'model_type': model_name,
            'creation_time': pd.Timestamp.now().isoformat()
        }
        
        # Save model
        save_model(model, output_dir, model_name, metadata)
    
    # Save preprocessor if provided
    if preprocessor is not None:
        preprocessor_file = os.path.join(output_dir, 'preprocessor.joblib')
        joblib.dump(preprocessor, preprocessor_file)
        print(f"Preprocessor saved to {preprocessor_file}")


def main(args):
    """
    Main function to train models and generate explanations.
    
    Args:
        args: Command-line arguments
    """
    # Load data
    print("Loading data...")
    data = load_data(args.data_path, args.data_file)
    
    # Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        data, 
        args.target_column
    )
    
    # Get feature names (after preprocessing)
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        # For older scikit-learn versions or custom preprocessors
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    
    # Save processed data
    if args.save_processed:
        processed_data_path = os.path.join(args.output_dir, 'processed_data')
        save_processed_data(X_train, X_test, y_train, y_test, processed_data_path)
    
    # Train models
    print("Training models...")
    models = train_models(
        X_train, 
        y_train, 
        models_to_train=args.models.split(','),
        optimize=args.optimize,
        X_test=X_test,
        y_test=y_test
    )
    
    # Evaluate models
    print("Evaluating models...")
    evaluation_dir = os.path.join(args.output_dir, 'evaluations')
    evaluate_models(models, X_test, y_test, evaluation_dir)
    
    # Generate SHAP explanations
    print("Generating SHAP explanations...")
    shap_dir = os.path.join(args.output_dir, 'shap_explanations')
    generate_shap_explanations(models, X_train, X_test, feature_names, shap_dir)
    
    # Generate LIME explanations
    print("Generating LIME explanations...")
    lime_dir = os.path.join(args.output_dir, 'lime_explanations')
    generate_lime_explanations(models, X_train, X_test, y_test, feature_names, lime_dir)
    
    # Save models
    if args.save_models:
        print("Saving models...")
        models_dir = os.path.join(args.output_dir, 'models')
        save_models(models, models_dir, preprocessor, feature_names)
    
    print("Done!")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train and explain credit scoring models')
    
    parser.add_argument('--data-path', type=str, default='data/raw', 
                        help='Path to the data directory')
    parser.add_argument('--data-file', type=str, default='credit_data.csv', 
                        help='Name of the data file')
    parser.add_argument('--target-column', type=str, default='loan_default', 
                        help='Name of the target column')
    parser.add_argument('--output-dir', type=str, default='output', 
                        help='Directory to save outputs')
    parser.add_argument('--models', type=str, default='logistic,rf,xgboost,lightgbm', 
                        help='Comma-separated list of models to train')
    parser.add_argument('--optimize', action='store_true', 
                        help='Whether to optimize hyperparameters')
    parser.add_argument('--save-models', action='store_true', 
                        help='Whether to save trained models')
    parser.add_argument('--save-processed', action='store_true', 
                        help='Whether to save processed data')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run main function
    main(args) 