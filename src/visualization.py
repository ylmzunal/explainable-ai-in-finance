"""
Module for visualization utilities.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


def plot_roc_curve(y_true, y_pred_proba, output_path=None, model_names=None, title='ROC Curve'):
    """
    Plot the ROC curve for one or more models.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (dict if multiple models)
        output_path: Path to save the plot (None for display)
        model_names: List of model names (if None, defaults to 'Model')
        title: Plot title
        
    Returns:
        fig: The created figure (None if display only)
    """
    plt.figure(figsize=(10, 8))
    
    # If y_pred_proba is not a dict, convert it to one
    if not isinstance(y_pred_proba, dict):
        if model_names is None:
            model_names = ['Model']
        y_pred_proba = {model_names[0]: y_pred_proba}
    elif model_names is None:
        model_names = list(y_pred_proba.keys())
    
    # Plot ROC curve for each model
    for i, (model_name, pred_proba) in enumerate(y_pred_proba.items()):
        fpr, tpr, _ = roc_curve(y_true, pred_proba)
        plt.plot(
            fpr, 
            tpr, 
            lw=2, 
            label=f'{model_name}'
        )
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save or display
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    else:
        fig = plt.gcf()
        return fig


def plot_precision_recall_curve(y_true, y_pred_proba, output_path=None, 
                                model_names=None, title='Precision-Recall Curve'):
    """
    Plot the Precision-Recall curve for one or more models.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (dict if multiple models)
        output_path: Path to save the plot (None for display)
        model_names: List of model names (if None, defaults to 'Model')
        title: Plot title
        
    Returns:
        fig: The created figure (None if display only)
    """
    plt.figure(figsize=(10, 8))
    
    # If y_pred_proba is not a dict, convert it to one
    if not isinstance(y_pred_proba, dict):
        if model_names is None:
            model_names = ['Model']
        y_pred_proba = {model_names[0]: y_pred_proba}
    elif model_names is None:
        model_names = list(y_pred_proba.keys())
    
    # Plot Precision-Recall curve for each model
    for i, (model_name, pred_proba) in enumerate(y_pred_proba.items()):
        precision, recall, _ = precision_recall_curve(y_true, pred_proba)
        plt.plot(
            recall, 
            precision, 
            lw=2, 
            label=f'{model_name}'
        )
    
    # Set labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save or display
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    else:
        fig = plt.gcf()
        return fig


def plot_confusion_matrix(y_true, y_pred, output_path=None, class_names=None, 
                         title='Confusion Matrix', normalize=False, cmap='Blues'):
    """
    Plot the confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the plot (None for display)
        class_names: List of class names
        title: Plot title
        normalize: Whether to normalize the confusion matrix
        cmap: Colormap to use
        
    Returns:
        fig: The created figure (None if display only)
    """
    plt.figure(figsize=(8, 6))
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Plot confusion matrix
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap=cmap, 
        cbar=False,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    # Set labels and title
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    
    # Save or display
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    else:
        fig = plt.gcf()
        return fig


def plot_feature_importances(feature_importances, feature_names=None, n_top=20, 
                             output_path=None, title='Feature Importances'):
    """
    Plot feature importances.
    
    Args:
        feature_importances: Feature importance values
        feature_names: List of feature names
        n_top: Number of top features to show
        output_path: Path to save the plot (None for display)
        title: Plot title
        
    Returns:
        fig: The created figure (None if display only)
    """
    plt.figure(figsize=(10, 8))
    
    # If feature_importances is a dict, convert to a DataFrame
    if isinstance(feature_importances, dict):
        feature_importances = pd.Series(feature_importances)
    
    # If feature_names is not provided, use indices
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(feature_importances))]
    
    # Create DataFrame
    importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    
    # Sort by importance
    importances_df = importances_df.sort_values('Importance', ascending=False)
    
    # Get top N features
    if n_top is not None and n_top < len(importances_df):
        importances_df = importances_df.head(n_top)
    
    # Plot bar chart
    sns.barplot(
        x='Importance',
        y='Feature',
        data=importances_df,
        palette='viridis'
    )
    
    # Set labels and title
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, axis='x', alpha=0.3)
    
    # Save or display
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    else:
        fig = plt.gcf()
        return fig


def plot_correlation_matrix(df, output_path=None, title='Correlation Matrix', 
                            mask_upper=True, cmap='coolwarm', figsize=(12, 10)):
    """
    Plot correlation matrix of features.
    
    Args:
        df: DataFrame with features
        output_path: Path to save the plot (None for display)
        title: Plot title
        mask_upper: Whether to mask the upper triangle
        cmap: Colormap to use
        figsize: Figure size
        
    Returns:
        fig: The created figure (None if display only)
    """
    plt.figure(figsize=figsize)
    
    # Compute correlation matrix
    corr = df.corr()
    
    # Generate mask for the upper triangle
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool))
    else:
        mask = None
    
    # Plot correlation matrix
    sns.heatmap(
        corr, 
        mask=mask,
        cmap=cmap,
        annot=True,
        fmt='.2f',
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8}
    )
    
    # Set title
    plt.title(title, fontsize=14)
    
    # Save or display
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    else:
        fig = plt.gcf()
        return fig


def plot_distribution(df, column, hue=None, output_path=None, 
                      title=None, kind='hist', kde=True, figsize=(10, 6)):
    """
    Plot distribution of a feature.
    
    Args:
        df: DataFrame with data
        column: Column to plot
        hue: Column to use for grouping
        output_path: Path to save the plot (None for display)
        title: Plot title (if None, uses column name)
        kind: Kind of plot ('hist' or 'box')
        kde: Whether to include KDE in histogram
        figsize: Figure size
        
    Returns:
        fig: The created figure (None if display only)
    """
    plt.figure(figsize=figsize)
    
    if title is None:
        title = f"Distribution of {column}"
    
    if kind == 'hist':
        sns.histplot(
            data=df,
            x=column,
            hue=hue,
            kde=kde,
            element='step',
            common_norm=False,
            alpha=0.7
        )
    elif kind == 'box':
        sns.boxplot(
            data=df,
            x=hue,
            y=column,
            palette='viridis'
        )
    else:
        raise ValueError(f"Unsupported plot kind: {kind}")
    
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Save or display
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    else:
        fig = plt.gcf()
        return fig 