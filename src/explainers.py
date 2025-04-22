"""
Module for model explainability methods (SHAP and LIME).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
from sklearn.pipeline import Pipeline


class ShapExplainer:
    """
    Class for SHAP (SHapley Additive exPlanations) model explanations.
    """
    
    def __init__(self, model, X_train, feature_names=None, model_type=None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model
            X_train: Training data used for explainer initialization
            feature_names: List of feature names
            model_type: Type of model ('tree', 'linear', or None for auto-detection)
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        
        # Determine model type if not provided
        if self.model_type is None:
            if hasattr(model, 'feature_importances_'):
                self.model_type = 'tree'
            elif hasattr(model, 'coef_'):
                self.model_type = 'linear'
            else:
                self.model_type = 'kernel'
        
        # Initialize the explainer based on model type
        if self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(model)
        elif self.model_type == 'linear':
            self.explainer = shap.LinearExplainer(model, X_train)
        else:
            # For other model types, use KernelExplainer
            if hasattr(model, 'predict_proba'):
                self.explainer = shap.KernelExplainer(
                    model.predict_proba, shap.sample(X_train, 100)
                )
            else:
                self.explainer = shap.KernelExplainer(
                    model.predict, shap.sample(X_train, 100)
                )
    
    def explain_instance(self, instance, class_index=1):
        """
        Generate SHAP explanation for a single instance.
        
        Args:
            instance: The instance to explain
            class_index: Index of the class to explain (default=1, for binary classification)
            
        Returns:
            shap_values: SHAP values for the instance
            base_value: Expected value (base value)
        """
        # Reshape instance if necessary
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        # Get SHAP values
        if self.model_type == 'linear' or self.model_type == 'tree':
            shap_values = self.explainer.shap_values(instance)
            base_value = self.explainer.expected_value
            
            # For tree models, select the class_index if it's a list
            if isinstance(shap_values, list):
                shap_values = shap_values[class_index]
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[class_index]
        else:
            shap_values = self.explainer.shap_values(instance, check_additivity=False)
            base_value = self.explainer.expected_value
            
            if isinstance(shap_values, list):
                shap_values = shap_values[class_index]
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[class_index]
        
        return shap_values, base_value
    
    def explain_dataset(self, X, n_samples=None, class_index=1):
        """
        Generate SHAP explanations for a dataset.
        
        Args:
            X: Dataset to explain
            n_samples: Number of samples to use (None for all)
            class_index: Index of the class to explain (default=1, for binary classification)
            
        Returns:
            shap_values: SHAP values for the dataset
            base_value: Expected value (base value)
        """
        # Sample data if requested
        if n_samples is not None and n_samples < X.shape[0]:
            indices = np.random.choice(X.shape[0], n_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Get SHAP values
        if self.model_type == 'linear' or self.model_type == 'tree':
            shap_values = self.explainer.shap_values(X_sample)
            base_value = self.explainer.expected_value
            
            # For tree models, select the class_index if it's a list
            if isinstance(shap_values, list):
                shap_values = shap_values[class_index]
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[class_index]
        else:
            shap_values = self.explainer.shap_values(X_sample, check_additivity=False)
            base_value = self.explainer.expected_value
            
            if isinstance(shap_values, list):
                shap_values = shap_values[class_index]
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[class_index]
        
        return shap_values, base_value
    
    def plot_summary(self, X, n_samples=100, class_index=1, plot_type='bar', max_display=10, 
                     output_path=None, title=None):
        """
        Create SHAP summary plot.
        
        Args:
            X: Dataset to explain
            n_samples: Number of samples to use
            class_index: Index of the class to explain
            plot_type: Type of plot ('bar', 'beeswarm', or 'violin')
            max_display: Maximum number of features to display
            output_path: Path to save the plot (None for display)
            title: Plot title
            
        Returns:
            fig: The created figure (None if display only)
        """
        # Sample data if requested
        if n_samples is not None and n_samples < X.shape[0]:
            indices = np.random.choice(X.shape[0], n_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Get SHAP values
        shap_values, _ = self.explain_dataset(X_sample, class_index=class_index)
        
        # Feature names for plot
        feature_names = self.feature_names
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Add title if provided
        if title:
            plt.title(title, fontsize=14)
        
        if plot_type == 'bar':
            shap.summary_plot(
                shap_values, X_sample, 
                feature_names=feature_names,
                plot_type='bar', 
                max_display=max_display,
                show=False
            )
        else:  # beeswarm or violin
            shap.summary_plot(
                shap_values, X_sample, 
                feature_names=feature_names,
                plot_type=plot_type, 
                max_display=max_display,
                show=False
            )
        
        # Save or display
        if output_path:
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return None
        else:
            fig = plt.gcf()
            return fig
    
    def plot_dependence(self, X, feature_idx, interaction_idx=None, class_index=1, 
                        n_samples=100, output_path=None, title=None):
        """
        Create SHAP dependence plot for a feature.
        
        Args:
            X: Dataset to explain
            feature_idx: Index or name of the feature to plot
            interaction_idx: Index or name of the interaction feature
            class_index: Index of the class to explain
            n_samples: Number of samples to use
            output_path: Path to save the plot (None for display)
            title: Plot title
            
        Returns:
            fig: The created figure (None if display only)
        """
        # Sample data if requested
        if n_samples is not None and n_samples < X.shape[0]:
            indices = np.random.choice(X.shape[0], n_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Get SHAP values
        shap_values, _ = self.explain_dataset(X_sample, class_index=class_index)
        
        # Feature names for plot
        feature_names = self.feature_names
        
        # Convert feature_idx and interaction_idx to names if they are indices
        if isinstance(feature_idx, int) and feature_names is not None:
            feature_name = feature_names[feature_idx]
        else:
            feature_name = feature_idx
            if feature_names is not None and feature_name in feature_names:
                feature_idx = feature_names.index(feature_name)
        
        if interaction_idx is not None:
            if isinstance(interaction_idx, int) and feature_names is not None:
                interaction_name = feature_names[interaction_idx]
            else:
                interaction_name = interaction_idx
                if feature_names is not None and interaction_name in feature_names:
                    interaction_idx = feature_names.index(interaction_name)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Add title if provided
        if title:
            plt.title(title, fontsize=14)
        
        # Create dependence plot
        shap.dependence_plot(
            feature_idx, 
            shap_values, 
            X_sample,
            interaction_index=interaction_idx,
            feature_names=feature_names,
            show=False
        )
        
        # Save or display
        if output_path:
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return None
        else:
            fig = plt.gcf()
            return fig
    
    def plot_force(self, instance, class_index=1, output_path=None, matplotlib=False):
        """
        Create SHAP force plot for an instance.
        
        Args:
            instance: The instance to explain
            class_index: Index of the class to explain
            output_path: Path to save the plot (None for display)
            matplotlib: Whether to use matplotlib (True) or JS (False)
            
        Returns:
            force_plot: The created force plot
        """
        # Reshape instance if necessary
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        # Get SHAP values
        shap_values, base_value = self.explain_instance(instance, class_index=class_index)
        
        # Feature names for plot
        feature_names = self.feature_names
        
        if matplotlib:
            # Create figure using matplotlib
            plt.figure(figsize=(12, 3))
            shap.force_plot(
                base_value, 
                shap_values[0], 
                instance[0],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            
            # Save or display
            if output_path:
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                return None
            else:
                fig = plt.gcf()
                return fig
        else:
            # Create force plot using JS (for notebooks)
            force_plot = shap.force_plot(
                base_value, 
                shap_values[0], 
                instance[0],
                feature_names=feature_names
            )
            
            # Save if requested
            if output_path:
                shap.save_html(output_path, force_plot)
            
            return force_plot


class LimeExplainer:
    """
    Class for LIME (Local Interpretable Model-agnostic Explanations).
    """
    
    def __init__(self, model, X_train, training_labels=None, feature_names=None, 
                 categorical_features=None, categorical_names=None, class_names=None,
                 discretize_continuous=True, mode='classification'):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained model
            X_train: Training data
            training_labels: Training labels (optional)
            feature_names: List of feature names
            categorical_features: List of indices of categorical features
            categorical_names: Dict mapping categorical feature indices to their names
            class_names: List of class names
            discretize_continuous: Whether to discretize continuous features
            mode: 'classification' or 'regression'
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.mode = mode
        
        if categorical_features is None:
            categorical_features = []
        
        if self.mode == 'classification':
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train,
                training_labels=training_labels,
                feature_names=feature_names,
                categorical_features=categorical_features,
                categorical_names=categorical_names,
                class_names=class_names,
                discretize_continuous=discretize_continuous,
                mode=mode
            )
        else:  # regression
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train,
                training_labels=training_labels,
                feature_names=feature_names,
                categorical_features=categorical_features,
                categorical_names=categorical_names,
                discretize_continuous=discretize_continuous,
                mode=mode
            )
    
    def explain_instance(self, instance, num_features=10, num_samples=5000, 
                         labels=(1,), top_labels=None):
        """
        Generate LIME explanation for a single instance.
        
        Args:
            instance: The instance to explain
            num_features: Maximum number of features to include in explanation
            num_samples: Number of samples to generate
            labels: List of labels/classes to explain
            top_labels: Instead of specifying labels, explain top K labels
            
        Returns:
            explanation: LIME explanation object
        """
        if isinstance(self.model, Pipeline):
            # For sklearn pipelines, use the predict_proba of the final estimator
            predict_fn = self.model.predict_proba
        else:
            # For regular models, use the predict_proba method
            predict_fn = self.model.predict_proba
        
        if self.mode == 'classification':
            explanation = self.explainer.explain_instance(
                instance, 
                predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                labels=labels,
                top_labels=top_labels
            )
        else:  # regression
            explanation = self.explainer.explain_instance(
                instance, 
                self.model.predict,
                num_features=num_features,
                num_samples=num_samples
            )
        
        return explanation
    
    def plot_explanation(self, explanation, label=1, output_path=None):
        """
        Plot LIME explanation.
        
        Args:
            explanation: LIME explanation object
            label: Class label to explain
            output_path: Path to save the plot (None for display)
            
        Returns:
            fig: The created figure (None if display only)
        """
        # Create figure
        if self.mode == 'classification':
            fig = explanation.as_pyplot_figure(label=label)
        else:  # regression
            fig = explanation.as_pyplot_figure()
        
        # Save or display
        if output_path:
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return None
        else:
            return fig
    
    def get_explanation_as_dataframe(self, explanation, label=1):
        """
        Get LIME explanation as a DataFrame.
        
        Args:
            explanation: LIME explanation object
            label: Class label to explain
            
        Returns:
            pd.DataFrame: DataFrame with explanation
        """
        if self.mode == 'classification':
            # Extract the explanation as a list of tuples
            feature_weight_tuples = explanation.as_list(label=label)
        else:  # regression
            # Extract the explanation as a list of tuples
            feature_weight_tuples = explanation.as_list()
        
        # Convert to DataFrame
        df = pd.DataFrame(feature_weight_tuples, columns=['Feature', 'Weight'])
        
        # Sort by absolute weight (importance)
        df['AbsWeight'] = df['Weight'].abs()
        df = df.sort_values('AbsWeight', ascending=False).drop('AbsWeight', axis=1)
        
        return df
    
    def get_explanation_as_html(self, explanation, label=1):
        """
        Get LIME explanation as HTML.
        
        Args:
            explanation: LIME explanation object
            label: Class label to explain
            
        Returns:
            str: HTML representation of the explanation
        """
        if self.mode == 'classification':
            html = explanation.as_html(label=label)
        else:  # regression
            html = explanation.as_html()
        
        return html
    
    def save_explanation_as_html(self, explanation, output_path, label=1):
        """
        Save LIME explanation as HTML file.
        
        Args:
            explanation: LIME explanation object
            output_path: Path to save the HTML file
            label: Class label to explain
        """
        html = self.get_explanation_as_html(explanation, label=label)
        
        with open(output_path, 'w') as f:
            f.write(html) 