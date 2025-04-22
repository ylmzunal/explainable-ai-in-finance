"""
Streamlit app for interactive model explanations.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
import shap
import lime
import lime.lime_tabular

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import load_model
from src.explainers import ShapExplainer, LimeExplainer
from src.visualization import (
    plot_roc_curve, 
    plot_precision_recall_curve, 
    plot_confusion_matrix,
    plot_feature_importances
)


# Set page config
st.set_page_config(
    page_title="Explainable AI in Finance: Credit Scoring",
    page_icon="💰",
    layout="wide"
)

# Default paths
DEFAULT_MODELS_PATH = os.path.join('output', 'models')
DEFAULT_DATA_PATH = os.path.join('data', 'raw', 'credit_data.csv')


@st.cache_data
def load_data(data_path):
    """Load and cache the dataset."""
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


@st.cache_resource
def load_models_and_preprocessor(models_path, model_names=None):
    """Load and cache models and preprocessor."""
    models = {}
    
    # Default model names if not specified
    if model_names is None:
        model_names = ['logistic', 'rf', 'xgboost', 'lightgbm']
    
    # Load models
    for model_name in model_names:
        try:
            model, metadata = load_model(models_path, model_name)
            models[model_name] = {'model': model, 'metadata': metadata}
        except Exception as e:
            st.warning(f"Could not load model '{model_name}': {str(e)}")
    
    # Load preprocessor
    preprocessor = None
    try:
        preprocessor_path = os.path.join(models_path, 'preprocessor.joblib')
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
    except Exception as e:
        st.warning(f"Could not load preprocessor: {str(e)}")
    
    return models, preprocessor


def preprocess_input(input_data, preprocessor, original_data=None):
    """Preprocess input data for prediction."""
    if preprocessor is None:
        st.error("Preprocessor not available. Cannot make predictions.")
        return None
    
    try:
        # If input is a single instance, convert to DataFrame
        if not isinstance(input_data, pd.DataFrame):
            if original_data is not None:
                # Create a DataFrame with the same columns as original_data
                input_df = pd.DataFrame([input_data], columns=original_data.columns.drop('loan_default'))
            else:
                st.error("Cannot process input data: original data structure unknown")
                return None
        else:
            input_df = input_data.copy()
        
        # Apply preprocessing
        X_processed = preprocessor.transform(input_df)
        
        return X_processed
    
    except Exception as e:
        st.error(f"Error preprocessing input data: {str(e)}")
        return None


def predict_loan_default(input_data, model, preprocessor, threshold=0.5):
    """Make loan default prediction."""
    # Preprocess input data
    X_processed = preprocess_input(input_data, preprocessor)
    
    if X_processed is None:
        return None, None
    
    # Make prediction
    try:
        # Get probability of default
        probabilities = model.predict_proba(X_processed)[0]
        default_prob = probabilities[1]
        
        # Classify based on threshold
        prediction = 1 if default_prob >= threshold else 0
        
        return prediction, default_prob
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None


def generate_lime_explanation(instance, model, X_train, feature_names, class_names):
    """Generate LIME explanation for a single instance."""
    try:
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=True,
            mode='classification'
        )
        
        # Generate explanation
        exp = explainer.explain_instance(
            instance,
            model.predict_proba,
            num_features=10,
            top_labels=1
        )
        
        return exp
    
    except Exception as e:
        st.error(f"Error generating LIME explanation: {str(e)}")
        return None


def generate_shap_explanation(instance, model, X_train, feature_names):
    """Generate SHAP explanation for a single instance."""
    try:
        # Determine model type
        if hasattr(model, 'feature_importances_'):
            model_type = 'tree'
        elif hasattr(model, 'coef_'):
            model_type = 'linear'
        else:
            model_type = None
        
        # Create SHAP explainer
        explainer = ShapExplainer(model, X_train, feature_names=feature_names, model_type=model_type)
        
        # Generate explanation
        shap_values, base_value = explainer.explain_instance(instance)
        
        return shap_values, base_value, explainer
    
    except Exception as e:
        st.error(f"Error generating SHAP explanation: {str(e)}")
        return None, None, None


def main():
    """Main function for the Streamlit app."""
    # App title and description
    st.title("Explainable AI in Finance: Credit Scoring")
    st.markdown("""
    This application demonstrates how Explainable AI (XAI) can be used to interpret 
    machine learning models for credit scoring and loan approval decisions.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Model selection
        st.subheader("Model")
        default_model_options = ['logistic', 'rf', 'xgboost', 'lightgbm']
        model_name = st.selectbox(
            "Select model",
            options=default_model_options,
            index=2  # Default to xgboost
        )
        
        # Data & models paths
        st.subheader("Data and Models")
        data_path = st.text_input("Data path", value=DEFAULT_DATA_PATH)
        models_path = st.text_input("Models path", value=DEFAULT_MODELS_PATH)
        
        # Load data and models buttons
        if st.button("Load Data and Models"):
            st.session_state['reload_data'] = True
        
        # Explanation method
        st.subheader("Explanation Method")
        explanation_method = st.radio(
            "Select explanation method",
            options=["SHAP", "LIME", "Both"],
            index=0  # Default to SHAP
        )
        
        # Prediction threshold
        st.subheader("Prediction Threshold")
        threshold = st.slider(
            "Default probability threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
    
    # Initialize session state if not already done
    if 'data' not in st.session_state or 'reload_data' in st.session_state:
        st.session_state['data'] = load_data(data_path)
        st.session_state['models'], st.session_state['preprocessor'] = load_models_and_preprocessor(
            models_path, [model_name]
        )
        if 'reload_data' in st.session_state:
            del st.session_state['reload_data']
    
    # Main content
    if st.session_state['data'] is not None and model_name in st.session_state['models']:
        # Get data and model
        data = st.session_state['data']
        model = st.session_state['models'][model_name]['model']
        preprocessor = st.session_state['preprocessor']
        
        # Get feature metadata if available
        if 'metadata' in st.session_state['models'][model_name] and 'feature_names' in st.session_state['models'][model_name]['metadata']:
            feature_names = st.session_state['models'][model_name]['metadata']['feature_names']
        else:
            # If no feature names in metadata, use data columns (without target)
            feature_names = data.columns.drop('loan_default').tolist()
        
        # Preprocess training data for use in explanations
        X_train = preprocessor.transform(data.drop(columns=['loan_default']))
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Loan Application", "Model Insights", "Data Exploration"])
        
        # Tab 1: Loan Application
        with tab1:
            st.header("Loan Application Evaluation")
            st.markdown("""
            Enter loan applicant information to evaluate credit risk and 
            understand the factors influencing the decision.
            """)
            
            # Create columns for form
            col1, col2, col3 = st.columns(3)
            
            with col1:
                income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=60000, step=5000)
                age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
                employment_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5, step=1)
                debt_to_income = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
                credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700, step=10)
            
            with col2:
                num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, max_value=30, value=5, step=1)
                num_late_payments = st.number_input("Number of Late Payments", min_value=0, max_value=20, value=1, step=1)
                loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=100000, value=20000, step=1000)
                loan_term = st.selectbox("Loan Term (months)", options=[12, 24, 36, 48, 60, 72, 84, 96, 120], index=2)
                interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=30.0, value=5.0, step=0.1)
            
            with col3:
                housing_status = st.selectbox("Housing Status", options=["Own", "Mortgage", "Rent"], index=1)
                loan_purpose = st.selectbox(
                    "Loan Purpose", 
                    options=[
                        "Debt Consolidation", 
                        "Credit Card Refinancing", 
                        "Home Improvement", 
                        "Major Purchase", 
                        "Medical Expenses",
                        "Business",
                        "Auto",
                        "Other"
                    ], 
                    index=0
                )
                employment_status = st.selectbox(
                    "Employment Status", 
                    options=["Full-time", "Part-time", "Self-employed", "Retired", "Other"],
                    index=0
                )
                education = st.selectbox(
                    "Education Level", 
                    options=[
                        "High School", 
                        "Associate Degree", 
                        "Bachelor's Degree", 
                        "Master's Degree",
                        "Doctoral Degree",
                        "Other"
                    ],
                    index=2
                )
            
            # Create input data
            input_data = {
                'income': income,
                'age': age,
                'employment_length': employment_length,
                'debt_to_income': debt_to_income,
                'credit_score': credit_score,
                'num_credit_lines': num_credit_lines,
                'num_late_payments': num_late_payments,
                'loan_amount': loan_amount,
                'loan_term': loan_term,
                'interest_rate': interest_rate,
                'housing_status': housing_status,
                'loan_purpose': loan_purpose,
                'employment_status': employment_status,
                'education': education
            }
            
            # Evaluate button
            if st.button("Evaluate Application"):
                # Make prediction
                prediction, default_prob = predict_loan_default(
                    input_data, 
                    model, 
                    preprocessor, 
                    threshold=threshold
                )
                
                if prediction is not None:
                    # Display result
                    st.subheader("Loan Evaluation Result")
                    
                    # Create columns for result
                    res_col1, res_col2 = st.columns([1, 2])
                    
                    with res_col1:
                        if prediction == 0:
                            st.success("Loan Approved")
                        else:
                            st.error("Loan Denied")
                        
                        st.metric(
                            "Default Probability",
                            f"{default_prob:.2%}",
                            delta=f"{(default_prob - threshold) * 100:.1f}%",
                            delta_color="inverse"
                        )
                    
                    with res_col2:
                        st.subheader("Decision Explanation")
                        
                        # Create instance for explanation
                        instance = preprocess_input(input_data, preprocessor, data)
                        instance = instance[0]  # Get first (and only) row
                        
                        # Generate explanations based on selected method
                        if explanation_method in ["SHAP", "Both"]:
                            shap_values, base_value, shap_explainer = generate_shap_explanation(
                                instance, 
                                model, 
                                X_train, 
                                feature_names
                            )
                            
                            if shap_values is not None:
                                st.write("SHAP Explanation (feature contribution to default probability):")
                                
                                # Create matplotlib force plot
                                plt.figure(figsize=(10, 3))
                                shap.force_plot(
                                    base_value, 
                                    shap_values[0], 
                                    instance,
                                    feature_names=feature_names,
                                    matplotlib=True,
                                    show=False
                                )
                                st.pyplot(plt)
                        
                        if explanation_method in ["LIME", "Both"]:
                            lime_exp = generate_lime_explanation(
                                instance, 
                                model, 
                                X_train, 
                                feature_names, 
                                class_names=['No Default', 'Default']
                            )
                            
                            if lime_exp is not None:
                                st.write("LIME Explanation (feature importance for prediction):")
                                
                                # Create explanation plot
                                fig = lime_exp.as_pyplot_figure(label=1)  # Default class
                                st.pyplot(fig)
        
        # Tab 2: Model Insights
        with tab2:
            st.header("Model Insights")
            st.markdown("""
            Explore global model explanations to understand how different features
            influence credit approval decisions across all applicants.
            """)
            
            # Feature importance plot
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importances")
                importances = model.feature_importances_
                
                fig = plot_feature_importances(importances, feature_names)
                st.pyplot(fig)
                plt.close()
            
            # SHAP summary plot
            st.subheader("SHAP Feature Impact")
            
            # Sample data for summary plots
            n_samples = min(200, len(data))
            sample_indices = np.random.choice(len(data), n_samples, replace=False)
            sample_data = data.iloc[sample_indices]
            
            # Preprocess sample data
            X_sample = preprocessor.transform(sample_data.drop(columns=['loan_default']))
            
            # Create SHAP explainer
            if hasattr(model, 'feature_importances_'):
                model_type = 'tree'
            elif hasattr(model, 'coef_'):
                model_type = 'linear'
            else:
                model_type = None
                
            shap_explainer = ShapExplainer(model, X_train, feature_names=feature_names, model_type=model_type)
            
            # Generate and display summary plot
            fig = shap_explainer.plot_summary(X_sample, plot_type='bar')
            st.pyplot(fig)
            plt.close()
            
            fig = shap_explainer.plot_summary(X_sample, plot_type='violin')
            st.pyplot(fig)
            plt.close()
            
            # Dependence plots for top features
            st.subheader("Feature Dependency Plots")
            
            # Get SHAP values for sample data
            shap_values, _ = shap_explainer.explain_dataset(X_sample)
            
            # Get top features by mean absolute SHAP value
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[-3:]  # Top 3 features
            
            for idx in top_indices:
                feature_name = feature_names[idx]
                st.write(f"SHAP Dependence Plot - {feature_name}")
                
                fig = shap_explainer.plot_dependence(X_sample, feature_idx=idx)
                st.pyplot(fig)
                plt.close()
        
        # Tab 3: Data Exploration
        with tab3:
            st.header("Data Exploration")
            st.markdown("""
            Explore the dataset used to train the model and understand the distribution
            of different features.
            """)
            
            # Show data sample
            st.subheader("Data Sample")
            st.dataframe(data.head(10))
            
            # Feature distribution plots
            st.subheader("Feature Distributions")
            
            # Let user select a feature to visualize
            feature_to_plot = st.selectbox(
                "Select feature to visualize",
                options=data.columns.drop('loan_default'),
                index=0
            )
            
            # Check if feature is categorical or numerical
            if data[feature_to_plot].dtype == 'object':
                # Categorical feature
                fig, ax = plt.subplots(figsize=(10, 6))
                data[feature_to_plot].value_counts().plot(kind='bar', ax=ax)
                plt.title(f"Distribution of {feature_to_plot}")
                plt.ylabel("Count")
                plt.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
                # Distribution by target
                fig, ax = plt.subplots(figsize=(12, 6))
                pd.crosstab(
                    data[feature_to_plot], 
                    data['loan_default'], 
                    normalize='index'
                ).plot(kind='bar', ax=ax)
                plt.title(f"Default Rate by {feature_to_plot}")
                plt.ylabel("Default Rate")
                plt.legend(['No Default', 'Default'])
                plt.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
            else:
                # Numerical feature
                fig, ax = plt.subplots(figsize=(10, 6))
                data[feature_to_plot].hist(bins=30, ax=ax)
                plt.title(f"Distribution of {feature_to_plot}")
                plt.ylabel("Count")
                plt.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
                # Distribution by target
                fig, ax = plt.subplots(figsize=(10, 6))
                data.boxplot(column=feature_to_plot, by='loan_default', ax=ax)
                plt.title(f"{feature_to_plot} by Loan Default")
                plt.suptitle("")  # Remove default suptitle
                plt.ylabel(feature_to_plot)
                plt.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            # Correlation matrix
            st.subheader("Correlation Matrix")
            
            # Calculate correlations for numerical features
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
            corr = data[numerical_cols].corr()
            
            # Plot correlation matrix
            fig, ax = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            cmap = "coolwarm"
            
            sns_heatmap = sns.heatmap(
                corr, 
                mask=mask,
                cmap=cmap,
                annot=True,
                fmt='.2f',
                square=True,
                linewidths=0.5,
                cbar_kws={'shrink': 0.8},
                ax=ax
            )
            plt.title("Correlation Matrix of Numerical Features")
            st.pyplot(fig)
            plt.close()
            
    else:
        st.warning("Please load the data and models using the button in the sidebar.")


if __name__ == "__main__":
    # Check for streamlit
    try:
        import streamlit as st
        import seaborn as sns
    except ImportError:
        print("Error: streamlit or seaborn package not found.")
        print("Please install it using: pip install streamlit seaborn")
        sys.exit(1)
    
    main() 