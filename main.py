"""
Main script to run the explainable AI credit scoring pipeline.
"""

import os
import argparse
import subprocess


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Explainable AI Credit Scoring Pipeline')
    
    parser.add_argument('--generate-data', action='store_true',
                        help='Generate synthetic credit data')
    parser.add_argument('--train-models', action='store_true',
                        help='Train and evaluate credit scoring models')
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize model hyperparameters')
    parser.add_argument('--run-app', action='store_true',
                        help='Run the Streamlit web application')
    parser.add_argument('--all', action='store_true',
                        help='Run the complete pipeline')
    
    # Data generation parameters
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Number of samples for synthetic data generation')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Model training parameters
    parser.add_argument('--models', type=str, default='logistic,rf',
                        help='Comma-separated list of models to train')
    parser.add_argument('--data-path', type=str, default='data/raw',
                        help='Path to the data directory')
    parser.add_argument('--data-file', type=str, default='credit_data.csv',
                        help='Name of the data file')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save outputs')
    
    return parser.parse_args()


def generate_data(args):
    """Generate synthetic credit data."""
    print("Generating synthetic credit data...")
    
    cmd = [
        'python', 'src/generate_data.py',
        '--n-samples', str(args.n_samples),
        '--random-seed', str(args.random_seed)
    ]
    
    subprocess.run(cmd, check=True)
    
    print("Data generation complete.")


def train_models(args):
    """Train and evaluate credit scoring models."""
    print("Training and evaluating credit scoring models...")
    
    cmd = [
        'python', 'src/train_and_explain.py',
        '--data-path', args.data_path,
        '--data-file', args.data_file,
        '--output-dir', args.output_dir,
        '--models', 'logistic,rf',
        '--save-models',
        '--save-processed'
    ]
    
    if args.optimize:
        cmd.append('--optimize')
    
    subprocess.run(cmd, check=True)
    
    print("Model training and evaluation complete.")


def run_app():
    """Run the Streamlit web application."""
    print("Launching Streamlit application...")
    
    cmd = ['streamlit', 'run', 'app/app.py']
    
    subprocess.run(cmd, check=True)


def main():
    """Main function to run the pipeline."""
    args = parse_args()
    
    # Create required directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Run the selected steps or the complete pipeline
    if args.all or args.generate_data:
        generate_data(args)
    
    if args.all or args.train_models:
        train_models(args)
    
    if args.all or args.run_app:
        run_app()


if __name__ == "__main__":
    main() 