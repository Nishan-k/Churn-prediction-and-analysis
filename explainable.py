import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

def global_shap_summary(model_path, X_train_path, sample_size=1000, random_state=42):
    """
    Compute and plot global SHAP feature importance.
    
    Args:
        model_path (str): Path to the saved model (.pkl).
        X_train_path (str): Path to the training data (CSV).
        sample_size (int): Subsample size for SHAP computation (faster results).
        random_state (int): Random seed for reproducibility.
    """
    # Load model and data
    model = joblib.load(model_path)
    X_train = pd.read_csv(X_train_path)
    
    # Sample data for efficiency
    if sample_size < len(X_train):
        X_train_sampled = X_train.sample(sample_size, random_state=random_state)
    else:
        X_train_sampled = X_train
    
    # Convert DataFrame to NumPy array
    X_train_sampled_np = X_train_sampled.to_numpy()

    # Define a wrapper to ensure predict_proba works with NumPy arrays
    def predict_proba_wrapper(X):
        return model.predict_proba(pd.DataFrame(X, columns=X_train_sampled.columns))

    # Use a small subset of X_train as background data
    background = shap.sample(X_train_sampled_np, 50)

    # Initialize KernelExplainer with the wrapped function
    explainer = shap.KernelExplainer(predict_proba_wrapper, background)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_train_sampled_np)
    
    # Plot global feature importance
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[1], X_train_sampled_np, feature_names=X_train_sampled.columns, plot_type='bar', show=False)
    plt.title("Global Feature Importance (SHAP)", fontsize=14)
    plt.tight_layout()
    plt.show()
