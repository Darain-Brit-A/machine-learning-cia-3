"""
SHAP Explainability Module

This module provides:
- Global feature importance analysis
- Individual prediction explanations
- SHAP visualization
- Integration with trained ML models
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """SHAP-based explainability for ML models."""
    
    def __init__(self, model, X_train, feature_names):
        """
        Initialize SHAP explainer.
        
        Parameters:
        -----------
        model : sklearn model
            Trained ML model
        X_train : array-like
            Training features (for SHAP reference)
        feature_names : list
            Feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def initialize_explainer(self, max_samples=100):
        """
        Initialize SHAP TreeExplainer or KernelExplainer.
        
        Parameters:
        -----------
        max_samples : int
            Maximum samples to use for SHAP background data
        """
        print("Initializing SHAP explainer...")
        
        # Use subset of training data for SHAP background
        if len(self.X_train) > max_samples:
            background_indices = np.random.choice(
                len(self.X_train), max_samples, replace=False
            )
            background_data = self.X_train[background_indices]
        else:
            background_data = self.X_train
        
        # Try to use TreeExplainer for tree-based models
        try:
            self.explainer = shap.TreeExplainer(self.model)
            print("  ✓ Using TreeExplainer")
        except:
            # Fall back to KernelExplainer for other models
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                shap.sample(background_data, min(50, len(background_data)))
            )
            print("  ✓ Using KernelExplainer")
        
        return self
    
    def explain_prediction(self, X_sample, return_dict=True):
        """
        Explain a single prediction or small batch.
        
        Parameters:
        -----------
        X_sample : array-like
            Input features (1D or 2D)
        return_dict : bool
            Return as dictionary (for API) or SHAP values
            
        Returns:
        --------
        explanation : dict or array
            Feature contributions or SHAP values
        """
        if self.explainer is None:
            self.initialize_explainer()
        
        # Ensure 2D input
        if len(X_sample.shape) == 1:
            X_sample = X_sample.reshape(1, -1)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_sample)
        
        # Handle multi-class output (use positive class)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 (higher risk)
        
        if return_dict:
            # For API/dashboard
            explanation = self._format_explanation(X_sample[0], shap_values[0])
            return explanation
        else:
            return shap_values
    
    def _format_explanation(self, features, shap_values):
        """
        Format SHAP values as dictionary.
        
        Parameters:
        -----------
        features : array-like
            Input feature values
        shap_values : array-like
            SHAP values for each feature
            
        Returns:
        --------
        dict : Formatted explanation
        """
        # Create feature importance dataframe
        importance_data = []
        for i, (feature, shap_val) in enumerate(zip(self.feature_names, shap_values)):
            importance_data.append({
                'feature': feature,
                'value': float(features[i]),
                'shap_value': float(shap_val),
                'contribution': 'positive' if shap_val > 0 else 'negative',
                'magnitude': float(abs(shap_val))
            })
        
        # Sort by magnitude
        importance_data.sort(key=lambda x: x['magnitude'], reverse=True)
        
        # Get top contributors
        top_positive = [x for x in importance_data if x['contribution'] == 'positive'][:3]
        top_negative = [x for x in importance_data if x['contribution'] == 'negative'][:3]
        
        explanation = {
            'all_features': importance_data,
            'top_positive_contributors': top_positive,
            'top_negative_contributors': top_negative,
            'base_value': float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.0
        }
        
        return explanation
    
    def global_feature_importance(self, X_data):
        """
        Calculate global feature importance using SHAP.
        
        Parameters:
        -----------
        X_data : array-like
            Dataset to explain
            
        Returns:
        --------
        importance_df : DataFrame
            Features ranked by importance
        """
        print("Computing global feature importance...")
        
        if self.explainer is None:
            self.initialize_explainer()
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_data)
        
        # Handle multi-class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': mean_abs_shap
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def plot_summary(self, X_data, plot_type='bar', max_display=15):
        """
        Create SHAP summary plot.
        
        Parameters:
        -----------
        X_data : array-like
            Dataset to explain
        plot_type : str
            'bar' or 'beeswarm'
        max_display : int
            Number of features to display
        """
        if self.explainer is None:
            self.initialize_explainer()
        
        print(f"Creating SHAP {plot_type} plot...")
        
        shap_values = self.explainer.shap_values(X_data)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        if plot_type == 'bar':
            shap.summary_plot(
                shap_values, 
                X_data, 
                feature_names=self.feature_names,
                plot_type='bar',
                max_display=max_display,
                show=False
            )
        else:
            shap.summary_plot(
                shap_values,
                X_data,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
        
        plt.tight_layout()
        return plt
    
    def save_explanation(self, explanation, filepath):
        """Save explanation as JSON."""
        with open(filepath, 'w') as f:
            json.dump(explanation, f, indent=2, default=str)
        print(f"  ✓ Saved explanation to {filepath}")


def load_shap_explainer(model_dir='models'):
    """Load trained model and initialize SHAP explainer."""
    
    # Load model and scaler
    model_path = Path(model_dir) / 'random_forest.pkl'
    scaler_path = Path(model_dir) / 'scaler.pkl'
    features_path = Path(model_dir) / 'feature_names.json'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(features_path, 'r') as f:
        feature_names = json.load(f)
    
    # Load training data for SHAP reference
    train_df = pd.read_csv('data/heart_iot_synthetic.csv')
    exclude_cols = ['patient_id', 'timestamp', 'label']
    X_train = train_df[[col for col in train_df.columns if col not in exclude_cols]].fillna(train_df.median())
    X_train = scaler.transform(X_train)
    
    # Create explainer
    explainer = SHAPExplainer(model, X_train, feature_names)
    explainer.initialize_explainer()
    
    return explainer


def demo_shap_explanation():
    """Demonstrate SHAP explainability."""
    
    print("="*80)
    print("SHAP EXPLAINABILITY DEMO")
    print("="*80)
    
    try:
        explainer = load_shap_explainer('models')
        
        # Load test data
        df = pd.read_csv('data/heart_iot_synthetic.csv')
        exclude_cols = ['patient_id', 'timestamp', 'label']
        X = df[[col for col in df.columns if col not in exclude_cols]].fillna(df.median())
        
        import joblib
        scaler = joblib.load('models/scaler.pkl')
        X_scaled = scaler.transform(X)
        
        # Explain first prediction
        print("\n[1] Explaining individual prediction...")
        explanation = explainer.explain_prediction(X_scaled[0])
        
        print("\nTop Positive Contributors (increase risk):")
        for contrib in explanation['top_positive_contributors'][:3]:
            print(f"  {contrib['feature']}: {contrib['shap_value']:.4f}")
        
        print("\nTop Negative Contributors (decrease risk):")
        for contrib in explanation['top_negative_contributors'][:3]:
            print(f"  {contrib['feature']}: {contrib['shap_value']:.4f}")
        
        # Global importance
        print("\n[2] Computing global feature importance...")
        importance = explainer.global_feature_importance(X_scaled[:1000])
        
        print("\nTop 10 Most Important Features:")
        print(importance.head(10).to_string(index=False))
        
        print("\n" + "="*80)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train models first using ml_pipeline.py")


if __name__ == '__main__':
    demo_shap_explanation()
