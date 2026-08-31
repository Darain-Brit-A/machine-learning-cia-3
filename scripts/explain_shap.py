"""
Step 1.3: SHAP Explainability

Generate SHAP explanations for model predictions - both global and local interpretability.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


def load_trained_model(model_name):
    """Load best trained model."""
    print(f"Loading trained model: {model_name}...")
    
    models_dir = Path('models')
    model_path = models_dir / f'{model_name.lower().replace(" ", "_")}.pkl'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    print(f"✓ Model loaded: {model_path}")
    
    return model


def load_test_data():
    """Load preprocessed test data."""
    preprocess_dir = Path('data/preprocessed')
    
    X_test = pd.read_csv(preprocess_dir / 'X_test.csv')
    y_test = pd.read_csv(preprocess_dir / 'y_test.csv').squeeze()
    
    with open(preprocess_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"✓ Loaded test data: {X_test.shape}")
    
    return X_test, y_test, metadata


def generate_shap_explainer(model, X_test, model_name):
    """Generate SHAP explainer."""
    print(f"\nGenerating SHAP explainer for {model_name}...")
    
    # Create SHAP explainer based on model type
    if 'XGBoost' in model_name or 'CatBoost' in model_name:
        print("  Using TreeExplainer (optimized for tree-based models)")
        explainer = shap.TreeExplainer(model)
    elif 'Random Forest' in model_name:
        print("  Using TreeExplainer (optimized for ensemble models)")
        explainer = shap.TreeExplainer(model)
    else:
        print("  Using KernelExplainer (model-agnostic)")
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1],
            shap.sample(X_test, min(100, len(X_test))),
            link="logit"
        )
    
    print("  ✓ Explainer created")
    return explainer


def compute_shap_values(explainer, X_test, model, model_name):
    """Compute SHAP values for test set."""
    print(f"\nComputing SHAP values for {len(X_test)} test samples...")
    
    # Get SHAP values
    if 'XGBoost' in model_name or 'Random Forest' in model_name or 'CatBoost' in model_name:
        shap_values = explainer.shap_values(X_test)
        # For binary classification, shap_values might be a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use class 1 (high risk)
    else:
        shap_values = explainer.shap_values(X_test)
    
    print(f"✓ SHAP values computed: shape {shap_values.shape}")
    
    return shap_values


def global_feature_importance(shap_values, X_test, metadata):
    """
    Compute global feature importance using mean absolute SHAP values.
    Shows which features are most important overall.
    """
    print("\nComputing global feature importance...")
    
    # Mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': mean_abs_shap,
        'importance_pct': mean_abs_shap / mean_abs_shap.sum() * 100
    }).sort_values('importance', ascending=False)
    
    print("\nGlobal Feature Importance (by mean |SHAP value|):\n")
    for idx, row in importance_df.iterrows():
        print(f"  {row['feature']:<25} {row['importance']:.4f} ({row['importance_pct']:>6.2f}%)")
    
    return importance_df


def plot_shap_summary(shap_values, X_test, model_name):
    """Plot SHAP summary plots."""
    print("\nGenerating SHAP summary plots...")
    
    # Summary plot (Bar) - feature importance
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f'SHAP Summary Plot (Bar) - {model_name}', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'models/shap_summary_bar_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: shap_summary_bar_{model_name.lower().replace(' ', '_')}.png")
    plt.close()
    
    # Summary plot (Dot) - feature values vs SHAP impact
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f'SHAP Summary Plot (Dot) - {model_name}', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'models/shap_summary_dot_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: shap_summary_dot_{model_name.lower().replace(' ', '_')}.png")
    plt.close()


def plot_shap_dependence(shap_values, X_test, importance_df, model_name):
    """Plot SHAP dependence plots for top features."""
    print("Generating SHAP dependence plots for top 3 features...")
    
    top_features = importance_df.head(3)['feature'].tolist()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    for idx, feature in enumerate(top_features):
        feature_idx = list(X_test.columns).index(feature)
        
        ax = axes[idx]
        scatter = ax.scatter(X_test.iloc[:, feature_idx], shap_values[:, feature_idx], 
                            c=shap_values[:, feature_idx], cmap='coolwarm', alpha=0.6, s=30)
        ax.set_xlabel(feature, fontweight='bold')
        ax.set_ylabel('SHAP Value', fontweight='bold')
        ax.set_title(f'Dependence: {feature}', fontweight='bold')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='SHAP Value')
    
    plt.tight_layout()
    plt.savefig(f'models/shap_dependence_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: shap_dependence_{model_name.lower().replace(' ', '_')}.png")
    plt.close()


def explain_individual_predictions(shap_values, X_test, y_test, model, 
                                  importance_df, model_name):
    """Generate explanations for individual predictions."""
    print("\nGenerating individual prediction explanations...")
    
    explanations = []
    
    # Select diverse samples: high confidence correct, high confidence wrong, low confidence
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # High confidence correct (true positive)
    high_conf_correct = np.where(
        (y_pred == y_test) & (y_pred == 1) & (y_pred_proba > 0.8)
    )[0]
    
    # High confidence wrong (false positive)
    high_conf_wrong = np.where(
        (y_pred != y_test) & (y_pred == 1) & (y_pred_proba > 0.8)
    )[0]
    
    # Uncertain predictions (near 0.5)
    uncertain = np.where((y_pred_proba > 0.4) & (y_pred_proba < 0.6))[0]
    
    # Get indices
    sample_indices = []
    if len(high_conf_correct) > 0:
        sample_indices.append(high_conf_correct[0])
    if len(high_conf_wrong) > 0:
        sample_indices.append(high_conf_wrong[0])
    if len(uncertain) > 0:
        sample_indices.append(uncertain[0])
    
    # Limit to 3 examples
    sample_indices = sample_indices[:3]
    
    for idx, sample_idx in enumerate(sample_indices, 1):
        # Get prediction
        pred = y_pred[sample_idx]
        pred_proba = y_pred_proba[sample_idx]
        actual = y_test.iloc[sample_idx]
        
        # Get top 3 features contributing to this prediction
        top_features_idx = np.argsort(np.abs(shap_values[sample_idx, :]))[-3:][::-1]
        
        explanation = {
            'example_id': idx,
            'sample_index': int(sample_idx),
            'prediction': int(pred),
            'prediction_label': 'HIGH RISK' if pred == 1 else 'LOW RISK',
            'probability': float(pred_proba),
            'confidence': 'HIGH' if pred_proba > 0.7 else 'MEDIUM' if pred_proba > 0.5 else 'LOW',
            'actual': int(actual),
            'correct': bool(pred == actual),
            'top_contributing_features': []
        }
        
        for rank, feat_idx in enumerate(top_features_idx, 1):
            feat_name = X_test.columns[feat_idx]
            feat_value = X_test.iloc[sample_idx, feat_idx]
            shap_value = shap_values[sample_idx, feat_idx]
            
            direction = 'increases' if shap_value > 0 else 'decreases'
            
            explanation['top_contributing_features'].append({
                'rank': rank,
                'feature': feat_name,
                'value': float(feat_value),
                'shap_contribution': float(shap_value),
                'direction': direction,
                'impact_magnitude': float(abs(shap_value))
            })
        
        explanations.append(explanation)
    
    return explanations


def save_shap_results(importance_df, explanations, model_name):
    """Save SHAP results."""
    print("\nSaving SHAP results...")
    
    models_dir = Path('models')
    
    # Save feature importance
    importance_df.to_csv(
        models_dir / f'shap_feature_importance_{model_name.lower().replace(" ", "_")}.csv',
        index=False
    )
    print(f"  ✓ Saved feature importance CSV")
    
    # Save explanations
    with open(
        models_dir / f'shap_explanations_{model_name.lower().replace(" ", "_")}.json', 'w'
    ) as f:
        json.dump(explanations, f, indent=2)
    print(f"  ✓ Saved individual explanations JSON")
    
    # Save summary
    summary = {
        'model': model_name,
        'feature_importance_top_5': importance_df.head(5).to_dict('records'),
        'example_explanations': explanations,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(
        models_dir / f'shap_summary_{model_name.lower().replace(" ", "_")}.json', 'w'
    ) as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved SHAP summary JSON")


def main():
    """Main SHAP explainability pipeline."""
    
    print("=" * 80)
    print("PHASE 1, STEP 1.3: SHAP EXPLAINABILITY")
    print("=" * 80)
    
    # Load best model info
    models_dir = Path('models')
    with open(models_dir / 'best_model_info.json', 'r') as f:
        best_model_info = json.load(f)
    
    best_model_name = best_model_info['best_model']
    print(f"\nUsing best model: {best_model_name}")
    
    # Load model and data
    model = load_trained_model(best_model_name)
    X_test, y_test, metadata = load_test_data()
    
    # Generate SHAP explainer
    explainer = generate_shap_explainer(model, X_test, best_model_name)
    
    # Compute SHAP values
    shap_values = compute_shap_values(explainer, X_test, model, best_model_name)
    
    # Global feature importance
    print("\n" + "-" * 80)
    print("GLOBAL FEATURE IMPORTANCE")
    print("-" * 80)
    importance_df = global_feature_importance(shap_values, X_test, metadata)
    
    # Generate visualizations
    print("\n" + "-" * 80)
    print("GENERATING VISUALIZATIONS")
    print("-" * 80)
    plot_shap_summary(shap_values, X_test, best_model_name)
    plot_shap_dependence(shap_values, X_test, importance_df, best_model_name)
    
    # Generate individual explanations
    print("\n" + "-" * 80)
    print("INDIVIDUAL PREDICTION EXPLANATIONS")
    print("-" * 80)
    explanations = explain_individual_predictions(
        shap_values, X_test, y_test, model, importance_df, best_model_name
    )
    
    for exp in explanations:
        print(f"\nExample {exp['example_id']}:")
        print(f"  Prediction: {exp['prediction_label']} (Prob: {exp['probability']:.2%})")
        print(f"  Actual: {'HIGH RISK' if exp['actual'] == 1 else 'LOW RISK'}")
        print(f"  Correct: {'✓' if exp['correct'] else '✗'}")
        print(f"  Top Contributing Features:")
        for feat in exp['top_contributing_features']:
            print(f"    {feat['rank']}. {feat['feature']}: {feat['value']:.2f} "
                  f"→ {feat['direction']} risk (SHAP: {feat['shap_contribution']:.4f})")
    
    # Save results
    save_shap_results(importance_df, explanations, best_model_name)
    
    print("\n" + "=" * 80)
    print("SHAP EXPLAINABILITY COMPLETE")
    print("=" * 80)
    print(f"\nNext step: Validate model performance")
    print(f"Run: python scripts/validate_model.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
