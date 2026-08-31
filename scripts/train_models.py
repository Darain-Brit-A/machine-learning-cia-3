"""
Step 1.2: Model Training

Train 4 different models (Logistic Regression, Random Forest, XGBoost, CatBoost)
and compare their performance.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, auc
)
import xgboost as xgb
from catboost import CatBoostClassifier

import matplotlib.pyplot as plt
import seaborn as sns


def load_preprocessed_data():
    """Load preprocessed data."""
    print("Loading preprocessed data...")
    
    preprocess_dir = Path('data/preprocessed')
    
    if not preprocess_dir.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {preprocess_dir}\n"
            "Please run: python scripts/preprocess_data.py"
        )
    
    X_train = pd.read_csv(preprocess_dir / 'X_train.csv')
    y_train = pd.read_csv(preprocess_dir / 'y_train.csv').squeeze()
    X_test = pd.read_csv(preprocess_dir / 'X_test.csv')
    y_test = pd.read_csv(preprocess_dir / 'y_test.csv').squeeze()
    
    # Load metadata
    with open(preprocess_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"✓ Loaded training data: {X_train.shape}")
    print(f"✓ Loaded test data: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, metadata


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model."""
    print("\n[1/4] Training Logistic Regression...")
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    print("  ✓ Training complete")
    
    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest model."""
    print("\n[2/4] Training Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    print("  ✓ Training complete")
    
    return model


def train_xgboost(X_train, y_train):
    """Train XGBoost model."""
    print("\n[3/4] Training XGBoost...")
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    print("  ✓ Training complete")
    
    return model


def train_catboost(X_train, y_train):
    """Train CatBoost model."""
    print("\n[4/4] Training CatBoost...")
    
    model = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=False,
        auto_class_weights='Balanced'
    )
    
    model.fit(X_train, y_train)
    print("  ✓ Training complete")
    
    return model


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance."""
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    metrics = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
        'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0
    }
    
    return metrics, y_pred, y_pred_proba, fpr, tpr, cm


def plot_model_comparison(results_list):
    """Plot comparison of all models."""
    print("\nGenerating model comparison plots...")
    
    models = [r['model_name'] for r in results_list]
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 4))
    
    for idx, metric in enumerate(metrics_to_plot):
        values = [r[metric] for r in results_list]
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        
        axes[idx].bar(models, values, color=colors, edgecolor='black', alpha=0.7)
        axes[idx].set_title(metric.replace('_', ' ').title(), fontweight='bold')
        axes[idx].set_ylabel('Score')
        axes[idx].set_ylim([0, 1])
        axes[idx].axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80% threshold')
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('models/comparison_metrics.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: models/comparison_metrics.png")
    plt.close()


def plot_confusion_matrices(results_list):
    """Plot confusion matrices for all models."""
    print("Generating confusion matrices...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, result in enumerate(results_list):
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    cbar_kws={'label': 'Count'})
        axes[idx].set_title(f"{result['model_name']}\n(Accuracy: {result['accuracy']:.3f})", 
                           fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_xticklabels(['Low Risk', 'High Risk'])
        axes[idx].set_yticklabels(['Low Risk', 'High Risk'])
    
    plt.tight_layout()
    plt.savefig('models/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: models/confusion_matrices.png")
    plt.close()


def plot_roc_curves(results_list):
    """Plot ROC curves for all models."""
    print("Generating ROC curves...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_list)))
    
    for idx, result in enumerate(results_list):
        fpr = result['fpr']
        tpr = result['tpr']
        roc_auc = result['roc_auc']
        
        ax.plot(fpr, tpr, label=f"{result['model_name']} (AUC = {roc_auc:.3f})",
                linewidth=2.5, color=colors[idx])
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier', alpha=0.5)
    
    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    ax.set_title('ROC Curves - Model Comparison', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('models/roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: models/roc_curves.png")
    plt.close()


def save_models_and_results(models_dict, results_list):
    """Save trained models and evaluation results."""
    print("\nSaving models and results...")
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Save each model
    for model_name, model in models_dict.items():
        model_path = models_dir / f'{model_name.lower().replace(" ", "_")}.pkl'
        joblib.dump(model, model_path)
        print(f"  ✓ Saved: {model_path}")
    
    # Save results
    results_df = pd.DataFrame(results_list)
    results_path = models_dir / 'model_evaluation_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"  ✓ Saved: {results_path}")
    
    # Save as JSON
    results_json = models_dir / 'model_evaluation_results.json'
    with open(results_json, 'w') as f:
        json.dump(results_list, f, indent=2)
    print(f"  ✓ Saved: {results_json}")
    
    return models_dir


def select_best_model(results_list):
    """Select best model based on ROC-AUC."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    # Sort by ROC-AUC (descending)
    sorted_results = sorted(results_list, key=lambda x: x['roc_auc'], reverse=True)
    
    print("\nRanking by ROC-AUC (primary metric):\n")
    for rank, result in enumerate(sorted_results, 1):
        print(f"{rank}. {result['model_name']:<25} ROC-AUC: {result['roc_auc']:.4f} | "
              f"Accuracy: {result['accuracy']:.4f} | Recall: {result['recall']:.4f}")
    
    best_model = sorted_results[0]
    
    print("\n" + "=" * 80)
    print("BEST MODEL SELECTED")
    print("=" * 80)
    print(f"\nModel: {best_model['model_name']}")
    print(f"ROC-AUC: {best_model['roc_auc']:.4f}")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print(f"Precision: {best_model['precision']:.4f}")
    print(f"Recall: {best_model['recall']:.4f}")
    print(f"F1-Score: {best_model['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Negatives: {best_model['true_negatives']}")
    print(f"  False Positives: {best_model['false_positives']}")
    print(f"  False Negatives: {best_model['false_negatives']}")
    print(f"  True Positives: {best_model['true_positives']}")
    print("=" * 80)
    
    return best_model['model_name']


def main():
    """Main model training pipeline."""
    
    print("=" * 80)
    print("PHASE 1, STEP 1.2: MODEL TRAINING")
    print("=" * 80)
    
    # Load data
    X_train, X_test, y_train, y_test, metadata = load_preprocessed_data()
    
    # Train all models
    print("\n" + "-" * 80)
    print("TRAINING MODELS")
    print("-" * 80)
    
    models = {}
    models['Logistic Regression'] = train_logistic_regression(X_train, y_train)
    models['Random Forest'] = train_random_forest(X_train, y_train)
    models['XGBoost'] = train_xgboost(X_train, y_train)
    models['CatBoost'] = train_catboost(X_train, y_train)
    
    # Evaluate all models
    print("\n" + "-" * 80)
    print("EVALUATING MODELS")
    print("-" * 80)
    
    results = []
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        metrics, y_pred, y_pred_proba, fpr, tpr, cm = evaluate_model(
            model, X_test, y_test, model_name
        )
        
        metrics['confusion_matrix'] = cm.tolist()
        metrics['fpr'] = fpr.tolist()
        metrics['tpr'] = tpr.tolist()
        metrics['y_pred'] = y_pred.tolist()
        metrics['y_pred_proba'] = y_pred_proba.tolist()
        
        results.append(metrics)
        
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
    
    # Save models and results
    print("\n" + "-" * 80)
    print("SAVING ARTIFACTS")
    print("-" * 80)
    
    models_dir = save_models_and_results(models, results)
    
    # Generate visualizations
    print("\n" + "-" * 80)
    print("GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    plot_model_comparison(results)
    plot_confusion_matrices(results)
    plot_roc_curves(results)
    
    # Select best model
    best_model_name = select_best_model(results)
    
    # Save best model info
    best_model_info = {
        'best_model': best_model_name,
        'training_timestamp': pd.Timestamp.now().isoformat(),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_count': len(metadata['feature_cols']),
        'features': metadata['feature_cols']
    }
    
    with open(models_dir / 'best_model_info.json', 'w') as f:
        json.dump(best_model_info, f, indent=2)
    
    print(f"\n✓ All artifacts saved to: {models_dir}")
    print(f"\nNext step: Generate SHAP explanations")
    print(f"Run: python scripts/explain_shap.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
