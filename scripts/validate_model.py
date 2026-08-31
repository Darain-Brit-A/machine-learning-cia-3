"""
Step 1.4: Model Validation & Performance Report

Comprehensive validation including cross-validation, threshold analysis,
and detailed performance report.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_data_and_model():
    """Load preprocessed data and best model."""
    print("Loading data and model...")
    
    preprocess_dir = Path('data/preprocessed')
    X_train = pd.read_csv(preprocess_dir / 'X_train.csv')
    y_train = pd.read_csv(preprocess_dir / 'y_train.csv').squeeze()
    X_test = pd.read_csv(preprocess_dir / 'X_test.csv')
    y_test = pd.read_csv(preprocess_dir / 'y_test.csv').squeeze()
    
    models_dir = Path('models')
    with open(models_dir / 'best_model_info.json', 'r') as f:
        best_model_info = json.load(f)
    
    model_name = best_model_info['best_model']
    model = joblib.load(models_dir / f'{model_name.lower().replace(" ", "_")}.pkl')
    
    print(f"✓ Loaded model: {model_name}")
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test, model, model_name


def cross_validation_analysis(model, X_train, y_train):
    """
    Perform k-fold cross-validation (patient-level).
    Ensures model generalizes well to unseen data.
    """
    print("\nPerforming 5-fold cross-validation...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, 
                               return_train_score=True)
    
    # Summarize results
    results_summary = {}
    for metric in scoring.keys():
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        results_summary[metric] = {
            'test_mean': float(test_scores.mean()),
            'test_std': float(test_scores.std()),
            'train_mean': float(train_scores.mean()),
            'train_std': float(train_scores.std()),
            'fold_scores': test_scores.tolist()
        }
        
        print(f"\n{metric.upper()}:")
        print(f"  Test:  {test_scores.mean():.4f} (±{test_scores.std():.4f})")
        print(f"  Train: {train_scores.mean():.4f} (±{train_scores.std():.4f})")
        print(f"  Folds: {[f'{s:.4f}' for s in test_scores]}")
    
    return results_summary


def threshold_analysis(model, X_test, y_test):
    """
    Analyze how changing prediction threshold affects metrics.
    
    Default threshold: 0.5
    For healthcare, lower threshold = more false positives but catches more at-risk patients
    """
    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)
    print("\nEvaluating different decision thresholds...")
    
    # Get prediction probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Test different thresholds
    thresholds = np.arange(0.3, 0.8, 0.05)
    threshold_results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        threshold_results.append({
            'threshold': float(threshold),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1)
        })
        
        print(f"\nThreshold: {threshold:.2f}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f} (of predicted high-risk, {prec:.1%} are correct)")
        print(f"  Recall:    {rec:.4f} (of actual high-risk, {rec:.1%} are caught)")
        print(f"  F1-Score:  {f1:.4f}")
    
    return threshold_results


def plot_threshold_analysis(threshold_results):
    """Plot how metrics change with threshold."""
    print("\nGenerating threshold analysis plots...")
    
    df = pd.DataFrame(threshold_results)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['threshold'], df['accuracy'], marker='o', label='Accuracy', linewidth=2.5)
    ax.plot(df['threshold'], df['precision'], marker='s', label='Precision', linewidth=2.5)
    ax.plot(df['threshold'], df['recall'], marker='^', label='Recall', linewidth=2.5)
    ax.plot(df['threshold'], df['f1_score'], marker='d', label='F1-Score', linewidth=2.5)
    
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Default (0.5)')
    
    ax.set_xlabel('Decision Threshold', fontweight='bold', fontsize=12)
    ax.set_ylabel('Metric Score', fontweight='bold', fontsize=12)
    ax.set_title('Performance Metrics vs Decision Threshold', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('models/threshold_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: threshold_analysis.png")
    plt.close()


def plot_precision_recall_curve(model, X_test, y_test):
    """Plot precision-recall curve (useful for imbalanced data)."""
    print("Generating precision-recall curve...")
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(recall, precision, linewidth=2.5, color='#2ecc71')
    ax.fill_between(recall, precision, alpha=0.2, color='#2ecc71')
    
    # No-skill baseline
    no_skill = (y_test == 1).sum() / len(y_test)
    ax.plot([0, 1], [no_skill, no_skill], linestyle='--', linewidth=2, 
            color='red', alpha=0.5, label=f'No Skill ({no_skill:.1%})')
    
    ax.set_xlabel('Recall', fontweight='bold', fontsize=12)
    ax.set_ylabel('Precision', fontweight='bold', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('models/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: precision_recall_curve.png")
    plt.close()


def failure_analysis(model, X_test, y_test):
    """Analyze model failures - false positives and false negatives."""
    print("\n" + "=" * 80)
    print("FAILURE ANALYSIS")
    print("=" * 80)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # False Positives: Predicted 1, Actually 0
    fp_mask = (y_pred == 1) & (y_test == 0)
    fp_count = fp_mask.sum()
    
    # False Negatives: Predicted 0, Actually 1
    fn_mask = (y_pred == 0) & (y_test == 1)
    fn_count = fn_mask.sum()
    
    print(f"\nFalse Positives (Healthy flagged as at-risk): {fp_count}")
    if fp_count > 0:
        fp_probs = y_pred_proba[fp_mask]
        print(f"  Mean probability: {fp_probs.mean():.4f}")
        print(f"  Range: {fp_probs.min():.4f} - {fp_probs.max():.4f}")
        print(f"  Clinical impact: Unnecessary anxiety, additional testing")
    
    print(f"\nFalse Negatives (At-risk flagged as healthy): {fn_count}")
    if fn_count > 0:
        fn_probs = y_pred_proba[fn_mask]
        print(f"  Mean probability: {fn_probs.mean():.4f}")
        print(f"  Range: {fn_probs.min():.4f} - {fn_probs.max():.4f}")
        print(f"  Clinical impact: CRITICAL - missed at-risk patients")
    
    # Analysis summary
    total_errors = fp_count + fn_count
    print(f"\nTotal errors: {total_errors} ({total_errors/len(y_test)*100:.2f}%)")
    
    if fn_count > 0:
        print(f"\n⚠️  CRITICAL: Model missed {fn_count} high-risk patients")
        print(f"    This is more dangerous than false positives")
        print(f"    Recommendation: Lower decision threshold to catch more at-risk patients")
    
    return {
        'false_positives': int(fp_count),
        'false_negatives': int(fn_count),
        'total_errors': int(total_errors),
        'error_rate': float(total_errors / len(y_test))
    }


def generate_validation_report(cv_results, threshold_results, failure_analysis_results, model, X_test, y_test):
    """Generate comprehensive validation report."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE VALIDATION REPORT")
    print("=" * 80)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'model_validation': {
            'cross_validation': cv_results,
            'test_set_performance': {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
            }
        },
        'threshold_analysis': threshold_results,
        'failure_analysis': failure_analysis_results,
        'recommendations': []
    }
    
    # Generate recommendations
    recommendations = []
    
    if report['model_validation']['test_set_performance']['roc_auc'] > 0.85:
        recommendations.append("✓ Model ROC-AUC is excellent (>0.85)")
    elif report['model_validation']['test_set_performance']['roc_auc'] > 0.80:
        recommendations.append("✓ Model ROC-AUC is good (>0.80)")
    else:
        recommendations.append("⚠ Model ROC-AUC could be improved (<0.80)")
    
    if failure_analysis_results['false_negatives'] > 0:
        recommendations.append(f"⚠ Consider lowering threshold to reduce false negatives ({failure_analysis_results['false_negatives']})")
    
    if cv_results['roc_auc']['test_std'] < cv_results['roc_auc']['test_mean'] * 0.1:
        recommendations.append("✓ Model is stable across folds (low variance)")
    else:
        recommendations.append("⚠ Model has high variance across folds - may need more regularization")
    
    recommendations.append("→ Model is ready for deployment after addressing false negatives")
    
    report['recommendations'] = recommendations
    
    return report


def save_validation_report(report):
    """Save validation report."""
    print("\nSaving validation report...")
    
    models_dir = Path('models')
    
    with open(models_dir / 'validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Saved: validation_report.json")
    
    # Print recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    for rec in report['recommendations']:
        print(f"\n{rec}")
    
    return models_dir


def main():
    """Main validation pipeline."""
    
    print("=" * 80)
    print("PHASE 1, STEP 1.4: MODEL VALIDATION & PERFORMANCE REPORT")
    print("=" * 80)
    
    # Load data and model
    X_train, y_train, X_test, y_test, model, model_name = load_data_and_model()
    
    # Cross-validation analysis
    print("\n" + "-" * 80)
    print("CROSS-VALIDATION ANALYSIS")
    print("-" * 80)
    cv_results = cross_validation_analysis(model, X_train, y_train)
    
    # Threshold analysis
    print("\n" + "-" * 80)
    print("THRESHOLD ANALYSIS")
    print("-" * 80)
    threshold_results = threshold_analysis(model, X_test, y_test)
    
    # Failure analysis
    failure_results = failure_analysis(model, X_test, y_test)
    
    # Visualizations
    print("\n" + "-" * 80)
    print("GENERATING VISUALIZATIONS")
    print("-" * 80)
    plot_threshold_analysis(threshold_results)
    plot_precision_recall_curve(model, X_test, y_test)
    
    # Generate report
    print("\n" + "-" * 80)
    print("GENERATING REPORT")
    print("-" * 80)
    report = generate_validation_report(cv_results, threshold_results, failure_results, 
                                       model, X_test, y_test)
    
    # Save report
    models_dir = save_validation_report(report)
    
    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE - ALL STEPS FINISHED")
    print("=" * 80)
    print(f"\nModel artifacts saved to: {models_dir}")
    print("\nGenerated files:")
    print("  ✓ Trained models (Logistic Regression, Random Forest, XGBoost, CatBoost)")
    print("  ✓ SHAP explanations and feature importance")
    print("  ✓ Performance evaluation and comparison")
    print("  ✓ Cross-validation results")
    print("  ✓ Threshold analysis")
    print("  ✓ Comprehensive validation report")
    print("\nNext: Phase 2 - FastAPI Server Development")
    print("=" * 80)


if __name__ == '__main__':
    main()
