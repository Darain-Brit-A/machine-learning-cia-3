"""
ML Pipeline: Preprocessing, Training, and Model Selection

This module handles:
- Data preprocessing (missing values, scaling, feature engineering)
- Patient-level train/test split
- Model training (Logistic Regression, Random Forest, XGBoost, CatBoost)
- Model evaluation and comparison
- Model serialization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class MLPipeline:
    """Complete ML pipeline for heart disease prediction."""
    
    def __init__(self, data_path='data/heart_iot_synthetic.csv'):
        """Initialize pipeline."""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_patient_ids = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.feature_names = None
        self.results = {}
        
    def load_data(self):
        """Load dataset."""
        print("[1/8] Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"  ✓ Loaded {len(self.df):,} rows, {len(self.df.columns)} columns")
        return self
    
    def preprocess_data(self):
        """Preprocess the dataset."""
        print("[2/8] Preprocessing data...")
        
        # Select features (exclude patient_id, timestamp, label)
        exclude_cols = ['patient_id', 'timestamp', 'label']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        X = self.df[feature_cols].copy()
        y = self.df['label'].copy()
        patient_ids = self.df['patient_id'].copy()
        
        self.feature_names = feature_cols
        
        # Handle missing values
        print(f"  - Handling missing values...")
        X = X.fillna(X.median())
        
        # Detect and handle outliers using IQR
        print(f"  - Detecting outliers...")
        for col in X.select_dtypes(include=[np.number]).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X[col] = X[col].clip(lower_bound, upper_bound)
        
        print(f"  ✓ Preprocessing complete")
        return X, y, patient_ids
    
    def patient_level_split(self, X, y, patient_ids, test_size=0.2, random_state=42):
        """
        Split data by patient, not random rows.
        This prevents data leakage when repeated measurements exist.
        """
        print("[3/8] Performing patient-level train/test split...")
        
        unique_patients = patient_ids.unique()
        np.random.seed(random_state)
        np.random.shuffle(unique_patients)
        
        split_point = int(len(unique_patients) * (1 - test_size))
        train_patients = unique_patients[:split_point]
        test_patients = unique_patients[split_point:]
        
        train_idx = patient_ids.isin(train_patients)
        test_idx = patient_ids.isin(test_patients)
        
        self.X_train = X[train_idx].copy()
        self.X_test = X[test_idx].copy()
        self.y_train = y[train_idx].copy()
        self.y_test = y[test_idx].copy()
        self.test_patient_ids = patient_ids[test_idx].values
        
        print(f"  - Train patients: {len(train_patients)} ({len(self.X_train):,} samples)")
        print(f"  - Test patients: {len(test_patients)} ({len(self.X_test):,} samples)")
        print(f"  ✓ Split complete")
        
        return self
    
    def scale_features(self):
        """Scale features using StandardScaler."""
        print("[4/8] Scaling features...")
        
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"  ✓ Features scaled")
        return self
    
    def train_models(self):
        """Train multiple candidate models."""
        print("[5/8] Training models...")
        
        # Logistic Regression
        print("  - Training Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr
        
        # Random Forest
        print("  - Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        
        # XGBoost
        print("  - Training XGBoost...")
        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = xgb_model
        
        # CatBoost
        print("  - Training CatBoost...")
        cb = CatBoostClassifier(iterations=100, random_state=42, verbose=0)
        cb.fit(self.X_train, self.y_train)
        self.models['CatBoost'] = cb
        
        print(f"  ✓ {len(self.models)} models trained")
        return self
    
    def evaluate_models(self):
        """Evaluate all models and compare performance."""
        print("[6/8] Evaluating models...")
        
        results = []
        
        for model_name, model in self.models.items():
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, zero_division=0)
            rec = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            results.append({
                'Model': model_name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1,
                'ROC-AUC': roc_auc
            })
            
            print(f"  {model_name}:")
            print(f"    Accuracy:  {acc:.4f}")
            print(f"    Precision: {prec:.4f}")
            print(f"    Recall:    {rec:.4f}")
            print(f"    F1-Score:  {f1:.4f}")
            print(f"    ROC-AUC:   {roc_auc:.4f}")
        
        self.results = pd.DataFrame(results)
        print(f"\n  ✓ Model evaluation complete")
        
        return self
    
    def select_best_model(self):
        """Select best model based on ROC-AUC (balanced metric)."""
        print("[7/8] Selecting best model...")
        
        best_idx = self.results['ROC-AUC'].idxmax()
        self.best_model_name = self.results.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"  ✓ Best model: {self.best_model_name} (ROC-AUC: {self.results.loc[best_idx, 'ROC-AUC']:.4f})")
        
        return self
    
    def save_models(self, output_dir='models'):
        """Save trained model and preprocessing pipeline."""
        print("[8/8] Saving models...")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save best model
        model_path = Path(output_dir) / f'{self.best_model_name.lower().replace(" ", "_")}.pkl'
        joblib.dump(self.best_model, model_path)
        print(f"  ✓ Saved model: {model_path}")
        
        # Save scaler
        scaler_path = Path(output_dir) / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"  ✓ Saved scaler: {scaler_path}")
        
        # Save feature names
        features_path = Path(output_dir) / 'feature_names.json'
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f)
        print(f"  ✓ Saved feature names: {features_path}")
        
        # Save results
        results_path = Path(output_dir) / 'model_comparison.csv'
        self.results.to_csv(results_path, index=False)
        print(f"  ✓ Saved model comparison: {results_path}")
        
        return self
    
    def print_summary(self):
        """Print final summary."""
        print("\n" + "=" * 80)
        print("ML PIPELINE SUMMARY")
        print("=" * 80)
        print(f"\nSelected Model: {self.best_model_name}")
        print("\nModel Performance on Test Set:")
        for idx, row in self.results.iterrows():
            if row['Model'] == self.best_model_name:
                print(f"  Accuracy:  {row['Accuracy']:.4f}")
                print(f"  Precision: {row['Precision']:.4f}")
                print(f"  Recall:    {row['Recall']:.4f}")
                print(f"  F1-Score:  {row['F1-Score']:.4f}")
                print(f"  ROC-AUC:   {row['ROC-AUC']:.4f}")
        
        print(f"\nAll Model Comparison:")
        print(self.results.to_string(index=False))
        print("\n" + "=" * 80)


def main():
    """Execute the ML pipeline."""
    
    pipeline = MLPipeline('data/heart_iot_synthetic.csv')
    
    # Execute pipeline
    X, y, patient_ids = pipeline.load_data().preprocess_data()
    pipeline.patient_level_split(X, y, patient_ids, test_size=0.2)
    pipeline.scale_features()
    pipeline.train_models()
    pipeline.evaluate_models()
    pipeline.select_best_model()
    pipeline.save_models('models')
    pipeline.print_summary()


if __name__ == '__main__':
    main()
