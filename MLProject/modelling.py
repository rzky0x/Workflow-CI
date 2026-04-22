"""
modelling.py (for MLProject / Workflow-CI)
Model training script for the CI pipeline.

Trains a RandomForestClassifier on preprocessed Wine Quality dataset
and logs to MLflow with manual logging.

Author: Rizky
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import warnings

warnings.filterwarnings("ignore")


def load_data():
    """Load preprocessed data."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "wine_quality_preprocessing")
    
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    target_col = "quality_label"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def main():
    """Main training pipeline for CI."""
    print("="*60)
    print("WINE QUALITY - CI MODEL TRAINING")
    print("="*60)
    
    X_train, X_test, y_train, y_test = load_data()
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid,
        cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"\n[TUNING] Best Params: {best_params}")
    
    # MLflow logging
    mlflow.set_experiment("wine-quality-ci")
    
    with mlflow.start_run(run_name="CI_RandomForest_Tuned"):
        # Log parameters
        for k, v in best_params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("random_state", 42)
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        # Log metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            print(f"  {name}: {value:.4f}")
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        # Extra artifacts
        artifact_dir = "ci_artifacts"
        os.makedirs(artifact_dir, exist_ok=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Bad", "Good"],
                    yticklabels=["Bad", "Good"], ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.tight_layout()
        cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        mlflow.log_artifact(cm_path, "plots")
        
        # Feature importance
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(importances)), importances[indices], color='steelblue')
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels([X_train.columns[i] for i in indices])
        ax.set_title('Feature Importance')
        ax.invert_yaxis()
        plt.tight_layout()
        fi_path = os.path.join(artifact_dir, "feature_importance.png")
        plt.savefig(fi_path, dpi=150)
        plt.close()
        mlflow.log_artifact(fi_path, "plots")
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = os.path.join(artifact_dir, "classification_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path, "reports")
        
        # Register model
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, "wine-quality-model")
        
        print(f"\n[MLFLOW] Run completed: {run_id}")
    
    print("\n[DONE] CI Training Complete!")


if __name__ == "__main__":
    main()
