import os
import sys
import shutil
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# Ensure working directory is the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Fix UnicodeEncodeError on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import ml_pipeline

def main():
    print("=" * 80)
    print("STARTING COMPARATIVE TRAINING PIPELINE")
    print("=" * 80)

    # 1. Arrange files: Move X_chi2.csv to data/ if it's in the root
    src_path = "X_chi2.csv"
    dest_dir = "data"
    dest_path = os.path.join(dest_dir, "X_chi2.csv")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    if os.path.exists(src_path):
        print(f"[1] Moving '{src_path}' to '{dest_path}'...")
        shutil.move(src_path, dest_path)
    else:
        print(f"[1] '{dest_path}' already in data directory or handled.")

    if not os.path.exists(dest_path):
        print(f"Error: {dest_path} not found! Please make sure X_chi2.csv is in the project.")
        return

    # Existing RF-selected features
    x_path = os.path.join(dest_dir, "X_rf.csv")
    y_path = os.path.join(dest_dir, "y.csv")

    if not os.path.exists(x_path):
        print(f"Error: {x_path} not found!")
        return

    # 2. Train on Both Datasets
    # We will use 20 trials for Optuna to keep it relatively fast but highly optimized.
    n_trials = 20
    
    datasets = {
        "X_chi2.csv (310 Features - Unfiltered)": dest_path,
        "X_rf.csv (310 Features - RF Filtered)": x_path
    }
    
    results_summary = {}

    for name, x_file in datasets.items():
        print("\n" + "="*50)
        print(f"TRAINING ON: {name}")
        print("="*50)
        
        # Load and split
        X_train, X_test, y_train, y_test = ml_pipeline.load_data(x_path=x_file, y_path=y_path)
        
        # Train and optimize
        print(f"    - Running Optuna optimization ({n_trials} trials)...")
        results = ml_pipeline.train_all_models(X_train, y_train, n_trials=n_trials)
        
        # Get the best model (using XGBoost as the primary model)
        xgb_pipeline, _, threshold = results["xgb"]
        
        # Predict on Test Set
        y_proba_test = xgb_pipeline.predict_proba(X_test)[:, 1]
        y_pred_test = (y_proba_test >= threshold).astype(int)
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_proba_test)
        ap_score = average_precision_score(y_test, y_proba_test)
        
        # Classification report dictionary to extract F1 and Recall
        report = classification_report(y_test, y_pred_test, output_dict=True)
        
        results_summary[name] = {
            "Accuracy": report["accuracy"],
            "F1-Macro": report["macro avg"]["f1-score"],
            "Recall-Resistant": report["1"]["recall"],
            "Precision-Resistant": report["1"]["precision"],
            "ROC-AUC": roc_auc,
            "PR-AUC": ap_score
        }
        
        print(f"    - Completed training for {name}")

    # 3. Print Comparison Report
    comparison_df = pd.DataFrame(results_summary).T
    report_text = f"""# COMPARATIVE TRAINING REPORT

## Performance Metrics on Test Set (20 Optuna Trials)

{comparison_df.round(4).to_markdown()}
"""
    
    print("\n" + "="*80)
    print("FINAL COMPARATIVE REPORT ON TEST SET")
    print("="*80)
    print(comparison_df.round(4).to_string())
    print("="*80)
    
    # Save to file
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[OK] Saved comparison report to: {report_path}")

if __name__ == "__main__":
    main()
