"""
Select Top 100 Most Important AA K-mers and Generate Final Datasets
====================================================================
1. Load AA k-mer matrix
2. Load Gene features and labels
3. Feature selection 1: Chi-square -> Top 100 k-mers
4. Feature selection 2: Random Forest Importance -> Top 100 k-mers
5. Combine selected k-mers with all gene features to create 2 datasets
6. Train and compare baseline models on both datasets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("SELECT TOP 100 AA K-mers (Chi2 & RF) & GENERATE DATASETS")
print("=" * 70)

# ===========================================================================
# STEP 1: Load data
# ===========================================================================
print("\n[1] Loading data...")

try:
    # Load AA k-mer features
    df_aa = pd.read_csv('kmer_features.csv')
    df_aa = df_aa.set_index('Sample') if 'Sample' in df_aa.columns else df_aa
    print(f"   - AA k-mer matrix: {df_aa.shape}")

    # Load gene features and labels
    df_gene = pd.read_csv('X_train.csv', index_col=0)
    df_y = pd.read_csv('y_train.csv')
    y_full = df_y.set_index('Genome ID')['Resistant'] if 'Genome ID' in df_y.columns else df_y.iloc[:, 1]
    
    print(f"   - Gene features: {df_gene.shape}")
    print(f"   - Labels: {len(y_full)}")
except FileNotFoundError as e:
    print(f"❌ Error loading data: {e}")
    print("Please ensure 'kmer_features.csv', 'X_train.csv', and 'y_train.csv' are in the current directory.")
    exit(1)

# ===========================================================================
# STEP 2: Align genomes
# ===========================================================================
print("\n[2] Aligning genomes...")

common_genomes = df_aa.index.intersection(df_gene.index).intersection(y_full.index)
print(f"   - Common genomes: {len(common_genomes)}")

df_aa_aligned = df_aa.loc[common_genomes]
df_gene_aligned = df_gene.loc[common_genomes]
y_aligned = y_full.loc[common_genomes]

X_aa = df_aa_aligned.values
y_vec = y_aligned.values

print(f"   - AA k-mers aligned: {df_aa_aligned.shape}")
print(f"   - Gene features aligned: {df_gene_aligned.shape}")

# ===========================================================================
# STEP 3: Feature selection - Chi-Square
# ===========================================================================
print("\n[3] Selecting top 100 AA k-mers by Chi-square...")

chi_selector = SelectKBest(score_func=chi2, k=100)
chi_selector.fit(X_aa, y_vec)

selected_mask_chi2 = chi_selector.get_support()
selected_kmers_chi2 = df_aa_aligned.columns[selected_mask_chi2]

kmer_scores_chi2 = pd.DataFrame({
    'kmer': selected_kmers_chi2,
    'chi2_score': chi_selector.scores_[selected_mask_chi2]
}).sort_values('chi2_score', ascending=False)

df_aa_top100_chi2 = df_aa_aligned[selected_kmers_chi2]
print(f"   - Selected Chi2 AA k-mers shape: {df_aa_top100_chi2.shape}")

# ===========================================================================
# STEP 4: Feature selection - Random Forest Importance
# ===========================================================================
print("\n[4] Selecting top 100 AA k-mers by Random Forest Importance...")

rf_selector = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
rf_selector.fit(X_aa, y_vec)

importances = rf_selector.feature_importances_
top_idx_rf = np.argsort(importances)[::-1][:100]

selected_kmers_rf = df_aa_aligned.columns[top_idx_rf]

kmer_scores_rf = pd.DataFrame({
    'kmer': selected_kmers_rf,
    'rf_importance': importances[top_idx_rf]
}).sort_values('rf_importance', ascending=False)

df_aa_top100_rf = df_aa_aligned[selected_kmers_rf]
print(f"   - Selected RF AA k-mers shape: {df_aa_top100_rf.shape}")

# ===========================================================================
# STEP 5: Combine with ALL Gene Features
# ===========================================================================
print("\n[5] Combining AA k-mers + ALL gene features...")

# We use all gene features as specified in PREPROCESSING.md to get 310 features total
df_combined_chi2 = pd.concat([df_gene_aligned, df_aa_top100_chi2], axis=1)
df_combined_rf = pd.concat([df_gene_aligned, df_aa_top100_rf], axis=1)

print(f"   - Combined Chi2 dataset: {df_combined_chi2.shape}")
print(f"   - Combined RF dataset: {df_combined_rf.shape}")

# ===========================================================================
# STEP 6: Save Results
# ===========================================================================
print("\n[6] Saving datasets...")

kmer_scores_chi2.to_csv('top_100_kmers_chi2.csv', index=False)
kmer_scores_rf.to_csv('top_100_kmers_rf.csv', index=False)

df_combined_chi2.to_csv('X_chi2.csv')
df_combined_rf.to_csv('X_rf.csv')

print("   ✅ Files saved:")
print("      - top_100_kmers_chi2.csv")
print("      - top_100_kmers_rf.csv")
print("      - X_chi2.csv")
print("      - X_rf.csv")

# ===========================================================================
# STEP 7: Quick Baseline Evaluation
# ===========================================================================
print("\n[7] Quick Baseline Evaluation (Logistic Regression)")

def evaluate_dataset(X_df, y_true, name="Dataset"):
    print(f"\n   --- Evaluating {name} ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values, y_true, test_size=0.2, random_state=42, stratify=y_true
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    
    roc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='roc_auc')
    
    print(f"   ROC-AUC (Test): {roc:.4f}")
    print(f"   Recall (Test):  {recall:.4f}")
    print(f"   F1 (Test):      {f1:.4f}")
    print(f"   CV ROC-AUC:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

evaluate_dataset(df_combined_chi2, y_aligned.values, "Chi2 Combined Features (310 features)")
evaluate_dataset(df_combined_rf, y_aligned.values, "RF Combined Features (310 features)")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)