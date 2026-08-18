"""
Generate Performance Curve Plots & External Dataset Validation for CSE 400 Dengue Thesis
Matches exact 16-feature leakage-free pipeline from notebook:
  1. ROC Curves (Pediatric, Adult, Pooled) across LR, RF, XGB, LGBM, Stacking
  2. Precision-Recall (PR) Curves across cohorts
  3. Calibration Curves (Reliability Diagrams) with Brier Scores
  4. Confusion Matrix Heatmaps (Pediatric & Adult)
  5. External Dataset Generalizability Validation on Moon Kaggle Cohort (N=1524)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    brier_score_loss, confusion_matrix, accuracy_score, f1_score,
    matthews_corrcoef, recall_score, precision_score
)
from sklearn.calibration import calibration_curve

# Paths
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(WORKSPACE, "data")
IMAGES_DIR = os.path.join(WORKSPACE, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# 1. Load Primary Dataset (Mendeley Data)
primary_path = os.path.join(DATA_DIR, "Comprehensive Dengue Hematology and Clinical Datas", "dengue_dataset_combined.csv")
if not os.path.exists(primary_path):
    primary_path = os.path.join(DATA_DIR, "Dengue Hematology Dataset_bd v1.csv")

df = pd.read_csv(primary_path)

# Filter for Dengue Positive patients only
if 'Dengue NS1' in df.columns:
    df = df[df['Dengue NS1'].astype(str).str.lower() == 'positive'].copy()

# Rename columns
if 'Sex' in df.columns:
    df = df.rename(columns={'Sex': 'Gender'})

# Convert Gender to numeric (Male: 0, Female: 1)
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].replace({'Male': 0, 'Female': 1, 'male': 0, 'female': 1, 0: 0, 1: 1}).fillna(0).astype(int)

# Median imputation for numeric features
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Binary Target: Severe (PLT < 50.0 K/uL) vs Non-Severe (PLT >= 50.0 K/uL)
df['Target'] = (df['PLT'] < 50.0).astype(int)

# Feature engineering matching notebook exactly
df['wbc_leukopenia'] = (df['WBC'] < 4.0).astype(int)
df['lymph_low'] = (df['Lymph %'] < 20.0).astype(int) if 'Lymph %' in df.columns else 0
df['neut_high'] = (df['Neut %'] > 70.0).astype(int) if 'Neut %' in df.columns else 0
df['ast_alt_ratio'] = df['AST'] / (df['ALT'] + 1e-5) if ('AST' in df.columns and 'ALT' in df.columns) else 1.0
df['liver_involvement'] = (((df['AST'] > 40.0) | (df['ALT'] > 40.0)).astype(int)) if ('AST' in df.columns and 'ALT' in df.columns) else 0
df['age_decade'] = pd.cut(df['Age'], bins=[0, 9, 19, 29, 39, 49, 59, 69, 79, 100], labels=False).fillna(2).astype(int)
df['gender_age_int'] = df['Gender'] * df['Age']

# Exact 16 predictive features (Zero target leakage: NO PLT or derivatives)
features = [
    'Age', 'Gender', 'WBC', 'HCT', 'RBC', 'Lymph %', 'Neut %', 'ALT', 'AST',
    'wbc_leukopenia', 'lymph_low', 'neut_high', 'ast_alt_ratio', 'liver_involvement',
    'age_decade', 'gender_age_int'
]
features = [f for f in features if f in df.columns]

print(f"Primary Dataset: Total N = {len(df)}")
ped_df = df[df['Age'] <= 18].copy()
adult_df = df[df['Age'] > 18].copy()
print(f"Pediatric Cohort: n = {len(ped_df)}, Severe = {ped_df['Target'].sum()}")
print(f"Adult Cohort: n = {len(adult_df)}, Severe = {adult_df['Target'].sum()}")

# Model factory
def get_models():
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    xgb = XGBClassifier(n_estimators=100, scale_pos_weight=2.0, eval_metric='logloss', random_state=42)
    lgbm = LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42, verbose=-1)
    
    estimators = [('rf', rf), ('xgb', xgb), ('lgbm', lgbm)]
    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
    
    return {'LR': lr, 'RF': rf, 'XGB': xgb, 'LGBM': lgbm, 'Stacking': stacking}

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
colors = {'LR': '#e74c3c', 'RF': '#2ecc71', 'XGB': '#3498db', 'LGBM': '#9b59b6', 'Stacking': '#e67e22'}

# -------------------------------------------------------------
# PLOT 1: Multi-Cohort ROC Curves (Figure: roc_curves_comparison.png)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
cohort_data = [
    ('Pooled Cohort (N=972)', df, axes[0]),
    ('Pediatric Cohort (n=158, <=18y)', ped_df, axes[1]),
    ('Adult Cohort (n=814, >18y)', adult_df, axes[2])
]

for name, c_df, ax in cohort_data:
    X = c_df[features].values
    y = c_df['Target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    models = get_models()
    for m_name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{m_name} (AUC = {roc_auc:.3f})", color=colors[m_name], lw=2)
        
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance (AUC = 0.50)')
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    ax.legend(loc='lower right', frameon=True, fontsize=9.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

plt.tight_layout()
roc_path = os.path.join(IMAGES_DIR, 'roc_curves_comparison.png')
plt.savefig(roc_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"[OK] Saved ROC curves to {roc_path}")

# -------------------------------------------------------------
# PLOT 2: Precision-Recall (PR) Curves (Figure: precision_recall_curves.png)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

for (name, c_df, _), ax in zip(cohort_data, axes):
    X = c_df[features].values
    y = c_df['Target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    models = get_models()
    for m_name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, label=f"{m_name} (AUPRC = {pr_auc:.3f})", color=colors[m_name], lw=2)
        
    no_skill = np.sum(y_test) / len(y_test)
    ax.plot([0, 1], [no_skill, no_skill], 'k--', alpha=0.5, label=f'Baseline ({no_skill:.2f})')
    ax.set_title(f"PR Curve: {name}", fontsize=13, fontweight='bold')
    ax.set_xlabel('Recall (Sensitivity)', fontsize=11)
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=11)
    ax.legend(loc='lower left', frameon=True, fontsize=9.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

plt.tight_layout()
pr_path = os.path.join(IMAGES_DIR, 'precision_recall_curves.png')
plt.savefig(pr_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"[OK] Saved PR curves to {pr_path}")

# -------------------------------------------------------------
# PLOT 3: Calibration Reliability Curves (Figure: calibration_curves.png)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

for idx, (name, c_df) in enumerate([('Pediatric Cohort (<=18y)', ped_df), ('Adult Cohort (>18y)', adult_df)]):
    ax = axes[idx]
    X = c_df[features].values
    y = c_df['Target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    models = get_models()
    for m_name in ['RF', 'Stacking', 'XGB', 'LR']:
        model = models[m_name]
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=5, strategy='uniform')
        brier = brier_score_loss(y_test, y_prob)
        ax.plot(prob_pred, prob_true, marker='o', label=f"{m_name} (Brier = {brier:.3f})", color=colors[m_name], lw=2)
        
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Perfect Calibration')
    ax.set_title(f"Reliability Diagram: {name}", fontsize=13, fontweight='bold')
    ax.set_xlabel('Mean Predicted Probability', fontsize=11)
    ax.set_ylabel('Fraction of Positives', fontsize=11)
    ax.legend(loc='upper left', frameon=True, fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

plt.tight_layout()
calib_path = os.path.join(IMAGES_DIR, 'calibration_curves.png')
plt.savefig(calib_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"[OK] Saved Calibration curves to {calib_path}")

# -------------------------------------------------------------
# PLOT 4: Confusion Matrices (Figure: confusion_matrices_pediatric_adult.png)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

for idx, (name, c_df) in enumerate([('Pediatric (Best: Random Forest / Stacking)', ped_df), ('Adult (Best: Random Forest)', adult_df)]):
    ax = axes[idx]
    X = c_df[features].values
    y = c_df['Target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                annot_kws={'size': 14, 'weight': 'bold'},
                xticklabels=['Non-Severe (0)', 'Severe (1)'],
                yticklabels=['Non-Severe (0)', 'Severe (1)'])
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)

plt.tight_layout()
cm_path = os.path.join(IMAGES_DIR, 'confusion_matrices_pediatric_adult.png')
plt.savefig(cm_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"[OK] Saved Confusion Matrices to {cm_path}")

# -------------------------------------------------------------
# 5. EXTERNAL DATASET VALIDATION (Moon Kaggle Cohort, N=1524)
# -------------------------------------------------------------
kaggle_path = os.path.join(DATA_DIR, "Dengue Fever Hematological Dataset - Md Mahmudul Hasan Moon - kaggle.csv")
if os.path.exists(kaggle_path):
    print("\n--- Running External Validation on Moon Kaggle Cohort (N=1524) ---")
    ext_df = pd.read_csv(kaggle_path)
    ext_df.columns = ext_df.columns.str.strip()
    
    ext_clean = pd.DataFrame()
    ext_clean['Age'] = ext_df['Age']
    ext_clean['Gender'] = ext_df['Gender'].map({'Male': 0, 'Female': 1}).fillna(0).astype(int)
    ext_clean['WBC'] = ext_df['Total WBC count(/cumm)'] / 1000.0 if ext_df['Total WBC count(/cumm)'].median() > 100 else ext_df['Total WBC count(/cumm)']
    ext_clean['HCT'] = ext_df['HCT(%)']
    ext_clean['RBC'] = ext_df['RBC']
    ext_clean['Lymph %'] = ext_df['Lymphocytes(%)']
    ext_clean['Neut %'] = ext_df['Neutrophils(%)']
    
    # Impute ALT and AST with training medians (zero-leakage transfer)
    ext_clean['ALT'] = df['ALT'].median()
    ext_clean['AST'] = df['AST'].median()
    
    # Feature engineering for external set
    ext_clean['wbc_leukopenia'] = (ext_clean['WBC'] < 4.0).astype(int)
    ext_clean['lymph_low'] = (ext_clean['Lymph %'] < 20.0).astype(int)
    ext_clean['neut_high'] = (ext_clean['Neut %'] > 70.0).astype(int)
    ext_clean['ast_alt_ratio'] = ext_clean['AST'] / (ext_clean['ALT'] + 1e-5)
    ext_clean['liver_involvement'] = ((ext_clean['AST'] > 40.0) | (ext_clean['ALT'] > 40.0)).astype(int)
    ext_clean['age_decade'] = pd.cut(ext_clean['Age'], bins=[0, 9, 19, 29, 39, 49, 59, 69, 79, 100], labels=False).fillna(2).astype(int)
    ext_clean['gender_age_int'] = ext_clean['Gender'] * ext_clean['Age']
    
    # Target in Kaggle: Platelet is in raw count (e.g. 112000) or thousands
    plt_vals = ext_df['Total Platelet Count(/cumm)']
    if plt_vals.median() > 1000:
        ext_target = (plt_vals < 50000).astype(int)
    else:
        ext_target = (plt_vals < 50.0).astype(int)
        
    X_train_full = df[features].values
    y_train_full = df['Target'].values
    X_ext = ext_clean[features].fillna(ext_clean[features].median()).values
    y_ext = ext_target.values
    
    rf_full = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_full.fit(X_train_full, y_train_full)
    
    ext_prob = rf_full.predict_proba(X_ext)[:, 1]
    ext_pred = rf_full.predict(X_ext)
    
    ext_fpr, ext_tpr, _ = roc_curve(y_ext, ext_prob)
    ext_auc = auc(ext_fpr, ext_tpr)
    ext_acc = accuracy_score(y_ext, ext_pred)
    ext_f1 = f1_score(y_ext, ext_pred)
    ext_sens = recall_score(y_ext, ext_pred)
    ext_spec = recall_score(y_ext, ext_pred, pos_label=0)
    ext_mcc = matthews_corrcoef(y_ext, ext_pred)
    
    print(f"External Validation Results (Kaggle Cohort, N={len(ext_df)}):")
    print(f"  - AUC-ROC:     {ext_auc:.3f}")
    print(f"  - Accuracy:    {ext_acc*100:.1f}%")
    print(f"  - Sensitivity: {ext_sens*100:.1f}%")
    print(f"  - Specificity: {ext_spec*100:.1f}%")
    print(f"  - F1-Score:    {ext_f1:.3f}")
    print(f"  - MCC:         {ext_mcc:.3f}")
    
    # Plot External ROC Curve
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot(ext_fpr, ext_tpr, color='#e74c3c', lw=2.5, label=f'External Validation (Kaggle N=1524, AUC = {ext_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance Baseline (AUC = 0.50)')
    ax.set_title('Cross-Dataset External Validation ROC Curve', fontsize=13, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.legend(loc='lower right', frameon=True, fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    ext_roc_path = os.path.join(IMAGES_DIR, 'external_validation_performance.png')
    plt.savefig(ext_roc_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[OK] Saved External Validation curve to {ext_roc_path}")

print("\n🎉 All performance curves and external validations generated successfully!")
