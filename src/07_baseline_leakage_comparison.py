import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

RANDOM_SEED = 42

def main():
    print("Executing Section 07: Baseline Target Leakage Comparison...")
    
    df = pd.read_csv('outputs/dataset_with_target.csv')
    
    # 1. Target and Gender handling
    y = df['Target_Binary']
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'Male': 0, 'Female': 1, 'male': 0, 'female': 1})
        
    # 2. Define Features for both models
    all_cols = df.columns.tolist()
    
    # "Literature Baseline" model (INCLUDES the Platelet count which is mathematically tied to the target)
    baseline_leakage_cols = ['NiaziMomand_SeverityProxy_v2', 'Target_Binary', 'Source', 'Source_Row', 'WHO_observable_severe_organ_flag', 'Dengue_NS1']
    baseline_features = [c for c in all_cols if c not in baseline_leakage_cols]
    
    # "Our Zero-Leakage Proxy" model (EXCLUDES the Platelet count)
    zero_leakage_cols = baseline_leakage_cols + ['PLT_k']
    zero_leakage_features = [c for c in all_cols if c not in zero_leakage_cols]
    
    print(f"Literature Baseline Features: {baseline_features}")
    print(f"Zero-Leakage Features: {zero_leakage_features}")
    
    # 3. Model setup
    model = RandomForestClassifier(class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
    
    def create_pipeline():
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    # Storage for ROC variables
    tprs_baseline = []
    tprs_zero = []
    aucs_baseline = []
    aucs_zero = []
    mean_fpr = np.linspace(0, 1, 100)
    
    print("Running cross-validation to compute ROC curves...")
    for train_idx, test_idx in cv.split(df, y):
        # Baseline (Leakage) Model
        X_train_b, X_test_b = df[baseline_features].iloc[train_idx], df[baseline_features].iloc[test_idx]
        y_train_b, y_test_b = y.iloc[train_idx], y.iloc[test_idx]
        
        pipe_b = create_pipeline()
        pipe_b.fit(X_train_b, y_train_b)
        y_prob_b = pipe_b.predict_proba(X_test_b)[:, 1]
        
        fpr_b, tpr_b, _ = roc_curve(y_test_b, y_prob_b)
        roc_auc_b = auc(fpr_b, tpr_b)
        aucs_baseline.append(roc_auc_b)
        interp_tpr_b = np.interp(mean_fpr, fpr_b, tpr_b)
        interp_tpr_b[0] = 0.0
        tprs_baseline.append(interp_tpr_b)
        
        # Zero-Leakage Proxy Model
        X_train_z, X_test_z = df[zero_leakage_features].iloc[train_idx], df[zero_leakage_features].iloc[test_idx]
        y_train_z, y_test_z = y.iloc[train_idx], y.iloc[test_idx]
        
        pipe_z = create_pipeline()
        pipe_z.fit(X_train_z, y_train_z)
        y_prob_z = pipe_z.predict_proba(X_test_z)[:, 1]
        
        fpr_z, tpr_z, _ = roc_curve(y_test_z, y_prob_z)
        roc_auc_z = auc(fpr_z, tpr_z)
        aucs_zero.append(roc_auc_z)
        interp_tpr_z = np.interp(mean_fpr, fpr_z, tpr_z)
        interp_tpr_z[0] = 0.0
        tprs_zero.append(interp_tpr_z)

    # 4. Plotting
    mean_tpr_b = np.mean(tprs_baseline, axis=0)
    mean_tpr_b[-1] = 1.0
    mean_auc_b = auc(mean_fpr, mean_tpr_b)
    
    mean_tpr_z = np.mean(tprs_zero, axis=0)
    mean_tpr_z[-1] = 1.0
    mean_auc_z = auc(mean_fpr, mean_tpr_z)
    
    plt.figure(figsize=(9, 7))
    plt.plot(mean_fpr, mean_tpr_b, color='red', lw=2.5, 
             label=f'Literature Baseline (Target Leakage): PLT Included\nAUC = {mean_auc_b:.3f}')
             
    plt.plot(mean_fpr, mean_tpr_z, color='blue', lw=2.5, 
             label=f'Our Proposed Approach (Zero-Leakage Proxy): PLT Excluded\nAUC = {mean_auc_z:.3f}')
             
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Target Leakage Artifact vs Genuine Physiological Proxy', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    os.makedirs('images', exist_ok=True)
    plt.tight_layout()
    plt.savefig('images/literature_leakage_comparison.png', dpi=300)
    print("Saved literature comparison plot to images/literature_leakage_comparison.png")

if __name__ == "__main__":
    main()
