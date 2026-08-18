import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import make_scorer, f1_score, balanced_accuracy_score, matthews_corrcoef

RANDOM_SEED = 42

def get_models(pos_weight):
    models = {
        'LR': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED),
        'RF': RandomForestClassifier(class_weight='balanced', random_state=RANDOM_SEED),
        'XGB': XGBClassifier(scale_pos_weight=pos_weight, random_state=RANDOM_SEED, eval_metric='logloss'),
        'LGBM': LGBMClassifier(class_weight='balanced', random_state=RANDOM_SEED, verbose=-1)
    }
    estimators = [(name, model) for name, model in models.items()]
    models['Stacking'] = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    return models

def compute_ci(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    return f"{mean:.3f} ± {1.96 * (std / np.sqrt(len(scores))):.3f}"

def evaluate_cohort(cohort_name, df, features, target_col):
    print(f"\n--- Benchmarking {cohort_name} Cohort (N={len(df)}) ---")
    X = df[features]
    y = df[target_col]
    
    if len(y.unique()) < 2:
        print(f"Skipping {cohort_name} (only 1 class present).")
        return None, None
        
    pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
    models = get_models(pos_weight)
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_SEED)
    
    scoring = {
        'f1': make_scorer(f1_score, average='macro'),
        'balanced_acc': make_scorer(balanced_accuracy_score),
        'mcc': make_scorer(matthews_corrcoef)
    }
    
    results = []
    best_f1 = 0
    best_model_name = None
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        cv_res = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        f1_mean = np.mean(cv_res['test_f1'])
        
        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_model_name = name
            
        results.append({
            'Cohort': cohort_name,
            'Model': name,
            'Macro-F1 (95% CI)': compute_ci(cv_res['test_f1']),
            'Balanced Acc (95% CI)': compute_ci(cv_res['test_balanced_acc']),
            'MCC (95% CI)': compute_ci(cv_res['test_mcc'])
        })
    
    # Train the best model on full cohort data for export
    best_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', models[best_model_name])
    ])
    best_pipeline.fit(X, y)
    
    # Save the pipeline
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_pipeline, f"models/{cohort_name.lower()}_best.pkl")
    print(f"Exported Best Model: {best_model_name} as {cohort_name.lower()}_best.pkl")
    
    return pd.DataFrame(results), best_pipeline

def main():
    print("Executing Section 06-09: Feature Policy and Benchmark...")
    df = pd.read_csv('outputs/dataset_with_target.csv')
    
    # Section 06 - Feature Policy & Leakage Audit
    # Target is derived from PLT, HCT, WBC.
    # To prevent circular leakage, we MUST drop PLT and its derivatives.
    # We will test "Proxy classification without PLT" (The main thesis result).
    
    # Exclude leakage variables and constant strings
    leakage_cols = ['PLT_k', 'NiaziMomand_SeverityProxy_v2', 'Target_Binary', 'Source', 'Source_Row', 'WHO_observable_severe_organ_flag', 'Dengue_NS1']
    
    all_cols = df.columns.tolist()
    features = [c for c in all_cols if c not in leakage_cols]
    
    # Convert Gender to numeric to prevent SimpleImputer median errors
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'Male': 0, 'Female': 1, 'male': 0, 'female': 1})
        
    print(f"Selected Features (Zero-Leakage): {features}")
    
    # Define Cohorts
    cohorts = {
        'Pooled': df,
        'Pediatric': df[df['Age'] <= 18],
        'Adult': df[df['Age'] > 18]
    }
    
    all_results = []
    
    for name, cohort_df in cohorts.items():
        res_df, _ = evaluate_cohort(name, cohort_df, features, 'Target_Binary')
        if res_df is not None:
            all_results.append(res_df)
            
    final_res = pd.concat(all_results, ignore_index=True)
    os.makedirs('outputs', exist_ok=True)
    final_res.to_csv('outputs/benchmark_results.csv', index=False)
    
    print("\nFinal Benchmark Results (Leakage-Controlled):")
    print(final_res.to_string(index=False))

if __name__ == "__main__":
    main()
