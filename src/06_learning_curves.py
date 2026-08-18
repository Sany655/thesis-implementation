import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

RANDOM_SEED = 42

def plot_learning_curve(estimator, title, X, y, cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Training examples", fontsize=12)
    plt.ylabel("Macro-F1 Score", fontsize=12)
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, 
        train_sizes=train_sizes, scoring='f1_macro'
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score", linewidth=2)
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score", linewidth=2)
    
    plt.legend(loc="lower right", fontsize=11)
    return plt

def main():
    print("Executing Section 06: Generating Learning Curves...")
    
    df = pd.read_csv('outputs/dataset_with_target.csv')
    
    # Exclude leakage variables
    leakage_cols = ['PLT_k', 'NiaziMomand_SeverityProxy_v2', 'Target_Binary', 'Source', 'Source_Row', 'WHO_observable_severe_organ_flag', 'Dengue_NS1']
    features = [c for c in df.columns if c not in leakage_cols]
    
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'Male': 0, 'Female': 1, 'male': 0, 'female': 1})
        
    X = df[features]
    y = df['Target_Binary']
    
    # Use the best performing stable model from benchmark (Random Forest)
    model = RandomForestClassifier(class_weight='balanced', random_state=RANDOM_SEED)
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_SEED)
    
    # Generate curves for the pooled cohort
    print("Computing learning curve metrics (this may take a minute)...")
    plt = plot_learning_curve(pipeline, "Learning Curve: Zero-Leakage Proxy Model (Random Forest)", X, y, cv=cv)
    
    os.makedirs('images', exist_ok=True)
    plt.tight_layout()
    plt.savefig('images/learning_curves.png', dpi=300)
    print("Saved learning curves to images/learning_curves.png")

if __name__ == "__main__":
    main()
