import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import joblib

from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier

def main():
    print("Loading dataset...")
    df = pd.read_csv('../data/Dengue Hematology Dataset_bd v4.csv')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    pediatric_df = df[df['Age'] < 18].copy()

    features = ['Age', 'Sex', 'AST', 'ALT', 'RBC', 'Lymph %', 'Neut %', 'PLT', 'HCT', 'WBC']
    target = 'Dengue NS1'

    pediatric_df['Sex'] = pediatric_df['Sex'].map({'Female': 0, 'Male': 1})
    pediatric_df[target] = pediatric_df[target].map({'Negative': 0, 'Positive': 1})

    valid_idx = pediatric_df[target].notna()
    pediatric_df = pediatric_df[valid_idx]
    
    X = pediatric_df[features]
    y = pediatric_df[target].astype(int)

    print("Imputing...")
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    print("Training Best RF Model...")
    # Best Params from 07: max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=50
    model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=1, min_samples_split=2, random_state=42, class_weight='balanced')
    model.fit(X_imputed, y)

    # Save model for future use
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/v4_pediatric_best.pkl')
    print("Saved model to ../models/v4_pediatric_best.pkl")

    print("Generating SHAP insights...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_imputed)

    # For Random Forest, shap_values is a list. We take the positive class [1]
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    else:
        shap_values_pos = shap_values

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_pos, X_imputed, show=False)
    plt.title('SHAP Summary: Feature Importance for Pediatric Dengue (v4)')
    plt.tight_layout()
    plt.savefig('../figures/pediatric/shap_summary_examiner.png', dpi=300)
    plt.close()
    print("Saved SHAP summary to ../figures/pediatric/shap_summary_examiner.png")

if __name__ == '__main__':
    main()
