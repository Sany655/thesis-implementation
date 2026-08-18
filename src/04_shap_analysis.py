import pandas as pd
import numpy as np
import shap
import joblib
import os
import matplotlib.pyplot as plt

def generate_shap_plots(cohort_name, df, features):
    print(f"\nGenerating SHAP plots for {cohort_name} cohort...")
    
    # Load model pipeline
    model_path = f"models/{cohort_name.lower()}_best.pkl"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
        
    pipeline = joblib.load(model_path)
    X = df[features]
    
    # Extract transformations and model
    imputer = pipeline.named_steps['imputer']
    scaler = pipeline.named_steps['scaler']
    classifier = pipeline.named_steps['classifier']
    
    # Transform data
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)
    
    # Check if tree-based for TreeExplainer
    tree_models = ['RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier']
    model_type = classifier.__class__.__name__
    
    if model_type in tree_models:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer(X_scaled_df)
        
        # Adjust for shape diffs in binary classification
        if isinstance(shap_values.values, list):
            sv = shap_values.values[1] # positive class
        elif len(shap_values.values.shape) == 3:
            sv = shap_values.values[:, :, 1]
        else:
            sv = shap_values.values
            
        shap_obj = shap.Explanation(values=sv, base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value, data=X_scaled_df, feature_names=features)
        
        # 1. Beeswarm plot
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(shap_obj, show=False)
        plt.title(f"{cohort_name} Cohort - SHAP Beeswarm (Zero-Leakage)", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"outputs/shap_beeswarm_{cohort_name.lower()}.png", dpi=300)
        plt.close()
        print(f"Saved outputs/shap_beeswarm_{cohort_name.lower()}.png")
        
        # 2. Extract Top 5 features based on mean absolute SHAP
        mean_abs_shap = np.abs(sv).mean(axis=0)
        shap_ranking = pd.DataFrame({'Feature': features, 'Mean_Abs_SHAP': mean_abs_shap}).sort_values(by='Mean_Abs_SHAP', ascending=False)
        shap_ranking.to_csv(f"outputs/shap_ranking_{cohort_name.lower()}.csv", index=False)
        print(f"Saved outputs/shap_ranking_{cohort_name.lower()}.csv")
        
    else:
        print(f"SHAP TreeExplainer not supported for {model_type}. Using KernelExplainer or skip...")
        # Since we use RF/XGB/LGBM mostly as best models, we assume one of them won.

def main():
    print("Executing Section 10: SHAP Analysis...")
    df = pd.read_csv('outputs/dataset_with_target.csv')
    
    # Must match exactly what was trained
    leakage_cols = ['PLT_k', 'NiaziMomand_SeverityProxy_v2', 'Target_Binary', 'Source', 'Source_Row', 'WHO_observable_severe_organ_flag', 'Dengue_NS1']
    
    # Handle Gender mapping like in training
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'Male': 0, 'Female': 1, 'male': 0, 'female': 1})
        
    all_cols = df.columns.tolist()
    features = [c for c in all_cols if c not in leakage_cols]
    
    cohorts = {
        'Pediatric': df[df['Age'] <= 18],
        'Adult': df[df['Age'] > 18],
        'Pooled': df
    }
    
    for name, cohort_df in cohorts.items():
        generate_shap_plots(name, cohort_df, features)

if __name__ == "__main__":
    main()
