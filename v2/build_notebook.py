import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 0: Markdown
cell_0 = nbf.v4.new_markdown_cell("""# Thesis: Pediatric vs Adult Dengue Severity Prediction with Subgroup-Specific SHAP Explanations

**Context & Methodology:**
- **Objective**: Identify divergent physiological drivers of Dengue severity between Pediatric (<=18) and Adult (>18) cohorts.
- **Dataset**: Mendeley Data - Comprehensive Dengue Hematology and Clinical Dataset from Bangladesh.
- **Approach**: Age-stratified ML framework involving pooled and subgroup-specific models, interpreted via localized SHAP analysis.
- **Severity Target Harmonization**: Consistent with Niazi & Momand (2026), severity is derived from platelet counts (Minor, Moderate, Severe). For binary classification, we use Severe vs Non-Severe.
""")
nb.cells.append(cell_0)

# Cell 1: Imports & Setup
cell_1 = nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from scipy.stats import kendalltau
from IPython.display import display

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
""")
nb.cells.append(cell_1)

# Cell 2: Data Ingestion & Cohort Stratification
cell_2 = nbf.v4.new_markdown_cell("""## 1. Data Ingestion & Preprocessing
We load the dataset and perform initial preprocessing. We harmonize the target with the 3-tier severity proxy from Niazi & Momand (2026).
""")
nb.cells.append(cell_2)

cell_3 = nbf.v4.new_code_cell("""# 1. Load Data
file_path = 'Comprehensive Dengue Hematology and Clinical Datas/Dengue Hematology Dataset_bd.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print(f"Dataset Loaded. Shape: {df.shape}")
else:
    raise FileNotFoundError(f"File not found at {file_path}")

# Display columns
print(df.columns)
""")
nb.cells.append(cell_3)

# Cell 4: Preprocessing and Severity harmonization
cell_4 = nbf.v4.new_code_cell("""# 2. Preprocessing & Target Harmonization

# Filter for Dengue Positive patients only
if 'Dengue NS1' in df.columns:
    df = df[df['Dengue NS1'].str.lower() == 'positive'].copy()

# Rename columns for ease of use
df = df.rename(columns={'Sex': 'Gender'})

# Convert Gender to binary
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].replace({'Male': 0, 'Female': 1, 'male': 0, 'female': 1})

# Median imputation for missing values
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Deriving Severity Target (Niazi & Momand 2026 criteria based on Platelet Count)
# PLT is in K/uL (e.g., 50.0 means 50,000)
def derive_severity_3tier(plt):
    if plt < 50.0:
        return 'Severe'
    elif plt < 100.0:
        return 'Moderate'
    else:
        return 'Minor'

df['Severity_3Tier'] = df['PLT'].apply(derive_severity_3tier)

# Binary Target: Severe vs Non-Severe (Minor/Moderate)
df['Target'] = df['Severity_3Tier'].apply(lambda x: 1 if x == 'Severe' else 0)

print(df['Severity_3Tier'].value_counts())
print(df['Target'].value_counts())
""")
nb.cells.append(cell_4)

# Cell 5: EDA
cell_5 = nbf.v4.new_markdown_cell("""## 2. Exploratory Data Analysis (EDA)
Visualize key hematological markers across Pediatric vs Adult cohorts.
""")
nb.cells.append(cell_5)

cell_6 = nbf.v4.new_code_cell("""df_plot = df.copy()
df_plot['Cohort'] = df_plot['Age'].apply(lambda x: 'Pediatric (<=18)' if x <= 18 else 'Adult (>18)')

eda_features = ['PLT', 'WBC', 'HCT', 'Lymph %']
eda_features = [f for f in eda_features if f in df.columns]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(eda_features):
    sns.boxplot(x='Cohort', y=col, data=df_plot, ax=axes[i], palette='Set2', hue='Cohort', legend=False)
    axes[i].set_title(f'Distribution of {col}', fontsize=14)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('pediatric_vs_adult_eda.png')
plt.close()
""")
nb.cells.append(cell_6)

# Cell 7: Feature Engineering
cell_7 = nbf.v4.new_markdown_cell("""## 3. Feature Engineering
We create domain-driven features based on clinical thresholds as requested in the proposal (Phase 4).
""")
nb.cells.append(cell_7)

cell_8 = nbf.v4.new_code_cell("""# 3. Feature Engineering

# Platelet Critical Low (< 50,000)
df['platelet_critical_low'] = (df['PLT'] < 50.0).astype(int)

# Platelet Severe Low (< 20,000)
df['platelet_severe_low'] = (df['PLT'] < 20.0).astype(int)

# WBC Leukopenia (< 4,000)
df['wbc_leukopenia'] = (df['WBC'] < 4.0).astype(int)

# Lymphopenia (< 20%)
if 'Lymph %' in df.columns:
    df['lymph_low'] = (df['Lymph %'] < 20.0).astype(int)

# Neutrophilia (> 70%)
if 'Neut %' in df.columns:
    df['neut_high'] = (df['Neut %'] > 70.0).astype(int)

# AST / ALT ratio and Liver Involvement
if 'AST' in df.columns and 'ALT' in df.columns:
    df['ast_alt_ratio'] = df['AST'] / (df['ALT'] + 1e-5)
    # Liver involvement proxy (upper limit > 40 U/L)
    df['liver_involvement'] = ((df['AST'] > 40.0) | (df['ALT'] > 40.0)).astype(int)
    # AST to Platelet Ratio Index (APRI proxy)
    df['ast_to_platelet_ratio_index'] = df['AST'] / (df['PLT'] + 1e-5)

# PLT to WBC ratio
df['plt_to_wbc_ratio'] = df['PLT'] / (df['WBC'] + 1e-5)

# Age Decade
df['age_decade'] = pd.cut(df['Age'], bins=[0, 9, 19, 29, 39, 49, 59, 69, 79, 100], labels=False)

# Gender x Age interaction
df['gender_age_int'] = df['Gender'] * df['Age']

# Select final features
excluded_cols = ['SN', 'Dengue NS1', 'Severity_3Tier', 'Target', 'Unnamed: 12']
features = [c for c in df.columns if c not in excluded_cols]

# Remove data leakage features that perfectly determine the proxy target
leaky_features = ['PLT', 'platelet_critical_low', 'platelet_severe_low', 'ast_to_platelet_ratio_index', 'plt_to_wbc_ratio']
features = [f for f in features if f not in leaky_features]

print(f"Engineered Features (Leakage Removed): {features}")
""")
nb.cells.append(cell_8)

# Cell 9: Stratified Model Training
cell_9 = nbf.v4.new_markdown_cell("""## 4. Stratified Model Training & Evaluation
We split the dataset into Pediatric and Adult cohorts, train five classifiers (LR, RF, XGB, LGBM, Stacking) per cohort, and evaluate.
""")
nb.cells.append(cell_9)

cell_10 = nbf.v4.new_code_cell("""# 4. Stratified Model Training

# Subgroups
pediatric_mask = df['Age'] <= 18
adult_mask = df['Age'] > 18

cohorts = {
    'Pooled': df,
    'Pediatric': df[pediatric_mask],
    'Adult': df[adult_mask]
}

# Define Base Models
def get_models():
    return {
        'LR': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED),
        'RF': RandomForestClassifier(class_weight='balanced', random_state=RANDOM_SEED),
        'XGB': XGBClassifier(scale_pos_weight=(df['Target'].value_counts()[0]/df['Target'].value_counts()[1]), random_state=RANDOM_SEED, eval_metric='logloss'),
        'LGBM': LGBMClassifier(class_weight='balanced', random_state=RANDOM_SEED, verbose=-1)
    }

# Function to train and evaluate
def train_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    
    models = get_models()
    
    # Define Stacking Classifier
    estimators = [(name, model) for name, model in models.items()]
    stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    models['Stacking'] = stacking_clf
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate Sensitivity & Specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        # Matthews Correlation Coefficient
        # Handling the import for matthews_corrcoef
        from sklearn.metrics import matthews_corrcoef
        mcc = matthews_corrcoef(y_test, y_pred)
        
        results[name] = {
            'Accuracy': acc,
            'AUC': auc,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'F1': f1,
            'MCC': mcc
        }
        trained_models[name] = model
        
    return results, trained_models, X_train, X_test

# Run for each cohort
all_results = {}
all_models = {}
all_data = {}

for cohort_name, cohort_df in cohorts.items():
    print(f"\\n--- Training {cohort_name} Cohort (n={len(cohort_df)}) ---")
    X = cohort_df[features]
    y = cohort_df['Target']
    
    # If a cohort has only 1 class in target, skip or handle
    if len(y.unique()) < 2:
        print(f"Skipping {cohort_name} due to single class in target.")
        continue
        
    res, models_dict, X_tr, X_te = train_evaluate(X, y)
    all_results[cohort_name] = res
    all_models[cohort_name] = models_dict
    all_data[cohort_name] = {'X_train': X_tr, 'X_test': X_te, 'y_test': y.loc[X_te.index]}

# Format results
res_dfs = []
for cohort, res in all_results.items():
    df_res = pd.DataFrame(res).T
    df_res['Cohort'] = cohort
    res_dfs.append(df_res)

final_results = pd.concat(res_dfs)
display(final_results)
""")
nb.cells.append(cell_10)


# Cell 11: Subgroup-Specific SHAP Interpretations
cell_11 = nbf.v4.new_markdown_cell("""## 5. Subgroup-Specific SHAP Interpretations
Using SHAP to extract feature importance and dependence plots for the best model (Random Forest/XGBoost).
""")
nb.cells.append(cell_11)

cell_12 = nbf.v4.new_code_cell("""# SHAP Analysis
shap_values_dict = {}
shap_explainers = {}

# Use XGBoost for SHAP as TreeExplainer works well
model_to_explain = 'XGB'

fig, axes = plt.subplots(1, 3, figsize=(24, 6))

for i, cohort in enumerate(['Pooled', 'Pediatric', 'Adult']):
    if cohort not in all_models:
        continue
        
    model = all_models[cohort][model_to_explain]
    X_test = all_data[cohort]['X_test']
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    shap_values_dict[cohort] = shap_values
    shap_explainers[cohort] = explainer
    
    plt.subplot(1, 3, i+1)
    plt.title(f"SHAP Summary: {cohort}")
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=10)

plt.tight_layout()
plt.savefig('shap_summary_comparison.png')
plt.show()
""")
nb.cells.append(cell_12)

# Cell 13: Feature Ranking Comparison
cell_13 = nbf.v4.new_markdown_cell("""## 6. Feature Ranking Comparison
We quantify the disagreement between paired SHAP rankings using Kendall's tau-b.
""")
nb.cells.append(cell_13)

cell_14 = nbf.v4.new_code_cell("""from scipy.stats import kendalltau

def get_feature_ranking(shap_vals, feature_names):
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    ranking = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
    return ranking

rankings = {}
for cohort in ['Pooled', 'Pediatric', 'Adult']:
    if cohort in shap_values_dict:
        X_test = all_data[cohort]['X_test']
        rankings[cohort] = get_feature_ranking(shap_values_dict[cohort], X_test.columns)

print("--- Top 5 Features ---")
for cohort, rank in rankings.items():
    print(f"\\n{cohort}:")
    print(rank.head(5))

if 'Pediatric' in rankings and 'Adult' in rankings:
    # Ensure same feature order
    features_ordered = rankings['Pooled'].index.tolist()
    
    rank_ped = [rankings['Pediatric'].index.get_loc(f) for f in features_ordered]
    rank_ad = [rankings['Adult'].index.get_loc(f) for f in features_ordered]
    rank_pool = [rankings['Pooled'].index.get_loc(f) for f in features_ordered]
    
    tau_ped_ad, _ = kendalltau(rank_ped, rank_ad)
    tau_ped_pool, _ = kendalltau(rank_ped, rank_pool)
    tau_ad_pool, _ = kendalltau(rank_ad, rank_pool)
    
    print(f"\\nKendall's tau-b (Pediatric vs Adult): {tau_ped_ad:.3f}")
    print(f"Kendall's tau-b (Pediatric vs Pooled): {tau_ped_pool:.3f}")
    print(f"Kendall's tau-b (Adult vs Pooled): {tau_ad_pool:.3f}")
""")
nb.cells.append(cell_14)


# Cell 15: Age-Aware Clinical Decision Framework
cell_15 = nbf.v4.new_markdown_cell("""## 7. Age-Aware Clinical Decision Framework
Translating SHAP outputs into a structured clinical decision framework by examining SHAP dependence plots.
""")
nb.cells.append(cell_15)

cell_16 = nbf.v4.new_code_cell("""# SHAP Dependence plot for AST
for cohort in ['Pediatric', 'Adult']:
    if cohort in shap_explainers:
        shap.dependence_plot("AST", shap_values_dict[cohort], all_data[cohort]['X_test'], show=False)
        plt.title(f"Dependence Plot: AST ({cohort})")
        plt.savefig(f'shap_dependence_AST_{cohort}.png')
        plt.close()

# SHAP Dependence plot for WBC
for cohort in ['Pediatric', 'Adult']:
    if cohort in shap_explainers:
        shap.dependence_plot("WBC", shap_values_dict[cohort], all_data[cohort]['X_test'], show=False)
        plt.title(f"Dependence Plot: WBC ({cohort})")
        plt.savefig(f'shap_dependence_WBC_{cohort}.png')
        plt.close()
""")
nb.cells.append(cell_16)

# Cell 17: Programmatic Threshold Extraction
cell_17 = nbf.v4.new_markdown_cell("""## 8. Programmatic SHAP-Derived Clinical Threshold Extraction
To solidify the Age-Aware Clinical Decision Framework (Phase 9 of proposal), we programmatically extract the exact clinical cutoff values for the top features where the risk of severe Dengue becomes significantly elevated (SHAP value > 0).
""")
nb.cells.append(cell_17)

cell_18 = nbf.v4.new_code_cell("""from sklearn.tree import DecisionTreeClassifier
import pandas as pd

print("--- Extracted Clinical Cutoffs for Severe Dengue Risk (SHAP > 0) ---")
for cohort in ['Pediatric', 'Adult']:
    if cohort in shap_explainers:
        print(f"\\nCohort: {cohort}")
        X_test = all_data[cohort]['X_test']
        shap_vals = shap_values_dict[cohort]
        
        # Identify top 3 continuous features by mean absolute SHAP
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        top_features = pd.Series(mean_abs_shap, index=X_test.columns).sort_values(ascending=False)
        continuous_top = [f for f in top_features.index if X_test[f].nunique() > 2][:3]
        
        for feature in continuous_top:
            feature_idx = X_test.columns.get_loc(feature)
            feature_values = X_test.iloc[:, feature_idx].values.reshape(-1, 1)
            feature_shap = shap_vals[:, feature_idx]
            
            # Binary target: Does this feature's value push the model towards Severe risk?
            risk_elevated = (feature_shap > 0).astype(int)
            
            # Use a depth-1 Decision Tree to find the optimal split point
            dt = DecisionTreeClassifier(max_depth=1, random_state=42)
            try:
                dt.fit(feature_values, risk_elevated)
                if dt.tree_.node_count > 1:
                    threshold = dt.tree_.threshold[0]
                    direction = "<=" if dt.tree_.value[1][0][1] > dt.tree_.value[2][0][1] else ">"
                    print(f"  - {feature}: High Risk when {direction} {threshold:.2f}")
                else:
                    print(f"  - {feature}: No single threshold found (monotonic impact)")
            except:
                pass
""")
nb.cells.append(cell_18)

with open('c:/All/semester 8/CSE 400 - thesis/v2/thesis_v2_implementation_sir_proposed.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook successfully generated!")
