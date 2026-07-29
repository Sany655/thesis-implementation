import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib

# 1. Data Ingestion
file_path = 'niazi and momand - kaggle dengue - Dengue Fever Hematological Dataset.csv'
if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
    exit()

df = pd.read_csv(file_path)
print("Dataset Loaded.")

# 2. Preprocessing
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].replace({'Male': 0, 'Female': 1, 'male': 0, 'female': 1})

target_col = 'Result' if 'Result' in df.columns else df.columns[-1]
df['Target'] = df[target_col].replace({'negative': 0, 'positive': 1})

# Define features explicitly based on methodology and dataset
features = ['Age', 'Gender', 'Hemoglobin(g/dl)', 'Total Platelet Count(/cumm)', 'HCT(%)', 'WBC']
features = [f for f in features if f in df.columns]

# Median Imputation
X = df[features].fillna(df[features].median())
y = df['Target']

# EDA Plot (Pediatric vs Adult)
df_plot = df.copy()
df_plot['Cohort'] = df_plot['Age'].apply(lambda x: 'Pediatric (<18)' if x < 18 else 'Adult (>=18)')
eda_features = ['Total Platelet Count(/cumm)', 'Hemoglobin(g/dl)', 'WBC', 'HCT(%)']
eda_features = [f for f in eda_features if f in df.columns]

if len(eda_features) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(eda_features):
        sns.boxplot(x='Cohort', y=col, data=df_plot, ax=axes[i], palette='Set2', hue='Cohort', legend=False)
        axes[i].set_title(f'Distribution of {col}', fontsize=14)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
    # Hide any unused axes
    for j in range(len(eda_features), len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig('pediatric_vs_adult_eda.png')
    print("EDA plot saved as pediatric_vs_adult_eda.png")


# 3. Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Evaluation
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

print("\n--- Model Performance (Training Set) ---")
print(f"Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Precision: {precision_score(y_train, y_train_pred, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_train, y_train_pred, average='weighted'):.4f}")
print(f"F1-Score:  {f1_score(y_train, y_train_pred, average='weighted'):.4f}")

print("\n--- Model Performance (Testing Set) ---")
print(f"Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_test_pred, average='weighted'):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_test_pred, average='weighted'):.4f}")

joblib.dump(rf_model, 'dengue_rf_model.joblib')

# 6 & 7. Age-Stratification and SHAP Analysis

explainer = shap.TreeExplainer(rf_model)

def generate_shap_plots(X_data, dataset_name):
    shap_values = explainer.shap_values(X_data, check_additivity=False)
    if isinstance(shap_values, list):
        sv_class1 = shap_values[1]
    else:
        sv_class1 = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values

    ped_mask = (X_data['Age'] < 18).values
    adult_mask = (X_data['Age'] >= 18).values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))
    
    plt.sca(ax1)
    if sum(ped_mask) > 0:
        shap.summary_plot(sv_class1[ped_mask], X_data[ped_mask], show=False)
    ax1.set_title(f"Pediatric Cohort SHAP (n={sum(ped_mask)})", fontsize=16)

    plt.sca(ax2)
    if sum(adult_mask) > 0:
        shap.summary_plot(sv_class1[adult_mask], X_data[adult_mask], show=False)
    ax2.set_title(f"Adult Cohort SHAP (n={sum(adult_mask)})", fontsize=16)

    plt.tight_layout()
    filename = f'shap_comparison_{dataset_name}.png'
    plt.savefig(filename)
    print(f"SHAP plot saved as {filename}")

# Run for Test Dataset
print("\nGenerating SHAP for Test Dataset...")
generate_shap_plots(X_test, "test_dataset")

# Run for Whole Dataset
print("\nGenerating SHAP for Whole Dataset...")
generate_shap_plots(X, "full_dataset")

print("\nPipeline Execution Complete!")