import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

def main():
    # Ensure output directory for figures exists
    os.makedirs('../figures/pediatric', exist_ok=True)

    # 1. Load Data & Filter Pediatric
    print("Loading data...")
    df = pd.read_csv('../data/Dengue Hematology Dataset_bd v4.csv')

    # Clean Age column and filter
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    pediatric_df = df[df['Age'] < 18].copy()
    print(f"Extracted {len(pediatric_df)} pediatric records.")

    # 2. Select requested features and target
    features = ['Age', 'Sex', 'AST', 'ALT', 'RBC', 'Lymph %', 'Neut %', 'PLT', 'HCT', 'WBC']
    target = 'Dengue NS1'

    # Preprocessing Categoricals
    pediatric_df['Sex'] = pediatric_df['Sex'].map({'Female': 0, 'Male': 1})
    pediatric_df[target] = pediatric_df[target].map({'Negative': 0, 'Positive': 1})

    # Isolate feature matrix X and target y
    X = pediatric_df[features]
    y = pediatric_df[target]

    # Drop rows where target is missing
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx].astype(int)

    # 3. KNN Imputation for AST, ALT, etc.
    print("Performing KNN Imputation...")
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # 4. EDA: Correlation Heatmap and Distribution
    print("Generating EDA plots...")
    plt.figure(figsize=(10, 8))
    corr_matrix = X_imputed.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Pediatric Features', fontsize=14)
    plt.tight_layout()
    plt.savefig('../figures/pediatric/correlation_heatmap.png', dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.countplot(x=y, palette='Set2')
    plt.title('Target Distribution (Dengue NS1) in Pediatric Cohort', fontsize=14)
    plt.xticks(ticks=[0, 1], labels=['Negative', 'Positive'])
    plt.xlabel('Dengue Diagnosis')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('../figures/pediatric/target_distribution.png', dpi=300)
    plt.close()

    # 5. Model Training and Hyperparameter Tuning for "Perfect Fit"
    print("Training Model with GridSearchCV...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # n_jobs=1 instead of -1 to avoid any possible issues on windows entirely for small dataset
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=1)
    grid_search.fit(X_scaled, y)

    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")

    # Evaluate Model
    y_pred = best_model.predict(X_scaled)
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Negative', 'Positive']))
    print(f"ROC-AUC Score: {roc_auc_score(y, best_model.predict_proba(X_scaled)[:, 1]):.4f}")

    # 6. Generate Learning Curve
    print("Generating Learning Curves...")
    train_sizes, train_scores, test_scores = learning_curve(
        best_model, X_scaled, y, cv=cv, scoring='accuracy', 
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=1, random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    plt.title('Learning Curve: Pediatric Dengue Model', fontsize=14)
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    plt.savefig('../figures/pediatric/learning_curve.png', dpi=300)
    plt.close()
    print("Pipeline complete! Artifacts saved to ../figures/pediatric/")

if __name__ == '__main__':
    main()
