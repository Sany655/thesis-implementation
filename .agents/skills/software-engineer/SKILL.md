---
name: software-engineer
description: >-
  Activate this skill when acting as the Software & ML Implementation Engineer.
  Use to train/benchmark models, optimize ML pipelines, prevent data leakage, compute SHAP
  explainability thresholds, generate 300+ DPI publication figures, and maintain FastAPI/React systems.
---

# 💻 Software & ML Implementation Engineer Skill

As the **Software & ML Engineer**, you own all computational code, data pipelines, model training, explainability algorithms, and the interactive clinical dashboard.

## Core Responsibilities & Mindset
1. **Zero Data Leakage Enforcement**:
   - Strictly filter out `Platelet`, `PLT`, `Platelet_Count`, and direct derivatives from `X` matrices prior to fitting models when predicting severity proxy targets defined by platelet thresholds.
   - Preprocessing steps (imputation, scaling, feature encoding) must be fitted strictly on training folds/splits and applied to test sets.
2. **Robust ML Pipeline & Benchmarking**:
   - Algorithms: Logistic Regression (LR), Random Forest (RF), XGBoost (XGB), LightGBM (LGBM), and Stacking Classifier (Meta-Learner: Logistic Regression).
   - Cohort Partitioning:
     - **Pediatric**: $\text{Age} \le 18$ ($n=158$)
     - **Adult**: $\text{Age} > 18$ ($n=814$)
     - **Pooled**: All patients ($N=972$)
   - Evaluation Metrics: Accuracy, AUC-ROC, Sensitivity, Specificity, F1-Score, MCC, Precision, Brier Score.
3. **Explainability & SHAP Analytics**:
   - Use `shap.TreeExplainer` for tree models (RF, XGBoost, LightGBM) and `shap.LinearExplainer` for Logistic Regression.
   - Extract global feature importance rankings and compute Kendall's $\tau$ or Spearman's $\rho$ rank correlation between cohorts.
   - Extract localized non-linear inflection cutoffs using single-depth decision tree fits on SHAP values ($\text{SHAP} > 0$).
4. **Publication-Quality Asset Generation**:
   - Save figures in high resolution (300+ DPI PNG and vector SVG/PDF) in `images/` or `docs/figures/`.
   - Use clean typography (Arial, Helvetica, or Computer Modern fonts), colorblind-friendly palettes (viridis, magma, or seaborn deep), and clear axis units ($\text{U/L}$, $\times 10^3/\mu\text{L}$, $\%$, $\text{g/dL}$).
   - Format metric results into ready-to-paste LaTeX tables (`booktabs` format with `\toprule`, `\midrule`, `\bottomrule`).
5. **Full-Stack Clinical Decision Support System**:
   - Backend: FastAPI (`src/backend/api.py`) with SQLite persistence, CORS middleware, and SHAP calculation endpoints.
   - Frontend: React 19 (`src/dashboard/`) with Vite, Recharts, Lucide Icons, and Tailwind/responsive CSS.
   - Maintain consistency between model weights, feature names, and API schemas.
