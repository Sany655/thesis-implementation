# Thesis Poster Content: Early Severe Dengue Prediction

## 1. Header
**Title**: Explainable AI-Based Predictive Dashboard for Early Dengue Shock & Platelet Nadir Forecasting  
**Authors**: [Your Name], [Supervisor Name]  
**Institution**: [Your University/Department]

---

## 2. Introduction
Dengue is a rapid-onset lethal threat where clinical status can deteriorate into life-threatening shock within hours. This research delivers an explainable ensemble system that transforms complex hematological shifts into trusted prognostic foresight for clinicians.

---

## 3. Problem Statement
Traditional triage remains too static for the high volatility of dengue progression, often failing to anticipate circulatory collapse. There is a critical need for an early warning system that bridges the gap between clinical intuition and automated AI diagnostics.

![Shock vs Platelet Comparison](shock_vs_platelet.png)

---

## 4. Objective
To develop a robust ensemble-based predictive dashboard that forecasts platelet nadir and shock risk. This system utilizes Explainable AI (XAI) to provide clinicians with interpretable visual insights into the key symptoms driving specific patient risk scores.

---

## 5. Methodology
*   **Dataset**: Prospective study of 2,301 high-risk dengue cases; features include Age, Baseline Platelets, HCT, and Clinical Symptoms. Applied **SMOTE** to balance shock outcomes.
*   **Architecture**: Optimized **Random Forest Regressor** for nadir forecasting and a **Weighted Soft-Voting Ensemble** (XGB/GB + RF) for shock precision.
*   **XAI Layer**: Integration of **SHAP** to provide feature-level transparency, mapping predictions to markers like HCT/PLT Ratio and Shock Index (Pulse/SBP).

![Methodology Diagram](methodology_diagram_final.png)

---

## 6. Result
*   **Performance**: Achieved **ROC AUC of 0.7142** for shock risk and **$R^2$ of 0.3495** (MAE ~28,496) for platelet nadir forecasting.
*   **EDA Observations**: Correlation analysis identified strong links between baseline hematocrit and shock risk, while distribution peaks show critical platelet drops on days 3-8.
*   **Feature Influence**: SHAP highlights **HCT/PLT Ratio**, **Pulse Rate**, and **Abdominal Pain** as the primary non-linear drivers of severity.

![Results Summary Table](results_table_final.png)
![Confusion Matrix](confusion_matrix.png)
![SHAP Global Importance](shap_summary.png)
![Platelet Distribution](platelet_dist.png)

---

## 7. Discussion
**Prognostic Utility & Synergy**: The ensemble model (XGB/GB/RF) achieved a robust 0.7142 ROC AUC using optimized SMOTE balancing to detect early shock risk. While the platelet nadir R² is 0.35, the quantitative 'floor' forecast provides a vital Early Warning System for clinical resource optimization.

**XAI & Clinical Trust**: Integration of SHAP provides feature-level transparency by mapping predictions to quantifiable markers like HCT/PLT ratio and pulse rate. This bridges the gap between automated forecasting and clinical intuition, ensuring the system is transparent and actionable for triage.

---

## 8. Conclusion
**Integrated Diagnostic Solution**: This research successfully developed an interactive dashboard that transforms static clinical markers into actionable prognostic foresight. The ensemble-based pipeline provides validated, real-time risk assessment for shock and platelet forecasting.

**Impact and Future Scope**: While currently validated on pediatric cohorts, future work will integrate diverse datasets to broaden the system's robustness across all age groups. This validated framework provides a critical blueprint for intelligent triage in resource-constrained environments.

---

## 9. Reference
1. Nguyen et al. (2013). *The value of daily platelet counts for predicting dengue shock syndrome*.
2. WHO. *Dengue Guidelines for Diagnosis, Treatment, Prevention and Control*.
3. Kaggle/OpenDengue. *Structured Clinical and Hematological Datasets*.
4. Islam et al. *Dengue Severity Level Prediction Using Machine Learning Models*.
