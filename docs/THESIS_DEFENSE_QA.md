# 🎓 Thesis Defense & Conference Presentation Q&A Cheatsheet

**Topic**: *Age-Stratified Dengue Severity Prediction and Subgroup-Conditioned Feature Attribution Analysis*  
**Authors**: Md. Aftabul Islam (`22070103`), Mohammad Mazharul Alam (`22070116`)  
**Supervisor**: Mohammad Arif Hasan Chowdhury  

Use this cheatsheet to ace questions from thesis defense examiners, conference session chairs, and peer reviewers.

---

### Q1: Why did you partition the dataset into pediatric ($\le 18$) and adult ($> 18$) cohorts rather than using Age as a continuous feature?
**Answer**:
"Treating Age as a single continuous feature in a pooled model assumes a uniform, monotonic relationship across all ages. However, dengue pathophysiology fundamentally differs between children and adults:
1. Children have greater microvascular permeability and distinct baseline lymphocyte distributions.
2. Adults experience higher rates of comorbid complications, hepatic involvement ($\text{AST} > \text{ALT}$), and hemoconcentration ($\text{HCT}$).
Our empirical findings showed that age-stratified models outperformed the pooled baseline ($\text{AUC } 0.947\text{--}0.961$ in pediatrics vs. $0.860$ pooled), and SHAP analysis revealed divergent non-linear decision thresholds between cohorts."

---

### Q2: How did you prevent circular data leakage in your model?
**Answer**:
"Many published dengue ML papers suffer from data leakage by including platelet count ($\text{PLT}$) or platelet-to-WBC ratios as predictive features while simultaneously defining severe dengue based on platelet thresholds. In our study, we strictly excluded $\text{PLT}$ and all direct platelet derivatives from the feature matrix $X$. The models rely solely on independent hematological and hepatic parameters ($\text{WBC}$, $\text{HCT}$, $\text{Lymphocyte \%}$, $\text{Neutrophil \%}$, $\text{AST}$, $\text{ALT}$), ensuring genuine prognostic utility."

---

### Q3: Why is your severe target defined as $\text{PLT} < 50,000/\mu\text{L}$ instead of the full WHO 2009 criteria?
**Answer**:
"The underlying clinical dataset is a retrospective hematology laboratory dataset that recorded complete blood counts and liver enzymes without continuous clinical ward charting of plasma leakage volumes or shock syndromes. Therefore, we harmonized our severity target with the established Niazi & Momand proxy criterion ($\text{PLT} < 50,000/\mu\text{L}$), representing severe thrombocytopenia. We explicitly disclose this as a proxy target in our limitations and emphasize the need for prospective validation on adjudicated WHO datasets."

---

### Q4: Why did Random Forest and Stacking Classifiers outperform XGBoost and Logistic Regression?
**Answer**:
"Logistic Regression is a linear classifier that cannot capture complex non-linear biomarker interactions (such as the joint effect of leukopenia and elevated transaminases). While XGBoost is effective, Random Forest and Stacking Ensembles demonstrated superior variance reduction and stability on the smaller pediatric cohort ($n=158$), achieving $100\%$ specificity and $0.961$ AUC without overfitting."

---

### Q5: How did you handle class imbalance in severe dengue cases?
**Answer**:
"Rather than applying synthetic oversampling (such as SMOTE), which can fabricate artificial biological measurements in clinical tabular data, we utilized balanced class weighting during cost-function optimization. This penalizes false negatives on severe cases without distorting the underlying clinical feature distributions. We evaluated with $\text{AUC-ROC}$, $\text{F1-score}$, and $\text{MCC}$ to ensure balanced assessment."

---

### Q6: What is the significance of Kendall's $\tau\text{-b} = 0.900$ rank correlation?
**Answer**:
"Kendall's $\tau\text{-b} = 0.900$ proves that at a macro level, both pediatric and adult models prioritize the same top clinical indicators ($\text{Lymphocytes}$, $\text{WBC}$, $\text{AST}$, $\text{HCT}$). However, the local SHAP dependence curves and single-depth decision tree splits revealed that the *inflection thresholds* differ significantly: for instance, positive risk contribution from AST begins at $\approx 50\text{ U/L}$ in children vs. $\approx 80\text{ U/L}$ in adults."

---

### Q7: Why is the lymphocyte percentage split threshold higher in children ($> 44.50\%$) than adults ($> 24.99\%$)?
**Answer**:
"Children naturally possess a higher baseline physiological lymphocyte percentage than adults (median $39.4\%$ pediatric baseline vs. $35.7\%$ adult). In acute severe dengue, pediatric patients frequently develop pronounced atypical reactive lymphocytosis as part of their hyperactive viral immune response, shifting the median lymphocyte percentage in severe pediatric cases to $49.5\%$, which drives the higher algorithmic decision threshold."

---

### Q8: What are the primary clinical limitations of this study?
**Answer**:
"We transparently highlight three limitations:
1. Retrospective, single-center origin from Bangladesh.
2. Use of severe thrombocytopenia ($\text{PLT} < 50,000/\mu\text{L}$) as a severity proxy.
3. Constrained sample size in the pediatric cohort ($n=158$).
Our study provides proof-of-concept for age-stratified XAI and serves as the analytical foundation for prospective multi-center validation."

---

### Q9: How is the clinical decision support dashboard deployed and structured?
**Answer**:
"The system is built as a production-grade full-stack application:
* **Frontend**: React 19, Vite, Recharts, and Lucide Icons with real-time SHAP waterfall rendering.
* **Backend**: FastAPI with SQLite persistence and RESTful inference endpoints.
* **Live URL**: [https://dengue-severity-risk-assessment.vercel.app/](https://dengue-severity-risk-assessment.vercel.app/)
* It includes clinical triage flags and longitudinal patient snapshot comparisons."

---

### Q10: What are the main recommendations for future work?
**Answer**:
"1. Prospective validation on multi-center international cohorts with adjudicated WHO 2009 labels.
2. Integrating day-of-illness temporal tracking into the age-stratified framework.
3. Expanding the pediatric cohort to enable multi-hospital prospective clinical trial validation."
