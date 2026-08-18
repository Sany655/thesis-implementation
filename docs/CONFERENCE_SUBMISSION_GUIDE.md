# 📄 Conference Submission Package Guide

**Target Venues**:
* **IEEE BHI** (International Conference on Biomedical and Health Informatics)
* **IEEE EMBC** (Engineering in Medicine and Biology Society)
* **IEEE JBHI** (Journal of Biomedical and Health Informatics - Fast Track)
* **ACM CHIL** (Conference on Health, Inference, and Learning)
* **Springer LNCS / AIME** (Artificial Intelligence in Medicine)

---

## 📌 Submission Metadata & Author Details

* **Paper Title**: *Age-Stratified Dengue Severity Prediction and Subgroup-Conditioned Feature Attribution Analysis*
* **Short Running Title**: *Age-Stratified Explainable Dengue Severity Prediction*
* **Primary Track**: *Machine Learning & Explainable AI in Healthcare / Clinical Informatics*
* **Keywords**: `Dengue severity prediction`, `Age stratification`, `Explainable AI (xAI)`, `SHAP`, `Pediatric hematology`, `Decision tree thresholds`, `Clinical decision support system`.

### Author Affiliations
1. **Md. Aftabul Islam** (Student ID: `22070103`)  
   Department of Computer Science & Engineering, Faculty of Science, Engineering & Technology,  
   University of Science and Technology Chittagong (USTC), Chittagong, Bangladesh.
2. **Mohammad Mazharul Alam** (Student ID: `22070116`)  
   Department of Computer Science & Engineering, Faculty of Science, Engineering & Technology,  
   University of Science and Technology Chittagong (USTC), Chittagong, Bangladesh.
3. **Mohammad Arif Hasan Chowdhury** (Supervised By / Corresponding Author)  
   Assistant Professor, Department of Computer Science & Engineering, FSET,  
   University of Science and Technology Chittagong (USTC), Chittagong, Bangladesh.

---

## 📝 Camera-Ready Abstract (Copy & Paste Ready)

> Machine learning models developed for dengue severity prediction frequently pool pediatric and adult cases into a single training cohort, assuming a uniform feature contribution structure across age groups. This study evaluates whether partitioning a clinical dataset into age-stratified cohorts alters classifier performance and SHapley Additive exPlanations (SHAP) feature attribution patterns compared to an age-pooled baseline. Using a public dataset of 972 laboratory-confirmed Dengue NS1-positive patients from Bangladesh (158 pediatric $\le 18$ years, 814 adult $> 18$ years), we trained five classifiers (Logistic Regression, Random Forest, XGBoost, LightGBM, and a Stacking Ensemble) using routine hematology parameters. To reduce target leakage, platelet count (PLT) and its direct derivatives were excluded prior to model training, using PLT solely to define a binary severe-dengue proxy target ($\text{PLT} < 50,000 / \mu\text{L}$). On an 80/20 train-test split, the Stacking Ensemble achieved 90.6% accuracy (AUC 0.947) in pediatric cases, whereas Random Forest achieved 84.7% accuracy (AUC 0.892) in adult cases. Kendall's $\tau$-b rank correlation between pediatric and adult SHAP rankings reached 0.900, indicating high relative rank agreement for top features such as lymphocyte percentage, white blood cell count, and AST. However, fitting depth-1 decision trees to positive SHAP contributions ($\text{SHAP} > 0$) identified age-dependent split points, including a higher lymphocyte percentage split in children ($> 44.50\%$) than adults ($> 24.99\%$). The findings indicate that while overall feature ranking remains consistent across age groups, decision split thresholds differ between cohorts. These thresholds reflect dataset-specific attributions on a proxy target and require prospective validation on independent clinical datasets with adjudicated WHO severity labels before consideration for clinical decision support.

---

## 📦 How to Upload to Overleaf / Conference Portal
1. **Primary LaTeX Manuscript**: [`docs/ieee_dengue_age_stratified_paper.tex`](file:///c:/All/semester%208/CSE%20400%20-%20thesis/docs/ieee_dengue_age_stratified_paper.tex)
2. **Figures Directory**: Upload all images in [`images/`](file:///c:/All/semester%208/CSE%20400%20-%20thesis/images/) into your Overleaf project root or `images/` subfolder:
   * `methodology.drawio.png` (System flowchart)
   * `pediatric_vs_adult_eda.png` (EDA distribution comparison)
   * `shap_summary_comparison.png` (Global SHAP feature importance)
   * `shap_dependence_AST_Pediatric.png` & `shap_dependence_AST_Adult.png`
   * `shap_dependence_WBC_Pediatric.png` & `shap_dependence_WBC_Adult.png`
3. **Compilation Mode**: Select **pdfLaTeX** with `IEEEtran.cls` (standard on Overleaf).

---

## 🌐 Live Demonstration Link for Reviewers
Reviewers and conference attendees can interact directly with the deployed clinical decision support dashboard:
👉 **[https://dengue-severity-risk-assessment.vercel.app/](https://dengue-severity-risk-assessment.vercel.app/)**
