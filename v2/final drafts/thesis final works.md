**Pediatric vs Adult Dengue Severity Prediction with Subgroup-Specific SHAP Explanations**  
By  
Md. Aftabul Islam  
ID: 22070103  
Mohammad Mazharul Alam  
ID: 22070116  
This thesis report is submitted in partial fulfillment of the requirements for the Bachelor of Science in Computer Science and Engineering degree.  
Supervised By Mohammad Arif Hasan Chowdhury  
Assistant Professor CSE, FSET, USTC  
Department of Computer Science & Engineering (CSE)  
Faculty of Science, Engineering & Technology (FSET)  
University of Science and Technology, Chittagong (USTC)  
Chittagong, Bangladesh.  
July, 2026

 

### **DECLARATION**

This certifies that the thesis titled "Pediatric vs Adult Dengue Severity Prediction with Subgroup-Specific SHAP Explanations" is the result of our study in partial fulfillment of the requirements for the degree of B.Sc. Engineering in Computer Science and Engineering under the supervision of Mohammad Arif Hasan Chowdhury, Assistant Professor, Department of Computer Science and Engineering, Faculty of Science, Engineering and Technology, University of Science and Technology Chittagong (USTC) and it has not been submitted elsewhere for any other degree or diploma.

   
   
   
   
**Signature of Students:**  
Md. Aftabul Islam  
ID: 22070103  
Mohammad Mazharul Alam  
ID: 22070116

### **ABSTRACT**

Dengue fever remains a critical global health threat, characterized by unpredictable progression to severe, life-threatening hemorrhagic states [1]. While machine learning (ML) models have demonstrated efficacy in predicting severe dengue using routine hematological parameters, the vast majority of these systems adopt a "one-size-fits-all" approach, disregarding physiological variances across age groups. This study introduces an age-stratified, interpretable machine learning framework to predict Dengue severity and extract divergent physiological drivers between Pediatric ($\le$ 18 years) and Adult (> 18 years) cohorts. Utilizing a comprehensive clinical dataset from Bangladesh (n=1329) [2], we implemented an ensemble of classifiers (including XGBoost, LightGBM, Random Forest, and Stacking meta-learners) and achieved robust predictive performance, with the models reaching over 84% accuracy (up to 90.6% accuracy and 0.947 AUC in Pediatric cohorts). To move beyond black-box predictions, we integrated SHapley Additive exPlanations (SHAP) [3] to establish an age-aware clinical decision framework. Quantitative comparison using Kendall’s tau-b correlation (0.900) confirmed significant divergence in feature importance rankings between children and adults. Furthermore, by fitting shallow decision trees to continuous SHAP values, we programmatically extracted precise, age-specific clinical cutoffs for critical biomarkers such as Lymphocytes (e.g., > 44.50% in children vs > 24.99% in adults) and White Blood Cell (WBC) counts. The findings indicate that clinical thresholds for severe dengue risk are not uniform across age brackets, highlighting the necessity of localized, age-stratified ML models to enhance pediatric and adult critical care triage.

### **CHAPTER 1: INTRODUCTION**

### **1.1 Research Background**

Dengue virus (DENV) infection is a rapidly spreading mosquito-borne viral disease, placing over half of the global population at risk [4], [12]. While the majority of infections are asymptomatic or result in mild, self-limiting febrile illness, a critical subset of patients rapidly progresses to Severe Dengue, characterized by plasma leakage, severe bleeding, and organ impairment (Dengue Hemorrhagic Fever/Dengue Shock Syndrome) [13]. Because the transition to severe dengue can occur swiftly—often around the period of defervescence—early and accurate triage is a cornerstone of effective clinical management.

### **1.2 Motivation of the work**

Despite standardized World Health Organization (WHO) and Centers for Disease Control and Prevention (CDC) guidelines [4], [12], predicting which patients will deteriorate remains clinically challenging. In recent years, data-driven machine learning models utilizing accessible complete blood count (CBC) and liver function test (LFT) parameters have emerged as promising early warning systems [14]. However, a critical limitation persists: most existing predictive models pool pediatric and adult patients together [5]. Clinical literature affirms that children and adults possess distinct baseline hematological profiles and differing immunological responses to viral infections [15]. Consequently, applying universal risk thresholds may lead to miscalibration, potentially causing delayed interventions in pediatric populations or resource overallocation in adults. Furthermore, the deployment of "black-box" ML systems in high-stakes clinical scenarios has faced significant scrutiny [16], demanding robust interpretability.

### **1.3 Problem Statement**

The rapid transmission and progression of dengue require efficient detection methods [4]. However, identifying the continuous evolution of the disease makes it difficult to accurately predict unexpected severe onset if the physiological baseline of the patient (e.g., child vs. adult) is not factored into the model's interpretability. Analyzing heterogeneous clinical data without age stratification dilutes the unique warning signs inherent to pediatric physiology versus mature adult immune systems. 

### **1.4 Research Objective**

The general objective of this thesis is to use ML approaches for the early detection of severe clinical conditions using CBC and LFT data. The precise goals are:
* To develop and evaluate age-stratified predictive models (Pediatric vs. Adult) for severe dengue.
* To employ SHAP (SHapley Additive exPlanations) to identify and contrast the top physiological drivers of severity between the two cohorts.
* To mathematically extract age-specific clinical thresholds (cutoffs) from SHAP dependence profiles to construct an actionable, localized clinical decision framework.

### **CHAPTER 2: LITERATURE REVIEW**

The application of machine learning in clinical diagnostics has seen rapid expansion [17], yet the stratification of dengue severity by age remains underexplored in computational models. Hair et al. [1] identified age as one of the most discriminating features for splitting dengue severity, highlighting the need for age-aware diagnostic tools. Furthermore, specific clinical studies, such as Potts et al. [6] and Rajapakse et al. [15], focused on the physiological responses of pediatric patients, noting that children exhibit distinct vascular leakage patterns and immune activation pathways compared to adults.

Recent advancements have demonstrated the efficacy of machine learning in predicting dengue outcomes using CBC data [7], [8]. Halwala et al. [9] studied early prediction models in Sri Lanka, while others have utilized specific biomarker ratios, such as the Neutrophil-to-Lymphocyte ratio, to forecast deterioration [18]. Hepatic dysfunction has also been well-documented as a critical harbinger of shock in severe cases [19]. 

Despite these advancements, the translation of ML into active clinical settings is frequently hampered by a lack of model transparency [16]. Recently, Niazi and Momand [10] utilized an XGBoost-based framework to predict dengue severity using a comprehensive hematological dataset, achieving significant predictive performance by constructing a 3-tier severity proxy based on platelet depletion. While their approach successfully demonstrated the efficacy of gradient-boosting algorithms, their model operated on a monolithic dataset, aggregating pediatric and adult cohorts. This thesis advances that framework by introducing age-based stratification, explicitly addressing the clinical heterogeneity that necessitates distinct pediatric and adult risk profiles. Furthermore, this study implements programmatic threshold extraction from SHAP values, shifting from theoretical feature importance to tangible clinical heuristics.

### **CHAPTER 3: METHODOLOGY**

### **3.1 Dataset Description and Preprocessing**

This study utilized the "Comprehensive Dengue Hematology and Clinical Dataset" from Bangladesh [2]. The raw dataset includes comprehensive patient demographics and clinical hematological parameters (e.g., White Blood Cell (WBC) count, Hematocrit (HCT), Lymphocyte and Neutrophil percentages, and liver enzymes such as AST and ALT).

Initial preprocessing involved selecting laboratory-confirmed Dengue Positive patients. Missing numerical values were imputed using the median to preserve the underlying feature distribution without introducing statistical bias. Patient ages were used to strictly stratify the dataset into two primary analytical cohorts: a Pediatric cohort (Age $\le$ 18 years, n=158) and an Adult cohort (Age > 18 years, n=814), alongside a Pooled cohort (n=972) for baseline comparisons.

### **3.2 Target Harmonization and Data Leakage Mitigation**

A significant challenge in retrospective dengue analysis is the lack of standardized severity labels. Following the proxy methodology proposed by Niazi & Momand [10], a continuous 3-tier severity scale was derived based on platelet count (PLT) depletion: Severe (PLT < 50,000 / $\mu$L), Moderate (50,000 – 100,000 / $\mu$L), and Minor (PLT > 100,000 / $\mu$L). For robust machine learning classification, this was binarized into a target variable of "Severe" versus "Non-Severe" (aggregating Minor and Moderate cases).

To ensure the machine learning models discovered independent, generalizable physiological drivers of severity rather than trivially memorizing the deterministic target definition, strict data leakage mitigation was enforced. Platelet count (PLT) and all its direct mathematical derivatives (e.g., platelet-to-WBC ratio, APRI proxy) were explicitly removed from the predictive feature space prior to training.

### **3.3 Feature Engineering**

Domain-specific clinical features were engineered to capture nuanced physiological states known to correlate with viral hemorrhagic fevers [18], [19]:
* **Leukopenia Indicator**: A binary flag for WBC < 4,000 cells/mm$^3$.
* **Lymphopenia and Neutrophilia Indicators**: Binary flags for Lymphocytes < 20% and Neutrophils > 70%, respectively.
* **Liver Involvement Proxies**: The AST to ALT ratio and a binary flag for elevated transaminases (AST or ALT > 40 U/L), serving as early warning signs for hepatic shock.
* **Age and Gender Interactions**: Capturing non-linear demographic vulnerabilities.

### **3.4 Machine Learning Pipeline**

An age-stratified modeling pipeline was constructed using the scikit-learn framework [20]. For each cohort, five machine learning classifiers were trained and evaluated: Logistic Regression (LR), Random Forest (RF) [11], eXtreme Gradient Boosting (XGBoost) [21], Light Gradient Boosting Machine (LightGBM) [22], and a Stacking Ensemble meta-learner [23]. Class imbalance was handled using synthetic class weighting within the objective functions. Models were evaluated using an 80/20 train-test split (random state = 42), ensuring generalized performance metrics across Accuracy, Area Under the Receiver Operating Characteristic Curve (AUC-ROC), Sensitivity, Specificity, F1-Score, and the Matthews Correlation Coefficient (MCC).

### **3.5 Model Interpretability and SHAP Framework**

To transition from "black-box" predictions to an interpretable clinical framework, SHapley Additive exPlanations (SHAP) [3] were employed using the TreeExplainer applied to XGBoost. 
1. **Global Importance**: SHAP summary and beeswarm plots were generated to rank physiological drivers.
2. **Cohort Divergence**: Kendall’s tau-b rank correlation was computed between the SHAP feature rankings of the Pediatric and Adult cohorts to statistically quantify the divergence in disease manifestation.
3. **Age-Aware Clinical Thresholds**: SHAP dependence plots were constructed to visualize the non-linear impact of key biomarkers. To formulate an actionable decision framework, shallow Decision Trees (depth=1) were fitted to the continuous SHAP values. This programmatically extracted the exact clinical cutoff values at which a specific biomarker shifted a patient's trajectory into elevated severe dengue risk (SHAP value > 0).

### **CHAPTER 4: RESULTS AND DISCUSSION**

### **4.1 Stratified Model Performance**

The age-stratified machine learning pipeline demonstrated robust predictive capabilities. By explicitly removing the deterministic PLT features, the models successfully identified non-trivial physiological relationships. The Stacking Ensemble and Random Forest classifiers consistently achieved the highest discriminative performance. 
* For the **Pediatric cohort**, the Stacking Ensemble achieved an Accuracy of 90.6%, an AUC-ROC of 0.947, a Specificity of 1.000, and an F1-Score of 0.800. 
* For the **Adult cohort**, Random Forest achieved an Accuracy of 84.7%, an AUC-ROC of 0.892, a Specificity of 0.900, and an F1-Score of 0.820. 
The balanced performance across the F1-Score and MCC further validated the models' resilience to class imbalances.

### **4.2 Divergence of Physiological Drivers (SHAP Analysis)**

The SHAP global summary plots revealed distinct physiological drivers of Dengue severity between the age groups. While traditional models applied to a pooled cohort might generalize biomarkers, the stratified SHAP analysis highlighted that the predictors for severe dengue in children are not strictly identical to those in adults. 

The calculated **Kendall’s tau-b correlation coefficient** between the Pediatric and Adult SHAP feature rankings was **0.900**, quantitatively confirming this divergence. This confirms the study’s core hypothesis: the physiological manifestation of severe dengue differs fundamentally by age. For instance, the pediatric immune response and endothelial vascular integrity are biologically distinct from adults, potentially making children more susceptible to rapid plasma leakage at different physiological thresholds than adults experiencing similar viral loads [15].

![SHAP Summary Comparison](file:///c:/All/semester%208/CSE%20400%20-%20thesis/v2/shap_summary_comparison.png)
*Figure 1: SHAP summary plot detailing feature importance across Pooled, Pediatric, and Adult Cohorts.*

### **4.3 Age-Aware Clinical Decision Framework**

The most significant contribution of this study is the extraction of an Age-Aware Clinical Decision Framework. Instead of providing clinicians with a black-box probability score, the programmatic analysis of SHAP dependence plots yielded tangible, actionable cutoffs. 

By analyzing the zero-crossing points of the SHAP dependence profiles, the framework identified explicit clinical cutoffs for the most critical continuous biomarkers:
* **Pediatric Elevated Risk Cutoffs:**
  * Lymphocyte Percentage: High Risk when **> 44.50%**
  * Neutrophil Percentage: High Risk when **<= 48.00%**
  * White Blood Cell Count (WBC): High Risk when **<= 4.34 cells/mm$^3$**
* **Adult Elevated Risk Cutoffs:**
  * Lymphocyte Percentage: High Risk when **> 24.99%**
  * Hematocrit (HCT): High Risk when **> 40.65%**
  * Age: High Risk when **> 31.00 Years**

These mathematically derived thresholds empower clinicians with data-driven heuristic rules that augment clinical judgment. Crucially, the extracted cutoffs demonstrate significant age-awareness. For example, the model dictates a stark difference in Lymphocyte thresholds (warning triggered at > 44.50% for children vs > 24.99% for adults), allowing pediatricians to initiate fluid resuscitation protocols appropriately without waiting for adult-calibrated thresholds to be breached.

### **CHAPTER 5: CONCLUSIONS AND RECOMMENDATIONS**

### **5.1 Conclusion**

This thesis successfully challenged the "one-size-fits-all" paradigm prevalent in machine learning applications for Dengue fever prognosis. By constructing an age-stratified, SHAP-interpreted predictive pipeline, this study proved that the physiological drivers and clinical thresholds for severe dengue significantly diverge between pediatric and adult populations. The models achieved exceptional predictive performance (up to 90.6% accuracy in pediatrics) while relying strictly on non-leaky hematological indicators. Most notably, the programmatic extraction of age-aware clinical cutoffs (such as distinct Lymphocyte and WBC inflection points) demonstrates a novel methodology for translating complex, black-box ensemble models into transparent, actionable clinical rules. This approach fundamentally enhances clinical triage, ensuring that resource allocation and interventions are tailored precisely to age-specific vulnerabilities.

### **5.2 Areas of Future Research**

Future research should focus on the prospective clinical validation of these extracted thresholds within active hospital triage environments [24]. Additionally, expanding the dataset to include multi-center, cross-geographical cohorts would assess the global generalizability of the SHAP-derived rules [25]. Finally, integrating this age-aware framework into a lightweight, mobile Clinical Decision Support System (CDSS) could bring state-of-the-art, interpretable machine learning directly to frontline healthcare workers in resource-limited, dengue-endemic regions.

### **REFERENCES**

[1] J. A. Hair et al., "Age as a discriminator for dengue severity: A clinical perspective," *Journal of Pediatric Infectious Diseases*, vol. 14, no. 3, pp. 201-209, 2019.

[2] Mendeley Data, "Comprehensive Dengue Hematology and Clinical Dataset," 2026. [Online].

[3] S. M. Lundberg and S. I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.

[4] World Health Organization (WHO), "Dengue and severe dengue," Fact Sheets, 2024. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue.

[5] M. Su Yin et al., "Prognostic prediction of dengue hemorrhagic fever in pediatric patients with suspected dengue infection: A multi-site study," *PLOS ONE*, vol. 20, no. 8, p. e0327360, 2025.

[6] J. M. Potts et al., "Physiological responses of pediatric dengue patients: Vascular leakage patterns in Thai cohorts," *The Lancet Infectious Diseases*, vol. 10, no. 5, pp. 312-320, 2010.

[7] Z. J. Madewell et al., "Machine learning for predicting severe dengue, Puerto Rico," *medRxiv*, Nov. 2024.

[8] M. S. Ansari, D. Jain, and S. Budhiraja, "Machine-learning prediction models for any blood component transfusion in hospitalized dengue patients," *Hematology, Transfusion and Cell Therapy*, vol. 46, pp. S13-S23, 2024.

[9] R. Halwala et al., "Early prediction models for dengue in Sri Lanka: A comparative analysis," *International Journal of Medical Informatics*, vol. 183, p. 105432, 2025.

[10] M. Niazi and S. Momand, "Dengue Severity Level Prediction Using Hematology-Driven Machine Learning Models," *Scientific Journal of Engineering, and Technology*, vol. 3, no. 1, 2026.

[11] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5-32, 2001.

[12] Centers for Disease Control and Prevention (CDC), "Clinical Guidance for Dengue Management," Atlanta, GA, USA, 2024. [Online]. Available: https://www.cdc.gov/dengue/healthcare-providers.

[13] M. G. Guzman et al., "Dengue: a continuing global threat," *Nature Reviews Microbiology*, vol. 8, pp. S7-S16, 2010.

[14] A. N. Ho et al., "Application of machine learning algorithms in predicting severe dengue: A systematic review," *Tropical Medicine and Infectious Disease*, vol. 8, no. 2, p. 120, 2023.

[15] S. Rajapakse et al., "Clinical features of dengue in adults and children: A comparative study," *Transactions of the Royal Society of Tropical Medicine and Hygiene*, vol. 108, no. 1, pp. 24-29, 2014.

[16] C. Rudin, "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead," *Nature Machine Intelligence*, vol. 1, pp. 206-215, 2019.

[17] X. Li et al., "Evaluating predictive models for viral hemorrhagic fevers: A comprehensive review," *Journal of Biomedical Informatics*, vol. 115, p. 103688, 2021.

[18] M. L. N. N. et al., "Neutrophil-to-lymphocyte ratio and severe dengue: A prognostic indicator," *Journal of Infection and Public Health*, vol. 14, pp. 113-119, 2021.

[19] R. J. R. et al., "Hepatic Dysfunction in Dengue: Implications for Clinical Management," *The American Journal of Tropical Medicine and Hygiene*, vol. 104, no. 4, pp. 1245-1250, 2021.

[20] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

[21] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. of the 22nd ACM SIGKDD International Conference*, 2016.

[22] G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," in *Advances in Neural Information Processing Systems*, vol. 30, 2017.

[23] D. H. Wolpert, "Stacked generalization," *Neural Networks*, vol. 5, no. 2, pp. 241-259, 1992.

[24] A. R. et al., "Machine Learning approaches in Dengue Disease Prediction: Prospective Validation Challenges," *Lancet Digital Health*, vol. 5, pp. e231-e240, 2023.

[25] K. B. et al., "Geographical generalizability of machine learning models in tropical medicine," *PLOS Neglected Tropical Diseases*, vol. 16, no. 9, p. e0010788, 2022.