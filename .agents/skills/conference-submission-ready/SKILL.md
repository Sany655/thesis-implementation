---
name: conference-submission-ready
description: >-
  Use this skill to perform end-to-end conference submission readiness checks,
  orchestrating the supervisor, engineer, and writer sub-agents to verify that code,
  data, LaTeX paper, figures, and artifacts meet top-tier conference submission standards.
---

# 🚀 Conference Submission Readiness Pipeline

This skill orchestrates the multi-agent workflow to bring the dengue thesis from current draft to final conference submission readiness.

## 📋 Comprehensive Conference Submission Checklist

### Phase 1: Experimental & Code Validation (`@engineer`)
- [ ] Feature leakage check: Platelet features completely isolated from input feature matrix `X`.
- [ ] Stratified cross-validation: Repeated 5-fold or 10-fold CV results recorded with standard deviations.
- [ ] Metric synchronization: All numbers in code output match `README.md` and `docs/ieee_dengue_age_stratified_paper.tex`.
- [ ] Figures exported at 300+ DPI with legible axis labels and colorblind-safe palettes.
- [ ] Code repository clean, documented, and reproducible with minimal setup.

### Phase 2: Manuscript & LaTeX Review (`@writer`)
- [ ] IEEE conference template compliance (no format deviations).
- [ ] Abstract contains clear quantitative results (AUC, Accuracy, F1) and takeaways.
- [ ] All equations mathematically rigorous with proper variable definitions.
- [ ] Figures and tables properly referenced in text (`\ref{...}`) with informative captions.
- [ ] Complete bibliography without duplicate or placeholder citations.
- [ ] Page count within conference limits (typically 6-8 pages for IEEE conferences).

### Phase 3: Senior Reviewer Audit (`@supervisor`)
- [ ] Clear scientific contribution highlighted in Introduction & Conclusion.
- [ ] Proper handling and disclosure of the proxy severity target ($\text{PLT} < 50,000/\mu\text{L}$).
- [ ] Clinical and physiological justification provided for observed pediatric vs. adult SHAP divergences.
- [ ] Comprehensive Limitations section addressing sample size, single-center data, and prospective validation.
- [ ] Final simulated peer-review score: Overall recommendation $\ge 4/5$ (Strong Accept).

## 🛠️ Automated Audit Tool
Run the multi-agent validation script:
```powershell
python .agents/scripts/research_team.py --check-all
```
