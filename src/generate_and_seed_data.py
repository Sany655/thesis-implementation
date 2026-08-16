import os
import random
import uuid
import numpy as np
import pandas as pd
import sqlite3
import joblib
import shap
from datetime import datetime, timedelta

# Set seed for reproducible synthesis
np.random.seed(42)
random.seed(42)

# Load trained models
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
if not os.path.exists(os.path.join(MODEL_DIR, 'pediatric_best.pkl')):
    MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

print("Loading ML models...")
ped_model = joblib.load(os.path.join(MODEL_DIR, 'pediatric_best.pkl'))
adult_model = joblib.load(os.path.join(MODEL_DIR, 'adult_best.pkl'))

print("Initializing SHAP TreeExplainers...")
ped_explainer = shap.TreeExplainer(ped_model)
adult_explainer = shap.TreeExplainer(adult_model)

def build_features_df(records):
    rows = []
    for r in records:
        gender_int = 0 if r["Sex"] == "Male" else 1
        rows.append({
            'Age': r["Age"],
            'Gender': gender_int,
            'WBC': r["WBC (×10³/µL)"],
            'HCT': r["HCT (%)"],
            'RBC': r["RBC"],
            'Lymph %': r["Lymph %"],
            'Neut %': r["Neut %"],
            'ALT': r["ALT"],
            'AST': r["AST"]
        })
    df = pd.DataFrame(rows)
    df['wbc_leukopenia']    = (df['WBC'] < 4.0).astype(int)
    df['lymph_low']         = (df['Lymph %'] < 20.0).astype(int)
    df['neut_high']         = (df['Neut %'] > 70.0).astype(int)
    df['ast_alt_ratio']     = df['AST'] / (df['ALT'] + 1e-5)
    df['liver_involvement'] = ((df['AST'] > 40.0) | (df['ALT'] > 40.0)).astype(int)

    bins = [0, 9, 19, 29, 39, 49, 59, 69, 79, 100]
    df['age_decade'] = pd.cut(df['Age'], bins=bins, labels=False)
    df['age_decade'] = df['age_decade'].fillna(0).astype(int)
    df['gender_age_int'] = df['Gender'] * df['Age']

    expected_cols = [
        'Age', 'Gender', 'WBC', 'HCT', 'RBC', 'Lymph %', 'Neut %', 'ALT', 'AST',
        'wbc_leukopenia', 'lymph_low', 'neut_high', 'ast_alt_ratio', 'liver_involvement',
        'age_decade', 'gender_age_int'
    ]
    return df[expected_cols]

# Define 46 patients: P005 to P050
patient_ids = [f"P{i:03d}" for i in range(5, 51)]

patients_meta = []
for i, pid in enumerate(patient_ids):
    # Balanced age distribution
    if i % 3 == 0:
        age = round(random.choice([2.5, 4.0, 6.5, 8.0, 10.0, 12.5, 14.0, 15.5, 16.0, 17.5, 18.0]), 1)
    elif i % 3 == 1:
        age = round(random.choice([19.0, 21.0, 23.0, 25.0, 27.0, 30.0, 32.0, 35.0, 38.0, 42.0, 45.0]), 1)
    else:
        age = round(random.choice([48.0, 52.0, 55.0, 58.0, 62.0, 65.0, 70.0, 74.0]), 1)

    gender_str = "Male" if i % 2 == 0 else "Female"
    gender_int = 0 if gender_str == "Male" else 1

    # Severity distribution
    if i % 5 in (0, 1):
        severity = "Severe"
    elif i % 5 in (2, 3):
        severity = "Moderate"
    else:
        severity = "Minor"

    patients_meta.append({
        "patient_id": pid,
        "age": age,
        "gender_str": gender_str,
        "gender_int": gender_int,
        "severity": severity
    })

longitudinal_records = []
base_date = datetime(2026, 8, 10)

for p in patients_meta:
    pid = p["patient_id"]
    age = p["age"]
    gender_str = p["gender_str"]
    sev = p["severity"]
    
    if sev == "Severe":
        start_plt = round(random.uniform(135, 155), 1)
        end_plt   = round(random.uniform(18, 38), 1)
        plt_traj  = np.linspace(start_plt, end_plt, 5) + np.random.normal(0, 1.5, 5)
        plt_traj  = np.clip(plt_traj, 15, 180)

        start_wbc = round(random.uniform(4.0, 4.8), 2)
        end_wbc   = round(random.uniform(1.8, 2.5), 2)
        wbc_traj  = np.linspace(start_wbc, end_wbc, 5) + np.random.normal(0, 0.08, 5)
        wbc_traj  = np.clip(wbc_traj, 1.5, 6.5)

        start_hct = round(random.uniform(37.5, 39.5), 1)
        end_hct   = round(random.uniform(46.0, 50.0), 1)
        hct_traj  = np.linspace(start_hct, end_hct, 5) + np.random.normal(0, 0.2, 5)

        start_rbc = round(random.uniform(4.5, 4.8), 2)
        end_rbc   = round(random.uniform(4.1, 4.4), 2)
        rbc_traj  = np.linspace(start_rbc, end_rbc, 5)

        start_lym = round(random.uniform(24.0, 28.0), 1)
        end_lym   = round(random.uniform(39.0, 45.0), 1)
        lym_traj  = np.linspace(start_lym, end_lym, 5)

        start_neut = round(random.uniform(66.0, 70.0), 1)
        end_neut   = round(random.uniform(49.0, 54.0), 1)
        neut_traj  = np.linspace(start_neut, end_neut, 5)

        start_ast = round(random.uniform(45, 65), 1)
        end_ast   = round(random.uniform(140, 230), 1)
        ast_traj  = np.linspace(start_ast, end_ast, 5)

        start_alt = round(random.uniform(35, 50), 1)
        end_alt   = round(random.uniform(95, 160), 1)
        alt_traj  = np.linspace(start_alt, end_alt, 5)

    elif sev == "Moderate":
        start_plt = round(random.uniform(155, 180), 1)
        end_plt   = round(random.uniform(58, 85), 1)
        plt_traj  = np.linspace(start_plt, end_plt, 5) + np.random.normal(0, 2, 5)
        plt_traj  = np.clip(plt_traj, 52, 195)

        start_wbc = round(random.uniform(5.2, 6.5), 2)
        end_wbc   = round(random.uniform(3.1, 3.8), 2)
        wbc_traj  = np.linspace(start_wbc, end_wbc, 5) + np.random.normal(0, 0.08, 5)

        start_hct = round(random.uniform(36.5, 38.5), 1)
        end_hct   = round(random.uniform(41.5, 44.0), 1)
        hct_traj  = np.linspace(start_hct, end_hct, 5)

        start_rbc = round(random.uniform(4.4, 4.7), 2)
        end_rbc   = round(random.uniform(4.3, 4.6), 2)
        rbc_traj  = np.linspace(start_rbc, end_rbc, 5)

        start_lym = round(random.uniform(22.0, 26.0), 1)
        end_lym   = round(random.uniform(32.0, 36.0), 1)
        lym_traj  = np.linspace(start_lym, end_lym, 5)

        start_neut = round(random.uniform(67.0, 71.0), 1)
        end_neut   = round(random.uniform(57.0, 61.0), 1)
        neut_traj  = np.linspace(start_neut, end_neut, 5)

        start_ast = round(random.uniform(32, 45), 1)
        end_ast   = round(random.uniform(65, 105), 1)
        ast_traj  = np.linspace(start_ast, end_ast, 5)

        start_alt = round(random.uniform(25, 38), 1)
        end_alt   = round(random.uniform(52, 80), 1)
        alt_traj  = np.linspace(start_alt, end_alt, 5)

    else: # Minor
        start_plt = round(random.uniform(185, 225), 1)
        end_plt   = round(random.uniform(120, 150), 1)
        plt_traj  = np.linspace(start_plt, end_plt, 5) + np.random.normal(0, 3, 5)
        plt_traj  = np.clip(plt_traj, 110, 240)

        start_wbc = round(random.uniform(6.5, 8.0), 2)
        end_wbc   = round(random.uniform(4.6, 5.6), 2)
        wbc_traj  = np.linspace(start_wbc, end_wbc, 5)

        start_hct = round(random.uniform(35.5, 38.0), 1)
        end_hct   = round(random.uniform(37.5, 39.5), 1)
        hct_traj  = np.linspace(start_hct, end_hct, 5)

        start_rbc = round(random.uniform(4.3, 4.7), 2)
        end_rbc   = round(random.uniform(4.3, 4.6), 2)
        rbc_traj  = np.linspace(start_rbc, end_rbc, 5)

        start_lym = round(random.uniform(25.0, 29.0), 1)
        end_lym   = round(random.uniform(28.0, 33.0), 1)
        lym_traj  = np.linspace(start_lym, end_lym, 5)

        start_neut = round(random.uniform(63.0, 67.0), 1)
        end_neut   = round(random.uniform(59.0, 63.0), 1)
        neut_traj  = np.linspace(start_neut, end_neut, 5)

        start_ast = round(random.uniform(22, 34), 1)
        end_ast   = round(random.uniform(30, 45), 1)
        ast_traj  = np.linspace(start_ast, end_ast, 5)

        start_alt = round(random.uniform(18, 28), 1)
        end_alt   = round(random.uniform(24, 38), 1)
        alt_traj  = np.linspace(start_alt, end_alt, 5)

    for day in range(1, 6):
        idx = day - 1
        curr_plt   = round(float(plt_traj[idx]), 1)
        curr_wbc   = round(float(wbc_traj[idx]), 2)
        curr_hct   = round(float(hct_traj[idx]), 1)
        curr_rbc   = round(float(rbc_traj[idx]), 2)
        curr_lym   = round(float(lym_traj[idx]), 1)
        curr_neut  = round(float(neut_traj[idx]), 1)
        curr_ast   = round(float(ast_traj[idx]), 1)
        curr_alt   = round(float(alt_traj[idx]), 1)

        assessment_time = base_date + timedelta(days=day-1, hours=8 + (idx*2), minutes=15)

        longitudinal_records.append({
            "Patient": pid,
            "Day": day,
            "Age": age,
            "Sex": gender_str,
            "PLT (×10³/µL)": curr_plt,
            "WBC (×10³/µL)": curr_wbc,
            "HCT (%)": curr_hct,
            "RBC": curr_rbc,
            "Lymph %": curr_lym,
            "Neut %": curr_neut,
            "ALT": curr_alt,
            "AST": curr_ast,
            "Severity": sev,
            "Timestamp": assessment_time.strftime("%Y-%m-%d %H:%M:%S")
        })

df_longitudinal = pd.DataFrame(longitudinal_records)

# Save to data directory
os.makedirs("data/Comprehensive Dengue Hematology and Clinical Datas", exist_ok=True)
csv_path1 = "data/synthesized_longitudinal_dengue_dataset.csv"
csv_path2 = "data/Comprehensive Dengue Hematology and Clinical Datas/synthesized_longitudinal_dengue_dataset.csv"
df_longitudinal.to_csv(csv_path1, index=False)
df_longitudinal.to_csv(csv_path2, index=False)
print(f"Generated {len(df_longitudinal)} longitudinal rows across {len(patients_meta)} patients (P005-P050).")

# Compute ML Predictions and SHAP in batch
print("Batch computing feature matrices and ML predictions...")
X_all = build_features_df(longitudinal_records)

ped_probs = ped_model.predict_proba(X_all)[:, 1] * 100
adult_probs = adult_model.predict_proba(X_all)[:, 1] * 100

print("Batch computing SHAP explanations (Pediatric)...")
ped_raw_shap = ped_explainer.shap_values(X_all)
if isinstance(ped_raw_shap, list):
    ped_shaps = np.array(ped_raw_shap[1])
elif np.array(ped_raw_shap).ndim == 3:
    ped_shaps = np.array(ped_raw_shap)[:, :, 1]
else:
    ped_shaps = np.array(ped_raw_shap)

print("Batch computing SHAP explanations (Adult)...")
adult_raw_shap = adult_explainer.shap_values(X_all)
if isinstance(adult_raw_shap, list):
    adult_shaps = np.array(adult_raw_shap[1])
elif np.array(adult_raw_shap).ndim == 3:
    adult_shaps = np.array(adult_raw_shap)[:, :, 1]
else:
    adult_shaps = np.array(adult_raw_shap)

feat_cols = X_all.columns.tolist()

# Seed SQLite databases
db_paths = [
    "src/backend/dengue_dashboard.db",
    "src/dengue_dashboard.db"
]

for db_path in db_paths:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE patient (
        patient_id VARCHAR(36) PRIMARY KEY,
        age FLOAT NOT NULL,
        gender VARCHAR(10) NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE lab_assessment (
        assessment_id VARCHAR(36) PRIMARY KEY,
        patient_id VARCHAR(36) NOT NULL REFERENCES patient(patient_id),
        assessment_time DATETIME NOT NULL,
        wbc FLOAT NOT NULL,
        hct FLOAT NOT NULL,
        rbc FLOAT NOT NULL,
        lymphocyte FLOAT NOT NULL,
        neutrophil FLOAT NOT NULL,
        ast FLOAT NOT NULL,
        alt FLOAT NOT NULL,
        plt FLOAT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE model_prediction (
        prediction_id VARCHAR(36) PRIMARY KEY,
        assessment_id VARCHAR(36) NOT NULL REFERENCES lab_assessment(assessment_id),
        cohort VARCHAR(20) NOT NULL,
        model VARCHAR(50) NOT NULL,
        predicted_class VARCHAR(50) NOT NULL,
        probability FLOAT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE shap_explanation (
        shap_id VARCHAR(36) PRIMARY KEY,
        prediction_id VARCHAR(36) NOT NULL REFERENCES model_prediction(prediction_id),
        feature VARCHAR(50) NOT NULL,
        shap_value FLOAT NOT NULL,
        feature_val FLOAT NOT NULL
    );
    """)

    for p in patients_meta:
        cur.execute("INSERT INTO patient (patient_id, age, gender) VALUES (?, ?, ?)",
                    (p["patient_id"], p["age"], p["gender_str"]))

    for row_idx, r in enumerate(longitudinal_records):
        assessment_id = str(uuid.uuid4())
        cur.execute("""
        INSERT INTO lab_assessment (
            assessment_id, patient_id, assessment_time,
            wbc, hct, rbc, lymphocyte, neutrophil, ast, alt, plt
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            assessment_id,
            r["Patient"],
            r["Timestamp"],
            r["WBC (×10³/µL)"],
            r["HCT (%)"],
            r["RBC"],
            r["Lymph %"],
            r["Neut %"],
            r["AST"],
            r["ALT"],
            r["PLT (×10³/µL)"]
        ))

        row_vals = X_all.iloc[row_idx].tolist()

        # Pediatric Prediction & SHAP
        p_prob = float(ped_probs[row_idx])
        p_pred_id = str(uuid.uuid4())
        p_class = "High-risk pattern" if p_prob > 50 else "Low-risk pattern"
        cur.execute("""
        INSERT INTO model_prediction (
            prediction_id, assessment_id, cohort, model, predicted_class, probability
        ) VALUES (?, ?, ?, ?, ?, ?)
        """, (p_pred_id, assessment_id, "Pediatric", "RandomForest", p_class, p_prob))

        for f_idx, f_name in enumerate(feat_cols):
            cur.execute("""
            INSERT INTO shap_explanation (
                shap_id, prediction_id, feature, shap_value, feature_val
            ) VALUES (?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()), p_pred_id, f_name,
                float(ped_shaps[row_idx, f_idx]), float(row_vals[f_idx])
            ))

        # Adult Prediction & SHAP
        a_prob = float(adult_probs[row_idx])
        a_pred_id = str(uuid.uuid4())
        a_class = "High-risk pattern" if a_prob > 50 else "Low-risk pattern"
        cur.execute("""
        INSERT INTO model_prediction (
            prediction_id, assessment_id, cohort, model, predicted_class, probability
        ) VALUES (?, ?, ?, ?, ?, ?)
        """, (a_pred_id, assessment_id, "Adult", "RandomForest", a_class, a_prob))

        for f_idx, f_name in enumerate(feat_cols):
            cur.execute("""
            INSERT INTO shap_explanation (
                shap_id, prediction_id, feature, shap_value, feature_val
            ) VALUES (?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()), a_pred_id, f_name,
                float(adult_shaps[row_idx, f_idx]), float(row_vals[f_idx])
            ))

    conn.commit()
    conn.close()
    print(f"Successfully seeded DB: {db_path}")

print("All datasets generated and databases seeded successfully!")
