import pandas as pd
import hashlib
import glob
import os
import numpy as np

def calculate_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def harmonize_schema(filepath, source_name):
    df = pd.read_csv(filepath)
    
    # Kaggle dataset
    if 'Total Platelet Count(/cumm)' in df.columns:
        df = df.rename(columns={
            'Gender': 'Gender',
            'Age': 'Age',
            'Result': 'Dengue_NS1',
            'Total Platelet Count(/cumm)': 'PLT_k',
            'Total WBC count(/cumm)': 'WBC_k',
            'HCT(%)': 'HCT_pct',
            'RBC': 'RBC',
            'Lymphocytes(%)': 'Lymph_pct',
            'Neutrophils(%)': 'Neut_pct',
            'Hemoglobin(g/dl)': 'HGB'
        })
        # Convert to thousands
        df['PLT_k'] = pd.to_numeric(df['PLT_k'], errors='coerce') / 1000.0
        df['WBC_k'] = pd.to_numeric(df['WBC_k'], errors='coerce') / 1000.0
        df['ALT'] = np.nan
        df['AST'] = np.nan
        
    # BD Datasets
    else:
        rename_map = {
            'Sex': 'Gender',
            'Age (year)': 'Age',
            'Age': 'Age',
            'Dengue NS1': 'Dengue_NS1',
            'PLT': 'PLT_k',
            'WBC': 'WBC_k',
            'HCT': 'HCT_pct',
            'RBC': 'RBC',
            'Lymph %': 'Lymph_pct',
            'Neut %': 'Neut_pct',
            'ALT': 'ALT',
            'AST': 'AST'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df['HGB'] = np.nan
        
    df['Source'] = source_name
    df['Source_Row'] = df.index
    
    # Filter to standard columns
    standard_cols = ['Age', 'Gender', 'Dengue_NS1', 'PLT_k', 'WBC_k', 'HCT_pct', 'RBC', 'Lymph_pct', 'Neut_pct', 'HGB', 'ALT', 'AST', 'Source', 'Source_Row']
    return df[[c for c in standard_cols if c in df.columns]]

def main():
    print("Executing Section 01, 02, 03: Data Registry and Harmonization...")
    os.makedirs('outputs', exist_ok=True)
    
    files = {
        'BD_v4': 'data/Dengue Hematology Dataset_bd v4.csv',
        'BD_v2': 'data/dengue_dataset v2.csv',
        'Kaggle': 'data/Dengue Fever Hematological Dataset - Md Mahmudul Hasan Moon - kaggle.csv',
        'BD_v1': 'data/Dengue Hematology Dataset_bd v1.csv',
        'BD_v3': 'data/Dengue Hematology Dataset_bd v3.csv'
    }
    
    registry = []
    dfs = []
    
    for source_name, filepath in files.items():
        if os.path.exists(filepath):
            hash_val = calculate_sha256(filepath)
            df_raw = pd.read_csv(filepath)
            registry.append({
                'Source': source_name,
                'File': os.path.basename(filepath),
                'SHA-256': hash_val,
                'Raw_Rows': len(df_raw),
                'Columns': len(df_raw.columns)
            })
            
            # According to Manus guide, bdv1 and bdv3 are exact duplicates of bdv3/v4 data and should not be pooled independently.
            if source_name not in ['BD_v1', 'BD_v3']:
                df_clean = harmonize_schema(filepath, source_name)
                dfs.append(df_clean)
                
    registry_df = pd.DataFrame(registry)
    registry_df.to_csv('outputs/data_registry.csv', index=False)
    print(f"Data registry saved. Discarded {len(files) - len(dfs)} duplicate source files.")
    
    # Combine harmonized
    all_rows = pd.concat(dfs, ignore_index=True)
    
    # Filter to Positive only
    if 'Dengue_NS1' in all_rows.columns:
        all_rows['Dengue_NS1'] = all_rows['Dengue_NS1'].astype(str).str.strip().str.lower()
        positive_mask = all_rows['Dengue_NS1'] == 'positive'
        all_rows = all_rows[positive_mask]
    
    all_rows.to_csv('outputs/harmonized_all_rows.csv', index=False)
    print(f"Filtered to NS1 Positive. Total rows before dedup: {len(all_rows)}")
    
    # Deduplicate based on core measurements
    core_key = ['Age', 'Gender', 'PLT_k', 'WBC_k', 'HCT_pct', 'RBC', 'Lymph_pct', 'Neut_pct']
    dedup = all_rows.drop_duplicates(subset=core_key, keep='first')
    
    duplicates_removed = len(all_rows) - len(dedup)
    
    dedup.to_csv('outputs/harmonized_positive_deduplicated.csv', index=False)
    
    flow = pd.DataFrame({
        'Step': ['Total Pooled Positive', 'Deduplicated Core', 'Duplicates Removed'],
        'Count': [len(all_rows), len(dedup), duplicates_removed]
    })
    flow.to_csv('outputs/flow_counts.csv', index=False)
    
    print(f"Deduplicated dataset saved. Final Cohort Size: {len(dedup)}")
    print(f"Duplicates removed: {duplicates_removed}")

if __name__ == "__main__":
    main()
