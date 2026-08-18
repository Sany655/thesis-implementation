import pandas as pd
import numpy as np

def compute_niazi_momand_proxy(row):
    """
    Constructs the NiaziMomand_SeverityProxy_v2.
    - Severe: PLT < 50 AND (HCT > 45 OR WBC < 2 OR WBC > 20)
    - Moderate: PLT 50-99 OR HCT > 40 OR WBC < 4 OR WBC > 11
    - Minor: Otherwise
    """
    plt = row['PLT_k']
    hct = row['HCT_pct']
    wbc = row['WBC_k']
    
    # Severe criteria
    is_plt_severe = plt < 50.0
    corroborating_severe = (hct > 45.0) or (wbc < 2.0) or (wbc > 20.0)
    
    if is_plt_severe and corroborating_severe:
        return 'Severe'
        
    # Moderate criteria
    is_plt_mod = (plt >= 50.0) and (plt < 100.0)
    corroborating_mod = (hct > 40.0) or (wbc < 4.0) or (wbc > 11.0)
    
    if is_plt_mod or corroborating_mod or is_plt_severe: # is_plt_severe here catches plt<50 but without corroboration
        return 'Moderate'
        
    return 'Minor'

def main():
    print("Executing Section 04: Target Construction...")
    df = pd.read_csv('outputs/harmonized_positive_deduplicated.csv')
    
    # Construct Primary Proxy
    df['NiaziMomand_SeverityProxy_v2'] = df.apply(compute_niazi_momand_proxy, axis=1)
    
    # Construct binary target (Severe vs Non-Severe)
    df['Target_Binary'] = df['NiaziMomand_SeverityProxy_v2'].apply(lambda x: 1 if x == 'Severe' else 0)
    
    # Observable severe organ flag (AST/ALT >= 1000)
    df['WHO_observable_severe_organ_flag'] = ((df['AST'] >= 1000) | (df['ALT'] >= 1000)).astype(int)
    
    df.to_csv('outputs/dataset_with_target.csv', index=False)
    
    print(df['NiaziMomand_SeverityProxy_v2'].value_counts())
    print("\nBinary Target (Severe=1):")
    print(df['Target_Binary'].value_counts())
    
    print(f"\nSaved dataset_with_target.csv with {len(df)} rows.")

if __name__ == "__main__":
    main()
