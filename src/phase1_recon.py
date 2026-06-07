import pandas as pd
import os
import io

def phase1_recon():
    base_dir = r"d:\Code\Projects\Internships\CodeAlpha\CodeAlpha_Desease_Detection-Model"
    train_path = os.path.join(base_dir, "pipeline_tabular", "data", "desease_test_train-dataset", "Training.csv")
    test_path = os.path.join(base_dir, "pipeline_tabular", "data", "desease_test_train-dataset", "Testing.csv")
    readme_path = os.path.join(base_dir, "ReadMe.md")
    
    # 1. Data Ingestion
    print("Loading data...")
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    log_content = []
    log_content.append("# Phase 1: Planning, Setup & Data Reconnaissance\n")
    
    # 2. Structural Reconnaissance
    train_shape = train_df.shape
    test_shape = test_df.shape
    log_content.append("## Structural Reconnaissance")
    log_content.append(f"- **Training Data Shape**: {train_shape[0]} rows, {train_shape[1]} columns")
    log_content.append(f"- **Testing Data Shape**: {test_shape[0]} rows, {test_shape[1]} columns\n")
    
    # 3. Feature Type Audit
    # Target column is 'prognosis'
    if 'prognosis' in train_df.columns:
        unique_diseases = train_df['prognosis'].nunique()
        log_content.append("## Feature Type Audit")
        log_content.append(f"- **Unique Target Diseases ('prognosis')**: {unique_diseases}")
        
        # Verify other columns are binary (0 or 1)
        feature_cols = [c for c in train_df.columns if c != 'prognosis']
        # Also remove 'Unnamed: 133' if exists which sometimes happens with pandas
        if 'Unnamed: 133' in feature_cols:
            feature_cols.remove('Unnamed: 133')
            
        non_binary_cols = []
        for col in feature_cols:
            unique_vals = train_df[col].dropna().unique()
            if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                non_binary_cols.append(col)
                
        if len(feature_cols) == 132:
            log_content.append("- **Symptom Columns Count**: 132 (Verified)")
        else:
            log_content.append(f"- **Symptom Columns Count**: {len(feature_cols)} (Expected 132)")
            
        if not non_binary_cols:
            log_content.append("- **Feature Values**: All symptom columns are binary (0 or 1).")
        else:
            log_content.append(f"- **Feature Values**: Found non-binary columns: {non_binary_cols}")
            
    log_content.append("\n")
    
    # 4. Missing Value Check
    train_nulls = train_df.isnull().sum().sum()
    test_nulls = test_df.isnull().sum().sum()
    
    log_content.append("## Missing Value Check")
    log_content.append(f"- **Training Data Null/Missing Values**: {train_nulls}")
    log_content.append(f"- **Testing Data Null/Missing Values**: {test_nulls}\n")
    
    log_content.append("---\n")
    
    # Write to ReadMe.md (Append-only)
    print("Writing to ReadMe.md...")
    with open(readme_path, "a", encoding="utf-8") as f:
        f.write("\n".join(log_content))
        
    print("Success! Phase 1 Data Reconnaissance complete.")
    print("ReadMe.md contents appended:")
    print("\n".join(log_content))

if __name__ == "__main__":
    phase1_recon()
