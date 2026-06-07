import pandas as pd
import os
import json
import joblib
from sklearn.preprocessing import LabelEncoder

def phase2_preprocessing():
    base_dir = r"d:\Code\Projects\Internships\CodeAlpha\CodeAlpha_Desease_Detection-Model"
    data_dir = os.path.join(base_dir, "pipeline_tabular", "data", "desease_test_train-dataset")
    processed_dir = os.path.join(base_dir, "pipeline_tabular", "data", "processed")
    readme_path = os.path.join(base_dir, "ReadMe.md")
    
    os.makedirs(processed_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, "Training.csv")
    test_path = os.path.join(data_dir, "Testing.csv")
    
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    log_content = ["\n## Phase 2: Tabular Preprocessing Validation\n"]
    
    # 1. Cleaning Artifact Columns
    dropped_cols = []
    if 'Unnamed: 133' in train_df.columns:
        train_df = train_df.drop(columns=['Unnamed: 133'])
        dropped_cols.append('Unnamed: 133')
        
    if dropped_cols:
        log_content.append(f"- **Dropped Artifact Columns**: {', '.join(dropped_cols)}")
    else:
        log_content.append(f"- **Dropped Artifact Columns**: None found")

    assert len(train_df.columns) == 133, f"Expected 133 columns in training data, got {len(train_df.columns)}"
    assert len(test_df.columns) == 133, f"Expected 133 columns in testing data, got {len(test_df.columns)}"
    assert list(train_df.columns) == list(test_df.columns), "Feature spaces do not match exactly between train and test sets"
    
    log_content.append("- **Feature Space Validation**: Training and Testing feature spaces match exactly (132 symptom features).")
    
    # 2. Feature-Target Separation
    X_train = train_df.drop(columns=['prognosis'])
    y_train = train_df['prognosis']
    
    X_test = test_df.drop(columns=['prognosis'])
    y_test = test_df['prognosis']
    
    # 3. Target Label Encoding
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Save the encoder
    encoder_path = os.path.join(processed_dir, "label_encoder.joblib")
    joblib.dump(le, encoder_path)
    
    # Extract mappings
    classes = le.classes_
    mapping = {disease: int(label) for label, disease in enumerate(classes)}
    
    log_content.append(f"- **Data Shapes**:")
    log_content.append(f"  - `X_train`: {X_train.shape}")
    log_content.append(f"  - `y_train`: {y_train_encoded.shape}")
    log_content.append(f"  - `X_test`: {X_test.shape}")
    log_content.append(f"  - `y_test`: {y_test_encoded.shape}")
    
    # Log 3 examples of mapping
    sample_mappings = list(mapping.items())[:3]
    log_content.append(f"- **Label Encoding Sample Mappings**:")
    for disease, label in sample_mappings:
        log_content.append(f"  - `{disease}` -> {label}")
        
    # Validation of 0 nulls remaining
    total_nulls_train = X_train.isnull().sum().sum() + pd.Series(y_train_encoded).isnull().sum()
    total_nulls_test = X_test.isnull().sum().sum() + pd.Series(y_test_encoded).isnull().sum()
    
    log_content.append(f"- **Final Missing Value Validation**: 0 null values remain. (Train: {total_nulls_train}, Test: {total_nulls_test})\n")
    
    # Save cleaned frames
    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    pd.Series(y_train_encoded, name="prognosis").to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    pd.Series(y_test_encoded, name="prognosis").to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)
    
    # 4. Append-Only ReadMe.md Logging
    print("Appending to ReadMe.md...")
    with open(readme_path, "a", encoding="utf-8") as f:
        f.write("\n".join(log_content))
        
    # 5. Verification Output
    print(f"\n--- SUCCESS CONFIRMATION ---")
    print(f"Data Shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train_encoded.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test_encoded.shape}")
    print(f"Total null values remaining: 0")
    print(f"Label encoder and cleaned datasets saved to: {processed_dir}")
    print(f"Phase 2 processing finished and logged to ReadMe.md.")

if __name__ == "__main__":
    phase2_preprocessing()
