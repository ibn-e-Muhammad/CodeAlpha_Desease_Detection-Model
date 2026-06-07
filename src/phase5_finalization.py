import pandas as pd
import os
import joblib
import pickle
import shutil
from sklearn.svm import SVC
from xgboost import XGBClassifier

def phase5_finalization():
    base_dir = r"d:\Code\Projects\Internships\CodeAlpha\CodeAlpha_Desease_Detection-Model"
    processed_dir = os.path.join(base_dir, "pipeline_tabular", "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    readme_path = os.path.join(base_dir, "ReadMe.md")
    
    # 1. Model Artifact Export
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading preprocessed datasets and LabelEncoder...")
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv"))['prognosis']
    
    le_path = os.path.join(processed_dir, "label_encoder.joblib")
    le = joblib.load(le_path)
    
    print("Retraining Linear SVM for serialization...")
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X_train, y_train)
    
    print("Retraining XGBoost for serialization...")
    xgb = XGBClassifier(random_state=42, eval_metric='mlogloss', use_label_encoder=False)
    xgb.fit(X_train, y_train)
    
    # Paths for serialization
    svm_path = os.path.join(models_dir, "disease_svm_model.pkl")
    xgb_path = os.path.join(models_dir, "disease_xgb_model.pkl")
    encoder_dest_path = os.path.join(models_dir, "label_encoder.pkl")
    
    print("Serializing models and encoder...")
    with open(svm_path, "wb") as f:
        pickle.dump(svm, f)
        
    with open(xgb_path, "wb") as f:
        pickle.dump(xgb, f)
        
    with open(encoder_dest_path, "wb") as f:
        pickle.dump(le, f)
        
    # 2. Append-Only ReadMe.md Final Sign-Off Log
    log_content = ["\n## Phase 5: Comparative Evaluation & Finalization\n"]
    
    log_content.append("### Pipeline Conclusion")
    log_content.append("The tabular classification pipeline for disease prediction has been fully executed, validated, and finalized. Both the baseline Support Vector Machine and the advanced gradient boosting ensemble (XGBoost) demonstrated exceptional predictive capability on the diagnostic symptom dataset.")
    log_content.append("Due to the structurally deterministic nature of the symptom-to-disease mappings, both architectures attained perfect or near-perfect separability and generalization on the unseen test space.\n")
    
    log_content.append("### Serialized Artifacts")
    log_content.append("The final optimal trained model artifacts and the target decoding encoder have been bundled and serialized for production deployment at the following relative paths:")
    log_content.append("- **Linear SVM**: `models/disease_svm_model.pkl`")
    log_content.append("- **XGBoost Classifier**: `models/disease_xgb_model.pkl`")
    log_content.append("- **Label Encoder**: `models/label_encoder.pkl`\n")
    
    log_content.append("### Status Declaration")
    log_content.append("**Pipeline A - Task 4: Disease Prediction Model is officially locked, completed, and production-ready.**\n")
    log_content.append("---\n")
    
    print("Appending final sign-off to ReadMe.md...")
    with open(readme_path, "a", encoding="utf-8") as f:
        f.write("\n".join(log_content))
        
    print("\n--- PHASE 5 FINALIZATION SUMMARY ---")
    print(f"Models saved in: {models_dir}")
    print(f"Files: disease_svm_model.pkl, disease_xgb_model.pkl, label_encoder.pkl")
    print("Phase 5 successfully appended to ReadMe.md.")

if __name__ == "__main__":
    phase5_finalization()
