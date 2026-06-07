import pandas as pd
import os
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np

def phase3_svm():
    base_dir = r"d:\Code\Projects\Internships\CodeAlpha\CodeAlpha_Desease_Detection-Model"
    processed_dir = os.path.join(base_dir, "pipeline_tabular", "data", "processed")
    readme_path = os.path.join(base_dir, "ReadMe.md")
    
    print("Loading preprocessed datasets and encoder...")
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv"))['prognosis']
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv"))['prognosis']
    
    le = joblib.load(os.path.join(processed_dir, "label_encoder.joblib"))
    
    print("Initializing and training SVC Baseline...")
    svc = SVC(kernel='linear', C=1.0, random_state=42)
    svc.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = svc.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    precision_mac, recall_mac, f1_mac, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    precision_wt, recall_wt, f1_wt, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # We want classification report with target names decoded
    # Determine the unique classes in y_test to pass to target_names
    # Actually, if we use all classes from the encoder, we must ensure all are passed or we use labels=range(len(classes))
    all_classes = le.classes_
    target_names = [str(c) for c in all_classes]
    
    class_report_str = classification_report(y_test, y_pred, labels=np.arange(len(all_classes)), target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(all_classes)))
    
    log_content = ["\n## Phase 3: Baseline Model Training (Linear SVM)\n"]
    
    log_content.append("### Hyperparameters")
    log_content.append("- **Model**: Support Vector Classifier (SVC)")
    log_content.append("- **Kernel**: Linear (`kernel='linear'`)")
    log_content.append("- **C**: 1.0")
    log_content.append("- **Random State**: 42\n")
    
    log_content.append("### Global Metrics")
    log_content.append(f"- **Global Accuracy**: {acc:.4f}")
    log_content.append(f"- **Macro Average**: Precision={precision_mac:.4f}, Recall={recall_mac:.4f}, F1-Score={f1_mac:.4f}")
    log_content.append(f"- **Weighted Average**: Precision={precision_wt:.4f}, Recall={recall_wt:.4f}, F1-Score={f1_wt:.4f}\n")
    
    log_content.append("### Detailed Classification Report")
    log_content.append("```text\n" + class_report_str + "\n```\n")
    
    log_content.append("### Full Confusion Matrix")
    log_content.append("Rows correspond to True labels, Columns correspond to Predicted labels (0-40).")
    
    # Convert CM to a nice string representation
    # To keep it legible in markdown, we can format it nicely
    cm_str = np.array2string(cm, max_line_width=200, separator=', ')
    log_content.append("```text\n" + cm_str + "\n```\n")
    
    # Write to ReadMe.md
    print("Appending results to ReadMe.md...")
    with open(readme_path, "a", encoding="utf-8") as f:
        f.write("\n".join(log_content))
        
    print("\n--- PHASE 3 METRICS SUMMARY ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_mac:.4f}")
    print(f"Weighted F1: {f1_wt:.4f}")
    print("Appended detailed report and confusion matrix to ReadMe.md successfully.")
    
if __name__ == "__main__":
    phase3_svm()
