import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os

st.set_page_config(page_title="AI Diagnostic Symptom Screener", layout="centered")

@st.cache_resource
def load_assets():
    base_dir = r"d:\Code\Projects\Internships\CodeAlpha\CodeAlpha_Desease_Detection-Model"
    models_dir = os.path.join(base_dir, "models")
    processed_dir = os.path.join(base_dir, "pipeline_tabular", "data", "processed")
    
    # Load Models
    with open(os.path.join(models_dir, "disease_svm_model.pkl"), "rb") as f:
        svm_model = pickle.load(f)
        
    with open(os.path.join(models_dir, "disease_xgb_model.pkl"), "rb") as f:
        xgb_model = pickle.load(f)
        
    # Load Label Encoder
    with open(os.path.join(models_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
        
    # Dynamically extract the exact 132 feature names from the training set
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"), nrows=0)
    symptoms_list = X_train.columns.tolist()
    
    return svm_model, xgb_model, label_encoder, symptoms_list

st.title("🏥 AI Diagnostic Symptom Screener")
st.markdown("Select your symptoms below to receive an AI-powered diagnostic prediction.")

try:
    svm_model, xgb_model, label_encoder, symptoms_list = load_assets()
    
    # UI Component: Model Selection
    model_choice = st.radio(
        "Select Diagnostic Engine:",
        ("Linear SVM Baseline", "XGBoost Advanced Ensemble"),
        horizontal=True
    )
    
    # UI Component: Multi-select for Symptoms
    selected_symptoms = st.multiselect(
        "Select your symptoms (132 available):",
        options=symptoms_list,
        default=None,
        help="Select all symptoms you are currently experiencing."
    )
    
    if st.button("Predict Disease", type="primary"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom for the diagnostic engine to run.")
        else:
            # Vector Reconstruction Engine
            # Initialize 132-dim array of zeros
            input_vector = np.zeros(len(symptoms_list))
            
            # Toggle chosen index positions to 1
            for symptom in selected_symptoms:
                if symptom in symptoms_list:
                    idx = symptoms_list.index(symptom)
                    input_vector[idx] = 1.0
                    
            # Convert to DataFrame to match model expected feature names (crucial for XGBoost)
            input_df = pd.DataFrame([input_vector], columns=symptoms_list)
            
            # Route to chosen model
            if model_choice == "Linear SVM Baseline":
                prediction_encoded = svm_model.predict(input_df)[0]
            else:
                prediction_encoded = xgb_model.predict(input_df)[0]
                
            # Decode the target string
            predicted_disease = label_encoder.inverse_transform([prediction_encoded])[0]
            
            st.success(f"### Predicted Diagnosis: **{predicted_disease}**")
            st.info(f"**Model Engine Used**: {model_choice}\n\n**Symptoms Provided**: {', '.join(selected_symptoms)}")
            
except Exception as e:
    st.error(f"Error loading dependencies or executing prediction: {e}")
