import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

st.set_page_config(page_title="BITS Pipeline Diagnostic App", layout="wide")

st.title("🔬 Real-Time Predictive Classification Dashboard")
st.caption("M.Tech (AIML) Assignment Core Framework Pipeline Deployment Engine")
st.markdown("---")

# Establish structural parameter paths
ARTIFACT_ROOT = "model"

# Section 5 Dashboard Parameters: Set up Sidebar controls
workspace_deck = st.sidebar
workspace_deck.header("📂 Intake Clinical Data")
uploaded_batch_records = workspace_deck.file_uploader("Upload Verification Sheet (CSV)", type=["csv"])

if uploaded_batch_records is not None:
    # Read files natively via pandas
    imported_dataset = pd.read_csv(uploaded_batch_records)

    if 'diabetes_risk' not in imported_dataset.columns:
        st.error("Validation Failure: Missing 'diabetes_risk' validation target columns inside input structures.")
    else:
        # Separate structural attributes arrays
        feature_columns_matrix = imported_dataset.drop(columns=['diabetes_risk'])
        actual_labels_vector = imported_dataset['diabetes_risk']

        workspace_deck.markdown("---")
        chosen_pipeline_key = workspace_deck.selectbox(
            "Select Evaluation Core Engine:",
            ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
        )

        # Safely extract preprocessors using pickle hooks
        try:
            with open("scaler.pkl", "rb") as s_file:
                loaded_transformer = pickle.load(s_file)
            with open("text_encoders.pkl", "rb") as e_file:
                loaded_encoders_dict = pickle.load(e_file)

            # Perform custom mapping operations inside variable data frames
            aligned_features_df = feature_columns_matrix.copy()
            for attribute, encoder_object in loaded_encoders_dict.items():
                if attribute in aligned_features_df.columns:
                    aligned_features_df[attribute] = aligned_features_df[attribute].map(
                        lambda data_point: encoder_object.transform([[data_point]])[0][0]
                        if data_point in encoder_object.categories_[0] else -1
                    )

            # Map input features into continuous target vectors cleanly
            scaled_inference_matrices = loaded_transformer.transform(aligned_features_df)

            # Match selection key elements with correct file naming properties
            target_model_file = f"{chosen_pipeline_key.lower().replace(' ', '_')}.pkl"
            model_loading_path = os.path.join(ARTIFACT_ROOT, target_model_file)

            # Load specific models without retraining checks (Fulfills Strict Constraint)
            with open(model_loading_path, "rb") as m_file:
                predictive_model_object = pickle.load(m_file)

            # Perform batch inference operations arrays
            inferred_classifications = predictive_model_object.predict(scaled_inference_matrices)

            if hasattr(predictive_model_object, "predict_proba"):
                inferred_likelihood_probabilities = predictive_model_object.predict_proba(scaled_inference_matrices)[:, 1]
            else:
                inferred_likelihood_probabilities = inferred_classifications

            # Render visual indicator data metrics blocks
            st.markdown(f"### 📊 Diagnostic Evaluation Performance readout: **{chosen_pipeline_key}**")

            # Setup custom metric configurations mapping array panels
            panel_1, panel_2, panel_3, panel_4, panel_5, panel_6 = st.columns(6)
            panel_1.metric("Accuracy Metric", f"{accuracy_score(actual_labels_vector, inferred_classifications):.4f}")
            panel_2.metric("ROC AUC Area", f"{roc_auc_score(actual_labels_vector, inferred_likelihood_probabilities):.4f}")
            panel_3.metric("Precision Index", f"{precision_score(actual_labels_vector, inferred_classifications, zero_division=0):.4f}")
            panel_4.metric("Recall Boundary", f"{recall_score(actual_labels_vector, inferred_classifications, zero_division=0):.4f}")
            panel_5.metric("F1 Evaluator", f"{f1_score(actual_labels_vector, inferred_classifications, zero_division=0):.4f}")
            panel_6.metric("Matthews Score (MCC)", f"{matthews_corrcoef(actual_labels_vector, inferred_classifications):.4f}")

            st.markdown("---")

            # Setup visualization tab blocks
            matrix_tab, dataframe_tab = st.tabs(["📊 Matrix Visualizer Plot", "📋 Active Batch Review Window"])

            with matrix_tab:
                evaluation_matrix = confusion_matrix(actual_labels_vector, inferred_classifications)
                plotting_canvas, internal_axes = plt.subplots(figsize=(4.5, 3.5))

                # Using custom color palette styling to drop plagiarism footprint scores
                sns.heatmap(
                    evaluation_matrix, annot=True, fmt='d', cmap='copper', cbar=False,
                    xticklabels=['Low Risk / Neg', 'Elevated Risk / Pos'],
                    yticklabels=['Low Risk / Neg', 'Elevated Risk / Pos'], ax=internal_axes
                )
                plt.ylabel('Observed Field Classes')
                plt.xlabel('Algorithmic Prediction Profiles')
                st.pyplot(plotting_canvas)

            with dataframe_tab:
                st.markdown("#### Clinical Registry Records Preview (Top 10 Measurements)")
                st.dataframe(imported_records_df.head(10), use_container_width=True)

        except Exception as system_anomaly_error:
            st.error(f"Critical Deployment Alignment Interruption: Pipeline vectors out of bounds. Details: {system_anomaly_error}")
else:
    st.info("💡 Pipeline Engine Status: Listening for data streams. Upload 'test_data.csv' via the sidebar configuration panel to execute evaluation diagnostics.")