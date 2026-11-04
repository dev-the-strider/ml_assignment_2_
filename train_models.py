import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

def execute_training_pipeline():
    print("[LOG] Starting secure diagnostic data preprocessing routine...")

    # Read raw patient dataset matrices
    health_data = pd.read_csv("diabetes_risk.csv")

    # Extract structural identifier codes safely if present
    if 'patient_id' in health_data.columns:
        health_data = health_data.drop(columns=['patient_id'])

    # Standardize the target target vector labels into a binary matrix format
    health_data['diabetes_risk'] = health_data['diabetes_risk'].map({'Low': 0, 'Moderate': 1, 'High': 1})

    # Type-Inspection loops to isolate numeric dimensions from descriptive strings
    categorical_features = []
    numerical_features = []

    for feature_lbl in health_data.columns:
        if feature_lbl != 'diabetes_risk':
            # Safe evaluation parsing block to see if column contains textual properties
            sample_value = str(health_data[feature_lbl].dropna().iloc[0]) if health_data[feature_lbl].dropna().size > 0 else ""
            try:
                float(sample_value)
                numerical_features.append(feature_lbl)
            except ValueError:
                categorical_features.append(feature_lbl)

    print(f"[INFO] Columns parsed -> Continuous Numeric: {len(numerical_features)} | Categorical Strings: {len(categorical_features)}")

    # Clean missing fields cleanly using type-safe imputation metrics
    for str_item in categorical_features:
        most_frequent_value = health_data[str_item].mode()
        fallback_string = most_frequent_value.iloc[0] if not most_frequent_value.empty else "Unknown"
        health_data[str_item] = health_data[str_item].fillna(fallback_string).astype(str)

    for num_item in numerical_features:
        # Cast explicitly to float matrices first, then resolve missing elements using median
        health_data[num_item] = pd.to_numeric(health_data[num_item], errors='coerce')
        health_data[num_item] = health_data[num_item].fillna(health_data[num_item].median())

    # Process text dimensions via unique Ordinal encoding transformers array mappings
    map_converters = {}
    for element in categorical_features:
        encoder_instance = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        health_data[[element]] = encoder_instance.fit_transform(health_data[[element]])
        map_converters[element] = encoder_instance

    # Split design fields into array matrices
    predictive_matrices = health_data.drop(columns=['diabetes_risk'])
    target_labels = health_data['diabetes_risk']

    # Stratified custom data segmentation split (using non-standard 78/22 partition ratio)
    X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(
        predictive_matrices, target_labels,
        test_size=0.22,
        random_state=77,
        stratify=target_labels
    )

    # Scale mathematical arrays bounded strictly within a 0 to 1 range
    feature_range_scaler = MinMaxScaler()
    X_train_scaled = feature_range_scaler.fit_transform(X_train_set)
    X_test_scaled = feature_range_scaler.transform(X_test_set)

    # Export clean evaluation files to local workspace directly
    experimental_test_records = X_test_set.copy()
    experimental_test_records['diabetes_risk'] = y_test_set
    experimental_test_records.to_csv("test_data.csv", index=False)

    # Serialize model preprocessor artifacts into storage loops using pickle hooks
    with open("scaler.pkl", "wb") as scaler_file:
        pickle.dump(feature_range_scaler, scaler_file)
    with open("text_encoders.pkl", "wb") as encoding_file:
        pickle.dump(map_converters, encoding_file)

    # Algorithmic instances initialization setup blocks
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import SVC

    model_inventory = {
        "Logistic Regression": LogisticRegression(solver='liblinear', C=0.9, random_state=77),
        "Decision Tree": DecisionTreeClassifier(criterion='entropy', max_depth=8, random_state=77),
        "KNN": KNeighborsClassifier(n_neighbors=9, metric='manhattan'),
        "Naive Bayes": GaussianNB(var_smoothing=1e-09),
        "Support Vector Machine": CalibratedClassifierCV(SVC(kernel='rbf', C=1.1, random_state=77), ensemble=False)
    }

    performance_records = []
    os.makedirs("model", exist_ok=True)

    for classifier_name, algorithm_object in model_inventory.items():
        # Train classifier configurations
        algorithm_object.fit(X_train_scaled, y_train_set)

        # Calculate evaluations and inference prediction arrays
        class_predictions = algorithm_object.predict(X_test_scaled)
        class_probabilities = algorithm_object.predict_proba(X_test_scaled)[:, 1]

        # Build individual metrics values
        metric_acc = accuracy_score(y_test_set, class_predictions)
        metric_auc = roc_auc_score(y_test_set, class_probabilities)
        metric_prc = precision_score(y_test_set, class_predictions, zero_division=0)
        metric_rcl = recall_score(y_test_set, class_predictions, zero_division=0)
        metric_f1s = f1_score(y_test_set, class_predictions, zero_division=0)
        metric_mcc = matthews_corrcoef(y_test_set, class_predictions)

        performance_records.append({
            "ML Model Name": classifier_name,
            "Accuracy": round(metric_acc, 4),
            "AUC": round(metric_auc, 4),
            "Precision": round(metric_prc, 4),
            "Recall": round(metric_rcl, 4),
            "F1": round(metric_f1s, 4),
            "MCC": round(metric_mcc, 4)
        })

        # Write serialized object binaries down into target folder paths
        binary_filename = f"model/{classifier_name.lower().replace(' ', '_')}.pkl"
        with open(binary_filename, "wb") as output_file:
            pickle.dump(algorithm_object, output_file)

    # Print out metrics dataframe summary view into terminal using standard pandas string conversion
    metrics_dataframe = pd.DataFrame(performance_records)
    print("\n" + "="*80)
    print("📈 ACCUMULATED EVALUATION RESULTS MATRIX (COPY PASTE TO YOUR SUBMISSION)")
    print("="*80 + "\n")
    print(metrics_dataframe.to_string(index=False))
    print("\n" + "="*80)

if __name__ == "__main__":
    execute_training_pipeline()