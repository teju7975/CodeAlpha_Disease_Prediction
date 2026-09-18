
import streamlit as st
import pandas as pd

from disease_prediction import (
    load_and_preprocess_data,
    train_and_evaluate_models
)

st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Disease Prediction System")
st.write(
    "Machine Learning based educational demonstration "
    "using Heart Disease, Diabetes and Breast Cancer datasets."
)

st.warning(
    "⚠️ This application is for educational/demo purposes only "
    "and is not a medical diagnosis tool."
)


@st.cache_resource
def prepare_model(dataset_name):
    X_train, X_test, y_train, y_test, features, scaler = \
        load_and_preprocess_data(dataset_name)

    results, models, best_model_name = train_and_evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
        features,
        dataset_name
    )

    return models[best_model_name], scaler, features, results, best_model_name


dataset = st.selectbox(
    "Select Dataset",
    ["heart", "diabetes", "breast_cancer"]
)

st.info(f"Selected Dataset: {dataset.replace('_', ' ').title()}")

model, scaler, features, results, best_model_name = prepare_model(dataset)

st.success(f"Best Model: {best_model_name}")

st.subheader("Enter Patient/Clinical Values")

inputs = {}

if dataset == "heart":

    inputs["age"] = st.number_input(
        "Age", min_value=1, max_value=120, value=50
    )

    inputs["sex"] = st.selectbox(
        "Sex (0 = Female, 1 = Male)",
        [0, 1]
    )

    inputs["cp"] = st.number_input(
        "Chest Pain Type (CP)",
        min_value=1,
        max_value=4,
        value=1
    )

    inputs["trestbps"] = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120
    )

    inputs["chol"] = st.number_input(
        "Serum Cholesterol",
        min_value=50,
        max_value=600,
        value=200
    )

    inputs["fbs"] = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1]
    )

    inputs["restecg"] = st.number_input(
        "Resting ECG Result",
        min_value=0,
        max_value=2,
        value=0
    )

    inputs["thalach"] = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )

    inputs["exang"] = st.selectbox(
        "Exercise Induced Angina",
        [0, 1]
    )

    inputs["oldpeak"] = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0
    )

    inputs["slope"] = st.number_input(
        "Slope",
        min_value=1,
        max_value=3,
        value=2
    )

    inputs["ca"] = st.number_input(
        "Number of Major Vessels (CA)",
        min_value=0,
        max_value=3,
        value=0
    )

    inputs["thal"] = st.number_input(
        "Thal",
        min_value=0,
        max_value=7,
        value=3
    )


elif dataset == "diabetes":

    inputs["pregnancies"] = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    inputs["glucose"] = st.number_input(
        "Glucose",
        min_value=0,
        max_value=300,
        value=120
    )

    inputs["blood_pressure"] = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70
    )

    inputs["skin_thickness"] = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )

    inputs["insulin"] = st.number_input(
        "Insulin",
        min_value=0,
        max_value=900,
        value=80
    )

    inputs["bmi"] = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

    inputs["pedigree_function"] = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    inputs["age"] = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )


else:

    feature_labels = {
        "mean radius": "Mean Radius",
        "mean texture": "Mean Texture",
        "mean perimeter": "Mean Perimeter",
        "mean area": "Mean Area",
        "mean smoothness": "Mean Smoothness",
        "mean compactness": "Mean Compactness",
        "mean concavity": "Mean Concavity",
        "mean concave points": "Mean Concave Points",
        "mean symmetry": "Mean Symmetry",
        "mean fractal dimension": "Mean Fractal Dimension"
    }

    for feature in features:
        label = feature_labels.get(feature, feature.title())

        inputs[feature] = st.number_input(
            label,
            value=0.0,
            format="%.6f"
        )


if st.button("🔍 Predict", type="primary"):

    input_df = pd.DataFrame(
        [[inputs[feature] for feature in features]],
        columns=features
    )

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result")

    if dataset == "heart":

        if prediction == 1:
            st.error("Prediction: Disease Present")
        else:
            st.success("Prediction: No Disease Detected")

    elif dataset == "diabetes":

        if prediction == 1:
            st.error("Prediction: Diabetes Class")
        else:
            st.success("Prediction: Non-Diabetes Class")

    else:

        if prediction == 0:
            st.error("Prediction: Malignant Class")
        else:
            st.success("Prediction: Benign Class")

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_scaled)[0]

        st.write("Prediction Probability:")

        probability_df = pd.DataFrame({
            "Class": range(len(probability)),
            "Probability": probability
        })

        st.dataframe(probability_df, use_container_width=True)


st.subheader("Model Performance")

st.dataframe(
    results,
    use_container_width=True
)

st.caption(
    "For academic demonstration only. Do not use this application "
    "to make real medical decisions."
)
