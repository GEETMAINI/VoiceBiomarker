import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit App
st.title("🌿 Parkinson's Disease Prediction")
st.write(
    "This application uses machine learning models to predict the likelihood of Parkinson's disease based on voice metrics. "
    "Upload your training dataset to train models, then provide a single-entry CSV file for prediction."
)

# File upload for training
uploaded_training_file = st.file_uploader("🔄 Upload your dataset for training", type=["csv"])
if uploaded_training_file is not None:
    # Load and preprocess training data
    data = pd.read_csv(uploaded_training_file)

    st.subheader("Data Preprocessing")
    st.write("Preparing the data by dropping unnecessary columns and setting up target variables.")

    # Drop unnecessary columns
    if 'name' in data.columns:
        data = data.drop(columns=['name'])
    data['status'] = (data['status'] >= 0.7).astype(int)

    # Correlation heatmap (optional for the user to see preprocessing insights)
    st.subheader("📊 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Drop highly correlated features
    correlation_matrix = data.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    threshold = 0.9
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    data = data.drop(columns=to_drop)

    # Prepare features and target variable
    X = data.drop(columns=['status']).values
    y = data['status'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(random_state=42, probability=True)
    }

    # Train all models and keep the best one based on accuracy
    best_model_name = None
    best_model = None
    best_accuracy = 0

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
            best_model = model

    st.success(f"Models trained successfully! Best model: **{best_model_name}** with accuracy: {best_accuracy * 100:.2f}%.")

# File upload for single-entry prediction
uploaded_single_entry = st.file_uploader("📁 Upload a single-entry CSV file for prediction", type=["csv"])
if uploaded_single_entry is not None:
    if best_model is None:
        st.error("Please upload a training dataset and train the models first.")
    else:
        # Load and preprocess the single entry
        single_entry = pd.read_csv(uploaded_single_entry)

        # Ensure the columns match the training dataset structure
        missing_columns = set(data.drop(columns=['status']).columns) - set(single_entry.columns)
        if missing_columns:
            st.error(f"The uploaded file is missing the following columns: {', '.join(missing_columns)}")
        else:
            single_entry = single_entry[data.drop(columns=['status']).columns]

            # Scale features to match the training scaler
            single_entry_scaled = scaler.transform(single_entry)

            # Predict using the best model
            prediction = best_model.predict(single_entry_scaled)[0]  # Get the prediction (0 or 1)
            probability = best_model.predict_proba(single_entry_scaled)[0]  # Get probability scores
            result = "Parkinson" if prediction == 1 else "No Parkinson"

            st.subheader("🚀 Prediction Result")
            st.write(f"The uploaded entry is predicted to be: **{result}**.")
            st.write(f"Confidence: {probability[prediction] * 100:.2f}%")
