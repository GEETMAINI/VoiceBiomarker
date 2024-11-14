import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# Function to display confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    conf_matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Parkinson', 'Parkinson'],
                yticklabels=['No Parkinson', 'Parkinson'], ax=ax)
    ax.set_title(f'Confusion Matrix for {model_name}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    st.pyplot(fig)


# Streamlit App
st.title("🌿 Parkinson's Disease Prediction")
st.write(
    "This application uses machine learning models to predict the likelihood of Parkinson's disease based on voice metrics. Upload your dataset to begin.")

# File upload
uploaded_file = st.file_uploader("🔄 Upload your dataset", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Data preprocessing
    st.subheader("Data Preprocessing")
    st.write("Dropping unnecessary columns and setting up target variable.")

    data = data.drop(columns=['name'])
    data['status'] = (data['status'] >= 0.5).astype(int)

    # Display correlation heatmap
    st.subheader("📊 Correlation Heatmap")
    st.write("This heatmap displays the correlation between features in the dataset.")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
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

    # Models and Results
    st.subheader("🚀 Model Training and Evaluation")
    st.write(
        "Four models are trained and evaluated for accuracy. Each model's confusion matrix and classification report are displayed below.")

    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(random_state=42)
    }

    model_accuracies = {}

    for model_name, model in models.items():
        st.markdown(f"### **{model_name}**")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[model_name] = accuracy

        st.write(f"**Accuracy:** {accuracy * 100:.2f}%")
        st.write("**Classification Report:**")
        st.text(classification_report(y_test, y_pred))

        plot_confusion_matrix(y_test, y_pred, model_name)

    # Model Comparison
    st.subheader("📈 Model Accuracy Comparison")
    st.write("Bar chart comparing model accuracies to determine the best-performing model.")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(model_accuracies.keys()), y=list(model_accuracies.values()), palette="viridis", ax=ax)
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison')
    st.pyplot(fig)
