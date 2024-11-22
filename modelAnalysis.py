# Import necessary libraries
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

# Load the data
data = pd.read_csv('dataset.csv')

# Preprocess the data
data = data.drop(columns=['name'])
data['status'] = (data['status'] >= 0.7).astype(int)

# Display correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Features", fontsize=16)
plt.show()

# Drop highly correlated features
correlation_matrix = data.corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
threshold = 0.9
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
data = data.drop(columns=to_drop)

# Prepare the features and target variable
X = data.drop(columns=['status']).values
y = data['status'].values

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Parkinson', 'Parkinson'], yticklabels=['No Parkinson', 'Parkinson'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Logistic Regression
print("\nLogistic Regression Model")
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg) * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred_log_reg))
plot_confusion_matrix(y_test, y_pred_log_reg, "Logistic Regression")

# Decision Tree
print("\nDecision Tree Model")
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_decision_tree = decision_tree.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_decision_tree) * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred_decision_tree))
plot_confusion_matrix(y_test, y_pred_decision_tree, "Decision Tree")

# Random Forest
print("\nRandom Forest Model")
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)
y_pred_random_forest = random_forest.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_random_forest) * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred_random_forest))
plot_confusion_matrix(y_test, y_pred_random_forest, "Random Forest")

# Support Vector Machine
print("\nSupport Vector Machine Model")
svm = SVC(random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm) * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred_svm))
plot_confusion_matrix(y_test, y_pred_svm, "Support Vector Machine")

# Plotting Model Comparison
model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Support Vector Machine']
model_accuracies = [
    accuracy_score(y_test, y_pred_log_reg),
    accuracy_score(y_test, y_pred_decision_tree),
    accuracy_score(y_test, y_pred_random_forest),
    accuracy_score(y_test, y_pred_svm)
]

plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=model_accuracies, palette="viridis")
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.show()
