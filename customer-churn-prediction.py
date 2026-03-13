import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("customer_churn.csv")

print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Data preprocessing
le = LabelEncoder()

data["ContractType"] = le.fit_transform(data["ContractType"])
data["InternetService"] = le.fit_transform(data["InternetService"])
data["Churn"] = le.fit_transform(data["Churn"])

# Feature selection
X = data[["Age","Tenure","MonthlyCharges","ContractType","InternetService","SupportCalls"]]
y = data["Churn"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

# Visualization
plt.scatter(data["Tenure"], data["MonthlyCharges"])
plt.xlabel("Tenure")
plt.ylabel("Monthly Charges")
plt.title("Tenure vs Monthly Charges")
plt.show()