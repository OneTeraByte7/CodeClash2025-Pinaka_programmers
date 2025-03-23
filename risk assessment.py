import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("risk_assessment_dataset.csv")

sns.set_style("darkgrid")

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
sns.histplot(df["Distance"], bins=30, kde=True, color="blue")
plt.title("Distance Distribution")

plt.subplot(2, 2, 2)
sns.histplot(df["Velocity"], bins=30, kde=True, color="green")
plt.title("Velocity Distribution")

plt.subplot(2, 2, 3)
sns.scatterplot(x=df["Distance"], y=df["Velocity"], hue=df["Risk Level"], palette="coolwarm")
plt.title("Distance vs Velocity (Risk Level)")


plt.subplot(2, 2, 4)
sns.countplot(x=df["Risk Level"], palette="coolwarm")
plt.title("Risk Level Distribution")
plt.xticks([0, 1], ["Low Risk", "High Risk"])

plt.tight_layout()
plt.show()

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

sample_input = np.array([[20, 5, 40, 23]])  # Test scenario
risk_prediction = model.predict(sample_input)
print("Risk Level:", "High" if risk_prediction[0] == 1 else "Low")
