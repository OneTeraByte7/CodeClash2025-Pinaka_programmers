import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


MODEL_PATH = "collision_risk_model.pkl"
DATA_PATH = "risk_assessment_dataset.csv"

try:
    model = joblib.load(MODEL_PATH)
    print(" Model Loaded Successfully!")
except FileNotFoundError:
    print(" Error: Trained model not found! Ensure 'collision_risk_model.pkl' exists.")
    exit()


try:
    df = pd.read_csv(DATA_PATH)
    print(f" Data Loaded Successfully! Total records: {len(df)}")
except FileNotFoundError:
    print(" Error: 'risk_assessment_dataset.csv' not found!")
    exit()


required_columns = ["Distance", "Velocity", "Angle", "Vehicle Speed", "Risk Level"]
if not all(col in df.columns for col in required_columns):
    print(" Error: CSV must contain columns: Distance, Velocity, Angle, Vehicle Speed, Risk Level")
    exit()


X = df[["Distance", "Velocity", "Angle", "Vehicle Speed"]].values
y_actual = df["Risk Level"].values  

df["Predicted Risk"] = model.predict(X)


accuracy = accuracy_score(y_actual, df["Predicted Risk"]) * 100


total_records = len(df)
risky_count = (df["Predicted Risk"] == 1).sum()
safe_count = total_records - risky_count
risk_percentage = (risky_count / total_records) * 100 if total_records > 0 else 0


risky_data = df[df["Predicted Risk"] == 1]
safe_data = df[df["Predicted Risk"] == 0]

avg_risky_distance = risky_data["Distance"].mean()
avg_safe_distance = safe_data["Distance"].mean()

avg_risky_speed = risky_data["Vehicle Speed"].mean()
avg_safe_speed = safe_data["Vehicle Speed"].mean()

most_common_risk_factor = risky_data.drop(columns=["Predicted Risk"]).mean().idxmax()

# Print insights
print("\n1. Risk Assessment Analysis")
print(f"2. Total Records: {total_records}")
print(f"3. Risky Cases: {risky_count} ({round(risk_percentage, 2)}%)")
print(f"4. Safe Cases: {safe_count} ({round(100 - risk_percentage, 2)}%)")
print(f"5. Model Accuracy: {round(accuracy, 2)}%")
print(f"6. Avg Distance (Risky): {round(avg_risky_distance, 2)}m | (Safe): {round(avg_safe_distance, 2)}m")
print(f"7. Avg Speed (Risky): {round(avg_risky_speed, 2)} km/h | (Safe): {round(avg_safe_speed, 2)} km/h")
print(f"8. Most Common Risk Factor: {most_common_risk_factor}")
