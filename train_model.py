from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Salary_Data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Clean column names
df.columns = df.columns.str.strip()

# Select input and target
X = df[["YearsExperience"]]
y = df["Salary"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict salary
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n----- MODEL PERFORMANCE -----")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Save trained model
joblib.dump(model, BASE_DIR / "salary_model.pkl")

# Graph 1: Experience vs Salary
plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Actual Salary")
plt.plot(X, model.predict(X), color="red", label="Prediction Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "salary_experience.png")
plt.close()

# Graph 2: Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png")
plt.close()

print("\nModel saved successfully!")
print("Graphs saved in outputs folder.")