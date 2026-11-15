from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "salary_model.pkl")

print("SALARY PREDICTION SYSTEM")
print("-----------------------")

experience = float(input("Enter years of experience: "))

input_data = pd.DataFrame({
    "YearsExperience": [experience]
})

predicted_salary = model.predict(input_data)[0]

print("\nYears of Experience:", experience)
print("Predicted Salary: Rs.", round(predicted_salary, 2))
