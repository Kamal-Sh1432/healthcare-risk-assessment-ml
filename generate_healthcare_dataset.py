import numpy as np
import pandas as pd

np.random.seed(42)
n = 10000

# -----------------------------
# Generate base features
# -----------------------------
age = np.random.normal(45, 12, n).clip(18, 90)
gender = np.random.choice([0, 1], n)
bmi = np.random.normal(27, 4, n).clip(15, 45)

systolic_bp = np.random.normal(125, 18, n)
diastolic_bp = np.random.normal(82, 12, n)

cholesterol = np.random.normal(205, 35, n).clip(120, 350)
glucose = np.random.normal(115, 30, n).clip(60, 250)

smoking = np.random.choice([0, 1], n, p=[0.7, 0.3])
alcohol = np.random.choice([0, 1], n, p=[0.6, 0.4])

physical_activity = np.random.normal(3, 1.5, n).clip(0, 10)
family_history = np.random.choice([0, 1], n, p=[0.6, 0.4])

stress_level = np.random.randint(1, 11, n)
heart_rate = np.random.normal(75, 10, n)

hba1c = np.random.normal(5.7, 1.1, n).clip(4, 12)
waist_circumference = np.random.normal(92, 14, n).clip(60, 150)

sleep_hours = np.random.normal(7, 1.2, n).clip(3, 10)
diet_quality = np.random.randint(1, 11, n)

bp_medication = np.random.choice([0, 1], n, p=[0.8, 0.2])
sugar_intake = np.random.normal(55, 20, n).clip(0, 150)

# -----------------------------
# Risk score (STRONGER SIGNAL)
# -----------------------------
risk_score = (
    0.04 * age +
    0.06 * bmi +
    0.05 * (cholesterol / 10) +
    0.06 * (glucose / 10) +
    0.8  * smoking +
    0.7  * family_history +
    0.6  * (bmi > 30) +
    0.6  * (glucose > 140) +
    0.5  * (hba1c > 6.5) +
    0.3  * bp_medication -
    0.05 * physical_activity -
    0.05 * diet_quality -
    0.04 * sleep_hours
)

# -----------------------------
# Convert risk score to probability
# -----------------------------
prob = 1 / (1 + np.exp(-0.15 * (risk_score - np.mean(risk_score))))

# -----------------------------
# Anchor extremes (KEY FIX)
# -----------------------------
high_threshold = np.percentile(risk_score, 85)
low_threshold = np.percentile(risk_score, 15)

disease = []

for rs, p in zip(risk_score, prob):
    if rs >= high_threshold:
        disease.append(1)        # definite high risk
    elif rs <= low_threshold:
        disease.append(0)        # definite low risk
    else:
        disease.append(np.random.binomial(1, p))  # uncertain middle

disease = np.array(disease)

# -----------------------------
# Create DataFrame
# -----------------------------
health_df = pd.DataFrame({
    "age": age,
    "gender": gender,
    "bmi": bmi,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "cholesterol": cholesterol,
    "glucose": glucose,
    "smoking": smoking,
    "alcohol": alcohol,
    "physical_activity": physical_activity,
    "family_history": family_history,
    "stress_level": stress_level,
    "heart_rate": heart_rate,
    "hba1c": hba1c,
    "waist_circumference": waist_circumference,
    "sleep_hours": sleep_hours,
    "diet_quality": diet_quality,
    "bp_medication": bp_medication,
    "sugar_intake": sugar_intake,
    "disease": disease
})

# -----------------------------
# Save dataset
# -----------------------------
health_df.to_csv("healthcare_disease_dataset.csv", index=False)

print("✅ Realistic healthcare dataset generated successfully!")
print("High risk %:", disease.mean())
