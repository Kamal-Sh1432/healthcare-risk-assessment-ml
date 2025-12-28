from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained artifacts
model = joblib.load("logreg_healthcare_model.pkl")
scaler = joblib.load("scaler_healthcare.pkl")
feature_columns = joblib.load("feature_columns.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    probability = None
    risk_label = None
    top_drivers = []
    recommendations = []

    # Auto-fill realistic defaults
    defaults = {
        "age": 45,
        "bmi": 26,
        "cholesterol": 190,
        "glucose": 105,
        "smoking": 0,
        "family_history": 0,
        "sleep_hours": 7,
        "diet_quality": 6
    }

    if request.method == "POST":
        # Build input dictionary with all model features
        input_data = dict.fromkeys(feature_columns, 0)

        for key in defaults:
            input_data[key] = float(request.form.get(key, defaults[key]))

        input_df = pd.DataFrame([input_data])[feature_columns]
        input_scaled = scaler.transform(input_df)

        probability = round(float(model.predict_proba(input_scaled)[0][1]), 2)

        # 🎯 Risk band logic (realistic)
        if probability < 0.30:
            risk_label = "Low"
        elif probability < 0.60:
            risk_label = "Medium"
        else:
            risk_label = "High"

        # 🔍 Top risk drivers
        if input_data["bmi"] > 30:
            top_drivers.append("High BMI")

        if input_data["glucose"] > 140:
            top_drivers.append("Elevated Blood Glucose")

        if input_data["cholesterol"] > 220:
            top_drivers.append("High Cholesterol")

        if input_data["smoking"] == 1:
            top_drivers.append("Smoking Habit")

        if input_data["sleep_hours"] < 6:
            top_drivers.append("Insufficient Sleep")

        if input_data["diet_quality"] < 4:
            top_drivers.append("Poor Diet Quality")

        # 🧠 Personalized recommendations
        if input_data["bmi"] > 30:
            recommendations.append("Follow a structured weight management plan.")

        if input_data["glucose"] > 140:
            recommendations.append("Monitor blood glucose and reduce sugar intake.")

        if input_data["cholesterol"] > 220:
            recommendations.append("Limit saturated fats and increase fiber intake.")

        if input_data["smoking"] == 1:
            recommendations.append("Consider smoking cessation programs.")

        if input_data["sleep_hours"] < 6:
            recommendations.append("Aim for 7–8 hours of quality sleep.")

        if input_data["diet_quality"] < 4:
            recommendations.append("Improve diet with fruits, vegetables, and whole grains.")

    return render_template(
        "index.html",
        probability=probability,
        risk_label=risk_label,
        top_drivers=top_drivers,
        recommendations=recommendations,
        defaults=defaults
    )

if __name__ == "__main__":
    app.run(debug=True)
