import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1️⃣ Load dataset
df = pd.read_csv("healthcare_disease_dataset.csv")

# 2️⃣ Split features & target
X = df.drop("disease", axis=1)
y = df["disease"]

feature_columns = X.columns.tolist()

# 3️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5️⃣ Create & TRAIN model (🔥 ORDER MATTERS)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

# 6️⃣ Evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# 7️⃣ Save artifacts
joblib.dump(model, "logreg_healthcare_model.pkl")
joblib.dump(scaler, "scaler_healthcare.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")

print("✅ Model and Scaler Saved Successfully!")
