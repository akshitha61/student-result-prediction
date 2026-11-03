
import numpy as np
import pandas as pd
import joblib
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate student dataset
np.random.seed(42)

data = pd.DataFrame({
    "internal_marks": np.random.randint(20, 101, 500),
    "assignment_marks": np.random.randint(20, 101, 500),
    "previous_score": np.random.randint(20, 101, 500),
    "attendance": np.random.randint(40, 101, 500)
})

# Calculate result
score = (
    0.3 * data["internal_marks"]
    + 0.2 * data["assignment_marks"]
    + 0.3 * data["previous_score"]
    + 0.2 * data["attendance"]
)

data["result"] = (score >= 50).astype(int)

# Prepare features and target
X = data.drop("result", axis=1)
y = data["result"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save model
joblib.dump(model, "student_result_model.pkl")

# Save metrics
with open("metrics.json", "w") as file:
    json.dump({"accuracy": accuracy}, file, indent=4)

print("Student Result Prediction Model Training Completed")
print("Total Records:", len(data))
print("Model Accuracy:", round(accuracy, 4))
print("Model saved successfully")
print("Metrics saved successfully")
