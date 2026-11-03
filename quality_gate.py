
import json

# Read model evaluation metrics
with open("metrics.json", "r") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]
required_accuracy = 0.85

print("Reading model evaluation metrics...")
print("Model Accuracy:", round(accuracy, 4))
print("Required Accuracy:", required_accuracy)

if accuracy >= required_accuracy:
    print("QUALITY GATE PASSED")
    print("Model performance satisfies the required threshold.")
else:
    print("QUALITY GATE FAILED")
    print("Model accuracy is below the required threshold.")
    raise SystemExit(1)
