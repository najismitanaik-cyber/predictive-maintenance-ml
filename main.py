print("Hello! Predictive Maintenance Project Started")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("machine_data.csv")

# Features & target
X = data[["temperature", "vibration", "runtime_hours"]]
y = data["failure"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Predict new machine data
new_data = [[85, 0.07, 280]]
result = model.predict(new_data)

if result[0] == 1:
    print("⚠️ Machine Failure Expected")
else:
    print("✅ Machine Safe")
new_data = pd.DataFrame(
    [[85, 0.07, 280]],
    columns=["temperature", "vibration", "runtime_hours"]
)

result = model.predict(new_data)

