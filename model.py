# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

print("Loading dataset...")

# Load dataset
df = pd.read_csv("winequality-red.csv")

# Features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- SCALER --------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------- MODEL --------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Evaluate
r2 = r2_score(y_test, model.predict(X_test_scaled))
print("R2 Score:", round(r2, 4))

# -------- SAVE TWO PICKLE FILES --------
pickle.dump(model, open("wine_model.pkl", "wb"))
pickle.dump(scaler, open("wine_scaler.pkl", "wb"))

print("✅ wine_model.pkl and wine_scaler.pkl saved successfully!")