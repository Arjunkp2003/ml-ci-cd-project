import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error

# Load test data
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# Load trained model
model = joblib.load("model.pkl")

# Predict
preds = model.predict(X_test)

# Compute RMSE (CI-safe for all sklearn versions)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)

print("RMSE:", rmse)

# Fail CI if model is bad
if rmse > 1.0:
    raise ValueError("❌ Model performance below threshold")

print("✅ Model evaluation passed")
