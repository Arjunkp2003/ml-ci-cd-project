import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

model = joblib.load("model.pkl")
preds = model.predict(X_test)

rmse = mean_squared_error(y_test, preds, squared=False)
print("RMSE:", rmse)

if rmse > 1.0:
    raise ValueError("❌ Model performance below threshold")

print("✅ Model evaluation passed")
