import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
print("✅ Model trained and saved")
