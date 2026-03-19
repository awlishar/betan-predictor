import os
import json
import numpy as np
from src.betan_predictor import *


# Example usage

X = np.load()
y = np.load()

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

print("First sample features:", X[0])
print("First label:", y[0])

X_train, X_test, y_train, y_test, scaler = preprocess_dataset(X, y)

model = train_model(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))

model_dir = "model_sklearn_0242"
save_model(model_dir, model, scaler)