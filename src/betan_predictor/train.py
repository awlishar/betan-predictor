"""
# Example usage
main_dirs = [
    "/scratch/project_2009007/data_JET_1H/success",
    "/scratch/project_2009007/data_JET_2H/success",
    "/scratch/project_2009007/data_JET_3H/success",
]

X, y = load_dataset(main_dirs)
X_train, X_test, y_train, y_test = preprocess_dataset(X, y)
model = train_model(X_train, y_train)
y_pred = model.predict(X_test)

"""

import os
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


def load_dataset(main_dirs: list):
    X = []
    y = []

    for main_dir in main_dirs:
        print(f"Loading data from: {main_dir}")
        for subdir in os.listdir(main_dir):
            subpath = os.path.join(main_dir, subdir)

            if not os.path.isdir(subpath):
                continue

            json_file = os.path.join(subpath, "summary.json")

            if not os.path.exists(json_file):
                continue

            with open(json_file, "r") as f:
                data = json.load(f)
            if not data.get("success", False):
                continue

            try:
                aminor = data["radius"]
                ip = data["ip"]
                bvac = data["bvac"]
                t_eped = data["t_eped"]
                n_eped = data["n_eped"]
                d_ped_te = data["d_ped_te"]
                d_ped_ne = data["d_ped_ne"]
                betan = data["betan"]
                core_slope_te = data["profile_fit"]["te"]["cte"]
                core_slope_ne = data["profile_fit"]["de"]["cde"]

                X.append([
                    aminor,
                    ip,
                    bvac,
                    t_eped,
                    n_eped,
                    d_ped_te,
                    d_ped_ne,
                    core_slope_ne,
                    betan])
                y.append(core_slope_te)

            except KeyError:
                # skip runs with missing data
                continue

    X = np.array(X)
    y = np.array(y)

    return X, y


def preprocess_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


def train_model(X_train, y_train, epochs=5000):

    model = MLPRegressor(
    hidden_layer_sizes=(9,9),
    activation="relu",
    solver="adam",
    max_iter=epochs,
    random_state=42
    )

    model.fit(X_train, y_train)
    return model


def save_model(model_dir, model, scaler):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "core_slope_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "feature_scaler.pkl"))
    return
