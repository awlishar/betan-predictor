"""
Example usage:
    model, scaler = load_model(model_dir)
    x = np.array([[ip, bvac, t_eped, n_eped, d_ped_te, d_ped_ne, core_slope_ne, betan]])
    x = scaler.transform(x)
    y_pred = model.predict(x)

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
import joblib


def load_model(model_dir):
    """ Given a directory, load the model and the scaler, """
    
    model = joblib.load(os.path.join(model_dir, "core_slope_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "feature_scaler.pkl"))
    return model, scaler


def plot_feature_histograms(X):
    feature_names = [
        "aminor",
        "ip",
        "bvac",
        "t_eped",
        "n_eped",
        "d_ped_te",
        "d_ped_ne",
        "core_slope_ne",
        "betan"
    ]

    n_features = X.shape[1]

    fig, axes = plt.subplots(3, 3, figsize=(12,10))
    axes = axes.flatten()

    for i in range(n_features):
        axes[i].hist(X[:, i], bins=40)
        axes[i].set_title(feature_names[i])
        axes[i].set_ylabel("Count")


    plt.tight_layout()
    plt.show()


def regression_diagnostics(y_true, y_pred):

    residuals = y_true - y_pred

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    fig, axes = plt.subplots(1, 3, figsize=(15,5))

    # -------------------------
    # 1. True vs Predicted
    # -------------------------
    ax = axes[0]

    ax.scatter(y_true, y_pred, alpha=0.6)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))

    ax.plot([min_val, max_val], [min_val, max_val], "--")

    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"True vs Predicted\nR²={r2:.3f}")
    ax.grid(True)

    # -------------------------
    # 2. Residuals vs Prediction
    # -------------------------
    ax = axes[1]

    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, linestyle="--")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Prediction")
    ax.grid(True)

    # -------------------------
    # 3. Residual Distribution
    # -------------------------
    ax = axes[2]

    ax.hist(residuals, bins=100)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution\nMSE={mse:.3e}")
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, X_test, y_test):

    feature_names = [
        "aminor",
        "ip",
        "bvac",
        "t_eped",
        "n_eped",
        "d_ped_te",
        "d_ped_ne",
        "core_slope_ne",
        "betan"
    ]

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42
    )

    importance = result.importances_mean
    std = result.importances_std

    # Sort features by importance
    idx = np.argsort(importance)

    plt.figure(figsize=(8,5))

    plt.barh(np.array(feature_names)[idx], importance[idx], xerr=std[idx])

    plt.xlabel("Permutation Importance")
    plt.title("Feature Importance for core_slope_te Prediction")

    plt.tight_layout()
    plt.show()
