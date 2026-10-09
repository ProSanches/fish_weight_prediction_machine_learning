"""Анализ ошибок лучшей модели (E3-hgb)."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocess import handle_outliers


def main():
    df = pd.read_csv("data/fish_participant.csv")
    df = handle_outliers(df)

    test_idx = np.intersect1d(np.load("configs/test_indices.npy"), df.index.values)
    test_df = df.loc[test_idx]

    X_test = test_df.drop(columns=["Weight"])
    y_test = test_df["Weight"]

    model = joblib.load("models/best_model.joblib")
    pred = model.predict(X_test)
    residuals = y_test.values - pred

    # --- График остатков ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(pred, residuals, alpha=0.6)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Предсказание, г")
    axes[0].set_ylabel("Остаток, г")
    axes[0].set_title("Остатки vs предсказание")

    axes[1].scatter(y_test, pred, alpha=0.6)
    lim = [y_test.min(), y_test.max()]
    axes[1].plot(lim, lim, "r--", label="y = x")
    axes[1].set_xlabel("Истина, г")
    axes[1].set_ylabel("Предсказание, г")
    axes[1].set_title("Предсказание vs истина")
    axes[1].legend()

    plt.tight_layout()
    Path("reports").mkdir(exist_ok=True)
    plt.savefig("reports/residuals_plot.png", dpi=120)
    plt.close()

    # --- Топ-5 крупнейших ошибок ---
    err = test_df.copy()
    err["y_pred"] = pred
    err["abs_error"] = np.abs(residuals)

    print("=== Топ-5 крупнейших ошибок ===")
    cols = ["Species", "Length3", "Height", "Width", "Weight", "y_pred", "abs_error"]
    print(err.nlargest(5, "abs_error")[cols].to_string())

    # --- MAE по видам ---
    print("\n=== MAE по видам ===")
    print(err.groupby("Species")["abs_error"].mean().round(1).sort_values(ascending=False))

    # --- MAE по размерным корзинам ---
    err["size_bin"] = pd.cut(
        err["Length3"],
        bins=[0, 25, 35, 45, 100],
        labels=["<25", "25–35", "35–45", ">45"],
    )
    print("\n=== MAE по размерным диапазонам Length3 ===")
    print(err.groupby("size_bin", observed=True)["abs_error"].mean().round(1))

    # --- Проверка физического смысла ---
    print(f"\nОтрицательных предсказаний: {(pred < 0).sum()}")
    print(f"Мин/Макс предсказание: {pred.min():.1f} / {pred.max():.1f}")
    print(f"Мин/Макс истина:       {y_test.min():.1f} / {y_test.max():.1f}")


if __name__ == "__main__":
    main()