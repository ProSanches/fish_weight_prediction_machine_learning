"""Обучение и оценка моделей E1–E4 с логированием в MLflow."""
import argparse

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.preprocess import (
    CATEGORICAL_COLUMNS, NUMERIC_COLUMNS,
    handle_outliers, build_preprocessor,
)

RANDOM_STATE = 42

TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def load_split(extra_features=None):
    """Загружает данные, чистит выбросы, восстанавливает зафиксированное разбиение."""
    df = pd.read_csv("data/fish_participant.csv")
    df = handle_outliers(df)

    if extra_features:
        for name, fn in extra_features.items():
            df[name] = fn(df)

    test_idx = np.intersect1d(np.load("configs/test_indices.npy"), df.index.values)
    train_idx = np.intersect1d(np.load("configs/train_indices.npy"), df.index.values)

    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]

    X_train = train_df.drop(columns=["Weight"])
    y_train = train_df["Weight"]
    X_test = test_df.drop(columns=["Weight"])
    y_test = test_df["Weight"]
    return X_train, y_train, X_test, y_test


def make_pipeline(model, extra_numeric=None):
    """Собирает Pipeline с ColumnTransformer. Опционально расширяет список числовых признаков."""
    if extra_numeric is None:
        pre = build_preprocessor()
    else:
        num_cols = NUMERIC_COLUMNS + extra_numeric
        numeric = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log, validate=False)),
            ("scale", StandardScaler()),
        ])
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        pre = ColumnTransformer([
            ("num", numeric, num_cols),
            ("cat", categorical, CATEGORICAL_COLUMNS),
        ])
    return Pipeline([("preprocess", pre), ("regressor", model)])


def evaluate(model, X_train, y_train, X_test, y_test, run_name, params):
    """Обучает модель, считает CV и тестовые метрики, логирует всё в MLflow."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("random_state", RANDOM_STATE)

        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_mse = -cross_val_score(model, X_train, y_train, cv=cv,
                                  scoring="neg_mean_squared_error")
        cv_mae = -cross_val_score(model, X_train, y_train, cv=cv,
                                  scoring="neg_mean_absolute_error")
        cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")

        mlflow.log_metric("cv_mse_mean", cv_mse.mean())
        mlflow.log_metric("cv_mse_std", cv_mse.std())
        mlflow.log_metric("cv_mae_mean", cv_mae.mean())
        mlflow.log_metric("cv_mae_std", cv_mae.std())
        mlflow.log_metric("cv_r2_mean", cv_r2.mean())
        mlflow.log_metric("cv_r2_std", cv_r2.std())

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)

        print(f"[{run_name}]")
        print(f"  CV MSE:  {cv_mse.mean():.2f} ± {cv_mse.std():.2f}")
        print(f"  Test MSE: {mse:.2f}")
        print(f"  Test MAE: {mae:.2f}")
        print(f"  Test R² : {r2:.4f}")
        print()

        residuals = pd.DataFrame({
            "y_true": y_test.values,
            "y_pred": pred,
            "residual": y_test.values - pred,
        })
        res_path = f"reports/{run_name}_residuals.csv"
        residuals.to_csv(res_path, index=False)
        mlflow.log_artifact(res_path)

        joblib.dump(model, f"models/{run_name}.joblib")
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=TRUSTED_TYPES,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True,
                        choices=["E1", "E2", "E3", "E4"])
    args = parser.parse_args()

    mlflow.set_experiment("fish-grow-lab02")

    if args.experiment == "E1":
        X_train, y_train, X_test, y_test = load_split()
        model = make_pipeline(LinearRegression())
        evaluate(model, X_train, y_train, X_test, y_test,
                 "E1-linear", {"model": "LinearRegression"})

    elif args.experiment == "E2":
        X_train, y_train, X_test, y_test = load_split()
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            model = make_pipeline(Ridge(alpha=alpha))
            evaluate(model, X_train, y_train, X_test, y_test,
                     f"E2-ridge-alpha{alpha}",
                     {"model": "Ridge", "alpha": alpha})

    elif args.experiment == "E3":
        X_train, y_train, X_test, y_test = load_split()
        model = make_pipeline(HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.1, max_depth=5,
            random_state=RANDOM_STATE))
        evaluate(model, X_train, y_train, X_test, y_test,
                 "E3-hgb",
                 {"model": "HistGradientBoosting", "max_iter": 200,
                  "learning_rate": 0.1, "max_depth": 5})

    elif args.experiment == "E4":
        extras = {"Vproxy": lambda d: d["Length3"] * d["Height"] * d["Width"]}
        X_train, y_train, X_test, y_test = load_split(extra_features=extras)
        model = make_pipeline(Ridge(alpha=1.0), extra_numeric=["Vproxy"])
        evaluate(model, X_train, y_train, X_test, y_test,
                 "E4-ridge-vproxy",
                 {"model": "Ridge", "alpha": 1.0, "extra_feature": "Vproxy"})


if __name__ == "__main__":
    main()