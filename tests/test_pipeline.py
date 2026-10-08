"""Минимальные автоматические проверки для лабораторной №2."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor, handle_outliers, prep_features


def load_clean_data():
    df = pd.read_csv("data/fish_participant.csv")
    return handle_outliers(df)


def test_split_indices_do_not_overlap():
    train_idx = np.load("configs/train_indices.npy")
    test_idx = np.load("configs/test_indices.npy")
    assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_pipeline_handles_unknown_category():
    df = load_clean_data()
    X, y = prep_features(df)

    model = Pipeline([("preprocess", build_preprocessor()),
                      ("regressor", Ridge())])
    model.fit(X, y)

    new_row = X.iloc[[0]].copy()
    new_row["Species"] = "UnknownFish"
    pred = model.predict(new_row)
    assert np.isfinite(pred).all()


def test_predictions_count_and_finiteness():
    df = load_clean_data()
    X, y = prep_features(df)

    model = Pipeline([("preprocess", build_preprocessor()),
                      ("regressor", Ridge())])
    model.fit(X, y)

    pred = model.predict(X.head(10))
    assert len(pred) == 10
    assert np.isfinite(pred).all()


def test_metrics_manual_calculation():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    assert mean_squared_error(y_true, y_pred) == pytest.approx(0.02, abs=1e-9)
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(0.1333, abs=1e-4)


def test_saved_model_reproduces_predictions():
    df = load_clean_data()
    X, _ = prep_features(df)

    model = joblib.load("models/best_model.joblib")
    pred1 = model.predict(X.head(5))
    pred2 = model.predict(X.head(5))
    assert np.allclose(pred1, pred2)


def test_no_negative_predictions_on_train():
    df = load_clean_data()
    X, _ = prep_features(df)

    model = joblib.load("models/best_model.joblib")
    pred = model.predict(X)
    assert (pred > 0).all(), "Модель выдала отрицательные предсказания"