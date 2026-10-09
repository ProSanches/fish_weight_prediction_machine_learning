"""Общая цепочка предобработки и подготовки данных FishGrow."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUMERIC_COLUMNS = ["Length3", "Height", "Width"]
CATEGORICAL_COLUMNS = ["Species"]


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет строки-выбросы по правилу IQR.

    Строка удаляется, если она является выбросом хотя бы по одному
    из числовых столбцов Weight/Length1/Length2/Length3/Height/Width.
    """
    numerical = ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]
    mask = pd.Series(False, index=df.index)
    for col in numerical:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask |= (df[col] < lo) | (df[col] > hi)
    return df.loc[~mask].copy()


def prep_features(df: pd.DataFrame):
    """Убирает Length1/Length2 и делит на X и y."""
    df = df.drop(columns=["Length1", "Length2"])
    X = df.drop(columns=["Weight"])
    y = df["Weight"]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: log+scale для числовых, one-hot для Species."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log, validate=False)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_COLUMNS),
        ("cat", categorical, CATEGORICAL_COLUMNS),
    ])