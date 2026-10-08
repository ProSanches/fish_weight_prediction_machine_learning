"""E0: наивные прогнозы (среднее и медиана), фиксация разбиения."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    df = pd.read_csv("data/fish_participant.csv")
    X = df.drop(columns=["Weight"])
    y = df["Weight"]

    idx_train, idx_test = train_test_split(
        df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    np.save("configs/test_indices.npy", idx_test.values)
    np.save("configs/train_indices.npy", idx_train.values)

    y_train = y.loc[idx_train]
    y_test = y.loc[idx_test]

    print(f"Размер обучающей выборки: {len(idx_train)}")
    print(f"Размер тестовой выборки:  {len(idx_test)}")
    print(f"Индексы сохранены в configs/")
    print()

    for name, value in [("среднее", y_train.mean()), ("медиана", y_train.median())]:
        pred = np.full(len(y_test), value)
        print(f"--- E0 ({name}): прогноз = {value:.2f} ---")
        print(f"MSE: {mean_squared_error(y_test, pred):.2f}")
        print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")
        print(f"R² : {r2_score(y_test, pred):.4f}")
        print()


if __name__ == "__main__":
    main()