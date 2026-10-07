# Fish Weight Prediction Model:

![Fish](images/Fishlength.jpg)

## Requirements

Hello! Welcome to the famous Tsukiji fish market of Tokyo, Japan! We came here to collect data on some of the fish they have here 
but we didn't wake up at 5am for the tuna auction and by the time we showed up they were only left with a few species of fish. 
We got to work and gathered measurements from a few different species of fish and want you to train a regression model to predict
the weight of a fish using some of the features we were able to measure. We have no idea which features will be good predictors. 

We will hold out 30% of the data before we hand it to you and we will use that csv for scoring.

Here's what we need from you:
1. A function that accepts a csv path and returns the predictions of your regression model using our csv. The csv we use will contain all the columns. 
2. Use a pipenv and scikit learn to submit the final model. You may use R for model selection. 

With this function and it's output, we will rank the students by how well their model performed on predicting weight based on naive data. 
Your grade will be determined by ranking according to Mean-squared Error.
If your function does not return a list of predictions or we cannot compute the accuracy of your model that it will be an automatic F. 

## Used

*   Linear Regression
*   Lasso Regression
*   Random Forest
*   Scikit Learn
*   Python
*   Joblib
*   Matplotlib
*   Pipenv

# Паспорт данных

## Источник
Датасет Fish Market (Tsukiji). Файлы: `fish_participant.csv` (train, 111 строк),
`fish_holdout_demo.csv` (демо, 28 строк).

## Поля

| Поле | Тип | Единицы | Роль | Что проверить |
|------|-----|---------|------|----------------|
| Species | категориальный | — | признак | 7 категорий, нет пустых строк |
| Weight | числовой | г | **целевая** | > 0, правосторонняя асимметрия |
| Length1 | числовой | см | признак (отброшен) | сильно коррелирует с Length2/3 |
| Length2 | числовой | см | признак (отброшен) | сильно коррелирует с Length1/3 |
| Length3 | числовой | см | признак | > 0, максимум среди длин |
| Height | числовой | см | признак | > 0 |
| Width | числовой | см | признак | > 0 |

## Решения
- Length1, Length2 исключены: корреляция с Length3 ≈ 0.99 → мультиколлинеарность.
- Length3 оставлена как «Total Length» — максимум из трёх длин в каждой строке.
- Один выраженный выброс: строка 13 (Pike, Weight=1550 г).
  В исходном репозитории удаляется. В лабе №2 решение нужно обосновать в отчёте.

## Пропуски
Нет (проверено `df.notna().all().all()`).

## Правило разбиения
Фиксированное: `train_test_split(test_size=0.2, random_state=42)`,
индексы теста сохраняются в `configs/test_indices.npy`.