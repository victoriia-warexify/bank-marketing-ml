
import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    build_preprocessing_pipeline,
    transform_month,
    transform_pdays,
)


# ------------------------------------------------------------
# Тестовые данные
# ------------------------------------------------------------

@pytest.fixture
def sample_data():
    n = 40

    X = pd.DataFrame({
        "age": [25 + i % 15 for i in range(n)],
        "campaign": [1 + i % 3 for i in range(n)],
        "previous": [i % 2 for i in range(n)],
        "pdays": [999 if i % 3 else 5 for i in range(n)],
        "emp_var_rate": [-1.8] * n,
        "cons_price_idx": [93.5] * n,
        "cons_conf_idx": [-40.0] * n,
        "euribor3m": [1.3] * n,
        "nr_employed": [5099.1] * n,
        "marital": ["married", "single"] * 20,
        "default": ["no"] * n,
        "housing": ["yes"] * n,
        "loan": ["no"] * n,
        "contact": ["cellular"] * n,
        "day_of_week": ["mon", "tue"] * 20,
        "poutcome": ["nonexistent"] * n,
        "job": ["admin.", "technician"] * 20,
        "education": [
            "high.school", "university.degree"
        ] * 20,
        "month": ["mar", "oct"] * 20,
    })

    y = pd.Series([0, 1] * 20)

    return X, y


# ------------------------------------------------------------
# 1. Преобразование pdays
# ------------------------------------------------------------

def test_transform_pdays():
    X = pd.DataFrame({
        "age": [30, 40, 50],
        "pdays": [999, 5, 0]
    })

    result = transform_pdays(X)

    assert "pdays" not in result.columns
    assert result["pdays_known"].tolist() == [0, 1, 1]
    assert result["pdays_days"].tolist() == [0, 5, 0]
    assert result["age"].tolist() == [30, 40, 50]


# ------------------------------------------------------------
# 2. Проверка неизменности исходных данных
# ------------------------------------------------------------

def test_transform_pdays_does_not_modify_input():
    X = pd.DataFrame({"pdays": [999, 7]})
    original = X.copy(deep=True)

    transform_pdays(X)

    pd.testing.assert_frame_equal(X, original)


# ------------------------------------------------------------
# 3. Циклическое кодирование месяца
# ------------------------------------------------------------

def test_transform_month():
    X = pd.DataFrame(
        {"month": ["mar", "sep"]},
        index=[10, 20]
    )

    result = transform_month(X)

    assert result.index.tolist() == [10, 20]
    assert result.columns.tolist() == [
        "month_sin", "month_cos"
    ]

    np.testing.assert_allclose(
        result["month_sin"],
        [1.0, -1.0],
        atol=1e-10
    )

    np.testing.assert_allclose(
        result["month_cos"],
        [0.0, 0.0],
        atol=1e-10
    )


# ------------------------------------------------------------
# 4. Обработка неизвестного месяца
# ------------------------------------------------------------

def test_transform_month_unknown_value():
    X = pd.DataFrame({"month": ["invalid"]})

    with pytest.raises(
        ValueError,
        match="неизвестные значения month"
    ):
        transform_month(X)


# ------------------------------------------------------------
# 5. Проверка полного пайплайна предобработки
# ------------------------------------------------------------

def test_preprocessing_pipeline(sample_data):
    X, y = sample_data

    pipeline = build_preprocessing_pipeline()

    X_transformed = pipeline.fit_transform(X, y)

    assert X_transformed.shape[0] == len(X)
    assert np.isfinite(X_transformed).all()

    # Новые данные с неизвестными категориями
    X_new = X.iloc[:2].copy()

    X_new.loc[0, "contact"] = "unknown_contact"
    X_new.loc[0, "job"] = "unknown_job"
    X_new.loc[0, "education"] = "unknown_education"

    X_new_transformed = pipeline.transform(X_new)

    assert X_new_transformed.shape[0] == 2
    assert (
        X_new_transformed.shape[1]
        == X_transformed.shape[1]
    )
    assert np.isfinite(X_new_transformed).all()


# ------------------------------------------------------------
# 6. Интеграционный тест обучения и прогнозирования
# ------------------------------------------------------------

def test_model_training_and_prediction(sample_data):
    X, y = sample_data

    model = Pipeline([
        (
            "preprocessing",
            build_preprocessing_pipeline()
        ),
        (
            "model",
            LogisticRegression(max_iter=1000)
        )
    ])

    # Обучаем модель вместе с предобработкой
    model.fit(X, y)

    # Получаем вероятности для пяти наблюдений
    probabilities = model.predict_proba(X.iloc[:5])

    # Проверяем размерность результата
    assert probabilities.shape == (5, 2)

    # Вероятности должны быть конечными
    assert np.isfinite(probabilities).all()

    # Каждая вероятность находится в диапазоне [0, 1]
    assert np.all(
        (probabilities >= 0)
        & (probabilities <= 1)
    )

    # Сумма вероятностей классов равна 1
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(5),
        atol=1e-10
    )
