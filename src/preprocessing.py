"""Преобразования признаков для моделей Bank Marketing."""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
)


NUMERIC_COLS = [
    "age",
    "campaign",
    "previous",
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor3m",
    "nr_employed",
    "pdays_days",
]

ONEHOT_COLS = [
    "marital",
    "default",
    "housing",
    "loan",
    "contact",
    "day_of_week",
    "poutcome",
]

TARGET_COLS = ["job", "education"]
CYCLICAL_COLS = ["month"]
BINARY_COLS = ["pdays_known"]

MONTH_MAPPING = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def transform_pdays(X: pd.DataFrame) -> pd.DataFrame:
    """Разделяет pdays на число дней и индикатор известного интервала."""
    X = X.copy()
    X["pdays_known"] = (X["pdays"] != 999).astype(int)
    X["pdays_days"] = X["pdays"].where(X["pdays"] != 999, 0)
    return X.drop(columns="pdays")


def transform_month(X: pd.DataFrame) -> pd.DataFrame:
    """Кодирует месяц с помощью синуса и косинуса."""
    month_num = X["month"].map(MONTH_MAPPING)
    if month_num.isna().any():
        raise ValueError("Обнаружены неизвестные значения month")

    return pd.DataFrame(
        {
            "month_sin": np.sin(2 * np.pi * month_num / 12),
            "month_cos": np.cos(2 * np.pi * month_num / 12),
        },
        index=X.index,
    )


def month_feature_names(transformer, input_features):
    """Возвращает названия признаков для month."""
    return np.array(["month_sin", "month_cos"], dtype=object)


def build_preprocessing_pipeline() -> Pipeline:
    """Собирает пайплайн предобработки. Обучение выполняется отдельно."""
    target_transformer = Pipeline(
        steps=[
            (
                "encoder",
                TargetEncoder(
                    target_type="binary",
                    cv=StratifiedKFold(
                        n_splits=5,
                        shuffle=True,
                        random_state=42,
                    ),
                    smooth="auto",
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    month_transformer = FunctionTransformer(
        transform_month,
        validate=False,
        feature_names_out=month_feature_names,
    )

    columns = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLS),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ONEHOT_COLS,
            ),
            ("target", target_transformer, TARGET_COLS),
            ("month", month_transformer, CYCLICAL_COLS),
            ("binary", "passthrough", BINARY_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Pipeline(
        steps=[
            ("pdays", FunctionTransformer(transform_pdays, validate=False)),
            ("columns", columns),
        ]
    )
