from typing import Any, Dict
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.learning.config import cat_features, config, features, target
from src.learning.ml_model import PriceModel
from src.learning.preprocessing import RealEstatePreprocessor




def run_training_pipeline(
    data_path: str = 'data.parquet',
    tune_hyperparams: bool = True,
    tune_trials: int = 10
) -> Dict[str, Any]:
    """
    Полный pipeline тренировки модели:
      - загрузка и предобработка данных,
      - разделение на train/val/test,
      - опциональная оптимизация гиперпараметров через Optuna,
      - обучение с ранней остановкой,
      - оценка на тесте,
      - сохранение модели.
    """
    preprocessor = RealEstatePreprocessor(**config)
    df = preprocessor.load_data(data_path)
    df = preprocessor.preprocess(df)
    df = preprocessor.feature_engineer(df)
    X, y = preprocessor.get_features_and_target(df, features, cat_features, target)

    mask = y.notna()
    if not mask.any():
        raise ValueError("Нет доступных значений таргета для обучения")
    X, y = X[mask], y[mask]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42
    )

    model = PriceModel()
    tuning_results: Dict[str, Any] = {}
    if tune_hyperparams:
        logger.info("Начало гиперпараметрического тюнинга Optuna")
        tuning_results = model.tune(
            X_train, y_train,
            cat_features=cat_features,
            n_trials=tune_trials
        )
        logger.info(f"Результаты тюнинга: {tuning_results}")

    logger.info("Обучение модели с early stopping на валидации")
    model.train(
        X_train, y_train,
        X_val, y_val,
        cat_features=cat_features
    )

    logger.info("Оценка модели на тестовом наборе данных")
    eval_results = model.evaluate(X_test, y_test)

    model.save('price_per_sqm_model.cbm')

    return {
        'tuning': tuning_results,
        'evaluation': eval_results
    }