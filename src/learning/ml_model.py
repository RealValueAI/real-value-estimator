import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool
from catboost import CatBoostError
from optuna.integration import CatBoostPruningCallback
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.utils.logger import logger


class PriceModel:
    """
    CatBoost wrapper with Optuna hyperparameter tuning and overfitting control.
    """
    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.params = params or {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'eval_metric': 'RMSE',
            'early_stopping_rounds': 50,
            'random_seed': 42,
            'verbose': 0
        }
        self.model: Optional[CatBoostRegressor] = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        cat_features: Optional[List[str]] = None
    ) -> None:
        """
        Train CatBoost model on training set, optionally with validation for early stopping.
        """
        model = CatBoostRegressor(**self.params)
        if X_val is not None and y_val is not None and not y_val.empty:
            pool_train = Pool(X_train, y_train, cat_features=cat_features)
            pool_val = Pool(X_val, y_val, cat_features=cat_features)
            model.fit(
                pool_train,
                eval_set=pool_val,
                verbose=100
            )
        else:
            model.fit(
                X_train,
                y_train,
                cat_features=cat_features,
                verbose=100
            )
        self.model = model

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Compute MAE, RMSE, MAPE and feature importances.
        """
        assert self.model is not None, "Model not trained"
        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mask = y_test != 0
        mape = np.nan
        if mask.any():
            mape = np.mean(np.abs((y_test[mask] - preds[mask]) / y_test[mask])) * 100
        # Feature importances
        names = X_test.columns.tolist()
        imps = self.model.get_feature_importance(type='FeatureImportance')
        feat_imp = sorted(zip(names, imps), key=lambda x: x[1], reverse=True)
        logger.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
        for name, score in feat_imp:
            logger.info(f" {name}: {score:.4f}")
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'feature_importances': feat_imp
        }

    def save(self, path: str) -> None:
        """Save trained model to file."""
        assert self.model is not None, "Model not trained"
        logger.info(f"Saving model to {path}")
        self.model.save_model(path)

    def load(self, path: str) -> None:
        """Load model from file."""
        logger.info(f"Loading model from {path}")
        self.model = CatBoostRegressor()
        self.model.load_model(path)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        assert self.model is not None, "Model not trained or loaded"
        return self.model.predict(X)

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cat_features: Optional[List[str]] = None,
        n_trials: int = 50,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Hyperparameter tuning with Optuna, returns best params and scores.
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        def objective(trial: optuna.Trial) -> float:
            params = {
                'iterations': trial.suggest_int(
                    'iterations',
                    500,
                    2500,
                    step=250,
                ),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'random_seed': random_state,
                'eval_metric': 'RMSE'
            }
            model = CatBoostRegressor(**params)
            try:
                model.fit(
                    Pool(X_train, y_train, cat_features=cat_features),
                    eval_set=Pool(X_val, y_val, cat_features=cat_features),
                    early_stopping_rounds=50,
                    callbacks=[CatBoostPruningCallback(trial, 'RMSE')],
                    verbose=False
                )
            except CatBoostError:
                return float('inf')
            preds = model.predict(X_val)
            return mean_squared_error(y_val, preds, squared=False)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best = study.best_params
        logger.info(f"Best params: {best}")
        self.params.update(best)
        return {'best_params': best, 'best_value': study.best_value}