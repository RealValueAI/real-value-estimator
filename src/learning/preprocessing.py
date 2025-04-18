import ast
import logging
import math
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.utils.logger import logger


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_array_field(x: Any) -> List:
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    if isinstance(x, list):
        return x
    return []

class RealEstatePreprocessor:
    def __init__(
        self,
        dtypes: Dict[str, Any],
        parse_dates: List[str],
        array_columns: List[str],
        numeric_fillna: Dict[str, Any],
        bool_columns: List[str],
        center_latitude: float,
        center_longitude: float
    ):
        self.dtypes = dtypes
        self.parse_dates = parse_dates
        self.array_columns = array_columns
        self.numeric_fillna = numeric_fillna
        self.bool_columns = bool_columns
        self.center_lat = center_latitude
        self.center_lon = center_longitude

    def load_data(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Загрузка данных из {file_path}")
        df = pd.read_parquet(file_path, engine='pyarrow')
        for col in self.array_columns:
            if col in df.columns:
                df[col] = df[col].apply(parse_array_field)
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Предобработка данных")
        # Заполнение числовых пропусков
        for col, val in self.numeric_fillna.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        # Булевы флаги
        for col in self.bool_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        # Подготовка subway_names как категориальной фичи
        if 'subway_names' in df.columns:
            df['subway_names'] = df['subway_names'].apply(
                lambda x: x if isinstance(x, list) and x else ['unknown']
            )
        return df

    def feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Feature Engineering: создание новых признаков")
        now = datetime.now()
        # Географические признаки
        if 'subway_distances' in df.columns:
            df['num_subways'] = df['subway_distances'].apply(len)
            df['subway_min_dist'] = df['subway_distances'].apply(lambda a: min(a) if a else np.nan)
            df['subway_mean_dist'] = df['subway_distances'].apply(lambda a: np.mean(a) if a else np.nan)
        # Основная станция метро
        if 'subway_names' in df.columns:
            df['primary_subway'] = df['subway_names'].apply(lambda a: a[0])
        # Координаты и расстояние до центра
        if {'latitude', 'longitude'}.issubset(df.columns):
            df['distance_to_center'] = df.apply(
                lambda r: haversine(r['latitude'], r['longitude'], self.center_lat, self.center_lon),
                axis=1
            )
            bins = [0, 1, 3, 5, 10, np.inf]
            labels = ['<1km', '1-3km', '3-5km', '5-10km', '>10km']
            df['distance_bucket'] = pd.cut(df['distance_to_center'], bins=bins, labels=labels)
            df['lat_bucket'] = df['latitude'].round(2)
            df['lon_bucket'] = df['longitude'].round(2)
        # Временные дельты
        if {'published_date', 'updated_date'}.issubset(df.columns):
            df['days_since_published'] = (now - df['published_date']).dt.days
            df['days_since_updated'] = (df['updated_date'] - df['published_date']).dt.days
        # Отношения
        if {'house_floors', 'floor'}.issubset(df.columns):
            df['floor_ratio'] = np.where(df['house_floors'] > 0, df['floor'] / df['house_floors'], -1)
        if {'rooms', 'area'}.issubset(df.columns):
            df['rooms_density'] = np.where(df['area'] > 0, df['rooms'] / df['area'], 0)
        # Текстовые признаки
        if 'description' in df.columns:
            df['desc_len'] = df['description'].str.len()
            df['desc_word_count'] = df['description'].str.split().apply(len)
        return df

    def get_features_and_target(
        self,
        df: pd.DataFrame,
        features: List[str],
        cat_features: List[str],
        target: str
    ) -> (pd.DataFrame, pd.Series):
        requested = features + cat_features
        available = [f for f in requested if f in df.columns]
        missing = set(requested) - set(available)
        if missing:
            logger.warning(f"Пропущены признаки: {sorted(missing)}")
        if target not in df.columns:
            raise KeyError(f"Целевой столбец '{target}' отсутствует")
        X = df[available]
        y = df[target]
        return X, y

