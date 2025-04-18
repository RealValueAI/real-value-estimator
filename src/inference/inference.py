import logging

import pandas as pd
from clickhouse_driver import Client

from src.learning.config import cat_features, config, features
from src.learning.ml_model import PriceModel
from src.learning.preprocessing import RealEstatePreprocessor
from src.utils.config import clickhouse_config
from src.utils.logger import logger


def run_inference_pipeline(
    data_path: str = 'x_inference.parquet',
    model_path: str = 'price_per_sqm_model.cbm'
) -> pd.DataFrame:
    """
    Pipeline инференса: загрузка данных, предсказания и запись в ClickHouse.
    Использует конфигурацию из clickhouse_config для подключения.
    """
    # Загрузка модели
    model = PriceModel()
    model.load(model_path)

    # Предобработка и feature engineering
    preprocessor = RealEstatePreprocessor(**config)
    df = preprocessor.load_data(data_path)
    df = preprocessor.preprocess(df)
    df = preprocessor.feature_engineer(df)

    # Проверка наличия всех признаков
    requested = features + cat_features
    missing = [f for f in requested if f not in df.columns]
    if missing:
        raise KeyError(f"Отсутствуют признаки в данных для инференса: {missing}")

    X = df[requested]
    preds = model.predict(X)
    df['pred_price_per_sqm'] = preds

    out_df = df[['listing_id', 'platform_id', 'pred_price_per_sqm']].copy()
    # Приведение типов перед вставкой
    out_df['listing_id'] = out_df['listing_id'].astype('int64')
    out_df['platform_id'] = out_df['platform_id'].astype('int64')
    out_df['pred_price_per_sqm'] = out_df['pred_price_per_sqm'].astype(
        'float64'
        )

    # Подключение к ClickHouse и запись
    client = Client(
        host=clickhouse_config.host,
        port=clickhouse_config.port_native,
        user=clickhouse_config.user,
        password=clickhouse_config.password,
        database=clickhouse_config.database
    )
    table = clickhouse_config.output_table
    if not table:
        raise ValueError("OUTPUT_TABLE_NAME не задана в конфиге")

    # Создаем таблицу, если не существует
    client.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            listing_id UInt64,
            platform_id UInt32,
            pred_price_per_sqm Float64
        ) ENGINE = MergeTree()
        ORDER BY (listing_id, platform_id)
    """)
    # Очищаем старые данные
    client.execute(f"TRUNCATE TABLE {table}")
    # Вставляем данные
    records = []
    for row in out_df.itertuples(index=False):
        lid, pid, val = row
        records.append((int(lid), int(pid), float(val)))
    client.execute(
        f"INSERT INTO {table} (listing_id, platform_id, pred_price_per_sqm) VALUES",
        records
    )

    logger.info(f"Записано {len(out_df)} строк в таблицу {table} на {clickhouse_config.host}")
    return out_df
