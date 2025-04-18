import logging

import pandas as pd

from src.utils.config import ClickHouseConfig, clickhouse_config
from sqlalchemy import create_engine, text


def create_clickhouse_engine(config: ClickHouseConfig):
    """
    Создает SQLAlchemy engine для подключения к ClickHouse.
    """
    connection_str = (
        f"clickhouse+native://{config.user}:{config.password}@"
        f"{config.host}:{config.port}/{config.database}"
    )
    engine = create_engine(connection_str)
    logging.info("ClickHouse engine создан успешно.")
    return engine


def read_table_to_dataframe(engine, table_full_name: str) -> pd.DataFrame:
    """
    Считывает данные из указанной таблицы ClickHouse и возвращает DataFrame.
    :param engine: SQLAlchemy engine для подключения к ClickHouse.
    :param table_full_name: Полное имя таблицы (например, dwh.listings_actual_c).
    :return: DataFrame с данными из таблицы.
    """
    query = f"SELECT * FROM {table_full_name}"
    logging.info(f"Выполняется запрос: {query}")
    df = pd.read_sql(query, engine)
    logging.info(f"Считано {len(df)} строк из таблицы {table_full_name}.")
    return df


def save_dataframe_to_parquet(df: pd.DataFrame, file_path: str) -> None:
    """
    Сохраняет DataFrame в файл формата Parquet.
    :param df: DataFrame для сохранения.
    :param file_path: Путь к выходному файлу Parquet.
    """
    df.to_parquet(file_path, index=False)
    logging.info(f"DataFrame сохранен в Parquet файл: {file_path}")


def refresh_data():

    engine = create_clickhouse_engine(clickhouse_config)
    table_full_name = f"dwh.{clickhouse_config.table_name}"
    df = read_table_to_dataframe(engine, table_full_name)
    output_file = "data.parquet"

    save_dataframe_to_parquet(df, output_file)

def upload_group_labels(parquet_path: str, table_full_name: str):
    """
    Загружает в ClickHouse содержимое parquet_path (listing_id, platform_id, group_id)
    в таблицу table_full_name: сначала дропаем, затем создаём и вставляем данные.
    """

    engine = create_clickhouse_engine(clickhouse_config)
    df = pd.read_parquet(parquet_path, columns=['listing_id', 'platform_id', 'group_id'])
    logging.info(f"Прочитано {len(df)} строк из {parquet_path}.")


    if '.' in table_full_name:
        schema, table = table_full_name.split('.', 1)
    else:
        schema, table = engine.url.database, table_full_name

    # 3) Дроп и создание новой таблицы
    ddl_drop = text(f"DROP TABLE IF EXISTS {table_full_name}")
    ddl_create = text(f"""
        CREATE TABLE {table_full_name} (
            listing_id   UInt64,
            platform_id  UInt8,
            group_id     Int64
        ) ENGINE = MergeTree()
        ORDER BY (listing_id)
    """)
    with engine.begin() as conn:
        conn.execute(ddl_drop)
        logging.info(f"Таблица {table_full_name} удалена (если существовала).")
        conn.execute(ddl_create)
        logging.info(f"Таблица {table_full_name} создана заново.")


    df = df.astype(
        {
            'listing_id': 'int64',
            'platform_id': 'int8',
            'group_id': 'int64'
        }
    )
    df.to_sql(
        name=table,
        con=engine,
        schema=schema,
        if_exists='append',
        index=False,
    )
    logging.info(f"Загружено {len(df)} записей в {table_full_name}.")