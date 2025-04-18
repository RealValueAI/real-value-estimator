import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

import pandas as pd
from sqlalchemy import create_engine

# Загружаем переменные окружения из .env файла
load_dotenv()

# Настройка логгирования для более красивого вывода
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class ClickHouseConfig:
    host: str = os.getenv("CLICKHOUSE_HOST")
    port: int = int(os.getenv("CLICKHOUSE_PORT"))
    user: str = os.getenv("CLICKHOUSE_USER")
    password: str = os.getenv("CLICKHOUSE_PASSWORD")
    database: str = os.getenv("CLICKHOUSE_DATABASE")
    table_name: str = os.getenv("TABLE_NAME")


clickhouse_config = ClickHouseConfig()
