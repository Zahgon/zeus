"""Server global configurations."""

from __future__ import annotations
from typing import Union

from dotenv import find_dotenv
from zeus.utils.pydantic_v1 import BaseSettings, validator


class ZeusBsoSettings(BaseSettings):
    """App setting.

    Attributes:
        database_url: url of database for the server
        echo_sql: log sql statements it executes
        log_level: level of log
    """

    database_url: str
    echo_sql: Union[bool, str] = False  # To prevent conversion error for empty string
    log_level: str = "INFO"

    class Config:
        """Model configuration.

        Set how to find the env variables and how to parse it.
        """

        env_prefix = "ZEUS_BSO_"
        env_file = find_dotenv(filename=".env")
        env_file_encoding = "utf-8"

    @validator("echo_sql")
    def _validate_echo_sql(cls, v) -> bool:
        pass

    @validator("log_level")
    def _validate_log_level(cls, v) -> str:
        pass


settings = ZeusBsoSettings()  # type: ignore
