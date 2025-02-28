import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 基础配置
    OPEN_CROSS_DOMAIN: bool = True
    API_SERVER: dict = {
        "host": "0.0.0.0",
        "port": 8000,
    }
    VERSION: str = "1.0.0"

    # SSL 配置
    SSL_KEYFILE: str = ""
    SSL_CERTFILE: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# 实例化设置
basic_settings = Settings()