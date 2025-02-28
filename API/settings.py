# server/settings.py
import os
from pydantic_settings import BaseSettings
from typing import Dict, Union  # 引入 Union

class Settings(BaseSettings):
    # 基础配置
    OPEN_CROSS_DOMAIN: bool = True
    API_SERVER: Dict[str, Union[str, int]] = {  # 使用 Union[str, int] 替代 str | int
        "host": "0.0.0.0",
        "port": 8000,
    }
    VERSION: str = "1.0.0"
    
    # SSL 配置
    SSL_KEYFILE: str = ""
    SSL_CERTFILE: str = ""
    
    # 模型路径配置
    SCRFD_ENGINE_PATH: str = "/home/aa/API/SCRFD/API_detection-l.engine"
    ARCFACE_ENGINE_PATH: str = "/home/aa/API/SCRFD/API_arcface.engine"
    
    # 数据存储路径
    DATA_DIR: str = "./data/face_recognition"
    DETECTED_FACES_DIR: str = "./data/detected_faces"
    
    # 人脸检测配置
    DEFAULT_CONF_THRES: float = 0.5
    DEFAULT_IOU_THRES: float = 0.4
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# 实例化设置
settings = Settings()