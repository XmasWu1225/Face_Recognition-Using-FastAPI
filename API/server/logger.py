# server/logger.py
import logging
import os
from pathlib import Path

# 日志文件路径（项目根目录下的 logs 文件夹）
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)  # 创建 logs 目录（如果不存在）
LOG_FILE = LOG_DIR / "app.log"

# 配置日志
def setup_logger(name: str = "FaceRecognitionAPI", level: int = logging.INFO) -> logging.Logger:
    """设置并返回一个全局日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 防止重复添加处理器
    if not logger.handlers:
        # 文件处理器：输出到 logs/app.log
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(level)

        # 控制台处理器：输出到终端
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # 日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# 默认全局日志记录器
logger = setup_logger()