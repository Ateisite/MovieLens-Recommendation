"""共用工具函数"""

import os
import yaml
import logging
from datetime import datetime


def load_config(config_path="config.yaml"):
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(name, log_file=None, level=logging.INFO):
    """配置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_raw_data(config):
    """加载原始数据文件"""
    raw_dir = config["data"]["raw_dir"]
    ratings = pd.read_csv(os.path.join(raw_dir, config["data"]["ratings_file"]))
    movies = pd.read_csv(os.path.join(raw_dir, config["data"]["movies_file"]))
    tags = pd.read_csv(os.path.join(raw_dir, config["data"]["tags_file"]))
    return ratings, movies, tags


def check_data_files(config):
    """检查数据文件是否存在"""
    raw_dir = config["data"]["raw_dir"]
    required_files = [
        config["data"]["ratings_file"],
        config["data"]["movies_file"],
        config["data"]["tags_file"],
    ]
    for f in required_files:
        path = os.path.join(raw_dir, f)
        if not os.path.exists(path):
            print(f"[ERROR] 缺少数据文件: {f}")
            print(f"请从 https://grouplens.org/datasets/movielens/32m/ 下载并放入 {raw_dir}/")
            return False
    return True


def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def get_timestamp():
    """获取当前时间戳字符串"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 延迟导入避免循环依赖
import pandas as pd
