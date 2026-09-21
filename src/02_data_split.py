"""任务二前置：数据划分

按时间戳划分训练/验证/测试集，防止数据泄漏。
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def temporal_split(ratings, config):
    """按时间顺序划分数据"""
    train_ratio = config["split"]["train_ratio"]
    val_ratio = config["split"]["val_ratio"]

    ratings = ratings.sort_values("timestamp")
    n = len(ratings)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = ratings.iloc[:train_end]
    val = ratings.iloc[train_end:val_end]
    test = ratings.iloc[val_end:]

    print(f"训练集: {len(train):,} 条")
    print(f"验证集: {len(val):,} 条")
    print(f"测试集: {len(test):,} 条")
    return train, val, test


def generate_candidates(test, config):
    """生成候选集：测试正样本 + 固定负样本"""
    np.random.seed(config["split"]["neg_sample_seed"])
    threshold = config["split"]["pos_threshold"]
    neg_count = config["split"]["neg_sample_count"]

    pos_items = test[test["rating"] >= threshold]["movieId"].unique()
    all_items = set(test["movieId"].unique())

    # TODO: 实现负采样逻辑
    print(f"正样本数: {len(pos_items):,}")
    return pos_items


def main():
    config = load_config()
    raw_dir = config["data"]["raw_dir"]
    splits_dir = config["data"]["splits_dir"]
    os.makedirs(splits_dir, exist_ok=True)

    ratings_path = os.path.join(raw_dir, config["data"]["ratings_file"])
    if not os.path.exists(ratings_path):
        print("[ERROR] 请先下载数据文件")
        sys.exit(1)

    ratings = pd.read_csv(ratings_path)
    train, val, test = temporal_split(ratings, config)

    train.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(splits_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(splits_dir, "test.csv"), index=False)

    print("\n[Done] 数据划分完成")


if __name__ == "__main__":
    main()
