"""任务三（算法模块2）：Item-based 协同过滤

基于物品的协同过滤推荐算法。
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_item_similarity(ratings, config):
    """构建物品相似度矩阵"""
    threshold = config["split"]["pos_threshold"]
    pos_ratings = ratings[ratings["rating"] >= threshold]

    users = pos_ratings["userId"].unique()
    items = pos_ratings["movieId"].unique()
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {m: i for i, m in enumerate(items)}

    row = pos_ratings["userId"].map(user_map).values
    col = pos_ratings["movieId"].map(item_map).values
    data = np.ones(len(pos_ratings))

    user_item = csr_matrix((data, (row, col)), shape=(len(users), len(items)))
    item_sim = cosine_similarity(user_item.T)

    return item_sim, user_map, item_map


def main():
    config = load_config()
    splits_dir = config["data"]["splits_dir"]
    train_path = os.path.join(splits_dir, "train.csv")

    if not os.path.exists(train_path):
        print("[ERROR] 请先运行 02_data_split.py")
        sys.exit(1)

    train = pd.read_csv(train_path)
    item_sim, user_map, item_map = build_item_similarity(train, config)

    print(f"物品相似度矩阵: {item_sim.shape}")
    print("[Done] ItemCF 训练完成")


if __name__ == "__main__":
    main()
